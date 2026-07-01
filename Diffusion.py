import torch
import math
import numpy as np
from helpers import *


# -----------------------------------------------------------------------------
# Schedules / utilities
# -----------------------------------------------------------------------------

def get_beta_schedule(num_diffusion_steps, name="cosine", gamma=1.0):
    """
    Return a beta schedule as a numpy array of shape [T].

    Supported:
      - "cosine"  (Nichol & Dhariwal style)
      - "linear"  (DDPM-style)
    """
    if name == "cosine":
        betas = []
        max_beta = 0.999
        cosine_s=0.008

        def alpha_bar_fn(t):
            t_eff = t ** gamma
            return np.cos((t_eff + cosine_s) / (1.0 + cosine_s) * np.pi / 2) ** 2

        for i in range(num_diffusion_steps):
            t1 = i / num_diffusion_steps
            t2 = (i + 1) / num_diffusion_steps

            beta = 1.0 - alpha_bar_fn(t2) / alpha_bar_fn(t1)
            betas.append(min(beta, max_beta))

        return np.array(betas, dtype=np.float64)

    if name == "linear":
        scale = 1000 / num_diffusion_steps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        return np.linspace(beta_start, beta_end, num_diffusion_steps, dtype=np.float64)

    raise NotImplementedError(f"unknown beta schedule: {name}")


def extract(a: torch.Tensor, t: torch.Tensor, x_shape, device):
    """
    Extract values from a 1-D schedule tensor `a` at indices `t` and reshape/broadcast
    to `x_shape`.

    a: [T] tensor
    t: [B] long tensor
    returns: [B, 1, 1, 1, 1] broadcastable to x_shape
    """
    out = a.to(device).gather(0, t)
    while out.ndim < len(x_shape):
        out = out[..., None]
    return out.expand(x_shape)


def mean_flat(tensor: torch.Tensor) -> torch.Tensor:
    """Mean over non-batch dimensions."""
    return torch.mean(tensor, dim=list(range(1, tensor.ndim)))


def normal_kl(mean1, logvar1, mean2, logvar2):
    """KL(N(mean1,var1) || N(mean2,var2)) elementwise."""
    return 0.5 * (
        -1.0
        + logvar2
        - logvar1
        + torch.exp(logvar1 - logvar2)
        + (mean1 - mean2).square() * torch.exp(-logvar2)
    )

def soft_dice_loss(pred_prob, target, eps: float = 1e-6):
    """
    Soft Dice loss for probabilities vs binary targets.
    pred_prob: probabilities in [0,1]
    target: binary mask {0,1}
    returns: per-sample Dice loss [B]
    """
    pred_prob = pred_prob.float()
    target = target.float()

    dims = list(range(1, pred_prob.ndim))
    intersection = torch.sum(pred_prob * target, dim=dims)
    denominator = torch.sum(pred_prob, dim=dims) + torch.sum(target, dim=dims)

    dice = (2.0 * intersection + eps) / (denominator + eps)
    return 1.0 - dice

def approx_standard_normal_cdf(x):
    """Fast approx to standard normal CDF."""
    return 0.5 * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x.pow(3))))


def discretised_gaussian_log_likelihood(x, means, log_scales, bins=256):
    """
    Discretized Gaussian log-likelihood for data scaled to [0,1].
    x, means, log_scales: same shape
    returns: log probs same shape
    """
    assert x.shape == means.shape == log_scales.shape

    bin_width = 1.0 / (bins - 1)

    centered_x = x - means
    inv_stdv = torch.exp(-log_scales)

    plus_in = inv_stdv * (centered_x + bin_width / 2.0)
    min_in = inv_stdv * (centered_x - bin_width / 2.0)

    cdf_plus = approx_standard_normal_cdf(plus_in)
    cdf_min = approx_standard_normal_cdf(min_in)

    log_cdf_plus = torch.log(cdf_plus.clamp(min=1e-6))
    log_one_minus_cdf_min = torch.log((1.0 - cdf_min).clamp(min=1e-6))

    cdf_delta = cdf_plus - cdf_min
    mid = torch.log(cdf_delta.clamp(min=1e-6))

    log_probs = torch.where(
        x <= (0.0 + bin_width / 2.0),
        log_cdf_plus,
        torch.where(
            x >= (1.0 - bin_width / 2.0),
            log_one_minus_cdf_min,
            mid,
        ),
    )
    return log_probs


def bernoulli_kl(p, q, eps: float = 1e-6):
    """KL(Bern(p) || Bern(q)) elementwise."""
    p = p.clamp(eps, 1.0 - eps)
    q = q.clamp(eps, 1.0 - eps)
    return p * (p.log() - q.log()) + (1.0 - p) * ((1.0 - p).log() - (1.0 - q).log())


def bernoulli_neg_log_likelihood(x, p, eps: float = 1e-6):
    """-log Bernoulli likelihood elementwise for target x in {0,1}."""
    p = p.clamp(eps, 1.0 - eps)
    return -(x * p.log() + (1.0 - x) * (1.0 - p).log())


def dice_per_sample(pred_prob, target, threshold: float = 0.5, eps: float = 1e-8):
    """Dice score per sample for prob map vs binary target."""
    pred = (pred_prob >= threshold).float()
    target = target.float()
    dims = list(range(1, pred.ndim))
    inter = torch.sum(pred * target, dim=dims)
    denom = torch.sum(pred, dim=dims) + torch.sum(target, dim=dims)
    return (2.0 * inter + eps) / (denom + eps)


# -----------------------------------------------------------------------------
# Diffusion model (Gaussian image + Bernoulli segmentation)
# -----------------------------------------------------------------------------

class DiffusionModel:
    """
    Mixed diffusion:
      - Image channel: Gaussian DDPM (predict eps)
      - Seg channel:   Bernoulli diffusion (BerDiff-style; predict eps logits)

    Conventions:
      - images are assumed in [0,1]
      - segmentations are binary {0,1}
      - model(x_t, t) returns:
          * image-only: eps_img
          * seg-only:   eps_seg_logits
          * mixed:      (eps_img, eps_seg_logits)
    """

    # -------------------------
    # Init / schedules
    # -------------------------

    def __init__(
        self,
        img_size,
        betas,
        in_channels=1,
        labels_channels=None,
        loss_types=("l2",),       # image: l2/l1/hybrid ; seg: bce/hybrid_seg
        loss_weight="none",       # "none" / "prop-t" / "uniform"
        noise="gauss",            # only "gauss" supported in this cleaned version
        separate_seg_schedule=False,
    ):
        super().__init__()

        if noise != "gauss":
            raise ValueError(f"Unknown noise type: {noise}")
        self.noise_fn_img = lambda x, t: torch.randn_like(x)

        self.img_size = img_size
        self.in_channels = in_channels

        # ----- modality config -----
        if labels_channels is None:
            labels_channels = ["image"] if in_channels == 1 else ["image", "segmentation"]

        # normalize legacy labels
        labels_channels = list(labels_channels)
        if self.in_channels == 2:
            organ_label = next(x for x in labels_channels if x != "image")
            labels_channels[labels_channels.index(organ_label)] = "segmentation"
        elif self.in_channels == 1 and 'image' not in labels_channels:
             labels_channels = ["segmentation"]

        self.labels_channels = labels_channels
        self.has_image = "image" in labels_channels
        self.has_seg = "segmentation" in labels_channels

        expected = len(labels_channels)
        if in_channels != expected:
            raise ValueError(
                f"in_channels={in_channels} but labels_channels={labels_channels} (expected {expected} channels)"
            )

        self.loss_types = loss_types

        # ----- betas / alphas -----
        if isinstance(betas, np.ndarray):
            betas_t = torch.from_numpy(betas).float()
        elif isinstance(betas, list):
            betas_t = torch.tensor(betas, dtype=torch.float32)
        elif torch.is_tensor(betas):
            betas_t = betas.float().detach().cpu()
        else:
            raise TypeError("betas must be np.ndarray, list, or torch.Tensor")

        self.betas = betas_t
        self.num_timesteps = int(betas_t.shape[0])

        # ----- loss timestep weighting -----
        self.loss_weight = loss_weight
        if loss_weight == "prop-t":
            self.weights = np.arange(self.num_timesteps, 0, -1)
        elif loss_weight == "uniform":
            self.weights = np.ones(self.num_timesteps)
        elif loss_weight == "none":
            self.weights = None
        else:
            raise ValueError(f"Unknown loss_weight: {loss_weight}")

        # ----- precompute schedules -----
        alphas = 1.0 - self.betas
        self.alphas = alphas
        self.sqrt_alphas = torch.sqrt(alphas)
        self.sqrt_betas = torch.sqrt(self.betas)

        self.alphas_cumprod = torch.cumprod(alphas, dim=0)  # [T]
        self.alphas_cumprod_prev = torch.cat(
            [torch.ones(1, dtype=self.betas.dtype), self.alphas_cumprod[:-1]], dim=0
        )

        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

        one_minus = torch.clamp(1.0 - self.alphas_cumprod, min=1e-20)
        self.log_one_minus_alphas_cumprod = torch.log(one_minus)

        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / torch.clamp(self.alphas_cumprod, min=1e-20))
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(
            torch.clamp(1.0 / torch.clamp(self.alphas_cumprod, min=1e-20) - 1.0, min=0.0)
        )

        # Gaussian posterior q(x_{t-1} | x_t, x0)
        self.posterior_variance = self.betas * (1.0 - self.alphas_cumprod_prev) / one_minus
        self.posterior_log_variance_clipped = torch.log(
            torch.clamp(torch.cat([self.posterior_variance[1:2], self.posterior_variance[1:]], dim=0), min=1e-20)
        )

        self.posterior_mean_coef1 = self.betas * torch.sqrt(self.alphas_cumprod_prev) / one_minus
        self.posterior_mean_coef2 = (1.0 - self.alphas_cumprod_prev) * torch.sqrt(alphas) / one_minus

        # ----- optional separate segmentation schedule -----
        self.separate_seg_schedule = separate_seg_schedule
        self.seg_schedule_scale = 2.0  # scale factor for seg betas if separate_seg_schedule is True

        if separate_seg_schedule:
            # More aggressive segmentation schedule.
            # This makes alpha_bar_seg decay faster, so p_flip approaches 0.5 earlier.
            betas_seg = torch.clamp(self.betas * float(self.seg_schedule_scale), min=1e-8, max=0.999)
        else:
            # Old behaviour: segmentation uses the same schedule as the image.
            betas_seg = self.betas.clone()

        self.betas_seg = betas_seg
        self.alphas_seg = 1.0 - self.betas_seg
        self.alphas_cumprod_seg = torch.cumprod(self.alphas_seg, dim=0)
        self.alphas_cumprod_prev_seg = torch.cat(
            [torch.ones(1, dtype=self.betas_seg.dtype), self.alphas_cumprod_seg[:-1]],
            dim=0,
        )

    # -------------------------
    # Modality helpers
    # -------------------------

    def _split_modalities(self, x):
        """Return (x_img, x_seg) with None for missing modalities."""
        if not (self.has_image or self.has_seg):
            raise ValueError("labels_channels must include 'image' and/or 'segmentation'.")

        if self.has_image and self.has_seg:
            img_idx = self.labels_channels.index("image")
            seg_idx = self.labels_channels.index("segmentation")
            x_img = x[:, img_idx:img_idx + 1, ...]
            x_seg = x[:, seg_idx:seg_idx + 1, ...]
            return x_img, x_seg

        if self.has_image:
            return x, None

        return None, x

    def _pack_modalities(self, x_img, x_seg):
        """Pack (x_img, x_seg) into channel order defined by self.labels_channels."""
        if self.has_image and self.has_seg:
            by_name = {"image": x_img, "segmentation": x_seg}
            return torch.cat([by_name[name] for name in self.labels_channels], dim=1)
        if self.has_image:
            return x_img
        return x_seg

    @staticmethod
    def xor_binary(a, b):
        """XOR for {0,1} tensors represented as float/bool."""
        return torch.abs(a.float() - b.float())

    # -------------------------
    # Forward (q) processes
    # -------------------------

    # Gaussian
    def q_sample_gaussian(self, x0, t, noise):
        """Sample x_t ~ q(x_t | x0) for Gaussian diffusion."""
        return (
            extract(self.sqrt_alphas_cumprod, t, x0.shape, x0.device) * x0
            + extract(self.sqrt_one_minus_alphas_cumprod, t, x0.shape, x0.device) * noise
        )

    def q_sample_gaussian_step(self, x_tm1, t, noise):
        """Sample x_t ~ q(x_t | x_{t-1}) for Gaussian diffusion."""
        return (
            extract(self.sqrt_alphas, t, x_tm1.shape, x_tm1.device) * x_tm1
            + extract(self.sqrt_betas, t, x_tm1.shape, x_tm1.device) * noise
        )

    def q_mean_variance_gaussian(self, x0, t):
        """Return mean/var/logvar of q(x_t|x0)."""
        mean = extract(self.sqrt_alphas_cumprod, t, x0.shape, x0.device) * x0
        var = extract(1.0 - self.alphas_cumprod, t, x0.shape, x0.device)
        logvar = extract(self.log_one_minus_alphas_cumprod, t, x0.shape, x0.device)
        return mean, var, logvar

    # Bernoulli (BerDiff reparameterization)
    def q_sample_bernoulli(self, y0, t, eps: float = 1e-6):
        """
        Sample y_t and the binary flip noise eps_t using BerDiff reparameterization:
          eps_t ~ Bernoulli((1 - abar_t)/2)
          y_t = y0 XOR eps_t
        Returns (y_t, eps_t), both in {0,1}.
        """
        y0 = y0.float()
        abar_t = extract(self.alphas_cumprod_seg, t, y0.shape, y0.device)
        p_eps = ((1.0 - abar_t) * 0.5).clamp(eps, 1.0 - eps)
        eps_t = torch.bernoulli(p_eps)
        y_t = self.xor_binary(y0, eps_t)
        return y_t, eps_t

    def q_sample_bernoulli_step(self, y_tm1, t, eps: float = 1e-6):
        """Sample y_t ~ q(y_t | y_{t-1}). Used only for visualizing gradual forward chains."""
        y_tm1 = y_tm1.float()
        alpha_t = extract(self.alphas_seg, t, y_tm1.shape, y_tm1.device)
        p_1 = alpha_t * y_tm1 + (1.0 - alpha_t) * 0.5
        return torch.bernoulli(p_1.clamp(eps, 1.0 - eps))

    def q_prob_bernoulli(self, y0, t, eps: float = 1e-6):
        """Return Bernoulli parameter P(y_t=1 | y0) under the marginal forward process."""
        y0 = y0.float()
        abar_t = extract(self.alphas_cumprod_seg, t, y0.shape, y0.device)
        p_t = abar_t * y0 + (1.0 - abar_t) * 0.5
        return p_t.clamp(eps, 1.0 - eps)

    # -------------------------
    # Reverse (p) processes
    # -------------------------

    # Gaussian helpers
    def predict_x0_from_eps(self, x_t, t, eps_hat):
        """Recover x0 estimate from eps prediction."""
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape, x_t.device) * x_t
            - extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape, x_t.device) * eps_hat
        )

    def predict_eps_from_x0(self, x_t, t, x0_hat):
        """Recover eps estimate from x0 prediction."""
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape, x_t.device) * x_t
            - x0_hat
        ) / extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape, x_t.device)

    def q_posterior_gaussian(self, x0, x_t, t):
        """Mean/var/logvar of q(x_{t-1} | x_t, x0) for Gaussian diffusion."""
        mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape, x_t.device) * x0
            + extract(self.posterior_mean_coef2, t, x_t.shape, x_t.device) * x_t
        )
        var = extract(self.posterior_variance, t, x_t.shape, x_t.device)
        logvar = extract(self.posterior_log_variance_clipped, t, x_t.shape, x_t.device)
        return mean, var, logvar

    def p_mean_variance_gaussian(self, x_t, t, eps_hat, clip_x0=True):
        """
        Compute p_theta(x_{t-1} | x_t) mean/var/logvar given predicted eps.
        """
        x0_hat = self.predict_x0_from_eps(x_t, t, eps_hat)
        if clip_x0:
            x0_hat = x0_hat.clamp(0.0, 1.0)

        mean, _, _ = self.q_posterior_gaussian(x0_hat, x_t, t)

        model_var = torch.cat([self.posterior_variance[1:2], self.betas[1:]], dim=0)
        model_logvar = torch.log(torch.clamp(model_var, min=1e-20))
        var = extract(model_var, t, x_t.shape, x_t.device)
        logvar = extract(model_logvar, t, x_t.shape, x_t.device)

        return {"mean": mean, "variance": var, "log_variance": logvar, "pred_x0": x0_hat}

    # Bernoulli posterior / model posterior
    def q_posterior_bernoulli_prob(self, y0, y_t, t, eps: float = 1e-6):
        """
        True posterior probability q(y_{t-1}=1 | y_t, y0) for Bernoulli diffusion.
        """
        alpha_t = extract(self.alphas_seg, t, y_t.shape, y_t.device)
        abar_prev = extract(self.alphas_cumprod_prev_seg, t, y_t.shape, y_t.device)

        yt = y_t.float()
        y0 = y0.float()

        yt2 = torch.stack([1.0 - yt, yt], dim=1)
        y02 = torch.stack([1.0 - y0, y0], dim=1)

        A = alpha_t.unsqueeze(1) * yt2 + (1.0 - alpha_t).unsqueeze(1) * 0.5
        B = abar_prev.unsqueeze(1) * y02 + (1.0 - abar_prev).unsqueeze(1) * 0.5

        unnorm = A * B
        denom = unnorm.sum(dim=1, keepdim=True).clamp(min=eps)
        probs2 = (unnorm / denom).clamp(eps, 1.0 - eps)
        return probs2[:, 1, ...]

    def p_bernoulli_prob_from_eps_logits(self, y_t, t, eps_logits=None, eps_prob=None, eps: float = 1e-6):
        """
        BerDiff noise-prediction posterior:
          eps_hat_prob = sigmoid(eps_logits)
          y0_hat_prob  = |y_t - eps_hat_prob|
          p_theta(y_{t-1}=1 | y_t) = q(y_{t-1}=1 | y_t, y0_hat_prob)
        """
        if eps_prob is None:
            if eps_logits is None:
                raise ValueError("Need eps_logits or eps_prob.")
            eps_prob = torch.sigmoid(eps_logits)

        eps_prob = eps_prob.clamp(eps, 1.0 - eps)
        y0_hat_prob = torch.abs(y_t.float() - eps_prob).clamp(eps, 1.0 - eps)
        prob_prev = self.q_posterior_bernoulli_prob(y0_hat_prob, y_t.float(), t, eps=eps)

        return {
            "prob": prob_prev,
            "pred_y0": y0_hat_prob,
            "pred_eps_prob": eps_prob,
            "pred_eps_logits": eps_logits,
        }

    # -------------------------
    # Training losses
    # -------------------------

    def _sample_timesteps(self, batch_size, device):
        if self.loss_weight == "none":
            t = torch.randint(0, self.num_timesteps, (batch_size,), device=device)
            weights = 1.0
            return t, weights

        p = self.weights / np.sum(self.weights)
        idx = np.random.choice(len(p), size=batch_size, p=p)
        t = torch.from_numpy(idx).long().to(device)

        # keep your original weighting scheme
        w = (1 / len(p)) * p[idx]
        weights = torch.from_numpy(w).float().to(device)
        return t, weights

    def p_loss(self, model, x0, args):
        """
        Entry-point used by your training loop.
        Returns:
          (scalar_loss, (loss_dict, x_t, model_out, t))
        """
        t, weights = self._sample_timesteps(x0.shape[0], x0.device)
        loss_dict, x_t, preds = self.calc_loss(model, x0, t, seg_weight=args["seg_weight"],model_name=args["model"])
        return ((loss_dict["loss"] * weights).mean(), (loss_dict, x_t, preds, t))

    def calc_loss(self, model, x0, t, seg_weight=1.0, model_name='DDPM'):
        """
        Compute per-modality losses and total loss.

        Image (Gaussian): MSE/L1 between eps_hat and true eps.
        Seg   (Bernoulli): BCEWithLogits between eps_logits and true eps (flip mask).
        """
        x0_img, x0_seg = self._split_modalities(x0)

        if model_name == 'DDPMCondSeg':
            if not (self.has_image and self.has_seg):
                raise ValueError("Conditional segmentation diffusion requires both image and segmentation channels.")

            x_t_seg, eps_seg = self.q_sample_bernoulli(x0_seg, t)
            x_t = self._pack_modalities(x0_img, x_t_seg)  # for previews: clean image condition + noisy mask

            eps_seg_logits = model(x_t_seg.float(), t, x0_img.float())

            lt_seg = self.loss_types[1] if (self.has_image and isinstance(self.loss_types, (list, tuple))) else (
                self.loss_types[0] if isinstance(self.loss_types, (list, tuple)) else self.loss_types
            )
            bce = torch.nn.BCEWithLogitsLoss(reduction="none")
            loss_eps_bce = mean_flat(bce(eps_seg_logits, eps_seg.float()))

            pred_eps_prob = torch.sigmoid(eps_seg_logits)
            pred_y0_prob = torch.abs(x_t_seg.float() - pred_eps_prob).clamp(1e-6, 1.0 - 1e-6)
            loss_dice = soft_dice_loss(pred_y0_prob, x0_seg)

            losses = {}
            if lt_seg == "bce":
                losses["loss_seg_bce"] = loss_eps_bce
                losses["loss_seg"] = loss_eps_bce
            elif lt_seg in ("bce_dice", "hybrid_seg_dsc"):
                losses["loss_seg_bce"] = loss_eps_bce
                losses["loss_seg_dice"] = loss_dice
                losses["loss_seg"] = loss_eps_bce + loss_dice
                if lt_seg == "hybrid_seg_dsc":
                    losses["vlb_seg"] = self.vlb_xt_bernoulli(x0_seg, x_t_seg, t, eps_seg_logits)["output"]
                    losses["loss_seg"] = losses["vlb_seg"] + losses["loss_seg"]
            elif lt_seg == "hybrid_seg":
                losses["vlb_seg"] = self.vlb_xt_bernoulli(x0_seg, x_t_seg, t, eps_seg_logits)["output"]
                losses["loss_seg"] = losses["vlb_seg"] + loss_eps_bce
            else:
                raise ValueError(f"Unknown seg loss type for conditional segmentation model: {lt_seg}")

            # There is no image reconstruction term in the conditional model.
            losses["loss"] = losses["loss_seg"]
            return losses, x_t, eps_seg_logits

        x_t_by_name = {}
        noise_img = None
        x_t_img = None
        x_t_seg = None
        eps_seg = None

        # forward sample image
        if self.has_image:
            noise_img = self.noise_fn_img(x0_img, t).float()
            x_t_img = self.q_sample_gaussian(x0_img, t, noise_img)
            x_t_by_name["image"] = x_t_img

        # forward sample seg (reparam + keep eps for supervision)
        if self.has_seg:
            x_t_seg, eps_seg = self.q_sample_bernoulli(x0_seg, t)
            x_t_by_name["segmentation"] = x_t_seg

        # pack into model input order
        x_t = torch.cat([x_t_by_name[name] for name in self.labels_channels], dim=1) if len(x_t_by_name) > 1 else list(x_t_by_name.values())[0]

        # model forward
        out = model(x_t, t)
        if self.has_image and self.has_seg:
            eps_img_hat, eps_seg_logits = out
        elif self.has_image:
            eps_img_hat, eps_seg_logits = out, None
        else:
            eps_img_hat, eps_seg_logits = None, out

        losses = {}

        # image loss
        if self.has_image:
            lt_img = self.loss_types[0] if isinstance(self.loss_types, (list, tuple)) else self.loss_types
            if lt_img == "l1":
                losses["loss_img"] = mean_flat((eps_img_hat - noise_img).abs())
            elif lt_img == "l2":
                losses["loss_img"] = mean_flat((eps_img_hat - noise_img).square())
            elif lt_img == "hybrid":
                losses["vlb_img"] = self.vlb_xt_gaussian(x0_img, x_t_img, t, eps_img_hat)["output"]
                losses["loss_img"] = losses["vlb_img"] + mean_flat((eps_img_hat - noise_img).square())
            else:
                raise ValueError(f"Unknown image loss type: {lt_img}")

        # seg loss
        if self.has_seg:
            lt_seg = self.loss_types[1] if (self.has_image and isinstance(self.loss_types, (list, tuple))) else (
                self.loss_types[0] if isinstance(self.loss_types, (list, tuple)) else self.loss_types
            )
            bce = torch.nn.BCEWithLogitsLoss(reduction="none")

            if lt_seg == "bce":
                losses["loss_seg"] = mean_flat(bce(eps_seg_logits, eps_seg.float()))
            elif lt_seg == "hybrid_seg":
                losses["vlb_seg"] = self.vlb_xt_bernoulli(x0_seg, x_t_seg, t, eps_seg_logits)["output"]
                losses["loss_seg"] = losses["vlb_seg"] + mean_flat(bce(eps_seg_logits, eps_seg.float()))
            elif lt_seg == "hybrid_seg_dsc":
                pred_eps_prob = torch.sigmoid(eps_seg_logits)
                pred_y0_prob = torch.abs(x_t_seg.float() - pred_eps_prob).clamp(1e-6, 1.0 - 1e-6)

                losses["vlb_seg"] = self.vlb_xt_bernoulli(x0_seg, x_t_seg, t, eps_seg_logits)["output"]
                losses["loss_seg"] = losses["vlb_seg"] + mean_flat(bce(eps_seg_logits, eps_seg.float())) + soft_dice_loss(pred_y0_prob, x0_seg)
            else:
                raise ValueError(f"Unknown seg loss type: {lt_seg}")

        # total
        total = 0.0
        if "loss_img" in losses:
            total = total + losses["loss_img"]
        if "loss_seg" in losses:
            total = total + (seg_weight * losses["loss_seg"] if self.has_image else losses["loss_seg"])
        losses["loss"] = total

        return losses, x_t, out

    # -------------------------
    # VLB terms
    # -------------------------

    def vlb_xt_gaussian(self, x0, x_t, t, eps_hat):
        """
        VLB term for Gaussian diffusion at timestep t (bits).
        """
        true_mean, _, true_logvar = self.q_posterior_gaussian(x0, x_t, t)
        out = self.p_mean_variance_gaussian(x_t, t, eps_hat, clip_x0=True)

        kl = normal_kl(true_mean, true_logvar, out["mean"], out["log_variance"])
        kl = mean_flat(kl) / np.log(2.0)

        decoder_nll = -discretised_gaussian_log_likelihood(x0, out["mean"], log_scales=0.5 * out["log_variance"])
        decoder_nll = mean_flat(decoder_nll) / np.log(2.0)

        nll = torch.where((t == 0), decoder_nll, kl)
        return {"output": nll, "pred_x0": out["pred_x0"]}

    def vlb_xt_bernoulli(self, y0, y_t, t, eps_logits):
        """
        VLB term for Bernoulli diffusion at timestep t (bits):
          KL(q(y_{t-1}|y_t,y0) || p_theta(y_{t-1}|y_t))   for t>0
          decoder NLL term at t==0
        """
        y0 = y0.float()
        y_t = y_t.float()

        true_prob = self.q_posterior_bernoulli_prob(y0, y_t, t)
        out = self.p_bernoulli_prob_from_eps_logits(y_t, t, eps_logits=eps_logits)

        kl = bernoulli_kl(true_prob, out["prob"])
        kl = mean_flat(kl) / np.log(2.0)

        decoder_nll = bernoulli_neg_log_likelihood(y0, out["pred_y0"])
        decoder_nll = mean_flat(decoder_nll) / np.log(2.0)

        nll = torch.where((t == 0), decoder_nll, kl)
        return {
            "output": nll,
            "pred_y0": out["pred_y0"],
            "pred_eps_prob": out["pred_eps_prob"],
            "pred_eps_logits": out["pred_eps_logits"],
        }

    # -------------------------
    # Priors / full VLB decompositions
    # -------------------------

    def prior_vlb_gaussian(self, x0_img):
        """KL(q(x_T|x0) || N(0,I)) in bits."""
        t = torch.full((x0_img.shape[0],), self.num_timesteps - 1, device=x0_img.device, dtype=torch.long)
        qt_mean, _, qt_logvar = self.q_mean_variance_gaussian(x0_img, t)
        kl = normal_kl(qt_mean, qt_logvar, torch.tensor(0.0, device=x0_img.device), torch.tensor(0.0, device=x0_img.device))
        return mean_flat(kl) / np.log(2.0)

    def prior_vlb_bernoulli(self, y0):
        """KL(q(y_T|y0) || Bernoulli(0.5)) in bits."""
        t = torch.full((y0.shape[0],), self.num_timesteps - 1, device=y0.device, dtype=torch.long)
        q_prob = self.q_prob_bernoulli(y0, t)
        prior_prob = torch.full_like(q_prob, 0.5)
        kl = bernoulli_kl(q_prob, prior_prob)
        return mean_flat(kl) / np.log(2.0)

    def calc_total_vlb_gaussian(self, x0, model, args):
        """
        Full Gaussian VLB decomposition across timesteps.
        If this DiffusionModel is mixed, pass full x0 and we will use only the image channel.
        """
        if not self.has_image:
            raise ValueError("calc_total_vlb_gaussian requires has_image=True")

        x0_img, x0_seg = self._split_modalities(x0)

        vb, x0_mse, mse = [], [], []
        for tt in reversed(range(self.num_timesteps)):
            t = torch.full((x0_img.shape[0],), tt, device=x0_img.device, dtype=torch.long)

            noise = torch.randn_like(x0_img)
            x_t_img = self.q_sample_gaussian(x0_img, t, noise)

            if self.has_seg:
                x_t_seg, _ = self.q_sample_bernoulli(x0_seg, t)
                x_t_full = self._pack_modalities(x_t_img, x_t_seg)
            else:
                x_t_full = x_t_img

            with torch.no_grad():
                out = model(x_t_full, t)
                eps_hat = out[0] if isinstance(out, (tuple, list)) else out

                vlb_t = self.vlb_xt_gaussian(x0_img, x_t_img, t, eps_hat)

            vb.append(vlb_t["output"])
            x0_mse.append(mean_flat((vlb_t["pred_x0"] - x0_img).square()))
            eps_from_x0 = self.predict_eps_from_x0(x_t_img, t, vlb_t["pred_x0"])
            mse.append(mean_flat((eps_from_x0 - noise).square()))

        vb = torch.stack(vb, dim=1)
        x0_mse = torch.stack(x0_mse, dim=1)
        mse = torch.stack(mse, dim=1)

        prior = self.prior_vlb_gaussian(x0_img)
        total = vb.sum(dim=1) + prior
        return {"total_vlb": total, "prior_vlb": prior, "vb": vb, "x0_mse": x0_mse, "mse": mse}

    def calc_total_vlb_bernoulli(self, x0, model, args=None):
        """
        Full Bernoulli VLB decomposition across timesteps.
        Works for seg-only and mixed models.
        """
        if not self.has_seg:
            raise ValueError("calc_total_vlb_bernoulli requires has_seg=True")

        x0_img, x0_seg = self._split_modalities(x0)

        vb, dsc = [], []
        for tt in reversed(range(self.num_timesteps)):
            t = torch.full((x0_seg.shape[0],), tt, device=x0_seg.device, dtype=torch.long)
            y_t, _ = self.q_sample_bernoulli(x0_seg, t)

            with torch.no_grad():
                if self.has_image:
                    if args["model"] == "DDPMCondSeg":
                        eps_logits = model(y_t.float(), t, x0_img.float())
                    else:
                        noise = torch.randn_like(x0_img)
                        x_t_img = self.q_sample_gaussian(x0_img, t, noise)
                        x_t_full = self._pack_modalities(x_t_img, y_t)
                        _, eps_logits = model(x_t_full, t)
                else:
                    eps_logits = model(y_t.float(), t)

                out = self.vlb_xt_bernoulli(x0_seg, y_t, t, eps_logits)

            vb.append(out["output"])
            dsc.append(dice_per_sample(out["pred_y0"], x0_seg))

        vb = torch.stack(vb, dim=1)
        dsc = torch.stack(dsc, dim=1)

        prior = self.prior_vlb_bernoulli(x0_seg)
        total = vb.sum(dim=1) + prior
        return {"total_vlb": total, "prior_vlb": prior, "vb": vb, "dsc": dsc}

    # -------------------------
    # Sampling (DDPM / DDIM)
    # -------------------------

    def forward_backward(
        self,
        model,
        x,
        see_whole_sequence="half",  # "whole" / "half" / None
        t_distance=None,
        denoise_fn="gauss",         # "gauss" or "noise_fn"
    ):
        """
        DDPM-style sampling (step-by-step reverse).

        Input x determines mode by channels:
          - C==2: mixed (image+seg)
          - C==1: image-only if self.has_image, else seg-only

        Returns:
          - if see_whole_sequence is None: final tensor
          - else: list of tensors over the trajectory
        """
        assert see_whole_sequence in ("whole", "half", None)

        if t_distance == 0:
            return x.detach()
        if t_distance is None:
            t_distance = self.num_timesteps

        B, C = x.shape[0], x.shape[1]
        device = x.device

        mixed_input = (C == 2)
        image_only_input = (C == 1) and self.has_image
        seg_only_input = (C == 1) and (self.has_seg and not self.has_image)

        if not (mixed_input or image_only_input or seg_only_input):
            raise ValueError(
                f"forward_backward got x with {C} channels, but has_image={self.has_image}, has_seg={self.has_seg}."
            )

        seq = [x.detach().cpu()]

        # ---- forward ----
        if see_whole_sequence == "whole":
            if mixed_input:
                x_img, x_seg = self._split_modalities(x)
                xt_img, xt_seg = x_img, x_seg
                for tt in range(int(t_distance)):
                    t = torch.full((B,), tt, device=device, dtype=torch.long)
                    xt_img = self.q_sample_gaussian_step(xt_img, t, self.noise_fn_img(xt_img, t).float())
                    xt_seg = self.q_sample_bernoulli_step(xt_seg, t)
                    seq.append(self._pack_modalities(xt_img, xt_seg).detach().cpu())

            elif image_only_input:
                xt = x
                for tt in range(int(t_distance)):
                    t = torch.full((B,), tt, device=device, dtype=torch.long)
                    xt = self.q_sample_gaussian_step(xt, t, self.noise_fn_img(xt, t).float())
                    seq.append(xt.detach().cpu())

            else:
                xt = x
                for tt in range(int(t_distance)):
                    t = torch.full((B,), tt, device=device, dtype=torch.long)
                    xt = self.q_sample_bernoulli_step(xt, t)
                    seq.append(xt.detach().cpu())

        else:
            t = torch.full((B,), int(t_distance) - 1, device=device, dtype=torch.long)
            if mixed_input:
                x_img, x_seg = self._split_modalities(x)
                xt_img = self.q_sample_gaussian(x_img, t, self.noise_fn_img(x_img, t).float())
                xt_seg, _ = self.q_sample_bernoulli(x_seg, t)
                xt = self._pack_modalities(xt_img, xt_seg)
            elif image_only_input:
                xt = self.q_sample_gaussian(x, t, self.noise_fn_img(x, t).float())
            else:
                xt, _ = self.q_sample_bernoulli(x, t)

            if see_whole_sequence == "half":
                seq.append(xt.detach().cpu())

        # ---- reverse ----
        x_curr = xt
        for tt in range(int(t_distance) - 1, -1, -1):
            t = torch.full((B,), tt, device=device, dtype=torch.long)

            if mixed_input:
                x_img, x_seg = self._split_modalities(x_curr)
                with torch.no_grad():
                    eps_img, eps_seg_logits = model(x_curr, t)

                    out_img = self.p_mean_variance_gaussian(x_img, t, eps_img, clip_x0=True)

                    if denoise_fn == "gauss":
                        noise = torch.randn_like(x_img)
                    elif denoise_fn == "noise_fn":
                        noise = self.noise_fn_img(x_img, t).float()
                    else:
                        raise ValueError(f"Unknown denoise_fn: {denoise_fn}")

                    nonzero = (t != 0).float().view(-1, *([1] * (x_img.ndim - 1)))
                    x_img_prev = out_img["mean"] + nonzero * torch.exp(0.5 * out_img["log_variance"]) * noise

                    out_seg = self.p_bernoulli_prob_from_eps_logits(x_seg.float(), t, eps_logits=eps_seg_logits)
                    x_seg_prev = torch.bernoulli(out_seg["prob"])

                    x_curr = self._pack_modalities(x_img_prev, x_seg_prev)

            elif image_only_input:
                with torch.no_grad():
                    if self.has_seg:
                        dummy_seg = torch.zeros_like(x_curr)
                        x_full = self._pack_modalities(x_curr, dummy_seg)
                        out = model(x_full, t)
                        eps_hat = out[0] if isinstance(out, (tuple, list)) else out
                    else:
                        out = model(x_curr, t)
                        eps_hat = out[0] if isinstance(out, (tuple, list)) else out

                    out_img = self.p_mean_variance_gaussian(x_curr, t, eps_hat, clip_x0=True)

                    if denoise_fn == "gauss":
                        noise = torch.randn_like(x_curr)
                    elif denoise_fn == "noise_fn":
                        noise = self.noise_fn_img(x_curr, t).float()
                    else:
                        raise ValueError(f"Unknown denoise_fn: {denoise_fn}")

                    nonzero = (t != 0).float().view(-1, *([1] * (x_curr.ndim - 1)))
                    x_curr = out_img["mean"] + nonzero * torch.exp(0.5 * out_img["log_variance"]) * noise

            else:
                with torch.no_grad():
                    eps_logits = model(x_curr.float(), t)
                    out_seg = self.p_bernoulli_prob_from_eps_logits(x_curr.float(), t, eps_logits=eps_logits)
                    x_curr = torch.bernoulli(out_seg["prob"])

            if see_whole_sequence is not None:
                seq.append(x_curr.detach().cpu())

        return x_curr.detach() if see_whole_sequence is None else seq

    # ---- DDIM helpers ----
    def forward_backward_ddim_conditional_seg(
        self,
        model,
        x,
        see_whole_sequence="half",
        t_distance=None,
        ddim_steps=50,
    ):
        """
        DDIM-style sampling for image-conditioned segmentation diffusion.

        x can be either:
          - full tensor [B,2,D,H,W] containing image + segmentation, useful for
            reconstruction previews because the starting y_t is made from the
            provided clean mask; or
          - image tensor [B,1,D,H,W], useful for inference because the mask is
            initialized from Bernoulli(0.5).

        Returns a sequence of packed [image, segmentation] tensors when
        see_whole_sequence is not None, otherwise returns the final packed tensor.
        """
        assert see_whole_sequence in ("whole", "half", None)

        if t_distance == 0:
            return x.detach()
        if t_distance is None:
            t_distance = self.num_timesteps

        B = x.shape[0]
        device = x.device
        t_start = int(t_distance) - 1
        t0 = torch.full((B,), t_start, device=device, dtype=torch.long)

        if x.shape[1] == 2 and self.has_image and self.has_seg:
            x_img, x_seg = self._split_modalities(x)
            y_curr, _ = self.q_sample_bernoulli(x_seg, t0)
        elif x.shape[1] == 1:
            x_img = x
            y_curr = torch.bernoulli(torch.full_like(x_img, 0.5))
        else:
            raise ValueError(
                "forward_backward_ddim_conditional_seg expects [B,2,D,H,W] full input "
                "or [B,1,D,H,W] image-only input."
            )

        seq = []
        if see_whole_sequence is not None:
            seq.append(self._pack_modalities(x_img, y_curr).detach().cpu())

        ddim_ts = self._make_ddim_timesteps(t_distance=t_distance, ddim_steps=ddim_steps, device=device)

        for i in range(len(ddim_ts) - 1):
            tt = ddim_ts[i].item()
            tt_prev = ddim_ts[i + 1].item()
            t = torch.full((B,), tt, device=device, dtype=torch.long)
            t_prev = torch.full((B,), tt_prev, device=device, dtype=torch.long)

            with torch.no_grad():
                eps_logits = model(y_curr.float(), t, x_img.float())
                eps_prob = torch.sigmoid(eps_logits)
                p_prev = self.bernoulli_ddim_prob(y_curr, t, t_prev, eps_prob)
                y_curr = torch.bernoulli(p_prev)

            if see_whole_sequence is not None:
                seq.append(self._pack_modalities(x_img, y_curr).detach().cpu())

        final = self._pack_modalities(x_img, y_curr)
        
        return final.detach() if see_whole_sequence is None else seq, ddim_ts.cpu().numpy()

    @staticmethod
    def _make_ddim_timesteps(t_distance: int, ddim_steps: int, device):
        """Return decreasing timesteps [t_start, ..., 0] as a 1-D long tensor."""
        ddim_steps = int(ddim_steps)
        t_distance = int(t_distance)
        if ddim_steps < 2:
            raise ValueError("ddim_steps must be >= 2")

        ts = np.linspace(0, t_distance - 1, ddim_steps, dtype=np.int64)
        ts = np.unique(ts)
        if ts[-1] != (t_distance - 1):
            ts = np.concatenate([ts, [t_distance - 1]])
        if ts[0] != 0:
            ts = np.concatenate([[0], ts])

        ts = ts[::-1].copy()
        return torch.from_numpy(ts).to(device=device, dtype=torch.long)

    def ddim_step_gaussian(self, x_t, t, t_prev, eps_hat, eta: float = 0.0, clip_x0: bool = True):
        """
        One Gaussian DDIM step: x_t -> x_{t_prev}.

        eta controls stochasticity (eta=0 => deterministic).
        """
        a_t = extract(self.alphas_cumprod, t, x_t.shape, x_t.device)
        a_prev = extract(self.alphas_cumprod, t_prev, x_t.shape, x_t.device)

        sqrt_a_t = torch.sqrt(torch.clamp(a_t, min=1e-20))
        sqrt_one_minus_a_t = torch.sqrt(torch.clamp(1.0 - a_t, min=1e-20))

        x0 = (x_t - sqrt_one_minus_a_t * eps_hat) / sqrt_a_t
        if clip_x0:
            x0 = x0.clamp(0.0, 1.0)

        frac = torch.clamp(a_t / torch.clamp(a_prev, min=1e-20), min=0.0, max=1.0)
        sigma = (
            eta
            * torch.sqrt(torch.clamp((1.0 - a_prev) / torch.clamp(1.0 - a_t, min=1e-20), min=0.0))
            * torch.sqrt(torch.clamp(1.0 - frac, min=0.0))
        )
        c = torch.sqrt(torch.clamp(1.0 - a_prev - sigma * sigma, min=0.0))

        z = torch.randn_like(x_t) if eta > 0 else torch.zeros_like(x_t)
        x_prev = torch.sqrt(torch.clamp(a_prev, min=1e-20)) * x0 + c * eps_hat + sigma * z
        return x_prev, x0

    def bernoulli_ddim_prob(self, y_t, t, t_prev, pred_eps_prob, eps: float = 1e-6):
        """
        BerDiff DDIM probability for y_{t_prev} given y_t and eps_hat_prob.

        Uses sigma = (1 - abar_prev) / (1 - abar_t) (BerDiff implementation),
        and the calibrated y0_hat = |y_t - eps_hat_prob|.
        """
        yt = y_t.float()
        pred_eps_prob = pred_eps_prob.clamp(eps, 1.0 - eps)

        abar_t = extract(self.alphas_cumprod_seg, t, yt.shape, yt.device)
        abar_prev = extract(self.alphas_cumprod_seg, t_prev, yt.shape, yt.device)

        sigma = (1.0 - abar_prev) / torch.clamp(1.0 - abar_t, min=eps)
        y0_hat = torch.abs(yt - pred_eps_prob)

        prob = (
            sigma * yt
            + (abar_prev - sigma * abar_t) * y0_hat
            + (((1.0 - abar_prev) - (1.0 - abar_t) * sigma) * 0.5)
        )
        return prob.clamp(eps, 1.0 - eps)

    def forward_backward_ddim(
        self,
        model,
        x,
        see_whole_sequence="half",
        t_distance=None,
        ddim_steps=50,
        eta=0.0,
    ):
        """
        DDIM sampling:
          - Gaussian image uses DDIM subsequence
          - Bernoulli segmentation uses BerDiff DDIM probability on the same subsequence
        """
        assert see_whole_sequence in ("whole", "half", None)

        if t_distance == 0:
            return x.detach()
        if t_distance is None:
            t_distance = self.num_timesteps

        B, C = x.shape[0], x.shape[1]
        device = x.device

        mixed_input = (C == 2)
        image_only_input = (C == 1) and self.has_image
        seg_only_input = (C == 1) and (self.has_seg and not self.has_image)

        if not (mixed_input or image_only_input or seg_only_input):
            raise ValueError(
                f"forward_backward_ddim got x with {C} channels, but has_image={self.has_image}, has_seg={self.has_seg}."
            )

        seq = []

        # ---- forward jump to t_start ----
        t_start = int(t_distance) - 1
        t0 = torch.full((B,), t_start, device=device, dtype=torch.long)

        if mixed_input:
            x_img, x_seg = self._split_modalities(x)
            xt_img = self.q_sample_gaussian(x_img, t0, self.noise_fn_img(x_img, t0).float())
            xt_seg, _ = self.q_sample_bernoulli(x_seg, t0)
            x_curr = self._pack_modalities(xt_img, xt_seg)
        elif image_only_input:
            x_curr = self.q_sample_gaussian(x, t0, self.noise_fn_img(x, t0).float())
        else:
            x_curr, _ = self.q_sample_bernoulli(x, t0)

        # ---- reverse on subsequence ----
        ddim_ts = self._make_ddim_timesteps(t_distance=t_distance, ddim_steps=ddim_steps, device=device)

        if seg_only_input:
            for i in range(len(ddim_ts) - 1):
                tt = ddim_ts[i].item()
                tt_prev = ddim_ts[i + 1].item()
                t = torch.full((B,), tt, device=device, dtype=torch.long)
                t_prev = torch.full((B,), tt_prev, device=device, dtype=torch.long)

                with torch.no_grad():
                    eps_logits = model(x_curr.float(), t)
                    eps_prob = torch.sigmoid(eps_logits)
                    p_prev = self.bernoulli_ddim_prob(x_curr, t, t_prev, eps_prob)
                    x_curr = torch.bernoulli(p_prev)

                if see_whole_sequence is not None:
                    seq.append(x_curr.detach().cpu())

            return x_curr.detach() if see_whole_sequence is None else seq

        for i in range(len(ddim_ts) - 1):
            tt = ddim_ts[i].item()
            tt_prev = ddim_ts[i + 1].item()
            t = torch.full((B,), tt, device=device, dtype=torch.long)
            t_prev = torch.full((B,), tt_prev, device=device, dtype=torch.long)

            if mixed_input:
                x_img, x_seg = self._split_modalities(x_curr)
                with torch.no_grad():
                    eps_img, eps_seg_logits = model(x_curr, t)

                    x_img_prev, _ = self.ddim_step_gaussian(x_img, t, t_prev, eps_img, eta=float(eta), clip_x0=True)

                    eps_prob = torch.sigmoid(eps_seg_logits)
                    p_seg_prev = self.bernoulli_ddim_prob(x_seg, t, t_prev, eps_prob)
                    x_seg_prev = torch.bernoulli(p_seg_prev)

                    x_curr = self._pack_modalities(x_img_prev, x_seg_prev)

            else:
                with torch.no_grad():
                    if self.has_seg:
                        dummy_seg = torch.zeros_like(x_curr)
                        x_full = self._pack_modalities(x_curr, dummy_seg)
                        out = model(x_full, t)
                        eps_hat = out[0] if isinstance(out, (tuple, list)) else out
                    else:
                        out = model(x_curr, t)
                        eps_hat = out[0] if isinstance(out, (tuple, list)) else out

                    x_curr, _ = self.ddim_step_gaussian(x_curr, t, t_prev, eps_hat, eta=float(eta), clip_x0=True)

            if see_whole_sequence is not None:
                seq.append(x_curr.detach().cpu())

        seq.append(x.detach().cpu())

        return x_curr.detach() if see_whole_sequence is None else seq, ddim_ts.cpu().numpy()