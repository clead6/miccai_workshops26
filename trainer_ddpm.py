import collections
import copy
import sys
import os
import time
from tqdm import trange

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import torch
from torch import optim
from torch.amp import autocast, GradScaler

from UNet3D import UNetModel, ConditionalSegUNetModel, update_ema_params
from Diffusion import DiffusionModel, get_beta_schedule
from helpers import save, plot_loss_history

def trainer_ddpm(training_dataset_loader, validation_dataset_loader, args, device):
    """
    Trainer
    """
    
    # --- config knobs (with safe defaults) ---
    use_amp = bool(args.get("amp", True))
    amp_dtype_str = str(args.get("amp_dtype", "bf16")).lower()
    amp_dtype = torch.bfloat16 if amp_dtype_str == "bf16" else torch.float16

    patience = int(args.get("patience", 10))

    if args['model'] == 'DDPM':
        model = UNetModel(
            args['patch_size'][0],
            args['init_channels'],
            channel_mults=args['channel_mults'],
            dropout=args["dropout"],
            n_heads=args["num_heads"],
            n_head_channels=args.get("num_head_channels", -1),
            in_channels=args['num_channels']
        )
    elif args['model'] == 'DDPMCondSeg':
        model = ConditionalSegUNetModel(
            args['patch_size'][0],
            args['init_channels'],
            channel_mults=args['channel_mults'],
            dropout=args["dropout"],
            n_heads=args["num_heads"],
            n_head_channels=args.get("num_head_channels", -1),
        )
    else:
        raise ValueError(f"Invalid model type {args['model']}. Must be 'DDPM' or 'DDPMCondSeg'.")

    betas = get_beta_schedule(args['num_diffusion_steps'], args['beta_schedule'], args['beta_schedule_gamma'])

    diffusion = DiffusionModel(
        args['patch_size'],
        betas,
        loss_weight=args['loss_weight'],
        loss_types=args['loss_types'],
        noise=args["noise_fn"],
        in_channels=args['num_channels'],
        labels_channels=args['labels_channels'],
        separate_seg_schedule=args['separate_seg_schedule'],
    )

    if args['load_model'] is not None:
        checkpoint = torch.load(args['load_model'], map_location=device)

        # --- robust key handling (supports older checkpoints too) ---
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        elif "unet" in checkpoint:
            model.load_state_dict(checkpoint["unet"])
        elif "ema" in checkpoint:
            # fall back: sometimes only ema was saved
            model.load_state_dict(checkpoint["ema"])
        else:
            raise KeyError(f"Unknown checkpoint keys: {list(checkpoint.keys())}")
        
        if args['model'] == 'DDPM':
            ema = UNetModel(
                args['patch_size'][0],
                args['init_channels'],
                channel_mults=args['channel_mults'],
                dropout=args["dropout"],
                n_heads=args["num_heads"],
                n_head_channels=args.get("num_head_channels", -1),
                in_channels=args['num_channels']
            )
        elif args['model'] == 'DDPMCondSeg':
            ema = ConditionalSegUNetModel(
                args['patch_size'][0],
                args['init_channels'],
                channel_mults=args['channel_mults'],
                dropout=args["dropout"],
                n_heads=args["num_heads"],
                n_head_channels=args.get("num_head_channels", -1),
            )
        else:
            raise ValueError(f"Invalid model type {args['model']}. Must be 'DDPM' or 'DDPMCondSeg'.")

        if "ema" in checkpoint:
            ema.load_state_dict(checkpoint["ema"])
        else:
            # if no EMA in ckpt, start EMA from loaded model
            ema.load_state_dict(model.state_dict())

        print(f"Loaded model from {args['load_model']}")
    else:
        ema = copy.deepcopy(model)

    # tqdm_epoch = range(start_epoch, args['epochs'] + 1)
    tqdm_epoch = trange(args['epochs'], leave=False, desc="epoch: 0")
    model.to(device)
    ema.to(device)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=args['learning_rate'],
        weight_decay=args['weight_decay'],
        betas=(0.9, 0.999)
    )

    # GradScaler only needed/used for fp16; disabled for bf16
    use_scaler = use_amp and (amp_dtype is torch.float16)
    scaler = GradScaler(enabled=use_scaler)

    start_time = time.time()
    vlb = collections.deque([], maxlen=10)

    # ---- loss component histories (epoch means) ----
    loss_hist = {
        "train_total": [],
        "train_img": [],
        "train_seg": [],
        "val_total": [],
    }

    # --- early stopping state ---
    best_val = float("inf")
    best_epoch = -1
    epochs_no_improve = 0
    best_model_state = None
    best_ema_state = None
    best_optim_state = None

    # dataset loop
    for epoch in tqdm_epoch:
        model.train()
        mean_loss = []
        mean_loss_img = []
        mean_loss_seg = []

        for i, data in enumerate(training_dataset_loader):
            x = data["im"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            # ---- forward with autocast ----
            with autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
                loss, estimates = diffusion.p_loss(model, x, args) 

            # ---- backward (scaled if fp16) ----
            if use_scaler:
                scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            # ---- EMA ----
            update_ema_params(ema, model)

            # estimates[0] is the loss dict from calc_loss
            loss_dict = estimates[0]

            # total (always present)
            mean_loss.append(loss_dict["loss"].detach().float().mean().cpu())

            if "loss_img" in loss_dict:
                mean_loss_img.append(loss_dict["loss_img"].detach().float().mean().cpu())

            if "loss_seg" in loss_dict:
                mean_loss_seg.append(loss_dict["loss_seg"].detach().float().mean().cpu())

            # ---- samples/preview ----
            if epoch % 50 == 0 and i == 0:
                row_size = min(8, x.shape[0])
                noisy, est, t = estimates[1], estimates[2], estimates[3]

                # use EMA for preview; no grad, can keep autocast for speed
                ema.eval()
                with torch.no_grad(), autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
                    training_outputs(
                        diffusion, x, est, noisy, t, epoch, row_size,
                        ema=ema, args=args,
                        device=device,
                        save_imgs=args['save_imgs'],
                        save_vids=args['save_vids']                    
                    )

        # ---- epoch logging ----
        train_total_epoch = float(torch.stack(mean_loss).mean().item()) if len(mean_loss) else float("nan")
        loss_hist["train_total"].append(train_total_epoch)

        if len(mean_loss_img):
            loss_hist["train_img"].append(float(torch.stack(mean_loss_img).mean().item()))
        else:
            loss_hist["train_img"].append(np.nan)

        if len(mean_loss_seg):
            loss_hist["train_seg"].append(float(torch.stack(mean_loss_seg).mean().item()))
        else:
            loss_hist["train_seg"].append(np.nan)

        # ---- compute validation loss with EMA (early stopping metric) ----
        ema.eval()
        val_losses = []
        with torch.no_grad(), autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
            for i, data in enumerate(validation_dataset_loader):
                vdata = data["im"]
                vx = vdata.to(device, non_blocking=True)
                vloss, _ = diffusion.p_loss(ema, vx, args)  # eval EMA for stability
                val_losses.append(float(vloss.detach().float().cpu()))
        val_loss = float(np.mean(val_losses))
        loss_hist["val_total"].append(val_loss)

        # ---- early stopping bookkeeping ----
        improved = val_loss < best_val - 1e-8  # tiny tolerance
        if improved:
            best_val = val_loss
            best_epoch = epoch
            epochs_no_improve = 0
            # snapshot best states
            best_model_state = copy.deepcopy(model.state_dict())
            best_ema_state = copy.deepcopy(ema.state_dict())
            best_optim_state = copy.deepcopy(optimizer.state_dict())
            # save "best" checkpoint immediately
            save(model=model, args=args, optimizer=optimizer, final=False, ema=ema, epoch=epoch)
        else:
            epochs_no_improve += 1

        if (epoch % 50 == 0 and epoch > 0):
            time_taken = time.time() - start_time

            ema.eval()
            with torch.no_grad(), autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
                # --- IMAGE (Gaussian) ---
                vlb_img_mean = None
                x0_mse_mean = None

                if diffusion.has_image and args['model'] == 'DDPM':
                    vlb_terms_img = diffusion.calc_total_vlb_gaussian(vx, ema, args)
                    vlb_img_mean = float(vlb_terms_img["total_vlb"].mean().detach().cpu())
                    x0_mse_mean = float(vlb_terms_img["x0_mse"].mean().detach().cpu())

                # --- SEG (Bernoulli) ---
                vlb_seg_mean = None
                dsc_mean = None

                if diffusion.has_seg:
                    vlb_terms_seg = diffusion.calc_total_vlb_bernoulli(vx, ema, args)
                    vlb_seg_mean = float(vlb_terms_seg["total_vlb"].mean().detach().cpu())
                    dsc_mean = float(vlb_terms_seg["dsc"].mean().detach().cpu())

            # Keep a single scalar history in `vlb` for display:
            # - image-only: use image VLB
            # - seg-only: use seg VLB
            # - mixed: use sum (so it's comparable epoch-to-epoch)
            if args['model'] == 'DDPMCondSeg':
                vlb_scalar = vlb_seg_mean
            elif diffusion.has_image and diffusion.has_seg: 
                vlb_scalar = vlb_img_mean + vlb_seg_mean 
            elif diffusion.has_image: 
                vlb_scalar = vlb_img_mean 
            else: 
                vlb_scalar = vlb_seg_mean

            vlb.append(vlb_scalar)
            mean_vlb_hist = float(np.mean(vlb)) if len(vlb) > 0 else float(vlb_scalar)

            # Build a readable status line depending on modality
            parts = [
                f"epoch: {epoch}",
                f"mean VLB {mean_vlb_hist:.4f}",
            ]

            if diffusion.has_image and args['model'] == 'DDPM':
                parts.append(f"mse img {x0_mse_mean:.6f}")

            if diffusion.has_seg:
                parts.append(f"dsc seg {dsc_mean:.4f}")

            total_train_loss = loss_hist["train_total"][-1]
            parts.append(f"mean loss {total_train_loss:.4f}")
            parts.append(f"time elapsed {int(time_taken / 3600)}:{((time_taken / 3600) % 1) * 60:02.0f}")

            tqdm_epoch.set_description(", ".join(parts))

        else:
            time_taken = time.time() - start_time
            total_train_loss = loss_hist["train_total"][-1]
            tqdm_epoch.set_description(
                f"epoch: {epoch}, train loss: {total_train_loss:.4f}, val loss: {val_loss:.4f}, "
                f"time elapsed {int(time_taken / 3600)}:{((time_taken / 3600) % 1) * 60:02.0f}"
            )

        # ---- check early stopping ----
        if epochs_no_improve >= patience:
            print(f"\nEarly stopping at epoch {epoch} (best val {best_val:.4f} @ {best_epoch}).")
            # restore best weights before exiting the loop
            if best_model_state is not None:
                model.load_state_dict(best_model_state)
            if best_ema_state is not None:
                ema.load_state_dict(best_ema_state)
            if best_optim_state is not None:
                optimizer.load_state_dict(best_optim_state)
            break  # exit epoch loop

    save(model=model, args=args, optimizer=optimizer, final=True, ema=ema)
    plot_loss_history(loss_hist, args)
    np.save(os.path.join(args["path_save"], "model", "loss_history.npy"), loss_hist)

def training_outputs(diffusion, x, est, noisy, t, epoch, row_size, ema, args, device, save_imgs=False, save_vids=False):
    """
    Training preview for 3 cases:
      1) image-only: Gaussian diffusion, model predicts eps
      2) seg-only: Bernoulli diffusion, model predicts x0 logits (prob via sigmoid)
      3) image+seg: model returns (eps_img, seg_logits)

    Notes:
      - `x` is x0 (B,C,D,H,W)
      - `noisy` is x_t as produced by diffusion.calc_loss (B,C,D,H,W)
      - `est` is model output from diffusion.calc_loss: either tensor or (eps_img, seg_logits)
    """
    if not (save_imgs or save_vids):
        return

    os.makedirs(os.path.join(args["path_save"], "training-images"), exist_ok=True)
    os.makedirs(os.path.join(args["path_save"], "training-gifs"), exist_ok=True)

    index_slice = x.shape[2] // 2
    row_size = min(8, x.shape[0])

    # ----- unpack modalities + ordering -----
    has_image = diffusion.has_image
    has_seg = diffusion.has_seg
    labels = diffusion.labels_channels  # e.g. ['image','segmentation'] or swapped

    def chan_idx(name: str) -> int:
        return labels.index(name)

    img_idx = chan_idx("image") if has_image else None
    seg_idx = chan_idx("segmentation") if has_seg else None

    # x0 per modality (shape [B,1,D,H,W])
    x0_img = x[:, img_idx:img_idx+1] if has_image else None
    x0_seg = x[:, seg_idx:seg_idx+1] if has_seg else None

    # xt per modality from "noisy"
    xt_img = noisy[:, img_idx:img_idx+1] if has_image else None
    xt_seg = noisy[:, seg_idx:seg_idx+1] if has_seg else None

    # unpack model outputs (est)
    if args['model'] == 'DDPMCondSeg':
        eps_img, seg_logits = None, est
    elif has_image and has_seg:
        eps_img, seg_logits = est
    elif has_image:
        eps_img, seg_logits = est, None
    else:
        eps_img, seg_logits = None, est

    # =========================
    # IMAGES (save_imgs)
    # =========================
    if save_imgs:
        # ---------- IMAGE MODALITY (Gaussian) ----------
        if has_image and args['model'] == 'DDPM':
            pred_x0_img = diffusion.predict_x0_from_eps(xt_img, t, eps_img).clamp(0, 1)
            xt_img = xt_img.clamp(0, 1)

            # panel: real / sample(mean) / pred_x0
            out = torch.cat((
                x0_img[:row_size, 0, index_slice].detach().cpu(),
                xt_img[:row_size, 0, index_slice].detach().cpu(),
                pred_x0_img[:row_size, 0, index_slice].detach().cpu(),
            ), dim=0)  

            out = torch.rot90(out, k=1, dims=(1, 2))
            grid = stack_to_grid(out, n_rows=3)

            save_panel(
                grid, cmap="gray", title=f"IMAGE diffusion (epoch {epoch}) t={t[0].item()}", 
                path=os.path.join(args["path_save"], "training-images", f"image_epoch{epoch}.png"), 
                row_labels=["x₀", "xₜ", "pred x̂₀"],
                H=out.shape[-2], W=out.shape[-1], row_size=row_size,
            )

        # ---------- SEG MODALITY (Bernoulli) ----------
        if has_seg:
            # seg_logits here are eps logits
            eps_prob = torch.sigmoid(seg_logits).detach()                 # \hat eps in [0,1]
            x0_hat_prob = torch.abs(xt_seg.float() - eps_prob).clamp(1e-6, 1.0 - 1e-6)
            x0_hat_bin = (x0_hat_prob >= 0.5).float()

            out = torch.cat((
                x0_seg[:row_size, 0, index_slice].detach().cpu(),         # x0 (GT)
                xt_seg[:row_size, 0, index_slice].detach().cpu(),         # xt (noisy)
                x0_hat_bin[:row_size, 0, index_slice].detach().cpu(),     # x0 hat (bin)
            ), dim=0)

            out = torch.rot90(out, k=1, dims=(1, 2))
            grid = stack_to_grid(out, n_rows=3)

            save_panel(
                grid, cmap="Reds", title=f"SEG diffusion (epoch {epoch}) t={t[0].item()}",
                path=os.path.join(args["path_save"], "training-images", f"segmentation_epoch{epoch}.png"),
                row_labels=["x₀", "xₜ", "binary x̂₀"],
                H=out.shape[-2], W=out.shape[-1], row_size=row_size,
            )

    # =========================
    # VIDEOS (save_vids)
    # =========================
    if save_vids:
        plt.rcParams["figure.dpi"] = 200

        # -------------------------
        # shared video config
        # -------------------------
        length = args["num_diffusion_steps"] // (20 if args.get("debug", False) else 4)
        ddim_steps = int(max(10, args.get("ddim_steps", 50)))
        ddim_eta = float(args.get("ddim_eta", 0.0))  # used only for Gaussian DDIM in your implementation
        frame_stride = int(args.get("gif_stride", 10))

        index_slice = x.shape[2] // 2

        def _steps_for_bounce(num_frames: int, t_distance: int):
            # produces the same "forward then backward" labeling you were using
            steps = []
            for k in range(0, num_frames * frame_stride, frame_stride):
                if k <= t_distance:
                    steps.append(k)
                else:
                    steps.append(2 * t_distance - k)
            return steps

        def _make_gif_from_seq(seq_tbdhw, out_path, cmap, title, vmin=None, vmax=None):
            """
            seq_tbdhw: Tensor [T, B, 1, D, H, W] (already on CPU or GPU; we detach/cpu inside)
            """
            seq = seq_tbdhw.detach().cpu().clamp(0, 1)  # safe for display

            frames = []
            for k in range(0, seq.shape[0], frame_stride):
                fr = seq[k, :row_size, 0, index_slice]         # [row_size,H,W]
                fr = torch.rot90(fr, k=1, dims=(1, 2))
                frames.append(fr)

            frames_rows = [f.reshape(row_size, f.shape[-2], f.shape[-1]) for f in frames]
            steps = _steps_for_bounce(num_frames=len(frames), t_distance=length)

            save_grid_gif(
                frames_rows=frames_rows,
                steps=steps,
                n_rows=1,
                row_size=row_size,
                path=out_path,
                cmap=cmap,
                title=title,
                interval=200,
                vmin=vmin,
                vmax=vmax,
            )

        # image + segmentation DDIM
        if has_image and has_seg:
            x0_full = x[:row_size].to(device)

            if args["model"] == "DDPMCondSeg":
                out_list, _ = diffusion.forward_backward_ddim_conditional_seg(
                    ema,
                    x0_full,
                    see_whole_sequence="whole",
                    t_distance=length,
                    ddim_steps=ddim_steps,
                )
            else:
                out_list, _ = diffusion.forward_backward_ddim(
                    ema,
                    x0_full,
                    see_whole_sequence="whole",
                    t_distance=length,
                    ddim_steps=ddim_steps,
                    eta=ddim_eta,
                )

            out_seq = torch.stack(out_list, dim=0)  # [T,B,2,D,H,W]

            img_idx = diffusion.labels_channels.index("image")
            seg_idx = diffusion.labels_channels.index("segmentation")

            img_seq = out_seq[:, :, img_idx:img_idx+1]  # [T,B,1,D,H,W]
            seg_seq = out_seq[:, :, seg_idx:seg_idx+1]  # [T,B,1,D,H,W]

            _make_gif_from_seq(
                img_seq,
                os.path.join(args["path_save"], "training-gifs", f"image_epoch{epoch}.gif"),
                cmap="gray",
                title=f"IMAGE DDIM (epoch {epoch})  |  columns = samples",
                vmin=0.0, vmax=1.0,
            )
            _make_gif_from_seq(
                seg_seq,
                os.path.join(args["path_save"], "training-gifs", f"seg_epoch{epoch}.gif"),
                cmap="Reds",
                title=f"SEG DDIM (epoch {epoch})  |  columns = samples",
                vmin=0.0, vmax=1.0,
            )

        else:
            # image-only DDIM
            if has_image:
                if args['model'] == 'DDPMCondSeg':
                    out_list, _ = diffusion.forward_backward_ddim_conditional_seg(
                        ema,
                        x0_full,
                        see_whole_sequence="whole",
                        t_distance=length,
                        ddim_steps=ddim_steps,
                    )
                else:
                    out_list, _ = diffusion.forward_backward_ddim(
                        ema,
                        x0_full,
                        see_whole_sequence="whole",
                        t_distance=length,
                        ddim_steps=ddim_steps,
                        eta=ddim_eta,
                    )

                img_seq = torch.stack(out_list, dim=0)  # [T,B,1,D,H,W]

                _make_gif_from_seq(
                    img_seq,
                    os.path.join(args["path_save"], "training-gifs", f"image_epoch{epoch}.gif"),
                    cmap="gray",
                    title=f"IMAGE DDIM (epoch {epoch})  |  columns = samples",
                    vmin=0.0, vmax=1.0,
                )

            # seg-only DDIM
            if has_seg:
                out_list, _ = diffusion.forward_backward_ddim(
                    ema,
                    x0_seg[:row_size].to(device),  # [B,1,D,H,W]
                    see_whole_sequence="whole",
                    t_distance=length,
                    ddim_steps=ddim_steps,
                    eta=ddim_eta,
                )
                seg_seq = torch.stack(out_list, dim=0)  # [T,B,1,D,H,W]

                _make_gif_from_seq(
                    seg_seq,
                    os.path.join(args["path_save"], "training-gifs", f"seg_epoch{epoch}.gif"),
                    cmap="Reds",
                    title=f"SEG DDIM (epoch {epoch})  |  columns = samples",
                    vmin=0.0, vmax=1.0,
                )

# ----- helpers for plotting -----
def stack_to_grid(out_hw, n_rows):
    """
    out_hw: tensor [K, H, W] where K = n_rows * n_cols
    returns: [n_rows*H, n_cols*W]
    """
    K, H, W = out_hw.shape
    assert K % n_rows == 0, f"K={K} must be divisible by n_rows={n_rows}"
    n_cols = K // n_rows
    out_hw = out_hw.view(n_rows, n_cols, H, W)          # [R, C, H, W]
    out_hw = out_hw.permute(0, 2, 1, 3).contiguous()    # [R, H, C, W]
    out_hw = out_hw.view(n_rows * H, n_cols * W)        # [R*H, C*W]
    return out_hw

def save_panel(tensor_2d, cmap, title, path, row_labels=None, H=None, W=None, row_size=1):
    plt.figure(figsize=(2*row_size, 4))
    plt.imshow(tensor_2d, cmap=cmap)
    plt.axis("off")
    plt.title(title, fontsize=12)

    # Row labels (left)
    if row_labels is not None and H is not None:
        for r, lbl in enumerate(row_labels):
            y = r * H + H / 2
            plt.text(-5, y, lbl, va="center", ha="right", fontsize=10)

    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()

def save_grid_gif(
    frames_rows,          # list: each tensor [R*row_size,H,W] or [R,row_size,H,W]
    steps,                # list/array same length as frames_rows (or None)
    n_rows,               # R
    row_size,             # number of samples (columns)
    path,                 # output .gif
    cmap="gray",
    title=None,
    dpi=150,
    interval=50,          # ms
    vmin=None,
    vmax=None,
):
    """
    Creates a GIF where each frame is a tiled grid:
    rows = semantic rows (R)
    cols = samples (row_size)
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)

    tiled_frames = []
    for fr in frames_rows:
        if torch.is_tensor(fr):
            fr = fr.detach().cpu()
        else:
            fr = torch.as_tensor(fr)

        if fr.ndim == 4:
            # [R, row_size, H, W] -> [R*row_size, H, W]
            R, C, H, W = fr.shape
            assert R == n_rows and C == row_size, f"Expected ({n_rows},{row_size},H,W), got {fr.shape}"
            fr = fr.reshape(R * C, H, W)

        assert fr.ndim == 3, f"Expected [R*row_size,H,W], got {fr.shape}"

        # tile: [R*row_size,H,W] -> [R*H, row_size*W]
        grid = stack_to_grid(fr, n_rows=n_rows)  # <-- your existing function
        tiled_frames.append(grid.numpy())

    if steps is None:
        steps = [None] * len(tiled_frames)
    assert len(steps) == len(tiled_frames), "steps must have same length as frames_rows (or be None)."

    # Optional: lock scaling to avoid flicker
    if vmin is None or vmax is None:
        all_vals = np.stack(tiled_frames, axis=0)
        if vmin is None:
            vmin = float(all_vals.min())
        if vmax is None:
            vmax = float(all_vals.max())

    fig, ax = plt.subplots(figsize=(10, 3))
    ax.axis("off")

    im = ax.imshow(tiled_frames[0], cmap=cmap, animated=True, vmin=vmin, vmax=vmax)

    def _frame_title(i):
        base = "" if title is None else str(title)
        if steps[i] is None:
            return base
        return f"{base}  |  t={steps[i]}"

    ax.set_title(_frame_title(0))

    def update(i):
        im.set_array(tiled_frames[i])
        ax.set_title(_frame_title(i))
        return (im,)

    ani = animation.FuncAnimation(
        fig, update,
        frames=len(tiled_frames),
        interval=interval,
        blit=True
    )

    fig.tight_layout()
    ani.save(path, writer="pillow", dpi=dpi)
    plt.close(fig)

