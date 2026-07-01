import os
import copy
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import ListedColormap
from skimage import measure

import torch
from scipy.spatial import cKDTree
from monai.networks.nets import VarAutoEncoder

import dataset
from Diffusion import DiffusionModel, get_beta_schedule
from UNet3D import UNetModel, ConditionalSegUNetModel
from helpers import parse_args
from trainer_ddpm import stack_to_grid, save_panel, save_grid_gif

def dice_per_sample(pred_prob, target, threshold: float = 0.5, eps: float = 1e-8):
    pred = (pred_prob >= threshold).float()
    target = target.float()
    dims = list(range(1, pred.ndim))
    inter = torch.sum(pred * target, dim=dims)
    denom = torch.sum(pred, dim=dims) + torch.sum(target, dim=dims)
    return (2.0 * inter + eps) / (denom + eps)

def mse_per_sample(recon, real):
    se = (real - recon).square()
    mse = torch.mean(se, dim=list(range(len(real.shape))))
    return mse.detach().cpu().numpy()

def preblend_over(bg_hex, color_hex, alpha):
    br, bg, bb = mcolors.to_rgb(bg_hex)
    r, g, b = mcolors.to_rgb(color_hex)
    R = r*alpha + br*(1-alpha)
    G = g*alpha + bg*(1-alpha)
    B = b*alpha + bb*(1-alpha)
    return (R, G, B)

def plot(img, img_recon, seg, seg_recon, dta_map, seg_no_error, title, path, plot_labels):
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(9, 3), constrained_layout=True)
    cmap_seg = ListedColormap([preblend_over("white", "#EE6677", 0.7)]) # red

    if img is not None:
        ax[0].imshow(img.T, cmap='gray', aspect="auto", origin='lower', vmin=0, vmax=1)
    ax[0].imshow(np.ma.masked_where(seg == 0., seg).T, cmap=cmap_seg, aspect="auto", origin='lower')

    if img_recon is not None:
        ax[1].imshow(img_recon.T, cmap='gray', aspect="auto", origin='lower', vmin=0, vmax=1)
    ax[1].imshow(np.ma.masked_where(seg_recon == 0., seg_recon).T, cmap=cmap_seg, aspect="auto", origin='lower')

    max_dta = 5
    dta_map = np.clip(dta_map, 0, max_dta)
    norm = mcolors.Normalize(vmin=0, vmax=max_dta)
    sd = ax[2].imshow(dta_map.T, cmap='inferno', origin='lower', norm=norm, aspect='auto')
    cbar = fig.colorbar(sd, ax=ax[2], orientation='vertical')
    ticks = cbar.get_ticks()
    ticks[-1] = max_dta
    cbar.set_ticks(ticks)
    tick_labels = [str(t) for t in ticks]  
    tick_labels[-1] = f'{max_dta} >' 
    cbar.set_ticklabels(tick_labels)

    if seg_no_error is not None:
        error_voxels = np.where(seg_no_error != seg, 1, 0)
        contours = measure.find_contours(error_voxels.T, level=0.5)
        for contour in contours:
            ax[0].plot(contour[:, 1], contour[:, 0], linewidth=0.75, color='cyan')
            ax[1].plot(contour[:, 1], contour[:, 0], linewidth=0.75, color='cyan')
            ax[2].plot(contour[:, 1], contour[:, 0], linewidth=0.75, color='cyan')

    for i, label in enumerate(plot_labels):
        ax[i].set_title(label)
        ax[i].axis("off")

    fig.suptitle(title, fontsize=12)
    plt.savefig(path, dpi=300)
    plt.close()

def error_distances_map(inputs, outputs):
    # Get coordinates of the foreground (nonzero) pixels
    inputs_coords = np.argwhere(inputs > 0)
    outputs_coords = np.argwhere(outputs > 0)

    # Initialize the distance map
    dmap = np.full(inputs.shape, np.nan, dtype=float)

    if inputs_coords.size == 0 or outputs_coords.size == 0:
        return dmap
    
    # Build KD-Trees
    tree_inputs = cKDTree(inputs_coords)
    tree_outputs = cKDTree(outputs_coords)
    
    # Distances from inputs -> outputs
    dist_inputs, _ = tree_outputs.query(inputs_coords)
    dmap[tuple(inputs_coords.T)] = dist_inputs

    # Distances from outputs -> inputs (and combine by max at overlapping pixels)
    dist_outputs, _ = tree_inputs.query(outputs_coords)
    idx_out = tuple(outputs_coords.T)
    existing = dmap[idx_out]
    dmap[idx_out] = np.where(np.isnan(existing), dist_outputs, np.maximum(existing, dist_outputs))
    
    return dmap

def boxplot_metrics(metrics_dict, metric_name, path):
    plt.boxplot(
        [list(metrics_dict[t].values()) for t in sorted(metrics_dict.keys())],
        tick_labels=[str(t) for t in sorted(metrics_dict.keys())],
    )
    plt.xlabel("t distance")
    plt.ylabel(metric_name)
    plt.title(f"{metric_name} vs t distance")
    plt.savefig(path)
    plt.close()

def metrics_single_step(testing_dataset_loader, diffusion, args, ema, labels, has_image, has_seg, img_idx, seg_idx, num_saved_batches, device):
    mses_model = {t: {} for t in range(100, args["num_diffusion_steps"] + 1, 100)}
    dscs_model = {t: {} for t in range(100, args["num_diffusion_steps"] + 1, 100)}

    os.makedirs(os.path.join(args["path_save"], f"testing-images-{args['test_sampling']}"), exist_ok=True)
    os.makedirs(os.path.join(args["path_save"], f"testing-boxplots-{args['test_sampling']}"), exist_ok=True)

    print('Computing testing single-step metrics and images...')

    for batch_idx, data in enumerate(testing_dataset_loader):
        x = data["im"].to(device)
        keys = data["key"]
        row_size = min(8, x.shape[0])
        index_slice = x.shape[2] // 2

        # split modalities (each [B,1,...] or None)
        x_img = x[:, img_idx:img_idx+1] if has_image else None
        x_seg = x[:, seg_idx:seg_idx+1] if has_seg else None

        for t_dist in list(range(100, args["num_diffusion_steps"] + 1, 100)):
            # print(f"  t = {t_dist}")
            # use a *fixed* t for the whole batch for simplicity + compatibility with sliding-window wrappers
            t = torch.full((x.shape[0],), int(t_dist) - 1, device=device, dtype=torch.long)

            # --- build xt for each modality ---
            xt_parts = {}

            if has_image and args['model'] == 'DDPM':
                noise_img = torch.randn_like(x_img)
                xt_img = diffusion.q_sample_gaussian(x_img, t, noise_img).clamp(0, 1)
                xt_parts["image"] = xt_img

            if has_seg:
                xt_seg, _ = diffusion.q_sample_bernoulli(x_seg, t)  # binary
                xt_parts["segmentation"] = xt_seg

            # pack in labels order (model input)
            if args['model'] == 'DDPMCondSeg':
                xt = xt_parts["segmentation"]
                with torch.no_grad():
                    out = ema(xt, t, x_img)
            else:
                xt = torch.cat([xt_parts[name] for name in labels], dim=1) if len(labels) > 1 else list(xt_parts.values())[0]
                with torch.no_grad():
                    out = ema(xt, t)

            # unpack model outputs
            if args['model'] == 'DDPMCondSeg':
                eps_img, seg_logits = None, out
            elif has_image and has_seg:
                eps_img, seg_logits = out
            elif has_image:
                eps_img, seg_logits = out, None
            else:
                eps_img, seg_logits = None, out

            # GRIDS
            if has_image and args['model'] == 'DDPM':
                pred_x0_img = diffusion.predict_x0_from_eps(xt_parts["image"], t, eps_img).clamp(0, 1)

            if has_seg:
                seg_prob = torch.sigmoid(seg_logits).detach()
                seg_bin = (seg_prob >= 0.5).float()

            for im in range(x.shape[0]):
                if has_image and args['model'] == 'DDPM':
                    mses_model[t_dist][keys[im]] = float(mse_per_sample(pred_x0_img[im], x_img[im]))

                if has_seg:
                    dscs_model[t_dist][keys[im]] = float(dice_per_sample(seg_bin[im], x_seg[im]).mean().detach().cpu())

            if args['save_imgs'] and batch_idx < num_saved_batches:
                if has_image and args['model'] == 'DDPM':
                    out_stack = torch.cat((
                        x_img[:row_size, 0, index_slice].detach().cpu(),
                        xt_parts["image"][:row_size, 0, index_slice].detach().cpu(),
                        pred_x0_img[:row_size, 0, index_slice].detach().cpu(),
                    ), dim=0)

                    out_stack = torch.rot90(out_stack, k=1, dims=(1, 2))
                    grid = stack_to_grid(out_stack, n_rows=3)

                    save_panel(
                        grid,
                        cmap="gray",
                        title=f"TEST IMAGE t={t_dist}",
                        path=os.path.join(args["path_save"], f"testing-images-{args['test_sampling']}", f"test_model_preds_image_{batch_idx}_t{t_dist}.png"),
                        row_labels=["x₀", "xₜ", "pred x̂₀"],
                        H=out_stack.shape[-2],
                        W=out_stack.shape[-1],
                        row_size=row_size,
                    )

                if has_seg:
                    out_stack = torch.cat((
                        x_seg[:row_size, 0, index_slice].detach().cpu(),
                        xt_parts["segmentation"][:row_size, 0, index_slice].detach().cpu(),
                        seg_bin[:row_size, 0, index_slice].detach().cpu(),
                    ), dim=0)

                    out_stack = torch.rot90(out_stack, k=1, dims=(1, 2))
                    grid = stack_to_grid(out_stack, n_rows=3)

                    save_panel(
                        grid,
                        cmap="Reds",
                        title=f"TEST SEG t={t_dist}",
                        path=os.path.join(args["path_save"], f"testing-images-{args['test_sampling']}", f"test_model_preds_seg_{batch_idx}_t{t_dist}.png"),
                        row_labels=["x₀", "xₜ", "binary x̂₀"],
                        H=out_stack.shape[-2],
                        W=out_stack.shape[-1],
                        row_size=row_size,
                    )

            if args['debug']:
                break

    # save metrics to JSON and plot
    if has_image:
        with open(os.path.join(args["path_save"], f"mses_model_preds.json"), 'w') as f:
            json.dump(mses_model, f, indent=4)
        boxplot_metrics(mses_model, "MSE", os.path.join(args["path_save"], f"testing-boxplots-{args['test_sampling']}", f"mses_model_preds_boxplot.png"))

    if has_seg:
        with open(os.path.join(args["path_save"], f"dscs_model_preds.json"), 'w') as f:
            json.dump(dscs_model, f, indent=4)
        boxplot_metrics(dscs_model, "DSC", os.path.join(args["path_save"], f"testing-boxplots-{args['test_sampling']}", f"dscs_model_preds_boxplot.png"))

def gif_single_step(testing_dataset_loader, diffusion, args, ema, img_idx, device):
    print('Generating diffusion testing GIFs...')

    for batch_idx, data in enumerate(testing_dataset_loader):
        x = data["im"].to(device)
        row_size = min(8, x.shape[0])
        index_slice = x.shape[2] // 2

        x_img = x[:, img_idx:img_idx+1]

        # full chain uses sliding-window wrapper
        if args['test_sampling'] == 'ddpm':
            if args['debug']:
                length = args['num_diffusion_steps'] // 20
            else:
                length = args['num_diffusion_steps'] // 4
            out_seq = diffusion.forward_backward(
                ema,
                x_img,
                see_whole_sequence="whole",
                t_distance=length,
            )

            out_seq = torch.stack(out_seq, dim=0).clamp(0, 1)  # [T',B,1,D,H,W]

            # sample every 10 steps (or fewer if short)
            stride = 10 

            frames = []
            steps = []
            for k in range(0, out_seq.shape[0], stride):
                fr = out_seq[k, :row_size, 0, index_slice]          # [row,H,W]
                fr = torch.rot90(fr, k=1, dims=(1, 2))
                frames.append(fr.cpu())
                if k <= length:   
                    steps.append(k) 
                else:
                    steps.append(2 * length - k) 

            frames_rows = [f.reshape(1 * row_size, f.shape[-2], f.shape[-1]) for f in frames]

        else:
            if args["model"] == "DDPMCondSeg":
                out_seq, steps= diffusion.forward_backward_ddim_conditional_seg(
                    ema,
                    x_img,
                    see_whole_sequence="whole",
                    t_distance=args['num_diffusion_steps'],
                    ddim_steps=args['ddim_steps'],  
                )
            else:
                out_seq, steps = diffusion.forward_backward_ddim(
                    ema,
                    x_img,
                    see_whole_sequence="whole",
                    t_distance=args['num_diffusion_steps'],
                    ddim_steps=args['ddim_steps'],  
                    eta=args['ddim_eta'],         
                )

            out_seq = torch.stack(out_seq, dim=0).clamp(0, 1)  # [T',B,1,D,H,W]

            frames_rows = out_seq[:, :row_size, 0, index_slice]
            frames_rows = torch.rot90(frames_rows, k=1, dims=(2, 3))  # [T', row_size, H, W]

            frames_rows = frames_rows[-20:]
            steps = steps[-20:]

        save_grid_gif(
            frames_rows=frames_rows,
            steps=steps,
            n_rows=1,
            row_size=row_size,
            path=os.path.join(args["path_save"], f"testing-gifs-{args['test_sampling']}", f"test_image.gif"),
            cmap="gray",
            title=f"TEST IMAGE",
            interval=200,
        )

def testing(testing_dataset_loader, args, diffusion, ema, vae, device):

    if args['num_channels'] == 2:
        has_image, has_seg = True, True
        img_idx, seg_idx = 0, 1
    elif args['num_channels'] == 1:
        if args['organ_channel'] is not None:
            has_image, has_seg = False, True
            seg_idx = 0
        else:
            has_image, has_seg = True, False
            img_idx = 0

    if diffusion is not None:
        ema.eval()
        num_saved_batches = 4
        os.makedirs(os.path.join(args["path_save"], f"testing-images-{args['test_sampling']}"), exist_ok=True)
        os.makedirs(os.path.join(args["path_save"], f"testing-boxplots-{args['test_sampling']}"), exist_ok=True)

        if args['save_metrics']:
            metrics_single_step(testing_dataset_loader, diffusion, args, ema, diffusion.labels_channels, has_image, has_seg, img_idx, seg_idx, num_saved_batches, device=device)

        if args["save_vids"] and has_image:
            gif_single_step(testing_dataset_loader, diffusion, args, ema, img_idx, device)

        mses = {t_dist: {} for t_dist in range(100, args["num_diffusion_steps"] + 1, 100)}
        dscs = {t_dist: {} for t_dist in range(100, args["num_diffusion_steps"] + 1, 100)}
        dtas = {t_dist: {} for t_dist in range(100, args["num_diffusion_steps"] + 1, 100)}
        dtas_95 = {t_dist: {} for t_dist in range(100, args["num_diffusion_steps"] + 1, 100)}

        t_dists_range = list(range(100, args["num_diffusion_steps"] + 1, 100))
        img_dir = f"testing-images-{args['test_sampling']}"

    elif vae is not None:
        vae.eval()
        num_saved_batches = 1
        os.makedirs(os.path.join(args["path_save"], f"testing-images"), exist_ok=True)

        mses = {0: {}}
        dscs = {0: {}}
        dtas = {0: {}}
        dtas_95 = {0: {}}

        t_dists_range = [0] # dummy variable for unified handling with diffusion
        img_dir = "testing-images"

    print('Computing testing quantitative metrics...')

    for t_dist in t_dists_range:
        for batch_idx, data in enumerate(testing_dataset_loader):
            x = data["im"].to(device)
            keys = data["key"]

            if diffusion is not None:
                if args['test_sampling'] == 'ddpm':
                    xrec = diffusion.forward_backward(ema, x, see_whole_sequence=None, t_distance=t_dist)
                else:
                    if args["model"] == "DDPMCondSeg":
                        xrec, _ = diffusion.forward_backward_ddim_conditional_seg(
                            ema,
                            x,
                            see_whole_sequence=None,
                            t_distance=t_dist,
                            ddim_steps=args["ddim_steps"],
                        )
                    else:
                        xrec, _ = diffusion.forward_backward_ddim(
                            ema,
                            x,
                            see_whole_sequence=None,
                            t_distance=t_dist,
                            ddim_steps=args["ddim_steps"],  
                            eta=args["ddim_eta"],         
                        )
            elif vae is not None:
                with torch.no_grad():
                    xrec, _, _, _ = vae(x)

            for im in range(len(x)):
                if has_image:
                    x_img = x[im][img_idx]
                    xrec_img = xrec[im][img_idx] 
                    mses[t_dist][keys[im]] = float(mse_per_sample(xrec_img, x_img))

                if has_seg:
                    x_seg = x[im][seg_idx] 
                    xrec_seg = xrec[im][seg_idx]
                    xrec_seg_bin = (xrec_seg >= 0.5).float()
                    dscs[t_dist][keys[im]] = float(dice_per_sample(xrec_seg_bin, x_seg).mean().detach().cpu())

                    dmap_vals = error_distances_map(x_seg.cpu().detach().numpy(), xrec_seg_bin.cpu().detach().numpy())
                    union = (x_seg.cpu().detach().numpy() > 0) | (xrec_seg_bin.cpu().detach().numpy() > 0)
                    vals = dmap_vals[union]
                    dtas[t_dist][keys[im]] = float(np.nanmean(vals)) if vals.size > 0 else float('nan')
                    dtas_95[t_dist][keys[im]] = float(np.nanpercentile(vals, 95)) if vals.size > 0 else float('nan')

                    dmap_to_plot = np.nan_to_num(dmap_vals, nan=0.0) 

                if batch_idx < num_saved_batches:
                    if diffusion is not None:
                        plot_name = f"test_{keys[im]}_t{t_dist}.png"
                    else:
                        plot_name = f"test_{keys[im]}.png"

                    if has_image and has_seg:
                        if args['organ_channel'] == 'brainstem' or args['organ_channel'] == 'spinalcord':
                            index_slice = int(np.mean(np.argwhere(x_seg.cpu().detach().numpy()), axis=0)[0]) if x_seg.cpu().detach().numpy().sum() > 0 else x_seg.shape[0] // 2
                            plot(
                                x_img[index_slice].detach().cpu(),
                                xrec_img[index_slice].detach().cpu(),
                                x_seg[index_slice].detach().cpu(),
                                xrec_seg_bin[index_slice].detach().cpu(),
                                dmap_to_plot[index_slice],
                                seg_no_error=None,
                                title=f"TEST t={t_dist}",
                                path=os.path.join(args["path_save"], img_dir, plot_name),
                                plot_labels=["x₀", "pred x̂₀", 'DTA map'],
                            )
                        elif args['organ_channel'] == 'parotid_l' or args['organ_channel'] == 'parotid_r':
                            index_slice = int(np.mean(np.argwhere(x_seg.cpu().detach().numpy()), axis=0)[2]) if x_seg.cpu().detach().numpy().sum() > 0 else x_seg.shape[2] // 2
                            plot(
                                x_img[:, :, index_slice].detach().cpu(),
                                xrec_img[:, :, index_slice].detach().cpu(),
                                x_seg[:, :, index_slice].detach().cpu(),
                                xrec_seg_bin[:, :, index_slice].detach().cpu(),
                                dmap_to_plot[:, :, index_slice],
                                seg_no_error=None,
                                title=f"TEST t={t_dist}",
                                path=os.path.join(args["path_save"], img_dir, plot_name),
                                plot_labels=["x₀", "pred x̂₀", 'DTA map'],
                            )
                    
                    elif has_seg:
                        if args['organ_channel'] == 'brainstem' or args['organ_channel'] == 'spinalcord':
                            index_slice = int(np.mean(np.argwhere(x_seg.cpu().detach().numpy()), axis=0)[0]) if x_seg.cpu().detach().numpy().sum() > 0 else x_seg.shape[0] // 2
                            plot(
                                None,
                                None,
                                x_seg[index_slice].detach().cpu(),
                                xrec_seg_bin[index_slice].detach().cpu(),
                                dmap_to_plot[index_slice],
                                seg_no_error=None,
                                title=f"TEST t={t_dist}",
                                path=os.path.join(args["path_save"], img_dir, plot_name),
                                plot_labels=["x₀", "pred x̂₀", 'DTA map'],
                            )
                        elif args['organ_channel'] == 'parotid_l' or args['organ_channel'] == 'parotid_r':
                            index_slice = int(np.mean(np.argwhere(x_seg.cpu().detach().numpy()), axis=0)[2]) if x_seg.cpu().detach().numpy().sum() > 0 else x_seg.shape[2] // 2
                            plot(
                                None,
                                None,
                                x_seg[:, :, index_slice].detach().cpu(),
                                xrec_seg_bin[:, :, index_slice].detach().cpu(),
                                dmap_to_plot[:, :, index_slice],
                                seg_no_error=None,
                                title=f"TEST t={t_dist}",
                                path=os.path.join(args["path_save"], img_dir, plot_name),
                                plot_labels=["x₀", "pred x̂₀", 'DTA map'],
                            )

    # -----------------------
    # printing helpers
    # -----------------------
    def mean_std(xs):
        xs = np.array(xs, dtype=float)
        return xs.mean(), xs.std()

    print("\n=== Test set metrics ===")

    # ---- recon quality ---
    if has_image:
        if diffusion is not None:
            for t_dist in t_dists_range:
                m, s = mean_std(list(mses[t_dist].values()))
                print(f"MSE (t={t_dist}):      {m:.6f} +- {s:.6f}")
            boxplot_metrics(mses, "MSE", os.path.join(args["path_save"], f"testing-boxplots-{args['test_sampling']}", f"mses_recon_boxplot.png"))
            filename = f"img_metrics_recon_{args['dataset']}_{args['test_sampling']}.json"
        
        elif vae is not None:
            m, s = mean_std(list(mses[0].values()))
            print(f"MSE:      {m:.6f} +- {s:.6f}")
            filename = f"img_metrics_{args['dataset']}.json"
        
        with open(os.path.join(args["path_save"], filename), 'w') as f:
            json.dump(mses, f, indent=4)

    if has_seg:
        if diffusion is not None:
            for t_dist in list(range(100, args["num_diffusion_steps"] + 1, 100)):
                m_dsc, s_dsc = mean_std(list(dscs[t_dist].values()))
                print(f"DSC (t={t_dist}):      {m_dsc:.4f} +- {s_dsc:.4f}")
            for t_dist in list(range(100, args["num_diffusion_steps"] + 1, 100)):
                m_dta, s_dta = mean_std(list(dtas[t_dist].values()))
                print(f"DTA (t={t_dist}):      {m_dta:.4f} +- {s_dta:.4f}")
            for t_dist in list(range(100, args["num_diffusion_steps"] + 1, 100)):
                m_dta95, s_dta95 = mean_std(list(dtas_95[t_dist].values()))
                print(f"95th DTA (t={t_dist}): {m_dta95:.4f} +- {s_dta95:.4f}")
            boxplot_metrics(dscs, "DSC", os.path.join(args["path_save"], f"testing-boxplots-{args['test_sampling']}", f"dscs_recon_boxplot.png"))
            boxplot_metrics(dtas, "DTA", os.path.join(args["path_save"], f"testing-boxplots-{args['test_sampling']}", f"dtas_recon_boxplot.png"))
            boxplot_metrics(dtas_95, "DTA 95th", os.path.join(args["path_save"], f"testing-boxplots-{args['test_sampling']}", f"dtas_95_recon_boxplot.png"))

            filename = f"seg_metrics_{args['dataset']}_{args['test_sampling']}.json"
        
        elif vae is not None:
            m_dsc, s_dsc = mean_std(list(dscs[0].values()))
            print(f"DSC:      {m_dsc:.4f} +- {s_dsc:.4f}")
            m_dta, s_dta = mean_std(list(dtas[0].values()))
            print(f"DTA:      {m_dta:.4f} +- {s_dta:.4f}")
            m_dta95, s_dta95 = mean_std(list(dtas_95[0].values()))
            print(f"95th DTA: {m_dta95:.4f} +- {s_dta95:.4f}")

            filename = f"seg_metrics_{args['dataset']}.json"

        seg_metrics_save = {
            "dscs": dscs,
            "dtas": dtas,
            "dtas_95": dtas_95,
        }
        with open(os.path.join(args["path_save"], filename), 'w') as f:
            json.dump(seg_metrics_save, f, indent=4)
    
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    args = parse_args()

    _, _, testing_dataset = dataset.init_datasets(args, debug=args["debug"], test_only=True)
    testing_dataset_loader = dataset.init_dataset_loader(testing_dataset, args, shuffle=False)

    output = torch.load(os.path.join(args["path_save"], "model", "model_final.pt"), map_location=device, weights_only=False)

    if args['model'] == 'DDPM' or args['model'] == 'DDPMCondSeg':
        if args['model'] == 'DDPM':
            unet = UNetModel(
                args['patch_size'][0],
                args['init_channels'],
                channel_mults=args['channel_mults'],
                dropout=args["dropout"],
                n_heads=args["num_heads"],
                n_head_channels=args.get("num_head_channels", -1),
                in_channels=args['num_channels']
            )
        else:
            unet = ConditionalSegUNetModel(
                args['patch_size'][0],
                args['init_channels'],
                channel_mults=args['channel_mults'],
                dropout=args["dropout"],
                n_heads=args["num_heads"],
                n_head_channels=args.get("num_head_channels", -1),
            )

        ema = copy.deepcopy(unet)

        betas = get_beta_schedule(args['num_diffusion_steps'], args['beta_schedule'])

        diff = DiffusionModel(
            args['patch_size'],
            betas,
            loss_weight=args['loss_weight'],
            loss_types=args['loss_types'],
            noise=args["noise_fn"],
            in_channels=args['num_channels'],
            labels_channels=args['labels_channels'],
            separate_seg_schedule=args['separate_seg_schedule'],
        )

        ema.load_state_dict(output["ema"], strict=False)
        ema.to(device)
        ema.eval()

        testing(testing_dataset_loader, args, diffusion=diff, ema=ema, vae=None, device=device)

    elif args['model'] == 'VAE':
        model = VarAutoEncoder(
            spatial_dims=3,
            in_shape=(args["num_channels"], *args["patch_size"]), 
            out_channels=args["num_channels"],
            latent_size=args["latent_size"],
            channels=args["model_channels"],
            strides=args["model_strides"],
            dropout=args["drop_rate"],
            num_res_units=args["num_res_units"],
        ).to(device)

        model.load_state_dict(output["model_state_dict"], strict=False)
        model.eval()

        testing(testing_dataset_loader, args, diffusion=None, ema=None, vae=model, device=device)

if __name__ == '__main__':
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    
    main()
