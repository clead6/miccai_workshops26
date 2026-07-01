import os
import copy
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import ListedColormap
from skimage import measure

import torch
import torch.nn.functional as F
from monai.networks.nets import VarAutoEncoder

import dataset
from Diffusion import DiffusionModel, get_beta_schedule
from UNet3D import UNetModel, ConditionalSegUNetModel
from helpers import parse_args

from evaluation import dice_per_sample, mse_per_sample, preblend_over, error_distances_map, plot

def plot_reviewed(img_ori, img_ori_recon, seg_ori, seg_ori_recon, dta_ori_map, 
                  img_cor, img_cor_recon, seg_cor, seg_cor_recon, dta_cor_map, 
                  title, path, plot_labels):
    fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(9, 6), constrained_layout=True)
    cmap_seg = ListedColormap([preblend_over("white", "#EE6677", 0.7)]) # red

    if img_ori is not None:
        ax[0, 0].imshow(img_ori.T, cmap='gray', aspect="auto", origin='lower', vmin=0, vmax=1)
        ax[1, 0].imshow(img_cor.T, cmap='gray', aspect="auto", origin='lower', vmin=0, vmax=1)
    if seg_ori is not None:
        ax[0, 0].imshow(np.ma.masked_where(seg_ori == 0., seg_ori).T, cmap=cmap_seg, aspect="auto", origin='lower')
        ax[1, 0].imshow(np.ma.masked_where(seg_cor == 0., seg_cor).T, cmap=cmap_seg, aspect="auto", origin='lower')

    if img_ori_recon is not None:
        ax[0, 1].imshow(img_ori_recon.T, cmap='gray', aspect="auto", origin='lower', vmin=0, vmax=1)
        ax[1, 1].imshow(img_cor_recon.T, cmap='gray', aspect="auto", origin='lower', vmin=0, vmax=1)
    if seg_ori_recon is not None:
        ax[0, 1].imshow(np.ma.masked_where(seg_ori_recon == 0., seg_ori_recon).T, cmap=cmap_seg, aspect="auto", origin='lower')
        ax[1, 1].imshow(np.ma.masked_where(seg_cor_recon == 0., seg_cor_recon).T, cmap=cmap_seg, aspect="auto", origin='lower')

    max_dta = 5
    dta_ori_map = np.clip(dta_ori_map, 0, max_dta)
    norm = mcolors.Normalize(vmin=0, vmax=max_dta)
    sd = ax[0, 2].imshow(dta_ori_map.T, cmap='inferno', origin='lower', norm=norm, aspect='auto')
    cbar = fig.colorbar(sd, ax=ax[0, 2], orientation='vertical')
    ticks = cbar.get_ticks()
    ticks[-1] = max_dta
    cbar.set_ticks(ticks)
    tick_labels = [str(t) for t in ticks]  
    tick_labels[-1] = f'{max_dta} >' 
    cbar.set_ticklabels(tick_labels)
    dta_cor_map = np.clip(dta_cor_map, 0, max_dta)
    norm = mcolors.Normalize(vmin=0, vmax=max_dta)
    sd = ax[1, 2].imshow(dta_cor_map.T, cmap='inferno', origin='lower', norm=norm, aspect='auto')
    cbar = fig.colorbar(sd, ax=ax[1, 2], orientation='vertical')
    ticks = cbar.get_ticks()
    ticks[-1] = max_dta
    cbar.set_ticks(ticks)
    tick_labels = [str(t) for t in ticks]  
    tick_labels[-1] = f'{max_dta} >' 
    cbar.set_ticklabels(tick_labels)

    if seg_ori is not None and img_ori is not None:
        corrected_voxels = np.where(seg_ori != seg_cor, 1, 0)
        contours = measure.find_contours(corrected_voxels.T, level=0.5)
        for contour in contours:
            ax[0, 0].plot(contour[:, 1], contour[:, 0], linewidth=0.75, color='cyan')
            ax[0, 1].plot(contour[:, 1], contour[:, 0], linewidth=0.75, color='cyan')
            ax[0, 2].plot(contour[:, 1], contour[:, 0], linewidth=0.75, color='cyan')
            ax[1, 0].plot(contour[:, 1], contour[:, 0], linewidth=0.75, color='cyan')
            ax[1, 1].plot(contour[:, 1], contour[:, 0], linewidth=0.75, color='cyan')
            ax[1, 2].plot(contour[:, 1], contour[:, 0], linewidth=0.75, color='cyan')

    for i, label in enumerate(plot_labels):
        ax[0, i].set_title(label)
        ax[0, i].axis("off")
        ax[1, i].axis("off")
    ax[0, 0].set_ylabel("Original")
    ax[1, 0].set_ylabel("Corrected")

    fig.suptitle(title, fontsize=12)
    plt.savefig(path, dpi=300)
    plt.close()

def testing_reviewed_aggreg(testing_dataset_original_loader, testing_dataset_corrected_loader,
                            args, diffusion, ema, vae, test_ts, device):

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
        num_saved_batches = 10
        os.makedirs(os.path.join(args["path_save"], f"testing-images-reviewed-{args['test_sampling']}"), exist_ok=True)
        img_dir = f"testing-images-reviewed-{args['test_sampling']}"
    elif vae is not None:
        vae.eval()
        num_saved_batches = 1
        os.makedirs(os.path.join(args["path_save"], f"testing-images-reviewed"), exist_ok=True)
        img_dir = "testing-images-reviewed"
    
    mses = {i: {} for i in ['original', 'corrected']}
    dscs = {i: {} for i in ['original', 'corrected']}
    dtas = {i: {} for i in ['original', 'corrected']}
    dtas_95 = {i: {} for i in ['original', 'corrected']}
    dtas_corrected = {i: {} for i in ['original', 'corrected']}
    dtas_95_corrected = {i: {} for i in ['original', 'corrected']}
    dtas_non_corrected = {i: {} for i in ['original', 'corrected']}
    dtas_95_non_corrected = {i: {} for i in ['original', 'corrected']}
    dtas_corrected_perc = {k: {i: {} for i in ['original', 'corrected']} for k in [0, 1, 2]}

    for batch_idx, (data_original, data_corrected) in enumerate(zip(testing_dataset_original_loader, testing_dataset_corrected_loader)):
        x_ori = data_original["im"].to(device)
        x_cor = data_corrected["im"].to(device)
        keys = data_original["key"]

        xrec_ori_aggreg = []
        xrec_cor_aggreg = []

        for test_t in test_ts:
            if diffusion is not None:
                if args['test_sampling'] == 'ddpm':
                    xrec_ori = diffusion.forward_backward(
                        ema, x_ori, see_whole_sequence=None, t_distance=test_t)
                    xrec_cor = diffusion.forward_backward(
                        ema, x_cor, see_whole_sequence=None, t_distance=test_t)
                else:
                    if args["model"] == "DDPMCondSeg":
                        xrec_ori, _ = diffusion.forward_backward_ddim_conditional_seg(
                            ema, x_ori, see_whole_sequence=None,
                            t_distance=test_t, ddim_steps=args["ddim_steps"])
                        xrec_cor, _ = diffusion.forward_backward_ddim_conditional_seg(
                            ema, x_cor, see_whole_sequence=None,
                            t_distance=test_t, ddim_steps=args["ddim_steps"])
                    else:
                        xrec_ori, _ = diffusion.forward_backward_ddim(
                            ema, x_ori, see_whole_sequence=None, t_distance=test_t,
                            ddim_steps=args["ddim_steps"],   eta=args["ddim_eta"])
                        xrec_cor, _ = diffusion.forward_backward_ddim(
                            ema, x_cor, see_whole_sequence=None, t_distance=test_t,
                            ddim_steps=args["ddim_steps"],   eta=args["ddim_eta"])
            
            elif vae is not None:
                with torch.no_grad():
                    xrec_ori, _, _, _ = vae(x_ori)
                    xrec_cor, _, _, _ = vae(x_cor)

            xrec_ori_aggreg.append(xrec_ori)
            xrec_cor_aggreg.append(xrec_cor)

        xrec_ori = torch.mean(torch.stack(xrec_ori_aggreg, dim=0), dim=0)
        xrec_cor = torch.mean(torch.stack(xrec_cor_aggreg, dim=0), dim=0)
            
        for im in range(len(keys)):
            if has_image:
                x_ori_img = x_ori[im][img_idx]
                x_cor_img = x_cor[im][img_idx]
                xrec_ori_img = xrec_ori[im][img_idx]
                xrec_cor_img = xrec_cor[im][img_idx]
                mses['original'][keys[im]] = float(mse_per_sample(xrec_ori_img, x_ori_img))
                mses['corrected'][keys[im]] = float(mse_per_sample(xrec_cor_img, x_cor_img))

            if has_seg:
                x_ori_seg = x_ori[im][seg_idx]
                x_cor_seg = x_cor[im][seg_idx]
                xrec_ori_seg = xrec_ori[im][seg_idx]
                xrec_cor_seg = xrec_cor[im][seg_idx]
                xrec_ori_seg_bin = (xrec_ori_seg >= 0.5).float()
                xrec_cor_seg_bin = (xrec_cor_seg >= 0.5).float()
                dscs['original'][keys[im]] = float(dice_per_sample(xrec_ori_seg_bin, x_ori_seg).mean().detach().cpu())
                dscs['corrected'][keys[im]] = float(dice_per_sample(xrec_cor_seg_bin, x_cor_seg).mean().detach().cpu())
                
                union = ((x_ori_seg.cpu().detach().numpy() > 0) | (xrec_ori_seg_bin.cpu().detach().numpy() > 0) | (x_cor_seg.cpu().detach().numpy() > 0) | (xrec_cor_seg_bin.cpu().detach().numpy() > 0))
                corrected_voxels = ((x_cor_seg.cpu().detach().numpy() == 1) & (x_ori_seg.cpu().detach().numpy() == 0)) | ((x_cor_seg.cpu().detach().numpy() == 0) & (x_ori_seg.cpu().detach().numpy() == 1))
                non_corrected_voxels = union & ~corrected_voxels

                dmap_vals_ori = error_distances_map(x_ori_seg.cpu().detach().numpy(), xrec_ori_seg_bin.cpu().detach().numpy())
                dmap_vals_cor = error_distances_map(x_cor_seg.cpu().detach().numpy(), xrec_cor_seg_bin.cpu().detach().numpy())
                vals_ori = dmap_vals_ori[union]
                vals_cor = dmap_vals_cor[union]
                vals_ori_corrected_voxels = dmap_vals_ori[corrected_voxels]
                vals_ori_non_corrected_voxels = dmap_vals_ori[non_corrected_voxels]
                vals_cor_corrected_voxels = dmap_vals_cor[corrected_voxels]
                vals_cor_non_corrected_voxels = dmap_vals_cor[non_corrected_voxels]

                dtas['original'][keys[im]] = float(np.nanmean(vals_ori))
                dtas['corrected'][keys[im]] = float(np.nanmean(vals_cor))
                dtas_95['original'][keys[im]] = float(np.nanpercentile(vals_ori, 95))
                dtas_95['corrected'][keys[im]] = float(np.nanpercentile(vals_cor, 95))
                dtas_corrected['original'][keys[im]] = float(np.nanmean(vals_ori_corrected_voxels)) if (np.sum(corrected_voxels) > 0 and np.nansum(vals_ori_corrected_voxels) > 0.) else float(0)
                dtas_corrected['corrected'][keys[im]] = float(np.nanmean(vals_cor_corrected_voxels)) if (np.sum(corrected_voxels) > 0 and np.nansum(vals_cor_corrected_voxels) > 0.) else float(0)
                dtas_95_corrected['original'][keys[im]] = float(np.nanpercentile(vals_ori_corrected_voxels, 95)) if (np.sum(corrected_voxels) > 0 and np.nansum(vals_ori_corrected_voxels) > 0.) else float(0)
                dtas_95_corrected['corrected'][keys[im]] = float(np.nanpercentile(vals_cor_corrected_voxels, 95)) if (np.sum(corrected_voxels) > 0 and np.nansum(vals_cor_corrected_voxels) > 0.) else float(0)
                dtas_non_corrected['original'][keys[im]] = float(np.nanmean(vals_ori_non_corrected_voxels)) if (np.sum(non_corrected_voxels) > 0 and np.nansum(vals_ori_non_corrected_voxels) > 0.) else float(0)
                dtas_non_corrected['corrected'][keys[im]] = float(np.nanmean(vals_cor_non_corrected_voxels)) if (np.sum(non_corrected_voxels) > 0 and np.nansum(vals_cor_non_corrected_voxels) > 0.) else float(0)
                dtas_95_non_corrected['original'][keys[im]] = float(np.nanpercentile(vals_ori_non_corrected_voxels, 95)) if (np.sum(non_corrected_voxels) > 0 and np.nansum(vals_ori_non_corrected_voxels) > 0.) else float(0)
                dtas_95_non_corrected['corrected'][keys[im]] = float(np.nanpercentile(vals_cor_non_corrected_voxels, 95)) if (np.sum(non_corrected_voxels) > 0 and np.nansum(vals_cor_non_corrected_voxels) > 0.) else float(0)

                for threshold in (0, 1, 2):
                    n_ori_all = np.sum(vals_ori > threshold)
                    n_ori_corrected_voxels = np.sum(vals_ori_corrected_voxels > threshold)
                    dtas_corrected_perc[threshold]['original'][keys[im]] = float(n_ori_corrected_voxels / n_ori_all) if n_ori_all > 0 else float('nan')

                    n_cor_all = np.sum(vals_cor > threshold)
                    n_cor_corrected_voxels = np.sum(vals_cor_corrected_voxels > threshold)
                    dtas_corrected_perc[threshold]['corrected'][keys[im]] = float(n_cor_corrected_voxels / n_cor_all) if n_cor_all > 0 else float('nan')

                dmap_ori_to_plot = np.nan_to_num(dmap_vals_ori, nan=0.0)
                dmap_cor_to_plot = np.nan_to_num(dmap_vals_cor, nan=0.0)

            if batch_idx < num_saved_batches:
                if diffusion is not None:
                    plot_name = f"test_{keys[im]}_t_aggreg.png"
                else:
                    plot_name = f"test_{keys[im]}.png"
                plot_title = f"TEST"

                if has_image and has_seg:
                    if args['organ_channel'] == 'brainstem' or args['organ_channel'] == 'spinalcord':
                        index_slice = int(np.mean(np.argwhere(x_cor_seg.cpu().detach().numpy()), axis=0)[0]) if x_cor_seg.cpu().detach().numpy().sum() > 0 else x_cor_seg.shape[0] // 2
                        plot_reviewed(
                            x_ori_img[index_slice].detach().cpu(),
                            xrec_ori_img[index_slice].detach().cpu(),
                            x_ori_seg[index_slice].detach().cpu(),
                            xrec_ori_seg_bin[index_slice].detach().cpu(),
                            dmap_ori_to_plot[index_slice],
                            x_cor_img[index_slice].detach().cpu(),
                            xrec_cor_img[index_slice].detach().cpu(),
                            x_cor_seg[index_slice].detach().cpu(),
                            xrec_cor_seg_bin[index_slice].detach().cpu(),
                            dmap_cor_to_plot[index_slice],
                            title=plot_title,
                            path=os.path.join(args["path_save"], img_dir, plot_name),
                            plot_labels=["x₀", "pred x̂₀", 'DTA map'],
                        )
                        plot_reviewed(
                            None,
                            None,
                            x_ori_seg[index_slice].detach().cpu(),
                            xrec_ori_seg_bin[index_slice].detach().cpu(),
                            dmap_ori_to_plot[index_slice],
                            None,
                            None,
                            x_cor_seg[index_slice].detach().cpu(),
                            xrec_cor_seg_bin[index_slice].detach().cpu(),
                            dmap_cor_to_plot[index_slice],
                            title=plot_title,
                            path=os.path.join(args["path_save"], img_dir, f"test_{keys[im]}_t_aggreg_seg_only.png"),
                            plot_labels=["x₀", "pred x̂₀", 'DTA map'],
                        )
                        plot_reviewed(
                            x_ori_img[index_slice].detach().cpu(),
                            xrec_ori_img[index_slice].detach().cpu(),
                            None,
                            None,
                            dmap_ori_to_plot[index_slice],
                            x_cor_img[index_slice].detach().cpu(),
                            xrec_cor_img[index_slice].detach().cpu(),
                            None,
                            None,
                            dmap_cor_to_plot[index_slice],
                            title=plot_title,
                            path=os.path.join(args["path_save"], img_dir, f"test_{keys[im]}_t_aggreg_img_only.png"),
                            plot_labels=["x₀", "pred x̂₀", 'DTA map'],
                        )
                        # exit()
                    elif args['organ_channel'] == 'parotid_l' or args['organ_channel'] == 'parotid_r':
                        index_slice = int(np.mean(np.argwhere(x_cor_seg.cpu().detach().numpy()), axis=0)[2]) if x_cor_seg.cpu().detach().numpy().sum() > 0 else x_cor_seg.shape[2] // 2
                        plot_reviewed(
                            x_ori_img[:, :, index_slice].detach().cpu(),
                            xrec_ori_img[:, :, index_slice].detach().cpu(),
                            x_ori_seg[:, :, index_slice].detach().cpu(),
                            xrec_ori_seg_bin[:, :, index_slice].detach().cpu(),
                            dmap_ori_to_plot[:, :, index_slice],
                            x_cor_img[:, :, index_slice].detach().cpu(),
                            xrec_cor_img[:, :, index_slice].detach().cpu(),
                            x_cor_seg[:, :, index_slice].detach().cpu(),
                            xrec_cor_seg_bin[:, :, index_slice].detach().cpu(),
                            dmap_cor_to_plot[:, :, index_slice],
                            title=plot_title,
                            path=os.path.join(args["path_save"], img_dir, plot_name),
                            plot_labels=["x₀", "pred x̂₀", 'DTA map'],
                        )                        

                elif has_seg:
                    if args['organ_channel'] == 'brainstem' or args['organ_channel'] == 'spinalcord':
                        index_slice = int(np.mean(np.argwhere(x_cor_seg.cpu().detach().numpy()), axis=0)[0]) if x_cor_seg.cpu().detach().numpy().sum() > 0 else x_cor_seg.shape[0] // 2
                        plot_reviewed(
                            None,
                            None,
                            x_ori_seg[index_slice].detach().cpu(),
                            xrec_ori_seg_bin[index_slice].detach().cpu(),
                            dmap_ori_to_plot[index_slice],
                            None,
                            None,
                            x_cor_seg[index_slice].detach().cpu(),
                            xrec_cor_seg_bin[index_slice].detach().cpu(),
                            dmap_cor_to_plot[index_slice],
                            title=plot_title,
                            path=os.path.join(args["path_save"], img_dir, plot_name),
                            plot_labels=["x₀", "pred x̂₀", 'DTA map'],
                        )
                        
                    elif args['organ_channel'] == 'parotid_l' or args['organ_channel'] == 'parotid_r':
                        index_slice = int(np.mean(np.argwhere(x_cor_seg.cpu().detach().numpy()), axis=0)[2]) if x_cor_seg.cpu().detach().numpy().sum() > 0 else x_cor_seg.shape[2] // 2
                        plot_reviewed(
                            None,
                            None,
                            x_ori_seg[:, :, index_slice].detach().cpu(),
                            xrec_ori_seg_bin[:, :, index_slice].detach().cpu(),
                            dmap_ori_to_plot[:, :, index_slice],
                            None,
                            None,
                            x_cor_seg[:, :, index_slice].detach().cpu(),
                            xrec_cor_seg_bin[:, :, index_slice].detach().cpu(),
                            dmap_cor_to_plot[:, :, index_slice],
                            title=plot_title,
                            path=os.path.join(args["path_save"], img_dir, plot_name),
                            plot_labels=["x₀", "pred x̂₀", 'DTA map'],
                        )                        

    def mean_std(xs):
        xs = np.array(xs, dtype=float)
        return np.nanmean(xs), np.nanstd(xs)
    
    print("\n=== Test set metrics ===")

    # ---- recon quality ----
    if has_image:
        m_ori, s_ori = mean_std(list(mses['original'].values()))
        m_cor, s_cor = mean_std(list(mses['corrected'].values()))
        print(f"[IMAGE] MSE: Original: {m_ori:.4f} +- {s_ori:.4f}, Corrected: {m_cor:.4f} +- {s_cor:.4f}")

        img_metrics_save = {
            "mses": mses
        }
        if diffusion is not None:
            filename = f"img_metrics_reviewed_{args['dataset']}_{args['test_sampling']}_aggreg.json"
        else:
            filename = f"img_metrics_reviewed_{args['dataset']}.json"

        with open(os.path.join(args["path_save"], filename), "w") as f:
            json.dump(img_metrics_save, f)

    if has_seg:      
        m_dsc_ori, s_dsc_ori = mean_std(list(dscs['original'].values()))
        m_dsc_cor, s_dsc_cor = mean_std(list(dscs['corrected'].values()))
        m_dta_ori, s_dta_ori = mean_std(list(dtas['original'].values()))
        m_dta_cor, s_dta_cor = mean_std(list(dtas['corrected'].values()))
        m_dta95_ori, s_dta95_ori = mean_std(list(dtas_95['original'].values()))
        m_dta95_cor, s_dta95_cor = mean_std(list(dtas_95['corrected'].values()))
        m_dta_corrected_ori, s_dta_corrected_ori = mean_std(list(dtas_corrected['original'].values()))
        m_dta_corrected_cor, s_dta_corrected_cor = mean_std(list(dtas_corrected['corrected'].values()))
        m_dta95_corrected_ori, s_dta95_corrected_ori = mean_std(list(dtas_95_corrected['original'].values()))
        m_dta95_corrected_cor, s_dta95_corrected_cor = mean_std(list(dtas_95_corrected['corrected'].values()))  
        m_dta_non_corrected_ori, s_dta_non_corrected_ori = mean_std(list(dtas_non_corrected['original'].values()))
        m_dta_non_corrected_cor, s_dta_non_corrected_cor = mean_std(list(dtas_non_corrected['corrected'].values()))
        m_dta95_non_corrected_ori, s_dta95_non_corrected_ori = mean_std(list(dtas_95_non_corrected['original'].values()))
        m_dta95_non_corrected_cor, s_dta95_non_corrected_cor = mean_std(list(dtas_95_non_corrected['corrected'].values()))
        print(f"[SEG]:")        
        print(f"   DSC: Original: {m_dsc_ori:.4f} +- {s_dsc_ori:.4f}, Corrected: {m_dsc_cor:.4f} +- {s_dsc_cor:.4f}")
        print(f"   DTA: Original: {m_dta_ori:.4f} +- {s_dta_ori:.4f}, Corrected: {m_dta_cor:.4f} +- {s_dta_cor:.4f}")
        print(f"   DTA 95th perc.: Original: {m_dta95_ori:.4f} +- {s_dta95_ori:.4f}, Corrected: {m_dta95_cor:.4f} +- {s_dta95_cor:.4f}")
        print(f"   DTA corrected voxels: Original: {m_dta_corrected_ori:.4f} +- {s_dta_corrected_ori:.4f}, Corrected: {m_dta_corrected_cor:.4f} +- {s_dta_corrected_cor:.4f}")
        print(f"   DTA 95th perc. corrected voxels: Original: {m_dta95_corrected_ori:.4f} +- {s_dta95_corrected_ori:.4f}, Corrected: {m_dta95_corrected_cor:.4f} +- {s_dta95_corrected_cor:.4f}")
        print(f"   DTA non-corrected voxels: Original: {m_dta_non_corrected_ori:.4f} +- {s_dta_non_corrected_ori:.4f}, Corrected: {m_dta_non_corrected_cor:.4f} +- {s_dta_non_corrected_cor:.4f}")
        print(f"   DTA 95th perc. non-corrected voxels: Original: {m_dta95_non_corrected_ori:.4f} +- {s_dta95_non_corrected_ori:.4f}, Corrected: {m_dta95_non_corrected_ori:.4f} +- {s_dta95_non_corrected_ori:.4f}, Corrected: {m_dta95_non_corrected_cor:.4f} +- {s_dta95_non_corrected_cor:.4f}")

        seg_metrics_save = {
            "dscs": dscs,
            "dtas": dtas,
            "dtas_95": dtas_95,
            "dtas_corrected": dtas_corrected,
            "dtas_95_corrected": dtas_95_corrected,
            "dtas_non_corrected": dtas_non_corrected,
            "dtas_95_non_corrected": dtas_95_non_corrected,
            "dtas_corrected_perc0": dtas_corrected_perc[0],
            "dtas_corrected_perc1": dtas_corrected_perc[1],
            "dtas_corrected_perc2": dtas_corrected_perc[2]
        }

        if diffusion is not None:
            filename = f"seg_metrics_reviewed_{args['dataset']}_{args['test_sampling']}_aggreg.json"
        else:
            filename = f"seg_metrics_reviewed_{args['dataset']}.json"

        with open(os.path.join(args["path_save"], filename), "w") as f:
            json.dump(seg_metrics_save, f)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    args = parse_args()

    testing_dataset_original, testing_dataset_corrected = dataset.init_reviewed_datasets(args)
    testing_dataset_original_loader = dataset.init_dataset_loader(testing_dataset_original, args, shuffle=False)
    testing_dataset_corrected_loader = dataset.init_dataset_loader(testing_dataset_corrected, args, shuffle=False)

    output = torch.load(os.path.join(args["path_save"], "model", "model_final.pt"), map_location=device, weights_only=False)

    test_ts = [700, 800, 900, 1000]
    if args['model'] == 'VAE': test_ts = [0]

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
            separate_seg_schedule=args['separate_seg_schedule']
        )

        ema.load_state_dict(output["ema"], strict=False)
        ema.to(device)
        ema.eval()

        testing_reviewed_aggreg(testing_dataset_original_loader, testing_dataset_corrected_loader,
                                args, diffusion=diff, ema=ema, vae=None, test_ts=test_ts, device=device)

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

        testing_reviewed_aggreg(testing_dataset_original_loader, testing_dataset_corrected_loader,
                                args, diffusion=None, ema=None, vae=model, test_ts=test_ts, device=device)

if __name__ == '__main__':
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    
    main()
