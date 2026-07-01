import os
import copy
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import ListedColormap

import torch
import torch.nn.functional as F
from monai.networks.nets import VarAutoEncoder

import dataset
from Diffusion import DiffusionModel, get_beta_schedule
from UNet3D import UNetModel, ConditionalSegUNetModel
from helpers import parse_args

from evaluation import dice_per_sample, mse_per_sample, preblend_over, error_distances_map, plot

# Simulated errors functions
def brainstem_top(inputs, expansion_size, channel, device):
    inputs_err = inputs.clone()

    for im in range(inputs_err.shape[0]):
        if expansion_size == 0:
            break
        mask_brainstem = inputs_err[im, channel]
        _, _, structure_rows = torch.where(mask_brainstem == 1)
        highest_row = structure_rows.max().item()

        if expansion_size < 0:
            mask_brainstem[:, :, highest_row - abs(expansion_size) + 1:highest_row + 1] = 0
        elif expansion_size > 0:
            com_brainstem = torch.mean(inputs_err[im, channel].nonzero().float(), dim=0)
            index = int(com_brainstem[2].item())

            if highest_row + expansion_size + 1 > inputs_err.shape[4]:
                mod_expansion_size = expansion_size - (highest_row + expansion_size + 1 - inputs_err.shape[4])
                mask_brainstem[:, :, index + mod_expansion_size + 1:highest_row + mod_expansion_size + 1] = mask_brainstem[:, :, index + 1:highest_row + 1] 
            else:
                mask_brainstem[:, :, index + expansion_size + 1:highest_row + expansion_size + 1] = mask_brainstem[:, :, index + 1:highest_row + 1] 

        inputs_err[im, channel] = mask_brainstem

    return inputs_err

def brainstem_bottom(inputs, expansion_size, channel, device):
    inputs_err = inputs.clone()

    for im in range(inputs_err.shape[0]):
        if expansion_size == 0:
            break
        mask = inputs_err[im, channel]
        _, _, structure_rows = torch.where(mask == 1)
        boundary_row = structure_rows.min().item()
        shifted_rows = abs(expansion_size)
        if expansion_size < 0:
            for row in range(boundary_row, boundary_row + shifted_rows):
                mask[:, :, row] = 0
        else:
            for row in range(boundary_row - shifted_rows, boundary_row):
                mask[:, :, row] = mask[:, :, boundary_row]

        inputs_err[im, channel] = mask

    return inputs_err

def spinalcord_bottom(inputs, expansion_size, channel, device):
    inputs_err = inputs.clone()

    for im in range(inputs_err.shape[0]):
        mask_spin = inputs_err[im, channel] 
        _, _, structure_rows = torch.where(mask_spin == 1)
        lowest_row = structure_rows.min().item()
        mask_spin[:, :, lowest_row:lowest_row+abs(expansion_size)] = 0
        inputs_err[im, channel] = mask_spin

    return inputs_err

def spinalcord_top(inputs, expansion_size, channel, device):
    inputs_err = inputs.clone()

    for im in range(inputs_err.shape[0]):
        mask_spin = inputs_err[im, channel]
        _, _, structure_rows = torch.where(mask_spin == 1)
        boundary_row = structure_rows.max().item()
        shifted_rows = abs(expansion_size)
        if expansion_size < 0:
            for row in range(boundary_row, boundary_row + shifted_rows + 1):
                if row >= mask_spin.shape[2]:
                    break
                mask_spin[:, :, row] = mask_spin[:, :, boundary_row]
        else:
            for row in range(boundary_row - shifted_rows + 1, boundary_row + 1):
                mask_spin[:, :, row] = 0

        inputs_err[im, channel] = mask_spin

    return inputs_err

def spinalcord_width(inputs, expansion_size, channel, device):
    inputs_err = inputs.clone()
    kernel_size = 2 * abs(expansion_size) + 1
    kernel = torch.ones((1, 1, kernel_size, kernel_size, 1), dtype=torch.float32, device=device)
    for im in range(inputs_err.shape[0]):
        if expansion_size == 0:
            break
        mask_spinalcord = inputs_err[im, channel].unsqueeze(0).unsqueeze(0).float()  # spinalcord mask 
        mask_spinalcord = F.conv3d(mask_spinalcord, kernel, padding='same')
        if expansion_size > 0:    
            mask_spinalcord = (mask_spinalcord > 0).squeeze(0).squeeze(0).float()
        else:
            mask_spinalcord = (mask_spinalcord == kernel.numel()).squeeze(0).squeeze(0).float()
        inputs_err[im, channel] = mask_spinalcord

    return inputs_err

def parotid_erod_dil(inputs, size, channel, device):
    inputs_err = inputs.clone()

    radius = abs(int(size))
    # 18-connected footprint: 3x3x3 cube minus 8 corners
    kernel = torch.ones((3, 3, 3), device=inputs.device, dtype=inputs.dtype)
    kernel[0, 0, 0] = kernel[0, 0, 2] = kernel[0, 2, 0] = kernel[0, 2, 2] = 0
    kernel[2, 0, 0] = kernel[2, 0, 2] = kernel[2, 2, 0] = kernel[2, 2, 2] = 0
    kernel = kernel.view(1, 1, 3, 3, 3)

    for im in range(inputs_err.shape[0]):
        mask = inputs_err[im, channel].unsqueeze(0).unsqueeze(0)

        morphed = mask
        for _ in range(radius):
            if size > 0:
                # dilation: any 18-connected neighbour is foreground
                morphed = (F.conv3d(morphed, kernel, padding=1) > 0).to(inputs.dtype)
            else:
                # erosion: all 18-connected neighbours are foreground
                morphed = (F.conv3d(morphed, kernel, padding=1) == kernel.sum()).to(inputs.dtype)

        inputs_err[im, channel] = morphed.squeeze(0).squeeze(0)

    return inputs_err

def testing_simulated_aggreg(testing_dataset_loader, args, diffusion, ema, vae, 
                             test_ts, error_func, step, device):

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
        img_dir = f"testing-images-simulated-v2-{args['test_sampling']}"
        os.makedirs(os.path.join(args["path_save"], img_dir), exist_ok=True)
    elif vae is not None:
        vae.eval()
        num_saved_batches = 1
        img_dir = "testing-images-simulated-v2"
        os.makedirs(os.path.join(args["path_save"], img_dir), exist_ok=True)        
    
    mses = {i: {} for i in ['original', 'simulated']}
    dscs = {i: {} for i in ['original', 'simulated']}
    dtas = {i: {} for i in ['original', 'simulated']}
    dtas_95 = {i: {} for i in ['original', 'simulated']}
    dtas_simulated = {i: {} for i in ['original', 'simulated']}
    dtas_95_simulated = {i: {} for i in ['original', 'simulated']}
    dtas_non_simulated = {i: {} for i in ['original', 'simulated']}
    dtas_95_non_simulated = {i: {} for i in ['original', 'simulated']}
    dtas_simulated_perc = {k: {i: {} for i in ['original', 'simulated']} for k in [0, 1, 2]}

    for batch_idx, data in enumerate(testing_dataset_loader):
        x_ori = data["im"].to(device)
        keys = data["key"]
        x_err = error_func(x_ori, step, seg_idx, device)

        xrec_ori_aggreg = []
        xrec_err_aggreg = []

        for test_t in test_ts:
            if diffusion is not None:
                if args['test_sampling'] == 'ddpm':
                    xrec_ori = diffusion.forward_backward(
                        ema, 
                        x_ori, 
                        see_whole_sequence=None, 
                        t_distance=test_t)
                    xrec_err = diffusion.forward_backward(
                        ema, 
                        x_err, 
                        see_whole_sequence=None, 
                        t_distance=test_t)
                else:
                    if args["model"] == "DDPMCondSeg":
                        xrec_ori, _ = diffusion.forward_backward_ddim_conditional_seg(
                            ema,
                            x_ori,
                            see_whole_sequence=None,
                            t_distance=test_t,
                            ddim_steps=args["ddim_steps"],
                        )
                        xrec_err, _ = diffusion.forward_backward_ddim_conditional_seg(
                            ema,
                            x_err,
                            see_whole_sequence=None,
                            t_distance=test_t,
                            ddim_steps=args["ddim_steps"],     
                        )
                    else:
                        xrec_ori, _ = diffusion.forward_backward_ddim(
                            ema,
                            x_ori,
                            see_whole_sequence=None,
                            t_distance=test_t,
                            ddim_steps=args["ddim_steps"],
                            eta=args["ddim_eta"],
                        )
                        xrec_err, _ = diffusion.forward_backward_ddim(
                            ema,
                            x_err,
                            see_whole_sequence=None,
                            t_distance=test_t,
                            ddim_steps=args["ddim_steps"],  
                            eta=args["ddim_eta"],         
                        )
            elif vae is not None:
                with torch.no_grad():
                    xrec_ori, _, _, _ = vae(x_ori)
                    xrec_err, _, _, _ = vae(x_err)

            xrec_ori_aggreg.append(xrec_ori)
            xrec_err_aggreg.append(xrec_err)

        xrec_ori = torch.mean(torch.stack(xrec_ori_aggreg, dim=0), dim=0)
        xrec_err = torch.mean(torch.stack(xrec_err_aggreg, dim=0), dim=0)
                
        for im in range(len(keys)):
            if has_image:
                x_ori_img = x_ori[im][img_idx]
                x_err_img = x_err[im][img_idx] 
                xrec_ori_img = xrec_ori[im][img_idx]
                xrec_err_img = xrec_err[im][img_idx] 
                mses['original'][keys[im]] = float(mse_per_sample(xrec_ori_img, x_ori_img))
                mses['simulated'][keys[im]] = float(mse_per_sample(xrec_err_img, x_err_img))

            if has_seg:
                x_ori_seg = x_ori[im][seg_idx] 
                x_err_seg = x_err[im][seg_idx] 
                xrec_ori_seg = xrec_ori[im][seg_idx]
                xrec_err_seg = xrec_err[im][seg_idx] 
                xrec_ori_seg_bin = (xrec_ori_seg >= 0.5).float()
                xrec_err_seg_bin = (xrec_err_seg >= 0.5).float()
                dscs['original'][keys[im]] = float(dice_per_sample(xrec_ori_seg_bin, x_ori_seg).mean().detach().cpu())
                dscs['simulated'][keys[im]] = float(dice_per_sample(xrec_err_seg_bin, x_err_seg).mean().detach().cpu())

                union = ((x_ori_seg.cpu().detach().numpy() > 0) | (xrec_ori_seg_bin.cpu().detach().numpy() > 0) | (x_err_seg.cpu().detach().numpy() > 0) | (xrec_err_seg_bin.cpu().detach().numpy() > 0))
                simulated_voxels = ((x_err_seg.cpu().detach().numpy() == 1) & (x_ori_seg.cpu().detach().numpy() == 0)) | ((x_err_seg.cpu().detach().numpy() == 0) & (x_ori_seg.cpu().detach().numpy() == 1))
                non_simulated_voxels = union & ~simulated_voxels

                dmap_vals_ori = error_distances_map(x_ori_seg.cpu().detach().numpy(), xrec_ori_seg_bin.cpu().detach().numpy())
                dmap_vals_err = error_distances_map(x_err_seg.cpu().detach().numpy(), xrec_err_seg_bin.cpu().detach().numpy())
                vals_ori = dmap_vals_ori[union]
                vals_err = dmap_vals_err[union]
                vals_ori_simulated = dmap_vals_ori[simulated_voxels]
                vals_err_simulated = dmap_vals_err[simulated_voxels]
                vals_ori_non_simulated = dmap_vals_ori[non_simulated_voxels]
                vals_err_non_simulated = dmap_vals_err[non_simulated_voxels]

                dtas['original'][keys[im]] = float(np.nanmean(vals_ori))
                dtas['simulated'][keys[im]] = float(np.nanmean(vals_err))
                dtas_95['original'][keys[im]] = float(np.nanpercentile(vals_ori, 95))
                dtas_95['simulated'][keys[im]] = float(np.nanpercentile(vals_err, 95))
                dtas_simulated['original'][keys[im]] = float(np.nanmean(vals_ori_simulated)) if (np.sum(simulated_voxels) > 0 and np.nansum(vals_ori_simulated) > 0.) else float(0)
                dtas_simulated['simulated'][keys[im]] = float(np.nanmean(vals_err_simulated)) if (np.sum(simulated_voxels) > 0 and np.nansum(vals_err_simulated) > 0.) else float(0)
                dtas_95_simulated['original'][keys[im]] = float(np.nanpercentile(vals_ori_simulated, 95)) if (np.sum(simulated_voxels) > 0 and np.nansum(vals_ori_simulated) > 0.) else float(0)
                dtas_95_simulated['simulated'][keys[im]] = float(np.nanpercentile(vals_err_simulated, 95)) if (np.sum(simulated_voxels) > 0 and np.nansum(vals_err_simulated) > 0.) else float(0)
                dtas_non_simulated['original'][keys[im]] = float(np.nanmean(vals_ori_non_simulated)) if (np.sum(non_simulated_voxels) > 0 and np.nansum(vals_ori_non_simulated) > 0.) else float(0)
                dtas_non_simulated['simulated'][keys[im]] = float(np.nanmean(vals_err_non_simulated)) if (np.sum(non_simulated_voxels) > 0 and np.nansum(vals_err_non_simulated) > 0.) else float(0)
                dtas_95_non_simulated['original'][keys[im]] = float(np.nanpercentile(vals_ori_non_simulated, 95)) if (np.sum(non_simulated_voxels) > 0 and np.nansum(vals_ori_non_simulated) > 0.) else float(0)
                dtas_95_non_simulated['simulated'][keys[im]] = float(np.nanpercentile(vals_err_non_simulated, 95)) if (np.sum(non_simulated_voxels) > 0 and np.nansum(vals_err_non_simulated) > 0.) else float(0)

                for threshold in (0, 1, 2):
                    n_ori_all = np.sum(vals_ori > threshold)
                    n_ori_simulated = np.sum(vals_ori_simulated > threshold)
                    dtas_simulated_perc[threshold]['original'][keys[im]] = float(n_ori_simulated / n_ori_all) if n_ori_all > 0 else float('nan')

                    n_err_all = np.sum(vals_err > threshold)
                    n_err_simulated = np.sum(vals_err_simulated > threshold)
                    dtas_simulated_perc[threshold]['simulated'][keys[im]] = float(n_err_simulated / n_err_all) if n_err_all > 0 else float('nan')

                dmap_to_plot = np.nan_to_num(dmap_vals_err, nan=0.0) 

            if batch_idx < num_saved_batches:
                if diffusion is not None:
                    plot_name = f"test_{keys[im]}_t_aggreg_{error_func.__name__}_{step}.png"
                    plot_title = f"TEST t-aggregated={test_ts}, {error_func.__name__}, {step}"
                else:
                    plot_name = f"test_{keys[im]}_{error_func.__name__}_{step}.png"
                    plot_title = f"TEST {error_func.__name__}, {step}"

                if has_image and has_seg:
                    if args['organ_channel'] == 'brainstem' or args['organ_channel'] == 'spinalcord':
                        index_slice = int(np.mean(np.argwhere(x_err_seg.cpu().detach().numpy()), axis=0)[0]) if x_err_seg.cpu().detach().numpy().sum() > 0 else x_err_seg.shape[0] // 2
                        plot(
                            x_err_img[index_slice].detach().cpu(),
                            xrec_err_img[index_slice].detach().cpu(),
                            x_err_seg[index_slice].detach().cpu(),
                            xrec_err_seg_bin[index_slice].detach().cpu(),
                            dmap_to_plot[index_slice],
                            seg_no_error=x_ori_seg[index_slice].detach().cpu(),
                            title=plot_title,
                            path=os.path.join(args["path_save"], img_dir, plot_name),
                            plot_labels=["x₀", "pred x̂₀", 'DTA map'],
                        )
                    elif args['organ_channel'] == 'parotid_l' or args['organ_channel'] == 'parotid_r':
                        index_slice = int(np.mean(np.argwhere(x_err_seg.cpu().detach().numpy()), axis=0)[2]) if x_err_seg.cpu().detach().numpy().sum() > 0 else x_err_seg.shape[2] // 2
                        plot(
                            x_err_img[:, :, index_slice].detach().cpu(),
                            xrec_err_img[:, :, index_slice].detach().cpu(),
                            x_err_seg[:, :, index_slice].detach().cpu(),
                            xrec_err_seg_bin[:, :, index_slice].detach().cpu(),
                            dmap_to_plot[:, :, index_slice],
                            seg_no_error=x_ori_seg[:, :, index_slice].detach().cpu(),
                            title=plot_title,
                            path=os.path.join(args["path_save"], img_dir, plot_name),
                            plot_labels=["x₀", "pred x̂₀", 'DTA map'],
                        )

                elif has_seg:
                    if args['organ_channel'] == 'brainstem' or args['organ_channel'] == 'spinalcord':
                        index_slice = int(np.mean(np.argwhere(x_err_seg.cpu().detach().numpy()), axis=0)[0]) if x_err_seg.cpu().detach().numpy().sum() > 0 else x_err_seg.shape[0] // 2
                        plot(
                            None,
                            None,
                            x_err_seg[index_slice].detach().cpu(),
                            xrec_err_seg_bin[index_slice].detach().cpu(),
                            dmap_to_plot[index_slice],
                            seg_no_error=x_ori_seg[index_slice].detach().cpu(),
                            title=plot_title,
                            path=os.path.join(args["path_save"], img_dir, plot_name),
                            plot_labels=["x₀", "pred x̂₀", 'DTA map'],
                        )
                    elif args['organ_channel'] == 'parotid_l' or args['organ_channel'] == 'parotid_r':
                        index_slice = int(np.mean(np.argwhere(x_err_seg.cpu().detach().numpy()), axis=0)[2]) if x_err_seg.cpu().detach().numpy().sum() > 0 else x_err_seg.shape[2] // 2
                        plot(
                            None,
                            None,
                            x_err_seg[:, :, index_slice].detach().cpu(),
                            xrec_err_seg_bin[:, :, index_slice].detach().cpu(),
                            dmap_to_plot[:, :, index_slice],
                            seg_no_error=x_ori_seg[:, :, index_slice].detach().cpu(),
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
        m_err, s_err = mean_std(list(mses['simulated'].values()))
        print(f"[IMAGE] MSE {error_func.__name__}, {step}: Original: {m_ori:.4f} +- {s_ori:.4f}, Simulated: {m_err:.4f} +- {s_err:.4f}")

        img_metrics_save = {
            "error_func": error_func.__name__,
            "step": step,
            "mses": mses
        }
        if diffusion is not None:
            filename = f"img_metrics_simulated_v2_{args['dataset']}_{args['test_sampling']}_aggreg.json"
        else:
            filename = f"img_metrics_simulated_v2_{args['dataset']}.json"

        with open(os.path.join(args["path_save"], filename), 'a') as f:
            f.write(json.dumps(img_metrics_save) + "\n")

    if has_seg:
        m_dsc_ori, s_dsc_ori = mean_std(list(dscs['original'].values()))
        m_dsc_err, s_dsc_err = mean_std(list(dscs['simulated'].values()))
        m_dta_ori, s_dta_ori = mean_std(list(dtas['original'].values()))
        m_dta_err, s_dta_err = mean_std(list(dtas['simulated'].values()))
        m_dta95_ori, s_dta95_ori = mean_std(list(dtas_95['original'].values()))
        m_dta95_err, s_dta95_err = mean_std(list(dtas_95['simulated'].values()))
        m_dta_simulated_ori, s_dta_simulated_ori = mean_std(list(dtas_simulated['original'].values()))
        m_dta_simulated_err, s_dta_simulated_err = mean_std(list(dtas_simulated['simulated'].values()))
        m_dta95_simulated_ori, s_dta95_simulated_ori = mean_std(list(dtas_95_simulated['original'].values()))
        m_dta95_simulated_err, s_dta95_simulated_err = mean_std(list(dtas_95_simulated['simulated'].values()))  
        m_dta_non_simulated_ori, s_dta_non_simulated_ori = mean_std(list(dtas_non_simulated['original'].values()))
        m_dta_non_simulated_err, s_dta_non_simulated_err = mean_std(list(dtas_non_simulated['simulated'].values()))
        m_dta95_non_simulated_ori, s_dta95_non_simulated_ori = mean_std(list(dtas_95_non_simulated['original'].values()))
        m_dta95_non_simulated_err, s_dta95_non_simulated_err = mean_std(list(dtas_95_non_simulated['simulated'].values()))
        print(f"[SEG] {error_func.__name__}, {step}:")        
        print(f"   DSC: Original: {m_dsc_ori:.4f} +- {s_dsc_ori:.4f}, Simulated: {m_dsc_err:.4f} +- {s_dsc_err:.4f}")
        print(f"   DTA: Original: {m_dta_ori:.4f} +- {s_dta_ori:.4f}, Simulated: {m_dta_err:.4f} +- {s_dta_err:.4f}")
        print(f"   DTA 95th perc.: Original: {m_dta95_ori:.4f} +- {s_dta95_ori:.4f}, Simulated: {m_dta95_err:.4f} +- {s_dta95_err:.4f}")
        print(f"   DTA simulated voxels: Original: {m_dta_simulated_ori:.4f} +- {s_dta_simulated_ori:.4f}, Simulated: {m_dta_simulated_err:.4f} +- {s_dta_simulated_err:.4f}")
        print(f"   DTA 95th perc. simulated voxels: Original: {m_dta95_simulated_ori:.4f} +- {s_dta95_simulated_ori:.4f}, Simulated: {m_dta95_simulated_err:.4f} +- {s_dta95_simulated_err:.4f}")
        print(f"   DTA non-simulated voxels: Original: {m_dta_non_simulated_ori:.4f} +- {s_dta_non_simulated_ori:.4f}, Simulated: {m_dta_non_simulated_err:.4f} +- {s_dta_non_simulated_err:.4f}")
        print(f"   DTA 95th perc. non-simulated voxels: Original: {m_dta95_non_simulated_ori:.4f} +- {s_dta95_non_simulated_ori:.4f}, Simulated: {m_dta95_non_simulated_err:.4f} +- {s_dta95_non_simulated_err:.4f}")

        seg_metrics_save = {
            "error_func": error_func.__name__,
            "step": step,
            "dscs": dscs,
            "dtas": dtas,
            "dtas_95": dtas_95,
            "dtas_simulated": dtas_simulated,
            "dtas_95_simulated": dtas_95_simulated,
            "dtas_non_simulated": dtas_non_simulated,
            "dtas_95_non_simulated": dtas_95_non_simulated,
            "dtas_simulated_perc0": dtas_simulated_perc[0],
            "dtas_simulated_perc1": dtas_simulated_perc[1],
            "dtas_simulated_perc2": dtas_simulated_perc[2],
        }

        if diffusion is not None:
            filename = f"seg_metrics_simulated_v2_{args['dataset']}_{args['test_sampling']}_aggreg.json"
        else:
            filename = f"seg_metrics_simulated_v2_{args['dataset']}.json"

        with open(os.path.join(args["path_save"], filename), 'a') as f:
            f.write(json.dumps(seg_metrics_save) + "\n")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    args = parse_args()

    _, _, testing_dataset = dataset.init_datasets(args, debug=args["debug"], test_only=True)
    testing_dataset_loader = dataset.init_dataset_loader(testing_dataset, args, shuffle=False)

    output = torch.load(os.path.join(args["path_save"], "model", "model_final.pt"), map_location=device, weights_only=False)

    if args["organ_channel"] == "brainstem":
        variables = {
            "simulated_error_func": ["brainstem_top", "brainstem_bottom"],
            "steps": {
                "brainstem_top": [-5, -3, -2, -1, 1, 2, 3, 5], # steps top
                "brainstem_bottom": [-5, -3, -2, -1, 1, 2, 3, 5], # steps bottom
            }
        }
        error_functions = {
            "brainstem_top": brainstem_top,
            "brainstem_bottom": brainstem_bottom,
        }
    elif args["organ_channel"] == "spinalcord":
        variables = {
            "simulated_error_func": ["spinalcord_top", "spinalcord_width"],
            "steps": {
                "spinalcord_top": [-5, -3, -2, -1, 1, 2, 3, 5], # steps top
                "spinalcord_width": [1, 2], # steps width
            }
        }
        error_functions = {
            "spinalcord_top": spinalcord_top,
            "spinalcord_width": spinalcord_width,
        }
    elif args["organ_channel"] == "parotid_l" or args["organ_channel"] == "parotid_r":
        variables = {
            "simulated_error_func": ["parotid_erod_dil"],
            "steps": {
                "no_error": [0], 
                "parotid_erod_dil": [-2, -1, 1, 2], # erosion/dilation sizes
            }
        }
        error_functions = {
            "parotid_erod_dil": parotid_erod_dil,
        }
        
    else:
        raise Exception("Invalid organ channel")
    
    test_ts = [700, 800, 900, 1000]
    if args['model'] == 'VAE': test_ts = [0]
    
    array_id = os.getenv("EVAL_TASK_ID", os.getenv("SGE_TASK_ID"))
    try:
        array_id = int(array_id)
    except:
        array_id = 1  # default to 1 for testing

    idx = array_id - 1

    func_step_pairs = []
    for func in variables["simulated_error_func"]:
        for step in variables["steps"][func]:
            func_step_pairs.append((func, step))

    n_pairs = len(func_step_pairs)

    pair_index = idx % n_pairs
    simulated_error_func, step = func_step_pairs[pair_index]

    print(f"Running evaluation for simulated_error_func={simulated_error_func}, step={step}")

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

        testing_simulated_aggreg(testing_dataset_loader, args, diffusion=diff, ema=ema, 
                                 vae=None, test_ts=test_ts, error_func=error_functions[simulated_error_func], 
                                 step=step, device=device)

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

        testing_simulated_aggreg(testing_dataset_loader, args, diffusion=None, ema=None, 
                                 vae=model, test_ts=test_ts, error_func=error_functions[simulated_error_func], 
                                 step=step, device=device)

if __name__ == '__main__':
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    
    main()
