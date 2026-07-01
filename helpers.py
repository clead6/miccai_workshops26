import os
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser()

    # model type
    parser.add_argument("--model", type=str, help="DDPM, DDPMCondSeg or VAE")

    # image parameters
    parser.add_argument("--patch_size", type=int, nargs=3, default=[64, 64, 64], help="Patch size (D H W)")

    # channels
    parser.add_argument("--num_channels", type=int, default=2)
    parser.add_argument("--organ_channel", type=str, default=None, help="Organ to include as additional channel (e.g., 'brainstem' or 'spinalcord')")

    # shared training parameters
    parser.add_argument("--epochs", type=int, default=3000)
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate for optimizer")

    # paths
    parser.add_argument("--arg_num", type=str, help="Current job number", required=True)
    parser.add_argument("--path_data", type=str, default='/SAN/medic/clead-phd2/May26_patches/files_data_radcure_isotropic/')
    parser.add_argument("--path_save", type=str, default='/SAN/medic/clead-phd2/May26_patches/outputs/')
    parser.add_argument("--dataset", type=str, default="radcure", help="Dataset name")
    parser.add_argument("--dataset_subset", type=float, default=1, help="Subset of dataset to use (for debugging)")

    # debug
    parser.add_argument("--debug", type=str2bool, default=False)

    # load existing model
    parser.add_argument("--load_model", type=str, default=None, help="Path to existing model to load (including filename)")

    if parser.parse_known_args()[0].model == "DDPM" or parser.parse_known_args()[0].model == "DDPMCondSeg":
        # training parameters
        parser.add_argument("--batch_size", type=int, default=8)
        parser.add_argument("--weight_decay", type=float, default=0.0)
        parser.add_argument("--patience", type=int, default=30)

        # diffusion parameters
        parser.add_argument("--num_diffusion_steps", type=int, default=1000)
        parser.add_argument("--beta_schedule", type=str, default="cosine")
        parser.add_argument("--beta_schedule_gamma", type=float, default=1.0)
        parser.add_argument("--noise_fn", type=str, default="gauss")
        parser.add_argument("--separate_seg_schedule", type=str2bool, default=False, help="Whether to use a separate noise schedule for the segmentation channel")

        # model architecture
        parser.add_argument("--init_channels", type=int, default=64)
        parser.add_argument("--channel_mults", type=str, default="1,2,4", help="Comma-separated channel multipliers")
        parser.add_argument("--num_res_blocks", type=int, default=1)
        parser.add_argument("--attention_resolutions", type=str, default="8")
        parser.add_argument("--num_heads", type=int, default=4)
        parser.add_argument("--dropout", type=float, default=0.2)

        # loss
        parser.add_argument("--loss_types", type=str, nargs="+", default=["hybrid", "hybrid_seg"])
        parser.add_argument("--loss_weight", type=str, default="none")
        parser.add_argument("--seg_weight", type=float, default=1.0)

        # testing parameters
        parser.add_argument("--test_sampling", type=str, default="ddim")
        parser.add_argument("--ddim_steps", type=int, default=50)
        parser.add_argument("--ddim_eta", type=float, default=0.0)

        # saving/logging
        parser.add_argument("--save_metrics", type=str2bool, default=True)
        parser.add_argument("--save_imgs", type=str2bool, default=True)
        parser.add_argument("--save_vids", type=str2bool, default=True)

    elif parser.parse_known_args()[0].model == "VAE":
        # parameters
        parser.add_argument("--beta_start", type=float, default=0.0, help="Starting value of beta for VAE loss")
        parser.add_argument("--beta_end", type=float, default=1.0, help="Ending value of beta for VAE loss")
        parser.add_argument("--latent_size", type=int, default=64, help="Dimensionality of the latent space")
        parser.add_argument("--model_num", type=int, default=3, help="Model architecture")
        parser.add_argument("--recon_loss", type=str, default="mse", help="Reconstruction loss type (e.g., 'mse', 'l1')")
        parser.add_argument("--annealing_max_epochs", type=int, default=100, help="Number of epochs over which to anneal beta from beta_start to beta_end")
        parser.add_argument("--batch_size", type=int, default=128, help="Batch size for training")
        parser.add_argument("--num_workers", type=int, default=0, help="Number of worker processes for data loading")
        parser.add_argument("--drop_rate", type=float, default=0.2, help="Dropout rate for the model")
        parser.add_argument("--l2", type=float, default=1e-4, help="L2 regularization weight")
        parser.add_argument("--num_res_units", type=int, default=0, help="Number of residual units in the model")
        parser.add_argument("--patience", type=int, default=50)

    else:
        raise ValueError("Model must be either 'DDPM' or 'VAE'")

    # get all the args
    args = parser.parse_args()
    args = vars(args) # convert to dict

    # channels
    if args["num_channels"] == 2:
        if args["organ_channel"] is None:
            raise ValueError("organ_channel must be specified when num_channels is 2")
        args["labels_channels"] = ["image", args["organ_channel"]]
    elif args["num_channels"] == 1:
        if args["organ_channel"] is None:
            args["labels_channels"] = ["image"]
        else:
            args["labels_channels"] = [args["organ_channel"]]
    else:
        raise ValueError("num_channels must be either 1 or 2")

    # VAE model architecture
    if args["model"] == "VAE":
        models = {
            1: {
                "channels": (16, 32, 64, 128),
                "strides": (2, 2, 2, 2)
            },
            2: {
                "channels": (16, 32, 64, 64, 128, 128),
                "strides": (1, 2, 1, 2, 1, 2)
            },
            3: {
                "channels": (16, 32, 64, 64, 128, 128, 256, 256),
                "strides": (1, 2, 1, 2, 1, 2, 1, 2)
            },
            4: {
                "channels": (16, 32, 64, 64, 128, 128, 256, 256, 512, 512),
                "strides": (1, 2, 1, 2, 1, 2, 1, 2, 1, 2)
            },
            5: {
                "channels": (16, 32, 64, 64, 128, 128, 256, 256, 512, 512, 1024, 1024),
                "strides": (1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2)
            }
        }
        args["model_channels"] = models[args["model_num"]]["channels"]
        args["model_strides"] = models[args["model_num"]]["strides"]

    # debug
    if args["debug"]:
        args["epochs"] = 2

    # print args
    for key, value in args.items():
        print(f"{key}: {value}")

    os.makedirs(args["path_save"], exist_ok=True)
    args["path_save"] = os.path.join(args["path_save"], args["arg_num"])
    os.makedirs(args["path_save"], exist_ok=True)

    return args

def save(final, model, optimizer, args, ema=None, loss=0, epoch=0):
    """
    Save model final or checkpoint
    """
    if final:
        torch.save(
                {
                    'n_epoch':              args["epochs"],
                    'model_state_dict':     model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    "ema":                  ema.state_dict() if ema is not None else None,
                    "args":                 args
                    # 'loss': LOSS,
                    }, os.path.join(args["path_save"], "model", 'model_final.pt')
                )
    else:
        torch.save(
                {
                    'n_epoch':              epoch,
                    'model_state_dict':     model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    "args":                 args,
                    "ema":                  ema.state_dict() if ema is not None else None,
                    'loss':                 loss,
                    }, os.path.join(args["path_save"], "model", "checkpoint", f'model_epoch{epoch}.pt')
                )

def plot_loss_history(loss_hist, args, out_name="loss_curves.png"):
    """
    Plot training / validation loss curves.

    Parameters
    ----------
    loss_hist : dict
        Dictionary with keys like:
            - "train_total"
            - "train_img"
            - "train_seg"
            - "val_total"
        Values are lists (or arrays) over epochs.
        Missing or all-NaN curves are skipped automatically.

    args : dict
        Must contain args["path_save"]

    out_name : str
        Output filename (saved under args["path_save"]/model/)
    """

    def _get_curve(key):
        if key not in loss_hist:
            return None
        y = np.asarray(loss_hist[key], dtype=float)
        if y.size == 0 or np.all(np.isnan(y)):
            return None
        return y

    train_total = _get_curve("train_total")
    train_img   = _get_curve("train_img")
    train_seg   = _get_curve("train_seg")
    val_total   = _get_curve("val_total")

    curves = [c for c in [train_total, train_img, train_seg, val_total] if c is not None]
    if not curves:
        raise ValueError("plot_loss_history: no valid curves found in loss_hist")

    plt.figure(figsize=(10, 5))

    if train_total is not None:
        plt.plot(np.arange(len(train_total)), train_total, label="train_total")

    if val_total is not None:
        plt.plot(np.arange(len(val_total)), val_total, label="val_total")

    if train_img is not None:
        plt.plot(np.arange(len(train_img)), train_img, label="train_img")

    if train_seg is not None:
        plt.plot(np.arange(len(train_seg)), train_seg, label="train_seg")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training / Validation Loss Curves")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()

    out_dir = os.path.join(args["path_save"], "model")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, out_name)

    plt.savefig(out_path, dpi=200)
    plt.close()

def str2bool(v):
    if isinstance(v, bool):
        return v
    v = v.lower()
    if v in ("yes", "true", "t", "1", "y"):
        return True
    if v in ("no", "false", "f", "0", "n"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")

def main():
    pass

if __name__ == '__main__':
    main()
