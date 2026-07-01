import sys 
import os
import copy
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE" 
os.environ["TF_ENABLE_ONEDNN_OPTS"]="0"
import random
import json

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from tqdm import trange

from monai.networks.nets import VarAutoEncoder

from helpers import save, plot_loss_history

def loss_function(recon_loss, recon_x, x, mu, log_var, beta):
    if recon_loss == "bce":
        recon_error = F.binary_cross_entropy(recon_x, x, reduction='sum')
    elif recon_loss == "mse":
        recon_error = F.mse_loss(recon_x, x, reduction='sum')

    kld = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return recon_error + beta * kld, recon_error.item(), kld.item()

def trainer_vae(training_dataset_loader, validation_dataset_loader, args, device):
    """
    Trainer
    """

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

    if args['load_model'] is not None:
        checkpoint = torch.load(args['load_model'], map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"], strict=False)

        print(f"Loaded model from {args['load_model']}")

    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=args["learning_rate"], weight_decay=args["l2"])

    # loss history 
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
    best_optim_state = None

    # retrain the model with the selected layer
    epoch = 0
    t = trange(args["epochs"], leave=False, desc="epoch 0, average train loss: ?, validation loss: ?")
    for epoch in t:
        model.train()

        beta_epoch = args["beta_start"] + (args["beta_end"] - args["beta_start"]) * (epoch / args["annealing_max_epochs"])
        if beta_epoch > args["beta_end"]:
            beta_epoch = args["beta_end"]

        epoch_loss = 0
        epoch_img_loss = 0
        epoch_seg_loss = 0

        for batch_data in training_dataset_loader:
            inputs = batch_data["im"].to(device)
            optimizer.zero_grad()
            recon_batch, mu, log_var, _ = model(inputs)
            batch_loss, _, _ = loss_function(args["recon_loss"], recon_batch, inputs, mu, log_var, beta_epoch)
            if args["num_channels"] == 2:
                batch_img_loss, _, _ = loss_function(args["recon_loss"], recon_batch[:, 0:1], inputs[:, 0:1], mu, log_var, beta_epoch)
                batch_seg_loss, _, _ = loss_function(args["recon_loss"], recon_batch[:, 1:2], inputs[:, 1:2], mu, log_var, beta_epoch)
                epoch_img_loss += batch_img_loss.item()
                epoch_seg_loss += batch_seg_loss.item()        
            batch_loss.backward()
            optimizer.step()
            epoch_loss += batch_loss.item()         

        loss_hist["train_total"].append(epoch_loss / len(training_dataset_loader.dataset))
        if args["num_channels"] == 2:
            loss_hist["train_img"].append(epoch_img_loss / len(training_dataset_loader.dataset))
            loss_hist["train_seg"].append(epoch_seg_loss / len(training_dataset_loader.dataset))

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_data in validation_dataset_loader:
                inputs = batch_data["im"].to(device)
                recon, mu, log_var, _ = model(inputs)
                batch_loss, _, _ = loss_function(args["recon_loss"], recon, inputs, mu, log_var, beta_epoch)
                val_loss += batch_loss.item() 
        loss_hist["val_total"].append(val_loss / len(validation_dataset_loader.dataset))

        improved = val_loss < best_val - 1e-8  # tiny tolerance
        if improved:
            best_val = val_loss
            best_epoch = epoch
            epochs_no_improve = 0
            # snapshot best states
            best_model_state = copy.deepcopy(model.state_dict())
            best_optim_state = copy.deepcopy(optimizer.state_dict())
            # save "best" checkpoint immediately
            save(model=model, args=args, optimizer=optimizer, final=False, ema=None, epoch=epoch)
        else:
            epochs_no_improve += 1

        # ---- check early stopping ----
        if epochs_no_improve >= args["patience"]:
            print(f"\nEarly stopping at epoch {epoch} (best val {best_val:.4f} @ {best_epoch}).")
            # restore best weights before exiting the loop
            if best_model_state is not None:
                model.load_state_dict(best_model_state)
            if best_optim_state is not None:
                optimizer.load_state_dict(best_optim_state)
            break  # exit epoch loop

        t.set_description( 
            f"epoch {epoch + 1}, train loss: {loss_hist['train_total'][-1]:.6f}, val loss: {loss_hist['val_total'][-1]:.6f}"
        )

    # Save final model and loss history
    save(model=model, args=args, optimizer=optimizer, final=True, ema=None)
    plot_loss_history(loss_hist, args)
    np.save(os.path.join(args["path_save"], "model", "loss_history.npy"), loss_hist)
