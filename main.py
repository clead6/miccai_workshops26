import os
import random
import json
import numpy as np
import torch

import dataset
from trainer_ddpm import trainer_ddpm
from trainer_vae import trainer_vae
from helpers import parse_args

torch.cuda.empty_cache()

def main():
    """
    Load arguments, run training and testing functions, then remove checkpoint directory
    """

    args = parse_args()

    # make training directories
    os.makedirs(os.path.join(args["path_save"], "model"), exist_ok=True)
    os.makedirs(os.path.join(args["path_save"], "model", "checkpoint"), exist_ok=True)

    with open(os.path.join(args["path_save"], "args.json"), 'w') as f:
        json.dump(args, f, indent=4)

    training_dataset, validation_dataset, _ = dataset.init_datasets(args, debug=args["debug"])
    training_dataset_loader = dataset.init_dataset_loader(training_dataset, args, shuffle=True)
    validation_dataset_loader = dataset.init_dataset_loader(validation_dataset, args, shuffle=False)

    # start training
    if args['model'] == 'DDPM' or args['model'] == 'DDPMCondSeg':
        trainer_ddpm(training_dataset_loader, validation_dataset_loader, args, device)
    elif args['model'] == 'VAE':
        trainer_vae(training_dataset_loader, validation_dataset_loader, args, device)
    else:
        raise ValueError(f"Invalid model type {args['model']}. Must be 'DDPM', 'DDPMCondSeg' or 'VAE'.")

    # remove checkpoints after final_param is saved (due to storage requirements)
    for file_remove in os.listdir(os.path.join(args["path_save"], "model", "checkpoint")):
        os.remove(os.path.join(args["path_save"], "model", "checkpoint", file_remove))

    os.removedirs(os.path.join(args["path_save"], "model", "checkpoint"))

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print(torch.cuda.get_device_properties(0).name)
        print(round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 1))
    else:
        print("Running on CPU")

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    
    main()
