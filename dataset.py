import json
import os
from torch.utils.data import IterableDataset
from monai.data import DataLoader, CacheDataset, ShuffleBuffer
from monai.transforms import (
    EnsureChannelFirstD,
    Compose,
    LoadImageD,
    EnsureTypeD,
    ConcatItemsd,
    DeleteItemsd,
)
from patch_brainstem import BrainstemPatchd
from patch_spinalcord import SpinalCordPatchd
from patch_parotid import ParotidPatchd

def init_datasets(args, debug=False, test_only=False):

    pre_transforms_3d = [
        LoadImageD(keys=args["labels_channels"]),
        EnsureChannelFirstD(keys=args["labels_channels"]),
    ]

    patch_size = tuple(args["patch_size"])  

    if args["organ_channel"] == 'brainstem':
        with open(os.path.join(args["path_data"], f'vertebrae_C2_coms_z.json'), 'r') as file:
            vertebrae = json.load(file)

        patches_transform = [
            BrainstemPatchd(
                keys=args["labels_channels"],
                label_key=args["organ_channel"],
                patch_size=patch_size,
                vertebrae=vertebrae
            ),
        ]
    elif args["organ_channel"] == 'spinalcord':
        with open(os.path.join(args["path_data"], f'vertebrae_C5_coms_z.json'), 'r') as file:
            vertebrae = json.load(file)

        patches_transform = [
            SpinalCordPatchd(
                keys=args["labels_channels"],
                label_key=args["organ_channel"],
                patch_size=patch_size,
                vertebrae=vertebrae
            ),
        ]
    elif args["organ_channel"] == 'parotid_l' or args["organ_channel"] == 'parotid_r':
        patches_transform = [
            ParotidPatchd(
                keys=args["labels_channels"],
                label_key=args["organ_channel"],
                patch_size=patch_size,
            ),
        ]
    else:
        raise Exception('Invalid channels')

    post_transforms_3d = [
        ConcatItemsd(keys=args["labels_channels"], name="im"),
        DeleteItemsd(keys=args["labels_channels"]),   
        EnsureTypeD(keys=["im"]), # converting to tensor
    ]

    transforms_trainval = Compose(
        pre_transforms_3d + patches_transform + post_transforms_3d
    )

    transforms_test = Compose(pre_transforms_3d + patches_transform + post_transforms_3d)

    path_train = os.path.join(args["path_data"], f'train_dict_{args["dataset"]}_{args["organ_channel"]}.json')
    path_val = os.path.join(args["path_data"], f'val_dict_{args["dataset"]}_{args["organ_channel"]}.json')
    path_test = os.path.join(args["path_data"], f'test_dict_{args["dataset"]}_{args["organ_channel"]}.json')
    if not os.path.exists(path_val):
        path_val = os.path.join(args["path_data"], f'val_dict_{args["dataset"]}_all.json')
    if not os.path.exists(path_test):
        path_test = os.path.join(args["path_data"], f'test_dict_{args["dataset"]}_all.json')

    if test_only:
        with open(path_test, 'r') as file:
            test_datadict = json.load(file)
        for i in range(len(test_datadict)):
            test_datadict[i]['key'] = test_datadict[i]['image'].split('/')[-2]

        if debug:
            if len(test_datadict) > 10:
                test_datadict = test_datadict[:10]

        testing_dataset = CacheDataset(test_datadict, transforms_test)

        return None, None, testing_dataset
    
    else:
        with open(path_train, 'r') as file:
            train_datadict = json.load(file)
        for i in range(len(train_datadict)):
            train_datadict[i]['key'] = train_datadict[i]['image'].split('/')[-2]

        with open(path_val, 'r') as file:
            val_datadict = json.load(file)
        for i in range(len(val_datadict)):
            val_datadict[i]['key'] = val_datadict[i]['image'].split('/')[-2]

        if args['dataset_subset'] < 1:
            subset_size_train = int(len(train_datadict) * args['dataset_subset'])
            subset_size_val = int(len(val_datadict) * args['dataset_subset'])
            train_datadict = train_datadict[:subset_size_train]
            val_datadict = val_datadict[:subset_size_val]

        if debug:
            train_datadict = train_datadict[:10]
            val_datadict = val_datadict[:10]

        training_dataset = CacheDataset(train_datadict, transforms_trainval)
        validation_dataset = CacheDataset(val_datadict, transforms_trainval)

        return training_dataset, validation_dataset, None

def init_reviewed_datasets(args):

    patch_size = tuple(args["patch_size"])  
    label_key_original = f"{args['organ_channel']} - original"

    pre_transforms_3d_original = [
        LoadImageD(keys=args["labels_channels"]),
        EnsureChannelFirstD(keys=args["labels_channels"]),
    ]
    pre_transforms_3d_corrected = [
        LoadImageD(keys=args["labels_channels"] + [label_key_original]),
        EnsureChannelFirstD(keys=args["labels_channels"] + [label_key_original]),
    ]

    if args["organ_channel"] == 'brainstem':
        with open(os.path.join(args["path_data"], f'vertebrae_C2_coms_z.json'), 'r') as file:
            vertebrae = json.load(file)

        patches_transform_original = [
            BrainstemPatchd(
                keys=args["labels_channels"],
                label_key=args["organ_channel"],
                patch_size=patch_size,
                vertebrae=vertebrae,
            ),
        ]
        patches_transform_corrected = [
            BrainstemPatchd(
                keys=args["labels_channels"],
                label_key=args["organ_channel"],
                patch_size=patch_size,
                vertebrae=vertebrae,
                label_key_original=label_key_original
            ),
        ]
    elif args["organ_channel"] == 'spinalcord':
        with open(os.path.join(args["path_data"], f'vertebrae_C5_coms_z.json'), 'r') as file:
            vertebrae = json.load(file)

        patches_transform_original = [
            SpinalCordPatchd(
                keys=args["labels_channels"],
                label_key=args["organ_channel"],
                patch_size=patch_size,
                vertebrae=vertebrae
            ),
        ]
        patches_transform_corrected = [
            SpinalCordPatchd(
                keys=args["labels_channels"],
                label_key=args["organ_channel"],
                patch_size=patch_size,
                vertebrae=vertebrae,
                label_key_original=label_key_original
            ),
        ]
    elif args["organ_channel"] == 'parotid_l' or args["organ_channel"] == 'parotid_r':
        patches_transform_original = [
            ParotidPatchd(
                keys=args["labels_channels"],
                label_key=args["organ_channel"],
                patch_size=patch_size,
            ),
        ]
        patches_transform_corrected = [
            ParotidPatchd(
                keys=args["labels_channels"],
                label_key=args["organ_channel"],
                patch_size=patch_size,
                label_key_original=label_key_original
            ),
        ]
    else:
        raise Exception('Invalid channels')

    post_transforms_3d_original = [
        ConcatItemsd(keys=args["labels_channels"], name="im"),
        DeleteItemsd(keys=args["labels_channels"]),   
        EnsureTypeD(keys=["im"]), # converting to tensor
    ]
    post_transforms_3d_corrected = [
        ConcatItemsd(keys=args["labels_channels"], name="im"),
        DeleteItemsd(keys=args["labels_channels"] + [label_key_original]),
        EnsureTypeD(keys=["im"]), # converting to tensor
    ]

    transforms_test_original = Compose(pre_transforms_3d_original + patches_transform_original + post_transforms_3d_original)
    transforms_test_corrected = Compose(pre_transforms_3d_corrected + patches_transform_corrected + post_transforms_3d_corrected)

    path_test_original = os.path.join(args["path_data"], f'test_dict_{args["dataset"]}_{args["organ_channel"]}_original.json')
    path_test_corrected = os.path.join(args["path_data"], f'test_dict_{args["dataset"]}_{args["organ_channel"]}_corrected.json')

    with open(path_test_original, 'r') as file:
        test_datadict_original = json.load(file)
    for i in range(len(test_datadict_original)):
        test_datadict_original[i]['key'] = test_datadict_original[i]['image'].split('/')[-2]

    with open(path_test_corrected, 'r') as file:
        test_datadict_corrected = json.load(file)
    for i in range(len(test_datadict_corrected)):
        test_datadict_corrected[i]['key'] = test_datadict_corrected[i]['image'].split('/')[-2]

    for i in range(len(test_datadict_corrected)):
        test_datadict_corrected[i][label_key_original] = test_datadict_original[i][args['organ_channel']]

    testing_dataset_original = CacheDataset(test_datadict_original, transforms_test_original)
    testing_dataset_corrected = CacheDataset(test_datadict_corrected, transforms_test_corrected)

    return testing_dataset_original, testing_dataset_corrected

def init_dataset_loader(dataset, args, shuffle=True):
    
    return DataLoader(
        dataset,
        batch_size=args["batch_size"],
        shuffle=shuffle,
        num_workers=0,
    )
