from datasets import MiewIdDataset, get_train_transforms, get_valid_transforms
from logging_utils import WandbContext
from models import MiewIdNet
from etl import preprocess_data, print_intersect_stats, preprocess_images
from losses import fetch_loss
from schedulers import MiewIdScheduler
from engine import run_fn
from helpers import get_config, write_config

import os
import torch
import random
import numpy as np
from dotenv import load_dotenv

import argparse

# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
# os.environ['TORCH_USE_CUDA_DSA'] = "1"

def parse_args():
    parser = argparse.ArgumentParser(description="Load configuration file.")
    parser.add_argument(
        '--config',
        type=str,
        default='configs/default_config.yaml',
        help='Path to the YAML configuration file. Default: configs/default_config.yaml'
    )
    return parser.parse_args()

def run(config):
    checkpoint_dir = f"{config.checkpoint_dir}/{config.project_name}/{config.exp_name}"
    os.makedirs(checkpoint_dir, exist_ok=False)
    print('Checkpoints will be saved at: ', checkpoint_dir)

    config_path_out = f'{checkpoint_dir}/{config.exp_name}.yaml'
    config.data.test.checkpoint_path = f'{checkpoint_dir}/model_best.bin'

    def set_seed_torch(seed):
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        
    set_seed_torch(config.engine.seed)

    df_train = preprocess_data(config.data.train.anno_path, 
                                name_keys=config.data.name_keys,
                                convert_names_to_ids=True, 
                                viewpoint_list=config.data.viewpoint_list, 
                                n_filter_min=config.data.train.n_filter_min, 
                                n_subsample_max=config.data.train.n_subsample_max,
                                use_full_image_path=config.data.use_full_image_path,
                                images_dir = config.data.images_dir,
                                )
    
    df_val = preprocess_data(config.data.val.anno_path, 
                                name_keys=config.data.name_keys,
                                convert_names_to_ids=True, 
                                viewpoint_list=config.data.viewpoint_list, 
                                n_filter_min=config.data.val.n_filter_min, 
                                n_subsample_max=config.data.val.n_subsample_max,
                                use_full_image_path=config.data.use_full_image_path,
                                images_dir = config.data.images_dir
                                )
    
    print_intersect_stats(df_train, df_val, individual_key='name_orig')
    
    n_train_classes = df_train['name'].nunique()

    crop_bbox = config.data.crop_bbox
    if config.data.preprocess_images:
        preprocess_dir_images = os.path.join(checkpoint_dir, 'images')
        preprocess_dir_train = os.path.join(preprocess_dir_images, 'train')
        preprocess_dir_val = os.path.join(preprocess_dir_images, 'val')
        print("Preprocessing images. Destination: ", preprocess_dir_images)
        os.makedirs(preprocess_dir_train)
        os.makedirs(preprocess_dir_val)

        target_size = (config.data.image_size[0],config.data.image_size[1])

        df_train = preprocess_images(df_train, crop_bbox, preprocess_dir_train, target_size)
        df_val = preprocess_images(df_val, crop_bbox, preprocess_dir_val, target_size)

        crop_bbox = False

    train_dataset = MiewIdDataset(
        csv=df_train,
        transforms=get_train_transforms(config),
        fliplr=config.test.fliplr,
        fliplr_view=config.test.fliplr_view,
        crop_bbox=crop_bbox,
    )
        
    valid_dataset = MiewIdDataset(
        csv=df_val,
        transforms=get_valid_transforms(config),
        fliplr=config.test.fliplr,
        fliplr_view=config.test.fliplr_view,
        crop_bbox=crop_bbox,
    )
        
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.engine.train_batch_size,
        num_workers=config.engine.num_workers,
        shuffle=True,
        pin_memory=True,
        drop_last=True,
    )

    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=config.engine.valid_batch_size,
        num_workers=config.engine.num_workers,
        shuffle=False,
        pin_memory=True,
        drop_last=False,
    )

    device = torch.device(config.engine.device)

    if config.model_params.n_classes != n_train_classes:
        print(f"WARNING: Overriding n_classes in config ({config.model_params.n_classes}) which is different from actual n_train_classes in the dataset - ({n_train_classes}).")
        config.model_params.n_classes = n_train_classes

    if config.model_params.loss_module == 'arcface_subcenter_dynamic':
        margin_min = 0.2
        margin_max = config.model_params.margin #0.5
        tmp = np.sqrt(1 / np.sqrt(df_train['name'].value_counts().sort_index().values))
        margins = (tmp - tmp.min()) / (tmp.max() - tmp.min()) * (margin_max - margin_min) + margin_min
    else:
        margins = None

    model = MiewIdNet(**dict(config.model_params), margins=margins)
    model.to(device)

    criterion = fetch_loss()
    criterion.to(device)
        

    optimizer = torch.optim.Adam(model.parameters(), lr=config.scheduler_params.lr_start)

    scheduler = MiewIdScheduler(optimizer,**dict(config.scheduler_params))

    write_config(config, config_path_out)


    with WandbContext(config):
        best_score = run_fn(config, model, train_loader, valid_loader, criterion, optimizer, scheduler, device, checkpoint_dir, use_wandb=config.engine.use_wandb)



    return best_score

if __name__ == '__main__':
    args = parse_args()
    config_path = args.config
    
    config = get_config(config_path)

    run(config)
