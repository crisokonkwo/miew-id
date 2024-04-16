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

import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler

# For DDP utils
from socket import gethostname

# Environment variables set by torch.distributed.launch
rank = int(os.environ["SLURM_PROCID"])
#rank = 0
#gpus_per_node = 2
#world_size = 4
world_size    = int(os.environ["WORLD_SIZE"])
gpus_per_node = int(os.environ["SLURM_GPUS_ON_NODE"])
local_rank = rank - gpus_per_node * (rank // gpus_per_node)

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

def run(config, checkpoint_dir):
    #checkpoint_dir = f"{config.checkpoint_dir}/{config.project_name}/{config.exp_name}"
    #os.makedirs(checkpoint_dir, exist_ok=False)
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

    # Restricts data loading to a subset of the dataset exclusive to the current process
    train_sampler = DistributedSampler(dataset=train_dataset)
    #train_sampler = torch.utils.data.SubsetRandomSampler(train_dataset)
    valid_sampler = DistributedSampler(dataset=valid_dataset)
    #valid_sampler = torch.utils.data.SubsetRandomSampler(valid_dataset)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.engine.train_batch_size,
        num_workers=config.engine.num_workers,
        sampler=train_sampler,
        shuffle=False,
        pin_memory=True,
        drop_last=True,
    )

    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=config.engine.valid_batch_size,
        num_workers=config.engine.num_workers,
        sampler=valid_sampler,
        shuffle=False,
        pin_memory=True,
        drop_last=False,
    )

    if config.engine.device=="cuda":
        device = torch.device("cuda:{}".format(local_rank))
    else:
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
    ddp_model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)

    criterion = fetch_loss()
    criterion.to(device)
        

    optimizer = torch.optim.Adam(model.parameters(), lr=config.scheduler_params.lr_start)

    scheduler = MiewIdScheduler(optimizer,**dict(config.scheduler_params))

    write_config(config, config_path_out)

    with WandbContext(config):
        best_score = run_fn(config, ddp_model, train_loader, valid_loader, criterion, optimizer, scheduler, device, checkpoint_dir, use_wandb=config.engine.use_wandb)

    return best_score

def init_processes(checkpoint_dir):
    print(rank)
    print(world_size)
    print(gpus_per_node)
    print(f"Hello from rank {rank} of {world_size} on {gethostname()} where there are" \
          f" {gpus_per_node} allocated GPUs per node.", flush=True)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    #dist.init_process_group(backend, rank=rank, world_size=world_size)
    #fn(dist.get_rank(), dist.get_world_size())
    if rank == 0: 
        print(f"Group initialized? {dist.is_initialized()}", flush=True)
        #checkpoint_dir = f"{config.checkpoint_dir}/{config.project_name}/{config.exp_name}"
        os.makedirs(checkpoint_dir, exist_ok=False)
    
    
if __name__ == '__main__':
    args = parse_args()
    print(args)
    
    config_path = args.config
    
    config = get_config(config_path)

    checkpoint_dir = f"{config.checkpoint_dir}/{config.project_name}/{config.exp_name}" 

    init_processes(checkpoint_dir)

    run(config, checkpoint_dir)
