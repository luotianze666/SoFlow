import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data.distributed
from torchvision import transforms
from torchvision.datasets import ImageFolder
import numpy as np
from PIL import Image
import argparse
from tqdm import tqdm
from diffusers.models import AutoencoderKL
import os
import torch.distributed as dist
import h5py

class ImagenetDataset(torch.utils.data.Dataset):
    def __init__(self, data_path):
        
        self.hdf5_path = os.path.join(data_path, 'imagenet_latent.hdf5')
        self.h5_file = None
        
        with h5py.File(self.hdf5_path, 'r', libver='latest', swmr=True) as f:
            self.amount = f['amount'][()]
            self.latent_size = f['latent_size'][()]
    
    def lazy_init(self):
        if self.h5_file is None:
            self.h5_file = h5py.File(self.hdf5_path, 'r', libver='latest',swmr=True)
            self.latents = self.h5_file['latents']
            self.labels = self.h5_file['labels']
            
    def __len__(self):
        return self.amount
    
    def __getitem__(self, idx):
        self.lazy_init()
        latent = torch.tensor(self.latents[idx]).float()
        label = torch.tensor(self.labels[idx]).long()
        return latent, label
    
    def __del__(self):
        if self.h5_file is not None:
            self.h5_file.close()
            self.h5_file = None
    
        
def center_crop_arr(pil_image, image_size):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type = str, default = './imagenet', help = 'Imagenet raw dataset path.')
    parser.add_argument("--save-path",type = str, default = './imagenet_latent', help = 'Imagenet latent dataset saving path.')
    parser.add_argument("--image-size", type = int, choices = [256, 512], default = 256)
    parser.add_argument("--device-batch-size", type = int, default = 256)
    parser.add_argument("--num-workers", type = int, default = 4)
    parser.add_argument("--seed", type = int, default = 42)
    args = parser.parse_args()
    latent_size = args.image_size // 8
    
    world_size = int(os.environ['WORLD_SIZE'])
    rank = int(os.environ['RANK'])
    local_rank = int(os.environ['LOCAL_RANK'])
    
    torch.cuda.set_device(local_rank)
    device = torch.device(f'cuda:{local_rank}')
    
    dist.init_process_group(backend='nccl', init_method='env://', 
                            rank=rank, world_size=world_size, device_id = device)
    torch.manual_seed(args.seed + rank)
    
    os.makedirs(args.save_path, exist_ok = True)
    
    transform = transforms.Compose([
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
    ])
    
    imagenet = ImageFolder(args.data_path, transform = transform)
    amount = len(imagenet)
    
    dist.barrier()

    block_size, remain = amount // world_size, amount % world_size
    if rank < remain:
        start = rank * (block_size + 1)
        end = start + (block_size + 1)
    else:
        start = remain * (block_size + 1) + (rank - remain) * block_size
        end = start + block_size
        
    num_samples = end - start    
    temp_hdf5_path = os.path.join(args.save_path, f'imagenet_latent_part_{rank}.hdf5')
    with h5py.File(temp_hdf5_path, 'w') as f:
        f.create_dataset('latents', shape=(num_samples, 8, latent_size, latent_size), dtype=np.float32)
        f.create_dataset('labels', shape=(num_samples,), dtype=np.int64)
        
    tqdm_loader = tqdm(torch.utils.data.DataLoader(
        torch.utils.data.Subset(imagenet, list(range(start, end))),
        batch_size = args.device_batch_size,
        shuffle = False,
        num_workers = args.num_workers,
        pin_memory = True,
        drop_last = False,
    ), desc = f'Proc {rank}: Extracting Imagenet-{args.image_size} Latents', position = rank)
    
    vae = AutoencoderKL.from_pretrained(f'stabilityai/sd-vae-ft-mse').to(device)
    vae.requires_grad_(False)
    vae.eval()

    with h5py.File(temp_hdf5_path, 'r+') as f:
        with torch.inference_mode():
            latents_dataset = f['latents']
            labels_dataset = f['labels']
            
            count = 0 
            dist.barrier()
            for images, labels in tqdm_loader:
                
                images = images.to(device)
                latent_dists = vae.encode(images).latent_dist
                latents = torch.cat([latent_dists.mean, latent_dists.std], dim = 1)
                
                batch_start = count
                batch_end = count + images.size(0)
                
                latents_dataset[batch_start:batch_end] = latents.cpu().numpy()
                labels_dataset[batch_start:batch_end] = labels.cpu().numpy()
                latents_dataset.flush()
                labels_dataset.flush()
                count = batch_end
            dist.barrier()

    if rank == 0:
        print("All ranks finished. Rank 0 is now merging files...")
        final_hdf5_path = os.path.join(args.save_path, 'imagenet_latent.hdf5')
        with h5py.File(final_hdf5_path, 'w') as f_final:

            f_final.create_dataset('latents', shape=(amount, 8, latent_size, latent_size), dtype=np.float32)
            f_final.create_dataset('labels', shape=(amount,), dtype=np.int64)
            f_final.create_dataset('amount', data=np.array(amount, dtype=np.int64))
            f_final.create_dataset('latent_size', data=np.array(latent_size, dtype=np.int64))

            current_pos = 0
            for i in range(world_size):
                part_file_path = os.path.join(args.save_path, f'imagenet_latent_part_{i}.hdf5')
                with h5py.File(part_file_path, 'r') as f_part:
                    num_in_part = f_part['latents'].shape[0]
                    print(f"Merging part {i}: {num_in_part} samples...")
                    f_final['latents'][current_pos:current_pos + num_in_part] = f_part['latents'][:]
                    f_final['labels'][current_pos:current_pos + num_in_part] = f_part['labels'][:]
                    current_pos += num_in_part
                
                os.remove(part_file_path)
        print(f"Merging complete. Final file with {current_pos} data points saved to: {final_hdf5_path}")             
    dist.destroy_process_group()
    