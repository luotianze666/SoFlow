import os
import shutil
import torch
import numpy as np
from models import Soflow
import tqdm
import torch.distributed as dist
from diffusers.models import AutoencoderKL
from latent_dataset import ImagenetDataset
import argparse
import yaml
import json

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type = str, required=True, help = "Training config path")
    parser.add_argument("--ckpt-steps", type = int, required = True, help = "Checkpoint number for inference")
    parser.add_argument("--eval-batch-size", type = int, default = 32, help= "Inference batch size")
    parser.add_argument("--eval-NFE", type = int, default = 1, help= "Inference time schedule")
    parser.add_argument("--seed", type = int, default = 42)
    args = parser.parse_args()
    
    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)
        
    if int(os.environ['RANK']) == 0:
        os.makedirs(config['working_dir'], exist_ok = True)
        if not os.path.exists(os.path.join(config['working_dir'],'config.yaml')) or \
            not os.path.samefile(args.config, os.path.join(config['working_dir'],'config.yaml')):
            shutil.copy(args.config, os.path.join(config['working_dir'],'config.yaml'))
            
    world_size = int(os.environ['WORLD_SIZE'])
    rank = int(os.environ['RANK'])
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.manual_seed(args.seed + rank)
    device = torch.device(f'cuda:{local_rank}')
    dist.init_process_group(backend='nccl', init_method='env://', rank = rank, world_size = world_size, device_id = device)
    
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
        
    if config['dataset_type'] == 'Imagenet':
        data_channels, data_size, num_classes = 4, config['data_size'], 1000  
    elif config['dataset_type'] == 'CIFAR10':
        data_channels, data_size, num_classes = 3, config['data_size'], 10

    ema_model = Soflow(data_channels = data_channels, data_size =  data_size, num_classes =  num_classes, cfg_drop_rate =  config['cfg_drop_rate'],
                           noising_type =  config['noising_type'], coefficient_type = config['coefficient_type'], model_type = config['model_type']).to(device)
        
    ema_model.load_state_dict(torch.load(os.path.join(config['working_dir'],'ckpts',f'{args.ckpt_steps}.pt'), map_location = device)['ema_model'])
    ema_model.requires_grad_(False)
    ema_model.eval()
    
    if  config['dataset_type'] == 'Imagenet':
        vae = AutoencoderKL.from_pretrained(f'stabilityai/sd-vae-ft-mse').to(device)
        vae.requires_grad_(False)
        vae.eval()
    
    eval_NFE, eval_batch_size = args.eval_NFE, args.eval_batch_size
    
    eval_time = [1.0 - i / eval_NFE for i in range(eval_NFE + 1)]
         
    with torch.inference_mode():
        results = [] # Producing FID-50K evaluation npz files
        generate_amount = 50000 // dist.get_world_size() if 50000 % dist.get_world_size() == 0 else 50000 // dist.get_world_size() + 1
        for start in tqdm.trange(0, generate_amount, args.eval_batch_size, desc = 'Inference for FID-50K Evaluation', disable=(dist.get_rank() != 0)):
            end = min(generate_amount, start +  args.eval_batch_size)
            size = end - start
            
            t_ones = torch.ones((size, ), device = device)  
            labels = torch.randint(0, ema_model.num_classes, (size,), device = device)
            data_channels, data_size = ema_model.data_channels, ema_model.data_size
            x_t = torch.randn((size, data_channels, data_size, data_size), device = device)

            if config['dataset_type'] != 'Imagenet':
                labels = None
           
            for i in range(len(eval_time) - 2):
                start = eval_time[i] * t_ones
                end = 0 * t_ones 
                x_t = ema_model.solution_operator(x_t, start, end, labels)
                # Adding noise to perform multi-step generation
                next_start = eval_time[i + 1] * t_ones
                alpha_next = ema_model.scheduler.alpha(next_start).view(-1, 1, 1, 1)
                beta_next = ema_model.scheduler.beta(next_start).view(-1, 1, 1, 1)
                x_t = alpha_next * x_t + beta_next * torch.randn_like(x_t)
            
            x_t = ema_model.solution_operator(x_t, eval_time[-2] * t_ones, eval_time[-1] * t_ones, labels)
            
            if config['dataset_type'] == 'Imagenet':
                x_t = vae.decode(x_t / 0.18215).sample   
                
            generated_samples = (x_t * 127.5 + 128.0).clamp(min=0,max=255).to(torch.uint8).permute(0, 2, 3, 1).contiguous()
            if dist.get_rank() == 0:
                gathered_results_list = [torch.zeros_like(generated_samples) for _ in range(dist.get_world_size())]
            else:
                gathered_results_list = None

            dist.gather(generated_samples, gather_list=gathered_results_list, dst=0)
            if dist.get_rank() == 0:
                results.append(torch.cat(gathered_results_list, dim = 0).cpu())
        
    if dist.get_rank() == 0:
        results = torch.cat(results, dim = 0)[:50000].numpy()
        os.makedirs(os.path.join( config['working_dir'], 'evals'), exist_ok = True)
        np.savez(os.path.join( config['working_dir'], 'evals', f'{args.ckpt_steps}_{str(eval_NFE)}NFE_{str(args.seed)}.npz'), results)
        print(f"50000 samples (NFE = {eval_NFE}) are saved to {os.path.join(config['working_dir'], 'evals', f'{args.ckpt_steps}_{str(eval_NFE)}NFE_{str(args.seed)}.npz')}")
    
    dist.destroy_process_group()
