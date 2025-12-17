import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import matplotlib.pyplot as plt
import numpy as np
from models import Soflow
import tqdm
from typing import Tuple, Literal, List
import torch.distributed as dist
from diffusers.models import AutoencoderKL
from latent_dataset import ImagenetDataset
from augment import AugmentPipe
import argparse
import time
import logging
import yaml
import shutil

class Learner:
    def __init__(self,
                 working_dir: str = './your_working_dir',
                 noising_type: Literal['Linear', 'Trig'] = 'Linear', 
                 coefficient_type: Literal['Euler', 'Trig'] = 'Euler',
                 model_type: Literal['UNet', 'DiT-B-4', 'DiT-B-2', 'DiT-M-2', 'DiT-L-2', 'DiT-XL-2'] = 'DiT-B-4',
                 dataset_type: Literal['CIFAR10', 'Imagenet'] = 'Imagenet',
                 data_size: int = 32, dataset_path = './imagenet_latent', 
                 velocity_loss_ratio: float = 0.75, p: float = 1.0, eps: float = 0.01, 
                 time_mean: Tuple[float, float, float] = (-0.2, -1.0, 0.2), time_std: Tuple[float, float, float] = (1.0, 0.8, 0.8), 
                 cfg_scale: float = 1.0, mix_ratio: float = 1.0, cfg_drop_rate: float = 0.0, cfg_decay_time: float = 1.0,
                 l_schedule: Literal['Const','Linear','Cosine','Exponent'] = 'Exponent', l_init_ratio: float = 0.1, l_end_ratio: float = 0.002,
                 total_steps: int = 400000, ema_decay_rate: float = 0.9999,
                 learning_rate: float = 1e-4, betas: Tuple[float, float] = (0.9, 0.99), weight_decay: float = 0.0, 
                 batch_size: int = 256, num_workers: int = 12, prefetch_factor: int = 6, 
                 eval_steps: int = 25000, eval_batch_size: int = 125, eval_NFE: int = 1, 
                 eval_demo_steps: int = 1000, eval_demo_shape: Tuple[int, int] = (4, 8),
                 logging_steps: int = 100, ckpt_saving_steps: int = 200000, seed: int = 42
                ):
        
        world_size = int(os.environ['WORLD_SIZE'])
        assert batch_size % world_size == 0, 'batch size shoule be multiple of GPU amount'
        rank = int(os.environ['RANK'])
        local_rank = int(os.environ['LOCAL_RANK'])
        torch.manual_seed(seed + rank)
        if rank == 0:
            os.makedirs(working_dir, exist_ok = True)
            logging.basicConfig(
                level=logging.INFO,
                format='[%(asctime)s] %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S',
                handlers=[logging.StreamHandler(), logging.FileHandler(os.path.join(working_dir,'log.txt'))]
            )
            self.logger= logging.getLogger(__name__)
            self.logger.info(f"Logger initialized with world size = {world_size}")

        self.device = torch.device(f'cuda:{local_rank}')
        dist.init_process_group(backend='nccl', init_method='env://', rank = rank, world_size = world_size, device_id = self.device)
        torch.cuda.set_device(self.device)

        torch.backends.cuda.enable_flash_sdp(False)  
        torch.backends.cuda.enable_mem_efficient_sdp(True) 
        torch.backends.cuda.enable_math_sdp(False) 
        
        assert dataset_type in ['CIFAR10', 'Imagenet'], 'Dataset type error'
        if dataset_type == 'Imagenet':
            assert model_type != 'UNet', 'UNet is for unconditonal generation on CIFAR10 & MNIST'
            if cfg_scale == 1.0:
                assert cfg_drop_rate == 0.0, 'CFG config error'
            else:
                assert (cfg_scale > 1.0 and cfg_drop_rate > 0.0), 'CFG config error'
            train_set = ImagenetDataset(dataset_path)
            assert data_size == train_set.latent_size, "data size unmatched"
            data_channels, num_classes = 4, 1000
            
        else:# dataset_type == 'CIFAR10':
            # assert model_type == 'UNet', 'UNet is for unconditonal generation on CIFAR10 & MNIST'
            assert cfg_scale == 1.0, 'Uncondional generation does not support CFG'
            # classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
            assert data_size == 32, "data size unmatched"
            data_channels, num_classes = 3, 10
            transform = torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
            
            if rank == 0:
                train_set = torchvision.datasets.CIFAR10(
                    root = dataset_path, train = True, 
                    download = True, transform = transform
                )
                
            dist.barrier()
            if rank != 0:
                train_set = torchvision.datasets.CIFAR10(
                    root = dataset_path, train = True,
                    download = False, transform = transform
                )
                
            self.augment_pipe = AugmentPipe(p=0.12, xflip=1e8, yflip=0, scale=1, rotate_frac=0, aniso=1, translate_frac=1)
    
        self.sampler = torch.utils.data.distributed.DistributedSampler(
            train_set,
            num_replicas = world_size,
            rank = rank,
            shuffle = True
        )
            
        self.train_loader = torch.utils.data.DataLoader(
            train_set,
            batch_size = batch_size // world_size,
            sampler = self.sampler,
            num_workers = num_workers,
            prefetch_factor = prefetch_factor,
            persistent_workers=True,
            pin_memory = True,
            drop_last = True
        )
        self.working_dir = working_dir
        
        self.dataset_type = dataset_type
        self.velocity_loss_ratio = velocity_loss_ratio
        self.p, self.eps = p, eps
        self.time_mean = time_mean
        self.time_std = time_std
        
        self.num_classes = num_classes
        self.cfg_scale = cfg_scale
        self.mix_ratio = mix_ratio
        self.cfg_drop_rate = cfg_drop_rate
        self.cfg_decay_time = cfg_decay_time
        
        self.eval_steps = eval_steps
        self.eval_batch_size = eval_batch_size
        self.eval_NFE = eval_NFE
        
        self.eval_time = [1.0 - i / self.eval_NFE for i in range(self.eval_NFE + 1)]
         
        self.eval_demo_steps = eval_demo_steps
        self.eval_demo_shape = eval_demo_shape
        
        self.total_steps = total_steps
        self.l_schedule = l_schedule
        self.l_init_ratio = l_init_ratio
        self.l_end_ratio = l_end_ratio
        self.ema_decay_rate = ema_decay_rate
        self.finished_steps = 0 
        self.finished_epochs = 0
        self.logging_steps = logging_steps
        self.ckpt_saving_steps = ckpt_saving_steps
        
        self.model = Soflow(data_channels = data_channels, data_size = data_size, num_classes = num_classes, cfg_drop_rate = cfg_drop_rate,
                           noising_type = noising_type, coefficient_type = coefficient_type, model_type = model_type).to(self.device)
        
        self.ema_model = Soflow(data_channels = data_channels, data_size = data_size, num_classes = num_classes, cfg_drop_rate = cfg_drop_rate,
                           noising_type = noising_type, coefficient_type = coefficient_type, model_type = model_type).to(self.device)
        
        self.ema_model.load_state_dict(self.model.state_dict())
        self.ema_model.requires_grad_(False)
        self.ema_model.eval()
        
        if dataset_type == 'Imagenet':
            self.vae = AutoencoderKL.from_pretrained(f'stabilityai/sd-vae-ft-mse').to(self.device)
            self.vae.requires_grad_(False)
            self.vae.eval()
            
        self.model = torch.nn.parallel.DistributedDataParallel(self.model, device_ids=[local_rank])
        if dataset_type == 'Imagenet':
            self.optimizer = torch.optim.AdamW(self.model.module.network.parameters(), lr = learning_rate, betas = betas, weight_decay = weight_decay)
        else:
            self.optimizer = torch.optim.RAdam(self.model.module.network.parameters(), lr = learning_rate, betas = betas, weight_decay = weight_decay)
        
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        
        os.makedirs(os.path.join(self.working_dir, 'ckpts'), exist_ok = True)
        ckpt_list = sorted([eval(one.split('.')[0]) for one in os.listdir(os.path.join(self.working_dir, 'ckpts')) if one.endswith('.pt')], reverse = True)
        
        dist.barrier()
        if len(ckpt_list) > 0:
            all = torch.load(os.path.join(self.working_dir, 'ckpts', f'{ckpt_list[0]}.pt'), map_location=self.device)
            self.model.module.load_state_dict(all['model'])
            self.ema_model.load_state_dict(all['ema_model'])
            self.optimizer.load_state_dict(all['optimizer'])
            self.finished_steps = all['finished_steps']
            self.finished_epochs = all['finished_epochs']
            if rank == 0:
                self.logger.info(f"Loaded checkpoint from {os.path.join(self.working_dir, 'ckpts', f'{ckpt_list[0]}.pt')}")
                self.logger.info(f"Dataset type: {self.dataset_type}, Model type: {model_type}, Parameters: {sum([p.numel() for p in self.model.parameters()])/1e6:.2f} M")
        else:     
            if rank == 0:
                torch.save({
                    'model':self.model.module.state_dict(),
                    'ema_model':self.ema_model.state_dict(),
                    'optimizer':self.optimizer.state_dict(),
                    'finished_steps':self.finished_steps,
                    'finished_epochs':self.finished_epochs
                }, os.path.join(self.working_dir, 'ckpts', '0.pt'))
                self.logger.info(f"Saving checkpoint to {os.path.join(self.working_dir, 'ckpts', '0.pt')}")
                self.logger.info(f"Dataset type: {self.dataset_type}, Model type: {model_type}, Parameters: {sum([p.numel() for p in self.model.parameters()])/1e6:.2f} M")

        dist.barrier()
            
    def train(self):
        avg_stats = {'vel': 0,'con': 0,'w_vel': 0,'w_con': 0,'grad': 0}
        start_time = time.time()
        while True:
            self.model.train()
            self.sampler.set_epoch(self.finished_epochs)
                
            for images_data, labels in self.train_loader:
                
                augment_labels = None
                if self.dataset_type == 'Imagenet':
                    mean, std = torch.chunk(images_data.to(self.device), chunks = 2, dim = 1)
                    samples = (mean + std * torch.randn_like(std)) * 0.18215
                else: # self.dataset_type == 'CIFAR10':
                    samples, augment_labels = self.augment_pipe(images_data.to(self.device))
                    
                labels = labels.to(self.device)
                        
                batch_size = samples.size(0)
                velocity_num = int(self.velocity_loss_ratio * batch_size)
                consistency_num = batch_size - velocity_num

                if velocity_num > 0:
                    samples_velocity = samples[:velocity_num]
                    labels_velocity = labels[:velocity_num]
                    augment_labels_velocity = None if augment_labels is None else augment_labels[:velocity_num]
                    
                    if self.dataset_type != 'Imagenet':
                        labels_velocity = None

                    # logit-normal sampling
                    t = torch.sigmoid(torch.randn(velocity_num, 1, 1, 1, device = self.device) * self.time_std[0] + self.time_mean[0]) 
                    if self.cfg_scale > 1.0:
                        cfg_mask = torch.rand(velocity_num, device = self.device) < self.cfg_drop_rate
                    else:
                        cfg_mask = torch.zeros(velocity_num, device = self.device).bool()
                        
                    velocity_loss, velocity_loss_mse, velocity_loss_weight = self.model(get_velocity_loss = True, 
                        samples = samples_velocity, t = t, labels = labels_velocity, cfg_mask = cfg_mask, cfg_scale = self.cfg_scale,
                        cfg_decay_time = self.cfg_decay_time, p = self.p, eps = self.eps, mix_ratio = self.mix_ratio, 
                        augment_labels = augment_labels_velocity)                 
                else:
                    velocity_loss = torch.zeros(1, device = self.device, requires_grad=True)
                    velocity_loss_mse, velocity_loss_weight = torch.zeros(1, device = self.device), torch.zeros(1, device = self.device)
                
                if consistency_num > 0:
                    samples_consistency = samples[velocity_num:]
                    labels_consistency = labels[velocity_num:]
                    augment_labels_consistency = None if augment_labels is None else augment_labels[velocity_num:]
                    
                    if self.dataset_type != 'Imagenet':
                        labels_consistency = None
                    
                    # logit-normal sampling
                    time1 = torch.sigmoid(torch.randn(consistency_num, 1, 1, 1, device = self.device) * self.time_std[1] + self.time_mean[1])
                    time2 = torch.sigmoid(torch.randn(consistency_num, 1, 1, 1, device = self.device) * self.time_std[2] + self.time_mean[2])

                    # the minimal gap is set to 1e-4
                    t = time2.clamp(min = 1e-4)
                    s = torch.min(time1, t - 1e-4)
                    l = self.model.module.get_middle_time(t, s, self.finished_steps, self.total_steps, init_ratio = self.l_init_ratio, end_ratio = self.l_end_ratio, method = self.l_schedule)
                    l = torch.min(l, t - 1e-4)
                    
                    if self.cfg_scale > 1.0:
                        cfg_mask = torch.rand(consistency_num, device = self.device) < self.cfg_drop_rate
                    else:
                        cfg_mask = torch.zeros(consistency_num, device = self.device).bool()
                    
                    consistency_loss, consistency_loss_mse, consistency_loss_weight = self.model(get_velocity_loss = False, 
                        samples = samples_consistency, t = t, l = l, s = s, labels = labels_consistency, cfg_mask = cfg_mask, 
                        cfg_scale = self.cfg_scale, cfg_decay_time = self.cfg_decay_time, p = self.p, eps = self.eps, 
                        mix_ratio = self.mix_ratio, augment_labels = augment_labels_consistency)
                else:
                    consistency_loss = torch.zeros(1, device = self.device, requires_grad=True)
                    consistency_loss_mse, consistency_loss_weight = torch.zeros(1, device = self.device), torch.zeros(1, device = self.device)

                loss = velocity_loss * (velocity_num / batch_size) + consistency_loss * (consistency_num / batch_size)
                    
                self.optimizer.zero_grad()
                loss.backward()
                
                grad = torch.zeros(1, device = self.device)
                for param in self.model.module.network.parameters():
                    if param.grad is not None:
                        grad += param.grad.pow(2).sum()
                grad = grad.sqrt()
                self.optimizer.step()

                with torch.inference_mode():
                    if self.dataset_type == 'Imagenet':
                        for ema_param, model_param in zip(self.ema_model.parameters(), self.model.module.parameters()):
                            ema_param.mul_(self.ema_decay_rate).add_(model_param, alpha = 1 - self.ema_decay_rate)
                        for ema_buffer, model_buffer in zip(self.ema_model.buffers(), self.model.module.buffers()):
                            ema_buffer.data.copy_(model_buffer.data)
                    else:
                        # CIFAR10 prefers a higher ema precision 
                        ema_accumulate_steps = 16
                        if self.finished_steps % ema_accumulate_steps == 0:
                            decay_accumulate = self.ema_decay_rate ** ema_accumulate_steps
                            for ema_param, model_param in zip(self.ema_model.parameters(), self.model.module.parameters()):
                                ema_param_double = ema_param.double()
                                model_param_double = model_param.double()
                                new_ema = decay_accumulate * ema_param_double + (1 - decay_accumulate) * model_param_double
                                ema_param.copy_(new_ema.float())
                
                            for ema_buffer, model_buffer in zip(self.ema_model.buffers(), self.model.module.buffers()):
                                ema_buffer.data.copy_(model_buffer.data)        
                    
                    avg_stats['vel'] += velocity_loss_mse
                    avg_stats['con'] += consistency_loss_mse
                    avg_stats['w_vel'] += velocity_loss_weight
                    avg_stats['w_con'] += consistency_loss_weight
                    avg_stats['grad'] += grad
                        
                    self.finished_steps += 1
                    if self.finished_steps % self.logging_steps == 0:
                        torch.cuda.synchronize()
                        end_time = time.time()
                        for key in avg_stats.keys():
                            avg_stats[key] /= self.logging_steps
                            dist.all_reduce(avg_stats[key], op = dist.ReduceOp.SUM)
                            avg_stats[key] = avg_stats[key].item() / dist.get_world_size()
                        
                        if dist.get_rank() == 0:
                            percentage = self.finished_steps / self.total_steps * 100
                            peak_memory = torch.cuda.max_memory_allocated() / 1024**3
                            torch.cuda.reset_peak_memory_stats()
                            self.logger.info(f"Steps = {self.finished_steps:07d}/{self.total_steps:07d} ({percentage:.3f}%), Vel_MSE = {avg_stats['vel']:.5f}, Vel_Weight = {avg_stats['w_vel']:.5f}, " + \
                                f"Con_MSE = {avg_stats['con']:.5f}, Con_Weight = {avg_stats['w_con']:.5f}, Grad_Norm = {avg_stats['grad']:.5f}, Mem = {peak_memory:.2f} GB, Time = {end_time - start_time:.2f} seconds")
                        
                        for key in avg_stats.keys():
                            avg_stats[key] = 0
                        start_time = time.time()
                        
                    if self.finished_steps % self.eval_demo_steps == 0 and dist.get_rank() == 0:
                        # Producing visual samples
                        sample_amount = self.eval_demo_shape[0] * self.eval_demo_shape[1]
                        t_ones = torch.ones((sample_amount,), device = self.device)    
                        C, H, W = self.ema_model.data_channels, self.ema_model.data_size, self.ema_model.data_size
                        x_t = torch.randn((sample_amount, C, H, W), device = self.device)
                        labels = torch.randint(0, self.ema_model.num_classes, (sample_amount, ), device = self.device)
                        
                        if self.dataset_type != 'Imagenet':
                            labels = None

                        for i in range(len(self.eval_time) - 2):
                            start = self.eval_time[i] * t_ones
                            end = 0 * t_ones 
                            x_t = self.ema_model.solution_operator(x_t, start, end, labels)
                            # Adding noise to perform multi-step generation
                            next_start = self.eval_time[i + 1] * t_ones
                            alpha_next = self.ema_model.scheduler.alpha(next_start).view(-1, 1, 1, 1)
                            beta_next = self.ema_model.scheduler.beta(next_start).view(-1, 1, 1, 1)
                            x_t = alpha_next * x_t + beta_next * torch.randn_like(x_t)
                    
                        x_t = self.ema_model.solution_operator(x_t, self.eval_time[-2] * t_ones, self.eval_time[-1] * t_ones, labels)
                                        
                        if self.dataset_type == 'Imagenet':
                            x_t = self.vae.decode(x_t / 0.18215).sample
                    
                        os.makedirs(os.path.join(self.working_dir, 'figs'), exist_ok = True)
                        self.visualize_batch(os.path.join(self.working_dir, 'figs', f'{self.finished_steps}.png'), x_t, self.eval_demo_shape[0], self.eval_demo_shape[1])
                        self.logger.info(f"{self.eval_demo_shape[0]} * {self.eval_demo_shape[1]} = {sample_amount} samples (NFE = {self.eval_NFE}) are saved to {os.path.join(self.working_dir, 'figs', f'{self.finished_steps}.png')}")

                    if self.finished_steps % self.ckpt_saving_steps == 0:
                        if dist.get_rank() == 0:
                            torch.save({
                                'model':self.model.module.state_dict(),
                                'ema_model':self.ema_model.state_dict(),
                                'optimizer':self.optimizer.state_dict(),
                                'finished_steps':self.finished_steps,
                                'finished_epochs':self.finished_epochs
                            }, os.path.join(self.working_dir, 'ckpts', f'{self.finished_steps}.pt'))
                            self.logger.info(f"Saving checkpoint to {os.path.join(self.working_dir, 'ckpts', f'{self.finished_steps}.pt')}")
                            
                    if self.finished_steps % self.eval_steps == 0:
                                    
                        for ema_param in self.ema_model.parameters():
                            dist.broadcast(ema_param.data, src = 0)
                        for ema_buffer in self.ema_model.buffers():
                            dist.broadcast(ema_buffer, src = 0) 
                    
                        results = [] # Producing FID-50K evaluation npz files
                        generate_amount = 50000 // dist.get_world_size() if 50000 % dist.get_world_size() == 0 else 50000 // dist.get_world_size() + 1
                        for start in tqdm.trange(0, generate_amount, self.eval_batch_size, desc = 'Inference for FID-50K Evaluation', disable=(dist.get_rank() != 0)):
                            end = min(generate_amount, start + self.eval_batch_size)
                            size = end - start
                            
                            t_ones = torch.ones((size, ), device = self.device)  
                            labels = torch.randint(0, self.ema_model.num_classes, (size,), device = self.device)
                            data_channels, data_size = self.ema_model.data_channels, self.ema_model.data_size
                            x_t = torch.randn((size, data_channels, data_size, data_size), device = self.device)

                            if self.dataset_type != 'Imagenet':
                                labels = None
                            
                            for i in range(len(self.eval_time) - 2):
                                start = self.eval_time[i] * t_ones
                                end = 0 * t_ones
                                x_t = self.ema_model.solution_operator(x_t, start, end, labels)
                                # Adding noise to perform multi-step generation
                                next_start = self.eval_time[i + 1] * t_ones
                                alpha_next = self.ema_model.scheduler.alpha(next_start).view(-1, 1, 1, 1)
                                beta_next = self.ema_model.scheduler.beta(next_start).view(-1, 1, 1, 1)
                                x_t = alpha_next * x_t + beta_next * torch.randn_like(x_t)
                        
                            x_t = self.ema_model.solution_operator(x_t, self.eval_time[-2] * t_ones, self.eval_time[-1] * t_ones, labels)
                        
                            if self.dataset_type == 'Imagenet':
                                x_t = self.vae.decode(x_t / 0.18215).sample   
                                
                            generated_samples = (x_t * 127.5 + 128.0).clamp(min=0,max=255).to(torch.uint8).permute(0, 2, 3, 1).contiguous()
                            if dist.get_rank() == 0:
                                gathered_results_list = [torch.zeros_like(generated_samples) for _ in range(dist.get_world_size())]
                            else:
                                gathered_results_list = None
                            dist.barrier()
                            dist.gather(generated_samples, gather_list=gathered_results_list, dst=0)
                            if dist.get_rank() == 0:
                                results.append(torch.cat(gathered_results_list, dim = 0).cpu())
                        
                        torch.cuda.empty_cache()
                        
                        if dist.get_rank() == 0:
                            results = torch.cat(results, dim = 0)[:50000].numpy()
                            os.makedirs(os.path.join(self.working_dir, 'evals'), exist_ok = True)
                            np.savez(os.path.join(self.working_dir, 'evals', f'{self.finished_steps}.npz'), results)
                            self.logger.info(f"50000 samples  (NFE = {self.eval_NFE}) are saved to {os.path.join(self.working_dir, 'evals', f'{self.finished_steps}.npz')}")
                        
                    
                if self.finished_steps >= self.total_steps:
                    dist.destroy_process_group()
                    return 
            self.finished_epochs += 1
            
    def visualize_batch(self, save_name, images, num_rows = 4, num_cols = 8):
        # images: [num_rows * num_cols, 1 or 3, 32, 32]
        plt.figure(figsize=(2 * num_cols, 2 * num_rows))
        for i in range(num_rows * num_cols):
            plt.subplot(num_rows, num_cols, i + 1)
            img = (images[i].cpu().permute(1,2,0) * 127.5 + 128.0).clamp(min=0,max=255).to(torch.uint8)
            plt.imshow(img, cmap = 'gray' if img.size(-1) == 1 else 'viridis')
            plt.axis('off')
        plt.tight_layout()
        plt.savefig(save_name)
        plt.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type = str, required=True, help = 'Training config path')
    args = parser.parse_args()
    
    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)
        
    if int(os.environ['RANK']) == 0: 
        os.makedirs(config['working_dir'], exist_ok = True)
        if not os.path.exists(os.path.join(config['working_dir'],'config.yaml')) or not os.path.samefile(args.config, os.path.join(config['working_dir'],'config.yaml')):
            shutil.copy(args.config, os.path.join(config['working_dir'],'config.yaml'))
        
    learner = Learner(**config)
    learner.train()



