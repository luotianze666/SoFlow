import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from unet import SongUNet
from dit import DiT
from typing import Literal

class Scheduler(nn.Module):
    def __init__(self, noising_type: Literal['Linear', 'Triangular'] = 'Linear', coefficient_type: Literal['Euler', 'Triangular'] = 'Euler'):
        super().__init__()
        self.noising_type = noising_type
        self.coefficient_type = coefficient_type

    def alpha(self, t: torch.Tensor) -> torch.Tensor:
        if self.noising_type == 'Linear':
            return 1 - t
        elif self.noising_type == 'Triangular':
            return torch.cos(torch.pi / 2.0 * t)
            
    def beta(self, t: torch.Tensor) -> torch.Tensor:
        if self.noising_type == 'Linear':
            return t
        elif self.noising_type == 'Triangular':
            return torch.sin(torch.pi / 2.0 * t)
        
    def alpha_grad(self, t: torch.Tensor) -> torch.Tensor:
        if self.noising_type == 'Linear':
            return -1 * torch.ones_like(t)
        elif self.noising_type == 'Triangular':
            return -1 * torch.pi / 2.0 * torch.sin(torch.pi / 2.0 * t)

    def beta_grad(self, t: torch.Tensor) -> torch.Tensor:
        if self.noising_type == 'Linear':
            return torch.ones_like(t)
        elif self.noising_type == 'Triangular':
            return torch.pi / 2.0 * torch.cos(torch.pi / 2.0 * t)
    
    def get_x_t_v_t(self, x_0: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # x_0: [B, C, H, W], t: [B]
        t = t.view(x_0.size(0), 1, 1, 1)
        x_1 = torch.randn_like(x_0)
        alpha, beta = self.alpha(t), self.beta(t)
        alpha_grad, beta_grad = self.alpha_grad(t), self.beta_grad(t)
        x_t, v_t = alpha * x_0 + beta * x_1, alpha_grad * x_0 + beta_grad * x_1
        return x_t, v_t
        
    def a(self, t: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        if self.coefficient_type == 'Euler':
            return torch.ones_like(t, requires_grad = t.requires_grad)
        else: # self.coefficient_type == 'Triangular':
            return torch.cos(torch.pi / 2.0 * (s - t))
        
    def b(self, t: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        if self.coefficient_type == 'Euler':
            return (s - t)
        else: # self.coefficient_type == 'Triangular':
            return torch.sin(torch.pi / 2.0 * (s - t))
    
    def a_grad(self, t: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        if self.coefficient_type == 'Euler':
            return torch.zeros_like(t, requires_grad = t.requires_grad), torch.zeros_like(s, requires_grad = s.requires_grad)
        else: # self.coefficient_type == 'Triangular':
            return torch.pi / 2.0 * torch.sin(torch.pi / 2.0 * (s - t)), - torch.pi / 2.0 * torch.sin(torch.pi / 2.0 * (s - t))
        
    def b_grad(self, t: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        if self.coefficient_type == 'Euler':
            return - torch.ones_like(t, requires_grad = t.requires_grad), torch.ones_like(s, requires_grad = s.requires_grad)
        else: # self.coefficient_type == 'Triangular':
            return - torch.pi / 2.0 * torch.cos(torch.pi / 2.0 * (s - t)), torch.pi / 2.0 * torch.cos(torch.pi / 2.0 * (s - t))

    
class Soflow(nn.Module):
    def __init__(self, data_channels: int = 3, data_size: int = 32, num_classes: int | None = None, cfg_drop_rate: float = 0.1,
                 noising_type: Literal['Linear', 'Triangular'] = 'Linear', coefficient_type: Literal['Euler', 'Triangular'] = 'Euler',
                 model_type: Literal['UNet','DiT-B-4', 'DiT-B-2', 'DiT-M-2', 'DiT-L-2', 'DiT-XL-2'] = 'DiT-B-4'):
        
        super().__init__()
        self.data_channels = data_channels
        self.data_size = data_size
        self.num_classes = num_classes
        self.model_type = model_type
        self.scheduler = Scheduler(noising_type, coefficient_type)
        
        # Parameterization: f(x,t,s) = a(t,s)x + b(t,s)NN(x,t,s)
        if self.model_type == 'UNet':
            # UNet is for unconditional generation
            self.network = SongUNet(img_resolution = data_size, in_channels = data_channels, out_channels = data_channels, label_dim = 0, 
                                    augment_dim = 6, model_channels = 128, channel_mult = [2,2,2], channel_mult_emb = 4, num_blocks = 4, 
                                    attn_resolutions = [16], dropout = 0.30,  label_dropout = 0, embedding_type = 'positional', 
                                    channel_mult_noise  = 2, encoder_type = 'standard',   decoder_type = 'standard',  resample_filter = [1,3,3,1]
                            )
        else:
            # Diffusion transformer is for conditional generation
            if self.model_type == 'DiT-B-4':
                patch_size, hidden_size, depth, num_heads = 4, 768, 12, 12
            elif self.model_type == 'DiT-B-2':
                patch_size, hidden_size, depth, num_heads = 2, 768, 12, 12
            elif self.model_type == 'DiT-M-2':
                patch_size, hidden_size, depth, num_heads = 2, 1024, 16, 16
            elif self.model_type == 'DiT-L-2':
                patch_size, hidden_size, depth, num_heads = 2, 1024, 24, 16
            else: # 'DiT-XL-2'
                patch_size, hidden_size, depth, num_heads = 2, 1152, 28, 16
                
            self.network = DiT(input_size = data_size, patch_size = patch_size, in_channels = data_channels, hidden_size= hidden_size, 
                               depth = depth, num_heads = num_heads, mlp_ratio = 4, class_dropout_prob = cfg_drop_rate, num_classes = num_classes, 
                               learn_sigma = False) 
            
    def predict_velocity(self, x_t: torch.Tensor, t: torch.Tensor, labels: torch.Tensor | None = None, augment_labels: torch.Tensor | None = None):
        # images: [B, C, H, W], t: [B], s: [B], output: [B, C, H, W]
        batch_size = x_t.size(0)
        paps, pbps = self.scheduler.a_grad(t, t)[1], self.scheduler.b_grad(t, t)[1]
        return paps.view(batch_size, 1, 1, 1) * x_t + pbps.view(batch_size, 1, 1, 1) * self.network(x_t, t, t, labels, augment_labels) 
        
    def solution_operator(self, x_t: torch.Tensor, t: torch.Tensor, s: torch.Tensor, labels: torch.Tensor | None = None, augment_labels: torch.Tensor | None = None):
        # images: [B, C, H, W], t: [B], s: [B], output: [B, C, H, W]
        batch_size = x_t.size(0)
        return self.scheduler.a(t, s).view(batch_size, 1, 1, 1) * x_t + \
               self.scheduler.b(t, s).view(batch_size, 1, 1, 1) * self.network(x_t , t , s, labels, augment_labels)
        
    def get_velocity_loss(self, samples: torch.Tensor, t: torch.Tensor, labels: torch.Tensor | None, cfg_mask : torch.Tensor, cfg_scale: float,
                          cfg_decay_time: float, p: float = 0.0, eps: float = 0.01, mix_ratio: float = 1.0, augment_labels: torch.Tensor | None = None):
        
        # x_t: [B, C, H, W], v_t: [B, C, H, W], t: [B], labels: [B] or None
        
        batch_size = samples.size(0)          
        x_t, v_t = self.scheduler.get_x_t_v_t(samples, t)
        
        if labels is not None and cfg_scale > 1.0:            
            
            labels_masked = torch.where(cfg_mask, self.num_classes, labels)
            v_t_predict = self.predict_velocity(x_t, t, labels_masked, augment_labels) 
            
            with torch.no_grad():
                v_t_uncond = self.predict_velocity(x_t, t, None)
                cfg_decay_scale = self.get_cfg_decay_scale(t, decay_time = cfg_decay_time)
                v_t_guided = (v_t + cfg_decay_scale * (cfg_scale - 1) * (v_t - v_t_uncond))
                v_t_guided = mix_ratio * v_t_guided + (1 - mix_ratio) * v_t_predict
                v_t_masked = torch.where(cfg_mask.view(-1, 1, 1, 1), v_t, v_t_guided)
                
            delta = (v_t_predict - v_t_masked).view(batch_size, -1)
        
        else: # Unconditional generation or No-guidance conditional generation
            
            v_t_predict = self.predict_velocity(x_t, t, labels, augment_labels) 
            delta = (v_t_predict - v_t).view(batch_size, -1)

                        
        pbps = self.scheduler.b_grad(t, t)[1]
        
        w = 1.0 / (pbps.abs().flatten() * (delta.pow(2).mean(dim = 1) + eps).pow(p))
        
        velocity_loss = (w.detach() * delta.pow(2).mean(dim = 1)).mean() 
        velocity_loss_mse = delta.pow(2).mean()
        velocity_loss_weight = w.mean()
        
        return velocity_loss, velocity_loss_mse, velocity_loss_weight
    
    
    def get_consistency_loss(self, samples: torch.Tensor, t: torch.Tensor, l: torch.Tensor, s: torch.Tensor, labels: torch.Tensor | None, cfg_mask: torch.Tensor | None, 
                             cfg_scale: float, cfg_decay_time: float, p: float = 0.0, eps: float = 0.01, mix_ratio: float = 1.0, augment_labels: torch.Tensor | None = None):   
                
        # x_t: [B, C, H, W], v_t: [B, C, H, W], t: [B], s: [B], labels: [B] or None
        batch_size = samples.size(0)
        t, l, s = t.view(batch_size, 1, 1, 1), l.view(batch_size, 1, 1, 1), s.view(batch_size, 1, 1, 1)

        x_t, v_t = self.scheduler.get_x_t_v_t(samples, t)
    
        if labels is not None and cfg_scale > 1.0:    
            
            with torch.no_grad():

                labels_masked = torch.where(cfg_mask, self.num_classes, labels) 
                cfg_decay_scale = self.get_cfg_decay_scale(t, decay_time = cfg_decay_time)
                v_t_uncond = self.predict_velocity(x_t, t, None, augment_labels)
                v_t_guided = v_t + cfg_decay_scale * (cfg_scale - 1) * (v_t - v_t_uncond)
                if mix_ratio != 1.0:
                    v_t_predict = self.predict_velocity(x_t, t, labels, augment_labels)
                    v_t_guided = mix_ratio * v_t_guided + (1 - mix_ratio) * v_t_predict
               
                v_t_masked = torch.where(cfg_mask.view(-1, 1, 1, 1), v_t, v_t_guided)
        
        else: # Unconditional generation or No-guidance conditional generation
            labels_masked = labels
            v_t_masked = v_t

        random_engine_state = torch.cuda.get_rng_state()
        NN_t = self.network(x_t , t , s, labels_masked, augment_labels)
        f_t = self.scheduler.a(t, s) * x_t + self.scheduler.b(t, s) * NN_t
        
        with torch.no_grad():
            x_l = x_t + v_t_masked * (l - t)
            torch.cuda.set_rng_state(random_engine_state)
            # Use same random engine
            NN_l = self.network(x_l, l, s, labels_masked, augment_labels)
            f_l = self.scheduler.a(l, s) * x_l + self.scheduler.b(l, s) * NN_l
            delta_quotient = ((f_t - f_l) / (t - l)).view(batch_size, -1)
            w = 1.0 / (delta_quotient.pow(2).mean(dim = 1) + eps).pow(p).view(batch_size, -1)
        
        sgn = torch.sign(self.scheduler.b(t, s)).flatten() # b(t,s) / |b(t,s)| = sgn(b(t,s))
        consistency_loss = (sgn * (NN_t.view(batch_size, -1) * (delta_quotient * w)).mean(dim = 1)).mean()
        # Stabler implementation of the MSE objective with same gradient (No division)
        consistency_loss_mse = delta_quotient.pow(2).mean()
        consistency_loss_weight = w.mean()
        
        return consistency_loss, consistency_loss_mse, consistency_loss_weight
    
    def get_cfg_decay_scale(self, t: torch.Tensor, decay_time: float = 1.0, k: float = 0.025):
        # Smoothly decay CFG strength to provide a continuously differentiable guided velocity field 
        if decay_time < 1.0:
            decay_t = ((1 - t) / (1 - decay_time)).clamp(min = 0, max = 1 - 1e-6)
            return ((t <= decay_time)).float() + ((t > decay_time)).float() * (1 - torch.exp(- k * (decay_t / (1 - decay_t))))
        else:
            return torch.ones_like(t)
        
    def get_middle_time(self, t: torch.Tensor, s: torch.Tensor, now_steps: int, total_steps: int, 
                        init_ratio: float = 0.1, end_ratio: float = 0.002, method: Literal['Const','Linear','Cosine','Exponent'] = 'Exponent'):
        
        assert method in ['Const','Linear','Cosine','Exponent'], 'method type error'
        if method == 'Const':
            return t + (s - t) * end_ratio
        elif method == 'Linear':
            return t + (s - t) * (init_ratio + (end_ratio - init_ratio) * (now_steps / total_steps))
        elif method == 'Cosine':
            return t + (s - t) * (end_ratio + (init_ratio - end_ratio) * 0.5 * (1.0 + math.cos(math.pi * (now_steps / total_steps))))
        else: # method == 'Exponent':
            return  t + (s - t) * (init_ratio * ((end_ratio / init_ratio) ** (now_steps / total_steps)))
                   
    def forward(self, get_velocity_loss: bool = True, ** kwargs):
        
        if get_velocity_loss == True:
            return self.get_velocity_loss(samples = kwargs['samples'], t = kwargs['t'], labels = kwargs['labels'], cfg_mask = kwargs['cfg_mask'], 
                                          cfg_scale = kwargs['cfg_scale'], cfg_decay_time = kwargs['cfg_decay_time'], p = kwargs['p'], 
                                          eps = kwargs['eps'], mix_ratio = kwargs['mix_ratio'], augment_labels = kwargs['augment_labels'])
        
        else:
            return self.get_consistency_loss(samples = kwargs['samples'], t = kwargs['t'], l = kwargs['l'], s = kwargs['s'], labels = kwargs['labels'], 
                                             cfg_mask = kwargs['cfg_mask'], cfg_scale = kwargs['cfg_scale'], cfg_decay_time = kwargs['cfg_decay_time'],
                                             p = kwargs['p'], eps = kwargs['eps'], mix_ratio = kwargs['mix_ratio'], augment_labels = kwargs['augment_labels'])
        