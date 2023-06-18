import torch
from tqdm import tqdm
from enum import Enum
from diffusers import UNet2DModel
from typing import Optional
import numpy as np


class ScheduleTypes(Enum):
    LINEAR = 1
    COSINE = 2


class PredTypes(Enum):
    eps = 1
    v = 2
    x = 3


class CustomScheduler:

    def __init__(self, device, schedule=ScheduleTypes.COSINE):
        self.device = device
        self.schedule = schedule

    def compute_alpha(self, t):
        return torch.cos((0.5 * torch.pi * t + 0.008) / 1.008) ** 2


    def get_betas(self, n, timesteps):
        betas = torch.zeros(n, device=self.device)
        if self.schedule == ScheduleTypes.LINEAR:
            beta_start = 0.0001
            beta_end = 0.02
            betas = torch.linspace(beta_start, beta_end, n, dtype=torch.float32, device=self.device)
        elif self.schedule == ScheduleTypes.COSINE:
            alpha_bars = torch.zeros(n, device=self.device)
            for t in timesteps:
                alpha_bars[t] = self.compute_alpha(t/n)
            for t in timesteps:
                alphas_t1 = alpha_bars[t - 1] if t > 0 else torch.tensor([1], device=t.device)
                betas[t] = min(1 - alpha_bars[t] / alphas_t1, 0.999)

        return betas


    # code from: https://github.com/huggingface/diffusers/blob/v0.12.0/src/diffusers/schedulers/scheduling_ddpm.py step()
    def denoise_step(self, model_output, t, sample, generator, alphas, alphas_cumprod, betas, schedule, pred_type):

        alpha_prod_t = alphas_cumprod[t]
        alpha_t1 = alphas_cumprod[t-1] if t > 0 else torch.tensor([1], device=self.device)
        alpha_prod_t_prev = alpha_t1
        if schedule == ScheduleTypes.LINEAR:
            beta_t = betas[t]
        elif schedule == ScheduleTypes.COSINE:
            beta_t = 1 - alpha_prod_t / alpha_t1 #https://arxiv.org/pdf/2102.09672.pdf
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev

        if pred_type == PredTypes.eps:
            pred_original_sample = (sample - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)
        elif pred_type == PredTypes.x:
            pred_original_sample = model_output
        elif pred_type == PredTypes.v:
            pred_original_sample = (alpha_prod_t ** 0.5) * sample - (beta_prod_t ** 0.5) * model_output

        pred_original_sample = torch.clamp(pred_original_sample, -1, 1)

        # 4. Compute coefficients for pred_original_sample x_0 and current sample x_t
        # See formula (7) from https://arxiv.org/pdf/2006.11239.pdf
        pred_original_sample_coeff = (alpha_prod_t_prev ** (0.5) * beta_t) / beta_prod_t
        current_sample_coeff = alphas[t] ** (0.5) * beta_prod_t_prev / beta_prod_t

        # 5. Compute predicted previous sample µ_t
        # See formula (7) from https://arxiv.org/pdf/2006.11239.pdf
        pred_prev_sample = pred_original_sample_coeff * pred_original_sample + current_sample_coeff * sample
        # 6. Add noise
        variance = 0
        if t > 0:
            variance_noise = torch.randn(
                model_output.shape, generator=generator, device=self.device, dtype=model_output.dtype
            )
            variance = (1 - alpha_prod_t_prev) / (1 - alpha_prod_t) * beta_t
            variance = torch.clamp(variance, min=1e-20) ** 0.5
            variance = variance * variance_noise
        pred_prev_sample = pred_prev_sample + variance
        return pred_prev_sample


    def denoise_ddim_step(self, model_output, t, sample, generator, alphas, alphas_cumprod, betas, schedule, pred_type):
        eta = 1.0
        alpha_prod_t = alphas_cumprod[t]
        alpha_t1 = alphas_cumprod[t-1] if t > 0 else torch.tensor([1], device=self.device)
        alpha_prod_t_prev = alpha_t1

        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev

        if pred_type == PredTypes.eps:
            pred_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
        elif pred_type == PredTypes.v:
            pred_original_sample = (alpha_prod_t ** 0.5) * sample - (beta_prod_t ** 0.5) * model_output
            # predict V
            #model_output = (alpha_prod_t ** 0.5) * model_output + (beta_prod_t ** 0.5) * sample
        elif pred_type == PredTypes.x:
            pred_original_sample = model_output

        pred_original_sample = torch.clamp(pred_original_sample, -1, 1)
        variance = (beta_prod_t_prev / beta_prod_t) * (1 - alpha_prod_t / alpha_prod_t_prev)
        std_dev_t = eta * variance ** 0.5

        model_output = (sample - alpha_prod_t ** 0.5 * pred_original_sample) / beta_prod_t ** 0.5
        # 6. compute "direction pointing to x_t" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
        pred_sample_direction = (1 - alpha_prod_t_prev - std_dev_t ** 2) ** 0.5 * model_output

        # 7. compute x_t without "random noise" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
        prev_sample = alpha_prod_t_prev ** 0.5 * pred_original_sample + pred_sample_direction
        if eta > 0:
            # randn_like does not support generator https://github.com/pytorch/pytorch/issues/27072
            device = model_output.device
            variance_noise = torch.randn(
                    model_output.shape, generator=generator, device=device, dtype=model_output.dtype)
            variance = std_dev_t * variance_noise

        prev_sample = prev_sample + variance
        return prev_sample


    def img_to_batch(self, img, batch_size):
        #img: numpy array of shape (dim,dim,3)
        i = 0
        dim = img.shape[0]
        img = img.transpose()
        batch_dim = int(np.sqrt(batch_size))
        red_dim = int(dim / batch_dim)
        result = np.zeros((batch_size, 3, red_dim, red_dim))
        for i in range(batch_dim):
            for j in range(batch_dim):
                pos = (i * batch_dim) + j
                result[pos] = img[:,red_dim * i: red_dim * (i+1),red_dim * j: red_dim * (j+1)]

        return result


    # code from https://github.com/huggingface/diffusers/blob/v0.7.0/src/diffusers/pipelines/ddpm/pipeline_ddpm.py __call__()
    @torch.no_grad()
    def denoise(
            self,
            model: UNet2DModel,
            image_shape: tuple = (3, 32, 32),
            generator: Optional[torch.Generator] = None,
            n: int = 1024,
            batch_size: int = 16,
            pred_type: PredTypes = PredTypes.eps,
            timestep_type="discrete"
            ):
        tensor_shape = tuple([batch_size]) + image_shape
        image = torch.randn(
                tensor_shape,
                generator=generator,
                device=self.device
            )
        ts = torch.from_numpy(np.arange(0, n)[::-1].copy()).to(self.device)

        betas = self.get_betas(n, ts)
        alphas = 1 - betas

        alphas_cumprod = torch.cumprod(alphas, dim=0)
        for t in tqdm(ts, position=1, desc="Step: ", leave=False, colour='blue', ncols=80):

            # 1. predict noise model_output
            if timestep_type == "discrete":
                model_output = model(image, t).sample
            else:
                model_output = model(image, t / n).sample

            # 2. compute previous image: x_t -> x_t-1
            image = self.denoise_ddim_step(
                model_output, t, image, generator, alphas, alphas_cumprod, betas, self.schedule, pred_type
            )
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy().astype(float)

        return image

