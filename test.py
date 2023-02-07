import torch
from dataclasses import dataclass
from datasets import load_dataset
from torchvision import transforms
from diffusers import UNet2DModel

from PIL import Image
import torch.nn.functional as F
from diffusers.optimization import get_cosine_schedule_with_warmup
from diffusers import DDPMPipeline
from accelerate import Accelerator
import numpy as np
from diffusers import DDPMScheduler
import copy
import torchvision.transforms as T
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import os
from typing import Optional, Tuple, Union
from anim_plot import PlotAnimation
@dataclass
class TrainingConfig:
    image_size = 32  # the generated image resolution
    train_batch_size = 16
    eval_batch_size = 4  # how many images to sample during evaluation
    num_epochs = 5
    gradient_accumulation_steps = 1
    learning_rate = 1e-4
    lr_warmup_steps = 500
    save_image_epochs = 1
    save_model_epochs = 1
    mixed_precision = 'fp16'  # `no` for float32, `fp16` for automatic mixed precision
    output_dir = 'c10'  # the model namy locally and on the HF Hub

    push_to_hub = False  # whether to upload the saved model to the HF Hub
    hub_private_repo = False
    overwrite_output_dir = True  # overwrite the old model when re-running the notebook
    seed = 0


config = TrainingConfig()
config.dataset_name = "cifar10"
dataset = load_dataset(config.dataset_name, split="train")
preprocess = transforms.Compose(
    [
        transforms.Resize((config.image_size, config.image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ]
)


def transform(examples):
    images = [preprocess(image.convert("RGB")) for image in examples["img"]]
    return {"img": images}


def make_grid(images, rows, cols):
    r, w, h = images[0].size()
    grid = Image.new('RGB', size=(cols * w, rows * h))

    for i, image in enumerate(images):
        im = T.ToPILImage()(image)
        grid.paste(im, box=(i % cols * w, i // cols * h))
    return grid


# code from: https://github.com/huggingface/diffusers/blob/v0.12.0/src/diffusers/schedulers/scheduling_ddpm.py step()
def denoise_step(model_output, t, sample, generator, N, alphas, alpha_bars, alphas_cumprod):

    #alphas_cumprod = torch.cumprod(alphas, dim=0)

    alpha_prod_t = alphas_cumprod[t]
    alpha_t1 = alphas_cumprod[t-1] if t > 0 else torch.tensor([1], device=t.device)
    alpha_prod_t_prev = alpha_t1
    beta_t = 1 - alpha_prod_t / alpha_t1 #https://arxiv.org/pdf/2102.09672.pdf
    beta_t = torch.clamp(beta_t, 0.000001, 0.999)
    beta_prod_t = 1 - alpha_prod_t
    beta_prod_t_prev = 1 - alpha_prod_t_prev
    betas = 1 - alphas
    # calculations for posterior q(x_{t-1} | x_t, x_0)
    posterior_variance = (
            betas[t] * (1.0 - alpha_prod_t_prev) / (1.0 - alpha_prod_t)
    )
    posterior_mean_coef1 = (
            betas[t] * torch.sqrt(alpha_prod_t_prev) / (1.0 - alpha_prod_t)
    )
    posterior_mean_coef2 = (
            (1.0 - alpha_prod_t_prev)
            * torch.sqrt(alphas[t])
            / (1.0 - alpha_prod_t)
    )

    pred_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** (0.5)
    pred_original_sample = torch.clamp(pred_original_sample, -1, 1)

    # 4. Compute coefficients for pred_original_sample x_0 and current sample x_t
    # See formula (7) from https://arxiv.org/pdf/2006.11239.pdf
    #pred_original_sample_coeff = (alpha_prod_t_prev ** (0.5) * beta_t) / beta_prod_t
    #current_sample_coeff = alphas[t] ** (0.5) * beta_prod_t_prev / beta_prod_t

    # 5. Compute predicted previous sample µ_t
    # See formula (7) from https://arxiv.org/pdf/2006.11239.pdf
    #pred_prev_sample = pred_original_sample_coeff * pred_original_sample + current_sample_coeff * sample
    pred_prev_sample = posterior_mean_coef1 * pred_original_sample + posterior_mean_coef2 * sample
    # 6. Add noise
    variance = 0
    if t > 0:
        device = model_output.device
        variance_noise = torch.randn(
            model_output.shape, generator=generator, device=device, dtype=model_output.dtype
        )
        variance = (1 - alpha_prod_t_prev) / (1 - alpha_prod_t) * betas[t]
        variance = torch.clamp(variance, min=1e-20)
        variance = variance * variance_noise
    pred_prev_sample = pred_prev_sample + variance
    return pred_prev_sample


#code from https://github.com/huggingface/diffusers/blob/v0.7.0/src/diffusers/pipelines/ddpm/pipeline_ddpm.py __call__()
@torch.no_grad()
def denoise(
        model: UNet2DModel,
        image_shape: tuple = (1, 3, 32, 32),
        device="cpu",
        generator: Optional[torch.Generator] = None,
        n: int = 1024
        ):
    image = torch.randn(
            image_shape,
            generator=generator,
            device=device
        )
    pa = PlotAnimation()
    alphas = torch.zeros(n).to(device)
    alpha_bars = torch.zeros(n).to(device)
    betas = torch.zeros(n).to(device)
    ts = torch.from_numpy(np.arange(0, n)[::-1].copy()).to(device)
    for t in ts:
        alpha_bars[t] = compute_alpha(t/n) #torch.clamp(compute_alpha(t/n), 0.00001, 0.999)
    for t in ts:
        alphas_t1 = alpha_bars[t - 1] if t > 0 else torch.tensor([1], device=t.device)
        betas[t] = min(1 - alpha_bars[t] / alphas_t1, 0.999)
        alphas[t] = 1 - betas[t]
    alphas_cumprod = torch.cumprod(alphas, dim=0)

    for t in ts:
        print(t)
        # 1. predict noise model_output

        model_output = model(image, t).sample

        # 2. compute previous image: x_t -> x_t-1
        image = denoise_step(
            model_output, t, image, generator, n, alphas, alpha_bars, alphas_cumprod
        )
        pa.add_tensor(image)


    image = (image / 2 + 0.5).clamp(0, 1)

    image = image.cpu().permute(0, 2, 3, 1).numpy().astype(float)
    pa.start_animation()
    return image

def evaluate(config, epoch, model, x):
    # Sample some images from random noise (this is the backward diffusion process).
    # The default pipeline output type is `List[PIL.Image]`

    t = torch.tensor(0).repeat(x.shape[0]).to(x.device)
    alpha_t = compute_alpha(t).to(x.device)
    sigma_t = compute_sigma(alpha_t).to(x.device)
    # Sample noise to add to the images
    epsilon = torch.randn(x.shape).to(x.device)

    # Add noise to the clean images according to the noise magnitude at each timestep
    # (this is the forward diffusion process)
    images = model(epsilon, t)[0]
    # images = images.permute(0,2,3,1)

    # Make a grid out of the images
    image_grid = make_grid(images, rows=4, cols=4)

    # Save the images
    test_dir = os.path.join(config.output_dir, "samples")
    os.makedirs(test_dir, exist_ok=True)
    image_grid.save(f"{test_dir}/{epoch:04d}.png")


dataset.set_transform(transform)
train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=config.train_batch_size, shuffle=True)

'''model = UNet2DModel(
    sample_size=config.image_size,  # the target image resolution
    in_channels=3,  # the number of input channels, 3 for RGB images
    out_channels=3,  # the number of output channels
    layers_per_block=2,  # how many ResNet layers to use per UNet block
    block_out_channels=(128, 128, 256, 256, 512, 512),  # the number of output channes for each UNet block
    down_block_types=(
        "DownBlock2D",  # a regular ResNet downsampling block
        "DownBlock2D",
        "DownBlock2D",
        "DownBlock2D",
        "AttnDownBlock2D",  # a ResNet downsampling block with spatial self-attention
        "DownBlock2D",
    ),
    up_block_types=(
        "UpBlock2D",  # a regular ResNet upsampling block
        "AttnUpBlock2D",  # a ResNet upsampling block with spatial self-attention
        "UpBlock2D",
        "UpBlock2D",
        "UpBlock2D",
        "UpBlock2D"
      ),
)'''
model_id = "google/ddpm-cifar10-32"

# load model and scheduler
teacher = DDPMPipeline.from_pretrained(model_id, device_map="auto") # you can replace DDPMPipeline with DDIMPipeline or PNDMPipeline for faster inference
noise_scheduler = DDPMScheduler(num_train_timesteps=1000)
teacher.scheduler = noise_scheduler
student = DDPMPipeline.from_pretrained(model_id, device_map="auto")
student.scheduler = noise_scheduler



optimizer = torch.optim.AdamW(student.unet.parameters(), lr=config.learning_rate)

lr_scheduler = get_cosine_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=config.lr_warmup_steps,
    num_training_steps=(len(train_dataloader) * config.num_epochs),
)


def compute_alpha(t):
    return (torch.cos((0.5 * torch.pi * t + 0.008) / (1.008)))**2


def compute_sigma(alpha):
    return torch.sqrt(1 - alpha ** 2)


def batch_first(a):
    return a.permute(3, 0, 1, 2)


def batch_last(a):
    return a.permute(1, 2, 3, 0)


def train_loop(config, student, optimizer, train_dataloader, lr_scheduler, teacher):
    # Initialize accelerator and tensorboard logging
    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        log_with="tensorboard",
        logging_dir=os.path.join(config.output_dir, "logs")
    )

    # Prepare everything
    # There is no specific order to remember, you just need to unpack the
    # objects in the same order you gave them to the prepare method.
    student, optimizer, train_dataloader, lr_scheduler, teacher = accelerator.prepare(
        student, optimizer, train_dataloader, lr_scheduler, teacher
    )

    global_step = 0
    K = 3
    N = 1024
    for i in range(1, K):
        # Student parameters = Teacher parameters
        student.load_state_dict(teacher.state_dict())
        # Now you train the model
        for epoch in range(config.num_epochs):
            progress_bar = tqdm(total=len(train_dataloader))
            progress_bar.set_description(f"Epoch {epoch}")

            for step, batch in enumerate(train_dataloader):
                #Sample data
                x = batch['img']  # Batch size last for timestep multiplication

                # Sample a random timestep for each image
                i = torch.randint(0, N, (x.shape[0],), device=x.device).long()
                ts = torch.from_numpy(np.arange(0, N)[::-1].copy()).to(x.device)
                tsN = ts / N

                t = tsN[i]

                alpha_t = compute_alpha(t).to(x.device)
                sigma_t = compute_sigma(alpha_t).to(x.device)
                # Sample noise to add to the images
                epsilon = torch.randn(x.shape).to(x.device)

                xT = batch_last(x)
                epsilonT = batch_last(epsilon)

                # Add noise to the clean images according to the noise magnitude at each timestep
                # (this is the forward diffusion process)

                zT_t = (alpha_t * xT + sigma_t * epsilonT)
                z_t = batch_first(zT_t)

                # 2 Steps of DDIM of teacher
                t1 = t - 0.5/N
                t2 = t - 1/N
                ts1 = t1 * N
                alpha_t1 = compute_alpha(t1).to(x.device)
                alpha_t2 = compute_alpha(t2).to(x.device)
                sigma_t1 = compute_sigma(alpha_t1).to(x.device)
                sigma_t2 = compute_sigma(alpha_t2).to(x.device)

                x_t = teacher(z_t, i, return_dict=False)[0]
                xT_t = batch_last(x_t)

                zT_t1 = alpha_t1 * xT_t + (sigma_t1 / sigma_t) * (zT_t - alpha_t * xT_t)
                z_t1 = batch_first(zT_t1)

                x_t1 = teacher(z_t1, ts1, return_dict=False)[0]
                xT_t1 = batch_last(x_t1)
                zT_t2 = alpha_t2 * xT_t1 + (sigma_t2 / sigma_t) * (zT_t1 - alpha_t1 * xT_t1)

                x_tilde = (zT_t2 - (sigma_t2 / sigma_t) * zT_t) / (alpha_t2 - (sigma_t2 / sigma_t) * alpha_t)
                x_tilde = batch_first(x_tilde)
                omega_t = torch.max(torch.tensor([alpha_t[0] ** 2 / sigma_t[0] ** 2, 1]))
                with accelerator.accumulate(student):
                    # Predict the noise residual

                    x_student = student(z_t, i, return_dict=False)[0]

                    loss = F.mse_loss(x_tilde, x_student)
                    accelerator.backward(loss)

                    accelerator.clip_grad_norm_(student.parameters(), 1.0)
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()

                progress_bar.update(1)
                logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], "step": global_step}
                progress_bar.set_postfix(**logs)
                global_step += 1

            # After each epoch you optionally sample some demo images with evaluate() and save the model
            if accelerator.is_main_process:
                if (epoch + 1) % config.save_image_epochs == 0 or epoch == config.num_epochs - 1:
                    evaluate(config, epoch, student, x)

                if (epoch + 1) % config.save_model_epochs == 0 or epoch == config.num_epochs - 1:
                    print("saved")
                    student.save_pretrained(config.output_dir)
                accelerator.init_trackers("train_example")
        N = int(N/2)
        teacher.load_state_dict(student.state_dict())


args = (config, student, optimizer, train_dataloader, lr_scheduler, teacher)

#train_loop(*args)

device = "cuda:0"

gen = torch.Generator(device)
gen.manual_seed(config.seed)
#img = denoise(teacher, (1,3,32,32), "cuda:0", gen)[0]


def make_grid_s(images, rows, cols):
    w, h = images[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))
    for i, image in enumerate(images):
        grid.paste(image, box=(i%cols*w, i//cols*h))
    return grid


def evaluate_s(config, epoch, pipeline):
    # Sample some images from random noise (this is the backward diffusion process).
    # The default pipeline output type is `List[PIL.Image]`
    images = pipeline(
        batch_size = config.eval_batch_size,
        generator=torch.manual_seed(config.seed),
    ).images

    # Make a grid out of the images
    image_grid = make_grid_s(images, rows=2, cols=2)

    # Save the images
    test_dir = os.path.join(config.output_dir, "samples")
    os.makedirs(test_dir, exist_ok=True)
    image_grid.save(f"{test_dir}/{epoch:04d}.png")



evaluate_s(config,1,teacher)




