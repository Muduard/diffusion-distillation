import torch
from dataclasses import dataclass
from datasets import load_dataset
from torchvision import transforms
from diffusers import UNet2DModel

from PIL import Image
import torch.nn.functional as F
from diffusers.optimization import get_cosine_schedule_with_warmup
from diffusers.schedulers import DDIMScheduler, DDPMScheduler
from accelerate import Accelerator
import numpy as np

from tqdm.auto import tqdm
import os
from scheduling import CustomScheduler, ScheduleTypes, PredTypes


@dataclass
class TrainingConfig:
    image_size = 32  # the generated image resolution
    train_batch_size = 32
    eval_batch_size = 16  # how many images to sample during evaluation
    num_epochs = 50
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

dataset.set_transform(transform)
#dataset = torch.utils.data.Subset(dataset, list(range(100)))
train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=config.train_batch_size, shuffle=True)

model = UNet2DModel(
    sample_size=config.image_size,# the target image resolution
    time_embedding_type="fourier",
    flip_sin_to_cos=False,
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
)






def make_grid(images, rows, cols):
    w, h, r = images[0].shape
    grid = Image.new('RGB', size=(cols * w, rows * h))

    for i, image in enumerate(images):
        im = (image * 255).astype(np.uint8)
        im = Image.fromarray(im)
        grid.paste(im, box=(i % cols * w, i // cols * h))
    return grid


def evaluate(config, epoch, model, gen, timestep_number, schedule, pred_type,timestep_type):
    # Sample some images from random noise (this is the backward diffusion process).
    # The default pipeline output type is `List[PIL.Image]`
    noise_scheduler = CustomScheduler(device, schedule)
    images = noise_scheduler.denoise(model, (3, 32, 32), gen, timestep_number, config.eval_batch_size, pred_type, timestep_type=timestep_type)

    # Make a grid out of the images
    image_grid = make_grid(images, rows=4, cols=4)

    # Save the images
    test_dir = os.path.join(config.output_dir, "samples")
    os.makedirs(test_dir, exist_ok=True)
    image_grid.save(f"{test_dir}/{epoch:04d}.png")
    return image_grid



def compute_sigma(alpha):
    return torch.sqrt(1 - alpha ** 2)


def batch_first(a):
    return a.permute(3, 0, 1, 2)


def batch_last(a):
    return a.permute(1, 2, 3, 0)


device = "cuda"
gen = torch.Generator(device)
gen.manual_seed(config.seed)

#model = UNet2DModel.from_pretrained("c10/eps_epoch5")
model = model.to(device)

noise_scheduler = CustomScheduler("cuda", ScheduleTypes.COSINE)#DDIMScheduler(num_train_timesteps=1024, beta_schedule="squaredcos_cap_v2")

optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

lr_scheduler = get_cosine_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=config.lr_warmup_steps,
    num_training_steps=(len(train_dataloader) * config.num_epochs),
)


def train_loop(config, model, noise_scheduler, optimizer, train_dataloader, lr_scheduler):
    # Initialize accelerator and tensorboard logging
    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        log_with="tensorboard",
        logging_dir=os.path.join(config.output_dir, "logs")
    )
    if accelerator.is_main_process:
        os.makedirs(config.output_dir, exist_ok=True)
        accelerator.init_trackers("train_example")

    # Prepare everything
    # There is no specific order to remember, you just need to unpack the
    # objects in the same order you gave them to the prepare method.
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )
    N = 1024
    global_step = 0
    ts = torch.from_numpy(np.arange(0, N)[::-1].copy()).to(device)

    betas = noise_scheduler.get_betas(N, ts)
    alphas = 1 - betas
    tsN = ts
    starting_epoch = 0
    alphas_cumprod = torch.cumprod(alphas, dim=0)

    # Now you train the model
    for epoch in range(starting_epoch, config.num_epochs):
        progress_bar = tqdm(total=len(train_dataloader), disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch}")

        for step, batch in enumerate(train_dataloader):
            x = batch['img']
            # Sample noise to add to the images

            bs = x.shape[0]

            # Sample a random timestep for each image
            timesteps = torch.randint(0, N, (bs,),
                                      device=x.device).long()

            alpha_t = alphas_cumprod[ts[timesteps]] ** 0.5
            alpha_t1 = alphas_cumprod[ts[timesteps]-1] ** 0.5
            beta_t = (1 - alpha_t**2) ** 0.5
            beta_t1 = (1 - alpha_t1 ** 2) ** 0.5
            sigma_t = compute_sigma(alpha_t).to(device)
            epsilon = torch.randn(x.shape).to(x.device)

            xT = batch_last(x)
            epsilonT = batch_last(epsilon)
            zT_t = (alpha_t * xT + sigma_t * epsilonT)
            z_t = batch_first(zT_t)
            den = beta_t1 - beta_t * (alpha_t1 / alpha_t)
            target = zT_t - (alpha_t1 / alpha_t) * xT / den


            # Add noise to the clean images according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            #noisy_images = #noise_scheduler.add_noise(clean_images, noise, timesteps)

            with accelerator.accumulate(model):
                # Predict the noise residual

                t_pred = model(z_t, tsN[timesteps], return_dict=False)[0]

                #v = alpha_t * epsilonT - sigma_t * xT
                #vT = batch_first(v)

                loss = F.mse_loss(t_pred, target)
                accelerator.backward(loss)

                accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            progress_bar.update(1)
            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], "step": global_step}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
            global_step += 1

        # After each epoch you optionally sample some demo images with evaluate() and save the model
        if accelerator.is_main_process:
            if (epoch + 1) % config.save_image_epochs == 0 or epoch == config.num_epochs - 1:
                evaluate(config, epoch, model, gen, N, ScheduleTypes.COSINE, PredTypes.eps, "continuous")

            if (epoch + 1) % config.save_model_epochs == 0 or epoch == config.num_epochs - 1:
                file_path = config.output_dir + f'/eps_four_epoch{epoch}/'
                if not os.path.exists(file_path):
                    os.mkdir(file_path)
                model.save_pretrained(file_path)
            accelerator.init_trackers("train_example")


args = (config, model, noise_scheduler, optimizer, train_dataloader, lr_scheduler)

train_loop(*args)