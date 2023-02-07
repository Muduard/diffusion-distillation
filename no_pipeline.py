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
from tqdm.auto import tqdm
import os
from scheduling import CustomScheduler, ScheduleTypes
@dataclass
class TrainingConfig:
    image_size = 32  # the generated image resolution
    train_batch_size = 16
    eval_batch_size = 16  # how many images to sample during evaluation
    num_epochs = 2
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

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def transform(examples):
    images = [preprocess(image.convert("RGB")) for image in examples["image"]]
    return {"img": images}


def make_grid(images, rows, cols):
    w, h, r = images[0].shape
    grid = Image.new('RGB', size=(cols * w, rows * h))

    for i, image in enumerate(images):
        im = (image * 255).astype(np.uint8)
        im = Image.fromarray(im)
        grid.paste(im, box=(i % cols * w, i // cols * h))
    return grid


def evaluate(config, epoch, model, gen, timestep_number, schedule):
    # Sample some images from random noise (this is the backward diffusion process).
    # The default pipeline output type is `List[PIL.Image]`
    noise_scheduler = CustomScheduler(device, schedule)
    images = noise_scheduler.denoise(model, (3, 32, 32), gen, timestep_number, config.eval_batch_size)

    # Make a grid out of the images
    image_grid = make_grid(images, rows=4, cols=4)

    # Save the images
    test_dir = os.path.join(config.output_dir, "samples")
    os.makedirs(test_dir, exist_ok=True)
    image_grid.save(f"{test_dir}/{epoch:04d}.png")


dataset.set_transform(transform)
train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=config.train_batch_size, shuffle=True)

pretrained_student = ""#config.output_dir + "/epoch8"


model_id = "google/ddpm-cifar10-32"

# load model and scheduler
teacher = DDPMPipeline.from_pretrained(model_id).unet
teacher = teacher.to("cuda:0")

if pretrained_student == "":
    student = copy.deepcopy(teacher)
else:
    student = UNet2DModel.from_pretrained(pretrained_student)


optimizer = torch.optim.AdamW(student.parameters(), lr=config.learning_rate)

lr_scheduler = get_cosine_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=config.lr_warmup_steps,
    num_training_steps=(len(train_dataloader) * config.num_epochs),
)

gen = torch.Generator(device)
gen.manual_seed(config.seed)

def compute_sigma(alpha):
    return torch.sqrt(1 - alpha ** 2)


def batch_first(a):
    return a.permute(3, 0, 1, 2)


def batch_last(a):
    return a.permute(1, 2, 3, 0)


def train_loop(config, student, optimizer, train_dataloader, lr_scheduler, teacher, schedule):
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
    student, optimizer, train_dataloader, lr_scheduler, teacher, schedule = accelerator.prepare(
        student, optimizer, train_dataloader, lr_scheduler, teacher, schedule
    )

    global_step = 0
    K = 2
    N = 1024
    noise_scheduler = CustomScheduler(device, schedule)

    for i in range(1, K):
        # Student parameters = Teacher parameters
        if i > 1:
            student.load_state_dict(teacher.state_dict())
        # Now you train the model
        ts = torch.from_numpy(np.arange(0, N)[::-1].copy()).to(device)
        one = torch.tensor([1], device=device, dtype=torch.int64)
        # Add 2 elements at the beggining to evaluate t-1 and t-2
        ts = torch.cat((one, one, ts), 0)
        betas = noise_scheduler.get_betas(N, ts)
        alphas = 1 - betas

        for epoch in range(config.num_epochs):
            progress_bar = tqdm(total=len(train_dataloader))
            progress_bar.set_description(f"Epoch {epoch}")

            for step, batch in enumerate(train_dataloader):
                #Sample data
                x = batch['img']  # Batch size last for timestep multiplication

                # Sample a random timestep for each image
                i = torch.randint(0, N, (x.shape[0],), device=x.device).long()

                i = torch.add(i, other=2)

                alpha_t = alphas[ts[i]]
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
                alpha_t1 = alphas[ts[i-1]]
                alpha_t2 = alphas[ts[i-2]]
                sigma_t1 = compute_sigma(alpha_t1).to(x.device)
                sigma_t2 = compute_sigma(alpha_t2).to(x.device)

                x_t = teacher(z_t, ts[i], return_dict=False)[0]
                xT_t = batch_last(x_t)

                zT_t1 = alpha_t1 * xT_t + (sigma_t1 / sigma_t) * (zT_t - alpha_t * xT_t)
                z_t1 = batch_first(zT_t1)

                x_t1 = teacher(z_t1, ts[i-1], return_dict=False)[0]
                xT_t1 = batch_last(x_t1)
                zT_t2 = alpha_t2 * xT_t1 + (sigma_t2 / sigma_t1) * (zT_t1 - alpha_t1 * xT_t1)

                x_tilde = (zT_t2 - (sigma_t2 / sigma_t) * zT_t) / (alpha_t2 - (sigma_t2 / sigma_t) * alpha_t)
                x_tilde = batch_first(x_tilde)
                omega_t = torch.max(torch.tensor([torch.mean(alpha_t ** 2 / sigma_t ** 2), 1]))

                with accelerator.accumulate(student):
                    # Predict the noise residual
                    x_student = student(z_t, ts[i], return_dict=False)[0]

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
                    evaluate(config, epoch, student, gen, 512, ScheduleTypes.COSINE)

                if (epoch + 1) % config.save_model_epochs == 0 or epoch == config.num_epochs - 1:
                    file_path = config.output_dir + f'/epoch{epoch}/'
                    if not os.path.exists(file_path):
                        os.mkdir(file_path)
                    student.save_pretrained(config.output_dir + f'/epoch{epoch}/')
                accelerator.init_trackers("train_example")
        N = int(N/2)
        teacher.load_state_dict(student.state_dict())


args = (config, student, optimizer, train_dataloader, lr_scheduler, teacher, ScheduleTypes.COSINE)

#train_loop(*args)


teacher = UNet2DModel.from_pretrained("./c10/epoch9", device_map="auto")
evaluate(config, 10, teacher, gen, 512, ScheduleTypes.LINEAR)




