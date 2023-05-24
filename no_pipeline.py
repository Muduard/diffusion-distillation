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
import matplotlib.pyplot as plt
import copy
from tqdm.auto import tqdm
import os
from scheduling import CustomScheduler, ScheduleTypes, PredTypes

from torch.nn import Sigmoid,Softplus
sigmoid = Sigmoid()
softplus = Softplus()
import math
import wandb

'''wandb.init(
    # set the wandb project where this run will be logged
    project="generative",

    # track hyperparameters and run metadata
    config={
        "dataset": "CIFAR-100",
        "epochs": 50,
        "down_block_types" : (
        "DownBlock2D",  # a regular ResNet downsampling block
        "DownBlock2D",  # a ResNet downsampling block with spatial self-attention
        "DownBlock2D",
        "AttnDownBlock2D",
        "DownBlock2D",
        ),
        "sigma": "new"
    }
)'''



@dataclass
class TrainingConfig:
    image_size = 32  # the generated image resolution
    train_batch_size = 16
    eval_batch_size = 16  # how many images to sample during evaluation
    num_epochs = 100
    gradient_accumulation_steps = 1
    learning_rate = 2e-4
    lr_warmup_steps = 1000
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
    images = [preprocess(image.convert("RGB")) for image in examples["img"]]
    return {"img": images}


def make_grid(images, rows, cols):
    w, h, r = images[0].shape
    grid = Image.new('RGB', size=(cols * w, rows * h))

    for i, image in enumerate(images):
        im = (image * 255).astype(np.uint8)
        im = Image.fromarray(im)
        grid.paste(im, box=(i % cols * w, i // cols * h))
    return grid


def evaluate(config, epoch, model, gen, timestep_number, schedule, pred_type, timestep_type):
    # Sample some images from random noise (this is the backward diffusion process).
    # The default pipeline output type is `List[PIL.Image]`
    noise_scheduler = CustomScheduler(device, schedule)
    images = noise_scheduler.denoise(model, (3, 32, 32), gen, timestep_number, config.eval_batch_size, pred_type=pred_type, timestep_type=timestep_type)
    #images = noise_scheduler.denoise(model, (3, 32, 32), gen, timestep_number, config.eval_batch_size)
    # Make a grid out of the images
    image_grid = make_grid(images, rows=4, cols=4)

    # Save the images
    test_dir = os.path.join(config.output_dir, "samples")
    os.makedirs(test_dir, exist_ok=True)
    image_grid.save(f"{test_dir}/{epoch:04d}.png")


def compute_sigma(alpha):
    return torch.sqrt(1 - alpha**2)


def batch_first(a):
    return a.permute(3, 0, 1, 2)


def batch_last(a):
    return a.permute(1, 2, 3, 0)

def eps_to_x(z, logsnr, eps):
        return torch.sqrt(1. + torch.exp(-logsnr)) * (
                z - eps * (1 / (torch.sqrt(1. + torch.exp(logsnr)))))


dataset.set_transform(transform)
train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=config.train_batch_size, shuffle=True)

pretrained_student = "c10/0eps_distill-17"#"c10/0eepoch3/"#"c10/0epoch2/"#""#config.output_dir + "/0epoch5/" #""##  #

model_id = "google/ddpm-cifar10-32"

# load model and scheduler
teacher = UNet2DModel.from_pretrained("c10/eps_good75")# #DDPMPipeline.from_pretrained(model_id).unet#UNet2DModel.from_pretrained("c10/eps_epoch22")#DDPMPipeline.from_pretrained(model_id).unet#DDPMPipeline.from_pretrained(model_id).unet#UNet2DModel.from_pretrained("c10/0epoch9")#DDPMPipeline.from_pretrained(model_id).unet#UNet2DModel.from_pretrained("c10/eps_epoch22")##UNet2DModel.from_pretrained("c10/1epoch5")
teacher = teacher.to(device)

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


def train_loop_v(config, student, optimizer, train_dataloader, lr_scheduler, teacher, schedule, timestep_type):
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
    K = 3
    starting_K = 0
    N = 512
    starting_epoch = 0
    for j in range(starting_K, K):
        noise_scheduler = CustomScheduler(device, schedule)

        # ts = torch.from_numpy(np.arange(0, N)[::-1].copy()).to(device)
        one = torch.tensor([0], device=device, dtype=torch.int64)
        # Add 2 elements at the beggining to evaluate t-1 and t-2
        # ts = torch.cat((one, one, ts), 0)
        # Timesteps for mid-steps e.g. t - 0.5/N
        # ts2 = torch.from_numpy(np.arange(0, 2 * N)[::-1].copy()).to(device)
        ts2 = torch.from_numpy(np.arange(0, N)[::-1].copy()).to(device)
        # ts2 = torch.cat((one, one, ts2), 0)
        if timestep_type == "discrete":
            ts2N = ts2
        else:
            ts2N = ts2 / N
        # tsN = ts/N
        betas = noise_scheduler.get_betas(N, ts2)
        # alphas = 1 - betas

        # Student parameters = Teacher parameters
        if j > 1:
            student.load_state_dict(teacher.state_dict())
        # Now you train the model
        for epoch in range(starting_epoch, config.num_epochs):
            progress_bar = tqdm(total=len(train_dataloader))
            progress_bar.set_description(f"Epoch {epoch}")

            for step, batch in enumerate(train_dataloader):
                # Sample data
                x = batch['img']  # Batch size last for timestep multiplication

                # Sample a random timestep for each image
                i = torch.randint(0, int(N), (x.shape[0],), device=x.device).long()
                i = torch.add(i, other=1)
                i = i / N
                # logsnr_t = compute_sigma_t(i)
                # logsnr_1 = compute_sigma_t(i - 0.5/N)
                # logsnr_2 = compute_sigma_t(i - 1/N)
                alpha_t = compute_alpha(i)  # torch.sqrt(sigmoid(logsnr_t))
                sigma_t = compute_sigma(alpha_t).to(x.device)
                # sigma_t = torch.sqrt(sigmoid(-logsnr_t))
                # Sample noise to add to the images
                epsilon = torch.randn(x.shape, device=device)

                xT = batch_last(x)
                epsilonT = batch_last(epsilon)

                # Add noise to the clean images according to the noise magnitude at each timestep
                # (this is the forward diffusion process)

                zT_t = (alpha_t * xT + sigma_t * epsilonT)
                z_t = batch_first(zT_t)

                # 2 Steps of DDIM of teacher
                alpha_t1 = compute_alpha(i - 0.5 / N)  # t - 0.5/N#torch.sqrt(sigmoid(logsnr_1))#
                alpha_t2 = compute_alpha(i - 1 / N)  # torch.sqrt(sigmoid(logsnr_2))#  # t - 1/N

                # sigma_t1 = torch.sqrt(sigmoid(-logsnr_1))
                # sigma_t2 = torch.sqrt(sigmoid(-logsnr_2))
                sigma_t1 = compute_sigma(alpha_t1).to(x.device)
                sigma_t2 = compute_sigma(alpha_t2).to(x.device)

                v_t = teacher(z_t, i, return_dict=False)[0]
                vT_t = batch_last(v_t)
                phi_t = torch.arctan(sigma_t / alpha_t)
                phi_t1 = torch.arctan(sigma_t1 / alpha_t1)
                phi_t2 = torch.arctan(sigma_t2 / alpha_t2)

                # xT_t = eps_to_x(zT_t, logsnr_t, epsT_t)
                xT_t = alpha_t * zT_t - sigma_t * vT_t

                #zT_t1 = alpha_t1 * xT_t + sigma_t1 * epsT_t
                zT_t1 = torch.cos(phi_t1 - phi_t) * zT_t + torch.sin(phi_t1 - phi_t) * vT_t
                z_t1 = batch_first(zT_t1)

                v_t1 = teacher(z_t1, i - 0.5 / N, return_dict=False)[0]
                vT_t1 = batch_last(v_t1)

                xT_t1 = alpha_t1 * zT_t1 - sigma_t1 * vT_t1  # eps_to_x(zT_t1, i - 1/N, epsT_t1)#

                zT_t2 = torch.cos(phi_t2 - phi_t1) * zT_t1 + torch.sin(phi_t2 - phi_t1) * vT_t1

                stdv_frac = sigma_t2 / sigma_t
                # stdv_frac = torch.exp(0.5 * (softplus(sigma_t) - softplus(sigma_t2)))
                x_tilde = (zT_t2 - stdv_frac * zT_t) / (alpha_t2 - stdv_frac * alpha_t)
                eps_tilde = (zT_t2 - alpha_t2 * x_tilde) / sigma_t2
                v_tilde = alpha_t * eps_tilde - sigma_t * x_tilde
                # omega_t = torch.max(torch.tensor([torch.mean(alpha_t ** 2 / sigma_t ** 2), 1]))
                # i2 = (i//2).to(torch.long)
                # alpha_s = alphas[ts2[i2]]
                # sigma_s = compute_sigma(alpha_s)
                # zT_s = (alpha_t * xT + sigma_t * epsilonT)
                # z_s = batch_first(zT_s)

                with accelerator.accumulate(student):
                    # Predict the noise residual
                    # student(z_t, tsN[i], return_dict=False)[0]

                    v_student = student(z_t, i, return_dict=False)[0]
                    vT_student = batch_last(v_student)
                    # x_student = eps_to_x(zT_t, logsnr_t, epsT_student)
                    # xT_student = batch_first(x_student)
                    # xT_student = batch_first(x_student)
                    '''x1 = x_student.detach().cpu().numpy().transpose()[0]
                    x2 = x_tilde.detach().cpu().numpy().transpose()[0]

                    plt.imshow(x1)
                    plt.show()
                    plt.imshow(x2)
                    plt.show()'''

                    # loss_x = F.mse_loss(x_tilde, x_student)
                    # loss = F.mse_loss(x_tilde, x_student)
                    loss = F.mse_loss(v_tilde, vT_student)
                    '''if(loss > 100):
                        print("alpha_t:")
                        print(alpha_t)
                        print("sigma_t:")
                        print(sigma_t)'''
                    # loss2 = F.mse_loss(eps_tilde, epsT_student)
                    # loss = torch.maximum(loss1, loss2)
                    accelerator.backward(loss)

                    accelerator.clip_grad_norm_(student.parameters(), 1.0)
                    #wandb.log({"loss": loss})
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
                    evaluate(config, epoch, student, gen, int(N), schedule, PredTypes.v, timestep_type)

                if (epoch + 1) % config.save_model_epochs == 0 or epoch == config.num_epochs - 1:
                    file_path = config.output_dir + f'/{j}v_new-{epoch}/'
                    if not os.path.exists(file_path):
                        os.mkdir(file_path)
                    student.save_pretrained(file_path)
                accelerator.init_trackers("train_example")
        N = int(N / 2)
        teacher.load_state_dict(student.state_dict())


def compute_sigma_t(t):
    logsnr_min = -20.
    logsnr_max = 20.
    b = math.atan(math.exp(-0.5 * logsnr_max))
    a = math.atan(math.exp(-0.5 * logsnr_min)) - b
    return -2. * torch.log(torch.tan(a * t + b))


def compute_alpha(t):
    return (torch.cos((0.5 * torch.pi * t + 0.008) / (1.008)))


def train_loop_eps(config, student, optimizer, train_dataloader, lr_scheduler, teacher, schedule, timestep_type):
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
    K = 3
    starting_K = 0
    N = 512
    starting_epoch = 18
    for j in range(starting_K, K):
        noise_scheduler = CustomScheduler(device, schedule)

        # ts = torch.from_numpy(np.arange(0, N)[::-1].copy()).to(device)
        one = torch.tensor([0], device=device, dtype=torch.int64)
        # Add 2 elements at the beggining to evaluate t-1 and t-2
        # ts = torch.cat((one, one, ts), 0)
        # Timesteps for mid-steps e.g. t - 0.5/N
        #ts2 = torch.from_numpy(np.arange(0, 2 * N)[::-1].copy()).to(device)
        ts2 = torch.from_numpy(np.arange(0, N)[::-1].copy()).to(device)
        #ts2 = torch.cat((one, one, ts2), 0)
        if timestep_type == "discrete":
            ts2N = ts2
        else:
            ts2N = ts2/N
        # tsN = ts/N
        betas = noise_scheduler.get_betas(N, ts2)
        #alphas = 1 - betas

        # Student parameters = Teacher parameters
        if j > 1:
            student.load_state_dict(teacher.state_dict())
        # Now you train the model
        for epoch in range(starting_epoch, config.num_epochs):
            progress_bar = tqdm(total=len(train_dataloader))
            progress_bar.set_description(f"Epoch {epoch}")

            for step, batch in enumerate(train_dataloader):
                #Sample data
                x = batch['img']  # Batch size last for timestep multiplication

                # Sample a random timestep for each image
                i = torch.randint(0, int(N), (x.shape[0],), device=x.device).long()
                i = torch.add(i, other=1)
                i = i / N
                #logsnr_t = compute_sigma_t(i)
                #logsnr_1 = compute_sigma_t(i - 0.5/N)
                #logsnr_2 = compute_sigma_t(i - 1/N)
                alpha_t = compute_alpha(i)#torch.sqrt(sigmoid(logsnr_t))
                sigma_t = compute_sigma(alpha_t).to(x.device)
                #sigma_t = torch.sqrt(sigmoid(-logsnr_t))
                # Sample noise to add to the images
                epsilon = torch.randn(x.shape, device=device)

                xT = batch_last(x)
                epsilonT = batch_last(epsilon)

                # Add noise to the clean images according to the noise magnitude at each timestep
                # (this is the forward diffusion process)

                zT_t = (alpha_t * xT + sigma_t * epsilonT)
                z_t = batch_first(zT_t)

                # 2 Steps of DDIM of teacher
                alpha_t1 = compute_alpha(i - 0.5/N)  # t - 0.5/N#torch.sqrt(sigmoid(logsnr_1))#
                alpha_t2 = compute_alpha(i - 1/N)#torch.sqrt(sigmoid(logsnr_2))#  # t - 1/N

                #sigma_t1 = torch.sqrt(sigmoid(-logsnr_1))
                #sigma_t2 = torch.sqrt(sigmoid(-logsnr_2))
                sigma_t1 = compute_sigma(alpha_t1).to(x.device)
                sigma_t2 = compute_sigma(alpha_t2).to(x.device)

                eps_t = teacher(z_t, i, return_dict=False)[0]
                epsT_t = batch_last(eps_t)

                #xT_t = eps_to_x(zT_t, logsnr_t, epsT_t)
                xT_t = (zT_t - sigma_t * epsT_t) / alpha_t

                zT_t1 = alpha_t1 * xT_t + sigma_t1 * epsT_t
                z_t1 = batch_first(zT_t1)

                eps_t1 = teacher(z_t1, i - 0.5/N, return_dict=False)[0]
                epsT_t1 = batch_last(eps_t1)

                xT_t1 = (zT_t1 - sigma_t1 * epsT_t1) / alpha_t1 #eps_to_x(zT_t1, i - 1/N, epsT_t1)#

                zT_t2 = alpha_t2 * xT_t1 + sigma_t2 * epsT_t1

                stdv_frac = sigma_t2 / sigma_t
                #stdv_frac = torch.exp(0.5 * (softplus(sigma_t) - softplus(sigma_t2)))
                x_tilde = (zT_t2 - stdv_frac * zT_t) / (alpha_t2 - stdv_frac * alpha_t)
                eps_tilde = (zT_t2 - alpha_t2 * x_tilde) / sigma_t2

                #omega_t = torch.max(torch.tensor([torch.mean(alpha_t ** 2 / sigma_t ** 2), 1]))
                #i2 = (i//2).to(torch.long)
                #alpha_s = alphas[ts2[i2]]
                #sigma_s = compute_sigma(alpha_s)
                #zT_s = (alpha_t * xT + sigma_t * epsilonT)
                #z_s = batch_first(zT_s)

                with accelerator.accumulate(student):
                    # Predict the noise residual
                    #student(z_t, tsN[i], return_dict=False)[0]

                    eps_student = student(z_t, i, return_dict=False)[0]
                    epsT_student = batch_last(eps_student)
                    #x_student = eps_to_x(zT_t, logsnr_t, epsT_student)
                    #xT_student = batch_first(x_student)
                    #x_student = (zT_t - sigma_t * epsT_student) / alpha_t

                    #xT_student = batch_first(x_student)
                    '''x1 = x_student.detach().cpu().numpy().transpose()[0]
                    x2 = x_tilde.detach().cpu().numpy().transpose()[0]

                    plt.imshow(x1)
                    plt.show()
                    plt.imshow(x2)
                    plt.show()'''

                    #loss_x = F.mse_loss(x_tilde, x_student)
                    #loss = F.mse_loss(x_tilde, x_student)
                    loss = F.mse_loss(eps_tilde, epsT_student)
                    '''if(loss > 100):
                        print("alpha_t:")
                        print(alpha_t)
                        print("sigma_t:")
                        print(sigma_t)'''
                    #loss2 = F.mse_loss(eps_tilde, epsT_student)
                    #loss = torch.maximum(loss1, loss2)
                    accelerator.backward(loss)

                    accelerator.clip_grad_norm_(student.parameters(), 1.0)
                    #wandb.log({"loss": loss})
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
                    evaluate(config, epoch, student, gen, int(N), schedule, PredTypes.eps, timestep_type)

                if (epoch + 1) % config.save_model_epochs == 0 or epoch == config.num_epochs - 1:
                    file_path = config.output_dir + f'/{j}eps_distill-{epoch}/'
                    if not os.path.exists(file_path):
                        os.mkdir(file_path)
                    student.save_pretrained(file_path)
                accelerator.init_trackers("train_example")
        N = int(N/2)
        teacher.load_state_dict(student.state_dict())


args = (config, student, optimizer, train_dataloader, lr_scheduler, teacher, ScheduleTypes.COSINE, "continuous")
#0epoch9
#train_loop_v(*args)
train_loop_eps(*args)
#teacher = UNet2DModel.from_pretrained("./c10/0eps_N512x-7", device_map="auto") #UNet2DModel.from_pretrained("./c10/eps_four_epoch7", device_map="auto")#DDPMPipeline.from_pretrained(model_id).unet#UNet2DModel.from_pretrained("./c10/0epoch5", device_map="auto")
teacher = UNet2DModel.from_pretrained("./c10/0eps_distill-7", device_map="auto")
teacher = teacher.to(device)
evaluate(config, 52, teacher, gen, 512, ScheduleTypes.COSINE, PredTypes.eps, "continuous")




