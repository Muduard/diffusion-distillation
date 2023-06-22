import torch
from dataclasses import dataclass
from diffusers import UNet2DModel
import torch.nn.functional as F
from diffusers.optimization import get_cosine_schedule_with_warmup
from accelerate import Accelerator

import copy
from tqdm.auto import tqdm
import os
from scheduling import ScheduleTypes, PredTypes
from train_utils import compute_alpha, compute_sigma, batch_first, batch_last, get_dataloader,evaluate, freeze_model


@dataclass
class trainingConfig:
    image_size = 32  # the generated image resolution
    train_batch_size = 16
    eval_batch_size = 16  # how many images to sample during evaluation
    num_epochs = 30
    learning_rate = 2e-4
    lr_warmup_steps = 1000
    save_image_epochs = 1
    save_model_epochs = 1
    mixed_precision = 'fp16'  # `no` for float32, `fp16` for automatic mixed precision
    output_dir = 'c10/distill-v'  # the model namy locally and on the HF Hub
    seed = 214
    pretrained_student = ""
    pretrained_teacher = "c10/distill-v/3_40/"


config = trainingConfig()
config.dataset_name = "cifar10"
train_dataloader = get_dataloader(config.dataset_name, config.train_batch_size)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

#pretrained_student = "c10/distill-epsilon/1_59"#"c10/distill-epsilon/1_67"

# load model and scheduler
teacher = UNet2DModel.from_pretrained(config.pretrained_teacher)
teacher = teacher.to(device)


if config.pretrained_student == "":
    student = copy.deepcopy(teacher)

else:
    student = UNet2DModel.from_pretrained(config.pretrained_student)

#Freeze teacher weights
teacher.eval()
freeze_model(teacher)

optimizer = torch.optim.AdamW(student.parameters(), lr=config.learning_rate)

starting_epoch = 0
lr_scheduler = get_cosine_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=config.lr_warmup_steps,
    num_training_steps=(len(train_dataloader) * config.num_epochs)
)
if config.pretrained_student != "":
    lr_scheduler.load_state_dict(torch.load(config.pretrained_student + "scheduler.pkl"))

gen = torch.Generator(device)
gen.manual_seed(config.seed)


def diffuse_from_x(x, z, alpha, sigma, alpha_next, sigma_next):
    eps = (z - alpha * x) / sigma
    z_next = alpha_next * x + sigma_next * eps
    return z_next


def diffuse_from_eps(eps, z, alpha, sigma, alpha_next, sigma_next):
    x = (z - sigma * eps) / alpha
    z_next = alpha_next * x + sigma_next * eps
    return z_next


def diffuse_from_v(v, z, alpha, sigma, alpha_next, sigma_next):
    phi = torch.arctan(sigma / alpha)
    phi_next = torch.arctan(sigma_next / alpha_next)
    z_next = torch.cos(phi_next - phi) * z + torch.sin(phi_next - phi) * v
    return z_next


def train_loop(config, student, optimizer, train_dataloader, lr_scheduler, teacher, schedule, timestep_type, pred_type, starting_epoch):
    # Initialize accelerator and tensorboard logging
    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        logging_dir=os.path.join(config.output_dir, "logs")
    )

    # Prepare everything
    # there is no specific order to remember, you just need to unpack the
    # objects in the same order you gave them to the prepare method.
    student, optimizer, train_dataloader, lr_scheduler, teacher, schedule, starting_epoch = accelerator.prepare(
        student, optimizer, train_dataloader, lr_scheduler, teacher, schedule, starting_epoch
    )

    global_step = 0
    K = 8
    starting_K = 5
    N = 512
    for j in range(starting_K, K):
        n = int(N/(2**j))
        print("Starting training with sample steps = " + str(n))
        # Now you train the model
        for epoch in range(starting_epoch, config.num_epochs+1):
            progress_bar = tqdm(total=len(train_dataloader))
            progress_bar.set_description(f"Epoch {epoch}")
            student.train()
            for step, batch in enumerate(train_dataloader):
                # Sample data
                x = batch['img']  # Batch size last for timestep multiplication

                # Sample a random timestep for each image
                t = torch.randint(0, int(n), (x.shape[0],), device=x.device).long()
                t = torch.add(t, other=1)
                t = t / n
                t1 = t - 0.5 / n
                t2 = t - 1 / n
                alpha_t0 = compute_alpha(t)
                sigma_t0 = compute_sigma(alpha_t0).to(x.device)

                # Sample noise to add to the images
                epsilon = torch.randn(x.shape, device=device)

                # Get transposed matrices
                xt = batch_last(x)
                epsilont = batch_last(epsilon)

                # Add noise to the clean images according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                zt_t0 = (alpha_t0 * xt + sigma_t0 * epsilont)
                z_t0 = batch_first(zt_t0)

                # 2 Steps of DDIM of teacher
                alpha_t1 = compute_alpha(t1)
                alpha_t2 = compute_alpha(t2)
                sigma_t1 = compute_sigma(alpha_t1).to(x.device)
                sigma_t2 = compute_sigma(alpha_t2).to(x.device)

                pred_t0 = teacher(z_t0, t, return_dict=False)[0]
                predt_t0 = batch_last(pred_t0)


                if pred_type == PredTypes.eps:
                    zt_t1 = diffuse_from_eps(predt_t0, zt_t0, alpha_t0, sigma_t0, alpha_t1, sigma_t1)
                elif pred_type == PredTypes.x:
                    zt_t1 = diffuse_from_x(predt_t0, zt_t0, alpha_t0, sigma_t0, alpha_t1, sigma_t1)
                elif pred_type == PredTypes.v:
                    zt_t1 = diffuse_from_v(predt_t0, zt_t0, alpha_t0, sigma_t0, alpha_t1, sigma_t1)

                z_t1 = batch_first(zt_t1)
                pred_t1 = teacher(z_t1, t1, return_dict=False)[0]
                predt_t1 = batch_last(pred_t1)

                if pred_type == PredTypes.eps:
                    zt_t2 = diffuse_from_eps(predt_t1, zt_t1, alpha_t1, sigma_t1, alpha_t2, sigma_t2)
                elif pred_type == PredTypes.x:
                    zt_t2 = diffuse_from_x(predt_t1, zt_t1, alpha_t1, sigma_t1, alpha_t2, sigma_t2)
                elif pred_type == PredTypes.v:
                    zt_t2 = diffuse_from_v(predt_t1, zt_t1, alpha_t1, sigma_t1, alpha_t2, sigma_t2)

                stdv_frac = sigma_t2 / sigma_t0
                x_tilde = (zt_t2 - stdv_frac * zt_t0) / (alpha_t2 - stdv_frac * alpha_t0)
                if pred_type == PredTypes.eps or pred_type == PredTypes.v:
                    eps_tilde = (zt_t0 - alpha_t2 * x_tilde) / sigma_t0
                if pred_type == PredTypes.v:
                    v_tilde = alpha_t0 * eps_tilde - sigma_t0 * x_tilde

                with accelerator.accumulate(student):
                    # Predict the noise residual
                    pred_student = student(z_t0, t, return_dict=False)[0]
                    predt_student = batch_last(pred_student)

                    if pred_type == PredTypes.eps:
                        loss = F.mse_loss(eps_tilde, predt_student)
                    elif pred_type == PredTypes.x:
                        loss = F.mse_loss(x_tilde, predt_student)
                    elif pred_type == PredTypes.v:
                        loss = F.mse_loss(v_tilde, predt_student)

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
                    evaluate(config, epoch, student, gen, int(n), schedule, pred_type, timestep_type, device)

                if (epoch + 1) % config.save_model_epochs == 0 or epoch == config.num_epochs - 1:
                    file_path = config.output_dir + f'/{j}_{epoch}/'
                    if not os.path.exists(file_path):
                        os.mkdir(file_path)
                    m = accelerator.unwrap_model(student)
                    m.save_pretrained(file_path)
                    torch.save(lr_scheduler.state_dict(), file_path + "scheduler.pkl")
                accelerator.init_trackers("train_example")

        starting_epoch = 0
        lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=config.lr_warmup_steps,
            num_training_steps=(len(train_dataloader) * config.num_epochs),
            last_epoch= starting_epoch
        )
        teacher.load_state_dict(student.state_dict())



args = (config, student, optimizer, train_dataloader, lr_scheduler, teacher, ScheduleTypes.COSINE, "continuous", PredTypes.v, starting_epoch)

train_loop(*args)




