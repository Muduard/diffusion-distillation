import torch
from dataclasses import dataclass
import torch.nn.functional as F
from diffusers.optimization import get_cosine_schedule_with_warmup
from accelerate import Accelerator
from tqdm.auto import tqdm
import os
from scheduling import CustomScheduler, ScheduleTypes, PredTypes
from train_utils import compute_sigma, compute_alpha, make_model, evaluate, get_dataloader, batch_last, batch_first, get_pretrained
#from evaluator import batch_sample
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
    output_dir = 'c10/v_base/'
    overwrite_output_dir = True  # overwrite the old model when re-running the notebook
    seed = 214
    device = "cuda"
    saved_state = ""

config = TrainingConfig()
config.dataset_name = "cifar10"
train_dataloader = get_dataloader(config.dataset_name, config.train_batch_size)

gen = torch.Generator(config.device)
gen.manual_seed(config.seed)

model = make_model(config.image_size, config.device)
#model = get_pretrained("c10/v_base/", config.device)#make_model(config.image_size, config.device)
#model = torch.load("b.pt")
#torch.save(model, "b.pt")
#epsilon = torch.randn((1,3,32,32))
#model = torch.jit.load("traced_diff.pt")

noise_scheduler = CustomScheduler(config.device, ScheduleTypes.COSINE)
optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

lr_scheduler = get_cosine_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=config.lr_warmup_steps,
    num_training_steps=(len(train_dataloader) * config.num_epochs),
)
#lr_scheduler.load_state_dict(torch.load("c10/v_base/34/scheduler.pkl"))

def train_loop(config, model, optimizer, train_dataloader, lr_scheduler, pred_type):
    # Initialize accelerator and tensorboard logging
    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        logging_dir=os.path.join(config.output_dir, "logs")
    )

    if accelerator.is_main_process:
        os.makedirs(config.output_dir, exist_ok=True)
        accelerator.init_trackers("teacher")

    # Prepare everything
    # There is no specific order to remember, you just need to unpack the
    # objects in the same order you gave them to the prepare method.
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )
    if config.saved_state != "":
        accelerator.load_state(config.saved_state)
    n = 1024
    global_step = 0
    starting_epoch = 0

    # Now you train the model
    for epoch in range(starting_epoch, config.num_epochs):
        model.train()
        progress_bar = tqdm(total=len(train_dataloader), disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch}")

        for step, batch in enumerate(train_dataloader):
            x = batch['img']
            bs = x.shape[0]
            # Sample a random timestep for each image
            timesteps = torch.randint(0, n, (bs,), device=x.device).long()
            t = timesteps / n

            alpha_t = compute_alpha(t)
            sigma_t = compute_sigma(alpha_t).to(x.device)

            epsilon = torch.randn(x.shape).to(x.device)
            epsilonT = batch_last(epsilon)
            xT = batch_last(x)
            zT_t = (xT * alpha_t + sigma_t * epsilonT)
            z_t = batch_first(zT_t)

            # Predict the noise residual
            pred = model(z_t, t, return_dict=False)[0]
            if pred_type == PredTypes.eps:
                loss = F.mse_loss(pred, epsilon)

            elif pred_type == PredTypes.x:
                loss = F.mse_loss(pred, x)

            elif pred_type == PredTypes.v:
                v = alpha_t * epsilonT - sigma_t * xT
                v = batch_first(v)
                loss = F.mse_loss(pred, v)

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
                 evaluate(config, epoch, model, gen, n, ScheduleTypes.COSINE, pred_type, "continuous", config.device)

            if (epoch + 1) % config.save_model_epochs == 0 or epoch == config.num_epochs - 1:
                file_path = config.output_dir + f'/{epoch}/'
                if not os.path.exists(file_path):
                    os.mkdir(file_path)
                m = accelerator.unwrap_model(model)
                m.save_pretrained(file_path)
                torch.save(lr_scheduler.state_dict(), file_path + "scheduler.pkl")


args = (config, model, optimizer, train_dataloader, lr_scheduler, PredTypes.v)

train_loop(*args)
