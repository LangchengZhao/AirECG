"""
Training script for AirECG using PyTorch DDP.
"""
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from collections import OrderedDict
from copy import deepcopy
from glob import glob
from time import time
import argparse
import logging
import os
from tqdm import tqdm

from models import AirECG_model
from diffusion import create_diffusion

import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = 23, 15

def extract_model(model_name):
    """
    load a pre-trained AirECG model from a local path.
    """
    assert os.path.isfile(model_name), f'Could not find AirECG checkpoint at {model_name}'
    checkpoint = torch.load(model_name, map_location=lambda storage, loc: storage)
    if "ema" in checkpoint:  # supports checkpoints from train.py
        checkpoint = checkpoint["model"]
    return checkpoint

@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag


def cleanup():
    """
    End DDP training.
    """
    dist.destroy_process_group()


def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    if dist.get_rank() == 0:  # real logger
        logging.basicConfig(
            level=logging.INFO,
            format='[\033[34m%(asctime)s\033[0m] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
        )
        logger = logging.getLogger(__name__)
    else:  # dummy logger (does nothing)
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    return logger

def sample_images(ref, y, x, samples, batchIdx, resultPath, isVal=True):
    """
    Saves generated signals from the validation set
    """

    current_img_dir = resultPath #+ '/%s_Val.png' % (batchIdx)

    fig, axes = plt.subplots(y.shape[0], 4)
    axes[0][0].set_title('Ref')
    axes[0][1].set_title('mmWave')
    axes[0][2].set_title('ECG GroundTruth')
    axes[0][3].set_title('Generated ECG')

    
    for idx, signal in enumerate(y):
        axes[idx][0].plot(ref[idx], color='c')
        axes[idx][1].plot(y[idx], color='c')
        axes[idx][2].plot(x[idx], color='m')
        axes[idx][3].plot(samples[idx], color='y')

    fig.canvas.draw()
    fig.savefig(current_img_dir)
    plt.close(fig)

from torch.utils.data import DataLoader,Dataset
def DataLoader_example(bs, personID=0, fold_idx=0, splitChannel = False, mmWaveNorm = True):
    #Load your data here
    train_mmwave = torch.randn(96, 8, 1024)
    train_ecg = torch.randn(96, 1024)

    test_mmwave = torch.randn(96, 8, 1024)
    test_ecg = torch.randn(96, 1024)

    ref_ecg = torch.randn(96, 1024)

    def patchingmmWave(inputSignal):
        B = inputSignal.shape[0]
        C = inputSignal.shape[1]
        inputSignal = inputSignal.reshape(B,C,32,32)
        return inputSignal
    
    def patchingECG(inputSignal):
        B = inputSignal.shape[0]
        inputSignal = inputSignal.reshape(B,32,32)
        inputSignal = inputSignal.unsqueeze(1)
        return inputSignal
    
    train_mmwave = patchingmmWave(train_mmwave)
    train_ecg = patchingECG(train_ecg)
    
    test_mmwave = patchingmmWave(test_mmwave)
    test_ecg = patchingECG(test_ecg)

    ref_ecg = patchingECG(ref_ecg)

    class DataSet(Dataset):
        def __init__(self, x : torch.Tensor, y : torch.Tensor, y_ref : torch.Tensor):
            self.x = x
            self.y = y
            self.y_ref = y_ref
        
        def __len__(self):
            return self.x.shape[0]

        def __getitem__(self, index):
            return [self.x[index], self.y[index], self.y_ref[index]] 

    trainECGDataSet = DataSet(train_mmwave.float(), train_ecg.float(), ref_ecg.float())
    testECGDataSet = DataSet(test_mmwave.float(), test_ecg.float(), ref_ecg.float())
    
    train_ecg_loader = DataLoader(trainECGDataSet, batch_size=bs, shuffle=False)
    test_ecg_loader = DataLoader(testECGDataSet, batch_size=bs, shuffle=False)

    return train_ecg_loader, test_ecg_loader

def main(args):
    """
    Trains AirECG model.
    """
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."

    # Setup DDP:
    dist.init_process_group("nccl")
    assert args.global_batch_size % dist.get_world_size() == 0, f"Batch size must be divisible by world size."
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")

    # Setup an experiment folder:
    if rank == 0:
        os.makedirs(args.results_dir, exist_ok=True)  # Make results folder (holds all experiment subfolders)
        experiment_index = len(glob(f"{args.results_dir}/*"))
        model_string_name = 'AirECG'
        experiment_dir = f"{args.results_dir}/{experiment_index:03d}-{model_string_name}"  # Create an experiment folder
        checkpoint_dir = f"{experiment_dir}/checkpoints"  # Stores saved model checkpoints
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(experiment_dir)
        logger.info(f"Experiment directory created at {experiment_dir}")
    else:
        logger = create_logger(None)

    # Create model:
    latent_size = 32
    model = AirECG_model(
        input_size=latent_size,
        mm_channels = args.mmWave_channels,
    )
    ckpt_path = args.ckpt
    if ckpt_path!= None:
        state_dict = extract_model(ckpt_path)
        model.load_state_dict(state_dict)

    # Note that parameter initialization is done within the DiT constructor
    ema = deepcopy(model).to(device)  # Create an EMA of the model for use after training
    requires_grad(ema, False)
    model = DDP(model.to(device), device_ids=[rank])
    diffusion = create_diffusion(timestep_respacing="")  # default: 1000 steps, linear noise schedule

    logger.info(f"DiT Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Setup optimizer (we used default Adam betas=(0.9, 0.999) and a constant learning rate of 1e-4 in our paper):
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0)

    in_channels = args.mmWave_channels

    if in_channels == 8:
        train_mmwave_loader, test_mmwave_loader = DataLoader_example(bs = int(args.global_batch_size), splitChannel=False, mmWaveNorm = True)
    else:
        pass

    # Prepare models for training:
    update_ema(ema, model.module, decay=0)  # Ensure EMA is initialized with synced weights
    model.train()  # important! This enables embedding dropout for classifier-free guidance
    ema.eval()  # EMA model should always be in eval mode

    # Variables for monitoring/logging purposes:
    train_steps = 0
    log_steps = 0
    running_loss = 0
    start_time = time()

    logger.info(f"Training for {args.epochs} epochs...")
    for epoch in range(args.epochs):
        # sampler.set_epoch(epoch)
        logger.info(f"Beginning epoch {epoch}...")
        model.train()
        for batch_idx, (mmwave, ecg, ref) in enumerate(tqdm(train_mmwave_loader)):
            n = ecg.shape[0]
            ecg = ecg.to(device)
            mmwave = mmwave.to(device)
            ref = ref.to(device)    #Reference ECG for calibration guidance

            t = torch.randint(0, diffusion.num_timesteps, (ecg.shape[0],), device=device)
            model_kwargs = dict(y1=mmwave, y2=ref)
            loss_dict = diffusion.training_losses(model, ecg, t, model_kwargs)
            loss = loss_dict["loss"].mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            update_ema(ema, model.module)

            # Log loss values:
            running_loss += loss.item()
            log_steps += 1
            train_steps += 1
            if train_steps % args.log_every == 0:
                # Measure training speed:
                torch.cuda.synchronize()
                end_time = time()
                steps_per_sec = log_steps / (end_time - start_time)
                # Reduce loss history over all processes:
                avg_loss = torch.tensor(running_loss / log_steps, device=device)
                dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                avg_loss = avg_loss.item() / dist.get_world_size()
                logger.info(f"(step={train_steps:07d}) Train Loss: {avg_loss:.4f}, Train Steps/Sec: {steps_per_sec:.2f}")
                # Reset monitoring variables:
                running_loss = 0
                log_steps = 0
                start_time = time()

            # Save DiT checkpoint:
            if train_steps % args.ckpt_every == 0 and train_steps > 0:
                if rank == 0:
                    checkpoint = {
                        "model": model.module.state_dict(),
                        "ema": ema.state_dict(),
                        "opt": opt.state_dict(),
                        "args": args
                    }
                    checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
                    torch.save(checkpoint, checkpoint_path)
                    logger.info(f"Saved checkpoint to {checkpoint_path}")
                dist.barrier()

        if (epoch+1)%30 == 0:
            model.eval()
            for batch_idx, (mmwave, ecg, ref) in enumerate(tqdm(test_mmwave_loader)):
                mmwave = mmwave[0:16]
                ecg = ecg[0:16]
                ref = ref[0:16]
                n = ecg.shape[0]
                mmwave = mmwave.to(device)
                ref = ref.to(device)

                z = torch.randn(n, 1, latent_size, latent_size, device=device)
                # Setup guidance:
                
                model_kwargs = dict(y1=mmwave, y2=ref)
                # Sample images:
                samples = diffusion.p_sample_loop(
                    model.module.forward, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=True, device=device
                )

                mmwave = mmwave.reshape(n,in_channels,1024)[:,0,:]
                ecg = ecg.reshape(n, 1024)
                samples = samples.reshape(n, 1024)
                ref = ref.reshape(n, 1024).cpu().numpy()
                mmwave = mmwave.cpu().numpy()
                ecg = ecg.cpu().numpy()
                samples = samples.cpu().numpy()
                sample_images(ref, mmwave, ecg, samples, batch_idx, f"{checkpoint_dir}/{train_steps:07d}_test.jpg")
                break
            
            for batch_idx, (mmwave, ecg, ref) in enumerate(tqdm(train_mmwave_loader)):
                mmwave = mmwave[0:16]
                ecg = ecg[0:16]
                ref = ref[0:16]
                n = ecg.shape[0]
                mmwave = mmwave.to(device)
                ref = ref.to(device)

                z = torch.randn(n, 1, latent_size, latent_size, device=device)
                # Setup guidance:
                
                model_kwargs = dict(y1=mmwave, y2=ref)
                # Sample images:
                samples = diffusion.p_sample_loop(
                    model.module.forward, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=True, device=device
                )

                mmwave = mmwave.reshape(n,in_channels,1024)[:,0,:]
                ecg = ecg.reshape(n, 1024)
                samples = samples.reshape(n, 1024)
                ref = ref.reshape(n, 1024).cpu().numpy()
                mmwave = mmwave.cpu().numpy()
                ecg = ecg.cpu().numpy()
                samples = samples.cpu().numpy()
                sample_images(ref, mmwave, ecg, samples, batch_idx, f"{checkpoint_dir}/{train_steps:07d}_train.jpg")
                break
    model.eval()  # important! This disables randomized embedding dropout
    # do any sampling/FID calculation/etc. with ema (or model) in eval mode ...

    logger.info("Done!")
    cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--mmWave-channels", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=1400)
    parser.add_argument("--global-batch-size", type=int, default=64)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=322)
    parser.add_argument("--ckpt-every", type=int, default=19320)
    parser.add_argument("--ckpt", type=str, default=None, help="Load your checkpoint here.")
    args = parser.parse_args()
    main(args)
