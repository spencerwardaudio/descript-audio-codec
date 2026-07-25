"""train_fsq.py — DAC-FSQ training script.

Adapted from scripts/train.py with three changes:
  1. Uses DAC_FSQ (FSQ quantizer) instead of DAC (RVQ).
  2. build_dataset() accepts a CSV filelist in addition to folders dict.
  3. W&B logging added alongside TensorBoard.

Usage:
    python scripts/train_fsq.py --args.load conf/fsd50k_fsq.yml --save_path ckpt/fsq_run

W&B environment variables:
    WANDB_PROJECT   project name (default: dac-fsq-fsd50k)
    WANDB_NAME      run name
"""

import os
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path

import argbind
import torch
from audiotools import AudioSignal
from audiotools import ml
from audiotools.core import util
from audiotools.data import transforms
from audiotools.data.datasets import AudioDataset
from audiotools.data.datasets import AudioLoader
from audiotools.data.datasets import ConcatDataset
from audiotools.ml.decorators import timer
from audiotools.ml.decorators import Tracker
from audiotools.ml.decorators import when
from torch.utils.tensorboard import SummaryWriter

import dac

warnings.filterwarnings("ignore", category=UserWarning)

# W&B (optional)
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    wandb = None

# Module-level run handle; set by train(), read by train_loop / val_loop.
_wandb_run = None

# Enable cudnn autotuner to speed up training.
torch.backends.cudnn.benchmark = bool(int(os.getenv("CUDNN_BENCHMARK", 1)))

# Optimizers
AdamW = argbind.bind(torch.optim.AdamW, "generator", "discriminator")
Accelerator = argbind.bind(ml.Accelerator, without_prefix=True)


@argbind.bind("generator", "discriminator")
def ExponentialLR(optimizer, gamma: float = 1.0):
    return torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma)


# Models — DAC_FSQ instead of DAC
DAC_FSQ = argbind.bind(dac.model.DAC_FSQ)
Discriminator = argbind.bind(dac.model.Discriminator)

# Data
AudioDataset = argbind.bind(AudioDataset, "train", "val")
AudioLoader = argbind.bind(AudioLoader, "train", "val")

# Transforms
filter_fn = lambda fn: hasattr(fn, "transform") and fn.__qualname__ not in [
    "BaseTransform",
    "Compose",
    "Choose",
]
tfm = argbind.bind_module(transforms, "train", "val", filter_fn=filter_fn)

# Loss
filter_fn = lambda fn: hasattr(fn, "forward") and "Loss" in fn.__name__
losses = argbind.bind_module(dac.nn.loss, filter_fn=filter_fn)


def get_infinite_loader(dataloader):
    while True:
        for batch in dataloader:
            yield batch


@argbind.bind("train", "val")
def build_transform(
    augment_prob: float = 1.0,
    preprocess: list = ["Identity"],
    augment: list = ["Identity"],
    postprocess: list = ["Identity"],
):
    to_tfm = lambda l: [getattr(tfm, x)() for x in l]
    preprocess = transforms.Compose(*to_tfm(preprocess), name="preprocess")
    augment = transforms.Compose(*to_tfm(augment), name="augment", prob=augment_prob)
    postprocess = transforms.Compose(*to_tfm(postprocess), name="postprocess")
    transform = transforms.Compose(preprocess, augment, postprocess)
    return transform


@argbind.bind("train", "val", "test")
def build_dataset(
    sample_rate: int,
    folders: dict = None,
    filelist: str = None,
):
    """Build an AudioDataset from either a CSV filelist (one path per line)
    or a dict of {name: [folder_paths]}.  filelist takes priority.

    The filelist format is identical to what Q2D2 uses
    (datasets/fsd50k_train.csv / datasets/fsd50k_val.csv).
    """
    if filelist is not None:
        with open(filelist) as f:
            sources = [line.strip() for line in f if line.strip()]
        loader = AudioLoader(sources=sources)
        transform = build_transform()
        dataset = AudioDataset(loader, sample_rate, transform=transform)
        return dataset

    # Fallback: original folders-based loading
    datasets = []
    for _, v in folders.items():
        loader = AudioLoader(sources=v)
        transform = build_transform()
        dataset = AudioDataset(loader, sample_rate, transform=transform)
        datasets.append(dataset)
    dataset = ConcatDataset(datasets)
    dataset.transform = transform
    return dataset


@dataclass
class State:
    generator: DAC_FSQ
    optimizer_g: AdamW
    scheduler_g: ExponentialLR

    discriminator: Discriminator
    optimizer_d: AdamW
    scheduler_d: ExponentialLR

    stft_loss: losses.MultiScaleSTFTLoss
    mel_loss: losses.MelSpectrogramLoss
    gan_loss: losses.GANLoss
    waveform_loss: losses.L1Loss

    train_data: AudioDataset
    val_data: AudioDataset

    tracker: Tracker


@argbind.bind(without_prefix=True)
def load(
    args,
    accel: ml.Accelerator,
    tracker: Tracker,
    save_path: str,
    resume: bool = False,
    tag: str = "latest",
    load_weights: bool = False,
):
    generator, g_extra = None, {}
    discriminator, d_extra = None, {}

    if resume:
        kwargs = {
            "folder": f"{save_path}/{tag}",
            "map_location": "cpu",
            "package": not load_weights,
        }
        tracker.print(f"Resuming from {str(Path('.').absolute())}/{kwargs['folder']}")
        if (Path(kwargs["folder"]) / "dac_fsq").exists():
            generator, g_extra = DAC_FSQ.load_from_folder(**kwargs)
        if (Path(kwargs["folder"]) / "discriminator").exists():
            discriminator, d_extra = Discriminator.load_from_folder(**kwargs)

    generator = DAC_FSQ() if generator is None else generator
    discriminator = Discriminator() if discriminator is None else discriminator

    tracker.print(generator)
    tracker.print(discriminator)

    generator = accel.prepare_model(generator)
    discriminator = accel.prepare_model(discriminator)

    # Debug: Check if models are actually on GPU after prepare_model
    try:
        gen_device = next(generator.parameters()).device
        disc_device = next(discriminator.parameters()).device
        tracker.print(f"Generator device after prepare_model: {gen_device}")
        tracker.print(f"Discriminator device after prepare_model: {disc_device}")
    except StopIteration:
        tracker.print("WARNING: Could not determine model device (no parameters)")

    # Explicitly move to device if prepare_model didn't do it
    if accel.device != 'cpu':
        generator = generator.to(accel.device)
        discriminator = discriminator.to(accel.device)
        tracker.print(f"Explicitly moved models to {accel.device}")

    # Verify models are on the correct device
    gen_device_final = next(generator.parameters()).device
    disc_device_final = next(discriminator.parameters()).device
    tracker.print(f"Generator final device: {gen_device_final}")
    tracker.print(f"Discriminator final device: {disc_device_final}")
    
    # Assert GPU placement if CUDA is requested
    if accel.device == 'cuda':
        assert str(gen_device_final).startswith('cuda'), f"Generator not on CUDA! Device: {gen_device_final}"
        assert str(disc_device_final).startswith('cuda'), f"Discriminator not on CUDA! Device: {disc_device_final}"
        tracker.print("✓ GPU placement verified")

    with argbind.scope(args, "generator"):
        optimizer_g = AdamW(generator.parameters(), use_zero=accel.use_ddp)
        scheduler_g = ExponentialLR(optimizer_g)
    with argbind.scope(args, "discriminator"):
        optimizer_d = AdamW(discriminator.parameters(), use_zero=accel.use_ddp)
        scheduler_d = ExponentialLR(optimizer_d)

    if "optimizer.pth" in g_extra:
        optimizer_g.load_state_dict(g_extra["optimizer.pth"])
    if "scheduler.pth" in g_extra:
        scheduler_g.load_state_dict(g_extra["scheduler.pth"])
    if "tracker.pth" in g_extra:
        tracker.load_state_dict(g_extra["tracker.pth"])

    if "optimizer.pth" in d_extra:
        optimizer_d.load_state_dict(d_extra["optimizer.pth"])
    if "scheduler.pth" in d_extra:
        scheduler_d.load_state_dict(d_extra["scheduler.pth"])

    sample_rate = accel.unwrap(generator).sample_rate
    with argbind.scope(args, "train"):
        train_data = build_dataset(sample_rate)
    with argbind.scope(args, "val"):
        val_data = build_dataset(sample_rate)

    waveform_loss = losses.L1Loss()
    stft_loss = losses.MultiScaleSTFTLoss()
    mel_loss = losses.MelSpectrogramLoss()
    gan_loss = losses.GANLoss(discriminator)

    return State(
        generator=generator,
        optimizer_g=optimizer_g,
        scheduler_g=scheduler_g,
        discriminator=discriminator,
        optimizer_d=optimizer_d,
        scheduler_d=scheduler_d,
        waveform_loss=waveform_loss,
        stft_loss=stft_loss,
        mel_loss=mel_loss,
        gan_loss=gan_loss,
        tracker=tracker,
        train_data=train_data,
        val_data=val_data,
    )


@timer()
@torch.no_grad()
def val_loop(batch, state, accel):
    state.generator.eval()
    batch = util.prepare_batch(batch, accel.device)
    signal = state.val_data.transform(
        batch["signal"].clone(), **batch["transform_args"]
    )

    out = state.generator(signal.audio_data, signal.sample_rate)
    recons = AudioSignal(out["audio"], signal.sample_rate)

    out_dict = {
        "loss": state.mel_loss(recons, signal),
        "mel/loss": state.mel_loss(recons, signal),
        "stft/loss": state.stft_loss(recons, signal),
        "waveform/loss": state.waveform_loss(recons, signal),
    }

    # W&B val logging
    global _wandb_run
    if _wandb_run is not None:
        wandb.log(
            {
                "val/" + k: v.item() if hasattr(v, "item") else v
                for k, v in out_dict.items()
            }
        )

    return out_dict


@timer()
def train_loop(state, batch, accel, lambdas):
    state.generator.train()
    state.discriminator.train()
    output = {}

    batch = util.prepare_batch(batch, accel.device)
    with torch.no_grad():
        signal = state.train_data.transform(
            batch["signal"].clone(), **batch["transform_args"]
        )

    with accel.autocast():
        out = state.generator(signal.audio_data, signal.sample_rate)
        recons = AudioSignal(out["audio"], signal.sample_rate)
        # FSQ returns zeros here — included for completeness / TensorBoard
        commitment_loss = out["vq/commitment_loss"]
        codebook_loss = out["vq/codebook_loss"]

    with accel.autocast():
        output["adv/disc_loss"] = state.gan_loss.discriminator_loss(recons, signal)

    state.optimizer_d.zero_grad()
    accel.backward(output["adv/disc_loss"])
    accel.scaler.unscale_(state.optimizer_d)
    output["other/grad_norm_d"] = torch.nn.utils.clip_grad_norm_(
        state.discriminator.parameters(), 10.0
    )
    accel.step(state.optimizer_d)
    state.scheduler_d.step()

    with accel.autocast():
        output["stft/loss"] = state.stft_loss(recons, signal)
        output["mel/loss"] = state.mel_loss(recons, signal)
        output["waveform/loss"] = state.waveform_loss(recons, signal)
        (
            output["adv/gen_loss"],
            output["adv/feat_loss"],
        ) = state.gan_loss.generator_loss(recons, signal)
        output["vq/commitment_loss"] = commitment_loss   # always 0 for FSQ
        output["vq/codebook_loss"] = codebook_loss       # always 0 for FSQ
        # Total loss: only non-zero lambdas from fsd50k_fsq.yml contribute
        output["loss"] = sum(
            [v * output[k] for k, v in lambdas.items() if k in output]
        )

    state.optimizer_g.zero_grad()
    accel.backward(output["loss"])
    accel.scaler.unscale_(state.optimizer_g)
    output["other/grad_norm"] = torch.nn.utils.clip_grad_norm_(
        state.generator.parameters(), 1e3
    )
    accel.step(state.optimizer_g)
    state.scheduler_g.step()
    accel.update()

    output["other/learning_rate"] = state.optimizer_g.param_groups[0]["lr"]
    output["other/batch_size"] = signal.batch_size * accel.world_size

    result = {k: v for k, v in sorted(output.items())}

    # W&B train logging (rank-0 only)
    global _wandb_run
    if _wandb_run is not None:
        wandb.log(
            {k: v.item() if hasattr(v, "item") else v for k, v in result.items()}
        )

    return result


def checkpoint(state, save_iters, save_path):
    metadata = {"logs": state.tracker.history}

    tags = ["latest"]
    state.tracker.print(f"Saving to {str(Path('.').absolute())}")
    if state.tracker.is_best("val", "mel/loss"):
        state.tracker.print(f"Best generator so far")
        tags.append("best")
    if state.tracker.step in save_iters:
        tags.append(f"{state.tracker.step // 1000}k")

    for tag in tags:
        generator_extra = {
            "optimizer.pth": state.optimizer_g.state_dict(),
            "scheduler.pth": state.scheduler_g.state_dict(),
            "tracker.pth": state.tracker.state_dict(),
            "metadata.pth": metadata,
        }
        accel.unwrap(state.generator).metadata = metadata
        accel.unwrap(state.generator).save_to_folder(
            f"{save_path}/{tag}", generator_extra
        )
        discriminator_extra = {
            "optimizer.pth": state.optimizer_d.state_dict(),
            "scheduler.pth": state.scheduler_d.state_dict(),
        }
        accel.unwrap(state.discriminator).save_to_folder(
            f"{save_path}/{tag}", discriminator_extra
        )


@torch.no_grad()
def save_samples(state, val_idx, writer):
    state.tracker.print("Saving audio samples to TensorBoard")
    state.generator.eval()

    samples = [state.val_data[idx] for idx in val_idx]
    batch = state.val_data.collate(samples)
    batch = util.prepare_batch(batch, accel.device)
    signal = state.train_data.transform(
        batch["signal"].clone(), **batch["transform_args"]
    )

    out = state.generator(signal.audio_data, signal.sample_rate)
    recons = AudioSignal(out["audio"], signal.sample_rate)

    audio_dict = {"recons": recons}
    if state.tracker.step == 0:
        audio_dict["signal"] = signal

    for k, v in audio_dict.items():
        for nb in range(v.batch_size):
            v[nb].cpu().write_audio_to_tb(
                f"{k}/sample_{nb}.wav", writer, state.tracker.step
            )


def validate(state, val_dataloader, accel):
    for batch in val_dataloader:
        output = val_loop(batch, state, accel)
    if hasattr(state.optimizer_g, "consolidate_state_dict"):
        state.optimizer_g.consolidate_state_dict()
        state.optimizer_d.consolidate_state_dict()
    return output


@argbind.bind(without_prefix=True)
def train(
    args,
    accel: ml.Accelerator,
    seed: int = 0,
    save_path: str = "ckpt",
    training_epochs: int = 50,
    num_iters: int = None,
    save_iters: list = [10000, 50000, 100000],
    sample_freq: int = 5000,
    valid_freq: int = 1000,
    batch_size: int = 12,
    val_batch_size: int = 12,
    num_workers: int = 4,
    val_idx: list = [0, 1, 2, 3, 4, 5, 6, 7],
    lambdas: dict = {
        "mel/loss": 15.0,
        "adv/feat_loss": 2.0,
        "adv/gen_loss": 1.0,
        # vq/commitment_loss and vq/codebook_loss intentionally absent:
        # FSQ returns zeros for them; excluding from lambdas keeps the loss clean.
    },
):
    util.seed(seed)
    Path(save_path).mkdir(exist_ok=True, parents=True)
    writer = (
        SummaryWriter(log_dir=f"{save_path}/logs") if accel.local_rank == 0 else None
    )
    tracker = Tracker(
        writer=writer, log_file=f"{save_path}/log.txt", rank=accel.local_rank
    )
    tracker.print(f"Accelerator device: {accel.device}")
    tracker.print(f"Accelerator amp: {accel.amp}")
    tracker.print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")

    state = load(args, accel, tracker, save_path)
    train_dataloader = accel.prepare_dataloader(
        state.train_data,
        start_idx=state.tracker.step * batch_size,
        num_workers=num_workers,
        batch_size=batch_size,
        collate_fn=state.train_data.collate,
        persistent_workers=True if num_workers > 0 else False,
        pin_memory=True,
    )
    steps_per_epoch = len(train_dataloader)
    if num_iters is None:
        num_iters = training_epochs * steps_per_epoch
    tracker.print(
        f"Training DAC-FSQ for {training_epochs} epochs "
        f"({num_iters} iterations, {steps_per_epoch} steps/epoch)"
    )

    # ── W&B init (rank-0 only) ────────────────────────────────────────────────
    global _wandb_run
    if WANDB_AVAILABLE and accel.local_rank == 0:
        _wandb_run = wandb.init(
            project=os.environ.get("WANDB_PROJECT", "dac-fsq-fsd50k"),
            name=os.environ.get("WANDB_NAME", "dac-fsq-run"),
            config={
                "training_epochs": training_epochs,
                "num_iters": num_iters,
                "steps_per_epoch": steps_per_epoch,
                "lambdas": lambdas,
                "save_path": save_path,
                "batch_size": batch_size,
            },
            resume="allow",
        )
        tracker.print(
            f"W&B run: {_wandb_run.url if hasattr(_wandb_run, 'url') else 'initialized'}"
        )

    train_dataloader = get_infinite_loader(train_dataloader)
    val_dataloader = accel.prepare_dataloader(
        state.val_data,
        start_idx=0,
        num_workers=num_workers,
        batch_size=val_batch_size,
        collate_fn=state.val_data.collate,
        persistent_workers=True if num_workers > 0 else False,
    )

    # Wrap functions for TensorBoard + progress bars.
    global train_loop, val_loop, validate, save_samples, checkpoint
    train_loop = tracker.log("train", "value", history=False)(
        tracker.track("train", num_iters, completed=state.tracker.step)(train_loop)
    )
    val_loop = tracker.track("val", len(val_dataloader))(val_loop)
    validate = tracker.log("val", "mean")(validate)

    save_samples = when(lambda: accel.local_rank == 0)(save_samples)
    checkpoint = when(lambda: accel.local_rank == 0)(checkpoint)

    with tracker.live:
        for tracker.step, batch in enumerate(train_dataloader, start=tracker.step):
            train_loop(state, batch, accel, lambdas)

            last_iter = (
                tracker.step == num_iters - 1 if num_iters is not None else False
            )
            if tracker.step % sample_freq == 0 or last_iter:
                save_samples(state, val_idx, writer)

            if tracker.step % valid_freq == 0 or last_iter:
                validate(state, val_dataloader, accel)
                checkpoint(state, save_iters, save_path)
                tracker.done("val", f"Iteration {tracker.step}")

            if last_iter:
                break

    # ── W&B finish ────────────────────────────────────────────────────────────
    if _wandb_run is not None:
        wandb.finish()
        _wandb_run = None


if __name__ == "__main__":
    args = argbind.parse_args()
    args["args.debug"] = int(os.getenv("LOCAL_RANK", 0)) == 0
    with argbind.scope(args):
        with Accelerator() as accel:
            if accel.local_rank != 0:
                sys.tracebacklimit = 0
            train(args, accel)
