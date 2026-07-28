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
from typing import Callable

import argbind
import torch
import soundfile
import torchaudio
import numpy as np
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

# Add project root to path for shared utilities
_PROJ_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJ_ROOT))

from dataloader_aug.audio_preprocessing import normalize_rms_snr
from dataloader_aug.dataset_paths import get_dataset_config

# Validate dataset paths on module load
_dataset_config = get_dataset_config()
assert _dataset_config.train_csv.exists(), \
    f"❌ DAC-FSQ training CSV missing: {_dataset_config.train_csv}"
assert _dataset_config.val_csv.exists(), \
    f"❌ DAC-FSQ validation CSV missing: {_dataset_config.val_csv}"

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
# Steps per epoch — set by train() so train_loop/val_loop can compute epoch number.
_steps_per_epoch: int = 1

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


class SimpleAudioDataset(torch.utils.data.Dataset):
    """Fast tensor-based CSV audio dataset compatible with multiprocess DataLoader.
    
    Uses soundfile.read() for fast loading (10-15x faster than AudioSignal),
    returns raw tensors with metadata, converted to AudioSignal in training loop only when needed.
    Modeled on Q2D2's VocosDataset for optimal GPU performance.
    
    Parameters
    ----------
    filelist : str
        Path to CSV file with one audio path per line
    sample_rate : int
        Target sample rate for resampling
    duration : float
        Duration of audio excerpts in seconds
    n_examples : int
        Number of examples (dataset length)
    transform : Callable, optional
        Transform to apply to audio samples
    train : bool, optional
        Whether this is a training dataset (random crops) or validation (deterministic), by default True
    loudness_cutoff : float, optional
        UNUSED - kept for API compatibility
    """
    
    def __init__(
        self,
        filelist: str,
        sample_rate: int,
        duration: float = 3.0,
        n_examples: int = 1000,
        transform: Callable = None,
        train: bool = True,
        loudness_cutoff: float = -40,
    ):
        print(f"[SimpleAudioDataset] Initializing from: {filelist}")
        print(f"[SimpleAudioDataset] Duration: {duration}s, Sample rate: {sample_rate} Hz, Examples: {n_examples}")
        
        with open(filelist) as f:
            self.file_list = [line.strip() for line in f if line.strip()]
        
        print(f"[SimpleAudioDataset] Read {len(self.file_list)} audio paths from CSV")
        print(f"[SimpleAudioDataset] First path: {self.file_list[0]}")
        
        # Check if first file exists
        if Path(self.file_list[0]).exists():
            print(f"[SimpleAudioDataset] \u2713 First file exists")
        else:
            print(f"[SimpleAudioDataset] \u2717 WARNING: First file NOT FOUND: {self.file_list[0]}")
        
        self.sample_rate = sample_rate
        self.duration = duration
        self.n_examples = n_examples
        self.transform = transform
        self.train = train
        self.loudness_cutoff = loudness_cutoff
        self.num_samples = int(duration * sample_rate)
        self.load_count = 0  # Track how many samples loaded
        
        print(f"[SimpleAudioDataset] Initialization complete")
        
    def __len__(self) -> int:
        return self.n_examples
    
    def __getitem__(self, index: int):
        self.load_count += 1
        
        # Debug: Log only during first epoch (first n_examples calls)
        # After that, logging is suppressed to avoid spam
        if self.load_count <= self.n_examples:
            # Log every 100th sample or first 5 samples during first epoch
            if index < 5 or index % 100 == 0:
                import sys
                print(f"[DataLoader] Loading sample {index}/{self.n_examples} (total loads: {self.load_count})", file=sys.stderr, flush=True)
                print(f"[DataLoader]   File index: {index % len(self.file_list)}/{len(self.file_list)}", file=sys.stderr, flush=True)
        elif self.load_count == self.n_examples + 1:
            # Log once when verbose logging is disabled
            import sys
            print(f"[DataLoader] First epoch complete ({self.n_examples} samples loaded). Verbose logging disabled.", file=sys.stderr, flush=True)
        
        # Wrap around if n_examples > len(file_list)
        file_idx = index % len(self.file_list)
        audio_path = self.file_list[file_idx]
        
        # Load audio - use simple random crop instead of expensive salient_excerpt
        try:
            # soundfile.read is ~10x faster than AudioSignal for full files
            y, sr = soundfile.read(audio_path)
            y = torch.tensor(y).float()
            
            # Handle mono/stereo
            if y.ndim == 1:
                y = y.unsqueeze(0)  # [samples] -> [1, samples]
            elif y.ndim == 2:
                y = y.mean(dim=-1, keepdim=True).T  # [samples, channels] -> [1, samples]
            
            # Apply RMS/SNR normalization (preserves amplitude relationships)
            y = normalize_rms_snr(
                y,
                target_snr_db=40.0,
                train_mode=self.train,
                snr_variation_db=5.0,
                audio_path=audio_path,
                source_identifier="DAC-FSQ/SimpleAudioDataset"
            )
            
            # Resample if needed
            if sr != self.sample_rate:
                y = torchaudio.functional.resample(y, orig_freq=sr, new_freq=self.sample_rate)
            
            # Crop or pad to exact duration
            if y.size(-1) < self.num_samples:
                # Pad by repeating
                pad_length = self.num_samples - y.size(-1)
                padding_tensor = y.repeat(1, 1 + pad_length // y.size(-1))
                y = torch.cat((y, padding_tensor[:, :pad_length]), dim=1)
            elif self.train:
                # Random crop
                start = np.random.randint(low=0, high=y.size(-1) - self.num_samples + 1)
                y = y[:, start : start + self.num_samples]
            else:
                # Deterministic first crop for validation
                y = y[:, :self.num_samples]
                
        except Exception as e:
            # Fallback to zeros if file loading fails
            import sys
            print(f"[DataLoader] ERROR loading {audio_path}: {e}", file=sys.stderr, flush=True)
            y = torch.zeros(1, self.num_samples)
        
        # Build return dict with raw tensor
        item = {
            "audio": y,
            "sample_rate": self.sample_rate,
            "path": str(audio_path),
            "index": file_idx,
        }
        
        # Transform args (no transform applied yet - done in training loop)
        item["transform_args"] = {}
        
        return item
    
    def collate(self, batch):
        """Collate function for DataLoader - stacks tensors instead of AudioSignals."""
        # Stack tensors into batch
        audio_batch = torch.stack([item["audio"] for item in batch], dim=0)  # [B, 1, T]
        return {
            "audio": audio_batch,
            "sample_rate": batch[0]["sample_rate"],
            "paths": [item["path"] for item in batch],
            "indices": [item["index"] for item in batch],
            "transform_args": batch[0]["transform_args"],
        }


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
    use_simple_dataset: bool = True,
    duration: float = 3.0,
    n_examples: int = 1000,
):
    """Build an AudioDataset from either a CSV filelist (one path per line)
    or a dict of {name: [folder_paths]}.  filelist takes priority.

    The filelist format is identical to what Q2D2 uses
    (datasets/fsd50k_train.csv / datasets/fsd50k_val.csv).
    
    Parameters
    ----------
    use_simple_dataset : bool
        If True and filelist is provided, use SimpleAudioDataset (fast, multiprocess-compatible).
        If False, use AudioLoader (slow initialization, ~4s per worker).
        Default True to enable num_workers > 0 on shared storage.
    """
    if filelist is not None:
        transform = build_transform()
        
        if use_simple_dataset:
            # Fast CSV loading compatible with multiprocess DataLoader
            dataset = SimpleAudioDataset(
                filelist=filelist,
                sample_rate=sample_rate,
                duration=duration,
                n_examples=n_examples,
                transform=transform,
                train=("train" in filelist.lower()),  # Heuristic: train vs val
            )
        else:
            # Original AudioLoader path (slow with num_workers > 0)
            with open(filelist) as f:
                sources = [line.strip() for line in f if line.strip()]
            loader = AudioLoader(sources=sources)
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
    expected_mem_mib = 0.0  # Will be calculated and returned

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
    
    # Count parameters and calculate expected memory footprint
    tracker.print("\n" + "="*70)
    tracker.print("=== Model Parameter Analysis ===")
    tracker.print("="*70)
    gen_params = sum(p.numel() for p in generator.parameters())
    disc_params = sum(p.numel() for p in discriminator.parameters())
    total_params = gen_params + disc_params
    # float32 = 4 bytes per parameter
    expected_mem_mib = (total_params * 4) / (1024**2)
    tracker.print(f"Generator parameters: {gen_params:,} ({gen_params * 4 / 1024**2:.1f} MiB)")
    tracker.print(f"Discriminator parameters: {disc_params:,} ({disc_params * 4 / 1024**2:.1f} MiB)")
    tracker.print(f"Total parameters: {total_params:,} ({expected_mem_mib:.1f} MiB expected)")
    tracker.print("="*70)
    
    # Check initial device placement
    gen_device_init = next(generator.parameters()).device
    disc_device_init = next(discriminator.parameters()).device
    tracker.print(f"\nGenerator initial device: {gen_device_init}")
    tracker.print(f"Discriminator initial device: {disc_device_init}")

    # CRITICAL FIX: Move to GPU BEFORE prepare_model (prepare_model wrapping prevents .to() from working)
    if accel.device != 'cpu':
        tracker.print("\n" + "="*70)
        tracker.print(f"=== Moving Models to {accel.device} ===")
        tracker.print("="*70)
        mem_before_move = torch.cuda.memory_allocated() / 1024**2
        tracker.print(f"GPU memory before .to(): {mem_before_move:.1f} MiB")
        
        # Try moving to GPU
        tracker.print(f"Calling generator.to({accel.device})...")
        generator = generator.to(accel.device)
        tracker.print(f"Calling discriminator.to({accel.device})...")
        discriminator = discriminator.to(accel.device)
        
        # Check memory IMMEDIATELY after .to() - this is the critical diagnostic
        mem_after_move = torch.cuda.memory_allocated() / 1024**2
        mem_delta = mem_after_move - mem_before_move
        tracker.print(f"\nGPU memory after .to(): {mem_after_move:.1f} MiB (delta: +{mem_delta:.1f} MiB)")
        tracker.print(f"Expected delta: ~{expected_mem_mib:.1f} MiB")
        
        # Verify parameter devices
        gen_device_before = next(generator.parameters()).device
        disc_device_before = next(discriminator.parameters()).device
        tracker.print(f"\nGenerator parameter device: {gen_device_before}")
        tracker.print(f"Discriminator parameter device: {disc_device_before}")
        
        # Check buffer devices (batch norm stats, etc.)
        gen_buffers = list(generator.buffers())
        disc_buffers = list(discriminator.buffers())
        if gen_buffers:
            gen_buffer_device = gen_buffers[0].device
            tracker.print(f"Generator buffer device: {gen_buffer_device}")
        if disc_buffers:
            disc_buffer_device = disc_buffers[0].device
            tracker.print(f"Discriminator buffer device: {disc_buffer_device}")
        
        # Assert device metadata is correct
        assert str(gen_device_before).startswith('cuda'), f"Failed to move generator to GPU: {gen_device_before}"
        assert str(disc_device_before).startswith('cuda'), f"Failed to move discriminator to GPU: {disc_device_before}"
        
        # CRITICAL: Assert memory was actually allocated
        if mem_delta < expected_mem_mib * 0.5:  # Should allocate at least 50% of expected
            tracker.print("\n" + "!"*70)
            tracker.print("!!! WARNING: .to() did not allocate expected memory !!!")
            tracker.print("!"*70)
            tracker.print(f"Expected: ~{expected_mem_mib:.1f} MiB")
            tracker.print(f"Actual delta: {mem_delta:.1f} MiB")
            tracker.print(f"Ratio: {mem_delta / expected_mem_mib * 100:.1f}%")
            tracker.print("\nAttempting alternative: .cuda() method...")
            
            # Try .cuda() as alternative
            generator = generator.cuda()
            discriminator = discriminator.cuda()
            mem_after_cuda = torch.cuda.memory_allocated() / 1024**2
            mem_delta_cuda = mem_after_cuda - mem_before_move
            tracker.print(f"GPU memory after .cuda(): {mem_after_cuda:.1f} MiB (delta: +{mem_delta_cuda:.1f} MiB)")
            
            if mem_delta_cuda < expected_mem_mib * 0.5:
                tracker.print("\n" + "X"*70)
                tracker.print("XXX CRITICAL ERROR: Cannot move models to GPU! XXX")
                tracker.print("X"*70)
                raise RuntimeError(
                    f"Cannot move models to GPU! Tried .to() and .cuda(), neither allocated memory.\n"
                    f"Expected: {expected_mem_mib:.1f} MiB, Got: {mem_delta_cuda:.1f} MiB\n"
                    f"Models may be empty or initialized with meta device."
                )
            else:
                tracker.print(f"✓ .cuda() worked! Using .cuda() instead of .to()")
        else:
            tracker.print(f"\n✓ Memory allocation successful ({mem_delta:.1f} MiB allocated)")
            tracker.print("="*70)

    # Now wrap with prepare_model (should preserve GPU placement)
    tracker.print("\n" + "="*70)
    tracker.print("=== Wrapping with prepare_model ===")
    tracker.print("="*70)
    mem_before_prepare = torch.cuda.memory_allocated() / 1024**2 if torch.cuda.is_available() else 0
    tracker.print(f"GPU memory before wrapping: {mem_before_prepare:.1f} MiB")
    
    generator = accel.prepare_model(generator)
    discriminator = accel.prepare_model(discriminator)
    
    mem_after_prepare = torch.cuda.memory_allocated() / 1024**2 if torch.cuda.is_available() else 0
    tracker.print(f"GPU memory after wrapping: {mem_after_prepare:.1f} MiB (delta: {mem_after_prepare - mem_before_prepare:+.1f} MiB)")

    # Verify models are still on the correct device after prepare_model
    gen_device_final = next(generator.parameters()).device
    disc_device_final = next(discriminator.parameters()).device
    tracker.print(f"\nGenerator device after prepare_model: {gen_device_final}")
    tracker.print(f"Discriminator device after prepare_model: {disc_device_final}")
    
    # Assert GPU placement if CUDA is requested
    if accel.device == 'cuda':
        assert str(gen_device_final).startswith('cuda'), f"Generator not on CUDA after prepare_model! Device: {gen_device_final}"
        assert str(disc_device_final).startswith('cuda'), f"Discriminator not on CUDA after prepare_model! Device: {disc_device_final}"
        
        # Verify memory didn't drop (prepare_model shouldn't move back to CPU)
        if mem_after_prepare < mem_before_prepare * 0.9:
            tracker.print("\n" + "X"*70)
            tracker.print("XXX CRITICAL ERROR: prepare_model moved models back to CPU! XXX")
            tracker.print("X"*70)
            raise RuntimeError(
                f"prepare_model moved models back to CPU!\n"
                f"Before: {mem_before_prepare:.1f} MiB, After: {mem_after_prepare:.1f} MiB"
            )
        
        tracker.print("\n✓ GPU placement verified after prepare_model")
        tracker.print("="*70)

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
    
    tracker.print("\n" + "="*70)
    tracker.print("=== Building Datasets ===")
    tracker.print("="*70)
    tracker.print(f"Sample rate: {sample_rate} Hz\n")
    
    with argbind.scope(args, "train"):
        tracker.print("Loading training dataset...")
        train_data = build_dataset(sample_rate)
        tracker.print(f"✓ Train dataset: {type(train_data).__name__}, length={len(train_data)}")
        if hasattr(train_data, 'file_list'):
            tracker.print(f"  Audio files: {len(train_data.file_list)}")
            tracker.print(f"  Duration: {train_data.duration}s ({train_data.num_samples} samples)")
    
    with argbind.scope(args, "val"):
        tracker.print("\nLoading validation dataset...")
        val_data = build_dataset(sample_rate)
        tracker.print(f"✓ Val dataset: {type(val_data).__name__}, length={len(val_data)}")
        if hasattr(val_data, 'file_list'):
            tracker.print(f"  Audio files: {len(val_data.file_list)}")
            tracker.print(f"  Duration: {val_data.duration}s ({val_data.num_samples} samples)")
    
    tracker.print("="*70)

    waveform_loss = losses.L1Loss()
    stft_loss = losses.MultiScaleSTFTLoss()
    mel_loss = losses.MelSpectrogramLoss()
    gan_loss = losses.GANLoss(discriminator)

    state = State(
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
    
    # Return both state and expected memory for validation
    return state, expected_mem_mib


@timer()
@torch.no_grad()
def val_loop(batch, state, accel):
    state.generator.eval()
    # Extract transform_args before prepare_batch (which filters custom keys)
    transform_args = batch.pop("transform_args", {})
    batch = util.prepare_batch(batch, accel.device)
    
    # Convert raw tensor batch to AudioSignal for transforms and losses
    audio_tensor = batch["audio"].squeeze(1)  # [B, 1, T] -> [B, T]
    sample_rate = batch["sample_rate"]
    signal = AudioSignal(audio_tensor, sample_rate)
    
    signal = state.val_data.transform(
        signal.clone(), **transform_args
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
    global _wandb_run, _steps_per_epoch
    if _wandb_run is not None:
        current_epoch = state.tracker.step // _steps_per_epoch
        log_dict = {
            "val/" + k: v.item() if hasattr(v, "item") else v
            for k, v in out_dict.items()
        }
        log_dict["epoch"] = current_epoch
        wandb.log(log_dict)

    return out_dict


@timer()
def train_loop(state, batch, accel, lambdas):
    state.generator.train()
    state.discriminator.train()
    output = {}

    # Extract transform_args before prepare_batch (which filters custom keys)
    transform_args = batch.pop("transform_args", {})
    batch = util.prepare_batch(batch, accel.device)
    
    # Convert raw tensor batch to AudioSignal for transforms and losses
    audio_tensor = batch["audio"].squeeze(1)  # [B, 1, T] -> [B, T]
    sample_rate = batch["sample_rate"]
    signal = AudioSignal(audio_tensor, sample_rate)
    
    with torch.no_grad():
        signal = state.train_data.transform(
            signal.clone(), **transform_args
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
    global _wandb_run, _steps_per_epoch
    if _wandb_run is not None:
        current_epoch = state.tracker.step // _steps_per_epoch
        log_dict = {k: v.item() if hasattr(v, "item") else v for k, v in result.items()}
        log_dict["epoch"] = current_epoch
        wandb.log(log_dict)

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
    # Extract transform_args before prepare_batch (which filters custom keys)
    transform_args = batch.pop("transform_args", {})
    batch = util.prepare_batch(batch, accel.device)
    
    # Convert raw tensor batch to AudioSignal for transforms and generation
    audio_tensor = batch["audio"].squeeze(1)  # [B, 1, T] -> [B, T]
    sample_rate = batch["sample_rate"]
    signal = AudioSignal(audio_tensor, sample_rate)
    
    signal = state.train_data.transform(
        signal.clone(), **transform_args
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
    
    # GPU ping test: ensure basic GPU operations work
    if torch.cuda.is_available():
        tracker.print("\n=== GPU Ping Test ===")
        try:
            test_tensor = torch.randn(1000, 1000).cuda()
            result = test_tensor @ test_tensor.T
            mem_after_ping = torch.cuda.memory_allocated() / 1024**2
            tracker.print(f"✓ GPU ping successful: allocated {mem_after_ping:.1f} MiB for test tensor")
            del test_tensor, result
            torch.cuda.empty_cache()
            tracker.print(f"✓ GPU memory released, ready for model loading")
        except Exception as e:
            tracker.print(f"✗ GPU ping FAILED: {e}")
            raise RuntimeError(f"GPU is not functional: {e}")
    
    # Baseline GPU memory usage (should be ~0 MiB before loading models)
    if torch.cuda.is_available():
        baseline_mem = torch.cuda.memory_allocated() / 1024**2
        tracker.print(f"\nBaseline GPU memory: {baseline_mem:.1f} MiB")

    state, expected_model_mem = load(args, accel, tracker, save_path)
    
    # Final GPU memory check after all models loaded
    final_mem = 0.0  # Initialize for later reference
    if torch.cuda.is_available():
        final_mem = torch.cuda.memory_allocated() / 1024**2
        tracker.print("\n" + "="*70)
        tracker.print("=== Final Memory Check ===")
        tracker.print("="*70)
        tracker.print(f"GPU memory after all models loaded: {final_mem:.1f} MiB")
        tracker.print(f"Expected model memory: {expected_model_mem:.1f} MiB")
        
        # Check if at least 80% of expected memory is allocated (allows some tolerance)
        min_required = expected_model_mem * 0.8
        tracker.print(f"Minimum required (80% of expected): {min_required:.1f} MiB")
        
        # This should pass if load() succeeded
        if final_mem < min_required:
            tracker.print("\n" + "X"*70)
            tracker.print("XXX CRITICAL ERROR: Models not properly loaded to GPU! XXX")
            tracker.print("X"*70)
            tracker.print(f"Expected: {expected_model_mem:.1f} MiB")
            tracker.print(f"Minimum (80%): {min_required:.1f} MiB")
            tracker.print(f"Actual: {final_mem:.1f} MiB")
            tracker.print(f"Shortfall: {min_required - final_mem:.1f} MiB")
            tracker.print(f"\nThis means the GPU placement in load() function failed.")
            tracker.print(f"Check the diagnostic output above for the root cause.")
            tracker.print("X"*70)
            raise RuntimeError(
                f"Models not properly loaded to GPU!\n"
                f"Expected: {expected_model_mem:.1f} MiB, Got: {final_mem:.1f} MiB\n"
                f"Check diagnostics above for root cause."
            )
        tracker.print(f"\n✓ Final GPU memory check passed ({final_mem:.1f} MiB allocated)")
        tracker.print(f"  Overhead: {final_mem - expected_model_mem:.1f} MiB ({(final_mem / expected_model_mem - 1) * 100:.1f}% over expected)")
        tracker.print("="*70)
    
    # DataLoader creation with diagnostics
    tracker.print("\n" + "="*70)
    tracker.print("=== Creating DataLoaders ===")
    tracker.print("="*70)
    tracker.print(f"Training DataLoader config:")
    tracker.print(f"  Batch size: {batch_size}")
    tracker.print(f"  Num workers: {num_workers}")
    tracker.print(f"  Start index: {state.tracker.step * batch_size}")
    tracker.print(f"  Persistent workers: {True if num_workers > 0 else False}")
    tracker.print(f"  Pin memory: True")
    
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
    
    tracker.print(f"\n✓ Training DataLoader created")
    tracker.print(f"  Steps per epoch: {steps_per_epoch}")
    tracker.print(f"  Total iterations: {num_iters} ({training_epochs} epochs)")
    tracker.print("="*70)
    
    # ── Test first batch load (fail fast if DataLoader hangs) ──────────────────
    tracker.print("\n" + "="*70)
    tracker.print("=== Testing First Batch Load ===")
    tracker.print("="*70)
    tracker.print("Timeout: 60 seconds")
    tracker.print("This will verify DataLoader can load data successfully...\n")
    
    import time
    import signal
    
    def timeout_handler(signum, frame):
        raise TimeoutError("DataLoader hang detected: first batch took >60s")
    
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(60)  # 60 second timeout
    
    try:
        train_dataloader_test = get_infinite_loader(train_dataloader)
        start_time = time.time()
        tracker.print("Calling next(dataloader)...")
        test_batch = next(train_dataloader_test)
        signal.alarm(0)  # Cancel alarm
        elapsed = time.time() - start_time
        
        tracker.print(f"\n✓ First batch loaded successfully in {elapsed:.1f}s")
        
        # Show batch details
        tracker.print(f"\nBatch details:")
        if isinstance(test_batch, dict):
            tracker.print(f"  Batch type: dict with keys: {list(test_batch.keys())}")
            if 'signal' in test_batch:
                signal_obj = test_batch['signal']
                if hasattr(signal_obj, 'audio_data'):
                    audio_data = signal_obj.audio_data
                    tracker.print(f"  Signal.audio_data shape: {audio_data.shape}")
                    tracker.print(f"  Signal.audio_data dtype: {audio_data.dtype}")
                    tracker.print(f"  Signal.audio_data device: {audio_data.device}")
                    tracker.print(f"  Signal.audio_data range: [{audio_data.min():.4f}, {audio_data.max():.4f}]")
        else:
            tracker.print(f"  Batch type: {type(test_batch)}")
            if hasattr(test_batch, 'shape'):
                tracker.print(f"  Batch shape: {test_batch.shape}")
        
        if torch.cuda.is_available():
            batch_mem = torch.cuda.memory_allocated() / 1024**2
            tracker.print(f"\nGPU memory after first batch: {batch_mem:.1f} MiB")
            if batch_mem <= final_mem:
                tracker.print(f"⚠ WARNING: GPU memory unchanged ({batch_mem:.1f} MiB), batch may not be on GPU")
            else:
                tracker.print(f"✓ Batch loaded to GPU successfully (delta: +{batch_mem-final_mem:.1f} MiB)")
        
        tracker.print("\nNote: Detailed dataset logging will only appear in first epoch")
        tracker.print("="*70)
    except TimeoutError as e:
        tracker.print(f"\n❌ FAILED: {e}")
        tracker.print("Possible causes: num_workers too high, slow file I/O, or dataset __getitem__ hang")
        tracker.print("="*70)
        raise
    except Exception as e:
        tracker.print(f"\n❌ FAILED to load first batch: {e}")
        tracker.print(f"Exception type: {type(e).__name__}")
        import traceback
        tracker.print(f"Traceback:\n{traceback.format_exc()}")
        tracker.print("="*70)
        raise

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
        # Define epoch as a custom x-axis so charts can be viewed per-epoch in W&B
        wandb.define_metric("epoch")
        wandb.define_metric("train/*", step_metric="epoch")
        wandb.define_metric("val/*", step_metric="epoch")
        wandb.define_metric("epoch_summary/*", step_metric="epoch")

    global _steps_per_epoch
    _steps_per_epoch = steps_per_epoch

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

            # ── Epoch-boundary summary log ─────────────────────────────────
            if _wandb_run is not None and steps_per_epoch > 0:
                if (tracker.step + 1) % steps_per_epoch == 0 or last_iter:
                    completed_epoch = (tracker.step + 1) // steps_per_epoch
                    train_hist = state.tracker.history.get("train", {})
                    summary: dict = {"epoch": completed_epoch}
                    for metric in ("loss", "mel/loss", "stft/loss", "adv/gen_loss",
                                   "adv/disc_loss", "other/grad_norm", "other/grad_norm_d"):
                        vals = train_hist.get(metric, [])
                        if vals:
                            summary[f"epoch_summary/{metric}"] = sum(vals) / len(vals)
                    wandb.log(summary)

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
