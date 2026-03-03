# 07_train_parakeet_aux.py
#!/usr/bin/env python3
import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import create_repo, upload_folder
from omegaconf import OmegaConf, open_dict
import lightning.pytorch as pl
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

import nemo.collections.asr as nemo_asr
from nemo.core import adapter_mixins
from omegaconf import OmegaConf, open_dict, DictConfig, ListConfig


def jsonable(x):
    try:
        json.dumps(x)
        return x
    except Exception:
        return str(x)


def read_manifest(path: Path) -> List[Dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def maybe_load_model(model_ref: str, enable_adapter_support: bool = False):
    model_path = Path(model_ref)

    if enable_adapter_support:
        if model_path.exists():
            cfg = nemo_asr.models.ASRModel.restore_from(
                str(model_path),
                return_config=True,
            )
        else:
            cfg = nemo_asr.models.ASRModel.from_pretrained(
                model_name=model_ref,
                return_config=True,
            )

        with open_dict(cfg):
            adapter_metadata = adapter_mixins.get_registered_adapter(cfg.encoder._target_)
            if adapter_metadata is None:
                raise RuntimeError(
                    f"No registered adapter-compatible encoder found for {cfg.encoder._target_}"
                )
            cfg.encoder._target_ = adapter_metadata.adapter_class_path

        print(f"Updated encoder target for adapter support: {cfg.encoder._target_}")

        if model_path.exists():
            print(f"Loading local NeMo checkpoint with adapter-compatible encoder: {model_path}")
            return nemo_asr.models.ASRModel.restore_from(
                str(model_path),
                override_config_path=cfg,
            )
        else:
            print(f"Loading pretrained NeMo model with adapter-compatible encoder: {model_ref}")
            return nemo_asr.models.ASRModel.from_pretrained(
                model_name=model_ref,
                override_config_path=cfg,
            )

    if model_path.exists():
        print(f"Loading local NeMo checkpoint: {model_path}")
        return nemo_asr.models.ASRModel.restore_from(str(model_path))

    print(f"Loading pretrained NeMo model: {model_ref}")
    return nemo_asr.models.ASRModel.from_pretrained(model_name=model_ref)


def build_lhotse_cfg(
    manifest_path: Path,
    batch_duration: float,
    num_workers: int,
    shuffle: bool,
    min_duration: float = 0.0,
    max_duration: float = 1000.0,
    num_buckets: int = 30,
    bucket_buffer_size: int = 10000,
    quadratic_duration: float | None = 30.0,
    use_bucketing: bool = True,
    return_sample_id: bool = True,
):
    cfg = {
        "manifest_filepath": str(manifest_path),
        "sample_rate": 16000,
        "use_lhotse": True,
        "batch_duration": float(batch_duration),
        "shuffle": shuffle,
        "num_workers": int(num_workers),
        "pin_memory": True,
        "use_start_end_token": False,
        "trim_silence": False,
        "min_duration": float(min_duration),
        "max_duration": float(max_duration),
        "return_sample_id": return_sample_id,
    }

    if use_bucketing:
        cfg["num_buckets"] = int(num_buckets)
        cfg["bucket_buffer_size"] = int(bucket_buffer_size)
    if quadratic_duration is not None:
        cfg["quadratic_duration"] = float(quadratic_duration)

    return OmegaConf.create(cfg)


def enable_adapter_peft(model, adapter_name: str, adapter_dim: int):
    if not hasattr(model, "add_adapter"):
        raise RuntimeError(
            "This NeMo ASR model does not expose add_adapter() even after adapter-compatible reload."
        )

    module_names = []
    try:
        module_names = list(model.adapter_module_names)
        print("Adapter-capable module names:", module_names)
    except Exception:
        pass

    full_name = f"encoder:{adapter_name}" if "encoder" in module_names else adapter_name

    adapter_cfg = OmegaConf.create(
        {
            "_target_": "nemo.collections.common.parts.adapter_modules.LinearAdapter",
            "in_features": None,
            "dim": int(adapter_dim),
            "dropout": 0.0,
            "norm_position": "pre",
        }
    )

    print(f"Adding adapter: {full_name}")
    model.add_adapter(full_name, cfg=adapter_cfg)

    model.freeze()
    model.unfreeze_enabled_adapters()

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(
        f"Trainable params after PEFT enable: {trainable:,} / {total:,} "
        f"({100.0 * trainable / total:.4f}%)"
    )

    return full_name


def maybe_load_existing_adapter(model, adapter_path: str | None):
    if not adapter_path:
        return
    if not hasattr(model, "load_adapters"):
        raise RuntimeError("This NeMo model does not expose load_adapters().")

    print(f"Loading existing adapters from: {adapter_path}")

    # PyTorch 2.6+ safe-unpickling compatibility for NeMo adapter bundles
    try:
        torch.serialization.add_safe_globals([DictConfig, ListConfig])
    except Exception as e:
        print(f"Warning: could not add safe globals: {e}")

    model.load_adapters(adapter_path)


def save_adapter_bundle(model, output_dir: Path, stem: str):
    output_dir.mkdir(parents=True, exist_ok=True)
    adapter_path = output_dir / f"{stem}.pt"
    if not hasattr(model, "save_adapters"):
        raise RuntimeError("This NeMo model does not expose save_adapters().")
    model.save_adapters(str(adapter_path))
    print(f"Saved adapters to: {adapter_path}")
    return adapter_path

def disable_rnnt_wer(model):
    class _NoOpWER(nn.Module):
        def __init__(self, device):
            super().__init__()
            self._device = device

        def update(self, *args, **kwargs):
            return None

        def compute(self):
            z = torch.tensor(0.0, device=self._device)
            return z, z, z

        def reset(self):
            return None

        def forward(self, *args, **kwargs):
            return None

    if not hasattr(model, "joint"):
        print("Model has no joint module; skipping RNNT WER disable.")
        return

    joint = model.joint
    device = next(model.parameters()).device
    noop = _NoOpWER(device)

    # Best path: replace registered child module with another nn.Module
    if hasattr(joint, "_wer"):
        try:
            joint._wer = noop
            print("Disabled model.joint._wer for aux training.")
            return
        except Exception as e:
            print(f"Could not replace joint._wer directly: {e}")

    # Fallback: monkey-patch existing metric module in place
    try:
        existing_wer = getattr(joint, "wer", None)
        if existing_wer is not None:
            existing_wer.update = noop.update
            existing_wer.compute = noop.compute
            existing_wer.reset = noop.reset
            if hasattr(existing_wer, "forward"):
                existing_wer.forward = noop.forward
            print("Monkey-patched existing model.joint.wer for aux training.")
            return
    except Exception as e:
        print(f"Could not patch joint.wer in place: {e}")

    print("Warning: could not disable RNNTJoint WER.")

def patch_rnnt_decoding(model):
    try:
        dec_cfg = model.cfg.get("rnnt_decoding", None)
        if dec_cfg is None:
            print("No rnnt_decoding config found on model.")
            return

        with open_dict(dec_cfg):
            if "strategy" in dec_cfg:
                dec_cfg.strategy = "greedy_batch"

            if hasattr(dec_cfg, "greedy") and dec_cfg.greedy is not None:
                if "use_cuda_graph_decoder" in dec_cfg.greedy:
                    dec_cfg.greedy.use_cuda_graph_decoder = False
                if "allow_cuda_graphs" in dec_cfg.greedy:
                    dec_cfg.greedy.allow_cuda_graphs = False
                if "cuda_graphs" in dec_cfg.greedy:
                    dec_cfg.greedy.cuda_graphs = False

            if hasattr(dec_cfg, "greedy_batch") and dec_cfg.greedy_batch is not None:
                if "use_cuda_graph_decoder" in dec_cfg.greedy_batch:
                    dec_cfg.greedy_batch.use_cuda_graph_decoder = False
                if "allow_cuda_graphs" in dec_cfg.greedy_batch:
                    dec_cfg.greedy_batch.allow_cuda_graphs = False
                if "cuda_graphs" in dec_cfg.greedy_batch:
                    dec_cfg.greedy_batch.cuda_graphs = False

        model.change_decoding_strategy(dec_cfg)
        print("Updated rnnt_decoding:", model.cfg.rnnt_decoding)

        try:
            if hasattr(model, "decoding") and hasattr(model.decoding, "decoding"):
                inner = model.decoding.decoding
                if hasattr(inner, "disable_cuda_graphs"):
                    inner.disable_cuda_graphs()
                    print("Disabled CUDA graphs on runtime decoder object.")
        except Exception as e:
            print(f"Runtime decoder disable_cuda_graphs() skipped: {e}")

    except Exception as e:
        print(f"Failed to patch RNNT decoding config: {e}")


def no_val_dataloader():
    return []


def no_validation_step(*args, **kwargs):
    return None


def no_validation_epoch_hook(*args, **kwargs):
    return None


def hard_disable_validation(obj):
    obj._validation_dl = []
    obj._validation_names = []
    if hasattr(obj, "_val_dl_idx"):
        obj._val_dl_idx = 0

    obj.val_dataloader = no_val_dataloader
    obj.validation_step = no_validation_step
    obj.on_validation_epoch_start = no_validation_epoch_hook
    obj.on_validation_epoch_end = no_validation_epoch_hook


class LabelEncoder:
    def __init__(self, labels: List[str]):
        uniq = sorted(set(labels))
        self.label_to_id = {x: i for i, x in enumerate(uniq)}
        self.id_to_label = {i: x for x, i in self.label_to_id.items()}

    def encode(self, x: str) -> int:
        return self.label_to_id[x]

    def num_classes(self) -> int:
        return len(self.label_to_id)


class ParakeetAuxModule(pl.LightningModule):
    def __init__(
        self,
        asr_model,
        train_manifest: Path,
        aux_task: str,
        learning_rate: float,
        weight_decay: float,
        train_batch_duration: float,
        num_workers: int,
        lambda_aux: float,
        min_duration: float,
        max_duration: float,
        num_buckets: int,
        bucket_buffer_size: int,
        quadratic_duration: float | None,
    ):
        super().__init__()
        self.asr_model = asr_model
        self.aux_task = aux_task
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.train_batch_duration = train_batch_duration
        self.num_workers = num_workers
        self.lambda_aux = lambda_aux
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.num_buckets = num_buckets
        self.bucket_buffer_size = bucket_buffer_size
        self.quadratic_duration = quadratic_duration

        self.train_rows = read_manifest(train_manifest)
        self._row_cursor = 0

        label_key = f"{aux_task}_label"
        all_labels = [str(r[label_key]) for r in self.train_rows]
        self.label_encoder = LabelEncoder(all_labels)

        hidden_size = self._infer_encoder_dim()
        self.aux_head = nn.Linear(hidden_size, self.label_encoder.num_classes())

        self.train_cfg = build_lhotse_cfg(
            manifest_path=train_manifest,
            batch_duration=train_batch_duration,
            num_workers=num_workers,
            shuffle=False,              # emergency deterministic mode
            min_duration=min_duration,
            max_duration=max_duration,
            num_buckets=num_buckets,
            bucket_buffer_size=bucket_buffer_size,
            quadratic_duration=quadratic_duration,
            use_bucketing=False,        # emergency deterministic mode
            return_sample_id=False,     # stop relying on unsupported sample ids
        )

        self.asr_model.setup_training_data(train_data_config=self.train_cfg)

        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        print(
            f"Total trainable params in aux module: {trainable:,} / {total:,} "
            f"({100.0 * trainable / total:.4f}%)"
        )

    def on_train_epoch_start(self):
        self._row_cursor = 0

    def _infer_encoder_dim(self) -> int:
        if hasattr(self.asr_model, "encoder") and hasattr(self.asr_model.encoder, "_feat_out"):
            return int(self.asr_model.encoder._feat_out)
        if hasattr(self.asr_model, "encoder") and hasattr(self.asr_model.encoder, "feat_out"):
            return int(self.asr_model.encoder.feat_out)
        return 1024

    def _normalize_sample_ids(self, sample_ids):
        if sample_ids is None:
            return None
        if torch.is_tensor(sample_ids):
            return [int(x) for x in sample_ids.detach().cpu().tolist()]
        if isinstance(sample_ids, (list, tuple)):
            out = []
            for x in sample_ids:
                if torch.is_tensor(x):
                    out.append(int(x.item()))
                else:
                    out.append(int(x))
            return out
        return None

    def _parse_batch(self, batch):
        if isinstance(batch, dict):
            signal = None
            signal_len = None
            sample_ids = None

            for k in ["audio_signal", "input_signal", "signal"]:
                if k in batch:
                    signal = batch[k]
                    break
            for k in ["audio_signal_length", "input_signal_length", "signal_length", "length"]:
                if k in batch:
                    signal_len = batch[k]
                    break
            for k in ["sample_id", "sample_ids"]:
                if k in batch:
                    sample_ids = batch[k]
                    break

            return signal, signal_len, self._normalize_sample_ids(sample_ids)

        if isinstance(batch, (tuple, list)):
            if len(batch) == 5:
                signal, signal_len, _, _, sample_ids = batch
                return signal, signal_len, self._normalize_sample_ids(sample_ids)
            if len(batch) == 4:
                signal, signal_len, _, _ = batch
                return signal, signal_len, None

        raise ValueError(f"Unexpected batch format: {type(batch)}")

    def _extract_asr_loss(self, batch, batch_idx):
        # NeMo's training_step tries to self.log(), which fails because asr_model
        # is nested inside this LightningModule and is not the trainer-managed module.
        orig_log = self.asr_model.log
        orig_log_dict = self.asr_model.log_dict

        def _no_log(*args, **kwargs):
            return None

        try:
            self.asr_model.log = _no_log
            self.asr_model.log_dict = _no_log

            out = self.asr_model.training_step(batch, batch_idx)

            if isinstance(out, dict):
                if "loss" in out:
                    return out["loss"]
                if "train_loss" in out:
                    return out["train_loss"]
            if torch.is_tensor(out):
                return out

            raise ValueError("Could not extract ASR loss from NeMo training_step output.")
        finally:
            self.asr_model.log = orig_log
            self.asr_model.log_dict = orig_log_dict


    def _get_encoder_repr(self, signal, signal_len):
        processed_signal, processed_signal_len = self.asr_model.preprocessor(
            input_signal=signal, length=signal_len
        )
        encoded, encoded_len = self.asr_model.encoder(
            audio_signal=processed_signal, length=processed_signal_len
        )

        if encoded.dim() != 3:
            raise ValueError(f"Unexpected encoder output shape: {tuple(encoded.shape)}")

        bsz, dim, time = encoded.shape
        mask = (
            torch.arange(time, device=encoded.device).unsqueeze(0).expand(bsz, time)
            < encoded_len.unsqueeze(1)
        )
        mask = mask.unsqueeze(1).float()
        pooled = (encoded * mask).sum(dim=2) / mask.sum(dim=2).clamp_min(1.0)
        return pooled

    def _get_aux_targets(self, sample_ids, batch_size: int):
        label_key = f"{self.aux_task}_label"

        # Preferred path if sample_ids ever show up
        if sample_ids is not None:
            labels = []
            for sid in sample_ids:
                label_str = str(self.train_rows[int(sid)][label_key])
                labels.append(self.label_encoder.encode(label_str))
            return torch.tensor(labels, dtype=torch.long, device=self.device)

        # Emergency fallback: deterministic sequential consumption
        start = self._row_cursor
        end = start + batch_size

        if end > len(self.train_rows):
            raise ValueError(
                f"Ran out of manifest rows while assigning aux labels: "
                f"start={start}, end={end}, total={len(self.train_rows)}"
            )

        rows = self.train_rows[start:end]
        self._row_cursor = end

        labels = [self.label_encoder.encode(str(r[label_key])) for r in rows]
        return torch.tensor(labels, dtype=torch.long, device=self.device)

    def training_step(self, batch, batch_idx):
        asr_loss = self._extract_asr_loss(batch, batch_idx)
        signal, signal_len, sample_ids = self._parse_batch(batch)

        pooled = self._get_encoder_repr(signal, signal_len)
        aux_logits = self.aux_head(pooled)
        batch_size = int(signal.shape[0])
        aux_targets = self._get_aux_targets(sample_ids, batch_size=batch_size)
        aux_loss = F.cross_entropy(aux_logits, aux_targets)

        loss = asr_loss + self.lambda_aux * aux_loss
        self.log("train_asr_loss", asr_loss, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log("train_aux_loss", aux_loss, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log("train_total_loss", loss, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
        return loss

    def train_dataloader(self):
        return self.asr_model._train_dl

    def configure_optimizers(self):
        trainable_params = [p for p in self.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.trainer.max_epochs,
            eta_min=1e-6,
        )

        # NeMo RNNT training_step expects these internal attributes
        self.asr_model._optimizer = optimizer
        self.asr_model._scheduler = scheduler

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", type=str, default="nvidia/parakeet-tdt-0.6b-v2")
    ap.add_argument(
        "--base_adapter_path",
        type=str,
        default=None,
        help="Optional previously saved adapter bundle to load before training.",
    )
    ap.add_argument("--manifests_dir", type=Path, default=Path("pipeline/manifests/parakeet"))
    ap.add_argument("--output_root", type=Path, default=Path("pipeline/checkpoints"))
    ap.add_argument("--aux_task", type=str, required=True, choices=["prosody", "phonation", "articulation"])

    ap.add_argument("--devices", type=int, default=1)
    ap.add_argument("--max_epochs", type=int, default=10)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--gradient_accumulation_steps", type=int, default=2)

    ap.add_argument("--train_batch_duration", type=float, default=100.0)
    ap.add_argument("--min_duration", type=float, default=0.0)
    ap.add_argument("--max_duration", type=float, default=1000.0)
    ap.add_argument("--num_buckets", type=int, default=30)
    ap.add_argument("--bucket_buffer_size", type=int, default=10000)
    ap.add_argument("--quadratic_duration", type=float, default=30.0)

    ap.add_argument("--adapter_name", type=str, default=None)
    ap.add_argument("--adapter_dim", type=int, default=64)

    ap.add_argument("--learning_rate", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-3)
    ap.add_argument("--lambda_aux", type=float, default=0.1)
    ap.add_argument("--precision", type=str, default="16-mixed")
    ap.add_argument("--seed", type=int, default=13)

    ap.add_argument("--enable_subsampling_chunking", action="store_true")

    ap.add_argument("--wandb_project", type=str, default="CanaryParakeet")
    ap.add_argument("--wandb_entity", type=str, default=None)
    ap.add_argument("--wandb_name", type=str, default=None)
    ap.add_argument("--wandb_offline", action="store_true")

    ap.add_argument("--push_to_hub", action="store_true")
    ap.add_argument("--hf_repo_id", type=str, default=None)
    ap.add_argument("--hf_private", action="store_true")

    args = ap.parse_args()
    pl.seed_everything(args.seed, workers=True)

    if args.adapter_name is None or str(args.adapter_name).strip() == "":
        args.adapter_name = f"hd_{args.aux_task}"

    if args.hf_repo_id is None or str(args.hf_repo_id).strip() == "":
        args.hf_repo_id = f"charleslwang/parakeet-tdt-0.6b-HD-{args.aux_task}"

    out_dir = args.output_root / f"parakeet_hd_{args.aux_task}"
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg_path = out_dir / "run_config.json"
    cfg_path.write_text(json.dumps({k: jsonable(v) for k, v in vars(args).items()}, indent=2))

    train_manifest = args.manifests_dir / "train.jsonl"
    if not train_manifest.exists():
        raise FileNotFoundError(train_manifest)

    asr_model = maybe_load_model(args.base_model, enable_adapter_support=True)

    if args.enable_subsampling_chunking:
        try:
            asr_model.change_subsampling_conv_chunking_factor(1)
            print("Enabled Parakeet subsampling conv auto-chunking.")
        except Exception as e:
            print(f"Could not enable subsampling conv chunking: {e}")

    patch_rnnt_decoding(asr_model)
    disable_rnnt_wer(asr_model)

    if args.base_adapter_path:
        maybe_load_existing_adapter(asr_model, args.base_adapter_path)
        asr_model.freeze()
        asr_model.unfreeze_enabled_adapters()
        adapter_full_name = f"encoder:{args.adapter_name}"
    else:
        adapter_full_name = enable_adapter_peft(
            model=asr_model,
            adapter_name=args.adapter_name,
            adapter_dim=args.adapter_dim,
        )

    module = ParakeetAuxModule(
        asr_model=asr_model,
        train_manifest=train_manifest,
        aux_task=args.aux_task,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        train_batch_duration=args.train_batch_duration,
        num_workers=args.num_workers,
        lambda_aux=args.lambda_aux,
        min_duration=args.min_duration,
        max_duration=args.max_duration,
        num_buckets=args.num_buckets,
        bucket_buffer_size=args.bucket_buffer_size,
        quadratic_duration=args.quadratic_duration,
    )

    label_map_path = out_dir / "label_map.json"
    label_map_path.write_text(json.dumps(module.label_encoder.label_to_id, indent=2))

    ckpt_dir = out_dir / "lightning_ckpts"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_cb = ModelCheckpoint(
        dirpath=str(ckpt_dir),
        filename=f"parakeet-hd-{args.aux_task}-peft" + "-{epoch:02d}",
        save_top_k=-1,
        save_last=True,
        every_n_epochs=1,
    )
    lr_cb = LearningRateMonitor(logging_interval="step")

    run_name = f"Parakeet-HD-{args.aux_task.capitalize()}"
    wandb_logger = WandbLogger(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_name or run_name,
        save_dir=str(out_dir),
        offline=args.wandb_offline,
        config={k: jsonable(v) for k, v in vars(args).items()},
    )

    hard_disable_validation(module)
    hard_disable_validation(module.asr_model)

    strategy = "ddp" if (torch.cuda.is_available() and args.devices > 1) else "auto"

    trainer = pl.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=args.devices,
        strategy=strategy,
        max_epochs=args.max_epochs,
        precision=args.precision if torch.cuda.is_available() else 32,
        accumulate_grad_batches=args.gradient_accumulation_steps,
        callbacks=[checkpoint_cb, lr_cb],
        default_root_dir=str(out_dir),
        log_every_n_steps=10,
        enable_progress_bar=True,
        use_distributed_sampler=False,
        logger=wandb_logger,
        num_sanity_val_steps=0,
        limit_val_batches=0,
        gradient_clip_val=1.0,
    )

    trainer.fit(module)

    final_ckpt = out_dir / "final_aux.ckpt"
    trainer.save_checkpoint(str(final_ckpt))
    print(f"Saved multitask Lightning checkpoint to: {final_ckpt}")

    final_adapter_path = save_adapter_bundle(module.asr_model, out_dir, "final_adapter")
    final_aux_head_path = out_dir / "final_aux_head.pt"
    torch.save(module.aux_head.state_dict(), final_aux_head_path)
    print(f"Saved final aux head to: {final_aux_head_path}")

    best_adapter_path = out_dir / "best_adapter.pt"
    best_aux_head_path = out_dir / "best_aux_head.pt"
    shutil.copyfile(final_adapter_path, best_adapter_path)
    shutil.copyfile(final_aux_head_path, best_aux_head_path)
    print(f"Copied final adapter to: {best_adapter_path}")
    print(f"Copied final aux head to: {best_aux_head_path}")

    meta = {
        "base_model": args.base_model,
        "adapter_name": adapter_full_name,
        "adapter_dim": args.adapter_dim,
        "peft_type": "nemo_linear_adapter",
        "aux_task": args.aux_task,
        "validation_disabled": True,
    }
    (out_dir / "peft_metadata.json").write_text(json.dumps(meta, indent=2))

    if args.push_to_hub:
        create_repo(args.hf_repo_id, private=args.hf_private, exist_ok=True)
        upload_folder(
            repo_id=args.hf_repo_id,
            folder_path=str(out_dir),
            path_in_repo=".",
            allow_patterns=[
                "best_adapter.pt",
                "final_adapter.pt",
                "best_aux_head.pt",
                "final_aux_head.pt",
                "label_map.json",
                "run_config.json",
                "peft_metadata.json",
            ],
            commit_message=f"Upload Parakeet-HD-{args.aux_task} PEFT adapters",
        )
        print(f"Uploaded PEFT artifacts to Hugging Face: {args.hf_repo_id}")


if __name__ == "__main__":
    main()
