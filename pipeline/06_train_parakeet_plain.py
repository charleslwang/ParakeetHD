# 06_train_parakeet_plain.py
#!/usr/bin/env python3
import argparse
import json
import shutil
from pathlib import Path

import torch
from huggingface_hub import create_repo, upload_folder
from omegaconf import OmegaConf, open_dict
import lightning.pytorch as pl
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

import nemo.collections.asr as nemo_asr
from nemo.core import adapter_mixins


def jsonable(x):
    try:
        json.dumps(x)
        return x
    except Exception:
        return str(x)


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
    return_sample_id: bool = False,
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
    model.load_adapters(adapter_path)


def save_adapter_bundle(model, output_dir: Path, stem: str):
    output_dir.mkdir(parents=True, exist_ok=True)
    adapter_path = output_dir / f"{stem}.pt"
    if not hasattr(model, "save_adapters"):
        raise RuntimeError("This NeMo model does not expose save_adapters().")
    model.save_adapters(str(adapter_path))
    print(f"Saved adapters to: {adapter_path}")
    return adapter_path


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


def hard_disable_validation(model):
    model._validation_dl = []
    model._validation_names = []
    if hasattr(model, "_val_dl_idx"):
        model._val_dl_idx = 0

    model.val_dataloader = no_val_dataloader
    model.validation_step = no_validation_step
    model.on_validation_epoch_start = no_validation_epoch_hook
    model.on_validation_epoch_end = no_validation_epoch_hook


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
    ap.add_argument("--output_dir", type=Path, default=Path("pipeline/checkpoints/parakeet_hd_plain"))
    ap.add_argument("--run_name", type=str, default="Parakeet-HD")

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

    ap.add_argument("--adapter_name", type=str, default="hd")
    ap.add_argument("--adapter_dim", type=int, default=64)

    ap.add_argument("--learning_rate", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-3)
    ap.add_argument("--precision", type=str, default="16-mixed")
    ap.add_argument("--seed", type=int, default=13)

    ap.add_argument("--enable_subsampling_chunking", action="store_true")

    ap.add_argument("--wandb_project", type=str, default="CanaryParakeet")
    ap.add_argument("--wandb_entity", type=str, default=None)
    ap.add_argument("--wandb_name", type=str, default=None)
    ap.add_argument("--wandb_offline", action="store_true")

    ap.add_argument("--push_to_hub", action="store_true")
    ap.add_argument("--hf_repo_id", type=str, default="charleslwang/parakeet-tdt-0.6b-HD")
    ap.add_argument("--hf_private", action="store_true")

    args = ap.parse_args()
    pl.seed_everything(args.seed, workers=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_manifest = args.manifests_dir / "train.jsonl"
    if not train_manifest.exists():
        raise FileNotFoundError(train_manifest)

    cfg_path = args.output_dir / "run_config.json"
    cfg_path.write_text(json.dumps({k: jsonable(v) for k, v in vars(args).items()}, indent=2))

    model = maybe_load_model(args.base_model, enable_adapter_support=True)

    if args.enable_subsampling_chunking:
        try:
            model.change_subsampling_conv_chunking_factor(1)
            print("Enabled Parakeet subsampling conv auto-chunking.")
        except Exception as e:
            print(f"Could not enable subsampling conv chunking: {e}")

    patch_rnnt_decoding(model)

    adapter_full_name = enable_adapter_peft(
        model=model,
        adapter_name=args.adapter_name,
        adapter_dim=args.adapter_dim,
    )
    maybe_load_existing_adapter(model, args.base_adapter_path)

    train_cfg = build_lhotse_cfg(
        manifest_path=train_manifest,
        batch_duration=args.train_batch_duration,
        num_workers=args.num_workers,
        shuffle=True,
        min_duration=args.min_duration,
        max_duration=args.max_duration,
        num_buckets=args.num_buckets,
        bucket_buffer_size=args.bucket_buffer_size,
        quadratic_duration=args.quadratic_duration,
        use_bucketing=True,
        return_sample_id=False,
    )
    model.setup_training_data(train_data_config=train_cfg)

    def custom_configure_optimizers():
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args.max_epochs,
            eta_min=1e-6,
        )

        # NeMo RNNT training_step expects these internal attributes
        model._optimizer = optimizer
        model._scheduler = scheduler

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }

    model.configure_optimizers = custom_configure_optimizers

    ckpt_dir = args.output_dir / "lightning_ckpts"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_cb = ModelCheckpoint(
        dirpath=str(ckpt_dir),
        filename="parakeet-hd-peft-{epoch:02d}",
        save_top_k=-1,
        save_last=True,
        every_n_epochs=1,
    )
    lr_cb = LearningRateMonitor(logging_interval="step")

    wandb_logger = WandbLogger(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_name or args.run_name,
        save_dir=str(args.output_dir),
        offline=args.wandb_offline,
        config={k: jsonable(v) for k, v in vars(args).items()},
    )

    hard_disable_validation(model)

    strategy = "ddp" if (torch.cuda.is_available() and args.devices > 1) else "auto"

    trainer = pl.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=args.devices,
        strategy=strategy,
        max_epochs=args.max_epochs,
        precision=args.precision if torch.cuda.is_available() else 32,
        accumulate_grad_batches=args.gradient_accumulation_steps,
        callbacks=[checkpoint_cb, lr_cb],
        default_root_dir=str(args.output_dir),
        log_every_n_steps=10,
        enable_progress_bar=True,
        use_distributed_sampler=False,
        logger=wandb_logger,
        num_sanity_val_steps=0,
        limit_val_batches=0,
        gradient_clip_val=1.0,
    )

    trainer.fit(model)

    final_adapter_path = save_adapter_bundle(model, args.output_dir, "final_adapter")

    best_adapter_path = args.output_dir / "best_adapter.pt"
    shutil.copyfile(final_adapter_path, best_adapter_path)
    print(f"Copied final adapter to: {best_adapter_path}")

    meta = {
        "base_model": args.base_model,
        "adapter_name": adapter_full_name,
        "adapter_dim": args.adapter_dim,
        "peft_type": "nemo_linear_adapter",
        "validation_disabled": True,
    }
    (args.output_dir / "peft_metadata.json").write_text(json.dumps(meta, indent=2))

    if args.push_to_hub:
        create_repo(args.hf_repo_id, private=args.hf_private, exist_ok=True)
        upload_folder(
            repo_id=args.hf_repo_id,
            folder_path=str(args.output_dir),
            path_in_repo=".",
            allow_patterns=[
                "best_adapter.pt",
                "final_adapter.pt",
                "run_config.json",
                "peft_metadata.json",
            ],
            commit_message="Upload Parakeet-HD PEFT adapters",
        )
        print(f"Uploaded PEFT artifacts to Hugging Face: {args.hf_repo_id}")


if __name__ == "__main__":
    main()
    