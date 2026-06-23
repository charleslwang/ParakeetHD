"""Load the Parakeet base model with a NeMo encoder adapter enabled."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import patch

import torch
from huggingface_hub import hf_hub_download
from omegaconf import DictConfig, ListConfig, open_dict
from omegaconf.base import ContainerMetadata
from typing import Any as TypingAny


def enable_nemo_safe_unpickling() -> None:
    """Allow NeMo's OmegaConf-based adapter bundles on PyTorch 2.6+."""

    safe_globals: list[Any] = [DictConfig, ListConfig, ContainerMetadata, TypingAny]
    try:
        from omegaconf.nodes import (
            AnyNode,
            BooleanNode,
            BytesNode,
            EnumNode,
            FloatNode,
            IntegerNode,
            PathNode,
            StringNode,
        )

        safe_globals.extend(
            [AnyNode, BooleanNode, BytesNode, EnumNode, FloatNode, IntegerNode, PathNode, StringNode]
        )
    except ImportError:
        pass

    torch.serialization.add_safe_globals(safe_globals)


def resolve_adapter(adapter_source: str, adapter_filename: str) -> Path:
    """Resolve either a local adapter bundle or a Hugging Face repository."""

    local_path = Path(adapter_source).expanduser()
    if local_path.is_file():
        return local_path.resolve()
    if local_path.is_dir():
        candidate = local_path / adapter_filename
        if not candidate.is_file():
            raise FileNotFoundError(f"Adapter directory does not contain {adapter_filename}: {local_path}")
        return candidate.resolve()

    return Path(
        hf_hub_download(
            repo_id=adapter_source,
            filename=adapter_filename,
            repo_type="model",
        )
    )


def load_parakeet_hd(
    base_model: str,
    adapter_source: str,
    adapter_filename: str = "best_adapter.pt",
    device: str = "cpu",
):
    """Reconstruct ParakeetHD from the NVIDIA base model and NeMo adapter.

    The Hugging Face ParakeetHD repository contains an encoder adapter, not a
    complete NeMo checkpoint. The base model's config must therefore be loaded
    first and changed to use NeMo's adapter-compatible encoder class.
    """

    import nemo.collections.asr as nemo_asr
    from nemo.core import adapter_mixins

    base_path = Path(base_model).expanduser()
    if base_path.exists():
        cfg = nemo_asr.models.ASRModel.restore_from(str(base_path), return_config=True)
    else:
        cfg = nemo_asr.models.ASRModel.from_pretrained(model_name=base_model, return_config=True)

    with open_dict(cfg):
        adapter_metadata = adapter_mixins.get_registered_adapter(cfg.encoder._target_)
        if adapter_metadata is None:
            raise RuntimeError(f"No adapter-compatible encoder is registered for {cfg.encoder._target_}")
        cfg.encoder._target_ = adapter_metadata.adapter_class_path

    if base_path.exists():
        model = nemo_asr.models.ASRModel.restore_from(str(base_path), override_config_path=cfg)
    else:
        model = nemo_asr.models.ASRModel.from_pretrained(
            model_name=base_model,
            override_config_path=cfg,
        )

    adapter_path = resolve_adapter(adapter_source, adapter_filename)
    if not hasattr(model, "load_adapters"):
        raise RuntimeError("The reconstructed NeMo model does not expose load_adapters().")

    enable_nemo_safe_unpickling()
    original_torch_load = torch.load

    def trusted_torch_load(*args, **kwargs):
        # This path is only appropriate for a trusted adapter checkpoint.
        kwargs.setdefault("weights_only", False)
        return original_torch_load(*args, **kwargs)

    with patch("torch.load", trusted_torch_load):
        model.load_adapters(str(adapter_path))

    model.eval()
    model.to(device)
    return model, adapter_path

