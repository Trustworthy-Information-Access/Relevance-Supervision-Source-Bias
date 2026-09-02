"""Model construction and execution for controlled retriever fine-tuning."""

from __future__ import annotations

import logging
import random
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from ..core.data import load_corpus, load_queries, write_json
from .losses import build_loss
from .plans import ControlledPlanDataset

LOGGER = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def build_model(config: Mapping[str, Any]):
    """Build a SentenceTransformer from a checkpoint or transformer plus pooler."""
    try:
        from sentence_transformers import SentenceTransformer, models
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("Install model dependencies with: pip install -e '.[models]'") from exc
    model_name = str(config["model_name"])
    if config.get("load_as_sentence_transformer", False):
        model = SentenceTransformer(model_name)
        model.max_seq_length = int(config.get("max_sequence_length", 512))
        return model
    transformer = models.Transformer(
        model_name, max_seq_length=int(config.get("max_sequence_length", 512))
    )
    pooling = str(config.get("pooling", "mean"))
    if pooling not in {"mean", "cls", "max"}:
        raise ValueError("pooling must be mean, cls, or max")
    pooler = models.Pooling(
        transformer.get_word_embedding_dimension(),
        pooling_mode_cls_token=pooling == "cls",
        pooling_mode_mean_tokens=pooling == "mean",
        pooling_mode_max_tokens=pooling == "max",
        pooling_mode_mean_sqrt_len_tokens=False,
    )
    return SentenceTransformer(modules=[transformer, pooler])


def train_from_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Run one controlled SentenceTransformers training condition."""
    try:
        import torch
        from torch.utils.data import DataLoader
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("Install model dependencies with: pip install -e '.[models]'") from exc
    seed = int(config.get("seed", 42))
    set_seed(seed)
    data = config["data"]
    dataset = ControlledPlanDataset(
        data["plan"],
        queries=load_queries(data["queries"]),
        corpus=load_corpus(data["corpus"]),
        query_prefix=str(config.get("query_prefix", "")),
        passage_prefix=str(config.get("passage_prefix", "")),
        include_title=bool(config.get("include_title", False)),
    )
    model = build_model(config)
    dataloader = DataLoader(
        dataset,
        batch_sampler=dataset.batches,
        collate_fn=model.smart_batching_collate,
        num_workers=int(config.get("num_workers", 0)),
    )
    similarity = str(config.get("similarity", "dot"))
    if similarity not in {"dot", "cosine"}:
        raise ValueError("similarity must be dot or cosine")
    default_scale = 1.0 if similarity == "dot" else 20.0
    loss = build_loss(
        model,
        strategy=str(config.get("negative_strategy", "standard")),
        similarity=similarity,
        scale=float(config.get("scale", default_scale)),
    )
    output = Path(config["output"])
    output.mkdir(parents=True, exist_ok=True)
    warmup_steps = int(config.get("warmup_steps", 1000))
    checkpoint_steps = int(config.get("checkpoint_steps", len(dataloader)))
    LOGGER.info(
        "Training %s batches with strategy=%s",
        len(dataloader),
        config.get("negative_strategy"),
    )
    # SentenceTransformers 3.x+ routes ``fit`` through its Trainer API, which
    # rebuilds the DataLoader and cannot preserve our explicit plan batches.
    # Its ``old_fit`` compatibility entry point retains the exact batch sampler
    # used by the paper; SentenceTransformers 2.x exposes the same behavior as
    # ``fit`` directly.
    fit = getattr(model, "old_fit", model.fit)
    fit(
        train_objectives=[(dataloader, loss)],
        epochs=1,
        warmup_steps=warmup_steps,
        optimizer_params={"lr": float(config.get("learning_rate", 2e-5))},
        use_amp=bool(config.get("use_amp", torch.cuda.is_available())),
        output_path=str(output),
        checkpoint_path=str(output / "checkpoints"),
        checkpoint_save_steps=checkpoint_steps,
        checkpoint_save_total_limit=int(config.get("checkpoint_limit", 2)),
        show_progress_bar=True,
    )
    metadata = {
        "model_name": str(config["model_name"]),
        "negative_strategy": str(config.get("negative_strategy", "standard")),
        "similarity": similarity,
        "seed": seed,
        "learning_rate": float(config.get("learning_rate", 2e-5)),
        "warmup_steps": warmup_steps,
        "num_batches": len(dataloader),
        "num_samples": len(dataset),
        "output": str(output),
    }
    write_json(metadata, output / "training_metadata.json")
    return metadata
