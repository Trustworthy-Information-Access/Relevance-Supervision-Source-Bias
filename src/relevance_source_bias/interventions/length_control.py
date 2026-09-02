"""Length-controlled passage rewriting used by the RQ1 robustness analysis."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from ..analysis.artifacts import get_tokenizer
from ..core.data import format_document, iter_jsonl, load_corpus

LENGTH_CONTROL_SYSTEM_PROMPT = "You are a helpful assistant."
LENGTH_CONTROL_USER_PROMPT = """Please follow the instructions below:
1. Maintain the original meaning of the input passage.
2. Make the paraphrased passage slightly longer than the original (e.g., 10% longer) \
while preserving the same information.
3. Output the paraphrased passage directly.

Following is the passage you need to paraphrase:
{text}
Your answer must be formatted as:
Rewritten Text:
<your rewritten text>"""


def clean_rewrite_text(text: str) -> str:
    return re.sub(r"^\s*Rewritten Text:\s*", "", text, count=1, flags=re.IGNORECASE).strip()


def length_controlled_rewrite_rows(
    corpus_path: str | Path,
    *,
    model_name: str = "Qwen/Qwen2.5-7B-Instruct",
    batch_size: int = 8,
    max_input_length: int = 4096,
    max_new_tokens: int = 2048,
    temperature: float = 0.2,
    top_p: float = 1.0,
    seed: int = 42,
    device: str | None = None,
) -> Iterable[dict[str, Any]]:
    """Generate meaning-preserving, length-extended corpus rows with the paper prompt."""
    if batch_size <= 0 or max_input_length <= 0 or max_new_tokens <= 0:
        raise ValueError("batch size and token limits must be positive")
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("Install model dependencies with: pip install -e '.[models]'") from exc

    selected_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=(torch.float16 if str(selected_device).startswith("cuda") else torch.float32),
    ).to(selected_device)
    model.eval()
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    rows = iter(iter_jsonl(corpus_path))
    while True:
        batch = []
        for _ in range(batch_size):
            try:
                batch.append(next(rows))
            except StopIteration:
                break
        if not batch:
            break
        conversations = [
            [
                {"role": "system", "content": LENGTH_CONTROL_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": LENGTH_CONTROL_USER_PROMPT.format(
                        text=format_document(row, include_title=False)
                    ),
                },
            ]
            for row in batch
        ]
        prompts = tokenizer.apply_chat_template(
            conversations, tokenize=False, add_generation_prompt=True
        )
        encoded = tokenizer(
            prompts,
            padding=True,
            truncation=True,
            max_length=max_input_length,
            return_tensors="pt",
        ).to(selected_device)
        generation_arguments: dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "pad_token_id": tokenizer.pad_token_id,
        }
        if temperature > 0:
            generation_arguments.update(
                {"do_sample": True, "temperature": temperature, "top_p": top_p}
            )
        else:
            generation_arguments["do_sample"] = False
        with torch.inference_mode():
            output_ids = model.generate(**encoded, **generation_arguments)
        new_ids = output_ids[:, encoded["input_ids"].shape[1] :]
        rewrites = tokenizer.batch_decode(new_ids, skip_special_tokens=True)
        for row, rewrite in zip(batch, rewrites, strict=True):
            output = dict(row)
            output["text"] = clean_rewrite_text(rewrite)
            output.pop("contents", None)
            yield output


def corpus_length_statistics(
    corpora: Mapping[str, str | Path],
    *,
    tokenizer: str = "lucene",
    include_title: bool = False,
) -> dict[str, Any]:
    """Compute Lucene-token length statistics and aligned changes across corpus versions."""
    if not corpora:
        raise ValueError("At least one corpus version is required")
    tokenize = get_tokenizer(tokenizer)
    lengths: dict[str, dict[str, int]] = {}
    for name, path in corpora.items():
        corpus = load_corpus(path)
        lengths[name] = {
            document_id: len(tokenize(format_document(row, include_title=include_title)))
            for document_id, row in corpus.items()
        }
    summaries = {}
    for name, values in lengths.items():
        numbers = list(values.values())
        if not numbers:
            raise ValueError(f"Corpus version {name!r} is empty")
        ordered = sorted(numbers)
        middle = len(ordered) // 2
        median = (
            float(ordered[middle])
            if len(ordered) % 2
            else float((ordered[middle - 1] + ordered[middle]) / 2)
        )
        summaries[name] = {
            "num_documents": len(numbers),
            "mean_tokens": float(sum(numbers) / len(numbers)),
            "median_tokens": median,
            "minimum_tokens": int(min(numbers)),
            "maximum_tokens": int(max(numbers)),
        }
    names = list(corpora)
    reference = names[0]
    comparisons = {}
    for name in names[1:]:
        shared = sorted(set(lengths[reference]) & set(lengths[name]))
        changes = [lengths[name][item] - lengths[reference][item] for item in shared]
        ratios = [
            lengths[name][item] / lengths[reference][item]
            for item in shared
            if lengths[reference][item] > 0
        ]
        comparisons[name] = {
            "reference": reference,
            "paired_documents": len(shared),
            "mean_token_change": float(sum(changes) / len(changes)) if changes else None,
            "mean_length_ratio": float(sum(ratios) / len(ratios)) if ratios else None,
            "fraction_longer": (
                float(sum(change > 0 for change in changes) / len(changes)) if changes else None
            ),
        }
    return {
        "tokenizer": tokenizer,
        "include_title": include_title,
        "reference": reference,
        "summaries": summaries,
        "comparisons": comparisons,
    }
