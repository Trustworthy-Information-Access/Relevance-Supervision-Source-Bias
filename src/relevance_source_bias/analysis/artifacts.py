"""Linguistic artifact measurements and their association with retrieval scores."""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats

from ..core.data import format_document, iter_jsonl

TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+(?:['’-][A-Za-z0-9]+)*")


def regex_tokenize(text: str) -> list[str]:
    return [match.group(0).lower() for match in TOKEN_PATTERN.finditer(text)]


def get_tokenizer(name: str) -> Callable[[str], list[str]]:
    """Return the requested tokenizer (`regex` or Apache Lucene through Pyserini)."""
    if name == "regex":
        return regex_tokenize
    if name == "lucene":
        try:
            from pyserini.analysis import Analyzer, get_lucene_analyzer
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("Install Lucene support with: pip install -e '.[lucene]'") from exc
        analyzer = Analyzer(get_lucene_analyzer())
        return lambda text: list(analyzer.analyze(text))
    raise ValueError("tokenizer must be 'regex' or 'lucene'")


def _row_id(row: Mapping[str, Any]) -> str:
    value = row.get("_id", row.get("id", row.get("docid")))
    if value is None:
        raise ValueError("A JSONL row has no _id, id, or docid")
    return str(value)


def _row_text(row: Mapping[str, Any], include_title: bool) -> str:
    return format_document(row, include_title=include_title)


def build_idf(
    corpus_path: str | Path,
    *,
    tokenizer: str = "lucene",
    include_title: bool = False,
) -> dict[str, Any]:
    """Compute `log(N / (1 + df))` on a JSONL collection."""
    tokenize = get_tokenizer(tokenizer)
    document_frequency: Counter[str] = Counter()
    document_count = 0
    for row in iter_jsonl(corpus_path):
        document_frequency.update(set(tokenize(_row_text(row, include_title))))
        document_count += 1
    values = {
        term: math.log(document_count / (1.0 + frequency))
        for term, frequency in document_frequency.items()
    }
    return {
        "num_documents": document_count,
        "tokenizer": tokenizer,
        "include_title": include_title,
        "formula": "log(N / (1 + df))",
        "idf": values,
    }


def score_idf_rows(
    corpus_path: str | Path,
    idf_data: Mapping[str, Any],
    *,
    tokenizer: str | None = None,
    include_title: bool | None = None,
) -> Iterable[dict[str, Any]]:
    tokenize = get_tokenizer(str(tokenizer or idf_data.get("tokenizer", "lucene")))
    use_title = bool(
        idf_data.get("include_title", False) if include_title is None else include_title
    )
    idf_map = idf_data.get("idf", idf_data)
    if not isinstance(idf_map, Mapping):
        raise ValueError("IDF input must be a term-to-value mapping or contain an 'idf' mapping")
    for row in iter_jsonl(corpus_path):
        tokens = tokenize(_row_text(row, use_title))
        values = np.asarray([float(idf_map[token]) for token in tokens if token in idf_map])
        yield {
            "_id": _row_id(row),
            "value": float(np.median(values)) if len(values) else 0.0,
            "num_tokens": len(tokens),
            "num_in_vocabulary": int(len(values)),
        }


def calculate_ppl_rows(
    corpus_path: str | Path,
    *,
    model_name: str,
    batch_size: int = 8,
    max_length: int = 512,
    include_title: bool = False,
    device: str | None = None,
) -> Iterable[dict[str, Any]]:
    """Yield per-document causal-LM perplexity, masking padding tokens."""
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("Install model dependencies with: pip install -e '.[models]'") from exc
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    selected_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=(torch.float16 if str(selected_device).startswith("cuda") else torch.float32),
    )
    model.to(selected_device)
    model.eval()
    rows = iter(iter_jsonl(corpus_path))
    while True:
        batch: list[Mapping[str, Any]] = []
        for _ in range(batch_size):
            try:
                batch.append(next(rows))
            except StopIteration:
                break
        if not batch:
            break
        texts = [_row_text(row, include_title) for row in batch]
        encoded = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(selected_device)
        with torch.inference_mode():
            logits = model(**encoded).logits[:, :-1, :].float()
        labels = encoded["input_ids"][:, 1:]
        mask = encoded["attention_mask"][:, 1:].bool()
        token_losses = torch.nn.functional.cross_entropy(
            logits.transpose(1, 2), labels, reduction="none"
        )
        token_losses = token_losses * mask
        counts = mask.sum(dim=1)
        mean_losses = token_losses.sum(dim=1) / counts.clamp_min(1)
        perplexities = torch.exp(mean_losses).cpu().numpy()
        for row, ppl, count in zip(batch, perplexities, counts.cpu().tolist(), strict=True):
            yield {"_id": _row_id(row), "value": float(ppl), "num_tokens": int(count)}


def load_feature_values(path: str | Path, field: str = "value") -> dict[str, float]:
    values: dict[str, float] = {}
    for row in iter_jsonl(path):
        document_id = _row_id(row)
        if field not in row:
            raise ValueError(f"Feature row {document_id!r} has no field {field!r}")
        values[document_id] = float(row[field])
    return values


def compare_feature_groups(left: Sequence[float], right: Sequence[float]) -> dict[str, float | int]:
    """Return descriptive statistics, Cohen's d, Welch t-test, and Mann-Whitney U."""
    left_array = np.asarray(left, dtype=np.float64)
    right_array = np.asarray(right, dtype=np.float64)
    if not len(left_array) or not len(right_array):
        raise ValueError("Both feature groups must be non-empty")
    left_var = float(left_array.var(ddof=1)) if len(left_array) > 1 else 0.0
    right_var = float(right_array.var(ddof=1)) if len(right_array) > 1 else 0.0
    pooled_denominator = max(len(left_array) + len(right_array) - 2, 1)
    pooled_std = math.sqrt(
        ((len(left_array) - 1) * left_var + (len(right_array) - 1) * right_var) / pooled_denominator
    )
    t_test = stats.ttest_ind(left_array, right_array, equal_var=False)
    mann_whitney = stats.mannwhitneyu(left_array, right_array, alternative="two-sided")
    degrees_of_freedom = len(left_array) + len(right_array) - 2
    cohens_d = (
        float((left_array.mean() - right_array.mean()) / pooled_std) if pooled_std > 0 else 0.0
    )
    # Small-sample correction for the effect size reported in the paper.
    correction = 1.0 - 3.0 / (4.0 * degrees_of_freedom - 1.0) if degrees_of_freedom > 1 else 1.0
    return {
        "left_count": int(len(left_array)),
        "right_count": int(len(right_array)),
        "left_mean": float(left_array.mean()),
        "right_mean": float(right_array.mean()),
        "left_median": float(np.median(left_array)),
        "right_median": float(np.median(right_array)),
        "mean_difference_left_minus_right": float(left_array.mean() - right_array.mean()),
        "cohens_d": cohens_d,
        "hedges_g": float(cohens_d * correction),
        "welch_t": float(t_test.statistic),
        "welch_p": float(t_test.pvalue),
        "mann_whitney_u": float(mann_whitney.statistic),
        "mann_whitney_p": float(mann_whitney.pvalue),
    }


def _base_id(document_id: str, sources: Sequence[str]) -> str:
    for source in sorted(sources, key=len, reverse=True):
        suffix = f"-{source}"
        if document_id.endswith(suffix):
            return document_id[: -len(suffix)]
    return document_id


def correlate_scores_with_feature(
    run: Mapping[str, Mapping[str, float]],
    features: Mapping[str, float],
    *,
    sources: Sequence[str] = (),
) -> dict[str, float | int]:
    """Correlate features with scores z-normalized separately within each query."""
    count = 0
    sum_x = sum_y = sum_xx = sum_yy = sum_xy = 0.0
    queries_used = 0
    for document_scores in run.values():
        pairs: list[tuple[float, float]] = []
        for document_id, score in document_scores.items():
            value = features.get(document_id)
            if value is None:
                value = features.get(_base_id(document_id, sources))
            if value is not None and math.isfinite(float(value)) and math.isfinite(float(score)):
                pairs.append((float(score), float(value)))
        if len(pairs) < 2:
            continue
        scores = np.asarray([pair[0] for pair in pairs], dtype=np.float64)
        standard_deviation = float(scores.std())
        if standard_deviation <= 0:
            continue
        normalized_scores = (scores - scores.mean()) / standard_deviation
        feature_values = np.asarray([pair[1] for pair in pairs], dtype=np.float64)
        count += len(pairs)
        queries_used += 1
        sum_x += float(normalized_scores.sum())
        sum_y += float(feature_values.sum())
        sum_xx += float(np.dot(normalized_scores, normalized_scores))
        sum_yy += float(np.dot(feature_values, feature_values))
        sum_xy += float(np.dot(normalized_scores, feature_values))
    numerator = count * sum_xy - sum_x * sum_y
    denominator = math.sqrt(
        max(count * sum_xx - sum_x * sum_x, 0.0) * max(count * sum_yy - sum_y * sum_y, 0.0)
    )
    correlation = numerator / denominator if denominator > 0 else 0.0
    if count > 2 and abs(correlation) < 1.0:
        t_value = correlation * math.sqrt((count - 2) / (1.0 - correlation**2))
        p_value = float(2.0 * stats.t.sf(abs(t_value), df=count - 2))
    elif count > 2:
        p_value = 0.0
    else:
        p_value = 1.0
    return {
        "pearson_r": float(correlation),
        "p_value": p_value,
        "num_pairs": int(count),
        "num_queries": int(queries_used),
    }
