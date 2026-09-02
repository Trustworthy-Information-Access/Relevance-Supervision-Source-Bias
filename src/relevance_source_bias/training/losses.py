"""Contrastive objectives for the controlled negative-sampling experiments."""

from __future__ import annotations


def aligned_positive_negative_scores(positive_matrix, negative_matrix):
    """Select each query's own positive and hard-negative scores."""
    if positive_matrix.shape != negative_matrix.shape:
        raise ValueError("Positive and hard-negative score matrices must have equal shapes")
    if len(positive_matrix.shape) != 2 or positive_matrix.shape[0] != positive_matrix.shape[1]:
        raise ValueError("Aligned score matrices must be square")
    return positive_matrix.diagonal(), negative_matrix.diagonal()


def build_loss(model, strategy: str, similarity: str, scale: float):
    """Build the requested controlled contrastive objective."""
    try:
        import torch
        from sentence_transformers import losses, util
        from torch import nn
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("Install model dependencies with: pip install -e '.[models]'") from exc
    similarity_function = util.dot_score if similarity == "dot" else util.cos_sim

    class InBatchOnlyLoss(nn.Module):
        def __init__(self):
            super().__init__()
            self.cross_entropy = nn.CrossEntropyLoss()

        def forward(self, sentence_features, labels):
            anchor = model(sentence_features[0])["sentence_embedding"]
            positive = model(sentence_features[1])["sentence_embedding"]
            scores = similarity_function(anchor, positive) * scale
            targets = torch.arange(len(scores), device=scores.device)
            return self.cross_entropy(scores, targets)

    class HardNegativeOnlyLoss(nn.Module):
        def __init__(self):
            super().__init__()
            self.cross_entropy = nn.CrossEntropyLoss()

        def forward(self, sentence_features, labels):
            anchor = model(sentence_features[0])["sentence_embedding"]
            positive = model(sentence_features[1])["sentence_embedding"]
            negative = model(sentence_features[2])["sentence_embedding"]
            positive_scores, negative_scores = aligned_positive_negative_scores(
                similarity_function(anchor, positive),
                similarity_function(anchor, negative),
            )
            scores = torch.stack((positive_scores, negative_scores), dim=1) * scale
            targets = torch.zeros(len(anchor), dtype=torch.long, device=scores.device)
            return self.cross_entropy(scores, targets)

    if strategy == "in_batch_only":
        return InBatchOnlyLoss()
    if strategy == "hard_neg_only":
        return HardNegativeOnlyLoss()
    if strategy == "standard":
        return losses.MultipleNegativesRankingLoss(
            model=model, similarity_fct=similarity_function, scale=scale
        )
    raise ValueError("negative_strategy must be in_batch_only, standard, or hard_neg_only")
