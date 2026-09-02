# Relevance Supervision Can Teach Neural Retrievers to Prefer LLM-Generated Text

This repository contains the implementation for our paper, **“Relevance Supervision Can
Teach Neural Retrievers to Prefer LLM-Generated Text,”** accepted to Findings of EMNLP
2026. The preprint is available on [arXiv](https://arxiv.org/abs/2604.06163).

The repository covers three experimental threads:

1. evaluating source preference with $\Delta\mathrm{NDSR}@k$ and retrieval quality;
2. analyzing PPL/IDF artifacts and embedding-space directions; and
3. mitigating source bias through controlled negative sampling or an inference-time
   projection.

## Installation

Python 3.10 or 3.11 is required. The tested dependency versions are recorded in
`requirements/constraints.txt`.

Create the complete Conda environment, including Java 11 for Pyserini:

```bash
conda env create -f environment.yml
conda activate relevance-source-bias
```

Alternatively, create a virtual environment:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install -c requirements/constraints.txt -e '.[models,dev]'
```

BM25 retrieval and Lucene IDF analysis additionally require the Lucene extra:

```bash
python -m pip install -c requirements/constraints.txt -e '.[lucene]'
```

The tested Lucene stack uses Pyserini 0.24.0 and Java 11.

## Quick start

Copy the evaluation template and change `data_root` to the local dataset path:

```bash
cp configs/evaluation.example.yaml configs/evaluation.local.yaml
```

The default configuration expects a Cocktail/BEIR-style layout:

```text
<data_root>/<dataset>/
├── corpus/
│   ├── human.jsonl
│   └── llama-2-7b-chat-tmp0.2.jsonl
├── queries.jsonl
└── qrels/test.tsv
```

Run dense retrieval on one dataset and model:

```bash
rsb retrieve \
  --config configs/evaluation.local.yaml \
  --models configs/models.yaml \
  --dataset scifact \
  --model ance
```

Evaluate source preference and retrieval effectiveness:

```bash
rsb evaluate \
  --run outputs/scifact/ance/run.json \
  --qrels /path/to/data/scifact/qrels/test.tsv \
  --human-source human \
  --llm-source llama-2-7b-chat-tmp0.2 \
  --k 5 \
  --output outputs/scifact/ance/metrics.json
```

A negative `delta_ndsr@5` means that the retriever prefers LLM-generated passages.
See [the input-format guide](docs/data_format.md) for accepted corpus, query, qrels,
run, feature, and embedding formats.

## Reproducing the paper

The full dataset–retriever matrix is resumable:

```bash
cp configs/matrix.example.yaml configs/matrix.local.yaml
rsb run-matrix --config configs/matrix.local.yaml
```

The main experiment entry points are:

| Paper component | Commands |
| --- | --- |
| RQ1 retrieval and source preference | `rsb retrieve`, `rsb evaluate`, `rsb run-matrix` |
| RQ1 relevance fine-tuning | `rsb prepare-training`, `rsb train` |
| RQ1 length control | `rsb length-rewrite`, `rsb length-stats` |
| RQ2 PPL and IDF artifacts | `rsb ppl`, `rsb idf-build`, `rsb idf-score` |
| RQ2 artifact–score correlation | `rsb compare-features`, `rsb correlate-scores` |
| RQ2 LH/PN embedding geometry | `rsb direction`, `rsb compare-directions`, `rsb sign-flip` |
| RQ3 controlled negative sampling | `rsb train` with the three RQ3 configurations |
| RQ3 positive-pool control | `rsb build-positive-pool`, `rsb bm25`, `rsb prepare-positive-pool-negatives` |
| RQ3 projection and CDC | `rsb debias`, `rsb search`, `rsb cdc`, `rsb evaluate` |

The model registry contains the 13 main retrievers and the appendix's NQ-only DPR
control. Complete commands, paper settings, experiment ordering, and output descriptions
are in [the reproduction guide](docs/reproduction.md).

## Repository layout

```text
configs/                 model registry and experiment templates
docs/                    input formats and complete reproduction guide
scripts/                 end-to-end command examples
src/relevance_source_bias/
  core/                   configuration, I/O, and embedding tables
  retrieval/              dense retrieval, Pyserini BM25, and metrics
  analysis/               artifact, geometry, and significance analyses
  training/               deterministic plans, losses, and training runner
  interventions/          projection, CDC, positive-pool, and length control
  workflows/              experiment matrices and result aggregation
  cli/                    command implementations
tests/                   synthetic unit and CLI tests
```

## Tests

```bash
python -m pip install -c requirements/constraints.txt -e '.[dev]'
pytest
```

GitHub Actions runs linting, formatting checks, tests, and wheel construction on Python
3.10 and 3.11. A separate manually triggered workflow runs the Pyserini/Java 11 BM25
integration test.

## Citation

```bibtex
@misc{huang2026relevance,
  title = {Relevance Supervision Can Teach Neural Retrievers to Prefer
           {LLM}-Generated Text},
  author = {Wei Huang and Keping Bi and Yinqiong Cai and Wei Chen and
            Jiafeng Guo and Xueqi Cheng},
  year = {2026},
  eprint = {2604.06163},
  archivePrefix = {arXiv},
  primaryClass = {cs.IR},
  url = {https://arxiv.org/abs/2604.06163}
}
```

## Acknowledgements

The controlled training pipeline builds on the Apache-2.0-licensed
[BEIR](https://github.com/beir-cellar/beir) and
[SentenceTransformers](https://github.com/UKPLab/sentence-transformers) projects.
Their attribution notices are retained in [`NOTICE`](NOTICE).

## License

Licensed under the Apache License 2.0. See [`LICENSE`](LICENSE).
