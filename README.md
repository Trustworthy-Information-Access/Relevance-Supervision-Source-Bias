# Relevance Supervision Can Teach Neural Retrievers to Prefer LLM-Generated Text

This repository contains the implementation for our paper, **“Relevance Supervision Can
Teach Neural Retrievers to Prefer LLM-Generated Text,”** accepted to Findings of EMNLP
2026. The preprint is available on [arXiv](https://arxiv.org/abs/2604.06163). It covers
the three experimental threads in the paper:

1. evaluating source preference with $\Delta\mathrm{NDSR}@k$ and retrieval quality;
2. analyzing PPL/IDF artifacts and embedding-space directions; and
3. mitigating source bias through controlled negative sampling or an inference-time
   projection.

The code was consolidated from the original experiment scripts into a small Python
package. Every command takes explicit paths, writes machine-readable outputs, and can be
run independently.

## Repository layout

```text
configs/                 model registry and experiment templates
docs/                    input formats and reproduction details
scripts/                 end-to-end command examples
src/relevance_source_bias/
  core/                   configuration, BEIR/Cocktail I/O, embedding tables
  retrieval/              dense encoding/search, Pyserini BM25, evaluation metrics
  analysis/               PPL/IDF, embedding geometry, significance tests
  training/               deterministic plans, controlled losses, training runner
  interventions/          projection, CDC, positive-pool, length control
  workflows/              resumable experiment matrices and result aggregation
  cli/                    command implementations grouped by experiment domain
tests/                   synthetic unit and CLI tests
```

## Installation

Python 3.10 or 3.11 is required. The tested versions of the main Python
dependencies are recorded in `requirements/constraints.txt`.

For a reproducible Conda environment, including the Java runtime used by
Pyserini, run:

```bash
conda env create -f environment.yml
conda activate relevance-source-bias
```

Alternatively, create a virtual environment and apply the same constraints:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install -c requirements/constraints.txt -e '.[models,dev]'
```

Install the tested Pyserini/Lucene stack when recomputing BM25 or the Lucene
IDF analysis. Pyserini 0.24.0 is tested with Java 11; `environment.yml`
installs that Java version automatically.

```bash
python -m pip install -c requirements/constraints.txt -e '.[lucene]'
java -version
```

## Configure paths

Copy the evaluation template and change `data_root` to the local dataset path:

```bash
cp configs/evaluation.example.yaml configs/evaluation.local.yaml
```

The expected dataset layout is the standard Cocktail/BEIR-style layout:

```text
<data_root>/<dataset>/
├── corpus/
│   ├── human.jsonl
│   └── llama-2-7b-chat-tmp0.2.jsonl
├── queries.jsonl
└── qrels/test.tsv              # dev.tsv for MS MARCO
```

Each generated passage has the same base `_id` as its human counterpart. See
[`docs/data_format.md`](docs/data_format.md) for all accepted formats.

## RQ1: retrieval and source preference

The model registry in [`configs/models.yaml`](configs/models.yaml) contains the 13
main retrievers used in the paper, plus the appendix's NQ-only DPR control. It also
records model-specific prefixes and separate DRAGON encoders.

Encode a mixed corpus and run exact dense retrieval:

```bash
rsb retrieve \
  --config configs/evaluation.local.yaml \
  --models configs/models.yaml \
  --dataset scifact \
  --model ance
```

The command writes query, human, and LLM embeddings, plus `run.json`. Evaluate both
source preference and retrieval effectiveness with:

```bash
rsb evaluate \
  --run outputs/scifact/ance/run.json \
  --qrels /path/to/data/scifact/qrels/test.tsv \
  --human-source human \
  --llm-source llama-2-7b-chat-tmp0.2 \
  --k 5 \
  --output outputs/scifact/ance/metrics.json
```

`delta_ndsr@5 < 0` means that the retriever prefers LLM-generated passages. The output
also contains the two-sided one-sample t-test over per-query differences and NDCG@5 on
the mixed corpus.

### Relevance fine-tuning

First build a deterministic training plan. Reuse the same plan for every loss condition
so the query/positive/hard-negative triples and batch composition are held fixed.

```bash
rsb prepare-training \
  --config configs/training/prepare.example.yaml
```

Fine-tuning templates are provided for Contriever, E5-unsup, and SimCSE. After changing
their paths, run the desired configuration:

```bash
rsb train --config configs/training/rq1.example.yaml
```

The other two templates are `rq1_e5_unsup.example.yaml` and
`rq1_simcse.example.yaml`. Add the resulting checkpoint path to `configs/models.yaml`
to evaluate it with the same retrieval pipeline.

The paper uses 10 epochs, batch size 75, learning rate `2e-5`, 1,000 warmup steps, and
seeds 42, 100, and 2026.

### Length-controlled evaluation

Create meaning-preserving longer variants with the paper's Qwen2.5-7B-Instruct prompt,
then verify the token-length change:

```bash
rsb length-rewrite \
  --corpus /path/to/cocktail/scifact/corpus/llama-2-7b-chat-tmp0.2.jsonl \
  --output outputs/length/scifact.jsonl

rsb length-stats \
  --corpus original=/path/to/cocktail/scifact/corpus/llama-2-7b-chat-tmp0.2.jsonl \
  --corpus longer=outputs/length/scifact.jsonl \
  --output outputs/length/scifact-stats.json
```

Point `corpus_pattern` in a copied evaluation configuration at the rewritten files and
run the same matrix. The generation settings are recorded alongside the output.

### Full evaluation matrix

After configuring paths, run or resume all dataset--retriever cells with:

```bash
cp configs/matrix.example.yaml configs/matrix.local.yaml
rsb run-matrix --config configs/matrix.local.yaml
```

Completed `metrics.json` files are skipped. The runner updates its manifest after every
cell and writes consolidated JSON, CSV, and LaTeX tables. To aggregate existing runs
without encoding, use `--evaluation-only`; to rebuild only the summaries, use:

```bash
rsb aggregate-matrix --config configs/matrix.local.yaml
```

Fine-tuned checkpoints from seeds 42, 100, and 2026 can be registered under templated
names as shown in `configs/matrix.example.yaml`. The aggregate output includes the
mean absolute Delta-NDSR for every model and seed.

## RQ2: artifact and embedding analyses

### PPL and IDF

Compute document PPL with a causal language model:

```bash
rsb ppl \
  --corpus /path/to/data/msmarco/corpus/human.jsonl \
  --model meta-llama/Meta-Llama-3-8B-Instruct \
  --batch-size 8 \
  --source human \
  --output outputs/artifacts/msmarco-human-ppl.jsonl
```

Build an IDF map on the full collection, then compute passage-level median IDF:

```bash
rsb idf-build \
  --corpus /path/to/data/msmarco/corpus/human.jsonl \
  --tokenizer lucene \
  --output outputs/artifacts/msmarco-idf.json

rsb idf-score \
  --corpus /path/to/data/msmarco/corpus/human.jsonl \
  --idf outputs/artifacts/msmarco-idf.json \
  --tokenizer lucene \
  --source human \
  --output outputs/artifacts/msmarco-human-idf.jsonl
```

Compare feature distributions or correlate a feature with within-query normalized
retrieval scores:

```bash
rsb compare-features --left positive.jsonl --right negative.jsonl --field value
rsb correlate-scores --run run.json --features all-document-ppl.jsonl --field value
```

Feature comparison reports descriptive statistics, Hedges' $g$, Cohen's $d$, and
two-sided significance tests. Score correlation first z-normalizes scores within each
query, as in the paper. For mixed-corpus correlations, compute human and LLM features
with their respective `--source` values and concatenate the two JSONL outputs.
The supplied evaluation configurations retain 200 results per query, matching the
retrieval depth used for the paper's artifact--score analysis.

### Embedding directions

Estimate the paired LLM-minus-human direction and its within-dataset consistency:

```bash
rsb direction \
  --human outputs/scifact/ance/human.npz \
  --llm outputs/scifact/ance/llama-2-7b-chat-tmp0.2.npz \
  --human-suffix=-human \
  --llm-suffix=-llama-2-7b-chat-tmp0.2 \
  --permutations 1000 \
  --output outputs/scifact/ance/lh-direction.json
```

For the MS MARCO positive-minus-negative (PN) analysis, first sample one negative per
query from its top-10 BM25 candidates, then align the resulting pairs with the original
passage embeddings:

```bash
rsb bm25 \
  --corpus /path/to/msmarco/corpus.jsonl \
  --queries /path/to/msmarco/queries.jsonl \
  --work-dir outputs/msmarco/bm25 \
  --hits 10 \
  --output outputs/msmarco/bm25/run.txt

rsb prepare-pn-pairs \
  --run outputs/msmarco/bm25/run.txt \
  --qrels /path/to/msmarco/qrels/train.tsv \
  --top-k 10 \
  --seed 42 \
  --output outputs/msmarco/pn-pairs.jsonl

rsb direction \
  --left outputs/msmarco/dragon/human.npz \
  --right outputs/msmarco/dragon/human.npz \
  --left-suffix=-human \
  --right-suffix=-human \
  --pairs outputs/msmarco/pn-pairs.jsonl \
  --permutations 1000 \
  --output outputs/msmarco/dragon/pn-direction.json
```

The command always computes `right - left`: human-to-LLM is therefore obtained by
passing human on the left and LLM on the right, while PN uses negatives on the left and
positives on the right. Compare saved dataset directions and their PN alignment with:

```bash
rsb compare-directions \
  --direction msmarco=outputs/msmarco/dragon/lh-direction.json \
  --direction scifact=outputs/scifact/dragon/lh-direction.json \
  --reference outputs/msmarco/dragon/pn-direction.json \
  --output outputs/dragon/direction-comparison.json
```

Run the appendix's empirical validation from the paired embeddings themselves:

```bash
cp configs/sign_flip.example.yaml configs/sign_flip.local.yaml
# Add the remaining dataset-level collections and change all paths.
rsb sign-flip \
  --config configs/sign_flip.local.yaml \
  --permutations 1000 \
  --seed 42 \
  --output outputs/dragon/sign-flip.json
```

This jointly recomputes the within-dataset LH, cross-dataset LH, and PN--LH
statistics in every permutation. Dataset-level statistics are macro-averaged within
each permutation, rather than averaging per-dataset p-values.

The descriptive 3-sigma reference used in the embedding figures is available without
model inference:

```bash
rsb random-reference --dimension 768 --sigma-multiplier 3
```

## RQ3: mitigation

### Controlled negative sampling

Run the same training plan with the three `negative_strategy` values:

```text
in_batch_only  other positives in the mini-batch are negatives
standard       other positives plus all mined hard negatives in the mini-batch
hard_neg_only  each query's own positive and sampled hard negative only
```

Templates are provided in [`configs/training`](configs/training). The implementation is
a configurable version of the controlled BEIR/SentenceTransformers training scripts used
for the paper.

For the appendix control that mines from other queries' annotated positives, build the
restricted corpus, retrieve it with BM25, and create a plan-compatible pair file:

```bash
rsb build-positive-pool \
  --corpus /path/to/msmarco/corpus.jsonl \
  --qrels /path/to/msmarco/qrels/train.tsv \
  --output outputs/positive-pool/corpus.jsonl

rsb bm25 \
  --corpus outputs/positive-pool/corpus.jsonl \
  --queries /path/to/msmarco/queries.jsonl \
  --work-dir outputs/positive-pool/bm25 \
  --output outputs/positive-pool/bm25/run.txt

rsb prepare-positive-pool-negatives \
  --run outputs/positive-pool/bm25/run.txt \
  --qrels /path/to/msmarco/qrels/train.tsv \
  --output outputs/positive-pool/training-pairs.jsonl

rsb prepare-training \
  --config configs/training/prepare_positive_pool.example.yaml
```

Point a copy of the standard training configuration at the resulting positive-pool
plan. The `dpr-nq` entry in `configs/models.yaml` reproduces the appendix's NQ-only DPR
evaluation through the regular retrieval or matrix command.

### Leave-one-dataset-out projection

Prepare a YAML file that maps calibration dataset groups to paired embedding files, then
run:

```bash
rsb debias \
  --calibration configs/calibration.example.yaml \
  --target scifact \
  --passages outputs/scifact/ance/human.npz outputs/scifact/ance/llama-2-7b-chat-tmp0.2.npz \
  --output outputs/scifact/ance/human.debiased.npz outputs/scifact/ance/llm.debiased.npz \
  --samples-per-group 100 \
  --seed 42
```

MS MARCO, DL19, and DL20 are represented by one shared collection group in the example.
The target group is excluded, 100 paired passages are sampled from each remaining group,
and the normalized mean LLM-minus-human displacement is projected out. Passage vectors
are not renormalized after projection, matching the paper.

To evaluate the projected vectors, rerun exact search without encoding:

```bash
rsb search \
  --queries outputs/scifact/ance/queries.npz \
  --corpus outputs/scifact/ance/human.debiased.npz \
  --corpus outputs/scifact/ance/llm.debiased.npz \
  --score dot \
  --top-k 200 \
  --output outputs/scifact/ance/run.debiased.json
```

### CDC and cross-generator controls

Apply the appendix's PPL-based CDC baseline using a 128-pair calibration sample:

```bash
rsb cdc \
  --calibration-run outputs/scifact/ance/run.json \
  --calibration-qrels /path/to/cocktail/scifact/qrels/test.tsv \
  --calibration-features outputs/scifact/all-ppl.jsonl \
  --run outputs/scifact/ance/run.json \
  --features outputs/scifact/all-ppl.jsonl \
  --output outputs/scifact/ance/run.cdc.json
```

Evaluate `run.cdc.json` with `rsb evaluate`. For Qwen3-32B counterparts, copy
`configs/evaluation_qwen.example.yaml` and `configs/calibration_qwen.example.yaml`,
change the paths, and run the same retrieve, debias, search, and evaluate sequence.

## Tests

The unit tests use only synthetic arrays and temporary files:

```bash
python -m pip install -c requirements/constraints.txt -e '.[dev]'
pytest
```

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
