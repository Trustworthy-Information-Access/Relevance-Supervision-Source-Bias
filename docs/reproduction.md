# Reproducing the paper

This guide covers the complete experimental workflow for **“Relevance Supervision Can
Teach Neural Retrievers to Prefer LLM-Generated Text.”** Run all commands from the
repository root after activating the tested environment.

## Environment and paths

Create the complete reference environment with:

```bash
conda env create -f environment.yml
conda activate relevance-source-bias
```

The reference stack uses Python 3.10, SentenceTransformers 2.7.0, Transformers 4.51.0,
PyTorch 2.2.0, Pyserini 0.24.0, and Java 11. The same Python versions can be installed
into a virtual environment with `requirements/constraints.txt`.

Copy the relevant example configuration and change its paths before running an
experiment:

```bash
cp configs/evaluation.example.yaml configs/evaluation.local.yaml
cp configs/matrix.example.yaml configs/matrix.local.yaml
```

The evaluation configuration expects `data_root/<dataset>/` to contain the corpus,
queries, and qrels paths described in [the input-format guide](data_format.md). Run
`rsb COMMAND --help` for the complete arguments of any command.

## Reproduction map

| Paper component | Command | Main output |
| --- | --- | --- |
| Full dataset/model/seed matrix | `rsb run-matrix`, `rsb aggregate-matrix` | resumable manifest, JSON, CSV, LaTeX |
| RQ1 source preference | `rsb retrieve`, `rsb evaluate` | `run.json`, `metrics.json` |
| RQ1 relevance fine-tuning | `rsb prepare-training`, `rsb train` | SentenceTransformer checkpoint |
| RQ1 length control | `rsb length-rewrite`, `rsb length-stats` | rewritten corpus and paired length statistics |
| RQ2 PPL | `rsb ppl` | document feature JSONL |
| RQ2 median IDF | `rsb idf-build`, `rsb idf-score` | IDF JSON and feature JSONL |
| RQ2 artifact–score correlation | `rsb correlate-scores` | Pearson correlation JSON |
| RQ2 LH/PN geometry | `rsb prepare-pn-pairs`, `rsb direction`, `rsb compare-directions` | paired contrasts, directions, cosine matrix |
| RQ2 empirical sign-flip validation | `rsb sign-flip` | observed/null statistics and corrected p-values |
| RQ2 random-vector reference | `rsb random-reference` | exact Beta and Gaussian tail probabilities |
| RQ3 negative sampling | `rsb train` with three configs | three controlled checkpoints |
| RQ3 positive-pool BM25 control | `rsb build-positive-pool`, `rsb bm25`, `rsb prepare-positive-pool-negatives` | fixed pair JSONL |
| RQ3 projection | `rsb debias`, `rsb search`, `rsb evaluate` | projected embeddings and metrics |
| RQ3 CDC baseline | `rsb cdc`, `rsb evaluate` | corrected run and metrics |
| Appendix NQ-only retriever | `rsb retrieve --model dpr-nq` | DPR mixed-corpus run |

## Recommended execution order

1. Copy the example configurations and change their paths.
2. Run the original RQ1 benchmark for the registered retrievers.
3. Prepare a fixed plan and fine-tune the RQ1 models for all three seeds.
4. Run the length-control evaluation.
5. Compute PPL/IDF features, PN/LH directions, and the sign-flip validation.
6. Train the three controlled negative-sampling conditions and positive-pool control.
7. Run leave-one-dataset-out projection and CDC, then evaluate the corrected runs.
8. Repeat the projection workflow with the Qwen cross-generator configurations.

## RQ1: retrieval and source preference

### Dense retrieval and evaluation

The model registry in `configs/models.yaml` contains the 13 main retrievers used in the
paper, plus the appendix's NQ-only DPR control. It records model-specific query and
passage prefixes and separate DRAGON encoders.

Encode the query and two source corpora, then run exact dense retrieval:

```bash
rsb retrieve \
  --config configs/evaluation.local.yaml \
  --models configs/models.yaml \
  --dataset scifact \
  --model ance
```

The command writes query, human, and LLM embedding tables and a `run.json`. Evaluate
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
also contains a two-sided one-sample t-test over per-query differences and NDCG@5 on the
mixed corpus.

### Full evaluation matrix

Run all configured dataset–retriever cells with:

```bash
rsb run-matrix --config configs/matrix.local.yaml
```

Completed `metrics.json` files are skipped. The runner updates its manifest after every
cell and writes consolidated JSON, CSV, and LaTeX tables. To aggregate existing runs
without encoding, use `--evaluation-only`. To rebuild only the summaries, run:

```bash
rsb aggregate-matrix --config configs/matrix.local.yaml
```

Fine-tuned checkpoints from seeds 42, 100, and 2026 can be registered under templated
names as shown in `configs/matrix.example.yaml`. The aggregate output includes the mean
absolute Delta-NDSR for every model and seed.

### Relevance fine-tuning

Build a deterministic training plan. Reuse the same plan for every loss condition so
that query, positive, hard-negative, and batch composition are held fixed:

```bash
rsb prepare-training \
  --config configs/training/prepare.example.yaml
```

Templates are provided for Contriever, E5-unsup, and SimCSE:

```bash
rsb train --config configs/training/rq1.example.yaml
rsb train --config configs/training/rq1_e5_unsup.example.yaml
rsb train --config configs/training/rq1_simcse.example.yaml
```

Generate a separate plan for each seed and change the matching output directory. Add
the resulting checkpoint paths to `configs/models.yaml` and evaluate them with the same
retrieval or matrix command.

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
run the same evaluation matrix. Generation settings are recorded alongside the output.

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
two-sided significance tests. Score correlation z-normalizes scores within each query.
For a mixed-corpus correlation, compute the human and LLM features with their respective
`--source` values and concatenate the two JSONL outputs. The evaluation configurations
retain 200 results per query for the artifact–score analysis.

### LLM–human direction

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

The command always computes `right - left`: passing human on the left and LLM on the
right produces the human-to-LLM direction.

### Positive–negative direction

For MS MARCO, retrieve BM25 candidates and sample one negative per query from its top 10:

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
```

Align the saved pairs with the passage embedding table. Negatives are on the left and
positives are on the right:

```bash
rsb direction \
  --left outputs/msmarco/dragon/human.npz \
  --right outputs/msmarco/dragon/human.npz \
  --left-suffix=-human \
  --right-suffix=-human \
  --pairs outputs/msmarco/pn-pairs.jsonl \
  --permutations 1000 \
  --output outputs/msmarco/dragon/pn-direction.json
```

Compare saved dataset directions and their PN alignment:

```bash
rsb compare-directions \
  --direction msmarco=outputs/msmarco/dragon/lh-direction.json \
  --direction scifact=outputs/scifact/dragon/lh-direction.json \
  --reference outputs/msmarco/dragon/pn-direction.json \
  --output outputs/dragon/direction-comparison.json
```

### Sign-flip validation and random reference

Copy the sign-flip configuration, add every dataset-level collection, and change the
paths:

```bash
cp configs/sign_flip.example.yaml configs/sign_flip.local.yaml
rsb sign-flip \
  --config configs/sign_flip.local.yaml \
  --permutations 1000 \
  --seed 42 \
  --output outputs/dragon/sign-flip.json
```

The validation jointly recomputes within-dataset LH, cross-dataset LH, and PN–LH
statistics in every permutation. Dataset-level statistics are macro-averaged inside
each permutation rather than averaging per-dataset p-values.

The descriptive 3-sigma random-vector reference is available without model inference:

```bash
rsb random-reference --dimension 768 --sigma-multiplier 3
```

## RQ3: mitigation

### Controlled negative sampling

Use one fixed training plan with the three `negative_strategy` conditions:

```text
in_batch_only  other positives in the mini-batch are negatives
standard       other positives plus all mined hard negatives in the mini-batch
hard_neg_only  each query's own positive and sampled hard negative only
```

Run the supplied templates:

```bash
rsb train --config configs/training/rq3_in_batch_only.example.yaml
rsb train --config configs/training/rq3_standard.example.yaml
rsb train --config configs/training/rq3_hard_neg_only.example.yaml
```

### Positive-pool BM25 control

Build the corpus restricted to annotated positives, retrieve it with BM25, and create a
plan-compatible pair file:

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
plan. The `dpr-nq` entry in `configs/models.yaml` reproduces the appendix's NQ-only
DPR evaluation through the regular retrieval or matrix command.

### Leave-one-dataset-out projection

Use a calibration configuration that maps dataset groups to paired embedding files:

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
The target group is excluded, and 100 paired passages are sampled from each remaining
group. The normalized mean LLM-minus-human displacement is projected out of passage
vectors without post-projection renormalization.

Search the projected vectors without re-encoding:

```bash
rsb search \
  --queries outputs/scifact/ance/queries.npz \
  --corpus outputs/scifact/ance/human.debiased.npz \
  --corpus outputs/scifact/ance/llm.debiased.npz \
  --score dot \
  --top-k 200 \
  --output outputs/scifact/ance/run.debiased.json
```

Evaluate `run.debiased.json` with the same `rsb evaluate` command used for RQ1.

### CDC and cross-generator controls

Apply the PPL-based CDC baseline with a 128-pair calibration sample:

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
change their paths, and repeat the retrieve, debias, search, and evaluate sequence.

## Fixed settings from the paper

- Retrieval uses independently encoded queries and passages, maximum passage length
  512, depth 200, and the checkpoint-specific prefixes in `configs/models.yaml`.
- Source preference is $\Delta\mathrm{NDSR}@5 =
  \mathrm{NDSR}_{Human}@5-\mathrm{NDSR}_{LLM}@5$.
- PPL uses Llama-3-8B-Instruct for the main analysis and Qwen3-8B for the cross-family
  check.
- Passage IDF is the median token IDF with
  $\mathrm{IDF}(t)=\log(N/(1+\mathrm{df}(t)))$ and Lucene tokenization.
- The PN direction pairs each judged MS MARCO positive with one randomly sampled
  negative from the query's top-10 BM25 candidates.
- The paired sign-flip validation uses 1,000 permutations. Within-dataset LH and PN–LH
  statistics are macro-averaged over dataset-level collections inside every
  permutation; the PN direction remains fixed.
- Fine-tuning uses InfoNCE, batch size 75, 10 epochs, AdamW learning rate `2e-5`,
  1,000 warmup steps, and seeds 42, 100, and 2026.
- The hard-neg-only loss compares each query only with its own annotated positive and
  sampled hard negative; cross-query candidates are excluded.
- The positive-pool control uses Pyserini BM25 with `k1=0.82` and `b=0.68`, then
  selects the highest-ranked positive belonging to another query.
- Length-controlled rewriting uses Qwen2.5-7B-Instruct, temperature 0.2, and the prompt
  printed in the paper appendix.
- Projection calibration samples 100 paired passages from each non-target collection
  group. MS MARCO, DL19, and DL20 form one group.
- Projection is applied only to passage embeddings, without post-projection
  renormalization.
