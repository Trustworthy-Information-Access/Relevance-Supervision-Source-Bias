# Reproduction map

Create the tested environment with `conda env create -f environment.yml`, or pass
`-c requirements/constraints.txt` to the installation commands in the main README.
The reference stack uses Python 3.10, SentenceTransformers 2.7.0, Transformers 4.51.0,
PyTorch 2.2.0, Pyserini 0.24.0, and Java 11.

| Paper component | Command | Main output |
| --- | --- | --- |
| Full dataset/model/seed matrix | `rsb run-matrix`, `rsb aggregate-matrix` | resumable manifest, JSON, CSV, LaTeX |
| RQ1 source preference | `rsb retrieve`, `rsb evaluate` | `run.json`, `metrics.json` |
| RQ1 relevance fine-tuning | `rsb prepare-training`, `rsb train` | SentenceTransformer checkpoint |
| RQ1 length control | `rsb length-rewrite`, `rsb length-stats` | rewritten corpus and paired length statistics |
| RQ2 PPL | `rsb ppl` | document feature JSONL |
| RQ2 median IDF | `rsb idf-build`, `rsb idf-score` | IDF JSON and feature JSONL |
| RQ2 artifact-score correlation | `rsb correlate-scores` | Pearson correlation JSON |
| RQ2 LH/PN geometry | `rsb prepare-pn-pairs`, `rsb direction`, `rsb compare-directions` | paired contrasts, directions, cosine matrix |
| RQ2 empirical sign-flip validation | `rsb sign-flip` | observed/null statistics and corrected p-values |
| RQ2 random-vector reference | `rsb random-reference` | exact Beta and Gaussian tail probabilities |
| RQ3 negative sampling | `rsb train` with three configs | three controlled checkpoints |
| RQ3 positive-pool BM25 control | `rsb build-positive-pool`, `rsb bm25`, `rsb prepare-positive-pool-negatives` | fixed pair JSONL |
| RQ3 projection | `rsb debias`, `rsb search`, `rsb evaluate` | projected embeddings and metrics |
| RQ3 CDC baseline | `rsb cdc`, `rsb evaluate` | corrected run and metrics |
| Appendix NQ-only retriever | `rsb retrieve --model dpr-nq` | DPR mixed-corpus run |

## Fixed settings from the paper

- Retrieval uses independently encoded queries and passages, maximum passage length
  512, depth 200, and the checkpoint-specific prefixes in `configs/models.yaml`.
- Source preference is $\Delta\mathrm{NDSR}@5 =
  \mathrm{NDSR}_{Human}@5-\mathrm{NDSR}_{LLM}@5$.
- PPL uses Llama-3-8B-Instruct for the main analysis and Qwen3-8B for the
  cross-family check.
- Passage IDF is the median token IDF with
  $\mathrm{IDF}(t)=\log(N/(1+\mathrm{df}(t)))$ and Lucene tokenization.
- The PN direction pairs each judged MS MARCO positive with one randomly sampled
  negative from the query's top-10 BM25 candidates.
- The paired sign-flip validation uses 1,000 permutations. Within-dataset LH and
  PN--LH statistics are macro-averaged over dataset-level collections inside every
  permutation; the PN direction remains fixed.
- Fine-tuning uses InfoNCE, batch size 75, 10 epochs, AdamW learning rate `2e-5`,
  1,000 warmup steps, and seeds 42, 100, and 2026.
- The controlled hard-neg-only loss compares each query only with its own annotated
  positive and sampled hard negative; all cross-query candidates are excluded.
- The positive-pool control uses Pyserini BM25 with `k1=0.82` and `b=0.68` by default,
  then selects the highest-ranked positive belonging to another query.
- Length-controlled rewriting uses Qwen2.5-7B-Instruct, temperature 0.2, and the exact
  prompt printed in the paper appendix.
- Projection calibration samples 100 paired passages from each non-target collection
  group. MS MARCO, DL19, and DL20 form one group.
- Projection is applied to passage embeddings only, with no post-projection
  renormalization.

For a multi-seed run, generate a separate plan for each seed and change the output
directory in the matching training configuration.

## Recommended execution order

1. Copy the example configuration files and change their paths.
2. Run `rsb run-matrix` for the original RQ1 benchmark and registered fine-tuned models.
3. Compute PPL/IDF features, PN/LH directions, and the sign-flip validation.
4. Train the three controlled negative-sampling conditions and the positive-pool control.
5. Run leave-one-dataset-out projection and CDC, then evaluate their corrected runs.
6. Repeat the projection workflow with the Qwen example configurations.
