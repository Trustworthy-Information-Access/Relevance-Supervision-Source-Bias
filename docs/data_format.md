# Input and output formats

All IDs are read as strings, including numeric MS MARCO IDs.

## Text collections

Corpora use one JSON object per line:

```json
{"_id":"doc-1","title":"Optional title","text":"Passage text","metadata":{}}
```

Queries use the same convention:

```json
{"_id":"query-1","text":"query text"}
```

`id` can be used instead of `_id`; `contents` can be used instead of `text` for
corpus files. Generated and human corpus files should use matching base IDs.

## Qrels

Qrels may use the three-column BEIR format, with or without its header:

```text
query-id	corpus-id	score
query-1	doc-1	1
```

Four-column TREC qrels (`query-id Q0 corpus-id score`) are also accepted.

Qrels contain base document IDs. During mixed-corpus evaluation the code applies the
same relevance label to every requested source counterpart.

## Retrieval runs

Runs may be nested JSON dictionaries. Mixed-corpus document IDs have a source suffix:

```json
{
  "query-1": {
    "doc-1-human": 12.4,
    "doc-1-llama-2-7b-chat-tmp0.2": 12.1
  }
}
```

The run loader also accepts standard six-column TREC run files. This is useful when
passing Pyserini BM25 output to `rsb prepare-pn-pairs`.

`rsb bm25` accepts the corpus and query JSONL forms above, converts them to Pyserini
input under its `--work-dir`, and writes this six-column TREC format.

## Embeddings

The CLI accepts `.npz`, `.h5`, and `.hdf5` files. Each file contains:

- `ids`: a one-dimensional UTF-8 string array;
- `embeddings`: a `[num_items, dimension]` float array.

NPZ is the default output. The HDF5 loader also understands the older `id_to_idx`
mapping used by the original experiment scripts.

## Artifact features

PPL and passage IDF are JSONL files with at least `_id` and `value`:

```json
{"_id":"doc-1","value":14.82,"num_tokens":107}
```

`rsb ppl` and `rsb idf-score` accept `--source` to append the same source suffix used
by mixed-corpus retrieval runs.

## Controlled training plan

`rsb prepare-training` writes one JSON object per mini-batch:

```json
{
  "epoch": 0,
  "batch_idx": 0,
  "samples": [
    {"query_id":"1","positive_id":"100","hard_negative_id":"200"}
  ]
}
```

The same plan is read without shuffling for all three RQ3 loss conditions. This keeps
the sampled triples, batch composition, and epoch order fixed.

For the positive-pool control, `rsb prepare-positive-pool-negatives` writes the same
three fields in a flat JSONL file. Set `data.pairs` in a prepare-training configuration
to batch and repeat those fixed pairs over the requested epochs.

A flat JSONL file with one pair per line can be supplied to
`rsb direction --pairs ...` to align negative and positive passage embeddings. The
`rsb prepare-pn-pairs` command produces the paper's BM25 top-10 pairing protocol in
this format. Use `--left-id-field` and `--right-id-field` when field names differ from
`hard_negative_id` and `positive_id`.

## Sign-flip configuration

`rsb sign-flip` reads a YAML mapping from dataset names to paired LH embedding files.
The files must have matching base IDs after their configured source suffixes are
removed. The `reference` field points to a PN direction JSON produced by
`rsb direction`; see `configs/sign_flip.example.yaml`.
