# DHS LLW & Immigration Enforcement Evaluation

This folder contains the evaluation materials for the DHS Less-Lethal Weapons and Immigration Enforcement case study used in the Rawabit master's thesis prototype.

It is intended to help a reviewer reproduce or inspect the workflow:

1. Generate or validate golden queries.
2. Create a Rawabit case from the evidence casepack.
3. Run the thesis evaluation script against an ingested case.
4. Inspect the published result artifacts.

## Folder Layout

```text
casepack/        Evidence PDFs used for the evaluation
goldens/         Canonical and generated golden query sets
results/         Evaluation outputs included for review, except results/old/
scripts/         Scripts for goldens, case setup, and thesis evaluation
```

The published repository should include the contents of `results/` except the `results/old/` directory. The non-`old` result folders are the artifacts to inspect when reviewing the reported thesis evaluation.

## Prerequisites

Start the Rawabit backend before running setup or evaluation:

```powershell
conda activate rawabit
python -m uvicorn backend.app.main:app --reload --host 127.0.0.1 --port 8000
```

Model-backed steps require API keys in the project root `.env` file or equivalent environment variables. Do not commit `.env`.

Common variables:

```text
RAWABIT_LLM_PROVIDER_API_KEY=your_key_here
RAWABIT_LLM_PROVIDER_BASE_URL=https://openrouter.ai/api/v1
RAWABIT_EMBEDDING_PROVIDER_API_KEY=your_key_here
RAWABIT_EMBEDDING_PROVIDER_BASE_URL=https://openrouter.ai/api/v1
EVALUATOR_MODEL=openai/gpt-5.4
```

## Golden Queries

The reviewed canonical dataset is:

```text
goldens/golden_queries.json
```

It contains 30 queries across link, temporal, flow, factual verification, and retrieval-honesty analysis types.


To generate a new synthetic dataset with an LLM:

```powershell
conda activate rawabit
python "evaluation\DHS LLW & Immigration Enforcement\scripts\generate_synthetic_goldens.py" `
  --model "gpt-5.4"
```

New synthetic runs are written under:

```text
goldens/generated/<timestamp_model>/golden_queries.json
```


## Casepack Setup

The casepack setup script creates a new Rawabit case, uploads the top-level files in `casepack/`, and waits for the related ingestion jobs to complete.

Run:

```powershell
conda activate rawabit
python "evaluation\DHS LLW & Immigration Enforcement\scripts\setup_casepack.py" `
  --casepack-dir "evaluation\DHS LLW & Immigration Enforcement\casepack" `
  --name "DHS LLW & Immigration Enforcement [Eval]" `
  --model "gpt-5.4-mini" `
  --base-url "http://localhost:8000" `
  --include "*.pdf" `
  --tags "dhs,llw,immigration,evaluation" `
  --notes-prefix "DHS LLW evaluation casepack"
```

The script prints the created `case_id`; use that value with `scripts/run_thesis_evaluation.py`.

The setup script uploads each evidence file with:

- MIME type inferred from the filename
- SHA-256 content hash
- confidence code `A1` by default
- `balanced_fast_intel` ingest profile by default
- `multimodal` processing mode by default
- tags and notes supplied by the command

## Running Evaluation

For reviewer reproduction in this repository, use the current script location:

```text
scripts/run_thesis_evaluation.py
```

The script reads goldens from `goldens/golden_queries.json`, queries the running Rawabit backend when `--run-queries` is passed, and writes a timestamped run directory under `results/` unless `--output-dir` is supplied. The commands below keep the substantive parameters used in the main evaluation runs.

The run command used for the main evaluation using GPT-5.4 mini:

```powershell
conda activate rawabit
python "evaluation\DHS LLW & Immigration Enforcement\scripts\run_thesis_evaluation.py" `
  --eval-dir "evaluation\DHS LLW & Immigration Enforcement" `
  --case-id 09fb881d-a36f-4757-b671-cf95c4f2fc73 `
  --model-slug "gpt-5.4-mini" `
  --RQs 1,2 `
  --modes hybrid,naive `
  --run-queries `
  --base-url "http://localhost:8000" `
  --judge-model "openai/gpt-5.4" `
  --query-concurrency 3 `
  --score-concurrency 3 `
  --top-k 10 `
  --chunk-top-k 10 `
  --max-context-chars 8000 `
  --overwrite `
  --model-provider-base-url "https://openrouter.ai/api/v1" `
  --model-provider-api-key "your_api_key_here" `
  --embedding-model "baai/bge-m3" `
  --entity-sim-threshold 0.7 `
  --relation-sim-threshold 0.7
```

The run command used for the evaluation using Gemma-4-31b:
```powershell
conda activate rawabit
python "evaluation\DHS LLW & Immigration Enforcement\scripts\run_thesis_evaluation.py" `
  --eval-dir "evaluation\DHS LLW & Immigration Enforcement" `
  --case-id a44a91c2-67c7-4dc7-8afd-9c9d7d6f7347 `
  --model-slug "gemma-4-31b-it" `
  --RQs 1,2 `
  --modes hybrid,naive `
  --run-queries `
  --base-url "http://localhost:8000" `
  --judge-model "openai/gpt-5.4" `
  --query-concurrency 3 `
  --score-concurrency 3 `
  --top-k 10 `
  --chunk-top-k 10 `
  --max-context-chars 8000 `
  --overwrite `
  --model-provider-base-url "https://openrouter.ai/api/v1" `
  --model-provider-api-key "your_api_key_here" `
  --embedding-model "baai/bge-m3" `
  --entity-sim-threshold 0.7 `
  --relation-sim-threshold 0.7
```

Pass `--model-provider-api-key` only for a local run if the key is not already available through `.env` or environment variables. Do not commit API keys.

### Evaluation Arguments

| Argument | Meaning | Values Used in the Thesis Runs |
| --- | --- | --- |
| `--eval-dir` | Evaluation folder containing `goldens/`, `results/`, and related assets. | `evaluation\DHS LLW & Immigration Enforcement` |
| `--case-id` | Rawabit case identifier to query and evaluate. | GPT: `09fb881d-a36f-4757-b671-cf95c4f2fc73`; Gemma: `a44a91c2-67c7-4dc7-8afd-9c9d7d6f7347` |
| `--model-slug` | Label used to identify the evaluated model/run in outputs and checkpoints. | `gpt-5.4-mini`; `gemma-4-31b-it` |
| `--RQs` | Research-question groups to evaluate. | `1,2` |
| `--modes` | Retrieval modes scored for each query. | `hybrid,naive` |
| `--run-queries` | Query the Rawabit backend before scoring, then cache the raw responses. | Enabled |
| `--base-url` | Local Rawabit backend URL. | `http://localhost:8000` |
| `--judge-model` | LLM judge used by the evaluation scorer. | `openai/gpt-5.4` |
| `--query-concurrency` | Number of concurrent Rawabit query requests. | `3` |
| `--score-concurrency` | Number of concurrent scoring workers. | `3` |
| `--top-k` | Graph/vector retrieval result limit passed to the Rawabit query API. | `10` |
| `--chunk-top-k` | Chunk retrieval result limit passed to the Rawabit query API. | `10` |
| `--max-context-chars` | Maximum retrieved-context characters retained per query for scoring. | `8000` |
| `--overwrite` | Clear existing checkpoints in the output directory before rerunning. | Enabled |
| `--model-provider-base-url` | OpenAI-compatible provider endpoint for evaluator model calls. | `https://openrouter.ai/api/v1` |
| `--model-provider-api-key` | Provider API key for local evaluator calls. Prefer `.env` or environment variables rather than command-line secrets. | Local credential, not published |
| `--embedding-model` | Embedding model used for semantic similarity scoring. | `baai/bge-m3` |
| `--entity-sim-threshold` | Similarity threshold for entity matching. | `0.7` |
| `--relation-sim-threshold` | Similarity threshold for relation matching. | `0.7` |

The evaluator writes query caches, checkpoints, scored CSVs, summaries, JSON score files, and a run manifest under:

```text
results/thesis_<timestamp>_<label>/
```

Use `--resume` to continue a partially completed run. Use `--query-ids` or `--only-query` for a small targeted reproduction.

## Published Results

For review, this repository includes the result folders under:

```text
results/
```
