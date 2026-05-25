# TorlakTag

Automatic morphosyntactic annotation pipeline for the 115 Torlak audio recordings transcripts. TorlakTag converts raw EXMARaLDA (EXB) transcription files into fully enriched CoNLL-U and TEI XML documents, with lemmatisation, UPOS, MTE XPOS, morphological features, speaker metadata, and ethnographic topic tags.

> **Anonymized supplementary repository.** Author-identifying information has been removed from this release for double-blind review. The citation placeholder below will be completed upon de-anonymization.

---

## Table of contents

1. [Repository structure](#repository-structure)
2. [Installation](#installation)
3. [Data access and input format](#data-access-and-input-format)
4. [Model weights](#model-weights)
5. [Running inference](#running-inference)
6. [Reproducing experiments](#reproducing-experiments)
7. [Pipeline overview](#pipeline-overview)
8. [Output formats](#output-formats)
9. [Compute and infrastructure](#compute-and-infrastructure)
10. [Intended use and limitations](#intended-use-and-limitations)
11. [Ethical and responsible use](#ethical-and-responsible-use)
12. [Citation](#citation)

---

## Repository structure

```
torlak-tag/
│
│  ── Inference & annotation pipeline ──────────────────────────────
├── TorlakTag_Inference_EXB2CoNLLU.ipynb   # Step 1: batch inference EXB → CoNLL-U
├── sentence_segmenter.py                  # Step 2: LLM-based sentence segmentation
├── run_segmenter_batch.py                 # Step 2: parallel batch runner
├── enrich_pred_conllu.py                  # Step 3: CoNLL-U enrichment + speaker metadata
├── run_enrich_batch.py                    # Step 3: parallel batch runner
├── add_topic_metadata.py                  # Step 4: ethnographic topic tag injection
├── conllu_to_tei.py                       # Step 5a: TEI XML generation
├── conllu_to_enriched_exb.py             # Step 5b: annotated EXB output (optional)
│
│  ── Training ─────────────────────────────────────────────────────
├── TorlakTag_Multirun_3models_EXB-2.ipynb # Train XLM-R / BERTić / RemBERT / ByT5
├── TorlakTag_LemmaOnly.ipynb              # Standalone ByT5-base lemmatization fine-tuning
│
│  ── Analysis & evaluation ────────────────────────────────────────
├── TorlakTag_FeatDifficulty_XLMRoberta2.ipynb  # Per-feature difficulty analysis
├── evaluate_with_gemini.py                # LLM-as-judge morphosyntactic evaluation
├── run_combined_eval.py                   # Multi-file evaluation report
│
│  ── Environment ──────────────────────────────────────────────────
└── requirements.txt
```

---

## Installation

**Requirements:** Python 3.10 or 3.11, CUDA 11.8+ (for GPU inference/training).

```bash
git clone https://anonymous.4open.science/r/torlak-tag-6008/
cd torlak-tag
pip install -r requirements.txt
```

Key packages and tested versions:

| Package | Version | Purpose |
|---|---|---|
| `torch` | ≥ 2.0.0 | Model training and inference |
| `transformers` | ≥ 4.40.0 | XLM-RoBERTa, RemBERT, BERTić, ByT5 |
| `datasets` | ≥ 2.18.0 | Dataset loading utilities |
| `sentencepiece` | ≥ 0.1.99 | ByT5 / mT5 tokenizers |
| `seqeval` | ≥ 1.2.2 | Sequence-labeling evaluation |
| `lxml` | ≥ 4.9.0 | EXB and TEI XML parsing |
| `google-generativeai` | ≥ 0.5.0 | Gemini API (sentence segmentation, evaluation) |
| `numpy` | ≥ 1.24.0 | Numerical utilities |

Training notebooks run on **Google Colab** with Google Drive for data storage. Inference scripts run locally or on any CUDA-capable machine.

---

## Data access and input format

The underlying corpora (EXB transcriptions, audio, and speaker metadata) are not included in this repository. Access is subject to the terms of the **Spoken Torlak Dialect Corpus** and **Spoken Lužnica Dialect Corpus**.

**Expected directory layout for inference:**

```
project_root/
├── TOR_C_EXB_transcripts/       # raw .exb transcription files
│   ├── TOR_C_0001.exb
│   └── ...
├── speaker_metadata/
│   └── TOR_C_speeches.csv       # speaker ID, age, gender, education, location
├── Torlak_tales-main/           # ethnographic topic markdown files
│   ├── 01_topic_clusters/
│   └── 01b_fine_customs/
└── mte2ud_output.txt            # MTE → UD UPOS/features mapping table
```

**EXB input format:** Standard EXMARaLDA XML. Each transcription tier maps speaker abbreviations to timed token sequences. The pipeline reads time-aligned utterances and tokenises them following the MTE tokenisation conventions used in the corpus.

**Training data format (`dataset/tor_train.tsv`):** Three-column TSV with blank-line sentence boundaries:

```
form    lemma    xpos
ja      ja       Pp1msn
sam     biti     Vcr1s
...
```

**Dataset split sizes:**

| Split | Tokens |
|---|---|
| Train | 52,349 |
| Dev | 6,486 |
| Test | 6,424 |
| **Total** | **65,259** |

The full annotated corpus (115 transcripts) contains **738,549 tokens** (677,840 lexical words).

---

## Model weights

Fine-tuned model weights are hosted externally. Download and place them under a `models/` directory before running inference.

| Model | Task | Download |
|---|---|---|
| XLM-RoBERTa-large (XPOS/UPOS/FEATS) | XPOS + UPOS + morphological features | *(link to be added)* |
| ByT5-base (Lemma) | Lemmatisation | *(link to be added)* |

**Expected directory structure after download:**

```
models/
├── xlmr_xpos/
│   ├── best_model.pt
│   ├── config.json
│   ├── tokenizer_config.json
│   └── label_maps.json          # xpos2id, upos2id, feat2id maps
└── byt5_lemma/
    ├── config.json
    ├── tokenizer_config.json
    └── pytorch_model.bin
```

**Verifying a download:**

```bash
python - <<'EOF'
import torch
ckpt = torch.load("models/xlmr_xpos/best_model.pt", map_location="cpu")
print("Keys:", list(ckpt.keys())[:5])
print("OK")
EOF
```

---

## Running inference

### Step 1 — EXB → raw CoNLL-U

Open `TorlakTag_Inference_EXB2CoNLLU.ipynb` in Colab or Jupyter. Set the paths in the CONFIG cell:

```python
MODEL_DIR  = Path("models/xlmr_xpos")
LEMMA_DIR  = Path("models/byt5_lemma")
EXB_DIR    = Path("TOR_C_EXB_transcripts")
OUTPUT_DIR = Path("final_conllu")
```

Run all cells. The notebook is resumable — already-completed files are skipped automatically.

### Step 2 — Sentence segmentation

```bash
# Single file
python sentence_segmenter.py final_conllu/TOR_C_0001.conllu \
    --output-dir sentence_segmented_conllu/ \
    --model gemini-2.5-flash \
    --api-key YOUR_GEMINI_KEY

# All files in parallel
python run_segmenter_batch.py --workers 5 --api-key YOUR_GEMINI_KEY
```

### Step 3 — CoNLL-U enrichment

```bash
# Single file
python enrich_pred_conllu.py \
    --corrected sentence_segmented_conllu/TOR_C_0001.corrected.conllup \
    --pred      final_conllu/TOR_C_0001.conllu \
    --speeches  speaker_metadata/TOR_C_speeches.csv \
    --output    final_conllu_relemmad/TOR_C_0001.enriched.conllu

# All files in parallel
python run_enrich_batch.py --workers 8
```

### Step 4 — Topic tagging

```bash
python add_topic_metadata.py \
    --input    final_conllu_relemmad/ \
    --output   topic_tagged_relemmad/ \
    --clusters Torlak_tales-main/01_topic_clusters/ \
    --customs  Torlak_tales-main/01b_fine_customs/
```

Safe to re-run: existing topic fields are stripped and rewritten each time.

### Step 5a — TEI XML

```bash
python conllu_to_tei.py \
    --conllu  topic_tagged_relemmad/ \
    --old-tei "TOR_C_TEI_transcripts/" \
    --output  tei_output/
```

### Step 5b — Enriched EXB (optional)

```bash
python conllu_to_enriched_exb.py \
    --conllu topic_tagged_relemmad/ \
    --exb    TOR_C_EXB_transcripts/ \
    --output enriched_exb/
```

---

## Reproducing experiments

### Model backbones compared

No new model architecture is introduced. The project fine-tunes existing pretrained encoders with a shared token-level linear classification head per task (LEMMA, UPOS, FEATS, XPOS):

| Backbone | HuggingFace ID | Role |
|---|---|---|
| XLM-RoBERTa-large | `FacebookAI/xlm-roberta-large` | Primary tagger (best results) |
| BERTić | `classla/bcms-bertic` | Comparison — BCMS-specialized |
| RemBERT | `google/rembert` | Comparison — large multilingual |
| ByT5-large | `google/byt5-large` | Comparison — byte-level encoder |
| ByT5-base | `google/byt5-base` | Lemmatisation — seq2seq generation |

### Hyperparameters

| Hyperparameter | XLM-R large | BERTić | RemBERT | ByT5-large |
|---|---|---|---|---|
| Max epochs | 30 | 30 | 30 | 30 |
| Batch size | 24 | 64 | 32 | 8 |
| Gradient accumulation | 4 | 2 | 2 | 8 |
| Effective batch size | 96 | 128 | 64 | 64 |
| Learning rate | 1e-5 | 2e-5 | 2e-5 | 1e-5 |
| Weight decay | 0.01 | 0.01 | 0.01 | 0.01 |
| Max token length | 16 | 16 | 16 | 64 (bytes) |
| Warmup ratio | 0.06 | 0.06 | 0.06 | 0.06 |
| LR schedule | cosine | cosine | cosine | cosine |
| Early stopping patience | 4 epochs | 4 epochs | 4 epochs | 4 epochs |
| Early stopping metric | dev joint acc. | dev joint acc. | dev joint acc. | dev joint acc. |
| Gradient clipping | 1.0 | 1.0 | 1.0 | 1.0 |
| Encoder frozen (epoch 1) | yes | yes | yes | yes |
| Mixed precision | fp16 | fp16 | fp16 | fp32 |
| Random seed | 13 | 13 | 13 | 13 |

**Optimizer:** AdamW (PyTorch default betas: β₁=0.9, β₂=0.999, ε=1e-8).  
**LR schedule:** cosine decay with linear warm-up (`get_cosine_schedule_with_warmup` from `transformers`).  
**Early stopping:** joint development-set accuracy — all four task predictions simultaneously correct for a token.

**Lemmatisation (ByT5-base seq2seq):**
- Input format: `lemmatize: {3 left tokens} [X] {form} [/X] {3 right tokens}`
- Target: lemma string (byte-level generation handles OOV dialect forms)
- Context window: 3 tokens left and right of the target

### Evaluation metrics

- UPOS accuracy (Universal Dependencies)
- XPOS accuracy (MulText-East tagset)
- Morphological feature accuracy — per-feature and joint
- Lemma exact-match accuracy
- LLM-as-judge evaluation via Gemini (`evaluate_with_gemini.py`)

### Running evaluation

```bash
# LLM-as-judge on a single CoNLL-U file
python evaluate_with_gemini.py --api-key YOUR_GEMINI_KEY

# Combined report across 4 held-out files
python run_combined_eval.py
```

Results: `evaluate_with_gemini_results.tsv`, `evaluate_with_gemini_summary.txt`.

---

## Pipeline overview

```
EXB transcripts
      │
      ▼  Step 1 — Model inference  (TorlakTag_Inference_EXB2CoNLLU.ipynb)
  raw CoNLL-U  (final_conllu/)
      │
      ▼  Step 2 — LLM sentence segmentation  (sentence_segmenter.py)
  segmented CoNLL-U  (sentence_segmented_conllu/)
      │
      ▼  Step 3 — Enrichment: speaker metadata + timing + provenance  (enrich_pred_conllu.py)
  enriched CoNLL-U  (final_conllu_relemmad/)
      │
      ▼  Step 4 — Topic tagging  (add_topic_metadata.py)
  topic-tagged CoNLL-U  (topic_tagged_relemmad/)
      │
      ├──▶  Step 5a — TEI XML  (conllu_to_tei.py)
      └──▶  Step 5b — Enriched EXB  (conllu_to_enriched_exb.py)
```

---

## Output formats

### Enriched CoNLL-U

Standard 10-column CoNLL-U with sentence-level metadata:

```
# sent_id           = TOR_C_0001-r000001
# text              = idem doma
# speaker           = TIM_SPK_0001
# speaker_abbr      = OS_1
# speaker_age       = 72
# speaker_gender    = f
# speaker_education = no education
# geographical_coordinates = 43.3244 N, 22.0711 E
# start_time        = 12.34
# end_time          = 14.56
# timing_precision  = exact
# EXB_sources       = TOR_C_0001-s0003
# EXB_token_map     = 1:TOR_C_0001-s0003-w1 2:TOR_C_0001-s0003-w2
# topic_clusters    = hearth_home_house
# topic_clusters_sr = Kuća, ognjište, život u domu
# fine_customs      = _
# fine_customs_topic = _
1  idem  ići   VERB  Vmr1s  Mood=Ind|Number=Sing|Person=1|Tense=Pres|VerbForm=Fin  _  _  _  OrigUttID=...|UttEnd=yes
2  doma  doma  ADV   Rgp    Degree=Pos  _  _  _  OrigUttID=...
```

### TEI XML

Spoken-corpus TEI P5 format:
- `<teiHeader>` — recording metadata, speaker demographics, tag-usage statistics, and topic taxonomies (`<classDecl>`)
- `<timeline>` — one `<when>` per unique timestamp, anchored to the `.wav` file
- `<u who="#TIM_SPK_0001" start="..." end="..." ana="#topc:hearth_home_house">` — one utterance per CoNLL-U sentence
- `<w lemma="ići" pos="VERB" ana="mte:Vmr1s" msd="Mood=Ind|...">idem</w>`
- `<pause/>`, `<vocal type="laugh"/>`, `<unclear/>`, `<incident type="noise"/>` for non-lexical events

### Topic metadata fields

| Comment key | Content |
|---|---|
| `topic_clusters` | `\|`-separated English topic slugs |
| `topic_clusters_sr` | Parallel Serbian topic labels |
| `fine_customs` | `\|`-separated fine-grained custom slugs |
| `fine_customs_topic` | Parent topic for each fine custom |
| `_` | No topic match found for this sentence |

---

## Compute and infrastructure

All training was run on **Google Colab Pro** (A100 or L4 GPU, 24 GB VRAM recommended). Inference and pipeline scripts run on CPU or any CUDA-capable GPU.

| Stage | Hardware | Approximate time |
|---|---|---|
| Training (one backbone, 30 epochs) | A100 40 GB | 2–4 hours |
| Batch inference (115 EXB files) | L4 24 GB | 1–2 hours |
| Sentence segmentation (115 files) | CPU + Gemini API | 3–5 hours |
| Enrichment (115 files, 8 workers) | CPU | ~15 minutes |
| Topic tagging + TEI export | CPU | ~5 minutes |

> Exact GPU-hour counts were not systematically recorded. The figures above are practical estimates; actual time will vary with GPU model, batch size, and Gemini API quota.

**Disk:** model checkpoints ~3 GB per backbone; full annotated corpus ~500 MB (CoNLL-U) + ~800 MB (TEI XML).

---

## Intended use and limitations

**Intended use:**
- Research in low-resource NLP, morphosyntactic tagging, and spoken language processing
- Language documentation and corpus linguistics for South Slavic dialects
- Reproducibility of experiments reported in the accompanying paper

**Limitations:**
- Models are trained on Torlak and Lužnica dialect data only; performance on standard Serbian or other South Slavic varieties is not evaluated
- Sentence segmentation relies on the Gemini API and may vary across model versions
- Topic tags derive from time-coded ethnographic excerpts and do not cover all recordings uniformly
- LLM-as-judge evaluation provides an approximate quality signal; it is not a replacement for expert human annotation

---

## Ethical and responsible use

- These artifacts are intended for **research, language documentation, and reproducibility** only
- They must **not** be used for speaker identification, speaker profiling, or any surveillance application
- They must **not** be treated as an authoritative linguistic reference for Torlak or Lužnica
- Speaker metadata (age, gender, education, location) is included solely to support linguistic analysis and corpus querying; it must be handled in accordance with applicable data protection principles
- All use is subject to the access, licensing, and ethical conditions of the underlying **Spoken Torlak Dialect Corpus** and **Spoken Lužnica Dialect Corpus**; contact the corpus holders for data access

---

## Citation

If you use this pipeline or the annotated corpus, please cite:

```bibtex
@inproceedings{anonymized2025torlaktag,
  title     = {TorlakTag: Automatic Morphosyntactic Annotation of Torlak Dialect Spoken Corpora},
  author    = {Anonymized for review},
  booktitle = {Anonymized for review},
  year      = {2025}
}
```

> This entry will be updated with full author and venue information upon de-anonymization.
