# TorlakTag

**Automatic morphosyntactic annotation pipeline for Torlak and Lužnica dialect spoken corpora.**

TorlakTag converts raw EXB (EXMARaLDA) transcription files into fully enriched, TEI-XML-formatted spoken-corpus documents. It runs fine-tuned XLM-RoBERTa models (lemmatisation, UPOS + XPOS + features) over dialectal South Slavic speech data, then layers sentence segmentation, speaker metadata, and ethnographic topic tags on top.

---

## Pipeline overview

```
EXB transcripts
      │
      ▼  (1) Model inference
  raw CoNLL-U  (final_conllu/)
      │
      ▼  (2) LLM sentence segmentation
  segmented CoNLL-U  (sentence_segmented_conllu/)
      │
      ▼  (3) CoNLL-U enrichment  (speaker metadata, timing, token provenance)
  enriched CoNLL-U  (final_conllu_relemmad/)
      │
      ▼  (4) Topic tagging
  topic-tagged CoNLL-U  (topic_tagged_relemmad/)
      │
      ├──▶  (5a) TEI XML  (tei_output_relemmad/)
      └──▶  (5b) Enriched EXB
```

---

## Step 1 — Model inference: EXB → CoNLL-U

**Training notebook:** `TorlakTag_Multirun_3models_EXB-2.ipynb`  
**Inference notebook:** `TorlakTag_Inference_EXB2CoNLLU.ipynb`

Two fine-tuned XLM-RoBERTa models are run sequentially over each EXB file:

| Model | Task | Output column(s) |
|---|---|---|
| Lemma model | Lemmatisation | `LEMMA` |
| XPOS model | MulText-East XPOS + UPOS + morphological features | `UPOS`, `XPOS`, `FEATS` |

The training notebook (`TorlakTag_Multirun_3models_EXB-2.ipynb`) compares multiple transformer backbones and saves the best checkpoint. The inference notebook (`TorlakTag_Inference_EXB2CoNLLU.ipynb`) then runs batch inference over all `.exb` files and writes one `.conllu` per recording to `final_conllu/`. It is resumable — already-completed files are skipped automatically.

### Model weights

The fine-tuned model weights are hosted on Google Drive. Download and place them in the project root before running inference:

| Model | Drive folder |
|---|---|
| Lemma model | *(link to be added)* |
| UPOS + XPOS + Features model | *(link to be added)* |

---

## Step 2 — Sentence segmentation

**Script:** `sentence_segmenter.py`  
**Batch runner:** `run_segmenter_batch.py`

The raw CoNLL-U output from Step 1 groups tokens by EXB utterance, not by grammatical sentence. This step uses the Gemini API to detect sentence boundaries and writes `.corrected.conllup` files to `sentence_segmented_conllu/`.

```bash
# Single file
python sentence_segmenter.py final_conllu/TOR_C_0001.conllu \
    --output-dir sentence_segmented_conllu/ \
    --model gemini-2.5-flash \
    --api-key YOUR_GEMINI_KEY

# All files in parallel (5 files × 20 Gemini calls)
python run_segmenter_batch.py --workers 5 --api-key YOUR_GEMINI_KEY
```

---

## Step 3 — CoNLL-U enrichment

**Script:** `enrich_pred_conllu.py`  
**Batch runner:** `run_enrich_batch.py`

Aligns sentence-segmented predictions back against their source utterances and adds speaker metadata, timing information, and per-token provenance. Writes `.enriched.conllu` to `final_conllu_relemmad/`.

```bash
# Single file
python enrich_pred_conllu.py \
    --corrected sentence_segmented_conllu/TOR_C_0001.corrected.conllup \
    --pred      final_conllu/TOR_C_0001.conllu \
    --speeches  speaker_metadata/TOR_C_speeches.csv \
    --output    final_conllu_relemmad/TOR_C_0001.enriched.conllu

# All files in parallel (8 workers)
python run_enrich_batch.py --workers 8
```

Each sentence in the output carries:

```
# sent_id          = TOR_C_0001-r000001
# text             = …
# speaker          = TIM_SPK_0001
# speaker_abbr     = OS_1
# speaker_age      = 72
# speaker_gender   = f
# speaker_education = no education
# start_time       = 12.34
# end_time         = 14.56
# timing_precision = exact
# EXB_sources      = TOR_C_0001-s0003|TOR_C_0001-s0004
# EXB_token_map    = 1:TOR_C_0001-s0003-w1 …
```

---

## Step 4 — Topic tagging

**Script:** `add_topic_metadata.py`

Matches each sentence's `[start_time, end_time]` window against time-coded ethnographic excerpts in the `Torlak_tales-main/` markdown files, adding four metadata fields per sentence:

```
# topic_clusters    = hearth_home_house|war_turks_bulgars_wwii
# topic_clusters_sr = Kuća, ognjište, život u domu|Ratovi, Turci, Bugari, Nemci, hajduci
# fine_customs      = bugari_bugarska_okup
# fine_customs_topic = Ratovi, Turci, Bugari, partizani
```

```bash
python add_topic_metadata.py \
    --input    final_conllu_relemmad/ \
    --output   topic_tagged_relemmad/ \
    --clusters Torlak_tales-main/01_topic_clusters/ \
    --customs  Torlak_tales-main/01b_fine_customs/
```

Safe to re-run: existing topic fields are stripped and rewritten on each pass.

---

## Step 5a — TEI XML output

**Script:** `conllu_to_tei.py`

Converts topic-tagged enriched CoNLL-U to TEI XML (spoken corpus format), recovering stress marks from the original TEI transcriptions where available.

```bash
python conllu_to_tei.py \
    --conllu  topic_tagged_relemmad/ \
    --old-tei "TOR_C_TEI_transcripts 2/" \
    --output  tei_output_relemmad/
```

Each TEI file contains:
- `<teiHeader>` with recording metadata, speaker demographics, tag-usage statistics, and topic/custom taxonomies
- `<timeline>` anchoring every utterance to the audio file
- `<u>` elements with `@start` / `@end` / `@who` / `@ana` attributes (topics encoded as `#topc:slug` / `#cust:slug`)
- `<w>` elements carrying `@lemma`, `@pos` (UPOS), `@ana` (MTE XPOS), and `@msd` (features)
- `<pause>`, `<vocal>`, `<unclear>`, `<incident>` elements for non-lexical events

---

## Step 5b — Enriched EXB output (optional)

**Script:** `conllu_to_enriched_exb.py`

Writes LEMMA, UPOS, XPOS, and topic-tag annotation tiers back into the original EXB files.

```bash
python conllu_to_enriched_exb.py \
    --conllu topic_tagged_relemmad/ \
    --exb    TOR_C_EXB_transcripts/ \
    --output enriched_exb/
```

---

## Training

| Notebook | Purpose |
|---|---|
| `TorlakTag_Multirun_3models_EXB-2.ipynb` | Train XLM-RoBERTa models (lemma, UPOS+XPOS+features) |
| `TorlakTag_FeatDifficulty_XLMRoberta2.ipynb` | Per-feature difficulty analysis |

---

## Evaluation

```bash
# LLM-as-judge evaluation via Gemini (single file)
python evaluate_with_gemini.py --api-key YOUR_KEY

# Combined 4-file evaluation report
python run_combined_eval.py
```

Results are written to `evaluate_with_gemini_results.tsv` and `evaluate_with_gemini_summary.txt`.

---

## Repository structure

```
torlak-tag/
├── sentence_segmenter.py               # Step 2: sentence boundary detection (Gemini)
├── run_segmenter_batch.py              # Step 2: batch runner
├── enrich_pred_conllu.py               # Step 3: CoNLL-U enrichment
├── run_enrich_batch.py                 # Step 3: batch runner
├── add_topic_metadata.py               # Step 4: topic tag injection
├── conllu_to_tei.py                    # Step 5a: TEI XML generation
├── conllu_to_enriched_exb.py          # Step 5b: enriched EXB output
├── augment_local.py                    # Training: Gemini data augmentation (local)
├── evaluate_with_gemini.py             # Evaluation: LLM-as-judge scorer
├── run_combined_eval.py                # Evaluation: combined multi-file report
│
├── TorlakTag_Multirun_3models_EXB-2.ipynb    # Training: multi-model comparison
├── TorlakTag_Inference_EXB2CoNLLU.ipynb      # Step 1: batch inference (EXB → CoNLL-U)
├── TorlakTag_FeatDifficulty_XLMRoberta2.ipynb # Analysis: per-feature difficulty
├── finetune_lemma.ipynb                        # Training: lemma model
├── finetune_xpos_2.ipynb                       # Training: XPOS model
└── augment_xpos.ipynb                          # Training: data augmentation (Colab)
```

---

## Dependencies

```bash
pip install transformers torch datasets conllu google-generativeai lxml
```

- Python 3.10+
- `transformers` ≥ 4.40 (XLM-RoBERTa inference and fine-tuning)
- `torch` ≥ 2.0
- `google-generativeai` (sentence segmentation and evaluation via Gemini API)
- Standard library: `xml.etree.ElementTree`, `pathlib`, `concurrent.futures`

---

## Data

The corpus covers **TOR_C** (Torlak dialect, 96 recordings) and **LUZ_C** (Lužnica dialect, 19 recordings). Raw EXB transcriptions and audio files are not included in this repository. Contact the corpus team for access.

Topic metadata is drawn from `Torlak_tales-main/`, a curated set of ethnographic topic-cluster and fine-custom markdown files that map time ranges in each recording to semantic topics (in both English slugs and Serbian labels).

---

## Citation

If you use this pipeline or the corpus, please cite:

> Vuković, T. (2025). *TorlakTag: Automatic morphosyntactic annotation of Torlak dialect spoken corpora*. Department of Slavonic Languages and Literatures, University of Zurich.
