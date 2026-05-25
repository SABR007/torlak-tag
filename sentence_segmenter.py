#!/usr/bin/env python3
"""
sentence_segmenter.py
---------------------
Local equivalent of sentence_.ipynb (no Google Colab dependency).

Usage:
    python sentence_segmenter.py INPUT.conllu [--api-key KEY] [--model MODEL]

Outputs (next to INPUT):
    INPUT.sentences.txt
    INPUT.sentences.json
    INPUT.corrected.conllup

Rules:
  - Tokens from different speakers are NEVER merged into one sentence.
  - Utterance gaps >= HARD_BOUNDARY_GAP (1.0 s) are ALWAYS sentence boundaries —
    Gemini never sees tokens across such a gap.
  - Within a same-speaker run, soft gaps (< 1 s) may be merged by Gemini.
    If merged, a pause marker is inserted at the boundary:
        < 0.5 s  →  ·
        0.5–1 s  →  ··
    Pause marker rows carry MISC = IsPause=yes|PauseDur=<seconds>.
"""

import os
import sys
import json
import re
import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from google import genai
from google.genai import types

# ── Config ────────────────────────────────────────────────────────────────────

MODEL_NAME        = "gemini-2.5-pro"
MAX_RETRIES       = 3
RETRY_DELAY       = 5    # seconds between retries
HARD_BOUNDARY_GAP = 2.0  # seconds — gaps >= this are ALWAYS mandatory sentence boundaries;
                          #   Gemini never sees tokens across such a gap

# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class TokenEntry:
    cols:  List[str]        # full 10 CoNLL-U columns
    token: str              # FORM
    meta:  Dict[str, str]   # sentence-level metadata (# key = value)


@dataclass
class UttBoundary:
    """Pause between two consecutive EXB utterances from the same speaker."""
    after_tok_idx: int    # 0-based index in speaker run; last token of prev utt
    gap_sec:       float  # next_start − prev_end in seconds
    is_hard:       bool   # True if gap >= HARD_BOUNDARY_GAP → mandatory sentence boundary


# ── Parsers ───────────────────────────────────────────────────────────────────

def parse_conllu_like(path: str) -> Tuple[List[TokenEntry], List[str]]:
    """Parse CoNLL-U/-UP file; attach per-sentence metadata to each token."""
    token_entries: List[TokenEntry] = []
    all_tokens:    List[str]        = []
    cur_meta:      Dict[str, str]   = {}

    with open(path, encoding="utf-8") as fh:
        for raw in fh:
            line = raw.rstrip("\n")
            if not line:
                cur_meta = {}
                continue
            if line.startswith("#"):
                m = re.match(r"#\s*([^=]+?)\s*=\s*(.*)\s*$", line)
                if m:
                    cur_meta[m.group(1).strip()] = m.group(2).strip()
                continue
            cols = line.split("\t")
            if len(cols) < 2:
                continue
            tok_id = cols[0]
            if "-" in tok_id or "." in tok_id:
                continue   # skip MWT / empty nodes
            form = cols[1]
            token_entries.append(TokenEntry(cols=cols, token=form, meta=dict(cur_meta)))
            all_tokens.append(form)

    return token_entries, all_tokens


# ── Speaker runs + pause utilities ────────────────────────────────────────────

def _safe_float(x: Optional[str]) -> Optional[float]:
    try:
        return float(x) if x is not None else None
    except (ValueError, TypeError):
        return None


def pause_marker(gap_sec: float) -> Optional[str]:
    """Middle-dot pause marker, or None for non-positive gaps."""
    if gap_sec <= 0:
        return None
    elif gap_sec < 0.5:
        return "\u00b7"            # ·
    elif gap_sec < 1.0:
        return "\u00b7\u00b7"      # ··
    else:
        return "\u00b7\u00b7\u00b7"  # ···


def build_speaker_runs(token_entries: List[TokenEntry]) -> List[Dict]:
    """
    Split flat token list into consecutive same-speaker runs.
    Within each run, EXB utterance boundaries are detected by changes in
    (start_time, end_time) — values shared by all tokens of one EXB utterance.
    """
    if not token_entries:
        return []

    runs: List[Dict] = []

    def flush(run_toks: List[TokenEntry]) -> None:
        if not run_toks:
            return
        speaker    = run_toks[0].meta.get("speaker_abbr", "_")
        boundaries: List[UttBoundary] = []
        prev_key   = (run_toks[0].meta.get("start_time"),
                      run_toks[0].meta.get("end_time"))

        for i in range(1, len(run_toks)):
            cur_key = (run_toks[i].meta.get("start_time"),
                       run_toks[i].meta.get("end_time"))
            if cur_key != prev_key:
                prev_end   = _safe_float(run_toks[i-1].meta.get("end_time"))   or 0.0
                next_start = _safe_float(run_toks[i].meta.get("start_time"))   or 0.0
                gap = next_start - prev_end
                boundaries.append(UttBoundary(
                    after_tok_idx = i - 1,
                    gap_sec       = gap,
                    is_hard       = gap >= HARD_BOUNDARY_GAP,
                ))
                prev_key = cur_key

        runs.append({"speaker": speaker, "tokens": run_toks,
                     "boundaries": boundaries})

    current: List[TokenEntry] = []
    current_spk: Optional[str] = None

    for te in token_entries:
        spk = te.meta.get("speaker_abbr", "_")
        if spk != current_spk:
            flush(current)
            current     = []
            current_spk = spk
        current.append(te)

    flush(current)
    return runs


# ── Hard boundary splitting ───────────────────────────────────────────────────

def split_run_at_hard_boundaries(run: Dict) -> List[Dict]:
    """
    Split a speaker run into sub-runs at every hard boundary (gap >= HARD_BOUNDARY_GAP).
    Each sub-run contains only the tokens and soft boundaries within that segment.
    Re-indexes soft boundaries to be local (0-based) within the sub-run.

    Hard boundaries become natural sentence boundaries — Gemini never sees tokens
    across them, so no pause marker is inserted there.
    """
    tokens     = run["tokens"]
    boundaries = run["boundaries"]
    speaker    = run["speaker"]

    hard_cuts = sorted(b.after_tok_idx for b in boundaries if b.is_hard)

    if not hard_cuts:
        return [run]

    # Build (start, end-exclusive) pairs for each sub-run
    starts = [0]           + [c + 1 for c in hard_cuts]
    ends   = [c + 1 for c in hard_cuts] + [len(tokens)]

    sub_runs: List[Dict] = []
    for seg_start, seg_end in zip(starts, ends):
        seg_toks = tokens[seg_start:seg_end]
        if not seg_toks:
            continue

        # Keep only soft boundaries strictly internal to this sub-run
        # (both left and right token must be inside [seg_start, seg_end-1])
        seg_bounds = [
            UttBoundary(
                after_tok_idx = b.after_tok_idx - seg_start,
                gap_sec       = b.gap_sec,
                is_hard       = False,
            )
            for b in boundaries
            if not b.is_hard
            and seg_start <= b.after_tok_idx
            and b.after_tok_idx + 1 < seg_end   # right token still inside segment
        ]

        sub_runs.append({"speaker": speaker, "tokens": seg_toks, "boundaries": seg_bounds})

    return sub_runs


# ── Gemini segmentation ───────────────────────────────────────────────────────

def _extract_json(text: str) -> Any:
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    for pat in [r"(\{.*\})", r"(\[.*\])"]:
        m = re.search(pat, text, flags=re.DOTALL)
        if m:
            try:
                return json.loads(m.group(1))
            except Exception:
                pass
    raise ValueError(f"Could not parse JSON:\n{text[:300]}")


def _build_utterances(sub_run: Dict) -> List[Dict]:
    """
    Convert a sub-run into a list of utterance dicts:
      id:            int
      text:          str   (space-joined tokens)
      tokens:        List[str]
      gap_after_sec: float | None   (gap to the NEXT utterance; None for the last)
    """
    tokens     = sub_run["tokens"]      # List[TokenEntry]
    boundaries = sub_run["boundaries"]  # List[UttBoundary], all soft (is_hard=False)

    if not boundaries:
        return [{"id": 0,
                 "text": " ".join(te.token for te in tokens),
                 "tokens": [te.token for te in tokens],
                 "gap_after_sec": None}]

    cuts   = [b.after_tok_idx for b in boundaries]
    starts = [0] + [c + 1 for c in cuts]
    ends   = [c + 1 for c in cuts] + [len(tokens)]

    utts = []
    for i, (s, e) in enumerate(zip(starts, ends)):
        toks = [te.token for te in tokens[s:e]]
        gap  = boundaries[i].gap_sec if i < len(boundaries) else None
        utts.append({"id": i, "text": " ".join(toks),
                     "tokens": toks, "gap_after_sec": gap})
    return utts


def segment_utterances(client, sub_run: Dict,
                       model_name: str = MODEL_NAME) -> List[List[str]]:
    """
    Semantic sentence grouping via Gemini.

    Instead of sending raw tokens for Gemini to split, we send the EXB utterances
    as pre-segmented units (with their inter-utterance gap durations) and ask
    Gemini to GROUP consecutive utterances into sentences.

    This lets Gemini read and understand each utterance in full before deciding
    whether it continues or completes the previous one.

    Token preservation is guaranteed by construction: we just concatenate the
    tokens of the utterances in each group — no token manipulation by Gemini.

    If Gemini merges utterances across a gap:
      - gap < 0.5 s  → · marker
      - 0.5–1 s      → ·· marker
      - 1–2 s        → ··· marker
    (inserted by write_corrected_conllup via boundary_map)
    """
    utts       = _build_utterances(sub_run)
    all_tokens = [te.token for te in sub_run["tokens"]]

    if len(utts) == 1:
        return [all_tokens]

    n = len(utts)
    prompt = {
        "task": (
            "You are segmenting spoken Torlak/BCS dialect speech into sentences. "
            "The utterances below are consecutive speech segments from the SAME speaker, "
            "each separated by a short pause (gap_after_sec). "
            "Decide which consecutive utterances belong to the SAME sentence "
            "and which start a NEW sentence."
        ),
        "utterances": [
            {
                "id": u["id"],
                "text": u["text"],
                **({"gap_after_sec": round(u["gap_after_sec"], 2)}
                   if u["gap_after_sec"] is not None else {}),
            }
            for u in utts
        ],
        "instructions": [
            "Read and understand each utterance before deciding.",
            "Two utterances belong to the SAME sentence when the second one "
            "syntactically completes or continues the first "
            "(e.g. a subordinate clause, a complement, a conjoined VP sharing the subject).",
            "Start a NEW sentence when an utterance introduces a new topic or "
            "is grammatically complete on its own.",
            "Use gap_after_sec as a cue: a longer gap makes a sentence boundary more likely, "
            "but semantics and grammar take priority.",
            "Never split a clause mid-way (e.g. do not separate a verb from its complement).",
            f"Return ONLY valid JSON: {{\"groups\": [[0], [1, 2], ...]}}",
            f"Every ID from 0 to {n - 1} must appear in exactly one group, in ascending order.",
        ],
        "output_format": {"groups": [[0], [1, 2]]},
    }

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = client.models.generate_content(
                model=model_name,
                contents=json.dumps(prompt, ensure_ascii=False),
                config=types.GenerateContentConfig(
                    temperature=0,
                    response_mime_type="application/json",
                ),
            )
            data   = _extract_json(resp.text)
            groups = data.get("groups", [])
            flat   = [uid for g in groups for uid in g]
            if flat == list(range(n)):
                # Reconstruct token lists by concatenating utterances in each group
                return [
                    [tok for uid in group for tok in utts[uid]["tokens"]]
                    for group in groups
                ]
            prompt["instructions"].insert(
                0,
                f"RETRY {attempt}: groups must contain every ID 0..{n - 1} "
                "exactly once in ascending order. Fix it.",
            )
        except Exception as e:
            print(f"  ⚠ attempt {attempt} error: {e}", file=sys.stderr)
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY * attempt)

    # Fallback: each utterance is its own sentence
    print("  ⚠ Gemini grouping failed — treating each utterance as a sentence.",
          file=sys.stderr)
    return [u["tokens"] for u in utts]


def segment_all_runs(client, runs: List[Dict],
                     model_name: str = MODEL_NAME) -> List[List[List[str]]]:
    """
    Segment every speaker run independently.

    Each run is first split at hard boundaries (gap >= HARD_BOUNDARY_GAP).
    Within each resulting sub-run, EXB utterances are sent to Gemini as named
    units (with gap durations) so it can understand each before grouping them
    into sentences.

    Returns sentences_per_run[i] = flat list of sentences for runs[i],
    preserving the 1-to-1 correspondence with `runs` so that
    write_corrected_conllup can use the original run boundaries unchanged.
    """
    sentences_per_run: List[List[List[str]]] = []
    total = sum(len(r["tokens"]) for r in runs)
    done  = 0

    for ri, run in enumerate(runs):
        sub_runs = split_run_at_hard_boundaries(run)
        n_hard   = len(sub_runs) - 1
        n_soft   = len(run["boundaries"]) - n_hard
        print(f"Run {ri+1}/{len(runs)}  speaker={run['speaker']}  "
              f"tokens={len(run['tokens'])}  "
              f"hard_boundaries={n_hard}  soft_boundaries={n_soft}")

        run_sents: List[List[str]] = []
        for sub in sub_runs:
            if not sub["tokens"]:
                continue
            if len(sub["boundaries"]) == 0:
                # Single EXB utterance — no grouping question, bypass Gemini
                run_sents.append([te.token for te in sub["tokens"]])
            else:
                # Multiple utterances — ask Gemini to group semantically
                sents = segment_utterances(client, sub, model_name)
                run_sents.extend(sents)

        sentences_per_run.append(run_sents)
        done += len(run["tokens"])
        print(f"  → {len(run_sents)} sentence(s)  ({done}/{total} tokens done)")

    return sentences_per_run


# ── Writers ───────────────────────────────────────────────────────────────────

def _all_sentences(sentences_per_run: List[List[List[str]]]) -> List[List[str]]:
    return [s for run_sents in sentences_per_run for s in run_sents]


def write_sentences_txt(sentences_per_run, out_path: str) -> None:
    with open(out_path, "w", encoding="utf-8") as fh:
        for s in _all_sentences(sentences_per_run):
            fh.write(" ".join(s) + "\n")


def write_sentences_json(sentences_per_run, out_path: str) -> None:
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump({"sentences": _all_sentences(sentences_per_run)},
                  fh, ensure_ascii=False, indent=2)


def _pause_row(marker: str, gap_sec: float, tok_id: int) -> str:
    misc = f"IsPause=yes|PauseDur={gap_sec:.3f}"
    return "\t".join([str(tok_id), marker, marker, "PUNCT",
                      "_", "_", "_", "_", "_", misc])


def write_corrected_conllup(runs: List[Dict],
                             sentences_per_run: List[List[List[str]]],
                             out_path: str,
                             base_id: str) -> None:
    """
    Write re-segmented CoNLL-U:
    - Sentence boundaries strictly within speaker runs (no cross-speaker merging).
    - Pause markers (·/··/···) inserted as real token rows at EXB utterance
      boundaries that fall *inside* a merged sentence.
    """
    s_i = 0
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write("# global.columns = ID FORM LEMMA UPOS XPOS FEATS HEAD DEPREL DEPS MISC\n\n")

        for run, run_sentences in zip(runs, sentences_per_run):
            run_tokens   = run["tokens"]
            boundary_map = {b.after_tok_idx: b.gap_sec for b in run["boundaries"]}
            run_offset   = 0

            for sent_tokens in run_sentences:
                s_i     += 1
                sent_len = len(sent_tokens)
                span     = run_tokens[run_offset : run_offset + sent_len]

                if [te.token for te in span] != sent_tokens:
                    raise RuntimeError(
                        f"Token mismatch at sentence {s_i}: "
                        f"expected {sent_tokens}, got {[te.token for te in span]}"
                    )

                # Build items: ('tok', TokenEntry) | ('pause', (marker, gap))
                items: List[Tuple[str, Any]] = []
                for local_idx, te in enumerate(span):
                    global_idx = run_offset + local_idx
                    items.append(("tok", te))
                    # Insert pause only for INTERNAL boundaries (both sides in sentence)
                    if global_idx in boundary_map and local_idx < sent_len - 1:
                        gap    = boundary_map[global_idx]
                        marker = pause_marker(gap)
                        if marker:
                            items.append(("pause", (marker, gap)))

                run_offset += sent_len

                tok_entries = [d for typ, d in items if typ == "tok"]

                speakers   = {te.meta.get("speaker")           for te in tok_entries if te.meta.get("speaker")}
                abbrs      = {te.meta.get("speaker_abbr")      for te in tok_entries if te.meta.get("speaker_abbr")}
                ages       = {te.meta.get("speaker_age")       for te in tok_entries if te.meta.get("speaker_age")}
                genders    = {te.meta.get("speaker_gender")    for te in tok_entries if te.meta.get("speaker_gender")}
                educations = {te.meta.get("speaker_education") for te in tok_entries if te.meta.get("speaker_education")}
                locations  = {te.meta.get("location")          for te in tok_entries if te.meta.get("location")}

                start_times = [_safe_float(te.meta.get("start_time")) for te in tok_entries]
                end_times   = [_safe_float(te.meta.get("end_time"))   for te in tok_entries]
                st = next((x for x in start_times if x is not None), None)
                et = next((x for x in reversed(end_times) if x is not None), None)

                text_parts = [d[0] if typ == "pause" else d.token for typ, d in items]
                sent_text  = " ".join(text_parts)

                fh.write(f"# sent_id = {base_id}-r{s_i:06d}\n")
                fh.write(f"# text = {sent_text}\n")
                if len(speakers)   == 1: fh.write(f"# speaker = {next(iter(speakers))}\n")
                if len(abbrs)      == 1: fh.write(f"# speaker_abbr = {next(iter(abbrs))}\n")
                if len(ages)       == 1: fh.write(f"# speaker_age = {next(iter(ages))}\n")
                if len(genders)    == 1: fh.write(f"# speaker_gender = {next(iter(genders))}\n")
                if len(educations) == 1: fh.write(f"# speaker_education = {next(iter(educations))}\n")
                if len(locations)  == 1: fh.write(f"# location = {next(iter(locations))}\n")
                if st is not None:       fh.write(f"# start_time = {st:.3f}\n")
                if et is not None:       fh.write(f"# end_time = {et:.3f}\n")

                row_id = 1
                for typ, dat in items:
                    if typ == "tok":
                        cols = list(dat.cols)
                        cols[0] = str(row_id)
                        fh.write("\t".join(cols) + "\n")
                    else:
                        marker, gap = dat
                        fh.write(_pause_row(marker, gap, row_id) + "\n")
                    row_id += 1

                fh.write("\n")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sentence segmentation for CoNLL-U files using Gemini (local, no Colab)."
    )
    parser.add_argument("input", help="Input .conllu / .conllup file")
    parser.add_argument("--api-key", default=None,
                        help="Gemini API key (default: GEMINI_API_KEY env or gemini_api_key.txt)")
    parser.add_argument("--model",   default=MODEL_NAME, help="Gemini model name")
    args = parser.parse_args()

    # Resolve API key
    api_key = (args.api_key
               or os.environ.get("GEMINI_API_KEY")
               or os.environ.get("GOOGLE_API_KEY"))
    if not api_key:
        key_file = Path(__file__).parent / "gemini_api_key.txt"
        if key_file.exists():
            api_key = key_file.read_text().strip()
    if not api_key:
        sys.exit("No Gemini API key found. Use --api-key, GEMINI_API_KEY env, "
                 "or place key in gemini_api_key.txt next to this script.")

    model_name = args.model
    client = genai.Client(api_key=api_key)

    in_path = Path(args.input)
    if not in_path.exists():
        sys.exit(f"Input file not found: {in_path}")

    # Parse
    token_entries, all_tokens = parse_conllu_like(str(in_path))
    print(f"Loaded  : {in_path}")
    print(f"Tokens  : {len(all_tokens)}")

    # Build speaker runs
    runs = build_speaker_runs(token_entries)
    print(f"\nSpeaker runs: {len(runs)}")
    for i, run in enumerate(runs):
        bcount = len(run["boundaries"])
        gap_str = (f"  gaps=[{', '.join(f'{b.gap_sec:.2f}s' for b in run['boundaries'])}]"
                   if bcount else "")
        print(f"  Run {i+1:2d}  speaker={run['speaker']:10s}  "
              f"tokens={len(run['tokens']):3d}  boundaries={bcount}{gap_str}")

    # Segment
    print("\nSegmenting with Gemini …")
    sentences_per_run = segment_all_runs(client, runs, model_name)
    total_sents = sum(len(s) for s in sentences_per_run)
    print(f"\nTotal sentences: {total_sents}")

    # Write outputs
    base_id       = in_path.stem
    out_dir       = in_path.parent
    out_sent_txt  = str(out_dir / f"{base_id}.sentences.txt")
    out_sent_json = str(out_dir / f"{base_id}.sentences.json")
    out_conll     = str(out_dir / f"{base_id}.corrected.conllup")

    write_sentences_txt(sentences_per_run,  out_sent_txt)
    write_sentences_json(sentences_per_run, out_sent_json)
    write_corrected_conllup(runs, sentences_per_run, out_conll, base_id=base_id)

    print(f"\nWrote:")
    print(f"  {out_sent_txt}")
    print(f"  {out_sent_json}")
    print(f"  {out_conll}")


if __name__ == "__main__":
    main()
