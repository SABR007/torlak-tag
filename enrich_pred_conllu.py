#!/usr/bin/env python3
"""
enrich_pred_conllu.py
---------------------
Enriches a sentence-segmented CoNLL-U file (output of sentence_segmenter.py)
by aligning tokens back to their original pred.conllu utterances and adding
provenance metadata. Also injects geographical_coordinates from TOR_C_speeches.csv.

Designed for TOR_C_0006/0025/0085 which have no EXB files — uses the pred.conllu
sent_ids as utterance references (equivalent to EXB utterance IDs).

Usage:
    python enrich_pred_conllu.py \\
        --corrected  TOR_C_0006.pred.corrected.conllup \\
        --pred       TOR_C_0006.pred.conllu \\
        --speeches   speaker_metadata/TOR_C_speeches.csv \\
        --output     TOR_C_0006.pred.enriched.conllu

Output format matches REMBERT_TOR_C_0031.corrected.enriched.conllu.
"""

import re
import sys
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Optional

PAUSE_FORMS = {"\u00b7", "\u00b7\u00b7", "\u00b7\u00b7\u00b7"}  # ·, ··, ···


# ── Parsers ────────────────────────────────────────────────────────────────────

def parse_pred_conllu(path: Path) -> List[Dict]:
    """
    Parse original pred.conllu into utterances.
    Returns list of dicts: {sent_id: str, tokens: List[str]}
    """
    utterances: List[Dict] = []
    cur_sent_id: Optional[str] = None
    cur_tokens:  List[str] = []

    with path.open(encoding="utf-8") as f:
        for raw in f:
            line = raw.rstrip("\n")
            if line.startswith("# sent_id"):
                cur_sent_id = line.split("=", 1)[1].strip()
            elif not line.strip():
                if cur_sent_id and cur_tokens:
                    utterances.append({"sent_id": cur_sent_id, "tokens": cur_tokens})
                cur_sent_id = None
                cur_tokens = []
            elif line and not line.startswith("#") and "\t" in line:
                cols = line.split("\t")
                tok_id = cols[0]
                if "-" not in tok_id and "." not in tok_id and len(cols) >= 2:
                    cur_tokens.append(cols[1])  # FORM

    if cur_sent_id and cur_tokens:
        utterances.append({"sent_id": cur_sent_id, "tokens": cur_tokens})

    return utterances


def parse_corrected(path: Path) -> List[Dict]:
    """
    Parse .corrected.conllup from sentence_segmenter.py.
    Returns list of dicts: {meta: Dict[str,str], rows: List[List[str]]}
    """
    sentences: List[Dict] = []
    cur_meta:  Dict[str, str] = {}
    cur_rows:  List[List[str]] = []

    with path.open(encoding="utf-8") as f:
        for raw in f:
            line = raw.rstrip("\n")
            if line.startswith("#"):
                m = re.match(r"#\s*(.+?)\s*=\s*(.*)\s*$", line)
                if m:
                    cur_meta[m.group(1).strip()] = m.group(2).strip()
            elif not line.strip():
                if cur_rows:
                    sentences.append({"meta": dict(cur_meta), "rows": cur_rows})
                cur_meta = {}
                cur_rows = []
            elif "\t" in line:
                cols = line.split("\t")
                tok_id = cols[0]
                if "-" not in tok_id and "." not in tok_id:
                    cur_rows.append(cols)

    if cur_rows:
        sentences.append({"meta": dict(cur_meta), "rows": cur_rows})

    return sentences


def load_speeches(path: Path) -> Dict[str, Tuple[float, float]]:
    """Parse TOR_C_speeches.csv → {transcript_id: (lat, lon)}"""
    coords: Dict[str, Tuple[float, float]] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(";")
            if len(parts) >= 4:
                tid, lat_s, lon_s = parts[0], parts[2], parts[3]
                try:
                    coords[tid] = (float(lat_s), float(lon_s))
                except ValueError:
                    pass
    return coords


# ── Alignment ─────────────────────────────────────────────────────────────────

def _is_pause(cols: List[str]) -> bool:
    return cols[1] in PAUSE_FORMS if len(cols) > 1 else False


def build_provenance(
    utterances: List[Dict],
    sentences:  List[Dict],
) -> Dict[Tuple[int, int], Tuple[str, int, bool]]:
    """
    Returns {(sent_idx, row_idx): (utt_sent_id, word_pos_1based, is_last_in_utt)}.

    Aligns the flat sequence of real (non-pause) tokens in `sentences` against
    the flat token list from `utterances`.
    """
    # Build flat list from utterances: (form, sent_id, word_pos, is_last)
    flat_utts: List[Tuple[str, str, int, bool]] = []
    for utt in utterances:
        n = len(utt["tokens"])
        for w, form in enumerate(utt["tokens"], 1):
            flat_utts.append((form, utt["sent_id"], w, w == n))

    # Build flat list of real tokens from corrected, with back-reference keys
    flat_corr: List[Tuple[int, int, List[str]]] = []  # (sent_idx, row_idx, cols)
    for si, sent in enumerate(sentences):
        for ri, cols in enumerate(sent["rows"]):
            if not _is_pause(cols):
                flat_corr.append((si, ri, cols))

    if len(flat_corr) != len(flat_utts):
        # Try to give a helpful error
        forms_corr = [c[1] for _, _, c in flat_corr]
        forms_utts = [f for f, *_ in flat_utts]
        mismatch = next(
            (i for i, (a, b) in enumerate(zip(forms_corr, forms_utts)) if a != b),
            None
        )
        raise ValueError(
            f"Token count mismatch: corrected={len(flat_corr)}, pred={len(flat_utts)}"
            + (f"\nFirst mismatch at pos {mismatch}: "
               f"corrected={forms_corr[mismatch]!r}, pred={forms_utts[mismatch]!r}"
               if mismatch is not None else "")
        )

    prov: Dict[Tuple[int, int], Tuple[str, int, bool]] = {}
    for (si, ri, cols), (form, sent_id, word_pos, is_last) in zip(flat_corr, flat_utts):
        prov[(si, ri)] = (sent_id, word_pos, is_last)

    return prov


# ── Writer ────────────────────────────────────────────────────────────────────

def write_enriched(
    sentences:   List[Dict],
    prov:        Dict[Tuple[int, int], Tuple[str, int, bool]],
    geo_str:     str,
    output_path: Path,
) -> None:

    with output_path.open("w", encoding="utf-8") as f:
        f.write("# global.columns = ID FORM LEMMA UPOS XPOS FEATS HEAD DEPREL DEPS MISC\n\n")

        for si, sent in enumerate(sentences):
            meta = sent["meta"]
            rows = sent["rows"]

            # ── Collect provenance info for this sentence ──────────────────────
            sent_prov: List[Tuple[str, int, bool]] = []
            for ri, cols in enumerate(rows):
                if not _is_pause(cols) and (si, ri) in prov:
                    sent_prov.append(prov[(si, ri)])

            # EXB_sources: unique utterance sent_ids in order of first appearance
            seen: List[str] = []
            for utt_id, _, _ in sent_prov:
                if not seen or seen[-1] != utt_id:
                    seen.append(utt_id)

            # EXB_token_map: position:utt_id-wN for real tokens only
            token_map_parts: List[str] = []
            real_pos = 0
            for ri, cols in enumerate(rows):
                if not _is_pause(cols):
                    real_pos += 1
                    if (si, ri) in prov:
                        utt_id, word_pos, _ = prov[(si, ri)]
                        token_map_parts.append(f"{real_pos}:{utt_id}-w{word_pos}")

            # ── Sentence metadata ──────────────────────────────────────────────
            f.write(f"# sent_id = {meta.get('sent_id', '_')}\n")
            f.write(f"# text = {meta.get('text', '_')}\n")

            # Speaker info (segmenter guarantees single-speaker per sentence)
            if "speakers" in meta:
                # Multi-speaker (shouldn't happen after segmenter, but handle it)
                f.write(f"# speakers = {meta['speakers']}\n")
            else:
                if "speaker" in meta:
                    f.write(f"# speaker = {meta['speaker']}\n")
                if "speaker_abbr" in meta:
                    f.write(f"# speaker_abbr = {meta['speaker_abbr']}\n")
                if "speaker_age" in meta:
                    f.write(f"# speaker_age = {meta['speaker_age']}\n")
                if "speaker_gender" in meta:
                    f.write(f"# speaker_gender = {meta['speaker_gender']}\n")
                if "speaker_education" in meta:
                    f.write(f"# speaker_education = {meta['speaker_education']}\n")

            if "location" in meta:
                f.write(f"# location = {meta['location']}\n")

            f.write(f"# geographical_coordinates = {geo_str}\n")

            if "start_time" in meta:
                f.write(f"# start_time = {meta['start_time']}\n")
            if "end_time" in meta:
                f.write(f"# end_time = {meta['end_time']}\n")
            f.write("# timing_precision = exact\n")

            f.write(f"# EXB_sources = {'|'.join(seen)}\n")
            f.write(f"# EXB_token_map = {' '.join(token_map_parts)}\n")

            # ── Token rows ─────────────────────────────────────────────────────
            row_id = 1
            for ri, cols in enumerate(rows):
                new_cols = list(cols)
                new_cols[0] = str(row_id)

                if _is_pause(cols):
                    # Keep pause row as-is (MISC has IsPause=yes|PauseDur=...)
                    pass
                elif (si, ri) in prov:
                    utt_id, word_pos, is_last = prov[(si, ri)]
                    misc = f"OrigUttID={utt_id}|EXBTokenRef={utt_id}-w{word_pos}"
                    if is_last:
                        misc += "|UttEnd=yes"
                    new_cols[9] = misc

                f.write("\t".join(new_cols) + "\n")
                row_id += 1

            f.write("\n")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Enrich sentence-segmented CoNLL-U with utterance provenance "
                    "from original pred.conllu."
    )
    parser.add_argument("--corrected", required=True,
                        help="Corrected .conllup from sentence_segmenter.py")
    parser.add_argument("--pred",      required=True,
                        help="Original .pred.conllu (utterance source)")
    parser.add_argument("--speeches",  required=True,
                        help="TOR_C_speeches.csv (for geographical coordinates)")
    parser.add_argument("--output",    required=True,
                        help="Output .enriched.conllu path")
    args = parser.parse_args()

    corrected_path = Path(args.corrected)
    pred_path      = Path(args.pred)
    speeches_path  = Path(args.speeches)
    output_path    = Path(args.output)

    # Validate inputs
    for p in (corrected_path, pred_path, speeches_path):
        if not p.exists():
            sys.exit(f"ERROR: File not found: {p}")

    # Load
    print(f"Loading pred.conllu:    {pred_path}")
    utterances = parse_pred_conllu(pred_path)
    print(f"  → {len(utterances)} utterances, "
          f"{sum(len(u['tokens']) for u in utterances)} tokens")

    print(f"Loading corrected:      {corrected_path}")
    sentences = parse_corrected(corrected_path)
    print(f"  → {len(sentences)} sentences")

    # Geo coordinates
    transcript_id = pred_path.stem.split(".")[0]  # e.g. "TOR_C_0006"
    speeches = load_speeches(speeches_path)
    geo = speeches.get(transcript_id)
    geo_str = f"{geo[0]}, {geo[1]}" if geo else "_"
    print(f"Transcript ID:          {transcript_id}")
    print(f"Geo coordinates:        {geo_str}")

    # Build provenance
    print("Aligning tokens …")
    prov = build_provenance(utterances, sentences)

    # Write
    print(f"Writing:                {output_path}")
    write_enriched(sentences, prov, geo_str, output_path)
    print("Done.")


if __name__ == "__main__":
    main()
