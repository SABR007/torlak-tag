#!/usr/bin/env python3
"""
conllu_to_enriched_exb.py
--------------------------
Adds annotation tiers (LEMMA, UPOS, XPOS, FEATS, topic_clusters, fine_customs)
to the TOR_C EXB files, using data from topic_tagged_conllu/*.enriched.conllu.

Each annotation tier is utterance-level: one event per sentence, spanning the
sentence's [start_time, end_time], with token values space-joined.

Usage:
    python conllu_to_enriched_exb.py \
        --conllu   topic_tagged_conllu/ \
        --exb      TOR_C_EXB_transcripts/ \
        --output   enriched_exb/
"""

import re
import sys
import argparse
from html import escape as xml_escape
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import xml.etree.ElementTree as ET

# ── Constants ─────────────────────────────────────────────────────────────────

ANNOTATION_COLS = ["LEMMA", "UPOS", "XPOS"]
SENTENCE_META   = []
ALL_TIERS       = ANNOTATION_COLS + SENTENCE_META
TLI_TOLERANCE      = 0.05   # seconds — tight match for most sentences
TLI_TOLERANCE_WIDE = 0.50   # seconds — fallback for annotator-re-segmented files

# ── CoNLL-U parser ────────────────────────────────────────────────────────────

def parse_conllu(path: Path) -> List[dict]:
    """Parse CoNLL-U into a list of sentence dicts."""
    sentences = []
    cur_meta: List[str] = []
    cur_tokens: List[List[str]] = []

    with path.open(encoding="utf-8") as f:
        for raw in f:
            line = raw.rstrip("\n")
            if line.startswith("# global"):
                continue
            if line.startswith("#"):
                cur_meta.append(line)
            elif line.strip() == "":
                if cur_meta or cur_tokens:
                    sentences.append({"meta": cur_meta, "tokens": cur_tokens})
                    cur_meta, cur_tokens = [], []
            else:
                parts = line.split("\t")
                if len(parts) == 10:
                    cur_tokens.append(parts)

    if cur_meta or cur_tokens:
        sentences.append({"meta": cur_meta, "tokens": cur_tokens})
    return sentences


def get_meta(meta_lines: List[str], key: str) -> Optional[str]:
    prefix = f"# {key} = "
    for line in meta_lines:
        if line.startswith(prefix):
            return line[len(prefix):].strip()
    return None


# ── EXB XML helpers ───────────────────────────────────────────────────────────

UD_META = (
    '<ud-meta-information>'
    '<ud-information attribute-name="Location"/>'
    '<ud-information attribute-name="Duration"/>'
    '<ud-information attribute-name="Dialect"/>'
    '<ud-information attribute-name="Date"/>'
    '</ud-meta-information>'
)
UD_SPEAKER = (
    '<ud-speaker-information>'
    '<ud-information attribute-name="Year"/>'
    '<ud-information attribute-name="Age of Birth"/>'
    '<ud-information attribute-name="Residence"/>'
    '<ud-information attribute-name="Education"/>'
    '</ud-speaker-information>'
    '<comment/>'
)


def fix_missing_ud_elements(raw: str) -> str:
    """Add ud-meta-information and ud-speaker-information blocks if absent.
    EXMARaLDA 1.8.x requires these elements even when empty."""
    if '<ud-meta-information>' not in raw:
        raw = raw.replace('</meta-information>', UD_META + '</meta-information>', 1)
    if '<ud-speaker-information>' not in raw:
        raw = raw.replace('</speaker>', UD_SPEAKER + '</speaker>')
    return raw


def parse_exb_for_structure(path: Path):
    """
    Read EXB file. Returns (raw_string, root_element).
    root_element is used for reading structure only — we never serialize it back.
    """
    raw = path.read_text(encoding="utf-8")
    idx = raw.index("<basic-transcription>")
    root = ET.fromstring(raw[idx:])
    return raw, root


def build_tli_lookup(root: ET.Element) -> Dict[str, float]:
    """Return {tli_id: time_seconds} from <common-timeline>."""
    lookup = {}
    for tli in root.iter("tli"):
        lookup[tli.get("id")] = float(tli.get("time"))
    return lookup


def find_tli_for_time(tli_lookup: Dict[str, float], target: float,
                       tolerance: float = TLI_TOLERANCE) -> Optional[str]:
    """Find TLI id whose time is within tolerance of target."""
    best_id = None
    best_diff = float("inf")
    for tli_id, t in tli_lookup.items():
        diff = abs(t - target)
        if diff < best_diff:
            best_diff = diff
            best_id = tli_id
    if best_diff <= tolerance:
        return best_id
    return None


def get_or_create_tli(tli_lookup: Dict[str, float],
                       new_tlis: Dict[float, str],
                       time_val: float) -> str:
    """Return existing TLI id or register a new one (stored in new_tlis)."""
    tli_id = find_tli_for_time(tli_lookup, time_val, TLI_TOLERANCE_WIDE)
    if tli_id:
        return tli_id
    for t, tid in new_tlis.items():
        if abs(t - time_val) <= TLI_TOLERANCE_WIDE:
            return tid
    new_id = f"T_ann_{len(tli_lookup) + len(new_tlis)}"
    new_tlis[time_val] = new_id
    return new_id


def build_speaker_lookup(root: ET.Element) -> Dict[str, str]:
    """Return {abbreviation: speaker_id} from <speakertable>."""
    lookup = {}
    for spk in root.findall(".//speaker"):
        spk_id = spk.get("id")
        abbr_el = spk.find("abbreviation")
        if abbr_el is not None and abbr_el.text:
            lookup[abbr_el.text.strip()] = spk_id
    return lookup


def build_tier_lookup(root: ET.Element) -> Dict[str, dict]:
    """Return {speaker_id: {tier_id, display_name}} for transcription tiers."""
    lookup = {}
    for tier in root.findall(".//tier"):
        if tier.get("type") == "t":
            spk_id = tier.get("speaker")
            if spk_id:
                lookup[spk_id] = {
                    "tier_id": tier.get("id"),
                    "display_name": tier.get("display-name", ""),
                }
    return lookup


def max_tier_number(root: ET.Element) -> int:
    nums = []
    for tier in root.findall(".//tier"):
        m = re.match(r"TIE(\d+)$", tier.get("id", ""))
        if m:
            nums.append(int(m.group(1)))
    return max(nums, default=-1)


def find_speaker_by_time(root: ET.Element, tli_lookup: Dict[str, float],
                          start: float, end: float) -> Optional[str]:
    """
    Fall back: find the speaker whose transcription-tier event starts at ~start.
    For sentences spanning multiple EXB utterances, the first utterance's speaker is used.
    Tries tight tolerance first, then wide tolerance (for annotator-re-segmented files).
    Returns speaker_id or None.
    """
    for tolerance in (TLI_TOLERANCE, TLI_TOLERANCE_WIDE):
        for tier in root.findall(".//tier"):
            if tier.get("type") != "t":
                continue
            spk_id = tier.get("speaker")
            for event in tier.findall("event"):
                t_start = tli_lookup.get(event.get("start"), -1)
                if abs(t_start - start) <= tolerance:
                    return spk_id
    return None


# ── Core enrichment ───────────────────────────────────────────────────────────

def enrich_exb(conllu_path: Path, exb_path: Path, output_path: Path) -> dict:
    raw_exb, root = parse_exb_for_structure(exb_path)
    raw_exb = fix_missing_ud_elements(raw_exb)
    tli_lookup    = build_tli_lookup(root)
    spk_lookup    = build_speaker_lookup(root)   # abbr → spk_id
    tier_lookup   = build_tier_lookup(root)       # spk_id → tier info
    sentences     = parse_conllu(conllu_path)

    new_tlis: Dict[float, str] = {}   # time → new TLI id (injected as strings)

    # Collect annotation events per (speaker, tier_type)
    ann_events: Dict[Tuple[str, str], List[Tuple[str, str, str]]] = {}

    skipped = 0
    for sent in sentences:
        start_s = get_meta(sent["meta"], "start_time")
        end_s   = get_meta(sent["meta"], "end_time")
        if start_s is None or end_s is None:
            skipped += 1
            continue

        start_f = float(start_s)
        end_f   = float(end_s)

        start_tli = get_or_create_tli(tli_lookup, new_tlis, start_f)
        end_tli   = get_or_create_tli(tli_lookup, new_tlis, end_f)

        spk_abbr = get_meta(sent["meta"], "speaker_abbr") or get_meta(sent["meta"], "speaker")
        spk_id   = spk_lookup.get(spk_abbr) if spk_abbr else None
        if spk_id is None:
            spk_id = find_speaker_by_time(root, tli_lookup, start_f, end_f)
        if spk_id is None:
            skipped += 1
            continue

        tokens  = sent["tokens"]
        col_idx = {"LEMMA": 2, "UPOS": 3, "XPOS": 4, "FEATS": 5}
        values: Dict[str, str] = {
            col: " ".join(tok[col_idx[col]] for tok in tokens)
            for col in ANNOTATION_COLS
        }
        for col in SENTENCE_META:
            v = get_meta(sent["meta"], col)
            values[col] = v if v else "_"

        for tier_type in ALL_TIERS:
            ann_events.setdefault((spk_id, tier_type), []).append(
                (start_tli, end_tli, values[tier_type])
            )

    spk_to_display: Dict[str, str] = {
        spk_id: tier_lookup[spk_id]["display_name"]
        for spk_id in tier_lookup
    }
    next_tie = max_tier_number(root) + 1

    # Build new TLI strings to inject before </common-timeline>
    tli_lines = "".join(
        f'<tli id="{tid}" time="{t}"/>\n'
        for t, tid in new_tlis.items()
    )

    # Build new tier strings to inject before </basic-body>
    tier_lines_parts: List[str] = []
    for (spk_id, tier_type), events in sorted(ann_events.items()):
        display = xml_escape(spk_to_display.get(spk_id, spk_id), quote=True)
        tier_type_esc = xml_escape(tier_type, quote=True)
        parts = [
            f'<tier id="TIE{next_tie}" speaker="{spk_id}" '
            f'category="{tier_type_esc}" type="a" '
            f'display-name="{display} [{tier_type_esc}]">'
        ]
        for start_tli, end_tli, text in events:
            parts.append(
                f'<event start="{start_tli}" end="{end_tli}">'
                f'{xml_escape(text)}</event>'
            )
        parts.append("</tier>")
        tier_lines_parts.append("".join(parts))
        next_tie += 1

    tier_lines = "\n".join(tier_lines_parts)

    # Inject into the original EXB string (preserves all original formatting exactly)
    if tli_lines:
        raw_exb = raw_exb.replace("</common-timeline>",
                                   tli_lines + "</common-timeline>", 1)
    raw_exb = raw_exb.replace("</basic-body>",
                               tier_lines + "\n</basic-body>", 1)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(raw_exb, encoding="utf-8")

    return {"sentences": len(sentences), "skipped": skipped,
            "new_tiers": len(ann_events)}


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Enrich TOR_C EXB files with annotation tiers from CoNLL-U."
    )
    parser.add_argument("--conllu",  required=True, help="topic_tagged_conllu/ folder")
    parser.add_argument("--exb",     required=True, help="TOR_C_EXB_transcripts/ folder")
    parser.add_argument("--output",  required=True, help="Output folder for enriched EXB files")
    args = parser.parse_args()

    conllu_dir = Path(args.conllu)
    exb_dir    = Path(args.exb)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    conllu_files = sorted(conllu_dir.glob("TOR_C_*.enriched.conllu"))
    print(f"Found {len(conllu_files)} TOR_C CoNLL-U files\n")

    total_sents = total_skipped = total_tiers = 0
    no_exb = []

    for cf in conllu_files:
        stem = cf.name[: -len(".enriched.conllu")]
        base = re.match(r"(TOR_C_\d+)", stem).group(1)
        exb_path = exb_dir / f"{base}.exb"

        if not exb_path.exists():
            no_exb.append(stem)
            continue

        out_path = output_dir / f"{base}.exb"
        stats = enrich_exb(cf, exb_path, out_path)

        total_sents   += stats["sentences"]
        total_skipped += stats["skipped"]
        total_tiers   += stats["new_tiers"]

        print(f"  {stem:<40}  sents={stats['sentences']:>5}  "
              f"skipped={stats['skipped']:>4}  new_tiers={stats['new_tiers']:>3}")

    print(f"\n{'─'*60}")
    print(f"Total sentences : {total_sents}")
    print(f"Skipped         : {total_skipped}")
    print(f"Total new tiers : {total_tiers}")
    print(f"Output folder   : {output_dir}")
    if no_exb:
        print(f"\nNo EXB found for: {no_exb}")


if __name__ == "__main__":
    main()
