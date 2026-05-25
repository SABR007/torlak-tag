#!/usr/bin/env python3
"""
enrich_conllu.py
----------------
Re-inserts special tokens that were silently stripped during EXB preprocessing
back into a CoNLL-U file produced by the TorlakTag notebook.

The preprocessing in the notebook strips trailing '/' (truncation markers) from
tokens like 'stojU/' → 'stoju' without preserving the '/' as a separate token.
This script:
  1. Parses the CoNLL-U predictions file.
  2. Parses the original EXB transcript.
  3. Matches each CoNLL-U sentence to its EXB utterance (via start_time + speaker_abbr).
  4. Re-tokenises the EXB raw text with an extended tokeniser that captures
     trailing '/' as a separate special token.
  5. Inserts those stripped specials back into the CoNLL-U with UPOS=X and all
     other fields blank ('_').
  6. Writes the enriched CoNLL-U to a new file.

Usage:
    python enrich_conllu.py \
        --conllu "REMBERT_TOR_C_0031.pred (1).conllu" \
        --exb    TOR_C_0031.exb \
        --output REMBERT_TOR_C_0031.enriched.conllu
"""

import re
import sys
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Set

try:
    from lxml import etree as ET
    _USE_LXML = True
except ImportError:
    import xml.etree.ElementTree as ET
    _USE_LXML = False

# ── Normalisation helpers (mirror the notebook exactly) ───────────────────────

def apply_spec_mapping(s: str) -> str:
    if s is None:
        return ""
    s = s.replace("#", "")
    s = s.replace("W", "Ə").replace("w", "ə")
    s = s.replace("1", "ḱ").replace("6", "ḱ")
    s = s.replace("2", "ǵ")
    s = s.replace("3", "č")
    s = s.replace("x", "š").replace("X", "š")
    s = s.replace("5", "ƨ")
    s = s.replace("ššš", "XXX")
    return s

RE_OVERLAP   = re.compile(r"\[[^\]]*\]")
RE_DOUBLEPAR = re.compile(r"^\(\(.*\)\)$")
RE_BULLETS   = re.compile(r"^[•]+$")
RE_LONGVOWEL = re.compile(r"([aeiouə])\1+")
RE_SPACES    = re.compile(r"\s+")
RE_TRAIL_SL  = re.compile(r"/+$")

_STRIP_EDGE = " \t\r\n\"'\u201c\u201d\u201e`\u00b4.,;:!?(){}[]<>\u2022"


def is_x_special_token(tok: str) -> bool:
    t = tok.strip()
    if not t:
        return True
    if RE_DOUBLEPAR.match(t):
        return True
    if RE_BULLETS.match(t):
        return True
    return False


def strip_attached_specials(tok: str) -> str:
    t = tok.strip()
    t = RE_TRAIL_SL.sub("", t)   # remove trailing slashes
    t = t.strip(_STRIP_EDGE)
    return t


def normalize_word(tok: str) -> str:
    t = apply_spec_mapping(tok).lower()
    t = RE_LONGVOWEL.sub(r"\1", t)
    t = RE_SPACES.sub(" ", t).strip()
    return t


# ── Extended tokeniser ─────────────────────────────────────────────────────────

def find_slash_insertions(raw_text: str) -> List[int]:
    """
    Simulates the notebook's tokenize_with_rules on *raw_text* and returns
    the 0-based indices (into the resulting orig_tokens list) AFTER WHICH a
    '/' special token should be inserted.

    These correspond to EXB tokens like 'stojU/' where the trailing '/' was
    stripped and NOT kept as a separate token.
    """
    if not raw_text:
        return []

    s = apply_spec_mapping(raw_text).lower()
    s = RE_OVERLAP.sub(" ", s)
    raw_tokens = [t for t in s.split() if t.strip()]

    insertions: List[int] = []
    orig_idx = 0   # tracks which orig_token we are currently producing

    for rt in raw_tokens:
        if is_x_special_token(rt):
            orig_idx += 1
            continue

        has_alnum = any(ch.isalpha() or ch.isdigit() for ch in rt)
        if has_alnum:
            slash_m = RE_TRAIL_SL.search(rt)
            w = strip_attached_specials(rt)
            w = normalize_word(w)
            if w:
                if slash_m:
                    insertions.append(orig_idx)  # insert '/' after this token
                orig_idx += 1
            else:
                # Stripped to nothing → kept as-is special token
                orig_idx += 1
        else:
            orig_idx += 1

    return insertions


# ── EXB parser ────────────────────────────────────────────────────────────────

def parse_exb_utterances(exb_path: Path) -> List[Dict]:
    """
    Parse an EXB file and return utterances sorted by start_time.
    Only tiers whose display-name contains '_' are included (same rule as
    the notebook).
    """
    if _USE_LXML:
        tree = ET.parse(str(exb_path))
        root = tree.getroot()
    else:
        tree = ET.parse(str(exb_path))
        root = tree.getroot()

    # Build timeline id → time mapping
    tli_time: Dict[str, float] = {}
    for tli in root.iter("tli"):
        tid = tli.attrib.get("id")
        if tid and "time" in tli.attrib:
            try:
                tli_time[tid] = float(tli.attrib["time"])
            except (ValueError, TypeError):
                pass

    file_id = exb_path.stem
    utterances: List[Dict] = []

    for tier in root.iter("tier"):
        display = tier.attrib.get("display-name", "")
        if "_" not in display:
            continue
        speaker_abbr = display

        for ev in tier.iter("event"):
            if ev.text is None:
                continue
            start = ev.attrib.get("start")
            end   = ev.attrib.get("end")
            if start not in tli_time or end not in tli_time:
                continue
            utterances.append({
                "file_id":      file_id,
                "speaker_abbr": speaker_abbr,
                "start_time":   tli_time[start],
                "end_time":     tli_time[end],
                "raw_text":     ev.text,
            })

    utterances.sort(key=lambda u: u["start_time"])
    return utterances


# ── CoNLL-U parser ─────────────────────────────────────────────────────────────

def parse_conllu(conllu_path: Path) -> Tuple[List[str], List[Dict]]:
    """
    Parse a CoNLL-U file.

    Returns:
        global_header : list of lines before the first blank line that start
                        with '# global'
        sentences     : list of dicts with keys 'meta' (list of comment lines)
                        and 'tokens' (list of 10-field rows as string lists)
    """
    global_header: List[str] = []
    sentences: List[Dict] = []
    current_meta: List[str] = []
    current_tokens: List[List[str]] = []
    in_header = True

    with conllu_path.open(encoding="utf-8") as fh:
        for raw in fh:
            line = raw.rstrip("\n")

            # Collect the very first global.columns header line
            if in_header and line.startswith("# global"):
                global_header.append(line)
                continue
            if in_header and line.strip() == "":
                # blank line separates global header from sentences
                in_header = False
                continue
            in_header = False  # any non-header content ends header mode

            if line.startswith("#"):
                current_meta.append(line)
            elif line.strip() == "":
                if current_meta or current_tokens:
                    sentences.append({
                        "meta":   current_meta,
                        "tokens": current_tokens,
                    })
                    current_meta   = []
                    current_tokens = []
            else:
                parts = line.split("\t")
                if len(parts) == 10:
                    current_tokens.append(parts)

    # flush last sentence if file doesn't end with blank line
    if current_meta or current_tokens:
        sentences.append({"meta": current_meta, "tokens": current_tokens})

    return global_header, sentences


def get_meta_value(meta_lines: List[str], key: str) -> Optional[str]:
    prefix = f"# {key} = "
    for line in meta_lines:
        if line.startswith(prefix):
            return line[len(prefix):].strip()
    return None


# ── Sentence ↔ utterance matching ─────────────────────────────────────────────

def match_sentences_to_utterances(
    sentences: List[Dict],
    utterances: List[Dict],
) -> List[Tuple[Dict, Optional[Dict]]]:
    """
    Match CoNLL-U sentences to EXB utterances.

    Primary key : (round(start_time, 3), speaker_abbr)
    Fallback    : positional (same sort order)

    Returns list of (sentence, utterance_or_None) pairs.
    """
    # Build lookup
    utt_by_key: Dict[Tuple, Dict] = {}
    for u in utterances:
        key = (round(u["start_time"], 3), u["speaker_abbr"])
        utt_by_key[key] = u

    pairs: List[Tuple[Dict, Optional[Dict]]] = []
    unmatched_utts = list(utterances)  # used for positional fallback

    for sent in sentences:
        start_s = get_meta_value(sent["meta"], "start_time")
        abbr_s  = get_meta_value(sent["meta"], "speaker_abbr")

        utt = None
        if start_s is not None and abbr_s is not None:
            key = (round(float(start_s), 3), abbr_s)
            utt = utt_by_key.get(key)

        pairs.append((sent, utt))

    return pairs


# ── Sentence rebuilder ─────────────────────────────────────────────────────────

_BLANK_X_ROW = ["_", "_", "_", "X", "_", "_", "_", "_", "_", "_"]
# fields: ID  FORM  LEMMA  UPOS  XPOS  FEATS  HEAD  DEPREL  DEPS  MISC


def rebuild_sentence(sent: Dict, utt: Optional[Dict]) -> Dict:
    """
    Given a CoNLL-U sentence and its matched EXB utterance, return a new
    sentence dict with stripped '/' tokens re-inserted as X rows.

    If the utterance is None or no insertions are needed, the original
    sentence is returned unchanged.
    """
    if utt is None:
        return sent

    insertions = find_slash_insertions(utt["raw_text"])
    if not insertions:
        return sent

    insertion_set: Set[int] = set(insertions)
    orig_tokens = sent["tokens"]

    # Safety: make sure insertion indices are in range
    out_of_range = [i for i in insertion_set if i >= len(orig_tokens)]
    if out_of_range:
        warn = (
            f"# WARNING: slash-insertion index out of range "
            f"{out_of_range} (sentence has {len(orig_tokens)} tokens) — skipped"
        )
        return {"meta": sent["meta"] + [warn], "tokens": orig_tokens}

    # Build new token list
    new_rows: List[List[str]] = []
    for i, row in enumerate(orig_tokens):
        new_rows.append(list(row))
        if i in insertion_set:
            slash_row = list(_BLANK_X_ROW)
            slash_row[1] = "/"      # FORM = '/'
            new_rows.append(slash_row)

    # Re-number IDs (1-based, sequential)
    for idx, row in enumerate(new_rows, start=1):
        row[0] = str(idx)

    # Update '# text' comment to include the new '/' tokens
    new_text = " ".join(row[1] for row in new_rows)
    new_meta = []
    for line in sent["meta"]:
        if line.startswith("# text = "):
            new_meta.append(f"# text = {new_text}")
        else:
            new_meta.append(line)

    return {"meta": new_meta, "tokens": new_rows}


# ── CoNLL-U writer ─────────────────────────────────────────────────────────────

def write_conllu(
    output_path: Path,
    global_header: List[str],
    sentences: List[Dict],
) -> None:
    with output_path.open("w", encoding="utf-8") as fh:
        for line in global_header:
            fh.write(line + "\n")
        if global_header:
            fh.write("\n")
        for sent in sentences:
            for line in sent["meta"]:
                fh.write(line + "\n")
            for row in sent["tokens"]:
                fh.write("\t".join(row) + "\n")
            fh.write("\n")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Re-insert stripped special tokens into a CoNLL-U file."
    )
    parser.add_argument(
        "--conllu",
        default="REMBERT_TOR_C_0031.pred (1).conllu",
        help="Path to the input CoNLL-U predictions file",
    )
    parser.add_argument(
        "--exb",
        default="TOR_C_0031.exb",
        help="Path to the original EXB transcript",
    )
    parser.add_argument(
        "--output",
        default="REMBERT_TOR_C_0031.enriched.conllu",
        help="Path for the enriched output CoNLL-U file",
    )
    args = parser.parse_args()

    conllu_path = Path(args.conllu)
    exb_path    = Path(args.exb)
    output_path = Path(args.output)

    print(f"Reading CoNLL-U : {conllu_path}")
    global_header, sentences = parse_conllu(conllu_path)
    print(f"  → {len(sentences)} sentences, "
          f"{sum(len(s['tokens']) for s in sentences)} tokens")

    print(f"Reading EXB     : {exb_path}")
    utterances = parse_exb_utterances(exb_path)
    print(f"  → {len(utterances)} utterances")

    pairs = match_sentences_to_utterances(sentences, utterances)
    n_matched = sum(1 for _, u in pairs if u is not None)
    n_unmatched = len(pairs) - n_matched
    print(f"  → {n_matched} matched, {n_unmatched} unmatched")

    if n_unmatched:
        print("  Unmatched sentences:")
        for sent, utt in pairs:
            if utt is None:
                sid  = get_meta_value(sent["meta"], "sent_id")  or "?"
                st   = get_meta_value(sent["meta"], "start_time") or "?"
                abbr = get_meta_value(sent["meta"], "speaker_abbr") or "?"
                print(f"    sent_id={sid}  start={st}  speaker={abbr}")

    # Rebuild sentences, inserting stripped '/' tokens
    new_sentences = []
    insertions_log: List[str] = []
    for sent, utt in pairs:
        rebuilt = rebuild_sentence(sent, utt)
        new_sentences.append(rebuilt)

        # Log any insertions made
        if utt is not None:
            ins = find_slash_insertions(utt["raw_text"])
            if ins:
                sid = get_meta_value(sent["meta"], "sent_id") or "?"
                insertions_log.append(
                    f"  {sid}: inserted '/' after token position(s) {[i+1 for i in ins]}"
                )

    if insertions_log:
        print(f"\nSpecial '/' tokens re-inserted ({len(insertions_log)} sentences affected):")
        for msg in insertions_log:
            print(msg)
    else:
        print("\nNo stripped '/' tokens found — output identical to input.")

    write_conllu(output_path, global_header, new_sentences)
    new_total = sum(len(s["tokens"]) for s in new_sentences)
    print(f"\nWritten: {output_path}")
    print(f"  {len(new_sentences)} sentences, {new_total} tokens "
          f"(+{new_total - sum(len(s['tokens']) for s in sentences)} inserted)")


if __name__ == "__main__":
    main()
