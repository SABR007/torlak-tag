#!/usr/bin/env python3
"""
enrich_conllu_corrected.py
--------------------------
Enriches a re-segmented / manually-corrected CoNLL-U file by:

  1. Aligning its flat token sequence against the flat EXB token sequence
     (the corrected file merges/splits EXB utterances, so time-based
      matching is not reliable).
  2. Mapping every token to its original EXB utterance (utt_id like
     TOR_C_0031-u1) and to its exact token position within that utterance
     (EXBTokenRef like TOR_C_0031-u1-w1), following TEI XML naming.
  3. Re-inserting stripped '/' truncation markers (from EXB forms like
     'stojU/', 'I/', 'jAm/') as separate X-tagged tokens at the correct
     positions within the merged sentences.
  4. Adding explicit provenance per token in the MISC field:
       OrigUttID=TOR_C_0031-u13|EXBTokenRef=TOR_C_0031-u13-w2
     plus a forced-alignment boundary marker on the last real token of
     each source utterance:
       UttEnd=yes
     and a sentence-level comment:
       # EXB_sources = TOR_C_0031-u13|TOR_C_0031-u14
  5. Adding explicit `# EXB_token_map = pos:TOR_C_0031-u1-w1 ...` comment
     showing every token's EXB origin by position (full TEI token reference).

Terminology note: EXB/TEI utterances are identified with the 'u' prefix
(e.g. TOR_C_0031-u1) and tokens within an utterance with the 'w' prefix
(e.g. TOR_C_0031-u1-w1), matching the TEI XML encoding.

Usage:
    python enrich_conllu_corrected.py \\
        --conllu  "REMBERT_TOR_C_0031.pred.corrected.conllup" \\
        --exb      TOR_C_0031.exb \\
        --output   REMBERT_TOR_C_0031.corrected.enriched.conllu
"""

import re
import sys
import argparse
import difflib
from pathlib import Path
from typing import List, Dict, Tuple, Optional

try:
    from lxml import etree as ET
    _USE_LXML = True
except ImportError:
    import xml.etree.ElementTree as ET
    _USE_LXML = False

# ── Normalisation helpers (identical to notebook) ─────────────────────────────

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
    t = RE_TRAIL_SL.sub("", t)
    t = t.strip(_STRIP_EDGE)
    return t


def normalize_word(tok: str) -> str:
    t = apply_spec_mapping(tok).lower()
    t = RE_LONGVOWEL.sub(r"\1", t)
    t = RE_SPACES.sub(" ", t).strip()
    return t


# ── EXB token extractor ────────────────────────────────────────────────────────

def extract_exb_tokens(raw_text: str) -> List[Tuple[str, bool, bool]]:
    """
    Tokenise one EXB event text.

    Returns list of (normalized_form, is_special, has_slash_after).

    'has_slash_after' is True when the original EXB token had a trailing '/'
    that was stripped during normalisation (e.g. 'stojU/' → form='stoju',
    has_slash_after=True).  In the output CoNLL-U, a '/' X-token should be
    inserted immediately after such a token.
    """
    if not raw_text:
        return []

    s = apply_spec_mapping(raw_text).lower()
    s = RE_OVERLAP.sub(" ", s)
    raw_tokens = [t for t in s.split() if t.strip()]

    result: List[Tuple[str, bool, bool]] = []
    for rt in raw_tokens:
        if is_x_special_token(rt):
            result.append((rt, True, False))
            continue

        has_alnum = any(ch.isalpha() or ch.isdigit() for ch in rt)
        if has_alnum:
            slash_m = RE_TRAIL_SL.search(rt)
            w = strip_attached_specials(rt)
            w = normalize_word(w)
            if w:
                result.append((w, False, bool(slash_m)))
            else:
                result.append((rt, True, False))
        else:
            result.append((rt, True, False))

    return result


# ── EXB parser ─────────────────────────────────────────────────────────────────

def parse_exb_utterances(exb_path: Path) -> List[Dict]:
    tree = ET.parse(str(exb_path))
    root = tree.getroot()

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

def parse_conllu(path: Path) -> Tuple[List[str], List[Dict]]:
    global_header: List[str] = []
    sentences: List[Dict]    = []
    current_meta: List[str]  = []
    current_tokens: List[List[str]] = []
    in_header = True

    with path.open(encoding="utf-8") as fh:
        for raw in fh:
            line = raw.rstrip("\n")

            if in_header and line.startswith("# global"):
                global_header.append(line)
                continue
            if in_header and line.strip() == "":
                in_header = False
                continue
            in_header = False

            if line.startswith("#"):
                current_meta.append(line)
            elif line.strip() == "":
                if current_meta or current_tokens:
                    sentences.append({"meta": current_meta, "tokens": current_tokens})
                    current_meta   = []
                    current_tokens = []
            else:
                parts = line.split("\t")
                if len(parts) == 10:
                    current_tokens.append(parts)

    if current_meta or current_tokens:
        sentences.append({"meta": current_meta, "tokens": current_tokens})

    return global_header, sentences


def get_meta_value(meta_lines: List[str], key: str) -> Optional[str]:
    prefix = f"# {key} = "
    for line in meta_lines:
        if line.startswith(prefix):
            return line[len(prefix):].strip()
    return None


def load_speaker_metadata(pred_path: Path) -> Dict[str, Dict[str, str]]:
    """
    Parse the original (un-corrected) pred CoNLL-U file and return a mapping
    of speaker_abbr → {speaker, speaker_abbr, speaker_age, speaker_gender,
    speaker_education} built from the first occurrence of each speaker.
    """
    _SPEAKER_FIELDS = (
        "speaker", "speaker_abbr", "speaker_age",
        "speaker_gender", "speaker_education",
    )
    result: Dict[str, Dict[str, str]] = {}
    _, sentences = parse_conllu(pred_path)
    for sent in sentences:
        abbr = get_meta_value(sent["meta"], "speaker_abbr")
        if abbr and abbr not in result:
            result[abbr] = {
                f: get_meta_value(sent["meta"], f) or "_"
                for f in _SPEAKER_FIELDS
            }
    return result


# ── Flat sequence alignment ────────────────────────────────────────────────────

class EXBToken:
    """One token from the EXB master sequence."""
    __slots__ = (
        "form", "is_special", "has_slash_after",
        "exb_utt_id",      # TEI-style utterance ID, e.g. "TOR_C_0031-u1"
        "exb_utt_idx",     # 0-based utterance index
        "exb_tok_pos",     # 1-based w-number within utterance; None for specials
        "utt_start_time",  # float seconds
        "utt_end_time",    # float seconds
        "is_last_in_utt",  # True for the last real token of its EXB utterance
        "speaker_abbr",    # EXB tier display-name (speaker abbreviation)
    )

    def __init__(self, form, is_special, has_slash_after,
                 exb_utt_id, exb_utt_idx, exb_tok_pos,
                 utt_start_time, utt_end_time, is_last_in_utt,
                 speaker_abbr=""):
        self.form            = form
        self.is_special      = is_special
        self.has_slash_after = has_slash_after
        self.exb_utt_id      = exb_utt_id
        self.exb_utt_idx     = exb_utt_idx
        self.exb_tok_pos     = exb_tok_pos
        self.utt_start_time  = utt_start_time
        self.utt_end_time    = utt_end_time
        self.is_last_in_utt  = is_last_in_utt
        self.speaker_abbr    = speaker_abbr


def build_exb_master(utterances: List[Dict]) -> List[EXBToken]:
    """
    Concatenate all EXB tokens (all utterances, sorted by start_time) into
    one flat list.

    Utterances are numbered u1, u2, … matching TEI XML (TOR_C_0031-u1 etc.).
    Each real (non-special) token gets a 1-based w-position (exb_tok_pos)
    matching TEI <w> element numbering. The last real token of each utterance
    is flagged with is_last_in_utt=True for forced-alignment boundary marking.
    """
    master: List[EXBToken] = []
    file_id = utterances[0]["file_id"] if utterances else "unknown"

    for i, utt in enumerate(utterances):
        exb_utt_id   = f"{file_id}-u{i+1}"   # TEI format: no zero-padding
        start_time   = utt["start_time"]
        end_time     = utt["end_time"]
        tokens       = extract_exb_tokens(utt["raw_text"])

        real_tok_counter = 0
        utt_tokens: List[EXBToken] = []

        for form, is_special, has_slash_after in tokens:
            if is_special:
                tok_pos = None
            else:
                real_tok_counter += 1
                tok_pos = real_tok_counter

            et = EXBToken(
                form, is_special, has_slash_after,
                exb_utt_id, i, tok_pos,
                start_time, end_time,
                False,                   # is_last_in_utt — set in post-pass below
                utt["speaker_abbr"],
            )
            utt_tokens.append(et)

        # Mark the last real token of this utterance
        for et in reversed(utt_tokens):
            if not et.is_special:
                et.is_last_in_utt = True
                break

        master.extend(utt_tokens)

    return master


class CorrToken:
    """One token from the corrected CoNLL-U flat sequence."""
    __slots__ = ("form", "sent_idx", "tok_idx")

    def __init__(self, form, sent_idx, tok_idx):
        self.form     = form
        self.sent_idx = sent_idx
        self.tok_idx  = tok_idx


_PAUSE_FORMS = frozenset({"\u00b7", "\u00b7\u00b7", "\u00b7\u00b7\u00b7"})  # ·, ··, ···


def is_pause_token(row: List[str]) -> bool:
    """Return True for synthetic pause marker rows (·, ··, ···) from sentence_.ipynb."""
    return row[1] in _PAUSE_FORMS or "IsPause=yes" in row[9]


def build_corr_master(sentences: List[Dict]) -> List[CorrToken]:
    """Build flat corrected token sequence, skipping pause marker pseudo-tokens."""
    master: List[CorrToken] = []
    for si, sent in enumerate(sentences):
        for ti, row in enumerate(sent["tokens"]):
            if not is_pause_token(row):
                master.append(CorrToken(row[1], si, ti))
    return master


def align_sequences(
    exb_master:  List[EXBToken],
    corr_master: List[CorrToken],
) -> List[Optional[EXBToken]]:
    """
    Align the corrected token sequence to the EXB token sequence.

    Returns a list the same length as corr_master, where each entry is the
    matching EXBToken (or None if unmatched).

    Tries exact 1-to-1 alignment first; falls back to difflib SequenceMatcher
    for robust handling of minor edits.
    """
    exb_forms  = [t.form for t in exb_master]
    corr_forms = [t.form for t in corr_master]

    if exb_forms == corr_forms:
        # Perfect match — simple zip
        return list(exb_master)

    # Fallback: use SequenceMatcher to find the best alignment
    print("  ⚠  Sequences differ — using SequenceMatcher alignment.")
    sm = difflib.SequenceMatcher(None, exb_forms, corr_forms, autojunk=False)
    result: List[Optional[EXBToken]] = [None] * len(corr_master)

    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            for di, (ei, ci) in enumerate(zip(range(i1, i2), range(j1, j2))):
                result[ci] = exb_master[ei]
        elif tag == "replace":
            # Map as many as possible 1-to-1 even in replace blocks
            for di, (ei, ci) in enumerate(zip(range(i1, i2), range(j1, j2))):
                result[ci] = exb_master[ei]
            # Remaining corrected tokens in a longer replace block stay None

    return result


# ── Sentence rebuilder ─────────────────────────────────────────────────────────

_BLANK_X_ROW = ["_", "/", "_", "X", "_", "_", "_", "_", "_", "_"]


def append_misc(existing: str, key: str, value: str) -> str:
    """Append key=value to a CoNLL-U MISC field."""
    if existing == "_":
        return f"{key}={value}"
    return f"{existing}|{key}={value}"


_SPEAKER_STRIP_KEYS = frozenset({
    "speaker", "speaker_abbr", "speaker_age", "speaker_gender", "speaker_education",
    "speakers",
})


def rebuild_sentences(
    sentences:    List[Dict],
    corr_master:  List[CorrToken],
    alignment:    List[Optional[EXBToken]],
    split_utt_ids: set,
    speaker_meta: Dict[str, Dict[str, str]],
    geo_coords:   str = "",
) -> List[Dict]:
    """
    For each corrected sentence:
      • Look up the EXB provenance of each token from the alignment.
      • Append OrigUttID=<exb_utt_id> and EXBTokenRef=<utt_id>-w<pos>
        to each token's MISC field.
      • Append UttEnd=yes to the last real token of each source utterance
        (forced-alignment boundary marker).
      • Insert '/' X-tokens wherever has_slash_after is True.
      • Recompute # start_time and # end_time from authoritative EXB values
        (min utt_start / max utt_end across all aligned source utterances).
      • Add # timing_precision = exact | approximate_bounds to document whether
        the time span is exact (all source utterances fully contained in this
        sentence) or only an outer bounding box (at least one source utterance
        is split across multiple sentences, so the true span is narrower).
      • Add sentence-level # EXB_sources and # EXB_token_map comments.
      • Re-number token IDs.
    """
    # Index alignment by (sent_idx, tok_idx) → EXBToken
    align_map: Dict[Tuple[int, int], Optional[EXBToken]] = {}
    for ct, et in zip(corr_master, alignment):
        align_map[(ct.sent_idx, ct.tok_idx)] = et

    new_sentences: List[Dict] = []

    for si, sent in enumerate(sentences):
        orig_rows = sent["tokens"]
        new_rows:  List[List[str]] = []

        # Per-token EXB info
        exb_tokens_for_sent: List[Tuple[int, Optional[EXBToken]]] = [
            (ti, align_map.get((si, ti))) for ti in range(len(orig_rows))
        ]

        # --- Build new_rows with provenance in MISC + slash insertions ---
        for ti, row in enumerate(orig_rows):
            et = align_map.get((si, ti))

            new_row = list(row)

            # Pause marker pseudo-tokens (·, ··, ···) are passed through as-is —
            # they have no EXB counterpart and carry their own IsPause/PauseDur MISC.
            if is_pause_token(row):
                new_rows.append(new_row)
                continue

            # Strip MulText= from MISC (redundant with XPOS column)
            misc = re.sub(r'MulText=[^|]*\|?', '', new_row[9]).strip('|')
            new_row[9] = misc if misc else '_'

            if et is not None:
                # Utterance-level reference
                new_row[9] = append_misc(new_row[9], "OrigUttID", et.exb_utt_id)
                # Token-level TEI reference (only for real, non-special tokens)
                if et.exb_tok_pos is not None:
                    tok_ref = f"{et.exb_utt_id}-w{et.exb_tok_pos}"
                    new_row[9] = append_misc(new_row[9], "EXBTokenRef", tok_ref)
                # Forced-alignment boundary: last real token of its utterance
                if et.is_last_in_utt:
                    new_row[9] = append_misc(new_row[9], "UttEnd", "yes")

            new_rows.append(new_row)

            # Insert '/' X-token if this EXB token had a trailing slash
            if et is not None and et.has_slash_after:
                slash_row = list(_BLANK_X_ROW)
                # Slash pseudo-tokens get utterance ID only (not a <w> element)
                slash_row[9] = f"OrigUttID={et.exb_utt_id}"
                new_rows.append(slash_row)

        # Re-number IDs
        for idx, row in enumerate(new_rows, start=1):
            row[0] = str(idx)

        # --- Build new # text (all tokens including inserted slashes) ---
        new_text = " ".join(row[1] for row in new_rows)

        # --- Build # EXB_sources comment (unique, ordered EXB utt_ids) ---
        seen_src: List[str] = []
        seen_abbrs: List[str] = []
        for _, et in exb_tokens_for_sent:
            if et is not None and et.exb_utt_id not in seen_src:
                seen_src.append(et.exb_utt_id)
            if et is not None and et.speaker_abbr and et.speaker_abbr not in seen_abbrs:
                seen_abbrs.append(et.speaker_abbr)
        exb_sources_str = "|".join(seen_src) if seen_src else "unknown"

        # --- Build # EXB_token_map comment with full TEI token references ---
        # Format: "1:TOR_C_0031-u1-w1 2:TOR_C_0031-u1-w2 ..."
        # Slash pseudo-tokens use the utterance ID only (no w-number).
        # Pause marker pseudo-tokens (·) are skipped — they have no EXB counterpart.
        token_map_parts: List[str] = []
        new_pos = 1
        for ti, row in enumerate(orig_rows):
            if is_pause_token(row):
                new_pos += 1   # counts toward renumbered ID; no map entry
                continue
            et = align_map.get((si, ti))
            if et is not None and et.exb_tok_pos is not None:
                ref = f"{et.exb_utt_id}-w{et.exb_tok_pos}"
            elif et is not None:
                ref = et.exb_utt_id   # special/unpositioned token
            else:
                ref = "?"
            token_map_parts.append(f"{new_pos}:{ref}")
            new_pos += 1
            if et is not None and et.has_slash_after:
                # The inserted slash pseudo-token
                token_map_parts.append(f"{new_pos}:{et.exb_utt_id}")
                new_pos += 1
        token_map_str = " ".join(token_map_parts)

        # --- Recompute start/end times from authoritative EXB values ---
        exb_toks_matched = [et for _, et in exb_tokens_for_sent if et is not None]
        if exb_toks_matched:
            new_start = round(min(et.utt_start_time for et in exb_toks_matched), 3)
            new_end   = round(max(et.utt_end_time   for et in exb_toks_matched), 3)
        else:
            new_start = new_end = None

        # --- Timing precision: exact iff no source utterance is split ---
        has_split_utt = any(uid in split_utt_ids for uid in seen_src)
        timing_precision = "approximate_bounds" if has_split_utt else "exact"

        # --- Update / inject meta comments ---
        # Prepare speaker lines to inject after # text
        speaker_lines: List[str] = []
        if len(seen_abbrs) == 1:
            abbr = seen_abbrs[0]
            sm = speaker_meta.get(abbr, {})
            if sm:
                speaker_lines = [
                    f"# speaker = {sm.get('speaker', '_')}",
                    f"# speaker_abbr = {sm.get('speaker_abbr', abbr)}",
                    f"# speaker_age = {sm.get('speaker_age', '_')}",
                    f"# speaker_gender = {sm.get('speaker_gender', '_')}",
                    f"# speaker_education = {sm.get('speaker_education', '_')}",
                ]
        elif len(seen_abbrs) > 1:
            speaker_lines = [f"# speakers = {'|'.join(seen_abbrs)}"]

        new_meta: List[str] = []
        text_replaced  = False
        start_replaced = False
        end_replaced   = False
        for line in sent["meta"]:
            # Strip old speaker lines — will be re-injected fresh below
            if line.startswith("# ") and " = " in line:
                key = line[2:].split(" = ", 1)[0]
                if key in _SPEAKER_STRIP_KEYS:
                    continue
            if line.startswith("# text = "):
                new_meta.append(f"# text = {new_text}")
                text_replaced = True
                new_meta.extend(speaker_lines)
            elif line.startswith("# start_time = ") and new_start is not None:
                new_meta.append(f"# start_time = {new_start}")
                start_replaced = True
            elif line.startswith("# end_time = ") and new_end is not None:
                new_meta.append(f"# end_time = {new_end}")
                end_replaced = True
            else:
                new_meta.append(line)
                # Inject geographical_coordinates right after location
                if geo_coords and line.startswith("# location = "):
                    new_meta.append(f"# geographical_coordinates = {geo_coords}")
        if not text_replaced:
            new_meta.insert(0, f"# text = {new_text}")
            for i, sl in enumerate(speaker_lines, start=1):
                new_meta.insert(i, sl)
        if not start_replaced and new_start is not None:
            new_meta.append(f"# start_time = {new_start}")
        if not end_replaced and new_end is not None:
            new_meta.append(f"# end_time = {new_end}")

        new_meta.append(f"# timing_precision = {timing_precision}")
        new_meta.append(f"# EXB_sources = {exb_sources_str}")
        new_meta.append(f"# EXB_token_map = {token_map_str}")

        new_sentences.append({"meta": new_meta, "tokens": new_rows})

    return new_sentences


# ── CoNLL-U writer ─────────────────────────────────────────────────────────────

def write_conllu(output_path: Path, global_header: List[str], sentences: List[Dict]) -> None:
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
        description="Enrich a corrected/re-segmented CoNLL-U with EXB token provenance and re-inserted '/' markers."
    )
    parser.add_argument(
        "--conllu",
        default="REMBERT_TOR_C_0031.pred.corrected.conllup",
        help="Path to the corrected CoNLL-U input file",
    )
    parser.add_argument(
        "--exb",
        default="TOR_C_0031.exb",
        help="Path to the original EXB transcript",
    )
    parser.add_argument(
        "--output",
        default="REMBERT_TOR_C_0031.corrected.enriched.conllu",
        help="Output path for the enriched CoNLL-U",
    )
    parser.add_argument(
        "--pred",
        default="REMBERT_TOR_C_0031.pred (1).conllu",
        help="Path to the original (un-corrected) pred CoNLL-U, used to recover speaker metadata",
    )
    parser.add_argument(
        "--geo",
        default="",
        help="Geographical coordinates to inject (e.g. '43.6396024, 22.3007863')",
    )
    args = parser.parse_args()

    # Resolve paths relative to this script's directory when not absolute,
    # so the script works regardless of the working directory (e.g. VS Code runner).
    script_dir  = Path(__file__).parent
    conllu_path = Path(args.conllu)
    exb_path    = Path(args.exb)
    output_path = Path(args.output)
    pred_path   = Path(args.pred)
    if not conllu_path.is_absolute():
        conllu_path = script_dir / conllu_path
    if not exb_path.is_absolute():
        exb_path = script_dir / exb_path
    if not output_path.is_absolute():
        output_path = script_dir / output_path
    if not pred_path.is_absolute():
        pred_path = script_dir / pred_path

    # ── Parse inputs ──────────────────────────────────────────────────────────
    print(f"Reading CoNLL-U : {conllu_path}")
    global_header, sentences = parse_conllu(conllu_path)
    n_corr_toks = sum(len(s["tokens"]) for s in sentences)
    print(f"  → {len(sentences)} sentences, {n_corr_toks} tokens")

    print(f"Reading EXB     : {exb_path}")
    utterances = parse_exb_utterances(exb_path)
    print(f"  → {len(utterances)} utterances")

    print(f"Loading speaker metadata from : {pred_path}")
    speaker_meta = load_speaker_metadata(pred_path) if pred_path.exists() else {}
    print(f"  → {len(speaker_meta)} speaker(s): {list(speaker_meta.keys())}")

    # ── Build flat sequences ───────────────────────────────────────────────────
    print("Building flat token sequences …")
    exb_master  = build_exb_master(utterances)
    corr_master = build_corr_master(sentences)
    print(f"  EXB  flat tokens : {len(exb_master)}")
    print(f"  Corr flat tokens : {len(corr_master)}")

    # ── Align ─────────────────────────────────────────────────────────────────
    print("Aligning sequences …")
    alignment = align_sequences(exb_master, corr_master)

    # Report alignment quality
    n_matched = sum(1 for et in alignment if et is not None)
    n_unmatched = len(alignment) - n_matched
    print(f"  Matched   : {n_matched}/{len(alignment)}")
    if n_unmatched:
        print(f"  ⚠  Unmatched : {n_unmatched} tokens")
        # Show the first few mismatches
        for ci, (ct, et) in enumerate(zip(corr_master, alignment)):
            if et is None:
                print(f"    pos {ci+1}: corr='{ct.form}'  →  no EXB match")
            elif et.form != ct.form:
                print(f"    pos {ci+1}: corr='{ct.form}'  ≠  exb='{et.form}' ({et.exb_utt_id})")
            if ci > 20:
                print("    … (more mismatches truncated)")
                break

    # ── Identify slash insertions ──────────────────────────────────────────────
    slash_sites = [(et.exb_utt_id, ct.sent_idx)
                   for ct, et in zip(corr_master, alignment)
                   if et is not None and et.has_slash_after]
    if slash_sites:
        print(f"\n'/' truncation markers to re-insert ({len(slash_sites)} sites):")
        for (exb_uid, si) in slash_sites:
            corr_sid = get_meta_value(sentences[si]["meta"], "sent_id") or f"sent[{si}]"
            print(f"  EXB {exb_uid}  →  corrected sentence {corr_sid}")
    else:
        print("\nNo stripped '/' markers found.")

    # ── Detect split utterances (one EXB utt → multiple CoNLL-U sentences) ─────
    from collections import defaultdict
    utt_to_sent_indices: Dict[str, set] = defaultdict(set)
    for ct, et in zip(corr_master, alignment):
        if et is not None:
            utt_to_sent_indices[et.exb_utt_id].add(ct.sent_idx)
    split_utt_ids = {uid for uid, idxs in utt_to_sent_indices.items() if len(idxs) > 1}
    if split_utt_ids:
        print(f"\n⚠  {len(split_utt_ids)} split utterance(s) detected (timing will be approximate_bounds):")
        for uid in sorted(split_utt_ids, key=lambda x: int(x.split('-u')[1])):
            sids = sorted(utt_to_sent_indices[uid])
            sent_ids = [get_meta_value(sentences[i]["meta"], "sent_id") or f"sent[{i}]"
                        for i in sids]
            print(f"    {uid}  →  {sent_ids}")
    else:
        print("\nNo split utterances — all sentence timings are exact.")

    # ── Rebuild sentences ──────────────────────────────────────────────────────
    print("\nRebuilding sentences with provenance mapping …")
    new_sentences = rebuild_sentences(sentences, corr_master, alignment, split_utt_ids, speaker_meta, geo_coords=args.geo)

    # ── Write output ───────────────────────────────────────────────────────────
    write_conllu(output_path, global_header, new_sentences)
    n_new_toks = sum(len(s["tokens"]) for s in new_sentences)
    print(f"\nWritten: {output_path}")
    print(f"  {len(new_sentences)} sentences, {n_new_toks} tokens "
          f"(+{n_new_toks - n_corr_toks} inserted '/' tokens)")


if __name__ == "__main__":
    main()
