#!/usr/bin/env python3
"""
conllu_to_tei.py
----------------
Convert enriched CoNLL-U files to TEI XML (spoken corpus format).

Stress-marked word forms are recovered from the old TOR_C TEI files where
available; LUZ files use plain CoNLL-U FORM column.

Usage:
    python conllu_to_tei.py \
        --conllu  topic_tagged_conllu/ \
        --old-tei "TOR_C_TEI_transcripts 2/" \
        --output  tei_output/
"""

import re
import sys
import argparse
import unicodedata
from collections import defaultdict
from html import escape as xe
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import xml.etree.ElementTree as ET

# ── Constants ──────────────────────────────────────────────────────────────────

TEI_NS  = "http://www.tei-c.org/ns/1.0"
XML_NS  = "http://www.w3.org/XML/1998/namespace"
PAUSE_RE = re.compile(r'^[·•]+$')
VOCAL_RE = re.compile(r'^\(\((\w+)\)\)$')   # ((laugh)), ((cough)), ((filler)) …
NOISE_RE = re.compile(r'^\(\(noise\)\)$', re.I)
UNCLEAR_RE = re.compile(r'^\(\([?x]+\)\)$', re.I)

# Per-file speaker ID mapping: CoNLL-U abbreviation → canonical TEI id.
# Built by matching CoNLL-U start_time values to EXB transcription-tier event
# times (tight tolerance 0.15 s), then voting per file.
# Researcher IDs (RS_*) map to themselves; [v] suffix is stripped at lookup time.
PER_FILE_SPEAKER_MAP: Dict[str, Dict[str, str]] = {
    "TOR_C_0001": {"OS_1": "TIM_SPK_0001", "RS_BS": "RS_BS", "RS_DK": "RS_DK"},
    "TOR_C_0002": {"DRV_1": "TIM_SPK_0002", "RS_BR": "RS_BR", "RS_SDJP": "RS_SDJP"},
    "TOR_C_0003": {"BALIN_1": "TIM_SPK_0003", "BALIN_2": "TIM_SPK_0004", "RS_BS": "RS_BS"},
    "TOR_C_0004": {"CUS_1": "TIM_SPK_0005", "CUS_2": "TIM_SPK_0006", "RS_BS": "RS_BS"},
    "TOR_C_0005": {"GORZUN_1": "TIM_SPK_0007", "RS_SC": "RS_SC", "RS_TV": "RS_TV"},
    "TOR_C_0006": {"NOVOK_1": "TIM_SPK_0008", "RS_BS": "RS_BS"},
    "TOR_C_0007": {"RS_BS": "RS_BS", "TRNOV_1": "TIM_SPK_0009", "TRNOV_2": "TIM_SPK_0010"},
    "TOR_C_0008": {"RS_BS": "RS_BS", "TRNOV_1": "TIM_SPK_0009"},
    "TOR_C_0009": {"JEL_1": "TIM_SPK_0011", "RS_KE": "RS_KE"},
    "TOR_C_0010": {"LEP_8": "TIM_SPK_0012", "RS_KE": "RS_KE"},
    "TOR_C_0011": {"RS_KE [v]": "RS_KE", "TRG_1 [v]": "TIM_SPK_0013", "TRG_2 [v]": "TIM_SPK_0014"},
    "TOR_C_0012": {"RS_KE": "RS_KE", "ZUK_1": "TIM_SPK_0015"},
    "TOR_C_0013": {"RS_KE": "RS_KE", "ZUK_1": "TIM_SPK_0015", "ZUK_2": "TIM_SPK_0016"},
    "TOR_C_0014": {"RS_SDJP": "RS_SDJP", "VRA_1": "TIM_SPK_0017"},
    "TOR_C_0015": {"RS_BS": "RS_BS", "RS_TV": "RS_TV", "ZLN_1": "TIM_SPK_0018", "ZLN_2": "TIM_SPK_0019"},
    "TOR_C_0016": {"RS_DPe": "RS_DPe", "RS_SDJ": "RS_SDJ", "VAS_1": "TIM_SPK_0020"},
    "TOR_C_0017": {"RS_BR": "RS_BR", "TIJ_1": "TIM_SPK_0021", "TIJ_2": "TIM_SPK_0022", "TIJ_3": "TIM_SPK_0023", "TIJ_4": "TIM_SPK_0024"},
    "TOR_C_0018": {"DREC_1": "TIM_SPK_0025", "DREC_2": "TIM_SPK_0026", "DREC_3": "TIM_SPK_0027", "RS_TV": "RS_TV"},
    "TOR_C_0019": {"GSOK_1": "TIM_SPK_0028", "GSOK_2": "TIM_SPK_0029", "GSOK_3": "TIM_SPK_0030", "RS_KE": "RS_KE", "RS_TV": "RS_TV"},
    "TOR_C_0020": {"MI_1": "TIM_SPK_0031", "MI_2": "TIM_SPK_0032", "MI_3": "TIM_SPK_0033", "MI_4": "TIM_SPK_0034", "RS_JMP": "RS_JMP", "RS_TV": "RS_TV"},
    "TOR_C_0021": {"RB_1": "TIM_SPK_0035", "RS_TV": "RS_TV"},
    "TOR_C_0022": {"REP_1": "TIM_SPK_0036", "RS_BR": "RS_BR", "RS_BS": "RS_BS"},
    "TOR_C_0023": {"RS_BR": "RS_BR", "RS_TV": "RS_TV", "SES_1": "TIM_SPK_0037", "SES_2": "TIM_SPK_0038", "SES_3": "TIM_SPK_0039"},
    "TOR_C_0024": {"BUCJ_1 [v]": "TIM_SPK_0040", "BUCJ_2 [v]": "TIM_SPK_0041", "BUCJ_3 [v]": "TIM_SPK_0042", "RS_TV [v]": "RS_TV"},
    "TOR_C_0025": {"KOZ_1": "TIM_SPK_0043", "KOZ_2": "TIM_SPK_0044", "RS_BS": "RS_BS"},
    "TOR_C_0026": {"GRL_1": "TIM_SPK_0045", "RS_KE": "RS_KE", "RS_MW": "RS_MW"},
    "TOR_C_0027": {"KAND_1": "TIM_SPK_0046", "RS_BS": "RS_BS"},
    "TOR_C_0028": {"INO_1": "TIM_SPK_0047", "RS_BR": "RS_BR", "RS_BS": "RS_BS", "RS_SC": "RS_SC"},
    "TOR_C_0029": {"INO_4": "TIM_SPK_0049", "RS_BR": "RS_BR", "RS_BS": "RS_BS", "RS_SC": "RS_SC"},
    "TOR_C_0030": {"JAK_1": "TIM_SPK_0050", "JAK_2": "TIM_SPK_0051", "JAK_3": "TIM_SPK_0052", "RS_BS": "RS_BS"},
    "TOR_C_0031": {"JAK_1": "TIM_SPK_0050", "JAK_2": "TIM_SPK_0051", "RS_BS": "RS_BS"},
    "TOR_C_0032": {"JAK_1": "TIM_SPK_0050", "JAK_2": "TIM_SPK_0051", "RS_BS": "RS_BS"},
    "TOR_C_0033": {"BAL-TRG_1": "TIM_SPK_0054", "BAL-TRG_2": "RS_SC", "RS_SC": "RS_SC"},
    "TOR_C_0034": {"PON_1": "TIM_SPK_0056", "PON_2": "TIM_SPK_0057", "PON_3": "TIM_SPK_0058", "RS_KE": "RS_KE", "RS_TV": "RS_TV"},
    "TOR_C_0035": {"RS_TV": "RS_TV", "SAS_1": "TIM_SPK_0059", "SAS_2": "TIM_SPK_0060"},
    "TOR_C_0036": {"RS_TV": "RS_TV", "SAS_1": "TIM_SPK_0059", "SAS_2": "TIM_SPK_0060"},
    "TOR_C_0037": {"RS_SDJ": "RS_SDJ", "VLAH_1": "TIM_SPK_0061"},
    "TOR_C_0038": {"JAN_1": "TIM_SPK_0063", "JAN_2": "TIM_SPK_0064", "JAN_3": "TIM_SPK_0065", "RS_SDJ": "RS_SDJ", "RS_SP": "RS_SP"},
    "TOR_C_0039": {"JAN_1": "TIM_SPK_0063", "JAN_2": "TIM_SPK_0064", "JAN_3": "TIM_SPK_0065", "JAN_4": "TIM_SPK_0066", "JAN_5": "TIM_SPK_0067", "RS_SDJ": "RS_SDJ", "RS_SP": "RS_SP"},
    "TOR_C_0040": {"JAN_1": "TIM_SPK_0063", "RS_SP": "RS_SP"},
    "TOR_C_0041": {"DKAM_1": "TIM_SPK_0068", "DKAM_2": "TIM_SPK_0069", "DKAM_3": "TIM_SPK_0070", "RS_SC": "RS_SC"},
    "TOR_C_0042": {"JAL_1": "TIM_SPK_0071", "JAL_2": "TIM_SPK_0072", "JAL_3": "TIM_SPK_0073", "JAL_4": "TIM_SPK_0074", "RS_BS": "RS_BS", "RS_TV": "RS_TV"},
    "TOR_C_0043": {"BAL_1": "TIM_SPK_0076", "RS_BR": "RS_BR", "RS_DP": "RS_DP", "RS_KE": "RS_KE"},
    "TOR_C_0044": {"BAL_1": "TIM_SPK_0076", "RS_BR": "RS_BR", "RS_DP": "RS_DP", "RS_KE": "RS_KE"},
    "TOR_C_0045": {"MIN_1": "TIM_SPK_0075", "MIN_2": "TIM_SPK_0077", "MIN_3": "TIM_SPK_0078", "MIN_4": "TIM_SPK_0079", "MIN_5": "TIM_SPK_0080", "MIN_6": "TIM_SPK_0081", "RS_SDJ": "RS_SDJ"},
    "TOR_C_0046": {"Mar_1": "TIM_SPK_0082", "Mar_2": "TIM_SPK_0083", "Mar_3": "TIM_SPK_0084", "RS_SP": "RS_SP"},
    "TOR_C_0047": {"Deb_2": "TIM_SPK_0085", "Deb_3": "TIM_SPK_0086", "RS_TV": "RS_TV"},
    "TOR_C_0048": {"RS_BR": "RS_BR", "RS_BS": "RS_BS", "SAR_1": "TIM_SPK_0087"},
    "TOR_C_0049": {"LEP_1": "TIM_SPK_0088", "LEP_2": "TIM_SPK_0089", "LEP_3": "TIM_SPK_0090", "LEP_4": "TIM_SPK_0091", "LEP_5": "TIM_SPK_0092", "RS_KE": "RS_KE", "RS_TV": "RS_TV"},
    "TOR_C_0050": {"RS_TV": "RS_TV", "STI_1": "TIM_SPK_0095"},
    "TOR_C_0051": {"ALD_1": "TIM_SPK_0096", "ALD_2": "TIM_SPK_0097", "RS_BS": "RS_BS"},
    "TOR_C_0052": {"PRIC_1": "TIM_SPK_0098", "PRIC_2": "TIM_SPK_0099", "RS_BR": "RS_BR", "RS_SDJP": "RS_SDJP"},
    "TOR_C_0053": {"RS_MW": "RS_MW", "RS_SDJ": "RS_SDJ", "VLAH_1": "TIM_SPK_0061", "VLAH_2": "TIM_SPK_0062"},
    "TOR_C_0054": {"CRVR_1": "TIM_SPK_0101", "CRVR_3": "TIM_SPK_0103", "RS_KE": "RS_KE"},
    "TOR_C_0055": {"CRVR_1": "TIM_SPK_0101", "CRVR_2": "TIM_SPK_0102", "CRVR_3": "TIM_SPK_0103", "RS_KE": "RS_KE"},
    "TOR_C_0056": {"GABR_1": "TIM_SPK_0104", "GABR_2": "TIM_SPK_0105", "RS_BR": "RS_BR", "RS_BS": "RS_BS"},
    "TOR_C_0057": {"GABR_1": "TIM_SPK_0104", "RS_BR": "RS_BR", "RS_BS": "RS_BS"},
    "TOR_C_0058": {"GABR_1": "TIM_SPK_0104", "RS_BR": "RS_BR", "RS_BS": "RS_BS"},
    "TOR_C_0059": {"GUS_1": "TIM_SPK_0106", "GUS_2": "TIM_SPK_0107", "RS_DP": "RS_DP"},
    "TOR_C_0060": {"GUS_1": "TIM_SPK_0106", "GUS_2": "TIM_SPK_0108", "RS_DP": "RS_DP"},
    "TOR_C_0061": {"KREN_6": "TIM_SPK_0110", "KREN_MOM": "TIM_SPK_0109", "RS_BS": "RS_BS", "RS_TV": "RS_TV"},
    "TOR_C_0062": {"ORES_3": "TIM_SPK_0111", "ORES_4": "TIM_SPK_0111", "RS_TV": "RS_TV"},
    "TOR_C_0063": {"ORES_1": "TIM_SPK_0113", "RS_KE": "RS_KE", "RS_TV": "RS_TV"},
    "TOR_C_0064": {"PETR_1": "TIM_SPK_0115", "PETR_2": "TIM_SPK_0116", "RS_TV": "RS_TV"},
    "TOR_C_0065": {"JAN_6": "TIM_SPK_0118", "RS_SP": "RS_SP"},
    "TOR_C_0066": {"RADIC_1": "TIM_SPK_0119", "RADIC_2": "TIM_SPK_0120", "RS_SC": "RS_SC"},
    "TOR_C_0067": {"RGO_1": "TIM_SPK_0121", "RGO_2": "TIM_SPK_0122", "RS_KE": "RS_KE", "RS_TV": "RS_TV"},
    "TOR_C_0068": {"RS_BR": "RS_BR", "RS_BS [v]": "RS_BS", "STKAL_1": "TIM_SPK_0123", "STKAL_2": "TIM_SPK_0123"},
    "TOR_C_0069": {"RS_BR": "RS_BR", "RS_BS": "RS_BS", "STKAL_3": "TIM_SPK_0125"},
    "TOR_C_0070": {"RS_KE": "RS_KE", "VRB_1": "TIM_SPK_0127", "VRB_2": "TIM_SPK_0128"},
    "TOR_C_0071": {"GLOG_1": "TIM_SPK_0129", "GLOG_2": "TIM_SPK_0130", "RS_TV": "RS_TV"},
    "TOR_C_0072": {"LESK_3": "TIM_SPK_0131", "RS_TV": "RS_TV"},
    "TOR_C_0073": {"GRADIS_1": "TIM_SPK_0133", "GRADIS_2": "TIM_SPK_0134", "RS_KE": "RS_KE"},
    "TOR_C_0074": {"LESK_1": "TIM_SPK_0135", "RS_SD": "RS_SD"},
    "TOR_C_0075": {"BALBERIL_1": "TIM_SPK_0137", "BALBERIL_2": "TIM_SPK_0138", "RS_BS": "RS_BS", "RS_TV": "RS_TV"},
    "TOR_C_0076": {"LOK_3": "TIM_SPK_0139", "RS_DP": "RS_DP"},
    "TOR_C_0077": {"LOK_3": "TIM_SPK_0139", "RS_DP": "RS_DP"},
    "TOR_C_0078": {"LOK_3": "TIM_SPK_0139", "RS_DP": "RS_DP"},
    "TOR_C_0079": {"RS_BS": "RS_BS", "STAROK_1": "TIM_SPK_0141"},
    "TOR_C_0080": {"BR_1": "TIM_SPK_0142", "BR_2": "TIM_SPK_0143", "RS_JMP": "RS_JMP", "RS_KE": "RS_KE", "RS_TV": "RS_TV"},
    "TOR_C_0081": {"BUL_1": "TIM_SPK_0145", "BUL_2": "TIM_SPK_0146", "RS_TV": "RS_TV"},
    "TOR_C_0082": {"RS_TV": "RS_TV", "ZOR_1": "TIM_SPK_0147", "ZOR_2": "TIM_SPK_0148", "ZOR_3": "TIM_SPK_0149", "ZOR_4": "TIM_SPK_0150"},
    "TOR_C_0083": {"RS_BS": "RS_BS", "RS_KE": "RS_KE", "SEL_1": "TIM_SPK_0151", "SEL_2": "TIM_SPK_0152", "SEL_3": "TIM_SPK_0153"},
    "TOR_C_0084": {"DRENOV_1": "TIM_SPK_0154", "DRENOV_2": "TIM_SPK_0155", "DRENOV_3": "TIM_SPK_0156", "RS_TV": "RS_TV"},
    "TOR_C_0085": {"DRENOV_4": "TIM_SPK_0157", "RS_KE": "RS_KE"},
    "TOR_C_0086": {"DRENOV_4": "TIM_SPK_0157", "DRENOV_5": "TIM_SPK_0158", "RS_KE": "RS_KE"},
    "TOR_C_0087": {"DRENOV_4": "TIM_SPK_0157", "RS_KE": "RS_KE"},
    "TOR_C_0088": {"DRENOV_4": "TIM_SPK_0157", "RS_KE": "RS_KE"},
    "TOR_C_0089": {"RS_MM": "RS_MM", "TEHS_5": "TIM_SPK_0159", "TEHS_8": "TIM_SPK_0166"},
    "TOR_C_0090": {"RS_MM": "RS_MM", "TEHS_2": "TIM_SPK_0160", "TEHS_3": "TIM_SPK_0161"},
    "TOR_C_0091": {"RS_MM": "RS_MM", "TEHS_3": "TIM_SPK_0161", "TEHS_9": "TIM_SPK_0167"},
    "TOR_C_0092": {"RS_MM": "RS_MM", "TEHS_6": "TIM_SPK_0162"},
    "TOR_C_0093": {"RS_MM": "RS_MM", "TEHS_4": "TIM_SPK_0163"},
    "TOR_C_0094": {"RS_MM": "RS_MM", "TEHS_1": "TIM_SPK_0164"},
    "TOR_C_0095": {"RS_MM": "RS_MM", "TEHS_7": "TIM_SPK_0165"},
    "TOR_C_0096": {"RS_KE": "RS_KE", "TRG_3": "TIM_SPK_0168"},
}


def resolve_speaker_id(file_id: str, abbr: str) -> str:
    """Return the canonical TEI speaker ID for a CoNLL-U speaker abbreviation."""
    file_map = PER_FILE_SPEAKER_MAP.get(file_id, {})
    mapped = file_map.get(abbr) or file_map.get(re.sub(r'\s*\[v\]\s*$', '', abbr)) or abbr
    # Strip [v] suffix from result (produces valid XML NCNames)
    return re.sub(r'\s*\[v\]\s*$', '', mapped)

CORPUS_STATIC = """\
      <funder xml:lang="en">Swiss National Science Foundation</funder>
      <funder xml:lang="en">EraNet Rus Plus</funder>
      <funder xml:lang="en">URPP Language and Space, University of Zurich</funder>
      <funder xml:lang="en">Doctoral Program Linguistics, University of Zurich</funder>
      <funder xml:lang="en">Department of Slavonic Languages and Literatures (Slavisches Seminar), University of Zurich</funder>
      <funder xml:lang="en">Ministry of Culture and Information of the Republic of Serbia</funder>
      <funder xml:lang="en">Swiss Government Excellence Scholarship for Foreign Scholars and Artists</funder>
      <respStmt>
         <resp xml:lang="en">Corpus creation</resp>
         <resp xml:lang="sr">Izrada korpusa</resp>
         <name>Teodora Vuković</name>
      </respStmt>
      <respStmt>
         <resp xml:lang="sr">Mentorisanje</resp>
         <resp xml:lang="en">Mentoring</resp>
         <name>Barbara Sonnenhauser</name>
         <name>Tanja Samardžić</name>
      </respStmt>
      <respStmt>
         <resp xml:lang="sr">Snimanje, vođenje razgovora</resp>
         <resp xml:lang="en">Recording, interviewing</resp>
         <name>Biljana_Sikimić</name>
         <name>Dejan Krstić</name>
      </respStmt>
      <respStmt>
         <resp xml:lang="sr">Osnovna transkripcija</resp>
         <resp xml:lang="en">Basic transcription</resp>
         <name>Aleksandra Pešić</name>
      </respStmt>
      <respStmt>
         <resp xml:lang="sr">Automatska anotacija</resp>
         <resp xml:lang="en">Automatic annotation</resp>
         <name>Teodora Vuković</name>
      </respStmt>"""

# ── CoNLL-U parser ─────────────────────────────────────────────────────────────

def parse_conllu(path: Path):
    global_header, sentences = [], []
    cur_meta, cur_tokens = [], []
    in_header = True
    with path.open(encoding="utf-8") as f:
        for raw in f:
            line = raw.rstrip("\n")
            if in_header and line.startswith("# global"):
                global_header.append(line)
                continue
            if in_header and line.strip() == "":
                in_header = False
                continue
            in_header = False
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
    return global_header, sentences


def get_meta(meta_lines: List[str], key: str) -> Optional[str]:
    prefix = f"# {key} = "
    for line in meta_lines:
        if line.startswith(prefix):
            return line[len(prefix):].strip()
    return None

# ── Stress-mark recovery from old TEI ─────────────────────────────────────────

def strip_accent(text: str) -> str:
    """Remove combining acute accent (U+0301) from text."""
    nfd = unicodedata.normalize("NFD", text)
    return "".join(c for c in nfd if unicodedata.category(c) != "Mn")


def build_stress_lookup(tei_dir: Path, file_id: str) -> Dict[str, str]:
    """
    Parse old TEI file for file_id and return {normalized_form: stressed_form}.
    Where multiple stressed forms exist for the same normalized form, keep the
    most frequent one (stress marks are corpus-wide patterns).
    """
    tei_path = tei_dir / f"{file_id}.xml"
    if not tei_path.exists():
        return {}
    try:
        tree = ET.parse(tei_path)
    except ET.ParseError:
        return {}
    root = tree.getroot()
    counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for w in root.iter(f"{{{TEI_NS}}}w"):
        if w.text:
            norm = strip_accent(w.text.lower())
            counts[norm][w.text] += 1
    return {norm: max(variants, key=variants.get)
            for norm, variants in counts.items()}


# ── Speaker demographics collector ────────────────────────────────────────────

def collect_speakers(sentences: List[dict], file_id: str = "") -> Dict[str, dict]:
    """Return {canonical_speaker_id: {age, gender, education, coords}} from sentence metadata."""
    speakers: Dict[str, dict] = {}
    for s in sentences:
        abbr = get_meta(s["meta"], "speaker_abbr") or get_meta(s["meta"], "speaker")
        if not abbr or abbr == "_":
            continue
        # Resolve to canonical ID for TOR_C files
        canonical = resolve_speaker_id(file_id, abbr) if file_id.startswith("TOR") else abbr
        if canonical not in speakers:
            speakers[canonical] = {"age": None, "gender": None,
                                   "education": None, "coords": None}
        rec = speakers[canonical]
        for field, key in (("age",       "speaker_age"),
                            ("gender",    "speaker_gender"),
                            ("education", "speaker_education"),
                            ("coords",    "geographical_coordinates")):
            if rec[field] is None:
                val = get_meta(s["meta"], key)
                if val and val != "_":
                    rec[field] = val
    return speakers


# ── Token classification ───────────────────────────────────────────────────────

def classify_token(parts: List[str]) -> str:
    form = parts[1]
    upos = parts[3]
    xpos = parts[4]
    misc = parts[9] if len(parts) > 9 else "_"

    if "IsPause=yes" in misc or PAUSE_RE.match(form):
        return "pause"
    if NOISE_RE.match(form):
        return "incident"
    m = VOCAL_RE.match(form)
    if m and m.group(1).lower() not in ("xxx", "?", "??"):
        return "vocal"
    if UNCLEAR_RE.match(form):
        return "unclear"
    return "word"


# ── Timeline builder ───────────────────────────────────────────────────────────

def build_timeline(file_id: str, sentences: List[dict]) -> Tuple[str, Dict[float, str]]:
    """
    Collect all unique start/end times, assign sequential IDs.
    Returns (xml_string, {time: when_id}).
    """
    times = set()
    for s in sentences:
        st = get_meta(s["meta"], "start_time")
        et = get_meta(s["meta"], "end_time")
        if st:
            times.add(float(st))
        if et:
            times.add(float(et))

    time_to_id: Dict[float, str] = {}
    lines = [f'         <timeline unit="s" origin="#{file_id}.t0" corresp="{file_id}.wav">',
             f'            <when xml:id="{file_id}.t0"/>']
    counter = 1
    for t in sorted(times):
        tid = f"{file_id}.t{counter}"
        time_to_id[t] = tid
        lines.append(f'            <when xml:id="{tid}" interval="{t}" since="#{file_id}.t0"/>')
        counter += 1
    lines.append("         </timeline>")
    return "\n".join(lines), time_to_id


# ── Utterance / word builder ───────────────────────────────────────────────────

def token_to_xml(parts: List[str], tok_class: str,
                  uid: str, w_num: int,
                  stress_lookup: Dict[str, str]) -> str:
    form  = parts[1]
    lemma = parts[2]
    upos  = parts[3]
    xpos  = parts[4]
    feats = parts[5]
    wid   = f"{uid}-w{w_num}"

    if tok_class == "pause":
        return f'<pause xml:id="{wid}"/>'
    if tok_class == "incident":
        return f'<incident xml:id="{wid}" type="noise"/>'
    if tok_class == "vocal":
        m = VOCAL_RE.match(form)
        vtype = m.group(1).lower() if m else "filler"
        return f'<vocal xml:id="{wid}" type="{vtype}"/>'
    if tok_class == "unclear":
        return f'<unclear xml:id="{wid}"/>'

    # Regular word
    norm = strip_accent(form.lower())
    stressed = stress_lookup.get(norm, form)

    attrs = f'xml:id="{wid}"'
    if lemma and lemma != "_" and lemma != form:
        attrs += f' lemma="{xe(lemma, quote=True)}"'
    if upos and upos not in ("_", "X"):
        attrs += f' pos="{xe(upos, quote=True)}"'
    if xpos and xpos not in ("_", "X"):
        attrs += f' ana="mte:{xe(xpos, quote=True)}"'
    if feats and feats != "_":
        attrs += f' msd="{xe(feats, quote=True)}"'
    return f'<w {attrs}>{xe(stressed)}</w>'


def build_utterances(file_id: str, sentences: List[dict],
                      time_to_id: Dict[float, str],
                      stress_lookup: Dict[str, str]) -> Tuple[str, dict]:
    lines = []
    u_count = 0
    stats = {"u": 0, "w": 0, "pause": 0, "unclear": 0, "vocal": 0}

    # collect all unique topics/customs for classDecl
    all_topics: set = set()
    all_customs: set = set()

    for sent in sentences:
        st = get_meta(sent["meta"], "start_time")
        et = get_meta(sent["meta"], "end_time")
        if st is None or et is None:
            continue

        start_id = time_to_id.get(float(st))
        end_id   = time_to_id.get(float(et))
        if not start_id or not end_id:
            continue

        raw_spk = get_meta(sent["meta"], "speaker_abbr") or get_meta(sent["meta"], "speaker") or "_"
        speaker = resolve_speaker_id(file_id, raw_spk) if file_id.startswith("TOR") else raw_spk
        u_count += 1
        uid = f"{file_id}-u{u_count}"
        stats["u"] += 1

        # Build ana from topic_clusters and fine_customs
        tc = get_meta(sent["meta"], "topic_clusters") or "_"
        fc = get_meta(sent["meta"], "fine_customs")   or "_"
        ana_parts = []
        if tc != "_":
            for slug in tc.split("|"):
                slug = slug.strip()
                if slug:
                    all_topics.add(slug)
                    ana_parts.append(f"#topc:{slug}")
        if fc != "_":
            for slug in fc.split("|"):
                slug = slug.strip()
                if slug:
                    all_customs.add(slug)
                    ana_parts.append(f"#cust:{slug}")

        ana_attr = f' ana="{" ".join(ana_parts)}"' if ana_parts else ""
        who_attr = f' who="#{xe(speaker, quote=True)}"' if speaker != "_" else ""

        u_open = (f'         <u xml:id="{uid}"'
                  f' start="{start_id}" end="{end_id}"'
                  f'{who_attr}{ana_attr}>')
        lines.append(u_open)

        w_num = 0
        for parts in sent["tokens"]:
            tok_class = classify_token(parts)
            w_num += 1
            tok_xml = token_to_xml(parts, tok_class, uid, w_num, stress_lookup)
            lines.append(f"            {tok_xml}")
            if tok_class == "word":
                stats["w"] += 1
            elif tok_class == "pause":
                stats["pause"] += 1
            elif tok_class == "unclear":
                stats["unclear"] += 1
            elif tok_class in ("vocal", "incident"):
                stats["vocal"] += 1

        lines.append("         </u>")

    return "\n".join(lines), stats, all_topics, all_customs


# ── Header builder ─────────────────────────────────────────────────────────────

def build_particdesc(speakers: Dict[str, dict]) -> str:
    lines = ["         <particDesc>", "            <listPerson>"]
    for abbr, info in sorted(speakers.items()):
        lines.append(f'               <person xml:id="{xe(abbr, quote=True)}">')
        if info["age"]:
            lines.append(f'                  <age>{xe(info["age"])}</age>')
        if info["gender"]:
            val = info["gender"].lower()
            lines.append(f'                  <sex value="{xe(val, quote=True)}"/>')
        if info["education"]:
            lines.append(f'                  <education>{xe(info["education"])}</education>')
        if info["coords"]:
            lines.append(f'                  <location><geo>{xe(info["coords"])}</geo></location>')
        lines.append("               </person>")
    lines += ["            </listPerson>", "         </particDesc>"]
    return "\n".join(lines)


def build_header(file_id: str, sentences: List[dict],
                  lang: str, stats: dict,
                  all_topics: set, all_customs: set,
                  speakers: Dict[str, dict] = None) -> str:
    # Derive duration from max end_time
    max_end = 0.0
    for s in sentences:
        et = get_meta(s["meta"], "end_time")
        if et:
            max_end = max(max_end, float(et))
    total_s = int(max_end)
    dur_iso = f"PT{total_s // 3600}H{(total_s % 3600) // 60}M{total_s % 60}S"

    corpus_prefix = file_id.split("_")[0] + "_" + file_id.split("_")[1]  # TOR_C or LUZ_C

    # Build classDecl if we have topics/customs
    classdecl = ""
    if all_topics or all_customs:
        cat_lines = []
        if all_topics:
            cat_lines.append('            <taxonomy xml:id="topic_clusters">')
            for slug in sorted(all_topics):
                cat_lines.append(f'               <category xml:id="topc:{xe(slug, quote=True)}"/>')
            cat_lines.append('            </taxonomy>')
        if all_customs:
            cat_lines.append('            <taxonomy xml:id="fine_customs">')
            for slug in sorted(all_customs):
                cat_lines.append(f'               <category xml:id="cust:{xe(slug, quote=True)}"/>')
            cat_lines.append('            </taxonomy>')
        classdecl = "         <classDecl>\n" + "\n".join(cat_lines) + "\n         </classDecl>\n"

    header = f"""\
  <teiHeader xml:lang="en">
      <fileDesc>
         <titleStmt>
            <title>{xe(file_id)}</title>
{CORPUS_STATIC}
         </titleStmt>
         <editionStmt>
            <edition>2.0</edition>
         </editionStmt>
         <publicationStmt>
            <publisher xml:lang="en">Department of Slavonic Languages and Literatures (Slavisches Seminar), University of Zurich</publisher>
            <publisher xml:lang="sr">Odsek za Slavistiku (Slavisches Seminar) Univerziteta u Cirihu</publisher>
            <availability>
               <p xml:lang="en">This work is licenced under <ref target="http://creativecommons.org/licenses/by-nc/4.0/">Creative Commons Attribution-NonCommercial 4.0</ref></p>
            </availability>
         </publicationStmt>
         <sourceDesc>
            <recordingStmt>
               <recording type="audio" dur="{dur_iso}">
                  <media xml:id="{xe(file_id)}.wav" mimeType="audio/wav" url="../{corpus_prefix}.wav/{xe(file_id)}.wav"/>
               </recording>
            </recordingStmt>
         </sourceDesc>
      </fileDesc>
      <encodingDesc>
{classdecl}         <appInfo>
            <application version="2.0" ident="torlak-tagger">
               <label>Torlak NLP pipeline</label>
               <p xml:lang="en">Lemmatisation and morphosyntactic tagging (UPOS, XPOS/MTE) with enriched CoNLL-U pipeline.</p>
            </application>
         </appInfo>
         <tagsDecl>
            <namespace name="http://www.tei-c.org/ns/1.0">
               <tagUsage gi="u"     occurs="{stats['u']}"/>
               <tagUsage gi="w"     occurs="{stats['w']}"/>
               <tagUsage gi="pause" occurs="{stats['pause']}"/>
               <tagUsage gi="unclear" occurs="{stats['unclear']}"/>
               <tagUsage gi="vocal" occurs="{stats['vocal']}"/>
            </namespace>
         </tagsDecl>
      </encodingDesc>
      <profileDesc>
{build_particdesc(speakers) if speakers else ""}
      </profileDesc>
  </teiHeader>"""
    return header


# ── Main converter ─────────────────────────────────────────────────────────────

def convert(conllu_path: Path, old_tei_dir: Optional[Path], output_path: Path) -> dict:
    _, sentences = parse_conllu(conllu_path)

    # Derive file_id from sent_id of first sentence
    first_sent_id = get_meta(sentences[0]["meta"], "sent_id") if sentences else ""
    file_id = re.match(r"((?:TOR|LUZ)_C_\d+)", first_sent_id).group(1) if first_sent_id else conllu_path.stem.split(".")[0]

    lang = "sr-tor" if file_id.startswith("TOR") else "sr-luz"

    # Build stress lookup (TOR_C only, if old TEI available)
    stress_lookup: Dict[str, str] = {}
    if old_tei_dir and file_id.startswith("TOR"):
        stress_lookup = build_stress_lookup(old_tei_dir, file_id)

    speakers = collect_speakers(sentences, file_id)
    timeline_xml, time_to_id = build_timeline(file_id, sentences)
    utt_xml, stats, all_topics, all_customs = build_utterances(
        file_id, sentences, time_to_id, stress_lookup)
    header_xml = build_header(file_id, sentences, lang, stats,
                               all_topics, all_customs, speakers)

    tei_xml = f"""\
<?xml version="1.0" encoding="UTF-8"?>
<TEI xmlns="http://www.tei-c.org/ns/1.0" xml:id="{xe(file_id, quote=True)}" xml:lang="{lang}">
{header_xml}
  <text>
      <body>
{timeline_xml}
{utt_xml}
      </body>
  </text>
</TEI>
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(tei_xml, encoding="utf-8")
    return {"file": file_id, **stats,
            "topics": len(all_topics), "customs": len(all_customs),
            "stress": len(stress_lookup)}


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Convert enriched CoNLL-U to TEI XML spoken corpus format.")
    parser.add_argument("--conllu",   required=True, help="topic_tagged_conllu/ folder")
    parser.add_argument("--old-tei",  help="TOR_C_TEI_transcripts/ folder (for stress recovery)")
    parser.add_argument("--output",   required=True, help="Output folder for TEI XML files")
    args = parser.parse_args()

    conllu_dir = Path(args.conllu)
    old_tei_dir = Path(args.old_tei) if args.old_tei else None
    output_dir  = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(conllu_dir.glob("*.enriched.conllu"))
    print(f"Found {len(files)} CoNLL-U files\n")

    totals = {"u": 0, "w": 0, "pause": 0, "stress": 0}
    for cf in files:
        out = output_dir / (cf.name[: -len(".enriched.conllu")] + ".xml")
        stats = convert(cf, old_tei_dir, out)
        for k in ("u", "w", "pause", "stress"):
            totals[k] += stats[k]
        print(f"  {stats['file']:<30}  u={stats['u']:>5}  w={stats['w']:>6}"
              f"  pause={stats['pause']:>4}  topics={stats['topics']:>3}"
              f"  stress_forms={stats['stress']:>5}")

    print(f"\n{'─'*60}")
    print(f"Total utterances : {totals['u']}")
    print(f"Total words      : {totals['w']}")
    print(f"Total pauses     : {totals['pause']}")
    print(f"Output           : {output_dir}")


if __name__ == "__main__":
    main()
