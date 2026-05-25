#!/usr/bin/env python3
"""
add_topic_metadata.py
---------------------
For every sentence in each enriched CoNLL-U file, check whether the
sentence's [start_time, end_time] overlaps with any excerpt entry in the
Torlak_tales topic-cluster or fine-custom markdown files.

Four metadata comments are appended to each sentence:

    # topic_clusters    = slug1|slug2
    # topic_clusters_sr = Serbian Name 1|Serbian Name 2
    # fine_customs      = subtopic_slug1|subtopic_slug2
    # fine_customs_topic = Parent Topic 1|Parent Topic 2

All fields use _ when no match is found.

Overlap condition (inclusive): sentence_start < excerpt_end AND sentence_end > excerpt_start

Usage:
    python add_topic_metadata.py \
        --input    enriched_conllu/ \
        --output   topic_tagged_conllu/ \
        --clusters Torlak_tales-main/01_topic_clusters/ \
        --customs  Torlak_tales-main/01b_fine_customs/
"""

import re
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# ── Time helpers ──────────────────────────────────────────────────────────────

def hms_to_sec(hms: str) -> float:
    """Convert 'H:MM:SS' or 'HH:MM:SS' to seconds (float)."""
    parts = hms.strip().split(":")
    h, m, s = int(parts[0]), int(parts[1]), float(parts[2])
    return h * 3600.0 + m * 60.0 + s


# Matches: *Izvor:* `TOR_C_0086.exb` ... *Vreme:* 01:06:58–01:12:52
RE_EXCERPT = re.compile(
    r"\*Izvor:\*\s+`([^`]+?)\.exb`"          # group 1: transcript ID (full, with spaces/underscores)
    r".*?"
    r"\*Vreme:\*\s*"
    r"(\d{1,2}:\d{2}:[\d.]+)"               # group 2: start time HH:MM:SS
    r"[–\-]"                                  # en-dash or hyphen
    r"(\d{1,2}:\d{2}:[\d.]+)"               # group 3: end time HH:MM:SS
)

# ── Parse cluster / custom markdown directories ───────────────────────────────

def parse_md_dir(md_dir: Path) -> Dict[str, List[Tuple[str, float, float]]]:
    """
    Scan every *.md file (except _INDEX.md and _unassigned/) in md_dir.

    Returns:
        {transcript_id: [(slug, start_sec, end_sec), ...]}

    The transcript_id is the part of the source filename before '.exb',
    e.g. "TOR_C_0086", "TOR_C_0004 BS", "TOR_C_0035_DS".
    """
    lookup: Dict[str, List[Tuple[str, float, float]]] = {}

    for md_file in sorted(md_dir.glob("*.md")):
        if md_file.name.startswith("_"):
            continue
        slug = md_file.stem  # filename without .md

        for line in md_file.read_text(encoding="utf-8").splitlines():
            m = RE_EXCERPT.search(line)
            if not m:
                continue
            transcript_id = m.group(1)
            try:
                start_sec = hms_to_sec(m.group(2))
                end_sec   = hms_to_sec(m.group(3))
            except (ValueError, IndexError):
                print(f"  WARNING: could not parse time in {md_file.name}: {line[:80]}",
                      file=sys.stderr)
                continue

            lookup.setdefault(transcript_id, []).append((slug, start_sec, end_sec))

    return lookup


def build_slug_label_map(md_dir: Path) -> Dict[str, str]:
    """
    Read each topic_cluster .md file and extract the H1 title (Serbian name).
    Returns {slug: "Serbian name"}.
    """
    labels: Dict[str, str] = {}
    for md_file in sorted(md_dir.glob("*.md")):
        if md_file.name.startswith("_"):
            continue
        slug = md_file.stem
        for line in md_file.read_text(encoding="utf-8").splitlines():
            if line.startswith("# "):
                labels[slug] = line[2:].strip()
                break
    return labels


RE_NADREDJENA = re.compile(r"\*Nadređena tema:\*\s*(.+)")

def build_custom_topic_map(md_dir: Path) -> Dict[str, str]:
    """
    Read each fine_customs .md file and extract the *Nadređena tema:* value.
    Returns {subtopic_slug: "parent topic string"}.
    """
    topics: Dict[str, str] = {}
    for md_file in sorted(md_dir.glob("*.md")):
        if md_file.name.startswith("_"):
            continue
        slug = md_file.stem
        for line in md_file.read_text(encoding="utf-8").splitlines():
            m = RE_NADREDJENA.search(line)
            if m:
                topics[slug] = m.group(1).strip()
                break
    return topics


# ── CoNLL-U helpers ────────────────────────────────────────────────────────────

def get_meta_value(meta_lines: List[str], key: str) -> Optional[str]:
    prefix = f"# {key} = "
    for line in meta_lines:
        if line.startswith(prefix):
            return line[len(prefix):].strip()
    return None


def parse_conllu(path: Path):
    global_header: List[str] = []
    sentences: List[dict] = []
    cur_meta:   List[str] = []
    cur_tokens: List[List[str]] = []
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


def write_conllu(path: Path, global_header: List[str], sentences: List[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for line in global_header:
            f.write(line + "\n")
        if global_header:
            f.write("\n")
        for sent in sentences:
            for line in sent["meta"]:
                f.write(line + "\n")
            for row in sent["tokens"]:
                f.write("\t".join(row) + "\n")
            f.write("\n")


# ── Matching ──────────────────────────────────────────────────────────────────

def find_slugs(
    transcript_id: str,
    s_start: float,
    s_end:   float,
    lookup:  Dict[str, List[Tuple[str, float, float]]],
) -> List[str]:
    """Return ordered, deduplicated list of slugs whose time range overlaps [s_start, s_end]."""
    seen:   set  = set()
    result: List[str] = []
    for slug, ex_start, ex_end in lookup.get(transcript_id, []):
        if s_start < ex_end and s_end > ex_start:
            if slug not in seen:
                seen.add(slug)
                result.append(slug)
    return result


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Add # topic_clusters / # fine_customs metadata to enriched CoNLL-U files."
    )
    parser.add_argument("--input",    required=True, help="Folder with *.enriched.conllu files")
    parser.add_argument("--output",   required=True, help="Output folder for tagged files")
    parser.add_argument("--clusters", required=True, help="01_topic_clusters/ directory")
    parser.add_argument("--customs",  required=True, help="01b_fine_customs/ directory")
    args = parser.parse_args()

    input_dir   = Path(args.input)
    output_dir  = Path(args.output)
    cluster_dir = Path(args.clusters)
    custom_dir  = Path(args.customs)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Build time-range lookups (slug → transcript excerpts)
    print("Parsing topic_clusters …")
    cluster_lookup = parse_md_dir(cluster_dir)
    print(f"  {len(cluster_lookup)} transcripts, "
          f"{sum(len(v) for v in cluster_lookup.values())} excerpt entries")

    print("Parsing fine_customs …")
    custom_lookup = parse_md_dir(custom_dir)
    print(f"  {len(custom_lookup)} transcripts, "
          f"{sum(len(v) for v in custom_lookup.values())} excerpt entries")

    # Build name maps for the additional label fields
    cluster_sr_map  = build_slug_label_map(cluster_dir)   # slug → Serbian name
    custom_topic_map = build_custom_topic_map(custom_dir)  # subtopic slug → Nadređena tema
    print(f"  cluster Serbian labels : {len(cluster_sr_map)}")
    print(f"  custom parent topics   : {len(custom_topic_map)}")

    # Process files
    input_files = sorted(input_dir.glob("*.enriched.conllu"))
    print(f"\nProcessing {len(input_files)} files …\n")

    total_sents = tagged_cluster = tagged_custom = 0

    for conllu_path in input_files:
        stem          = conllu_path.name[: -len(".enriched.conllu")]
        output_path   = output_dir / conllu_path.name

        global_header, sentences = parse_conllu(conllu_path)

        new_sentences: List[dict] = []
        file_cluster_hits = file_custom_hits = 0

        for sent in sentences:
            start_s = get_meta_value(sent["meta"], "start_time")
            end_s   = get_meta_value(sent["meta"], "end_time")

            if start_s is None or end_s is None:
                c_slugs = []
                k_slugs = []
            else:
                s_start = float(start_s)
                s_end   = float(end_s)
                c_slugs = find_slugs(stem, s_start, s_end, cluster_lookup)
                k_slugs = find_slugs(stem, s_start, s_end, custom_lookup)

            if c_slugs:
                file_cluster_hits += 1
            if k_slugs:
                file_custom_hits += 1

            # topic_clusters: English slugs + parallel Serbian names
            cluster_val    = "|".join(c_slugs) if c_slugs else "_"
            cluster_sr_val = "|".join(cluster_sr_map.get(s, s) for s in c_slugs) if c_slugs else "_"

            # fine_customs: subtopic slugs + parallel parent topics (Nadređena tema)
            custom_val       = "|".join(k_slugs) if k_slugs else "_"
            custom_topic_val = "|".join(custom_topic_map.get(s, "_") for s in k_slugs) if k_slugs else "_"

            # Strip any previously written topic fields to avoid duplication on re-runs
            _topic_keys = ("# topic_clusters ", "# topic_clusters_sr ",
                           "# fine_customs ", "# fine_customs_topic ")
            new_meta = [l for l in sent["meta"]
                        if not any(l.startswith(k) for k in _topic_keys)]
            new_meta.append(f"# topic_clusters = {cluster_val}")
            new_meta.append(f"# topic_clusters_sr = {cluster_sr_val}")
            new_meta.append(f"# fine_customs = {custom_val}")
            new_meta.append(f"# fine_customs_topic = {custom_topic_val}")

            new_sentences.append({"meta": new_meta, "tokens": sent["tokens"]})

        write_conllu(output_path, global_header, new_sentences)

        n = len(sentences)
        total_sents   += n
        tagged_cluster += file_cluster_hits
        tagged_custom  += file_custom_hits

        print(f"  {stem:<40}  {n:>5} sents  "
              f"topic_cluster={file_cluster_hits:>4}  fine_custom={file_custom_hits:>4}")

    print(f"\n{'─'*60}")
    print(f"Total sentences : {total_sents}")
    print(f"Tagged w/ cluster: {tagged_cluster}  ({100*tagged_cluster/max(total_sents,1):.1f}%)")
    print(f"Tagged w/ custom : {tagged_custom}  ({100*tagged_custom/max(total_sents,1):.1f}%)")
    print(f"Output folder   : {output_dir}")


if __name__ == "__main__":
    main()
