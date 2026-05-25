#!/usr/bin/env python3
"""
run_combined_eval.py
--------------------
Runs evaluate_with_gemini across 4 CoNLL-U files and produces a combined report.

Files evaluated:
  1. TOR_C_0006.pred.conllu
  2. TOR_C_0025.pred.conllu
  3. TOR_C_0085.pred.conllu
  4. REMBERT_TOR_C_0031.corrected.enriched.conllu  (TOR_C_0031.xml is TEI source, not CoNLL-U)

Outputs:
  combined_eval_results.tsv    per-token detail across all files
  combined_eval_summary.txt    aggregate + per-file breakdown
"""

import re, os, sys, json, time, textwrap
from pathlib import Path
from typing import List, Dict, Optional

# ── API key ────────────────────────────────────────────────────────────────────
SCRIPT_DIR   = Path(__file__).parent
API_KEY_FILE = SCRIPT_DIR / "gemini_api_key.txt"

if API_KEY_FILE.exists():
    API_KEY = API_KEY_FILE.read_text().strip()
elif os.environ.get("GEMINI_API_KEY"):
    API_KEY = os.environ["GEMINI_API_KEY"]
else:
    sys.exit("ERROR: No API key. Add gemini_api_key.txt or set GEMINI_API_KEY.")

try:
    from google import genai
    from google.genai import types as genai_types
except ImportError:
    sys.exit("ERROR: pip install google-genai")

GEMINI_MODEL = "gemini-2.5-flash"
DELAY        = 1.5   # seconds between calls

FILES = [
    ("TOR_C_0006", SCRIPT_DIR / "TOR_C_0006.pred.conllu"),
    ("TOR_C_0025", SCRIPT_DIR / "TOR_C_0025.pred.conllu"),
    ("TOR_C_0085", SCRIPT_DIR / "TOR_C_0085.pred.conllu"),
    ("TOR_C_0031", SCRIPT_DIR / "REMBERT_TOR_C_0031.corrected.enriched.conllu"),
]

# ── CoNLL-U parser ─────────────────────────────────────────────────────────────
def parse_conllu(path: Path) -> List[Dict]:
    sents, meta, toks = [], {}, []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            m = re.match(r"# ([\w_]+) = (.+)", line)
            if m:
                meta[m.group(1)] = m.group(2)
            elif line.strip() == "":
                if meta:
                    sents.append({"meta": dict(meta), "tokens": toks})
                meta, toks = {}, []
            elif "\t" in line:
                parts = line.split("\t")
                if len(parts) == 10 and parts[1] not in ("/", ""):
                    toks.append({
                        "id": parts[0], "form": parts[1], "lemma": parts[2],
                        "upos": parts[3], "xpos": parts[4], "feats": parts[5],
                    })
    if meta:
        sents.append({"meta": dict(meta), "tokens": toks})
    return sents

# ── Prompts ────────────────────────────────────────────────────────────────────
_XPOS_GUIDE = """
MulText-East (MTE) XPOS structure (first letter = POS):
  N Noun: Nc=common, Np=proper; gender(m/f/n); number(s/p); case(n/g/d/a/v/l/i)
  V Verb: Vm=main, Va=auxiliary; mood(i=ind,m=imp,c=cond); tense; person; number
  A Adjective: Ag=general; degree(p/c/s); gender; number; case; definiteness(y/n)
  P Pronoun: Pp=personal, Pd=demonstrative, Pi=indefinite, Px=reflexive, Ps=possessive
  R Adverb: Rg=general; degree(p/c/s)
  S Adposition | C Conjunction: Cc=coord, Cs=subord | M Numeral | Q Particle | I Interjection
"""

_SYSTEM = textwrap.dedent("""
You are an expert linguist specialising in South Slavic morphosyntax, including
Serbian, Bulgarian, Macedonian, and the Torlak dialect continuum (southeast Serbia).

Torlak distinctive features:
- Reduced nominal case morphology (more analytic than standard Serbian)
- Retention of archaic forms (aorist, imperfect)
- Bulgarian/Macedonian influence (postposed definiteness markers)
- Dialectal verb forms and lexicon

Judge each token's predicted LEMMA, UPOS, and XPOS:
- LEMMA: canonical/dictionary form. Be lenient for dialectal variation
  (e.g. 'tolko'→'tolko' or 'toliko' are both acceptable).
- UPOS: Universal Dependencies tag. Be strict for clear POS errors.
- XPOS: MulText-East tag. Check category + key features.

Return ONLY a valid JSON array — no markdown, no explanation.
""").strip()

def build_prompt(text: str, tokens: List[Dict]) -> str:
    rows = "\n".join(
        f"  {t['id']:>3}. form={t['form']:<18} lemma={t['lemma']:<18} "
        f"upos={t['upos']:<8} xpos={t['xpos']}"
        for t in tokens
    )
    return textwrap.dedent(f"""
Sentence (Torlak dialect, lowercase normalised):
  "{text}"

Predicted annotations:
{rows}
{_XPOS_GUIDE}
For EACH token return a JSON array with objects having keys:
  "id", "form", "lemma_ok" (bool), "upos_ok" (bool), "xpos_ok" (bool),
  "note" (empty string if all correct, else brief error description)
""").strip()

# ── Gemini caller ──────────────────────────────────────────────────────────────
def call_gemini(client, prompt: str, retries=3) -> Optional[str]:
    for attempt in range(1, retries + 1):
        try:
            resp = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt,
                config=genai_types.GenerateContentConfig(
                    system_instruction=_SYSTEM,
                    temperature=0.0,
                    response_mime_type="application/json",
                ),
            )
            return resp.text
        except Exception as e:
            print(f"    [attempt {attempt}/{retries}] {e}")
            if attempt < retries:
                time.sleep(5 * attempt)
    return None

def parse_response(raw: str, tokens: List[Dict]) -> List[Dict]:
    try:
        text = re.sub(r"```(?:json)?|```", "", raw).strip()
        data = json.loads(text)
        results = []
        for item, tok in zip(data, tokens):
            results.append({
                "form": tok["form"], "lemma": tok["lemma"],
                "upos": tok["upos"], "xpos": tok["xpos"],
                "lemma_ok": bool(item.get("lemma_ok")),
                "upos_ok":  bool(item.get("upos_ok")),
                "xpos_ok":  bool(item.get("xpos_ok")),
                "note":     str(item.get("note", "")).strip(),
                "parse_error": False,
            })
        for tok in tokens[len(results):]:
            results.append({
                "form": tok["form"], "lemma": tok["lemma"],
                "upos": tok["upos"], "xpos": tok["xpos"],
                "lemma_ok": None, "upos_ok": None, "xpos_ok": None,
                "note": "MISSING from response", "parse_error": True,
            })
        return results
    except Exception as e:
        return [{
            "form": t["form"], "lemma": t["lemma"],
            "upos": t["upos"], "xpos": t["xpos"],
            "lemma_ok": None, "upos_ok": None, "xpos_ok": None,
            "note": f"PARSE_ERROR: {e}", "parse_error": True,
        } for t in tokens]

# ── Stats ──────────────────────────────────────────────────────────────────────
def compute_stats(rows: List[Dict]) -> Dict:
    result = {}
    for cat, key in [("lemma","lemma_ok"), ("upos","upos_ok"), ("xpos","xpos_ok")]:
        ev  = [r for r in rows if r[key] is not None]
        ok  = [r for r in ev  if r[key] is True]
        result[cat] = {
            "correct": len(ok), "evaluable": len(ev),
            "accuracy": len(ok)/len(ev) if ev else 0.0,
        }
    return result

def fmt_stats(stats: Dict) -> str:
    lines = [f"  {'cat':<8} {'correct':>8} {'evaluable':>10} {'accuracy':>9}"]
    lines.append("  " + "-"*38)
    for cat, s in stats.items():
        lines.append(f"  {cat:<8} {s['correct']:>8} {s['evaluable']:>10} {s['accuracy']:>8.1%}")
    return "\n".join(lines)

# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    client    = genai.Client(api_key=API_KEY)
    all_rows:  List[Dict] = []   # all tokens across all files
    per_file:  Dict[str, List[Dict]] = {}

    for file_tag, conllu_path in FILES:
        if not conllu_path.exists():
            print(f"\n✗ SKIP {file_tag}: {conllu_path} not found")
            continue

        print(f"\n{'='*60}")
        print(f"Evaluating: {file_tag}  ({conllu_path.name})")
        print(f"{'='*60}")

        sentences  = parse_conllu(conllu_path)
        file_rows: List[Dict] = []
        n_err = 0

        for si, sent in enumerate(sentences, 1):
            sid  = sent["meta"].get("sent_id", f"{file_tag}-s{si:04d}")
            text = sent["meta"].get("text", "")
            toks = sent["tokens"]
            if not toks:
                continue

            print(f'  [{si:>3}/{len(sentences)}] {sid}  ({len(toks)} tok)  "{text[:60]}"')

            raw = call_gemini(client, build_prompt(text, toks))
            if raw is None:
                print("    ⚠ No response — skipping")
                n_err += 1
                for t in toks:
                    file_rows.append({
                        "file": file_tag, "sent_id": sid, "token_id": t["id"],
                        **{k: t[k] for k in ("form","lemma","upos","xpos")},
                        "lemma_ok": None, "upos_ok": None, "xpos_ok": None, "note": "API_ERROR",
                    })
                continue

            results = parse_response(raw, toks)
            for t, r in zip(toks, results):
                row = {"file": file_tag, "sent_id": sid, "token_id": t["id"], **r}
                file_rows.append(row)
                if not r["parse_error"]:
                    errors = []
                    if r["lemma_ok"] is False: errors.append(f"LEMMA({t['lemma']})")
                    if r["upos_ok"]  is False: errors.append(f"UPOS({t['upos']})")
                    if r["xpos_ok"]  is False: errors.append(f"XPOS({t['xpos']})")
                    if errors:
                        print(f"    ✗ {t['form']:<16} {', '.join(errors)}  ← {r['note']}")

            time.sleep(DELAY)

        per_file[file_tag] = file_rows
        all_rows.extend(file_rows)

        stats = compute_stats(file_rows)
        print(f"\n  {file_tag} results:")
        print(fmt_stats(stats))
        if n_err:
            print(f"  ⚠ {n_err} sentences skipped (API errors)")

    # ── Combined summary ──────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("COMBINED RESULTS")
    print(f"{'='*60}")
    combined = compute_stats(all_rows)
    print(fmt_stats(combined))

    # ── Write TSV ─────────────────────────────────────────────────────────────
    tsv_path = SCRIPT_DIR / "combined_eval_results.tsv"
    headers  = ["file","sent_id","token_id","form","lemma","upos","xpos",
                "lemma_ok","upos_ok","xpos_ok","note"]
    with tsv_path.open("w", encoding="utf-8") as f:
        f.write("\t".join(headers) + "\n")
        for r in all_rows:
            f.write("\t".join(str(r.get(h,"")) for h in headers) + "\n")
    print(f"\nPer-token TSV → {tsv_path}")

    # ── Write summary ─────────────────────────────────────────────────────────
    summary_path = SCRIPT_DIR / "combined_eval_summary.txt"
    with summary_path.open("w", encoding="utf-8") as f:
        f.write("Gemini Evaluation — Combined Summary\n")
        f.write(f"Model: {GEMINI_MODEL}\n")
        f.write(f"Files: {', '.join(tag for tag,_ in FILES)}\n\n")

        # Per-file
        for file_tag, rows in per_file.items():
            s = compute_stats(rows)
            n_sents = len({r["sent_id"] for r in rows})
            f.write(f"── {file_tag}  ({n_sents} sentences, {len(rows)} tokens)\n")
            for cat, st in s.items():
                f.write(f"   {cat:<8} {st['correct']:>5}/{st['evaluable']:<5}  {st['accuracy']:.1%}\n")
            f.write("\n")

        # Combined
        f.write(f"── COMBINED  ({len(all_rows)} tokens total)\n")
        for cat, st in combined.items():
            f.write(f"   {cat:<8} {st['correct']:>5}/{st['evaluable']:<5}  {st['accuracy']:.1%}\n")

        # Errors
        f.write("\n\nErrors (token-level):\n")
        f.write(f"{'file':<10} {'sent_id':<20} {'form':<18} {'lemma':>6} {'upos':>5} {'xpos':>5}  note\n")
        f.write("-"*90 + "\n")
        for r in all_rows:
            if r.get("lemma_ok") is False or r.get("upos_ok") is False or r.get("xpos_ok") is False:
                f.write(
                    f"{r['file']:<10} {r['sent_id']:<20} {r['form']:<18} "
                    f"{'OK' if r['lemma_ok'] else 'ERR':>6} "
                    f"{'OK' if r['upos_ok']  else 'ERR':>5} "
                    f"{'OK' if r['xpos_ok']  else 'ERR':>5}  {r['note']}\n"
                )

    print(f"Summary       → {summary_path}")


if __name__ == "__main__":
    main()
