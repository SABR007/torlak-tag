#!/usr/bin/env python3
"""
evaluate_with_gemini.py
-----------------------
Uses the Gemini API as an LLM judge to evaluate the automatic morphosyntactic
tagging (LEMMA, UPOS, XPOS) on the Torlak dialect CoNLL-U transcript.

For each sentence, Gemini is given:
  - the sentence text (Torlak dialect, South Slavic)
  - each token's predicted LEMMA, UPOS (Universal Dependencies), and
    XPOS (MulText-East tagset)
  - dialect context and tagset guidance

Gemini returns a JSON array indicating whether each prediction is correct,
with optional short notes.

Results are written to:
  - evaluate_with_gemini_results.tsv   (per-token detail)
  - evaluate_with_gemini_summary.txt   (aggregate accuracy by category)

Usage:
    python evaluate_with_gemini.py --api-key YOUR_KEY
    # or set GEMINI_API_KEY in the environment
"""

import re
import os
import sys
import json
import time
import argparse
import textwrap
from pathlib import Path
from typing import List, Dict, Optional, Tuple

# ── Gemini client ──────────────────────────────────────────────────────────────
try:
    from google import genai
    from google.genai import types as genai_types
except ImportError:
    sys.exit("ERROR: google-genai not installed. Run: pip install google-genai")

# ── CoNLL-U parser ─────────────────────────────────────────────────────────────

def parse_enriched_conllu(path: Path) -> List[Dict]:
    """Parse the enriched CoNLL-U; skip '/' pseudo-tokens."""
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
                if len(parts) == 10 and parts[1] != "/":
                    toks.append({
                        "id":    parts[0],
                        "form":  parts[1],
                        "lemma": parts[2],
                        "upos":  parts[3],
                        "xpos":  parts[4],
                        "feats": parts[5],
                        "misc":  parts[9],
                    })
    if meta:
        sents.append({"meta": dict(meta), "tokens": toks})
    return sents

# ── Prompt builder ─────────────────────────────────────────────────────────────

# Brief MulText-East XPOS prefix guide for the most common categories
_XPOS_GUIDE = """
MulText-East (MTE) XPOS tag structure (first letter = POS category):
  N  Noun       : Nc=common, Np=proper; gender (m/f/n); number (s/p); case (n/g/d/a/v/l/i)
  V  Verb        : Vm=main, Va=auxiliary, Vl=copula; mood (i=ind,m=imp); tense; person; number
  A  Adjective   : Ag=general; degree (p/c/s); gender; number; case; definiteness (y/n)
  P  Pronoun     : Pp=personal, Pd=demonstrative, Pi=indefinite, Px=reflexive, Ps=possessive
  R  Adverb      : Rg=general; degree (p/c/s)
  S  Adposition  : S (all)
  C  Conjunction : Cc=coordinating, Cs=subordinating
  M  Numeral     : Ml=cardinal, Mo=ordinal
  Q  Particle    : Qz=negation, Qo=other, Qr=response
  I  Interjection: I
  X  Residual/unknown
  _  (blank) for punctuation or special tokens
"""

_SYSTEM_PROMPT = textwrap.dedent("""
You are an expert linguist specialising in South Slavic morphosyntax, including
Serbian, Bulgarian, Macedonian, and — crucially — the Torlak dialect continuum
spoken in southeast Serbia and northwest North Macedonia.

Torlak has several distinctive features compared to standard Serbian:
- Reduced or absent nominal case morphology (more analytic)
- Retention of some archaic forms
- Influence from Bulgarian/Macedonian (e.g. postposed definiteness markers)
- Dialectal stress and vowel changes (capitalised vowels in the EXB source
  indicate primary stress; the CoNLL-U forms are lowercase normalised)
- Dialectal verb forms (e.g. aorist preserved; present-tense auxiliaries differ)

Your task: given a sentence in Torlak and the automatic annotations produced by
a transformer-based tagger, judge whether each token's predicted
  LEMMA  – the canonical/dictionary form (in standard Serbian spelling)
  UPOS   – Universal Dependencies part-of-speech tag
  XPOS   – MulText-East morphosyntactic tag

are CORRECT for the word in its sentence context.

Be lenient for normalisation differences in lemmas (e.g. dialectal 'tolko' may
legitimately lemmatise to itself OR to standard 'toliko'; accept both).
Be strict for clear POS errors (e.g. labelling a verb as NOUN).

""").strip()


def build_eval_prompt(sent_text: str, tokens: List[Dict]) -> str:
    """Build the per-sentence evaluation prompt."""
    rows = []
    for t in tokens:
        rows.append(
            f"  {t['id']:>3}.  form={t['form']:<18} lemma={t['lemma']:<18} "
            f"upos={t['upos']:<8} xpos={t['xpos']}"
        )
    token_block = "\n".join(rows)

    prompt = textwrap.dedent(f"""
Sentence (Torlak dialect, lowercase normalised):
  "{sent_text}"

Predicted annotations (ID, form, lemma, UPOS, XPOS):
{token_block}

{_XPOS_GUIDE}

For EACH token, return a JSON array (one object per token, in order).
Each object must have EXACTLY these keys:
  "id"        : token ID (string, matches the list above)
  "form"      : token surface form (copy from list)
  "lemma_ok"  : true/false — is the predicted lemma correct?
  "upos_ok"   : true/false — is the predicted UPOS correct?
  "xpos_ok"   : true/false — is the predicted XPOS correct?
  "note"      : short string, empty "" if all correct; otherwise briefly say
                what is wrong (e.g. "lemma should be 'ići'", "UPOS should be VERB")

Return ONLY a valid JSON array, no markdown fences, no explanation outside the JSON.
""").strip()
    return prompt


# ── Gemini caller ──────────────────────────────────────────────────────────────

def call_gemini(
    client,
    model_id: str,
    system_prompt: str,
    user_prompt: str,
    retries: int = 3,
    backoff: float = 5.0,
) -> Optional[str]:
    """Call Gemini, retry on transient errors, return raw text or None."""
    for attempt in range(1, retries + 1):
        try:
            response = client.models.generate_content(
                model=model_id,
                contents=user_prompt,
                config=genai_types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    temperature=0.0,
                    response_mime_type="application/json",
                ),
            )
            return response.text
        except Exception as exc:
            print(f"    [attempt {attempt}/{retries}] Gemini error: {exc}")
            if attempt < retries:
                time.sleep(backoff * attempt)
    return None


def parse_gemini_response(raw: str, tokens: List[Dict]) -> List[Dict]:
    """
    Parse Gemini's JSON response.
    Returns a list of result dicts (one per token) with keys:
      form, lemma_ok, upos_ok, xpos_ok, note, parse_error
    Falls back to all-None on parse failure.
    """
    try:
        # Strip accidental markdown fences
        text = re.sub(r"```(?:json)?|```", "", raw).strip()
        data = json.loads(text)
        if not isinstance(data, list):
            raise ValueError("Expected JSON array")
        results = []
        for item, tok in zip(data, tokens):
            results.append({
                "form":        tok["form"],
                "lemma":       tok["lemma"],
                "upos":        tok["upos"],
                "xpos":        tok["xpos"],
                "lemma_ok":    bool(item.get("lemma_ok")),
                "upos_ok":     bool(item.get("upos_ok")),
                "xpos_ok":     bool(item.get("xpos_ok")),
                "note":        str(item.get("note", "")).strip(),
                "parse_error": False,
            })
        # If Gemini returned fewer items than tokens, pad with None
        for tok in tokens[len(results):]:
            results.append({
                "form": tok["form"], "lemma": tok["lemma"],
                "upos": tok["upos"], "xpos": tok["xpos"],
                "lemma_ok": None, "upos_ok": None, "xpos_ok": None,
                "note": "MISSING from Gemini response", "parse_error": True,
            })
        return results
    except Exception as exc:
        print(f"      ⚠  Response parse error: {exc}")
        return [{
            "form": t["form"], "lemma": t["lemma"],
            "upos": t["upos"], "xpos": t["xpos"],
            "lemma_ok": None, "upos_ok": None, "xpos_ok": None,
            "note": f"PARSE_ERROR: {exc}", "parse_error": True,
        } for t in tokens]


# ── Results writer ─────────────────────────────────────────────────────────────

def write_tsv(output_path: Path, rows: List[Dict]) -> None:
    headers = [
        "sent_id", "token_id", "form", "lemma", "upos", "xpos",
        "lemma_ok", "upos_ok", "xpos_ok", "note",
    ]
    with output_path.open("w", encoding="utf-8") as f:
        f.write("\t".join(headers) + "\n")
        for r in rows:
            f.write("\t".join(str(r.get(h, "")) for h in headers) + "\n")


def compute_summary(rows: List[Dict]) -> Dict:
    """Compute accuracy for lemma, upos, xpos over all evaluable tokens."""
    cats = {"lemma": "lemma_ok", "upos": "upos_ok", "xpos": "xpos_ok"}
    stats = {}
    for cat, key in cats.items():
        evaluable = [r for r in rows if r[key] is not None]
        correct   = [r for r in evaluable if r[key] is True]
        stats[cat] = {
            "correct":   len(correct),
            "evaluable": len(evaluable),
            "skipped":   len(rows) - len(evaluable),
            "accuracy":  len(correct) / len(evaluable) if evaluable else 0.0,
        }
    return stats


def write_summary(output_path: Path, stats: Dict, model_id: str,
                  n_sents: int, n_toks: int) -> None:
    lines = [
        f"Gemini evaluation summary",
        f"  Model   : {model_id}",
        f"  Sentences evaluated : {n_sents}",
        f"  Tokens evaluated    : {n_toks}",
        f"",
        f"{'Category':<10}  {'Correct':>8}  {'Evaluable':>10}  {'Skipped':>8}  {'Accuracy':>9}",
        f"{'-'*55}",
    ]
    for cat, s in stats.items():
        lines.append(
            f"{cat:<10}  {s['correct']:>8}  {s['evaluable']:>10}  "
            f"{s['skipped']:>8}  {s['accuracy']:>8.1%}"
        )
    lines += ["", "Per-token breakdown (errors only):"]
    with output_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate CoNLL-U tagging (LEMMA/UPOS/XPOS) with Gemini as judge."
    )
    parser.add_argument(
        "--conllu",
        default="REMBERT_TOR_C_0031.corrected.enriched.conllu",
        help="Enriched CoNLL-U file to evaluate",
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("GEMINI_API_KEY", ""),
        help="Gemini API key (or set GEMINI_API_KEY env var)",
    )
    parser.add_argument(
        "--model",
        default="gemini-2.5-flash",
        help="Gemini model ID (default: gemini-2.5-flash)",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=1.5,
        help="Seconds to wait between API calls (default 1.5)",
    )
    parser.add_argument(
        "--max-sents",
        type=int,
        default=0,
        help="Evaluate only the first N sentences (0 = all)",
    )
    parser.add_argument(
        "--output-tsv",
        default="evaluate_with_gemini_results.tsv",
        help="Output TSV with per-token results",
    )
    parser.add_argument(
        "--output-summary",
        default="evaluate_with_gemini_summary.txt",
        help="Output summary file",
    )
    args = parser.parse_args()

    script_dir = Path(__file__).parent
    conllu_path = Path(args.conllu)
    if not conllu_path.is_absolute():
        conllu_path = script_dir / conllu_path
    tsv_path     = script_dir / args.output_tsv
    summary_path = script_dir / args.output_summary

    if not args.api_key:
        sys.exit("ERROR: Gemini API key required. Pass --api-key or set GEMINI_API_KEY.")

    # ── Load data ──────────────────────────────────────────────────────────────
    print(f"Reading CoNLL-U: {conllu_path}")
    sentences = parse_enriched_conllu(conllu_path)
    if args.max_sents > 0:
        sentences = sentences[: args.max_sents]
    n_toks = sum(len(s["tokens"]) for s in sentences)
    print(f"  → {len(sentences)} sentences, {n_toks} tokens to evaluate")

    # ── Gemini client ──────────────────────────────────────────────────────────
    client = genai.Client(api_key=args.api_key)
    print(f"Using model: {args.model}")

    # ── Evaluate sentence by sentence ──────────────────────────────────────────
    all_rows: List[Dict] = []
    n_api_errors = 0

    for si, sent in enumerate(sentences, 1):
        sid   = sent["meta"].get("sent_id", f"sent_{si}").replace("REMBERT_TOR_C_0031.pred-", "")
        text  = sent["meta"].get("text", "")
        toks  = sent["tokens"]

        print(f'  [{si:>3}/{len(sentences)}] {sid}  ({len(toks)} tokens)  "{text}"')

        prompt   = build_eval_prompt(text, toks)
        raw_resp = call_gemini(client, args.model, _SYSTEM_PROMPT, prompt)

        if raw_resp is None:
            print(f"    ⚠  No response — skipping sentence")
            n_api_errors += 1
            for t in toks:
                all_rows.append({
                    "sent_id": sid, "token_id": t["id"],
                    "form": t["form"], "lemma": t["lemma"],
                    "upos": t["upos"],  "xpos": t["xpos"],
                    "lemma_ok": None, "upos_ok": None, "xpos_ok": None,
                    "note": "API_ERROR",
                })
            continue

        results = parse_gemini_response(raw_resp, toks)

        for t, r in zip(toks, results):
            row = {
                "sent_id":  sid,
                "token_id": t["id"],
                **r,
            }
            all_rows.append(row)

            # Print errors inline
            if not r["parse_error"]:
                errors = []
                if r["lemma_ok"] is False:
                    errors.append(f"LEMMA({t['lemma']})")
                if r["upos_ok"] is False:
                    errors.append(f"UPOS({t['upos']})")
                if r["xpos_ok"] is False:
                    errors.append(f"XPOS({t['xpos']})")
                if errors:
                    note = r["note"] or "—"
                    print(f"    ✗  {t['form']:<18} {', '.join(errors)}  ← {note}")

        time.sleep(args.delay)

    # ── Aggregate results ──────────────────────────────────────────────────────
    stats = compute_summary(all_rows)

    print("\n" + "="*55)
    print("RESULTS")
    print("="*55)
    print(f"{'Category':<10}  {'Correct':>8}  {'Evaluable':>10}  {'Accuracy':>9}")
    print("-"*42)
    for cat, s in stats.items():
        print(f"{cat:<10}  {s['correct']:>8}  {s['evaluable']:>10}  {s['accuracy']:>8.1%}")
    if n_api_errors:
        print(f"\n⚠  {n_api_errors} sentence(s) skipped due to API errors")

    # ── Write outputs ──────────────────────────────────────────────────────────
    write_tsv(tsv_path, all_rows)
    print(f"\nPer-token TSV  → {tsv_path}")

    # Append error list to summary
    with summary_path.open("w", encoding="utf-8") as f:
        f.write(f"Gemini evaluation summary\n")
        f.write(f"  Model   : {args.model}\n")
        f.write(f"  Sentences: {len(sentences)}\n")
        f.write(f"  Tokens  : {n_toks}\n\n")
        f.write(f"{'Category':<10}  {'Correct':>8}  {'Evaluable':>10}  {'Skipped':>8}  {'Accuracy':>9}\n")
        f.write("-"*55 + "\n")
        for cat, s in stats.items():
            f.write(
                f"{cat:<10}  {s['correct']:>8}  {s['evaluable']:>10}  "
                f"{s['skipped']:>8}  {s['accuracy']:>8.1%}\n"
            )
        f.write("\nErrors (token-level):\n")
        f.write(f"{'sent_id':<12}  {'form':<18}  {'lemma_ok':>9}  {'upos_ok':>8}  {'xpos_ok':>8}  note\n")
        f.write("-"*90 + "\n")
        for r in all_rows:
            if r["lemma_ok"] is False or r["upos_ok"] is False or r["xpos_ok"] is False:
                f.write(
                    f"{r['sent_id']:<12}  {r['form']:<18}  "
                    f"{'OK' if r['lemma_ok'] else 'ERR':>9}  "
                    f"{'OK' if r['upos_ok'] else 'ERR':>8}  "
                    f"{'OK' if r['xpos_ok'] else 'ERR':>8}  "
                    f"{r['note']}\n"
                )

    print(f"Summary        → {summary_path}")


if __name__ == "__main__":
    main()
