#!/usr/bin/env python3
"""
run_enrich_batch.py
-------------------
Run enrich_pred_conllu.py on every .corrected.conllup file in INPUT_DIR in
parallel, writing all outputs to OUTPUT_DIR.
Resumable: files whose .enriched.conllu already exists in OUTPUT_DIR are skipped.

Usage:
    python run_enrich_batch.py [--workers N]
"""

import argparse
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

INPUT_DIR   = Path(__file__).parent / "sentence_segmented_conllu"
PRED_DIR    = Path(__file__).parent / "final_conllu"
OUTPUT_DIR  = Path(__file__).parent / "enriched_conllu"
SCRIPT      = Path(__file__).parent / "enrich_pred_conllu.py"
SPEECHES    = Path(__file__).parent / "speaker_metadata" / "TOR_C_speeches.csv"
DEFAULT_WORKERS = 8


def process_file(corrected_path: Path, pred_dir: Path, output_dir: Path,
                 speeches_path: Path) -> dict:
    stem = corrected_path.stem.replace(".corrected", "")  # e.g. TOR_C_0001
    pred_path   = pred_dir   / (stem + ".conllu")
    output_path = output_dir / (stem + ".enriched.conllu")

    if not pred_path.exists():
        return {"name": corrected_path.name, "ok": False, "elapsed": 0,
                "stderr": f"pred not found: {pred_path}"}

    cmd = [
        sys.executable, str(SCRIPT),
        "--corrected", str(corrected_path),
        "--pred",      str(pred_path),
        "--speeches",  str(speeches_path),
        "--output",    str(output_path),
    ]

    t0 = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - t0

    return {
        "name":    corrected_path.name,
        "ok":      result.returncode == 0,
        "elapsed": elapsed,
        "stderr":  result.stderr.strip(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_files = sorted(INPUT_DIR.glob("*.corrected.conllup"))
    if not all_files:
        sys.exit(f"No .corrected.conllup files found in {INPUT_DIR}")

    pending = [p for p in all_files
               if not (OUTPUT_DIR / (p.stem.replace(".corrected", "") + ".enriched.conllu")).exists()]
    skipped = len(all_files) - len(pending)

    print(f"Workers : {args.workers}")
    print(f"Input   : {INPUT_DIR}")
    print(f"Output  : {OUTPUT_DIR}")
    print(f"Total   : {len(all_files)} files  |  {skipped} already done  |  {len(pending)} to process\n")

    if not pending:
        print("All files already processed.")
        return

    done_count = 0
    failed = []
    t_batch = time.time()

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(process_file, p, PRED_DIR, OUTPUT_DIR, SPEECHES): p
            for p in pending
        }

        for future in as_completed(futures):
            res = future.result()
            done_count += 1
            status = "✅" if res["ok"] else "❌"
            print(f"[{done_count + skipped:>3}/{len(all_files)}] {status}  "
                  f"{res['name']:<50}  {res['elapsed']:.1f}s")
            if not res["ok"]:
                failed.append(res["name"])
                if res["stderr"]:
                    for line in res["stderr"].splitlines()[:3]:
                        print(f"           {line}")

    total_elapsed = time.time() - t_batch
    print("\n" + "=" * 60)
    print(f"Finished.  done={done_count}  skipped={skipped}  failed={len(failed)}")
    print(f"Total time: {total_elapsed:.1f}s")
    if failed:
        print("\nFailed files:")
        for f in failed:
            print(f"  {f}")


if __name__ == "__main__":
    main()
