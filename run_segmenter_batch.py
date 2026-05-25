#!/usr/bin/env python3
"""
run_segmenter_batch.py
----------------------
Run sentence_segmenter.py on every .conllu file in INPUT_DIR in parallel,
writing all outputs to OUTPUT_DIR.  Resumable: files whose
.corrected.conllup already exists in OUTPUT_DIR are skipped.

Usage:
    python run_segmenter_batch.py [--workers N] [--api-key KEY] [--model MODEL]
"""

import argparse
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

INPUT_DIR   = Path(__file__).parent / "final_conllu"
OUTPUT_DIR  = Path(__file__).parent / "sentence_segmented_conllu"
SCRIPT      = Path(__file__).parent / "sentence_segmenter.py"
DEFAULT_MODEL   = "gemini-2.5-flash"
DEFAULT_WORKERS = 5


def process_file(conllu_path: Path, output_dir: Path,
                 api_key: str | None, model: str, max_workers: int) -> dict:
    cmd = [sys.executable, str(SCRIPT), str(conllu_path),
           "--output-dir", str(output_dir),
           "--model", model,
           "--max-workers", str(max_workers)]
    if api_key:
        cmd += ["--api-key", api_key]

    t0 = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - t0

    return {
        "name":    conllu_path.name,
        "ok":      result.returncode == 0,
        "quota":   result.returncode == 2,
        "elapsed": elapsed,
        "stderr":  result.stderr.strip(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers",     type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--api-key",     default=None)
    parser.add_argument("--model",       default=DEFAULT_MODEL)
    parser.add_argument("--max-workers", type=int, default=20,
                        help="Concurrent Gemini calls within each file (default: 20)")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    conllu_files = sorted(INPUT_DIR.glob("*.conllu"))
    if not conllu_files:
        sys.exit(f"No .conllu files found in {INPUT_DIR}")

    # Resumability: skip files already completed
    pending = [p for p in conllu_files
               if not (OUTPUT_DIR / (p.stem + ".corrected.conllup")).exists()]
    skipped = len(conllu_files) - len(pending)

    print(f"Model   : {args.model}")
    print(f"Workers : {args.workers}")
    print(f"Input   : {INPUT_DIR}")
    print(f"Output  : {OUTPUT_DIR}")
    print(f"Total   : {len(conllu_files)} files  |  {skipped} already done  |  {len(pending)} to process\n")

    if not pending:
        print("All files already processed.")
        return

    done_count = 0
    failed     = []
    t_batch    = time.time()

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(process_file, p, OUTPUT_DIR, args.api_key, args.model, args.max_workers): p
            for p in pending
        }

        for future in as_completed(futures):
            res = future.result()
            done_count += 1

            if res.get("quota"):
                print(f"\n✗ QUOTA/CREDITS EXHAUSTED (from {res['name']}) — stopping batch immediately.",
                      file=sys.stderr)
                if res["stderr"]:
                    print(res["stderr"], file=sys.stderr)
                pool.shutdown(wait=False, cancel_futures=True)
                sys.exit(2)

            status = "✅" if res["ok"] else "❌"
            print(f"[{done_count + skipped:>3}/{len(conllu_files)}] {status}  "
                  f"{res['name']:<45}  {res['elapsed']:.0f}s")
            if not res["ok"]:
                failed.append(res["name"])
                if res["stderr"]:
                    for line in res["stderr"].splitlines()[:3]:
                        print(f"           {line}")

    total_elapsed = time.time() - t_batch
    print("\n" + "=" * 60)
    print(f"Finished.  done={done_count}  skipped={skipped}  failed={len(failed)}")
    print(f"Total time: {total_elapsed/60:.1f} min")
    if failed:
        print("\nFailed files:")
        for f in failed:
            print(f"  {f}")


if __name__ == "__main__":
    main()
