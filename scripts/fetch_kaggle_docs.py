"""Download the source PDF documents referenced in the Kaggle labeled dataset.

Reads doc_names from the labeled CSV and resolves download URLs:
  - arxiv IDs  (e.g. "2311.16502v3")  → https://arxiv.org/pdf/2311.16502v3
  - ACL IDs    (e.g. "D18-1003")      → https://aclanthology.org/D18-1003.pdf
  - Other/hash → skipped with a warning (manual download required)

Usage:
    python scripts/fetch_kaggle_docs.py
    python scripts/fetch_kaggle_docs.py --out data/kaggle_docs --csv path/to/custom.csv
"""

from __future__ import annotations

import argparse
import re
import time
from glob import glob
from pathlib import Path

import requests

KAGGLE_OUTPUTS = Path(__file__).parent.parent.parent / "kaggle" / "outputs"
DEFAULT_OUT = Path(__file__).parent.parent / "data" / "kaggle_docs"

ARXIV_RE = re.compile(r"^\d{4}\.\d{4,5}(v\d+)?$")
ACL_RE = re.compile(r"^[A-Z]\d{2}-\d{4}$")


def find_latest_csv(outputs_dir: Path) -> Path:
    pattern = str(outputs_dir / "pilot_labeled_full_*.csv")
    matches = sorted(glob(pattern))
    if not matches:
        raise FileNotFoundError(
            f"No pilot_labeled_full_*.csv found in {outputs_dir}\n"
            "Run build_analysis_dataset.py in the kaggle project first."
        )
    return Path(matches[-1])


def resolve_url(doc_name: str) -> str | None:
    """Return download URL for a doc_name, or None if unknown format."""
    if ARXIV_RE.match(doc_name):
        return f"https://arxiv.org/pdf/{doc_name}"
    if ACL_RE.match(doc_name):
        return f"https://aclanthology.org/{doc_name}.pdf"
    return None


def download_pdf(url: str, out_path: Path, retries: int = 3) -> bool:
    """Download a PDF to out_path. Returns True on success."""
    headers = {"User-Agent": "Mozilla/5.0 (research; eval pipeline)"}
    for attempt in range(1, retries + 1):
        try:
            resp = requests.get(url, headers=headers, timeout=30, stream=True)
            if resp.status_code == 200:
                content = resp.content
                if not content.startswith(b"%PDF"):
                    print(f"    [WARN] Response is not a PDF for {url}")
                    return False
                out_path.write_bytes(content)
                return True
            else:
                print(f"    [WARN] HTTP {resp.status_code} (attempt {attempt}/{retries})")
        except requests.RequestException as e:
            print(f"    [WARN] Request failed (attempt {attempt}/{retries}): {e}")
        time.sleep(2 ** attempt)  # exponential backoff
    return False


def main():
    parser = argparse.ArgumentParser(description="Download Kaggle source PDFs")
    parser.add_argument("--csv", default=None, help="Path to labeled CSV")
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="Output directory for PDFs")
    args = parser.parse_args()

    try:
        import pandas as pd
    except ImportError:
        raise ImportError("pandas required: pip install pandas")

    csv_path = Path(args.csv) if args.csv else find_latest_csv(KAGGLE_OUTPUTS)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"CSV: {csv_path}")
    print(f"Output dir: {out_dir}\n")

    df = pd.read_csv(csv_path)
    doc_names = sorted(df["doc_name"].unique())
    print(f"Found {len(doc_names)} unique documents\n")

    success, skipped, failed = [], [], []

    for doc_name in doc_names:
        out_path = out_dir / f"{doc_name}.pdf"

        if out_path.exists():
            print(f"  [SKIP] Already exists: {out_path.name}")
            success.append(doc_name)
            continue

        url = resolve_url(doc_name)
        if url is None:
            print(f"  [SKIP] Unknown format, manual download needed: {doc_name}")
            skipped.append(doc_name)
            continue

        print(f"  [GET]  {doc_name} → {url}")
        ok = download_pdf(url, out_path)
        if ok:
            print(f"         ✓ saved ({out_path.stat().st_size // 1024} KB)")
            success.append(doc_name)
        else:
            print(f"         ✗ FAILED")
            failed.append(doc_name)

        time.sleep(1)  # be polite to arxiv/acl servers

    print(f"\n--- Summary ---")
    print(f"  Downloaded: {len(success)}")
    print(f"  Skipped (unknown format): {len(skipped)}")
    print(f"  Failed: {len(failed)}")

    if skipped:
        print(f"\nManual download needed for:")
        for d in skipped:
            print(f"  {d}")

    if failed:
        print(f"\nFailed downloads (retry or download manually):")
        for d in failed:
            print(f"  {d}")

    if success and not failed:
        print(f"\nAll documents ready in: {out_dir}")
        print("\nNext step — ingest into RAG:")
        print("  POST /ingest  {\"source\": \"local_files\", \"file_paths\": [\"data/kaggle_docs/*.pdf\"]}")
        print("  Or use the eval pipeline directly (bypasses API).")


if __name__ == "__main__":
    main()
