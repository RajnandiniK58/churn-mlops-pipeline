"""Download the raw Telco churn CSV. Safe to re-run (skips if file exists)."""
import requests

from src.config import RAW_DATA_PATH, RAW_DATA_URL


def download(url: str = RAW_DATA_URL, dest=RAW_DATA_PATH, force: bool = False):
    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.exists() and not force:
        print(f"[download] already exists: {dest}")
        return dest

    print(f"[download] fetching {url}")
    response = requests.get(url, timeout=30)
    response.raise_for_status()          # fail loudly on 404 / network problems
    dest.write_bytes(response.content)
    print(f"[download] saved {len(response.content):,} bytes -> {dest}")
    return dest


if __name__ == "__main__":
    download()