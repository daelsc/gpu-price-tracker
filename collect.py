#!/usr/bin/env python3
"""Collect GPU pricing data from gpuhunt's S3 catalogs into a single JSON file."""

import argparse
import csv
import io
import json
import os
import sys
import zipfile
from datetime import datetime, timedelta
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
OUTPUT_FILE = os.path.join(DATA_DIR, "prices.json")

V1_BASE = "https://dstack-gpu-pricing.s3.eu-west-1.amazonaws.com/v1"
V2_BASE = "https://dstack-gpu-pricing.s3.eu-west-1.amazonaws.com/v2"
V2_VERSION_URL = f"{V2_BASE}/version"

# Weekly Mondays from 2024-01-01 through 2024-06-08
V1_START = datetime(2024, 1, 1)
V1_END = datetime(2024, 6, 8)


def log(msg):
    print(msg, file=sys.stderr)


def generate_v1_dates():
    """Generate Monday dates from V1_START through V1_END."""
    d = V1_START
    # Align to first Monday on or after start
    while d.weekday() != 0:
        d += timedelta(days=1)
    while d <= V1_END:
        yield d.strftime("%Y%m%d")
        d += timedelta(days=7)
    # Also include the exact start/end if they aren't Mondays
    start_str = V1_START.strftime("%Y%m%d")
    end_str = V1_END.strftime("%Y%m%d")
    # These will be deduplicated by the caller


def fetch_url(url, timeout=30):
    """Fetch URL contents as bytes. Returns None on 403/404."""
    try:
        with urlopen(url, timeout=timeout) as resp:
            return resp.read()
    except HTTPError as e:
        if e.code in (403, 404):
            return None
        raise
    except URLError:
        return None


def parse_catalog_zip(zip_bytes):
    """Parse a catalog.zip, returning {provider: [rows]} where each row is a dict."""
    providers = {}
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        for name in zf.namelist():
            if not name.endswith(".csv"):
                continue
            provider = name.replace(".csv", "").split("/")[-1]
            with zf.open(name) as f:
                text = io.TextIOWrapper(f, encoding="utf-8")
                reader = csv.DictReader(text)
                rows = []
                for row in reader:
                    rows.append(row)
                providers[provider] = rows
    return providers


def parse_float(val):
    """Parse a float, returning None on failure."""
    if val is None or val == "":
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


def parse_bool(val):
    """Parse a boolean string."""
    if isinstance(val, bool):
        return val
    if isinstance(val, str):
        return val.strip().upper() == "TRUE"
    return False


def aggregate_offers(providers):
    """Aggregate per-provider, per-GPU, per-spot pricing from parsed CSV rows.

    Returns a list of offer dicts.
    """
    # Key: (provider, gpu_name, gpu_memory, spot, gpu_count)
    buckets = {}

    for provider, rows in providers.items():
        for row in rows:
            gpu_name = row.get("gpu_name", "").strip()
            if not gpu_name:
                continue
            price = parse_float(row.get("price"))
            if price is None or price <= 0:
                continue
            gpu_memory = parse_float(row.get("gpu_memory"))
            gpu_count_raw = row.get("gpu_count", "1")
            gpu_count = int(parse_float(gpu_count_raw) or 1)
            spot = parse_bool(row.get("spot", "FALSE"))
            location = row.get("location", "").strip()
            instance_name = row.get("instance_name", "").strip()

            key = (provider, gpu_name, gpu_memory, spot, gpu_count)
            if key not in buckets:
                buckets[key] = {
                    "prices": [],
                    "locations": set(),
                    "instance_names": set(),
                }
            b = buckets[key]
            b["prices"].append(price)
            if location:
                b["locations"].add(location)
            if instance_name:
                b["instance_names"].add(instance_name)

    offers = []
    for (provider, gpu_name, gpu_memory, spot, gpu_count), b in sorted(buckets.items()):
        prices = b["prices"]
        offers.append({
            "provider": provider,
            "gpu_name": gpu_name,
            "gpu_memory": gpu_memory,
            "spot": spot,
            "min_price": round(min(prices), 2),
            "avg_price": round(sum(prices) / len(prices), 2),
            "locations": sorted(b["locations"]),
            "instance_names": sorted(b["instance_names"]),
            "gpu_count": gpu_count,
            "offer_count": len(prices),
        })

    return offers


def fetch_v1_snapshot(date_str):
    """Fetch and aggregate a v1 catalog snapshot. Returns None if unavailable."""
    url = f"{V1_BASE}/{date_str}/catalog.zip"
    log(f"  Fetching v1/{date_str} ...")
    data = fetch_url(url)
    if data is None:
        log(f"  Skipped v1/{date_str} (not available)")
        return None
    try:
        providers = parse_catalog_zip(data)
    except (zipfile.BadZipFile, Exception) as e:
        log(f"  Skipped v1/{date_str} (bad zip: {e})")
        return None
    offers = aggregate_offers(providers)
    if not offers:
        log(f"  Skipped v1/{date_str} (no GPU offers)")
        return None
    # Format date as YYYY-MM-DD
    formatted = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
    log(f"  Got v1/{date_str}: {len(offers)} offer groups from {len(providers)} providers")
    return {
        "date": formatted,
        "source": "v1",
        "offers": offers,
    }


def fetch_v2_snapshot():
    """Fetch the current v2 catalog snapshot."""
    log("Fetching v2 version...")
    version_bytes = fetch_url(V2_VERSION_URL)
    if version_bytes is None:
        log("ERROR: Could not fetch v2 version")
        return None
    version = version_bytes.decode("utf-8").strip().strip('"')
    log(f"  v2 version: {version}")

    # Extract date from version string (format: YYYYMMDD-nnnnn)
    date_part = version.split("-")[0]
    formatted_date = f"{date_part[:4]}-{date_part[4:6]}-{date_part[6:8]}"

    url = f"{V2_BASE}/{version}/catalog.zip"
    log(f"  Fetching v2/{version} ...")
    data = fetch_url(url, timeout=60)
    if data is None:
        log("ERROR: Could not fetch v2 catalog")
        return None
    try:
        providers = parse_catalog_zip(data)
    except (zipfile.BadZipFile, Exception) as e:
        log(f"ERROR: Bad v2 zip: {e}")
        return None
    offers = aggregate_offers(providers)
    log(f"  Got v2/{version}: {len(offers)} offer groups from {len(providers)} providers")
    return {
        "date": formatted_date,
        "source": "v2",
        "offers": offers,
    }


def build_metadata(snapshots):
    """Build metadata summary from all snapshots."""
    providers = set()
    gpu_types = set()
    dates = []

    for snap in snapshots:
        dates.append(snap["date"])
        for offer in snap["offers"]:
            providers.add(offer["provider"])
            gpu_types.add(offer["gpu_name"])

    return {
        "providers": sorted(providers),
        "gpu_types": sorted(gpu_types),
        "date_range": [min(dates), max(dates)] if dates else [],
        "snapshot_count": len(snapshots),
    }


def main():
    parser = argparse.ArgumentParser(description="Collect GPU pricing data from gpuhunt S3 catalogs")
    parser.add_argument("--full", action="store_true", help="Force re-download of all data (v1 + v2)")
    args = parser.parse_args()

    os.makedirs(DATA_DIR, exist_ok=True)

    existing = None
    if os.path.exists(OUTPUT_FILE) and not args.full:
        log(f"Loading existing {OUTPUT_FILE} ...")
        with open(OUTPUT_FILE) as f:
            existing = json.load(f)

    if existing and not args.full:
        # Incremental mode: keep v1 snapshots, only refresh v2
        log("Incremental mode: keeping existing v1 snapshots, refreshing v2")
        v1_snapshots = [s for s in existing.get("snapshots", []) if s.get("source") == "v1"]
        log(f"  Kept {len(v1_snapshots)} existing v1 snapshots")
    else:
        # Full mode: fetch all v1 snapshots
        log("Full mode: fetching v1 snapshots...")
        v1_dates = sorted(set(generate_v1_dates()))
        v1_snapshots = []
        for date_str in v1_dates:
            snap = fetch_v1_snapshot(date_str)
            if snap is not None:
                v1_snapshots.append(snap)
        log(f"Collected {len(v1_snapshots)} v1 snapshots")

    # Always fetch current v2
    log("Fetching current v2 snapshot...")
    v2_snap = fetch_v2_snapshot()

    # Combine snapshots
    snapshots = list(v1_snapshots)
    if v2_snap:
        # Remove any existing v2 snapshot for the same date
        snapshots = [s for s in snapshots if not (s["source"] == "v2" and s["date"] == v2_snap["date"])]
        snapshots.append(v2_snap)

    # Sort by date
    snapshots.sort(key=lambda s: s["date"])

    metadata = build_metadata(snapshots)

    output = {
        "last_updated": datetime.now().strftime("%Y-%m-%d"),
        "metadata": metadata,
        "snapshots": snapshots,
    }

    with open(OUTPUT_FILE, "w") as f:
        json.dump(output, f, indent=2)

    log(f"Wrote {OUTPUT_FILE}")
    log(f"  {metadata['snapshot_count']} snapshots, {len(metadata['providers'])} providers, {len(metadata['gpu_types'])} GPU types")
    log(f"  Date range: {metadata['date_range'][0]} to {metadata['date_range'][1]}" if metadata["date_range"] else "  No data")


if __name__ == "__main__":
    main()
