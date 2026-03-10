#!/usr/bin/env python3
"""Collect GPU pricing data from gpuhunt's S3 catalogs and Epoch AI hardware data."""

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

EPOCH_CSV_URL = "https://epoch.ai/data/ml_hardware.csv"
FSDL_CSV_URL = "https://raw.githubusercontent.com/the-full-stack/website/main/docs/cloud-gpus/cloud-gpus.csv"
OUTPUT_FILE = os.path.join(DATA_DIR, "prices.json")

# Map FSDL "Cloud" column (lowercased, spaces removed) to gpuhunt provider names
FSDL_PROVIDER_MAP = {
    "aws": "aws",
    "azure": "azure",
    "gcp": "gcp",
    "lambda": "lambdalabs",
    "oraclecloud": "oci",
    "runpod": "runpod",
    "cudocompute": "cudo",
    "datacrunch": "datacrunch",
}

# Map FSDL "GPU Type" to (gpu_name, gpu_memory) matching gpuhunt naming
FSDL_GPU_MAP = {
    "A100 (80 GB)": ("A100", 80.0),
    "A100 (40 GB)": ("A100", 40.0),
    "V100 (16 GB)": ("V100", 16.0),
    "V100 (32 GB)": ("V100", 32.0),
    "A10G (24 GB)": ("A10G", 24.0),
    "A10": ("A10", None),
    "A40 (48 GB)": ("A40", 48.0),
    "A4000 (16 GB)": ("A4000", 16.0),
    "A5000 (24 GB)": ("A5000", 24.0),
    "A6000 (48 GB)": ("A6000", 48.0),
    "RTX A6000 (48 GB)": ("A6000", 48.0),
    "H100 (80 GB)": ("H100", 80.0),
    "H100 HGX (80GB)": ("H100", 80.0),
    "H100 PCIe (80GB)": ("H100", 80.0),
    "GH200 (96 GB)": ("GH200", 96.0),
    "T4 (16 GB)": ("T4", 16.0),
    "L4 (24 GB)": ("L4", 24.0),
    "P100 (16 GB)": ("P100", 16.0),
    "P4 (8 GB)": ("P4", 8.0),
    "K80 (12 GB)": ("K80", 12.0),
    "Quadro RTX 6000 (24 GB)": ("RTX6000", 24.0),
    "RTX4000 (8 GB)": ("RTX4000", 8.0),
    "RTX5000 (16 GB)": ("RTX5000", 16.0),
    "3080Ti (12 GB)": ("RTX3080Ti", 12.0),
}

V1_BASE = "https://dstack-gpu-pricing.s3.eu-west-1.amazonaws.com/v1"
V2_BASE = "https://dstack-gpu-pricing.s3.eu-west-1.amazonaws.com/v2"
V2_VERSION_URL = f"{V2_BASE}/version"

# v1 bucket: available daily from 2024-01-01 through 2024-06-12
V1_START = datetime(2024, 1, 1)
V1_END = datetime(2024, 6, 12)

# Known v2 weekly snapshots (discovered by probing the S3 bucket)
# Format: "YYYYMMDD-BUILD"
V2_KNOWN_VERSIONS = [
    "20250226-6552", "20250304-6720", "20250311-6888", "20250318-7056",
    "20250325-7224", "20250408-7560", "20250429-8064", "20250506-8232",
    "20250513-8400", "20250526-8736", "20250609-9072", "20250616-9240",
    "20250714-9912", "20250721-10080", "20250728-10248", "20250804-10416",
    "20250811-10584", "20250818-10752", "20250825-10920", "20250901-11088",
    "20250915-11424", "20250922-11592", "20250929-11760", "20251006-11928",
    "20251013-12096", "20251019-12264", "20251026-12432", "20251102-12600",
    "20251109-12768", "20251123-13104", "20251130-13272", "20251207-13440",
    "20251214-13608", "20260104-14112", "20260111-14280", "20260118-14448",
    "20260125-14616", "20260201-14784", "20260208-14952", "20260215-15120",
    "20260222-15288",
]


def log(msg):
    print(msg, file=sys.stderr)


def generate_v1_dates():
    """Generate weekly Monday dates from V1_START through V1_END."""
    d = V1_START
    # Align to first Monday on or after start
    while d.weekday() != 0:
        d += timedelta(days=1)
    while d <= V1_END:
        yield d.strftime("%Y%m%d")
        d += timedelta(days=7)


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


def fetch_v2_version(version):
    """Fetch and aggregate a specific v2 catalog snapshot. Returns None if unavailable."""
    date_part = version.split("-")[0]
    formatted_date = f"{date_part[:4]}-{date_part[4:6]}-{date_part[6:8]}"
    url = f"{V2_BASE}/{version}/catalog.zip"
    log(f"  Fetching v2/{version} ...")
    data = fetch_url(url, timeout=60)
    if data is None:
        log(f"  Skipped v2/{version} (not available)")
        return None
    try:
        providers = parse_catalog_zip(data)
    except (zipfile.BadZipFile, Exception) as e:
        log(f"  Skipped v2/{version} (bad zip: {e})")
        return None
    offers = aggregate_offers(providers)
    if not offers:
        log(f"  Skipped v2/{version} (no GPU offers)")
        return None
    log(f"  Got v2/{version}: {len(offers)} offer groups from {len(providers)} providers")
    return {
        "date": formatted_date,
        "source": "v2",
        "offers": offers,
    }


def fetch_v2_current():
    """Fetch the current (latest) v2 catalog snapshot."""
    log("Fetching v2 version...")
    version_bytes = fetch_url(V2_VERSION_URL)
    if version_bytes is None:
        log("ERROR: Could not fetch v2 version")
        return None
    version = version_bytes.decode("utf-8").strip().strip('"')
    log(f"  v2 version: {version}")
    return fetch_v2_version(version)


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


def fetch_epoch_data():
    """Fetch Epoch AI ml_hardware.csv and return structured hardware price-performance data."""
    log("Fetching Epoch AI hardware data...")
    data = fetch_url(EPOCH_CSV_URL)
    if data is None:
        log("ERROR: Could not fetch Epoch AI CSV")
        return None

    text = data.decode("utf-8")
    reader = csv.DictReader(io.StringIO(text))

    items = []
    for row in reader:
        release_date = row.get("Release date", "").strip()
        release_price = row.get("Release price (USD)", "").strip()
        if not release_date or not release_price:
            continue
        price_val = parse_float(release_price)
        if price_val is None or price_val <= 0:
            continue

        name = row.get("Name", "").strip()
        max_flops = parse_float(row.get("Max performance (FLOP/s)", ""))
        price_perf = parse_float(row.get("Price-performance (FLOP/s per $)", ""))
        tdp = parse_float(row.get("TDP (W)", ""))
        memory_bytes = parse_float(row.get("Memory (bytes)", ""))

        item = {
            "name": name,
            "release_date": release_date,
            "release_price_usd": price_val,
        }
        if max_flops is not None:
            item["max_flops"] = max_flops
        if price_perf is not None:
            item["price_performance"] = price_perf
            item["dollars_per_flop"] = 1.0 / price_perf
        if tdp is not None:
            item["tdp_watts"] = tdp
        if memory_bytes is not None:
            item["memory_gb"] = round(memory_bytes / (1024 ** 3), 1)

        items.append(item)

    items.sort(key=lambda x: x["release_date"])
    log(f"  Epoch AI: {len(items)} hardware entries with price data")

    return {
        "description": "GPU hardware price-performance from Epoch AI (purchase price, not cloud rental)",
        "source_url": EPOCH_CSV_URL,
        "license": "CC-BY",
        "items": items,
    }


def normalize_fsdl_provider(cloud):
    """Normalize FSDL Cloud column to a provider name."""
    key = cloud.strip().lower().replace(" ", "").replace(".", "")
    return FSDL_PROVIDER_MAP.get(key, key)


def normalize_fsdl_gpu(gpu_type):
    """Map FSDL GPU Type to (gpu_name, gpu_memory). Returns None if unmappable."""
    gpu_type = gpu_type.strip()
    if gpu_type in FSDL_GPU_MAP:
        return FSDL_GPU_MAP[gpu_type]
    # Fallback: strip parenthetical, use as-is
    name = gpu_type.split("(")[0].strip().replace(" ", "")
    return (name, None) if name else None


def fetch_fsdl_snapshot():
    """Fetch FSDL cloud-gpus.csv and aggregate into gpuhunt-compatible offer groups."""
    log("Fetching FSDL cloud GPU data...")
    data = fetch_url(FSDL_CSV_URL)
    if data is None:
        log("ERROR: Could not fetch FSDL CSV")
        return None

    text = data.decode("utf-8")
    reader = csv.DictReader(io.StringIO(text))

    # Key: (provider, gpu_name, gpu_memory, spot, gpu_count)
    buckets = {}

    for row in reader:
        cloud = row.get("Cloud", "").strip()
        gpu_type = row.get("GPU Type", "").strip()
        if not cloud or not gpu_type:
            continue

        gpu_info = normalize_fsdl_gpu(gpu_type)
        if gpu_info is None:
            continue
        gpu_name, gpu_memory = gpu_info
        provider = normalize_fsdl_provider(cloud)

        gpu_count = int(parse_float(row.get("GPUs", "1")) or 1)
        instance_name = row.get("Name", "").strip()

        # Extract prices: prefer Per-GPU, fall back to On-demand / gpu_count
        per_gpu = parse_float(row.get("Per-GPU", ""))
        on_demand = parse_float(row.get("On-demand", ""))
        spot_price = parse_float(row.get("Spot", ""))

        if per_gpu is not None and per_gpu > 0:
            od_price = per_gpu
        elif on_demand is not None and on_demand > 0:
            od_price = round(on_demand / gpu_count, 4)
        else:
            od_price = None

        # Add on-demand offer
        if od_price is not None and od_price > 0:
            key = (provider, gpu_name, gpu_memory, False, gpu_count)
            if key not in buckets:
                buckets[key] = {"prices": [], "locations": set(), "instance_names": set()}
            buckets[key]["prices"].append(od_price)
            if instance_name:
                buckets[key]["instance_names"].add(instance_name)

        # Add spot offer if spot price available
        if spot_price is not None and spot_price > 0:
            spot_per_gpu = round(spot_price / gpu_count, 4)
            key = (provider, gpu_name, gpu_memory, True, gpu_count)
            if key not in buckets:
                buckets[key] = {"prices": [], "locations": set(), "instance_names": set()}
            buckets[key]["prices"].append(spot_per_gpu)
            if instance_name:
                buckets[key]["instance_names"].add(instance_name)

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

    log(f"  FSDL: {len(offers)} offer groups")

    return {
        "date": datetime.now().strftime("%Y-%m-%d"),
        "source": "fsdl",
        "offers": offers,
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
        # Incremental mode: keep existing snapshots, only refresh current v2
        log("Incremental mode: keeping existing snapshots, refreshing current v2")
        existing_snapshots = existing.get("snapshots", [])
        log(f"  Kept {len(existing_snapshots)} existing snapshots")
    else:
        # Full mode: fetch all v1 + historical v2 snapshots
        log("Full mode: fetching v1 snapshots...")
        v1_dates = sorted(set(generate_v1_dates()))
        existing_snapshots = []
        for date_str in v1_dates:
            snap = fetch_v1_snapshot(date_str)
            if snap is not None:
                existing_snapshots.append(snap)
        log(f"Collected {len(existing_snapshots)} v1 snapshots")

        log(f"Fetching {len(V2_KNOWN_VERSIONS)} historical v2 snapshots...")
        for version in V2_KNOWN_VERSIONS:
            snap = fetch_v2_version(version)
            if snap is not None:
                existing_snapshots.append(snap)
        log(f"Total after v2 historical: {len(existing_snapshots)} snapshots")

    # Always fetch current v2
    log("Fetching current v2 snapshot...")
    v2_snap = fetch_v2_current()

    # Combine snapshots
    snapshots = list(existing_snapshots)
    if v2_snap:
        # Remove any existing snapshot for the same date
        snapshots = [s for s in snapshots if s["date"] != v2_snap["date"]]
        snapshots.append(v2_snap)

    # Sort by date
    snapshots.sort(key=lambda s: s["date"])

    # Fetch Epoch AI hardware data
    epoch_data = fetch_epoch_data()

    # Always refresh FSDL (single HTTP request)
    fsdl_data = fetch_fsdl_snapshot()

    metadata = build_metadata(snapshots)

    output = {
        "last_updated": datetime.now().strftime("%Y-%m-%d"),
        "metadata": metadata,
        "snapshots": snapshots,
    }
    if epoch_data:
        output["epoch_hardware"] = epoch_data
    if fsdl_data:
        output["fsdl"] = fsdl_data

    with open(OUTPUT_FILE, "w") as f:
        json.dump(output, f, indent=2)

    log(f"Wrote {OUTPUT_FILE}")
    log(f"  {metadata['snapshot_count']} snapshots, {len(metadata['providers'])} providers, {len(metadata['gpu_types'])} GPU types")
    log(f"  Date range: {metadata['date_range'][0]} to {metadata['date_range'][1]}" if metadata["date_range"] else "  No data")


if __name__ == "__main__":
    main()
