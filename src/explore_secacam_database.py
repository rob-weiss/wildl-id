#!/usr/bin/env python3
"""
Diagnostic script to explore all metadata in the Zeiss Secacam Realm database.
"""

import re
import subprocess
from collections import Counter
from pathlib import Path

# Paths
CONTAINER_PATH = Path(
    "/Users/wri2lr/Library/Containers/4225BDAD-C278-4C1E-81B0-726B325A096D/Data"
)
DOCUMENTS_DIR = CONTAINER_PATH / "Documents"

# Find the realm database file
REALM_DB = None
for file in DOCUMENTS_DIR.glob("user_*_realm.realm"):
    REALM_DB = file
    break

print("Zeiss Secacam Database Explorer")
print("=" * 70)
print(f"Database: {REALM_DB}\n")

if not REALM_DB or not REALM_DB.exists():
    print("Realm database not found!")
    exit(1)

# Extract all strings from the database
print("Extracting strings from database...")
result = subprocess.run(
    ["strings", str(REALM_DB)], capture_output=True, text=True, timeout=60
)

lines = result.stdout.split("\n")
print(f"Found {len(lines)} text strings in database\n")

# Look for various patterns
print("=" * 70)
print("SEARCHING FOR METADATA PATTERNS")
print("=" * 70)

# 1. Camera serials
print("\n1. CAMERA SERIAL NUMBERS:")
serial_pattern = r"\bSEC[0-9]+-[WR][0-9]+-[0-9]+\b"
serials = set()
for line in lines:
    matches = re.findall(serial_pattern, line)
    serials.update(matches)
print(f"   Found {len(serials)} unique serials:")
for serial in sorted(serials):
    print(f"   - {serial}")

# 2. Camera IDs (UUIDs)
print("\n2. CAMERA IDs (UUIDs):")
uuid_pattern = r"\b[A-F0-9]{8}-[A-F0-9]{4}-[A-F0-9]{4}-[A-F0-9]{4}-[A-F0-9]{12}\b"
uuids = set()
for line in lines:
    matches = re.findall(uuid_pattern, line)
    uuids.update(matches)
print(f"   Found {len(uuids)} unique UUIDs:")
for uuid in sorted(uuids)[:10]:  # Show first 10
    print(f"   - {uuid}")
if len(uuids) > 10:
    print(f"   ... and {len(uuids) - 10} more")

# 3. Look for readable text strings (potential camera names/locations)
print("\n3. POTENTIAL CAMERA NAMES/LOCATIONS:")
# Look for strings that are readable text (3-30 chars, mostly alphanumeric)
text_pattern = r"\b[A-Za-z][A-Za-z0-9\s_-]{2,29}\b"
texts = []
for line in lines:
    # Skip lines with URLs, UUIDs, or hex
    if "http" in line.lower() or re.search(r"[0-9A-F]{8}-[0-9A-F]{4}", line):
        continue
    matches = re.findall(text_pattern, line)
    texts.extend(matches)

# Filter for common/repeated strings that might be names
text_counts = Counter(texts)
common_texts = [
    (text, count)
    for text, count in text_counts.most_common(50)
    if count > 5 and len(text) > 3
]
print(f"   Found {len(common_texts)} frequently repeated text strings:")
for text, count in common_texts[:20]:  # Show top 20
    print(f"   - '{text}' (appears {count} times)")

# 4. Date/Time patterns
print("\n4. DATE/TIME PATTERNS:")
date_pattern = r"20\d{2}[-/]\d{2}[-/]\d{2}"
datetime_pattern = r"20\d{2}[-/]\d{2}[-/]\d{2}\s+\d{2}:\d{2}:\d{2}"
dates = set()
datetimes = set()
for line in lines:
    dates.update(re.findall(date_pattern, line))
    datetimes.update(re.findall(datetime_pattern, line))
print(f"   Found {len(dates)} unique dates")
print(f"   Found {len(datetimes)} unique date-times")
if datetimes:
    print(f"   Sample dates: {sorted(list(datetimes))[:5]}")

# 5. Look for structured data near camera serials
print("\n5. CONTEXT AROUND CAMERA SERIALS:")
for serial in sorted(serials)[:3]:  # Show first 3
    print(f"\n   Serial: {serial}")
    for i, line in enumerate(lines):
        if serial in line:
            # Show 5 lines before and after
            start = max(0, i - 5)
            end = min(len(lines), i + 6)
            context = lines[start:end]
            # Filter out binary/noise
            context = [l for l in context if len(l) > 2 and len(l) < 100]
            print(f"   Context (lines {start}-{end}):")
            for j, ctx_line in enumerate(context):
                marker = " >>> " if ctx_line == line else "     "
                print(f"   {marker}{ctx_line[:80]}")
            break  # Only show first occurrence

# 6. Look for key-value patterns
print("\n6. POTENTIAL KEY-VALUE PAIRS:")
kv_pattern = r'"?([a-zA-Z_][a-zA-Z0-9_]{2,20})"?\s*[=:]\s*"?([^"\n]{1,50})"?'
kv_pairs = {}
for line in lines:
    matches = re.findall(kv_pattern, line)
    for key, value in matches:
        if key not in kv_pairs:
            kv_pairs[key] = []
        kv_pairs[key].append(value)

if kv_pairs:
    print(f"   Found {len(kv_pairs)} potential keys:")
    for key, values in sorted(kv_pairs.items())[:20]:  # Show first 20
        unique_values = set(values)
        if len(unique_values) <= 3:
            print(f"   - {key}: {unique_values}")
        else:
            print(f"   - {key}: {len(unique_values)} different values")

# 7. Image URLs summary
print("\n7. IMAGE URLs:")
url_pattern = r'https://media\.secacam\.com/getImage/[^\s"<>]+'
url_count = 0
for line in lines:
    url_count += len(re.findall(url_pattern, line))
print(f"   Found {url_count} image URLs")

# Save detailed output to file
output_file = Path.home() / "SecacamImages" / "database_exploration.txt"
output_file.parent.mkdir(parents=True, exist_ok=True)

with open(output_file, "w") as f:
    f.write("ZEISS SECACAM DATABASE EXPLORATION\n")
    f.write("=" * 70 + "\n\n")
    f.write(f"Camera Serials ({len(serials)}):\n")
    for serial in sorted(serials):
        f.write(f"  {serial}\n")
    f.write(f"\nCamera UUIDs ({len(uuids)}):\n")
    for uuid in sorted(uuids)[:20]:
        f.write(f"  {uuid}\n")
    f.write("\nCommon Text Strings:\n")
    for text, count in common_texts[:50]:
        f.write(f"  '{text}' x{count}\n")

print(f"\n{'=' * 70}")
print(f"Detailed output saved to: {output_file}")
print(f"{'=' * 70}")
