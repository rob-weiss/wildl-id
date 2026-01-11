#!/usr/bin/env python3
"""
Script to extract and download all Zeiss Secacam images from the app's databases.

Requirements:
    pip install realm requests
"""

import sqlite3
import os
import sys
from pathlib import Path
from typing import List, Dict, Set
import requests
from datetime import datetime
import json

# Paths
CONTAINER_PATH = Path("/Users/wri2lr/Library/Containers/4225BDAD-C278-4C1E-81B0-726B325A096D/Data")
CACHE_DB = CONTAINER_PATH / "Library/Caches/de.zeiss.cop.caledonia/Cache.db"
FS_CACHE_DIR = CONTAINER_PATH / "Library/Caches/de.zeiss.cop.caledonia/fsCachedData"
DOCUMENTS_DIR = CONTAINER_PATH / "Documents"

# Find the realm database file
REALM_DB = None
for file in DOCUMENTS_DIR.glob("user_*_realm.realm"):
    REALM_DB = file
    break

# Output directory
OUTPUT_DIR = Path.home() / "SecacamImages"


def extract_urls_from_cache_db() -> List[Dict]:
    """Extract image URLs from the Cache.db SQLite database."""
    print(f"Reading Cache.db: {CACHE_DB}")
    
    if not CACHE_DB.exists():
        print(f"Cache.db not found at {CACHE_DB}")
        return []
    
    urls = []
    conn = sqlite3.connect(str(CACHE_DB))
    cursor = conn.cursor()
    
    try:
        # Get all cached image URLs with their file references
        query = """
        SELECT 
            r.entry_ID,
            r.request_key as url,
            r.time_stamp,
            hex(d.receiver_data) as file_uuid_hex,
            d.isDataOnFS
        FROM cfurl_cache_response r 
        JOIN cfurl_cache_receiver_data d ON r.entry_ID = d.entry_ID 
        WHERE r.request_key LIKE '%getImage%'
        ORDER BY r.time_stamp DESC
        """
        
        cursor.execute(query)
        rows = cursor.fetchall()
        
        for row in rows:
            entry_id, url, timestamp, file_uuid_hex, is_on_fs = row
            
            # Decode the UUID from hex
            file_uuid = None
            if file_uuid_hex:
                try:
                    file_uuid = bytes.fromhex(file_uuid_hex).decode('utf-8')
                except:
                    pass
            
            urls.append({
                'entry_id': entry_id,
                'url': url,
                'timestamp': timestamp,
                'file_uuid': file_uuid,
                'is_on_fs': is_on_fs,
                'source': 'cache_db'
            })
        
        print(f"Found {len(urls)} image URLs in Cache.db")
        
    except sqlite3.Error as e:
        print(f"SQLite error: {e}")
    finally:
        conn.close()
    
    return urls


def extract_urls_from_realm_db() -> List[Dict]:
    """Extract image URLs from the Realm database."""
    if not REALM_DB or not REALM_DB.exists():
        print(f"Realm database not found")
        return []
    
    print(f"Reading Realm database: {REALM_DB}")
    
    try:
        import realm
        
        # Open the Realm database
        config = realm.Configuration(path=str(REALM_DB), read_only=True)
        r = realm.Realm(configuration=config)
        
        # Get all objects - we need to discover the schema first
        print("Realm database opened successfully")
        print(f"Available schemas: {[obj_schema.name for obj_schema in r.schema]}")
        
        urls = []
        
        # Try to find image-related objects
        for obj_schema in r.schema:
            schema_name = obj_schema.name
            print(f"\nExamining schema: {schema_name}")
            
            # Look for schemas that might contain image data
            if any(keyword in schema_name.lower() for keyword in ['image', 'photo', 'media', 'picture', 'event']):
                objects = r.objects(schema_name)
                print(f"  Found {len(objects)} objects")
                
                # Examine first object to understand structure
                if len(objects) > 0:
                    first_obj = objects[0]
                    print(f"  Properties: {list(first_obj.__dict__.keys())}")
                
                # Extract URLs
                for obj in objects:
                    url = None
                    timestamp = None
                    
                    # Try to find URL field
                    for attr in ['url', 'imageUrl', 'image_url', 'path', 'uri']:
                        if hasattr(obj, attr):
                            url = getattr(obj, attr)
                            break
                    
                    # Try to find timestamp
                    for attr in ['timestamp', 'date', 'createdAt', 'created_at', 'time']:
                        if hasattr(obj, attr):
                            timestamp = getattr(obj, attr)
                            break
                    
                    if url:
                        urls.append({
                            'url': url,
                            'timestamp': timestamp,
                            'schema': schema_name,
                            'source': 'realm_db'
                        })
        
        print(f"\nFound {len(urls)} image URLs in Realm database")
        return urls
        
    except ImportError:
        print("\nRealm library not installed. Install with: pip install realm")
        print("Skipping Realm database extraction.\n")
        return []
    except Exception as e:
        print(f"Error reading Realm database: {e}")
        print("This is normal - Realm format may have changed or require specific SDK version")
        return []


def copy_cached_files():
    """Copy currently cached files from fsCachedData."""
    print(f"\nCopying cached files from: {FS_CACHE_DIR}")
    
    if not FS_CACHE_DIR.exists():
        print("fsCachedData directory not found")
        return
    
    cached_dir = OUTPUT_DIR / "cached_files"
    cached_dir.mkdir(parents=True, exist_ok=True)
    
    copied = 0
    for file in FS_CACHE_DIR.iterdir():
        if file.is_file():
            # Check if it's an image
            result = os.popen(f'file -b "{file}"').read()
            if 'JPEG' in result or 'PNG' in result:
                # Copy with .jpg extension
                dest = cached_dir / f"{file.name}.jpg"
                os.system(f'cp "{file}" "{dest}"')
                copied += 1
    
    print(f"Copied {copied} cached image files to {cached_dir}")


def download_image(url: str, output_path: Path) -> bool:
    """Download an image from URL."""
    try:
        response = requests.get(url, timeout=30, stream=True)
        response.raise_for_status()
        
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        return True
    except Exception as e:
        print(f"  Error downloading: {e}")
        return False


def download_all_images(urls: List[Dict]):
    """Download all images from URLs."""
    if not urls:
        print("No URLs to download")
        return
    
    print(f"\n{'='*60}")
    print(f"Downloading {len(urls)} images...")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"{'='*60}\n")
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Save URL list for reference
    urls_file = OUTPUT_DIR / "image_urls.json"
    with open(urls_file, 'w') as f:
        json.dump(urls, f, indent=2, default=str)
    print(f"Saved URL list to: {urls_file}\n")
    
    success_count = 0
    fail_count = 0
    
    for i, item in enumerate(urls, 1):
        url = item['url']
        
        # Create filename from URL or timestamp
        timestamp = item.get('timestamp', 'unknown')
        filename = f"image_{i:04d}_{timestamp}.jpg"
        output_path = OUTPUT_DIR / filename
        
        # Skip if already exists
        if output_path.exists():
            print(f"[{i}/{len(urls)}] Skipping (exists): {filename}")
            success_count += 1
            continue
        
        print(f"[{i}/{len(urls)}] Downloading: {filename}")
        
        if download_image(url, output_path):
            success_count += 1
            print(f"  ✓ Success")
        else:
            fail_count += 1
            print(f"  ✗ Failed")
    
    print(f"\n{'='*60}")
    print(f"Download complete!")
    print(f"  Success: {success_count}")
    print(f"  Failed:  {fail_count}")
    print(f"  Total:   {len(urls)}")
    print(f"{'='*60}\n")


def main():
    print("Zeiss Secacam Image Extractor")
    print("=" * 60)
    
    # Step 1: Copy currently cached files
    copy_cached_files()
    
    # Step 2: Extract URLs from Cache.db
    cache_urls = extract_urls_from_cache_db()
    
    # Step 3: Extract URLs from Realm database
    realm_urls = extract_urls_from_realm_db()
    
    # Step 4: Combine and deduplicate URLs
    all_urls = cache_urls + realm_urls
    unique_urls = []
    seen_urls = set()
    
    for item in all_urls:
        url = item['url']
        if url not in seen_urls:
            seen_urls.add(url)
            unique_urls.append(item)
    
    print(f"\nTotal unique URLs found: {len(unique_urls)}")
    
    if not unique_urls:
        print("\nNo image URLs found in databases.")
        print("This might mean:")
        print("  1. The app hasn't synced any images yet")
        print("  2. The Realm database uses a different schema")
        print("  3. Images are only stored temporarily in the cache")
        return
    
    # Step 5: Ask user if they want to download
    response = input(f"\nDownload {len(unique_urls)} images? (y/n): ")
    if response.lower() == 'y':
        download_all_images(unique_urls)
    else:
        print("Download cancelled.")
        print(f"URLs saved to: {OUTPUT_DIR / 'image_urls.json'}")


if __name__ == "__main__":
    main()
