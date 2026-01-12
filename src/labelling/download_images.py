#!/usr/bin/env python3
"""
Download images from ZEISS Secacam gallery carousel.
This script assumes the carousel is already open in Safari.
"""

import hashlib
import json
import re
import subprocess
import urllib.request
from datetime import datetime
from pathlib import Path

from dateutil.relativedelta import relativedelta


def get_current_carousel_image():
    """Get the current full-resolution image URL and original filename from the carousel."""
    applescript = """
    tell application "Safari"
        tell current tab of front window
            do JavaScript "
                (function() {
                    // Find the currently visible/active carousel image - prefer the most centered one
                    const allImages = Array.from(document.querySelectorAll('img'))
                        .filter(img => {
                            const src = img.src || '';
                            if (!src.includes('media.secacam.com/getImage')) return false;
                            if (img.naturalWidth < 1000 || img.naturalHeight < 500) return false;
                            
                            // Must be actually visible with reasonable size
                            const rect = img.getBoundingClientRect();
                            return rect.width > 500 && rect.height > 500;
                        })
                        .map(img => {
                            const rect = img.getBoundingClientRect();
                            // Calculate how centered the image is
                            const centerX = rect.left + rect.width / 2;
                            const centerY = rect.top + rect.height / 2;
                            const viewportCenterX = window.innerWidth / 2;
                            const viewportCenterY = window.innerHeight / 2;
                            const distanceFromCenter = Math.sqrt(
                                Math.pow(centerX - viewportCenterX, 2) + 
                                Math.pow(centerY - viewportCenterY, 2)
                            );
                            
                            // Also check for active indicators
                            const parent = img.closest('[class*=active], [class*=current], [class*=selected], [aria-current]');
                            const hasActiveClass = parent !== null;
                            const zIndex = parseInt(window.getComputedStyle(img).zIndex) || 0;
                            
                            return {
                                img: img,
                                distanceFromCenter: distanceFromCenter,
                                hasActiveClass: hasActiveClass,
                                zIndex: zIndex,
                                rect: rect
                            };
                        })
                        .sort((a, b) => {
                            // Prioritize: active class > higher z-index > more centered
                            if (a.hasActiveClass !== b.hasActiveClass) {
                                return b.hasActiveClass ? 1 : -1;
                            }
                            if (Math.abs(a.zIndex - b.zIndex) > 0) {
                                return b.zIndex - a.zIndex;
                            }
                            return a.distanceFromCenter - b.distanceFromCenter;
                        });
                    
                    let mainImg = allImages.length > 0 ? allImages[0].img : null;
                    
                    if (!mainImg) {
                        // Fallback: Find any large visible image
                        const largeImages = Array.from(document.querySelectorAll('img'))
                            .filter(img => {
                                const src = img.src || '';
                                const skip = ['ic_', 'icon', 'logo', 'svg'];
                                const isUI = skip.some(s => src.toLowerCase().includes(s));
                                if (isUI) return false;
                                
                                const rect = img.getBoundingClientRect();
                                const isVisible = rect.width > 500 && rect.height > 500;
                                return isVisible && img.naturalWidth > 1000 && img.naturalHeight > 500;
                            })
                            .sort((a, b) => (b.naturalWidth * b.naturalHeight) - (a.naturalWidth * a.naturalHeight));
                        
                        mainImg = largeImages.length > 0 ? largeImages[0] : null;
                    }
                    
                    if (mainImg) {
                        const url = mainImg.src;
                        const alt = mainImg.getAttribute('alt');
                        const title = mainImg.getAttribute('title');
                        
                        let cameraName = '';
                        let timestamp = '';
                        let fullText = '';
                        let textCount = 0;
                        
                        // Strategy 1: Look for aria-label attributes or title elements near the image
                        const modal = mainImg.closest('[role=dialog], [class*=modal], [class*=viewer], [class*=lightbox]');
                        if (modal) {
                            // Check for aria-label on parent elements
                            const labeledElement = modal.querySelector('[aria-label]');
                            if (labeledElement) {
                                const ariaText = labeledElement.getAttribute('aria-label');
                                if (ariaText && ariaText.indexOf(' - ') !== -1 && ariaText.indexOf('/') !== -1) {
                                    fullText = ariaText;
                                    const dashIndex = ariaText.indexOf(' - ');
                                    cameraName = ariaText.substring(0, dashIndex).trim();
                                    timestamp = ariaText.substring(dashIndex + 3).trim();
                                }
                            }
                            
                            // Check for heading elements with the title
                            if (!fullText) {
                                const headings = modal.querySelectorAll('h1, h2, h3, h4, h5, h6');
                                for (let i = 0; i < headings.length; i++) {
                                    const text = headings[i].textContent.trim();
                                    if (text.indexOf(' - ') !== -1 && text.indexOf('/') !== -1) {
                                        fullText = text;
                                        const dashIndex = text.indexOf(' - ');
                                        cameraName = text.substring(0, dashIndex).trim();
                                        timestamp = text.substring(dashIndex + 3).trim();
                                        break;
                                    }
                                }
                            }
                        }
                        
                        // Strategy 2: Search all text content if not found yet
                        if (!fullText) {
                            const allElements = Array.from(document.querySelectorAll('*'));
                            const allText = [];
                            
                            allElements.forEach(function(el) {
                                if (allText.length >= 200) return;
                                const elText = el.textContent;
                                if (elText && elText.length > 20 && elText.length < 200) {
                                    const childTexts = Array.from(el.children).map(c => c.textContent).join('');
                                    const directText = elText.replace(childTexts, '').trim();
                                    if (directText && directText.length > 5) {
                                        allText.push(directText);
                                    }
                                }
                            });
                            
                            textCount = allText.length;
                            
                            allText.forEach(function(text) {
                                if (fullText) return;
                                const hasDash = text.indexOf(' - ') !== -1;
                                const hasSlash = text.indexOf('/') !== -1;
                                const hasDigit = /[0-9]/.test(text);
                                const notURL = text.indexOf('http') === -1;
                                
                                if (hasDash && hasSlash && hasDigit && notURL && text.length < 100) {
                                    fullText = text.trim();
                                    const dashIndex = text.indexOf(' - ');
                                    cameraName = text.substring(0, dashIndex).trim();
                                    timestamp = text.substring(dashIndex + 3).trim();
                                }
                            });
                        }
                        
                        // Fallback to searching for any camera name
                        if (!cameraName) {
                            if (alt && alt !== 'undefined' && alt !== 'null' && alt.trim() && alt.trim() !== 'undefined') {
                                cameraName = alt.trim();
                            } else if (title && title !== 'undefined' && title !== 'null' && title.trim() && title.trim() !== 'undefined') {
                                cameraName = title.trim();
                            }
                        }
                        
                        return JSON.stringify({
                            url: url,
                            cameraName: cameraName,
                            timestamp: timestamp,
                            fullText: fullText,
                            textCount: textCount,
                            width: mainImg.naturalWidth,
                            height: mainImg.naturalHeight,
                            debug_alt: mainImg.getAttribute('alt'),
                            debug_title: mainImg.getAttribute('title')
                        });
                    } else {
                        const allImgs = Array.from(document.querySelectorAll('img'));
                        const imgInfo = allImgs.slice(0, 5).map(i => ({
                            src: (i.src || '').substring(0, 80),
                            width: i.naturalWidth,
                            height: i.naturalHeight,
                            alt: i.alt || ''
                        }));
                        
                        return JSON.stringify({
                            error: 'not_found',
                            totalImages: allImgs.length,
                            largeImages: allImgs.filter(i => i.naturalWidth > 1000).length,
                            mediaImages: allImgs.filter(i => (i.src || '').includes('media.secacam.com')).length,
                            sampleImages: imgInfo
                        });
                    }
                })()
            "
        end tell
    end tell
    """

    try:
        result = subprocess.run(
            ["osascript", "-e", applescript], capture_output=True, text=True, timeout=10
        )

        print(f"  Debug: returncode={result.returncode}")
        print(f"  Debug: stdout='{result.stdout.strip()[:200]}'")
        print(f"  Debug: stderr='{result.stderr.strip()[:200]}'")

        if result.returncode == 0 and result.stdout.strip():
            data = json.loads(result.stdout.strip())
            if "error" in data:
                print(f"  Debug: Found {data.get('totalImages', 0)} total images")
                print(f"  Debug: {data.get('largeImages', 0)} large images (>1000px)")
                print(
                    f"  Debug: {data.get('mediaImages', 0)} images from media.secacam.com"
                )
                if data.get("sampleImages"):
                    print("  Debug: Sample images:")
                    for i, img in enumerate(data.get("sampleImages", [])[:3], 1):
                        print(
                            f"    {i}. {img.get('width')}x{img.get('height')} - {img.get('src', '')[:60]}"
                        )
                return None

            # Debug output for filename
            if data.get("debug_alt") is not None:
                print(f"  Debug: Raw alt='{data.get('debug_alt')}'")
            if data.get("debug_title") is not None:
                print(f"  Debug: Raw title='{data.get('debug_title')}'")

            return data
        return None
    except Exception as e:
        print(f"  Error: {e}")
        return None


def click_next_in_carousel():
    """Click the 'next' button in the carousel."""
    applescript = """
    tell application "Safari"
        tell current tab of front window
            do JavaScript "(function() {
                // Use keyboard navigation only - clicking buttons can close the carousel
                const carouselElements = document.querySelectorAll('[class*=carousel], [class*=slider], [class*=gallery], [class*=modal], [class*=lightbox], [role=dialog]');
                
                for (const elem of carouselElements) {
                    if (elem.offsetParent !== null) {
                        try {
                            elem.focus();
                            const event = new KeyboardEvent('keydown', {
                                key: 'ArrowRight',
                                keyCode: 39,
                                code: 'ArrowRight',
                                which: 39,
                                bubbles: true,
                                cancelable: true
                            });
                            elem.dispatchEvent(event);
                            break;
                        } catch(e) {}
                    }
                }
                
                // Also dispatch to document and body
                const evt = new KeyboardEvent('keydown', {
                    key: 'ArrowRight',
                    keyCode: 39,
                    code: 'ArrowRight',
                    which: 39,
                    bubbles: true,
                    cancelable: true
                });
                document.dispatchEvent(evt);
                document.body.dispatchEvent(evt);
                
                return 'key_pressed';
            })();"
            
            delay 1.5
        end tell
    end tell
    """

    try:
        result = subprocess.run(
            ["osascript", "-e", applescript], capture_output=True, text=True, timeout=10
        )

        print(f"  Debug: Click returned '{result.stdout.strip()}'")
        if result.stderr:
            print(f"  Debug: Click error '{result.stderr.strip()[:100]}'")

        # Always return True for keyboard events since we can't easily detect if they worked
        return result.returncode == 0
    except:
        return False


def download_with_safari(url, filepath):
    """Download a file using Safari's session."""
    applescript = f"""
    tell application "Safari"
        tell current tab of front window
            set jsCode to "
                fetch('{url}')
                    .then(r => r.blob())
                    .then(blob => {{
                        const reader = new FileReader();
                        reader.onload = () => {{
                            const base64 = reader.result.split(',')[1];
                            document.body.setAttribute('data-download-result', base64);
                        }};
                        reader.readAsDataURL(blob);
                    }})
                    .catch(e => {{
                        document.body.setAttribute('data-download-error', e.toString());
                    }});
                'downloading';
            "
            
            do JavaScript jsCode
            delay 2
            
            set resultJS to "document.body.getAttribute('data-download-result') || document.body.getAttribute('data-download-error') || 'pending'"
            set downloadResult to do JavaScript resultJS
            
            return downloadResult
        end tell
    end tell
    """

    try:
        result = subprocess.run(
            ["osascript", "-e", applescript], capture_output=True, text=True, timeout=30
        )

        if result.returncode == 0 and result.stdout.strip() not in [
            "pending",
            "downloading",
            "",
        ]:
            if "error" in result.stdout.lower():
                return False

            import base64

            try:
                image_data = base64.b64decode(result.stdout.strip())
                with open(filepath, "wb") as f:
                    f.write(image_data)
                return True
            except:
                return False
        return False
    except:
        return False


def main():
    """Main function."""
    print("ZEISS Secacam Gallery Carousel Downloader")
    print("=" * 60)
    print("💡 Make sure the carousel is already open in Safari!")
    print()

    # Configuration: How far back to download
    # Examples: "1w", "2m", "3d", "one week back", "two months back", "0" (max 6 months)
    DOWNLOAD_RANGE = "0"  # Change this to control download range

    # Parse the download range
    def parse_time_range(range_str):
        """Parse time range string and return timedelta."""
        if not range_str or range_str == "0":
            # 0 means download up to 6 months back, stop on first existing
            from datetime import timedelta

            return timedelta(days=180), True  # (timedelta, stop_on_existing)

        # Normalize the string
        range_str = range_str.lower().strip()

        # Try to extract number and unit
        import re

        # Match patterns like "1w", "2 weeks", "3 days back", "one month back"
        match = re.search(
            r"(\d+|one|two|three|four|five|six)\s*([dwmy]|day|week|month|year)",
            range_str,
        )

        if not match:
            print(f"⚠ Could not parse range '{range_str}', using 1 week")
            from datetime import timedelta

            return timedelta(weeks=1), False

        # Convert word numbers to integers
        word_to_num = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6}
        num_str = match.group(1)
        num = word_to_num.get(num_str, int(num_str))

        # Get unit
        unit = match.group(2)[0]  # First letter: d, w, m, y

        from datetime import timedelta

        if unit == "d":
            return timedelta(days=num), False
        elif unit == "w":
            return timedelta(weeks=num), False
        elif unit == "m":
            return relativedelta(months=num), False
        elif unit == "y":
            return relativedelta(years=num), False
        else:
            return timedelta(weeks=1), False

    time_delta, stop_on_existing = parse_time_range(DOWNLOAD_RANGE)
    cutoff_date = datetime.now() - time_delta

    print(f"📅 Download range: {DOWNLOAD_RANGE}")
    print(f"   Cutoff date: {cutoff_date.strftime('%Y-%m-%d %H:%M:%S')}")
    print(
        f"   Mode: {'Stop on first existing file' if stop_on_existing else 'Skip existing, continue to cutoff'}"
    )
    print()

    # Set download directory
    download_dir = Path(__file__).parent.parent.parent / "data"
    download_dir.mkdir(parents=True, exist_ok=True)

    print(f"✓ Download directory: {download_dir}")
    print()

    downloaded = 0
    skipped = 0
    failed = 0
    image_index = 1

    import time

    print("Starting download from carousel...")
    print("=" * 60)

    while True:
        print(f"\n[Image {image_index}]")

        # Get current image info
        img_data = get_current_carousel_image()

        if not img_data:
            print("  ✗ Could not get image data - end of carousel or no image found")
            break

        img_url = img_data.get("url")
        camera_name = img_data.get("cameraName", "").strip()
        timestamp = img_data.get("timestamp", "").strip()
        full_text = img_data.get("fullText", "").strip()
        text_count = img_data.get("textCount", 0)
        width = img_data.get("width", 0)
        height = img_data.get("height", 0)

        # Debug output
        if full_text:
            print(f"  Found: {full_text}")
        if not camera_name:
            print(f"  Text elements searched: {text_count}")
        print(f"  Camera: {camera_name if camera_name else 'Unknown'}")
        if timestamp:
            print(f"  Timestamp: {timestamp}")

        # Convert timestamp to ISO format and check against cutoff
        iso_timestamp = None
        image_date = None
        if timestamp:
            try:
                # Parse format like "01/11/2026 09:53 AM"
                dt = datetime.strptime(timestamp, "%m/%d/%Y %I:%M %p")
                image_date = dt
                # Convert to ISO format suitable for filename: YYYY-MM-DD_HH-MM-SS
                iso_timestamp = dt.strftime("%Y-%m-%d_%H-%M-%S")
            except ValueError:
                # If parsing fails, use sanitized original
                iso_timestamp = re.sub(r'[<>:"/\\|?*]', "_", timestamp)

        # Check if image is older than cutoff date
        if image_date and image_date < cutoff_date:
            print(
                f"  ⏸ Image date {image_date.strftime('%Y-%m-%d')} is before cutoff {cutoff_date.strftime('%Y-%m-%d')}"
            )
            print(f"\n{'=' * 60}")
            print("✓ Reached cutoff date - stopping download")
            break

        # Create hash from URL for unique filename
        url_hash = hashlib.md5(img_url.encode()).hexdigest()[:16]

        # Build filename with camera name and timestamp
        if camera_name and iso_timestamp:
            original_filename = f"{camera_name}_{iso_timestamp}_{url_hash}"
        elif camera_name:
            original_filename = f"{camera_name}_{url_hash}"
        else:
            original_filename = f"secacam_{url_hash}"

        print(f"  Filename: {original_filename}")
        print(f"  Size: {width}x{height}px")

        # Use camera name for folder organization
        camera_folder_name = camera_name if camera_name else "Unknown"

        # Sanitize camera name for folder
        camera_folder_name = re.sub(r'[<>:"/\\|?*]', "_", camera_folder_name)

        # Create camera-specific subfolder
        camera_path = download_dir / camera_folder_name
        camera_path.mkdir(parents=True, exist_ok=True)

        # Sanitize filename and ensure .jpg extension
        safe_filename = re.sub(r'[<>:"/\\|?*]', "_", original_filename)
        # Add .jpg extension if not present
        if not safe_filename.lower().endswith((".jpg", ".jpeg", ".png", ".gif")):
            safe_filename += ".jpg"
        filepath = camera_path / safe_filename

        # Check if already downloaded
        if filepath.exists():
            existing_size = filepath.stat().st_size / 1024
            print(f"  ⊘ Already exists ({existing_size:.1f} KB)")
            skipped += 1

            if stop_on_existing:
                # Mode: Stop on first existing file (for downloading latest only)
                print(f"\n{'=' * 60}")
                print("✓ Found existing file - stopping here!")
                print(
                    "  This file was already downloaded, assuming all newer images are also downloaded."
                )
                break
            else:
                # Mode: Skip existing and continue (for ensuring completeness to date)
                print("  → Skipping, continuing to next image...")
                # Don't download, just move to next
                pass  # Continue to navigation below

        if not filepath.exists():
            print("  ⬇ Downloading...")

            try:
                # Try downloading with Safari first
                success = download_with_safari(img_url, filepath)

                if not success:
                    # Fallback to direct download
                    try:
                        urllib.request.urlretrieve(img_url, filepath)
                        success = True
                    except:
                        success = False

                if success and filepath.exists():
                    downloaded += 1
                    size_kb = filepath.stat().st_size / 1024
                    print(f"  ✓ Saved ({size_kb:.1f} KB)")
                else:
                    failed += 1
                    print("  ✗ Failed to download")

            except Exception as e:
                failed += 1
                print(f"  ✗ Error: {e}")

        # Move to next image
        print("  → Next...")
        prev_url = img_url
        prev_filename = original_filename

        if not click_next_in_carousel():
            print("  ⓘ Could not navigate to next image - end of carousel")
            break

        # Wait for the new image to load
        print("  ⏳ Waiting for new image to load...")
        time.sleep(1.5)

        # Verify we got a different image
        next_img_data = get_current_carousel_image()
        if not next_img_data:
            print("  ⓘ No image found after navigation - carousel may have closed")
            break

        next_url = next_img_data.get("url", "")
        next_filename = next_img_data.get("filename", "")

        print(
            f"  Debug: URL match={next_url == prev_url}, Filename match={next_filename == prev_filename}"
        )
        print(f"  Debug: Prev='{prev_filename}', Next='{next_filename}'")

        # Compare both URL and filename to detect if we actually moved
        if next_url == prev_url and next_filename == prev_filename:
            print("  ⓘ Same image detected after navigation - reached end of carousel")
            break

        image_index += 1

    # Summary
    print(f"\n{'=' * 60}")
    print("✓ Download Complete!")
    print(f"  Images processed: {image_index}")
    print(f"  Downloaded: {downloaded}")
    print(f"  Skipped (existing): {skipped}")
    print(f"  Failed: {failed}")
    print(f"  Saved to: {download_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
