#!/usr/bin/env python3
"""
Download images from ZEISS Secacam gallery carousel.
This script assumes the carousel is already open in Safari.
"""

import json
import re
import subprocess
import urllib.request
from pathlib import Path


def get_current_carousel_image():
    """Get the current full-resolution image URL and original filename from the carousel."""
    applescript = """
    tell application "Safari"
        tell current tab of front window
            do JavaScript "
                (function() {
                    const allImages = Array.from(document.querySelectorAll('img'))
                        .filter(img => {
                            const src = img.src || '';
                            const isGalleryImage = src.includes('media.secacam.com/getImage');
                            const isLarge = img.naturalWidth > 1000 && img.naturalHeight > 500;
                            return isGalleryImage && isLarge;
                        })
                        .sort((a, b) => (b.naturalWidth * b.naturalHeight) - (a.naturalWidth * a.naturalHeight));
                    
                    let mainImg = allImages.length > 0 ? allImages[0] : null;
                    
                    if (!mainImg) {
                        const largeImages = Array.from(document.querySelectorAll('img'))
                            .filter(img => {
                                const src = img.src || '';
                                const skip = ['ic_', 'icon', 'logo', 'svg'];
                                const isUI = skip.some(s => src.toLowerCase().includes(s));
                                return !isUI && img.naturalWidth > 1000 && img.naturalHeight > 500;
                            })
                            .sort((a, b) => (b.naturalWidth * b.naturalHeight) - (a.naturalWidth * a.naturalHeight));
                        
                        mainImg = largeImages.length > 0 ? largeImages[0] : null;
                    }
                    
                    if (mainImg) {
                        const url = mainImg.src;
                        const alt = mainImg.getAttribute('alt') || '';
                        const title = mainImg.getAttribute('title') || '';
                        let filename = alt || title || 'image_' + Date.now();
                        
                        return JSON.stringify({
                            url: url,
                            filename: filename,
                            width: mainImg.naturalWidth,
                            height: mainImg.naturalHeight
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
            set jsCode to "
                // Look for next button
                const nextSelectors = [
                    'button[aria-label*=next]', 'button[aria-label*=Next]',
                    '[class*=next]:not([class*=prev])', 
                    '[class*=arrow-right]', '[class*=arrowright]',
                    'button[class*=right]:not([class*=left])',
                    '[data-action*=next]', 
                    'button:has(svg[class*=right])',
                    'button:has([class*=chevron-right])',
                    'button:has([class*=arrow-right])'
                ];
                
                let nextBtn = null;
                for (const sel of nextSelectors) {
                    try {
                        const btns = document.querySelectorAll(sel);
                        for (const btn of btns) {
                            if (btn.offsetParent !== null) {
                                const text = btn.textContent.toLowerCase();
                                const classes = (btn.className || '').toLowerCase();
                                const aria = (btn.getAttribute('aria-label') || '').toLowerCase();
                                
                                const isPrev = text.includes('prev') || text.includes('back') || 
                                             classes.includes('prev') || classes.includes('back') ||
                                             aria.includes('prev') || aria.includes('back');
                                
                                if (!isPrev) {
                                    nextBtn = btn;
                                    break;
                                }
                            }
                        }
                        if (nextBtn) break;
                    } catch(e) { continue; }
                }
                
                if (nextBtn) {
                    try {
                        nextBtn.click();
                        'clicked';
                    } catch(e) {
                        try {
                            const event = new MouseEvent('click', {
                                view: window,
                                bubbles: true,
                                cancelable: true
                            });
                            nextBtn.dispatchEvent(event);
                            'clicked';
                        } catch(e2) {
                            'error';
                        }
                    }
                } else {
                    // Try arrow key as fallback
                    document.dispatchEvent(new KeyboardEvent('keydown', { 
                        key: 'ArrowRight', 
                        code: 'ArrowRight', 
                        keyCode: 39,
                        bubbles: true
                    }));
                    'key_pressed';
                }
            "
            
            set result to do JavaScript jsCode
            delay 1.5
            return result
        end tell
    end tell
    """

    try:
        result = subprocess.run(
            ["osascript", "-e", applescript], capture_output=True, text=True, timeout=10
        )
        return result.returncode == 0 and result.stdout.strip() in [
            "clicked",
            "key_pressed",
        ]
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

    # Set download directory
    download_dir = Path("/Users/wri2lr/Downloads/ZEISS_Secacam_Gallery_Images")
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
        original_filename = img_data.get("filename", f"image_{image_index:04d}.jpg")
        width = img_data.get("width", 0)
        height = img_data.get("height", 0)

        print(f"  Filename: {original_filename}")
        print(f"  Size: {width}x{height}px")

        # Extract camera name from filename (format: "Camera Name - timestamp.jpg")
        camera_folder_name = "Unknown"
        if " - " in original_filename:
            camera_folder_name = original_filename.split(" - ")[0].strip()

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

        print(f"  Camera: {camera_folder_name}")

        # Check if already downloaded
        if filepath.exists():
            existing_size = filepath.stat().st_size / 1024
            print(f"  ⊘ Already exists ({existing_size:.1f} KB)")
            print(f"\n{'=' * 60}")
            print("✓ Found existing file - stopping here!")
            print(
                "  This file was already downloaded, assuming all newer images are also downloaded."
            )
            skipped += 1
            break

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
        if not click_next_in_carousel():
            print("  ⓘ Could not navigate to next image - end of carousel")
            break

        image_index += 1
        time.sleep(0.5)

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
