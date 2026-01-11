#!/usr/bin/env python3
"""
Download images from ZEISS Secacam gallery automatically.
This script uses AppleScript to get data from Safari's logged-in session.
"""

import json
import os
import re
import subprocess
import urllib.request
from pathlib import Path
from urllib.parse import urlparse


def get_safari_page_content():
    """Get the HTML content of the current Safari tab using AppleScript."""
    applescript = """
    tell application "Safari"
        if (count of windows) = 0 then
            return "ERROR: No Safari windows open"
        end if
        
        set currentURL to URL of current tab of front window
        
        -- Execute JavaScript to get page HTML
        tell current tab of front window
            set pageHTML to do JavaScript "document.documentElement.outerHTML"
        end tell
        
        return pageHTML
    end tell
    """

    try:
        result = subprocess.run(
            ["osascript", "-e", applescript], capture_output=True, text=True, timeout=10
        )

        if result.returncode == 0:
            return result.stdout
        else:
            print(f"❌ AppleScript error: {result.stderr}")
            return None
    except Exception as e:
        print(f"❌ Error running AppleScript: {e}")
        return None


def navigate_to_gallery():
    """Navigate to the gallery page from wherever we are."""
    applescript = """
    tell application "Safari"
        tell current tab of front window
            -- First check current URL
            set currentURL to do JavaScript "window.location.href"
            
            if currentURL contains "gallery" then
                return "already_on_gallery"
            end if
            
            -- Try to find and click Gallery link/button
            set jsCode to "
                // Look for Gallery link or button
                let found = false;
                
                // Try direct selectors first
                const galleryLink = document.querySelector('a[href*=gallery], a[href*=\\\\/gallery]');
                if (galleryLink) {
                    galleryLink.click();
                    found = true;
                }
                
                // If not found, search through all links and buttons
                if (!found) {
                    const allElements = document.querySelectorAll('a, button, [role=button], nav *');
                    for (const el of allElements) {
                        const text = el.textContent.trim().toLowerCase();
                        const ariaLabel = el.getAttribute('aria-label') || '';
                        if (text === 'gallery' || text === 'galerie' || ariaLabel.toLowerCase().includes('gallery')) {
                            el.click();
                            found = true;
                            break;
                        }
                    }
                }
                
                // Last resort: try to navigate directly
                if (!found) {
                    const currentUrl = window.location.href;
                    const baseUrl = currentUrl.split('/').slice(0, 3).join('/');
                    const lang = currentUrl.includes('/en/') ? 'en' : 'de';
                    window.location.href = baseUrl + '/' + lang + '/gallery';
                    found = true;
                }
                
                found ? 'clicked' : 'not_found';
            "
            
            set result to do JavaScript jsCode
            return result
        end tell
    end tell
    """

    try:
        result = subprocess.run(
            ["osascript", "-e", applescript], capture_output=True, text=True, timeout=10
        )

        if result.returncode == 0:
            output = result.stdout.strip()
            if "already_on_gallery" in output:
                return "already"
            elif "clicked" in output:
                return "navigated"
        return "failed"
    except Exception as e:
        print(f"⚠ Navigation error: {e}")
        return "failed"


def get_camera_buttons():
    """Get all camera/album buttons from the current Safari page."""
    applescript = """
    tell application "Safari"
        tell current tab of front window
            set jsCode to "
                JSON.stringify(
                    Array.from(document.querySelectorAll('button, a, [role=button], div[onclick]'))
                        .filter(btn => {
                            const text = btn.textContent.trim();
                            const hasValidLength = text.length >= 3 && text.length <= 50;
                            // Exclude navigation and common UI elements
                            const excludeWords = ['login', 'logout', 'back', 'next', 'close', 'menu', 'settings', 
                                                 'dashboard', 'gallery', 'profile', 'account', 'help', 'support',
                                                 'home', 'cancel', 'ok', 'yes', 'no', 'save', 'edit', 'delete'];
                            const notCommon = !excludeWords.includes(text.toLowerCase());
                            const isVisible = btn.offsetParent !== null;
                            // Only include if it doesn't look like it navigates to another page
                            const isNotNavigation = !btn.href || btn.href.includes('gallery');
                            return hasValidLength && notCommon && isVisible && isNotNavigation;
                        })
                        .map(btn => ({
                            text: btn.textContent.trim(),
                            selector: btn.id ? '#' + btn.id : 
                                     btn.className ? '.' + btn.className.split(' ').join('.') :
                                     btn.tagName.toLowerCase()
                        }))
                )
            "
            
            set jsonData to do JavaScript jsCode
            return jsonData
        end tell
    end tell
    """

    try:
        result = subprocess.run(
            ["osascript", "-e", applescript], capture_output=True, text=True, timeout=10
        )

        if result.returncode == 0:
            return json.loads(result.stdout.strip())
        return []
    except:
        return []


def click_camera_button(button_text):
    """Click a camera/album button by its text content."""
    # Escape single quotes in button text
    safe_text = button_text.replace("'", "\\'")

    applescript = f"""
    tell application "Safari"
        activate
        delay 0.5
        
        tell current tab of front window
            -- Try to click using multiple methods
            set jsCodeClick to "
                const allButtons = Array.from(document.querySelectorAll('button, a, [role=button], div[onclick]'));
                const targetButton = allButtons.find(btn => btn.textContent.trim() === '{safe_text}' && btn.offsetParent !== null);
                
                if (targetButton) {{
                    // Scroll into view first
                    targetButton.scrollIntoView({{ behavior: 'auto', block: 'center' }});
                    
                    // Try multiple click methods
                    try {{
                        // Method 1: Direct click
                        targetButton.click();
                    }} catch(e1) {{
                        try {{
                            // Method 2: Dispatch mouse event
                            const event = new MouseEvent('click', {{
                                view: window,
                                bubbles: true,
                                cancelable: true
                            }});
                            targetButton.dispatchEvent(event);
                        }} catch(e2) {{
                            // Method 3: If it has onclick, call it
                            if (targetButton.onclick) {{
                                targetButton.onclick();
                            }}
                        }}
                    }}
                    'clicked';
                }} else {{
                    'not_found';
                }}
            "
            
            set clickResult to do JavaScript jsCodeClick
            delay 1.5
            return clickResult
        end tell
    end tell
    """

    try:
        result = subprocess.run(
            ["osascript", "-e", applescript], capture_output=True, text=True, timeout=15
        )

        if result.returncode == 0 and "clicked" in result.stdout:
            return True
        return False
    except:
        return False


def get_safari_image_urls():
    """Extract all image URLs and camera/album info from the current Safari page using JavaScript."""
    applescript = """
    tell application "Safari"
        if (count of windows) = 0 then
            return "ERROR: No Safari windows open"
        end if
        
        tell current tab of front window
            -- JavaScript to extract all image URLs and album info
            set jsCode to "
                JSON.stringify({
                    images: Array.from(document.querySelectorAll('img')).map(img => ({
                        src: img.src,
                        alt: img.alt || '',
                        title: img.title || '',
                        parent: img.closest('[data-album], [data-camera], section, article')?.textContent?.trim().split('\\\\n')[0] || ''
                    })),
                    backgrounds: Array.from(document.querySelectorAll('*')).map(el => {
                        const bg = window.getComputedStyle(el).backgroundImage;
                        const match = bg.match(/url\\\\(['\\\"]?([^'\\\"]+)['\\\"]?\\\\)/);
                        return match ? match[2] : null;
                    }).filter(url => url && url.startsWith('http')),
                    links: Array.from(document.querySelectorAll('a[href*=jpg], a[href*=jpeg], a[href*=png], a[href*=webp]')).map(a => a.href),
                    currentURL: window.location.href,
                    albumName: document.querySelector('h1, h2, .album-name, .camera-name, [class*=album], [class*=camera]')?.textContent?.trim() || 
                               document.title.split('|')[0].trim() ||
                               'Unknown',
                    availableAlbums: Array.from(document.querySelectorAll('button, a')).map(el => el.textContent.trim()).filter(text => text && text.length > 2 && text.length < 50)
                })
            "
            
            set jsonData to do JavaScript jsCode
            return jsonData
        end tell
    end tell
    """

    try:
        result = subprocess.run(
            ["osascript", "-e", applescript], capture_output=True, text=True, timeout=15
        )

        if result.returncode == 0:
            data = json.loads(result.stdout.strip())
            return data
        else:
            print(f"❌ AppleScript error: {result.stderr}")
            return None
    except Exception as e:
        print(f"❌ Error extracting image URLs: {e}")
        return None


def download_with_safari(url, filepath):
    """Download a file using Safari's session via AppleScript."""
    # AppleScript to download using Safari's session
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
            
            -- Wait for download to complete
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
            # Check if it's an error
            if "error" in result.stdout.lower():
                return False

            # Decode base64 and save
            import base64

            try:
                image_data = base64.b64decode(result.stdout.strip())
                with open(filepath, "wb") as f:
                    f.write(image_data)
                return True
            except:
                return False
        return False
    except Exception as e:
        print(f"  ⚠ Safari download failed: {e}")
        return False


def click_first_thumbnail():
    """Click the first thumbnail in the gallery to open the carousel."""
    applescript = """
    tell application "Safari"
        activate
        delay 0.3
        
        tell current tab of front window
            set jsCode to "
                // Find all gallery thumbnails (exclude UI images)
                const thumbnails = Array.from(document.querySelectorAll('img'))
                    .filter(img => {
                        const src = img.src || '';
                        const skip = ['logo', 'icon', 'avatar', 'bg_', 'ic_', 'button'];
                        const isUIElement = skip.some(s => src.toLowerCase().includes(s));
                        const isVisible = img.offsetParent !== null;
                        const hasReasonableSize = img.width > 50 && img.height > 50;
                        return !isUIElement && isVisible && hasReasonableSize;
                    });
                
                if (thumbnails.length > 0) {
                    const first = thumbnails[0];
                    first.scrollIntoView({ behavior: 'auto', block: 'center' });
                    
                    // Try clicking the thumbnail or its parent link
                    const clickTarget = first.closest('a, button, [onclick], [role=button]') || first;
                    clickTarget.click();
                    'clicked';
                } else {
                    'not_found';
                }
            "
            
            set result to do JavaScript jsCode
            delay 2
            return result
        end tell
    end tell
    """

    try:
        result = subprocess.run(
            ["osascript", "-e", applescript], capture_output=True, text=True, timeout=15
        )
        return result.returncode == 0 and "clicked" in result.stdout
    except:
        return False


def get_current_carousel_image():
    """Get the current full-resolution image URL and original filename from the carousel."""
    applescript = """
    tell application "Safari"
        tell current tab of front window
            set jsCode to "
                // Look for the full-resolution image in the carousel
                const selectors = [
                    '[class*=carousel] img:not([src*=thumb]):not([src*=icon]):not([src*=logo])',
                    '[class*=modal] img:not([src*=thumb]):not([src*=icon]):not([src*=logo])',
                    '[class*=lightbox] img:not([src*=thumb]):not([src*=icon]):not([src*=logo])',
                    '[id*=carousel] img:not([src*=thumb]):not([src*=icon]):not([src*=logo])',
                    '[id*=modal] img:not([src*=thumb]):not([src*=icon]):not([src*=logo])',
                    '[role=dialog] img:not([src*=thumb]):not([src*=icon]):not([src*=logo])',
                    'img[src*=original]', 'img[src*=full]', 'img[src*=large]'
                ];
                
                let fullResImg = null;
                for (const sel of selectors) {
                    const img = document.querySelector(sel);
                    if (img && img.src && img.src.startsWith('http') && img.naturalWidth > 500) {
                        fullResImg = img;
                        break;
                    }
                }
                
                // Fallback: get the largest visible image
                if (!fullResImg) {
                    const allImages = Array.from(document.querySelectorAll('img'))
                        .filter(img => {
                            const skip = ['logo', 'icon', 'avatar', 'bg_', 'ic_', 'button', 'thumb'];
                            const isUI = skip.some(s => (img.src || '').toLowerCase().includes(s));
                            return !isUI && img.offsetParent !== null && img.src.startsWith('http');
                        })
                        .sort((a, b) => (b.naturalWidth * b.naturalHeight) - (a.naturalWidth * a.naturalHeight));
                    fullResImg = allImages[0];
                }
                
                if (fullResImg) {
                    // Try to extract original filename from URL or data attributes
                    const url = fullResImg.src;
                    const urlParts = url.split('/');
                    const urlFilename = urlParts[urlParts.length - 1].split('?')[0];
                    
                    // Look for data attributes with filename
                    const dataFilename = fullResImg.getAttribute('data-filename') || 
                                        fullResImg.getAttribute('data-name') ||
                                        fullResImg.closest('[data-filename]')?.getAttribute('data-filename');
                    
                    JSON.stringify({
                        url: url,
                        filename: dataFilename || urlFilename,
                        width: fullResImg.naturalWidth,
                        height: fullResImg.naturalHeight
                    });
                } else {
                    'not_found';
                }
            "
            
            set result to do JavaScript jsCode
            return result
        end tell
    end tell
    """

    try:
        result = subprocess.run(
            ["osascript", "-e", applescript], capture_output=True, text=True, timeout=10
        )

        if result.returncode == 0 and result.stdout.strip() != "not_found":
            return json.loads(result.stdout.strip())
        return None
    except:
        return None


def click_next_in_carousel():
    """Click the 'next' button in the carousel to go to the next image."""
    applescript = """
    tell application "Safari"
        tell current tab of front window
            set jsCode to "
                // Look for next button with various selectors
                const nextSelectors = [
                    '[class*=next]', '[aria-label*=next]', '[aria-label*=Next]',
                    '[class*=arrow][class*=right]', '[class*=forward]',
                    'button[class*=right]', '[data-action*=next]',
                    '.carousel-control-next', '.slick-next',
                    '[title*=next]', '[title*=Next]'
                ];
                
                let nextBtn = null;
                for (const sel of nextSelectors) {
                    const btn = document.querySelector(sel);
                    if (btn && btn.offsetParent !== null) {
                        // Make sure it's not a 'previous' button
                        const text = btn.textContent.toLowerCase();
                        const classes = btn.className.toLowerCase();
                        const aria = (btn.getAttribute('aria-label') || '').toLowerCase();
                        
                        if (!text.includes('prev') && !classes.includes('prev') && !aria.includes('prev')) {
                            nextBtn = btn;
                            break;
                        }
                    }
                }
                
                // Try keyboard shortcut as fallback
                if (nextBtn) {
                    nextBtn.click();
                    'clicked';
                } else {
                    // Try arrow key
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


def close_carousel():
    """Close the carousel/modal/lightbox."""
    applescript = """
    tell application "Safari"
        tell current tab of front window
            set jsCode to "
                // Try various methods to close the carousel
                const closeSelectors = [
                    '[class*=close]', '[aria-label*=close]', '[aria-label*=Close]',
                    'button[class*=close]', '[data-dismiss]', '.modal-close',
                    '[class*=overlay]', '[class*=backdrop]'
                ];
                
                let closed = false;
                for (const sel of closeSelectors) {
                    const closeBtn = document.querySelector(sel);
                    if (closeBtn && closeBtn.offsetParent !== null) {
                        closeBtn.click();
                        closed = true;
                        break;
                    }
                }
                
                // Try Escape key
                if (!closed) {
                    document.dispatchEvent(new KeyboardEvent('keydown', { 
                        key: 'Escape', 
                        code: 'Escape', 
                        keyCode: 27,
                        bubbles: true
                    }));
                    closed = true;
                }
                
                closed ? 'closed' : 'attempted';
            "
            
            do JavaScript jsCode
        end tell
    end tell
    """

    try:
        subprocess.run(
            ["osascript", "-e", applescript], capture_output=True, text=True, timeout=5
        )
    except:
        pass


def download_gallery_images(download_dir, camera_name=None):
    """
    Download all images from the ZEISS Secacam gallery using Safari's current session.

    Args:
        download_dir: Directory to save images
        camera_name: Name of the camera/album (optional, will be detected if not provided)
    """

    # Create download directory if it doesn't exist
    download_path = Path(download_dir)
    download_path.mkdir(parents=True, exist_ok=True)

    print("Extracting image URLs from Safari...")

    # Get image data from Safari
    data = get_safari_image_urls()

    if not data:
        print("❌ Failed to extract data from Safari.")
        print("Make sure Safari is open with the gallery page loaded.")
        return False

    current_url = data.get("currentURL", "")

    # Use provided camera name or try to detect it
    if camera_name:
        album_name = camera_name
    else:
        album_name = data.get("albumName", "Unknown")

    # Sanitize album name for use as folder name
    album_name = re.sub(r'[<>:"/\\|?*]', "_", album_name)
    album_name = album_name.strip()
    if not album_name or album_name == "Unknown":
        album_name = "Gallery"

    # Create subfolder for this camera/album
    album_path = download_path / album_name
    album_path.mkdir(parents=True, exist_ok=True)

    print(f"✓ Current page: {current_url}")
    print(f"✓ Album/Camera: {album_name}")
    print(f"✓ Saving to: {album_path}")

    if "gallery" not in current_url.lower():
        print("⚠ Warning: Current Safari page doesn't appear to be the gallery.")
        proceed = input("Continue anyway? (y/n): ").strip().lower()
        if proceed != "y":
            return False

    # Collect all image URLs
    found_images = set()

    # Add img src URLs
    for img_data in data.get("images", []):
        if isinstance(img_data, dict):
            url = img_data.get("src", "")
        else:
            url = img_data
        if url and url.startswith("http"):
            found_images.add(url)

    # Add background image URLs
    for url in data.get("backgrounds", []):
        if url:
            found_images.add(url)

    # Add linked images
    for url in data.get("links", []):
        if url:
            found_images.add(url)

    # Also parse HTML for additional patterns
    page_html = get_safari_page_content()
    if page_html:
        # Look for __NEXT_DATA__
        next_data_match = re.search(
            r'<script id="__NEXT_DATA__"[^>]*>(.*?)</script>', page_html, re.DOTALL
        )
        if next_data_match:
            try:
                next_data = json.loads(next_data_match.group(1))
                print("✓ Found Next.js data, parsing...")
                extract_images_from_json(next_data, found_images, current_url)
            except json.JSONDecodeError:
                pass

        # Look for other patterns
        patterns = [
            r'"(https://[^"]+\.(?:jpg|jpeg|png|gif|webp)[^"]*)"',
            r"'(https://[^']+\.(?:jpg|jpeg|png|gif|webp)[^']*)'",
        ]
        for pattern in patterns:
            matches = re.findall(pattern, page_html, re.IGNORECASE)
            found_images.update(matches)

    # Filter out UI elements
    filtered_images = [
        url
        for url in found_images
        if not any(
            skip in url.lower()
            for skip in ["bg_", "ic_", "logo", "icon", "button", "avatar"]
        )
    ]

    if not filtered_images:
        print("❌ No gallery images found.")
        print(
            f"   Found {len(found_images)} total images, but all appear to be UI elements."
        )
        print(
            "\nTIP: Make sure you've scrolled through the gallery to load all images."
        )

        # Show available albums/cameras
        available = data.get("availableAlbums", [])
        if available:
            unique_albums = sorted(
                set([a for a in available if len(a) > 3 and len(a) < 30])
            )[:10]
            if unique_albums:
                print("\nDetected camera/album buttons:")
                for alb in unique_albums:
                    print(f"  - {alb}")
                print(
                    "\nClick on a camera button to view its images, then run this script again."
                )

        return False

    print(f"\n✓ Found {len(filtered_images)} gallery thumbnails")
    print("\n🎯 Using carousel navigation to download full-resolution images...")

    # Click first thumbnail to open carousel
    print("\nOpening carousel...")
    if not click_first_thumbnail():
        print("❌ Could not open carousel by clicking first thumbnail")
        return False

    print("✓ Carousel opened")

    # Download images by navigating through carousel
    downloaded = 0
    failed = 0
    image_index = 1

    import time

    while True:
        print(f"\n[Image {image_index}] Extracting image data...")

        # Get current image info
        img_data = get_current_carousel_image()

        if not img_data:
            print("  ✗ Could not get image data (might be end of carousel)")
            break

        img_url = img_data.get("url")
        original_filename = img_data.get("filename", f"image_{image_index:04d}.jpg")
        width = img_data.get("width", 0)
        height = img_data.get("height", 0)

        # Sanitize filename
        original_filename = re.sub(r'[<>:"/\\|?*]', "_", original_filename)
        filepath = album_path / original_filename

        print(f"  Original filename: {original_filename}")
        print(f"  Dimensions: {width}x{height}px")

        # Check if already downloaded
        if filepath.exists():
            existing_size = filepath.stat().st_size / 1024
            print(f"  ⊘ Already exists ({existing_size:.1f} KB) - stopping here")
            break

        print("  ⬇ Downloading...")

        try:
            # Try downloading with Safari first (uses authenticated session)
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
                print(f"  ✓ Saved: {original_filename} ({size_kb:.1f} KB)")
            else:
                failed += 1
                print("  ✗ Failed to download")

        except Exception as e:
            failed += 1
            print(f"  ✗ Error: {e}")

        # Try to go to next image
        print("  → Navigating to next image...")
        if not click_next_in_carousel():
            print("  ⓘ No more images or couldn't click next")
            break

        image_index += 1
        time.sleep(0.5)  # Brief pause between images

    # Close carousel when done
    print("\nClosing carousel...")
    close_carousel()
    time.sleep(1)

    print(f"\n{'=' * 60}")
    print(f"✓ Download complete for album: {album_name}")
    print(f"  Images processed: {image_index}")
    print(f"  Successfully downloaded: {downloaded}")
    print(f"  Failed: {failed}")
    print(f"  Saved to: {album_path}")

    # Show available albums for next download
    available = data.get("availableAlbums", [])
    if available:
        unique_albums = sorted(
            set([a for a in available if 3 < len(a) < 30 and a != album_name])
        )[:10]
        if unique_albums:
            print(f"\n{'=' * 60}")
            print("Other detected cameras/albums:")
            for alb in unique_albums:
                if (
                    any(
                        keyword in alb.lower()
                        for keyword in ["kirschensitz", "sauenburg", "camera", "album"]
                    )
                    or alb[0].isupper()
                ):
                    print(f"  - {alb}")
            print("\nTo download another camera's images:")
            print("1. Click the camera button in Safari")
            print("2. Wait for images to load")
            print("3. Run this script again")

    return downloaded > 0


def extract_images_from_json(data, image_set, base_url):
    """Recursively extract image URLs from JSON data."""
    if isinstance(data, dict):
        for key, value in data.items():
            if key in ["src", "url", "image", "thumbnail", "href"] and isinstance(
                value, str
            ):
                if re.match(r".*\.(?:jpg|jpeg|png|gif|webp)", value, re.IGNORECASE):
                    image_set.add(value)
            else:
                extract_images_from_json(value, image_set, base_url)
    elif isinstance(data, list):
        for item in data:
            extract_images_from_json(item, image_set, base_url)


def get_filename_from_url(url, index):
    """Extract or generate a filename from URL."""
    parsed = urlparse(url)
    filename = os.path.basename(parsed.path)

    # If no filename in path, try to extract from query parameters
    if not filename or "." not in filename:
        # Check for filename in query parameters
        from urllib.parse import parse_qs

        query_params = parse_qs(parsed.query)
        for key in ["filename", "file", "name"]:
            if key in query_params:
                filename = query_params[key][0]
                break

    # If still no filename, generate one with index
    if not filename or "." not in filename:
        filename = f"image_{index:04d}.jpg"

    # Sanitize filename (keep alphanumeric, dots, dashes, underscores)
    # But preserve the file extension
    name_parts = filename.rsplit(".", 1)
    if len(name_parts) == 2:
        name, ext = name_parts
        name = re.sub(r"[^\w\-]", "_", name)
        filename = f"{name}.{ext}"
    else:
        filename = re.sub(r"[^\w\-.]", "_", filename)

    return filename


def main():
    """Main function."""
    print("ZEISS Secacam Gallery Image Downloader")
    print("=" * 60)
    print("\nThis script uses your active Safari session - no login needed!")

    # Gallery URL
    gallery_url = "https://secacam-webapp.zeiss.com/en/gallery"

    # Check if Safari is running and open gallery if needed
    check_and_open_safari = f'''
    tell application "Safari"
        if not running then
            activate
            delay 1
        end if
        
        -- Check if gallery page is already open in any tab
        set galleryOpen to false
        repeat with w in windows
            repeat with t in tabs of w
                if URL of t contains "secacam-webapp.zeiss.com" then
                    set galleryOpen to true
                    set current tab of window 1 to t
                    set index of w to 1
                    exit repeat
                end if
            end repeat
            if galleryOpen then exit repeat
        end repeat
        
        -- If not open, create new tab with gallery URL
        if not galleryOpen then
            if (count of windows) = 0 then
                make new document with properties {{URL:"{gallery_url}"}}
            else
                tell front window
                    set current tab to (make new tab with properties {{URL:"{gallery_url}"}})
                end tell
            end if
            delay 3
        end if
        
        activate
        return "opened"
    end tell
    '''

    print("\nChecking Safari...")
    try:
        result = subprocess.run(
            ["osascript", "-e", check_and_open_safari],
            capture_output=True,
            text=True,
            timeout=15,
        )
        if result.returncode == 0:
            print("✓ Safari is ready with Secacam gallery")
        else:
            print("⚠ Could not open Safari automatically")
            print("   Please open Safari and navigate to:")
            print(f"   {gallery_url}")
            return
    except Exception as e:
        print(f"⚠ Error opening Safari: {e}")
        print(f"   Please manually open: {gallery_url}")
        return

    # Set download directory
    download_dir = "/Users/wri2lr/Downloads/Secacam_Gallery_Images"

    print(f"Download directory: {download_dir}")

    # Navigate to Gallery page first
    print("\n" + "=" * 60)
    print("Step 1: Navigating to Gallery page...")
    print("=" * 60)

    nav_result = navigate_to_gallery()

    if nav_result == "already":
        print("✓ Already on Gallery page")
    elif nav_result == "navigated":
        print("✓ Successfully navigated to Gallery")
        import time

        time.sleep(3)  # Wait for gallery page to fully load
    else:
        print("❌ Failed to navigate to Gallery page")
        print(
            "   Please manually click on 'Gallery' in Safari and run the script again."
        )
        return

    # Get all camera buttons
    print("\n" + "=" * 60)
    print("Step 2: Detecting available cameras/albums...")
    print("=" * 60)
    buttons = get_camera_buttons()

    # Filter to likely camera names (capitalized, reasonable length)
    # Skip "All images", "Favorites", and navigation buttons
    skip_albums = {
        "all images",
        "favorites",
        "all",
        "favorite",
        "dashboard",
        "gallery",
        "settings",
        "profile",
        "logout",
        "login",
        "account",
        "help",
        "support",
        "home",
        "back",
        "menu",
        "close",
        "cancel",
        "reload",
        "cameras",
        "contacts",
        "shop",
        "language",
        "other",
        "sign out",
        "sign in",
    }

    # More strict filtering for camera buttons
    camera_buttons = []
    for btn in buttons:
        text = btn["text"]
        if not text or text.lower() in skip_albums:
            continue

        # Length check
        if len(text) < 3 or len(text) > 30:
            continue

        # Must start with uppercase OR be all uppercase
        if not (text[0].isupper() or text.isupper()):
            continue

        # Skip if it contains common UI words
        ui_words = [
            "click",
            "view",
            "show",
            "hide",
            "more",
            "less",
            "new",
            "add",
            "remove",
            "filter",
        ]
        if any(word in text.lower() for word in ui_words):
            continue

        # Skip if it's mostly numbers or special characters
        alpha_chars = sum(c.isalpha() for c in text)
        if alpha_chars < len(text) * 0.5:
            continue

        camera_buttons.append(btn)

    if not camera_buttons:
        print("⚠ No camera buttons detected. Downloading from current view...")
        print("=" * 60 + "\n")
        download_gallery_images(download_dir)
        return

    # Remove duplicates
    seen = set()
    unique_cameras = []
    for btn in camera_buttons:
        if btn["text"] not in seen:
            seen.add(btn["text"])
            unique_cameras.append(btn)

    print(f"\n✓ Found {len(unique_cameras)} cameras/albums:")
    for i, btn in enumerate(unique_cameras, 1):
        print(f"  {i}. {btn['text']}")

    print("\n" + "=" * 60)
    print("Starting automatic download for all cameras...")
    print("=" * 60 + "\n")

    total_downloaded = 0
    successful_cameras = 0

    for idx, camera in enumerate(unique_cameras, 1):
        camera_name = camera["text"]
        print(f"\n{'=' * 60}")
        print(f"[{idx}/{len(unique_cameras)}] Processing: {camera_name}")
        print("=" * 60)

        # Click the camera button
        print(f"Clicking button: {camera_name}...")
        if click_camera_button(camera_name):
            print("✓ Button clicked")

            # Wait for images to load
            print("Waiting for images to load...")
            import time

            time.sleep(3)  # Give time for images to load

            # Download images for this camera (pass the camera name)
            success = download_gallery_images(download_dir, camera_name)

            if success:
                successful_cameras += 1

            # Small delay between cameras
            time.sleep(1)
        else:
            print(f"✗ Could not click button for {camera_name}")

    # Final summary
    print(f"\n{'=' * 60}")
    print("=" * 60)
    print("ALL CAMERAS PROCESSED")
    print("=" * 60)
    print(f"Total cameras processed: {len(unique_cameras)}")
    print(f"Successful downloads: {successful_cameras}")
    print(f"Download directory: {download_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
