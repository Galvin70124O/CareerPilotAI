"""
Build the CareerPilot AI brand assets into static/img/.

    python tools/make_brand_assets.py

Needs Pillow (pip install pillow; a dev tool only, the app doesn't need it).
The Open Graph image is rendered from tools/og-image.html with headless Microsoft Edge
or Chrome if one is installed (see make_og_image below).

The logo: an indigo rounded tile with a compass ring and a needle that points up and to the
right. The bright half of the needle is an arrowhead that pokes out of the ring: direction
(compass) + growth (upward arrow). Everything is drawn on a 32 x 32 grid so it stays sharp at 16px.
"""
import os
import shutil
import subprocess
import tempfile

from PIL import Image, ImageDraw

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "static", "img")

INDIGO = "#4F46E5"        # light theme accent (same as --accent in app.css)
INDIGO_DARK = "#6366F1"   # dark theme accent

# Geometry on a 32 x 32 grid
TILE_RADIUS = 8
RING = {"cx": 16, "cy": 16, "r": 10, "width": 2, "opacity": 0.35}
NEEDLE_HEAD = [(25, 7), (13.6, 13.6), (18.4, 18.4)]     # arrowhead, pointing up-right, past the ring
NEEDLE_TAIL = [(8.5, 23.5), (13.6, 13.6), (18.4, 18.4)]  # the other half of the compass needle
TAIL_OPACITY = 0.5


def points(triangle):
    return " ".join(f"{x:g},{y:g}" for x, y in triangle)


def logo_svg(tile_color, dark_media_color=None):
    style = ""
    if dark_media_color:   # favicon: switch colour with the browser/OS theme
        style = (f"<style>.tile{{fill:{tile_color}}}@media (prefers-color-scheme:dark)"
                 f"{{.tile{{fill:{dark_media_color}}}}}</style>")
    fill = "" if dark_media_color else f' fill="{tile_color}"'
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32" width="32" height="32" '
        'role="img" aria-label="CareerPilot AI">'
        f"{style}"
        f'<rect class="tile" width="32" height="32" rx="{TILE_RADIUS}"{fill}/>'
        f'<circle cx="{RING["cx"]}" cy="{RING["cy"]}" r="{RING["r"]}" fill="none" stroke="#fff" '
        f'stroke-opacity="{RING["opacity"]}" stroke-width="{RING["width"]}"/>'
        f'<polygon points="{points(NEEDLE_TAIL)}" fill="#fff" fill-opacity="{TAIL_OPACITY}"/>'
        f'<polygon points="{points(NEEDLE_HEAD)}" fill="#fff"/>'
        "</svg>\n"
    )


def logo_png(size, tile_color, rounded=True):
    """Draw the same shapes with Pillow at 8x size, then shrink (smooth edges)."""
    scale = size * 8 / 32
    big = Image.new("RGBA", (size * 8, size * 8), (0, 0, 0, 0))
    draw = ImageDraw.Draw(big)
    radius = TILE_RADIUS * scale if rounded else 0
    draw.rounded_rectangle([0, 0, size * 8 - 1, size * 8 - 1], radius=radius, fill=tile_color)

    overlay = Image.new("RGBA", big.size, (0, 0, 0, 0))
    o = ImageDraw.Draw(overlay)
    r, w = RING["r"] * scale, RING["width"] * scale
    cx, cy = RING["cx"] * scale, RING["cy"] * scale
    o.ellipse([cx - r - w / 2, cy - r - w / 2, cx + r + w / 2, cy + r + w / 2],
              outline=(255, 255, 255, int(255 * RING["opacity"])), width=round(w))
    o.polygon([(x * scale, y * scale) for x, y in NEEDLE_TAIL], fill=(255, 255, 255, int(255 * TAIL_OPACITY)))
    o.polygon([(x * scale, y * scale) for x, y in NEEDLE_HEAD], fill=(255, 255, 255, 255))
    big = Image.alpha_composite(big, overlay)
    return big.resize((size, size), Image.LANCZOS)


def find_browser():
    for path in [r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
                 r"C:\Program Files\Google\Chrome\Application\chrome.exe",
                 shutil.which("google-chrome"), shutil.which("chromium"), shutil.which("microsoft-edge")]:
        if path and os.path.exists(path):
            return path
    return None


def make_og_image():
    """Render tools/og-image.html (1200 x 630) to static/img/og-image.png with a headless browser."""
    browser = find_browser()
    if not browser:
        print("No Edge/Chrome found: skipped og-image.png")
        return
    html = os.path.join(ROOT, "tools", "og-image.html")
    out = os.path.join(OUT, "og-image.png")
    with tempfile.TemporaryDirectory() as profile:
        subprocess.run([browser, "--headless=new", "--disable-gpu", "--hide-scrollbars", f"--user-data-dir={profile}",
                        "--force-device-scale-factor=1", "--window-size=1200,630", "--virtual-time-budget=4000",
                        f"--screenshot={out}", "file:///" + html.replace("\\", "/")],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=120)
    image = Image.open(out).convert("RGB").crop((0, 0, 1200, 630))
    image.save(out, optimize=True)
    print("wrote", out, image.size)


def main():
    os.makedirs(OUT, exist_ok=True)
    files = {
        "logo.svg": logo_svg(INDIGO),                      # for light backgrounds
        "logo-dark.svg": logo_svg(INDIGO_DARK),            # for dark backgrounds
        "favicon.svg": logo_svg(INDIGO, INDIGO_DARK),      # follows the browser theme
    }
    for name, svg in files.items():
        with open(os.path.join(OUT, name), "w", encoding="utf-8") as f:
            f.write(svg)
        print("wrote", name)
    logo_png(32, INDIGO).save(os.path.join(OUT, "favicon-32.png"), optimize=True)
    # iOS rounds the corners itself and doesn't like transparency: full square tile
    logo_png(180, INDIGO, rounded=False).convert("RGB").save(os.path.join(OUT, "apple-touch-icon.png"), optimize=True)
    logo_png(512, INDIGO).save(os.path.join(OUT, "logo-512.png"), optimize=True)
    print("wrote favicon-32.png, apple-touch-icon.png, logo-512.png")
    make_og_image()


if __name__ == "__main__":
    main()
