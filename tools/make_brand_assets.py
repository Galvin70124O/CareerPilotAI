"""
Build the CareerPilot AI brand assets into static/img/.

    python tools/make_brand_assets.py

Needs Pillow (pip install pillow; a dev tool only, the app doesn't need it).
The Open Graph image is rendered from tools/og-image.html with headless Microsoft Edge
or Chrome if one is installed (see make_og_image below).

The logo: a compass ring and a needle that points up and to the right, drawn in black ink with
no tile (printed-report style, see static/css/theme.css). The solid half of the needle is an
arrowhead that pokes out of the ring: direction (compass) + growth (upward arrow). Everything is
drawn on a 32 x 32 grid so it stays sharp at 16px. The PNG icons put the ink mark on a paper
square, so they read on light and dark browser tabs and home screens.
"""
import os
import shutil
import subprocess
import tempfile

from PIL import Image, ImageDraw

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "static", "img")

INK = "#1A1917"           # --text (light) in static/css/theme.css
PAPER = "#F7F5F0"         # --bg (light): ink on paper
INK_DARK = "#EEEBE4"      # --text (dark): the mark on dark backgrounds

# Geometry on a 32 x 32 grid (no tile, so the mark uses more of the square)
ICON_RADIUS = 6           # corner of the paper square in the PNG icons
RING = {"cx": 16, "cy": 16, "r": 12, "width": 2, "opacity": 0.4}
NEEDLE_HEAD = [(26.8, 5.2), (13.1, 13.1), (18.9, 18.9)]   # arrowhead, pointing up-right, past the ring
NEEDLE_TAIL = [(7, 25), (13.1, 13.1), (18.9, 18.9)]       # the other half of the compass needle
TAIL_OPACITY = 0.45


def points(triangle):
    return " ".join(f"{x:g},{y:g}" for x, y in triangle)


def logo_svg(ink, dark_media_ink=None):
    """The ink mark on a transparent background. With dark_media_ink (the favicon) the ink
    switches colour with the browser/OS theme, so it stays visible on a dark tab bar."""
    style = ""
    if dark_media_ink:
        style = (f"<style>.glyph{{fill:{ink}}}.ring{{stroke:{ink}}}"
                 f"@media (prefers-color-scheme:dark){{.glyph{{fill:{dark_media_ink}}}.ring{{stroke:{dark_media_ink}}}}}</style>")
    glyph = "" if dark_media_ink else f' fill="{ink}"'
    ring = "" if dark_media_ink else f' stroke="{ink}"'
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32" width="32" height="32" '
        'role="img" aria-label="CareerPilot AI">'
        f"{style}"
        f'<circle class="ring" cx="{RING["cx"]}" cy="{RING["cy"]}" r="{RING["r"]}" fill="none"{ring} '
        f'stroke-opacity="{RING["opacity"]}" stroke-width="{RING["width"]}"/>'
        f'<polygon class="glyph" points="{points(NEEDLE_TAIL)}"{glyph} fill-opacity="{TAIL_OPACITY}"/>'
        f'<polygon class="glyph" points="{points(NEEDLE_HEAD)}"{glyph}/>'
        "</svg>\n"
    )


def hex_rgb(color):
    color = color.lstrip("#")
    return tuple(int(color[i:i + 2], 16) for i in (0, 2, 4))


def logo_png(size, rounded=True):
    """The ink mark on a paper square. Drawn with Pillow at 8x size, then shrunk (smooth edges)."""
    scale = size * 8 / 32
    big = Image.new("RGBA", (size * 8, size * 8), (0, 0, 0, 0))
    draw = ImageDraw.Draw(big)
    radius = ICON_RADIUS * scale if rounded else 0
    draw.rounded_rectangle([0, 0, size * 8 - 1, size * 8 - 1], radius=radius, fill=PAPER)
    ink = hex_rgb(INK)

    overlay = Image.new("RGBA", big.size, (0, 0, 0, 0))
    o = ImageDraw.Draw(overlay)
    r, w = RING["r"] * scale, RING["width"] * scale
    cx, cy = RING["cx"] * scale, RING["cy"] * scale
    o.ellipse([cx - r - w / 2, cy - r - w / 2, cx + r + w / 2, cy + r + w / 2],
              outline=(*ink, int(255 * RING["opacity"])), width=round(w))
    o.polygon([(x * scale, y * scale) for x, y in NEEDLE_TAIL], fill=(*ink, int(255 * TAIL_OPACITY)))
    o.polygon([(x * scale, y * scale) for x, y in NEEDLE_HEAD], fill=(*ink, 255))
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
        "logo.svg": logo_svg(INK),                        # for light backgrounds
        "logo-dark.svg": logo_svg(INK_DARK),              # for dark backgrounds
        "favicon.svg": logo_svg(INK, INK_DARK),           # follows the browser theme
    }
    for name, svg in files.items():
        with open(os.path.join(OUT, name), "w", encoding="utf-8") as f:
            f.write(svg)
        print("wrote", name)
    logo_png(32).save(os.path.join(OUT, "favicon-32.png"), optimize=True)
    # iOS rounds the corners itself and doesn't like transparency: full square of paper
    logo_png(180, rounded=False).convert("RGB").save(os.path.join(OUT, "apple-touch-icon.png"), optimize=True)
    logo_png(512).save(os.path.join(OUT, "logo-512.png"), optimize=True)
    print("wrote favicon-32.png, apple-touch-icon.png, logo-512.png")
    make_og_image()


if __name__ == "__main__":
    main()
