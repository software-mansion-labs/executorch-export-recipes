"""Render `layout_verify.png` — the trace input for the PP-DocLayoutV3 export.

The export must be traced on a page with real document structure: torch.export
records the `.contiguous()` copies the deformable attention needs only when the
input looks like a document, and tracing on random noise elides them, which
yields a PTE that silently computes garbage on real pages (trick #2 in
`builder.py`). The page is generated rather than vendored so its provenance is
this file and nothing else: no third-party or user document is committed here.

    uv run --project exporter_models/paddleocr_layout \
        python exporter_models/paddleocr_layout/_layout/make_verify_page.py
"""
import os

from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "layout_verify.png")
W, H = 1000, 1414
MARGIN = 80

TITLE = "Document Layout Analysis"
SUBTITLE = "A synthetic page for export tracing"
BODY = [
    "A layout detector reads a page as a set of regions rather than a stream of",
    "characters. Each region carries a class, a box and a score, and the reading",
    "order is recovered from the geometry afterwards. This page exists only to give",
    "the exporter a realistic input to trace on, so the shapes and the strides that",
    "the traced graph records match the ones a real document produces at runtime.",
]
BODY2 = [
    "The blocks below cover the classes that matter most for the pipeline: headings,",
    "paragraphs, a table with ruled cells, and a figure with its caption. Nothing on",
    "the page is copied from another source; every glyph here is drawn by the script",
    "that generated it.",
]
TABLE = [
    ["Region", "Count", "Mean score"],
    ["paragraph_title", "4", "0.94"],
    ["text", "11", "0.91"],
    ["table", "1", "0.88"],
    ["figure", "1", "0.86"],
]

# Any grotesque with a full Latin set works; these cover macOS and Linux CI.
_FONTS = [
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
]
_BOLD = [
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
]


def _font(paths, size):
    for path in paths:
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    return ImageFont.load_default(size)


def render():
    page = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(page)
    title, bold, body, small = (
        _font(_BOLD, 46),
        _font(_BOLD, 26),
        _font(_FONTS, 22),
        _font(_FONTS, 18),
    )
    y = MARGIN

    d.text((MARGIN, y), TITLE, font=title, fill="black")
    y += 62
    d.text((MARGIN, y), SUBTITLE, font=small, fill=(90, 90, 90))
    y += 34
    d.line([(MARGIN, y), (W - MARGIN, y)], fill=(160, 160, 160), width=2)
    y += 40

    for line in BODY:
        d.text((MARGIN, y), line, font=body, fill=(20, 20, 20))
        y += 34
    y += 30

    d.text((MARGIN, y), "1. Regions", font=bold, fill="black")
    y += 44
    for line in BODY2:
        d.text((MARGIN, y), line, font=body, fill=(20, 20, 20))
        y += 34
    y += 40

    # Ruled table — the horizontal/vertical rules are what the table class keys on.
    d.text((MARGIN, y), "2. Detected regions", font=bold, fill="black")
    y += 44
    col_w = (W - 2 * MARGIN) // 3
    row_h = 46
    for r, row in enumerate(TABLE):
        top = y + r * row_h
        if r == 0:
            d.rectangle([MARGIN, top, W - MARGIN, top + row_h], fill=(238, 238, 238))
        for c, cell in enumerate(row):
            x = MARGIN + c * col_w
            d.rectangle([x, top, x + col_w, top + row_h], outline=(120, 120, 120))
            d.text((x + 14, top + 12), cell, font=small, fill=(20, 20, 20))
    y += len(TABLE) * row_h + 50

    # Figure block + caption.
    d.text((MARGIN, y), "3. Figure", font=bold, fill="black")
    y += 44
    fig_h = 260
    d.rectangle([MARGIN, y, W - MARGIN, y + fig_h], fill=(228, 230, 236))
    for i in range(6):
        x = MARGIN + 40 + i * 130
        h = 40 + 30 * ((i * 7) % 5)
        d.rectangle([x, y + fig_h - 40 - h, x + 70, y + fig_h - 40], fill=(70, 90, 150))
    y += fig_h + 16
    d.text((MARGIN, y), "Figure 1: a bar chart, drawn as a figure region.", font=small,
           fill=(90, 90, 90))
    y += 50

    for line in ("A closing paragraph keeps the lower half of the page occupied so the",
                 "detector sees text above and below the figure, as it would on a real page."):
        d.text((MARGIN, y), line, font=body, fill=(20, 20, 20))
        y += 34

    d.text((W // 2 - 10, H - MARGIN), "1", font=small, fill=(120, 120, 120))
    page.save(OUT, optimize=True)
    print(f"wrote {OUT} ({os.path.getsize(OUT) / 1024:.0f} KiB)")


if __name__ == "__main__":
    render()
