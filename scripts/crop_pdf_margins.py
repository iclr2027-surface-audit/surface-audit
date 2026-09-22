#!/usr/bin/env python3
"""Trim the empty top and bottom margin of a one-page PDF figure, in place or to a new file.

This is the exact crop applied to the paper's figures (Figures 1-4): render the page at 400 dpi,
find the first and last rows containing ink (any pixel darker than 250/255), and move the page's
top and bottom edges to that ink plus 0.75 pt of padding. Left and right edges are left unchanged,
so the figure keeps its width and scales to \\textwidth exactly as before.

Usage:  python scripts/crop_pdf_margins.py IN.pdf [OUT.pdf]      (OUT defaults to IN, overwritten)
Requires poppler's `pdftoppm` on PATH.
"""
import glob, os, subprocess, sys, tempfile
import numpy as np
from PIL import Image
from pypdf import PdfReader, PdfWriter

DPI, INK, PAD_PT = 400, 250, 0.75

def crop(src, out):
    r = PdfReader(src); pg = r.pages[0]; mb = pg.mediabox
    x0, y0, x1, y1 = float(mb.left), float(mb.bottom), float(mb.right), float(mb.top); H = y1 - y0
    with tempfile.TemporaryDirectory() as d:
        subprocess.run(["pdftoppm", "-r", str(DPI), "-f", "1", "-l", "1", "-png", src, os.path.join(d, "p")], check=True)
        a = np.array(Image.open(sorted(glob.glob(os.path.join(d, "p*.png")))[0]).convert("L")) < INK
    rows = a.any(axis=1); ph = a.shape[0]
    t = int(np.argmax(rows)); b = int(ph - 1 - np.argmax(rows[::-1])); s = ph / H
    nt = min(y1 - t / s + PAD_PT, y1); nb = max(y1 - (b + 1) / s - PAD_PT, y0)
    for box in (pg.mediabox, pg.cropbox):
        box.lower_left = (x0, nb); box.upper_right = (x1, nt)
    w = PdfWriter(); w.add_page(pg)
    with open(out, "wb") as f: w.write(f)
    print(f"cropped {os.path.basename(src)}: {H:.1f}pt -> {nt - nb:.1f}pt tall", flush=True)

if __name__ == "__main__":
    if len(sys.argv) not in (2, 3): sys.exit(__doc__)
    crop(sys.argv[1], sys.argv[2] if len(sys.argv) == 3 else sys.argv[1])
