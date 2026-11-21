#!/usr/bin/env python3
"""Draw the figure of the paper's evaluation: IOP proof length against
argument size, with and without zero knowledge.

    zo0k paper                       # writes paper/frontiers.csv
    uv run --with reportlab paper/plot_frontiers.py paper/frontiers.csv paper/size-frontiers.pdf

Straight segments join the configurations of a frontier. Open circles mark
the configurations of the tables. The labels of the two panels with the table
sizes are typeset by the paper on top of the graphic.
"""
import csv
import math
import sys

from reportlab.lib.colors import HexColor
from reportlab.pdfgen import canvas

PANELS = [
    ("sublinear", "Sublinear", "MiB", 2**20, [64, 128, 512, 2048, 8192]),
    ("polylog", "Polylogarithmic verifier", "GiB", 2**30, [4, 8, 16]),
]
VARIANTS = [("baseline", "#203d61", "Non-ZK", []), ("zero_knowledge", "#00896b", "ZK", [4, 2])]
GRID, AXIS, TEXT = "#e1e3e5", "#55585b", "#202329"


def main(source, target):
    points = {}
    with open(source, newline="") as stream:
        for row in csv.DictReader(stream):
            curve = points.setdefault(row["protocol"], {}).setdefault(row["variant"], [])
            curve.append((float(row["expected_argument_bytes"]) / 1024, int(row["iop_bytes"]), row["selected"] == "true"))

    page = canvas.Canvas(target, pagesize=(540, 230), invariant=1)
    page.setTitle("IOP proof length versus expected argument size")
    for panel, (protocol, title, unit, unit_bytes, levels) in enumerate(PANELS):
        curves = {variant: [(x, y / unit_bytes, s) for x, y, s in points[protocol][variant]] for variant, *_ in VARIANTS}
        left, right, bottom, top = panel * 270 + 47, panel * 270 + 261, 40, 192
        xs = [x for curve in curves.values() for x, _, _ in curve]
        ys = [y for curve in curves.values() for _, y, _ in curve]
        xmin, xmax, ymin, ymax = min(xs) / 1.12, max(xs) * 1.09, min(ys) / 1.12, max(ys) * 1.15

        def position(x, y):
            return (left + math.log(x / xmin) / math.log(xmax / xmin) * (right - left),
                    bottom + math.log(y / ymin) / math.log(ymax / ymin) * (top - bottom))

        page.setFillColor(HexColor(TEXT))
        page.setFont("Helvetica-Bold", 10)
        page.drawCentredString((left + right) / 2, 214, title)
        page.setLineWidth(0.35)
        page.setFont("Helvetica", 8)
        label_end = float("-inf")
        for x in [200, 500, 1000, 2000, 5000, 10000, 20000]:
            if xmin <= x <= xmax:
                px, _ = position(x, ymin)
                page.setStrokeColor(HexColor(GRID))
                page.line(px, bottom, px, top)
                half = page.stringWidth(f"{x:,}", "Helvetica", 8) / 2
                if px - half >= label_end + 4:
                    page.setFillColor(HexColor(AXIS))
                    page.drawCentredString(px, bottom - 12, f"{x:,}")
                    label_end = px + half
        for y in levels:
            if ymin <= y <= ymax:
                _, py = position(xmin, y)
                page.setStrokeColor(HexColor(GRID))
                page.line(left, py, right, py)
                page.setFillColor(HexColor(AXIS))
                page.drawRightString(left - 5, py - 3, f"{y:,}")
        page.setStrokeColor(HexColor(AXIS))
        page.line(left, top, left, bottom)
        page.line(left, bottom, right, bottom)

        for row, (variant, color, label, dash) in enumerate(VARIANTS):
            page.setStrokeColor(HexColor(color))
            page.setFillColor(HexColor(color))
            page.setLineWidth(1.6)
            page.setDash(dash)
            spots = [position(x, y) for x, y, _ in curves[variant]]
            path = page.beginPath()
            path.moveTo(*spots[0])
            for spot in spots[1:]:
                path.lineTo(*spot)
            page.drawPath(path)
            page.setDash([])
            for x, y in spots[::max(1, len(spots) // 12)]:
                if dash:
                    page.rect(x - 1.6, y - 1.6, 3.2, 3.2, stroke=0, fill=1)
                else:
                    page.circle(x, y, 1.7, stroke=0, fill=1)
            ly, lx = top - 12 - 13 * row, right - 65
            page.setDash(dash)
            page.line(lx, ly, lx + 18, ly)
            page.setDash([])
            page.setFont("Helvetica", 8.5)
            page.drawString(lx + 23, ly - 3, label)
        for variant, color, _, _ in VARIANTS:
            x, y = next(position(x, y) for x, y, selected in curves[variant] if selected)
            page.setStrokeColor(HexColor(color))
            page.setFillColor(HexColor("#ffffff"))
            page.setLineWidth(1.3)
            page.circle(x, y, 3.4, stroke=1, fill=1)

        page.setFillColor(HexColor(TEXT))
        page.setFont("Helvetica", 9)
        page.drawCentredString((left + right) / 2, 11, "Argument size (KiB)")
        page.saveState()
        page.translate(panel * 270 + 10, (bottom + top) / 2)
        page.rotate(90)
        page.drawCentredString(0, 0, f"IOP length ({unit})")
        page.restoreState()
    page.showPage()
    page.save()


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2])
