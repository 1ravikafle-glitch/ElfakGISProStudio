import os
import zipfile
import tempfile
import pandas as pd
import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from flask import Flask, render_template, request, jsonify, send_file
from shapely.geometry import Polygon, Point, LineString, box, MultiPolygon

app = Flask(__name__)

UPLOAD = "uploads"
OUTPUT = "outputs"
STATIC = "static"

os.makedirs(UPLOAD, exist_ok=True)
os.makedirs(OUTPUT, exist_ok=True)
os.makedirs(STATIC, exist_ok=True)


# =====================================================
# CRS
# =====================================================
def get_crs(zone):
    return "EPSG:32644" if str(zone) == "44" else "EPSG:32645"


# =====================================================
# SAFE MULTIPOLYGON HANDLER (FIX CRASH)
# =====================================================
def normalize_polygon(poly):
    if poly is None or poly.is_empty:
        return None

    if isinstance(poly, Polygon):
        return poly

    if isinstance(poly, MultiPolygon):
        # take largest polygon safely
        return max(poly.geoms, key=lambda g: g.area)

    return None


# =====================================================
# POINTS
# =====================================================
def build_points(df):
    df = df.copy()
    df["X"] = pd.to_numeric(df["X"], errors="coerce")
    df["Y"] = pd.to_numeric(df["Y"], errors="coerce")
    df = df.dropna(subset=["X", "Y"])

    points = [Point(xy) for xy in zip(df["X"], df["Y"])]
    return df, points


# =====================================================
# WHOLE BOUNDARY
# =====================================================
def build_whole(df):
    df = df.sort_values("Order")
    df, pts = build_points(df)

    coords = list(zip(df["X"], df["Y"]))
    if len(coords) < 3:
        return None, None, None, None

    if coords[0] != coords[-1]:
        coords.append(coords[0])

    line = LineString(coords)
    poly = normalize_polygon(Polygon(coords).buffer(0))

    return df, pts, line, poly


# =====================================================
# SEGMENTED BOUNDARY (SAFE)
# =====================================================
def build_segmented(df):
    df, _ = build_points(df)

    if "Compartment" not in df.columns:
        return []

    results = []

    for comp, g in df.groupby("Compartment"):
        g = g.sort_values("Order")

        coords = list(zip(g["X"], g["Y"]))
        if len(coords) < 3:
            continue

        if coords[0] != coords[-1]:
            coords.append(coords[0])

        poly = normalize_polygon(Polygon(coords).buffer(0))
        if poly is None:
            continue

        line = LineString(coords)
        pts = [Point(xy) for xy in coords]

        results.append({
            "comp": comp,
            "polygon": poly,
            "line": line,
            "points": pts
        })

    return results


# =====================================================
# SAMPLE FISHNET
# =====================================================
def build_sample(poly, w, h):
    poly = normalize_polygon(poly)
    if poly is None:
        return []

    minx, miny, maxx, maxy = poly.bounds
    samples = []

    x = minx
    while x < maxx:
        y = miny
        while y < maxy:
            cell = box(x, y, x + w, y + h)
            if cell.intersects(poly):
                samples.append(cell.centroid)
            y += h
        x += w

    return samples


# =====================================================
# PLOT SAVE
# =====================================================
def save_plot():
    path = os.path.join(STATIC, "preview.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return "/static/preview.png"


# =====================================================
# PREVIEWS
# =====================================================

def preview_whole(poly, line, pts):
    fig, ax = plt.subplots(figsize=(7, 6))

    if poly:
        x, y = poly.exterior.xy
        ax.plot(x, y, "green")

    if line:
        x, y = line.xy
        ax.plot(x, y, "black")

    for p in pts:
        ax.scatter(p.x, p.y, s=8)

    ax.set_axis_off()
    ax.set_title("WHOLE BOUNDARY")
    return save_plot()


def preview_segmented(seg):
    fig, ax = plt.subplots(figsize=(7, 6))

    for s in seg:
        p = s["polygon"]
        x, y = p.exterior.xy
        ax.plot(x, y, "red", linewidth=1)

        x, y = s["line"].xy
        ax.plot(x, y, "black")

        for pt in s["points"]:
            ax.scatter(pt.x, pt.y, s=6)

        c = p.centroid
        ax.text(c.x, c.y, str(s["comp"]))

    ax.set_axis_off()
    ax.set_title("SEGMENTED BOUNDARY")
    return save_plot()


def preview_sample(poly, samples):
    fig, ax = plt.subplots(figsize=(7, 6))

    if poly:
        x, y = poly.exterior.xy
        ax.plot(x, y, "red")

    for p in samples:
        ax.scatter(p.x, p.y, s=10)

    ax.set_axis_off()
    ax.set_title("SAMPLE GRID")
    return save_plot()


# =====================================================
# ROUTES
# =====================================================
@app.route("/")
def home():
    return render_template("index.html")


@app.route("/preview", methods=["POST"])
def preview():
    file = request.files["file"]
    mode = request.form.get("mode", "boundary")

    path = os.path.join(UPLOAD, file.filename)
    file.save(path)

    df = pd.read_excel(path)

    if mode == "boundary":
        df, pts, line, poly = build_whole(df)
        return jsonify({"image": preview_whole(poly, line, pts)})

    if mode == "compartment":
        seg = build_segmented(df)
        return jsonify({"image": preview_segmented(seg)})

    if mode == "sample":
        df, pts, line, poly = build_whole(df)
        w = float(request.form.get("cell_width", 100))
        h = float(request.form.get("cell_height", 100))
        samples = build_sample(poly, w, h)
        return jsonify({"image": preview_sample(poly, samples)})

    return jsonify({"error": "Invalid mode"})


if __name__ == "__main__":
    app.run(debug=True)
