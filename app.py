import os
import zipfile
import tempfile
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from flask import Flask, render_template, request, send_file, jsonify
from shapely.geometry import Polygon, Point, LineString, box

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
# LOAD SHAPEFILE FROM ZIP
# =====================================================
def load_shapefile(zip_path):
    temp_dir = tempfile.mkdtemp()

    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(temp_dir)

    for f in os.listdir(temp_dir):
        if f.endswith(".shp"):
            return gpd.read_file(os.path.join(temp_dir, f)).geometry.unary_union

    raise Exception("No shapefile found")


# =====================================================
# POINT GENERATION (EXCEL → POINT CORE)
# =====================================================
def build_points(df):
    df = df.copy()
    df["X"] = pd.to_numeric(df["X"], errors="coerce")
    df["Y"] = pd.to_numeric(df["Y"], errors="coerce")
    df = df.dropna(subset=["X", "Y"])

    points = [Point(xy) for xy in zip(df["X"], df["Y"])]
    return df, points


# =====================================================
# WHOLE BOUNDARY (POINT → LINE → POLYGON)
# =====================================================
def build_whole_boundary(df):
    df = df.sort_values("Order")
    df, points = build_points(df)

    coords = list(zip(df["X"], df["Y"]))

    if coords[0] != coords[-1]:
        coords.append(coords[0])

    line = LineString(coords)
    poly = Polygon(coords).buffer(0)

    return df, points, line, poly


# =====================================================
# SEGMENTED BOUNDARY (EXACT LOGIC)
# =====================================================
def build_segmented(df):

    df, _ = build_points(df)

    results = []

    for comp, group in df.groupby("Compartment"):
        group = group.sort_values("Order")

        coords = list(zip(group["X"], group["Y"]))
        if len(coords) < 3:
            continue

        if coords[0] != coords[-1]:
            coords.append(coords[0])

        line = LineString(coords)
        poly = Polygon(coords).buffer(0)

        if not poly.is_valid:
            continue

        pts = [Point(xy) for xy in coords]

        results.append({
            "comp": comp,
            "points": pts,
            "line": line,
            "polygon": poly
        })

    return results


# =====================================================
# SAMPLE PLOT (FISHNET GRID)
# =====================================================
def build_sample(poly, cell_w, cell_h):

    minx, miny, maxx, maxy = poly.bounds

    samples = []

    x = minx
    while x < maxx:
        y = miny
        while y < maxy:

            cell = box(x, y, x + cell_w, y + cell_h)

            if cell.intersects(poly):
                samples.append(cell.centroid)

            y += cell_h
        x += cell_w

    return samples


# =====================================================
# ================= PREVIEW 1 =========================
# WHOLE BOUNDARY PREVIEW
# =====================================================
def preview_whole(poly, line, points):

    fig, ax = plt.subplots(figsize=(7, 6))

    x, y = poly.exterior.xy
    ax.plot(x, y, "green", linewidth=2)

    x2, y2 = line.xy
    ax.plot(x2, y2, "black", linewidth=1)

    for p in points:
        ax.scatter(p.x, p.y, color="blue", s=10)

    ax.set_title("WHOLE BOUNDARY (Point → Line → Polygon)")
    ax.set_axis_off()

    path = os.path.join(STATIC, "preview.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()

    return "/static/preview.png"


# =====================================================
# ================= PREVIEW 2 =========================
# SEGMENTED PREVIEW
# =====================================================
def preview_segmented(seg):

    fig, ax = plt.subplots(figsize=(7, 6))

    for s in seg:
        x, y = s["polygon"].exterior.xy
        ax.plot(x, y, "red", linewidth=1)

        x2, y2 = s["line"].xy
        ax.plot(x2, y2, "black", linewidth=1)

        for p in s["points"]:
            ax.scatter(p.x, p.y, color="blue", s=8)

        c = s["polygon"].centroid
        ax.text(c.x, c.y, str(s["comp"]), fontsize=8)

    ax.set_title("SEGMENTED FOREST BOUNDARY")
    ax.set_axis_off()

    path = os.path.join(STATIC, "preview.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()

    return "/static/preview.png"


# =====================================================
# ================= PREVIEW 3 =========================
# SAMPLE PLOT PREVIEW
# =====================================================
def preview_sample(poly, samples):

    fig, ax = plt.subplots(figsize=(7, 6))

    x, y = poly.exterior.xy
    ax.plot(x, y, "red", linewidth=1)

    for p in samples:
        ax.scatter(p.x, p.y, color="blue", s=12)

    ax.set_title("SAMPLE PLOT (FISHNET GRID)")
    ax.set_axis_off()

    path = os.path.join(STATIC, "preview.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()

    return "/static/preview.png"


# =====================================================
# HOME
# =====================================================
@app.route("/")
def home():
    return render_template("index.html")


# =====================================================
# PREVIEW API (3 DIFFERENT OUTPUTS)
# =====================================================
@app.route("/preview", methods=["POST"])
def preview():

    file = request.files["file"]
    mode = request.form.get("mode", "boundary")

    path = os.path.join(UPLOAD, file.filename)
    file.save(path)

    try:

        # ---------------- ZIP ----------------
        if path.endswith(".zip"):
            geom = load_shapefile(path)

            return jsonify({
                "image": preview_whole(geom, LineString(geom.exterior.coords),
                                       [Point(xy) for xy in geom.exterior.coords])
            })

        # ---------------- EXCEL ----------------
        df = pd.read_excel(path)

        # ================= WHOLE =================
        if mode == "boundary":
            df, pts, line, poly = build_whole_boundary(df)
            return jsonify({"image": preview_whole(poly, line, pts)})

        # ================= SEGMENTED =================
        if mode == "compartment":
            seg = build_segmented(df)
            return jsonify({"image": preview_segmented(seg)})

        # ================= SAMPLE =================
        if mode == "sample":
            df, _, _, poly = build_whole_boundary(df)

            cell_w = float(request.form.get("cell_width", 100))
            cell_h = float(request.form.get("cell_height", 100))

            samples = build_sample(poly, cell_w, cell_h)

            return jsonify({"image": preview_sample(poly, samples)})

        return jsonify({"error": "Invalid mode"})

    except Exception as e:
        return jsonify({"error": str(e)})


# =====================================================
# RUN
# =====================================================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
