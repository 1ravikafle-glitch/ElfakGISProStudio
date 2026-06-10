import os
import zipfile
import tempfile
import pandas as pd
import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from flask import Flask, render_template, request, jsonify
from shapely.geometry import Polygon, Point, LineString, box, MultiPolygon

app = Flask(__name__)

UPLOAD = "uploads"
STATIC = "static"

os.makedirs(UPLOAD, exist_ok=True)
os.makedirs(STATIC, exist_ok=True)


# =====================================================
# CRS
# =====================================================
def get_crs(zone):
    return "EPSG:32644" if str(zone) == "44" else "EPSG:32645"


# =====================================================
# SAFE GEOMETRY NORMALIZER (IMPORTANT FIX)
# =====================================================
def normalize_geom(geom):
    if geom is None:
        return []

    if isinstance(geom, Polygon):
        return [geom]

    if isinstance(geom, MultiPolygon):
        return [g for g in geom.geoms if g.is_valid]

    return []


# =====================================================
# LOAD SHAPEFILE FROM ZIP (SAFE)
# =====================================================
def load_shapefile(zip_path):
    temp_dir = tempfile.mkdtemp()

    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(temp_dir)

    shp = None
    for f in os.listdir(temp_dir):
        if f.endswith(".shp"):
            shp = os.path.join(temp_dir, f)
            break

    if shp is None:
        raise Exception("No .shp file found in ZIP")

    gdf = gpd.read_file(shp)

    if gdf.empty:
        raise Exception("Empty shapefile")

    return gdf.geometry.unary_union


# =====================================================
# POINTS
# =====================================================
def build_points(df):
    df = df.copy()

    if not {"X", "Y"}.issubset(df.columns):
        raise Exception("Missing X or Y column")

    df["X"] = pd.to_numeric(df["X"], errors="coerce")
    df["Y"] = pd.to_numeric(df["Y"], errors="coerce")
    df = df.dropna(subset=["X", "Y"])

    if df.empty:
        raise Exception("No valid coordinates found")

    points = [Point(xy) for xy in zip(df["X"], df["Y"])]
    return df, points


# =====================================================
# WHOLE BOUNDARY
# =====================================================
def build_whole_boundary(df):
    df = df.sort_values("Order")
    df, points = build_points(df)

    coords = list(zip(df["X"], df["Y"]))

    if len(coords) < 3:
        raise Exception("Not enough points for polygon")

    if coords[0] != coords[-1]:
        coords.append(coords[0])

    line = LineString(coords)
    poly = Polygon(coords)

    if not poly.is_valid:
        poly = poly.buffer(0)

    return df, points, line, poly


# =====================================================
# SEGMENTED BOUNDARY (SAFE)
# =====================================================
def build_segmented(df):
    df, _ = build_points(df)

    results = []

    if "Compartment" not in df.columns:
        raise Exception("Compartment column missing")

    for comp, group in df.groupby("Compartment"):
        group = group.sort_values("Order")

        coords = list(zip(group["X"], group["Y"]))

        if len(coords) < 3:
            continue

        if coords[0] != coords[-1]:
            coords.append(coords[0])

        poly = Polygon(coords)

        if not poly.is_valid:
            poly = poly.buffer(0)

        if poly.is_empty:
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
def build_sample(poly, cell_w, cell_h):

    if poly is None or poly.is_empty:
        return []

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
# PREVIEW HELPER
# =====================================================
def save_plot():
    path = os.path.join(STATIC, "preview.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return "/static/preview.png"


# =====================================================
# PREVIEW WHOLE
# =====================================================
def preview_whole(poly, line, points):

    fig, ax = plt.subplots(figsize=(7, 6))

    if poly and not poly.is_empty:
        x, y = poly.exterior.xy
        ax.plot(x, y, "green", linewidth=2)

    if line:
        x, y = line.xy
        ax.plot(x, y, "black", linewidth=1)

    for p in points:
        ax.scatter(p.x, p.y, color="blue", s=10)

    ax.set_axis_off()
    ax.set_title("WHOLE BOUNDARY")

    return save_plot()


# =====================================================
# PREVIEW SEGMENTED
# =====================================================
def preview_segmented(seg):

    fig, ax = plt.subplots(figsize=(7, 6))

    for s in seg:

        if s["polygon"] and not s["polygon"].is_empty:
            x, y = s["polygon"].exterior.xy
            ax.plot(x, y, "red", linewidth=1)

        x, y = s["line"].xy
        ax.plot(x, y, "black", linewidth=1)

        for p in s["points"]:
            ax.scatter(p.x, p.y, color="blue", s=8)

        c = s["polygon"].centroid
        ax.text(c.x, c.y, str(s["comp"]), fontsize=8)

    ax.set_axis_off()
    ax.set_title("SEGMENTED BOUNDARY")

    return save_plot()


# =====================================================
# PREVIEW SAMPLE
# =====================================================
def preview_sample(poly, samples):

    fig, ax = plt.subplots(figsize=(7, 6))

    if poly:
        x, y = poly.exterior.xy
        ax.plot(x, y, "red", linewidth=1)

    for p in samples:
        ax.scatter(p.x, p.y, color="blue", s=12)

    ax.set_axis_off()
    ax.set_title("SAMPLE FISHNET")

    return save_plot()


# =====================================================
# HOME
# =====================================================
@app.route("/")
def home():
    return render_template("index.html")


# =====================================================
# PREVIEW API
# =====================================================
@app.route("/preview", methods=["POST"])
def preview():

    file = request.files["file"]
    mode = request.form.get("mode", "boundary")

    path = os.path.join(UPLOAD, file.filename)
    file.save(path)

    try:

        # ZIP
        if path.endswith(".zip"):
            geom = normalize_geom(load_shapefile(path))

            if not geom:
                return jsonify({"error": "Empty geometry"})

            poly = geom[0]

            return jsonify({
                "image": preview_whole(
                    poly,
                    LineString(poly.exterior.coords),
                    [Point(xy) for xy in poly.exterior.coords]
                )
            })

        # EXCEL
        df = pd.read_excel(path)

        if mode == "boundary":
            df, pts, line, poly = build_whole_boundary(df)
            return jsonify({"image": preview_whole(poly, line, pts)})

        if mode == "compartment":
            seg = build_segmented(df)
            return jsonify({"image": preview_segmented(seg)})

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
