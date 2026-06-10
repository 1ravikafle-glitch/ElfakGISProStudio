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
# LOAD ZIP SHAPEFILE
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
# WHOLE BOUNDARY (Excel → ordered polygon)
# =====================================================
def build_whole_boundary(df):
    df = df.sort_values("Order")

    coords = list(zip(df["X"], df["Y"]))
    if coords[0] != coords[-1]:
        coords.append(coords[0])

    return Polygon(coords).buffer(0)


# =====================================================
# SEGMENTED BOUNDARY (Compartment-wise)
# =====================================================
def build_segmented(df):
    polygons = []

    df = df.copy()
    df["X"] = pd.to_numeric(df["X"], errors="coerce")
    df["Y"] = pd.to_numeric(df["Y"], errors="coerce")
    df = df.dropna(subset=["X", "Y"])

    for comp, group in df.groupby("Compartment"):
        group = group.sort_values("Order")

        coords = list(zip(group["X"], group["Y"]))

        if len(coords) < 3:
            continue

        if coords[0] != coords[-1]:
            coords.append(coords[0])

        poly = Polygon(coords).buffer(0)

        if poly.is_valid:
            polygons.append(poly)

    return polygons


# =====================================================
# SAMPLE PLOT (REAL FISHNET SYSTEM)
# =====================================================
def build_sample_plots(polygon, cell_w, cell_h):

    minx, miny, maxx, maxy = polygon.bounds

    sample_points = []
    grid_cells = []

    x = minx
    while x < maxx:
        y = miny
        while y < maxy:

            cell = box(x, y, x + cell_w, y + cell_h)

            if cell.intersects(polygon):
                grid_cells.append(cell)
                sample_points.append(cell.centroid)

            y += cell_h

        x += cell_w

    return grid_cells, sample_points


# =====================================================
# PREVIEW RENDER
# =====================================================
def make_preview(boundary, samples=None, title="Preview", mode="boundary"):

    fig, ax = plt.subplots(figsize=(7, 6))

    # ---- boundary ----
    if isinstance(boundary, Polygon):
        x, y = boundary.exterior.xy
        ax.plot(x, y, "r-", linewidth=1)

    elif isinstance(boundary, list):
        for poly in boundary:
            x, y = poly.exterior.xy
            ax.plot(x, y, "r-", linewidth=1)

    # ---- sample plots ----
    if samples:
        for p in samples:
            ax.scatter(p.x, p.y, color="blue", s=10)

    ax.set_title(title)
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
# PREVIEW API
# =====================================================
@app.route("/preview", methods=["POST"])
def preview():

    file = request.files["file"]
    mode = request.form.get("mode", "boundary")

    path = os.path.join(UPLOAD, file.filename)
    file.save(path)

    try:
        # ---- ZIP ----
        if path.endswith(".zip"):
            geom = load_shapefile(path)
            return jsonify({"image": make_preview(geom, title="Shapefile Boundary")})

        # ---- EXCEL ----
        df = pd.read_excel(path)

        if mode == "boundary":
            poly = build_whole_boundary(df)
            return jsonify({"image": make_preview(poly, title="Whole Boundary")})

        if mode == "compartment":
            polys = build_segmented(df)
            return jsonify({"image": make_preview(polys, title="Segmented Boundary")})

        if mode == "sample":

            poly = build_whole_boundary(df)

            cell_w = float(request.form.get("cell_width", 100))
            cell_h = float(request.form.get("cell_height", 100))

            grids, samples = build_sample_plots(poly, cell_w, cell_h)

            return jsonify({
                "image": make_preview(poly, samples, "Sample Plot (Fishnet)")
            })

        return jsonify({"error": "Invalid mode"})

    except Exception as e:
        return jsonify({"error": str(e)})


# =====================================================
# MAIN PROCESS (EXPORT SHP)
# =====================================================
@app.route("/upload", methods=["POST"])
def upload():

    file = request.files["file"]
    mode = request.form["mode"]
    zone = request.form["zone"]

    path = os.path.join(UPLOAD, file.filename)
    file.save(path)

    crs = get_crs(zone)

    # ================= INPUT =================
    if path.endswith(".zip"):
        geom = load_shapefile(path)
        gdf = gpd.GeoDataFrame(geometry=[geom], crs=crs)

    else:
        df = pd.read_excel(path)

        df["X"] = pd.to_numeric(df["X"], errors="coerce")
        df["Y"] = pd.to_numeric(df["Y"], errors="coerce")
        df = df.dropna(subset=["X", "Y"])

        if mode == "compartment":
            gdf = gpd.GeoDataFrame(geometry=build_segmented(df), crs=crs)

        else:
            gdf = gpd.GeoDataFrame(geometry=[build_whole_boundary(df)], crs=crs)

    # ================= SAMPLE MODE =================
    if mode == "sample":

        cell_w = float(request.form.get("cell_width", 100))
        cell_h = float(request.form.get("cell_height", 100))

        poly = gdf.geometry.iloc[0]
        grids, samples = build_sample_plots(poly, cell_w, cell_h)

        sample_gdf = gpd.GeoDataFrame(geometry=samples, crs=crs)

        points = sample_gdf

        lines = gdf.copy()
        lines["geometry"] = gdf.geometry.apply(lambda g: LineString(g.exterior.coords))

        polygons = gdf.copy()

    else:
        points = gdf.copy()
        points["geometry"] = points.centroid

        lines = gdf.copy()
        lines["geometry"] = gdf.geometry.apply(lambda g: LineString(g.exterior.coords))

        polygons = gdf.copy()

    # ================= OUTPUT =================
    base = os.path.splitext(file.filename)[0]
    out_dir = os.path.join(OUTPUT, base)
    os.makedirs(out_dir, exist_ok=True)

    points.to_file(os.path.join(out_dir, "points.shp"))
    lines.to_file(os.path.join(out_dir, "lines.shp"))
    polygons.to_file(os.path.join(out_dir, "polygons.shp"))

    zip_name = f"{base}.zip"
    zip_path = os.path.join(OUTPUT, zip_name)

    with zipfile.ZipFile(zip_path, "w") as z:
        for f in os.listdir(out_dir):
            z.write(os.path.join(out_dir, f), arcname=f)

    return render_template(
        "index.html",
        download_file=zip_name,
        stats={
            "count": len(gdf),
            "crs": crs,
            "mode": mode
        }
    )


# =====================================================
# DOWNLOAD
# =====================================================
@app.route("/download/<filename>")
def download(filename):
    return send_file(os.path.join(OUTPUT, filename), as_attachment=True)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
