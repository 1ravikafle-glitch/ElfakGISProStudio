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
# ZIP SHAPEFILE LOADER
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
# POINT GENERATION (Excel → Point)
# =====================================================
def build_points(df):
    df = df.copy()
    df["X"] = pd.to_numeric(df["X"], errors="coerce")
    df["Y"] = pd.to_numeric(df["Y"], errors="coerce")
    df = df.dropna(subset=["X", "Y"])

    points = [Point(xy) for xy in zip(df["X"], df["Y"])]
    return df, points


# =====================================================
# WHOLE BOUNDARY (Point → Line → Polygon)
# =====================================================
def build_whole_boundary(df):

    df = df.sort_values("Order")
    df, points = build_points(df)

    coords = list(zip(df["X"], df["Y"]))

    if coords[0] != coords[-1]:
        coords.append(coords[0])

    line = LineString(coords)
    polygon = Polygon(coords).buffer(0)

    return df, points, line, polygon


# =====================================================
# SEGMENTED BOUNDARY (FULL TKINTER LOGIC MATCH)
# =====================================================
def build_segmented(df):

    df = df.copy()
    df["X"] = pd.to_numeric(df["X"], errors="coerce")
    df["Y"] = pd.to_numeric(df["Y"], errors="coerce")
    df = df.dropna(subset=["X", "Y"])

    results = []

    for comp, group in df.groupby("Compartment"):

        group = group.sort_values("Order")

        # POINTS FROM ROWS (IMPORTANT FIX)
        points = [
            Point(row.X, row.Y)
            for _, row in group.iterrows()
        ]

        coords = [(p.x, p.y) for p in points]

        if len(coords) < 3:
            continue

        if coords[0] != coords[-1]:
            coords.append(coords[0])

        line = LineString(coords)
        polygon = Polygon(coords).buffer(0)

        if not polygon.is_valid:
            continue

        results.append((points, line, polygon))

    return results


# =====================================================
# SAMPLE PLOTS (FISHNET GRID)
# =====================================================
def build_sample_plots(poly, cell_w, cell_h):

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
# PREVIEW ENGINE
# =====================================================
def make_preview(polys=None, lines=None, points=None, title="Preview"):

    fig, ax = plt.subplots(figsize=(7, 6))

    # ---------------- POLYGONS ----------------
    if polys is not None:
        if isinstance(polys, Polygon):
            polys = [polys]

        for p in polys:
            x, y = p.exterior.xy
            ax.plot(x, y, "r-", linewidth=1)

    # ---------------- LINES ----------------
    if lines is not None:
        if isinstance(lines, LineString):
            lines = [lines]

        for l in lines:
            x, y = l.xy
            ax.plot(x, y, "black", linewidth=1)

    # ---------------- POINTS ----------------
    if points is not None:
        for p in points:
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

        # ---------------- ZIP ----------------
        if path.endswith(".zip"):
            geom = load_shapefile(path)
            return jsonify({"image": make_preview(polys=geom, title="Shapefile Boundary")})

        # ---------------- EXCEL ----------------
        df = pd.read_excel(path)

        # ---------------- WHOLE ----------------
        if mode == "boundary":
            df, pts, line, poly = build_whole_boundary(df)
            return jsonify({
                "image": make_preview(poly, line, pts, "Whole Boundary")
            })

        # ---------------- SEGMENTED ----------------
        if mode == "compartment":

            seg = build_segmented(df)

            polys = [s[2] for s in seg]
            lines = [s[1] for s in seg]
            pts = []
            for p, _, _ in seg:
                pts += p

            return jsonify({
                "image": make_preview(polys, lines, pts, "Segmented Boundary")
            })

        # ---------------- SAMPLE ----------------
        if mode == "sample":

            df, _, _, poly = build_whole_boundary(df)

            cell_w = float(request.form.get("cell_width", 100))
            cell_h = float(request.form.get("cell_height", 100))

            samples = build_sample_plots(poly, cell_w, cell_h)

            return jsonify({
                "image": make_preview(poly, None, samples, "Sample Plot (Fishnet)")
            })

        return jsonify({"error": "Invalid mode"})

    except Exception as e:
        return jsonify({"error": str(e)})


# =====================================================
# MAIN EXPORT
# =====================================================
@app.route("/upload", methods=["POST"])
def upload():

    file = request.files["file"]
    mode = request.form["mode"]
    zone = request.form["zone"]

    path = os.path.join(UPLOAD, file.filename)
    file.save(path)

    crs = get_crs(zone)

    # ---------------- ZIP ----------------
    if path.endswith(".zip"):
        geom = load_shapefile(path)
        gdf = gpd.GeoDataFrame(geometry=[geom], crs=crs)

    # ---------------- EXCEL ----------------
    else:
        df = pd.read_excel(path)

        if mode == "compartment":
            seg = build_segmented(df)
            gdf = gpd.GeoDataFrame(geometry=[s[2] for s in seg], crs=crs)

        else:
            df, _, _, poly = build_whole_boundary(df)
            gdf = gpd.GeoDataFrame(geometry=[poly], crs=crs)

    # ---------------- POINT / LINE / POLYGON ----------------
    points = gdf.copy()
    points["geometry"] = points.centroid

    lines = gdf.copy()
    lines["geometry"] = gdf.geometry.apply(lambda g: LineString(g.exterior.coords))

    polygons = gdf.copy()

    # ---------------- OUTPUT ----------------
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
