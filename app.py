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
# EXCEL → POINTS (COMMON CORE LOGIC)
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
    polygon = Polygon(coords).buffer(0)

    return df, points, line, polygon


# =====================================================
# SEGMENTED BOUNDARY (COMPARTMENT LOGIC)
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
        polygon = Polygon(coords).buffer(0)

        if not polygon.is_valid:
            continue

        results.append((group, line, polygon))

    return results


# =====================================================
# SAMPLE PLOT (FISHNET SYSTEM)
# =====================================================
def build_sample_plots(polygon, cell_w, cell_h):

    minx, miny, maxx, maxy = polygon.bounds

    sample_points = []

    x = minx
    while x < maxx:
        y = miny
        while y < maxy:

            cell = box(x, y, x + cell_w, y + cell_h)

            if cell.intersects(polygon):
                sample_points.append(cell.centroid)

            y += cell_h

        x += cell_w

    return sample_points


# =====================================================
# PREVIEW
# =====================================================
def make_preview(boundary=None, lines=None, points=None, title="Preview"):

    fig, ax = plt.subplots(figsize=(7, 6))

    # polygon
    if isinstance(boundary, Polygon):
        x, y = boundary.exterior.xy
        ax.plot(x, y, "r-", linewidth=1)

    elif isinstance(boundary, list):
        for g in boundary:
            x, y = g.exterior.xy
            ax.plot(x, y, "r-", linewidth=1)

    # lines
    if lines:
        if isinstance(lines, LineString):
            x, y = lines.xy
            ax.plot(x, y, "black", linewidth=1)
        else:
            for l in lines:
                x, y = l.xy
                ax.plot(x, y, "black", linewidth=1)

    # points
    if points:
        for p in points:
            ax.scatter(p.x, p.y, color="blue", s=8)

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

        # ================= ZIP =================
        if path.endswith(".zip"):
            geom = load_shapefile(path)
            return jsonify({"image": make_preview(boundary=geom, title="Shapefile")})

        # ================= EXCEL =================
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

            polys = [p[2] for p in seg]
            lines = [p[1] for p in seg]
            pts = []

            for g, _, _ in seg:
                pts += [Point(xy) for xy in zip(g["X"], g["Y"])]

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

    # ================= ZIP =================
    if path.endswith(".zip"):
        geom = load_shapefile(path)
        gdf = gpd.GeoDataFrame(geometry=[geom], crs=crs)

        points = gdf.copy()
        points["geometry"] = gdf.centroid

        lines = gdf.copy()
        lines["geometry"] = gdf.geometry.apply(lambda g: LineString(g.exterior.coords))

        polygons = gdf.copy()

    # ================= EXCEL =================
    else:
        df = pd.read_excel(path)

        if mode == "compartment":
            seg = build_segmented(df)

            gdf = gpd.GeoDataFrame(
                geometry=[p[2] for p in seg],
                crs=crs
            )

        else:
            df, pts, line, poly = build_whole_boundary(df)

            gdf = gpd.GeoDataFrame(geometry=[poly], crs=crs)

        # POINT + LINE + POLYGON
        points = gdf.copy()
        points["geometry"] = gdf.centroid

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
