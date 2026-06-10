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
# SAFE MULTIPOLYGON FIX
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
# LOAD ZIP SHAPEFILE
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

    gdf = gpd.read_file(shp)
    return gdf.geometry.unary_union


# =====================================================
# BUILD POINTS
# =====================================================
def build_points(df):
    df["X"] = pd.to_numeric(df["X"], errors="coerce")
    df["Y"] = pd.to_numeric(df["Y"], errors="coerce")
    df = df.dropna(subset=["X", "Y"])

    return df, [Point(xy) for xy in zip(df["X"], df["Y"])]


# =====================================================
# WHOLE BOUNDARY
# =====================================================
def build_whole(df):
    df = df.sort_values("Order")
    df, pts = build_points(df)

    coords = list(zip(df["X"], df["Y"]))
    if coords[0] != coords[-1]:
        coords.append(coords[0])

    poly = Polygon(coords)
    if not poly.is_valid:
        poly = poly.buffer(0)

    line = LineString(coords)

    return pts, line, poly


# =====================================================
# SEGMENTED BOUNDARY
# =====================================================
def build_segmented(df):
    df, _ = build_points(df)

    if "Compartment" not in df.columns:
        return []

    result = []

    for comp, g in df.groupby("Compartment"):
        g = g.sort_values("Order")

        coords = list(zip(g["X"], g["Y"]))
        if len(coords) < 3:
            continue

        if coords[0] != coords[-1]:
            coords.append(coords[0])

        poly = Polygon(coords)
        if not poly.is_valid:
            poly = poly.buffer(0)

        line = LineString(coords)
        pts = [Point(xy) for xy in coords]

        result.append((comp, pts, line, poly))

    return result


# =====================================================
# SAMPLE PLOT (FISHNET)
# =====================================================
def build_sample(poly, w, h, rows, cols):

    minx, miny, maxx, maxy = poly.bounds
    samples = []

    for i in range(cols):
        for j in range(rows):

            x = minx + i * w
            y = miny + j * h

            cell = box(x, y, x + w, y + h)

            if cell.intersects(poly):
                samples.append(cell.centroid)

    return samples


# =====================================================
# SAVE PLOT
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
    fig, ax = plt.subplots()

    ax.plot(*poly.exterior.xy, "green")
    ax.plot(*line.xy, "black")

    for p in pts:
        ax.scatter(p.x, p.y, s=10)

    ax.set_title("WHOLE BOUNDARY")
    ax.set_axis_off()
    return save_plot()


def preview_segmented(seg):
    fig, ax = plt.subplots()

    for comp, pts, line, poly in seg:
        ax.plot(*poly.exterior.xy, "red")
        ax.plot(*line.xy, "black")

        for p in pts:
            ax.scatter(p.x, p.y, s=8)

        c = poly.centroid
        ax.text(c.x, c.y, str(comp))

    ax.set_title("SEGMENTED")
    ax.set_axis_off()
    return save_plot()


def preview_sample(poly, samples):
    fig, ax = plt.subplots()

    ax.plot(*poly.exterior.xy, "red")

    for s in samples:
        ax.scatter(s.x, s.y, s=12)

    ax.set_title("SAMPLE PLOTS")
    ax.set_axis_off()
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
    mode = request.form.get("mode")

    path = os.path.join(UPLOAD, file.filename)
    file.save(path)

    zone = request.form.get("zone", "45")

    try:

        # ZIP
        if path.endswith(".zip"):
            geom = normalize_geom(load_shapefile(path))
            poly = geom[0]

            return jsonify({
                "image": preview_whole(
                    poly,
                    LineString(poly.exterior.coords),
                    [Point(xy) for xy in poly.exterior.coords]
                )
            })

        df = pd.read_excel(path)

        # WHOLE
        if mode == "boundary":
            pts, line, poly = build_whole(df)
            return jsonify({"image": preview_whole(poly, line, pts)})

        # SEGMENTED
        if mode == "compartment":
            seg = build_segmented(df)
            return jsonify({"image": preview_segmented(seg)})

        # SAMPLE
        if mode == "sample":

            pts, line, poly = build_whole(df)

            w = float(request.form.get("cell_width", 100))
            h = float(request.form.get("cell_height", 100))
            rows = int(request.form.get("rows", 10))
            cols = int(request.form.get("cols", 10))

            samples = build_sample(poly, w, h, rows, cols)

            return jsonify({"image": preview_sample(poly, samples)})

        return jsonify({"error": "invalid mode"})

    except Exception as e:
        return jsonify({"error": str(e)})


# =====================================================
# RUN
# =====================================================
if __name__ == "__main__":
    app.run(debug=True)
