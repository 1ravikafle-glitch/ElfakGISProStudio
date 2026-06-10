import os
import zipfile
import tempfile
import pandas as pd
import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from flask import Flask, render_template, request, jsonify
from shapely.geometry import Point, LineString, Polygon, box, MultiPolygon

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
# SAFE MULTIPOLYGON
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
# LOAD SHAPEFILE (ZIP)
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
        return None

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
# BOUNDARY
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
# COMPARTMENT
# =====================================================
def build_compartment(df, crs):

    results = []

    for comp, g in df.groupby("Compartment"):

        g = g.sort_values("Order")

        coords = list(zip(g["X"], g["Y"]))
        if len(coords) < 3:
            continue

        coords.append(coords[0])

        poly = Polygon(coords).buffer(0)
        if not poly.is_valid:
            continue

        line = LineString(coords)

        results.append({
            "Comp": comp,
            "polygon": poly,
            "line": line,
            "points": [Point(xy) for xy in coords],
            "area": poly.area / 10000,
            "perimeter": poly.length
        })

    return results


# =====================================================
# SAMPLE FISHNET (SAFE FIXED)
# =====================================================
def build_sample(poly, w, h, rows, cols):

    if poly is None or poly.is_empty:
        return []

    if not poly.is_valid:
        poly = poly.buffer(0)

    minx, miny, maxx, maxy = poly.bounds
    samples = []

    # auto-limit grid to avoid empty output
    max_cols = max(1, int((maxx - minx) // w) + 1)
    max_rows = max(1, int((maxy - miny) // h) + 1)

    cols = min(cols, max_cols)
    rows = min(rows, max_rows)

    for i in range(cols):
        for j in range(rows):

            x = minx + i * w
            y = miny + j * h

            cell = box(x, y, x + w, y + h)

            if not cell.intersects(poly):
                continue

            clipped = cell.intersection(poly)

            if not clipped.is_empty:
                samples.append(clipped.centroid)

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

    ax.set_title("BOUNDARY")
    ax.set_axis_off()
    return save_plot()


def preview_compartment(data):
    fig, ax = plt.subplots()

    for d in data:
        ax.plot(*d["polygon"].exterior.xy, "red")
        ax.plot(*d["line"].xy, "black")

        for p in d["points"]:
            ax.scatter(p.x, p.y, s=8)

        c = d["polygon"].centroid
        ax.text(c.x, c.y, str(d["Comp"]))

    ax.set_title("COMPARTMENT MAP")
    ax.set_axis_off()
    return save_plot()


def preview_sample(poly, samples):
    fig, ax = plt.subplots()

    ax.plot(*poly.exterior.xy, "blue", linewidth=2)

    for s in samples:
        ax.scatter(s.x, s.y, s=15, color="red")

    ax.set_title("BOUNDARY + SAMPLE PLOTS")
    ax.set_axis_off()

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
    mode = request.form.get("mode")

    path = os.path.join(UPLOAD, file.filename)
    file.save(path)

    zone = request.form.get("zone", "45")

    try:

        poly = None
        line = None
        pts = None
        df = None
        crs = get_crs(zone)

        # ================= ZIP =================
        if path.endswith(".zip"):
            geom = load_shapefile(path)
            geom_list = normalize_geom(geom)

            if not geom_list:
                return jsonify({"error": "No valid geometry in shapefile"})

            poly = geom_list[0]

            if poly.geom_type == "MultiPolygon":
                poly = list(poly.geoms)[0]

            line = LineString(poly.exterior.coords)
            pts = [Point(xy) for xy in poly.exterior.coords]

        # ================= EXCEL =================
        else:
            df = pd.read_excel(path)

        # ================= BOUNDARY =================
        if mode == "boundary":

            if poly:
                return jsonify({"image": preview_whole(poly, line, pts)})

            pts, line, poly = build_whole(df)
            return jsonify({"image": preview_whole(poly, line, pts)})

        # ================= COMPARTMENT =================
        if mode == "compartment":

            if df is None:
                return jsonify({"error": "Compartment needs Excel file"})

            data = build_compartment(df, crs)
            return jsonify({"image": preview_compartment(data)})

        # ================= SAMPLE (FIXED) =================
        if mode == "sample":

            base_poly = poly if poly else build_whole(df)[2]

            if base_poly is None:
                return jsonify({"error": "No boundary found for sampling"})

            w = float(request.form.get("cell_width", 100))
            h = float(request.form.get("cell_height", 100))
            rows = int(request.form.get("rows", 10))
            cols = int(request.form.get("cols", 10))

            samples = build_sample(base_poly, w, h, rows, cols)

            if len(samples) == 0:
                return jsonify({"error": "No sample points generated. Adjust grid size."})

            return jsonify({"image": preview_sample(base_poly, samples)})

        return jsonify({"error": "invalid mode"})

    except Exception as e:
        return jsonify({"error": str(e)})


if __name__ == "__main__":
    app.run(debug=True)
