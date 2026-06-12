import os
import uuid
import zipfile
import shutil
import json
import pandas as pd
import geopandas as gpd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from flask import Flask, request, jsonify, send_file, send_from_directory

from shapely.geometry import Polygon, Point, LineString

app = Flask(__name__)

UPLOAD = "uploads"
OUTPUT = "outputs"

os.makedirs(UPLOAD, exist_ok=True)
os.makedirs(OUTPUT, exist_ok=True)

# ================= FILE READER =================
def read_input(file):
    name = file.filename.lower()
    if name.endswith(".csv"):
        try:
            return pd.read_csv(file, encoding="utf-8-sig")
        except:
            file.seek(0)
            return pd.read_csv(file, encoding="latin1")
    return pd.read_excel(file)

# ================= CRS =================
def get_crs(zone):
    return f"EPSG:326{int(zone)}"

# ================= NORMALIZE =================
def normalize_columns(df):
    df.columns = (
        df.columns.astype(str)
        .str.strip()
        .str.lower()
        .str.replace(" ", "")
        .str.replace("-", "")
        .str.replace("_", "")
    )
    return df

# ================= COLUMN RESOLVER =================
def resolve_col(df, mapping, key):
    df = normalize_columns(df)
    mapping = mapping or {}

    def norm(x):
        return str(x).strip().lower().replace(" ", "").replace("-", "").replace("_", "")

    clean = {norm(k): v for k, v in mapping.items()}
    cols = {norm(c): c for c in df.columns}

    k = norm(key)

    if k in clean and clean[k]:
        target = norm(clean[k])
        if target in cols:
            return cols[target]

        raise ValueError(f"Column '{clean[k]}' not found")

    raise ValueError(f"Missing UI mapping for '{key}'")

# ================= SAFE POLYGON =================
def safe_polygon(coords):
    if len(coords) < 3:
        return None
    if coords[0] != coords[-1]:
        coords.append(coords[0])

    poly = Polygon(coords)
    if not poly.is_valid:
        poly = poly.buffer(0)

    return poly if not poly.is_empty else None

# ================= GROUP A =================
def group_a(df, forest, crs, out, mapping):
    df = normalize_columns(df)

    x = resolve_col(df, mapping, "X")
    y = resolve_col(df, mapping, "Y")
    order = resolve_col(df, mapping, "Order")

    df = df.sort_values(order)

    coords = list(zip(df[x], df[y]))

    poly = safe_polygon(coords)
    line = LineString(coords)

    poly_gdf = gpd.GeoDataFrame([{
        "Forest": forest,
        "geometry": poly
    }], crs=crs)

    line_gdf = gpd.GeoDataFrame([{
        "Forest": forest,
        "geometry": line
    }], crs=crs)

    pts_gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df[x], df[y]),
        crs=crs
    )

    poly_gdf.to_file(os.path.join(out, "polygon.shp"))
    line_gdf.to_file(os.path.join(out, "line.shp"))
    pts_gdf.to_file(os.path.join(out, "points.shp"))

    return poly_gdf, line_gdf, pts_gdf

# ================= ROUTES =================

@app.route("/")
def home():
    return "Forest Survey Plotter API is running"

@app.route("/get-columns", methods=["POST"])
def get_columns():
    file = request.files["file"]
    df = read_input(file)
    df = normalize_columns(df)
    return jsonify(list(df.columns))

@app.route("/upload", methods=["POST"])
def upload():
    file = request.files["file"]
    mode = request.form["mode"]
    zone = request.form["zone"]
    forest = request.form.get("forest", "FOREST")
    mapping = json.loads(request.form.get("mapping", "{}"))

    run_id = str(uuid.uuid4())
    out = os.path.join(OUTPUT, run_id)
    os.makedirs(out, exist_ok=True)

    crs = get_crs(zone)

    df = read_input(file)

    if mode == "A":
        poly, line, pts = group_a(df, forest, crs, out, mapping)

        # preview
        fig, ax = plt.subplots()
        poly.plot(ax=ax, color="yellow")
        line.plot(ax=ax, color="black")
        pts.plot(ax=ax, color="red")
        plt.axis("off")

        img_path = os.path.join(out, "output.png")
        plt.savefig(img_path)
        plt.close()

    zip_path = os.path.join(out, "result.zip")
    shutil.make_archive(zip_path.replace(".zip",""), "zip", out)

    return jsonify({
        "run_id": run_id,
        "download": f"/download/{run_id}"
    })

@app.route("/download/<run_id>")
def download(run_id):
    path = os.path.join(OUTPUT, run_id, "result.zip")
    return send_file(path, as_attachment=True)

@app.route("/outputs/<run_id>/<file>")
def outputs(run_id, file):
    return send_from_directory(os.path.join(OUTPUT, run_id), file)

# ================= RUN =================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
