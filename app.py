import os
import uuid
import zipfile
import shutil
import json
import re
import pandas as pd
import geopandas as gpd
import matplotlib
matplotlib.use("Agg")

from flask import Flask, request, jsonify, send_file
from shapely.geometry import Polygon, Point, LineString

app = Flask(__name__)

UPLOAD = "uploads"
OUTPUT = "outputs"

os.makedirs(UPLOAD, exist_ok=True)
os.makedirs(OUTPUT, exist_ok=True)

# ================= NORMALIZER =================
def norm(col):
    return re.sub(r"[^a-z0-9]", "", str(col).lower())

# ================= KEY MAPS =================
X_KEYS = {"x", "xcoord", "xcoordinate", "east", "easting", "longitude", "lon", "lng"}
Y_KEYS = {"y", "ycoord", "ycoordinate", "north", "northing", "latitude", "lat"}
ORDER_KEYS = {"sn", "sno", "s.n", "sno", "serial", "id", "index", "order", "seq", "rowid", "fid"}

# ================= SAFE GEOMETRY =================
def safe_geom(g):
    if g is None:
        return None
    try:
        if not g.is_valid:
            g = g.buffer(0)

        if g.geom_type == "GeometryCollection":
            g = g.buffer(0)

        if g.geom_type not in ["Polygon", "MultiPolygon", "LineString"]:
            g = g.buffer(0)

        return g
    except:
        return None

# ================= FILE READER =================
def read_input(file):
    name = file.filename.lower()

    # ZIP FIX
    if name.endswith(".zip"):
        tmp = os.path.join(UPLOAD, str(uuid.uuid4()))
        os.makedirs(tmp, exist_ok=True)

        zip_path = os.path.join(tmp, "data.zip")
        file.save(zip_path)

        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(tmp)

        for root, _, files in os.walk(tmp):
            for f in files:
                p = os.path.join(root, f)
                try:
                    if f.endswith(".csv"):
                        return pd.read_csv(p, encoding="utf-8-sig")
                    if f.endswith((".xlsx", ".xls")):
                        return pd.read_excel(p)
                except:
                    continue

        raise ValueError("ZIP contains no valid CSV/XLSX")

    # CSV FIX
    if name.endswith(".csv"):
        try:
            return pd.read_csv(file, encoding="utf-8-sig")
        except:
            file.seek(0)
            return pd.read_csv(file, encoding="latin1")

    # EXCEL FIX
    if name.endswith((".xlsx", ".xls")):
        return pd.read_excel(file)

    raise ValueError("Only CSV / Excel / ZIP supported")

# ================= COLUMN RESOLVER (FIXED CORE ERROR) =================
def resolve_xyz(df, mapping=None):

    df.columns = [str(c).strip() for c in df.columns]

    x = y = order = None

    # 1. UI mapping override
    if mapping:
        x = mapping.get("X")
        y = mapping.get("Y")
        order = mapping.get("Order")

    # 2. AUTO DETECTION
    for col in df.columns:
        c = norm(col)

        if x is None and c in X_KEYS:
            x = col
        if y is None and c in Y_KEYS:
            y = col
        if order is None and c in ORDER_KEYS:
            order = col

    # 3. HARD FIX (handles X,Y,S.N weird headers)
    if x is None:
        for c in df.columns:
            if "x" in c.lower():
                x = c
                break

    if y is None:
        for c in df.columns:
            if "y" in c.lower():
                y = c
                break

    # 4. ORDER fallback
    if order is None:
        df["Order"] = range(1, len(df) + 1)
        order = "Order"

    if x is None or y is None:
        raise ValueError(f"Missing UI mapping X/Y. Columns: {df.columns.tolist()}")

    return x, y, order

# ================= SAFE POLYGON =================
def safe_polygon(coords):
    poly = Polygon(coords)
    if not poly.is_valid:
        poly = poly.buffer(0)
    return poly

# ================= GROUP C (FIXED + EXCEL OUTPUT) =================
def group_c(df, crs, out, mapping):

    x, y, order = resolve_xyz(df, mapping)

    df = df.sort_values(order)

    coords = list(zip(df[x], df[y]))

    if len(coords) < 3:
        raise ValueError("Need at least 3 points")

    if coords[0] != coords[-1]:
        coords.append(coords[0])

    poly = safe_polygon(coords)

    poly_gdf = gpd.GeoDataFrame([{
        "Area_ha": poly.area / 10000,
        "Perimeter": poly.length,
        "geometry": poly
    }], crs=crs)

    line_gdf = gpd.GeoDataFrame([{
        "geometry": LineString(coords)
    }], crs=crs)

    pts_gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df[x], df[y]),
        crs=crs
    )

    # SHP EXPORT SAFE
    poly_gdf.to_file(os.path.join(out, "polygon.shp"))
    line_gdf.to_file(os.path.join(out, "line.shp"))
    pts_gdf.to_file(os.path.join(out, "points.shp"))

    # ================= EXCEL FIX =================
    excel_path = os.path.join(out, "output.xlsx")

    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="input", index=False)
        poly_gdf.drop(columns="geometry").to_excel(writer, sheet_name="polygon", index=False)
        line_gdf.drop(columns="geometry").to_excel(writer, sheet_name="line", index=False)
        pts_gdf.drop(columns="geometry").to_excel(writer, sheet_name="points", index=False)

    return poly_gdf, line_gdf, pts_gdf

# ================= ROUTE =================
@app.route("/upload", methods=["POST"])
def upload():
    try:
        file = request.files["file"]
        mode = request.form.get("mode", "A")
        zone = request.form.get("zone", "44")

        # 🔥 FIX: mapping must come from frontend
        mapping_raw = request.form.get("mapping", "{}")
        mapping = json.loads(mapping_raw) if mapping_raw else {}

        run_id = str(uuid.uuid4())
        out = os.path.join(OUTPUT, run_id)
        os.makedirs(out, exist_ok=True)

        crs = f"EPSG:326{zone}"
        df = read_input(file)

        if mode == "C":
            group_c(df, crs, out, mapping)
        else:
            return jsonify({"error": "Only Group C enabled in fixed version"}), 400

        return jsonify({
            "run_id": run_id,
            "download": f"/download/{run_id}"
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ================= DOWNLOAD =================
@app.route("/download/<run_id>")
def download(run_id):
    folder = os.path.join(OUTPUT, run_id)

    zip_path = shutil.make_archive(
        os.path.join(OUTPUT, f"export_{run_id}"),
        "zip",
        root_dir=folder
    )

    return send_file(zip_path, as_attachment=True)

if __name__ == "__main__":
    app.run(debug=True)
