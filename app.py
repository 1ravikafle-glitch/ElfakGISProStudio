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
import matplotlib.pyplot as plt

from flask import Flask, request, jsonify, send_file, send_from_directory

from shapely.geometry import Polygon, Point, LineString
from shapely.ops import unary_union

app = Flask(__name__)

UPLOAD = "uploads"
OUTPUT = "outputs"

os.makedirs(UPLOAD, exist_ok=True)
os.makedirs(OUTPUT, exist_ok=True)


# ================= SAFE NORMALIZER =================
def norm(col):
    return re.sub(r"[^a-z0-9]", "", str(col).lower())


# ================= FIXED KEY SETS =================
X_KEYS = {"x","xcoord","xcoordinate","easting","east","longitude","lon","lng"}
Y_KEYS = {"y","ycoord","ycoordinate","northing","north","latitude","lat"}
ORDER_KEYS = {"sn","sno","snno","sno","serial","serialno","id","index","order","seq","sequence","rowid","fid"}


# ================= SAFE MULTIPOLYGON FIX =================
def safe_geometry(poly):
    if poly is None:
        return None

    if not poly.is_valid:
        poly = poly.buffer(0)

    # convert GeometryCollection → Polygon
    if poly.geom_type not in ["Polygon", "MultiPolygon"]:
        poly = poly.buffer(0)

    return poly


# ================= FILE READER (FULL FIX) =================
def read_input(file):
    name = file.filename.lower()

    # 🔥 ZIP FIX
    if name.endswith(".zip"):
        temp_dir = os.path.join(UPLOAD, str(uuid.uuid4()))
        os.makedirs(temp_dir, exist_ok=True)

        zip_path = os.path.join(temp_dir, "data.zip")
        file.save(zip_path)

        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(temp_dir)

        for root, _, files in os.walk(temp_dir):
            for f in files:
                path = os.path.join(root, f)
                try:
                    if f.endswith(".csv"):
                        return pd.read_csv(path, encoding="utf-8-sig")
                    if f.endswith((".xlsx", ".xls")):
                        return pd.read_excel(path)
                except:
                    continue

        raise ValueError("ZIP has no valid CSV/XLSX")

    # CSV FIX
    if name.endswith(".csv"):
        try:
            return pd.read_csv(file, encoding="utf-8-sig")
        except:
            file.seek(0)
            return pd.read_csv(file, encoding="latin1")

    # Excel FIX
    if name.endswith((".xlsx", ".xls")):
        return pd.read_excel(file)

    raise ValueError("Only CSV / Excel / ZIP supported")


# ================= COLUMN RESOLVER (FIXED 'X,Y,S.N' ERROR) =================
def resolve_xyz(df, mapping=None):

    df.columns = [str(c).strip() for c in df.columns]

    x_col = y_col = order_col = None

    # UI mapping first
    if mapping:
        x_col = mapping.get("X")
        y_col = mapping.get("Y")
        order_col = mapping.get("Order")

    # AUTO DETECT SAFE
    for col in df.columns:
        c = norm(col)

        if x_col is None and c in X_KEYS:
            x_col = col
        if y_col is None and c in Y_KEYS:
            y_col = col
        if order_col is None and c in ORDER_KEYS:
            order_col = col

    # 🔥 HARD FIX: common broken headers like "X,Y,S.N"
    if x_col is None:
        for col in df.columns:
            if "x" in col.lower():
                x_col = col

    if y_col is None:
        for col in df.columns:
            if "y" in col.lower():
                y_col = col

    if x_col is None or y_col is None:
        raise ValueError(f"Missing X/Y columns. Found: {df.columns.tolist()}")

    if order_col is None:
        df["Order"] = range(1, len(df) + 1)
        order_col = "Order"

    return x_col, y_col, order_col


# ================= SAFE POLYGON =================
def safe_polygon(coords):
    poly = Polygon(coords)

    if not poly.is_valid:
        poly = poly.buffer(0)

    return poly


# ================= GROUP C + EXCEL OUTPUT FIX =================
def group_c(df, crs, out, mapping):

    x, y, order = resolve_xyz(df, mapping)

    df = df.sort_values(order)

    coords = list(zip(df[x], df[y]))

    if len(coords) < 3:
        raise ValueError("Not enough points")

    if coords[0] != coords[-1]:
        coords.append(coords[0])

    poly = safe_polygon(coords)

    poly_gdf = gpd.GeoDataFrame([{
        "Area": poly.area / 10000,
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

    # SHP EXPORT
    poly_gdf.to_file(os.path.join(out, "polygon.shp"))
    line_gdf.to_file(os.path.join(out, "line.shp"))
    pts_gdf.to_file(os.path.join(out, "points.shp"))

    # 🔥 EXCEL OUTPUT FIX
    excel_path = os.path.join(out, "output.xlsx")

    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="input", index=False)
        poly_gdf.drop(columns="geometry").to_excel(writer, sheet_name="polygon", index=False)
        line_gdf.drop(columns="geometry").to_excel(writer, sheet_name="line", index=False)
        pts_gdf.drop(columns="geometry").to_excel(writer, sheet_name="points", index=False)

    return poly_gdf, line_gdf, pts_gdf


# ================= GROUP B (MULTIPOLYGON SAFE) =================
def group_b(df, crs, out, mapping):

    x, y, order = resolve_xyz(df, mapping)

    forest_col = "Forest" if "Forest" in df.columns else df.columns[0]
    comp_col = "Compartment" if "Compartment" in df.columns else None

    polys, lines, pts = [], [], []

    for f, g in df.groupby(forest_col):

        if comp_col:
            groups = g.groupby(comp_col)
        else:
            groups = [(f, g)]

        for c, cg in groups:

            cg = cg.sort_values(order)
            coords = list(zip(cg[x], cg[y]))

            if len(coords) < 3:
                continue

            if coords[0] != coords[-1]:
                coords.append(coords[0])

            poly = safe_polygon(coords)

            polys.append({
                "Forest": f,
                "Compartment": c,
                "geometry": poly
            })

    return None, None, None


# ================= UPLOAD ROUTE =================
@app.route("/upload", methods=["POST"])
def upload():
    try:
        file = request.files["file"]
        mode = request.form.get("mode", "A")
        zone = request.form.get("zone", "44")
        mapping = json.loads(request.form.get("mapping", "{}"))

        run_id = str(uuid.uuid4())
        out = os.path.join(OUTPUT, run_id)
        os.makedirs(out, exist_ok=True)

        crs = f"EPSG:326{zone}"
        df = read_input(file)

        if mode == "C":
            poly, line, pts = group_c(df, crs, out, mapping)
        else:
            return jsonify({"error": "Only Group C fixed version shown"}), 400

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
