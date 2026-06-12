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

from flask import Flask, request, jsonify, send_file
from shapely.geometry import Polygon, Point, LineString, MultiPolygon

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
ORDER_KEYS = {"sn", "sno", "s.n", "serial", "id", "index", "order", "seq", "rowid", "fid"}

# ================= SAFE GEOMETRY =================
def safe_geom(g):
    if g is None:
        return None
    try:
        if not g.is_valid:
            g = g.buffer(0)
        if g.geom_type == "GeometryCollection":
            g = g.buffer(0)
        return g
    except:
        return None

# ================= FILE READER =================
def read_input(file):
    name = file.filename.lower()

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

    if name.endswith(".csv"):
        try:
            return pd.read_csv(file, encoding="utf-8-sig")
        except:
            file.seek(0)
            return pd.read_csv(file, encoding="latin1")

    if name.endswith((".xlsx", ".xls")):
        return pd.read_excel(file)

    raise ValueError("Only CSV / Excel / ZIP supported")

# ================= COLUMN RESOLVER =================
def resolve_xyz(df, mapping=None):
    df.columns = [str(c).strip() for c in df.columns]
    x = y = order = None

    if mapping:
        x = mapping.get("X")
        y = mapping.get("Y")
        order = mapping.get("Order")

    for col in df.columns:
        c = norm(col)
        if x is None and c in X_KEYS: x = col
        if y is None and c in Y_KEYS: y = col
        if order is None and c in ORDER_KEYS: order = col

    if x is None:
        for c in df.columns:
            if "x" in c.lower(): x = c; break
    if y is None:
        for c in df.columns:
            if "y" in c.lower(): y = c; break

    if order is None:
        df["Order"] = range(1, len(df) + 1)
        order = "Order"

    if x is None or y is None:
        raise ValueError(f"Missing UI mapping X/Y. Columns: {df.columns.tolist()}")

    return x, y, order

# ================= RENDERING PIPELINE (FIXES DYNAMIC PREVIEW) =================
def render_preview(poly_gdf, line_gdf, pts_gdf, output_dir):
    try:
        fig, ax = plt.subplots(figsize=(8, 6), facecolor="#0f0a12")
        ax.set_facecolor("#0f0a12")
        
        # Plot structural GIS data with modern theme aesthetics
        if not poly_gdf.empty:
            poly_gdf.plot(ax=ax, color="#10b981", alpha=0.25, edgecolor="#10b981", linewidth=1.5)
        if not line_gdf.empty:
            line_gdf.plot(ax=ax, color="#65a30d", linewidth=1, linestyle="--", alpha=0.7)
        if not pts_gdf.empty:
            pts_gdf.plot(ax=ax, color="#ffffff", markersize=12, edgecolor="#10b981", alpha=0.9, zorder=5)
            
        ax.axis("off")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "output.png"), dpi=150, facecolor=fig.get_facecolor(), edgecolor='none')
        plt.close()
    except Exception as e:
        print(f"Canvas render skipped or warning generated: {str(e)}")

# ================= MASTER PIPELINE ENGINE =================
def process_gis_pipeline(df, crs, out, mapping, mode):
    x, y, order = resolve_xyz(df, mapping)
    df = df.sort_values(by=order)

    # Dynamic segmentation detection keys based on matching UI requirements
    group_col = None
    if mode in ["B", "D"]:
        for col in df.columns:
            n = norm(col)
            if "compartment" in n or "comp" in n:
                group_col = col
                break
        if not group_col:
            for col in df.columns:
                if "forest" in norm(col) or "id" in norm(col):
                    group_col = col; break

    poly_records, line_records = [], []

    if group_col:
        # Segmented handling for Mode B, Mode C (Sub-segmented options), and Complex Engine D
        for group_id, group_df in df.groupby(group_col):
            group_df = group_df.sort_values(by=order)
            coords = list(zip(group_df[x], group_df[y]))
            if len(coords) < 3: continue
            if coords[0] != coords[-1]: coords.append(coords[0])
            
            p = Polygon(coords)
            if not p.is_valid: p = p.buffer(0)
            
            poly_records.append({"ID": str(group_id), "Area_ha": p.area / 10000, "Perimeter": p.length, "geometry": p})
            line_records.append({"ID": str(group_id), "geometry": LineString(coords)})
    else:
        # Standard contiguous whole boundary compilation (Mode A & Basic Core loops)
        coords = list(zip(df[x], df[y]))
        if len(coords) < 3: raise ValueError("Need at least 3 spatial points to form vector boundaries.")
        if coords[0] != coords[-1]: coords.append(coords[0])
        
        p = Polygon(coords)
        if not p.is_valid: p = p.buffer(0)
        
        poly_records.append({"ID": "Whole_Canopy", "Area_ha": p.area / 10000, "Perimeter": p.length, "geometry": p})
        line_records.append({"ID": "Whole_Canopy", "geometry": LineString(coords)})

    if not poly_records:
        raise ValueError("Spatial mapping failed to resolve topology configurations.")

    poly_gdf = gpd.GeoDataFrame(poly_records, crs=crs)
    line_gdf = gpd.GeoDataFrame(line_records, crs=crs)
    pts_gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df[x], df[y]), crs=crs)

    # Export structural GIS files safely
    poly_gdf.to_file(os.path.join(out, "polygon.shp"))
    line_gdf.to_file(os.path.join(out, "line.shp"))
    pts_gdf.to_file(os.path.join(out, "points.shp"))

    # Generate Multi-sheet Spreadsheet matrix summaries
    excel_path = os.path.join(out, "output.xlsx")
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="input_raw", index=False)
        poly_gdf.drop(columns="geometry").to_excel(writer, sheet_name="polygon_metrics", index=False)
        line_gdf.drop(columns="geometry").to_excel(writer, sheet_name="line_vectors", index=False)

    # Render missing preview workspace visualization layer
    render_preview(poly_gdf, line_gdf, pts_gdf, out)
    return poly_gdf

# ================= ROUTE CONTROL MODULES =================
@app.route("/upload", methods=["POST"])
def upload():
    try:
        file = request.files["file"]
        mode = request.form.get("mode", "A")
        zone = request.form.get("zone", "44")

        mapping_raw = request.form.get("mapping", "{}")
        mapping = json.loads(mapping_raw) if mapping_raw else {}

        run_id = str(uuid.uuid4())
        out = os.path.join(OUTPUT, run_id)
        os.makedirs(out, exist_ok=True)

        crs = f"EPSG:326{zone}"
        df = read_input(file)

        # Process all system modules safely
        process_gis_pipeline(df, crs, out, mapping, mode)

        return jsonify({
            "run_id": run_id,
            "download": f"/download/{run_id}"
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/download/<run_id>")
def download(run_id):
    folder = os.path.join(OUTPUT, run_id)
    zip_path = shutil.make_archive(
        os.path.join(OUTPUT, f"export_{run_id}"),
        "zip",
        root_dir=folder
    )
    return send_file(zip_path, as_attachment=True)

# Static assets exposure rule to let frontend serve output.png natively
@app.route("/outputs/<run_id>/output.png")
def serve_preview_image(run_id):
    img_file = os.path.join(OUTPUT, run_id, "output.png")
    if os.path.exists(img_file):
        return send_file(img_file, mimetype="image/png")
    return jsonify({"error": "Preview rendering asset not found"}), 404

if __name__ == "__main__":
    app.run(debug=True, port=5000)
