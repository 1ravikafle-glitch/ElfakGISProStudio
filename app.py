import os
import uuid
import zipfile
import shutil
import json
import re
import pandas as pd
import geopandas as gpd
import matplotlib
import io
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from flask import Flask, request, jsonify, send_file, send_from_directory, render_template
from shapely.geometry import Polygon, Point, LineString
from io import BytesIO

app = Flask(_name_)
UPLOAD = "uploads"
OUTPUT = "outputs"

os.makedirs(UPLOAD, exist_ok=True)
os.makedirs(OUTPUT, exist_ok=True)

# ================= CORS CONTROLLER =================
@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type,Authorization"
    response.headers["Access-Control-Allow-Methods"] = "GET,POST,OPTIONS"
    return response

# ================= NORMALIZER =================
def norm(col):
    return re.sub(r"[^a-z0-9]", "", str(col).lower())

# ================= KEY MAPS =================
X_KEYS = {"x", "xcoord", "xcoordinate", "east", "easting", "longitude", "lon", "lng"}
Y_KEYS = {"y", "ycoord", "ycoordinate", "north", "northing", "latitude", "lat"}
ORDER_KEYS = {"sn", "sno", "s.n", "serial", "id", "index", "order", "seq", "rowid", "fid"}

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
        raise ValueError(f"Missing Coordinate Mapping X/Y. Headers: {df.columns.tolist()}")

    return x, y, order

# ================= SAFE POLYGON BUILDER =================
def safe_polygon(coords):
    poly = Polygon(coords)
    if not poly.is_valid:
        poly = poly.buffer(0)
    return poly

# ================= UNIVERSAL VISUALIZER PREVIEW =================
def generate_preview_plot(poly_gdf, line_gdf, pts_gdf, out_dir):
    try:
        fig, ax = plt.subplots(figsize=(6, 6), dpi=100)
        fig.patch.set_facecolor('#0f0a12')
        ax.set_facecolor('#0f0a12')
        
        # Plot structural elements
        if poly_gdf is not None and not poly_gdf.empty:
            poly_gdf.plot(ax=ax, facecolor='#10b981', alpha=0.25, edgecolor='#10b981', linewidth=1.5)
        if line_gdf is not None and not line_gdf.empty:
            line_gdf.plot(ax=ax, color='#65a30d', linewidth=2, linestyle='--')
        if pts_gdf is not None and not pts_gdf.empty:
            pts_gdf.plot(ax=ax, color='#ffffff', edgecolor='#10b981', markersize=40, zorder=5)
            
        ax.axis('off')
        plt.tight_layout(pad=0)
        
        preview_path = os.path.join(out_dir, "output.png")
        plt.savefig(preview_path, facecolor=fig.get_facecolor(), edgecolor='none', bbox_inches='tight')
        plt.close(fig)
    except Exception as e:
        print(f"Visualization engine fallback error: {str(e)}")

# ================= GROUP C RUNTIME (SUPPORTS TABULAR & ZIP LAYER INTERPOLATION) =================
def process_group_c(file_path, crs, out, mapping):
    
    # Check if a specific target layer within a Shapefile archive was requested
    target_shp = mapping.get("target_shp") if mapping else None

    if file_path.lower().endswith(".zip") and target_shp:
        # 1. DIRECT VECTOR SHAPEFILE PROCESSING ENGINE
        tmp_extract = os.path.join(out, "extracted_shp")
        os.makedirs(tmp_extract, exist_ok=True)
        
        with zipfile.ZipFile(file_path, "r") as z:
            z.extractall(tmp_extract)
        
        shp_absolute_path = os.path.join(tmp_extract, target_shp)
        if not os.path.exists(shp_absolute_path):
            raise FileNotFoundError(f"Requested file layer target not found inside bundle: {target_shp}")
            
        # Parse vector layers directly from spatial files
        src_gdf = gpd.read_file(shp_absolute_path)
        if src_gdf.crs is None:
            src_gdf.set_crs(crs, inplace=True)
        else:
            src_gdf = src_gdf.to_crs(crs)
            
        # Convert all incoming structural elements into geometric segments
        poly_geoms = [geom for geom in src_gdf.geometry if geom.geom_type in ["Polygon", "MultiPolygon"]]
        
        if not poly_geoms:
            # Dropdown selection fallback if points or line objects were provided
            coords = []
            for geom in src_gdf.geometry:
                if geom.geom_type == "Point":
                    coords.append((geom.x, geom.y))
                elif geom.geom_type == "LineString":
                    coords.extend(list(geom.coords))
            if len(coords) >= 3:
                poly_geoms = [safe_polygon(coords)]
            else:
                raise ValueError("The vector file selected does not contain sufficient coordinates to form a closed polygon.")

        # Build clean output datasets
        poly = poly_geoms[0] if poly_geoms else safe_polygon([])
        poly_gdf = gpd.GeoDataFrame([{"Area_ha": poly.area / 10000, "Perimeter": poly.length, "geometry": poly}], crs=crs)
        
        exterior_coords = list(poly.exterior.coords) if hasattr(poly, 'exterior') and poly.exterior else []
        line_gdf = gpd.GeoDataFrame([{"geometry": LineString(exterior_coords) if len(exterior_coords) > 1 else None}], crs=crs)
        
        pts_list = [{"geometry": Point(pt), "Order": i+1} for i, pt in enumerate(exterior_coords[:-1])]
        pts_gdf = gpd.GeoDataFrame(pts_list, crs=crs) if pts_list else gpd.GeoDataFrame(columns=['geometry'], crs=crs)
        
        # Export data frames
        df_excel_input = pd.DataFrame([{"X": pt[0], "Y": pt[1], "Order": i+1} for i, pt in enumerate(exterior_coords[:-1])])

    else:
        # 2. TABULAR INPUT PIPELINE (CSV/EXCEL / ZIP-FLAT)
        if file_path.lower().endswith(".zip"):
            tmp_extract = os.path.join(out, "extracted_flat")
            os.makedirs(tmp_extract, exist_ok=True)
            with zipfile.ZipFile(file_path, "r") as z:
                z.extractall(tmp_extract)
            
            df = None
            for root, _, files in os.walk(tmp_extract):
                for f in files:
                    p = os.path.join(root, f)
                    if f.endswith(".csv"):
                        df = pd.read_csv(p, encoding="utf-8-sig")
                        break
                    if f.endswith((".xlsx", ".xls")):
                        df = pd.read_excel(p)
                        break
                if df is not None: break
            if df is None:
                raise ValueError("ZIP archive folder structure has no recognizable tabular file logs (CSV/XLSX).")
        elif file_path.lower().endswith(".csv"):
            try:
                df = pd.read_csv(file_path, encoding="utf-8-sig")
            except:
                df = pd.read_csv(file_path, encoding="latin1")
        else:
            df = pd.read_excel(file_path)

        x, y, order = resolve_xyz(df, mapping)
        df = df.sort_values(order)
        coords = list(zip(df[x], df[y]))

        if len(coords) < 3:
            raise ValueError("Insufficient coordinates inside matrix logs (Need minimum 3 items).")

        if coords[0] != coords[-1]:
            coords.append(coords[0])

        poly = safe_polygon(coords)
        poly_gdf = gpd.GeoDataFrame([{"Area_ha": poly.area / 10000, "Perimeter": poly.length, "geometry": poly}], crs=crs)
        line_gdf = gpd.GeoDataFrame([{"geometry": LineString(coords)}], crs=crs)
        pts_gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df[x], df[y]), crs=crs)
        df_excel_input = df

    # Export Shapefiles safely
    poly_gdf.to_file(os.path.join(out, "polygon.shp"))
    if line_gdf is not None and not line_gdf.empty and line_gdf.geometry.iloc[0] is not None:
        line_gdf.to_file(os.path.join(out, "line.shp"))
    if not pts_gdf.empty:
        pts_gdf.to_file(os.path.join(out, "points.shp"))

    # Export cross-platform Excel summaries
    excel_path = os.path.join(out, "output.xlsx")
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        df_excel_input.to_excel(writer, sheet_name="input", index=False)
        poly_gdf.drop(columns="geometry", errors="ignore").to_excel(writer, sheet_name="polygon", index=False)
        if line_gdf is not None and not line_gdf.empty:
            line_gdf.drop(columns="geometry", errors="ignore").to_excel(writer, sheet_name="line", index=False)
        if not pts_gdf.empty:
            pts_gdf.drop(columns="geometry", errors="ignore").to_excel(writer, sheet_name="points", index=False)

    # Plot preview image
    generate_preview_plot(poly_gdf, line_gdf, pts_gdf, out)

# ================= CORE GATEWAY CONTROLLER =================
@app.route("/upload", methods=["POST"])
def upload():
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded in form boundary context"}), 400
            
        file = request.files["file"]
        mode = request.form.get("mode", "A")
        zone = request.form.get("zone", "44")

        mapping_raw = request.form.get("mapping", "{}")
        mapping = json.loads(mapping_raw) if mapping_raw else {}

        run_id = str(uuid.uuid4())
        run_output_dir = os.path.join(OUTPUT, run_id)
        os.makedirs(run_output_dir, exist_ok=True)

        # Stash original payload source
        saved_input_name = f"input_source_{run_id}_{file.filename}"
        saved_input_path = os.path.join(UPLOAD, saved_input_name)
        file.save(saved_input_path)

        crs = f"EPSG:326{zone}"

        # Route processing to Group C
        if mode in ["A", "B", "C", "D"]:
            process_group_c(saved_input_path, crs, run_output_dir, mapping)
        else:
            return jsonify({"error": f"Mode configuration signature context invalid: {mode}"}), 400

        return jsonify({
            "run_id": run_id,
            "download": f"/download/{run_id}"
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

# ================= STATIC RESOURCE ROUTING =================
@app.route("/outputs/<run_id>/<filename>")
def serve_output_assets(run_id, filename):
    return send_from_directory(os.path.join(OUTPUT, run_id), filename)

# ================= DOWNLOAD CONTROLLER =================
@app.route("/download/<run_id>")
def download(run_id):
    folder = os.path.join(OUTPUT, run_id)
    archive_store_path = os.path.join(OUTPUT, f"export_{run_id}")
    
    zip_path = shutil.make_archive(
        archive_store_path,
        "zip",
        root_dir=folder
    )
    return send_file(zip_path, as_attachment=True)
    # ================= HOME =================
@app.route("/")
def home():
    return render_template("index.html")


if name == "main":
    app.run(debug=True)
