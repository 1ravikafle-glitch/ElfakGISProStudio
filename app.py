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

from flask import (
    Flask, render_template, request,
    send_file, jsonify, send_from_directory
)

from shapely.geometry import Polygon, Point, LineString

app = Flask(__name__)

UPLOAD = "uploads"
OUTPUT = "outputs"

os.makedirs(UPLOAD, exist_ok=True)
os.makedirs(OUTPUT, exist_ok=True)


# ================= SAFE FILE READER =================
def read_input(file):
    name = file.filename.lower()

    if name.endswith(".csv"):
        try:
            return pd.read_csv(file, encoding="utf-8-sig")
        except Exception:
            file.seek(0)
            return pd.read_csv(file, encoding="latin1")

    elif name.endswith((".xlsx", ".xls")):
        return pd.read_excel(file)

    raise ValueError("Only CSV/Excel supported")


# ================= CRS FIX =================
def get_crs(zone):
    zone = str(zone).replace("UTM Zone", "").replace("N", "").strip()
    return f"EPSG:326{zone}"


# ================= NORMALIZE =================
def normalize_order(df):
    df.columns = [str(c).strip() for c in df.columns]

    for c in df.columns:
        if str(c).lower() in ["sn", "s.n", "order"]:
            df = df.rename(columns={c: "Order"})
    return df


# ================= SAFE COLUMN RESOLVER =================
def resolve_col(df, mapping, key, fallback_list):
    if mapping and key in mapping and mapping[key]:
        return mapping[key]

    cols_lower = {c.lower().strip(): c for c in df.columns}

    for f in fallback_list:
        if f.lower() in cols_lower:
            return cols_lower[f.lower()]

    # last fallback
    if fallback_list:
        return fallback_list[0]

    raise ValueError(f"Missing column for {key}")


# ================= OUTPUT SERVE =================
@app.route("/outputs/<run_id>/<filename>")
def outputs(run_id, filename):
    return send_from_directory(os.path.join(OUTPUT, run_id), filename)


# ================= GET COLUMNS =================
@app.route("/get-columns", methods=["POST"])
def get_columns():
    try:
        df = read_input(request.files["file"])
        return jsonify(list(df.columns))
    except Exception as e:
        return jsonify({"error": str(e)}), 400


# ================= GROUP A (FIXED) =================
def group_a(df, forest, crs, out, mapping):
    df = normalize_order(df).sort_values("Order")

    x = resolve_col(df, mapping, "X", ["X", "x", "Longitude"])
    y = resolve_col(df, mapping, "Y", ["Y", "y", "Latitude"])

    coords = list(zip(df[x], df[y]))
    if len(coords) < 3:
        raise ValueError("Not enough points for polygon")

    if coords[0] != coords[-1]:
        coords.append(coords[0])

    poly = Polygon(coords)
    line = LineString(coords)

    poly_gdf = gpd.GeoDataFrame([{
        "Forest": forest,
        "Area": poly.area / 10000,
        "Perim": poly.length,
        "geometry": poly
    }], crs=crs)

    line_gdf = gpd.GeoDataFrame([{"Forest": forest, "geometry": line}], crs=crs)
    pts_gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df[x], df[y]), crs=crs)

    poly_gdf.to_file(os.path.join(out, "polygon.shp"))
    line_gdf.to_file(os.path.join(out, "line.shp"))
    pts_gdf.to_file(os.path.join(out, "points.shp"))

    return poly_gdf, line_gdf, pts_gdf


# ================= GROUP B (FIXED) =================
def group_b(df, crs, out, mapping):
    df = normalize_order(df)

    x = resolve_col(df, mapping, "X", ["X", "x"])
    y = resolve_col(df, mapping, "Y", ["Y", "y"])
    order = resolve_col(df, mapping, "Order", ["Order"])
    forest_col = resolve_col(df, mapping, "Forest", ["Forest"])
    comp_col = resolve_col(df, mapping, "Compartment", ["Compartment"])

    polys, lines, pts = [], [], []

    for f, g in df.groupby(forest_col):
        for c, cg in g.groupby(comp_col):

            cg = cg.sort_values(order)
            coords = list(zip(cg[x], cg[y]))

            if len(coords) < 3:
                continue

            if coords[0] != coords[-1]:
                coords.append(coords[0])

            poly = Polygon(coords)
            line = LineString(coords)

            polys.append({
                "Forest": f,
                "Compartment": c,
                "Area": poly.area / 10000,
                "Perim": poly.length,
                "geometry": poly
            })

            lines.append({
                "Forest": f,
                "Compartment": c,
                "geometry": line
            })

            for _, r in cg.iterrows():
                pts.append({
                    "Forest": f,
                    "Compartment": c,
                    "Order": r[order],
                    "geometry": Point(r[x], r[y])
                })

    poly_gdf = gpd.GeoDataFrame(polys, crs=crs)
    line_gdf = gpd.GeoDataFrame(lines, crs=crs)
    pts_gdf = gpd.GeoDataFrame(pts, crs=crs)

    poly_gdf.to_file(os.path.join(out, "polygon.shp"))
    line_gdf.to_file(os.path.join(out, "line.shp"))
    pts_gdf.to_file(os.path.join(out, "points.shp"))

    return poly_gdf, line_gdf, pts_gdf


# ================= GROUP C (UNCHANGED, SAFE) =================
def group_c(file, crs, w, h, rows, cols, out, mapping=None, selected_shp=None):
    import tempfile

    polygons = []

    # ===================== CASE 1: ZIP INPUT =====================
    if file.filename.lower().endswith(".zip"):
        temp_dir = os.path.join(UPLOAD, f"tmp_{uuid.uuid4()}")
        os.makedirs(temp_dir, exist_ok=True)

        try:
            zip_path = os.path.join(temp_dir, "input.zip")
            file.save(zip_path)

            with zipfile.ZipFile(zip_path) as z:
                z.extractall(temp_dir)

            if not selected_shp:
                raise ValueError("Please select a shapefile from ZIP")

            shp_path = None
            for root, _, files in os.walk(temp_dir):
            for f in files:
            if f.lower().endswith(".shp") and f == os.path.basename(selected_shp):
                shp_path = os.path.join(root, f)
            break

            if not shp_path:
                raise ValueError("Shapefile not found in ZIP")

            gdf = gpd.read_file(shp_path)
        if gdf.crs is None:
                gdf.set_crs(crs, inplace=True)
            geom = gdf.geometry.unary_union
        if geom.geom_type == "Polygon":
            polygons = [geom]
        elif geom.geom_type == "MultiPolygon":
            polygons = list(geom.geoms)
        else:
            raise ValueError(f"Unsupported geometry: {geom.geom_type}")

        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    # ===================== CASE 2: CSV/EXCEL INPUT =====================
    else:
        df = normalize_order(read_input(file))

        x_col = resolve_col(df, mapping, "X", ["X", "x", "Longitude", "Lon"])
        y_col = resolve_col(df, mapping, "Y", ["Y", "y", "Latitude", "Lat"])
        order_col = resolve_col(df, mapping, "Order", ["Order", "S.N", "SN", "s.n"])

        df = df.sort_values(order_col)

        coords = list(zip(df[x_col], df[y_col]))

        if len(coords) < 3:
            raise ValueError("Not enough points to form polygon")

        if coords[0] != coords[-1]:
            coords.append(coords[0])

        polygons = [Polygon(coords)]

    # ===================== BUILD GEODATAFRAME =====================
    poly_gdf = gpd.GeoDataFrame(
        [{"geometry": p} for p in polygons],
        crs=crs
    )

    union = poly_gdf.unary_union

    # ===================== GRID GENERATION =====================
    minx, miny, maxx, maxy = union.bounds

    pts = []
    sn = 1

    for r in range(rows):
        for c in range(cols):
            x = minx + c * w
            y = miny + r * h

            center = Point(x + w / 2, y + h / 2)

            if union.contains(center):
                pts.append({
                    "SN": sn,
                    "X": center.x,
                    "Y": center.y,
                    "geometry": center
                })
                sn += 1

    pts_gdf = gpd.GeoDataFrame(pts, crs=crs)

    # ===================== OUTPUT FILES =====================
    poly_gdf.to_file(os.path.join(out, "poly.shp"))

    gpd.GeoDataFrame(
        [{"geometry": LineString(p.exterior.coords)} for p in polygons],
        crs=crs
    ).to_file(os.path.join(out, "line.shp"))

    pts_gdf.to_file(os.path.join(out, "sample.shp"))

    return poly_gdf, \
           gpd.GeoDataFrame(
               [{"geometry": LineString(p.exterior.coords)} for p in polygons],
               crs=crs
           ), \
           pts_gdf


# ================= GROUP D (FIXED LIGHTLY) =================
def group_d(df, crs, out):
    df = normalize_order(df).sort_values("Order")

    if "Forest" not in df.columns:
        raise ValueError("Forest column missing")

    polys, lines, pts = [], [], []

    for f, g in df.groupby("Forest"):
        coords = list(zip(g["X"], g["Y"]))

        if len(coords) < 3:
            continue

        if coords[0] != coords[-1]:
            coords.append(coords[0])

        poly = Polygon(coords)
        line = LineString(coords)

        polys.append({"Forest": f, "Area": poly.area / 10000, "Perim": poly.length, "geometry": poly})
        lines.append({"Forest": f, "geometry": line})

        for _, r in g.iterrows():
            pts.append({"Forest": f, "Order": r["Order"], "geometry": Point(r["X"], r["Y"])})

    poly_gdf = gpd.GeoDataFrame(polys, crs=crs)
    line_gdf = gpd.GeoDataFrame(lines, crs=crs)
    pts_gdf = gpd.GeoDataFrame(pts, crs=crs)

    poly_gdf.to_file(os.path.join(out, "poly.shp"))
    line_gdf.to_file(os.path.join(out, "line.shp"))
    pts_gdf.to_file(os.path.join(out, "points.shp"))

    return poly_gdf, line_gdf, pts_gdf


# ================= PREVIEW (UNCHANGED) =================
def preview(poly, line, pts, path, pc, lc, ptc):
    fig, ax = plt.subplots(figsize=(6, 6))

    try:
        # WHITE BACKGROUND
        fig.patch.set_facecolor("white")
        ax.set_facecolor("white")

        # POLYGON (optional outline, keep subtle or same as line)
        if not poly.empty:
            poly.plot(ax=ax, facecolor="none", edgecolor="black", linewidth=1)

        # LINE → BLACK
        if not line.empty:
            line.plot(ax=ax, color="black", linewidth=1.5)

        # POINTS → RED
        if not pts.empty:
            pts.plot(ax=ax, color="red", markersize=20)

        ax.set_axis_off()

        fig.savefig(
            path,
            dpi=150,
            bbox_inches="tight",
            facecolor="white"
        )

    finally:
        plt.close(fig)


# ================= UPLOAD (UNCHANGED) =================
@app.route("/upload", methods=["POST"])
def upload():
    try:
        file = request.files["file"]
        mode = request.form.get("mode", "A")
        zone = request.form.get("zone", "44")

        mapping = json.loads(request.form.get("mapping", "{}"))

        w = float(request.form.get("w", 50))
        h = float(request.form.get("h", 50))
        rows = int(request.form.get("rows", 10))
        cols = int(request.form.get("cols", 10))
        forest = request.form.get("forest", "FOREST")

        run_id = str(uuid.uuid4())
        out = os.path.join(OUTPUT, run_id)
        os.makedirs(out, exist_ok=True)

        crs = get_crs(zone)

        if mode == "A":
            df = read_input(file)
            poly, line, pts = group_a(df, forest, crs, out, mapping)
        elif mode == "B":
            df = read_input(file)
            poly, line, pts = group_b(df, crs, out, mapping)
        elif mode == "C":
            selected_shp = request.form.get("selected_shp")
            poly, line, pts = group_c(file, crs, w, h, rows, cols, out, selected_shp)
        else:
            df = read_input(file)
            poly, line, pts = group_d(df, crs, out)

        preview_path = os.path.join(out, "output.png")
        preview(poly, line, pts, preview_path, "#34d399", "#6b7280", "#f59e0b")

        return jsonify({"run_id": run_id, "download": f"/download/{run_id}"})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ================= DOWNLOAD =================
@app.route("/download/<run_id>")
def download(run_id):
    folder = os.path.join(OUTPUT, run_id)

    if not os.path.exists(folder):
        return "Output not found", 404

    zip_path = shutil.make_archive(
        os.path.join(OUTPUT, f"export_{run_id}"),
        "zip",
        root_dir=folder
    )

    return send_file(zip_path, as_attachment=True, download_name="gis_export.zip")


# ================= HOME =================
@app.route("/")
def home():
    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)
