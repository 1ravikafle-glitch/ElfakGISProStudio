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


# ================= CRS =================
def get_crs(zone):
    return "EPSG:32644" if zone == "44" else "EPSG:32645"


# ================= NORMALIZE ORDER =================
def normalize_order(df):
    for c in df.columns:
        if str(c).lower() in ["sn", "s.n", "order"]:
            df = df.rename(columns={c: "Order"})
    return df


# ================= COLUMN FALLBACK =================
def col(mapping, key, fallback):
    return mapping.get(key, fallback) if mapping else fallback


# ================= SERVE OUTPUTS =================
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


# ================= GROUP A =================
def group_a(df, forest, crs, out, mapping):
    df = normalize_order(df).sort_values("Order")

    x = col(mapping, "X", "X")
    y = col(mapping, "Y", "Y")

    coords = list(zip(df[x], df[y]))
    if coords and coords[0] != coords[-1]:
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


# ================= GROUP B =================
def group_b(df, crs, out, mapping):
    df = normalize_order(df)

    x = col(mapping, "X", "X")
    y = col(mapping, "Y", "Y")
    order = col(mapping, "Order", "Order")
    forest_col = col(mapping, "Forest", "Forest")
    comp_col = col(mapping, "Compartment", "Compartment")

    if forest_col not in df.columns or comp_col not in df.columns:
        raise ValueError("Missing Forest or Compartment columns")

    polys, lines, pts = [], [], []

    for f, g in df.groupby(forest_col):
        for c, cg in g.groupby(comp_col):

            cg = cg.sort_values(order)
            coords = list(zip(cg[x], cg[y]))
            if coords and coords[0] != coords[-1]:
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

    # ✅ FIX: EXPORT SHAPEFILES (WAS MISSING BEFORE)
    poly_gdf = gpd.GeoDataFrame(polys, crs=crs)
    line_gdf = gpd.GeoDataFrame(lines, crs=crs)
    pts_gdf = gpd.GeoDataFrame(pts, crs=crs)

    poly_gdf.to_file(os.path.join(out, "polygon.shp"))
    line_gdf.to_file(os.path.join(out, "line.shp"))
    pts_gdf.to_file(os.path.join(out, "points.shp"))

    return poly_gdf, line_gdf, pts_gdf


# ================= GROUP C =================
def group_c(file, crs, w, h, rows, cols, out):
    polygons = []

    if file.filename.lower().endswith(".zip"):
        temp = os.path.join(UPLOAD, f"tmp_{uuid.uuid4()}")
        os.makedirs(temp, exist_ok=True)

        try:
            zip_path = os.path.join(temp, "in.zip")
            file.save(zip_path)

            with zipfile.ZipFile(zip_path) as z:
                z.extractall(temp)

            gdf = None
            for root, _, files in os.walk(temp):
                for f in files:
                    if f.lower().endswith(".shp"):
                        gdf = gpd.read_file(os.path.join(root, f))
                        break

            if gdf is None:
                raise ValueError("No shapefile found in zip")

            geom = gdf.unary_union
            polygons = list(geom.geoms) if geom.geom_type != "Polygon" else [geom]

        finally:
            shutil.rmtree(temp, ignore_errors=True)

    else:
        df = normalize_order(read_input(file)).sort_values("Order")
        coords = list(zip(df["X"], df["Y"]))
        if coords and coords[0] != coords[-1]:
            coords.append(coords[0])
        polygons = [Polygon(coords)]

    poly_gdf = gpd.GeoDataFrame([{"geometry": p} for p in polygons], crs=crs)
    union = poly_gdf.unary_union
    minx, miny, _, _ = union.bounds

    pts = []
    sn = 1

    for r in range(rows):
        for c in range(cols):
            x = minx + c * w
            y = miny + r * h
            center = Point(x + w / 2, y + h / 2)

            if union.contains(center):
                pts.append({"SN": sn, "X": center.x, "Y": center.y, "geometry": center})
                sn += 1

    # ❗ FIX: EXPORT MISSING SHAPEFILES
    poly_gdf.to_file(os.path.join(out, "poly.shp"))
    gpd.GeoDataFrame([{"geometry": LineString(p.exterior.coords)} for p in polygons], crs=crs)\
        .to_file(os.path.join(out, "line.shp"))
    gpd.GeoDataFrame(pts, crs=crs).to_file(os.path.join(out, "sample.shp"))

    return poly_gdf, \
        gpd.GeoDataFrame([{"geometry": LineString(p.exterior.coords)} for p in polygons], crs=crs), \
        gpd.GeoDataFrame(pts, crs=crs)


# ================= GROUP D =================
def group_d(df, crs, out):
    df = normalize_order(df).sort_values("Order")

    if "Forest" not in df.columns:
        raise ValueError("Forest column missing")

    polys, lines, pts = [], [], []

    for f, g in df.groupby("Forest"):
        coords = list(zip(g["X"], g["Y"]))
        if coords and coords[0] != coords[-1]:
            coords.append(coords[0])

        poly = Polygon(coords)
        line = LineString(coords)

        polys.append({
            "Forest": f,
            "Area": poly.area / 10000,
            "Perim": poly.length,
            "geometry": poly
        })

        lines.append({"Forest": f, "geometry": line})

        for _, r in g.iterrows():
            pts.append({
                "Forest": f,
                "Order": r["Order"],
                "geometry": Point(r["X"], r["Y"])
            })

    # ❗ FIX: EXPORT SHAPEFILES (WAS MISSING BEFORE)
    poly_gdf = gpd.GeoDataFrame(polys, crs=crs)
    line_gdf = gpd.GeoDataFrame(lines, crs=crs)
    pts_gdf = gpd.GeoDataFrame(pts, crs=crs)

    poly_gdf.to_file(os.path.join(out, "poly.shp"))
    line_gdf.to_file(os.path.join(out, "line.shp"))
    pts_gdf.to_file(os.path.join(out, "points.shp"))

    return poly_gdf, line_gdf, pts_gdf


# ================= PREVIEW =================
def preview(poly, line, pts, path, pc, lc, ptc):
    fig, ax = plt.subplots(figsize=(6, 6))
    try:
        if not poly.empty:
            poly.plot(ax=ax, facecolor="none", edgecolor=pc)
        if not line.empty:
            line.plot(ax=ax, color=lc)
        if not pts.empty:
            pts.plot(ax=ax, color=ptc, markersize=10)

        ax.set_axis_off()
        fig.savefig(path, dpi=150, bbox_inches="tight", transparent=True)
    finally:
        plt.close(fig)


# ================= UPLOAD =================
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
            poly, line, pts = group_c(file, crs, w, h, rows, cols, out)
        else:
            df = read_input(file)
            poly, line, pts = group_d(df, crs, out)

        preview_path = os.path.join(out, "output.png")
        preview(poly, line, pts, preview_path, "#34d399", "#6b7280", "#f59e0b")

        return jsonify({"run_id": run_id, "download": f"/download/{run_id}"})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ================= DOWNLOAD (SAFE ZIP FIX) =================
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
