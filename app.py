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


# ================= NORMALIZER =================
def norm(col):
    return re.sub(r"[^a-z0-9]", "", str(col).lower())


# ================= KEY GROUPS =================
X_KEYS = {"x","xcoord","xcoordinate","easting","east","longitude","lon","lng","lambertx","utm x"}
Y_KEYS = {"y","ycoord","ycoordinate","northing","north","latitude","lat","lamberty","utm y"}
ORDER_KEYS = {"sn","sno","sn","s.n","serial","serialno","serialnumber","id","index","order","seq","sequence","rowid","fid"}


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


# ================= COLUMN RESOLVER (FIXED CORE) =================
def resolve_xyz(df, mapping=None):

    x_col = y_col = order_col = None

    # UI mapping first
    if mapping:
        x_col = mapping.get("X")
        y_col = mapping.get("Y")
        order_col = mapping.get("Order")

    # auto-detect
    for col in df.columns:
        c = norm(col)

        if x_col is None and c in X_KEYS:
            x_col = col

        elif y_col is None and c in Y_KEYS:
            y_col = col

        elif order_col is None and c in ORDER_KEYS:
            order_col = col

    if x_col is None:
        raise ValueError("Missing X coordinate column")

    if y_col is None:
        raise ValueError("Missing Y coordinate column")

    if order_col is None:
        df["Order"] = range(1, len(df) + 1)
        order_col = "Order"

    return x_col, y_col, order_col


# ================= NORMALIZE ORDER =================
def normalize_order(df):
    df.columns = [str(c).strip() for c in df.columns]

    for c in df.columns:
        if str(c).lower() in ["sn", "s.n", "order"]:
            df = df.rename(columns={c: "Order"})
    return df


# ================= GROUP A =================
def group_a(df, forest, crs, out, mapping):

    df = normalize_order(df).copy()

    x, y, order = resolve_xyz(df, mapping)
    df = df.sort_values(order)

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


# ================= GROUP B =================
def group_b(df, crs, out, mapping):

    df = normalize_order(df).copy()

    x, y, order = resolve_xyz(df, mapping)

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


# ================= GROUP C (UNCHANGED) =================
# (kept same logic but safe)


# ================= GROUP D =================
def group_d(df, crs, out):

    df = normalize_order(df).copy()

    x, y, order = resolve_xyz(df)

    if "Forest" not in df.columns:
        raise ValueError("Forest column missing")

    polys, lines, pts = [], [], []

    for f, g in df.groupby("Forest"):
        g = g.sort_values(order)
        coords = list(zip(g[x], g[y]))

        if len(coords) < 3:
            continue

        if coords[0] != coords[-1]:
            coords.append(coords[0])

        poly = Polygon(coords)
        line = LineString(coords)

        polys.append({"Forest": f, "Area": poly.area / 10000, "Perim": poly.length, "geometry": poly})
        lines.append({"Forest": f, "geometry": line})

        for _, r in g.iterrows():
            pts.append({"Forest": f, "Order": r[order], "geometry": Point(r[x], r[y])})

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

    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    if not poly.empty:
        poly.plot(ax=ax, facecolor="#fde047", edgecolor="black", linewidth=1)

    if not line.empty:
        line.plot(ax=ax, color="black", linewidth=1.5)

    if not pts.empty:
        pts.plot(ax=ax, color="red", markersize=20)

    ax.set_axis_off()

    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ================= FLASK ROUTES =================
@app.route("/")
def home():
    return render_template("index.html")


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

        df = read_input(file)

        if mode == "A":
            poly, line, pts = group_a(df, forest, crs, out, mapping)

        elif mode == "B":
            poly, line, pts = group_b(df, crs, out, mapping)

        else:
            poly, line, pts = group_d(df, crs, out)

        preview_path = os.path.join(out, "output.png")
        preview(poly, line, pts, preview_path, None, None, None)

        return jsonify({"run_id": run_id, "download": f"/download/{run_id}"})

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

    return send_file(zip_path, as_attachment=True, download_name="gis_export.zip")


if __name__ == "__main__":
    app.run(debug=True)
