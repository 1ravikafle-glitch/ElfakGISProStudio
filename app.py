import os
import uuid
import zipfile
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt

from flask import Flask, render_template, request, send_file, send_from_directory
from shapely.geometry import Polygon, Point, LineString
from shapely.ops import unary_union
from io import BytesIO

app = Flask(__name__)

UPLOAD = "uploads"
OUTPUT = "outputs"

os.makedirs(UPLOAD, exist_ok=True)
os.makedirs(OUTPUT, exist_ok=True)


# ================= CRS =================
def get_crs(zone):
    return "EPSG:32644" if zone == "44" else "EPSG:32645"


# ================= ORDER =================
def normalize_order(df):
    for c in df.columns:
        if c.lower() in ["sn", "s.n", "order"]:
            df = df.rename(columns={c: "Order"})
    return df


# ================= SERVE PREVIEW =================
@app.route("/outputs/<run_id>/<filename>")
def serve_output(run_id, filename):
    return send_from_directory(os.path.join(OUTPUT, run_id), filename)


# ================= ZIP =================
def zip_folder(folder):
    buf = BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for f in os.listdir(folder):
            fp = os.path.join(folder, f)
            if os.path.isfile(fp):
                z.write(fp, f)
    buf.seek(0)
    return buf


# ================= SAFE POLYGON FIX =================
def safe_polygon(gdf):
    geom = unary_union(gdf.geometry)

    if geom.geom_type == "MultiPolygon":
        geom = max(geom.geoms, key=lambda g: g.area)

    if geom.geom_type == "GeometryCollection":
        geom = [g for g in geom.geoms if g.geom_type in ["Polygon", "MultiPolygon"]]
        geom = unary_union(geom)

    return geom


# ================= GROUP C (ZIP FIXED) =================
def group_c(file, crs, w, h, rows, cols, out):

    folder = os.path.join(UPLOAD, str(uuid.uuid4()))
    os.makedirs(folder, exist_ok=True)

    zip_path = os.path.join(folder, "input.zip")
    file.save(zip_path)

    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(folder)

    shp = None
    for root, _, files in os.walk(folder):
        for f in files:
            if f.endswith(".shp"):
                shp = gpd.read_file(os.path.join(root, f))
                break

    if shp is None:
        raise Exception("No shapefile found in ZIP")

    poly = safe_polygon(shp)

    line = LineString(poly.exterior.coords)

    minx, miny, _, _ = poly.bounds

    pts = []
    for i in range(rows):
        for j in range(cols):
            x = minx + j * w
            y = miny + i * h

            cell = Polygon([
                (x, y),
                (x + w, y),
                (x + w, y + h),
                (x, y + h)
            ])

            pts.append(cell.centroid)

    inside = [p for p in pts if poly.contains(p)]

    gdf = gpd.GeoDataFrame(
        {"SN": range(1, len(inside) + 1)},
        geometry=inside,
        crs=crs
    )

    gdf["X"] = gdf.geometry.x
    gdf["Y"] = gdf.geometry.y

    poly_gdf = gpd.GeoDataFrame([{"geometry": poly}], crs=crs)
    line_gdf = gpd.GeoDataFrame([{"geometry": line}], crs=crs)

    run_id = str(uuid.uuid4())
    out = os.path.join(OUTPUT, run_id)
    os.makedirs(out, exist_ok=True)

    poly_gdf.to_file(os.path.join(out, "boundary.shp"))
    line_gdf.to_file(os.path.join(out, "line.shp"))
    gdf.to_file(os.path.join(out, "sampleplot.shp"))

    preview_name = f"{run_id}_preview.png"
    preview_path = os.path.join(out, preview_name)

    fig, ax = plt.subplots()

    poly_gdf.plot(ax=ax, facecolor="none", edgecolor="red")
    line_gdf.plot(ax=ax, color="yellow")
    gdf.plot(ax=ax, color="cyan", markersize=8)

    plt.axis("off")
    plt.savefig(preview_path, dpi=220)
    plt.close()

    return run_id, preview_name


# ================= MAIN =================
@app.route("/upload", methods=["POST"])
def upload():

    file = request.files["file"]
    mode = request.form["mode"]
    zone = request.form["zone"]

    w = float(request.form.get("w", 50))
    h = float(request.form.get("h", 50))
    rows = int(request.form.get("rows", 10))
    cols = int(request.form.get("cols", 10))

    crs = get_crs(zone)

    run_id = str(uuid.uuid4())
    out = os.path.join(OUTPUT, run_id)
    os.makedirs(out, exist_ok=True)

    preview_name = f"{run_id}_preview.png"
    preview_path = os.path.join(out, preview_name)

    # ================= GROUP C =================
    if mode == "C":
        run_id, preview_name = group_c(file, crs, w, h, rows, cols, out)

    else:
        df = pd.read_excel(file)

        coords = list(zip(df["X"], df["Y"]))
        coords.append(coords[0])

        poly = Polygon(coords)
        line = LineString(coords)

        poly_gdf = gpd.GeoDataFrame([{"geometry": poly}], crs=crs)
        line_gdf = gpd.GeoDataFrame([{"geometry": line}], crs=crs)

        poly_gdf.to_file(os.path.join(out, "poly.shp"))
        line_gdf.to_file(os.path.join(out, "line.shp"))

        fig, ax = plt.subplots()
        poly_gdf.plot(ax=ax, facecolor="none", edgecolor="red")
        line_gdf.plot(ax=ax, color="yellow")

        plt.axis("off")
        plt.savefig(preview_path, dpi=220)
        plt.close()

    zip_buffer = zip_folder(out)

    return send_file(
        zip_buffer,
        mimetype="application/zip",
        as_attachment=True,
        download_name="output.zip"
    )


@app.route("/")
def home():
    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)
