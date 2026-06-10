import os
import zipfile
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt

from flask import Flask, render_template, request, send_file
from shapely.geometry import Polygon
from werkzeug.utils import secure_filename
from io import BytesIO
import math

app = Flask(__name__)

UPLOAD = "uploads"
OUTPUT = "outputs"
STATIC = "static"

os.makedirs(UPLOAD, exist_ok=True)
os.makedirs(OUTPUT, exist_ok=True)
os.makedirs(STATIC, exist_ok=True)


# ================= CRS =================
def get_crs(z):
    return "EPSG:32644" if z == "44" else "EPSG:32645"


# ================= FISHNET =================
def fishnet_clip(polygon, plot_size_m2, crs):
    cell = math.sqrt(plot_size_m2)

    minx, miny, maxx, maxy = polygon.bounds
    cells = []

    x = minx
    while x < maxx:
        y = miny
        while y < maxy:

            poly = Polygon([
                (x, y),
                (x + cell, y),
                (x + cell, y + cell),
                (x, y + cell)
            ])

            clipped = poly.intersection(polygon)

            if not clipped.is_empty:
                cells.append(clipped)

            y += cell
        x += cell

    return gpd.GeoDataFrame(geometry=cells, crs=crs)


# ================= MAP =================
def make_map(gdf, name, out_name):
    fig, ax = plt.subplots(figsize=(10, 6))
    gdf.plot(ax=ax, edgecolor="black", alpha=0.7)
    ax.set_title(name)
    ax.set_axis_off()

    path = os.path.join(STATIC, out_name + ".png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path


# ================= EXPORT EXCEL =================
def export_excel(gdf, path):
    df = gdf.copy()

    if df.geometry.iloc[0].geom_type == "Point":
        df["X"] = df.geometry.x
        df["Y"] = df.geometry.y

    df = df.drop(columns="geometry", errors="ignore")
    df.to_excel(path, index=False)


# ================= PROCESS =================
def process(file_path, mode, zone, order_mode, intensity, plot_size):

    crs = get_crs(zone)
    base = os.path.splitext(os.path.basename(file_path))[0]

    df = pd.read_excel(file_path)

    out_dir = os.path.join(OUTPUT, base)
    os.makedirs(out_dir, exist_ok=True)

    zip_buffer = BytesIO()
    preview_img = None

    # ================= BOUNDARY / COMPARTMENT =================
    if mode in ["boundary", "compartment"]:

        for forest, group in df.groupby("Forest"):

            group["X"] = pd.to_numeric(group["X"], errors="coerce")
            group["Y"] = pd.to_numeric(group["Y"], errors="coerce")
            group = group.dropna()

            if mode == "boundary":

                coords = list(zip(group["X"], group["Y"]))
                coords.append(coords[0])

                poly = Polygon(coords)
                gdf = gpd.GeoDataFrame([{"geometry": poly}], crs=crs)

                shp_path = os.path.join(out_dir, f"{forest}_boundary.shp")
                xlsx_path = os.path.join(out_dir, f"{forest}_boundary.xlsx")

                gdf.to_file(shp_path)
                export_excel(gdf.explode(index_parts=False), xlsx_path)

                preview_img = make_map(gdf, forest, "boundary_preview")

            else:

                polys = []

                for comp, cgroup in group.groupby("Compartment"):

                    if order_mode == "auto":
                        cgroup = cgroup.sort_values("Order").reset_index(drop=True)

                    coords = list(zip(cgroup["X"], cgroup["Y"]))
                    coords.append(coords[0])

                    polys.append({
                        "Forest": forest,
                        "Compartment": comp,
                        "geometry": Polygon(coords)
                    })

                gdf = gpd.GeoDataFrame(polys, crs=crs)

                shp_path = os.path.join(out_dir, f"{forest}_compartment.shp")
                xlsx_path = os.path.join(out_dir, f"{forest}_compartment.xlsx")

                gdf.to_file(shp_path)
                gdf.drop(columns="geometry").to_excel(xlsx_path, index=False)

                preview_img = make_map(gdf, forest, "compartment_preview")

    # ================= SAMPLE =================
    if mode == "sample":

        for forest, group in df.groupby("Forest"):

            group["X"] = pd.to_numeric(group["X"], errors="coerce")
            group["Y"] = pd.to_numeric(group["Y"], errors="coerce")
            group = group.dropna()

            coords = list(zip(group["X"], group["Y"]))
            coords.append(coords[0])

            poly = Polygon(coords)

            grid = fishnet_clip(poly, plot_size, crs)

            points = grid.copy()
            points["geometry"] = points.centroid

            shp_path = os.path.join(out_dir, f"{forest}_sample_points.shp")
            xlsx_path = os.path.join(out_dir, f"{forest}_sample_points.xlsx")

            points.to_file(shp_path)
            export_excel(points, xlsx_path)

            preview_img = make_map(points, forest, "sample_preview")

    # ================= ZIP OUTPUT =================
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as z:
        for root, _, files in os.walk(out_dir):
            for f in files:
                full_path = os.path.join(root, f)
                arcname = os.path.relpath(full_path, out_dir)
                z.write(full_path, arcname)

    zip_buffer.seek(0)

    return zip_buffer, preview_img


# ================= ROUTES =================
@app.route("/")
def home():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():

    file = request.files["file"]
    mode = request.form["mode"]
    zone = request.form["zone"]
    order_mode = request.form.get("order_mode", "auto")

    intensity = float(request.form.get("intensity", 0.5))
    plot_size = float(request.form.get("plot_size", 500))

    filename = secure_filename(file.filename)
    path = os.path.join(UPLOAD, filename)
    file.save(path)

    zip_file, img = process(path, mode, zone, order_mode, intensity, plot_size)

    return send_file(
        zip_file,
        mimetype="application/zip",
        as_attachment=True,
        download_name="gis_output.zip"
    )


# ================= RENDER ENTRY =================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
