import os
import zipfile
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt

from flask import Flask, render_template, request
from shapely.geometry import Point, Polygon

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


# ================= FISHNET + CLIP =================
def fishnet_clip(polygon, plot_size_m2, crs):

    import math

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
def make_map(gdf, name):

    fig, ax = plt.subplots(figsize=(10, 6))

    gdf.plot(ax=ax, color="lightgreen", edgecolor="black")

    ax.set_title(name)
    ax.set_axis_off()

    path = os.path.join(STATIC, "preview.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()

    return path


# ================= PROCESS =================
def process(file_path, mode, zone, order_mode, intensity, plot_size):

    crs = get_crs(zone)
    base = os.path.splitext(os.path.basename(file_path))[0]

    df = pd.read_excel(file_path)

    out_dir = os.path.join(OUTPUT, base)
    os.makedirs(out_dir, exist_ok=True)

    zip_path = os.path.join(OUTPUT, f"{base}.zip")
    preview_img = None

    # ================= BOUNDARY =================
    if mode in ["boundary", "compartment"]:

        for forest, group in df.groupby("Forest"):

            group["X"] = pd.to_numeric(group["X"])
            group["Y"] = pd.to_numeric(group["Y"])
            group = group.dropna()

            if mode == "boundary":

                coords = list(zip(group["X"], group["Y"]))
                coords.append(coords[0])

                poly = Polygon(coords)

                gdf = gpd.GeoDataFrame([{"geometry": poly}], crs=crs)
                path_shp = os.path.join(out_dir, f"{forest}_poly.shp")
                gdf.to_file(path_shp)

                preview_img = make_map(gdf, forest)

            else:

                polys = []

                for comp, cgroup in group.groupby("Compartment"):

                    if order_mode == "auto":
                        cgroup = cgroup.sort_values("Order").reset_index(drop=True)
                        cgroup["Order"] = range(1, len(cgroup) + 1)

                    coords = list(zip(cgroup["X"], cgroup["Y"]))
                    coords.append(coords[0])

                    polys.append({
                        "Forest": forest,
                        "Compartment": comp,
                        "geometry": Polygon(coords)
                    })

                gdf = gpd.GeoDataFrame(polys, crs=crs)

                gdf.to_file(os.path.join(out_dir, f"{forest}_compartment.shp"))

                preview_img = make_map(gdf, forest)

    # ================= SAMPLE =================
    if mode == "sample":

        for forest, group in df.groupby("Forest"):

            group["X"] = pd.to_numeric(group["X"])
            group["Y"] = pd.to_numeric(group["Y"])
            group = group.dropna()

            coords = list(zip(group["X"], group["Y"]))
            coords.append(coords[0])

            poly = Polygon(coords)

            grid = fishnet_clip(poly, plot_size, crs)

            # OPTIONAL: convert to centroid points (BEST FOR FIELD WORK)
            grid_points = grid.copy()
            grid_points["geometry"] = grid_points.centroid

            shp_path = os.path.join(out_dir, f"{forest}_sample.shp")
            grid_points.to_file(shp_path)

            preview_img = make_map(grid_points, forest)

    # ================= ZIP =================
    with zipfile.ZipFile(zip_path, "w") as z:
        for root, _, files in os.walk(out_dir):
            for f in files:
                z.write(os.path.join(root, f), arcname=f)

    return zip_path, preview_img


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

    path = os.path.join(UPLOAD, file.filename)
    file.save(path)

    zip_file, img = process(path, mode, zone, order_mode, intensity, plot_size)

    return render_template(
        "index.html",
        map_image=img,
        download_file=zip_file
    )


if __name__ == "__main__":
    app.run(debug=True)
