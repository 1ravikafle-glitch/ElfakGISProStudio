import os
import zipfile
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt

from flask import Flask, render_template, request
from shapely.geometry import Point, Polygon, LineString

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "outputs"
STATIC_FOLDER = "static"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(STATIC_FOLDER, exist_ok=True)


# ================= CRS =================
def get_crs(zone):
    return "EPSG:32644" if zone == "44" else "EPSG:32645"


# ================= SAMPLE PLOT (FISHNET) =================
def fishnet(bounds, plot_size, crs):

    minx, miny, maxx, maxy = bounds
    step = plot_size ** 0.5

    grids = []

    x = minx
    while x < maxx:
        y = miny
        while y < maxy:

            grids.append(Polygon([
                (x, y),
                (x + step, y),
                (x + step, y + step),
                (x, y + step)
            ]))

            y += step
        x += step

    return gpd.GeoDataFrame(geometry=grids, crs=crs)


# ================= MAP =================
def make_map(poly, points, lines, name):

    fig, ax = plt.subplots(figsize=(10, 6))

    if poly is not None:
        poly.plot(ax=ax, color="lightgreen", edgecolor="darkgreen")

    if lines is not None:
        lines.plot(ax=ax, color="black")

    if points is not None:
        points.plot(ax=ax, color="red", markersize=15)

    ax.set_title(name)
    ax.set_axis_off()

    path = os.path.join(STATIC_FOLDER, "preview.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()

    return path


# ================= PROCESS =================
def process(file_path, mode, zone, intensity, plot_size):

    crs = get_crs(zone)
    base = os.path.splitext(os.path.basename(file_path))[0]

    df = pd.read_excel(file_path)

    out_folder = os.path.join(OUTPUT_FOLDER, base)
    os.makedirs(out_folder, exist_ok=True)

    all_polygons = []

    # ================= MODE 1 + 2 =================
    if mode in ["boundary", "compartment"]:

        for forest, group in df.groupby("Forest"):

            group["X"] = pd.to_numeric(group["X"], errors="coerce")
            group["Y"] = pd.to_numeric(group["Y"], errors="coerce")
            group = group.dropna()

            if mode == "boundary":

                coords = list(zip(group["X"], group["Y"]))
                coords.append(coords[0])

                poly = Polygon(coords)
                line = LineString(coords)
                pts = [Point(xy) for xy in coords]

                poly_gdf = gpd.GeoDataFrame([{"geometry": poly}], crs=crs)
                line_gdf = gpd.GeoDataFrame([{"geometry": line}], crs=crs)
                point_gdf = gpd.GeoDataFrame(geometry=pts, crs=crs)

                poly_gdf.to_file(os.path.join(out_folder, f"{forest}_poly.shp"))
                line_gdf.to_file(os.path.join(out_folder, f"{forest}_line.shp"))
                point_gdf.to_file(os.path.join(out_folder, f"{forest}_pts.shp"))

                img = make_map(poly_gdf, point_gdf, line_gdf, forest)

            # ================= COMPARTMENT MODE =================
            else:

                polys = []

                for comp, cgroup in group.groupby("Compartment"):

                    cgroup = cgroup.sort_values("Order").reset_index(drop=True)
                    cgroup["Order"] = range(1, len(cgroup) + 1)

                    coords = list(zip(cgroup["X"], cgroup["Y"]))
                    coords.append(coords[0])

                    poly = Polygon(coords)

                    polys.append({
                        "Forest": forest,
                        "Compartment": comp,
                        "geometry": poly
                    })

                gdf = gpd.GeoDataFrame(polys, crs=crs)
                gdf.to_file(os.path.join(out_folder, f"{forest}_compartment.shp"))

                img = make_map(gdf, None, None, forest)

    # ================= SAMPLE MODE =================
    if mode == "sample":

        for forest, group in df.groupby("Forest"):

            group["X"] = pd.to_numeric(group["X"])
            group["Y"] = pd.to_numeric(group["Y"])
            group = group.dropna()

            coords = list(zip(group["X"], group["Y"]))
            coords.append(coords[0])

            poly = Polygon(coords)

            fish = fishnet(poly.bounds, plot_size, crs)

            fish = fish[fish.intersects(poly)]

            fish.to_file(os.path.join(out_folder, f"{forest}_sample.shp"))

            poly_gdf = gpd.GeoDataFrame([{"geometry": poly}], crs=crs)

            img = make_map(poly_gdf, None, None, forest)

    # ================= ZIP OUTPUT =================
    zip_path = os.path.join(OUTPUT_FOLDER, f"{base}.zip")

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for root, _, files in os.walk(out_folder):
            for f in files:
                z.write(os.path.join(root, f), arcname=f)

    return zip_path, img


# ================= ROUTES =================
@app.route("/")
def home():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():

    file = request.files["file"]
    mode = request.form["mode"]
    zone = request.form["zone"]

    intensity = float(request.form.get("intensity", 0.5))
    plot_size = float(request.form.get("plot_size", 500))

    path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(path)

    zip_file, img = process(path, mode, zone, intensity, plot_size)

    return render_template("index.html",
                           map_image=img,
                           download_file=zip_file)


if __name__ == "__main__":
    app.run()
