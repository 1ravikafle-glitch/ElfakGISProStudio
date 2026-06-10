import os
import zipfile
import math
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


# ================= AREA =================
def calculate_n(area_ha, intensity, plot_size_m2):
    a_ha = plot_size_m2 / 10000
    n = (area_ha * intensity) / (a_ha * 100)
    return max(1, int(round(n)))


# ================= FISHNET + CLIP =================
def fishnet_clip(polygon, n, crs):

    minx, miny, maxx, maxy = polygon.bounds
    area_m2 = polygon.area
    spacing = math.sqrt(area_m2 / (n + 1))

    cells = []
    x = minx

    while x < maxx:
        y = miny
        while y < maxy:

            cell = Polygon([
                (x, y),
                (x + spacing, y),
                (x + spacing, y + spacing),
                (x, y + spacing)
            ])

            clipped = cell.intersection(polygon)

            if not clipped.is_empty:
                cells.append(clipped.centroid)

            y += spacing
        x += spacing

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
                gdf.to_file(os.path.join(out_dir, f"{forest}_poly.shp"))

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

    # ================= SAMPLE (FIXED + EXCEL + SHP) =================
    if mode == "sample":

        for forest, group in df.groupby("Forest"):

            group["X"] = pd.to_numeric(group["X"])
            group["Y"] = pd.to_numeric(group["Y"])
            group = group.dropna()

            coords = list(zip(group["X"], group["Y"]))
            coords.append(coords[0])

            poly = Polygon(coords)

            # AREA
            area_ha = poly.area / 10000

            # NUMBER OF PLOTS
            n = calculate_n(area_ha, intensity, plot_size)

            # FISHNET
            grid_points = fishnet_clip(poly, n, crs)

            # RESET S.N
            grid_points = grid_points.reset_index(drop=True)
            grid_points["S.N"] = range(1, len(grid_points) + 1)
            grid_points["Forest"] = forest

            # ================= SHP OUTPUT =================
            shp_path = os.path.join(out_dir, f"{forest}_sample.shp")
            grid_points.to_file(shp_path)

            # ================= EXCEL OUTPUT =================
            excel_df = pd.DataFrame({
                "S.N": grid_points["S.N"],
                "Forest": grid_points["Forest"],
                "X": grid_points.geometry.x,
                "Y": grid_points.geometry.y
            })

            excel_path = os.path.join(out_dir, f"{forest}_sample.xlsx")
            excel_df.to_excel(excel_path, index=False)

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
