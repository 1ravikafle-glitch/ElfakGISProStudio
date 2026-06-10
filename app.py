import os
import zipfile
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt

from flask import Flask, render_template, request
from shapely.geometry import Polygon

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
def fishnet_clip(polygon, cell_size, crs):

    minx, miny, maxx, maxy = polygon.bounds
    cells = []

    x = minx
    while x < maxx:
        y = miny
        while y < maxy:

            cell = Polygon([
                (x, y),
                (x + cell_size, y),
                (x + cell_size, y + cell_size),
                (x, y + cell_size)
            ])

            clipped = cell.intersection(polygon)
            if not clipped.is_empty:
                cells.append(clipped)

            y += cell_size
        x += cell_size

    return gpd.GeoDataFrame(geometry=cells, crs=crs)


# ================= MAP =================
def make_map(gdf, name, filename):

    fig, ax = plt.subplots(figsize=(10, 6))

    gdf.plot(ax=ax, edgecolor="black", alpha=0.7)

    ax.set_title(name)
    ax.set_axis_off()

    path = os.path.join(STATIC, filename + ".png")
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()

    return path


# ================= EXCEL EXPORT =================
def export_points(gdf, path):
    df = gdf.copy()
    df["X"] = df.geometry.x
    df["Y"] = df.geometry.y
    df.drop(columns="geometry").to_excel(path, index=False)


# ================= PROCESS =================
def process(file_path, mode, zone, order_mode, intensity, plot_size):

    crs = get_crs(zone)
    base = os.path.splitext(os.path.basename(file_path))[0]

    df = pd.read_excel(file_path)

    out_dir = os.path.join(OUTPUT, base)
    os.makedirs(out_dir, exist_ok=True)

    zip_path = os.path.join(OUTPUT, f"{base}.zip")

    map_images = {}
    stats = {
        "total_area": 0,
        "plot_count": 0,
        "plot_size": plot_size,
        "crs": zone
    }

    # ================= BOUNDARY / COMPARTMENT =================
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

                shp = os.path.join(out_dir, f"{forest}_boundary.shp")
                xlsx = os.path.join(out_dir, f"{forest}_boundary.xlsx")

                gdf.to_file(shp)
                gdf.to_file(xlsx.replace(".xlsx", ".gpkg"))

                map_images[forest] = make_map(gdf, forest, f"{forest}_boundary")

                stats["total_area"] += poly.area

            else:

                polys = []

                for i, (comp, cgroup) in enumerate(group.groupby("Compartment")):

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

                shp = os.path.join(out_dir, f"{forest}_compartment.shp")
                gdf.to_file(shp)

                map_images[forest] = make_map(gdf, forest, f"{forest}_compartment")

    # ================= SAMPLE (FISHNET + CLIP) =================
    if mode == "sample":

        for forest, group in df.groupby("Forest"):

            group["X"] = pd.to_numeric(group["X"])
            group["Y"] = pd.to_numeric(group["Y"])
            group = group.dropna()

            coords = list(zip(group["X"], group["Y"]))
            coords.append(coords[0])

            poly = Polygon(coords)

            cell_size = float(plot_size)

            grid = fishnet_clip(poly, cell_size, crs)

            points = grid.copy()
            points["geometry"] = points.centroid

            shp = os.path.join(out_dir, f"{forest}_sample.shp")
            xlsx = os.path.join(out_dir, f"{forest}_sample.xlsx")

            points.to_file(shp)
            export_points(points, xlsx)

            map_images[forest] = make_map(points, forest, f"{forest}_sample")

            stats["plot_count"] += len(points)

    # ================= ZIP =================
    with zipfile.ZipFile(zip_path, "w") as z:
        for root, _, files in os.walk(out_dir):
            for f in files:
                z.write(os.path.join(root, f), arcname=f)

    return zip_path, map_images, stats


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

    zip_file, map_images, stats = process(
        path, mode, zone, order_mode, intensity, plot_size
    )

    return render_template(
        "index.html",
        map_images=map_images,
        download_file=zip_file,
        stats=stats,
        chosen_mode=mode,
        chosen_zone=zone,
        chosen_order=order_mode,
        chosen_intensity=intensity,
        chosen_plot_size=plot_size
    )


if __name__ == "__main__":
    app.run(debug=True)
