import os
import zipfile
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import math

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


# ================= EXPORT =================
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

    # ================= METRICS =================
    total_area_m2 = 0.0
    total_perimeter_m = 0.0
    total_plots = 0

    # ================= BOUNDARY / COMPARTMENT =================
    if mode in ["boundary", "compartment"]:

        for forest, group in df.groupby("Forest"):

            group["X"] = pd.to_numeric(group["X"], errors="coerce")
            group["Y"] = pd.to_numeric(group["Y"], errors="coerce")
            group = group.dropna(subset=["X", "Y"])

            if group.empty:
                continue

            # ---------------- BOUNDARY ----------------
            if mode == "boundary":

                if order_mode == "auto":
                    group = group.sort_values("Order")

                coords = list(zip(group["X"], group["Y"]))
                coords.append(coords[0])

                poly = Polygon(coords)

                total_area_m2 += poly.area
                total_perimeter_m += poly.length

                gdf = gpd.GeoDataFrame([{
                    "Forest": forest,
                    "Area_ha": poly.area / 10000,
                    "Perimeter_m": poly.length,
                    "geometry": poly
                }], crs=crs)

                shp = os.path.join(out_dir, f"{forest}_boundary.shp")
                gpkg = shp.replace(".shp", ".gpkg")

                gdf.to_file(shp)
                gdf.to_file(gpkg)

                map_images[forest] = make_map(gdf, forest, f"{forest}_boundary")

           # ---------------- COMPARTMENT (REAL TKINTER LOGIC) ----------------

else:

    for forest, group in df.groupby("Forest"):

        group["X"] = pd.to_numeric(group["X"], errors="coerce")
        group["Y"] = pd.to_numeric(group["Y"], errors="coerce")
        group = group.dropna(subset=["X", "Y"])

        if group.empty:
            continue

        if order_mode == "auto":
            group = group.sort_values("Order")

        forest_folder = os.path.join(out_dir, str(forest))
        os.makedirs(forest_folder, exist_ok=True)

        all_points = []
        all_lines = []
        all_polygons = []

        # 🔥 REAL COMPARTMENT PROCESSING
        for comp, comp_group in group.groupby("Compartment"):

            comp_group = comp_group.sort_values("Order")

            coords = list(zip(comp_group["X"], comp_group["Y"]))

            if len(coords) < 3:
                continue

            if coords[0] != coords[-1]:
                coords.append(coords[0])

            poly = Polygon(coords)

            if not poly.is_valid:
                poly = poly.buffer(0)

            if poly.is_empty:
                continue

            line = poly.boundary

            total_area_m2 += poly.area
            total_perimeter_m += poly.length

            all_polygons.append({
                "Forest": forest,
                "Comp": comp,
                "Area_ha": round(poly.area / 10000, 4),
                "Perimeter_m": round(poly.length, 2),
                "geometry": poly
            })

            all_lines.append({
                "Forest": forest,
                "Comp": comp,
                "geometry": line
            })

            for _, r in comp_group.iterrows():
                all_points.append({
                    "Forest": forest,
                    "Comp": comp,
                    "Order": r["Order"],
                    "geometry": Point(r["X"], r["Y"])
                })

        if not all_polygons:
            continue

        # ================= GEO DATAFRAMES =================
        gdf_poly = gpd.GeoDataFrame(all_polygons, crs=crs)
        gdf_line = gpd.GeoDataFrame(all_lines, crs=crs)
        gdf_point = gpd.GeoDataFrame(all_points, crs=crs)

        # ================= SAVE FILES =================
        gdf_poly.to_file(os.path.join(forest_folder, "polygons.shp"))
        gdf_line.to_file(os.path.join(forest_folder, "lines.shp"))
        gdf_point.to_file(os.path.join(forest_folder, "points.shp"))

        gpkg = os.path.join(forest_folder, f"{forest}_compartment.gpkg")
        gdf_poly.to_file(gpkg)

        # ================= MAP =================
        map_images[forest] = make_map(
            gdf_poly,
            forest,
            f"{forest}_compartment"
        )

    # ================= SAMPLE MODE =================
    elif mode == "sample":

        for forest, group in df.groupby("Forest"):

            group["X"] = pd.to_numeric(group["X"], errors="coerce")
            group["Y"] = pd.to_numeric(group["Y"], errors="coerce")
            group = group.dropna(subset=["X", "Y"])

            if group.empty:
                continue

            if order_mode == "auto":
                group = group.sort_values("Order")

            coords = list(zip(group["X"], group["Y"]))
            coords.append(coords[0])

            poly = Polygon(coords)

            total_area_m2 += poly.area
            total_perimeter_m += poly.length

            cell_size = math.sqrt(float(plot_size))
            grid = fishnet_clip(poly, cell_size, crs)

            if not grid.empty:
                points = grid.copy()
                points["geometry"] = points.centroid

                total_plots += len(points)

                shp = os.path.join(out_dir, f"{forest}_sample.shp")
                xlsx = os.path.join(out_dir, f"{forest}_sample.xlsx")

                points.to_file(shp)
                export_points(points, xlsx)

                map_images[forest] = make_map(points, forest, f"{forest}_sample")

    # ================= ZIP =================
    with zipfile.ZipFile(zip_path, "w") as z:
        for root, _, files in os.walk(out_dir):
            for f in files:
                z.write(os.path.join(root, f), arcname=f)

    # ================= FINAL STATS =================
    stats = {
        "total_area": f"{round(total_area_m2 / 10000, 2)} ha",
        "total_perimeter": f"{round(total_perimeter_m, 2)} m",
        "plot_count": str(total_plots) if mode == "sample" else "N/A",
        "plot_size": f"{int(plot_size)} m²" if mode == "sample" else "N/A",
        "crs": f"UTM {zone}N"
    }

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
