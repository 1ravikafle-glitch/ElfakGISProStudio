import os
import zipfile
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt

from flask import Flask, render_template, request, send_file
from shapely.geometry import Polygon, Point, LineString

app = Flask(__name__)

UPLOAD = "uploads"
OUTPUT = "outputs"
STATIC = "static"

os.makedirs(UPLOAD, exist_ok=True)
os.makedirs(OUTPUT, exist_ok=True)
os.makedirs(STATIC, exist_ok=True)


# ================= CRS =================
def get_crs(z):
    return "EPSG:32644" if str(z) == "44" else "EPSG:32645"


# ================= GEOMETRY BUILDER =================
def build_geometry(group):

    group = group.dropna()

    if "Order" in group.columns:
        group = group.sort_values("Order")

    coords = list(zip(group["X"], group["Y"]))

    if len(coords) == 0:
        return None, None, None

    # POINTS
    points = [Point(xy) for xy in coords]

    # LINE
    line = LineString(coords) if len(coords) >= 2 else None

    # POLYGON (CLOSED)
    polygon = None
    if len(coords) >= 3:
        polygon = Polygon(coords + [coords[0]])

    return points, line, polygon


# ================= FISHNET =================
def fishnet_clip(
    polygon,
    cell_width,
    cell_height,
    rows,
    cols,
    crs
):

    minx, miny, maxx, maxy = polygon.bounds

    cells = []

    for r in range(rows):
        for c in range(cols):

            x1 = minx + (c * cell_width)
            y1 = miny + (r * cell_height)

            x2 = x1 + cell_width
            y2 = y1 + cell_height

            cell = Polygon([
                (x1, y1),
                (x2, y1),
                (x2, y2),
                (x1, y2)
            ])

            clipped = cell.intersection(polygon)

            if not clipped.is_empty:
                cells.append(clipped)

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

    return filename + ".png"


# ================= EXPORT =================
def export_points(gdf, path):

    df = gdf.copy()
    df["X"] = df.geometry.x
    df["Y"] = df.geometry.y

    df.drop(columns="geometry").to_excel(path, index=False)


# ================= PROCESS =================
def process(
    file_path,
    mode,
    zone,
    order_mode,
    intensity,
    plot_size,
    cell_width,
    cell_height,
    rows,
    cols
):

    crs = get_crs(zone)

    base = os.path.splitext(os.path.basename(file_path))[0]

    df = pd.read_excel(file_path)

    required_cols = ["Forest", "X", "Y"]

    for c in required_cols:
        if c not in df.columns:
            raise Exception(f"Missing column: {c}")

    out_dir = os.path.join(OUTPUT, base)
    os.makedirs(out_dir, exist_ok=True)

    zip_name = f"{base}.zip"
    zip_path = os.path.join(OUTPUT, zip_name)

    map_images = {}

    total_area_m2 = 0.0
    total_perimeter_m = 0.0
    total_plots = 0

    # ================= BOUNDARY =================
    if mode == "boundary":

        for forest, group in df.groupby("Forest"):

            points, line, polygon = build_geometry(group)

            if polygon is None:
                continue

            total_area_m2 += polygon.area
            total_perimeter_m += polygon.length

            gdf = gpd.GeoDataFrame([{
                "Forest": forest,
                "Area_ha": polygon.area / 10000,
                "Perimeter_m": polygon.length,
                "geometry": polygon
            }], crs=crs)

            shp = os.path.join(out_dir, f"{forest}_boundary.shp")
            gdf.to_file(shp)

            # optional extra layers
            if line:
                gpd.GeoDataFrame([{"geometry": line}], crs=crs).to_file(
                    os.path.join(out_dir, f"{forest}_line.shp")
                )

            if points:
                gpd.GeoDataFrame([{"geometry": p} for p in points], crs=crs).to_file(
                    os.path.join(out_dir, f"{forest}_points.shp")
                )

            map_images[forest] = make_map(gdf, forest, f"{forest}_boundary")


    # ================= COMPARTMENT =================
    elif mode == "compartment":

        for forest, group in df.groupby("Forest"):

            forest_folder = os.path.join(out_dir, str(forest))
            os.makedirs(forest_folder, exist_ok=True)

            polys = []

            for comp, cg in group.groupby("Compartment"):

                points, line, polygon = build_geometry(cg)

                if polygon is None:
                    continue

                total_area_m2 += polygon.area
                total_perimeter_m += polygon.length

                polys.append({
                    "Forest": forest,
                    "Comp": comp,
                    "Area_ha": polygon.area / 10000,
                    "geometry": polygon
                })

            gdf = gpd.GeoDataFrame(polys, crs=crs)

            shp = os.path.join(forest_folder, "compartment.shp")
            gdf.to_file(shp)

            map_images[forest] = make_map(gdf, forest, f"{forest}_compartment")


    # ================= SAMPLE =================
    elif mode == "sample":

        cell_width = float(cell_width)
        cell_height = float(cell_height)
        rows = int(rows)
        cols = int(cols)

        for forest, group in df.groupby("Forest"):

            group = group.dropna()

            coords = list(zip(group["X"], group["Y"]))

            if len(coords) < 3:
                continue

            coords.append(coords[0])

            poly = Polygon(coords)

            total_area_m2 += poly.area
            total_perimeter_m += poly.length

            fishnet = fishnet_clip(
                poly,
                cell_width,
                cell_height,
                rows,
                cols,
                crs
            )

            if not fishnet.empty:

                fishnet["geometry"] = fishnet.centroid

                fishnet["Plot_No"] = range(1, len(fishnet) + 1)

                total_plots += len(fishnet)

                shp = os.path.join(out_dir, f"{forest}_sample.shp")
                xlsx = os.path.join(out_dir, f"{forest}_sample.xlsx")

                fishnet.to_file(shp)
                export_points(fishnet, xlsx)

                map_images[forest] = make_map(
                    fishnet,
                    forest,
                    f"{forest}_sample"
                )


    # ================= ZIP =================
    with zipfile.ZipFile(zip_path, "w") as z:
        for root, _, files in os.walk(out_dir):
            for f in files:
                z.write(os.path.join(root, f), arcname=f)

    stats = {
        "total_area": f"{round(total_area_m2 / 10000, 2)} ha",
        "total_perimeter": f"{round(total_perimeter_m, 2)} m",
        "plot_count": str(total_plots),
        "plot_size": f"{cell_width} m × {cell_height} m",
        "crs": f"UTM {zone}N"
    }

    return zip_name, map_images, stats


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
    intensity = request.form.get("intensity", 0.5)
    plot_size = request.form.get("plot_size", 500)

    cell_width = request.form.get("cell_width", 100)
    cell_height = request.form.get("cell_height", 100)
    rows = request.form.get("rows", 10)
    cols = request.form.get("cols", 10)

    path = os.path.join(UPLOAD, file.filename)
    file.save(path)

    try:
        zip_file, map_images, stats = process(
            path, mode, zone,
            order_mode, intensity,
            plot_size,
            cell_width, cell_height,
            rows, cols
        )

        return render_template(
            "index.html",
            map_images=map_images,
            download_file=zip_file,
            stats=stats
        )

    except Exception as e:
        return render_template("index.html", error=str(e))


# ================= DOWNLOAD =================
@app.route("/download/<filename>")
def download(filename):
    return send_file(os.path.join(OUTPUT, filename), as_attachment=True)


if __name__ == "__main__":
    app.run(debug=True)
