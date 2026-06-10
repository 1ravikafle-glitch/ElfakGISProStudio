import os
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt

from flask import Flask, render_template, request, send_file
from shapely.geometry import Point, Polygon, LineString

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "CF_OUTPUT"
STATIC_FOLDER = "static"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(STATIC_FOLDER, exist_ok=True)


# ================= CRS =================
def get_crs(zone):
    return "EPSG:32644" if zone == "44" else "EPSG:32645"


# ================= MAP =================
def create_map(poly_gdf, point_gdf, line_gdf, name):

    fig, ax = plt.subplots(figsize=(10, 6))

    poly_gdf.plot(ax=ax, color="lightgreen", edgecolor="darkgreen")
    line_gdf.plot(ax=ax, color="black")
    point_gdf.plot(ax=ax, color="red", markersize=20)

    ax.set_title(name, fontsize=14)
    ax.set_axis_off()

    img_path = os.path.join(STATIC_FOLDER, "preview.png")
    plt.savefig(img_path, dpi=150, bbox_inches="tight")
    plt.close()

    return img_path


# ================= PROCESS =================
def process(file_path, mode, crs_zone):

    crs = get_crs(crs_zone)

    base_name = os.path.splitext(os.path.basename(file_path))[0]

    df = pd.read_excel(file_path)

    all_points = []
    all_lines = []
    all_polygons = []

    for forest, group in df.groupby("Forest"):

        group["X"] = pd.to_numeric(group["X"], errors="coerce")
        group["Y"] = pd.to_numeric(group["Y"], errors="coerce")
        group = group.dropna()

        coords = list(zip(group["X"], group["Y"]))

        if len(coords) < 3:
            continue

        if coords[0] != coords[-1]:
            coords.append(coords[0])

        poly = Polygon(coords).buffer(0)
        line = LineString(coords)

        points = [Point(xy) for xy in coords]

        poly_gdf = gpd.GeoDataFrame([{"geometry": poly}], crs=crs)
        line_gdf = gpd.GeoDataFrame([{"geometry": line}], crs=crs)
        point_gdf = gpd.GeoDataFrame(geometry=points, crs=crs)

        # folder per file
        out_folder = os.path.join(OUTPUT_FOLDER, base_name)
        os.makedirs(out_folder, exist_ok=True)

        # save shapefiles
        poly_gdf.to_file(os.path.join(out_folder, "polygon.shp"))
        line_gdf.to_file(os.path.join(out_folder, "line.shp"))
        point_gdf.to_file(os.path.join(out_folder, "points.shp"))

        # map preview
        img = create_map(poly_gdf, point_gdf, line_gdf, forest)

    # ZIP OUTPUT (named like upload file)
    zip_path = os.path.join(OUTPUT_FOLDER, f"{base_name}.zip")

    import zipfile
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(out_folder):
            for f in files:
                zipf.write(os.path.join(root, f), arcname=f)

    return zip_path, img


# ================= ROUTES =================
@app.route("/")
def home():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():

    file = request.files["file"]
    mode = request.form["mode"]
    crs = request.form["crs"]

    path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(path)

    zip_file, map_img = process(path, mode, crs)

    return render_template(
        "index.html",
        map_image=map_img,
        download_file=zip_file
    )


if __name__ == "__main__":
    app.run()
