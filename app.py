import os
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt

from flask import Flask, render_template, request, send_file
from shapely.geometry import Point, Polygon, LineString

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "CF_OUTPUT"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)


# ================= CRS =================
def get_crs(zone):
    return "EPSG:32644" if zone == "44" else "EPSG:32645"


# ================= COMPARTMENT MODE =================
def process_compartment(file_path, crs, out_dir):

    df = pd.read_excel(file_path)

    for forest, group in df.groupby("Forest"):

        folder = os.path.join(out_dir, "COMPARTMENT_CF", str(forest))
        os.makedirs(folder, exist_ok=True)

        group["X"] = pd.to_numeric(group["X"], errors="coerce")
        group["Y"] = pd.to_numeric(group["Y"], errors="coerce")
        group = group.dropna()

        points, lines, polys = [], [], []

        for comp, cgroup in group.groupby("Compartment"):

            cgroup = cgroup.sort_values("Order")
            coords = list(zip(cgroup["X"], cgroup["Y"]))

            if len(coords) < 3:
                continue

            if coords[0] != coords[-1]:
                coords.append(coords[0])

            poly = Polygon(coords).buffer(0)
            line = LineString(coords)

            if not poly.is_valid:
                continue

            polys.append({"Comp": comp, "geometry": poly})
            lines.append({"Comp": comp, "geometry": line})

            for _, r in cgroup.iterrows():
                points.append({"Comp": comp, "geometry": Point(r["X"], r["Y"])})

        gpd.GeoDataFrame(points, crs=crs).to_file(os.path.join(folder, "points.shp"))
        gpd.GeoDataFrame(lines, crs=crs).to_file(os.path.join(folder, "lines.shp"))
        gpd.GeoDataFrame(polys, crs=crs).to_file(os.path.join(folder, "polygons.shp"))


# ================= SINGLE FOREST =================
def process_single(file_path, crs, out_dir):

    df = pd.read_excel(file_path)

    for forest, group in df.groupby("Forest"):

        folder = os.path.join(out_dir, "SINGLE_CF", str(forest))
        os.makedirs(folder, exist_ok=True)

        group = group.sort_values("Order")
        coords = list(zip(group["X"], group["Y"]))

        if coords[0] != coords[-1]:
            coords.append(coords[0])

        poly = Polygon(coords).buffer(0)
        line = LineString(coords)

        gpd.GeoDataFrame([{"geometry": poly}]).to_file(os.path.join(folder, "polygon.shp"))
        gpd.GeoDataFrame([{"geometry": line}]).to_file(os.path.join(folder, "line.shp"))


# ================= SAMPLE POINTS =================
def process_points(file_path, crs, out_dir):

    if file_path.endswith(".xlsx"):
        df = pd.read_excel(file_path)
        gdf = gpd.GeoDataFrame(
            df,
            geometry=[Point(xy) for xy in zip(df["X"], df["Y"])],
            crs=crs
        )
    else:
        gdf = gpd.read_file(file_path)

    folder = os.path.join(out_dir, "SAMPLE_POINTS")
    os.makedirs(folder, exist_ok=True)

    gdf.to_file(os.path.join(folder, "sample_points.shp"))


# ================= PROCESS ROUTER =================
def process(file_path, mode, crs_zone):

    crs = get_crs(crs_zone)

    if mode == "comp":
        process_compartment(file_path, crs, OUTPUT_FOLDER)

    elif mode == "single":
        process_single(file_path, crs, OUTPUT_FOLDER)

    elif mode == "points":
        process_points(file_path, crs, OUTPUT_FOLDER)


# ================= FLASK ROUTES =================
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

    process(path, mode, crs)

    return "Processing completed ✔ Check CF_OUTPUT folder on server"


if __name__ == "__main__":
    app.run()
