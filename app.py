import os
import math
import zipfile
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt

from flask import Flask, render_template, request, send_file
from shapely.geometry import Polygon, Point, LineString, box

app = Flask(__name__)

UPLOAD = "uploads"
OUT = "CF_OUTPUT"
STATIC = "static"

os.makedirs(UPLOAD, exist_ok=True)
os.makedirs(OUT, exist_ok=True)
os.makedirs(STATIC, exist_ok=True)


# ================= CRS =================
def get_crs(zone):
    return "EPSG:32644" if zone == "44" else "EPSG:32645"


# ================= SAMPLE FORMULA =================
def calc_n(area_ha, plot_area_ha, intensity):
    return max(1, math.ceil((area_ha * intensity) / (plot_area_ha * 100)))


def spacing(area_m2, n):
    return math.sqrt(area_m2 / (n + 1))


# ================= FISHNET =================
def fishnet(poly, dist):
    minx, miny, maxx, maxy = poly.bounds
    pts = []

    x = minx
    while x < maxx:
        y = miny
        while y < maxy:
            p = Point(x + dist/2, y + dist/2)
            if poly.contains(p):
                pts.append(p)
            y += dist
        x += dist

    return pts


# ================= MAP =================
def save_map(poly, pts, name):
    fig, ax = plt.subplots(figsize=(8, 6))

    gpd.GeoSeries([poly]).plot(ax=ax, color="lightgreen", edgecolor="green")

    if pts:
        gpd.GeoSeries(pts).plot(ax=ax, color="red", markersize=10)

    ax.set_title(name)
    ax.set_axis_off()

    path = os.path.join(STATIC, "preview.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()

    return path


# ================= PROCESS =================
def process(file_path, mode, zone, intensity=0.5, plot_size_m2=500):

    crs = get_crs(zone)
    df = pd.read_excel(file_path)
    base = os.path.splitext(os.path.basename(file_path))[0]

    out_folder = os.path.join(OUT, base)
    os.makedirs(out_folder, exist_ok=True)

    img_path = None

    for forest, group in df.groupby("Forest"):

        group = group.sort_values("Order")
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

        area_m2 = poly.area
        area_ha = area_m2 / 10000

        pts = []

        # ================= MODE SWITCH =================
        if mode == "sample":

            n = calc_n(area_ha, plot_size_m2/10000, intensity)
            dist = spacing(area_m2, n)

            pts = fishnet(poly, dist)

            df_out = pd.DataFrame(
                [(i+1, p.x, p.y) for i, p in enumerate(pts)],
                columns=["S.N", "X", "Y"]
            )

            df_out.to_excel(os.path.join(out_folder, "sample_points.xlsx"), index=False)

        else:
            pts = [Point(xy) for xy in coords]

        # ================= GEO =================
        gdf_poly = gpd.GeoDataFrame([{"geometry": poly}], crs=crs)
        gdf_line = gpd.GeoDataFrame([{"geometry": line}], crs=crs)
        gdf_pts = gpd.GeoDataFrame(geometry=pts, crs=crs)

        gdf_poly.to_file(os.path.join(out_folder, "polygon.shp"))
        gdf_line.to_file(os.path.join(out_folder, "line.shp"))
        gdf_pts.to_file(os.path.join(out_folder, "points.shp"))

        img_path = save_map(poly, pts, forest)

    # ================= ZIP =================
    zip_path = os.path.join(OUT, f"{base}.zip")

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for f in os.listdir(out_folder):
            z.write(os.path.join(out_folder, f), f)

    return zip_path, img_path


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

    path = os.path.join(UPLOAD, file.filename)
    file.save(path)

    zipf, img = process(path, mode, zone, intensity, plot_size)

    return render_template(
        "index.html",
        map_image=img,
        download_file=zipf
    )


if __name__ == "__main__":
    app.run(debug=True)
