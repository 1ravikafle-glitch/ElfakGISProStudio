import os
import zipfile
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt

from flask import Flask, render_template, request, send_file, jsonify
from shapely.geometry import Polygon, Point, LineString
from io import BytesIO

app = Flask(__name__)

UPLOAD = "uploads"
OUTPUT = "outputs"
STATIC = "static"

os.makedirs(UPLOAD, exist_ok=True)
os.makedirs(OUTPUT, exist_ok=True)
os.makedirs(STATIC, exist_ok=True)


# ================= CRS =================
def get_crs(zone):
    return "EPSG:32644" if zone == "44" else "EPSG:32645"


# ================= GROUP A =================
def process_a(df, forest, crs):
    df = df.sort_values("Order")
    coords = list(zip(df["X"], df["Y"]))
    coords.append(coords[0])

    poly = Polygon(coords)
    line = LineString(coords)

    gdf_poly = gpd.GeoDataFrame([{
        "Forest": forest,
        "Area_Ha": poly.area / 10000,
        "Perim_M": poly.length,
        "geometry": poly
    }], crs=crs)

    gdf_point = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df["X"], df["Y"]),
        crs=crs
    )

    return gdf_poly, gdf_point


# ================= GROUP B =================
def process_b(df, crs):
    polys, pts = [], []

    for comp, g in df.groupby("Compartment"):
        g = g.sort_values("Order")

        coords = list(zip(g["X"], g["Y"]))
        coords.append(coords[0])

        poly = Polygon(coords)

        polys.append({
            "Forest": g["Forest"].iloc[0],
            "Comp": comp,
            "Area_Ha": poly.area / 10000,
            "Perim_M": poly.length,
            "geometry": poly
        })

        for _, r in g.iterrows():
            pts.append({
                "Forest": r["Forest"],
                "Comp": comp,
                "Order": r["Order"],
                "geometry": Point(r["X"], r["Y"])
            })

    return gpd.GeoDataFrame(polys, crs=crs), gpd.GeoDataFrame(pts, crs=crs)


# ================= FISHNET =================
def fishnet(poly, w, h):
    minx, miny, maxx, maxy = poly.bounds
    pts = []

    y = miny
    while y < maxy:
        x = minx
        while x < maxx:
            cell = Polygon([(x,y),(x+w,y),(x+w,y+h),(x,y+h)])
            pts.append(cell.centroid)
            x += w
        y += h

    return pts


# ================= GROUP C =================
def process_c(poly, crs, w, h):
    centroids = fishnet(poly, w, h)
    inside = [p for p in centroids if poly.contains(p)]

    gdf = gpd.GeoDataFrame({
        "S.N": range(1, len(inside)+1)
    }, geometry=inside, crs=crs)

    gdf["X"] = gdf.geometry.x
    gdf["Y"] = gdf.geometry.y

    return gdf


# ================= PREVIEW =================
def preview(poly_gdf, pts_gdf, path, color="red"):
    fig, ax = plt.subplots()
    poly_gdf.plot(ax=ax, color="none", edgecolor="red")
    pts_gdf.plot(ax=ax, color=color, markersize=10)
    plt.axis("off")
    plt.savefig(path, dpi=200)
    plt.close()


# ================= UPLOAD =================
@app.route("/upload", methods=["POST"])
def upload():

    file = request.files["file"]
    mode = request.form["mode"]
    zone = request.form["zone"]
    forest = request.form.get("forest", "FOREST")

    w = float(request.form.get("w", 50))
    h = float(request.form.get("h", 50))

    path = os.path.join(UPLOAD, file.filename)
    file.save(path)

    df = pd.read_excel(path)
    crs = get_crs(zone)

    out = os.path.join(OUTPUT, "result")
    os.makedirs(out, exist_ok=True)

    preview_path = os.path.join(STATIC, "preview.png")

    # ================= A =================
    if mode == "A":
        poly, pts = process_a(df, forest, crs)
        preview(poly, pts, preview_path)

    # ================= B =================
    elif mode == "B":
        poly, pts = process_b(df, crs)
        preview(poly, pts, preview_path)

    # ================= C =================
    else:
        poly = gpd.GeoDataFrame({"geometry":[Polygon(list(zip(df["X"],df["Y"])))]}, crs=crs)
        pts = process_c(poly.geometry.iloc[0], crs, w, h)
        preview(poly, pts, preview_path, "yellow")

    zip_buffer = BytesIO()

    with zipfile.ZipFile(zip_buffer, "w") as z:
        for root, _, files in os.walk(out):
            for f in files:
                z.write(os.path.join(root, f), f)

    zip_buffer.seek(0)

    return jsonify({"preview": "/static/preview.png"})


# ================= UI =================
@app.route("/")
def home():
    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)
