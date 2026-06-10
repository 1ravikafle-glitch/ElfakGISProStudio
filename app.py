import os
import zipfile
import tempfile
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt

from flask import Flask, render_template, request, send_file
from shapely.geometry import Polygon, MultiPolygon

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


# ================= SHP ZIP LOADER =================
def load_shapefile_from_zip(zip_path):

    temp_dir = tempfile.mkdtemp()

    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(temp_dir)

    shp_file = None

    for f in os.listdir(temp_dir):
        if f.endswith(".shp"):
            shp_file = os.path.join(temp_dir, f)
            break

    if shp_file is None:
        raise Exception("No .shp file found inside ZIP")

    gdf = gpd.read_file(shp_file)

    return gdf.geometry.unary_union


# ================= FISHNET =================
def fishnet_clip(polygon, cell_width, cell_height, rows, cols, crs):

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

            cells.append(cell)

    return gpd.GeoDataFrame(geometry=cells, crs=crs)


# ================= PLOT =================
def make_preview(polygon, fishnet_gdf, title):

    fig, ax = plt.subplots(figsize=(8, 6))

    # ---- polygon boundary ----
    if isinstance(polygon, Polygon):
        x, y = polygon.exterior.xy
        ax.plot(x, y, color="red", linewidth=2)

    elif isinstance(polygon, MultiPolygon):
        for poly in polygon.geoms:
            x, y = poly.exterior.xy
            ax.plot(x, y, color="red", linewidth=2)

    # ---- fishnet points ----
    fishnet_gdf.plot(ax=ax, color="blue", markersize=5)

    ax.set_title(title)
    ax.set_axis_off()

    path = os.path.join(STATIC, "preview.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()

    return "/static/preview.png"


# ================= PROCESS =================
def build_polygon(file_path):

    if file_path.endswith(".zip"):
        return load_shapefile_from_zip(file_path)

    df = pd.read_excel(file_path)
    coords = list(zip(df["X"], df["Y"]))
    coords.append(coords[0])
    return Polygon(coords)


# ================= ROUTES =================
@app.route("/")
def home():
    return render_template("index.html")


# ================= PREVIEW (IMPORTANT FIX) =================
@app.route("/preview", methods=["POST"])
def preview():

    file = request.files["file"]
    path = os.path.join(UPLOAD, file.filename)
    file.save(path)

    try:

        polygon = build_polygon(path)

        # SAME LOGIC FOR BOTH ZIP + EXCEL
        fishnet = fishnet_clip(
            polygon,
            100, 100,
            10, 10,
            "EPSG:32645"
        )

        fishnet["geometry"] = fishnet.centroid
        fishnet = fishnet[fishnet.within(polygon)]

        img = make_preview(polygon, fishnet, "Preview")

        return {"image": img}

    except Exception as e:
        return {"error": str(e)}


# ================= FINAL PROCESS =================
@app.route("/upload", methods=["POST"])
def upload():

    file = request.files["file"]
    mode = request.form["mode"]
    zone = request.form["zone"]

    cell_width = float(request.form.get("cell_width", 100))
    cell_height = float(request.form.get("cell_height", 100))
    rows = int(request.form.get("rows", 10))
    cols = int(request.form.get("cols", 10))

    path = os.path.join(UPLOAD, file.filename)
    file.save(path)

    try:

        polygon = build_polygon(path)

        fishnet = fishnet_clip(
            polygon,
            cell_width,
            cell_height,
            rows,
            cols,
            get_crs(zone)
        )

        fishnet["geometry"] = fishnet.centroid
        fishnet = fishnet[fishnet.within(polygon)]

        fishnet["Plot_No"] = range(1, len(fishnet) + 1)

        # output
        base = os.path.splitext(file.filename)[0]
        out_dir = os.path.join(OUTPUT, base)
        os.makedirs(out_dir, exist_ok=True)

        shp_path = os.path.join(out_dir, "sample.shp")
        xlsx_path = os.path.join(out_dir, "sample.xlsx")

        fishnet.to_file(shp_path)
        fishnet.drop(columns="geometry").to_excel(xlsx_path, index=False)

        # zip output
        zip_name = f"{base}.zip"
        zip_path = os.path.join(OUTPUT, zip_name)

        with zipfile.ZipFile(zip_path, "w") as z:
            for f in os.listdir(out_dir):
                z.write(os.path.join(out_dir, f), arcname=f)

        return render_template(
            "index.html",
            download_file=zip_name,
            stats={
                "plot_count": len(fishnet),
                "plot_size": f"{cell_width} × {cell_height}",
                "crs": f"UTM {zone}N"
            }
        )

    except Exception as e:
        return render_template("index.html", error=str(e))


# ================= DOWNLOAD =================
@app.route("/download/<filename>")
def download(filename):
    return send_file(os.path.join(OUTPUT, filename), as_attachment=True)


if __name__ == "__main__":
    app.run(debug=True)
