import os
import zipfile
import tempfile
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt

from flask import Flask, render_template, request, send_file
from shapely.geometry import Polygon, Point

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

    # ✅ Direct polygon (ArcGIS compatible)
    polygon = gdf.geometry.unary_union

    return polygon


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

            # KEEP FULL GRID (ArcGIS behavior)
            cells.append(cell)

    return gpd.GeoDataFrame(geometry=cells, crs=crs)


def make_map(gdf, polygon, name, filename):

    fig, ax = plt.subplots(figsize=(10, 6))

    from shapely.geometry import Polygon, MultiPolygon

    # --- polygon boundary (thin red line) ---
    if isinstance(polygon, Polygon):
        x, y = polygon.exterior.xy
        ax.plot(x, y, color="red", linewidth=1)

    elif isinstance(polygon, MultiPolygon):
        for poly in polygon.geoms:
            x, y = poly.exterior.xy
            ax.plot(x, y, color="red", linewidth=1)

    # --- fishnet points ---
    gdf.plot(ax=ax, color="blue", markersize=5, alpha=0.7)

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
def process(file_path, mode, zone, cell_width, cell_height, rows, cols):

    crs = get_crs(zone)
    base = os.path.splitext(os.path.basename(file_path))[0]

    out_dir = os.path.join(OUTPUT, base)
    os.makedirs(out_dir, exist_ok=True)

    map_images = {}
    total_plots = 0

    # ================= SAMPLE =================
    if mode == "sample":

        # ---------- INPUT POLYGON ----------
        if file_path.endswith(".zip"):
            polygon = load_shapefile_from_zip(file_path)

        else:
            df = pd.read_excel(file_path)

            coords = list(zip(df["X"], df["Y"]))
            coords.append(coords[0])
            polygon = Polygon(coords)

        # ================= FISHNET =================
        fishnet = fishnet_clip(
            polygon,
            float(cell_width),
            float(cell_height),
            int(rows),
            int(cols),
            crs
        )

        # ================= ARC GIS STYLE FILTER =================
        fishnet["geometry"] = fishnet.centroid

        # ✔ KEEP ONLY CENTROIDS INSIDE POLYGON
        fishnet = fishnet[fishnet.within(polygon)].reset_index(drop=True)

        fishnet["Plot_No"] = range(1, len(fishnet) + 1)

        total_plots = len(fishnet)

        # ================= OUTPUT =================
        shp_path = os.path.join(out_dir, "sample.shp")
        xlsx_path = os.path.join(out_dir, "sample.xlsx")

        fishnet.to_file(shp_path)
        export_points(fishnet, xlsx_path)

        map_images["sample"] = make_map(fishnet, polygon, "Sample Plot", "sample_plot")

    # ================= ZIP =================
    zip_name = f"{base}.zip"
    zip_path = os.path.join(OUTPUT, zip_name)

    with zipfile.ZipFile(zip_path, "w") as z:
        for root, _, files in os.walk(out_dir):
            for f in files:
                z.write(os.path.join(root, f), arcname=f)

    stats = {
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

    cell_width = request.form.get("cell_width", 100)
    cell_height = request.form.get("cell_height", 100)
    rows = request.form.get("rows", 10)
    cols = request.form.get("cols", 10)

    path = os.path.join(UPLOAD, file.filename)
    file.save(path)

    try:
        zip_file, map_images, stats = process(
            path,
            mode,
            zone,
            cell_width,
            cell_height,
            rows,
            cols
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
