import os
import zipfile
import tempfile
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt

from flask import Flask, render_template, request, send_file, jsonify
from shapely.geometry import Polygon, Point, LineString, MultiPolygon

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


# ================= LOAD ZIP SHAPEFILE =================
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
        raise Exception("No .shp found in ZIP")

    gdf = gpd.read_file(shp_file)
    return gdf.geometry.unary_union


# ================= BUILD POLYGON (WHOLE) =================
def build_polygon_excel(df):
    coords = list(zip(df["X"], df["Y"]))
    coords.append(coords[0])
    return Polygon(coords)


# ================= SEGMENTED POLYGONS =================
def build_segmented_polygons(df):
    """
    Uses Compartment grouping:
    C1 -> rows 1-7
    C2 -> rows 8-15 etc.
    """

    polygons = {}

    for comp, group in df.groupby("Compartment"):
        group = group.sort_values("Order")

        coords = list(zip(group["X"], group["Y"]))
        coords.append(coords[0])

        polygons[comp] = Polygon(coords)

    return polygons


# ================= GEOMETRY EXPORT (POINT/LINE/POLYGON) =================
def export_layers(gdf, out_dir, name):
    point_gdf = gdf.copy()
    line_gdf = gdf.copy()
    poly_gdf = gdf.copy()

    point_gdf["geometry"] = point_gdf.centroid
    line_gdf["geometry"] = LineString(list(gdf.geometry.iloc[0].coords)) if hasattr(gdf.geometry.iloc[0], "coords") else None
    poly_gdf = gdf.copy()

    point_path = os.path.join(out_dir, f"{name}_point.shp")
    line_path = os.path.join(out_dir, f"{name}_line.shp")
    poly_path = os.path.join(out_dir, f"{name}_polygon.shp")

    point_gdf.to_file(point_path)
    poly_gdf.to_file(poly_path)

    if line_gdf.geometry.iloc[0] is not None:
        line_gdf.to_file(line_path)

    return point_path, line_path, poly_path


# ================= PREVIEW =================
def make_preview(geom, title):
    fig, ax = plt.subplots(figsize=(7, 6))

    if isinstance(geom, Polygon):
        x, y = geom.exterior.xy
        ax.plot(x, y, "r")

    elif isinstance(geom, MultiPolygon):
        for g in geom.geoms:
            x, y = g.exterior.xy
            ax.plot(x, y, "r")

    ax.set_title(title)
    ax.set_axis_off()

    path = os.path.join(STATIC, "preview.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()

    return "/static/preview.png"


# ================= ROUTE =================
@app.route("/")
def home():
    return render_template("index.html")


# ================= PREVIEW API (MODE CONTROLLED) =================
@app.route("/preview", methods=["POST"])
def preview():
    file = request.files["file"]
    mode = request.form.get("mode", "boundary")

    path = os.path.join(UPLOAD, file.filename)
    file.save(path)

    try:
        if path.endswith(".zip"):
            geom = load_shapefile_from_zip(path)
            return jsonify({"image": make_preview(geom, "Shapefile Boundary Preview")})

        df = pd.read_excel(path)

        # ================= WHOLE FOREST =================
        if mode == "boundary":
            poly = build_polygon_excel(df)
            return jsonify({"image": make_preview(poly, "Whole Forest Boundary")})

        # ================= SEGMENTED =================
        if mode == "compartment":
            comps = build_segmented_polygons(df)
            union = gpd.GeoSeries(list(comps.values())).unary_union
            return jsonify({"image": make_preview(union, "Segmented Forest Boundary")})

        # ================= SAMPLE PLOT =================
        if mode == "sample":
            poly = build_polygon_excel(df)
            return jsonify({"image": make_preview(poly, "Sample Plot Preview")})

        return jsonify({"error": "Invalid mode"})

    except Exception as e:
        return jsonify({"error": str(e)})


# ================= MAIN PROCESS =================
@app.route("/upload", methods=["POST"])
def upload():
    file = request.files["file"]
    mode = request.form["mode"]
    zone = request.form["zone"]

    path = os.path.join(UPLOAD, file.filename)
    file.save(path)

    try:

        # ================= ZIP INPUT =================
        if path.endswith(".zip"):
            geom = load_shapefile_from_zip(path)

            gdf = gpd.GeoDataFrame(geometry=[geom], crs=get_crs(zone))

        # ================= EXCEL INPUT =================
        else:
            df = pd.read_excel(path)

            if mode == "compartment":
                comps = build_segmented_polygons(df)
                gdf = gpd.GeoDataFrame(geometry=list(comps.values()), crs=get_crs(zone))

            else:
                poly = build_polygon_excel(df)
                gdf = gpd.GeoDataFrame(geometry=[poly], crs=get_crs(zone))

        # ================= DERIVED LAYERS =================
        points = gdf.copy()
        points["geometry"] = points.centroid

        lines = gdf.copy()
        lines["geometry"] = gdf.geometry.apply(lambda g: LineString(g.exterior.coords))

        polygons = gdf.copy()

        # ================= OUTPUT =================
        base = os.path.splitext(file.filename)[0]
        out_dir = os.path.join(OUTPUT, base)
        os.makedirs(out_dir, exist_ok=True)

        point_path = os.path.join(out_dir, "point.shp")
        line_path = os.path.join(out_dir, "line.shp")
        poly_path = os.path.join(out_dir, "polygon.shp")

        points.to_file(point_path)
        lines.to_file(line_path)
        polygons.to_file(poly_path)

        xlsx_path = os.path.join(out_dir, "output.xlsx")
        polygons.drop(columns="geometry").to_excel(xlsx_path, index=False)

        # ZIP
        zip_name = f"{base}.zip"
        zip_path = os.path.join(OUTPUT, zip_name)

        with zipfile.ZipFile(zip_path, "w") as z:
            for f in os.listdir(out_dir):
                z.write(os.path.join(out_dir, f), arcname=f)

        return render_template(
            "index.html",
            download_file=zip_name,
            stats={
                "plot_count": len(gdf),
                "crs": f"UTM {zone}N",
                "mode": mode
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
