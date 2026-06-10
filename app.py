import os
import zipfile
import tempfile
import pandas as pd
import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from flask import Flask, render_template, request, jsonify, send_file
from shapely.geometry import Point, LineString, Polygon, box, MultiPolygon

app = Flask(__name__)

UPLOAD = "uploads"
STATIC = "static"
OUTPUT = "output"

os.makedirs(UPLOAD, exist_ok=True)
os.makedirs(STATIC, exist_ok=True)
os.makedirs(OUTPUT, exist_ok=True)

# =====================================================
# CRS
# =====================================================
def get_crs(zone):
    return "EPSG:32644" if str(zone) == "44" else "EPSG:32645"


# =====================================================
# SAFE GEOMETRY
# =====================================================
def normalize_geom(geom):
    if geom is None:
        return []
    if isinstance(geom, Polygon):
        return [geom]
    if isinstance(geom, MultiPolygon):
        return [g for g in geom.geoms if g.is_valid]
    return []


# =====================================================
# LOAD SHAPEFILE ZIP
# =====================================================
def load_shapefile(zip_path):
    temp_dir = tempfile.mkdtemp()

    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(temp_dir)

    shp = None
    for f in os.listdir(temp_dir):
        if f.endswith(".shp"):
            shp = os.path.join(temp_dir, f)
            break

    if shp is None:
        return None

    gdf = gpd.read_file(shp)
    return gdf.geometry.unary_union


# =====================================================
# BUILD POINTS
# =====================================================
def build_points(df):
    df["X"] = pd.to_numeric(df["X"], errors="coerce")
    df["Y"] = pd.to_numeric(df["Y"], errors="coerce")
    df = df.dropna(subset=["X", "Y"])
    return df, [Point(xy) for xy in zip(df["X"], df["Y"])]


# =====================================================
# BUILD POLYGON
# =====================================================
def build_polygon(coords):
    if len(coords) < 3:
        return None

    if coords[0] != coords[-1]:
        coords.append(coords[0])

    poly = Polygon(coords).buffer(0)
    if not poly.is_valid:
        return None

    line = LineString(coords)
    return poly, line


# =====================================================
# SAMPLE GRID
# =====================================================
def build_sample(poly, w, h, rows, cols):
    if poly is None or poly.is_empty:
        return []

    minx, miny, maxx, maxy = poly.bounds
    samples = []

    for i in range(cols):
        for j in range(rows):

            x = minx + i * w
            y = miny + j * h

            cell = box(x, y, x + w, y + h)

            if not cell.intersects(poly):
                continue

            clipped = cell.intersection(poly)
            if not clipped.is_empty:
                samples.append(clipped.centroid)

    return samples


# =====================================================
# SAVE MAP (PROFESSIONAL STYLE)
# =====================================================
def save_map(fig, name):
    path = os.path.join(STATIC, f"{name}.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return f"/static/{name}.png"


# =====================================================
# FOREST EXPORT (NEW FEATURE FROM TKINTER)
# =====================================================
def export_forest(df, crs, forest_name):

    forest_folder = os.path.join(OUTPUT, forest_name)
    os.makedirs(forest_folder, exist_ok=True)

    all_points = []
    all_lines = []
    all_polygons = []

    for comp, g in df.groupby("Compartment"):

        g = g.sort_values("Order")

        coords = list(zip(g["X"], g["Y"]))
        poly_line = build_polygon(coords)

        if not poly_line:
            continue

        poly, line = poly_line

        all_polygons.append({
            "Forest": forest_name,
            "Comp": comp,
            "Area_Ha": poly.area / 10000,
            "geometry": poly
        })

        all_lines.append({
            "Forest": forest_name,
            "Comp": comp,
            "geometry": line
        })

        for _, r in g.iterrows():
            all_points.append({
                "Forest": forest_name,
                "Comp": comp,
                "geometry": Point(r["X"], r["Y"])
            })

    if not all_polygons:
        return None

    gdf_p = gpd.GeoDataFrame(all_points, crs=crs)
    gdf_l = gpd.GeoDataFrame(all_lines, crs=crs)
    gdf_poly = gpd.GeoDataFrame(all_polygons, crs=crs)

    p1 = os.path.join(forest_folder, "points.shp")
    p2 = os.path.join(forest_folder, "lines.shp")
    p3 = os.path.join(forest_folder, "polygons.shp")

    gdf_p.to_file(p1)
    gdf_l.to_file(p2)
    gdf_poly.to_file(p3)

    # ZIP
    zip_path = os.path.join(OUTPUT, f"{forest_name}.zip")
    with zipfile.ZipFile(zip_path, "w") as z:
        for f in os.listdir(forest_folder):
            z.write(os.path.join(forest_folder, f), arcname=f)

    return zip_path


# =====================================================
# ROUTE
# =====================================================
@app.route("/preview", methods=["POST"])
def preview():

    file = request.files["file"]
    mode = request.form.get("mode")
    zone = request.form.get("zone", "45")

    path = os.path.join(UPLOAD, file.filename)
    file.save(path)

    crs = get_crs(zone)

    try:

        # ================= ZIP =================
        if path.endswith(".zip"):
            geom = load_shapefile(path)
            geom_list = normalize_geom(geom)

            if not geom_list:
                return jsonify({"error": "Invalid shapefile"})

            poly = geom_list[0]

        # ================= EXCEL =================
        else:
            df = pd.read_excel(path)

        # ================= COMPARTMENT =================
        if mode == "compartment":

            if df is None:
                return jsonify({"error": "Excel required"})

            forest_name = "FOREST"
            zip_file = export_forest(df, crs, forest_name)

            if not zip_file:
                return jsonify({"error": "No valid geometry"})

            return jsonify({"download": f"/download/{os.path.basename(zip_file)}"})

        # ================= SAMPLE =================
        if mode == "sample":

            base_poly = poly if "poly" in locals() else build_polygon(list(zip(df["X"], df["Y"])))[0]

            w = float(request.form.get("cell_width", 100))
            h = float(request.form.get("cell_height", 100))

            samples = build_sample(base_poly, w, h, 10, 10)

            fig, ax = plt.subplots()

            ax.plot(*base_poly.exterior.xy, "blue")
            for s in samples:
                ax.scatter(s.x, s.y, color="red")

            return jsonify({"image": save_map(fig, "sample")})

        return jsonify({"error": "Invalid mode"})

    except Exception as e:
        return jsonify({"error": str(e)})


# =====================================================
# DOWNLOAD ROUTE (NEW)
# =====================================================
@app.route("/download/<filename>")
def download(filename):
    return send_file(os.path.join(OUTPUT, filename), as_attachment=True)


if __name__ == "__main__":
    app.run(debug=True)
