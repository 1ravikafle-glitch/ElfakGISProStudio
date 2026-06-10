import os
import zipfile
import traceback
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt

from flask import Flask, render_template, request, send_file
from shapely.geometry import Point, LineString, Polygon

# ================= SETUP =================
app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "CF_OUTPUT"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

plt.rcParams["font.family"] = "Times New Roman"


# ================= CRS =================
def get_crs(zone):
    if zone == "44":
        return "EPSG:32644"
    return "EPSG:32645"


# ================= CLEAN OLD SHAPEFILES =================
def remove_old_shapefile(path):
    base = os.path.splitext(path)[0]
    for ext in [".shp", ".shx", ".dbf", ".prj", ".cpg"]:
        f = base + ext
        if os.path.exists(f):
            try:
                os.remove(f)
            except:
                pass


# ================= CORE PROCESS =================
def process_excel(file_path, crs_zone, map_title, legend_text):

    df = pd.read_excel(file_path)

    required_columns = ["Forest", "X", "Y", "Order"]
    for col in required_columns:
        if col not in df.columns:
            return None, f"Missing column: {col}"

    crs = get_crs(crs_zone)

    grouped = list(df.groupby("Forest"))

    zip_files = []

    for forest_name, group in grouped:

        safe_name = str(forest_name)
        for ch in '<>:"/\\|?*':
            safe_name = safe_name.replace(ch, "_")
        safe_name = safe_name.replace(" ", "_")

        group = group.sort_values(by="Order")
        group["X"] = pd.to_numeric(group["X"], errors="coerce")
        group["Y"] = pd.to_numeric(group["Y"], errors="coerce")
        group = group.dropna(subset=["X", "Y"])

        coords = list(zip(group["X"], group["Y"]))

        if len(coords) < 3:
            continue

        points = [Point(xy) for xy in coords]
        gdf_points = gpd.GeoDataFrame(group.copy(), geometry=points, crs=crs)

        if coords[0] != coords[-1]:
            coords.append(coords[0])

        line = LineString(coords)
        polygon = Polygon(coords).buffer(0)

        if not polygon.is_valid:
            continue

        area_ha = polygon.area / 10000

        gdf_line = gpd.GeoDataFrame(geometry=[line], crs=crs)

        gdf_polygon = gpd.GeoDataFrame(
            {"Forest": [forest_name], "Area_Ha": [round(area_ha, 4)]},
            geometry=[polygon],
            crs=crs
        )

        forest_folder = os.path.join(OUTPUT_FOLDER, safe_name)
        os.makedirs(forest_folder, exist_ok=True)

        points_path = os.path.join(forest_folder, f"{safe_name}_points.shp")
        line_path = os.path.join(forest_folder, f"{safe_name}_line.shp")
        poly_path = os.path.join(forest_folder, f"{safe_name}_polygon.shp")

        remove_old_shapefile(points_path)
        remove_old_shapefile(line_path)
        remove_old_shapefile(poly_path)

        gdf_points.to_file(points_path)
        gdf_line.to_file(line_path)
        gdf_polygon.to_file(poly_path)

        # ================= MAP =================
        fig, ax = plt.subplots(figsize=(10, 6))

        gdf_polygon.plot(ax=ax, color="lightgreen", edgecolor="darkgreen")
        gdf_line.plot(ax=ax, color="black")
        gdf_points.plot(ax=ax, color="red", markersize=20)

        ax.set_title(map_title)
        ax.set_axis_off()

        img_path = os.path.join(forest_folder, f"{safe_name}_map.jpg")
        plt.savefig(img_path, dpi=150, bbox_inches="tight")
        plt.close()

        # ================= ZIP =================
        zip_path = os.path.join(OUTPUT_FOLDER, f"{safe_name}.zip")

        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
            for f in os.listdir(forest_folder):
                zipf.write(os.path.join(forest_folder, f), arcname=f)

        zip_files.append(zip_path)

    return zip_files, None


# ================= ROUTES =================
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():

    try:
        file = request.files["file"]
        crs_zone = request.form["crs"]
        title = request.form["title"]
        legend = request.form["legend"]

        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(file_path)

        zips, error = process_excel(file_path, crs_zone, title, legend)

        if error:
            return error

        return send_file(zips[0], as_attachment=True)

    except Exception as e:
        traceback.print_exc()
        return str(e)


if __name__ == "__main__":
    app.run()