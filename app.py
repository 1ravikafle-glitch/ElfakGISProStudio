import os
import uuid
import zipfile
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt

from flask import Flask, render_template, request, send_file, send_from_directory
from shapely.geometry import Polygon, Point, LineString
from io import BytesIO

app = Flask(__name__)

UPLOAD = "uploads"
OUTPUT = "outputs"

os.makedirs(UPLOAD, exist_ok=True)
os.makedirs(OUTPUT, exist_ok=True)


# ================= CRS =================
def get_crs(zone):
    return "EPSG:32644" if zone == "44" else "EPSG:32645"


# ================= ORDER =================
def normalize_order(df):
    for c in df.columns:
        if c.lower() in ["sn", "s.n", "order"]:
            df = df.rename(columns={c: "Order"})
    return df


# ================= SERVE OUTPUT =================
@app.route("/outputs/<path:filename>")
def outputs(filename):
    return send_from_directory(OUTPUT, filename)


# ================= ZIP EXTRACT =================
def extract_zip(file):
    folder = os.path.join(UPLOAD, str(uuid.uuid4()))
    os.makedirs(folder, exist_ok=True)

    zip_path = os.path.join(folder, "input.zip")
    file.save(zip_path)

    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(folder)

    return folder


def read_shp_from_zip(folder):
    for root, _, files in os.walk(folder):
        for f in files:
            if f.endswith(".shp"):
                return gpd.read_file(os.path.join(root, f))
    return None


# ================= ZIP OUTPUT =================
def zip_folder(folder):
    buf = BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for f in os.listdir(folder):
            fp = os.path.join(folder, f)
            if os.path.isfile(fp):
                z.write(fp, f)
    buf.seek(0)
    return buf


# ================= GROUP A =================
def group_a(df, forest, crs, out):
    df = normalize_order(df).sort_values("Order")

    coords = list(zip(df["X"], df["Y"]))
    coords.append(coords[0])

    poly = Polygon(coords)
    line = LineString(coords)

    poly_gdf = gpd.GeoDataFrame([{
        "Forest": forest,
        "Area": poly.area / 10000,
        "Perim": poly.length,
        "geometry": poly
    }], crs=crs)

    line_gdf = gpd.GeoDataFrame([{
        "Forest": forest,
        "geometry": line
    }], crs=crs)

    pts_gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df["X"], df["Y"]),
        crs=crs
    )

    poly_gdf.to_file(os.path.join(out, "polygon.shp"))
    line_gdf.to_file(os.path.join(out, "line.shp"))
    pts_gdf.to_file(os.path.join(out, "points.shp"))

    return poly_gdf, line_gdf, pts_gdf


# ================= GROUP B =================
def group_b(df, crs, out):
    df = normalize_order(df)

    polys, lines, pts = [], [], []

    for f, g in df.groupby("Forest"):
        for c, cg in g.groupby("Compartment"):

            cg = cg.sort_values("Order")
            coords = list(zip(cg["X"], cg["Y"]))
            coords.append(coords[0])

            poly = Polygon(coords)
            line = LineString(coords)

            polys.append({
                "Forest": f,
                "Compartment": c,
                "Area": poly.area / 10000,
                "Perim": poly.length,
                "geometry": poly
            })

            lines.append({
                "Forest": f,
                "Compartment": c,
                "geometry": line
            })

            for _, r in cg.iterrows():
                pts.append({
                    "Forest": f,
                    "Compartment": c,
                    "Order": r["Order"],
                    "geometry": Point(r["X"], r["Y"])
                })

    poly_gdf = gpd.GeoDataFrame(polys, crs=crs)
    line_gdf = gpd.GeoDataFrame(lines, crs=crs)
    pts_gdf = gpd.GeoDataFrame(pts, crs=crs)

    poly_gdf.to_file(os.path.join(out, "polygons.shp"))
    line_gdf.to_file(os.path.join(out, "lines.shp"))
    pts_gdf.to_file(os.path.join(out, "points.shp"))

    return poly_gdf, line_gdf, pts_gdf


# ================= GROUP C (FIXED ZIP SUPPORT) =================
def group_c(file, crs, w, h, rows, cols, out):

    # CASE 1: ZIP INPUT (SHAPEFILE)
    if file.filename.endswith(".zip"):
        folder = extract_zip(file)
        gdf = read_shp_from_zip(folder)

        if gdf is None:
            raise Exception("No shapefile found in ZIP")

        poly = gdf.unary_union

    # CASE 2: EXCEL INPUT
    else:
        df = pd.read_excel(file)
        poly = Polygon(list(zip(df["X"], df["Y"])))

    line = LineString(poly.exterior.coords)

    minx, miny, _, _ = poly.bounds

    pts = []
    for i in range(rows):
        for j in range(cols):
            x = minx + j * w
            y = miny + i * h

            cell = Polygon([
                (x, y),
                (x + w, y),
                (x + w, y + h),
                (x, y + h)
            ])

            pts.append(cell.centroid)

    inside = [p for p in pts if poly.contains(p)]

    gdf = gpd.GeoDataFrame({"SN": range(1, len(inside) + 1)}, geometry=inside, crs=crs)
    gdf["X"] = gdf.geometry.x
    gdf["Y"] = gdf.geometry.y

    poly_gdf = gpd.GeoDataFrame([{"geometry": poly}], crs=crs)
    line_gdf = gpd.GeoDataFrame([{"geometry": line}], crs=crs)

    poly_gdf.to_file(os.path.join(out, "boundary.shp"))
    line_gdf.to_file(os.path.join(out, "boundary_line.shp"))
    gdf.to_file(os.path.join(out, "sampleplot.shp"))

    return poly_gdf, line_gdf, gdf


# ================= GROUP D =================
def group_d(df, crs, out):
    df = normalize_order(df).sort_values("Order")

    polys, lines, pts = [], [], []

    for f, g in df.groupby("Forest"):

        coords = list(zip(g["X"], g["Y"]))
        coords.append(coords[0])

        poly = Polygon(coords)
        line = LineString(coords)

        polys.append({
            "Forest": f,
            "Area": poly.area / 10000,
            "Perim": poly.length,
            "geometry": poly
        })

        lines.append({
            "Forest": f,
            "geometry": line
        })

        for _, r in g.iterrows():
            pts.append({
                "Forest": f,
                "Order": r["Order"],
                "geometry": Point(r["X"], r["Y"])
            })

    poly_gdf = gpd.GeoDataFrame(polys, crs=crs)
    line_gdf = gpd.GeoDataFrame(lines, crs=crs)
    pts_gdf = gpd.GeoDataFrame(pts, crs=crs)

    poly_gdf.to_file(os.path.join(out, "polygons.shp"))
    line_gdf.to_file(os.path.join(out, "lines.shp"))
    pts_gdf.to_file(os.path.join(out, "points.shp"))

    return poly_gdf, line_gdf, pts_gdf


# ================= PREVIEW (ALWAYS SAVED) =================
def preview(poly_gdf, line_gdf, pts_gdf, path, pc, lc, ptc):

    fig, ax = plt.subplots()

    poly_gdf.plot(ax=ax, facecolor="none", edgecolor=pc)
    line_gdf.plot(ax=ax, color=lc, linewidth=2)
    pts_gdf.plot(ax=ax, color=ptc, markersize=8)

    plt.axis("off")
    plt.savefig(path, dpi=220, bbox_inches="tight")
    plt.close()


# ================= MAIN =================
@app.route("/upload", methods=["POST"])
def upload():

    file = request.files["file"]
    mode = request.form["mode"]
    zone = request.form["zone"]
    forest = request.form.get("forest", "FOREST")

    w = float(request.form.get("w", 50))
    h = float(request.form.get("h", 50))
    rows = int(request.form.get("rows", 10))
    cols = int(request.form.get("cols", 10))

    run_id = str(uuid.uuid4())
    out = os.path.join(OUTPUT, run_id)
    os.makedirs(out, exist_ok=True)

    crs = get_crs(zone)
    preview_path = os.path.join(out, f"{mode}_preview.png")

    if mode == "A":
        poly, line, pts = group_a(df := pd.read_excel(file), forest, crs, out)
        preview(poly, line, pts, preview_path, "red", "black", "red")

    elif mode == "B":
        df = pd.read_excel(file)
        poly, line, pts = group_b(df, crs, out)
        preview(poly, line, pts, preview_path, "blue", "orange", "green")

    elif mode == "C":
        poly, line, pts = group_c(file, crs, w, h, rows, cols, out)
        preview(poly, line, pts, preview_path, "red", "yellow", "yellow")

    else:
        df = pd.read_excel(file)
        poly, line, pts = group_d(df, crs, out)
        preview(poly, line, pts, preview_path, "green", "cyan", "cyan")

    zip_buffer = zip_folder(out)

    return send_file(zip_buffer, mimetype="application/zip", as_attachment=True, download_name="output.zip")


@app.route("/")
def home():
    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)
