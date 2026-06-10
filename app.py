import os
import uuid
import zipfile
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt

from flask import (
    Flask,
    render_template,
    request,
    send_file,
    jsonify,
    send_from_directory
)
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


# ================= FILE SERVE =================
@app.route("/outputs/<run_id>/<filename>")
def outputs(run_id, filename):
    return send_from_directory(
        os.path.join(OUTPUT, run_id),
        filename
    )


# ================= ZIP HELP =================
def extract_zip(file, folder):
    zip_path = os.path.join(folder, "input.zip")
    file.save(zip_path)

    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(folder)


def read_shp(folder):
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

    return (
        gpd.GeoDataFrame(polys, crs=crs),
        gpd.GeoDataFrame(lines, crs=crs),
        gpd.GeoDataFrame(pts, crs=crs)
    )


# ================= GROUP C (FIXED) =================
def group_c(file, crs, w, h, rows, cols, out):

    if file.filename.lower().endswith(".zip"):
        folder = os.path.join(UPLOAD, str(uuid.uuid4()))
        os.makedirs(folder, exist_ok=True)

        extract_zip(file, folder)
        gdf = read_shp(folder)

        if gdf is None:
            raise Exception("No shapefile found in ZIP")

        poly = gdf.unary_union

        if poly.geom_type == "MultiPolygon":
            poly = max(poly.geoms, key=lambda p: p.area)

        elif poly.geom_type == "GeometryCollection":
            polys = [g for g in poly.geoms if g.geom_type == "Polygon"]
            if not polys:
                raise Exception("No polygon found")
            poly = max(polys, key=lambda p: p.area)

    else:
        df = pd.read_excel(file)
        coords = list(zip(df["X"], df["Y"]))
        coords.append(coords[0])
        poly = Polygon(coords)

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

    gdf_pts = gpd.GeoDataFrame(
        {"SN": range(1, len(inside) + 1)},
        geometry=inside,
        crs=crs
    )

    gdf_pts["X"] = gdf_pts.geometry.x
    gdf_pts["Y"] = gdf_pts.geometry.y

    poly_gdf = gpd.GeoDataFrame([{"geometry": poly}], crs=crs)
    line_gdf = gpd.GeoDataFrame([{"geometry": line}], crs=crs)

    poly_gdf.to_file(os.path.join(out, "boundary.shp"))
    line_gdf.to_file(os.path.join(out, "boundary_line.shp"))
    gdf_pts.to_file(os.path.join(out, "sampleplot.shp"))

    return poly_gdf, line_gdf, gdf_pts


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

    return (
        gpd.GeoDataFrame(polys, crs=crs),
        gpd.GeoDataFrame(lines, crs=crs),
        gpd.GeoDataFrame(pts, crs=crs)
    )


# ================= PREVIEW =================
def preview(poly_gdf, line_gdf, pts_gdf, path, pc, lc, ptc):
    fig, ax = plt.subplots()

    poly_gdf.plot(ax=ax, facecolor="none", edgecolor=pc)
    line_gdf.plot(ax=ax, color=lc, linewidth=2)
    pts_gdf.plot(ax=ax, color=ptc, markersize=8)

    plt.axis("off")
    plt.savefig(path, dpi=220, bbox_inches="tight")
    plt.close()


# ================= UPLOAD =================
@app.route("/upload", methods=["POST"])
def upload():

    file = request.files["file"]
    mode = request.form["mode"]
    zone = request.form["zone"]

    w = float(request.form.get("w", 50))
    h = float(request.form.get("h", 50))
    rows = int(request.form.get("rows", 10))
    cols = int(request.form.get("cols", 10))
    forest = request.form.get("forest", "FOREST")

    run_id = str(uuid.uuid4())
    out = os.path.join(OUTPUT, run_id)
    os.makedirs(out, exist_ok=True)

    crs = get_crs(zone)
    preview_path = os.path.join(out, "output.png")

    # PROCESS
    if mode == "A":
        poly, line, pts = group_a(pd.read_excel(file), forest, crs, out)

    elif mode == "B":
        poly, line, pts = group_b(pd.read_excel(file), crs, out)

    elif mode == "C":
        poly, line, pts = group_c(file, crs, w, h, rows, cols, out)

    else:
        poly, line, pts = group_d(pd.read_excel(file), crs, out)

    # SAFE PREVIEW
    try:
        preview(poly, line, pts, preview_path, "red", "black", "yellow")
    except Exception as e:
        print("Preview error:", e)

    zip_buffer = zip_folder(out)

    return jsonify({
        "run_id": run_id,
        "download": "/download/" + run_id
    })


# ================= DOWNLOAD =================
@app.route("/download/<run_id>")
def download(run_id):
    folder = os.path.join(OUTPUT, run_id)
    zip_buffer = zip_folder(folder)

    return send_file(
        zip_buffer,
        mimetype="application/zip",
        as_attachment=True,
        download_name="output.zip"
    )


# ================= HOME =================
@app.route("/")
def home():
    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)
