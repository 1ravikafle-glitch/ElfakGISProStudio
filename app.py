import os
import uuid
import zipfile
import shutil
import json
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
    return send_from_directory(os.path.join(OUTPUT, run_id), filename)


# ================= GET COLUMNS =================
@app.route("/get-columns", methods=["POST"])
def get_columns():
    file = request.files["file"]

    filter_forest = request.form.get("filterForest")
    filter_comp = request.form.get("filterCompartment")

    df = pd.read_excel(file)

    if filter_forest and "Forest" in df.columns:
        df = df[df["Forest"] == filter_forest]

    if filter_comp and "Compartment" in df.columns:
        df = df[df["Compartment"] == filter_comp]

    return jsonify(list(df.columns))

# ================= GROUP A =================
def group_a(df, forest, crs, out, mapping=None):
    df = normalize_order(df).sort_values("Order")

    x_col = mapping.get("X", "X") if mapping else "X"
    y_col = mapping.get("Y", "Y") if mapping else "Y"

    coords = list(zip(df[x_col], df[y_col]))
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
        geometry=gpd.points_from_xy(df[x_col], df[y_col]),
        crs=crs
    )

    poly_gdf.to_file(os.path.join(out, "polygon.shp"))
    line_gdf.to_file(os.path.join(out, "line.shp"))
    pts_gdf.to_file(os.path.join(out, "points.shp"))

    return poly_gdf, line_gdf, pts_gdf


# ================= GROUP B =================
def group_b(df, crs, out, mapping=None):
    df = normalize_order(df)

    x_col = mapping.get("X", "X") if mapping else "X"
    y_col = mapping.get("Y", "Y") if mapping else "Y"
    order_col = mapping.get("Order", "Order") if mapping else "Order"

    polys, lines, pts = [], [], []

    for f, g in df.groupby("Forest"):
        for c, cg in g.groupby("Compartment"):

            cg = cg.sort_values(order_col)

            coords = list(zip(cg[x_col], cg[y_col]))
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
                    "Order": r[order_col],
                    "geometry": Point(r[x_col], r[y_col])
                })

    poly_gdf = gpd.GeoDataFrame(polys, crs=crs)
    line_gdf = gpd.GeoDataFrame(lines, crs=crs)
    pts_gdf = gpd.GeoDataFrame(pts, crs=crs)

    poly_gdf.to_file(os.path.join(out, "polygons.shp"))
    line_gdf.to_file(os.path.join(out, "lines.shp"))
    pts_gdf.to_file(os.path.join(out, "points.shp"))

    return poly_gdf, line_gdf, pts_gdf


# ================= GROUP C =================
def group_c(file, crs, w, h, rows, cols, out):

    polygons = []

    if file.filename.lower().endswith(".zip"):
        folder = os.path.join(UPLOAD, str(uuid.uuid4()))
        os.makedirs(folder, exist_ok=True)

        zip_path = os.path.join(folder, "input.zip")
        file.save(zip_path)

        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(folder)

        gdf = None
        for root, _, files in os.walk(folder):
            for f in files:
                if f.endswith(".shp"):
                    gdf = gpd.read_file(os.path.join(root, f))
                    break

        geom = gdf.unary_union

        if geom.geom_type == "Polygon":
            polygons = [geom]
        elif geom.geom_type == "MultiPolygon":
            polygons = list(geom.geoms)
        else:
            polygons = [g for g in geom.geoms if g.geom_type == "Polygon"]

    else:
        df = pd.read_excel(file)
        if {"Forest", "Compartment"}.issubset(df.columns):
            for (f, c), g in df.groupby(["Forest", "Compartment"]):
                g = g.sort_values("Order")
                coords = list(zip(g["X"], g["Y"]))
                coords.append(coords[0])
                polygons.append(Polygon(coords))
        else:
            df = normalize_order(df).sort_values("Order")
            coords = list(zip(df["X"], df["Y"]))
            coords.append(coords[0])
            polygons.append(Polygon(coords))

    poly_gdf = gpd.GeoDataFrame([{"geometry": p} for p in polygons], crs=crs)
    poly_union = poly_gdf.unary_union

    line_gdf = gpd.GeoDataFrame(
        [{"geometry": LineString(p.exterior.coords)} for p in polygons],
        crs=crs
    )

    minx, miny, _, _ = poly_union.bounds

    inside_points = []
    sn = 1

    for r in range(rows):
        for c in range(cols):
            x = minx + c * w
            y = miny + r * h
            center = Point(x + w / 2, y + h / 2)

            if poly_union.contains(center):
                inside_points.append({
                    "SN": sn,
                    "X": center.x,
                    "Y": center.y,
                    "geometry": center
                })
                sn += 1

    pts_gdf = gpd.GeoDataFrame(inside_points, crs=crs)

    poly_gdf.to_file(os.path.join(out, "boundary_polygons.shp"))
    line_gdf.to_file(os.path.join(out, "boundary_lines.shp"))
    pts_gdf.to_file(os.path.join(out, "sampleplot.shp"))

    pd.DataFrame(inside_points)[["SN", "X", "Y"]].to_excel(
        os.path.join(out, "sampleplot.xlsx"),
        index=False
    )

    return poly_gdf, line_gdf, pts_gdf


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

    mapping_raw = request.form.get("mapping")
    mapping = json.loads(mapping_raw) if mapping_raw else None

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

    if mode == "A":
        df = pd.read_excel(file)
        poly, line, pts = group_a(df, forest, crs, out, mapping)

    elif mode == "B":
        df = pd.read_excel(file)
        poly, line, pts = group_b(df, crs, out, mapping)

    elif mode == "C":
        poly, line, pts = group_c(file, crs, w, h, rows, cols, out)

    else:
        df = pd.read_excel(file)
        poly, line, pts = group_d(df, crs, out)

    preview(poly, line, pts, preview_path, "yellow", "black", "red")

    return jsonify({
        "run_id": run_id,
        "download": "/download/" + run_id
    })
    
def zip_folder(folder_path):
    zip_path = folder_path + ".zip"
    shutil.make_archive(folder_path, 'zip', folder_path)
    return zip_path

@app.route("/download/<run_id>")
def download(run_id):
    folder = os.path.join(OUTPUT, run_id)

    zip_path = zip_folder(folder)

    return send_file(
        zip_path,
        mimetype="application/zip",
        as_attachment=True,
        download_name="output.zip"
    )


@app.route("/")
def home():
    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)
