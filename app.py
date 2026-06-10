import os
import uuid
import zipfile
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt

from flask import Flask, render_template, request, send_file, send_from_directory, jsonify
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


# ================= SERVE OUTPUT FILES =================
@app.route("/outputs/<run_id>/<path:filename>")
def outputs(run_id, filename):
    return send_from_directory(os.path.join(OUTPUT, run_id), filename)


# ================= ZIP =================
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
                    "geometry": Point(r["X"], r["Y"])
                })

    return (
        gpd.GeoDataFrame(polys, crs=crs),
        gpd.GeoDataFrame(lines, crs=crs),
        gpd.GeoDataFrame(pts, crs=crs)
    )


# ================= GROUP C =================
def group_c(df, crs, w, h, rows, cols, out):

    poly = Polygon(list(zip(df["X"], df["Y"])))
    line = LineString(list(zip(df["X"], df["Y"])))

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

    gdf = gpd.GeoDataFrame({
        "SN": range(1, len(inside) + 1)
    }, geometry=inside, crs=crs)

    poly_gdf = gpd.GeoDataFrame([{"geometry": poly}], crs=crs)
    line_gdf = gpd.GeoDataFrame([{"geometry": line}], crs=crs)

    gdf.to_file(os.path.join(out, "sample.shp"))
    poly_gdf.to_file(os.path.join(out, "boundary.shp"))
    line_gdf.to_file(os.path.join(out, "boundary_line.shp"))

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
            "geometry": poly
        })

        lines.append({
            "Forest": f,
            "geometry": line
        })

        for _, r in g.iterrows():
            pts.append({
                "Forest": f,
                "geometry": Point(r["X"], r["Y"])
            })

    return (
        gpd.GeoDataFrame(polys, crs=crs),
        gpd.GeoDataFrame(lines, crs=crs),
        gpd.GeoDataFrame(pts, crs=crs)
    )


# ================= PREVIEW =================
def preview(poly_gdf, line_gdf, pts_gdf, path):

    fig, ax = plt.subplots()

    poly_gdf.plot(ax=ax, facecolor="none", edgecolor="red")
    line_gdf.plot(ax=ax, color="black", linewidth=2)
    pts_gdf.plot(ax=ax, color="yellow", markersize=8)

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

    df = pd.read_excel(file)
    crs = get_crs(zone)

    preview_path = os.path.join(out, f"{mode}_preview.png")

    if mode == "A":
        poly, line, pts = group_a(df, forest, crs, out)

    elif mode == "B":
        poly, line, pts = group_b(df, crs, out)

    elif mode == "C":
        poly, line, pts = group_c(df, crs, w, h, rows, cols, out)

    else:
        poly, line, pts = group_d(df, crs, out)

    preview(poly, line, pts, preview_path)

    zip_buffer = zip_folder(out)

    return jsonify({
        "zip": f"/download/{run_id}",
        "preview": f"/outputs/{run_id}/{mode}_preview.png"
    })


@app.route("/download/<run_id>")
def download(run_id):
    folder = os.path.join(OUTPUT, run_id)
    return send_file(
        zip_folder(folder),
        mimetype="application/zip",
        as_attachment=True,
        download_name="output.zip"
    )


@app.route("/")
def home():
    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)
