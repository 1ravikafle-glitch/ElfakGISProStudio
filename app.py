import os
import uuid
import zipfile
import shutil
import json
import re
import pandas as pd
import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from flask import Flask, request, jsonify, send_file, send_from_directory, render_template
from shapely.geometry import Polygon, Point, LineString

app = Flask(__name__)

UPLOAD = "uploads"
OUTPUT = "outputs"

os.makedirs(UPLOAD, exist_ok=True)
os.makedirs(OUTPUT, exist_ok=True)

# ================= CORS =================
@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type,Authorization"
    response.headers["Access-Control-Allow-Methods"] = "GET,POST,OPTIONS"
    return response

# ================= UTIL =================
def read_input(file):
    name = file.filename.lower()
    if name.endswith(".csv"):
        return pd.read_csv(file, encoding="utf-8-sig")
    elif name.endswith((".xlsx", ".xls")):
        return pd.read_excel(file)
    else:
        raise ValueError("Only CSV/Excel supported")


def get_crs(zone):
    return f"EPSG:326{zone}"


def normalize_order(df):
    for c in df.columns:
        if c.lower() in ["sn", "s.n", "order"]:
            df = df.rename(columns={c: "Order"})
    return df


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


# ================= GROUP C (UPDATED MULTI-MODE) =================
def group_c(file, crs, w, h, rows, cols, out, mode, mapping=None):

    import pandas as pd
    from shapely.ops import unary_union

    polygons = []
    df = None

    # ================= LOAD INPUT =================
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

        if gdf is None:
            raise ValueError("No shapefile found in ZIP")

        # detect structure
        has_segment = ("Forest" in gdf.columns and "Compartment" in gdf.columns)

        if mode == "B" and has_segment:
            for (f, c), g in gdf.groupby(["Forest", "Compartment"]):
                polygons.append(g.unary_union)
        else:
            polygons = [gdf.unary_union]

    else:
        df = read_input(file)
        df = normalize_order(df).sort_values("Order")

        # ================= SINGLE MODE =================
        if mode == "A":

            x_col = mapping.get("X", "X") if mapping else "X"
            y_col = mapping.get("Y", "Y") if mapping else "Y"

            coords = list(zip(df[x_col], df[y_col]))
            coords.append(coords[0])

            polygons = [Polygon(coords)]

        # ================= SEGMENTED MODE =================
        else:

            x_col = mapping.get("X", "X") if mapping else "X"
            y_col = mapping.get("Y", "Y") if mapping else "Y"
            f_col = mapping.get("Forest", "Forest") if mapping else "Forest"
            c_col = mapping.get("Compartment", "Compartment") if mapping else "Compartment"

            if "Forest" in df.columns and "Compartment" in df.columns:
                for (f, c), g in df.groupby([f_col, c_col]):
                    g = g.sort_values("Order")
                    coords = list(zip(g[x_col], g[y_col]))
                    coords.append(coords[0])
                    polygons.append(Polygon(coords))
            else:
                raise ValueError("Segmented mode requires Forest & Compartment columns")

    # ================= UNIFIED PROCESS =================
    poly_gdf = gpd.GeoDataFrame([{"geometry": p} for p in polygons], crs=crs)
    union = poly_gdf.unary_union

    line_gdf = gpd.GeoDataFrame(
        [{"geometry": LineString(p.exterior.coords)} for p in polygons],
        crs=crs
    )

    minx, miny, _, _ = union.bounds

    pts = []
    sn = 1

    for r in range(rows):
        for c in range(cols):
            x = minx + c * w
            y = miny + r * h
            center = Point(x + w / 2, y + h / 2)

            if union.contains(center):
                pts.append({
                    "SN": sn,
                    "X": center.x,
                    "Y": center.y,
                    "geometry": center
                })
                sn += 1

    pts_gdf = gpd.GeoDataFrame(pts, crs=crs)

    # ================= OUTPUT =================
    poly_gdf.to_file(os.path.join(out, "boundary_polygons.shp"))
    line_gdf.to_file(os.path.join(out, "boundary_lines.shp"))
    pts_gdf.to_file(os.path.join(out, "sampleplot.shp"))

    pd.DataFrame(pts)[["SN", "X", "Y"]].to_excel(
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


# ================= UPLOAD ROUTE =================
@app.route("/upload", methods=["POST"])
def upload():
    file = request.files["file"]
    mode = request.form["mode"]
    zone = request.form["zone"]

    mapping = json.loads(request.form.get("mapping", "{}"))

    w = float(request.form.get("w", 50))
    h = float(request.form.get("h", 50))
    rows = int(request.form.get("rows", 10))
    cols = int(request.form.get("cols", 10))
    forest = request.form.get("forest", "FOREST")

    run_id = str(uuid.uuid4())
    out = os.path.join(OUTPUT, run_id)
    os.makedirs(out, exist_ok=True)

    crs = get_crs(zone)

    if mode == "A":
        df = read_input(file)
        poly, line, pts = group_a(df, forest, crs, out, mapping)

    elif mode == "B":
        df = read_input(file)
        poly, line, pts = group_b(df, crs, out, mapping)

    elif mode == "C":
        poly, line, pts = group_c(file, crs, w, h, rows, cols, out, mode, mapping)

    else:
        df = read_input(file)
        poly, line, pts = group_d(df, crs, out)

    preview_path = os.path.join(out, "output.png")
    preview(poly, line, pts, preview_path, "yellow", "black", "red")

    return jsonify({
        "run_id": run_id,
        "download": f"/download/{run_id}"
    })


# ================= DOWNLOAD =================
def zip_folder(folder):
    return shutil.make_archive(folder, "zip", folder)


@app.route("/download/<run_id>")
def download(run_id):
    folder = os.path.join(OUTPUT, run_id)

    if not os.path.exists(folder):
        return jsonify({"error": "Run not found"}), 404

    zip_path = zip_folder(folder)

    return send_file(zip_path, as_attachment=True)


# ================= HOME =================
@app.route("/")
def home():
    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)
