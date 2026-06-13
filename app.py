import os
import uuid
import zipfile
import shutil
import json
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


def safe_col(df, mapping, key, fallback):
    """Resolve a column name using the user mapping first, then an
    exact fallback, then a case-insensitive fallback match."""
    if mapping and mapping.get(key) and mapping[key] in df.columns:
        return mapping[key]

    if fallback in df.columns:
        return fallback

    for c in df.columns:
        if c.lower() == fallback.lower():
            return c

    return None


# ================= GROUP A =================
def group_a(df, forest, crs, out, mapping=None):
    df = normalize_order(df)

    x_col = safe_col(df, mapping, "X", "X")
    y_col = safe_col(df, mapping, "Y", "Y")
    order_col = safe_col(df, mapping, "Order", "Order")

    if not x_col or not y_col:
        raise ValueError("Missing X/Y columns")

    if order_col:
        df = df.sort_values(order_col)

    coords = list(zip(df[x_col], df[y_col]))

    if len(coords) < 3:
        raise ValueError("Not enough points to build polygon")

    coords.append(coords[0])

    poly = Polygon(coords)
    line = LineString(coords)

    poly_gdf = gpd.GeoDataFrame([{
        "Forest": forest,
        "Area": poly.area / 10000,
        "Perim": poly.length,
        "geometry": poly
    }], crs=crs)

    line_gdf = gpd.GeoDataFrame([{"Forest": forest, "geometry": line}], crs=crs)

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

    x_col = safe_col(df, mapping, "X", "X")
    y_col = safe_col(df, mapping, "Y", "Y")
    order_col = safe_col(df, mapping, "Order", "Order")
    forest_col = safe_col(df, mapping, "Forest", "Forest")
    comp_col = safe_col(df, mapping, "Compartment", "Compartment")

    if not x_col or not y_col:
        raise ValueError("Missing X/Y columns")

    if not forest_col:
        raise ValueError("Missing Forest column")

    polys, lines, pts = [], [], []

    for f, g in df.groupby(forest_col):

        if order_col:
            g = g.sort_values(order_col)

        if comp_col:
            groups = g.groupby(comp_col)
        else:
            groups = [(None, g)]

        for c, cg in groups:

            coords = list(zip(cg[x_col], cg[y_col]))
            if len(coords) < 3:
                continue

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
                    "Order": r.get(order_col, None) if order_col else None,
                    "geometry": Point(r[x_col], r[y_col])
                })

    poly_gdf = gpd.GeoDataFrame(polys, crs=crs)
    line_gdf = gpd.GeoDataFrame(lines, crs=crs)
    pts_gdf = gpd.GeoDataFrame(pts, crs=crs)

    if not poly_gdf.empty:
        poly_gdf.to_file(os.path.join(out, "polygon.shp"))
    if not line_gdf.empty:
        line_gdf.to_file(os.path.join(out, "line.shp"))
    if not pts_gdf.empty:
        pts_gdf.to_file(os.path.join(out, "points.shp"))

    return poly_gdf, line_gdf, pts_gdf


# ================= GROUP C (MULTI-MODE) =================
def group_c(file, crs, w, h, rows, cols, out, mode, mapping=None):

    polygons = []

    # ================= ZIP INPUT =================
    if file.filename.lower().endswith(".zip"):

        folder = os.path.join(UPLOAD, str(uuid.uuid4()))
        os.makedirs(folder, exist_ok=True)

        zip_path = os.path.join(folder, "input.zip")
        file.save(zip_path)

        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(folder)

        shp_candidates = []
        for root, _, files in os.walk(folder):
            for f in files:
                if f.endswith(".shp"):
                    shp_candidates.append(os.path.join(root, f))

        if not shp_candidates:
            raise ValueError("No shapefile found in ZIP")

        shp_path = shp_candidates[0]
        target_shp = (mapping or {}).get("target_shp")

        if target_shp:
            target_name = os.path.basename(target_shp)
            for cand in shp_candidates:
                if os.path.basename(cand) == target_name:
                    shp_path = cand
                    break

        gdf = gpd.read_file(shp_path)

        if gdf.empty:
            raise ValueError("Empty shapefile")

        if gdf.crs is None:
            gdf = gdf.set_crs(crs)
        else:
            gdf = gdf.to_crs(crs)

        union_geom = gdf.unary_union

        if union_geom.geom_type == "Polygon":
            polygons = [union_geom]
        elif union_geom.geom_type == "MultiPolygon":
            polygons = list(union_geom.geoms)
        else:
            raise ValueError("Unsupported geometry type in shapefile")

    # ================= CSV/EXCEL INPUT =================
    else:
        df = read_input(file)
        df = normalize_order(df)

        x_col = safe_col(df, mapping, "X", "X")
        y_col = safe_col(df, mapping, "Y", "Y")
        order_col = safe_col(df, mapping, "Order", "Order")

        if not x_col or not y_col:
            raise ValueError("Missing X/Y columns")

        if mode == "A":

            coords = list(zip(df[x_col], df[y_col]))
            if len(coords) < 3:
                raise ValueError("Not enough points")

            coords.append(coords[0])
            polygons = [Polygon(coords)]

        else:

            forest_col = safe_col(df, mapping, "Forest", "Forest")
            comp_col = safe_col(df, mapping, "Compartment", "Compartment")

            if not forest_col:
                raise ValueError("Segmented mode requires a Forest column")

            group_keys = [forest_col, comp_col] if comp_col else [forest_col]

            for key, g in df.groupby(group_keys):

                if order_col:
                    g = g.sort_values(order_col)

                coords = list(zip(g[x_col], g[y_col]))
                if len(coords) < 3:
                    continue

                coords.append(coords[0])
                polygons.append(Polygon(coords))

    if not polygons:
        raise ValueError("No valid polygons could be built from the input")

    # ================= SAFE GEODATAFRAMES =================
    poly_gdf = gpd.GeoDataFrame([{"geometry": p} for p in polygons], crs=crs)

    line_gdf = gpd.GeoDataFrame(
        [{"geometry": LineString(p.exterior.coords)} for p in polygons],
        crs=crs
    )

    union = poly_gdf.unary_union
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

    poly_gdf.to_file(os.path.join(out, "boundary_polygons.shp"))
    line_gdf.to_file(os.path.join(out, "boundary_lines.shp"))

    if not pts_gdf.empty:
        pts_gdf.to_file(os.path.join(out, "sampleplot.shp"))
        pd.DataFrame(pts)[["SN", "X", "Y"]].to_excel(
            os.path.join(out, "sampleplot.xlsx"),
            index=False
        )

    return poly_gdf, line_gdf, pts_gdf


# ================= GROUP D =================
def group_d(df, crs, out, mapping=None):
    df = normalize_order(df)

    x_col = safe_col(df, mapping, "X", "X")
    y_col = safe_col(df, mapping, "Y", "Y")
    order_col = safe_col(df, mapping, "Order", "Order")
    forest_col = safe_col(df, mapping, "Forest", "Forest")

    if not x_col or not y_col:
        raise ValueError("Missing X/Y columns")

    if not forest_col:
        raise ValueError("Missing Forest column")

    polys, lines, pts = [], [], []

    for f, g in df.groupby(forest_col):

        if order_col:
            g = g.sort_values(order_col)

        coords = list(zip(g[x_col], g[y_col]))
        if len(coords) < 3:
            continue

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
                "Order": r.get(order_col, None) if order_col else None,
                "geometry": Point(r[x_col], r[y_col])
            })

    poly_gdf = gpd.GeoDataFrame(polys, crs=crs)
    line_gdf = gpd.GeoDataFrame(lines, crs=crs)
    pts_gdf = gpd.GeoDataFrame(pts, crs=crs)

    if not poly_gdf.empty:
        poly_gdf.to_file(os.path.join(out, "polygon.shp"))
    if not line_gdf.empty:
        line_gdf.to_file(os.path.join(out, "line.shp"))
    if not pts_gdf.empty:
        pts_gdf.to_file(os.path.join(out, "points.shp"))

    return poly_gdf, line_gdf, pts_gdf


# ================= PREVIEW =================
def preview(poly_gdf, line_gdf, pts_gdf, path, pc, lc, ptc):
    fig, ax = plt.subplots()

    if poly_gdf is not None and not poly_gdf.empty:
        poly_gdf.plot(ax=ax, facecolor="none", edgecolor=pc)

    if line_gdf is not None and not line_gdf.empty:
        line_gdf.plot(ax=ax, color=lc, linewidth=2)

    if pts_gdf is not None and not pts_gdf.empty:
        pts_gdf.plot(ax=ax, color=ptc, markersize=8)

    plt.axis("off")
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


# ================= UPLOAD ROUTE =================
@app.route("/upload", methods=["POST"])
def upload():
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files["file"]
        mode = request.form.get("mode", "A")
        # 'module' identifies which workspace tab (A/B/C/D) sent the request.
        # 'mode' is reused by Group C as its sub-mode (A = Simple Poly,
        # B = Segmented), so it alone can't disambiguate Group A vs Group C.
        module = request.form.get("module", mode)
        zone = request.form.get("zone", "44")

        try:
            mapping = json.loads(request.form.get("mapping", "{}"))
        except (TypeError, ValueError):
            mapping = {}

        w = float(request.form.get("w", 50))
        h = float(request.form.get("h", 50))
        rows = int(request.form.get("rows", 10))
        cols = int(request.form.get("cols", 10))
        forest = request.form.get("forest") or mapping.get("forest") or "FOREST"

        run_id = str(uuid.uuid4())
        out = os.path.join(OUTPUT, run_id)
        os.makedirs(out, exist_ok=True)

        crs = get_crs(zone)

        if module == "B":
            df = read_input(file)
            poly, line, pts = group_b(df, crs, out, mapping)

        elif module == "C":
            poly, line, pts = group_c(file, crs, w, h, rows, cols, out, mode, mapping)

        elif module == "D":
            df = read_input(file)
            poly, line, pts = group_d(df, crs, out, mapping)

        else:  # module == "A" (default)
            df = read_input(file)
            poly, line, pts = group_a(df, forest, crs, out, mapping)

        preview_path = os.path.join(out, "output.png")
        preview(poly, line, pts, preview_path, "yellow", "black", "red")

        return jsonify({
            "run_id": run_id,
            "download": f"/download/{run_id}"
        })

    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": f"Unexpected error: {e}"}), 500


# ================= STATIC OUTPUT FILES (preview images etc.) =================
@app.route("/outputs/<run_id>/<path:filename>")
def serve_output(run_id, filename):
    folder = os.path.join(OUTPUT, run_id)

    if not os.path.exists(os.path.join(folder, filename)):
        return jsonify({"error": "File not found"}), 404

    return send_from_directory(folder, filename)


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
