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


def safe_polygon(coords):
    """
    Build a Polygon from coords and ensure it is topologically valid.
    Self-intersecting rings (figure-8s, bowtie shapes, etc.) raise a
    TopologyException during area / containment tests.  buffer(0) is the
    standard Shapely idiom to repair such geometries without external libs.
    """
    poly = Polygon(coords)
    if not poly.is_valid:
        poly = poly.buffer(0)
    return poly


# ─────────────────────────────────────────────────────────────────────────────
# COLUMN ALIAS RESOLUTION
# ─────────────────────────────────────────────────────────────────────────────

def _norm(s):
    return "".join(c for c in str(s).lower() if c.isalnum())


_X_ALIASES = {
    "x", "xcoord", "xcoordinate", "xcord", "xcords", "xcoords",
    "east", "easting", "eastings", "lon", "long", "longitude", "lng",
    "pointx", "coordx", "utme", "utmx",
}
_Y_ALIASES = {
    "y", "ycoord", "ycoordinate", "ycord", "ycords", "ycoords",
    "north", "northing", "northings", "lat", "latitude",
    "pointy", "coordy", "utmn", "utmy",
}
_ORDER_ALIASES = {
    "order", "id", "sn", "sno", "serial", "serialno", "serialnumber",
    "seq", "sequence", "index", "rowid", "fid", "no", "num", "number",
    "plotid", "plotno", "pointid", "pointno", "pid",
}
_FOREST_ALIASES = {
    "forest", "forestname", "forestid", "forestno", "fname",
    "forestblock", "block",
}
_COMPARTMENT_ALIASES = {
    "compartment", "comp", "compartmentno", "compartmentid",
    "compno", "compid", "section", "sectionno",
}


def _find_col(df, aliases):
    for col in df.columns:
        if _norm(col) in aliases:
            return col
    return None


def safe_col(df, mapping, key, fallback):
    if mapping and mapping.get(key) and mapping[key] in df.columns:
        return mapping[key]
    if fallback in df.columns:
        return fallback
    for c in df.columns:
        if c.lower() == fallback.lower():
            return c
    alias_map = {
        "X":           _X_ALIASES,
        "Y":           _Y_ALIASES,
        "Order":       _ORDER_ALIASES,
        "Forest":      _FOREST_ALIASES,
        "Compartment": _COMPARTMENT_ALIASES,
    }
    if key in alias_map:
        hit = _find_col(df, alias_map[key])
        if hit:
            return hit
    return None


# ================= NORMALIZE ORDER COLUMN =================
def normalize_order(df):
    for c in df.columns:
        if _norm(c) in _ORDER_ALIASES and c != "Order":
            df = df.rename(columns={c: "Order"})
            break
    return df


# ================= GROUP A =================
def group_a(df, forest, crs, out, mapping=None):
    df = normalize_order(df)

    x_col = safe_col(df, mapping, "X", "X")
    y_col = safe_col(df, mapping, "Y", "Y")
    order_col = safe_col(df, mapping, "Order", "Order")

    if not x_col:
        raise ValueError("Could not find an X / Easting / Longitude column.")
    if not y_col:
        raise ValueError("Could not find a Y / Northing / Latitude column.")

    if order_col:
        df = df.sort_values(order_col)

    coords = list(zip(df[x_col], df[y_col]))
    if len(coords) < 3:
        raise ValueError("Not enough points to build a polygon (need at least 3).")

    coords.append(coords[0])
    poly = safe_polygon(coords)
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

    x_col        = safe_col(df, mapping, "X",           "X")
    y_col        = safe_col(df, mapping, "Y",           "Y")
    order_col    = safe_col(df, mapping, "Order",       "Order")
    forest_col   = safe_col(df, mapping, "Forest",      "Forest")
    comp_col     = safe_col(df, mapping, "Compartment", "Compartment")

    if not x_col:
        raise ValueError("Could not find an X / Easting / Longitude column.")
    if not y_col:
        raise ValueError("Could not find a Y / Northing / Latitude column.")
    if not forest_col:
        raise ValueError("Could not find a Forest column.")

    polys, lines, pts = [], [], []

    for f, g in df.groupby(forest_col):
        if order_col:
            g = g.sort_values(order_col)

        sub_groups = g.groupby(comp_col) if comp_col else [(None, g)]

        for c, cg in sub_groups:
            coords = list(zip(cg[x_col], cg[y_col]))
            if len(coords) < 3:
                continue
            coords.append(coords[0])
            poly = safe_polygon(coords)
            line = LineString(coords)

            polys.append({"Forest": f, "Compartment": c, "Area": poly.area / 10000, "Perim": poly.length, "geometry": poly})
            lines.append({"Forest": f, "Compartment": c, "geometry": line})
            for _, r in cg.iterrows():
                pts.append({"Forest": f, "Compartment": c, "Order": r.get(order_col) if order_col else None, "geometry": Point(r[x_col], r[y_col])})

    poly_gdf = gpd.GeoDataFrame(polys, crs=crs)
    line_gdf = gpd.GeoDataFrame(lines, crs=crs)
    pts_gdf  = gpd.GeoDataFrame(pts,   crs=crs)

    if not poly_gdf.empty: poly_gdf.to_file(os.path.join(out, "polygon.shp"))
    if not line_gdf.empty: line_gdf.to_file(os.path.join(out, "line.shp"))
    if not pts_gdf.empty:  pts_gdf.to_file(os.path.join(out, "points.shp"))

    return poly_gdf, line_gdf, pts_gdf


# ================= GROUP C (MULTI-MODE) =================
def group_c(file, crs, w, h, rows, cols, out, mode, mapping=None):
    polygons = []

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
            raise ValueError("No shapefile (.shp) found inside the ZIP archive.")

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
            raise ValueError("The selected shapefile contains no features.")
        if gdf.crs is None:
            gdf = gdf.set_crs(crs)
        else:
            gdf = gdf.to_crs(crs)

        for geom in gdf.geometry:
            if geom is None or geom.is_empty:
                continue
            gtype = geom.geom_type
            if gtype == "Polygon":
                polygons.append(geom if geom.is_valid else geom.buffer(0))
            elif gtype == "MultiPolygon":
                for part in geom.geoms:
                    polygons.append(part if part.is_valid else part.buffer(0))
            else:
                if hasattr(geom, "geoms"):
                    for sub in geom.geoms:
                        if sub.geom_type == "Polygon":
                            polygons.append(sub if sub.is_valid else sub.buffer(0))

        if not polygons:
            raise ValueError("No polygon geometries found in the shapefile.")

    else:
        df = read_input(file)
        df = normalize_order(df)
        x_col     = safe_col(df, mapping, "X",     "X")
        y_col     = safe_col(df, mapping, "Y",     "Y")
        order_col = safe_col(df, mapping, "Order", "Order")

        if not x_col:
            raise ValueError("Could not find an X / Easting / Longitude column.")
        if not y_col:
            raise ValueError("Could not find a Y / Northing / Latitude column.")

        if mode == "A":
            if order_col:
                df = df.sort_values(order_col)
            coords = list(zip(df[x_col], df[y_col]))
            if len(coords) < 3:
                raise ValueError("Not enough points for a polygon (need at least 3).")
            coords.append(coords[0])
            polygons = [safe_polygon(coords)]
        else:
            forest_col = safe_col(df, mapping, "Forest",      "Forest")
            comp_col   = safe_col(df, mapping, "Compartment", "Compartment")
            if not forest_col:
                raise ValueError("Segmented mode requires a Forest column.")

            group_keys = [forest_col, comp_col] if comp_col else [forest_col]
            skipped = 0
            for key, g in df.groupby(group_keys):
                if order_col:
                    g = g.sort_values(order_col)
                coords = list(zip(g[x_col], g[y_col]))
                if len(coords) < 3:
                    skipped += 1
                    continue
                coords.append(coords[0])
                polygons.append(safe_polygon(coords))

            if not polygons:
                raise ValueError("No valid polygons could be built from the data.")

    if not polygons:
        raise ValueError("No valid polygons could be built from the input data.")

    poly_gdf = gpd.GeoDataFrame([{"geometry": p} for p in polygons], crs=crs)
    line_gdf = gpd.GeoDataFrame([{"geometry": LineString(p.exterior.coords)} for p in polygons], crs=crs)

    union  = poly_gdf.unary_union
    minx, miny, _, _ = union.bounds
    pts  = []
    sn   = 1
    for r in range(rows):
        for c_idx in range(cols):
            x      = minx + c_idx * w
            y      = miny + r * h
            center = Point(x + w / 2, y + h / 2)
            if union.contains(center):
                pts.append({"SN": sn, "X": center.x, "Y": center.y, "geometry": center})
                sn += 1

    pts_gdf = gpd.GeoDataFrame(pts, crs=crs)

    poly_gdf.to_file(os.path.join(out, "boundary_polygons.shp"))
    line_gdf.to_file(os.path.join(out, "boundary_lines.shp"))
    if not pts_gdf.empty:
        pts_gdf.to_file(os.path.join(out, "sampleplot.shp"))
        pd.DataFrame(pts)[["SN", "X", "Y"]].to_excel(os.path.join(out, "sampleplot.xlsx"), index=False)

    return poly_gdf, line_gdf, pts_gdf


# ================= GROUP D =================
# Group D supports two sub-modes passed via `mode`:
#
#   mode="A"  →  Whole Forest  (one polygon per Forest)
#       output/run_id/
#           Forest_A/  Polygon.shp  Line.shp  Point.shp
#           Forest_B/  ...
#
#   mode="B"  →  Segmented     (one polygon per Forest × Compartment)
#       output/run_id/
#           Forest_A/
#               Comp_1/  Polygon.shp  Line.shp  Point.shp
#               Comp_2/  ...
#           Forest_B/
#               ...
#
# Returns merged GeoDataFrames for the preview PNG.
# Topology fix: safe_polygon() repairs self-intersecting rings via buffer(0).
def _safe_dirname(s):
    """Return a filesystem-safe directory name from any string."""
    return str(s).strip().replace("/", "_").replace("\\", "_").replace(":", "_")


def _save_forest_layer(poly_rec, line_rec, pt_recs, save_dir, crs):
    """Write Polygon / Line / Point shapefiles into save_dir."""
    os.makedirs(save_dir, exist_ok=True)
    gpd.GeoDataFrame([poly_rec], crs=crs).to_file(os.path.join(save_dir, "Polygon.shp"))
    gpd.GeoDataFrame([line_rec], crs=crs).to_file(os.path.join(save_dir, "Line.shp"))
    gpd.GeoDataFrame(pt_recs,    crs=crs).to_file(os.path.join(save_dir, "Point.shp"))


def group_d(df, crs, out, mapping=None, mode="A"):
    df = normalize_order(df)

    x_col      = safe_col(df, mapping, "X",           "X")
    y_col      = safe_col(df, mapping, "Y",           "Y")
    order_col  = safe_col(df, mapping, "Order",       "Order")
    forest_col = safe_col(df, mapping, "Forest",      "Forest")
    comp_col   = safe_col(df, mapping, "Compartment", "Compartment")

    if not x_col:
        raise ValueError("Could not find an X / Easting / Longitude column. Please map the X column manually.")
    if not y_col:
        raise ValueError("Could not find a Y / Northing / Latitude column. Please map the Y column manually.")
    if not forest_col:
        raise ValueError("Could not find a Forest column. Please map it manually or add a 'Forest' column to your data.")

    if mode == "B" and not comp_col:
        raise ValueError(
            "Segmented mode requires a Compartment column. "
            "Please map it manually or switch to Whole Forest mode."
        )

    # Accumulators for the combined preview PNG
    all_polys, all_lines, all_pts = [], [], []
    skipped = 0

    for f, fg in df.groupby(forest_col):
        forest_dir = os.path.join(out, _safe_dirname(f))

        if mode == "B":
            # ── Segmented: one subfolder per compartment inside the forest folder ──
            for c, cg in fg.groupby(comp_col):
                if order_col:
                    cg = cg.sort_values(order_col)

                coords = list(zip(cg[x_col], cg[y_col]))
                if len(coords) < 3:
                    skipped += 1
                    continue
                coords.append(coords[0])

                poly = safe_polygon(coords)
                line = LineString(coords)

                poly_rec = {
                    "Forest": f, "Compartment": c,
                    "Area_ha": round(poly.area / 10000, 4),
                    "Perim_m": round(poly.length, 4),
                    "geometry": poly,
                }
                line_rec = {"Forest": f, "Compartment": c, "geometry": line}
                pt_recs  = [
                    {"Forest": f, "Compartment": c,
                     "Order": r[order_col] if order_col else None,
                     "geometry": Point(r[x_col], r[y_col])}
                    for _, r in cg.iterrows()
                ]

                comp_dir = os.path.join(forest_dir, _safe_dirname(c))
                _save_forest_layer(poly_rec, line_rec, pt_recs, comp_dir, crs)

                all_polys.append(poly_rec)
                all_lines.append(line_rec)
                all_pts.extend(pt_recs)

        else:
            # ── Whole Forest: one polygon for the entire forest ──────────────
            if order_col:
                fg = fg.sort_values(order_col)

            coords = list(zip(fg[x_col], fg[y_col]))
            if len(coords) < 3:
                skipped += 1
                continue
            coords.append(coords[0])

            poly = safe_polygon(coords)
            line = LineString(coords)

            poly_rec = {
                "Forest":  f,
                "Area_ha": round(poly.area / 10000, 4),
                "Perim_m": round(poly.length, 4),
                "geometry": poly,
            }
            line_rec = {"Forest": f, "geometry": line}
            pt_recs  = [
                {"Forest": f,
                 "Order": r[order_col] if order_col else None,
                 "geometry": Point(r[x_col], r[y_col])}
                for _, r in fg.iterrows()
            ]

            _save_forest_layer(poly_rec, line_rec, pt_recs, forest_dir, crs)

            all_polys.append(poly_rec)
            all_lines.append(line_rec)
            all_pts.extend(pt_recs)

    if not all_polys:
        detail = f" ({skipped} group(s) had fewer than 3 points)" if skipped else ""
        raise ValueError("No valid polygons could be built from the data." + detail)

    # Combined GeoDataFrames used only for the preview PNG
    poly_gdf = gpd.GeoDataFrame(all_polys, crs=crs)
    line_gdf = gpd.GeoDataFrame(all_lines, crs=crs)
    pts_gdf  = gpd.GeoDataFrame(all_pts,   crs=crs)

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

        file   = request.files["file"]
        mode   = request.form.get("mode",   "A")
        module = request.form.get("module", mode)
        zone   = request.form.get("zone",   "44")

        try:
            mapping = json.loads(request.form.get("mapping", "{}"))
        except (TypeError, ValueError):
            mapping = {}

        w      = float(request.form.get("w",    50))
        h      = float(request.form.get("h",    50))
        rows   = int(  request.form.get("rows", 10))
        cols   = int(  request.form.get("cols", 10))
        forest = request.form.get("forest") or mapping.get("forest") or "FOREST"

        run_id = str(uuid.uuid4())
        out    = os.path.join(OUTPUT, run_id)
        os.makedirs(out, exist_ok=True)

        crs = get_crs(zone)

        if module == "B":
            df = read_input(file)
            poly, line, pts = group_b(df, crs, out, mapping)
        elif module == "C":
            poly, line, pts = group_c(file, crs, w, h, rows, cols, out, mode, mapping)
        elif module == "D":
            df = read_input(file)
            d_mode = request.form.get("d_mode", "A")  # "A"=Whole Forest  "B"=Segmented
            poly, line, pts = group_d(df, crs, out, mapping, mode=d_mode)
        else:
            df = read_input(file)
            poly, line, pts = group_a(df, forest, crs, out, mapping)

        preview_path = os.path.join(out, "output.png")
        preview(poly, line, pts, preview_path, "yellow", "black", "red")

        return jsonify({"run_id": run_id, "download": f"/download/{run_id}"})

    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": f"Unexpected error: {e}"}), 500


# ================= STATIC OUTPUT FILES =================
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
