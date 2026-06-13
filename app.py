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


# ─────────────────────────────────────────────────────────────────────────────
# COLUMN ALIAS RESOLUTION
# Normalise a column header to a plain lowercase alphanumeric token so that
# all of these resolve correctly:
#   X          → x           x coordinate  → xcoordinate
#   X Cord     → xcord       X-Cord        → xcord
#   Easting    → easting     East          → east
#   Y          → y           Y Coord       → ycoord
#   Northing   → northing    North         → north
#   Order      → order       S.N           → sn
#   SN         → sn          S/N           → sn
#   Serial No  → serialno    Plot ID       → plotid
#   PlotId     → plotid      Plot_Id       → plotid
#   Forest     → forest      Forest Name   → forestname
#   Compartment→ compartment Comp          → comp
# ─────────────────────────────────────────────────────────────────────────────

def _norm(s):
    """Strip spaces, punctuation, lower-case → plain alphanumeric token."""
    return "".join(c for c in str(s).lower() if c.isalnum())


# Aliases keyed by semantic role → list of normalised strings that match it.
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
    """
    Return the first df column whose normalised name is in *aliases*.
    Returns None if no match found.
    """
    for col in df.columns:
        if _norm(col) in aliases:
            return col
    return None


def safe_col(df, mapping, key, fallback):
    """
    Resolve a column:
    1. Explicit user mapping (from the frontend dropdown)
    2. Exact match on fallback name
    3. Case-insensitive exact match on fallback name
    4. Alias table lookup (handles all spelling variants)
    """
    # 1. User-supplied mapping
    if mapping and mapping.get(key) and mapping[key] in df.columns:
        return mapping[key]

    # 2. Exact fallback
    if fallback in df.columns:
        return fallback

    # 3. Case-insensitive fallback
    for c in df.columns:
        if c.lower() == fallback.lower():
            return c

    # 4. Alias lookup
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
    """
    Rename any recognisable order/serial column to a canonical "Order"
    so downstream code always finds it.
    """
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
        raise ValueError(
            "Could not find an X / Easting / Longitude column. "
            "Please map the X column manually or rename it to 'X'."
        )
    if not y_col:
        raise ValueError(
            "Could not find a Y / Northing / Latitude column. "
            "Please map the Y column manually or rename it to 'Y'."
        )

    if order_col:
        df = df.sort_values(order_col)

    coords = list(zip(df[x_col], df[y_col]))

    if len(coords) < 3:
        raise ValueError("Not enough points to build a polygon (need at least 3).")

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

    x_col        = safe_col(df, mapping, "X",           "X")
    y_col        = safe_col(df, mapping, "Y",           "Y")
    order_col    = safe_col(df, mapping, "Order",       "Order")
    forest_col   = safe_col(df, mapping, "Forest",      "Forest")
    comp_col     = safe_col(df, mapping, "Compartment", "Compartment")

    if not x_col:
        raise ValueError(
            "Could not find an X / Easting / Longitude column. "
            "Please map the X column manually."
        )
    if not y_col:
        raise ValueError(
            "Could not find a Y / Northing / Latitude column. "
            "Please map the Y column manually."
        )
    if not forest_col:
        raise ValueError(
            "Could not find a Forest column. "
            "Please map it manually or add a 'Forest' column to your data."
        )

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

            poly = Polygon(coords)
            line = LineString(coords)

            polys.append({
                "Forest":      f,
                "Compartment": c,
                "Area":        poly.area / 10000,
                "Perim":       poly.length,
                "geometry":    poly
            })
            lines.append({
                "Forest":      f,
                "Compartment": c,
                "geometry":    line
            })
            for _, r in cg.iterrows():
                pts.append({
                    "Forest":      f,
                    "Compartment": c,
                    "Order":       r.get(order_col) if order_col else None,
                    "geometry":    Point(r[x_col], r[y_col])
                })

    poly_gdf = gpd.GeoDataFrame(polys, crs=crs)
    line_gdf = gpd.GeoDataFrame(lines, crs=crs)
    pts_gdf  = gpd.GeoDataFrame(pts,   crs=crs)

    if not poly_gdf.empty: poly_gdf.to_file(os.path.join(out, "polygon.shp"))
    if not line_gdf.empty: line_gdf.to_file(os.path.join(out, "line.shp"))
    if not pts_gdf.empty:  pts_gdf.to_file(os.path.join(out, "points.shp"))

    return poly_gdf, line_gdf, pts_gdf


# ================= GROUP C (MULTI-MODE) =================
def group_c(file, crs, w, h, rows, cols, out, mode, mapping=None):
    """
    mode == "A" → Simple Poly  (single boundary from one set of XY coords)
    mode == "B" → Segmented    (multiple boundaries grouped by Forest/Compartment)

    For ZIP input the mode is irrelevant for polygon construction (the shapefile
    already contains the geometries); mode only affects CSV/Excel input.

    MULTIPOLYGON FIX: We always collect *all* polygons instead of taking only
    the first one.  For each forest/compartment group we build one polygon and
    add it to the list.  The grid sampling then tests every sample point against
    the union of all polygons.
    """
    polygons = []

    # ─── ZIP / Shapefile input ────────────────────────────────────────────────
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

        # ── MULTIPOLYGON FIX: explode all geometries into individual Polygons ──
        for geom in gdf.geometry:
            if geom is None or geom.is_empty:
                continue
            gtype = geom.geom_type
            if gtype == "Polygon":
                polygons.append(geom)
            elif gtype == "MultiPolygon":
                # explode every part — this is the critical fix for Group C ZIP
                polygons.extend(list(geom.geoms))
            else:
                # GeometryCollection or other — extract any Polygon children
                if hasattr(geom, "geoms"):
                    for sub in geom.geoms:
                        if sub.geom_type == "Polygon":
                            polygons.append(sub)

        if not polygons:
            raise ValueError(
                "No polygon geometries found in the shapefile. "
                "Supported types: Polygon, MultiPolygon."
            )

    # ─── CSV / Excel input ────────────────────────────────────────────────────
    else:
        df = read_input(file)
        df = normalize_order(df)

        x_col     = safe_col(df, mapping, "X",     "X")
        y_col     = safe_col(df, mapping, "Y",     "Y")
        order_col = safe_col(df, mapping, "Order", "Order")

        if not x_col:
            raise ValueError(
                "Could not find an X / Easting / Longitude column. "
                "Please map the X column manually."
            )
        if not y_col:
            raise ValueError(
                "Could not find a Y / Northing / Latitude column. "
                "Please map the Y column manually."
            )

        if mode == "A":
            # Simple polygon — all rows form a single boundary
            if order_col:
                df = df.sort_values(order_col)

            coords = list(zip(df[x_col], df[y_col]))
            if len(coords) < 3:
                raise ValueError("Not enough points for a polygon (need at least 3).")
            coords.append(coords[0])
            polygons = [Polygon(coords)]

        else:
            # ── Segmented (mode B) — each Forest [+ Compartment] = one polygon ──
            forest_col = safe_col(df, mapping, "Forest",      "Forest")
            comp_col   = safe_col(df, mapping, "Compartment", "Compartment")

            if not forest_col:
                raise ValueError(
                    "Segmented mode requires a Forest column. "
                    "Please map it manually or rename the column to 'Forest'."
                )

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
                polygons.append(Polygon(coords))

            if not polygons:
                detail = (
                    f" ({skipped} group(s) had fewer than 3 points and were skipped)"
                    if skipped else ""
                )
                raise ValueError(
                    "No valid polygons could be built from the data. "
                    "Each Forest (or Forest+Compartment) group needs at least 3 points."
                    + detail
                )

    if not polygons:
        raise ValueError("No valid polygons could be built from the input data.")

    # ─── Build GeoDataFrames ─────────────────────────────────────────────────
    poly_gdf = gpd.GeoDataFrame(
        [{"geometry": p} for p in polygons], crs=crs
    )
    line_gdf = gpd.GeoDataFrame(
        [{"geometry": LineString(p.exterior.coords)} for p in polygons], crs=crs
    )

    # ─── Sample-plot grid ─────────────────────────────────────────────────────
    # Use the union of ALL polygons so the grid covers every boundary.
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
                pts.append({
                    "SN":      sn,
                    "X":       center.x,
                    "Y":       center.y,
                    "geometry": center
                })
                sn += 1

    pts_gdf = gpd.GeoDataFrame(pts, crs=crs)

    poly_gdf.to_file(os.path.join(out, "boundary_polygons.shp"))
    line_gdf.to_file(os.path.join(out, "boundary_lines.shp"))

    if not pts_gdf.empty:
        pts_gdf.to_file(os.path.join(out, "sampleplot.shp"))
        pd.DataFrame(pts)[["SN", "X", "Y"]].to_excel(
            os.path.join(out, "sampleplot.xlsx"), index=False
        )

    return poly_gdf, line_gdf, pts_gdf


# ================= GROUP D =================
def group_d(df, crs, out, mapping=None):
    df = normalize_order(df)

    x_col      = safe_col(df, mapping, "X",      "X")
    y_col      = safe_col(df, mapping, "Y",      "Y")
    order_col  = safe_col(df, mapping, "Order",  "Order")
    forest_col = safe_col(df, mapping, "Forest", "Forest")

    if not x_col:
        raise ValueError(
            "Could not find an X / Easting / Longitude column. "
            "Please map the X column manually."
        )
    if not y_col:
        raise ValueError(
            "Could not find a Y / Northing / Latitude column. "
            "Please map the Y column manually."
        )
    if not forest_col:
        raise ValueError(
            "Could not find a Forest column. "
            "Please map it manually or add a 'Forest' column to your data."
        )

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
            "Area":   poly.area / 10000,
            "Perim":  poly.length,
            "geometry": poly
        })
        lines.append({"Forest": f, "geometry": line})

        for _, r in g.iterrows():
            pts.append({
                "Forest": f,
                "Order":  r.get(order_col) if order_col else None,
                "geometry": Point(r[x_col], r[y_col])
            })

    poly_gdf = gpd.GeoDataFrame(polys, crs=crs)
    line_gdf = gpd.GeoDataFrame(lines, crs=crs)
    pts_gdf  = gpd.GeoDataFrame(pts,   crs=crs)

    if not poly_gdf.empty: poly_gdf.to_file(os.path.join(out, "polygon.shp"))
    if not line_gdf.empty: line_gdf.to_file(os.path.join(out, "line.shp"))
    if not pts_gdf.empty:  pts_gdf.to_file(os.path.join(out, "points.shp"))

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
            poly, line, pts = group_d(df, crs, out, mapping)

        else:  # module == "A" (default)
            df = read_input(file)
            poly, line, pts = group_a(df, forest, crs, out, mapping)

        preview_path = os.path.join(out, "output.png")
        preview(poly, line, pts, preview_path, "yellow", "black", "red")

        return jsonify({
            "run_id":   run_id,
            "download": f"/download/{run_id}"
        })

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
