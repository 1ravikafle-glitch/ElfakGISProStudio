import os
import uuid
import zipfile
import shutil
import json
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

from flask import Flask, request, jsonify, send_file, send_from_directory, render_template
from shapely.geometry import Polygon, Point, LineString, MultiPolygon
from shapely.ops import unary_union

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
    """Build a Polygon and repair topology (buffer(0) fixes self-intersections)."""
    poly = Polygon(coords)
    if not poly.is_valid:
        poly = poly.buffer(0)
    return poly


def _safe_dirname(s):
    """Return a filesystem-safe directory name from any string."""
    return str(s).strip().replace("/", "_").replace("\\", "_").replace(":", "_")


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


def normalize_order(df):
    for c in df.columns:
        if _norm(c) in _ORDER_ALIASES and c != "Order":
            df = df.rename(columns={c: "Order"})
            break
    return df


# ================= GROUP A =================
def group_a(df, forest, crs, out, mapping=None):
    df = normalize_order(df)
    x_col     = safe_col(df, mapping, "X", "X")
    y_col     = safe_col(df, mapping, "Y", "Y")
    order_col = safe_col(df, mapping, "Order", "Order")

    if not x_col: raise ValueError("Could not find an X / Easting / Longitude column.")
    if not y_col: raise ValueError("Could not find a Y / Northing / Latitude column.")

    if order_col: df = df.sort_values(order_col)
    coords = list(zip(df[x_col], df[y_col]))
    if len(coords) < 3: raise ValueError("Not enough points to build a polygon (need at least 3).")
    coords.append(coords[0])
    poly = safe_polygon(coords)
    line = LineString(coords)

    poly_gdf = gpd.GeoDataFrame([{"Forest": forest, "Area": poly.area/10000, "Perim": poly.length, "geometry": poly}], crs=crs)
    line_gdf = gpd.GeoDataFrame([{"Forest": forest, "geometry": line}], crs=crs)
    pts_gdf  = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df[x_col], df[y_col]), crs=crs)

    poly_gdf.to_file(os.path.join(out, f"{_safe_dirname(forest)}_polygon.shp"))
    line_gdf.to_file(os.path.join(out, f"{_safe_dirname(forest)}_line.shp"))
    pts_gdf.to_file(os.path.join(out,  f"{_safe_dirname(forest)}_point.shp"))
    return poly_gdf, line_gdf, pts_gdf


# ================= GROUP B =================
def group_b(df, crs, out, mapping=None):
    df = normalize_order(df)
    x_col      = safe_col(df, mapping, "X",           "X")
    y_col      = safe_col(df, mapping, "Y",           "Y")
    order_col  = safe_col(df, mapping, "Order",       "Order")
    forest_col = safe_col(df, mapping, "Forest",      "Forest")
    comp_col   = safe_col(df, mapping, "Compartment", "Compartment")

    if not x_col:      raise ValueError("Could not find an X / Easting / Longitude column.")
    if not y_col:      raise ValueError("Could not find a Y / Northing / Latitude column.")
    if not forest_col: raise ValueError("Could not find a Forest column.")

    polys, lines, pts = [], [], []
    for f, g in df.groupby(forest_col):
        if order_col: g = g.sort_values(order_col)
        sub_groups = g.groupby(comp_col) if comp_col else [(None, g)]
        for c, cg in sub_groups:
            coords = list(zip(cg[x_col], cg[y_col]))
            if len(coords) < 3: continue
            coords.append(coords[0])
            poly = safe_polygon(coords)
            line = LineString(coords)
            polys.append({"Forest": f, "Compartment": c, "Area": poly.area/10000, "Perim": poly.length, "geometry": poly})
            lines.append({"Forest": f, "Compartment": c, "geometry": line})
            for _, r in cg.iterrows():
                pts.append({"Forest": f, "Compartment": c, "Order": r[order_col] if order_col else None, "geometry": Point(r[x_col], r[y_col])})

    poly_gdf = gpd.GeoDataFrame(polys, crs=crs)
    line_gdf = gpd.GeoDataFrame(lines, crs=crs)
    pts_gdf  = gpd.GeoDataFrame(pts,   crs=crs)
    if not poly_gdf.empty: poly_gdf.to_file(os.path.join(out, "forest_polygon.shp"))
    if not line_gdf.empty: line_gdf.to_file(os.path.join(out, "forest_line.shp"))
    if not pts_gdf.empty:  pts_gdf.to_file(os.path.join(out, "forest_point.shp"))
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
        if gdf.empty: raise ValueError("The selected shapefile contains no features.")
        if gdf.crs is None: gdf = gdf.set_crs(crs)
        else: gdf = gdf.to_crs(crs)

        for geom in gdf.geometry:
            if geom is None or geom.is_empty: continue
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
        if not polygons: raise ValueError("No polygon geometries found in the shapefile.")

    else:
        df = read_input(file)
        df = normalize_order(df)
        x_col     = safe_col(df, mapping, "X",     "X")
        y_col     = safe_col(df, mapping, "Y",     "Y")
        order_col = safe_col(df, mapping, "Order", "Order")

        if not x_col: raise ValueError("Could not find an X / Easting / Longitude column.")
        if not y_col: raise ValueError("Could not find a Y / Northing / Latitude column.")

        if mode == "A":
            if order_col: df = df.sort_values(order_col)
            coords = list(zip(df[x_col], df[y_col]))
            if len(coords) < 3: raise ValueError("Not enough points for a polygon (need at least 3).")
            coords.append(coords[0])
            polygons = [safe_polygon(coords)]
        else:
            forest_col = safe_col(df, mapping, "Forest", "Forest")
            comp_col   = safe_col(df, mapping, "Compartment", "Compartment")
            if not forest_col: raise ValueError("Segmented mode requires a Forest column.")
            group_keys = [forest_col, comp_col] if comp_col else [forest_col]
            skipped = 0
            for key, g in df.groupby(group_keys):
                if order_col: g = g.sort_values(order_col)
                coords = list(zip(g[x_col], g[y_col]))
                if len(coords) < 3: skipped += 1; continue
                coords.append(coords[0])
                polygons.append(safe_polygon(coords))
            if not polygons: raise ValueError("No valid polygons could be built from the data.")

    if not polygons: raise ValueError("No valid polygons could be built from the input data.")

    poly_gdf = gpd.GeoDataFrame([{"geometry": p} for p in polygons], crs=crs)
    line_gdf = gpd.GeoDataFrame([{"geometry": LineString(p.exterior.coords)} for p in polygons], crs=crs)

    union = poly_gdf.unary_union
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
    poly_gdf.to_file(os.path.join(out, "boundary_polygon.shp"))
    line_gdf.to_file(os.path.join(out, "boundary_line.shp"))
    if not pts_gdf.empty:
        pts_gdf.to_file(os.path.join(out, "sampleplot_point.shp"))
        pd.DataFrame(pts)[["SN", "X", "Y"]].to_excel(os.path.join(out, "sampleplot.xlsx"), index=False)

    return poly_gdf, line_gdf, pts_gdf


# ================= GROUP D =================
def _save_forest_layer(poly_rec, line_rec, pt_recs, save_dir, crs):
    os.makedirs(save_dir, exist_ok=True)
    # Use the forest (or compartment) folder name as the file prefix
    prefix = os.path.basename(save_dir)
    gpd.GeoDataFrame([poly_rec], crs=crs).to_file(os.path.join(save_dir, f"{prefix}_polygon.shp"))
    gpd.GeoDataFrame([line_rec], crs=crs).to_file(os.path.join(save_dir, f"{prefix}_line.shp"))
    gpd.GeoDataFrame(pt_recs,    crs=crs).to_file(os.path.join(save_dir, f"{prefix}_point.shp"))


def group_d(df, crs, out, mapping=None, mode="A"):
    df = normalize_order(df)
    x_col      = safe_col(df, mapping, "X",           "X")
    y_col      = safe_col(df, mapping, "Y",           "Y")
    order_col  = safe_col(df, mapping, "Order",       "Order")
    forest_col = safe_col(df, mapping, "Forest",      "Forest")
    comp_col   = safe_col(df, mapping, "Compartment", "Compartment")

    if not x_col:      raise ValueError("Could not find an X / Easting / Longitude column.")
    if not y_col:      raise ValueError("Could not find a Y / Northing / Latitude column.")
    if not forest_col: raise ValueError("Could not find a Forest column.")
    if mode == "B" and not comp_col:
        raise ValueError("Segmented mode requires a Compartment column.")

    all_polys, all_lines, all_pts = [], [], []
    skipped = 0

    for f, fg in df.groupby(forest_col):
        forest_dir = os.path.join(out, _safe_dirname(f))

        if mode == "B":
            for c, cg in fg.groupby(comp_col):
                if order_col: cg = cg.sort_values(order_col)
                coords = list(zip(cg[x_col], cg[y_col]))
                if len(coords) < 3: skipped += 1; continue
                coords.append(coords[0])
                poly = safe_polygon(coords)
                line = LineString(coords)
                poly_rec = {"Forest": f, "Compartment": c, "Area_ha": round(poly.area/10000,4), "Perim_m": round(poly.length,4), "geometry": poly}
                line_rec = {"Forest": f, "Compartment": c, "geometry": line}
                pt_recs  = [{"Forest": f, "Compartment": c, "Order": r[order_col] if order_col else None, "geometry": Point(r[x_col], r[y_col])} for _, r in cg.iterrows()]
                comp_dir = os.path.join(forest_dir, _safe_dirname(c))
                _save_forest_layer(poly_rec, line_rec, pt_recs, comp_dir, crs)
                all_polys.append(poly_rec); all_lines.append(line_rec); all_pts.extend(pt_recs)
        else:
            if order_col: fg = fg.sort_values(order_col)
            coords = list(zip(fg[x_col], fg[y_col]))
            if len(coords) < 3: skipped += 1; continue
            coords.append(coords[0])
            poly = safe_polygon(coords)
            line = LineString(coords)
            poly_rec = {"Forest": f, "Area_ha": round(poly.area/10000,4), "Perim_m": round(poly.length,4), "geometry": poly}
            line_rec = {"Forest": f, "geometry": line}
            pt_recs  = [{"Forest": f, "Order": r[order_col] if order_col else None, "geometry": Point(r[x_col], r[y_col])} for _, r in fg.iterrows()]
            _save_forest_layer(poly_rec, line_rec, pt_recs, forest_dir, crs)
            all_polys.append(poly_rec); all_lines.append(line_rec); all_pts.extend(pt_recs)

    if not all_polys:
        raise ValueError(f"No valid polygons could be built from the data.{' (' + str(skipped) + ' groups skipped)' if skipped else ''}")

    return (gpd.GeoDataFrame(all_polys, crs=crs),
            gpd.GeoDataFrame(all_lines, crs=crs),
            gpd.GeoDataFrame(all_pts,   crs=crs))


# ================= GROUP E — POLYGON SUBDIVIDER =================
#
# Subdivides forest polygon(s) into N near-equal-area compartments using
# iterative area-balanced bisection along the longest bounding-box axis.
#
# e_mode = "A"  →  Single-forest  (like Group A: all rows = one polygon)
#   Output:  out/compartments.shp + compartment_lines.shp + compartment_points.shp
#            out/compartment_summary.xlsx
#
# e_mode = "B"  →  Multi-forest   (like Group B: Forest column groups rows)
#   Output:  out/Forest_A/compartments.shp  …
#            out/Forest_A/compartment_summary.xlsx  …
#
# Algorithm: recursive bisection — always splits the largest remaining piece
#   along its longest axis at an area-balanced position (binary search).
#   After reaching N pieces the compartments are numbered by area (largest first).

def _repair_geom(g):
    """Fix invalid geometry with buffer(0)."""
    return g if g.is_valid else g.buffer(0)


def _find_equal_area_cut(poly, axis, lo_bound, hi_bound, target_cumulative_area,
                         minx, miny, maxx, maxy, n_tries=80):
    from shapely.geometry import box as shapely_box
    lo, hi = lo_bound, hi_bound
    mid = (lo + hi) / 2.0   # safe default in case loop breaks immediately

    for _ in range(n_tries):
        mid = (lo + hi) / 2.0
        try:
            if axis == "x":
                clip = poly.intersection(shapely_box(minx - 1, miny - 1, mid, maxy + 1))
            else:
                clip = poly.intersection(shapely_box(minx - 1, miny - 1, maxx + 1, mid))
        except Exception:
            break

        got = clip.area if not clip.is_empty else 0.0
        rel_err = abs(got - target_cumulative_area) / (target_cumulative_area + 1e-10)
        if rel_err < 0.0005:
            return mid
        elif got < target_cumulative_area:
            lo = mid
        else:
            hi = mid

    return mid


def _subdivide_polygon(poly, n):
    """
    Slice *poly* into exactly *n* near-equal-area compartments.

    Algorithm — direct N-strip cumulative-area slicing:
    ────────────────────────────────────────────────────
    1. Choose the longest bounding-box axis (X or Y).
    2. For k = 1 … n-1, binary-search for cut_k such that
         area( poly ∩ [axis_start, cut_k] ) = k/n × total_area
       Each cut is searched over the full axis range using CUMULATIVE area
       from the polygon start — this is the key fix.
    3. Clip between consecutive cuts to get exactly n strips.

    Why this works for any N (odd, even, prime):
      Each cut_k is independently positioned so the cumulative slice is
      exactly k/n of total area.  The strip between cut_{k-1} and cut_k is
      therefore always 1/n of total area regardless of N.
      No recursive bisection → no power-of-2 artefacts.
    """
    from shapely.geometry import box as shapely_box

    if n <= 1:
        return [_repair_geom(poly)]

    poly = _repair_geom(poly)
    total_area = poly.area
    if total_area < 1e-10:
        return [poly]

    minx, miny, maxx, maxy = poly.bounds
    dx = maxx - minx
    dy = maxy - miny

    # Choose longest axis
    if dx >= dy:
        axis = "x"; lo_bound, hi_bound = minx, maxx
    else:
        axis = "y"; lo_bound, hi_bound = miny, maxy

    # Compute n-1 cut positions, each at cumulative area = k/n × total
    cut_positions = []
    for k in range(1, n):
        target_cum = total_area * k / n
        c = _find_equal_area_cut(
            poly, axis, lo_bound, hi_bound,
            target_cum, minx, miny, maxx, maxy
        )
        cut_positions.append(c)

    # Build strips: clip between consecutive cut lines
    pieces = []
    cuts = [lo_bound - 1] + cut_positions + [hi_bound + 1]

    for i in range(n):
        c_lo = cuts[i]
        c_hi = cuts[i + 1]
        try:
            if axis == "x":
                strip = poly.intersection(shapely_box(c_lo, miny - 1, c_hi, maxy + 1))
            else:
                strip = poly.intersection(shapely_box(minx - 1, c_lo, maxx + 1, c_hi))
        except Exception:
            strip = None

        if strip is not None and not strip.is_empty and strip.area > 1e-6:
            pieces.append(_repair_geom(strip))

    # If clipping produced fewer pieces (degenerate geometry), return what we have
    if not pieces:
        return [poly]

    return pieces


def _save_compartments(pieces, forest_name, crs, save_dir):
    """
    Write three shapefiles + Excel summary for a set of compartment polygons.
    Files: compartments.shp, compartment_lines.shp, compartment_points.shp,
           compartment_summary.xlsx
    """
    os.makedirs(save_dir, exist_ok=True)

    total_area = sum(p.area for p in pieces)
    records, line_recs, pt_recs = [], [], []

    for i, p in enumerate(pieces, start=1):
        p = _repair_geom(p)
        comp_id   = f"Comp_{i:03d}"
        area_ha   = round(p.area / 10000, 4)
        perim_m   = round(p.length, 4)
        pct_area  = round(p.area / total_area * 100, 2) if total_area > 0 else 0
        centroid  = p.centroid

        records.append({
            "Forest":   forest_name,
            "Comp_ID":  comp_id,
            "Area_ha":  area_ha,
            "Perim_m":  perim_m,
            "Pct_Area": pct_area,
            "geometry": p,
        })
        line_recs.append({
            "Forest":  forest_name,
            "Comp_ID": comp_id,
            "geometry": LineString(p.exterior.coords) if p.geom_type == "Polygon" else LineString(list(p.geoms)[0].exterior.coords),
        })
        pt_recs.append({
            "Forest":  forest_name,
            "Comp_ID": comp_id,
            "Area_ha": area_ha,
            "Pct_Area": pct_area,
            "geometry": centroid,
        })

    poly_gdf = gpd.GeoDataFrame(records,   crs=crs)
    line_gdf = gpd.GeoDataFrame(line_recs, crs=crs)
    pts_gdf  = gpd.GeoDataFrame(pt_recs,   crs=crs)

    prefix = _safe_dirname(forest_name)
    poly_gdf.to_file(os.path.join(save_dir, f"{prefix}_compartment_polygon.shp"))
    line_gdf.to_file(os.path.join(save_dir, f"{prefix}_compartment_line.shp"))
    pts_gdf.to_file( os.path.join(save_dir, f"{prefix}_compartment_point.shp"))

    summary_df = pd.DataFrame([
        {k: v for k, v in r.items() if k != "geometry"} for r in records
    ])
    summary_df.to_excel(os.path.join(save_dir, f"{prefix}_compartment_summary.xlsx"), index=False)

    return poly_gdf, line_gdf, pts_gdf


def _df_to_polygon(df, x_col, y_col, order_col):
    """Build and validate a single closed Polygon from a DataFrame of boundary points."""
    if order_col and order_col in df.columns:
        df = df.sort_values(order_col)
    coords = list(zip(df[x_col], df[y_col]))
    if len(coords) < 3:
        raise ValueError("Need at least 3 points to build a polygon.")
    coords.append(coords[0])
    return _repair_geom(Polygon(coords))


def _load_polygons_from_zip(file, target_shp, crs):
    """
    Extract a ZIP archive, find the selected .shp, read it as a GeoDataFrame
    reprojected to *crs*.  Returns a list of (feature_name, polygon) tuples
    where feature_name comes from any Name/Forest/Label attribute or the FID.
    """
    folder = os.path.join(UPLOAD, str(uuid.uuid4()))
    os.makedirs(folder, exist_ok=True)
    zip_path = os.path.join(folder, "input.zip")
    file.save(zip_path)

    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(folder)

    # Collect all .shp paths
    shp_candidates = []
    for root, _, files in os.walk(folder):
        for f in files:
            if f.endswith(".shp"):
                shp_candidates.append(os.path.join(root, f))

    if not shp_candidates:
        raise ValueError("No shapefile (.shp) found inside the ZIP archive.")

    # Pick target shp
    shp_path = shp_candidates[0]
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

    # Try to find a name column (Forest / Name / Label / first text column)
    name_col = None
    for candidate in ["Forest", "forest", "Name", "name", "NAME",
                       "Label", "label", "LABEL", "ID", "id"]:
        if candidate in gdf.columns:
            name_col = candidate
            break
    if name_col is None:
        for col in gdf.columns:
            if col == "geometry":
                continue
            if gdf[col].dtype == object:
                name_col = col
                break

    results = []
    for i, row in gdf.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue

        feat_name = str(row[name_col]) if name_col else f"Feature_{i+1}"

        # Flatten to individual Polygon objects
        if geom.geom_type == "Polygon":
            polys = [_repair_geom(geom)]
        elif geom.geom_type == "MultiPolygon":
            polys = [_repair_geom(g) for g in geom.geoms]
        else:
            polys = [_repair_geom(g) for g in geom.geoms
                     if g.geom_type == "Polygon"] if hasattr(geom, "geoms") else []

        # For MultiPolygon features merge into one polygon for subdivision
        if len(polys) > 1:
            merged = unary_union(polys)
            polys = [_repair_geom(merged)]

        for p in polys:
            if p.area > 1e-6:
                results.append((feat_name, p))

    if not results:
        raise ValueError("No polygon geometries found in the selected shapefile.")

    return results, shp_candidates


def group_e(file_or_df, crs, out, mapping=None, e_mode="A", n_compartments=4,
            is_zip=False):
    """
    Group E — Polygon Subdivider.

    Accepts either:
      • CSV/Excel DataFrame  (is_zip=False) — build polygon from XY boundary points
      • ZIP file             (is_zip=True)  — read polygon directly from shapefile

    e_mode="A"  →  Single-forest / single-polygon
        ZIP:   subdivides each selected polygon feature independently;
               if multiple features exist, each gets its own output folder.
        CSV:   all rows → one boundary polygon → subdivided.

    e_mode="B"  →  Multi-forest (CSV only)
        Each Forest group subdivided independently into its own folder.
        (For ZIP input e_mode is forced to "A" — each feature is already one polygon.)
    """
    if n_compartments < 2:
        raise ValueError("Number of compartments must be at least 2.")
    if n_compartments > 200:
        raise ValueError("Number of compartments cannot exceed 200.")

    all_poly_gdfs, all_line_gdfs, all_pts_gdfs = [], [], []

    # ── ZIP / Shapefile input ─────────────────────────────────────────────────
    if is_zip:
        target_shp = (mapping or {}).get("target_shp")
        features, _ = _load_polygons_from_zip(file_or_df, target_shp, crs)

        if len(features) == 1:
            # Single polygon — save directly in run output folder
            feat_name, poly = features[0]
            pieces = _subdivide_polygon(poly, n_compartments)
            pg, lg, ptg = _save_compartments(pieces, feat_name, crs, out)
            all_poly_gdfs.append(pg)
            all_line_gdfs.append(lg)
            all_pts_gdfs.append(ptg)
        else:
            # Multiple features — one subfolder per feature
            for feat_name, poly in features:
                pieces     = _subdivide_polygon(poly, n_compartments)
                feat_dir   = os.path.join(out, _safe_dirname(feat_name))
                pg, lg, ptg = _save_compartments(pieces, feat_name, crs, feat_dir)
                all_poly_gdfs.append(pg)
                all_line_gdfs.append(lg)
                all_pts_gdfs.append(ptg)

    # ── CSV / Excel input ─────────────────────────────────────────────────────
    else:
        df = file_or_df
        df = normalize_order(df)

        x_col      = safe_col(df, mapping, "X",      "X")
        y_col      = safe_col(df, mapping, "Y",      "Y")
        order_col  = safe_col(df, mapping, "Order",  "Order")
        forest_col = safe_col(df, mapping, "Forest", "Forest")

        if not x_col: raise ValueError("Could not find an X / Easting / Longitude column.")
        if not y_col: raise ValueError("Could not find a Y / Northing / Latitude column.")
        if e_mode == "B" and not forest_col:
            raise ValueError("Multi-Forest mode requires a Forest column.")

        if e_mode == "A":
            forest_name = (mapping or {}).get("forest") or "FOREST"
            poly   = _df_to_polygon(df, x_col, y_col, order_col)
            pieces = _subdivide_polygon(poly, n_compartments)
            pg, lg, ptg = _save_compartments(pieces, forest_name, crs, out)
            all_poly_gdfs.append(pg)
            all_line_gdfs.append(lg)
            all_pts_gdfs.append(ptg)
        else:
            for f, fg in df.groupby(forest_col):
                try:
                    poly = _df_to_polygon(fg, x_col, y_col, order_col)
                except ValueError:
                    continue
                pieces     = _subdivide_polygon(poly, n_compartments)
                forest_dir = os.path.join(out, _safe_dirname(str(f)))
                pg, lg, ptg = _save_compartments(pieces, str(f), crs, forest_dir)
                all_poly_gdfs.append(pg)
                all_line_gdfs.append(lg)
                all_pts_gdfs.append(ptg)

    if not all_poly_gdfs:
        raise ValueError("No valid polygons could be built from the data.")

    poly_gdf = gpd.GeoDataFrame(pd.concat(all_poly_gdfs, ignore_index=True), crs=crs)
    line_gdf = gpd.GeoDataFrame(pd.concat(all_line_gdfs, ignore_index=True), crs=crs)
    pts_gdf  = gpd.GeoDataFrame(pd.concat(all_pts_gdfs,  ignore_index=True), crs=crs)

    return poly_gdf, line_gdf, pts_gdf


# ================= GROUP F — DEM SLOPE ANALYSIS =================
#
# Workflow:
#   1. Accept forest boundary (CSV/Excel XY points OR ZIP/SHP polygon)
#   2. Accept raw DEM GeoTIFF
#   3. Compute slope in degrees from DEM using numpy gradient
#   4. Clip slope raster to forest boundary (using rasterio.mask)
#   5. Classify into 3 classes: <19°, 19-31°, >31°
#   6. Calculate area per class (pixel count × pixel area in ha)
#   7. Export: clipped_slope.tif, slope_classes.tif, slope_summary.xlsx,
#              slope_polygon.shp (vectorised class polygons), preview PNG
#
# Required pip packages (add to requirements.txt):
#   rasterio, scipy

def _boundary_polygon_from_df(df, mapping):
    """Build a single WGS84-like polygon from XY boundary DataFrame."""
    df = normalize_order(df)
    x_col     = safe_col(df, mapping, "X",     "X")
    y_col     = safe_col(df, mapping, "Y",     "Y")
    order_col = safe_col(df, mapping, "Order", "Order")
    if not x_col: raise ValueError("Could not find X column for boundary.")
    if not y_col: raise ValueError("Could not find Y column for boundary.")
    if order_col: df = df.sort_values(order_col)
    coords = list(zip(df[x_col], df[y_col]))
    if len(coords) < 3: raise ValueError("Need at least 3 points for boundary polygon.")
    coords.append(coords[0])
    return safe_polygon(coords)


def _boundary_polygon_from_zip(zip_file, target_shp, src_crs, dem_crs):
    """
    Extract polygon from ZIP shapefile, reproject to DEM CRS.
    Returns a shapely Polygon and the reprojected GeoDataFrame.
    """
    folder = os.path.join(UPLOAD, str(uuid.uuid4()))
    os.makedirs(folder, exist_ok=True)
    zip_path = os.path.join(folder, "boundary.zip")
    zip_file.save(zip_path)
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(folder)

    shp_candidates = []
    for root, _, files in os.walk(folder):
        for f in files:
            if f.endswith(".shp"):
                shp_candidates.append(os.path.join(root, f))
    if not shp_candidates:
        raise ValueError("No .shp found in boundary ZIP.")

    shp_path = shp_candidates[0]
    if target_shp:
        for cand in shp_candidates:
            if os.path.basename(cand) == os.path.basename(target_shp):
                shp_path = cand; break

    gdf = gpd.read_file(shp_path)
    if gdf.empty: raise ValueError("Boundary shapefile has no features.")
    if gdf.crs is None: gdf = gdf.set_crs(src_crs)
    gdf = gdf.to_crs(dem_crs)
    union = gdf.unary_union
    return _repair_geom(union), gdf


def group_f(boundary_file, dem_file, crs, out, mapping=None,
            boundary_is_zip=False, forest_name="FOREST"):
    """
    Group F — DEM Slope Analysis & Clipping.

    Steps:
      1. Build/load forest boundary polygon
      2. Load DEM, compute slope in degrees
      3. Clip slope to boundary
      4. Classify: Class1 <19°, Class2 19-31°, Class3 >31°
      5. Vectorise classes → slope_polygon.shp
      6. Summary table → slope_summary.xlsx
      7. Return clipped slope array + class array for preview
    """
    try:
        import rasterio
        from rasterio.mask import mask as rio_mask
        from rasterio.features import shapes as rio_shapes
        import scipy.ndimage as ndimage
    except ImportError:
        raise ValueError(
            "Group F requires 'rasterio' and 'scipy'. "
            "Install with: pip install rasterio scipy"
        )

    os.makedirs(out, exist_ok=True)
    prefix = _safe_dirname(forest_name)

    # ── Step 1: Load DEM ────────────────────────────────────────────────────
    dem_path = os.path.join(UPLOAD, f"{uuid.uuid4()}_dem.tif")
    dem_file.save(dem_path)

    with rasterio.open(dem_path) as src:
        dem_crs    = src.crs
        dem_arr    = src.read(1).astype(np.float32)
        nodata     = src.nodata
        transform  = src.transform
        profile    = src.profile.copy()
        res_x      = abs(transform.a)   # pixel width  in CRS units
        res_y      = abs(transform.e)   # pixel height in CRS units

    # Replace nodata with NaN
    if nodata is not None:
        dem_arr[dem_arr == nodata] = np.nan

    # ── Step 2: Load / build boundary polygon in DEM CRS ───────────────────
    if boundary_is_zip:
        target_shp = (mapping or {}).get("target_shp")
        boundary_poly, boundary_gdf = _boundary_polygon_from_zip(
            boundary_file, target_shp, crs, str(dem_crs)
        )
    else:
        df = read_input(boundary_file)
        boundary_poly = _boundary_polygon_from_df(df, mapping)
        # Build GDF in the user-specified UTM CRS then reproject to DEM CRS
        boundary_gdf = gpd.GeoDataFrame(
            [{"Forest": forest_name, "geometry": boundary_poly}], crs=crs
        ).to_crs(str(dem_crs))
        boundary_poly = boundary_gdf.unary_union

    # ── Step 3: Compute slope in degrees ───────────────────────────────────
    # Use scipy Sobel filters for robust gradient on irregular grids
    # dz/dx and dz/dy in map units; divide by pixel size to get rise/run
    dzdx = ndimage.sobel(np.nan_to_num(dem_arr, nan=0.0), axis=1) / (8 * res_x)
    dzdy = ndimage.sobel(np.nan_to_num(dem_arr, nan=0.0), axis=0) / (8 * res_y)
    slope_arr = np.degrees(np.arctan(np.sqrt(dzdx**2 + dzdy**2))).astype(np.float32)

    # Mask NaN DEM areas back to nodata in slope
    if nodata is not None:
        slope_arr[np.isnan(dem_arr)] = nodata
    else:
        slope_arr[np.isnan(dem_arr)] = -9999
        nodata = -9999

    # ── Step 4: Save full slope raster then clip to boundary ───────────────
    slope_profile = profile.copy()
    slope_profile.update(dtype="float32", nodata=nodata, count=1)

    slope_full_path = os.path.join(out, f"{prefix}_slope_full.tif")
    with rasterio.open(slope_full_path, "w", **slope_profile) as dst:
        dst.write(slope_arr, 1)

    # Clip to boundary
    with rasterio.open(slope_full_path) as src:
        clipped_arr, clipped_transform = rio_mask(
            src,
            [boundary_poly.__geo_interface__],
            crop=True, filled=True, nodata=nodata
        )
    clipped_slope = clipped_arr[0].astype(np.float32)

    clipped_profile = slope_profile.copy()
    rows_c, cols_c = clipped_slope.shape
    clipped_profile.update(
        height=rows_c, width=cols_c,
        transform=clipped_transform,
    )

    clipped_path = os.path.join(out, f"{prefix}_slope_clipped.tif")
    with rasterio.open(clipped_path, "w", **clipped_profile) as dst:
        dst.write(clipped_slope, 1)

    # ── Step 5: Classify slope ─────────────────────────────────────────────
    # Class 0 = nodata, Class 1 = <19°, Class 2 = 19-31°, Class 3 = >31°
    # Use np.isclose for float nodata comparison to avoid float precision issues
    if nodata is not None:
        nodata_mask = np.isclose(clipped_slope, nodata, atol=1e-3) | np.isnan(clipped_slope)
    else:
        nodata_mask = np.isnan(clipped_slope)
    valid_mask = ~nodata_mask

    class_arr  = np.zeros_like(clipped_slope, dtype=np.uint8)
    class_arr[valid_mask & (clipped_slope < 19)]                          = 1
    class_arr[valid_mask & (clipped_slope >= 19) & (clipped_slope <= 31)] = 2
    class_arr[valid_mask & (clipped_slope > 31)]                          = 3

    class_profile = clipped_profile.copy()
    class_profile.update(dtype="uint8", nodata=0, count=1)

    class_path = os.path.join(out, f"{prefix}_slope_classes.tif")
    with rasterio.open(class_path, "w", **class_profile) as dst:
        dst.write(class_arr, 1)

    # ── Step 6: Calculate area per class ───────────────────────────────────
    # Pixel area in m² then convert to ha
    pixel_area_m2 = abs(clipped_transform.a * clipped_transform.e)
    pixel_area_ha = pixel_area_m2 / 10000.0

    class_defs = [
        (1, "< 19°",   "Gentle",   "#2ecc71"),
        (2, "19 – 31°","Moderate", "#f39c12"),
        (3, "> 31°",   "Steep",    "#e74c3c"),
    ]

    summary_rows = []
    for cls_id, cls_label, cls_desc, _ in class_defs:
        count   = int(np.sum(class_arr == cls_id))
        area_ha = round(count * pixel_area_ha, 4)
        pct     = round(count / max(np.sum(valid_mask), 1) * 100, 2)
        summary_rows.append({
            "Class":       cls_id,
            "Slope_Range": cls_label,
            "Description": cls_desc,
            "Pixel_Count": count,
            "Area_ha":     area_ha,
            "Pct_Area":    pct,
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_path = os.path.join(out, f"{prefix}_slope_summary.xlsx")
    summary_df.to_excel(summary_path, index=False)

    # ── Step 7: Vectorise class raster → polygon shapefile ─────────────────
    # Pass valid_mask as the mask so nodata cells (0) are not vectorised,
    # which avoids thousands of tiny border slivers.
    vec_records = []
    with rasterio.open(class_path) as src:
        img        = src.read(1).astype(np.uint8)
        valid_uint = valid_mask.astype(np.uint8)   # 1=valid pixel, 0=nodata
        for shape_geom, shape_val in rio_shapes(img, mask=valid_uint,
                                                 transform=clipped_transform):
            cls_id = int(shape_val)
            if cls_id == 0:
                continue
            info    = {d[0]: d for d in class_defs}[cls_id]
            # shape_geom is a GeoJSON-like dict; coordinates[0] = exterior ring
            coords  = shape_geom["coordinates"][0]
            geom    = _repair_geom(Polygon(coords))
            if geom.is_empty or geom.area < 1e-10:
                continue
            area_ha = round(geom.area / 10000, 6)
            vec_records.append({
                "Class":       cls_id,
                "Slope_Range": info[1],
                "Descr":       info[2],
                "Area_ha":     area_ha,
                "geometry":    geom,
            })

    vec_crs = str(dem_crs)
    if vec_records:
        vec_gdf = gpd.GeoDataFrame(vec_records, crs=vec_crs)
        vec_path = os.path.join(out, f"{prefix}_slope_polygon.shp")
        vec_gdf.to_file(vec_path)
    else:
        vec_gdf = gpd.GeoDataFrame(columns=["geometry"], crs=vec_crs)

    # Also save the boundary itself
    boundary_gdf_out = boundary_gdf.copy()
    boundary_gdf_out.to_file(os.path.join(out, f"{prefix}_boundary_polygon.shp"))

    return clipped_slope, class_arr, clipped_transform, summary_rows, vec_gdf, nodata


def preview_slope(clipped_slope, class_arr, summary_rows, path, nodata):
    """
    Render a two-panel preview:
    Left:  clipped slope raster (greyscale gradient)
    Right: classified slope map with legend + area table
    """
    valid = (class_arr > 0)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), dpi=150)
    fig.patch.set_facecolor("white")

    # ── Left: raw slope ─────────────────────────────────────────────────────
    disp = np.where(valid, clipped_slope, np.nan)
    im = ax1.imshow(disp, cmap="terrain", interpolation="bilinear")
    plt.colorbar(im, ax=ax1, fraction=0.04, pad=0.03, label="Slope (degrees)")
    ax1.set_title("Clipped Slope Raster", fontsize=11, fontweight="bold", pad=8)
    ax1.axis("off")

    # ── Right: classified ───────────────────────────────────────────────────
    colors_map = {0: (1,1,1,0), 1: (0.18,0.8,0.44,1),
                  2: (0.95,0.61,0.07,1), 3: (0.91,0.29,0.24,1)}
    rgb = np.zeros((*class_arr.shape, 4), dtype=np.float32)
    for cls_id, rgba in colors_map.items():
        mask = class_arr == cls_id
        rgb[mask] = rgba

    ax2.imshow(rgb, interpolation="nearest")
    ax2.set_title("Slope Classification", fontsize=11, fontweight="bold", pad=8)
    ax2.axis("off")

    # Legend patches
    import matplotlib.patches as mpatches
    legend_items = [
        mpatches.Patch(facecolor="#2ecc71", label="< 19° (Gentle)"),
        mpatches.Patch(facecolor="#f39c12", label="19–31° (Moderate)"),
        mpatches.Patch(facecolor="#e74c3c", label="> 31° (Steep)"),
    ]
    ax2.legend(handles=legend_items, loc="lower left",
               fontsize=7, framealpha=0.85, title="Slope Class")

    # Area table inside right panel
    if summary_rows:
        table_data = [[r["Slope_Range"], r["Description"],
                       f"{r['Area_ha']} ha", f"{r['Pct_Area']}%"]
                      for r in summary_rows]
        tbl = ax2.table(
            cellText=table_data,
            colLabels=["Range", "Class", "Area (ha)", "%"],
            cellLoc="center", loc="upper right",
            bbox=[0.48, 0.72, 0.52, 0.27]
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(6.5)
        for (r, c), cell in tbl.get_celld().items():
            cell.set_edgecolor("#cccccc")
            if r == 0:
                cell.set_facecolor("#e8f5e9")
                cell.set_text_props(fontweight="bold")

    plt.tight_layout(pad=1.5)
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ================= PREVIEW =================
def preview(poly_gdf, line_gdf, pts_gdf, path, pc, lc, ptc,
            label_col=None, label_pts_gdf=None):
    """
    Render a preview PNG.
    label_col       : column name in label_pts_gdf (or pts_gdf) to annotate points with.
    label_pts_gdf   : separate GDF whose points get labelled (used for Group C SN labels).
    """
    fig, ax = plt.subplots(figsize=(8, 8), dpi=180)

    if poly_gdf is not None and not poly_gdf.empty:
        poly_gdf.plot(ax=ax, facecolor="none", edgecolor=pc, linewidth=1.2)

    if line_gdf is not None and not line_gdf.empty:
        line_gdf.plot(ax=ax, color=lc, linewidth=1.5)

    if pts_gdf is not None and not pts_gdf.empty:
        pts_gdf.plot(ax=ax, color=ptc, markersize=6, zorder=5)

    # ── Sample-plot SN labels (Group C) ──────────────────────────────────
    lbl_src = label_pts_gdf if label_pts_gdf is not None else pts_gdf
    if label_col and lbl_src is not None and not lbl_src.empty and label_col in lbl_src.columns:
        for _, row in lbl_src.iterrows():
            ax.annotate(
                str(int(row[label_col])),
                xy=(row.geometry.x, row.geometry.y),
                xytext=(0, 7),
                textcoords="offset points",
                ha="center", va="bottom",
                fontsize=5,
                fontweight="bold",
                color="black",
                path_effects=[
                    pe.Stroke(linewidth=1.8, foreground="white"),
                    pe.Normal()
                ],
                zorder=6,
            )

    # ── Compartment ID labels (Group E) ──────────────────────────────────
    if poly_gdf is not None and not poly_gdf.empty and "Comp_ID" in poly_gdf.columns:
        for _, row in poly_gdf.iterrows():
            cx = row.geometry.centroid.x
            cy = row.geometry.centroid.y
            area_txt = f"{row['Area_ha']:.2f} ha" if "Area_ha" in row else ""
            label = f"{row['Comp_ID']}\n{area_txt}"
            ax.annotate(
                label,
                xy=(cx, cy),
                ha="center", va="center",
                fontsize=5.5,
                fontweight="bold",
                color="black",
                path_effects=[
                    pe.Stroke(linewidth=2, foreground="white"),
                    pe.Normal()
                ],
                zorder=7,
            )

    plt.axis("off")
    fig.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ================= KMZ GENERATOR =================
def _gdf_to_kml_placemarks(gdf, style_id, name_col=None):
    """
    Convert a GeoDataFrame (already in WGS84) to KML Placemark XML strings.
    Supports Polygon, LineString, MultiPolygon, and Point geometry types.
    """
    lines = []
    for i, row in gdf.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue

        # Build a label from attributes
        if name_col and name_col in row.index and row[name_col]:
            label = str(row[name_col])
        elif "Comp_ID" in row.index and row["Comp_ID"]:
            parts = [str(row["Comp_ID"])]
            if "Area_ha" in row.index:
                parts.append(f"{row['Area_ha']} ha")
            label = " · ".join(parts)
        elif "Forest" in row.index and row["Forest"]:
            label = str(row["Forest"])
        else:
            label = f"Feature {i+1}"

        # Build description from all non-geometry attributes
        desc_parts = []
        for col in row.index:
            if col == "geometry": continue
            val = row[col]
            if val is not None and str(val) not in ("None", "nan", ""):
                desc_parts.append(f"{col}: {val}")
        description = " | ".join(desc_parts)

        def coords_str(coords):
            return " ".join(f"{x},{y},0" for x, y in coords)

        def polygon_kml(geom):
            outer = coords_str(list(geom.exterior.coords))
            rings = [f"<outerBoundaryIs><LinearRing><coordinates>{outer}</coordinates></LinearRing></outerBoundaryIs>"]
            for interior in geom.interiors:
                inner = coords_str(list(interior.coords))
                rings.append(f"<innerBoundaryIs><LinearRing><coordinates>{inner}</coordinates></LinearRing></innerBoundaryIs>")
            return f"<Polygon>{''.join(rings)}</Polygon>"

        if geom.geom_type == "Polygon":
            geo_kml = polygon_kml(geom)
        elif geom.geom_type == "MultiPolygon":
            parts = "".join(polygon_kml(g) for g in geom.geoms)
            geo_kml = f"<MultiGeometry>{parts}</MultiGeometry>"
        elif geom.geom_type == "LineString":
            geo_kml = f"<LineString><coordinates>{coords_str(list(geom.coords))}</coordinates></LineString>"
        elif geom.geom_type == "MultiLineString":
            parts = "".join(f"<LineString><coordinates>{coords_str(list(g.coords))}</coordinates></LineString>" for g in geom.geoms)
            geo_kml = f"<MultiGeometry>{parts}</MultiGeometry>"
        elif geom.geom_type == "Point":
            geo_kml = f"<Point><coordinates>{geom.x},{geom.y},0</coordinates></Point>"
        else:
            continue

        lines.append(
            f"<Placemark>"
            f"<name>{label}</name>"
            f"<description>{description}</description>"
            f"<styleUrl>#{style_id}</styleUrl>"
            f"{geo_kml}"
            f"</Placemark>"
        )
    return "\n".join(lines)


def generate_kmz(poly_gdf, line_gdf, pts_gdf, out_dir, run_id):
    """
    Build a KMZ (zipped KML) from the three output GeoDataFrames.
    Reprojects everything to WGS84 (EPSG:4326) first.
    Returns the URL path to serve the KMZ.
    """
    import zipfile as zf

    def to_wgs84(gdf):
        if gdf is None or gdf.empty:
            return None
        try:
            if gdf.crs is None:
                return None   # can't reproject without a CRS
            return gdf.to_crs("EPSG:4326")
        except Exception:
            return None

    poly_w = to_wgs84(poly_gdf)
    line_w = to_wgs84(line_gdf)
    pts_w  = to_wgs84(pts_gdf)

    # Compute centroid for GE camera position
    ref_gdf = poly_w if poly_w is not None and not poly_w.empty else line_w
    if ref_gdf is not None and not ref_gdf.empty:
        union   = ref_gdf.unary_union
        cx, cy  = union.centroid.x, union.centroid.y
        # Approximate altitude to fit the bounding box
        minx, miny, maxx, maxy = union.bounds
        span_deg = max(maxx - minx, maxy - miny)
        alt_m    = max(500, int(span_deg * 111_000 * 2))   # rough scale
    else:
        cx, cy, alt_m = 0, 0, 10000

    kml_styles = """
  <Style id="poly_style">
    <LineStyle><color>ff00ff00</color><width>2</width></LineStyle>
    <PolyStyle><color>4400cc00</color></PolyStyle>
  </Style>
  <Style id="line_style">
    <LineStyle><color>ff0000ff</color><width>2</width></LineStyle>
  </Style>
  <Style id="point_style">
    <IconStyle>
      <color>ff0000ff</color>
      <scale>0.8</scale>
      <Icon><href>http://maps.google.com/mapfiles/kml/shapes/placemark_circle.png</href></Icon>
    </IconStyle>
    <LabelStyle><scale>0.7</scale></LabelStyle>
  </Style>
"""

    folders = []

    if poly_w is not None and not poly_w.empty:
        placemarks = _gdf_to_kml_placemarks(poly_w, "poly_style", name_col="Forest")
        folders.append(f"<Folder><name>Polygons</name>{placemarks}</Folder>")

    if line_w is not None and not line_w.empty:
        placemarks = _gdf_to_kml_placemarks(line_w, "line_style", name_col="Forest")
        folders.append(f"<Folder><name>Lines</name>{placemarks}</Folder>")

    if pts_w is not None and not pts_w.empty:
        placemarks = _gdf_to_kml_placemarks(pts_w, "point_style", name_col="SN")
        folders.append(f"<Folder><name>Points</name>{placemarks}</Folder>")

    kml_body = "\n".join(folders)

    kml = f"""<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2">
<Document>
  <name>Elfak GIS Output</name>
  <LookAt>
    <longitude>{cx}</longitude>
    <latitude>{cy}</latitude>
    <altitude>0</altitude>
    <range>{alt_m}</range>
    <tilt>0</tilt>
    <heading>0</heading>
    <altitudeMode>relativeToGround</altitudeMode>
  </LookAt>
  {kml_styles}
  {kml_body}
</Document>
</kml>"""

    kmz_path = os.path.join(out_dir, "output.kmz")
    with zf.ZipFile(kmz_path, "w", zf.ZIP_DEFLATED) as kmz:
        kmz.writestr("doc.kml", kml.encode("utf-8"))

    # Return centroid info alongside URL so the frontend can build the GE link
    return {
        "url":   f"/outputs/{run_id}/output.kmz",
        "lat":   round(cy, 6),
        "lon":   round(cx, 6),
        "alt":   alt_m,
    }


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
        forest = request.form.get("forest") or (mapping or {}).get("forest") or "FOREST"

        run_id = str(uuid.uuid4())
        out    = os.path.join(OUTPUT, run_id)
        os.makedirs(out, exist_ok=True)
        crs = get_crs(zone)

        label_col      = None
        label_pts_gdf  = None

        if module == "B":
            df = read_input(file)
            poly, line, pts = group_b(df, crs, out, mapping)

        elif module == "C":
            poly, line, pts = group_c(file, crs, w, h, rows, cols, out, mode, mapping)
            # Pass SN labels for the preview
            label_col     = "SN"
            label_pts_gdf = pts

        elif module == "D":
            df = read_input(file)
            d_mode = request.form.get("d_mode", "A")
            poly, line, pts = group_d(df, crs, out, mapping, mode=d_mode)

        elif module == "E":
            e_mode         = request.form.get("e_mode", "A")
            n_compartments = int(request.form.get("n_compartments", 4))
            is_zip         = file.filename.lower().endswith(".zip")

            if mapping and "forest" not in mapping:
                mapping["forest"] = forest

            if is_zip:
                poly, line, pts = group_e(
                    file, crs, out, mapping,
                    e_mode=e_mode, n_compartments=n_compartments, is_zip=True
                )
            else:
                df = read_input(file)
                poly, line, pts = group_e(
                    df, crs, out, mapping,
                    e_mode=e_mode, n_compartments=n_compartments, is_zip=False
                )

        elif module == "F":
            dem_file_upload = request.files.get("dem_file")
            if not dem_file_upload:
                return jsonify({"error": "No DEM file uploaded. Please upload a GeoTIFF DEM."}), 400

            f_forest        = request.form.get("f_forest") or forest
            boundary_is_zip = file.filename.lower().endswith(".zip")

            clipped_slope, class_arr, clipped_transform, summary_rows, vec_gdf, nodata = group_f(
                boundary_file   = file,
                dem_file        = dem_file_upload,
                crs             = crs,
                out             = out,
                mapping         = mapping,
                boundary_is_zip = boundary_is_zip,
                forest_name     = f_forest,
            )

            preview_path = os.path.join(out, "output.png")
            preview_slope(clipped_slope, class_arr, summary_rows, preview_path, nodata)

            # Build minimal GDFs for KMZ using the vectorised slope polygons
            poly = vec_gdf if (vec_gdf is not None and not vec_gdf.empty) else gpd.GeoDataFrame()
            line = gpd.GeoDataFrame()
            pts  = gpd.GeoDataFrame()

            kmz_url = None
            try:
                kmz_url = generate_kmz(poly, line, pts, out, run_id)
            except Exception:
                pass

            return jsonify({
                "run_id":   run_id,
                "download": f"/download/{run_id}",
                "kmz_url":  kmz_url,
            })

        else:  # module == "A"
            df = read_input(file)
            poly, line, pts = group_a(df, forest, crs, out, mapping)

        preview_path = os.path.join(out, "output.png")
        preview(poly, line, pts, preview_path,
                pc="yellow", lc="black", ptc="red",
                label_col=label_col, label_pts_gdf=label_pts_gdf)

        # Generate KMZ for Google Earth preview
        kmz_url = None
        try:
            kmz_url = generate_kmz(poly, line, pts, out, run_id)
        except Exception:
            pass  # KMZ is optional — don't fail the whole request

        return jsonify({
            "run_id":   run_id,
            "download": f"/download/{run_id}",
            "kmz_url":  kmz_url,
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


# ================= KMZ SERVE =================
@app.route("/outputs/<run_id>/output.kmz")
def serve_kmz(run_id):
    folder = os.path.join(OUTPUT, run_id)
    kmz_path = os.path.join(folder, "output.kmz")
    if not os.path.exists(kmz_path):
        return jsonify({"error": "KMZ not found"}), 404
    return send_file(
        kmz_path,
        mimetype="application/vnd.google-earth.kmz",
        as_attachment=False,
        download_name="output.kmz"
    )


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
