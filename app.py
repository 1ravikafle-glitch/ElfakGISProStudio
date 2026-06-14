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
from shapely.ops import split as shp_split, unary_union

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

    poly_gdf.to_file(os.path.join(out, "polygon.shp"))
    line_gdf.to_file(os.path.join(out, "line.shp"))
    pts_gdf.to_file(os.path.join(out, "points.shp"))
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
    poly_gdf.to_file(os.path.join(out, "boundary_polygons.shp"))
    line_gdf.to_file(os.path.join(out, "boundary_lines.shp"))
    if not pts_gdf.empty:
        pts_gdf.to_file(os.path.join(out, "sampleplot.shp"))
        pd.DataFrame(pts)[["SN", "X", "Y"]].to_excel(os.path.join(out, "sampleplot.xlsx"), index=False)

    return poly_gdf, line_gdf, pts_gdf


# ================= GROUP D =================
def _save_forest_layer(poly_rec, line_rec, pt_recs, save_dir, crs):
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


def _area_balanced_split(poly, n_tries=60):
    """
    Split poly into two near-equal-area halves using box-clipping.

    Instead of shp_split (which fails on concave / self-touching polygons),
    we clip the polygon with a half-bounding-box rectangle.  This is 100%
    robust for any valid polygon shape, including highly concave forests.

    Binary-searches the cut position along the longest bbox axis until both
    halves differ by < 0.5 % of total area.

    Returns a list of exactly 2 Polygon (or MultiPolygon) objects, or [poly]
    if clipping fails completely.
    """
    from shapely.geometry import box as shapely_box

    minx, miny, maxx, maxy = poly.bounds
    dx = maxx - minx
    dy = maxy - miny

    # Choose axis to cut along (longest dimension)
    if dx >= dy:
        axis = "x"; lo, hi = minx, maxx
    else:
        axis = "y"; lo, hi = miny, maxy

    target   = poly.area / 2.0
    best_lo  = None
    best_hi  = None

    for _ in range(n_tries):
        mid = (lo + hi) / 2.0

        # Clip boxes — slightly oversized to avoid boundary artefacts
        try:
            if axis == "x":
                clip_lo = poly.intersection(shapely_box(minx - 1, miny - 1, mid,      maxy + 1))
                clip_hi = poly.intersection(shapely_box(mid,      miny - 1, maxx + 1, maxy + 1))
            else:
                clip_lo = poly.intersection(shapely_box(minx - 1, miny - 1, maxx + 1, mid     ))
                clip_hi = poly.intersection(shapely_box(minx - 1, mid,      maxx + 1, maxy + 1))
        except Exception:
            break

        # Reject empty or near-zero pieces
        if clip_lo.is_empty or clip_hi.is_empty:
            if clip_lo.is_empty: lo = mid
            else:                hi = mid
            continue

        lo_area = clip_lo.area
        if abs(lo_area - target) / (target + 1e-10) < 0.005:
            best_lo, best_hi = clip_lo, clip_hi
            break
        elif lo_area < target:
            lo = mid
        else:
            hi = mid

        best_lo, best_hi = clip_lo, clip_hi

    if best_lo is None or best_hi is None or best_lo.is_empty or best_hi.is_empty:
        return [poly]

    def _extract_polys(geom):
        """Flatten a geometry into a list of individual Polygons."""
        geom = _repair_geom(geom)
        if geom.geom_type == "Polygon":
            return [geom] if geom.area > 1e-6 else []
        elif geom.geom_type in ("MultiPolygon", "GeometryCollection"):
            out = []
            for g in geom.geoms:
                if g.geom_type == "Polygon" and g.area > 1e-6:
                    out.append(_repair_geom(g))
            return out
        return []

    # Merge MultiPolygon fragments back into single geometries per side
    lo_polys = _extract_polys(best_lo)
    hi_polys = _extract_polys(best_hi)

    if not lo_polys or not hi_polys:
        return [poly]

    # Use unary_union so each side is one geometry (possibly MultiPolygon for
    # concave forests where the cut produces disconnected pieces)
    lo_geom = unary_union(lo_polys) if len(lo_polys) > 1 else lo_polys[0]
    hi_geom = unary_union(hi_polys) if len(hi_polys) > 1 else hi_polys[0]

    return [_repair_geom(lo_geom), _repair_geom(hi_geom)]


def _subdivide_polygon(poly, n):
    """
    Bisect *poly* recursively until we have exactly n near-equal sub-polygons.

    Key improvements over naive bisection:
    - Always splits the LARGEST remaining piece (greedy equalisation)
    - Tracks failed pieces separately — a failure on one piece does NOT
      stop the loop; it tries the next largest instead
    - Retries up to (n * 3) times to handle concave shapes that need
      multiple attempts before a clean split is found
    - Final pieces are area-sorted and truncated to exactly n
    """
    if n <= 1:
        return [_repair_geom(poly)]

    pieces  = [_repair_geom(poly)]
    failed  = set()          # indices of pieces that cannot be split further
    max_iter = n * 4         # safety cap

    for _ in range(max_iter):
        if len(pieces) >= n:
            break

        # Sort by area descending; skip already-failed ones
        candidates = sorted(
            [(i, p) for i, p in enumerate(pieces) if i not in failed],
            key=lambda t: t[1].area,
            reverse=True,
        )

        if not candidates:
            break  # every piece is unsplittable

        idx, biggest = candidates[0]
        halves = _area_balanced_split(biggest)

        if len(halves) < 2:
            failed.add(idx)
            continue

        # Replace the chosen piece with the two halves
        pieces.pop(idx)
        # Rebuild failed set with adjusted indices
        failed = {fi if fi < idx else fi - 1 for fi in failed}
        pieces.extend(halves)

    # Sort by area descending and return exactly n pieces
    pieces.sort(key=lambda g: g.area, reverse=True)
    return pieces[:n]


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
            "geometry": LineString(p.exterior.coords),
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

    poly_gdf.to_file(os.path.join(save_dir, "compartments.shp"))
    line_gdf.to_file(os.path.join(save_dir, "compartment_lines.shp"))
    pts_gdf.to_file( os.path.join(save_dir, "compartment_points.shp"))

    summary_df = pd.DataFrame([
        {k: v for k, v in r.items() if k != "geometry"} for r in records
    ])
    summary_df.to_excel(os.path.join(save_dir, "compartment_summary.xlsx"), index=False)

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

        else:  # module == "A"
            df = read_input(file)
            poly, line, pts = group_a(df, forest, crs, out, mapping)

        preview_path = os.path.join(out, "output.png")
        preview(poly, line, pts, preview_path,
                pc="yellow", lc="black", ptc="red",
                label_col=label_col, label_pts_gdf=label_pts_gdf)

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
