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


def _chaikin_smooth(coords, iterations=2):
    """Chaikin corner-cutting on a closed ring. Returns smoothed coords."""
    pts = list(coords)
    if len(pts) > 1 and pts[0] == pts[-1]:
        pts = pts[:-1]
    for _ in range(iterations):
        new_pts = []
        n = len(pts)
        for i in range(n):
            p0 = pts[i]
            p1 = pts[(i + 1) % n]
            new_pts.append((0.75*p0[0] + 0.25*p1[0], 0.75*p0[1] + 0.25*p1[1]))
            new_pts.append((0.25*p0[0] + 0.75*p1[0], 0.25*p0[1] + 0.75*p1[1]))
        pts = new_pts
    pts.append(pts[0])
    return pts


def _best_cut(poly, target_area, n_tries=80):
    """
    Binary-search for a cut (horizontal or vertical) that gives a LEFT/BOTTOM
    piece of exactly target_area.  Tries both axes and picks the one that
    produces the most compact (lowest perimeter/area ratio) pieces.
    Returns (left_piece, right_piece) both guaranteed ⊆ poly with no gap.
    """
    from shapely.geometry import box as B

    minx, miny, maxx, maxy = poly.bounds
    total = poly.area

    best = None
    best_score = 1e18

    for axis in ("x", "y"):
        lo, hi = (minx, maxx) if axis == "x" else (miny, maxy)
        mid = (lo + hi) / 2.0

        for _ in range(n_tries):
            mid = (lo + hi) / 2.0
            try:
                if axis == "x":
                    left = poly.intersection(B(minx-1, miny-1, mid, maxy+1))
                else:
                    left = poly.intersection(B(minx-1, miny-1, maxx+1, mid))
            except Exception:
                break

            got = left.area if not left.is_empty else 0.0
            err = abs(got - target_area) / (target_area + 1e-10)
            if err < 0.0005:
                break
            elif got < target_area:
                lo = mid
            else:
                hi = mid

        # Right piece = exactly what's left — GUARANTEED no gap
        try:
            if axis == "x":
                left  = _repair_geom(poly.intersection(B(minx-1, miny-1, mid, maxy+1)))
                right = _repair_geom(poly.difference(left))
            else:
                left  = _repair_geom(poly.intersection(B(minx-1, miny-1, maxx+1, mid)))
                right = _repair_geom(poly.difference(left))
        except Exception:
            continue

        if left.is_empty or right.is_empty:
            continue

        # Score = total perimeter (lower = more compact shapes)
        score = left.length + right.length
        if score < best_score:
            best_score = score
            best = (left, right)

    return best if best else None


def _recursive_split(poly, n):
    """
    Split poly into exactly n pieces with equal area using recursive bisection.
    At each level, split into k and (n-k) pieces where k = n//2,
    choosing the cut that minimises total perimeter (most compact blocks).
    The RIGHT piece is always poly.difference(left) — guarantees zero gap.
    """
    if n <= 1:
        return [_repair_geom(poly)]
    if poly.is_empty or poly.area < 1e-10:
        return [poly]

    k     = n // 2          # left gets k pieces
    m     = n - k           # right gets m pieces
    # Target area for the left half
    target = poly.area * k / n

    cut = _best_cut(poly, target)
    if cut is None:
        return [poly] * n   # degenerate — return copies

    left_poly, right_poly = cut

    left_pieces  = _recursive_split(left_poly,  k)
    right_pieces = _recursive_split(right_poly, m)
    return left_pieces + right_pieces


def _smooth_internal_only(cell, boundary_line, poly):
    """
    Apply one pass of Chaikin smoothing ONLY to edges that do NOT lie on the
    original polygon boundary.  Boundary edges keep their exact vertices.
    The smoothed cell is then intersected with poly so it never overshoots,
    and the caller fills any residual gap separately.
    """
    try:
        if cell.geom_type != "Polygon" or len(cell.exterior.coords) < 4:
            return cell

        ext  = list(cell.exterior.coords)
        n    = len(ext) - 1     # last == first
        new  = []

        for i in range(n):
            p0 = ext[i]
            p1 = ext[(i+1) % n]
            seg = LineString([p0, p1])
            try:
                on_boundary = seg.intersection(boundary_line).length > seg.length * 0.05
            except Exception:
                on_boundary = True

            if on_boundary:
                new.append(p0)   # pin exactly
            else:
                new.append((0.75*p0[0]+0.25*p1[0], 0.75*p0[1]+0.25*p1[1]))
                new.append((0.25*p0[0]+0.75*p1[0], 0.25*p0[1]+0.75*p1[1]))

        if len(new) < 3:
            return cell
        new.append(new[0])
        s = _repair_geom(Polygon(new))
        # Clip to original polygon — boundary edges are pinned so overshoot
        # is only possible on internal curves; clip removes it cleanly.
        clipped = _repair_geom(s.intersection(poly))
        if clipped.is_empty or clipped.area < cell.area * 0.5:
            return cell
        return clipped
    except Exception:
        return cell


def _subdivide_polygon(poly, n):
    """
    Divide *poly* into *n* near-equal-area compartments.

    Pipeline
    ────────
    1. Recursive balanced bisection (guaranteed zero gaps, equal area)
    2. Internal-only Chaikin smoothing (boundary vertices pinned exactly)
    3. Gap fill — any residual area (from smoothing clip) given to the
       neighbour with the longest shared boundary
    """
    if n <= 1:
        return [_repair_geom(poly)]
    poly = _repair_geom(poly)
    if poly.is_empty or poly.area < 1e-10:
        return [poly]

    # ── 1. Recursive bisection → n raw pieces, zero gaps by construction ────
    pieces = _recursive_split(poly, n)
    if not pieces:
        return [poly]

    # ── 1b. Guarantee full coverage: rebuild using cumulative difference ─────
    # Even after recursive split, floating-point clipping can leave
    # micro-slivers between pieces.  Rebuild by assigning each piece
    # exactly what's left of the original polygon after removing all
    # previously-assigned pieces.
    rebuilt  = []
    remaining = _repair_geom(poly)   # starts as the full polygon

    # Sort pieces spatially (left→right or bottom→top) so rebuild order is stable
    pieces_sorted = sorted(pieces, key=lambda p: (p.centroid.x, p.centroid.y))

    for i, piece in enumerate(pieces_sorted):
        if i == len(pieces_sorted) - 1:
            # Last piece = everything not yet claimed → guaranteed no gap
            rebuilt.append(_repair_geom(remaining))
        else:
            # Clip piece to what's still available
            try:
                c = _repair_geom(piece.intersection(remaining))
            except Exception:
                c = piece
            if c.is_empty or c.area < 1e-8:
                continue
            rebuilt.append(c)
            try:
                remaining = _repair_geom(remaining.difference(c))
            except Exception:
                pass

    if not rebuilt:
        return [poly]
    pieces = rebuilt

    # ── 2. Smooth internal edges only ───────────────────────────────────────
    boundary_line = poly.boundary
    smoothed = [_smooth_internal_only(p, boundary_line, poly) for p in pieces]

    # ── 3. Gap fill ──────────────────────────────────────────────────────────
    # After smoothing clip, re-union all cells and compute residual gap.
    covered = None
    valid   = []
    for c in smoothed:
        c = _repair_geom(c)
        if not c.is_empty and c.area > 1e-8:
            valid.append(c)
            covered = c if covered is None else _repair_geom(covered.union(c))

    if not valid:
        return [poly]

    try:
        gap = _repair_geom(poly.difference(covered))
        if not gap.is_empty and gap.area > 1e-8:
            # Assign gap to neighbour with longest shared boundary
            best_i, best_len = 0, -1.0
            for i, c in enumerate(valid):
                try:
                    s = gap.intersection(c).length
                except Exception:
                    s = 0.0
                if s > best_len:
                    best_len, best_i = s, i
            valid[best_i] = _repair_geom(unary_union([valid[best_i], gap]))
    except Exception:
        pass

    return valid if valid else [poly]


def _cvt_lloyd(poly, seeds, iterations=60):
    """
    Lloyd's relaxation for Centroidal Voronoi Tessellation inside *poly*.
    Seeds are moved toward the centroid of their Voronoi cell (clipped to poly)
    so compartments become spatially compact and nearly equal in area.

    Returns list of (cell_polygon, seed_point) after convergence.
    """
    from shapely.ops import voronoi_diagram
    from shapely.geometry import MultiPoint, GeometryCollection

    boundary = poly.buffer(0)       # ensure valid
    pts = list(seeds)               # list of (x, y) tuples

    cells = []
    for iteration in range(iterations):
        mp     = MultiPoint([Point(x, y) for x, y in pts])
        # voronoi_diagram envelope must fully contain the polygon
        envelope = boundary.envelope.buffer(
            max(boundary.envelope.area ** 0.5 * 0.1, 1)
        )
        try:
            vd = voronoi_diagram(mp, envelope=envelope, tolerance=0.0)
        except Exception:
            break

        # Clip each Voronoi region to the forest boundary
        cells = []
        used_pts = []
        for geom in vd.geoms:
            try:
                cell = geom.intersection(boundary)
            except Exception:
                continue
            if cell.is_empty or cell.area < 1e-6:
                continue
            cells.append(_repair_geom(cell))
            used_pts.append(cell.centroid)

        if not cells:
            break

        # Move each seed to the centroid of its Voronoi cell
        new_pts = [(c.x, c.y) for c in used_pts]

        # Check convergence (max seed movement < 0.01% of bbox diagonal)
        if pts and new_pts and len(new_pts) == len(pts):
            diag = ((boundary.bounds[2] - boundary.bounds[0]) ** 2 +
                    (boundary.bounds[3] - boundary.bounds[1]) ** 2) ** 0.5
            max_move = max(
                ((a[0]-b[0])**2 + (a[1]-b[1])**2) ** 0.5
                for a, b in zip(pts, new_pts)
            )
            if max_move < diag * 0.0001:
                pts = new_pts
                break

        pts = new_pts

    return cells if cells else None


def _balance_areas(poly, cells, n, max_passes=8):
    """
    Area-balance pass: if any cell deviates > 3% from target area,
    move the shared boundary between large and small neighbours to equalise.
    Uses iterative shrink/grow via buffer on the dividing line.
    After balancing, clips all cells back to the original polygon and
    fills any coverage gaps so the total area is preserved.
    """
    from shapely.geometry import box as shapely_box

    if not cells or len(cells) < 2:
        return cells

    target = poly.area / n
    tolerance = target * 0.03    # 3 % tolerance

    for _ in range(max_passes):
        areas = [c.area for c in cells]
        max_dev = max(abs(a - target) for a in areas)
        if max_dev <= tolerance:
            break

        # Find the largest and smallest cell
        big_i = max(range(len(cells)), key=lambda i: cells[i].area)
        sml_i = min(range(len(cells)), key=lambda i: cells[i].area)
        if big_i == sml_i:
            break

        big = cells[big_i]
        sml = cells[sml_i]

        # Shared boundary between them
        try:
            shared = big.intersection(sml)
        except Exception:
            break
        if shared.is_empty or shared.length < 1e-3:
            break

        # Transfer a thin buffer strip from big → small
        transfer_area = (big.area - target) * 0.5
        strip_width   = transfer_area / (shared.length + 1e-10)
        strip_width   = min(strip_width, (big.area ** 0.5) * 0.1)

        try:
            strip = big.intersection(sml.buffer(strip_width))
        except Exception:
            break

        if strip.is_empty or strip.area < 1e-6:
            break

        try:
            cells[big_i] = _repair_geom(big.difference(strip))
            cells[sml_i] = _repair_geom(sml.union(strip))
        except Exception:
            break

    # Final clip to original polygon boundary + gap fill
    clipped = []
    covered = None
    for c in cells:
        try:
            c2 = _repair_geom(c.intersection(poly))
        except Exception:
            c2 = c
        if not c2.is_empty and c2.area > 1e-6:
            clipped.append(c2)
            covered = c2 if covered is None else covered.union(c2)

    # Fill any uncovered gap (rounding artefacts) into nearest cell
    if covered is not None:
        try:
            gap = _repair_geom(poly.difference(covered))
            if not gap.is_empty and gap.area > 1e-6:
                best_i, best_shared = 0, -1.0
                for i, c in enumerate(clipped):
                    try:
                        s = gap.intersection(c).length
                    except Exception:
                        s = 0.0
                    if s > best_shared:
                        best_shared, best_i = s, i
                clipped[best_i] = _repair_geom(clipped[best_i].union(gap))
        except Exception:
            pass

    return clipped if clipped else cells


def _place_seeds(poly, n):
    """
    Place n well-distributed seed points inside *poly* using a
    grid-jitter approach (fast and reliable for any polygon shape).
    Falls back to random interior points if grid yields too few.
    """
    minx, miny, maxx, maxy = poly.bounds
    dx = maxx - minx
    dy = maxy - miny

    # Try a dense grid first
    cols = max(int(np.ceil((n * dx / max(dy, 1e-6)) ** 0.5)) + 2, 3)
    rows = max(int(np.ceil(n * dy / max(dx, 1e-6) / (cols / n))) + 2, 3)
    step_x = dx / cols
    step_y = dy / rows

    rng   = np.random.default_rng(42)
    seeds = []
    for r in range(rows):
        for c in range(cols):
            x = minx + (c + 0.5) * step_x + rng.uniform(-step_x * 0.3, step_x * 0.3)
            y = miny + (r + 0.5) * step_y + rng.uniform(-step_y * 0.3, step_y * 0.3)
            pt = Point(x, y)
            if poly.contains(pt):
                seeds.append((x, y))

    # If not enough, sample random interior points
    attempts = 0
    while len(seeds) < n and attempts < 50000:
        x = rng.uniform(minx, maxx)
        y = rng.uniform(miny, maxy)
        if poly.contains(Point(x, y)):
            seeds.append((x, y))
        attempts += 1

    # Return exactly n seeds (trim or duplicate last if needed)
    if len(seeds) >= n:
        # Pick n most spread-out seeds using farthest-point selection
        chosen = [seeds[0]]
        remaining = seeds[1:]
        while len(chosen) < n and remaining:
            farthest = max(
                remaining,
                key=lambda s: min(
                    (s[0]-c[0])**2 + (s[1]-c[1])**2 for c in chosen
                )
            )
            chosen.append(farthest)
            remaining.remove(farthest)
        return chosen
    else:
        # Duplicate last seed slightly perturbed
        while len(seeds) < n:
            s = seeds[-1]
            seeds.append((s[0] + 1e-3, s[1] + 1e-3))
        return seeds[:n]


def _subdivide_polygon(poly, n):
    """
    Subdivide *poly* into *n* near-equal-area compartments using:

      1. Centroidal Voronoi Tessellation (CVT) via Lloyd's relaxation
         → spatially compact, blob-shaped compartments (not strips)
      2. Area-balancing pass
         → equalises areas across all compartments (target ± 3%)
      3. Chaikin corner-cutting smoothing
         → smooth, natural-looking compartment boundaries
         (smoothed only for the output geometry; topology preserved)

    Works correctly for any n ≥ 2 on any valid polygon shape,
    including highly concave and irregular forest boundaries.
    """
    if n <= 1:
        return [_repair_geom(poly)]

    poly = _repair_geom(poly)
    if poly.is_empty or poly.area < 1e-10:
        return [poly]

    # ── Step 1: Place initial seeds ─────────────────────────────────────────
    seeds = _place_seeds(poly, n)

    # ── Step 2: CVT Lloyd relaxation ────────────────────────────────────────
    cells = _cvt_lloyd(poly, seeds, iterations=80)

    # Fallback if CVT fails completely (very thin/degenerate polygon)
    if not cells or len(cells) < 1:
        from shapely.geometry import box as shapely_box
        minx, miny, maxx, maxy = poly.bounds
        dx = maxx - minx
        dy = maxy - miny
        axis = "x" if dx >= dy else "y"
        pieces = []
        for k in range(1, n):
            target_cum = poly.area * k / n
            lo, hi = (minx, maxx) if axis == "x" else (miny, maxy)
            mid = (lo + hi) / 2.0
            for _ in range(60):
                mid = (lo + hi) / 2.0
                try:
                    if axis == "x":
                        cl = poly.intersection(shapely_box(minx-1, miny-1, mid, maxy+1))
                    else:
                        cl = poly.intersection(shapely_box(minx-1, miny-1, maxx+1, mid))
                    got = cl.area if not cl.is_empty else 0.0
                except Exception:
                    break
                if abs(got - target_cum) / (target_cum + 1e-10) < 0.001:
                    break
                if got < target_cum:
                    lo = mid
                else:
                    hi = mid
            pieces.append(mid)
        cells = []
        cuts = [(minx if axis == "x" else miny) - 1] + pieces + \
               [(maxx if axis == "x" else maxy) + 1]
        for i in range(n):
            try:
                if axis == "x":
                    s = poly.intersection(shapely_box(cuts[i], miny-1, cuts[i+1], maxy+1))
                else:
                    s = poly.intersection(shapely_box(minx-1, cuts[i], maxx+1, cuts[i+1]))
                if s and not s.is_empty and s.area > 1e-6:
                    cells.append(_repair_geom(s))
            except Exception:
                pass
        if not cells:
            return [poly]

    # ── Step 3: Gap-free tiling ──────────────────────────────────────────────
    # After CVT + area balancing, floating-point clipping leaves micro-gaps
    # between cells and between cells and the boundary.
    # Fix: rebuild from scratch using a coverage-difference approach:
    #   cell[0] = cells[0] ∩ poly
    #   cell[1] = cells[1] ∩ poly  minus  cell[0]
    #   cell[2] = cells[2] ∩ poly  minus  cell[0] ∪ cell[1]
    #   ...
    #   cell[n-1] = poly  minus  everything already claimed
    # This guarantees: (a) every cell ⊆ poly, (b) no overlaps, (c) no gaps.

    tiled   = []
    covered = None   # union of all cells placed so far

    for i, cell in enumerate(cells):
        if i == len(cells) - 1:
            # Last cell = whatever of poly hasn't been claimed yet
            try:
                if covered is not None:
                    piece = _repair_geom(poly.difference(covered))
                else:
                    piece = poly
            except Exception:
                piece = poly
        else:
            try:
                piece = _repair_geom(cell.intersection(poly))
                if covered is not None:
                    piece = _repair_geom(piece.difference(covered))
            except Exception:
                piece = _repair_geom(cell.intersection(poly))

        if piece.is_empty or piece.area < 1e-8:
            continue

        tiled.append(piece)
        covered = piece if covered is None else _repair_geom(covered.union(piece))

    if not tiled:
        return [poly]

    # ── Step 4: Merge slivers (< 5 % of target) into longest-shared neighbour
    target_area = poly.area / n
    min_area    = target_area * 0.05

    changed = True
    while changed and len(tiled) > 1:
        changed = False
        tiny_idx = next(
            (i for i, p in sorted(enumerate(tiled), key=lambda t: t[1].area)
             if p.area < min_area),
            None
        )
        if tiny_idx is None:
            break
        sliver = tiled.pop(tiny_idx)
        best_j, best_len = 0, -1.0
        for j, other in enumerate(tiled):
            try:
                s = sliver.intersection(other).length
            except Exception:
                s = 0.0
            if s > best_len:
                best_len, best_j = s, j
        try:
            tiled[best_j] = _repair_geom(unary_union([tiled[best_j], sliver]))
        except Exception:
            tiled.append(sliver)
        changed = True

    # ── Step 5: Smooth internal edges only, pin boundary vertices exactly ───
    # boundary_line = the original polygon outline.
    # Any edge of a compartment that lies ON boundary_line is kept exactly.
    # Only the internal cut edges (between compartments) get Chaikin-smoothed.
    # After smoothing we do NOT re-clip (which would create gaps).
    # Instead we assign any unclaimed residual to the nearest neighbour so the
    # total coverage = poly exactly.

    boundary_line = poly.boundary
    smoothed = []

    for cell in tiled:
        try:
            if cell.geom_type != "Polygon" or len(cell.exterior.coords) < 4:
                smoothed.append(cell)
                continue

            ext_coords = list(cell.exterior.coords)
            n_pts = len(ext_coords) - 1  # closed ring: last == first

            # Classify each edge as boundary (False) or internal (True)
            is_internal = []
            for i in range(n_pts):
                seg = LineString([ext_coords[i], ext_coords[(i + 1) % n_pts]])
                try:
                    shared_len = seg.intersection(boundary_line).length
                    # internal if less than 5% of the edge overlaps the boundary
                    is_internal.append(shared_len < seg.length * 0.05)
                except Exception:
                    is_internal.append(False)

            if not any(is_internal):
                # All edges on the boundary — no smoothing possible
                smoothed.append(cell)
                continue

            # Build new coordinate ring:
            # - boundary edges: emit the start vertex unchanged
            # - internal edges: emit two Chaikin subdivision vertices
            new_coords = []
            for i in range(n_pts):
                p0 = ext_coords[i]
                p1 = ext_coords[(i + 1) % n_pts]
                if is_internal[i]:
                    q = (0.75*p0[0] + 0.25*p1[0], 0.75*p0[1] + 0.25*p1[1])
                    r = (0.25*p0[0] + 0.75*p1[0], 0.25*p0[1] + 0.75*p1[1])
                    new_coords.append(q)
                    new_coords.append(r)
                else:
                    new_coords.append(p0)   # pin exactly on boundary

            if len(new_coords) < 3:
                smoothed.append(cell)
                continue

            new_coords.append(new_coords[0])  # close ring
            s_poly = _repair_geom(Polygon(new_coords))

            if s_poly.is_empty or s_poly.area < cell.area * 0.3:
                smoothed.append(cell)
            else:
                smoothed.append(s_poly)

        except Exception:
            smoothed.append(cell)

    # ── Step 6: Final gap-fill ───────────────────────────────────────────────
    # Smoothing internal edges can push vertices very slightly outside
    # the original polygon OR leave hair-thin gaps between cells.
    # Re-clip each cell to poly, then fill any residual gap into the
    # neighbour with the longest shared boundary.

    final = []
    covered2 = None
    for cell in smoothed:
        try:
            c = _repair_geom(cell.intersection(poly))
        except Exception:
            c = cell
        if not c.is_empty and c.area > 1e-8:
            final.append(c)
            covered2 = c if covered2 is None else _repair_geom(covered2.union(c))

    # Fill gap
    if covered2 is not None:
        try:
            gap = _repair_geom(poly.difference(covered2))
            if not gap.is_empty and gap.area > 1e-8 and final:
                best_i, best_shared = 0, -1.0
                for i, c in enumerate(final):
                    try:
                        s = gap.intersection(c).length
                    except Exception:
                        s = 0.0
                    if s > best_shared:
                        best_shared, best_i = s, i
                final[best_i] = _repair_geom(unary_union([final[best_i], gap]))
        except Exception:
            pass

    return final if final else [poly]


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
            boundary_is_zip=False, forest_name="FOREST",
            f_mode="A", comp_col_name=None):
    """
    Group F — DEM Slope Analysis.
    f_mode A = whole boundary, B = per-Forest group, E = per-Compartment group.
    """
    try:
        import rasterio
        from rasterio.mask import mask as rio_mask
        from rasterio.features import shapes as rio_shapes
        import scipy.ndimage as ndimage
    except ImportError:
        raise ValueError("Group F requires rasterio and scipy. Install: pip install rasterio scipy")

    os.makedirs(out, exist_ok=True)
    prefix = _safe_dirname(forest_name)

    # Load DEM
    dem_path = os.path.join(UPLOAD, f"{uuid.uuid4()}_dem.tif")
    dem_file.save(dem_path)
    with rasterio.open(dem_path) as src:
        dem_crs   = src.crs
        dem_arr   = src.read(1).astype(np.float32)
        nodata    = src.nodata
        transform = src.transform
        profile   = src.profile.copy()
        res_x     = abs(transform.a)
        res_y     = abs(transform.e)
    if nodata is not None:
        dem_arr[dem_arr == nodata] = np.nan

    # Load boundary
    if boundary_is_zip:
        target_shp = (mapping or {}).get("target_shp")
        boundary_poly, boundary_gdf = _boundary_polygon_from_zip(
            boundary_file, target_shp, crs, str(dem_crs))
    else:
        df = read_input(boundary_file)
        boundary_poly = _boundary_polygon_from_df(df, mapping)
        boundary_gdf = gpd.GeoDataFrame(
            [{"Forest": forest_name, "geometry": boundary_poly}], crs=crs
        ).to_crs(str(dem_crs))
        boundary_poly = boundary_gdf.unary_union

    # Compute slope
    dzdx = ndimage.sobel(np.nan_to_num(dem_arr, nan=0.0), axis=1) / (8 * res_x)
    dzdy = ndimage.sobel(np.nan_to_num(dem_arr, nan=0.0), axis=0) / (8 * res_y)
    slope_arr = np.degrees(np.arctan(np.sqrt(dzdx**2 + dzdy**2))).astype(np.float32)
    if nodata is not None:
        slope_arr[np.isnan(dem_arr)] = nodata
    else:
        slope_arr[np.isnan(dem_arr)] = -9999
        nodata = -9999

    slope_profile = profile.copy()
    slope_profile.update(dtype="float32", nodata=nodata, count=1)
    slope_full_path = os.path.join(out, f"{prefix}_slope_full.tif")
    with rasterio.open(slope_full_path, "w", **slope_profile) as dst:
        dst.write(slope_arr, 1)

    class_defs = [
        (1, "< 19°",    "Gentle",   "#2ecc71"),
        (2, "19 - 31°", "Moderate", "#f39c12"),
        (3, "> 31°",    "Steep",    "#e74c3c"),
    ]

    def _clip_classify(clip_poly, label):
        with rasterio.open(slope_full_path) as src:
            arr, tr = rio_mask(src, [clip_poly.__geo_interface__],
                               crop=True, filled=True, nodata=nodata)
        cs = arr[0].astype(np.float32)
        nd_mask = np.isclose(cs, nodata, atol=1e-3) | np.isnan(cs) if nodata is not None else np.isnan(cs)
        valid = ~nd_mask
        ca = np.zeros_like(cs, dtype=np.uint8)
        ca[valid & (cs < 19)]               = 1
        ca[valid & (cs >= 19) & (cs <= 31)] = 2
        ca[valid & (cs > 31)]               = 3
        pix_ha     = abs(tr.a * tr.e) / 10000.0
        total_v    = max(int(np.sum(valid)), 1)
        total_ha   = round(total_v * pix_ha, 4)
        rows = []
        for cls_id, cls_label, cls_desc, _ in class_defs:
            count   = int(np.sum(ca == cls_id))
            area_ha = round(count * pix_ha, 4)
            pct     = round(count / total_v * 100, 2)
            rows.append({"Label": label, "Class": cls_id, "Slope_Range": cls_label,
                         "Description": cls_desc, "Pixel_Count": count,
                         "Area_ha": area_ha, "Pct_Area": pct, "Total_ha": total_ha})
        valid_uint = valid.astype(np.uint8)
        vec_recs = []
        for shape_geom, shape_val in rio_shapes(ca, mask=valid_uint, transform=tr):
            cid = int(shape_val)
            if cid == 0: continue
            info  = {d[0]: d for d in class_defs}[cid]
            coords = shape_geom["coordinates"][0]
            geom   = _repair_geom(Polygon(coords))
            if geom.is_empty or geom.area < 1e-10: continue
            vec_recs.append({"Label": label, "Class": cid, "Slope_Range": info[1],
                             "Descr": info[2], "Area_ha": round(geom.area/10000,6),
                             "geometry": geom})
        return cs, ca, tr, rows, vec_recs, valid

    # Build compartment polygon list
    comp_polygons = []
    if f_mode == "A":
        comp_polygons = [(forest_name, boundary_poly)]
    else:
        grp_col = None
        if comp_col_name:
            for c in boundary_gdf.columns:
                if c.lower() == comp_col_name.lower():
                    grp_col = c; break
        if grp_col is None:
            grp_col = _find_col(boundary_gdf, _FOREST_ALIASES if f_mode == "B" else _COMPARTMENT_ALIASES)
        if grp_col is None:
            comp_polygons = [(forest_name, boundary_poly)]
        else:
            for val, grp in boundary_gdf.groupby(grp_col):
                comp_polygons.append((str(val), _repair_geom(grp.unary_union)))

    all_summary, all_vec = [], []
    first_cs = first_ca = first_tr = first_valid = None

    for label, cpoly in comp_polygons:
        cs, ca, tr, rows, vec_recs, valid = _clip_classify(cpoly, label)
        all_summary.extend(rows)
        all_vec.extend(vec_recs)
        if first_cs is None:
            first_cs, first_ca, first_tr, first_valid = cs, ca, tr, valid

    if len(comp_polygons) > 1:
        first_cs, first_ca, first_tr, _, _, first_valid = _clip_classify(boundary_poly, forest_name)

    # Save clipped rasters
    clipped_profile = slope_profile.copy()
    clipped_profile.update(height=first_cs.shape[0], width=first_cs.shape[1], transform=first_tr)
    with rasterio.open(os.path.join(out, f"{prefix}_slope_clipped.tif"), "w", **clipped_profile) as dst:
        dst.write(first_cs, 1)
    class_profile = clipped_profile.copy()
    class_profile.update(dtype="uint8", nodata=0, count=1)
    with rasterio.open(os.path.join(out, f"{prefix}_slope_classes.tif"), "w", **class_profile) as dst:
        dst.write(first_ca, 1)

    # Summary Excel
    summary_df = pd.DataFrame(all_summary)
    if f_mode == "A":
        summary_df = summary_df.drop(columns=["Label"], errors="ignore")
    summary_df.to_excel(os.path.join(out, f"{prefix}_slope_summary.xlsx"), index=False)

    # Vector polygons
    vec_crs = str(dem_crs)
    if all_vec:
        vec_gdf = gpd.GeoDataFrame(all_vec, crs=vec_crs)
        vec_gdf.to_file(os.path.join(out, f"{prefix}_slope_polygon.shp"))
    else:
        vec_gdf = gpd.GeoDataFrame(columns=["geometry"], crs=vec_crs)

    boundary_gdf.to_file(os.path.join(out, f"{prefix}_boundary_polygon.shp"))
    return first_cs, first_ca, first_tr, all_summary, vec_gdf, nodata, boundary_gdf, f_mode


def preview_slope(clipped_slope, class_arr, summary_rows, path, nodata,
                  boundary_gdf=None, f_mode="A"):
    """
    Preview layout:
      Row 1 (top): [Slope Raster] [Classified Map + boundary outline]
      Row 2 (bottom): Full-width area summary table
    Table is never on top of the maps.
    """
    import matplotlib.patches as mpatches
    import matplotlib.gridspec as gridspec

    valid = (class_arr > 0)
    has_label = f_mode in ("B", "E")

    # Taller figure when compartment table has many rows
    n_rows = len(summary_rows)
    table_height = max(0.22, min(0.45, 0.06 + n_rows * 0.032))
    fig_h = 7 + (2 if n_rows > 6 else 0)

    fig = plt.figure(figsize=(14, fig_h), dpi=150, facecolor="white")
    gs  = gridspec.GridSpec(
        2, 2,
        height_ratios=[1, table_height],
        hspace=0.32, wspace=0.06,
        left=0.02, right=0.98, top=0.94, bottom=0.03
    )
    ax1 = fig.add_subplot(gs[0, 0])   # slope raster
    ax2 = fig.add_subplot(gs[0, 1])   # classified map
    ax3 = fig.add_subplot(gs[1, :])   # full-width table

    # ── Panel 1: raw slope raster ────────────────────────────────────────
    disp = np.where(valid, clipped_slope, np.nan)
    im   = ax1.imshow(disp, cmap="terrain", interpolation="bilinear")
    fig.colorbar(im, ax=ax1, fraction=0.046, pad=0.04, label="Degrees")
    ax1.set_title("Slope Raster (degrees)", fontsize=10, fontweight="bold", pad=6)
    ax1.axis("off")

    # ── Panel 2: classified map + boundary ───────────────────────────────
    colors_map = {0:(1,1,1,0), 1:(0.18,0.8,0.44,1),
                  2:(0.95,0.61,0.07,1), 3:(0.91,0.29,0.24,1)}
    rgb = np.zeros((*class_arr.shape, 4), dtype=np.float32)
    for cid, rgba in colors_map.items():
        rgb[class_arr == cid] = rgba
    ax2.imshow(rgb, interpolation="nearest")
    ax2.set_title("Slope Classification + Boundary", fontsize=10, fontweight="bold", pad=6)
    ax2.axis("off")

    # Draw original boundary outline on classified panel
    if boundary_gdf is not None and not boundary_gdf.empty:
        try:
            h_px, w_px = class_arr.shape
            bounds = boundary_gdf.total_bounds   # [minx, miny, maxx, maxy]
            bw = bounds[2] - bounds[0]
            bh = bounds[3] - bounds[1]
            def to_px(x, y):
                px = (x - bounds[0]) / bw * w_px
                py = (bounds[3] - y) / bh * h_px
                return px, py
            for geom in boundary_gdf.geometry:
                if geom is None or geom.is_empty: continue
                parts = list(geom.geoms) if geom.geom_type.startswith("Multi") else [geom]
                for g in parts:
                    xs, ys = g.exterior.xy
                    pxs, pys = zip(*[to_px(x, y) for x, y in zip(xs, ys)])
                    ax2.plot(pxs, pys, color="black", linewidth=2.5, zorder=6)
                    for interior in g.interiors:
                        xi, yi = interior.xy
                        pi_x, pi_y = zip(*[to_px(x, y) for x, y in zip(xi, yi)])
                        ax2.plot(pi_x, pi_y, color="black", linewidth=1.5, zorder=6)
        except Exception:
            pass

    legend_items = [
        mpatches.Patch(facecolor="#2ecc71", label="< 19°  Gentle"),
        mpatches.Patch(facecolor="#f39c12", label="19–31° Moderate"),
        mpatches.Patch(facecolor="#e74c3c", label="> 31°  Steep"),
    ]
    ax2.legend(handles=legend_items, loc="lower left",
               fontsize=7, framealpha=0.92, title="Slope Class", title_fontsize=7)

    # ── Panel 3: full-width area summary table ────────────────────────────
    ax3.axis("off")
    ax3.set_title("Slope Area Summary Table", fontsize=10, fontweight="bold", pad=4)

    if summary_rows:
        total_ha = sum(r["Area_ha"] for r in summary_rows
                       if not has_label or r.get("Class") == 1 or True)
        # For compartment modes show per-compartment totals
        if has_label:
            col_labels = ["Compartment", "Slope Range", "Description",
                          "Pixel Count", "Area (ha)", "% of Compartment"]
            table_data = [
                [r.get("Label", ""), r["Slope_Range"], r["Description"],
                 f"{r['Pixel_Count']:,}", f"{r['Area_ha']:.2f}", f"{r['Pct_Area']:.1f}%"]
                for r in summary_rows
            ]
        else:
            total_ha = sum(r["Area_ha"] for r in summary_rows)
            col_labels = ["Slope Range", "Description",
                          "Pixel Count", "Area (ha)", "% of Total"]
            table_data = [
                [r["Slope_Range"], r["Description"],
                 f"{r['Pixel_Count']:,}", f"{r['Area_ha']:.2f}", f"{r['Pct_Area']:.1f}%"]
                for r in summary_rows
            ]
            # Add total row
            table_data.append(["TOTAL", "", "", f"{total_ha:.2f}", "100%"])

        tbl = ax3.table(
            cellText=table_data,
            colLabels=col_labels,
            cellLoc="center",
            loc="center",
            bbox=[0.0, 0.0, 1.0, 1.0]
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(7.5 if has_label else 8.5)
        tbl.auto_set_column_width(list(range(len(col_labels))))

        row_colors = {1: "#d5f5e3", 2: "#fdebd0", 3: "#fadbd8"}
        for (r, c), cell in tbl.get_celld().items():
            cell.set_edgecolor("#cccccc")
            if r == 0:
                cell.set_facecolor("#1a5276")
                cell.set_text_props(color="white", fontweight="bold")
            else:
                ri = r - 1
                if ri < len(summary_rows):
                    cid = summary_rows[ri].get("Class", 0)
                    cell.set_facecolor(row_colors.get(cid, "#f8f9fa"))
                else:
                    # Total row
                    cell.set_facecolor("#ddeeff")
                    cell.set_text_props(fontweight="bold")

    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ================= PREVIEW =================
# Distinct fill palette for compartments (up to 12, then cycles)
_COMP_COLORS = [
    "#C8E6C9","#B3E5FC","#FFE0B2","#F8BBD0","#E1BEE7",
    "#DCEDC8","#B2EBF2","#FFF9C4","#D7CCC8","#CFD8DC",
    "#C5CAE9","#F0F4C3",
]


def preview_compartments(poly_gdf, path):
    """
    Dedicated Group E preview:
    - Each compartment filled with a distinct colour
    - Shared/outer edges all drawn in the same black, linewidth=1.8
    - Comp_ID + area label centred in each compartment
    - No separate line layer drawn on top (avoids double/misaligned outline)
    - White background
    """
    fig, ax = plt.subplots(figsize=(10, 10), dpi=180)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    for i, (_, row) in enumerate(poly_gdf.iterrows()):
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue
        color = _COMP_COLORS[i % len(_COMP_COLORS)]
        # Draw filled compartment
        gpd.GeoDataFrame([{"geometry": geom}], crs=poly_gdf.crs).plot(
            ax=ax, facecolor=color, edgecolor="#1a1a1a", linewidth=1.8
        )
        # Label at centroid
        cx, cy = geom.centroid.x, geom.centroid.y
        comp_id = row.get("Comp_ID", f"Comp_{i+1:03d}")
        area_ha = row.get("Area_ha", "")
        label   = f"{comp_id}\n{area_ha:.2f} ha" if isinstance(area_ha, float) else comp_id
        ax.annotate(
            label,
            xy=(cx, cy),
            ha="center", va="center",
            fontsize=7, fontweight="bold",
            color="#1a1a1a",
            path_effects=[
                pe.Stroke(linewidth=2.5, foreground="white"),
                pe.Normal()
            ],
            zorder=10,
        )

    ax.set_aspect("equal")
    ax.axis("off")
    plt.tight_layout(pad=0.5)
    fig.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
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

            # Dedicated compartment preview — coloured fills, no separate line layer
            preview_path = os.path.join(out, "output.png")
            preview_compartments(poly, preview_path)

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

        elif module == "F":
            dem_file_upload = request.files.get("dem_file")
            if not dem_file_upload:
                return jsonify({"error": "No DEM file uploaded. Please upload a GeoTIFF DEM."}), 400

            f_forest        = request.form.get("f_forest") or forest
            f_mode          = request.form.get("f_mode", "A")
            comp_col_name   = request.form.get("comp_col") or None
            boundary_is_zip = file.filename.lower().endswith(".zip")

            result = group_f(
                boundary_file   = file,
                dem_file        = dem_file_upload,
                crs             = crs,
                out             = out,
                mapping         = mapping,
                boundary_is_zip = boundary_is_zip,
                forest_name     = f_forest,
                f_mode          = f_mode,
                comp_col_name   = comp_col_name,
            )
            clipped_slope, class_arr, _clipped_tr, summary_rows, vec_gdf, nodata, boundary_gdf, f_mode_out = result

            preview_path = os.path.join(out, "output.png")
            preview_slope(clipped_slope, class_arr, summary_rows, preview_path, nodata,
                          boundary_gdf=boundary_gdf, f_mode=f_mode_out)

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
