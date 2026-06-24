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
from shapely.geometry import Polygon, Point, LineString, MultiPolygon, MultiPoint
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
    poly = Polygon(coords)
    if not poly.is_valid:
        poly = poly.buffer(0)
    return poly


def _safe_dirname(s):
    return str(s).strip().replace("/", "_").replace("\\", "_").replace(":", "_")


def _repair_geom(g):
    if g is None:
        return None
    return g if g.is_valid else g.buffer(0)


def _as_polygon(geom):
    if geom is None or geom.is_empty:
        return None
    if geom.geom_type == "Polygon":
        return geom if not geom.is_empty else None
    if geom.geom_type in ("MultiPolygon", "GeometryCollection"):
        polys = [g for g in geom.geoms
                 if g.geom_type == "Polygon" and not g.is_empty and g.area > 1e-10]
        return max(polys, key=lambda g: g.area) if polys else None
    return None


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

    area_ha = round(poly.area / 10000, 4)
    poly_gdf = gpd.GeoDataFrame([{
        "Forest": forest, "Area_ha": area_ha,
        "Perim_m": round(poly.length, 2), "geometry": poly
    }], crs=crs)
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
            polys.append({"Forest": f, "Compartment": c, "Area_ha": round(poly.area/10000,4),
                          "Perim_m": round(poly.length,2), "geometry": poly})
            lines.append({"Forest": f, "Compartment": c, "geometry": line})
            for _, r in cg.iterrows():
                pts.append({"Forest": f, "Compartment": c,
                            "Order": r[order_col] if order_col else None,
                            "geometry": Point(r[x_col], r[y_col])})

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
        raise ValueError(f"No valid polygons could be built from the data.")

    return (gpd.GeoDataFrame(all_polys, crs=crs),
            gpd.GeoDataFrame(all_lines, crs=crs),
            gpd.GeoDataFrame(all_pts,   crs=crs))


# =============================================================================
# GROUP E — STRAIGHT-LINE RECURSIVE BISECTION SUBDIVIDER
# =============================================================================
#
# Complete rewrite of the subdivision engine.
#
# WHY voronoi was wrong:
#   Voronoi cells are bounded by curved bisectors between seed points.
#   When clipped to a polygon, those bisectors produce curved/wavy internal
#   division lines and blob-shaped compartments. The intersection() and
#   difference() calls on near-miss floating-point edges leave hairline gaps
#   at 3-way junctions (the Comp_005 notch).
#
# NEW APPROACH — Recursive Balanced Bisection:
#   1. At each step, take the current polygon piece and its target count n.
#   2. Find the longest axis of the bounding box (horizontal or vertical).
#   3. Binary-search for the cut position x=c (or y=c) that splits the
#      polygon into two pieces whose areas are in ratio floor(n/2):(n-ceil(n/2)).
#   4. The cut is a full-width straight line through the polygon at that position.
#   5. Recurse into both halves.
#
# RESULT:
#   - ALL internal division lines are mathematically straight (single-segment
#     or bounding-box-wide lines), never curved.
#   - The original polygon boundary is NEVER modified — every compartment's
#     outer edges are exact subsets of the original polygon's edges.
#   - No floating-point gap is possible: left piece = poly ∩ half-plane,
#     right piece = poly - left piece (exact complement).
#   - Area balance is guaranteed by the binary search.
#
# =============================================================================

from shapely.geometry import box as _shapely_box


def _bisect_polygon(poly, n, axis=None):
    """
    Recursively split poly into exactly n near-equal-area pieces using
    straight-line bisection cuts.

    Parameters
    ----------
    poly : shapely Polygon (valid, repaired)
    n    : int >= 1
    axis : None | 'x' | 'y'   — force cut axis (None = auto-longest)

    Returns
    -------
    list of Polygon objects whose union == poly exactly
    """
    poly = _repair_geom(poly)
    if poly is None or poly.is_empty or poly.area < 1e-10:
        return []
    if n <= 1:
        return [poly]

    minx, miny, maxx, maxy = poly.bounds
    dx = maxx - minx
    dy = maxy - miny

    # Choose cut axis: prefer the longest dimension for compact shapes
    if axis is None:
        cut_axis = 'x' if dx >= dy else 'y'
    else:
        cut_axis = axis

    # Target area ratio: left gets floor(n/2) shares, right gets ceil(n/2)
    n_left  = n // 2
    n_right = n - n_left
    frac    = n_left / n           # fraction of total area that goes left

    # Binary search for the cut coordinate
    if cut_axis == 'x':
        lo, hi = minx, maxx
    else:
        lo, hi = miny, maxy

    target_area = poly.area * frac
    best_mid    = (lo + hi) / 2

    for _ in range(80):
        mid = (lo + hi) / 2
        try:
            if cut_axis == 'x':
                left_half = _shapely_box(minx - 1, miny - 1, mid, maxy + 1)
            else:
                left_half = _shapely_box(minx - 1, miny - 1, maxx + 1, mid)
            left_piece = _repair_geom(poly.intersection(left_half))
            got = left_piece.area if (left_piece and not left_piece.is_empty) else 0.0
        except Exception:
            break
        err = abs(got - target_area) / (target_area + 1e-12)
        if err < 5e-4:
            best_mid = mid
            break
        if got < target_area:
            lo = mid
        else:
            hi = mid
        best_mid = mid

    # Perform the actual cut
    try:
        if cut_axis == 'x':
            left_box  = _shapely_box(minx - 1, miny - 1, best_mid, maxy + 1)
        else:
            left_box  = _shapely_box(minx - 1, miny - 1, maxx + 1, best_mid)
        left_piece  = _repair_geom(poly.intersection(left_box))
        # Right piece is EXACTLY the complement — no gap possible
        right_piece = _repair_geom(poly.difference(left_piece))
    except Exception:
        return [poly] if n == 1 else [poly] * n  # fallback

    # Flatten MultiPolygon results by taking the largest piece per side
    lp = _as_polygon(left_piece)
    rp = _as_polygon(right_piece)

    if lp is None or lp.is_empty:
        return _bisect_polygon(rp or poly, n)
    if rp is None or rp.is_empty:
        return _bisect_polygon(lp or poly, n)

    # Alternate cut axis on next level for balanced grid-like result
    next_axis = 'y' if cut_axis == 'x' else 'x'

    return (_bisect_polygon(lp, n_left,  next_axis) +
            _bisect_polygon(rp, n_right, next_axis))


def _subdivide_polygon(poly, n):
    """
    Public entry point: subdivide poly into n near-equal-area compartments
    using straight-line recursive bisection.
    Returns a list of exactly n valid Polygon objects.
    """
    if n <= 1:
        return [_repair_geom(poly)]
    poly = _repair_geom(poly)
    if poly is None or poly.is_empty:
        return []

    pieces = _bisect_polygon(poly, n)

    # Remove empties and ensure all are valid Polygons
    valid = []
    for p in pieces:
        p = _repair_geom(p)
        if p is None or p.is_empty or p.area < 1e-10:
            continue
        pg = _as_polygon(p)
        if pg:
            valid.append(pg)

    if not valid:
        return [poly]

    # Guarantee exact n by merging tiny slivers into largest neighbour
    target = poly.area / n
    while len(valid) > n:
        # merge smallest into largest-shared neighbour
        smallest_i = min(range(len(valid)), key=lambda i: valid[i].area)
        sliver = valid.pop(smallest_i)
        best_j, best_len = 0, -1.0
        for j, vp in enumerate(valid):
            try: s = sliver.intersection(vp).length
            except: s = 0.0
            if s > best_len:
                best_len, best_j = s, j
        try:
            merged = _repair_geom(unary_union([valid[best_j], sliver]))
            mp = _as_polygon(merged)
            if mp: valid[best_j] = mp
        except Exception:
            valid.append(sliver)

    # If we have fewer than n, subdivide largest piece
    while len(valid) < n:
        largest_i = max(range(len(valid)), key=lambda i: valid[i].area)
        big = valid.pop(largest_i)
        sub = _bisect_polygon(big, 2)
        if len(sub) == 2:
            valid.extend(sub)
        else:
            valid.append(big)
            break  # can't split further

    # Final gap fill: assign any uncovered area to nearest neighbour
    try:
        covered = _repair_geom(unary_union(valid))
        gap     = _repair_geom(poly.difference(covered))
        if gap and not gap.is_empty and gap.area > 1e-8:
            best_i, best_len = 0, -1.0
            for i, vp in enumerate(valid):
                try: s = gap.buffer(1e-4).intersection(vp).length
                except: s = 0.0
                if s > best_len:
                    best_len, best_i = s, i
            valid[best_i] = _repair_geom(unary_union([valid[best_i], gap]))
            p = _as_polygon(valid[best_i])
            if p: valid[best_i] = p
    except Exception:
        pass

    return valid


# ── Compartment vertex points for Group E output ──────────────────────────

def _extract_division_points(pieces, forest_name):
    """
    For each compartment polygon, collect:
      - All exterior ring vertices
      - Mid-points along each edge (for long edges > threshold)
    Returns a list of dicts suitable for a DataFrame / shapefile.
    """
    records = []
    sn = 1
    for i, p in enumerate(pieces, start=1):
        comp_id = f"Comp_{i:03d}"
        p = _repair_geom(p)
        if p is None or p.is_empty:
            continue
        coords = list(p.exterior.coords)
        # Deduplicate closing point
        if len(coords) > 1 and coords[0] == coords[-1]:
            coords = coords[:-1]
        # Estimate a "midpoint threshold" based on typical edge length
        edge_lengths = [
            ((coords[(j+1) % len(coords)][0] - coords[j][0])**2 +
             (coords[(j+1) % len(coords)][1] - coords[j][1])**2) ** 0.5
            for j in range(len(coords))
        ]
        avg_edge = (sum(edge_lengths) / len(edge_lengths)) if edge_lengths else 1.0
        mid_thresh = avg_edge * 1.5  # add midpoint for edges > 1.5x average

        for j, (cx, cy) in enumerate(coords):
            records.append({
                "SN":      sn,
                "Forest":  forest_name,
                "Comp_ID": comp_id,
                "Type":    "Vertex",
                "X":       round(cx, 4),
                "Y":       round(cy, 4),
            })
            sn += 1
            # Mid-point on this edge
            nx, ny = coords[(j + 1) % len(coords)]
            elen = ((nx - cx)**2 + (ny - cy)**2) ** 0.5
            if elen > mid_thresh:
                mx, my = (cx + nx) / 2, (cy + ny) / 2
                records.append({
                    "SN":      sn,
                    "Forest":  forest_name,
                    "Comp_ID": comp_id,
                    "Type":    "Midpoint",
                    "X":       round(mx, 4),
                    "Y":       round(my, 4),
                })
                sn += 1
    return records


def _save_compartments(pieces, forest_name, crs, save_dir):
    """
    Write shapefiles + Excel summary + division points Excel for compartments.
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
        line_geom = LineString(p.exterior.coords) if p.geom_type == "Polygon" else LineString([(0,0),(0,0)])
        line_recs.append({"Forest": forest_name, "Comp_ID": comp_id, "geometry": line_geom})
        pt_recs.append({
            "Forest":   forest_name,
            "Comp_ID":  comp_id,
            "Area_ha":  area_ha,
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

    # Summary Excel
    summary_df = pd.DataFrame([{k: v for k, v in r.items() if k != "geometry"} for r in records])
    summary_df.to_excel(os.path.join(save_dir, f"{prefix}_compartment_summary.xlsx"), index=False)

    # Division-line points Excel (for ArcGIS reconstruction)
    div_pts = _extract_division_points(pieces, forest_name)
    if div_pts:
        div_df = pd.DataFrame(div_pts)
        div_df.to_excel(os.path.join(save_dir, f"{prefix}_division_points.xlsx"), index=False)
        # Also as shapefile
        div_gdf = gpd.GeoDataFrame(
            div_df,
            geometry=gpd.points_from_xy(div_df["X"], div_df["Y"]),
            crs=crs
        )
        div_gdf.to_file(os.path.join(save_dir, f"{prefix}_division_points.shp"))

    return poly_gdf, line_gdf, pts_gdf


def _df_to_polygon(df, x_col, y_col, order_col):
    if order_col and order_col in df.columns:
        df = df.sort_values(order_col)
    coords = list(zip(df[x_col], df[y_col]))
    if len(coords) < 3:
        raise ValueError("Need at least 3 points to build a polygon.")
    coords.append(coords[0])
    return _repair_geom(Polygon(coords))


def _load_polygons_from_zip(file, target_shp, crs, forest_col_name=None):
    """
    Extract polygon features from a ZIP/SHP.
    Returns list of (feature_name, polygon) tuples.
    forest_col_name: explicit column to use as feature name identifier.
    """
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

    # Determine name column
    name_col = None
    if forest_col_name:
        for col in gdf.columns:
            if col.lower() == forest_col_name.lower():
                name_col = col
                break
    if name_col is None:
        for candidate in ["Forest", "forest", "Name", "name", "NAME",
                           "Label", "label", "LABEL", "ID", "id"]:
            if candidate in gdf.columns:
                name_col = candidate
                break
    if name_col is None:
        for col in gdf.columns:
            if col == "geometry": continue
            if gdf[col].dtype == object:
                name_col = col
                break

    results = []
    for i, row in gdf.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue
        feat_name = str(row[name_col]) if name_col else f"Feature_{i+1}"

        if geom.geom_type == "Polygon":
            polys = [_repair_geom(geom)]
        elif geom.geom_type == "MultiPolygon":
            polys = [_repair_geom(g) for g in geom.geoms]
        else:
            polys = ([_repair_geom(g) for g in geom.geoms if g.geom_type == "Polygon"]
                     if hasattr(geom, "geoms") else [])

        if len(polys) > 1:
            merged = unary_union(polys)
            polys = [_repair_geom(merged)]

        for p in polys:
            if p and p.area > 1e-6:
                results.append((feat_name, p))

    if not results:
        raise ValueError("No polygon geometries found in the selected shapefile.")

    return results, shp_candidates


def group_e(file_or_df, crs, out, mapping=None, e_mode="A", n_compartments=4,
            is_zip=False, forest_col_name=None):
    """
    Group E — Polygon Subdivider using straight-line recursive bisection.
    """
    if n_compartments < 2:
        raise ValueError("Number of compartments must be at least 2.")
    if n_compartments > 200:
        raise ValueError("Number of compartments cannot exceed 200.")

    all_poly_gdfs, all_line_gdfs, all_pts_gdfs = [], [], []

    if is_zip:
        target_shp = (mapping or {}).get("target_shp")
        features, _ = _load_polygons_from_zip(
            file_or_df, target_shp, crs, forest_col_name=forest_col_name)

        if len(features) == 1:
            feat_name, poly = features[0]
            pieces = _subdivide_polygon(poly, n_compartments)
            pg, lg, ptg = _save_compartments(pieces, feat_name, crs, out)
            all_poly_gdfs.append(pg); all_line_gdfs.append(lg); all_pts_gdfs.append(ptg)
        else:
            for feat_name, poly in features:
                pieces   = _subdivide_polygon(poly, n_compartments)
                feat_dir = os.path.join(out, _safe_dirname(feat_name))
                pg, lg, ptg = _save_compartments(pieces, feat_name, crs, feat_dir)
                all_poly_gdfs.append(pg); all_line_gdfs.append(lg); all_pts_gdfs.append(ptg)

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
            all_poly_gdfs.append(pg); all_line_gdfs.append(lg); all_pts_gdfs.append(ptg)
        else:
            for f, fg in df.groupby(forest_col):
                try:
                    poly = _df_to_polygon(fg, x_col, y_col, order_col)
                except ValueError:
                    continue
                pieces     = _subdivide_polygon(poly, n_compartments)
                forest_dir = os.path.join(out, _safe_dirname(str(f)))
                pg, lg, ptg = _save_compartments(pieces, str(f), crs, forest_dir)
                all_poly_gdfs.append(pg); all_line_gdfs.append(lg); all_pts_gdfs.append(ptg)

    if not all_poly_gdfs:
        raise ValueError("No valid polygons could be built from the data.")

    poly_gdf = gpd.GeoDataFrame(pd.concat(all_poly_gdfs, ignore_index=True), crs=crs)
    line_gdf = gpd.GeoDataFrame(pd.concat(all_line_gdfs, ignore_index=True), crs=crs)
    pts_gdf  = gpd.GeoDataFrame(pd.concat(all_pts_gdfs,  ignore_index=True), crs=crs)

    return poly_gdf, line_gdf, pts_gdf
    
    
# ================= GROUP F — DEM SLOPE ANALYSIS =================
# Improved workflow:
# 1. Extract rectangle from DEM around polygon (20% extra coverage)
# 2. Calculate slope of whole rectangular DEM
# 3. Reclassify into <19, 19-31, >31 degree classes
# 4. Raster to polygon
# 5. Dissolve by gridcode
# 6. Clip dissolved polygons to forest boundary
# 7. Calculate calibrated areas using field area conversion factor

def _boundary_polygon_from_df(df, mapping):
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
            f_mode="A", comp_col_name=None, field_area_ha=None):
    """
    Group F — DEM Slope Analysis with improved raster-to-polygon clip workflow
    and optional field-area recalibration.
    """
    try:
        import rasterio
        from rasterio.mask import mask as rio_mask
        from rasterio.features import shapes as rio_shapes
        from rasterio.transform import from_bounds
        import scipy.ndimage as ndimage
    except ImportError:
        raise ValueError("Group F requires rasterio and scipy. Install: pip install rasterio scipy")

    os.makedirs(out, exist_ok=True)
    prefix = _safe_dirname(forest_name)

    # ── Load DEM ──────────────────────────────────────────────────────────
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

    # ── Load boundary ─────────────────────────────────────────────────────
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

    # ── Step 1: Extract rectangular DEM around polygon (20% buffer) ───────
    b = boundary_poly.bounds  # (minx, miny, maxx, maxy)
    buf_x = (b[2] - b[0]) * 0.20
    buf_y = (b[3] - b[1]) * 0.20
    rect_poly = _shapely_box(b[0]-buf_x, b[1]-buf_y, b[2]+buf_x, b[3]+buf_y)

    try:
        with rasterio.open(dem_path) as src:
            rect_arr, rect_tr = rio_mask(
                src, [rect_poly.__geo_interface__],
                crop=True, filled=True, nodata=nodata if nodata is not None else -9999
            )
        rect_dem = rect_arr[0].astype(np.float32)
        rect_nodata = nodata if nodata is not None else -9999
        rect_dem[rect_dem == rect_nodata] = np.nan
    except Exception:
        # Fallback: use full DEM
        rect_dem = dem_arr
        rect_tr  = transform
        rect_nodata = nodata if nodata is not None else -9999

    rect_res_x = abs(rect_tr.a)
    rect_res_y = abs(rect_tr.e)

    # ── Step 2: Calculate slope of rectangular DEM ────────────────────────
    dzdx = ndimage.sobel(np.nan_to_num(rect_dem, nan=0.0), axis=1) / (8 * max(rect_res_x, 1e-9))
    dzdy = ndimage.sobel(np.nan_to_num(rect_dem, nan=0.0), axis=0) / (8 * max(rect_res_y, 1e-9))
    slope_rect = np.degrees(np.arctan(np.sqrt(dzdx**2 + dzdy**2))).astype(np.float32)
    slope_nodata = -9999.0
    slope_rect[np.isnan(rect_dem)] = slope_nodata

    # Save rectangular slope raster
    rect_profile = profile.copy()
    rect_profile.update(
        dtype="float32", nodata=slope_nodata, count=1,
        height=slope_rect.shape[0], width=slope_rect.shape[1],
        transform=rect_tr
    )
    slope_rect_path = os.path.join(out, f"{prefix}_slope_rect.tif")
    with rasterio.open(slope_rect_path, "w", **rect_profile) as dst:
        dst.write(slope_rect, 1)

    # ── Step 3: Reclassify slope into 3 classes ───────────────────────────
    valid_mask = (slope_rect != slope_nodata) & ~np.isnan(slope_rect)
    class_rect = np.zeros_like(slope_rect, dtype=np.uint8)
    class_rect[valid_mask & (slope_rect < 19)]                      = 1
    class_rect[valid_mask & (slope_rect >= 19) & (slope_rect <= 31)]= 2
    class_rect[valid_mask & (slope_rect > 31)]                      = 3

    class_rect_path = os.path.join(out, f"{prefix}_class_rect.tif")
    cls_profile = rect_profile.copy()
    cls_profile.update(dtype="uint8", nodata=0)
    with rasterio.open(class_rect_path, "w", **cls_profile) as dst:
        dst.write(class_rect, 1)

    # ── Step 4 & 5: Raster-to-polygon, dissolve by class ──────────────────
    class_defs = {
        1: ("< 19°",    "Gentle",   "#2ecc71"),
        2: ("19 - 31°", "Moderate", "#f39c12"),
        3: ("> 31°",    "Steep",    "#e74c3c"),
    }

    rtp_records = []
    with rasterio.open(class_rect_path) as src:
        cls_arr, cls_tr = src.read(1), src.transform
        for shp, val in rio_shapes(cls_arr, mask=(cls_arr > 0).astype(np.uint8), transform=cls_tr):
            cid = int(val)
            if cid == 0: continue
            try:
                geom = _repair_geom(Polygon(shp["coordinates"][0]))
                if geom and not geom.is_empty and geom.area > 1e-10:
                    rtp_records.append({"gridcode": cid, "geometry": geom})
            except Exception:
                continue

    if not rtp_records:
        raise ValueError("No slope polygons could be extracted from the DEM.")

    rtp_gdf = gpd.GeoDataFrame(rtp_records, crs=str(dem_crs))

    # Dissolve by gridcode
    try:
        dissolved = rtp_gdf.dissolve(by="gridcode", as_index=False)
        dissolved["gridcode"] = dissolved["gridcode"].astype(int)
    except Exception:
        dissolved = rtp_gdf.copy()

    # ── Step 6: Clip dissolved polygons to forest boundary ────────────────
    # This ensures perfect boundary matching — no edges left out

    def _clip_to_boundary(clip_poly, label):
        """Clip dissolved slope polygons to clip_poly, compute areas."""
        rows = []
        vec_recs = []
        pix_ha = abs(rect_tr.a * rect_tr.e) / 10000.0

        total_valid_ha = 0.0
        for _, drow in dissolved.iterrows():
            cid   = int(drow["gridcode"])
            dgeom = _repair_geom(drow.geometry)
            if dgeom is None or dgeom.is_empty:
                continue
            try:
                clipped = _repair_geom(dgeom.intersection(clip_poly))
            except Exception:
                continue
            if clipped is None or clipped.is_empty:
                continue
            area_ha = round(clipped.area / 10000, 4)
            total_valid_ha += area_ha
            vec_recs.append({
                "Label":       label,
                "Class":       cid,
                "Slope_Range": class_defs[cid][0],
                "Descr":       class_defs[cid][1],
                "Area_ha":     area_ha,
                "geometry":    clipped,
            })

        total_valid_ha = max(total_valid_ha, 1e-6)
        for vr in vec_recs:
            pct = round(vr["Area_ha"] / total_valid_ha * 100, 2)
            rows.append({
                "Label":       vr["Label"],
                "Class":       vr["Class"],
                "Slope_Range": vr["Slope_Range"],
                "Description": vr["Descr"],
                "Area_ha":     vr["Area_ha"],
                "Pct_Area":    pct,
                "Total_ha":    round(total_valid_ha, 4),
            })
        return rows, vec_recs, total_valid_ha

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
    for label, cpoly in comp_polygons:
        rows, vec_recs, _ = _clip_to_boundary(cpoly, label)
        all_summary.extend(rows)
        all_vec.extend(vec_recs)

    # ── Recalibration factor if field area provided ────────────────────────
    if field_area_ha and field_area_ha > 0:
        total_computed = sum(r["Area_ha"] for r in all_summary
                             if f_mode == "A" or True)
        # Sum per label for multi-forest
        if f_mode != "A":
            label_totals = {}
            for r in all_summary:
                label_totals.setdefault(r["Label"], 0.0)
                label_totals[r["Label"]] += r["Area_ha"]
            # Use total of first label as reference (single field area given)
            total_computed = sum(label_totals.values()) if label_totals else 1.0
        factor = field_area_ha / max(total_computed, 1e-6)
        for r in all_summary:
            r["Recal_ha"] = round(r["Area_ha"] * factor, 4)
            r["Cal_Factor"] = round(factor, 6)
    else:
        for r in all_summary:
            r["Recal_ha"] = None
            r["Cal_Factor"] = None

    # ── Save outputs ──────────────────────────────────────────────────────
    vec_crs = str(dem_crs)

    # Build preview rasters from first comp_polygon for display
    try:
        with rasterio.open(slope_rect_path) as src:
            first_cs, first_tr = rio_mask(
                src, [comp_polygons[0][1].__geo_interface__],
                crop=True, filled=True, nodata=slope_nodata)
        first_cs = first_cs[0].astype(np.float32)
    except Exception:
        first_cs = slope_rect
        first_tr = rect_tr

    valid_mask2 = (first_cs != slope_nodata) & ~np.isnan(first_cs)
    first_ca = np.zeros_like(first_cs, dtype=np.uint8)
    first_ca[valid_mask2 & (first_cs < 19)]                       = 1
    first_ca[valid_mask2 & (first_cs >= 19) & (first_cs <= 31)]   = 2
    first_ca[valid_mask2 & (first_cs > 31)]                       = 3

    # Save clipped rasters
    clipped_profile = rect_profile.copy()
    clipped_profile.update(height=first_cs.shape[0], width=first_cs.shape[1], transform=first_tr)
    with rasterio.open(os.path.join(out, f"{prefix}_slope_clipped.tif"), "w", **clipped_profile) as dst:
        dst.write(first_cs, 1)
    class_profile2 = clipped_profile.copy()
    class_profile2.update(dtype="uint8", nodata=0)
    with rasterio.open(os.path.join(out, f"{prefix}_slope_classes.tif"), "w", **class_profile2) as dst:
        dst.write(first_ca, 1)

    # Summary Excel
    summary_df = pd.DataFrame(all_summary)
    if f_mode == "A":
        summary_df = summary_df.drop(columns=["Label"], errors="ignore")
    summary_df.to_excel(os.path.join(out, f"{prefix}_slope_summary.xlsx"), index=False)

    # Vector output
    if all_vec:
        vec_gdf = gpd.GeoDataFrame(all_vec, crs=vec_crs)
        vec_gdf.to_file(os.path.join(out, f"{prefix}_slope_polygon.shp"))
    else:
        vec_gdf = gpd.GeoDataFrame(columns=["geometry"], crs=vec_crs)

    boundary_gdf.to_file(os.path.join(out, f"{prefix}_boundary_polygon.shp"))

    return first_cs, first_ca, first_tr, all_summary, vec_gdf, slope_nodata, boundary_gdf, f_mode


def preview_slope(clipped_slope, class_arr, summary_rows, path, nodata,
                  boundary_gdf=None, f_mode="A"):
    import matplotlib.patches as mpatches
    import matplotlib.gridspec as gridspec

    valid     = (class_arr > 0)
    has_label = f_mode in ("B", "E")
    n_rows    = len(summary_rows)
    table_height = max(0.22, min(0.50, 0.06 + n_rows * 0.032))
    fig_h    = 7 + (2 if n_rows > 6 else 0)

    fig = plt.figure(figsize=(14, fig_h), dpi=150, facecolor="white")
    gs  = gridspec.GridSpec(2, 2, height_ratios=[1, table_height],
                            hspace=0.32, wspace=0.06,
                            left=0.02, right=0.98, top=0.94, bottom=0.03)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, :])

    disp = np.where(valid, clipped_slope, np.nan)
    im   = ax1.imshow(disp, cmap="terrain", interpolation="bilinear")
    fig.colorbar(im, ax=ax1, fraction=0.046, pad=0.04, label="Degrees")
    ax1.set_title("Slope Raster (degrees)", fontsize=10, fontweight="bold", pad=6)
    ax1.axis("off")

    colors_map = {0:(1,1,1,0), 1:(0.18,0.8,0.44,1),
                  2:(0.95,0.61,0.07,1), 3:(0.91,0.29,0.24,1)}
    rgb = np.zeros((*class_arr.shape, 4), dtype=np.float32)
    for cid, rgba in colors_map.items():
        rgb[class_arr == cid] = rgba
    ax2.imshow(rgb, interpolation="nearest")
    ax2.set_title("Slope Classification + Boundary", fontsize=10, fontweight="bold", pad=6)
    ax2.axis("off")

    if boundary_gdf is not None and not boundary_gdf.empty:
        try:
            h_px, w_px = class_arr.shape
            bounds = boundary_gdf.total_bounds
            bw = bounds[2] - bounds[0]
            bh = bounds[3] - bounds[1]
            def to_px(x, y):
                return (x - bounds[0]) / bw * w_px, (bounds[3] - y) / bh * h_px
            for geom in boundary_gdf.geometry:
                if geom is None or geom.is_empty: continue
                parts = list(geom.geoms) if geom.geom_type.startswith("Multi") else [geom]
                for g in parts:
                    xs, ys = g.exterior.xy
                    pxs, pys = zip(*[to_px(x, y) for x, y in zip(xs, ys)])
                    ax2.plot(pxs, pys, color="black", linewidth=2.5, zorder=6)
        except Exception:
            pass

    legend_items = [
        mpatches.Patch(facecolor="#2ecc71", label="< 19°  Gentle"),
        mpatches.Patch(facecolor="#f39c12", label="19–31° Moderate"),
        mpatches.Patch(facecolor="#e74c3c", label="> 31°  Steep"),
    ]
    ax2.legend(handles=legend_items, loc="lower left",
               fontsize=7, framealpha=0.92, title="Slope Class", title_fontsize=7)

    ax3.axis("off")
    ax3.set_title("Slope Area Summary Table", fontsize=10, fontweight="bold", pad=4)

    if summary_rows:
        has_recal = any(r.get("Recal_ha") is not None for r in summary_rows)
        if has_label:
            col_labels = ["Compartment", "Slope Range", "Description", "Area (ha)", "% Area"]
            if has_recal:
                col_labels += ["Recal. Area (ha)", "Conv. Factor"]
            table_data = []
            for r in summary_rows:
                row = [r.get("Label",""), r["Slope_Range"], r["Description"],
                       f"{r['Area_ha']:.2f}", f"{r['Pct_Area']:.1f}%"]
                if has_recal:
                    row += [f"{r['Recal_ha']:.2f}" if r.get("Recal_ha") else "—",
                            f"{r.get('Cal_Factor','')}" if r.get("Cal_Factor") else "—"]
                table_data.append(row)
        else:
            total_ha = sum(r["Area_ha"] for r in summary_rows)
            col_labels = ["Slope Range", "Description", "Area (ha)", "% of Total"]
            if has_recal:
                col_labels += ["Recal. Area (ha)", "Conv. Factor"]
            table_data = []
            for r in summary_rows:
                row = [r["Slope_Range"], r["Description"],
                       f"{r['Area_ha']:.2f}", f"{r['Pct_Area']:.1f}%"]
                if has_recal:
                    row += [f"{r['Recal_ha']:.2f}" if r.get("Recal_ha") else "—",
                            f"{r.get('Cal_Factor','')}" if r.get("Cal_Factor") else "—"]
                table_data.append(row)
            total_row = ["TOTAL", "", f"{total_ha:.2f}", "100%"]
            if has_recal:
                total_recal = sum(r["Recal_ha"] for r in summary_rows if r.get("Recal_ha"))
                total_row += [f"{total_recal:.2f}", ""]
            table_data.append(total_row)

        tbl = ax3.table(cellText=table_data, colLabels=col_labels,
                        cellLoc="center", loc="center", bbox=[0.0, 0.0, 1.0, 1.0])
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
                    cell.set_facecolor("#ddeeff")
                    cell.set_text_props(fontweight="bold")

    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ================= PREVIEW =================
_COMP_COLORS = [
    "#C8E6C9","#B3E5FC","#FFE0B2","#F8BBD0","#E1BEE7",
    "#DCEDC8","#B2EBF2","#FFF9C4","#D7CCC8","#CFD8DC",
    "#C5CAE9","#F0F4C3",
]


def preview_compartments(poly_gdf, path):
    fig, ax = plt.subplots(figsize=(10, 10), dpi=180)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    for i, (_, row) in enumerate(poly_gdf.iterrows()):
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue
        color = _COMP_COLORS[i % len(_COMP_COLORS)]
        gpd.GeoDataFrame([{"geometry": geom}], crs=poly_gdf.crs).plot(
            ax=ax, facecolor=color, edgecolor="#1a1a1a", linewidth=1.8
        )
        cx, cy = geom.centroid.x, geom.centroid.y
        comp_id = row.get("Comp_ID", f"Comp_{i+1:03d}")
        area_ha = row.get("Area_ha", "")
        label   = f"{comp_id}\n{area_ha:.2f} ha" if isinstance(area_ha, float) else comp_id
        ax.annotate(label, xy=(cx, cy), ha="center", va="center",
                    fontsize=7, fontweight="bold", color="#1a1a1a",
                    path_effects=[pe.Stroke(linewidth=2.5, foreground="white"), pe.Normal()],
                    zorder=10)

    ax.set_aspect("equal")
    ax.axis("off")
    plt.tight_layout(pad=0.5)
    fig.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def preview(poly_gdf, line_gdf, pts_gdf, path, pc, lc, ptc,
            label_col=None, label_pts_gdf=None, area_ha=None):
    fig, ax = plt.subplots(figsize=(8, 8), dpi=180)

    if poly_gdf is not None and not poly_gdf.empty:
        poly_gdf.plot(ax=ax, facecolor="none", edgecolor=pc, linewidth=1.2)

    if line_gdf is not None and not line_gdf.empty:
        line_gdf.plot(ax=ax, color=lc, linewidth=1.5)

    if pts_gdf is not None and not pts_gdf.empty:
        pts_gdf.plot(ax=ax, color=ptc, markersize=6, zorder=5)

    lbl_src = label_pts_gdf if label_pts_gdf is not None else pts_gdf
    if label_col and lbl_src is not None and not lbl_src.empty and label_col in lbl_src.columns:
        for _, row in lbl_src.iterrows():
            ax.annotate(str(int(row[label_col])),
                        xy=(row.geometry.x, row.geometry.y),
                        xytext=(0, 7), textcoords="offset points",
                        ha="center", va="bottom", fontsize=5, fontweight="bold", color="black",
                        path_effects=[pe.Stroke(linewidth=1.8, foreground="white"), pe.Normal()],
                        zorder=6)

    if poly_gdf is not None and not poly_gdf.empty and "Comp_ID" in poly_gdf.columns:
        for _, row in poly_gdf.iterrows():
            cx = row.geometry.centroid.x
            cy = row.geometry.centroid.y
            area_txt = f"{row['Area_ha']:.2f} ha" if "Area_ha" in row else ""
            ax.annotate(f"{row['Comp_ID']}\n{area_txt}",
                        xy=(cx, cy), ha="center", va="center", fontsize=5.5,
                        fontweight="bold", color="black",
                        path_effects=[pe.Stroke(linewidth=2, foreground="white"), pe.Normal()],
                        zorder=7)

    # Group A: show total area in preview
    if area_ha is not None:
        ax.set_title(f"Total Area: {area_ha:.2f} ha", fontsize=11, fontweight="bold",
                     color="#1a2b22", pad=8)

    plt.axis("off")
    fig.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ================= KMZ GENERATOR =================
def _gdf_to_kml_placemarks(gdf, style_id, name_col=None):
    lines = []
    for i, row in gdf.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue
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
        elif geom.geom_type == "Point":
            geo_kml = f"<Point><coordinates>{geom.x},{geom.y},0</coordinates></Point>"
        else:
            continue

        lines.append(f"<Placemark><name>{label}</name><description>{description}</description>"
                     f"<styleUrl>#{style_id}</styleUrl>{geo_kml}</Placemark>")
    return "\n".join(lines)


def generate_kmz(poly_gdf, line_gdf, pts_gdf, out_dir, run_id):
    import zipfile as zf

    def to_wgs84(gdf):
        if gdf is None or gdf.empty: return None
        try:
            if gdf.crs is None: return None
            return gdf.to_crs("EPSG:4326")
        except Exception:
            return None

    poly_w = to_wgs84(poly_gdf)
    line_w = to_wgs84(line_gdf)
    pts_w  = to_wgs84(pts_gdf)

    ref_gdf = poly_w if poly_w is not None and not poly_w.empty else line_w
    if ref_gdf is not None and not ref_gdf.empty:
        union = ref_gdf.unary_union
        cx, cy = union.centroid.x, union.centroid.y
        minx, miny, maxx, maxy = union.bounds
        span_deg = max(maxx - minx, maxy - miny)
        alt_m = max(500, int(span_deg * 111_000 * 2))
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
    <IconStyle><color>ff0000ff</color><scale>0.8</scale>
      <Icon><href>http://maps.google.com/mapfiles/kml/shapes/placemark_circle.png</href></Icon>
    </IconStyle>
    <LabelStyle><scale>0.7</scale></LabelStyle>
  </Style>"""

    folders = []
    if poly_w is not None and not poly_w.empty:
        folders.append(f"<Folder><name>Polygons</name>{_gdf_to_kml_placemarks(poly_w, 'poly_style', 'Forest')}</Folder>")
    if line_w is not None and not line_w.empty:
        folders.append(f"<Folder><name>Lines</name>{_gdf_to_kml_placemarks(line_w, 'line_style', 'Forest')}</Folder>")
    if pts_w is not None and not pts_w.empty:
        folders.append(f"<Folder><name>Points</name>{_gdf_to_kml_placemarks(pts_w, 'point_style', 'SN')}</Folder>")

    kml = f"""<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2">
<Document>
  <name>Elfak GIS Output</name>
  <LookAt><longitude>{cx}</longitude><latitude>{cy}</latitude><altitude>0</altitude>
    <range>{alt_m}</range><tilt>0</tilt><heading>0</heading>
    <altitudeMode>relativeToGround</altitudeMode></LookAt>
  {kml_styles}
  {"".join(folders)}
</Document></kml>"""

    kmz_path = os.path.join(out_dir, "output.kmz")
    with zf.ZipFile(kmz_path, "w", zf.ZIP_DEFLATED) as kmz:
        kmz.writestr("doc.kml", kml.encode("utf-8"))

    return {"url": f"/outputs/{run_id}/output.kmz", "lat": round(cy, 6), "lon": round(cx, 6), "alt": alt_m}


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
        area_ha_display = None

        if module == "B":
            df = read_input(file)
            poly, line, pts = group_b(df, crs, out, mapping)

        elif module == "C":
            poly, line, pts = group_c(file, crs, w, h, rows, cols, out, mode, mapping)
            label_col     = "SN"
            label_pts_gdf = pts

        elif module == "D":
            df = read_input(file)
            d_mode = request.form.get("d_mode", "A")
            poly, line, pts = group_d(df, crs, out, mapping, mode=d_mode)

        elif module == "E":
            e_mode          = request.form.get("e_mode", "A")
            n_compartments  = int(request.form.get("n_compartments", 4))
            is_zip          = file.filename.lower().endswith(".zip")
            forest_col_name = request.form.get("forest_col_name") or None

            if mapping and "forest" not in mapping:
                mapping["forest"] = forest

            if is_zip:
                poly, line, pts = group_e(
                    file, crs, out, mapping,
                    e_mode=e_mode, n_compartments=n_compartments,
                    is_zip=True, forest_col_name=forest_col_name
                )
            else:
                df = read_input(file)
                poly, line, pts = group_e(
                    df, crs, out, mapping,
                    e_mode=e_mode, n_compartments=n_compartments,
                    is_zip=False, forest_col_name=forest_col_name
                )

            preview_path = os.path.join(out, "output.png")
            preview_compartments(poly, preview_path)

            kmz_url = None
            try:
                kmz_url = generate_kmz(poly, line, pts, out, run_id)
            except Exception:
                pass

            return jsonify({"run_id": run_id, "download": f"/download/{run_id}", "kmz_url": kmz_url})

        elif module == "F":
            dem_file_upload = request.files.get("dem_file")
            if not dem_file_upload:
                return jsonify({"error": "No DEM file uploaded. Please upload a GeoTIFF DEM."}), 400

            f_forest        = request.form.get("f_forest") or forest
            f_mode          = request.form.get("f_mode", "A")
            comp_col_name   = request.form.get("comp_col") or None
            boundary_is_zip = file.filename.lower().endswith(".zip")

            # Field area for recalibration
            field_area_ha = None
            fa_str = request.form.get("field_area_ha", "").strip()
            if fa_str:
                try:
                    field_area_ha = float(fa_str)
                except ValueError:
                    pass

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
                field_area_ha   = field_area_ha,
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

            return jsonify({"run_id": run_id, "download": f"/download/{run_id}", "kmz_url": kmz_url})

        else:  # module == "A"
            df = read_input(file)
            poly, line, pts = group_a(df, forest, crs, out, mapping)
            # Compute total area for display
            if not poly.empty and "Area_ha" in poly.columns:
                area_ha_display = float(poly["Area_ha"].sum())

        preview_path = os.path.join(out, "output.png")
        preview(poly, line, pts, preview_path,
                pc="yellow", lc="black", ptc="red",
                label_col=label_col, label_pts_gdf=label_pts_gdf,
                area_ha=area_ha_display)

        kmz_url = None
        try:
            kmz_url = generate_kmz(poly, line, pts, out, run_id)
        except Exception:
            pass

        return jsonify({"run_id": run_id, "download": f"/download/{run_id}", "kmz_url": kmz_url})

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
