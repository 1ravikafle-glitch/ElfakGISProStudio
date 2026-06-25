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

from flask import Flask, request, jsonify, send_file, send_from_directory, render_template, session
from shapely.geometry import Polygon, Point, LineString, MultiPolygon, MultiPoint, box as shapely_box
from shapely.ops import unary_union

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "elfak-gis-engine-2025-secret")

UPLOAD = "uploads"
OUTPUT = "outputs"
USERS_FILE = "users.json"

os.makedirs(UPLOAD, exist_ok=True)
os.makedirs(OUTPUT, exist_ok=True)


# ================= USER STORE =================
def _load_users():
    if not os.path.exists(USERS_FILE):
        return {}
    try:
        with open(USERS_FILE, "r") as f:
            return json.load(f)
    except Exception:
        return {}

def _save_users(users):
    try:
        with open(USERS_FILE, "w") as f:
            json.dump(users, f, indent=2)
    except Exception:
        pass

def _user_exists(username):
    return username.strip() in _load_users()

def _create_user(username):
    username = username.strip()
    users = _load_users()
    if username in users:
        raise ValueError(f"Username '{username}' is already taken.")
    users[username] = {"username": username, "created_at": pd.Timestamp.now().isoformat(), "runs": []}
    _save_users(users)
    return users[username]

def _get_user(username):
    return _load_users().get(username.strip())

def _append_run(username, run_id, module, description=""):
    if not username:
        return
    users = _load_users()
    if username in users:
        users[username].setdefault("runs", []).append({
            "run_id": run_id, "module": module,
            "description": description,
            "timestamp": pd.Timestamp.now().isoformat()
        })
        users[username]["runs"] = users[username]["runs"][-50:]
        _save_users(users)

def _require_login():
    return session.get("username")


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

def _collect_polygons(geom):
    if geom is None or geom.is_empty:
        return []
    if geom.geom_type == "Polygon":
        return [geom] if geom.area > 1e-10 else []
    if geom.geom_type in ("MultiPolygon", "GeometryCollection"):
        out = []
        for g in geom.geoms:
            out.extend(_collect_polygons(g))
        return out
    return []


# ─────────────────────────────────────────────────────────────────────────────
# COLUMN ALIAS RESOLUTION
# ─────────────────────────────────────────────────────────────────────────────
def _norm(s):
    return "".join(c for c in str(s).lower() if c.isalnum())

_X_ALIASES = {"x","xcoord","xcoordinate","xcord","xcords","xcoords","east","easting","eastings","lon","long","longitude","lng","pointx","coordx","utme","utmx"}
_Y_ALIASES = {"y","ycoord","ycoordinate","ycord","ycords","ycoords","north","northing","northings","lat","latitude","pointy","coordy","utmn","utmy"}
_ORDER_ALIASES = {"order","id","sn","sno","serial","serialno","serialnumber","seq","sequence","index","rowid","fid","no","num","number","plotid","plotno","pointid","pointno","pid"}
_FOREST_ALIASES = {"forest","forestname","forestid","forestno","fname","forestblock","block"}
_COMPARTMENT_ALIASES = {"compartment","comp","compartmentno","compartmentid","compno","compid","section","sectionno"}

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
    alias_map = {"X":_X_ALIASES,"Y":_Y_ALIASES,"Order":_ORDER_ALIASES,"Forest":_FOREST_ALIASES,"Compartment":_COMPARTMENT_ALIASES}
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
    poly_gdf = gpd.GeoDataFrame([{"Forest": forest, "Area_ha": area_ha, "Perim_m": round(poly.length,2), "geometry": poly}], crs=crs)
    line_gdf = gpd.GeoDataFrame([{"Forest": forest, "geometry": line}], crs=crs)
    pts_gdf  = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df[x_col], df[y_col]), crs=crs)
    poly_gdf.to_file(os.path.join(out, f"{_safe_dirname(forest)}_polygon.shp"))
    line_gdf.to_file(os.path.join(out, f"{_safe_dirname(forest)}_line.shp"))
    pts_gdf.to_file(os.path.join(out, f"{_safe_dirname(forest)}_point.shp"))
    return poly_gdf, line_gdf, pts_gdf


# ================= GROUP B =================
def group_b(df, crs, out, mapping=None):
    df = normalize_order(df)
    x_col      = safe_col(df, mapping, "X", "X")
    y_col      = safe_col(df, mapping, "Y", "Y")
    order_col  = safe_col(df, mapping, "Order", "Order")
    forest_col = safe_col(df, mapping, "Forest", "Forest")
    comp_col   = safe_col(df, mapping, "Compartment", "Compartment")
    if not x_col: raise ValueError("Could not find an X / Easting / Longitude column.")
    if not y_col: raise ValueError("Could not find a Y / Northing / Latitude column.")
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
            polys.append({"Forest": f, "Compartment": c, "Area_ha": round(poly.area/10000,4), "Perim_m": round(poly.length,2), "geometry": poly})
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


# ================= GROUP C =================
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
                if f.endswith(".shp"): shp_candidates.append(os.path.join(root, f))
        if not shp_candidates: raise ValueError("No shapefile (.shp) found inside the ZIP archive.")
        shp_path = shp_candidates[0]
        target_shp = (mapping or {}).get("target_shp")
        if target_shp:
            for cand in shp_candidates:
                if os.path.basename(cand) == os.path.basename(target_shp): shp_path = cand; break
        gdf = gpd.read_file(shp_path)
        if gdf.empty: raise ValueError("The selected shapefile contains no features.")
        if gdf.crs is None: gdf = gdf.set_crs(crs)
        else: gdf = gdf.to_crs(crs)
        for geom in gdf.geometry:
            if geom is None or geom.is_empty: continue
            if geom.geom_type == "Polygon": polygons.append(geom if geom.is_valid else geom.buffer(0))
            elif geom.geom_type == "MultiPolygon":
                for part in geom.geoms: polygons.append(part if part.is_valid else part.buffer(0))
            elif hasattr(geom, "geoms"):
                for sub in geom.geoms:
                    if sub.geom_type == "Polygon": polygons.append(sub if sub.is_valid else sub.buffer(0))
        if not polygons: raise ValueError("No polygon geometries found in the shapefile.")
    else:
        df = read_input(file)
        df = normalize_order(df)
        x_col     = safe_col(df, mapping, "X", "X")
        y_col     = safe_col(df, mapping, "Y", "Y")
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
            for key, g in df.groupby(group_keys):
                if order_col: g = g.sort_values(order_col)
                coords = list(zip(g[x_col], g[y_col]))
                if len(coords) < 3: continue
                coords.append(coords[0])
                polygons.append(safe_polygon(coords))
            if not polygons: raise ValueError("No valid polygons could be built from the data.")
    if not polygons: raise ValueError("No valid polygons could be built from the input data.")
    poly_gdf = gpd.GeoDataFrame([{"geometry": p} for p in polygons], crs=crs)
    line_gdf = gpd.GeoDataFrame([{"geometry": LineString(p.exterior.coords)} for p in polygons], crs=crs)
    union = poly_gdf.unary_union
    minx, miny, _, _ = union.bounds
    pts = []
    sn = 1
    for r in range(rows):
        for c_idx in range(cols):
            x = minx + c_idx * w
            y = miny + r * h
            center = Point(x + w/2, y + h/2)
            if union.contains(center):
                pts.append({"SN": sn, "X": center.x, "Y": center.y, "geometry": center})
                sn += 1
    pts_gdf = gpd.GeoDataFrame(pts, crs=crs)
    poly_gdf.to_file(os.path.join(out, "boundary_polygon.shp"))
    line_gdf.to_file(os.path.join(out, "boundary_line.shp"))
    if not pts_gdf.empty:
        pts_gdf.to_file(os.path.join(out, "sampleplot_point.shp"))
        pd.DataFrame(pts)[["SN","X","Y"]].to_excel(os.path.join(out, "sampleplot.xlsx"), index=False)
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
    x_col      = safe_col(df, mapping, "X", "X")
    y_col      = safe_col(df, mapping, "Y", "Y")
    order_col  = safe_col(df, mapping, "Order", "Order")
    forest_col = safe_col(df, mapping, "Forest", "Forest")
    comp_col   = safe_col(df, mapping, "Compartment", "Compartment")
    if not x_col: raise ValueError("Could not find an X / Easting / Longitude column.")
    if not y_col: raise ValueError("Could not find a Y / Northing / Latitude column.")
    if not forest_col: raise ValueError("Could not find a Forest column.")
    if mode == "B" and not comp_col: raise ValueError("Segmented mode requires a Compartment column.")
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
                poly = safe_polygon(coords); line = LineString(coords)
                poly_rec = {"Forest": f, "Compartment": c, "Area_ha": round(poly.area/10000,4), "Perim_m": round(poly.length,4), "geometry": poly}
                line_rec = {"Forest": f, "Compartment": c, "geometry": line}
                pt_recs  = [{"Forest": f, "Compartment": c, "Order": r[order_col] if order_col else None, "geometry": Point(r[x_col], r[y_col])} for _, r in cg.iterrows()]
                _save_forest_layer(poly_rec, line_rec, pt_recs, os.path.join(forest_dir, _safe_dirname(c)), crs)
                all_polys.append(poly_rec); all_lines.append(line_rec); all_pts.extend(pt_recs)
        else:
            if order_col: fg = fg.sort_values(order_col)
            coords = list(zip(fg[x_col], fg[y_col]))
            if len(coords) < 3: skipped += 1; continue
            coords.append(coords[0])
            poly = safe_polygon(coords); line = LineString(coords)
            poly_rec = {"Forest": f, "Area_ha": round(poly.area/10000,4), "Perim_m": round(poly.length,4), "geometry": poly}
            line_rec = {"Forest": f, "geometry": line}
            pt_recs  = [{"Forest": f, "Order": r[order_col] if order_col else None, "geometry": Point(r[x_col], r[y_col])} for _, r in fg.iterrows()]
            _save_forest_layer(poly_rec, line_rec, pt_recs, forest_dir, crs)
            all_polys.append(poly_rec); all_lines.append(line_rec); all_pts.extend(pt_recs)
    if not all_polys:
        raise ValueError("No valid polygons could be built from the data.")
    return gpd.GeoDataFrame(all_polys, crs=crs), gpd.GeoDataFrame(all_lines, crs=crs), gpd.GeoDataFrame(all_pts, crs=crs)


# =============================================================================
# GROUP E — COMPACT SQUARE-ISH RECURSIVE BISECTION SUBDIVIDER
# =============================================================================
# Algorithm: At each recursion, choose the cut axis that produces the most
# SQUARE (compact) sub-pieces rather than always cutting the longest axis.
# This means we alternate X/Y cuts based on which axis makes the resulting
# pieces closest to square, preventing long thin strips.
#
# Square-ness metric: for a piece, compute aspect_ratio = long_side/short_side
# of its bounding box. We want this as close to 1.0 as possible.
#
# Division points: for each compartment, extract all boundary vertex coords
# plus midpoints on straight cut edges, exported as:
#   Forest Sub_compartment Vertices Table
#   S.N. | Sub_compar | x | y
# =============================================================================

def _bbox_aspect(poly):
    """Return bounding-box aspect ratio (>= 1.0). 1.0 = perfect square."""
    minx, miny, maxx, maxy = poly.bounds
    dx = max(maxx - minx, 1e-10)
    dy = max(maxy - miny, 1e-10)
    return max(dx, dy) / min(dx, dy)


def _bisect_one(poly, frac, axis):
    """
    Split poly along `axis` at position that gives `frac` of area on left/bottom.
    Returns (left_or_bottom_piece, right_or_top_piece).
    Both pieces are exact intersections — union == poly, no gaps.
    """
    poly = _repair_geom(poly)
    minx, miny, maxx, maxy = poly.bounds
    lo, hi = (minx, maxx) if axis == 'x' else (miny, maxy)
    target = poly.area * frac
    best   = (lo + hi) / 2.0

    for _ in range(80):
        mid = (lo + hi) / 2.0
        try:
            if axis == 'x':
                left_box = shapely_box(minx - 1, miny - 1, mid, maxy + 1)
            else:
                left_box = shapely_box(minx - 1, miny - 1, maxx + 1, mid)
            left_piece = _repair_geom(poly.intersection(left_box))
            got = left_piece.area if (left_piece and not left_piece.is_empty) else 0.0
        except Exception:
            break
        if abs(got - target) / (target + 1e-12) < 5e-5:
            best = mid
            break
        if got < target:
            lo = mid
        else:
            hi = mid
        best = mid

    try:
        if axis == 'x':
            left_box = shapely_box(minx - 1, miny - 1, best, maxy + 1)
        else:
            left_box = shapely_box(minx - 1, miny - 1, maxx + 1, best)
        left_piece  = _repair_geom(poly.intersection(left_box))
        # Right piece = exact difference → zero gap guaranteed
        right_piece = _repair_geom(poly.difference(left_piece))
    except Exception:
        return poly, None

    return left_piece, right_piece


def _score_cut(poly, frac, axis):
    """
    Score a proposed cut: lower is better (more square sub-pieces).
    Returns sum of bounding-box aspect ratios of the two resulting pieces.
    """
    try:
        left, right = _bisect_one(poly, frac, axis)
        score = 0.0
        for piece in (left, right):
            pg = _as_polygon(piece)
            if pg:
                score += _bbox_aspect(pg)
            else:
                score += 999.0
        return score
    except Exception:
        return 999.0


def _bisect_polygon(poly, n, _depth=0):
    """
    Recursively bisect poly into n near-equal-area compact pieces.

    At each step:
      1. Try both X and Y cuts at the balanced fraction.
      2. Choose whichever axis gives more square sub-pieces (lower aspect ratio sum).
      3. Split n into floor(n/2) and ceil(n/2), recurse.

    This produces grid-like, roughly square compartments instead of long strips.
    """
    poly = _repair_geom(poly)
    if poly is None or poly.is_empty or poly.area < 1e-10:
        return []
    if n <= 1:
        p = _as_polygon(poly)
        return [p] if p else [poly]

    n_left  = n // 2
    n_right = n - n_left
    frac    = n_left / n

    # Choose axis that gives most compact (square-ish) result
    score_x = _score_cut(poly, frac, 'x')
    score_y = _score_cut(poly, frac, 'y')
    axis    = 'x' if score_x <= score_y else 'y'

    left_geom, right_geom = _bisect_one(poly, frac, axis)

    def recurse(geom, count):
        if geom is None or geom.is_empty:
            return []
        pieces = _collect_polygons(geom)
        if not pieces:
            return []
        if len(pieces) == 1:
            return _bisect_polygon(pieces[0], count, _depth + 1)
        # MultiPolygon: distribute count by area proportion
        pieces.sort(key=lambda p: p.area, reverse=True)
        total_area = sum(p.area for p in pieces)
        result = []
        remaining = count
        for idx, piece in enumerate(pieces):
            if idx == len(pieces) - 1:
                sub_n = remaining
            else:
                sub_n = max(1, round(piece.area / total_area * count))
                sub_n = min(sub_n, remaining - (len(pieces) - idx - 1))
            result.extend(_bisect_polygon(piece, sub_n, _depth + 1))
            remaining -= sub_n
            if remaining <= 0:
                break
        return result

    return recurse(left_geom, n_left) + recurse(right_geom, n_right)


def _subdivide_polygon(poly, n):
    """
    Public entry: subdivide poly into n near-equal-area compact compartments.
    Returns list of n valid Polygon objects. Union == poly exactly.
    """
    if n <= 1:
        return [_repair_geom(poly)]
    poly = _repair_geom(poly)
    if poly is None or poly.is_empty:
        return []

    pieces = _bisect_polygon(poly, n)

    # Ensure all pieces are valid Polygons
    valid = []
    for p in pieces:
        p = _repair_geom(p) if p else None
        if p is None or p.is_empty or p.area < 1e-10:
            continue
        pg = _as_polygon(p)
        if pg:
            valid.append(pg)

    if not valid:
        return [poly]

    # Merge excess slivers
    target = poly.area / n
    while len(valid) > n:
        smallest_i = min(range(len(valid)), key=lambda i: valid[i].area)
        sliver = valid.pop(smallest_i)
        best_j, best_len = 0, -1.0
        for j, vp in enumerate(valid):
            try: s = sliver.intersection(vp).length
            except: s = 0.0
            if s > best_len: best_len, best_j = s, j
        try:
            merged = _repair_geom(unary_union([valid[best_j], sliver]))
            mp = _as_polygon(merged)
            if mp: valid[best_j] = mp
        except Exception:
            valid.append(sliver)

    # Subdivide largest if we have too few
    while len(valid) < n:
        largest_i = max(range(len(valid)), key=lambda i: valid[i].area)
        big = valid.pop(largest_i)
        sub = _bisect_polygon(big, 2)
        if len(sub) >= 2:
            valid.extend(sub[:2])
        else:
            valid.append(big)
            break

    # Fill any residual gap
    try:
        covered = _repair_geom(unary_union(valid))
        gap     = _repair_geom(poly.difference(covered))
        if gap and not gap.is_empty and gap.area > 1e-8:
            best_i, best_len = 0, -1.0
            for i, vp in enumerate(valid):
                try: s = gap.buffer(1e-4).intersection(vp).length
                except: s = 0.0
                if s > best_len: best_len, best_i = s, i
            valid[best_i] = _repair_geom(unary_union([valid[best_i], gap]))
            p = _as_polygon(valid[best_i])
            if p: valid[best_i] = p
    except Exception:
        pass

    return valid


# ── Division-line points for Group E export ─────────────────────────────────

def _extract_division_points(pieces, forest_name):
    """
    For each compartment polygon, extract all boundary vertices as a table:
      S.N. | Sub_compar | x | y
    Also extracts midpoints on long edges (> 1.5x avg edge length).
    Returns list of dicts for DataFrame output.
    """
    records = []
    sn = 1
    for i, p in enumerate(pieces, start=1):
        comp_id = f"C{i}"
        p = _repair_geom(p)
        if p is None or p.is_empty:
            continue
        coords = list(p.exterior.coords)
        # Remove duplicate closing coord
        if len(coords) > 1 and coords[0] == coords[-1]:
            coords = coords[:-1]
        if not coords:
            continue
        # Average edge length for midpoint threshold
        n_c = len(coords)
        edge_lengths = []
        for j in range(n_c):
            cx1, cy1 = coords[j]
            cx2, cy2 = coords[(j + 1) % n_c]
            edge_lengths.append(((cx2-cx1)**2 + (cy2-cy1)**2)**0.5)
        avg_edge = sum(edge_lengths) / len(edge_lengths) if edge_lengths else 1.0
        mid_thresh = avg_edge * 1.5

        for j, (cx, cy) in enumerate(coords):
            records.append({
                "S.N.":       sn,
                "Sub_compar": f"C{i}S{j+1}",
                "x":          round(cx, 2),
                "y":          round(cy, 2),
                "Forest":     forest_name,
                "Comp_ID":    comp_id,
                "Type":       "Vertex",
            })
            sn += 1
            # Add midpoint for long edges
            nx, ny = coords[(j + 1) % n_c]
            elen = edge_lengths[j]
            if elen > mid_thresh:
                mx, my = (cx + nx) / 2, (cy + ny) / 2
                records.append({
                    "S.N.":       sn,
                    "Sub_compar": f"C{i}M{j+1}",
                    "x":          round(mx, 2),
                    "y":          round(my, 2),
                    "Forest":     forest_name,
                    "Comp_ID":    comp_id,
                    "Type":       "Midpoint",
                })
                sn += 1
    return records


def _save_compartments(pieces, forest_name, crs, save_dir):
    """Write SHP + Excel summary + compartment boundary-vertices Excel."""
    os.makedirs(save_dir, exist_ok=True)
    total_area = sum(p.area for p in pieces)
    records, line_recs, pt_recs = [], [], []

    for i, p in enumerate(pieces, start=1):
        p = _repair_geom(p)
        comp_id  = f"Comp_{i:03d}"
        area_ha  = round(p.area / 10000, 4)
        perim_m  = round(p.length, 4)
        pct_area = round(p.area / total_area * 100, 2) if total_area > 0 else 0
        centroid = p.centroid
        records.append({"Forest": forest_name, "Comp_ID": comp_id, "Area_ha": area_ha,
                        "Perim_m": perim_m, "Pct_Area": pct_area, "geometry": p})
        line_geom = LineString(p.exterior.coords) if p.geom_type == "Polygon" else LineString([(0,0),(0,0)])
        line_recs.append({"Forest": forest_name, "Comp_ID": comp_id, "geometry": line_geom})
        pt_recs.append({"Forest": forest_name, "Comp_ID": comp_id, "Area_ha": area_ha,
                        "Pct_Area": pct_area, "geometry": centroid})

    poly_gdf = gpd.GeoDataFrame(records,   crs=crs)
    line_gdf = gpd.GeoDataFrame(line_recs, crs=crs)
    pts_gdf  = gpd.GeoDataFrame(pt_recs,   crs=crs)
    prefix   = _safe_dirname(forest_name)

    poly_gdf.to_file(os.path.join(save_dir, f"{prefix}_compartment_polygon.shp"))
    line_gdf.to_file(os.path.join(save_dir, f"{prefix}_compartment_line.shp"))
    pts_gdf.to_file( os.path.join(save_dir, f"{prefix}_compartment_point.shp"))

    # Summary Excel
    summary_df = pd.DataFrame([{k: v for k, v in r.items() if k != "geometry"} for r in records])
    summary_df.to_excel(os.path.join(save_dir, f"{prefix}_compartment_summary.xlsx"), index=False)

    # ── Compartment Boundary Vertices Excel ─────────────────────────────
    # Format: Forest Sub_compartment Vertices Table
    # S.N. | Sub_compar | x | y
    div_pts = _extract_division_points(pieces, forest_name)
    if div_pts:
        div_df = pd.DataFrame(div_pts)[["S.N.", "Sub_compar", "x", "y", "Forest", "Comp_ID", "Type"]]
        # Write with a header label row like the example
        xlsx_path = os.path.join(save_dir, f"{prefix}_compartment_vertices.xlsx")
        with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
            # Title row
            title_df = pd.DataFrame([[f"Forest Sub_compartment Vertices Table — {forest_name}"]])
            title_df.to_excel(writer, sheet_name="Vertices", index=False, header=False, startrow=0)
            # Actual data starting at row 2
            div_df.to_excel(writer, sheet_name="Vertices", index=True, header=True, startrow=2)
        # Also as SHP
        try:
            div_gdf = gpd.GeoDataFrame(
                div_df,
                geometry=gpd.points_from_xy(div_df["x"], div_df["y"]),
                crs=crs
            )
            div_gdf.to_file(os.path.join(save_dir, f"{prefix}_division_points.shp"))
        except Exception:
            pass

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
    folder = os.path.join(UPLOAD, str(uuid.uuid4()))
    os.makedirs(folder, exist_ok=True)
    zip_path = os.path.join(folder, "input.zip")
    file.save(zip_path)
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(folder)
    shp_candidates = []
    for root, _, files in os.walk(folder):
        for f in files:
            if f.endswith(".shp"): shp_candidates.append(os.path.join(root, f))
    if not shp_candidates:
        raise ValueError("No shapefile (.shp) found inside the ZIP archive.")
    shp_path = shp_candidates[0]
    if target_shp:
        for cand in shp_candidates:
            if os.path.basename(cand) == os.path.basename(target_shp): shp_path = cand; break
    gdf = gpd.read_file(shp_path)
    if gdf.empty: raise ValueError("The selected shapefile contains no features.")
    if gdf.crs is None: gdf = gdf.set_crs(crs)
    else: gdf = gdf.to_crs(crs)
    name_col = None
    if forest_col_name:
        for col in gdf.columns:
            if col.lower() == forest_col_name.lower(): name_col = col; break
    if name_col is None:
        for candidate in ["Forest","forest","Name","name","NAME","Label","label","ID","id"]:
            if candidate in gdf.columns: name_col = candidate; break
    if name_col is None:
        for col in gdf.columns:
            if col != "geometry" and gdf[col].dtype == object: name_col = col; break
    results = []
    for i, row in gdf.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty: continue
        feat_name = str(row[name_col]) if name_col else f"Feature_{i+1}"
        if geom.geom_type == "Polygon": polys = [_repair_geom(geom)]
        elif geom.geom_type == "MultiPolygon": polys = [_repair_geom(g) for g in geom.geoms]
        elif hasattr(geom, "geoms"): polys = [_repair_geom(g) for g in geom.geoms if g.geom_type == "Polygon"]
        else: polys = []
        if len(polys) > 1:
            merged = unary_union(polys)
            polys = [_repair_geom(merged)]
        for p in polys:
            if p and p.area > 1e-6: results.append((feat_name, p))
    if not results: raise ValueError("No polygon geometries found in the selected shapefile.")
    return results, shp_candidates


def group_e(file_or_df, crs, out, mapping=None, e_mode="A", n_compartments=4, is_zip=False, forest_col_name=None):
    if n_compartments < 2: raise ValueError("Number of compartments must be at least 2.")
    if n_compartments > 200: raise ValueError("Number of compartments cannot exceed 200.")
    all_poly_gdfs, all_line_gdfs, all_pts_gdfs = [], [], []

    if is_zip:
        target_shp = (mapping or {}).get("target_shp")
        features, _ = _load_polygons_from_zip(file_or_df, target_shp, crs, forest_col_name=forest_col_name)
        if len(features) == 1:
            feat_name, poly = features[0]
            pieces = _subdivide_polygon(poly, n_compartments)
            pg, lg, ptg = _save_compartments(pieces, feat_name, crs, out)
            all_poly_gdfs.append(pg); all_line_gdfs.append(lg); all_pts_gdfs.append(ptg)
        else:
            for feat_name, poly in features:
                pieces = _subdivide_polygon(poly, n_compartments)
                feat_dir = os.path.join(out, _safe_dirname(feat_name))
                pg, lg, ptg = _save_compartments(pieces, feat_name, crs, feat_dir)
                all_poly_gdfs.append(pg); all_line_gdfs.append(lg); all_pts_gdfs.append(ptg)
    else:
        df = file_or_df
        df = normalize_order(df)
        x_col      = safe_col(df, mapping, "X", "X")
        y_col      = safe_col(df, mapping, "Y", "Y")
        order_col  = safe_col(df, mapping, "Order", "Order")
        forest_col = safe_col(df, mapping, "Forest", "Forest")
        if not x_col: raise ValueError("Could not find an X / Easting / Longitude column.")
        if not y_col: raise ValueError("Could not find a Y / Northing / Latitude column.")
        if e_mode == "B" and not forest_col: raise ValueError("Multi-Forest mode requires a Forest column.")
        if e_mode == "A":
            forest_name = (mapping or {}).get("forest") or "FOREST"
            poly = _df_to_polygon(df, x_col, y_col, order_col)
            pieces = _subdivide_polygon(poly, n_compartments)
            pg, lg, ptg = _save_compartments(pieces, forest_name, crs, out)
            all_poly_gdfs.append(pg); all_line_gdfs.append(lg); all_pts_gdfs.append(ptg)
        else:
            for f, fg in df.groupby(forest_col):
                try: poly = _df_to_polygon(fg, x_col, y_col, order_col)
                except ValueError: continue
                pieces = _subdivide_polygon(poly, n_compartments)
                forest_dir = os.path.join(out, _safe_dirname(str(f)))
                pg, lg, ptg = _save_compartments(pieces, str(f), crs, forest_dir)
                all_poly_gdfs.append(pg); all_line_gdfs.append(lg); all_pts_gdfs.append(ptg)

    if not all_poly_gdfs: raise ValueError("No valid polygons could be built from the data.")
    poly_gdf = gpd.GeoDataFrame(pd.concat(all_poly_gdfs, ignore_index=True), crs=crs)
    line_gdf = gpd.GeoDataFrame(pd.concat(all_line_gdfs, ignore_index=True), crs=crs)
    pts_gdf  = gpd.GeoDataFrame(pd.concat(all_pts_gdfs,  ignore_index=True), crs=crs)
    return poly_gdf, line_gdf, pts_gdf


# =============================================================================
# GROUP F — DEM SLOPE ANALYSIS (ArcGIS-style workflow)
# =============================================================================
# Exact workflow:
#   1. Extract rectangular DEM patch around polygon (20% buffer on each side)
#      → avoids losing boundary pixels when clipping slope to boundary
#   2. Compute slope on the FULL rectangular DEM using Horn's 3x3 method
#      (same as ArcGIS Slope tool)
#   3. Reclassify slope raster into 3 classes (< 19°, 19–31°, > 31°)
#   4. Raster-to-polygon conversion on the FULL rectangular classified raster
#      (NOT clipped yet — this matches ArcGIS "Raster to Polygon" on full raster)
#   5. Dissolve by class (gridcode)
#   6. Clip dissolved polygons to forest boundary
#      → boundary pixels stay intact, no partial-pixel loss
#   7. Calculate areas from clipped polygons (more accurate than pixel counting)
#   8. Optional field-area recalibration
#
# This workflow exactly replicates ArcGIS:
#   Slope → Reclassify → Raster to Polygon → Dissolve → Clip → Area calculation
# =============================================================================

def _boundary_polygon_from_df(df, mapping):
    df = normalize_order(df)
    x_col     = safe_col(df, mapping, "X", "X")
    y_col     = safe_col(df, mapping, "Y", "Y")
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
            if f.endswith(".shp"): shp_candidates.append(os.path.join(root, f))
    if not shp_candidates: raise ValueError("No .shp found in boundary ZIP.")
    shp_path = shp_candidates[0]
    if target_shp:
        for cand in shp_candidates:
            if os.path.basename(cand) == os.path.basename(target_shp): shp_path = cand; break
    gdf = gpd.read_file(shp_path)
    if gdf.empty: raise ValueError("Boundary shapefile has no features.")
    if gdf.crs is None: gdf = gdf.set_crs(src_crs)
    gdf = gdf.to_crs(dem_crs)
    return _repair_geom(gdf.unary_union), gdf


def group_f(boundary_file, dem_file, crs, out, mapping=None,
            boundary_is_zip=False, forest_name="FOREST",
            f_mode="A", comp_col_name=None, field_area_ha=None):
    """
    Group F — DEM Slope Analysis, ArcGIS-equivalent workflow.
    Steps: Rect extract → Slope (Horn) → Reclassify → Raster-to-Poly →
           Dissolve → Clip to boundary → Area calc → Calibrate
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

    # ── Step 0: Load full DEM ─────────────────────────────────────────────
    dem_path = os.path.join(UPLOAD, f"{uuid.uuid4()}_dem.tif")
    dem_file.save(dem_path)

    with rasterio.open(dem_path) as src:
        dem_crs    = src.crs
        dem_nodata = src.nodata
        dem_profile = src.profile.copy()
        dem_transform = src.transform
        dem_res_x = abs(dem_transform.a)
        dem_res_y = abs(dem_transform.e)

    # ── Step 1: Load boundary, reproject to DEM CRS ────────────────────────
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
    boundary_poly = _repair_geom(boundary_poly)

    # ── Step 2: Extract rectangular DEM patch (20% buffer) ────────────────
    # Why: computing slope on the FULL rectangle then clipping polygons to the
    # boundary avoids losing edge pixels. ArcGIS does the same — slope is
    # computed on the full raster BEFORE any masking.
    b = boundary_poly.bounds  # (minx, miny, maxx, maxy)
    buf_x = (b[2] - b[0]) * 0.20
    buf_y = (b[3] - b[1]) * 0.20
    rect_poly = shapely_box(b[0]-buf_x, b[1]-buf_y, b[2]+buf_x, b[3]+buf_y)

    try:
        with rasterio.open(dem_path) as src:
            rect_arr, rect_tr = rio_mask(
                src,
                [rect_poly.__geo_interface__],
                crop=True,
                filled=True,
                nodata=dem_nodata if dem_nodata is not None else -9999
            )
        rect_dem = rect_arr[0].astype(np.float64)
        rect_nodata_val = dem_nodata if dem_nodata is not None else -9999
        # Mark nodata as NaN for math
        rect_valid = rect_dem != rect_nodata_val
        rect_dem_nan = rect_dem.copy().astype(np.float64)
        rect_dem_nan[~rect_valid] = np.nan
        rect_res_x = abs(rect_tr.a)
        rect_res_y = abs(rect_tr.e)
    except Exception as e:
        # If extraction fails (DEM smaller than buffer), use full DEM
        with rasterio.open(dem_path) as src:
            rect_arr_full = src.read(1).astype(np.float64)
            rect_tr = src.transform
            rect_res_x = abs(rect_tr.a)
            rect_res_y = abs(rect_tr.e)
            rect_nodata_val = dem_nodata if dem_nodata is not None else -9999
        rect_valid = rect_arr_full != rect_nodata_val
        rect_dem_nan = rect_arr_full.copy()
        rect_dem_nan[~rect_valid] = np.nan

    # ── Step 3: Compute slope using Horn's method (= ArcGIS Slope tool) ───
    # ArcGIS uses the 8-direction neighbourhood:
    #   dz/dx = [(c + 2f + i) - (a + 2d + g)] / (8 * cell_size_x)
    #   dz/dy = [(g + 2h + i) - (a + 2b + c)] / (8 * cell_size_y)
    # This is equivalent to Sobel operator / (8 * cell_size).
    # We fill NaN with 0 for the convolution (same as ArcGIS at NoData edges),
    # then invalidate output cells whose 3x3 window touched any NoData cell.

    filled = np.nan_to_num(rect_dem_nan, nan=0.0)

    import scipy.ndimage as ndimage
    dzdx = ndimage.sobel(filled, axis=1) / (8.0 * max(rect_res_x, 1e-12))
    dzdy = ndimage.sobel(filled, axis=0) / (8.0 * max(rect_res_y, 1e-12))
    slope_deg = np.degrees(np.arctan(np.sqrt(dzdx**2 + dzdy**2))).astype(np.float32)

    # ArcGIS marks a cell NoData if ANY of its 8 neighbours is NoData.
    # Replicate with binary erosion: a cell is "fully valid" only if its
    # entire 3x3 neighbourhood was valid. border_value=0 treats the raster
    # edge as NoData (matches ArcGIS boundary behavior).
    fully_valid = ndimage.binary_erosion(
        rect_valid, structure=np.ones((3, 3), dtype=bool), border_value=0
    )
    SLOPE_NODATA = np.float32(-9999)
    slope_deg[~fully_valid] = SLOPE_NODATA

    # Save rectangular slope raster
    rect_profile = dem_profile.copy()
    rect_profile.update(
        dtype="float32", nodata=float(SLOPE_NODATA), count=1,
        height=slope_deg.shape[0], width=slope_deg.shape[1],
        transform=rect_tr, crs=dem_crs
    )
    slope_rect_path = os.path.join(out, f"{prefix}_slope_rect.tif")
    with rasterio.open(slope_rect_path, "w", **rect_profile) as dst:
        dst.write(slope_deg, 1)

    # ── Step 4: Reclassify slope into 3 classes ───────────────────────────
    # Class 1: < 19°  (Gentle)
    # Class 2: 19–31° (Moderate)
    # Class 3: > 31°  (Steep)
    # NoData cells → class 0 (background, excluded from polygons)
    valid_mask = (slope_deg != SLOPE_NODATA) & ~np.isnan(slope_deg)
    class_rect = np.zeros(slope_deg.shape, dtype=np.uint8)
    class_rect[valid_mask & (slope_deg < 19)]                       = 1
    class_rect[valid_mask & (slope_deg >= 19) & (slope_deg <= 31)]  = 2
    class_rect[valid_mask & (slope_deg > 31)]                       = 3

    cls_rect_profile = rect_profile.copy()
    cls_rect_profile.update(dtype="uint8", nodata=0)
    cls_rect_path = os.path.join(out, f"{prefix}_class_rect.tif")
    with rasterio.open(cls_rect_path, "w", **cls_rect_profile) as dst:
        dst.write(class_rect, 1)

    # ── Step 5: Raster-to-polygon on FULL rectangular classified raster ────
    # Convert each connected region of same class to a polygon.
    # This matches ArcGIS "Raster to Polygon" on the full rect raster.
    # We DON'T clip to boundary yet — clipping happens in Step 7.
    class_defs = {
        1: ("< 19°",    "Gentle",   "#2ecc71"),
        2: ("19 - 31°", "Moderate", "#f39c12"),
        3: ("> 31°",    "Steep",    "#e74c3c"),
    }

    rtp_records = []  # list of {gridcode, geometry}
    with rasterio.open(cls_rect_path) as src:
        cls_arr = src.read(1)
        valid_pixels = (cls_arr > 0).astype(np.uint8)
        for shp_geom, val in rio_shapes(cls_arr, mask=valid_pixels, transform=rect_tr):
            cid = int(val)
            if cid == 0:
                continue
            try:
                rings    = shp_geom["coordinates"]
                exterior = rings[0]
                holes    = rings[1:] if len(rings) > 1 else []
                # Preserve holes — same as ArcGIS preserving interior rings
                geom = _repair_geom(Polygon(exterior, holes))
                if geom and not geom.is_empty and geom.area > 1e-12:
                    rtp_records.append({"gridcode": cid, "geometry": geom})
            except Exception:
                continue

    if not rtp_records:
        raise ValueError("No slope polygons could be extracted from the DEM. "
                         "Check that the DEM overlaps the boundary polygon.")

    rtp_gdf = gpd.GeoDataFrame(rtp_records, crs=str(dem_crs))

    # ── Step 6: Dissolve by gridcode ──────────────────────────────────────
    # Merges all polygons of same class into one multipolygon per class.
    # This matches ArcGIS "Dissolve" by gridcode.
    try:
        dissolved = rtp_gdf.dissolve(by="gridcode", as_index=False)
        dissolved["gridcode"] = dissolved["gridcode"].astype(int)
    except Exception:
        dissolved = rtp_gdf.copy()
        dissolved["gridcode"] = dissolved["gridcode"].astype(int)

    # ── Step 7: Clip dissolved polygons to forest boundary ────────────────
    # This is the KEY step that ensures boundary accuracy:
    # - Pixels inside boundary contribute their FULL area
    # - Pixels crossing the boundary are cut precisely at the boundary line
    # - No pixels are silently dropped due to centre-point inclusion rules
    # This exactly matches ArcGIS "Clip" after Raster-to-Polygon.

    def _clip_one_boundary(clip_poly, label):
        """Clip dissolved slope polygons to clip_poly. Return summary rows + vector records."""
        rows     = []
        vec_recs = []
        total_area_ha = 0.0

        for _, drow in dissolved.iterrows():
            cid   = int(drow["gridcode"])
            dgeom = _repair_geom(drow.geometry)
            if dgeom is None or dgeom.is_empty:
                continue
            try:
                clipped = _repair_geom(dgeom.intersection(clip_poly))
            except Exception:
                # Try buffering clip_poly slightly if intersection fails
                try:
                    clipped = _repair_geom(dgeom.intersection(clip_poly.buffer(1e-6)))
                except Exception:
                    continue
            if clipped is None or clipped.is_empty:
                continue
            area_ha = round(clipped.area / 10000.0, 4)
            if area_ha <= 0:
                continue
            total_area_ha += area_ha
            vec_recs.append({
                "Label":       label,
                "Class":       cid,
                "Slope_Range": class_defs[cid][0],
                "Descr":       class_defs[cid][1],
                "Area_ha":     area_ha,
                "geometry":    clipped,
            })

        total_area_ha = max(total_area_ha, 1e-8)
        for vr in vec_recs:
            rows.append({
                "Label":       vr["Label"],
                "Class":       vr["Class"],
                "Slope_Range": vr["Slope_Range"],
                "Description": vr["Descr"],
                "Area_ha":     vr["Area_ha"],
                "Pct_Area":    round(vr["Area_ha"] / total_area_ha * 100, 2),
                "Total_ha":    round(total_area_ha, 4),
            })
        return rows, vec_recs

    # Build list of (label, polygon) to process
    comp_polygons = []
    if f_mode == "A":
        comp_polygons = [(forest_name, boundary_poly)]
    else:
        grp_col = None
        if comp_col_name:
            for c in boundary_gdf.columns:
                if c.lower() == comp_col_name.lower(): grp_col = c; break
        if grp_col is None:
            grp_col = _find_col(boundary_gdf, _FOREST_ALIASES if f_mode == "B" else _COMPARTMENT_ALIASES)
        if grp_col is None:
            comp_polygons = [(forest_name, boundary_poly)]
        else:
            for val, grp in boundary_gdf.groupby(grp_col):
                comp_polygons.append((str(val), _repair_geom(grp.unary_union)))

    all_summary, all_vec = [], []
    for label, cpoly in comp_polygons:
        rows, vec_recs = _clip_one_boundary(cpoly, label)
        all_summary.extend(rows)
        all_vec.extend(vec_recs)

    if not all_summary:
        raise ValueError("Clipping produced no slope data. Verify boundary and DEM overlap.")

    # ── Step 8: Field-area recalibration ──────────────────────────────────
    # If user provides measured field area (ha), compute a single correction
    # factor = field_area / computed_total and apply to all rows uniformly.
    # This matches ArcGIS field-calibrated area adjustment.
    if field_area_ha and field_area_ha > 0:
        total_computed = sum(r["Area_ha"] for r in all_summary)
        factor = field_area_ha / max(total_computed, 1e-8)
        for r in all_summary:
            r["Recal_ha"]   = round(r["Area_ha"] * factor, 4)
            r["Cal_Factor"] = round(factor, 6)
    else:
        for r in all_summary:
            r["Recal_ha"]   = None
            r["Cal_Factor"] = None

    # ── Save outputs ──────────────────────────────────────────────────────
    vec_crs = str(dem_crs)

    # Clipped slope raster (first compartment, for preview)
    try:
        with rasterio.open(slope_rect_path) as src:
            first_cs_arr, first_tr = rio_mask(
                src, [comp_polygons[0][1].__geo_interface__],
                crop=True, filled=True, nodata=float(SLOPE_NODATA)
            )
        first_cs = first_cs_arr[0].astype(np.float32)
    except Exception:
        first_cs = slope_deg
        first_tr = rect_tr

    first_valid = (first_cs != SLOPE_NODATA) & ~np.isnan(first_cs)
    first_ca = np.zeros(first_cs.shape, dtype=np.uint8)
    first_ca[first_valid & (first_cs < 19)]                        = 1
    first_ca[first_valid & (first_cs >= 19) & (first_cs <= 31)]    = 2
    first_ca[first_valid & (first_cs > 31)]                        = 3

    # Save preview rasters
    clip_prof = rect_profile.copy()
    clip_prof.update(height=first_cs.shape[0], width=first_cs.shape[1], transform=first_tr)
    with rasterio.open(os.path.join(out, f"{prefix}_slope_clipped.tif"), "w", **clip_prof) as dst:
        dst.write(first_cs, 1)
    cls_prof2 = clip_prof.copy()
    cls_prof2.update(dtype="uint8", nodata=0)
    with rasterio.open(os.path.join(out, f"{prefix}_slope_classes.tif"), "w", **cls_prof2) as dst:
        dst.write(first_ca, 1)

    # Summary Excel
    summary_df = pd.DataFrame(all_summary)
    if f_mode == "A":
        summary_df = summary_df.drop(columns=["Label"], errors="ignore")
    summary_df.to_excel(os.path.join(out, f"{prefix}_slope_summary.xlsx"), index=False)

    # Vector slope polygons (clipped)
    if all_vec:
        vec_gdf = gpd.GeoDataFrame(all_vec, crs=vec_crs)
        vec_gdf.to_file(os.path.join(out, f"{prefix}_slope_polygon.shp"))
    else:
        vec_gdf = gpd.GeoDataFrame(columns=["geometry"], crs=vec_crs)

    boundary_gdf.to_file(os.path.join(out, f"{prefix}_boundary_polygon.shp"))

    return first_cs, first_ca, first_tr, all_summary, vec_gdf, float(SLOPE_NODATA), boundary_gdf, f_mode


def preview_slope(clipped_slope, class_arr, summary_rows, path, nodata,
                  boundary_gdf=None, f_mode="A"):
    import matplotlib.patches as mpatches
    import matplotlib.gridspec as gridspec

    valid = (class_arr > 0)
    has_label = f_mode in ("B", "E")
    n_rows = len(summary_rows)
    table_height = max(0.22, min(0.55, 0.06 + n_rows * 0.035))
    fig_h = 7 + (2 if n_rows > 6 else 0)

    fig = plt.figure(figsize=(14, fig_h), dpi=150, facecolor="white")
    gs  = gridspec.GridSpec(2, 2, height_ratios=[1, table_height],
                            hspace=0.32, wspace=0.06, left=0.02, right=0.98, top=0.94, bottom=0.03)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, :])

    disp = np.where(valid, clipped_slope, np.nan)
    im   = ax1.imshow(disp, cmap="terrain", interpolation="bilinear")
    fig.colorbar(im, ax=ax1, fraction=0.046, pad=0.04, label="Degrees")
    ax1.set_title("Slope Raster (degrees)", fontsize=10, fontweight="bold", pad=6)
    ax1.axis("off")

    colors_map = {0:(1,1,1,0), 1:(0.18,0.8,0.44,1), 2:(0.95,0.61,0.07,1), 3:(0.91,0.29,0.24,1)}
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
            bw, bh = bounds[2]-bounds[0], bounds[3]-bounds[1]
            def to_px(x, y): return (x-bounds[0])/bw*w_px, (bounds[3]-y)/bh*h_px
            for geom in boundary_gdf.geometry:
                if geom is None or geom.is_empty: continue
                parts = list(geom.geoms) if geom.geom_type.startswith("Multi") else [geom]
                for g in parts:
                    xs, ys = g.exterior.xy
                    pxs, pys = zip(*[to_px(x,y) for x,y in zip(xs,ys)])
                    ax2.plot(pxs, pys, color="black", linewidth=2.5, zorder=6)
        except Exception:
            pass

    legend_items = [
        mpatches.Patch(facecolor="#2ecc71", label="< 19°  Gentle"),
        mpatches.Patch(facecolor="#f39c12", label="19–31° Moderate"),
        mpatches.Patch(facecolor="#e74c3c", label="> 31°  Steep"),
    ]
    ax2.legend(handles=legend_items, loc="lower left", fontsize=7, framealpha=0.92,
               title="Slope Class", title_fontsize=7)

    ax3.axis("off")
    ax3.set_title("Slope Area Summary Table", fontsize=10, fontweight="bold", pad=4)

    if summary_rows:
        has_recal = any(r.get("Recal_ha") is not None for r in summary_rows)
        if has_label:
            col_labels = ["Compartment", "Slope Range", "Description", "Area (ha)", "% Area"]
            if has_recal: col_labels += ["Recal. Area (ha)", "Conv. Factor"]
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
            if has_recal: col_labels += ["Recal. Area (ha)", "Conv. Factor"]
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
        if geom is None or geom.is_empty: continue
        color = _COMP_COLORS[i % len(_COMP_COLORS)]
        gpd.GeoDataFrame([{"geometry": geom}], crs=poly_gdf.crs).plot(
            ax=ax, facecolor=color, edgecolor="#1a1a1a", linewidth=1.8)
        cx, cy  = geom.centroid.x, geom.centroid.y
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
            ax.annotate(str(int(row[label_col])), xy=(row.geometry.x, row.geometry.y),
                        xytext=(0,7), textcoords="offset points", ha="center", va="bottom",
                        fontsize=5, fontweight="bold", color="black",
                        path_effects=[pe.Stroke(linewidth=1.8, foreground="white"), pe.Normal()], zorder=6)
    if poly_gdf is not None and not poly_gdf.empty and "Comp_ID" in poly_gdf.columns:
        for _, row in poly_gdf.iterrows():
            cx = row.geometry.centroid.x; cy = row.geometry.centroid.y
            area_txt = f"{row['Area_ha']:.2f} ha" if "Area_ha" in row else ""
            ax.annotate(f"{row['Comp_ID']}\n{area_txt}", xy=(cx, cy),
                        ha="center", va="center", fontsize=5.5, fontweight="bold", color="black",
                        path_effects=[pe.Stroke(linewidth=2, foreground="white"), pe.Normal()], zorder=7)
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
        if geom is None or geom.is_empty: continue
        if name_col and name_col in row.index and row[name_col]: label = str(row[name_col])
        elif "Comp_ID" in row.index and row["Comp_ID"]:
            parts = [str(row["Comp_ID"])]
            if "Area_ha" in row.index: parts.append(f"{row['Area_ha']} ha")
            label = " · ".join(parts)
        elif "Forest" in row.index and row["Forest"]: label = str(row["Forest"])
        else: label = f"Feature {i+1}"
        desc_parts = []
        for col in row.index:
            if col == "geometry": continue
            val = row[col]
            if val is not None and str(val) not in ("None","nan",""): desc_parts.append(f"{col}: {val}")
        description = " | ".join(desc_parts)
        def coords_str(coords): return " ".join(f"{x},{y},0" for x,y in coords)
        def polygon_kml(g):
            outer = coords_str(list(g.exterior.coords))
            rings = [f"<outerBoundaryIs><LinearRing><coordinates>{outer}</coordinates></LinearRing></outerBoundaryIs>"]
            for interior in g.interiors:
                inner = coords_str(list(interior.coords))
                rings.append(f"<innerBoundaryIs><LinearRing><coordinates>{inner}</coordinates></LinearRing></innerBoundaryIs>")
            return f"<Polygon>{''.join(rings)}</Polygon>"
        if geom.geom_type == "Polygon": geo_kml = polygon_kml(geom)
        elif geom.geom_type == "MultiPolygon":
            geo_kml = f"<MultiGeometry>{''.join(polygon_kml(g) for g in geom.geoms)}</MultiGeometry>"
        elif geom.geom_type == "LineString":
            geo_kml = f"<LineString><coordinates>{coords_str(list(geom.coords))}</coordinates></LineString>"
        elif geom.geom_type == "Point":
            geo_kml = f"<Point><coordinates>{geom.x},{geom.y},0</coordinates></Point>"
        else: continue
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
        except Exception: return None
    poly_w = to_wgs84(poly_gdf); line_w = to_wgs84(line_gdf); pts_w = to_wgs84(pts_gdf)
    ref_gdf = poly_w if poly_w is not None and not poly_w.empty else line_w
    if ref_gdf is not None and not ref_gdf.empty:
        union = ref_gdf.unary_union
        cx, cy = union.centroid.x, union.centroid.y
        minx,miny,maxx,maxy = union.bounds
        alt_m = max(500, int(max(maxx-minx,maxy-miny)*111_000*2))
    else:
        cx, cy, alt_m = 0, 0, 10000
    kml_styles = """
  <Style id="poly_style"><LineStyle><color>ff00ff00</color><width>2</width></LineStyle><PolyStyle><color>4400cc00</color></PolyStyle></Style>
  <Style id="line_style"><LineStyle><color>ff0000ff</color><width>2</width></LineStyle></Style>
  <Style id="point_style"><IconStyle><color>ff0000ff</color><scale>0.8</scale><Icon><href>http://maps.google.com/mapfiles/kml/shapes/placemark_circle.png</href></Icon></IconStyle><LabelStyle><scale>0.7</scale></LabelStyle></Style>"""
    folders = []
    if poly_w is not None and not poly_w.empty:
        folders.append(f"<Folder><name>Polygons</name>{_gdf_to_kml_placemarks(poly_w,'poly_style','Forest')}</Folder>")
    if line_w is not None and not line_w.empty:
        folders.append(f"<Folder><name>Lines</name>{_gdf_to_kml_placemarks(line_w,'line_style','Forest')}</Folder>")
    if pts_w is not None and not pts_w.empty:
        folders.append(f"<Folder><name>Points</name>{_gdf_to_kml_placemarks(pts_w,'point_style','SN')}</Folder>")
    kml = f"""<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2"><Document><name>Elfak GIS Output</name>
  <LookAt><longitude>{cx}</longitude><latitude>{cy}</latitude><altitude>0</altitude>
    <range>{alt_m}</range><tilt>0</tilt><heading>0</heading><altitudeMode>relativeToGround</altitudeMode></LookAt>
  {kml_styles}{"".join(folders)}</Document></kml>"""
    kmz_path = os.path.join(out_dir, "output.kmz")
    with zf.ZipFile(kmz_path, "w", zf.ZIP_DEFLATED) as kmz:
        kmz.writestr("doc.kml", kml.encode("utf-8"))
    return {"url": f"/outputs/{run_id}/output.kmz", "lat": round(cy,6), "lon": round(cx,6), "alt": alt_m}


# ================= AUTH ROUTES =================
@app.route("/login", methods=["POST"])
def login():
    import re
    data     = request.get_json(silent=True) or {}
    username = (data.get("username") or "").strip()
    if not username or len(username) < 2:
        return jsonify({"error": "Username must be at least 2 characters."}), 400
    if len(username) > 40:
        return jsonify({"error": "Username too long (max 40 characters)."}), 400
    if not re.match(r'^[A-Za-z0-9 _\-]+$', username):
        return jsonify({"error": "Username may only contain letters, numbers, spaces, - and _."}), 400
    if _user_exists(username):
        return jsonify({"error": f"Username \"{username}\" is already taken.", "taken": True}), 409
    try:
        user = _create_user(username)
    except ValueError as e:
        return jsonify({"error": str(e), "taken": True}), 409
    session["username"] = username
    session.permanent   = True
    return jsonify({"ok": True, "username": username, "runs": []})

@app.route("/logout", methods=["POST"])
def logout():
    session.clear()
    return jsonify({"ok": True})

@app.route("/me")
def me():
    username = _require_login()
    if not username: return jsonify({"error": "Not logged in"}), 401
    users = _load_users()
    user  = users.get(username, {})
    return jsonify({"username": username, "runs": user.get("runs", [])[-20:]})

@app.route("/history")
def history():
    username = _require_login()
    if not username: return jsonify({"error": "Not logged in"}), 401
    users = _load_users()
    user  = users.get(username, {})
    return jsonify({"runs": user.get("runs", [])})


# ================= UPLOAD ROUTE =================
@app.route("/upload", methods=["POST"])
def upload():
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400
        file   = request.files["file"]
        mode   = request.form.get("mode", "A")
        module = request.form.get("module", mode)
        zone   = request.form.get("zone", "44")
        try:
            mapping = json.loads(request.form.get("mapping", "{}"))
        except (TypeError, ValueError):
            mapping = {}
        w      = float(request.form.get("w",    50))
        h      = float(request.form.get("h",    50))
        rows   = int(  request.form.get("rows", 10))
        cols   = int(  request.form.get("cols", 10))
        forest = request.form.get("forest") or (mapping or {}).get("forest") or "FOREST"
        run_id   = str(uuid.uuid4())
        username = _require_login() or "guest"
        out      = os.path.join(OUTPUT, run_id)
        os.makedirs(out, exist_ok=True)
        crs = get_crs(zone)
        label_col = None; label_pts_gdf = None; area_ha_display = None

        if module == "B":
            df = read_input(file)
            poly, line, pts = group_b(df, crs, out, mapping)

        elif module == "C":
            poly, line, pts = group_c(file, crs, w, h, rows, cols, out, mode, mapping)
            label_col = "SN"; label_pts_gdf = pts

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
                poly, line, pts = group_e(file, crs, out, mapping,
                    e_mode=e_mode, n_compartments=n_compartments, is_zip=True,
                    forest_col_name=forest_col_name)
            else:
                df = read_input(file)
                poly, line, pts = group_e(df, crs, out, mapping,
                    e_mode=e_mode, n_compartments=n_compartments, is_zip=False,
                    forest_col_name=forest_col_name)
            preview_path = os.path.join(out, "output.png")
            preview_compartments(poly, preview_path)
            kmz_url = None
            try: kmz_url = generate_kmz(poly, line, pts, out, run_id)
            except Exception: pass
            _append_run(username, run_id, "E")
            return jsonify({"run_id": run_id, "download": f"/download/{run_id}", "kmz_url": kmz_url})

        elif module == "F":
            dem_file_upload = request.files.get("dem_file")
            if not dem_file_upload:
                return jsonify({"error": "No DEM file uploaded. Please upload a GeoTIFF DEM."}), 400
            f_forest        = request.form.get("f_forest") or forest
            f_mode          = request.form.get("f_mode", "A")
            comp_col_name   = request.form.get("comp_col") or None
            boundary_is_zip = file.filename.lower().endswith(".zip")
            field_area_ha   = None
            fa_str = request.form.get("field_area_ha", "").strip()
            if fa_str:
                try: field_area_ha = float(fa_str)
                except ValueError: pass
            result = group_f(
                boundary_file=file, dem_file=dem_file_upload, crs=crs, out=out,
                mapping=mapping, boundary_is_zip=boundary_is_zip,
                forest_name=f_forest, f_mode=f_mode, comp_col_name=comp_col_name,
                field_area_ha=field_area_ha)
            clipped_slope, class_arr, _tr, summary_rows, vec_gdf, nodata, boundary_gdf, f_mode_out = result
            preview_path = os.path.join(out, "output.png")
            preview_slope(clipped_slope, class_arr, summary_rows, preview_path, nodata,
                          boundary_gdf=boundary_gdf, f_mode=f_mode_out)
            poly = vec_gdf if (vec_gdf is not None and not vec_gdf.empty) else gpd.GeoDataFrame()
            line = gpd.GeoDataFrame(); pts = gpd.GeoDataFrame()
            kmz_url = None
            try: kmz_url = generate_kmz(poly, line, pts, out, run_id)
            except Exception: pass
            _append_run(username, run_id, "F")
            return jsonify({"run_id": run_id, "download": f"/download/{run_id}", "kmz_url": kmz_url})

        else:  # module == "A"
            df = read_input(file)
            poly, line, pts = group_a(df, forest, crs, out, mapping)
            if not poly.empty and "Area_ha" in poly.columns:
                area_ha_display = float(poly["Area_ha"].sum())

        preview_path = os.path.join(out, "output.png")
        preview(poly, line, pts, preview_path, pc="yellow", lc="black", ptc="red",
                label_col=label_col, label_pts_gdf=label_pts_gdf, area_ha=area_ha_display)
        kmz_url = None
        try: kmz_url = generate_kmz(poly, line, pts, out, run_id)
        except Exception: pass
        _append_run(username, run_id, module)
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
