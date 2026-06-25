import os
import uuid
import zipfile
import shutil
import json
import math
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D

from flask import Flask, request, jsonify, send_file, send_from_directory, render_template, session
from shapely.geometry import Polygon, Point, LineString, MultiPolygon, MultiPoint, GeometryCollection
from shapely.geometry import box as _shapely_box
from shapely.ops import unary_union

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "elfak-gis-engine-2025-secret")

UPLOAD = "uploads"
OUTPUT = "outputs"
USERS_FILE = "users.json"

os.makedirs(UPLOAD, exist_ok=True)
os.makedirs(OUTPUT, exist_ok=True)

# ─── A4 figure size in inches at 150 dpi ────────────────────────────────────
A4_W_IN = 8.27   # 210mm
A4_H_IN = 11.69  # 297mm
A4_DPI  = 150

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
    users[username] = {
        "username":   username,
        "created_at": pd.Timestamp.now().isoformat(),
        "runs":       []
    }
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
            "run_id":      run_id,
            "module":      module,
            "description": description,
            "timestamp":   pd.Timestamp.now().isoformat()
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
    if g.is_empty:
        return None
    return g if g.is_valid else g.buffer(0)


def _as_polygon(geom):
    """Extract best polygon from any geometry type."""
    if geom is None or geom.is_empty:
        return None
    t = geom.geom_type
    if t == "Polygon":
        return geom if not geom.is_empty else None
    if t in ("MultiPolygon", "GeometryCollection"):
        polys = [g for g in geom.geoms
                 if g.geom_type == "Polygon" and not g.is_empty and g.area > 1e-10]
        return max(polys, key=lambda g: g.area) if polys else None
    return None


def _force_polygon_only(geom):
    """
    CRITICAL FIX: Force any geometry to Polygon/MultiPolygon.
    Prevents LINESTRING written to POLYGON shapefile error.
    """
    if geom is None or geom.is_empty:
        return None
    t = geom.geom_type
    if t == "Polygon":
        return geom if geom.area > 1e-12 else None
    if t == "MultiPolygon":
        polys = [g for g in geom.geoms if g.geom_type == "Polygon" and g.area > 1e-12]
        return MultiPolygon(polys) if polys else None
    if t == "GeometryCollection":
        polys = []
        for g in geom.geoms:
            p = _force_polygon_only(g)
            if p is not None:
                if p.geom_type == "Polygon":
                    polys.append(p)
                elif p.geom_type == "MultiPolygon":
                    polys.extend(p.geoms)
        if not polys:
            return None
        return MultiPolygon(polys) if len(polys) > 1 else polys[0]
    # LineString, Point, etc → discard
    return None


# ─── COLUMN ALIAS RESOLUTION ────────────────────────────────────────────────

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


# ================= MAP DECORATIONS =================

def _add_north_arrow(ax, x=0.93, y=0.93, size=0.055):
    """Add ArcGIS-style north arrow to axes."""
    ax.annotate(
        '', xy=(x, y + size * 0.8), xytext=(x, y - size * 0.3),
        xycoords='axes fraction', textcoords='axes fraction',
        arrowprops=dict(arrowstyle='->', color='black', lw=1.5)
    )
    ax.text(x, y + size, 'N', transform=ax.transAxes,
            ha='center', va='bottom', fontsize=9, fontweight='bold',
            color='black', fontfamily='sans-serif')


def _add_scale_bar(ax, gdf_union=None, length_m=None, x=0.05, y=0.04):
    """Add ArcGIS-style scale bar."""
    try:
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        map_w = xlim[1] - xlim[0]
        if length_m is None:
            nice = [50, 100, 200, 250, 500, 1000, 2000, 5000, 10000, 20000, 50000]
            raw = map_w * 0.2
            length_m = min(nice, key=lambda v: abs(v - raw))

        bar_frac = length_m / max(map_w, 1e-9)
        bx0 = xlim[0] + map_w * x
        by0 = ylim[0] + (ylim[1] - ylim[0]) * y
        bx1 = bx0 + length_m

        # Draw alternating black/white bar
        ax.plot([bx0, bx1], [by0, by0], color='black', lw=4, solid_capstyle='butt',
                zorder=12, transform=ax.transData)
        mid = (bx0 + bx1) / 2
        ax.plot([bx0, mid], [by0, by0], color='black', lw=4, solid_capstyle='butt',
                zorder=13, transform=ax.transData)
        ax.plot([mid, bx1], [by0, by0], color='white', lw=4, solid_capstyle='butt',
                zorder=13, transform=ax.transData)

        # Labels
        lbl = f"{length_m:,.0f} m" if length_m < 1000 else f"{length_m/1000:.1f} km"
        ax.text(bx0, by0 - (ylim[1]-ylim[0])*0.012, "0",
                ha='center', va='top', fontsize=7, color='black', zorder=14,
                path_effects=[pe.Stroke(linewidth=2, foreground='white'), pe.Normal()])
        ax.text(bx1, by0 - (ylim[1]-ylim[0])*0.012, lbl,
                ha='center', va='top', fontsize=7, color='black', zorder=14,
                path_effects=[pe.Stroke(linewidth=2, foreground='white'), pe.Normal()])
    except Exception:
        pass


def _add_graticule(ax, n_lines=4):
    """
    Add coordinate graticule: tick marks and labels on axes edges only,
    NO grid lines inside the map (as requested).
    """
    try:
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        # Round to nice intervals
        dx = xlim[1] - xlim[0]
        dy = ylim[1] - ylim[0]
        step_x = _nice_step(dx / n_lines)
        step_y = _nice_step(dy / n_lines)
        x_ticks = _arange_ticks(xlim[0], xlim[1], step_x)
        y_ticks = _arange_ticks(ylim[0], ylim[1], step_y)

        # Only draw tick marks on axes border — no interior gridlines
        for xt in x_ticks:
            if xlim[0] <= xt <= xlim[1]:
                # bottom tick
                ax.plot([xt, xt], [ylim[0], ylim[0] + dy*0.01],
                        color='black', lw=0.8, zorder=11, transform=ax.transData)
                # top tick
                ax.plot([xt, xt], [ylim[1] - dy*0.01, ylim[1]],
                        color='black', lw=0.8, zorder=11, transform=ax.transData)
                ax.text(xt, ylim[0] - dy*0.025, f"{int(xt)}",
                        ha='center', va='top', fontsize=6, color='black',
                        rotation=45, zorder=12)

        for yt in y_ticks:
            if ylim[0] <= yt <= ylim[1]:
                ax.plot([xlim[0], xlim[0] + dx*0.01], [yt, yt],
                        color='black', lw=0.8, zorder=11, transform=ax.transData)
                ax.plot([xlim[1] - dx*0.01, xlim[1]], [yt, yt],
                        color='black', lw=0.8, zorder=11, transform=ax.transData)
                ax.text(xlim[0] - dx*0.025, yt, f"{int(yt)}",
                        ha='right', va='center', fontsize=6, color='black', zorder=12)
    except Exception:
        pass


def _nice_step(raw):
    if raw <= 0:
        return 1
    exp = math.floor(math.log10(raw))
    frac = raw / 10**exp
    if frac < 1.5:   return 1 * 10**exp
    elif frac < 3.5: return 2 * 10**exp
    elif frac < 7.5: return 5 * 10**exp
    else:            return 10 * 10**exp


def _arange_ticks(lo, hi, step):
    start = math.ceil(lo / step) * step
    ticks = []
    v = start
    while v <= hi + step * 0.01:
        ticks.append(v)
        v += step
    return ticks


def _add_legend(ax, handles, title="Legend", loc="lower right"):
    """Add ArcGIS-style legend box."""
    if not handles:
        return
    leg = ax.legend(
        handles=handles, title=title, loc=loc,
        framealpha=0.92, fancybox=True, shadow=False,
        fontsize=7, title_fontsize=8,
        edgecolor='#cccccc',
        borderpad=0.8, handlelength=1.6,
    )
    leg.get_frame().set_linewidth(0.8)


def _add_map_title(ax, title, subtitle=None):
    """Add ArcGIS-style map title."""
    ax.set_title(title, fontsize=12, fontweight='bold', pad=8,
                 fontfamily='sans-serif', color='#1a2b22')
    if subtitle:
        ax.text(0.5, 1.01, subtitle, transform=ax.transAxes,
                ha='center', va='bottom', fontsize=8,
                color='#666', style='italic')


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
        raise ValueError("No valid polygons could be built from the data.")

    return (gpd.GeoDataFrame(all_polys, crs=crs),
            gpd.GeoDataFrame(all_lines, crs=crs),
            gpd.GeoDataFrame(all_pts,   crs=crs))


# =============================================================================
# GROUP E — POLYGON SUBDIVIDER
# =============================================================================

def _principal_axis_angle(poly):
    try:
        coords = np.array(poly.exterior.coords[:-1])
        coords -= coords.mean(axis=0)
        cov = np.cov(coords.T)
        eigvals, eigvecs = np.linalg.eigh(cov)
        principal = eigvecs[:, np.argmax(eigvals)]
        return np.arctan2(principal[1], principal[0])
    except Exception:
        return 0.0


def _elongation_ratio(poly):
    minx, miny, maxx, maxy = poly.bounds
    dx, dy = maxx - minx, maxy - miny
    short = min(dx, dy)
    if short < 1e-9:
        return 999.0
    return max(dx, dy) / short


def _rotate_polygon(poly, angle):
    from shapely import affinity
    cx, cy = poly.centroid.x, poly.centroid.y
    return affinity.rotate(poly, -np.degrees(angle), origin=(cx, cy))


def _unrotate_polygon(poly, angle, cx, cy):
    from shapely import affinity
    return affinity.rotate(poly, np.degrees(angle), origin=(cx, cy))


def _close_polygon(poly):
    """
    Guarantee the polygon exterior ring is explicitly closed.
    Fixes the 'unclosed ring' artefact seen with 8+ bisections.
    """
    if poly is None or poly.is_empty:
        return poly
    ext = list(poly.exterior.coords)
    if ext[0] != ext[-1]:
        ext.append(ext[0])
    holes = [list(interior.coords) for interior in poly.interiors]
    for h in holes:
        if h[0] != h[-1]:
            h.append(h[0])
    try:
        closed = Polygon(ext, holes)
        if not closed.is_valid:
            closed = closed.buffer(0)
        return closed if not closed.is_empty else poly
    except Exception:
        return poly


def _bisect_polygon(poly, n, axis=None, _depth=0):
    """
    Recursively split poly into exactly n near-equal-area pieces.
    Each piece is explicitly clipped to parent to prevent open rings (8+).
    """
    poly = _repair_geom(poly)
    if poly is None or poly.is_empty or poly.area < 1e-10:
        return []
    if n <= 1:
        return [_close_polygon(poly)]

    minx, miny, maxx, maxy = poly.bounds
    dx = maxx - minx
    dy = maxy - miny

    use_principal = False
    rot_angle     = 0.0
    cx0, cy0      = poly.centroid.x, poly.centroid.y

    if axis == 'principal' or (axis is None and _depth < 2):
        elong = _elongation_ratio(poly)
        if elong > 1.6:
            rot_angle = _principal_axis_angle(poly)
            abs_deg   = abs(np.degrees(rot_angle)) % 90
            if 10 < abs_deg < 80:
                use_principal = True

    if use_principal:
        poly_rot = _rotate_polygon(poly, rot_angle)
        poly_rot = _repair_geom(poly_rot)
        if poly_rot and not poly_rot.is_empty:
            pieces_rot = _bisect_polygon(poly_rot, n, axis='x', _depth=_depth+1)
            pieces = []
            for p in pieces_rot:
                p_back = _unrotate_polygon(p, rot_angle, cx0, cy0)
                p_back = _repair_geom(p_back)
                pg = _as_polygon(p_back)
                if pg and not pg.is_empty and pg.area > 1e-10:
                    pieces.append(pg)
            clipped = []
            remaining = _repair_geom(poly)
            try:
                pieces.sort(key=lambda p: p.centroid.x)
            except Exception:
                pass
            for i, p in enumerate(pieces):
                if remaining is None or remaining.is_empty:
                    break
                if i == len(pieces) - 1:
                    last = _as_polygon(_repair_geom(remaining))
                    if last and last.area > 1e-10:
                        clipped.append(_close_polygon(last))
                else:
                    try:
                        c = _repair_geom(p.intersection(remaining))
                        c = _as_polygon(c)
                    except Exception:
                        c = None
                    if c and c.area > 1e-10:
                        clipped.append(_close_polygon(c))
                        try:
                            remaining = _repair_geom(remaining.difference(c))
                        except Exception:
                            pass
            if clipped:
                return clipped
        use_principal = False

    n_left  = n // 2
    n_right = n - n_left
    frac    = n_left / n

    def _cut_along(cand_axis):
        lo, hi      = (minx, maxx) if cand_axis == 'x' else (miny, maxy)
        target_area = poly.area * frac
        best_mid    = (lo + hi) / 2
        for _ in range(80):
            mid = (lo + hi) / 2
            try:
                if cand_axis == 'x':
                    left_half = _shapely_box(minx - 1, miny - 1, mid, maxy + 1)
                else:
                    left_half = _shapely_box(minx - 1, miny - 1, maxx + 1, mid)
                left_piece = _repair_geom(poly.intersection(left_half))
                got = left_piece.area if (left_piece and not left_piece.is_empty) else 0.0
            except Exception:
                return None
            err = abs(got - target_area) / (target_area + 1e-12)
            if err < 5e-4:
                best_mid = mid
                break
            if got < target_area:
                lo = mid
            else:
                hi = mid
            best_mid = mid
        try:
            if cand_axis == 'x':
                left_box = _shapely_box(minx - 1, miny - 1, best_mid, maxy + 1)
            else:
                left_box = _shapely_box(minx - 1, miny - 1, maxx + 1, best_mid)
            left_piece  = _repair_geom(poly.intersection(left_box))
            right_piece = _repair_geom(poly.difference(left_piece))
        except Exception:
            return None
        lp = _as_polygon(left_piece)
        rp = _as_polygon(right_piece)
        if lp is None or lp.is_empty or rp is None or rp.is_empty:
            return None
        return lp, rp

    def _bbox_aspect(p):
        b = p.bounds
        w, h  = b[2] - b[0], b[3] - b[1]
        short = min(w, h) or 1e-9
        return max(w, h) / short

    candidate_axes = [axis] if axis in ('x', 'y') else ['x', 'y']
    best = None
    for cand_axis in candidate_axes:
        cut = _cut_along(cand_axis)
        if cut is None:
            continue
        lp_c, rp_c = cut
        score = max(_bbox_aspect(lp_c), _bbox_aspect(rp_c))
        if best is None or score < best[0]:
            best = (score, lp_c, rp_c)

    if best is None:
        return [_close_polygon(poly)]

    _, lp, rp = best

    # Clip each child back to parent before recursing — prevents open rings
    try:
        lp = _as_polygon(_repair_geom(lp.intersection(poly))) or lp
        rp = _as_polygon(_repair_geom(rp.intersection(poly))) or rp
    except Exception:
        pass

    return (_bisect_polygon(lp, n_left,  None, _depth=_depth+1) +
            _bisect_polygon(rp, n_right, None, _depth=_depth+1))


def _subdivide_voronoi(poly, n, area_tol_ha=0.3):
    """Voronoi-based subdivision with iterative refinement."""
    try:
        from shapely.ops import voronoi_diagram
        import random
        # Generate seed points
        minx, miny, maxx, maxy = poly.bounds
        seeds = []
        attempts = 0
        while len(seeds) < n and attempts < n * 100:
            px = random.uniform(minx, maxx)
            py = random.uniform(miny, maxy)
            pt = Point(px, py)
            if poly.contains(pt):
                seeds.append(pt)
            attempts += 1
        if len(seeds) < n:
            return _subdivide_polygon(poly, n, area_tol_ha, method='bisect')

        mp = MultiPoint(seeds)
        regions = voronoi_diagram(mp, envelope=poly)
        pieces = []
        for region in regions.geoms:
            clipped = _repair_geom(region.intersection(poly))
            p = _as_polygon(clipped)
            if p and p.area > 1e-10:
                pieces.append(_close_polygon(p))

        if len(pieces) != n:
            # fallback
            return _subdivide_polygon(poly, n, area_tol_ha, method='bisect')
        return pieces
    except Exception:
        return _subdivide_polygon(poly, n, area_tol_ha, method='bisect')


def _subdivide_polygon(poly, n, area_tol_ha=0.3, method='bisect'):
    """
    Public entry: subdivide poly into n near-equal-area compartments.
    method: 'bisect', 'voronoi', 'grid'
    """
    n = max(2, min(200, int(n)))

    poly = _repair_geom(poly)
    if poly is None or poly.is_empty:
        return []

    if method == 'voronoi':
        return _subdivide_voronoi(poly, n, area_tol_ha)
    elif method == 'grid':
        return _subdivide_grid(poly, n, area_tol_ha)

    pieces = _bisect_polygon(poly, n)

    # Close all rings
    valid = []
    for p in pieces:
        p = _repair_geom(p)
        if p is None or p.is_empty or p.area < 1e-10:
            continue
        pg = _as_polygon(p)
        if pg:
            valid.append(_close_polygon(pg))

    if not valid:
        return [_close_polygon(poly)]

    # Merge excess slivers
    while len(valid) > n:
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
            if mp: valid[best_j] = _close_polygon(mp)
        except Exception:
            valid.append(sliver)

    # Split largest if short
    while len(valid) < n:
        largest_i = max(range(len(valid)), key=lambda i: valid[i].area)
        big = valid.pop(largest_i)
        sub = _bisect_polygon(big, 2)
        sub_valid = [_close_polygon(s) for s in sub if s and not s.is_empty and s.area > 1e-10]
        if len(sub_valid) == 2:
            valid.extend(sub_valid)
        else:
            valid.append(big)
            break

    # Gap fill
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
            merged = _repair_geom(unary_union([valid[best_i], gap]))
            p = _as_polygon(merged)
            if p: valid[best_i] = _close_polygon(p)
    except Exception:
        pass

    # Final close
    final = []
    for p in valid:
        p = _close_polygon(_repair_geom(p) or p)
        final.append(p)

    return final


def _subdivide_grid(poly, n, area_tol_ha=0.3):
    """Grid-based subdivision."""
    minx, miny, maxx, maxy = poly.bounds
    cols_n = math.ceil(math.sqrt(n))
    rows_n = math.ceil(n / cols_n)
    dx = (maxx - minx) / cols_n
    dy = (maxy - miny) / rows_n
    pieces = []
    for r in range(rows_n):
        for c in range(cols_n):
            cell = _shapely_box(minx + c*dx, miny + r*dy,
                                minx + (c+1)*dx, miny + (r+1)*dy)
            clipped = _repair_geom(poly.intersection(cell))
            p = _as_polygon(clipped)
            if p and p.area > 1e-10:
                pieces.append(_close_polygon(p))
    if not pieces:
        return [_close_polygon(poly)]
    return pieces


def _extract_division_points(pieces, forest_name):
    records = []
    sn = 1
    for i, p in enumerate(pieces, start=1):
        comp_id = f"Comp_{i:03d}"
        p = _close_polygon(_repair_geom(p))
        if p is None or p.is_empty:
            continue
        coords = list(p.exterior.coords)
        if len(coords) > 1 and coords[0] == coords[-1]:
            coords = coords[:-1]
        edge_lengths = [
            ((coords[(j+1) % len(coords)][0] - coords[j][0])**2 +
             (coords[(j+1) % len(coords)][1] - coords[j][1])**2) ** 0.5
            for j in range(len(coords))
        ]
        avg_edge = (sum(edge_lengths) / len(edge_lengths)) if edge_lengths else 1.0
        mid_thresh = avg_edge * 1.5

        for j, (cx, cy) in enumerate(coords):
            records.append({
                "SN": sn, "Forest": forest_name, "Comp_ID": comp_id,
                "Type": "Vertex", "X": round(cx, 4), "Y": round(cy, 4),
            })
            sn += 1
            nx, ny = coords[(j + 1) % len(coords)]
            elen = ((nx - cx)**2 + (ny - cy)**2) ** 0.5
            if elen > mid_thresh:
                mx, my = (cx + nx) / 2, (cy + ny) / 2
                records.append({
                    "SN": sn, "Forest": forest_name, "Comp_ID": comp_id,
                    "Type": "Midpoint", "X": round(mx, 4), "Y": round(my, 4),
                })
                sn += 1
    return records


def _extract_internal_cut_lines(pieces):
    lines = []
    n = len(pieces)
    for i in range(n):
        pi = pieces[i]
        if pi is None or pi.is_empty:
            continue
        bi = pi.bounds
        for j in range(i + 1, n):
            pj = pieces[j]
            if pj is None or pj.is_empty:
                continue
            bj = pj.bounds
            if bi[2] < bj[0] or bj[2] < bi[0] or bi[3] < bj[1] or bj[3] < bi[1]:
                continue
            try:
                shared = pi.exterior.intersection(pj.exterior)
            except Exception:
                continue
            if shared is None or shared.is_empty or shared.length < 1e-6:
                continue
            lines.append({"comp_i": i + 1, "comp_j": j + 1, "geometry": shared})
    return lines


def _save_compartments(pieces, forest_name, crs, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    # Close all rings before saving
    pieces = [_close_polygon(_repair_geom(p)) for p in pieces if p and not p.is_empty]
    pieces = [p for p in pieces if p and not p.is_empty]

    total_area = sum(p.area for p in pieces)
    records, line_recs, pt_recs = [], [], []

    for i, p in enumerate(pieces, start=1):
        comp_id  = f"Comp_{i:03d}"
        area_ha  = round(p.area / 10000, 4)
        perim_m  = round(p.length, 4)
        pct_area = round(p.area / total_area * 100, 2) if total_area > 0 else 0
        centroid = p.centroid

        records.append({
            "Forest":   forest_name,
            "Comp_ID":  comp_id,
            "Area_ha":  area_ha,
            "Perim_m":  perim_m,
            "Pct_Area": pct_area,
            "geometry": p,
        })
        ext_coords = list(p.exterior.coords)
        if ext_coords[0] != ext_coords[-1]:
            ext_coords.append(ext_coords[0])
        line_geom = LineString(ext_coords)
        line_recs.append({
            "Forest":  forest_name,
            "Comp_ID": comp_id,
            "geometry": line_geom,
        })
        pt_recs.append({
            "Forest":  forest_name,
            "Comp_ID": comp_id,
            "Area_ha": area_ha,
            "X":       round(centroid.x, 4),
            "Y":       round(centroid.y, 4),
            "geometry": centroid,
        })

    poly_gdf = gpd.GeoDataFrame(records,   crs=crs)
    line_gdf = gpd.GeoDataFrame(line_recs, crs=crs)
    pts_gdf  = gpd.GeoDataFrame(pt_recs,   crs=crs)

    if not poly_gdf.empty:
        poly_gdf.to_file(os.path.join(save_dir, "compartment_polygon.shp"))
    if not line_gdf.empty:
        line_gdf.to_file(os.path.join(save_dir, "compartment_line.shp"))
    if not pts_gdf.empty:
        pts_gdf.to_file(os.path.join(save_dir, "compartment_centroid.shp"))

    # Division points Excel
    div_pts = _extract_division_points(pieces, forest_name)
    if div_pts:
        dp_df = pd.DataFrame(div_pts)
        dp_df.to_excel(os.path.join(save_dir, "division_points.xlsx"), index=False)
        dp_gdf = gpd.GeoDataFrame(dp_df,
                                   geometry=gpd.points_from_xy(dp_df.X, dp_df.Y),
                                   crs=crs)
        dp_gdf.to_file(os.path.join(save_dir, "division_points.shp"))

    # Cut lines
    cut_lines = _extract_internal_cut_lines(pieces)
    if cut_lines:
        cl_gdf = gpd.GeoDataFrame(cut_lines, crs=crs)
        cl_gdf.to_file(os.path.join(save_dir, "cut_lines.shp"))

    # Summary Excel
    summary_rows = [{
        "Forest": forest_name, "Comp_ID": r["Comp_ID"],
        "Area_ha": r["Area_ha"], "Perim_m": r["Perim_m"], "Pct_Area": r["Pct_Area"],
    } for r in records]
    total_row = {
        "Forest": forest_name, "Comp_ID": "TOTAL",
        "Area_ha": round(sum(r["Area_ha"] for r in records), 4),
        "Perim_m": "", "Pct_Area": 100.0,
    }
    summary_rows.append(total_row)
    pd.DataFrame(summary_rows).to_excel(
        os.path.join(save_dir, "compartment_summary.xlsx"), index=False)

    return poly_gdf, line_gdf, pts_gdf


def _df_to_polygon(df, x_col, y_col, order_col):
    if order_col: df = df.sort_values(order_col)
    coords = list(zip(df[x_col], df[y_col]))
    if len(coords) < 3: raise ValueError("Need at least 3 points for a polygon.")
    coords.append(coords[0])
    return safe_polygon(coords)


def _load_polygons_from_zip(zip_file_or_path, target_shp, crs, forest_col_name=None):
    folder = os.path.join(UPLOAD, str(uuid.uuid4()))
    os.makedirs(folder, exist_ok=True)
    zip_path = os.path.join(folder, "input.zip")

    if hasattr(zip_file_or_path, "save"):
        zip_file_or_path.save(zip_path)
    else:
        shutil.copy(zip_file_or_path, zip_path)

    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(folder)

    shp_candidates = []
    for root, _, files in os.walk(folder):
        for f in files:
            if f.endswith(".shp"):
                shp_candidates.append(os.path.join(root, f))
    if not shp_candidates:
        raise ValueError("No .shp found in ZIP.")

    shp_path = shp_candidates[0]
    if target_shp:
        for cand in shp_candidates:
            if os.path.basename(cand) == os.path.basename(target_shp):
                shp_path = cand; break

    gdf = gpd.read_file(shp_path)
    if gdf.empty:
        raise ValueError("The selected shapefile contains no features.")
    if gdf.crs is None:
        gdf = gdf.set_crs(crs)
    else:
        gdf = gdf.to_crs(crs)

    name_col = None
    if forest_col_name:
        for col in gdf.columns:
            if col.lower() == forest_col_name.lower():
                name_col = col; break
    if name_col is None:
        for candidate in ["Forest", "forest", "Name", "name", "NAME", "Label", "label", "ID", "id"]:
            if candidate in gdf.columns:
                name_col = candidate; break
    if name_col is None:
        for col in gdf.columns:
            if col == "geometry": continue
            if gdf[col].dtype == object:
                name_col = col; break

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
                results.append((feat_name, _close_polygon(p)))

    if not results:
        raise ValueError("No polygon geometries found in the selected shapefile.")

    return results, shp_candidates


def group_e(file_or_df, crs, out, mapping=None, e_mode="A", n_compartments=4,
            is_zip=False, forest_col_name=None, area_tol_ha=0.3, method='bisect'):
    """Group E — Polygon Subdivider (2–200 compartments per polygon)."""
    n_compartments = max(2, min(200, int(n_compartments)))

    all_poly_gdfs, all_line_gdfs, all_pts_gdfs = [], [], []

    if is_zip:
        target_shp = (mapping or {}).get("target_shp")
        features, _ = _load_polygons_from_zip(
            file_or_df, target_shp, crs, forest_col_name=forest_col_name)

        if len(features) == 1:
            feat_name, poly = features[0]
            pieces = _subdivide_polygon(poly, n_compartments, area_tol_ha, method=method)
            pg, lg, ptg = _save_compartments(pieces, feat_name, crs, out)
            all_poly_gdfs.append(pg); all_line_gdfs.append(lg); all_pts_gdfs.append(ptg)
        else:
            for feat_name, poly in features:
                pieces   = _subdivide_polygon(poly, n_compartments, area_tol_ha, method=method)
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
            pieces = _subdivide_polygon(poly, n_compartments, area_tol_ha, method=method)
            pg, lg, ptg = _save_compartments(pieces, forest_name, crs, out)
            all_poly_gdfs.append(pg); all_line_gdfs.append(lg); all_pts_gdfs.append(ptg)
        else:
            for f, fg in df.groupby(forest_col):
                try:
                    poly = _df_to_polygon(fg, x_col, y_col, order_col)
                except ValueError:
                    continue
                pieces     = _subdivide_polygon(poly, n_compartments, area_tol_ha, method=method)
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
    Group F — DEM Slope Analysis.
    FIXED: vec_gdf now guaranteed to contain only Polygon geometries.
    Workflow:
      1. Extract rect DEM (20% buffer)
      2. Slope (Horn method)
      3. Reclassify: <19, 19-31, >31 degrees
      4. Raster-to-polygon on whole rectangle
      5. Dissolve by gridcode
      6. Clip dissolved rect polygon to boundary (per compartment/forest)
      7. Build slope tables per group
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

    # ── Load DEM ──────────────────────────────────────────────────────
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

    # ── Load boundary ─────────────────────────────────────────────────
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

    # ── Step 1: Rect DEM extract (20% buffer) ─────────────────────────
    b = boundary_poly.bounds
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
        rect_dem = dem_arr
        rect_tr  = transform
        rect_nodata = nodata if nodata is not None else -9999

    rect_res_x = abs(rect_tr.a)
    rect_res_y = abs(rect_tr.e)

    # ── Step 2: Slope (Horn method) ───────────────────────────────────
    valid_dem = ~np.isnan(rect_dem)
    filled    = np.nan_to_num(rect_dem, nan=0.0)

    dzdx = ndimage.sobel(filled, axis=1) / (8 * max(rect_res_x, 1e-9))
    dzdy = ndimage.sobel(filled, axis=0) / (8 * max(rect_res_y, 1e-9))
    slope_rect = np.degrees(np.arctan(np.sqrt(dzdx**2 + dzdy**2))).astype(np.float32)

    full_neighborhood_valid = ndimage.binary_erosion(
        valid_dem, structure=np.ones((3, 3), dtype=bool), border_value=0
    )
    slope_nodata = -9999.0
    slope_rect[~full_neighborhood_valid] = slope_nodata

    rect_profile = profile.copy()
    rect_profile.update(
        dtype="float32", nodata=slope_nodata, count=1,
        height=slope_rect.shape[0], width=slope_rect.shape[1],
        transform=rect_tr
    )
    slope_rect_path = os.path.join(out, f"{prefix}_slope_rect.tif")
    with rasterio.open(slope_rect_path, "w", **rect_profile) as dst:
        dst.write(slope_rect, 1)

    # ── Step 3: Reclassify ────────────────────────────────────────────
    valid_mask = (slope_rect != slope_nodata) & ~np.isnan(slope_rect)
    class_rect = np.zeros_like(slope_rect, dtype=np.uint8)
    class_rect[valid_mask & (slope_rect < 19)]                       = 1
    class_rect[valid_mask & (slope_rect >= 19) & (slope_rect <= 31)] = 2
    class_rect[valid_mask & (slope_rect > 31)]                       = 3

    class_rect_path = os.path.join(out, f"{prefix}_class_rect.tif")
    cls_profile = rect_profile.copy()
    cls_profile.update(dtype="uint8", nodata=0)
    with rasterio.open(class_rect_path, "w", **cls_profile) as dst:
        dst.write(class_rect, 1)

    # ── Step 4 & 5: Raster-to-polygon, dissolve by class ─────────────
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
                rings    = shp["coordinates"]
                exterior = rings[0]
                holes    = rings[1:] if len(rings) > 1 else None
                geom = _repair_geom(Polygon(exterior, holes=holes))
                if geom and not geom.is_empty and geom.area > 1e-10:
                    # CRITICAL: ensure only Polygon stored
                    pg = _force_polygon_only(geom)
                    if pg:
                        rtp_records.append({"gridcode": cid, "geometry": pg})
            except Exception:
                continue

    if not rtp_records:
        raise ValueError("No slope polygons could be extracted from the DEM.")

    rtp_gdf = gpd.GeoDataFrame(rtp_records, crs=str(dem_crs))

    try:
        dissolved = rtp_gdf.dissolve(by="gridcode", as_index=False)
        dissolved["gridcode"] = dissolved["gridcode"].astype(int)
        # CRITICAL FIX: Force all geometries in dissolved to be Polygon/MultiPolygon only
        def _fix_dissolved_geom(g):
            pg = _force_polygon_only(_repair_geom(g))
            return pg if pg is not None else Polygon()
        dissolved["geometry"] = dissolved["geometry"].apply(_fix_dissolved_geom)
        dissolved = dissolved[~dissolved["geometry"].is_empty].copy()
    except Exception:
        dissolved = rtp_gdf.copy()

    # ── Step 6: Clip to boundary per group ───────────────────────────

    def _clip_group_to_poly(clip_poly, label):
        """
        Clip dissolved slope polygons to clip_poly.
        FIXED: All returned geometries are forced to Polygon type.
        """
        vec_recs = []
        total_valid_ha = 0.0

        for _, drow in dissolved.iterrows():
            cid   = int(drow["gridcode"])
            dgeom = _repair_geom(drow.geometry)
            if dgeom is None or dgeom.is_empty:
                continue
            try:
                clipped_raw = _repair_geom(dgeom.intersection(clip_poly))
            except Exception:
                continue
            if clipped_raw is None or clipped_raw.is_empty:
                continue

            # CRITICAL FIX: Force polygon-only — eliminates LINESTRING error
            clipped = _force_polygon_only(clipped_raw)
            if clipped is None or clipped.is_empty:
                continue

            area_ha = round(clipped.area / 10000, 4)
            if area_ha < 1e-6:
                continue
            total_valid_ha += area_ha
            vec_recs.append({
                "Label":       label,
                "Class":       cid,
                "Slope_Range": class_defs[cid][0],
                "Descr":       class_defs[cid][1],
                "Area_ha":     area_ha,
                "geometry":    clipped,   # guaranteed Polygon/MultiPolygon
            })

        total_valid_ha = max(total_valid_ha, 1e-6)
        rows = []
        for vr in vec_recs:
            rows.append({
                "Label":       vr["Label"],
                "Class":       vr["Class"],
                "Slope_Range": vr["Slope_Range"],
                "Description": vr["Descr"],
                "Area_ha":     vr["Area_ha"],
                "Pct_Area":    round(vr["Area_ha"] / total_valid_ha * 100, 2),
                "Total_ha":    round(total_valid_ha, 4),
            })
        return rows, vec_recs

    # ── Build group polygon list based on f_mode ──────────────────────
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
            if f_mode == "B":
                grp_col = _find_col(boundary_gdf, _FOREST_ALIASES)
            else:
                grp_col = _find_col(boundary_gdf, _COMPARTMENT_ALIASES)
                if grp_col is None:
                    grp_col = _find_col(boundary_gdf, _FOREST_ALIASES)

        if grp_col is None:
            comp_polygons = [(forest_name, boundary_poly)]
        else:
            for val, grp in boundary_gdf.groupby(grp_col):
                union_poly = _repair_geom(grp.unary_union)
                if union_poly and not union_poly.is_empty:
                    comp_polygons.append((str(val), union_poly))

        if not comp_polygons:
            comp_polygons = [(forest_name, boundary_poly)]

    # ── Run clipping for every group ──────────────────────────────────
    all_summary, all_vec = [], []
    per_group_summaries = {}

    for label, cpoly in comp_polygons:
        rows, vec_recs = _clip_group_to_poly(cpoly, label)
        all_summary.extend(rows)
        all_vec.extend(vec_recs)
        per_group_summaries[label] = rows

    # ── Recalibration ─────────────────────────────────────────────────
    if field_area_ha and field_area_ha > 0:
        total_computed = sum(r["Area_ha"] for r in all_summary)
        factor = field_area_ha / max(total_computed, 1e-6)
        for r in all_summary:
            r["Recal_ha"]   = round(r["Area_ha"] * factor, 4)
            r["Cal_Factor"] = round(factor, 6)
    else:
        for r in all_summary:
            r["Recal_ha"]   = None
            r["Cal_Factor"] = None

    # ── Save vector output ────────────────────────────────────────────
    # CRITICAL FIX: Ensure ALL geometries in vec_gdf are Polygon only
    vec_crs = str(dem_crs)
    if all_vec:
        # Force polygon conversion for every record before GDF creation
        clean_vec = []
        for rec in all_vec:
            geom = rec.get("geometry")
            forced = _force_polygon_only(_repair_geom(geom))
            if forced is not None and not forced.is_empty:
                rec2 = {k: v for k, v in rec.items() if k != "geometry"}
                rec2["geometry"] = forced
                clean_vec.append(rec2)

        if clean_vec:
            vec_gdf = gpd.GeoDataFrame(clean_vec, crs=vec_crs)
            # Final validation pass — drop any non-polygon rows
            vec_gdf = vec_gdf[vec_gdf.geometry.geom_type.isin(["Polygon", "MultiPolygon"])].copy()
            if not vec_gdf.empty:
                vec_gdf.to_file(os.path.join(out, f"{prefix}_slope_polygon.shp"))
        else:
            vec_gdf = gpd.GeoDataFrame(columns=["geometry"], crs=vec_crs)
    else:
        vec_gdf = gpd.GeoDataFrame(columns=["geometry"], crs=vec_crs)

    boundary_gdf.to_file(os.path.join(out, f"{prefix}_boundary_polygon.shp"))

    # Bonus clipped rasters
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
    first_ca[valid_mask2 & (first_cs < 19)]                      = 1
    first_ca[valid_mask2 & (first_cs >= 19) & (first_cs <= 31)]  = 2
    first_ca[valid_mask2 & (first_cs > 31)]                      = 3

    clipped_profile = rect_profile.copy()
    clipped_profile.update(height=first_cs.shape[0], width=first_cs.shape[1], transform=first_tr)
    with rasterio.open(os.path.join(out, f"{prefix}_slope_clipped.tif"), "w", **clipped_profile) as dst:
        dst.write(first_cs, 1)
    class_profile2 = clipped_profile.copy()
    class_profile2.update(dtype="uint8", nodata=0)
    with rasterio.open(os.path.join(out, f"{prefix}_slope_classes.tif"), "w", **class_profile2) as dst:
        dst.write(first_ca, 1)

    # ── Excel summary ─────────────────────────────────────────────────
    summary_df = pd.DataFrame(all_summary)
    excel_path = os.path.join(out, f"{prefix}_slope_summary.xlsx")

    if f_mode == "A":
        out_df = summary_df.drop(columns=["Label"], errors="ignore")
        out_df.to_excel(excel_path, index=False)
    else:
        try:
            with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
                summary_df.to_excel(writer, sheet_name="All_Groups", index=False)
                for label, group_rows in per_group_summaries.items():
                    if not group_rows:
                        continue
                    g_df = pd.DataFrame(group_rows)
                    total_ha = sum(r["Area_ha"] for r in group_rows)
                    total_row = {
                        "Label": label, "Class": "", "Slope_Range": "TOTAL",
                        "Description": "", "Area_ha": round(total_ha, 4),
                        "Pct_Area": 100.0, "Total_ha": round(total_ha, 4),
                    }
                    if group_rows[0].get("Recal_ha") is not None:
                        total_row["Recal_ha"]   = round(sum(r["Recal_ha"] for r in group_rows if r.get("Recal_ha")), 4)
                        total_row["Cal_Factor"] = group_rows[0].get("Cal_Factor", "")
                    sheet_name = str(label)[:31]
                    g_df_with_total = pd.concat([g_df, pd.DataFrame([total_row])], ignore_index=True)
                    g_df_with_total.to_excel(writer, sheet_name=sheet_name, index=False)
        except Exception:
            summary_df.to_excel(excel_path, index=False)

    return all_summary, vec_gdf, boundary_gdf, f_mode, per_group_summaries


# ================= PREVIEW FUNCTIONS =================

_COMP_COLORS = [
    "#C8E6C9","#B3E5FC","#FFE0B2","#F8BBD0","#E1BEE7",
    "#DCEDC8","#B2EBF2","#FFF9C4","#D7CCC8","#CFD8DC",
    "#C5CAE9","#F0F4C3","#FFCCBC","#B2DFDB","#E8EAF6",
    "#FCE4EC","#F3E5F5","#E8F5E9","#E3F2FD","#FFF3E0",
]


def _setup_a4_fig(title="", subtitle=""):
    """Create A4-sized figure with map axes."""
    fig = plt.figure(figsize=(A4_W_IN, A4_H_IN), dpi=A4_DPI, facecolor='white')
    # Layout: title bar, map, legend/table
    fig.subplots_adjust(left=0.08, right=0.92, top=0.94, bottom=0.08)
    ax = fig.add_subplot(111)
    ax.set_facecolor('white')
    if title:
        fig.suptitle(title, fontsize=14, fontweight='bold', y=0.97,
                     fontfamily='sans-serif', color='#1a2b22')
    if subtitle:
        ax.set_title(subtitle, fontsize=9, color='#555', style='italic', pad=4)
    return fig, ax


def preview_compartments(poly_gdf, path, title="Compartment Division Map",
                          legend_title="Legend", label_col=None):
    """Group E A4 preview with north arrow, scale bar, legend, graticule."""
    fig, ax = _setup_a4_fig(title=title)

    legend_handles = []
    for i, (_, row) in enumerate(poly_gdf.iterrows()):
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue
        if geom.geom_type == "Polygon":
            ext = list(geom.exterior.coords)
            if ext[0] != ext[-1]:
                ext.append(ext[0])
            geom = Polygon(ext)
        color = _COMP_COLORS[i % len(_COMP_COLORS)]
        comp_id = row.get("Comp_ID", f"Comp_{i+1:03d}")
        area_ha = row.get("Area_ha", None)

        gpd.GeoDataFrame([{"geometry": geom}], crs=poly_gdf.crs).plot(
            ax=ax, facecolor=color, edgecolor="#1a1a1a", linewidth=1.2, zorder=3
        )

        lbl_val = str(row[label_col]) if label_col and label_col in row.index else comp_id
        lbl_txt = f"{lbl_val}"
        if area_ha is not None:
            lbl_txt += f"\n{area_ha:.2f} ha"

        cx, cy = geom.centroid.x, geom.centroid.y
        ax.annotate(lbl_txt, xy=(cx, cy), ha="center", va="center",
                    fontsize=6.5, fontweight="bold", color="#1a1a1a",
                    path_effects=[pe.Stroke(linewidth=2, foreground="white"), pe.Normal()],
                    zorder=10)
        legend_handles.append(mpatches.Patch(facecolor=color, edgecolor="#1a1a1a", label=comp_id))

    ax.set_aspect("equal")
    ax.axis("off")

    # Decorations
    _add_graticule(ax)
    _add_north_arrow(ax)
    _add_scale_bar(ax)
    _add_legend(ax, legend_handles[:15], title=legend_title)

    # Neat border
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.2)
        spine.set_color('#333')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(path, dpi=A4_DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def preview(poly_gdf, line_gdf, pts_gdf, path, pc="blue", lc="black", ptc="red",
            label_col=None, label_pts_gdf=None, area_ha=None,
            title="Forest Boundary Map", legend_title="Legend",
            poly_label="Forest Boundary", pts_label="Survey Points"):
    """A4 preview with north arrow, scale bar, legend, graticule — Groups A/B/C/D."""
    fig, ax = _setup_a4_fig(title=title)

    legend_handles = []

    if poly_gdf is not None and not poly_gdf.empty:
        poly_gdf.plot(ax=ax, facecolor="none", edgecolor=pc, linewidth=1.8, zorder=3)
        legend_handles.append(
            mpatches.Patch(facecolor='none', edgecolor=pc, linewidth=1.8, label=poly_label)
        )

    if line_gdf is not None and not line_gdf.empty:
        line_gdf.plot(ax=ax, color=lc, linewidth=1.2, zorder=2)

    if pts_gdf is not None and not pts_gdf.empty:
        pts_gdf.plot(ax=ax, color=ptc, markersize=8, zorder=5)
        legend_handles.append(
            Line2D([0], [0], marker='o', color='w', markerfacecolor=ptc,
                   markersize=6, label=pts_label)
        )

    # Labels for survey points
    lbl_src = label_pts_gdf if label_pts_gdf is not None else pts_gdf
    if label_col and lbl_src is not None and not lbl_src.empty and label_col in lbl_src.columns:
        for _, row in lbl_src.iterrows():
            ax.annotate(str(int(row[label_col])),
                        xy=(row.geometry.x, row.geometry.y),
                        xytext=(0, 7), textcoords="offset points",
                        ha="center", va="bottom", fontsize=5, fontweight="bold", color="black",
                        path_effects=[pe.Stroke(linewidth=1.8, foreground="white"), pe.Normal()],
                        zorder=6)

    # Compartment labels
    if poly_gdf is not None and not poly_gdf.empty and "Comp_ID" in poly_gdf.columns:
        for _, row in poly_gdf.iterrows():
            if row.geometry is None or row.geometry.is_empty:
                continue
            cx = row.geometry.centroid.x
            cy = row.geometry.centroid.y
            area_txt = f"{row['Area_ha']:.2f} ha" if "Area_ha" in row else ""
            ax.annotate(f"{row['Comp_ID']}\n{area_txt}",
                        xy=(cx, cy), ha="center", va="center", fontsize=5.5,
                        fontweight="bold", color="black",
                        path_effects=[pe.Stroke(linewidth=2, foreground="white"), pe.Normal()],
                        zorder=7)

    if area_ha is not None:
        ax.text(0.02, 0.98, f"Total Area: {area_ha:.3f} ha",
                transform=ax.transAxes, ha='left', va='top',
                fontsize=9, fontweight='bold', color='#1a2b22',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8), zorder=15)

    ax.set_aspect("equal")
    ax.axis("off")

    _add_graticule(ax)
    _add_north_arrow(ax)
    _add_scale_bar(ax)
    if legend_handles:
        _add_legend(ax, legend_handles, title=legend_title)

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.2)
        spine.set_color('#333')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(path, dpi=A4_DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def preview_slope(vec_gdf, boundary_gdf, summary_rows, path,
                  f_mode="A", per_group_summaries=None,
                  title="Slope Classification Map", legend_title="Legend"):
    """
    A4 slope preview with map + table.
    Mode A: map + single summary table
    Mode B/E: map + per-group tables
    """
    import matplotlib.gridspec as gridspec

    class_colors = {1: "#2ecc71", 2: "#f39c12", 3: "#e74c3c"}
    class_labels_map = {1: "< 19°  Gentle", 2: "19–31° Moderate", 3: "> 31°  Steep"}

    has_groups = f_mode in ("B", "E") and per_group_summaries and len(per_group_summaries) > 1
    n_rows_total = len(summary_rows) if summary_rows else 0

    # A4 figure
    fig = plt.figure(figsize=(A4_W_IN, A4_H_IN), dpi=A4_DPI, facecolor="white")
    if title:
        fig.suptitle(title, fontsize=13, fontweight='bold', y=0.98, color='#1a2b22')

    table_ratio = max(0.22, min(0.45, 0.06 + n_rows_total * 0.025))
    gs = gridspec.GridSpec(2, 1, height_ratios=[1, table_ratio], hspace=0.15,
                           left=0.06, right=0.94, top=0.95, bottom=0.04)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])

    # ── Map ────────────────────────────────────────────────────────────
    if vec_gdf is not None and not vec_gdf.empty and "Class" in vec_gdf.columns:
        for cid, color in class_colors.items():
            sub = vec_gdf[vec_gdf["Class"] == cid]
            if not sub.empty:
                sub.plot(ax=ax1, facecolor=color, edgecolor="#2c2c2c",
                         linewidth=0.4, alpha=0.92, zorder=3)

    if boundary_gdf is not None and not boundary_gdf.empty:
        try:
            boundary_gdf.boundary.plot(ax=ax1, color="black", linewidth=1.8, zorder=5)
        except Exception:
            pass

    # Label groups on map
    if has_groups and boundary_gdf is not None and not boundary_gdf.empty:
        grp_col = None
        for c in boundary_gdf.columns:
            if c.lower() not in ("geometry",):
                grp_col = c; break
        if grp_col:
            for _, row in boundary_gdf.iterrows():
                geom = row.geometry
                if geom and not geom.is_empty:
                    cx, cy = geom.centroid.x, geom.centroid.y
                    ax1.annotate(str(row[grp_col]), xy=(cx, cy),
                                 ha="center", va="center", fontsize=7,
                                 fontweight="bold", color="#1a1a1a",
                                 path_effects=[pe.Stroke(linewidth=2.5, foreground="white"), pe.Normal()],
                                 zorder=8)

    ax1.set_aspect("equal")
    ax1.axis("off")

    # Map decorations
    _add_graticule(ax1)
    _add_north_arrow(ax1)
    _add_scale_bar(ax1)

    # Legend
    legend_items = [mpatches.Patch(facecolor=c, edgecolor="#2c2c2c", label=class_labels_map[cid])
                    for cid, c in class_colors.items()]
    _add_legend(ax1, legend_items, title=legend_title)

    for spine in ax1.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.0)
        spine.set_color('#333')

    # ── Table ──────────────────────────────────────────────────────────
    ax2.axis("off")
    ax2.set_title("Slope Area Summary", fontsize=10, fontweight="bold", pad=4)

    if not summary_rows:
        fig.savefig(path, dpi=A4_DPI, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        return

    has_recal = any(r.get("Recal_ha") is not None for r in summary_rows)

    if f_mode == "A":
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
        row_class_map = {i: summary_rows[i].get("Class", 0) for i in range(len(summary_rows))}
    else:
        col_labels = ["Group", "Slope Range", "Description", "Area (ha)", "% of Total"]
        if has_recal:
            col_labels += ["Recal. (ha)"]
        table_data = []
        row_class_map = {}
        row_idx = 0

        if per_group_summaries:
            for label, group_rows in per_group_summaries.items():
                for r in group_rows:
                    row = [str(label), r["Slope_Range"], r["Description"],
                           f"{r['Area_ha']:.2f}", f"{r['Pct_Area']:.1f}%"]
                    if has_recal:
                        row += [f"{r['Recal_ha']:.2f}" if r.get("Recal_ha") else "—"]
                    table_data.append(row)
                    row_class_map[row_idx] = r.get("Class", 0)
                    row_idx += 1
                g_total = sum(r["Area_ha"] for r in group_rows)
                subtotal_row = [f"  ↳ {label} Total", "", "", f"{g_total:.2f}", ""]
                if has_recal:
                    g_recal = sum(r["Recal_ha"] for r in group_rows if r.get("Recal_ha"))
                    subtotal_row += [f"{g_recal:.2f}"]
                table_data.append(subtotal_row)
                row_class_map[row_idx] = -1
                row_idx += 1
        else:
            for r in summary_rows:
                row = [r.get("Label",""), r["Slope_Range"], r["Description"],
                       f"{r['Area_ha']:.2f}", f"{r['Pct_Area']:.1f}%"]
                if has_recal:
                    row += [f"{r['Recal_ha']:.2f}" if r.get("Recal_ha") else "—"]
                table_data.append(row)
                row_class_map[row_idx] = r.get("Class", 0)
                row_idx += 1

    if not table_data:
        fig.savefig(path, dpi=A4_DPI, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        return

    tbl = ax2.table(cellText=table_data, colLabels=col_labels,
                    cellLoc="center", loc="center", bbox=[0.0, 0.0, 1.0, 1.0])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(7.0 if has_groups else 8.5)
    tbl.auto_set_column_width(list(range(len(col_labels))))

    row_colors = {1: "#d5f5e3", 2: "#fdebd0", 3: "#fadbd8",
                  0: "#f8f9fa", -1: "#ddeeff"}
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor("#cccccc")
        if r == 0:
            cell.set_facecolor("#1a5276")
            cell.set_text_props(color="white", fontweight="bold")
        else:
            ri = r - 1
            cid = row_class_map.get(ri, 0)
            if cid == -1:
                cell.set_facecolor("#ddeeff")
                cell.set_text_props(fontweight="bold")
            elif cid == 0 and ri >= len(summary_rows):
                cell.set_facecolor("#ddeeff")
                cell.set_text_props(fontweight="bold")
            else:
                cell.set_facecolor(row_colors.get(cid, "#f8f9fa"))

    fig.savefig(path, dpi=A4_DPI, bbox_inches="tight", facecolor="white")
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

        def polygon_kml(g):
            outer = coords_str(list(g.exterior.coords))
            rings = [f"<outerBoundaryIs><LinearRing><coordinates>{outer}</coordinates></LinearRing></outerBoundaryIs>"]
            for interior in g.interiors:
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
    <LineStyle><color>ff0000ff</color><width>2</width></LineStyle>
    <PolyStyle><color>44ff0000</color></PolyStyle>
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


# ================= GeoJSON for OSM preview =================
def _gdf_to_geojson_wgs84(gdf):
    """Convert GeoDataFrame to WGS84 GeoJSON string."""
    if gdf is None or gdf.empty:
        return None
    try:
        wgs = gdf.to_crs("EPSG:4326")
        return wgs.to_json()
    except Exception:
        return None


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
        return jsonify({
            "error": f"Username \"{username}\" is already taken. Please choose a different name.",
            "taken": True
        }), 409

    try:
        user = _create_user(username)
    except ValueError as e:
        return jsonify({"error": str(e), "taken": True}), 409

    session["username"]  = username
    session.permanent    = True
    return jsonify({"ok": True, "username": username, "runs": []})


@app.route("/logout", methods=["POST"])
def logout():
    session.clear()
    return jsonify({"ok": True})


@app.route("/me")
def me():
    username = _require_login()
    if not username:
        return jsonify({"error": "Not logged in"}), 401
    users = _load_users()
    user  = users.get(username, {})
    return jsonify({"username": username, "runs": user.get("runs", [])[-20:]})


@app.route("/history")
def history():
    username = _require_login()
    if not username:
        return jsonify({"error": "Not logged in"}), 401
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

        # Title and legend params
        map_title     = request.form.get("map_title", "").strip() or None
        legend_title  = request.form.get("legend_title", "Legend").strip() or "Legend"
        poly_label    = request.form.get("poly_label", "Forest Boundary").strip() or "Forest Boundary"
        pts_label     = request.form.get("pts_label", "Survey Points").strip() or "Survey Points"
        label_col_req = request.form.get("label_col", "").strip() or None

        run_id   = str(uuid.uuid4())
        username = _require_login() or "guest"
        out      = os.path.join(OUTPUT, run_id)
        os.makedirs(out, exist_ok=True)
        crs = get_crs(zone)

        label_col      = label_col_req
        label_pts_gdf  = None
        area_ha_display = None

        if module == "B":
            df = read_input(file)
            poly, line, pts = group_b(df, crs, out, mapping)
            default_title = map_title or f"Forest Boundary Map — {forest}"

        elif module == "C":
            poly, line, pts = group_c(file, crs, w, h, rows, cols, out, mode, mapping)
            if not label_col:
                label_col = "SN"
            label_pts_gdf = pts
            default_title = map_title or "Forest Sample Plot Map"

        elif module == "D":
            df = read_input(file)
            d_mode = request.form.get("d_mode", "A")
            poly, line, pts = group_d(df, crs, out, mapping, mode=d_mode)
            default_title = map_title or f"Multi-Forest Map — {d_mode} Mode"

        elif module == "E":
            e_mode          = request.form.get("e_mode", "A")
            n_compartments  = max(2, min(200, int(request.form.get("n_compartments", 4))))
            area_tol_ha     = float(request.form.get("area_tol_ha", "0.3") or "0.3")
            e_method        = request.form.get("e_method", "bisect").strip() or "bisect"
            is_zip          = file.filename.lower().endswith(".zip")
            forest_col_name = request.form.get("forest_col_name") or None

            if mapping and "forest" not in mapping:
                mapping["forest"] = forest

            if is_zip:
                poly, line, pts = group_e(
                    file, crs, out, mapping,
                    e_mode=e_mode, n_compartments=n_compartments,
                    is_zip=True, forest_col_name=forest_col_name,
                    area_tol_ha=area_tol_ha, method=e_method
                )
            else:
                df = read_input(file)
                poly, line, pts = group_e(
                    df, crs, out, mapping,
                    e_mode=e_mode, n_compartments=n_compartments,
                    is_zip=False, forest_col_name=forest_col_name,
                    area_tol_ha=area_tol_ha, method=e_method
                )

            preview_path = os.path.join(out, "output.png")
            preview_compartments(poly, preview_path,
                                  title=map_title or f"Compartment Division — {n_compartments} Parts",
                                  legend_title=legend_title,
                                  label_col=label_col)

            # GeoJSON for OSM
            geojson_str = _gdf_to_geojson_wgs84(poly)
            geojson_path = os.path.join(out, "output.geojson")
            if geojson_str:
                with open(geojson_path, "w") as gf:
                    gf.write(geojson_str)

            kmz_url = None
            try:
                kmz_url = generate_kmz(poly, line, pts, out, run_id)
            except Exception:
                pass

            _append_run(username, run_id, "E")
            return jsonify({
                "run_id": run_id,
                "download": f"/download/{run_id}",
                "kmz_url": kmz_url,
                "has_geojson": geojson_str is not None
            })

        elif module == "F":
            dem_file_upload = request.files.get("dem_file")
            if not dem_file_upload:
                return jsonify({"error": "No DEM file uploaded. Please upload a GeoTIFF DEM."}), 400

            f_forest        = request.form.get("f_forest") or forest
            f_mode          = request.form.get("f_mode", "A")
            comp_col_name   = request.form.get("comp_col") or None
            boundary_is_zip = file.filename.lower().endswith(".zip")

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
            summary_rows, vec_gdf, boundary_gdf, f_mode_out, per_group_summaries = result

            preview_path = os.path.join(out, "output.png")
            preview_slope(vec_gdf, boundary_gdf, summary_rows, preview_path,
                          f_mode=f_mode_out, per_group_summaries=per_group_summaries,
                          title=map_title or f"Slope Classification — {f_forest}",
                          legend_title=legend_title)

            # GeoJSON for OSM
            geojson_str = _gdf_to_geojson_wgs84(vec_gdf)
            if geojson_str:
                with open(os.path.join(out, "output.geojson"), "w") as gf:
                    gf.write(geojson_str)

            poly = vec_gdf if (vec_gdf is not None and not vec_gdf.empty) else gpd.GeoDataFrame()
            line = gpd.GeoDataFrame()
            pts  = gpd.GeoDataFrame()

            kmz_url = None
            try:
                kmz_url = generate_kmz(poly, line, pts, out, run_id)
            except Exception:
                pass

            _append_run(username, run_id, "F")
            return jsonify({
                "run_id": run_id,
                "download": f"/download/{run_id}",
                "kmz_url": kmz_url,
                "has_geojson": geojson_str is not None
            })

        else:  # module == "A"
            df = read_input(file)
            poly, line, pts = group_a(df, forest, crs, out, mapping)
            if not poly.empty and "Area_ha" in poly.columns:
                area_ha_display = float(poly["Area_ha"].sum())
            default_title = map_title or f"Forest Boundary — {forest}"

        # ── Standard preview (A, B, C, D) ───────────────────────────────
        preview_path = os.path.join(out, "output.png")
        preview(poly, line, pts, preview_path,
                pc="blue", lc="black", ptc="red",
                label_col=label_col, label_pts_gdf=label_pts_gdf,
                area_ha=area_ha_display,
                title=map_title or default_title,
                legend_title=legend_title,
                poly_label=poly_label,
                pts_label=pts_label)

        # GeoJSON for OSM
        geojson_str = _gdf_to_geojson_wgs84(poly)
        if geojson_str:
            with open(os.path.join(out, "output.geojson"), "w") as gf:
                gf.write(geojson_str)

        kmz_url = None
        try:
            kmz_url = generate_kmz(poly, line, pts, out, run_id)
        except Exception:
            pass

        _append_run(username, run_id, module)
        return jsonify({
            "run_id": run_id,
            "download": f"/download/{run_id}",
            "kmz_url": kmz_url,
            "has_geojson": geojson_str is not None
        })

    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        return jsonify({"error": f"Unexpected error: {e}", "traceback": tb}), 500


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
