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

from flask import (
    Flask,
    render_template,
    request,
    send_file,
    jsonify,
    send_from_directory
)

from shapely.geometry import Polygon, Point, LineString

app = Flask(__name__)

UPLOAD = "uploads"
OUTPUT = "outputs"

os.makedirs(UPLOAD, exist_ok=True)
os.makedirs(OUTPUT, exist_ok=True)

# ================= SAFE FILE READER =================
def read_input(file):
    name = file.filename.lower()
    if name.endswith(".csv"):
        try:
            return pd.read_csv(file, encoding="utf-8-sig")
        except Exception:
            file.seek(0)
            return pd.read_csv(file, encoding="latin1")
    elif name.endswith((".xlsx", ".xls")):
        return pd.read_excel(file)
    raise ValueError("Only CSV/Excel supported")

# ================= CRS =================
def get_crs(zone):
    return f"EPSG:326{int(zone)}"

# ================= NORMALIZE =================
def normalize_columns(df):
    df.columns = (
        df.columns.astype(str)
        .str.strip()
        .str.lower()
        .str.replace(" ", "")
        .str.replace("-", "")
        .str.replace("_", "")
    )
    return df

# ================= ORDER FIX =================
def normalize_order(df):
    df = normalize_columns(df)

    rename_map = {}
    for col in df.columns:
        if col in ["order", "ordering", "ord", "serial", "sno", "sn", "sn."]:
            rename_map[col] = "order"

    if rename_map:
        df = df.rename(columns=rename_map)

    return df

# ================= SAFE COLUMN RESOLVER (FIXED) =================
def resolve_col(df, mapping, key):
    df = normalize_columns(df)
    mapping = mapping or {}

    def norm(x):
        return str(x).strip().lower().replace(" ", "").replace("-", "").replace("_", "")

    clean_mapping = {norm(k): v for k, v in mapping.items()}
    cols = {norm(c): c for c in df.columns}

    key_norm = norm(key)

    if key_norm in clean_mapping and clean_mapping[key_norm]:
        target = norm(clean_mapping[key_norm])

        if target in cols:
            return cols[target]

        raise ValueError(
            f"UI mapping error: '{clean_mapping[key_norm]}' not found in file columns."
        )

    raise ValueError(f"Missing UI mapping for required field: '{key}'")

# ================= SAFE POLYGON =================
def safe_polygon(coords):
    if len(coords) < 3:
        return None

    if coords[0] != coords[-1]:
        coords.append(coords[0])

    poly = Polygon(coords)

    if poly.is_empty:
        return None

    if not poly.is_valid:
        poly = poly.buffer(0)

    if poly.is_empty:
        return None

    return poly

# ================= GROUP A =================
def group_a(df, forest, crs, out, mapping):
    df = normalize_columns(df)

    x = resolve_col(df, mapping, "X")
    y = resolve_col(df, mapping, "Y")
    order = resolve_col(df, mapping, "Order")

    df = df.sort_values(order)

    coords = list(zip(df[x], df[y]))

    poly = safe_polygon(coords)
    if poly is None:
        raise ValueError("Invalid polygon (Group A)")

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
        geometry=gpd.points_from_xy(df[x], df[y]),
        crs=crs
    )

    poly_gdf.to_file(os.path.join(out, "polygon.shp"))
    line_gdf.to_file(os.path.join(out, "line.shp"))
    pts_gdf.to_file(os.path.join(out, "points.shp"))

    return poly_gdf, line_gdf, pts_gdf

# ================= GROUP B =================
def group_b(df, crs, out, mapping):
    df = normalize_order(df)

    x = resolve_col(df, mapping, "X")
    y = resolve_col(df, mapping, "Y")
    order = resolve_col(df, mapping, "Order")
    forest = resolve_col(df, mapping, "Forest")
    comp = resolve_col(df, mapping, "Compartment")

    polys, lines, pts = [], [], []

    for f, g in df.groupby(forest):
        for c, cg in g.groupby(comp):

            cg = cg.sort_values(order)
            coords = list(zip(cg[x], cg[y]))

            poly = safe_polygon(coords)
            if poly is None:
                continue

            lines.append({
                "Forest": f,
                "Compartment": c,
                "geometry": LineString(coords)
            })

            polys.append({
                "Forest": f,
                "Compartment": c,
                "Area": poly.area / 10000,
                "Perim": poly.length,
                "geometry": poly
            })

            for _, r in cg.iterrows():
                pts.append({
                    "Forest": f,
                    "Compartment": c,
                    "Order": r[order],
                    "geometry": Point(r[x], r[y])
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

# ================= GROUP C (FIXED UNION + CRS SAFE) =================
def group_c(file, crs, w, h, rows, cols, out,
            base_mode="A", mapping=None, selected_shp=None):

    polygons = []

    if file.filename.lower().endswith(".zip"):
        temp_dir = os.path.join(UPLOAD, f"tmp_{uuid.uuid4()}")
        os.makedirs(temp_dir, exist_ok=True)

        try:
            zip_path = os.path.join(temp_dir, "input.zip")
            file.save(zip_path)

            with zipfile.ZipFile(zip_path) as z:
                z.extractall(temp_dir)

            shp_path = None
            for root, _, files in os.walk(temp_dir):
                for f in files:
                    if f.lower().endswith(".shp") and f == selected_shp:
                        shp_path = os.path.join(root, f)
                        break

            if not shp_path:
                raise ValueError("Shapefile not found")

            gdf = gpd.read_file(shp_path)

            if gdf.crs is None:
                gdf.set_crs(crs, inplace=True)

            geom = gdf.geometry.union_all()

            if geom.geom_type == "Polygon":
                polygons = [geom]
            elif geom.geom_type == "MultiPolygon":
                polygons = list(geom.geoms)
            else:
                raise ValueError("Unsupported geometry")

        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    else:
        df = normalize_order(read_input(file))

        if base_mode == "A":
            x = resolve_col(df, mapping, "X")
            y = resolve_col(df, mapping, "Y")
            order = resolve_col(df, mapping, "Order")

            df = df.sort_values(order)
            coords = list(zip(df[x], df[y]))

            poly = safe_polygon(coords)
            polygons = [poly] if poly else []

        elif base_mode == "B":
            x = resolve_col(df, mapping, "X")
            y = resolve_col(df, mapping, "Y")
            order = resolve_col(df, mapping, "Order")
            forest = resolve_col(df, mapping, "Forest")
            comp = resolve_col(df, mapping, "Compartment")

            for _, g in df.groupby([forest, comp]):
                g = g.sort_values(order)
                coords = list(zip(g[x], g[y]))

                poly = safe_polygon(coords)
                if poly:
                    polygons.append(poly)

    if not polygons:
        raise ValueError("No polygons generated")

    poly_gdf = gpd.GeoDataFrame([{"geometry": p} for p in polygons], crs=crs)

    union = poly_gdf.geometry.union_all()
    minx, miny, maxx, maxy = union.bounds

    pts = []
    sn = 1

    for r in range(rows):
        for c in range(cols):
            x = minx + c * w
            y = miny + r * h

            center = Point(x + w/2, y + h/2)

            if union.contains(center):
                pts.append({"SN": sn, "X": center.x, "Y": center.y, "geometry": center})
                sn += 1

    pts_gdf = gpd.GeoDataFrame(pts, crs=crs)

    return poly_gdf, pts_gdf, pts_gdf

# ================= PREVIEW =================
def preview(poly, line, pts, path):
    fig, ax = plt.subplots(figsize=(6, 6))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    if poly is not None and not poly.empty:
        poly.plot(ax=ax, facecolor="#fde047", edgecolor="black")

    if line is not None and not line.empty:
        line.plot(ax=ax, color="black")

    if pts is not None and not pts.empty:
        pts.plot(ax=ax, color="red", markersize=20)

    ax.set_axis_off()
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
