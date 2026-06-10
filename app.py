import os
import zipfile
import pandas as pd
import geopandas as gpd
import streamlit as st
import matplotlib.pyplot as plt

from shapely.geometry import Point, LineString, Polygon
from io import BytesIO

# =========================
# APP CONFIG
# =========================
st.set_page_config(page_title="Community Forest GIS Tool", layout="centered")

OUTPUT_FOLDER = "OUTPUT"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

st.title("🌲 Community Forest GIS Tool (Unified App)")


# =========================
# CRS FUNCTION
# =========================
def get_crs(zone):
    return "EPSG:32644" if zone == "44" else "EPSG:32645"


# =========================
# ZIP FUNCTION
# =========================
def make_zip(folder_path):
    zip_buffer = BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(folder_path):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, folder_path)
                zipf.write(file_path, arcname)
    zip_buffer.seek(0)
    return zip_buffer


# =========================
# MODE SELECTION
# =========================
mode = st.radio(
    "Select Processing Mode",
    [
        "🌳 Whole Forest Boundary",
        "🌲 Segmented Forest (Compartment-wise)"
    ]
)

uploaded_file = st.file_uploader("📤 Upload Excel File", type=["xlsx", "xls"])
zone = st.radio("UTM Zone", ["44", "45"], index=1)


# ============================================================
# MODE 1: WHOLE FOREST BOUNDARY
# ============================================================
def process_whole_forest(file, zone):
    df = pd.read_excel(file)

    required = ["Forest", "X", "Y", "Order"]
    for c in required:
        if c not in df.columns:
            st.error(f"Missing column: {c}")
            return {}

    crs = get_crs(zone)
    grouped = df.groupby("Forest")

    results = {}

    for forest, group in grouped:
        group = group.sort_values("Order")
        group["X"] = pd.to_numeric(group["X"], errors="coerce")
        group["Y"] = pd.to_numeric(group["Y"], errors="coerce")
        group = group.dropna()

        coords = list(zip(group["X"], group["Y"]))
        if len(coords) < 3:
            continue

        if coords[0] != coords[-1]:
            coords.append(coords[0])

        polygon = Polygon(coords).buffer(0)
        if not polygon.is_valid:
            continue

        line = LineString(coords)

        area_ha = polygon.area / 10000

        gdf_points = gpd.GeoDataFrame(
            group.copy(),
            geometry=[Point(xy) for xy in coords[:len(group)]],
            crs=crs
        )

        gdf_line = gpd.GeoDataFrame(geometry=[line], crs=crs)

        gdf_poly = gpd.GeoDataFrame(
            {"Forest": [forest], "Area_Ha": [round(area_ha, 4)]},
            geometry=[polygon],
            crs=crs
        )

        folder = os.path.join(OUTPUT_FOLDER, f"whole_{forest}")
        os.makedirs(folder, exist_ok=True)

        gdf_points.to_file(os.path.join(folder, "points.shp"))
        gdf_line.to_file(os.path.join(folder, "line.shp"))
        gdf_poly.to_file(os.path.join(folder, "polygon.shp"))

        results[forest] = make_zip(folder)

    return results


# ============================================================
# MODE 2: SEGMENTED FOREST
# ============================================================
def process_segmented(file, zone):
    df = pd.read_excel(file)

    required = ["Forest", "Compartment", "X", "Y", "Order"]
    for c in required:
        if c not in df.columns:
            st.error(f"Missing column: {c}")
            return {}

    crs = get_crs(zone)
    grouped = df.groupby("Forest")

    results = {}

    for forest_name, group in grouped:

        all_points = []
        all_lines = []
        all_polygons = []

        group["X"] = pd.to_numeric(group["X"], errors="coerce")
        group["Y"] = pd.to_numeric(group["Y"], errors="coerce")
        group = group.dropna(subset=["X", "Y"])

        for comp, comp_group in group.groupby("Compartment"):
            comp_group = comp_group.sort_values("Order")

            coords = list(zip(comp_group["X"], comp_group["Y"]))
            if len(coords) < 3:
                continue

            if coords[0] != coords[-1]:
                coords.append(coords[0])

            polygon = Polygon(coords).buffer(0)
            if not polygon.is_valid:
                continue

            line = LineString(coords)

            area_ha = polygon.area / 10000
            perimeter = polygon.length

            all_polygons.append({
                "Forest": str(forest_name),
                "Comp": str(comp),
                "Area_Ha": round(area_ha, 4),
                "Perim_M": round(perimeter, 2),
                "geometry": polygon
            })

            all_lines.append({
                "Forest": str(forest_name),
                "Comp": str(comp),
                "geometry": line
            })

            for _, r in comp_group.iterrows():
                all_points.append({
                    "Forest": str(forest_name),
                    "Comp": str(comp),
                    "Order": r["Order"],
                    "geometry": Point(r["X"], r["Y"])
                })

        if not all_polygons:
            continue

        gdf_points = gpd.GeoDataFrame(all_points, crs=crs)
        gdf_lines = gpd.GeoDataFrame(all_lines, crs=crs)
        gdf_polygons = gpd.GeoDataFrame(all_polygons, crs=crs)

        folder = os.path.join(OUTPUT_FOLDER, f"seg_{forest_name}")
        os.makedirs(folder, exist_ok=True)

        gdf_points.to_file(os.path.join(folder, "points.shp"))
        gdf_lines.to_file(os.path.join(folder, "lines.shp"))
        gdf_polygons.to_file(os.path.join(folder, "polygons.shp"))

        # Map preview
        fig, ax = plt.subplots(figsize=(6, 6))
        gdf_polygons.plot(ax=ax, alpha=0.4, edgecolor="black")
        gdf_points.plot(ax=ax, color="red", markersize=15)

        for _, row in gdf_polygons.iterrows():
            c = row.geometry.centroid
            ax.text(c.x, c.y, str(row["Comp"]), fontsize=8)

        ax.set_title(forest_name)
        ax.axis("off")

        st.pyplot(fig)

        results[forest_name] = make_zip(folder)

    return results


# =========================
# RUN APP
# =========================
if uploaded_file:

    if st.button("🚀 Run Processing"):

        if mode == "🌳 Whole Forest Boundary":
            results = process_whole_forest(uploaded_file, zone)

        else:
            results = process_segmented(uploaded_file, zone)

        if not results:
            st.warning("No valid outputs generated.")
        else:
            st.success("Processing completed!")

            for name, zip_buffer in results.items():
                st.download_button(
                    label=f"⬇ Download {name} ZIP",
                    data=zip_buffer,
                    file_name=f"{name}.zip",
                    mime="application/zip"
                )
