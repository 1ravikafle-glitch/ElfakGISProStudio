import os
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import threading

from shapely.geometry import Polygon, Point, LineString

import tkinter as tk
from tkinter import filedialog, messagebox


# ======================================================
# CRS (FIXED SAFE VERSION)
# ======================================================
CRS_VAR = None


def get_crs(zone):
    return "EPSG:32644" if zone == "44" else "EPSG:32645"


# ======================================================
# FILE SELECT
# ======================================================
def select_file():
    return filedialog.askopenfilename(
        filetypes=[("Excel Files", "*.xlsx *.xls")]
    )


def select_output_folder():
    return filedialog.askdirectory(title="Select Output Folder")


# ======================================================
# WHOLE FOREST BOUNDARY (1–24)  [UNCHANGED LOGIC]
# ======================================================
def process_whole_forest(df, crs, output_dir):

    grouped = df.groupby("Forest")

    for forest, group in grouped:

        forest_dir = os.path.join(output_dir, "Whole_Forest_Boundary", str(forest))
        os.makedirs(forest_dir, exist_ok=True)

        group["X"] = pd.to_numeric(group["X"], errors="coerce")
        group["Y"] = pd.to_numeric(group["Y"], errors="coerce")
        group = group.dropna()

        group = group.sort_values("Order")

        coords = list(zip(group["X"], group["Y"]))

        # FIX: safety check added
        if len(coords) < 3:
            continue

        coords.append(coords[0])

        polygon = Polygon(coords)
        line = LineString(coords)

        gdf_poly = gpd.GeoDataFrame(
            [{
                "Forest": forest,
                "Area_Ha": polygon.area / 10000,
                "Perim_M": polygon.length,
                "geometry": polygon
            }],
            crs=crs
        )

        gdf_line = gpd.GeoDataFrame(
            [{"Forest": forest, "geometry": line}],
            crs=crs
        )

        gdf_points = gpd.GeoDataFrame(
            group,
            geometry=gpd.points_from_xy(group["X"], group["Y"]),
            crs=crs
        )

        gdf_points.to_file(os.path.join(forest_dir, "points.shp"))
        gdf_line.to_file(os.path.join(forest_dir, "line.shp"))
        gdf_poly.to_file(os.path.join(forest_dir, "polygon.shp"))

        # preview
        fig, ax = plt.subplots()
        gdf_poly.plot(ax=ax, alpha=0.4)
        gdf_points.plot(ax=ax, color="red")
        plt.title(forest)
        plt.axis("off")
        plt.savefig(os.path.join(forest_dir, "map.png"), dpi=300)
        plt.close()


# ======================================================
# PART FOREST BOUNDARY (25–28) [UNCHANGED LOGIC]
# ======================================================
def process_part_forest(df, crs, output_dir):

    grouped = df.groupby("Forest")

    for forest, group in grouped:

        forest_dir = os.path.join(output_dir, "Part_Forest_Boundary", str(forest))
        os.makedirs(forest_dir, exist_ok=True)

        group["X"] = pd.to_numeric(group["X"], errors="coerce")
        group["Y"] = pd.to_numeric(group["Y"], errors="coerce")
        group = group.dropna()

        all_polygons = []
        all_lines = []
        all_points = []

        for comp, cgroup in group.groupby("Compartment"):

            cgroup = cgroup.sort_values("Order")

            coords = list(zip(cgroup["X"], cgroup["Y"]))

            if len(coords) < 3:
                continue

            coords.append(coords[0])

            poly = Polygon(coords).buffer(0)
            line = LineString(coords)

            all_polygons.append({
                "Forest": forest,
                "Comp": comp,
                "Area_Ha": poly.area / 10000,
                "Perim_M": poly.length,
                "geometry": poly
            })

            all_lines.append({
                "Forest": forest,
                "Comp": comp,
                "geometry": line
            })

            for _, row in cgroup.iterrows():
                all_points.append({
                    "Forest": forest,
                    "Comp": comp,
                    "Order": row["Order"],
                    "geometry": Point(row["X"], row["Y"])
                })

        gdf_poly = gpd.GeoDataFrame(all_polygons, crs=crs)
        gdf_line = gpd.GeoDataFrame(all_lines, crs=crs)
        gdf_point = gpd.GeoDataFrame(all_points, crs=crs)

        gdf_poly.to_file(os.path.join(forest_dir, "polygons.shp"))
        gdf_line.to_file(os.path.join(forest_dir, "lines.shp"))
        gdf_point.to_file(os.path.join(forest_dir, "points.shp"))

        # map
        fig, ax = plt.subplots()
        gdf_poly.plot(ax=ax, alpha=0.4)
        gdf_point.plot(ax=ax, color="red")
        plt.title(forest)
        plt.axis("off")
        plt.savefig(os.path.join(forest_dir, "map.png"), dpi=300)
        plt.close()


# ======================================================
# PROCESS CONTROLLER (FIXED)
# ======================================================
def process_file(file_path, output_dir, status_label, mode):

    try:
        df = pd.read_excel(file_path)

        zone = CRS_VAR.get()
        crs = get_crs(zone)

        output_dir = os.path.join(output_dir, "OUTPUT")
        os.makedirs(output_dir, exist_ok=True)

        status_label.config(text="Processing...")

        if mode == "whole":
            process_whole_forest(df, crs, output_dir)

        elif mode == "part":
            required = ["Forest", "Compartment", "X", "Y", "Order"]
            for c in required:
                if c not in df.columns:
                    raise ValueError(f"Missing column: {c}")

            process_part_forest(df, crs, output_dir)

        status_label.config(text="Completed ✔")
        messagebox.showinfo("Success", "Processing Done")

    except Exception as e:
        messagebox.showerror("Error", str(e))
        status_label.config(text="Error ❌")


# ======================================================
# START THREAD
# ======================================================
def start():

    file_path = select_file()
    if not file_path:
        return

    output_dir = select_output_folder()
    if not output_dir:
        return

    mode = mode_var.get()

    threading.Thread(
        target=process_file,
        args=(file_path, output_dir, status_label, mode),
        daemon=True
    ).start()


# ======================================================
# GUI (UNCHANGED)
# ======================================================
root = tk.Tk()
root.title("CF Boundary Generator")
root.geometry("500x320")


tk.Label(root, text="Community Forest GIS Tool",
         font=("Arial", 14, "bold")).pack(pady=10)


mode_var = tk.StringVar(value="whole")

tk.Label(root, text="Select Mode:").pack()

tk.Radiobutton(root, text="Whole Forest Boundary", variable=mode_var, value="whole").pack()
tk.Radiobutton(root, text="Part Forest Boundary", variable=mode_var, value="part").pack()


CRS_VAR = tk.StringVar(value="45")

frame = tk.Frame(root)
frame.pack(pady=5)

tk.Label(frame, text="UTM Zone:").grid(row=0, column=0)

tk.Radiobutton(frame, text="44N", variable=CRS_VAR, value="44").grid(row=0, column=1)
tk.Radiobutton(frame, text="45N", variable=CRS_VAR, value="45").grid(row=0, column=2)


tk.Button(root, text="Select File & Run",
          command=start, width=30, height=2).pack(pady=15)

status_label = tk.Label(root, text="Waiting...")
status_label.pack()

root.mainloop()
