# ElfakGISProStudio — Architecture

## Stack

| Layer | Technology |
|---|---|
| Backend | Python 3, Flask |
| GIS Processing | GeoPandas, Shapely, Fiona |
| Map Rendering | Matplotlib (A4, 300 DPI) |
| DEM / Slope | rasterio, scipy.ndimage (optional) |
| Frontend | Vanilla JS, HTML5, CSS3 |
| Map Preview | Leaflet.js + leaflet-geoman |
| Auth | Session cookies, SHA-256 token, users.json |
| Storage | Local filesystem (outputs/, uploads/, dem_catalog/) |
| Deployment | Gunicorn (Procfile), Heroku/Render compatible |

---

## Directory Structure

```
ElfakGISProStudio/
│
├── app.py                  # Entire backend — 4958 lines, single file
├── index.html              # Frontend SPA — 6336 lines
├── requirements.txt        # Python deps (unpinned)
├── Procfile                # gunicorn app:app
│
├── templates/
│   └── map_result.html     # Standalone map editor page (deep-link)
│
├── static/ / Static/       # CSS, JS, images (served by Flask)
│
├── uploads/                # User-uploaded files (temp, per-request)
│   └── dem_cache/          # Downloaded DEM tiles cached here
│
├── outputs/                # Generated run output directories
│   └── <run_id>/
│       ├── output.png      # Rendered map (A4, 300 DPI)
│       ├── meta.json       # {forest_name, area_ha}
│       ├── output.zip      # Shapefiles + map
│       ├── output.kmz      # Google Earth file
│       └── *.shp / *.geojson
│
├── dem_catalog/
│   └── 44N/                # Nepal UTM Zone 44N DEM tiles
│
└── users.json              # Auth store {username: {token_hash, runs[]}}
```

---

## Backend (app.py) Sections

### 1. Constants & Imports (lines 1–120)
- `FIG_W=8.27, FIG_H=11.69, DPI=300` — A4 portrait
- `POLY_COLOR, POINT_COL, GRID_COL` — map style constants
- Optional: `_HAS_RASTERIO` guard for DEM operations

### 2. Auth & Security (lines 121–320)
- `_rate_limit(limit, window)` — per-IP sliding window decorator
- `_safe_path(base, *parts)` — blocks directory traversal
- `_safe_runid(run_id)` — alphanumeric-only run ID validation
- `_lu() / _su()` — read/write users.json with file lock
- `_USERS_LOCK` — threading.Lock for concurrent writes
- `_PIPELINE_SEM` — semaphore limits concurrent heavy GIS ops (default: 4)

### 3. Geometry Helpers (lines 321–600)
- `_repair(geom)` — Shapely topology repair chain
- `_as_poly(geom)` — GeometryCollection → largest Polygon extraction
- `_enforce_poly_gdf(gdf)` — ensure GeoDataFrame has valid polygons only
- `_close_poly(coords)` — close open rings
- `_rotated_bbox(rect, angle, center)` — rotation-aware bounding box

### 4. Map Layout (lines 601–900)
- `FreeSpaceManager` — tracks free canvas area, scores regions for overlay placement
- `_place_labels(ax, gdf, col, ...)` — 8-direction label placement with collision avoidance
- `get_default_layout_state()` — default overlay positions (%)
- `compute_safe_rect(layout_state, aspect)` — compute safe matplotlib axes rect

### 5. Map Renderer (lines 901–1150)
- `_add_north_arrow(fig, pos, size)` — baked N arrow (slope mode only)
- `_add_scale_bar(fig, ax, ...)` — baked scale bar (slope mode only)
- `_setup_utm_grid(ax, ...)` — UTM tick labels all 4 sides, dashed grid
- `_label_points_export(ax, pts_gdf, sn_col)` — SN labels with white stroke
- `render_map(path, ...)` — main entry point:
  - **Standard mode**: A4, 70% fill, UTM grid, blue boundary, red dots, no baked overlays
  - **Slope mode** (`slope_mode=True`): adds baked north arrow + scale bar + slope table

### 6. Processing Groups (lines 1151–4300)

| Group | Function | Input | Output |
|---|---|---|---|
| A | `_run_group_a()` | Boundary SHP + Survey CSV/Excel | Boundary polygon + survey points map |
| B | `_run_group_b()` | Multiple boundary SHPs | Segmented/coloured boundary map |
| C | `_run_group_c()` | Boundary + Survey + Sample intervals | Sample plot point placement map |
| D | `_run_group_d()` | Multiple forest packages | Multi-forest coloured map |
| E | `_run_group_e()` | Boundary + Compartment CSV | Voronoi-subdivided compartment map |
| F | `_run_group_f()` | Boundary + DEM tiles | Slope analysis map (3 classes) |
| G | `_run_group_g()` | Boundary SHP | Survey point generator (vertex/boundary/divider) |
| H | `_run_group_h()` | Boundary + DEM + Sample points | Sample point based slope map |

### 7. KMZ Generator (lines 4301–4420)
- `generate_kmz(poly, line, pts, out_dir, run_id)` — builds KML from GeoDataFrames, zips to .kmz
- Handles Polygon, MultiPolygon, LineString, Point geometries

### 8. Routes (lines 4421–4958)

| Route | Method | Description |
|---|---|---|
| `/` | GET | Serve index.html |
| `/login` | POST | Username auth, create/validate session |
| `/logout` | POST | Clear session |
| `/me` | GET | Return current session user |
| `/upload` | POST | Main pipeline entry — detects module A–H, runs pipeline |
| `/run_g` | POST | Group G dedicated route |
| `/run_f` | POST | Group F dedicated route |
| `/progress/<run_id>` | GET | SSE stream of pipeline progress (0–100%) |
| `/compose/<run_id>` | POST | Re-render map with updated layout/title/labels |
| `/geojson/<run_id>` | GET | Return GeoJSON of run output for Leaflet |
| `/outputs/<path>` | GET | Serve output files (PNG, SHP, etc.) |
| `/download/<run_id>` | GET | Stream ZIP of all run outputs |
| `/history` | GET | Return user's run history |
| `/map_editor/<run_id>` | GET | Render standalone map editor page |
| `/save_edit` | POST | Save vertex edits back to shapefiles |

### 9. Meta Helper
- `_save_run_meta(out_dir, forest_name, area_ha)` — writes `meta.json` alongside every run

---

## Frontend (index.html) Sections

### Layout
```
┌─────────────┬─────────────────────────────────────────────┐
│             │  Tab bar: 🖼 Map | 🌍 OSM | 🎨 Composer      │
│  Left Nav   ├─────────────────────────────────────────────┤
│  A B C D   │                                              │
│  E F G H   │     Map Canvas (canvas-wrap-static)          │
│             │                                              │
│  Module     │     ┌── overlay-layer ──────────────────┐   │
│  Forms      │     │  ov-north  ov-scale  ov-legend    │   │
│             │     │  ov-title  ov-area                 │   │
│  History    │     └───────────────────────────────────┘   │
│  Drawer     ├─────────────────────────────────────────────┤
│             │  [✥ Edit Layout]  [metrics]  [🔍 Full] [ZIP]│
└─────────────┴─────────────────────────────────────────────┘
```

### Key Global State Variables
```js
let currentRunId     = null;   // active run UUID
let currentGeoJSON   = null;   // GeoJSON of active run
let activeModule     = 'A';    // current group tab
let _layoutEditActive= false;  // overlay edit mode on/off
let leafMap          = null;   // Leaflet map instance
let leafLayers       = [];     // all Leaflet layers for current run
let activeTool       = null;   // 'vertex' | null
```

### Overlay System
Five overlay items float above the map image:

| ID | Type | Draggable | Resizable | Text Editable |
|---|---|---|---|---|
| `ov-north` | North arrow (SVG) | ✅ | ✅ 8 handles | ✅ N label, colour picker on hover |
| `ov-scale` | Scale bar + text | ✅ | ✅ | ✅ span contenteditable |
| `ov-legend` | Legend box | ✅ | ✅ | ✅ title + all labels |
| `ov-title` | Map title | ✅ | ✅ | ✅ |
| `ov-area` | Area display | ✅ | ✅ | ✅ |

State saved to `localStorage` key `ov-<run_id>` on every change.

### SSE Progress
`startSSE(runId)` opens an `EventSource` to `/progress/<run_id>`.
Server sends `data: {"pct": 45, "msg": "Processing boundary…"}` events.
Frontend updates progress bar and status text in real time.

---

## Security Model

| Concern | Mechanism |
|---|---|
| Path traversal | `_safe_path()` with `os.realpath()` comparison |
| Rate limiting | `_rate_limit` decorator, per-IP sliding window |
| Concurrent pipelines | `_PIPELINE_SEM` semaphore |
| Session fixation | `secrets.token_hex(32)`, stored as SHA-256 hash |
| Username abuse | Allowlist regex + blocklist check |
| File upload size | `MAX_CONTENT_LENGTH = 2 GB` |
| CORS | Response headers on all routes |

---

## Known Gaps

| Issue | Notes |
|---|---|
| `export_layout` route missing | Frontend calls `/export_layout` but route not implemented in app.py |
| `users.json` race condition | Read-modify-write not atomic under multi-worker Gunicorn |
| Unpinned requirements.txt | Shapely 2.x has breaking API changes vs 1.x |
| `_PROG` dict unbounded | Progress dict grows until cleanup (every 3600s, threshold 10000) |
| KMZ drops GeometryCollection | Silently skips features that aren't Polygon/Line/Point |
