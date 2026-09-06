# ElfakGISProStudio — Request & Data Flows

## 1. Application Load & Login

```
Browser                        Flask (app.py)           users.json
  │                                │                        │
  ├─── GET /  ───────────────────► │                        │
  │◄── index.html ─────────────── │                        │
  │                                │                        │
  │  [checkSession() runs]         │                        │
  ├─── GET /me ──────────────────► │                        │
  │◄── {username} or 401 ────────  │                        │
  │                                │                        │
  │  [if 401: show login overlay]  │                        │
  │  [user types username]         │                        │
  ├─── POST /login ──────────────► │                        │
  │    {username: "ravi"}          ├── read ──────────────► │
  │                                │◄─ {users dict} ─────── │
  │                                │  hash token            │
  │                                ├── write ─────────────► │
  │◄── {username, is_new, runs} ── │                        │
  │  [onLoginSuccess()]            │                        │
  │  [hide overlay, show UI]       │                        │
```

---

## 2. Module Selection & Form Fill

```
User clicks nav item (e.g. "C")
  │
  └─► switchTab('C')
        ├── sets activeModule = 'C'
        ├── hides all card-* divs
        ├── shows card-C
        └── resets progress bar
```

---

## 3. Pipeline Run (Groups A–E, H)

```
User fills form, clicks Run
  │
  ├─► sendRequest()
  │     ├── builds FormData (files + fields)
  │     ├── POST /upload (multipart)
  │     └─► startSSE(run_id)  ← opens EventSource /progress/<run_id>
  │
Flask /upload
  │
  ├── _rate_limit check
  ├── _safe_path validation
  ├── acquire _PIPELINE_SEM
  ├── detect module from form field
  │
  ├── [Group A]
  │     ├── parse boundary SHP → poly_gdf (GeoPandas)
  │     ├── parse survey CSV/Excel → pts_gdf
  │     ├── reproject to UTM 44N/45N
  │     ├── compute area_ha
  │     ├── render_map() → output.png (A4, 300 DPI, 70% fill)
  │     ├── _save_run_meta() → meta.json
  │     ├── generate_kmz() → output.kmz
  │     ├── zip all outputs → output.zip
  │     └── _prog(run_id, "Complete.", 100)
  │
  └─► return JSON:
        {run_id, download, kmz_url, map_editor_url}
  │
  SSE stream (/progress/<run_id>)
  │  data: {"pct": 20, "msg": "Reading boundary..."}
  │  data: {"pct": 55, "msg": "Reprojecting..."}
  │  data: {"pct": 88, "msg": "Rendering export map..."}
  │  data: {"pct": 100, "msg": "Complete."}
  │
Browser receives JSON response:
  ├── set out-img.src → /outputs/<run_id>/output.png
  ├── show download button
  ├── loadOSM(run_id) → GET /geojson/<run_id> → add Leaflet layers
  └── showOverlayForRun()
        ├── _rebuildOverlayLegend(activeModule, currentGeoJSON)
        ├── _wireOvEvents()
        ├── _ovRestoreState()  ← from localStorage
        └── update scale bar text from GeoJSON bounds
```

---

## 4. Pipeline Run (Group G)

```
User fills Group G form, clicks Run
  │
  ├─► runG()
  │     ├── POST /run_g (JSON body)
  │     └─► startSSE(run_id)
  │
Flask /run_g
  │
  ├── parse boundary SHP
  ├── generate vertex points (all polygon vertices)
  ├── generate boundary points (evenly spaced along perimeter)
  ├── generate divider points (compartment intersections)
  ├── assign SN numbers, build output GeoDataFrame
  ├── render_map() → output.png
  ├── _save_run_meta()
  └─► return {run_id, download, kmz_url, summary, map_editor_url}
```

---

## 5. Group F — Slope Analysis

```
User uploads boundary SHP, clicks Run F
  │
Flask /run_f
  │
  ├── parse boundary SHP → poly_gdf
  ├── determine UTM zone (44N or 45N) from centroid
  ├── _fetch_dem_tiles(poly_gdf) 
  │     ├── check dem_catalog/44N/ for local tiles
  │     └── download missing tiles from GitHub raw
  ├── clip DEM to boundary
  ├── compute slope (scipy.ndimage or rasterio.warp)
  ├── classify: 0-19° gentle, 19-31° moderate, >31° steep
  ├── vectorize slope classes → vgdf (GeoDataFrame)
  ├── render_map(slope_mode=True)  ← bakes north arrow + scale bar
  └─► return {run_id, download, kmz_url}
```

---

## 6. OSM / Leaflet Preview

```
After run completes:
  loadOSM(run_id, kmz_url)
  │
  ├── GET /geojson/<run_id>
  │     └── Flask reads output shapefiles → returns GeoJSON
  │
  ├── L.tileLayer(OpenStreetMap)
  ├── L.geoJSON(polyGJ)  → polyLayer (blue boundary)
  ├── L.geoJSON(ptGJ)    → ptLayer (red CircleMarkers)
  └── leafMap.fitBounds(polyLayer.getBounds().pad(0.06))
```

---

## 7. Vertex Editing (OSM Tab)

```
User clicks "✏️ Vertices"
  │
  setTool('vertex')
  │
  ├── [Module C or G]: pointOnly = true
  │     ├── for each layer in leafLayers:
  │     │     ├── isPoint? → l.pm.enable()   ← only sample/survey points
  │     │     └── isPolygon? → l.pm.disable() ← boundary stays locked
  │     └── user drags point markers
  │
  └── [Other modules]: enable pm on all layers
  
User clicks "💾 Save"
  │
  saveEdits()
  ├── collect edited GeoJSON from leafLayers
  └── POST /save_edit
        ├── {run_id, poly_geojson, point_geojson}
        ├── Flask writes updated shapefiles
        └── re-renders output.png via render_map()
```

---

## 8. Layout Edit & Export

```
User clicks "✥ Edit Layout"
  │
  toggleLayoutEdit()
  ├── overlay-layer gets class "editing"
  ├── all .ov-editable get contentEditable = "true"
  ├── all .ov-legend-label get contentEditable = "true"
  ├── resize handles become visible
  └── north arrow colour picker enabled on hover

User drags overlay item:
  _ovPointerDown → setPointerCapture → _ovDragMove → _ovDragEnd
  └── _ovSaveState() → localStorage.setItem('ov-<run_id>', JSON)

User resizes overlay item:
  _ovHandleDown → _ovResizeMove → _ovResizeEnd
  ├── north arrow: aspect ratio locked
  └── _ovSaveState()

User edits legend label:
  click on .ov-legend-label → contentEditable → type → blur
  └── _ovSaveState()

User changes arrow colour:
  hover on ov-north → colour picker appears → pick colour
  └── updates fill/stroke on SVG polygons + N text

User clicks "🔍 Full" (viewFullMap):
  ├── html2canvas(canvas-wrap-static, scale:2.0)
  ├── composites map PNG + all overlay items
  └── shows in modal, download as edited PNG

User clicks "⬇ Export Official Map" (exportLayout):
  ├── collect _getLayoutStateForServer()
  ├── POST /export_layout  [NOT YET IMPLEMENTED in app.py]
  └── download PDF/PNG/SVG
```

---

## 9. Composer Tab — Re-render

```
User changes title/legend in Composer, clicks "Re-Render":
  │
  doCompose()
  ├── _ovSaveState()
  ├── collect legend positions, title, label_col
  └── POST /compose/<run_id>
        ├── Flask re-runs render_map() with new params
        ├── overwrites output.png
        └── returns {png: "/outputs/<run_id>/output.png?t=..."}
        
Browser:
  └── out-img.src = new PNG URL → reload → showOverlayForRun()
```

---

## 10. History & Run Reload

```
User clicks history icon → toggleHist()
  │
  ├── GET /history → [{run_id, module, timestamp, description}...]
  └── renderHistory(runs) → renders run cards

User clicks "🗺 Preview" on a run card:
  │
  loadRunPrev(run_id)
  ├── set currentRunId
  ├── out-img.src = /outputs/<run_id>/output.png
  ├── dl-btn.href = /download/<run_id>
  ├── switchPView('static')
  ├── loadOSM(run_id, null)
  ├── GET /geojson/<run_id> → currentGeoJSON
  └── showOverlayForRun()
```

---

## Data Formats

### GeoDataFrame Convention
All GeoDataFrames in processing use **UTM Zone 44N (EPSG:32644)** or **45N (EPSG:32645)** depending on centroid longitude.

### meta.json
```json
{
  "forest_name": "Salghari CF",
  "area_ha": 464.405
}
```

### Overlay State (localStorage)
```json
{
  "ov-legend": {
    "left": "12px", "top": "auto", "right": "8px", "bottom": "40px",
    "width": "160px", "height": "auto",
    "zIndex": "10", "display": ""
  },
  "ov-north": { "left": "auto", "top": "8px", "right": "8px", ... },
  "ov-scale": { "left": "50%", "bottom": "8px", ... }
}
```

### SSE Progress Event
```
data: {"pct": 45, "msg": "Reprojecting to UTM..."}
```

### /upload Response
```json
{
  "run_id": "a3f8c2d1...",
  "download": "/download/a3f8c2d1...",
  "kmz_url": "/outputs/a3f8c2d1.../output.kmz",
  "map_editor_url": "/map_editor/a3f8c2d1..."
}
```
