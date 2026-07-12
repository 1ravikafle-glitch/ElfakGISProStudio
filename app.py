import os, uuid, zipfile, shutil, json, re, io, traceback, math, time, gc
import threading, hashlib, html, secrets, logging
from collections import defaultdict, OrderedDict
from functools import wraps
from datetime import datetime
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
from matplotlib.transforms import Bbox
from shapely.geometry import Polygon, Point, LineString, MultiPolygon, box
from shapely.geometry import box as _sbox
from shapely.ops import unary_union
from shapely import affinity

try:
    import rasterio
    from rasterio.mask import mask as rio_mask
    from rasterio.features import shapes as rio_shapes
    from rasterio.plot import show
    import scipy.ndimage as ndi
    _HAS_RASTERIO = True
except ImportError:
    _HAS_RASTERIO = False

from flask import (Flask, request, jsonify, send_file, send_from_directory,
                   render_template, session, Response, stream_with_context, abort, g)

# ----------------------------------------------------------------------
# Configuration & Constants
# ----------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
log = logging.getLogger("elfak-gis")

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY") or secrets.token_hex(32)
if not os.environ.get("SECRET_KEY"):
    log.warning("SECRET_KEY not set in environment — using random key (sessions won't persist across restarts)")

app.config.update(
    SESSION_COOKIE_SECURE    = os.environ.get("HTTPS","0") == "1",
    SESSION_COOKIE_HTTPONLY  = True,
    SESSION_COOKIE_SAMESITE  = "Lax",
    PERMANENT_SESSION_LIFETIME = 86400 * 7,
    MAX_CONTENT_LENGTH       = 2 * 1024 * 1024 * 1024,
)

UPLOAD, OUTPUT, USERS_FILE = "uploads", "outputs", "users.json"
DEM_CATALOG_DIR = os.environ.get("DEM_CATALOG_DIR", "dem_catalog")
GITHUB_DEM_BASE = os.environ.get(
    "GITHUB_DEM_BASE",
    "https://raw.githubusercontent.com/1ravikafle-glitch/ElfakGISProStudio/main/dem_catalog"
)
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN", "").strip()
DEM_CACHE_DIR = os.path.join(UPLOAD, "dem_cache")
for _d in (UPLOAD, OUTPUT, DEM_CATALOG_DIR, DEM_CACHE_DIR):
    os.makedirs(_d, exist_ok=True)

FIG_W, FIG_H, DPI = 10.0, 7.5, 200   # landscape A4-like
EPS = 1e-6
DEFAULT_PADDING = 0.02  # 2% of figure size

# ----------------------------------------------------------------------
# Helper: Human-readable Run ID
# ----------------------------------------------------------------------

def _generate_run_id(base_name, max_len=40):
    """
    Create a human-readable run ID from a base name, with timestamp and random suffix.
    Returns: e.g. "Salghari_CF_20250101_143022_8x9k"
    """
    clean = re.sub(r'[^A-Za-z0-9_-]', '_', base_name)
    clean = re.sub(r'_+', '_', clean)
    max_base = max_len - 20
    if len(clean) > max_base:
        clean = clean[:max_base]
    clean = clean.rstrip('_')
    if not clean:
        clean = "forest"
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = secrets.token_hex(2)
    return f"{clean}_{ts}_{suffix}"

def _safe_runid(rid):
    rid = str(rid).strip()
    if not re.fullmatch(r'[A-Za-z0-9_-]+', rid):
        log.warning(f"Invalid run_id rejected: {rid!r}")
        abort(400, "Invalid run ID.")
    return rid

# ----------------------------------------------------------------------
# KMZ Generation (was missing)
# ----------------------------------------------------------------------

def generate_kmz(poly_gdf, line_gdf, pts_gdf, out_dir, run_id):
    import zipfile as zf

    def w84(gdf):
        if gdf is None or gdf.empty:
            return None
        try:
            return gdf.to_crs("EPSG:4326") if gdf.crs else None
        except:
            return None

    pw = w84(poly_gdf)
    lw = w84(line_gdf)
    ptw = w84(pts_gdf)
    ref = pw if pw is not None and not pw.empty else lw

    if ref is not None and not ref.empty:
        u = ref.unary_union
        cx, cy = u.centroid.x, u.centroid.y
        minx, miny, maxx, maxy = u.bounds
        span = max(maxx - minx, maxy - miny)
        alt = max(500, int(span * 111000 * 2))
    else:
        cx, cy, alt = 0, 0, 10000

    st = """<Style id="poly_style"><LineStyle><color>ff00ff00</color><width>2</width></LineStyle><PolyStyle><color>4400cc00</color></PolyStyle></Style>
<Style id="line_style"><LineStyle><color>ff0000ff</color><width>2</width></LineStyle></Style>
<Style id="point_style"><IconStyle><color>ff0000ff</color><scale>0.8</scale><Icon><href>http://maps.google.com/mapfiles/kml/shapes/placemark_circle.png</href></Icon></IconStyle></Style>"""

    def kml_pm(gdf, sid, nc=None):
        lines = []
        for i, row in gdf.iterrows():
            g = row.geometry
            if g is None or g.is_empty:
                continue
            if nc and nc in row.index and row[nc]:
                label = str(row[nc])
            elif "Comp_ID" in row.index and row["Comp_ID"]:
                label = str(row["Comp_ID"])
            elif "Forest" in row.index and row["Forest"]:
                label = str(row["Forest"])
            else:
                label = f"Feature {i+1}"

            def cs(c):
                return " ".join(f"{x},{y},0" for x, y in c)

            def pk(geom):
                o = cs(list(geom.exterior.coords))
                r = [f"<outerBoundaryIs><LinearRing><coordinates>{o}</coordinates></LinearRing></outerBoundaryIs>"]
                for interior in geom.interiors:
                    inn = cs(list(interior.coords))
                    r.append(f"<innerBoundaryIs><LinearRing><coordinates>{inn}</coordinates></LinearRing></innerBoundaryIs>")
                return f"<Polygon>{''.join(r)}</Polygon>"

            if g.geom_type == "Polygon":
                gk = pk(g)
            elif g.geom_type == "MultiPolygon":
                gk = f"<MultiGeometry>{''.join(pk(x) for x in g.geoms)}</MultiGeometry>"
            elif g.geom_type == "LineString":
                gk = f"<LineString><coordinates>{cs(list(g.coords))}</coordinates></LineString>"
            elif g.geom_type == "Point":
                gk = f"<Point><coordinates>{g.x},{g.y},0</coordinates></Point>"
            else:
                continue
            lines.append(f"<Placemark><name>{label}</name><styleUrl>#{sid}</styleUrl>{gk}</Placemark>")
        return "\n".join(lines)

    folders = []
    if pw is not None and not pw.empty:
        folders.append(f"<Folder><name>Polygons</name>{kml_pm(pw,'poly_style','Forest')}</Folder>")
    if lw is not None and not lw.empty:
        folders.append(f"<Folder><name>Lines</name>{kml_pm(lw,'line_style','Forest')}</Folder>")
    if ptw is not None and not ptw.empty:
        folders.append(f"<Folder><name>Points</name>{kml_pm(ptw,'point_style','SN')}</Folder>")

    kml = f"""<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2">
<Document>
<name>Elfak GIS</name>
<LookAt><longitude>{cx}</longitude><latitude>{cy}</latitude><altitude>0</altitude><range>{alt}</range><tilt>0</tilt><heading>0</heading></LookAt>
{st}
{"".join(folders)}
</Document>
</kml>"""

    kmz = os.path.join(out_dir, "output.kmz")
    with zf.ZipFile(kmz, "w", zf.ZIP_DEFLATED) as z:
        z.writestr("doc.kml", kml.encode("utf-8"))
    return {
        "url": f"/outputs/{run_id}/output.kmz",
        "lat": round(cy, 6),
        "lon": round(cx, 6),
        "alt": alt
    }

# ----------------------------------------------------------------------
# User Management, Progress, Rate Limiting
# ----------------------------------------------------------------------

_PROG: dict = {}
_PROG_LOCK = threading.Lock()

def _prog(rid, msg, pct=None):
    o = {
        "msg": str(msg)[:500],
        "pct": max(0, min(100, int(pct))) if pct is not None else None,
        "ts": time.time()
    }
    with _PROG_LOCK:
        if rid not in _PROG: _PROG[rid] = []
        _PROG[rid].append(json.dumps(o))
        _PROG[rid] = _PROG[rid][-500:]
    time.sleep(0.01)

def _cleanup_old_prog():
    while True:
        time.sleep(3600)
        with _PROG_LOCK:
            keys = list(_PROG.keys())
            if len(keys) > 10000:
                for k in keys[:len(keys)//2]:
                    _PROG.pop(k, None)

threading.Thread(target=_cleanup_old_prog, daemon=True).start()

_USERS_LOCK = threading.RLock()

def _lu():
    with _USERS_LOCK:
        try:
            if not os.path.exists(USERS_FILE): return {}
            with open(USERS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            log.error(f"_lu error: {e}"); return {}

def _su(users_dict):
    with _USERS_LOCK:
        try:
            tmp = USERS_FILE + ".tmp." + uuid.uuid4().hex[:8]
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(users_dict, f, indent=2, ensure_ascii=False)
            os.replace(tmp, USERS_FILE)
        except Exception as e:
            log.error(f"_su error: {e}")

def _register_user(name):
    name = name.strip()
    with _USERS_LOCK:
        u = _lu()
        if name in u:
            raise ValueError(f'Username "{name}" is already taken. Choose a different name.')
        token = secrets.token_hex(32)
        u[name] = {
            "username":   name,
            "created_at": pd.Timestamp.now().isoformat(),
            "token_hash": hashlib.sha256(token.encode()).hexdigest(),
            "runs":       [],
            "active_sessions": 0,
        }
        _su(u)
        log.info(f"New user registered: {name!r}")
        return u[name], token

def _login_existing(name):
    name = name.strip()
    with _USERS_LOCK:
        u = _lu()
        if name not in u:
            raise KeyError(f'User "{name}" not found.')
        u[name]["active_sessions"] = u[name].get("active_sessions", 0) + 1
        u[name]["last_login"] = pd.Timestamp.now().isoformat()
        _su(u)
        return u[name]

def _logout_user(name):
    if not name: return
    with _USERS_LOCK:
        u = _lu()
        if name in u:
            u[name]["active_sessions"] = max(0, u[name].get("active_sessions", 1) - 1)
            _su(u)

def _append_run(uname, rid, mod, desc=""):
    if not uname: return
    with _USERS_LOCK:
        u = _lu()
        if uname in u:
            u[uname].setdefault("runs", []).append({
                "run_id":    rid,
                "module":    mod,
                "description": desc[:200],
                "timestamp": pd.Timestamp.now().isoformat(),
            })
            u[uname]["runs"] = u[uname]["runs"][-100:]
            _su(u)

def _require_login():
    return session.get("username")

def _login_required(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        if not _require_login():
            return jsonify({"error": "Authentication required. Please log in."}), 401
        return fn(*args, **kwargs)
    return wrapper

def _cleanup_old_outputs():
    while True:
        time.sleep(1800)
        try:
            dirs = [(os.path.join(OUTPUT, d),
                     os.path.getmtime(os.path.join(OUTPUT, d)))
                    for d in os.listdir(OUTPUT)
                    if os.path.isdir(os.path.join(OUTPUT, d))]
            dirs.sort(key=lambda x: x[1])
            for path, _ in dirs[:-500]:
                shutil.rmtree(path, ignore_errors=True)
        except Exception as e:
            log.error(f"Cleanup error: {e}")

threading.Thread(target=_cleanup_old_outputs, daemon=True).start()

_RL: dict = defaultdict(list)
_RL_LOCK = threading.Lock()

def _rate_limit(limit=30, window=60, key_fn=None):
    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            ip = _get_client_ip()
            key = (key_fn(request) if key_fn else ip)
            now = time.time()
            with _RL_LOCK:
                _RL[key] = [t for t in _RL[key] if now - t < window]
                if len(_RL[key]) >= limit:
                    retry_after = int(window - (now - _RL[key][0])) + 1
                    log.warning(f"Rate limit hit: {key} on {request.path}")
                    return jsonify({"error": f"Too many requests. Try again in {retry_after}s.",
                                    "retry_after": retry_after}), 429
                _RL[key].append(now)
            return fn(*args, **kwargs)
        return wrapper
    return decorator

def _clean_rl():
    while True:
        time.sleep(300)
        with _RL_LOCK:
            now = time.time()
            dead = [k for k, ts in _RL.items() if not ts or now - ts[-1] > 3600]
            for k in dead: del _RL[k]

threading.Thread(target=_clean_rl, daemon=True).start()

_ALLOWED_EXTS = {
    ".csv", ".xls", ".xlsx", ".zip",
    ".shp", ".dbf", ".prj", ".shx", ".cpg",
    ".tif", ".tiff"
}
_BLOCKED_NAMES = {
    "admin","root","system","null","undefined","guest","test","demo",
    "anonymous","api","static","login","logout","upload","download",
    "server","config","env","app","index","user","users","data",
    "script","style","public","private","backend","frontend"
}

def _safe_filename(fname):
    fname = os.path.basename((fname or "upload").replace("\\", "/"))
    fname = re.sub(r"[^A-Za-z0-9._-]", "_", fname)[:200]
    if not fname: fname = "upload"
    ext = os.path.splitext(fname)[1].lower()
    if ext not in _ALLOWED_EXTS:
        raise ValueError(f"File type '{ext}' not allowed. Allowed: {sorted(_ALLOWED_EXTS)}")
    return fname

def _safe_path(base, rel):
    base = os.path.realpath(os.path.abspath(base))
    full = os.path.realpath(os.path.abspath(os.path.join(base, str(rel))))
    if not (full == base or full.startswith(base + os.sep)):
        log.warning(f"Path traversal blocked: base={base!r} rel={rel!r} full={full!r}")
        abort(400, "Path traversal detected.")
    return full

def _validate_username(name):
    name = (name or "").strip()
    if not name:
        raise ValueError("Username is required.")
    if len(name) < 2:
        raise ValueError("Username too short (minimum 2 characters).")
    if len(name) > 40:
        raise ValueError("Username too long (maximum 40 characters).")
    if not re.match(r"^[A-Za-z0-9][A-Za-z0-9 _-]*$", name):
        raise ValueError("Username must start with a letter/digit and contain only letters, numbers, spaces, hyphens, or underscores.")
    if name.lower() in _BLOCKED_NAMES:
        raise ValueError("That username is reserved. Please choose a different name.")
    _bad = set("<>'\";&|`")
    if any(c in _bad for c in name):
        raise ValueError("Username contains invalid characters.")
    return name

def _get_client_ip():
    xff = request.headers.get("X-Forwarded-For", "")
    if xff:
        ip = xff.split(",")[0].strip()
        if re.match(r"^\d{1,3}(\.\d{1,3}){3}$", ip) or ":" in ip:
            return ip
    return request.remote_addr or "unknown"

@app.errorhandler(413)
def too_large(e):
    return jsonify({"error": "File too large. Maximum upload is 2GB."}), 413

@app.errorhandler(429)
def rate_limited(e):
    return jsonify({"error": "Too many requests. Please wait a moment.", "retry_after": 5}), 429

@app.errorhandler(500)
def server_error(e):
    log.error(f"Unhandled 500: {e}")
    return jsonify({"error": f"Internal server error: {str(e)[:200]}"}), 500

@app.errorhandler(Exception)
def unhandled(e):
    log.error(f"Unhandled exception: {type(e).__name__}: {e}")
    return jsonify({"error": f"Unexpected error: {type(e).__name__}: {str(e)[:200]}"}), 500

@app.after_request
def _security_headers(resp):
    resp.headers["X-Content-Type-Options"]   = "nosniff"
    resp.headers["X-Frame-Options"]          = "DENY"
    resp.headers["X-XSS-Protection"]         = "1; mode=block"
    resp.headers["Referrer-Policy"]          = "strict-origin-when-cross-origin"
    resp.headers["Permissions-Policy"]       = "geolocation=(), camera=(), microphone=()"
    if os.environ.get("HTTPS") == "1":
        resp.headers["Strict-Transport-Security"] = "max-age=63072000; includeSubDomains"
    allowed = os.environ.get("ALLOWED_ORIGINS", "*")
    resp.headers["Access-Control-Allow-Origin"]       = allowed
    resp.headers["Access-Control-Allow-Headers"]      = "Content-Type,Authorization,X-Requested-With"
    resp.headers["Access-Control-Allow-Methods"]      = "GET,POST,OPTIONS"
    resp.headers["Access-Control-Allow-Credentials"]  = "true"
    if request.path.startswith(("/login", "/me", "/history", "/progress")):
        resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, private"
        resp.headers["Pragma"]        = "no-cache"
    resp.headers["X-Accel-Buffering"] = "no"
    return resp

# ─── PIPELINE SEMAPHORE ──────────────────────────────────────────────────
_PIPELINE_SEM = threading.Semaphore(int(os.environ.get("MAX_PIPELINES", "4")))
_ACTIVE_PIPELINES = 0
_AP_LOCK = threading.Lock()

def _with_pipeline_sem(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        global _ACTIVE_PIPELINES
        acquired = _PIPELINE_SEM.acquire(timeout=5)
        if not acquired:
            log.warning(f"Pipeline semaphore exhausted for {_get_client_ip()}")
            return jsonify({
                "error": "Server is at capacity. Please wait a moment and retry.",
                "retry_after": 30
            }), 503
        with _AP_LOCK: _ACTIVE_PIPELINES += 1
        try:
            log.info(f"Pipeline started by {session.get('username','?')} "
                     f"[active={_ACTIVE_PIPELINES}] [{request.path}]")
            return fn(*args, **kwargs)
        finally:
            _PIPELINE_SEM.release()
            with _AP_LOCK: _ACTIVE_PIPELINES = max(0, _ACTIVE_PIPELINES - 1)
    return wrapper

# ----------------------------------------------------------------------
# Helper: Column detection, reading, CRS, geometry
# ----------------------------------------------------------------------

def _norm(s): return "".join(c for c in str(s).lower() if c.isalnum())
_XA={"x","xcoord","xcoordinate","xcord","east","easting","eastings","lon","long","longitude","lng","pointx","coordx","utme","utmx"}
_YA={"y","ycoord","ycoordinate","ycord","north","northing","northings","lat","latitude","pointy","coordy","utmn","utmy"}
_OA={"order","id","sn","sno","serial","serialno","seq","sequence","index","rowid","fid","no","num","number","plotid","plotno","pointid","pointno","pid"}
_FA={"forest","forestname","forestid","forestno","fname","forestblock","block"}
_CA={"compartment","comp","compartmentno","compartmentid","compno","compid","section","sectionno"}
def _find_col(df, aliases):
    for c in df.columns:
        if _norm(c) in aliases: return c
    return None
def safe_col(df, mapping, key, fallback):
    if mapping and mapping.get(key) and mapping[key] in df.columns: return mapping[key]
    if fallback in df.columns: return fallback
    for c in df.columns:
        if c.lower() == fallback.lower(): return c
    am = {"X":_XA,"Y":_YA,"Order":_OA,"Forest":_FA,"Compartment":_CA}
    if key in am:
        h = _find_col(df, am[key])
        if h: return h
    return None
def normalize_order(df):
    for c in df.columns:
        if _norm(c) in _OA and c != "Order": df = df.rename(columns={c:"Order"}); break
    return df
def read_input(file):
    n = file.filename.lower()
    if n.endswith(".csv"): return pd.read_csv(file, encoding="utf-8-sig")
    elif n.endswith((".xlsx",".xls")): return pd.read_excel(file)
    raise ValueError("Only CSV/Excel supported.")
def get_crs(zone): return f"EPSG:326{zone}"
def _safe_dn(s): return str(s).strip().replace("/","_").replace("\\","_").replace(":","_")

# Geometry helpers
def safe_polygon(coords):
    p = Polygon(coords); return p if p.is_valid else p.buffer(0)
def _repair(g):
    if g is None: return None
    try:
        if g.is_empty: return None
        if not g.is_valid:
            g = g.buffer(0)
            if g is None or g.is_empty: return None
        if g.geom_type == "MultiPolygon":
            parts = [p for p in g.geoms if p and not p.is_empty]
            if len(parts) == 1:
                g = parts[0]
        elif g.geom_type == "GeometryCollection":
            polys = [p for p in g.geoms
                     if p.geom_type in ("Polygon","MultiPolygon") and not p.is_empty]
            if polys:
                g = unary_union(polys)
                if g.is_empty: return None
            else:
                return None
        return g if (g and not g.is_empty) else None
    except Exception:
        try:
            fixed = g.buffer(0) if g else None
            return fixed if (fixed and not fixed.is_empty) else None
        except Exception:
            return None
def _as_poly(g):
    if g is None: return None
    try:
        if g.is_empty: return None
        if g.geom_type == "Polygon": return g
        if g.geom_type == "MultiPolygon":
            parts = [x for x in g.geoms if x.geom_type=="Polygon" and not x.is_empty]
            return max(parts, key=lambda x: x.area) if parts else None
        if g.geom_type == "GeometryCollection":
            polys = []
            for x in g.geoms:
                if x.geom_type == "Polygon" and not x.is_empty and x.area > 1e-10:
                    polys.append(x)
                elif x.geom_type == "MultiPolygon":
                    polys.extend([p for p in x.geoms if not p.is_empty and p.area>1e-10])
            return max(polys, key=lambda x: x.area) if polys else None
    except Exception:
        pass
    return None
def _close_poly(p):
    if p is None: return p
    try:
        if p.is_empty: return p
        if p.geom_type == "MultiPolygon":
            parts = [x for x in p.geoms if not x.is_empty]
            if not parts: return p
            p = max(parts, key=lambda x: x.area)
        if p.geom_type != "Polygon":
            return p
        ext = list(p.exterior.coords)
        if ext[0] != ext[-1]: ext.append(ext[0])
        holes = [list(i.coords) for i in p.interiors]
        for h in holes:
            if h[0] != h[-1]: h.append(h[0])
        c = Polygon(ext, holes)
        if c.is_empty: return p
        return c if c.is_valid else c.buffer(0)
    except Exception:
        return p
def _enforce_poly_gdf(gdf):
    if gdf is None or gdf.empty:
        return gdf
    keep = []
    for _, row in gdf.iterrows():
        g = row.geometry
        if g is None:
            continue
        try:
            g = _repair(g)
            if g is None or g.is_empty:
                continue
            if g.geom_type in ("Polygon", "MultiPolygon"):
                pg = g
            elif g.geom_type == "GeometryCollection":
                polys = [p for p in g.geoms
                         if p.geom_type in ("Polygon","MultiPolygon")
                         and not p.is_empty and p.area > 1e-10]
                if not polys:
                    continue
                pg = _repair(unary_union(polys))
            else:
                pg = _repair(g.buffer(0))
                if pg is None or pg.geom_type not in ("Polygon","MultiPolygon"):
                    continue
            if pg is None or pg.is_empty or pg.area < 1e-12:
                continue
            if pg.geom_type == "Polygon":
                pg = _close_poly(pg)
            elif pg.geom_type == "MultiPolygon":
                fixed = []
                for part in pg.geoms:
                    if part is None or part.is_empty: continue
                    cp = _close_poly(part)
                    if cp is not None and cp.geom_type == "Polygon" and not cp.is_empty and cp.area > 1e-12:
                        fixed.append(cp)
                if not fixed:
                    continue
                pg = MultiPolygon(fixed) if len(fixed) > 1 else fixed[0]
            r2 = row.copy()
            r2["geometry"] = pg
            keep.append(r2)
        except Exception:
            continue
    if not keep:
        return gpd.GeoDataFrame(columns=gdf.columns, crs=gdf.crs)
    result = gpd.GeoDataFrame(keep, crs=gdf.crs)
    result = result.reset_index(drop=True)
    return result

# ----------------------------------------------------------------------
# LAYOUT ENGINE (FreeSpaceManager, compute_safe_rect, label engine)
# ----------------------------------------------------------------------

class FreeSpaceManager:
    """Manages free rectangles in normalized figure coordinates (0..1)."""
    def __init__(self, page=(0.0, 0.0, 1.0, 1.0)):
        self.rects = [page]

    def _merge(self):
        while True:
            merged = False
            new_rects = []
            skip = set()
            n = len(self.rects)
            for i in range(n):
                if i in skip:
                    continue
                r1 = self.rects[i]
                for j in range(i+1, n):
                    if j in skip:
                        continue
                    r2 = self.rects[j]
                    x0a, y0a, x1a, y1a = r1
                    x0b, y0b, x1b, y1b = r2

                    if abs(y0a - y0b) < EPS and abs(y1a - y1b) < EPS:
                        if abs(x0a - x1b) < EPS:
                            merged_rect = (x0b, y0a, x1a, y1a)
                            skip.add(i); skip.add(j)
                            new_rects.append(merged_rect)
                            merged = True
                            break
                        elif abs(x0b - x1a) < EPS:
                            merged_rect = (x0a, y0a, x1b, y1a)
                            skip.add(i); skip.add(j)
                            new_rects.append(merged_rect)
                            merged = True
                            break

                    if abs(x0a - x0b) < EPS and abs(x1a - x1b) < EPS:
                        if abs(y0a - y1b) < EPS:
                            merged_rect = (x0a, y0b, x1a, y1a)
                            skip.add(i); skip.add(j)
                            new_rects.append(merged_rect)
                            merged = True
                            break
                        elif abs(y0b - y1a) < EPS:
                            merged_rect = (x0a, y0a, x1a, y1b)
                            skip.add(i); skip.add(j)
                            new_rects.append(merged_rect)
                            merged = True
                            break
                    if merged:
                        break
                if merged:
                    break

            for i, r in enumerate(self.rects):
                if i not in skip:
                    new_rects.append(r)
            new_rects = [r for r in new_rects if (r[2]-r[0])*(r[3]-r[1]) > EPS]
            if not merged:
                self.rects = new_rects
                break
            else:
                self.rects = new_rects

    def subtract(self, rect):
        new_rects = []
        for (fx0, fy0, fx1, fy1) in self.rects:
            if rect[2] <= fx0 + EPS or rect[0] >= fx1 - EPS or \
               rect[3] <= fy0 + EPS or rect[1] >= fy1 - EPS:
                new_rects.append((fx0, fy0, fx1, fy1))
                continue

            if fy1 > rect[3] + EPS:
                new_rects.append((fx0, rect[3], fx1, fy1))
            if fy0 < rect[1] - EPS:
                new_rects.append((fx0, fy0, fx1, rect[1]))
            if fx0 < rect[0] - EPS:
                y0 = max(fy0, rect[1])
                y1 = min(fy1, rect[3])
                if y0 < y1 - EPS:
                    new_rects.append((fx0, y0, rect[0], y1))
            if fx1 > rect[2] + EPS:
                y0 = max(fy0, rect[1])
                y1 = min(fy1, rect[3])
                if y0 < y1 - EPS:
                    new_rects.append((rect[2], y0, fx1, y1))

        self.rects = new_rects
        self._merge()

    def get_best_rect(self, poly_aspect=None, center_weight=0.4):
        if not self.rects:
            return (0.0, 0.0, 1.0, 1.0)

        def score(r):
            x0, y0, x1, y1 = r
            w = x1 - x0
            h = y1 - y0
            area = w * h
            if area < EPS:
                return -1e9
            aspect = w / h if h > EPS else 1.0
            thinness = min(aspect, 1.0/aspect) if aspect > 0 else 0.0
            thin_penalty = 1.0 - (1.0 - thinness) * 0.5
            cx = (x0 + x1) / 2
            cy = (y0 + y1) / 2
            dist = ((cx - 0.5)**2 + (cy - 0.5)**2)**0.5
            center_score = 1.0 - dist * 2.0
            if poly_aspect is not None and poly_aspect > 0:
                aspect_penalty = min(abs(poly_aspect - aspect) / max(poly_aspect, aspect, EPS), 1.0)
                aspect_score = 1.0 - aspect_penalty * 0.8
            else:
                aspect_score = 1.0
            return area * thin_penalty * (0.6 * aspect_score + 0.4 * max(center_score, 0))

        best = max(self.rects, key=score)
        return (best[0], best[1], best[2]-best[0], best[3]-best[1])

def get_default_layout_state():
    return {
        "ov-legend": {"left": 74, "top": 76, "width": 24, "height": 18, "visible": True, "padding": 0.025, "rotation": 0},
        "ov-north":  {"left": 86, "top": 2,  "width": 10, "height": 15, "visible": True, "padding": 0.02,  "rotation": 0},
        "ov-scale":  {"left": 2,  "top": 86, "width": 18, "height": 10, "visible": True, "padding": 0.02,  "rotation": 0},
        "ov-title":  {"left": 25, "top": 2,  "width": 50, "height": 8,  "visible": True, "padding": 0.015, "rotation": 0},
        "ov-area":   {"left": 74, "top": 12, "width": 20, "height": 6,  "visible": True, "padding": 0.015, "rotation": 0},
    }

def _rotated_bbox(rect, rotation_deg, center=None):
    x, y, w, h = rect
    if center is None:
        cx, cy = x + w/2, y + h/2
    else:
        cx, cy = center
    angle = math.radians(rotation_deg)
    corners = [(-w/2, -h/2), (w/2, -h/2), (w/2, h/2), (-w/2, h/2)]
    rot_corners = []
    for dx, dy in corners:
        rx = dx * math.cos(angle) - dy * math.sin(angle)
        ry = dx * math.sin(angle) + dy * math.cos(angle)
        rot_corners.append((cx + rx, cy + ry))
    xs = [p[0] for p in rot_corners]
    ys = [p[1] for p in rot_corners]
    return (min(xs), min(ys), max(xs), max(ys))

def compute_safe_rect(layout_state, poly_aspect=None):
    mgr = FreeSpaceManager()
    for key, item in layout_state.items():
        if not item.get('visible', True):
            continue
        left = item.get('left', 0) / 100.0
        top = item.get('top', 0) / 100.0
        width = item.get('width', 0) / 100.0
        height = item.get('height', 0) / 100.0
        bottom = 1.0 - top - height
        pad = item.get('padding', DEFAULT_PADDING)
        rot = item.get('rotation', 0)
        base = (left - pad, bottom - pad, left + width + pad, top + height + pad)
        if abs(rot) > EPS:
            cx = (base[0] + base[2]) / 2
            cy = (base[1] + base[3]) / 2
            bbox = _rotated_bbox((base[0], base[1], base[2]-base[0], base[3]-base[1]), rot, (cx, cy))
            bbox = (max(0, bbox[0]), max(0, bbox[1]), min(1, bbox[2]), min(1, bbox[3]))
            if bbox[0] < bbox[2] and bbox[1] < bbox[3]:
                mgr.subtract(bbox)
        else:
            mgr.subtract(base)
    return mgr.get_best_rect(poly_aspect)

# Label engine with coordinate-system correction
def _place_labels(ax, gdf, label_col, excluded_overlay_rects, fig,
                  fontsize=5.5, color='black', offset=8):
    if gdf is None or gdf.empty or label_col not in gdf.columns:
        return []

    trans = ax.transData.inverted()
    data_excluded = []
    for rect in excluded_overlay_rects:
        corners = [(rect[0], rect[1]), (rect[2], rect[1]), (rect[2], rect[3]), (rect[0], rect[3])]
        data_corners = [trans.transform_point(p) for p in corners]
        xs = [p[0] for p in data_corners]
        ys = [p[1] for p in data_corners]
        data_excluded.append((min(xs), min(ys), max(xs), max(ys)))

    placed = []
    for _, row in gdf.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue
        cx, cy = geom.centroid.x, geom.centroid.y

        best_pos = None
        best_score = None
        offsets = [(0, offset), (offset, offset), (offset, 0), (offset, -offset),
                   (0, -offset), (-offset, -offset), (-offset, 0), (-offset, offset)]
        for scale in [1.0, 0.6, 0.3]:
            for dx, dy in offsets:
                px, py = cx + dx*scale, cy + dy*scale
                label_text = str(row[label_col])
                size = 0.2 * fontsize * len(label_text) * 0.05
                rect = (px - size, py - size, px + size, py + size)

                overlap = False
                for ex in data_excluded:
                    if not (rect[2] <= ex[0] or rect[0] >= ex[2] or rect[3] <= ex[1] or rect[1] >= ex[3]):
                        overlap = True
                        break
                if overlap:
                    continue

                overlap_placed = False
                for p in placed:
                    if not (rect[2] <= p[0] or rect[0] >= p[2] or rect[3] <= p[1] or rect[1] >= p[3]):
                        overlap_placed = True
                        break
                if overlap_placed:
                    continue

                dist = ((px - cx)**2 + (py - cy)**2)**0.5
                if best_score is None or dist < best_score:
                    best_score = dist
                    best_pos = (px, py, rect)

        if best_pos is None:
            for scale in [1.0, 0.6, 0.3]:
                for dx, dy in offsets:
                    px, py = cx + dx*scale, cy + dy*scale
                    overlap = False
                    for ex in data_excluded:
                        if ex[0] <= px <= ex[2] and ex[1] <= py <= ex[3]:
                            overlap = True
                            break
                    if not overlap:
                        best_pos = (px, py, (px-0.01, py-0.01, px+0.01, py+0.01))
                        break
                if best_pos:
                    break
            if best_pos is None:
                best_pos = (cx, cy, (cx-0.01, cy-0.01, cx+0.01, cy+0.01))

        px, py, rect = best_pos
        placed.append(rect)
        ax.annotate(str(row[label_col]),
                    xy=(cx, cy),
                    xytext=(px, py),
                    textcoords="data",
                    ha='left' if px > cx else 'right',
                    va='bottom' if py > cy else 'top',
                    fontsize=fontsize,
                    fontweight='bold',
                    color=color,
                    path_effects=[pe.Stroke(linewidth=1.8, foreground='white'), pe.Normal()],
                    zorder=9)
    return placed

# ----------------------------------------------------------------------
# COMMON MAP PLOTTING HELPERS
# ----------------------------------------------------------------------

_COMP_COLORS_RICH = [
    "#2196F3","#FF9800","#4CAF50","#9C27B0","#F48FB1",
    "#607D8B","#CDDC39","#00BCD4","#FF5722","#795548",
    "#E91E63","#009688","#FFC107","#3F51B5","#8BC34A"
]

def _get_bounds(gdf):
    if gdf is None or gdf.empty:
        return None
    bounds = gdf.total_bounds
    if bounds is None or any(np.isnan(b) for b in bounds):
        return None
    return bounds

def _get_combined_bounds(gdfs):
    all_bounds = []
    for gdf in gdfs:
        b = _get_bounds(gdf)
        if b is not None:
            all_bounds.append(b)
    if not all_bounds:
        return (0, 0, 1, 1)
    minx = min(b[0] for b in all_bounds)
    miny = min(b[1] for b in all_bounds)
    maxx = max(b[2] for b in all_bounds)
    maxy = max(b[3] for b in all_bounds)
    return (minx, miny, maxx, maxy)

def _fit_bounds_to_axes(ax, bounds, safe_rect, fig_w, fig_h):
    minx, miny, maxx, maxy = bounds
    data_w = maxx - minx
    data_h = maxy - miny
    if data_w < EPS and data_h < EPS:
        data_w = data_h = 1.0

    pos = ax.get_position()
    ax_w = pos.width * fig_w
    ax_h = pos.height * fig_h
    ax_aspect = ax_w / ax_h if ax_h > EPS else 1.0

    data_aspect = data_w / data_h if data_h > EPS else 1.0

    extent = max(data_w, data_h)
    margin = 0.05 * extent
    margin = min(margin, 0.15 * extent)
    margin = max(margin, 0.02 * (ax_w + ax_h) / 2)

    data_w_m = data_w + 2 * margin
    data_h_m = data_h + 2 * margin

    if data_w_m / data_h_m > ax_aspect:
        new_w = data_w_m
        new_h = new_w / ax_aspect
    else:
        new_h = data_h_m
        new_w = new_h * ax_aspect

    center_x = (minx + maxx) / 2
    center_y = (miny + maxy) / 2
    half_w = new_w / 2
    half_h = new_h / 2
    ax.set_xlim(center_x - half_w, center_x + half_w)
    ax.set_ylim(center_y - half_h, center_y + half_h)

def _plot_polygons(ax, poly_gdf, label_col, excluded_overlay_rects, fig):
    if poly_gdf is None or poly_gdf.empty:
        return
    if "Comp_ID" in poly_gdf.columns:
        comps = poly_gdf["Comp_ID"].unique()
        color_map = {comp: _COMP_COLORS_RICH[i % len(_COMP_COLORS_RICH)] for i, comp in enumerate(comps)}
        poly_gdf = poly_gdf.copy()
        poly_gdf["display_color"] = poly_gdf["Comp_ID"].map(color_map)
        poly_gdf.plot(ax=ax, column="display_color", categorical=True,
                      edgecolor="#111111", linewidth=1.6)
    else:
        poly_gdf.plot(ax=ax, facecolor="#d4edda", edgecolor="#1a3a22", linewidth=1.5)

    if label_col and label_col in poly_gdf.columns:
        _place_labels(ax, poly_gdf, label_col, excluded_overlay_rects, fig,
                      fontsize=8, color="#1565C0", offset=6)

def _plot_points(ax, pts_gdf, point_label_col, excluded_overlay_rects, fig):
    if pts_gdf is None or pts_gdf.empty:
        return
    pts_gdf.plot(ax=ax, color="#ff0000", markersize=16, zorder=8, marker="o")
    if point_label_col and point_label_col in pts_gdf.columns:
        _place_labels(ax, pts_gdf, point_label_col, excluded_overlay_rects, fig,
                      fontsize=5.5, color="black", offset=4)

def _plot_lines(ax, line_gdf):
    if line_gdf is not None and not line_gdf.empty:
        line_gdf.plot(ax=ax, color="#000000", linewidth=1.2)

def _add_north_arrow(fig, pos=(0.90, 0.92)):
    from matplotlib.patches import Polygon
    x, y = pos
    ax = fig.add_axes([0,0,1,1], frameon=False)
    ax.set_axis_off()
    arrow = Polygon([[x, y-0.03], [x-0.015, y-0.01], [x+0.015, y-0.01]],
                    closed=True, facecolor='black', edgecolor='black',
                    transform=ax.transAxes, zorder=20)
    ax.add_patch(arrow)
    ax.text(x, y+0.015, 'N', transform=ax.transAxes,
            fontsize=12, fontweight='bold', ha='center', va='bottom',
            path_effects=[pe.Stroke(linewidth=2, foreground='white'), pe.Normal()],
            zorder=21)

def _add_scale_bar(fig, pos=(0.50, 0.06), width_frac=0.18, height_frac=0.015):
    x, y = pos
    ax = fig.add_axes([0,0,1,1], frameon=False)
    ax.set_axis_off()
    segs = 3
    seg_w = width_frac / segs
    for i in range(segs):
        fc = "black" if i % 2 == 0 else "white"
        rect = plt.Rectangle((x + i*seg_w, y), seg_w, height_frac,
                             linewidth=0.8, edgecolor="black", facecolor=fc,
                             transform=ax.transAxes, zorder=15, clip_on=False)
        ax.add_patch(rect)
    ax.text(x, y - 0.015, "0", ha='center', va='top', fontsize=7,
            fontweight='bold', color='black', transform=ax.transAxes, zorder=16)
    ax.text(x + width_frac/2, y - 0.015, "500 m", ha='center', va='top', fontsize=7,
            fontweight='bold', color='black', transform=ax.transAxes, zorder=16)
    ax.text(x + width_frac, y - 0.015, "1000 m", ha='center', va='top', fontsize=7,
            fontweight='bold', color='black', transform=ax.transAxes, zorder=16)

def _draw_slope_table(ax, slope_areas):
    """Draw a compact slope area table on the axes."""
    table_data = []
    for cls, info in SLOPE_CLASSES.items():
        area = slope_areas.get(info['range'], 0)
        table_data.append([info['range'], f"{area:.2f} ha"])
    total = sum(slope_areas.values())
    table_data.append(["Total", f"{total:.2f} ha"])
    table = ax.table(cellText=table_data, colLabels=["Slope", "Area"],
                     loc='lower left', bbox=[0.02, 0.02, 0.3, 0.15],
                     cellLoc='center', colWidths=[0.15, 0.10])
    table.auto_set_font_size(False)
    table.set_fontsize(7)
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight='bold', color='white')
            cell.set_facecolor('#1a5276')
        else:
            cell.set_facecolor('#f8f9fa')

SLOPE_CLASSES = {
    1: {"range": "< 19°",   "color": "#2e8b57", "label": "Gentle"},
    2: {"range": "19–31°",  "color": "#ffd700", "label": "Moderate"},
    3: {"range": "> 31°",   "color": "#ef4444", "label": "Steep"},
}

def render_map(path, poly_gdf=None, line_gdf=None, pts_gdf=None,
               label_col=None, point_label_col=None,
               safe_rect=None, layout_state=None,
               title=None, slope_mode=False, summary_rows=None,
               slope_areas=None):
    if safe_rect is None:
        safe_rect = (0.0, 0.0, 1.0, 1.0)

    fig = plt.figure(figsize=(FIG_W, FIG_H), dpi=DPI)
    fig.patch.set_facecolor("white")

    excluded_rects = []
    if layout_state:
        for key, item in layout_state.items():
            if not item.get('visible', True):
                continue
            left = item.get('left', 0) / 100.0
            top = item.get('top', 0) / 100.0
            width = item.get('width', 0) / 100.0
            height = item.get('height', 0) / 100.0
            bottom = 1.0 - top - height
            pad = item.get('padding', DEFAULT_PADDING)
            rot = item.get('rotation', 0)
            base = (left - pad, bottom - pad, left + width + pad, top + height + pad)
            if abs(rot) > EPS:
                cx = (base[0] + base[2]) / 2
                cy = (base[1] + base[3]) / 2
                bbox = _rotated_bbox((base[0], base[1], base[2]-base[0], base[3]-base[1]), rot, (cx, cy))
                bbox = (max(0, bbox[0]), max(0, bbox[1]), min(1, bbox[2]), min(1, bbox[3]))
                if bbox[0] < bbox[2] and bbox[1] < bbox[3]:
                    excluded_rects.append(bbox)
            else:
                excluded_rects.append(base)

    if slope_mode:
        height_ratios = [1, 0.18]
        gs = gridspec.GridSpec(2, 1, height_ratios=height_ratios,
                               left=safe_rect[0], right=safe_rect[0]+safe_rect[2],
                               bottom=safe_rect[1], top=safe_rect[1]+safe_rect[3],
                               hspace=0.15)
        ax_map = fig.add_subplot(gs[0])
        ax_tbl = fig.add_subplot(gs[1])
        ax_tbl.axis('off')
        ax = ax_map
    else:
        ax = fig.add_axes(safe_rect)

    ax.set_facecolor("white")
    ax.axis('off')

    _plot_polygons(ax, poly_gdf, label_col, excluded_rects, fig)
    _plot_lines(ax, line_gdf)
    _plot_points(ax, pts_gdf, point_label_col, excluded_rects, fig)

    gdfs = [poly_gdf, line_gdf, pts_gdf]
    bounds = _get_combined_bounds(gdfs)
    if bounds is not None:
        _fit_bounds_to_axes(ax, bounds, safe_rect, FIG_W, FIG_H)

    # North arrow and scale bar
    _add_north_arrow(fig)
    _add_scale_bar(fig)

    # Slope table (if slope_mode and slope_areas provided)
    if slope_mode and slope_areas:
        _draw_slope_table(ax, slope_areas)

    if title:
        fig.suptitle(title, fontsize=14, weight='bold')

    fig.savefig(path, dpi=DPI, facecolor="white", pad_inches=0)
    plt.close(fig)
    gc.collect()

# ----------------------------------------------------------------------
# GROUP A – BOUNDARY WHOLE
# ----------------------------------------------------------------------

def group_a(df, forest, crs, out, mapping=None):
    df = normalize_order(df)
    xc = safe_col(df,mapping,"X","X"); yc = safe_col(df,mapping,"Y","Y")
    oc = safe_col(df,mapping,"Order","Order")
    if not xc: raise ValueError("X/Easting column not found.")
    if not yc: raise ValueError("Y/Northing column not found.")
    if oc: df = df.sort_values(oc)
    coords = list(zip(df[xc], df[yc]))
    if len(coords) < 3: raise ValueError("Need ≥3 points for a polygon.")
    coords.append(coords[0])
    poly = safe_polygon(coords); line = LineString(coords)
    ah = round(poly.area/10000, 4); pfx = _safe_dn(forest)
    pg = gpd.GeoDataFrame([{"Forest":forest,"Area_ha":ah,"Perim_m":round(poly.length,2),"geometry":poly}],crs=crs)
    lg = gpd.GeoDataFrame([{"Forest":forest,"geometry":line}],crs=crs)
    ptg = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df[xc],df[yc]), crs=crs)
    pg.to_file(os.path.join(out,f"{pfx}_polygon.shp"))
    lg.to_file(os.path.join(out,f"{pfx}_line.shp"))
    ptg.to_file(os.path.join(out,f"{pfx}_point.shp"))
    return pg, lg, ptg

# ----------------------------------------------------------------------
# GROUP B – SEGMENTED FOREST
# ----------------------------------------------------------------------

def group_b(df, crs, out, mapping=None):
    df = normalize_order(df)
    xc=safe_col(df,mapping,"X","X"); yc=safe_col(df,mapping,"Y","Y")
    oc=safe_col(df,mapping,"Order","Order"); fc=safe_col(df,mapping,"Forest","Forest")
    cc=safe_col(df,mapping,"Compartment","Compartment")
    if not xc: raise ValueError("X column not found.")
    if not yc: raise ValueError("Y column not found.")
    if not fc: raise ValueError("Forest column not found.")
    polys,lines,pts=[],[],[]
    for f,g in df.groupby(fc):
        if oc: g=g.sort_values(oc)
        subs = g.groupby(cc) if cc else [(None,g)]
        for c,cg in subs:
            coords = list(zip(cg[xc],cg[yc]))
            if len(coords)<3: continue
            coords.append(coords[0]); poly=safe_polygon(coords)
            polys.append({"Forest":f,"Compartment":c,"Area_ha":round(poly.area/10000,4),"Perim_m":round(poly.length,2),"geometry":poly})
            lines.append({"Forest":f,"Compartment":c,"geometry":LineString(coords)})
            for _,r in cg.iterrows():
                pts.append({"Forest":f,"Compartment":c,"Order":r[oc] if oc else None,"geometry":Point(r[xc],r[yc])})
    p=gpd.GeoDataFrame(polys,crs=crs); l=gpd.GeoDataFrame(lines,crs=crs); pt=gpd.GeoDataFrame(pts,crs=crs)
    p=_enforce_poly_gdf(p)
    if not p.empty: p.to_file(os.path.join(out,"forest_polygon.shp"))
    if not l.empty: l.to_file(os.path.join(out,"forest_line.shp"))
    if not pt.empty: pt.to_file(os.path.join(out,"forest_point.shp"))
    return p,l,pt

# ----------------------------------------------------------------------
# GROUP C – SAMPLE PLOT GENERATOR
# ----------------------------------------------------------------------
import tempfile
import zipfile
import shutil
import traceback

def group_c(file, crs, w, h, rows, cols, out, mode, mapping=None, base_name="boundary", run_id=None):
    """
    Process boundary file (CSV/Excel or ZIP shapefile) and generate grid points.
    """
    polygons = []
    is_zip = file.filename.lower().endswith(".zip")

    if is_zip:
        tmp_dir = tempfile.mkdtemp(prefix="group_c_")
        zip_path = os.path.join(tmp_dir, "upload.zip")
        try:
            # Save uploaded ZIP to temp
            file.save(zip_path)

            # Extract
            with zipfile.ZipFile(zip_path, 'r') as zf:
                zf.extractall(tmp_dir)

            # Find all .shp files
            shps = []
            for root, _, files in os.walk(tmp_dir):
                for fname in files:
                    if fname.lower().endswith(".shp"):
                        shps.append(os.path.join(root, fname))

            if not shps:
                raise ValueError("No .shp file found in the uploaded ZIP.")

            # Choose target shapefile (if specified)
            sp = shps[0]
            ts = (mapping or {}).get("target_shp")
            if ts:
                for s in shps:
                    if os.path.basename(s) == os.path.basename(ts):
                        sp = s
                        break

            # Read shapefile
            try:
                gdf = gpd.read_file(sp)
            except Exception as e:
                raise ValueError(f"Failed to read shapefile: {e}")

            if gdf.empty:
                raise ValueError("Shapefile is empty.")

            # Ensure CRS
            if gdf.crs is None:
                gdf = gdf.set_crs(crs)
            else:
                try:
                    gdf = gdf.to_crs(crs)
                except Exception as e:
                    # Fallback: use the user-provided CRS
                    gdf = gdf.set_crs(crs)
                    log.warning(f"CRS conversion failed, set to {crs}: {e}")

            # Collect valid polygons
            for geom in gdf.geometry:
                if geom is None or geom.is_empty:
                    continue
                if geom.geom_type == "Polygon":
                    polygons.append(geom if geom.is_valid else geom.buffer(0))
                elif geom.geom_type == "MultiPolygon":
                    for p in geom.geoms:
                        polygons.append(p if p.is_valid else p.buffer(0))

            if not polygons:
                raise ValueError("No valid polygon geometries found in shapefile.")

        except Exception as e:
            # Log the full traceback
            log.error(f"Group C ZIP processing error: {traceback.format_exc()}")
            raise ValueError(f"ZIP processing failed: {e}")
        finally:
            # Clean up temp directory
            shutil.rmtree(tmp_dir, ignore_errors=True)

    else:
        # CSV/Excel handling (unchanged)
        df = read_input(file)
        df = normalize_order(df)
        xc = safe_col(df, mapping, "X", "X")
        yc = safe_col(df, mapping, "Y", "Y")
        oc = safe_col(df, mapping, "Order", "Order")

        if not xc or not yc:
            raise ValueError("X/Y columns not found.")

        if mode == "A":
            if oc:
                df = df.sort_values(oc)
            coords = list(zip(df[xc], df[yc]))
            coords.append(coords[0])
            polygons = [safe_polygon(coords)]
        else:  # mode B – segmented
            fc = safe_col(df, mapping, "Forest", "Forest")
            cc = safe_col(df, mapping, "Compartment", "Compartment")
            if not fc:
                raise ValueError("Forest column required for segmented mode.")
            gkeys = [fc, cc] if cc else [fc]
            for _, g in df.groupby(gkeys):
                if oc:
                    g = g.sort_values(oc)
                coords = list(zip(g[xc], g[yc]))
                if len(coords) < 3:
                    continue
                coords.append(coords[0])
                polygons.append(safe_polygon(coords))
            if not polygons:
                raise ValueError("No valid polygons could be constructed from the data.")

    if not polygons:
        raise ValueError("No valid polygons from input.")

    # Build GeoDataFrames
    p_gdf = gpd.GeoDataFrame([{"geometry": p} for p in polygons], crs=crs)
    l_gdf = gpd.GeoDataFrame([{"geometry": LineString(p.exterior.coords)} for p in polygons], crs=crs)

    # Union for point‑in‑polygon tests
    union = p_gdf.unary_union
    minx, miny, _, _ = union.bounds

    pts = []
    sn = 1
    for ri in range(rows):
        for ci in range(cols):
            center = Point(minx + ci * w + w / 2, miny + ri * h + h / 2)
            if union.contains(center):
                pts.append({"SN": sn, "X": center.x, "Y": center.y, "geometry": center})
                sn += 1

    pt_gdf = gpd.GeoDataFrame(pts, crs=crs)

    # Save outputs
    p_gdf = _enforce_poly_gdf(p_gdf)
    if not p_gdf.empty:
        p_gdf.to_file(os.path.join(out, f"{base_name}_polygon.shp"))
    l_gdf.to_file(os.path.join(out, f"{base_name}_line.shp"))
    if not pt_gdf.empty:
        pt_gdf.to_file(os.path.join(out, f"{base_name}_point.shp"))
        pd.DataFrame(pts)[["SN", "X", "Y"]].to_excel(os.path.join(out, f"{base_name}_sampleplot.xlsx"), index=False)

    return p_gdf, l_gdf, pt_gdf
# ----------------------------------------------------------------------
# GROUP D – MULTI-FOREST COMPLEX
# ----------------------------------------------------------------------

def _save_fl(pr,lr,pts,d,crs):
    os.makedirs(d,exist_ok=True); pfx=os.path.basename(d)
    _pdf=_enforce_poly_gdf(gpd.GeoDataFrame([pr],crs=crs))
    if not _pdf.empty: _pdf.to_file(os.path.join(d,f"{pfx}_polygon.shp"))
    gpd.GeoDataFrame([lr],crs=crs).to_file(os.path.join(d,f"{pfx}_line.shp"))
    gpd.GeoDataFrame(pts,crs=crs).to_file(os.path.join(d,f"{pfx}_point.shp"))

def group_d(df, crs, out, mapping=None, mode="A"):
    df=normalize_order(df)
    xc=safe_col(df,mapping,"X","X"); yc=safe_col(df,mapping,"Y","Y")
    oc=safe_col(df,mapping,"Order","Order"); fc=safe_col(df,mapping,"Forest","Forest")
    cc=safe_col(df,mapping,"Compartment","Compartment")
    if not xc: raise ValueError("X column not found.")
    if not yc: raise ValueError("Y column not found.")
    if not fc: raise ValueError("Forest column not found.")
    if mode=="B" and not cc: raise ValueError("Compartment column required.")
    ap,al,apt=[],[],[]
    for f,fg in df.groupby(fc):
        fd=os.path.join(out,_safe_dn(f))
        if mode=="B":
            for c,cg in fg.groupby(cc):
                if oc: cg=cg.sort_values(oc)
                coords=list(zip(cg[xc],cg[yc]))
                if len(coords)<3: continue
                coords.append(coords[0]); poly=safe_polygon(coords)
                pr={"Forest":f,"Compartment":c,"Area_ha":round(poly.area/10000,4),"Perim_m":round(poly.length,4),"geometry":poly}
                lr={"Forest":f,"Compartment":c,"geometry":LineString(coords)}
                ptl=[{"Forest":f,"Compartment":c,"Order":r[oc] if oc else None,"geometry":Point(r[xc],r[yc])} for _,r in cg.iterrows()]
                _save_fl(pr,lr,ptl,os.path.join(fd,_safe_dn(c)),crs)
                ap.append(pr); al.append(lr); apt.extend(ptl)
        else:
            if oc: fg=fg.sort_values(oc)
            coords=list(zip(fg[xc],fg[yc]))
            if len(coords)<3: continue
            coords.append(coords[0]); poly=safe_polygon(coords)
            pr={"Forest":f,"Area_ha":round(poly.area/10000,4),"Perim_m":round(poly.length,4),"geometry":poly}
            lr={"Forest":f,"geometry":LineString(coords)}
            ptl=[{"Forest":f,"Order":r[oc] if oc else None,"geometry":Point(r[xc],r[yc])} for _,r in fg.iterrows()]
            _save_fl(pr,lr,ptl,fd,crs); ap.append(pr); al.append(lr); apt.extend(ptl)
    if not ap: raise ValueError("No valid polygons built.")
    return gpd.GeoDataFrame(ap,crs=crs),gpd.GeoDataFrame(al,crs=crs),gpd.GeoDataFrame(apt,crs=crs)

# ----------------------------------------------------------------------
# GROUP E – POLYGON SUBDIVIDER
# ----------------------------------------------------------------------
"""
Polygon Subdivision Module  (fixed v3)
───────────────────────────────────────
Key fixes in this version
  • snap_tol auto-scaled to geometry units (degrees vs metres)
  • hard_clip() applied to EVERY piece before validation — no leaks possible
  • orig_poly padding fallback removed; _ensure_count raises instead
  • safe_overlay never silently returns an unclipped geometry
  • _validate_subdivision tolerance is unit-aware
  • _bisect PA-rotation path clips against original polygon, not running rem
  • _enforce_area_tolerance transfer always ≥ 0; uses join_style for compat
  • _clip_to_original guards every .length call against None
  • _ensure_count loop-capped; sub-pieces clipped before use
  • _subdivide_voronoi seeds never fewer than needed; GeometryCollection handled
  • _subdivide_grid explicit guard loops
"""

import math
import os
import uuid
import zipfile

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.affinity import rotate as sh_rotate
from shapely.geometry import (
    LineString, MultiPoint, Point, Polygon, MultiPolygon, box as _sbox,
)
from shapely.ops import snap, unary_union, voronoi_diagram
from shapely.validation import make_valid

# ═══════════════════════════════════════════════════════════
# SECTION 1 – SNAP TOLERANCE (unit-aware)
# ═══════════════════════════════════════════════════════════

def _snap_tol(geom):
    """
    Return a snap/residual tolerance appropriate for the geometry's coordinate scale.
    Projected CRS (metres): coords ~ 1e4–1e6  → tol = 0.01 m
    Geographic CRS (degrees): coords ~ -180..180 → tol = 1e-8 °
    """
    try:
        minx, miny, maxx, maxy = geom.bounds
        span = max(maxx - minx, maxy - miny, 1e-9)
        # If span > 1000, almost certainly metres
        return 0.01 if span > 1000 else 1e-8
    except Exception:
        return 1e-7

# residual tolerance for validation (m² or deg²); derived lazily from geometry
_GEOM_TOL_FRAC = 1e-6   # fraction of orig area that counts as "leak"


# ═══════════════════════════════════════════════════════════
# SECTION 2 – LOW-LEVEL GEOMETRY HELPERS
# ═══════════════════════════════════════════════════════════

def _force_valid(geom):
    if geom is None or geom.is_empty:
        return None
    if geom.is_valid:
        return geom
    for tol in (1e-8, 1e-7, 1e-6):
        try:
            g = geom.simplify(tol, preserve_topology=True)
            if g and not g.is_empty and g.is_valid:
                return g
        except Exception:
            pass
    try:
        g = geom.buffer(0)
        if g and not g.is_empty and g.is_valid:
            return g
    except Exception:
        pass
    try:
        g = make_valid(geom)
        if g and not g.is_empty and g.is_valid:
            return g
    except Exception:
        pass
    for eps in (1e-7, 1e-6, 1e-5):
        try:
            g = geom.buffer(eps).buffer(-eps)
            if g and not g.is_empty and g.is_valid:
                return g
        except Exception:
            continue
    try:
        g = geom.convex_hull
        if g and not g.is_empty and g.is_valid:
            return g
    except Exception:
        pass
    return None


def _close_poly(p):
    """Return a Polygon with explicitly closed rings; picks largest part of Multi."""
    if p is None:
        return None
    try:
        if p.is_empty:
            return p
        if p.geom_type == "MultiPolygon":
            parts = [x for x in p.geoms if not x.is_empty]
            if not parts:
                return p
            p = max(parts, key=lambda x: x.area)
        if p.geom_type != "Polygon":
            return p
        ext = list(p.exterior.coords)
        if ext[0] != ext[-1]:
            ext.append(ext[0])
        holes = []
        for ring in p.interiors:
            h = list(ring.coords)
            if h[0] != h[-1]:
                h.append(h[0])
            holes.append(h)
        c = Polygon(ext, holes)
        if c.is_empty:
            return p
        return c if c.is_valid else c.buffer(0)
    except Exception:
        return p


def _as_poly(g):
    """Extract the largest Polygon from any geometry, or None."""
    if g is None:
        return None
    try:
        if g.is_empty:
            return None
        if g.geom_type == "Polygon":
            return g
        if g.geom_type == "MultiPolygon":
            parts = [x for x in g.geoms if x.geom_type == "Polygon" and not x.is_empty]
            return max(parts, key=lambda x: x.area) if parts else None
        if g.geom_type in ("GeometryCollection",):
            polys = []
            for x in g.geoms:
                if x.geom_type == "Polygon" and not x.is_empty and x.area > 1e-10:
                    polys.append(x)
                elif x.geom_type == "MultiPolygon":
                    polys.extend(pp for pp in x.geoms
                                 if not pp.is_empty and pp.area > 1e-10)
            return max(polys, key=lambda x: x.area) if polys else None
    except Exception:
        pass
    return None


def _repair(g):
    """Full repair: validate → collapse Multi/Collection → close rings."""
    if g is None:
        return None
    try:
        if g.is_empty:
            return None
        g = _force_valid(g)
        if g is None or g.is_empty:
            return None
        if g.geom_type == "MultiPolygon":
            parts = [p for p in g.geoms if p and not p.is_empty and p.is_valid]
            if not parts:
                return None
            g = parts[0] if len(parts) == 1 else unary_union(parts)
            if g is None or g.is_empty:
                return None
        elif g.geom_type == "GeometryCollection":
            polys = [p for p in g.geoms
                     if p.geom_type in ("Polygon", "MultiPolygon")
                     and not p.is_empty and p.is_valid]
            if not polys:
                return None
            g = unary_union(polys)
            if g is None or g.is_empty:
                return None
        if g.geom_type == "Polygon":
            g = _close_poly(g)
        return g if (g and not g.is_empty) else None
    except Exception:
        try:
            fixed = g.buffer(0) if g else None
            return fixed if (fixed and not fixed.is_empty) else None
        except Exception:
            return None


# ═══════════════════════════════════════════════════════════
# SECTION 3 – HARD CLIP (guaranteed containment)
# ═══════════════════════════════════════════════════════════

def _hard_clip(piece, orig_poly, tol=None):
    """
    Guarantee that `piece` lies entirely within `orig_poly`.
    Uses progressively larger snap tolerances until the intersection succeeds
    and the result has no residual outside orig_poly.

    Returns a valid Polygon fully inside orig_poly, or None if impossible.
    """
    if piece is None or orig_poly is None:
        return None

    tol = tol or _snap_tol(orig_poly)
    tolerances = [tol, tol * 10, tol * 100, tol * 1000]

    for t in tolerances:
        try:
            ps = snap(piece, orig_poly, t)
            os_ = snap(orig_poly, piece, t)
            result = ps.intersection(os_)
            if result is None or result.is_empty:
                continue
            result = _repair(result)
            if result is None or result.is_empty:
                continue
            # Verify no residual leak
            diff = result.difference(orig_poly.buffer(t))
            if diff is None or diff.is_empty or diff.area < orig_poly.area * _GEOM_TOL_FRAC:
                # Final clip to be safe
                final = result.intersection(orig_poly.buffer(t * 2))
                if final is None or final.is_empty:
                    final = result
                return _close_poly(_repair(final))
        except Exception:
            continue

    # Last resort: use buffer(tol) on orig to absorb floating-point boundary
    try:
        padded = orig_poly.buffer(tol * 100)
        result = piece.intersection(padded)
        if result and not result.is_empty:
            result = result.intersection(orig_poly)
            if result and not result.is_empty:
                return _close_poly(_repair(result))
    except Exception:
        pass
    return None


# ═══════════════════════════════════════════════════════════
# SECTION 4 – SAFE OVERLAY
# ═══════════════════════════════════════════════════════════

def safe_overlay(a, b, op):
    """
    Robust two-operand overlay. snap_tol is derived from geometry scale.
    Never returns a geometry that is larger than both inputs (for intersection).
    """
    a = _repair(a)
    b = _repair(b)
    if a is None or b is None:
        return None

    tol = _snap_tol(a)

    def _do(ga, gb):
        if op == "intersection":
            return ga.intersection(gb)
        if op == "difference":
            return ga.difference(gb)
        if op == "union":
            return ga.union(gb)
        if op == "symmetric_difference":
            return ga.symmetric_difference(gb)
        raise ValueError(f"Unknown op: {op}")

    # Try with snapping
    try:
        result = _do(snap(a, b, tol), snap(b, a, tol))
        r = _repair(result)
        if r is not None:
            return r
    except Exception:
        pass

    # Fallback: buffer(0) repair before overlay
    try:
        result = _do(a.buffer(0), b.buffer(0))
        r = _repair(result)
        if r is not None:
            return r
    except Exception:
        pass

    # Fallback: make_valid both sides
    try:
        result = _do(make_valid(a), make_valid(b))
        return _repair(result)
    except Exception:
        return None


# ═══════════════════════════════════════════════════════════
# SECTION 5 – SHAPE DESCRIPTORS
# ═══════════════════════════════════════════════════════════

def _elong(p):
    minx, miny, maxx, maxy = p.bounds
    dx, dy = maxx - minx, maxy - miny
    return max(dx, dy) / max(min(dx, dy), 1e-9)

def _asp(p):
    b = p.bounds
    w, h = b[2] - b[0], b[3] - b[1]
    return max(w, h) / max(min(w, h), 1e-9)

def _pa_angle(p):
    try:
        c = np.array(p.exterior.coords[:-1])
        c -= c.mean(0)
        _, v = np.linalg.eigh(np.cov(c.T))
        return float(np.arctan2(v[1, 1], v[0, 1]))
    except Exception:
        return 0.0


# ═══════════════════════════════════════════════════════════
# SECTION 6 – AREA-BALANCE REFINEMENT
# ═══════════════════════════════════════════════════════════

def _enforce_area_tolerance(pieces, orig_poly, n, tol_ha):
    if not pieces or n < 2:
        return pieces

    tol_m2 = tol_ha * 10_000.0
    ideal = orig_poly.area / n

    for _iter in range(50):
        areas = [p.area for p in pieces]
        if max(areas) - min(areas) <= tol_m2:
            break

        i_max = int(np.argmax(areas))
        i_min = int(np.argmin(areas))
        if i_max == i_min:
            break

        excess  = areas[i_max] - ideal
        deficit = ideal - areas[i_min]
        transfer = max(min(excess, deficit) * 0.5, 0.0)
        if transfer < 1.0:
            break

        shared = safe_overlay(
            pieces[i_max].boundary, pieces[i_min].boundary, "intersection"
        )
        if shared is None or shared.is_empty or shared.length < 0.5:
            break

        strip_w = max(transfer / shared.length, 0.05)
        buf = shared.buffer(strip_w, join_style=2)
        strip = safe_overlay(pieces[i_max], buf, "intersection")
        if strip is None or strip.is_empty or strip.area < 0.1:
            continue

        ni = safe_overlay(pieces[i_max], strip, "difference")
        nj = safe_overlay(pieces[i_min], strip, "union")
        if (ni is not None and nj is not None
                and ni.area > ideal * 0.05 and nj.area > ideal * 0.05):
            pieces[i_max] = _close_poly(ni)
            pieces[i_min] = _close_poly(nj)

    return [_close_poly(_repair(p)) for p in pieces]


# ═══════════════════════════════════════════════════════════
# SECTION 7 – CLIP TO ORIGINAL & FILL GAPS
# ═══════════════════════════════════════════════════════════

def _clip_to_original(pieces, orig_poly):
    """
    Hard-clip every piece, then fill any residual gap back into
    the best-touching neighbour.
    """
    if not pieces:
        return [orig_poly]

    tol = _snap_tol(orig_poly)
    clipped = []
    for p in pieces:
        c = _hard_clip(p, orig_poly, tol)
        if c is not None and not c.is_empty:
            clipped.append(_close_poly(c))

    if not clipped:
        return [orig_poly]

    union_all = clipped[0]
    for p in clipped[1:]:
        union_all = safe_overlay(union_all, p, "union")

    gap = safe_overlay(orig_poly, union_all, "difference")
    if gap is None or gap.is_empty or gap.area <= orig_poly.area * _GEOM_TOL_FRAC:
        return clipped

    gap_geoms = (
        list(gap.geoms)
        if gap.geom_type in ("MultiPolygon", "GeometryCollection")
        else [gap]
    )

    for frag in gap_geoms:
        if frag is None or frag.is_empty or frag.area < orig_poly.area * _GEOM_TOL_FRAC:
            continue
        best_i, best_len = 0, -1.0
        for idx, cell in enumerate(clipped):
            inter = safe_overlay(cell.boundary, frag.boundary, "intersection")
            length = inter.length if (inter is not None and not inter.is_empty) else 0.0
            if length > best_len:
                best_len = length
                best_i = idx
        merged = safe_overlay(clipped[best_i], frag, "union")
        if merged is not None and not merged.is_empty:
            clipped[best_i] = _close_poly(merged)

    return clipped


# ═══════════════════════════════════════════════════════════
# SECTION 8 – ENSURE EXACT PIECE COUNT
# ═══════════════════════════════════════════════════════════

def _ensure_count(pieces, n, orig_poly):
    def _clean(lst):
        out = []
        for p in lst:
            p2 = _close_poly(_repair(p))
            if p2 is not None and not p2.is_empty:
                out.append(p2)
        return out

    pieces = _clean(pieces)
    if not pieces:
        # Bootstrap with grid
        pieces = _subdivide_grid(orig_poly, n)
        pieces = _clean(pieces)

    # Grow: split largest
    max_attempts = n * 4
    attempt = 0
    while len(pieces) < n and attempt < max_attempts:
        attempt += 1
        li = max(range(len(pieces)), key=lambda i: pieces[i].area)
        big = pieces.pop(li)
        sub = _bisect(big, 2)
        sub = _clean(sub)
        sub = [_close_poly(safe_overlay(p, orig_poly, "intersection")) for p in sub]
        sub = _clean(sub)
        if len(sub) >= 2:
            pieces.extend(sub)
        else:
            pieces.append(big)
            break

    # Shrink: merge smallest into best neighbour
    while len(pieces) > n:
        si = min(range(len(pieces)), key=lambda i: pieces[i].area)
        small = pieces.pop(si)
        if not pieces:
            pieces.append(small)
            break
        best_j, best_len = 0, -1.0
        for j, other in enumerate(pieces):
            inter = safe_overlay(small.boundary, other.boundary, "intersection")
            length = inter.length if (inter is not None and not inter.is_empty) else 0.0
            if length > best_len:
                best_len = length
                best_j = j
        merged = safe_overlay(small, pieces[best_j], "union")
        merged = safe_overlay(merged, orig_poly, "intersection") if merged else None
        if merged is not None and not merged.is_empty:
            pieces[best_j] = _close_poly(merged)
        else:
            pieces.append(small)
            break

    pieces = [_close_poly(safe_overlay(p, orig_poly, "intersection")) for p in pieces]
    return _clean(pieces)


# ═══════════════════════════════════════════════════════════
# SECTION 9 – FINAL HARD-CLIP PASS BEFORE VALIDATION
# ═══════════════════════════════════════════════════════════

def _final_clip_pass(pieces, orig_poly):
    """
    Last-chance hard clip: ensure EVERY piece is strictly inside orig_poly.
    Any piece that still leaks after _hard_clip is replaced by its intersection.
    """
    tol = _snap_tol(orig_poly)
    result = []
    for p in pieces:
        if p is None or p.is_empty:
            continue
        # Quick check: does it leak?
        try:
            diff = p.difference(orig_poly)
            leaks = diff is not None and not diff.is_empty and diff.area > orig_poly.area * _GEOM_TOL_FRAC
        except Exception:
            leaks = True

        if leaks:
            clipped = _hard_clip(p, orig_poly, tol)
            if clipped is not None and not clipped.is_empty:
                result.append(_close_poly(clipped))
        else:
            result.append(_close_poly(p))
    return result


# ═══════════════════════════════════════════════════════════
# SECTION 10 – VALIDATION
# ═══════════════════════════════════════════════════════════

def _validate_subdivision(pieces, orig_poly, n, tol_ha):
    if len(pieces) != n:
        raise ValueError(f"Expected {n} pieces, got {len(pieces)}")

    # Residual tolerance: 0.01% of orig area, minimum 0.01 m²
    res_tol = max(orig_poly.area * _GEOM_TOL_FRAC, 0.01)

    for i, p in enumerate(pieces):
        if p is None or p.is_empty:
            raise ValueError(f"Piece {i} is empty")
        if not p.is_valid:
            raise ValueError(f"Piece {i} is invalid geometry")
        try:
            diff = p.difference(orig_poly)
            leak = diff.area if (diff is not None and not diff.is_empty) else 0.0
        except Exception:
            leak = 0.0
        if leak > res_tol:
            raise ValueError(
                f"Piece {i} leaks outside original boundary "
                f"(residual {leak:.3e} m², tolerance {res_tol:.3e} m²)"
            )

    union_all = pieces[0]
    for p in pieces[1:]:
        union_all = safe_overlay(union_all, p, "union")
    try:
        gap = orig_poly.difference(union_all) if union_all else orig_poly
        gap_area = gap.area if (gap is not None and not gap.is_empty) else 0.0
    except Exception:
        gap_area = 0.0
    if gap_area > res_tol:
        raise ValueError(
            f"Pieces do not fully cover original polygon "
            f"(uncovered area: {gap_area:.3e} m²)"
        )

    areas = [p.area for p in pieces]
    spread = max(areas) - min(areas)
    tol_m2 = tol_ha * 10_000.0
    if spread > tol_m2:
        raise ValueError(
            f"Area spread {spread:.2f} m² ({spread/10000:.4f} ha) "
            f"exceeds tolerance {tol_m2:.2f} m² ({tol_ha} ha)"
        )
    return True


# ═══════════════════════════════════════════════════════════
# SECTION 11 – FOUR SUBDIVISION METHODS
# ═══════════════════════════════════════════════════════════

def _bisect(poly, n, axis=None, depth=0):
    from shapely.ops import split as sh_split

    poly = _repair(poly)
    if poly is None or poly.is_empty or poly.area < 1e-10:
        return []
    if n <= 1:
        return [_close_poly(poly)]

    minx, miny, maxx, maxy = poly.bounds
    cx0, cy0 = poly.centroid.x, poly.centroid.y

    # PA rotation for elongated polygons
    if axis is None and depth < 2 and _elong(poly) > 1.6:
        rot_deg = np.degrees(_pa_angle(poly))
        if 10 < abs(rot_deg) % 90 < 80:
            try:
                pr = sh_rotate(poly, -rot_deg, origin=(cx0, cy0))
                pr = _repair(pr)
                if pr and not pr.is_empty:
                    rp = _bisect(pr, n, "x", depth + 1)
                    if len(rp) == n:
                        pieces = []
                        for p in rp:
                            pb = sh_rotate(p, rot_deg, origin=(cx0, cy0))
                            # Clip against the ORIGINAL (unrotated) polygon
                            pb = safe_overlay(pb, poly, "intersection")
                            pg = _as_poly(pb)
                            if pg and pg.area > 1e-10:
                                pieces.append(_close_poly(pg))
                        if len(pieces) == n:
                            return pieces
            except Exception:
                pass

    nl = n // 2
    nr = n - nl
    frac = nl / n

    def _cut(cax):
        lo, hi = (minx, maxx) if cax == "x" else (miny, maxy)
        target = poly.area * frac
        best_m = (lo + hi) / 2.0

        for _ in range(120):
            m = (lo + hi) / 2.0
            box = (
                _sbox(minx - 1, miny - 1, m, maxy + 1)
                if cax == "x"
                else _sbox(minx - 1, miny - 1, maxx + 1, m)
            )
            lp = safe_overlay(poly, box, "intersection")
            got = lp.area if lp else 0.0
            err = abs(got - target) / (target + 1e-12)
            if err < 5e-5:
                best_m = m
                break
            if got < target:
                lo = m
            else:
                hi = m
            best_m = m
            if hi - lo < 1e-10:
                break

        box = (
            _sbox(minx - 1, miny - 1, best_m, maxy + 1)
            if cax == "x"
            else _sbox(minx - 1, miny - 1, maxx + 1, best_m)
        )
        lp = safe_overlay(poly, box, "intersection")
        rp = safe_overlay(poly, lp, "difference") if lp else None

        la = _as_poly(lp)
        ra = _as_poly(rp)
        if la and ra and la.area > 1e-10 and ra.area > 1e-10:
            return la, ra

        # LineString split fallback
        try:
            if cax == "x":
                cut_line = LineString([(best_m, miny - 1), (best_m, maxy + 1)])
            else:
                cut_line = LineString([(minx - 1, best_m), (maxx + 1, best_m)])
            parts = [p for p in sh_split(poly, cut_line).geoms if p.area > 1e-10]
            if len(parts) >= 2:
                parts.sort(key=lambda p: p.centroid.x if cax == "x" else p.centroid.y)
                la = _as_poly(parts[0])
                ra = _as_poly(parts[-1])
                if la and ra and la.area > 1e-10 and ra.area > 1e-10:
                    return la, ra
        except Exception:
            pass
        return None

    best = None
    for cax in (["x", "y"] if axis is None else [axis]):
        cut = _cut(cax)
        if cut is None:
            continue
        lp, rp = cut
        score = max(_asp(lp), _asp(rp))
        if best is None or score < best[0]:
            best = (score, lp, rp)

    if best is None:
        return [_close_poly(poly)]

    _, lp, rp = best
    lp = _close_poly(_repair(safe_overlay(lp, poly, "intersection")))
    rp = _close_poly(_repair(safe_overlay(rp, poly, "intersection")))

    left = _bisect(lp, nl, None, depth + 1)
    right = _bisect(rp, nr, None, depth + 1)
    return [
        _close_poly(_repair(p))
        for p in left + right
        if p and not p.is_empty
    ]


def _subdivide_ba(poly, n, area_tol_ha=0.3):
    pieces = _bisect(poly, n)
    if not pieces:
        return [_close_poly(poly)]
    pieces = [_close_poly(_repair(p)) for p in pieces if p and not p.is_empty]
    if len(pieces) < n:
        return _subdivide_grid(poly, n)

    ideal = poly.area / n

    for _iter in range(15):
        areas = [p.area for p in pieces]
        if max(areas) - min(areas) <= area_tol_ha * 10_000.0:
            break
        pairs = sorted(
            [(i, j) for i in range(len(pieces))
             for j in range(i + 1, len(pieces))],
            key=lambda ij: abs(areas[ij[0]] - areas[ij[1]]),
            reverse=True,
        )
        changed = False
        for i, j in pairs:
            ai, aj = pieces[i].area, pieces[j].area
            if abs(ai - aj) < 1.0:
                continue
            shared = safe_overlay(
                pieces[i].boundary, pieces[j].boundary, "intersection"
            )
            if shared is None or shared.is_empty or shared.length < 0.5:
                continue
            big, small = (i, j) if ai > aj else (j, i)
            buf_w = max(abs(ai - aj) * 0.5 / shared.length, 0.05)
            buf = shared.buffer(buf_w, join_style=2)
            strip = safe_overlay(pieces[big], buf, "intersection")
            if strip is None or strip.is_empty:
                continue
            nb = safe_overlay(pieces[big], strip, "difference")
            ns = safe_overlay(pieces[small], strip, "union")
            if (nb is not None and ns is not None
                    and nb.area > ideal * 0.05 and ns.area > ideal * 0.05):
                pieces[big] = _close_poly(nb)
                pieces[small] = _close_poly(ns)
                areas[big] = pieces[big].area
                areas[small] = pieces[small].area
                changed = True
        if not changed:
            break

    return pieces


def _subdivide_voronoi(poly, n, area_tol_ha=0.3, max_iter=30):
    import random
    try:
        minx, miny, maxx, maxy = poly.bounds
        seeds = []

        cn = int(math.ceil(math.sqrt(n)))
        rn = int(math.ceil(n / cn))
        for ri in range(rn):
            for ci in range(cn):
                px = minx + (ci + 0.5) * (maxx - minx) / cn
                py = miny + (ri + 0.5) * (maxy - miny) / rn
                pt = Point(px, py)
                if poly.contains(pt):
                    seeds.append(pt)

        for _ in range(n * 500):
            if len(seeds) >= n:
                break
            pt = Point(random.uniform(minx, maxx), random.uniform(miny, maxy))
            if poly.contains(pt):
                seeds.append(pt)

        if len(seeds) < 2:
            return _subdivide_grid(poly, n)
        seeds = seeds[:n]

        for _ in range(max_iter):
            mp = MultiPoint(seeds)
            try:
                diagram = voronoi_diagram(mp, envelope=poly.buffer(
                    max(_snap_tol(poly) * 100, 1.0)
                ))
            except Exception:
                break

            cells = []
            for region in diagram.geoms:
                cl = safe_overlay(region, poly, "intersection")
                if cl is not None and not cl.is_empty and cl.area > 1e-10:
                    pg = _as_poly(cl)
                    if pg:
                        cells.append(pg)

            if not cells:
                break
            cells.sort(key=lambda c: c.area, reverse=True)
            cells = cells[:n]

            new_seeds = [c.centroid for c in cells]
            moved = max(
                (ns.distance(s) for ns, s in zip(new_seeds, seeds[: len(new_seeds)])),
                default=0.0,
            )
            seeds = new_seeds
            if moved < _snap_tol(poly) * 10 or len(seeds) < 2:
                break

        cells = [_close_poly(_repair(c)) for c in cells if c and not c.is_empty]
        return cells if len(cells) >= 2 else _subdivide_grid(poly, n)

    except Exception:
        return _subdivide_grid(poly, n)


def _subdivide_grid(poly, n):
    cn = int(math.ceil(math.sqrt(n)))
    rn = int(math.ceil(n / cn))
    minx, miny, maxx, maxy = poly.bounds
    cw = (maxx - minx) / cn
    rh = (maxy - miny) / rn

    cells = []
    for ri in range(rn):
        for ci in range(cn):
            box = _sbox(
                minx + ci * cw, miny + ri * rh,
                minx + (ci + 1) * cw, miny + (ri + 1) * rh,
            )
            cl = safe_overlay(box, poly, "intersection")
            if cl is not None and cl.area > 1e-10:
                cells.append(_close_poly(cl))

    # Merge excess
    max_merge = len(cells) * 3
    attempt = 0
    while len(cells) > n and attempt < max_merge:
        attempt += 1
        si = min(range(len(cells)), key=lambda i: cells[i].area)
        small = cells.pop(si)
        if not cells:
            cells.append(small)
            break
        best_j, best_len = 0, -1.0
        for j, c in enumerate(cells):
            inter = safe_overlay(small.boundary, c.boundary, "intersection")
            length = inter.length if (inter is not None and not inter.is_empty) else 0.0
            if length > best_len:
                best_len = length
                best_j = j
        merged = safe_overlay(cells[best_j], small, "union")
        if merged is not None and not merged.is_empty:
            cells[best_j] = _close_poly(merged)
        else:
            cells.append(small)
            break

    # Split deficit
    max_split = n * 4
    attempt = 0
    while len(cells) < n and attempt < max_split:
        attempt += 1
        li = max(range(len(cells)), key=lambda i: cells[i].area)
        big = cells.pop(li)
        sub = _bisect(big, 2)
        sub = [_close_poly(p) for p in sub if p and not p.is_empty]
        if len(sub) >= 2:
            cells.extend(sub)
        else:
            cells.append(big)
            break

    return cells[:n]


# ═══════════════════════════════════════════════════════════
# SECTION 12 – MASTER SUBDIVISION PIPELINE
# ═══════════════════════════════════════════════════════════

def _subdivide(poly, n, method="bisect", area_tol_ha=0.3):
    """
    Full pipeline with guaranteed boundary preservation.

    Order of operations:
      1  Repair & validate input
      2  Run chosen method
      3  _ensure_count  → exactly n pieces
      4  _clip_to_original  → clip + fill gaps
      5  _enforce_area_tolerance  → balance areas
      6  _clip_to_original  → re-clip after balance moves
      7  _ensure_count  → restore count if balance changed it
      8  _final_clip_pass  → hard-clip every piece (catches any remaining leak)
      9  _validate_subdivision  → raise on any remaining error
    """
    n = max(2, min(15, int(n)))

    orig_poly = _repair(poly)
    if orig_poly is None or orig_poly.is_empty:
        return []
    if not orig_poly.is_valid:
        orig_poly = make_valid(orig_poly)
    orig_poly = orig_poly.buffer(0)
    orig_poly = _repair(orig_poly)
    if orig_poly is None or orig_poly.is_empty:
        return []

    method_map = {
        "voronoi": _subdivide_voronoi,
        "grid":    _subdivide_grid,
        "ba":      _subdivide_ba,
        "bisect":  _bisect,
    }
    fn = method_map.get(method, _bisect)
    try:
        if method in ("ba", "voronoi"):
            pieces = fn(orig_poly, n, area_tol_ha)
        else:
            pieces = fn(orig_poly, n)
    except Exception as e:
        print(f"[subdivision] method={method!r} failed ({e}); falling back to grid.")
        pieces = _subdivide_grid(orig_poly, n)

    # Post-processing pipeline
    pieces = _ensure_count(pieces, n, orig_poly)
    pieces = _clip_to_original(pieces, orig_poly)
    pieces = _enforce_area_tolerance(pieces, orig_poly, n, area_tol_ha)
    pieces = _clip_to_original(pieces, orig_poly)
    pieces = _ensure_count(pieces, n, orig_poly)

    # ── CRITICAL: hard-clip every piece before validation ──────────────────
    pieces = _final_clip_pass(pieces, orig_poly)

    # If count drifted (shouldn't), fix it
    if len(pieces) != n:
        pieces = _ensure_count(pieces, n, orig_poly)
        pieces = _final_clip_pass(pieces, orig_poly)

    # Final cleanup
    pieces = [_close_poly(_repair(p)) for p in pieces if p is not None and not p.is_empty]

    if len(pieces) < n:
        # Absolute last resort: pad from grid (already clipped to orig_poly)
        grid_extras = _subdivide_grid(orig_poly, n)
        grid_extras = _final_clip_pass(grid_extras, orig_poly)
        pieces = pieces + grid_extras[len(pieces):]
        pieces = pieces[:n]

    _validate_subdivision(pieces, orig_poly, n, area_tol_ha)
    return pieces[:n]


# ═══════════════════════════════════════════════════════════
# SECTION 13 – OUTPUT HELPERS
# ═══════════════════════════════════════════════════════════

def _extract_div_pts(pieces, fname):
    recs = []
    sn = 1
    for i, p in enumerate(pieces, 1):
        cid = f"Comp_{i:03d}"
        p = _close_poly(_repair(p))
        if p is None or p.is_empty:
            continue
        coords = list(p.exterior.coords)
        if len(coords) > 1 and coords[0] == coords[-1]:
            coords = coords[:-1]
        if not coords:
            continue
        n_c = len(coords)
        edge_lens = [
            math.hypot(
                coords[(j + 1) % n_c][0] - coords[j][0],
                coords[(j + 1) % n_c][1] - coords[j][1],
            )
            for j in range(n_c)
        ]
        mean_e = sum(edge_lens) / len(edge_lens) if edge_lens else 1.0
        thresh = mean_e * 1.5

        for j, (cx, cy) in enumerate(coords):
            recs.append({
                "SN": sn, "Forest": fname, "Comp_ID": cid,
                "Type": "Vertex", "X": round(cx, 4), "Y": round(cy, 4),
            })
            sn += 1
            nx, ny = coords[(j + 1) % n_c]
            if edge_lens[j] > thresh:
                recs.append({
                    "SN": sn, "Forest": fname, "Comp_ID": cid,
                    "Type": "Midpoint",
                    "X": round((cx + nx) / 2, 4),
                    "Y": round((cy + ny) / 2, 4),
                })
                sn += 1
    return recs


def _save_compartments(pieces, fname, crs, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    pieces = [_close_poly(_repair(p)) for p in pieces if p and not p.is_empty]
    total_area = sum(p.area for p in pieces)

    poly_recs, line_recs, pt_recs = [], [], []
    for i, p in enumerate(pieces, 1):
        cid = f"Comp_{i:03d}"
        ah  = round(p.area / 10_000, 4)
        pm  = round(p.length, 4)
        pct = round(p.area / total_area * 100, 2) if total_area > 0 else 0
        poly_recs.append({
            "Forest": fname, "Comp_ID": cid,
            "Area_ha": ah, "Perim_m": pm, "Pct_Area": pct, "geometry": p,
        })
        ext = list(p.exterior.coords)
        if ext[0] != ext[-1]:
            ext.append(ext[0])
        line_recs.append({"Forest": fname, "Comp_ID": cid, "geometry": LineString(ext)})
        pt_recs.append({
            "Forest": fname, "Comp_ID": cid,
            "Area_ha": ah, "Pct_Area": pct, "geometry": p.centroid,
        })

    pg  = gpd.GeoDataFrame(poly_recs, crs=crs)
    lg  = gpd.GeoDataFrame(line_recs, crs=crs)
    ptg = gpd.GeoDataFrame(pt_recs,   crs=crs)

    pg  = _enforce_poly_gdf(pg)
    pfx = _safe_dn(fname)

    if not pg.empty:
        pg.to_file(os.path.join(save_dir, f"{pfx}_compartment_polygon.shp"))
    if not lg.empty:
        lg.to_file(os.path.join(save_dir, f"{pfx}_compartment_line.shp"))
    if not ptg.empty:
        ptg.to_file(os.path.join(save_dir, f"{pfx}_compartment_point.shp"))

    pd.DataFrame(
        [{k: v for k, v in r.items() if k != "geometry"} for r in poly_recs]
    ).to_excel(
        os.path.join(save_dir, f"{pfx}_compartment_summary.xlsx"), index=False
    )

    dp = _extract_div_pts(pieces, fname)
    if dp:
        ddf = pd.DataFrame(dp)
        ddf.to_excel(
            os.path.join(save_dir, f"{pfx}_division_points.xlsx"), index=False
        )
        gpd.GeoDataFrame(
            ddf,
            geometry=gpd.points_from_xy(ddf["X"], ddf["Y"]),
            crs=crs,
        ).to_file(os.path.join(save_dir, f"{pfx}_division_points.shp"))

    return pg, lg, ptg


# ═══════════════════════════════════════════════════════════
# SECTION 14 – INPUT LOADERS
# ═══════════════════════════════════════════════════════════

def _df_to_poly(df, xc, yc, oc):
    if oc and oc in df.columns:
        df = df.sort_values(oc)
    coords = list(zip(df[xc], df[yc]))
    if len(coords) < 3:
        raise ValueError("Need ≥ 3 points to build a polygon.")
    coords.append(coords[0])
    return _close_poly(_repair(Polygon(coords)))


def _load_polys_from_zip(file, target_shp, crs, fcol=None):
    folder = os.path.join(UPLOAD, str(uuid.uuid4()))
    os.makedirs(folder, exist_ok=True)
    zp = os.path.join(folder, "i.zip")
    file.save(zp)
    with zipfile.ZipFile(zp) as z:
        z.extractall(folder)

    shps = [
        os.path.join(r, f)
        for r, _, fs in os.walk(folder)
        for f in fs if f.endswith(".shp")
    ]
    if not shps:
        raise ValueError("No .shp found in ZIP.")

    sp = shps[0]
    if target_shp:
        tn = os.path.basename(target_shp)
        for s in shps:
            if os.path.basename(s) == tn:
                sp = s
                break

    gdf = gpd.read_file(sp)
    if gdf.empty:
        raise ValueError("Shapefile is empty.")
    gdf = gdf.set_crs(crs) if gdf.crs is None else gdf.to_crs(crs)

    nc = None
    if fcol:
        for c in gdf.columns:
            if c.lower() == fcol.lower():
                nc = c
                break
    if nc is None:
        for cand in ("Forest","forest","Name","name","NAME","Label","label","ID","id"):
            if cand in gdf.columns:
                nc = cand
                break
    if nc is None:
        for c in gdf.columns:
            if c != "geometry" and gdf[c].dtype == object:
                nc = c
                break

    results = []
    for i, row in gdf.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue
        fn = str(row[nc]) if nc else f"Feature_{i + 1}"
        if geom.geom_type == "Polygon":
            pls = [_repair(geom)]
        elif geom.geom_type == "MultiPolygon":
            pls = [_repair(unary_union(list(geom.geoms)))]
        elif hasattr(geom, "geoms"):
            pls = [_repair(unary_union(
                [g for g in geom.geoms if g.geom_type == "Polygon"]
            ))]
        else:
            pls = []
        for p in pls:
            if p and p.area > 1e-6:
                results.append((fn, _close_poly(p)))

    if not results:
        raise ValueError("No polygon geometries found in shapefile.")
    return results, shps


# ═══════════════════════════════════════════════════════════
# SECTION 15 – MAIN ENTRY POINT
# ═══════════════════════════════════════════════════════════

def group_e(
    file_or_df,
    crs,
    out,
    mapping=None,
    e_mode="A",
    n_compartments=4,
    is_zip=False,
    fcol=None,
    area_tol_ha=0.3,
    method="bisect",
    run_id=None,
):
    n_compartments = max(2, min(15, int(n_compartments)))
    ap, al, apts = [], [], []

    if is_zip:
        ts = (mapping or {}).get("target_shp")
        features, _ = _load_polys_from_zip(file_or_df, ts, crs, fcol)
        total = len(features)
        for idx, (fn, poly) in enumerate(features):
            pct = 20 + int(60 * idx / max(total, 1))
            if run_id:
                _prog(run_id, f"[{idx+1}/{total}] Subdividing {fn}…", pct)
            pieces = _subdivide(poly, n_compartments, method, area_tol_ha)
            areas  = [round(p.area / 10_000, 2) for p in pieces if p]
            ideal  = round(poly.area / 10_000 / n_compartments, 2)
            diff   = max((abs(a - ideal) for a in areas), default=0)
            if run_id:
                _prog(run_id,
                      f"[{idx+1}/{total}] {fn}: {len(pieces)} parts ✓  "
                      f"ideal={ideal} ha  max_diff={diff:.2f} ha",
                      pct + 5)
            fd = os.path.join(out, _safe_dn(fn)) if total > 1 else out
            pg, lg, ptg = _save_compartments(pieces, fn, crs, fd)
            ap.append(pg); al.append(lg); apts.append(ptg)

    else:
        df = file_or_df
        df = normalize_order(df)
        xc = safe_col(df, mapping, "X", "X")
        yc = safe_col(df, mapping, "Y", "Y")
        oc = safe_col(df, mapping, "Order", "Order")
        fc = safe_col(df, mapping, "Forest", "Forest")

        if not xc:
            raise ValueError("X column not found.")
        if not yc:
            raise ValueError("Y column not found.")
        if e_mode == "B" and not fc:
            raise ValueError("Forest column required for mode B.")

        if e_mode == "A":
            fn = (mapping or {}).get("forest") or "FOREST"
            if run_id:
                _prog(run_id, "Building & subdividing polygon…", 15)
            poly   = _df_to_poly(df, xc, yc, oc)
            pieces = _subdivide(poly, n_compartments, method, area_tol_ha)
            pg, lg, ptg = _save_compartments(pieces, fn, crs, out)
            ap.append(pg); al.append(lg); apts.append(ptg)

        else:
            groups = list(df.groupby(fc))
            total  = len(groups)
            for idx, (f, fg) in enumerate(groups):
                if run_id:
                    _prog(run_id, f"Processing {f}…",
                          15 + int(65 * idx / max(total, 1)))
                try:
                    poly   = _df_to_poly(fg, xc, yc, oc)
                    pieces = _subdivide(poly, n_compartments, method, area_tol_ha)
                    fd     = os.path.join(out, _safe_dn(str(f)))
                    pg, lg, ptg = _save_compartments(pieces, str(f), crs, fd)
                    ap.append(pg); al.append(lg); apts.append(ptg)
                except Exception as ex:
                    if run_id:
                        _prog(run_id, f"Warning: {f} skipped — {ex}")

    if not ap:
        raise ValueError("No valid polygons were built.")

    p_out  = gpd.GeoDataFrame(pd.concat(ap,    ignore_index=True), crs=crs)
    l_out  = gpd.GeoDataFrame(pd.concat(al,    ignore_index=True), crs=crs)
    pt_out = gpd.GeoDataFrame(pd.concat(apts,  ignore_index=True), crs=crs)
    return p_out, l_out, pt_out
# ----------------------------------------------------------------------
# GROUP F – SLOPE ANALYSIS
# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
# Helper functions for Group F (add these before group_f if not already present)
# ----------------------------------------------------------------------

def _bnd_from_df(df, mapping):
    """Build boundary polygon from DataFrame (CSV/Excel) using X/Y columns."""
    df = normalize_order(df)
    xc = safe_col(df, mapping, "X", "X")
    yc = safe_col(df, mapping, "Y", "Y")
    oc = safe_col(df, mapping, "Order", "Order")
    if not xc:
        raise ValueError("X column not found.")
    if not yc:
        raise ValueError("Y column not found.")
    if oc:
        df = df.sort_values(oc)
    coords = list(zip(df[xc], df[yc]))
    if len(coords) < 3:
        raise ValueError("Need ≥3 boundary points.")
    coords.append(coords[0])
    return safe_polygon(coords)

def _bnd_from_zip(zip_file, target_shp, src_crs, dem_crs):
    """
    Extract boundary polygon from a ZIP containing a shapefile.
    Returns (boundary_polygon, boundary_gdf, basename).
    """
    import io, zipfile, tempfile
    zip_bytes = zip_file.read()
    if len(zip_bytes) < 100:
        raise ValueError("Uploaded ZIP file is empty or too small.")
    tmp_dir = os.path.join(UPLOAD, "bnd_tmp_" + uuid.uuid4().hex[:8])
    os.makedirs(tmp_dir, exist_ok=True)
    try:
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
            zf.extractall(tmp_dir)
        shps = []
        for root, _, files in os.walk(tmp_dir):
            for f in files:
                if f.lower().endswith(".shp"):
                    shps.append(os.path.join(root, f))
        if not shps:
            raise ValueError("No .shp file found inside the uploaded ZIP.")
        if target_shp:
            target_base = os.path.basename(target_shp)
            chosen = None
            for s in shps:
                if os.path.basename(s) == target_base:
                    chosen = s
                    break
            if chosen is None:
                raise ValueError(f"Specified shapefile '{target_shp}' not found in ZIP.")
        else:
            chosen = shps[0]
        shp_basename = os.path.splitext(os.path.basename(chosen))[0]
        gdf = gpd.read_file(chosen)
        if gdf.empty:
            raise ValueError("Boundary shapefile is empty.")
        if gdf.crs is None:
            gdf = gdf.set_crs(src_crs)
        else:
            gdf = gdf.to_crs(dem_crs)
        union = _repair(gdf.unary_union)
        if union is None or union.is_empty:
            raise ValueError("Boundary geometry is empty after union.")
        return union, gdf, shp_basename
    except Exception as e:
        raise ValueError(f"Failed to process boundary ZIP: {e}")
    finally:
        try:
            shutil.rmtree(tmp_dir)
        except Exception:
            pass


# ----------------------------------------------------------------------
# GROUP F – SLOPE ANALYSIS (complete, corrected version)
# ----------------------------------------------------------------------

# ----------------------------------------------------------------------
# Helper functions for Group F (add these before group_f if not already present)
# ----------------------------------------------------------------------

# ----------------------------------------------------------------------
# Helper functions for Group F (add these before group_f if not already present)
# ----------------------------------------------------------------------

def _bnd_from_df(df, mapping):
    """Build boundary polygon from DataFrame (CSV/Excel) using X/Y columns."""
    df = normalize_order(df)
    xc = safe_col(df, mapping, "X", "X")
    yc = safe_col(df, mapping, "Y", "Y")
    oc = safe_col(df, mapping, "Order", "Order")
    if not xc:
        raise ValueError("X column not found.")
    if not yc:
        raise ValueError("Y column not found.")
    if oc:
        df = df.sort_values(oc)
    coords = list(zip(df[xc], df[yc]))
    if len(coords) < 3:
        raise ValueError("Need ≥3 boundary points.")
    coords.append(coords[0])
    return safe_polygon(coords)

def _bnd_from_zip(zip_file, target_shp, src_crs, dem_crs):
    """
    Extract boundary polygon from a ZIP containing a shapefile.
    Returns (boundary_polygon, boundary_gdf, basename).
    """
    import io, zipfile, tempfile
    zip_bytes = zip_file.read()
    if len(zip_bytes) < 100:
        raise ValueError("Uploaded ZIP file is empty or too small.")
    tmp_dir = os.path.join(UPLOAD, "bnd_tmp_" + uuid.uuid4().hex[:8])
    os.makedirs(tmp_dir, exist_ok=True)
    try:
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
            zf.extractall(tmp_dir)
        shps = []
        for root, _, files in os.walk(tmp_dir):
            for f in files:
                if f.lower().endswith(".shp"):
                    shps.append(os.path.join(root, f))
        if not shps:
            raise ValueError("No .shp file found inside the uploaded ZIP.")
        if target_shp:
            target_base = os.path.basename(target_shp)
            chosen = None
            for s in shps:
                if os.path.basename(s) == target_base:
                    chosen = s
                    break
            if chosen is None:
                raise ValueError(f"Specified shapefile '{target_shp}' not found in ZIP.")
        else:
            chosen = shps[0]
        shp_basename = os.path.splitext(os.path.basename(chosen))[0]
        gdf = gpd.read_file(chosen)
        if gdf.empty:
            raise ValueError("Boundary shapefile is empty.")
        if gdf.crs is None:
            gdf = gdf.set_crs(src_crs)
        else:
            gdf = gdf.to_crs(dem_crs)
        union = _repair(gdf.unary_union)
        if union is None or union.is_empty:
            raise ValueError("Boundary geometry is empty after union.")
        return union, gdf, shp_basename
    except Exception as e:
        raise ValueError(f"Failed to process boundary ZIP: {e}")
    finally:
        try:
            shutil.rmtree(tmp_dir)
        except Exception:
            pass


# ----------------------------------------------------------------------
# GROUP F – SLOPE ANALYSIS (FINAL CORRECTED VERSION)
# ----------------------------------------------------------------------

def group_f(boundary_file, dem_file, crs, out, mapping=None,
            boundary_is_zip=False, forest_name="FOREST",
            f_mode="A", comp_col_name=None, field_area_ha=None, run_id=None):
    """
    Slope analysis with raster-to-polygon pipeline.
    Area_ha is the raw polygon area from raster-to-polygon after clipping to the compartment.
    No buffer is applied during clipping, ensuring the three classes partition the compartment exactly.
    If field_area_ha is provided, Recal_ha and Cal_Factor columns are added (Area_ha remains raw).
    Returns: (summary_rows, vector_gdf, boundary_gdf, f_mode, per_group)
    """
    if not _HAS_RASTERIO:
        raise ValueError("rasterio and scipy are not installed.")

    os.makedirs(out, exist_ok=True)
    pfx = _safe_dn(forest_name)
    SN = -9999.0

    if run_id:
        _prog(run_id, "Step 0 — Loading DEM…", 5)

    # Save uploaded DEM to disk
    dem_path = os.path.join(UPLOAD, f"{uuid.uuid4()}_dem.tif")
    try:
        dem_file.save(dem_path)
    except Exception as e:
        raise ValueError(f"Could not save DEM file: {e}")
    if not os.path.exists(dem_path) or os.path.getsize(dem_path) < 100:
        raise ValueError("DEM file is empty or could not be written to disk.")

    with rasterio.open(dem_path) as _src:
        dem_crs = _src.crs
        dem_nodata = _src.nodata
        dem_profile = _src.profile.copy()

    if run_id:
        _prog(run_id, "Step 1 — Loading & reprojecting boundary…", 10)

    # --- Load boundary polygon (either from ZIP shapefile or from CSV/Excel) ---
    if boundary_is_zip:
        ts = (mapping or {}).get("target_shp")
        bpoly, bgdf, shp_basename = _bnd_from_zip(boundary_file, ts, crs, str(dem_crs))
        if forest_name in (None, "", "FOREST"):
            forest_name = shp_basename
            pfx = _safe_dn(forest_name)
    else:
        try:
            df = read_input(boundary_file)
            bpoly = _bnd_from_df(df, mapping)
            bgdf = gpd.GeoDataFrame(
                [{"Forest": forest_name, "geometry": bpoly}], crs=crs
            ).to_crs(str(dem_crs))
            bpoly = bgdf.unary_union
        except Exception as e:
            raise ValueError(f"Failed to read/project boundary: {e}")

    if bpoly is None or bpoly.is_empty:
        raise ValueError("Boundary polygon is empty after loading.")

    # --- Prepare rectangular DEM clip (20% buffer) ---
    b = bpoly.bounds
    bx = (b[2] - b[0]) * 0.20
    by = (b[3] - b[1]) * 0.20
    rect_poly = _sbox(b[0] - bx, b[1] - by, b[2] + bx, b[3] + by)

    nd = dem_nodata if dem_nodata is not None else SN

    if run_id:
        _prog(run_id, "Step 1 — Clipping rectangular DEM (20% buffer)…", 18)

    # Clip DEM to rectangle
    try:
        with rasterio.open(dem_path) as _src:
            ra, rt = rio_mask(
                _src, [rect_poly.__geo_interface__],
                crop=True, filled=True, nodata=nd, all_touched=False
            )
        rdm = ra[0].astype(np.float32)
        rdm[rdm == nd] = np.nan
    except Exception as e:
        log.warning(f"Rect clip failed ({e}), reading full DEM")
        with rasterio.open(dem_path) as _src:
            ra = _src.read(1).astype(np.float32)
            rt = _src.transform
        nd_val = nd if nd is not None else SN
        rdm = ra.copy()
        rdm[rdm == nd_val] = np.nan

    rx = abs(rt.a)
    ry = abs(rt.e)
    if rx < 0.01 or ry < 0.01:
        raise ValueError(f"DEM pixel size is too small ({rx}m × {ry}m). Check DEM is in a projected CRS (e.g. UTM).")

    if run_id:
        _prog(run_id, "Step 2 — Computing slope (Horn method)…", 28)

    valid = ~np.isnan(rdm)
    if not valid.any():
        raise ValueError("DEM has no valid elevation data inside the boundary+buffer rectangle.")

    # Fill NoData holes using nearest neighbour
    if (~valid).any():
        ind = ndi.distance_transform_edt(
            ~valid, return_distances=False, return_indices=True
        )
        filled = rdm[tuple(ind)]
    else:
        filled = rdm.copy()

    # Slope calculation
    dzdx = ndi.sobel(filled, axis=1) / (8.0 * rx)
    dzdy = ndi.sobel(filled, axis=0) / (8.0 * ry)
    slope = np.degrees(np.arctan(np.sqrt(dzdx**2 + dzdy**2))).astype(np.float32)

    # Mask outer pixels (edge effect)
    outer_valid = np.ones_like(valid, dtype=bool)
    outer_valid[0, :] = False
    outer_valid[-1, :] = False
    outer_valid[:, 0] = False
    outer_valid[:, -1] = False
    slope[~outer_valid] = SN

    # Save rectangular slope raster
    rp = dem_profile.copy()
    rp.update(dtype="float32", nodata=SN, count=1,
              height=slope.shape[0], width=slope.shape[1], transform=rt)
    sp_path = os.path.join(out, f"{pfx}_slope_rect.tif")
    with rasterio.open(sp_path, "w", **rp) as dst:
        dst.write(slope, 1)

    if run_id:
        _prog(run_id, f"Step 2 — Slope raster saved ({slope.shape[1]}×{slope.shape[0]}px)", 35)

    # --- Reclassify ---
    if run_id:
        _prog(run_id, "Step 3 — Reclassifying slope into 3 classes…", 40)

    vm = (slope != SN) & ~np.isnan(slope)
    cls = np.zeros_like(slope, dtype=np.uint8)
    cls[vm & (slope < 19)] = 1
    cls[vm & (slope >= 19) & (slope <= 31)] = 2
    cls[vm & (slope > 31)] = 3

    cp = rp.copy()
    cp.update(dtype="uint8", nodata=0)
    cls_path = os.path.join(out, f"{pfx}_class_rect.tif")
    with rasterio.open(cls_path, "w", **cp) as dst:
        dst.write(cls, 1)

    # --- Raster to polygon ---
    if run_id:
        _prog(run_id, "Step 4 — Vectorising classified raster…", 50)

    rtp = []
    with rasterio.open(cls_path) as _src:
        ca, ct = _src.read(1), _src.transform
        mask_valid = (ca > 0).astype(np.uint8)
        for shp, val in rio_shapes(ca, mask=mask_valid, transform=ct):
            cid = int(val)
            if cid == 0:
                continue
            try:
                coords = shp["coordinates"]
                ext = coords[0]
                holes = coords[1:] if len(coords) > 1 else None
                geom = _repair(Polygon(ext, holes=holes))
                if geom and not geom.is_empty and geom.area > 1e-10:
                    rtp.append({"gridcode": cid, "geometry": geom})
            except Exception:
                continue

    if not rtp:
        raise ValueError("Raster-to-polygon produced no features. Check DEM overlap and CRS.")

    rtp_gdf = gpd.GeoDataFrame(rtp, crs=str(dem_crs))
    _rtp_save = _enforce_poly_gdf(rtp_gdf)
    if not _rtp_save.empty:
        _rtp_save.to_file(os.path.join(out, f"{pfx}_rtp_raw.shp"))

    if run_id:
        _prog(run_id, f"Step 4 — {len(rtp)} slope polygons vectorised", 56)

    # --- Dissolve by gridcode ---
    if run_id:
        _prog(run_id, "Step 5 — Dissolving by gridcode…", 62)

    try:
        dissolved = rtp_gdf.dissolve(by="gridcode", as_index=False)
        dissolved["gridcode"] = dissolved["gridcode"].astype(int)
    except Exception as e:
        log.warning(f"dissolve failed ({e}), using raw rtp")
        dissolved = rtp_gdf.copy()

    _dis_save = _enforce_poly_gdf(dissolved)
    if not _dis_save.empty:
        _dis_save.to_file(os.path.join(out, f"{pfx}_rtp_dissolved.shp"))

    if run_id:
        _prog(run_id, f"Step 5 — Dissolved into {len(dissolved)} gridcode classes", 66)

    # --- Determine compartments to clip to ---
    if run_id:
        _prog(run_id, "Step 6 — Clipping dissolved polygons to boundary…", 70)

    class_defs = {
        1: ("0-19 degree",  "Gentle",   "#2e8b57"),
        2: ("19-31 degree", "Moderate", "#ffd700"),
        3: (">31 degree",   "Steep",    "#ef4444"),
    }

    comp_polygons = []
    if f_mode == "A":
        comp_polygons = [(forest_name, bpoly)]
    else:
        # Find compartment column in bgdf
        grp_col = None
        if comp_col_name:
            for c in bgdf.columns:
                if c.lower() == comp_col_name.lower():
                    grp_col = c
                    break
        if grp_col is None:
            grp_col = _find_col(bgdf, _FA if f_mode == "B" else _CA)
        if grp_col:
            for val, grp in bgdf.groupby(grp_col):
                up = _repair(grp.unary_union)
                if up and not up.is_empty:
                    comp_polygons.append((str(val), up))
        if not comp_polygons:
            comp_polygons = [(forest_name, bpoly)]

    # --- Clip dissolved polygons to each compartment ---
    all_sum = []   # flattened list of rows (for summary Excel)
    per_grp = {}   # dict: label -> list of rows for that compartment
    all_vec = []   # list of geometry records for shapefile

    for i, (label, clip_poly) in enumerate(comp_polygons):
        if run_id:
            pct = 70 + int(16 * i / max(len(comp_polygons), 1))
            _prog(run_id, f"Step 6 — Clipping {label} ({i+1}/{len(comp_polygons)})…", pct)

        vrecs = []
        for _, drow in dissolved.iterrows():
            cid = int(drow["gridcode"])
            dgeom = _repair(drow.geometry)
            if dgeom is None or dgeom.is_empty:
                continue
            if cid not in class_defs:
                continue
            try:
                # --- FIX: removed buffer to avoid double-counting ---
                clipped = _repair(dgeom.intersection(clip_poly))
            except Exception:
                continue
            if clipped is None or clipped.is_empty:
                continue

            # Ensure polygon/multipolygon
            if clipped.geom_type == "Polygon":
                pg = clipped if clipped.is_valid else _repair(clipped)
            elif clipped.geom_type == "MultiPolygon":
                valid_parts = [p for p in clipped.geoms if p and not p.is_empty and p.area > 1e-10]
                if not valid_parts:
                    continue
                pg = MultiPolygon(valid_parts) if len(valid_parts) > 1 else valid_parts[0]
            elif hasattr(clipped, "geoms"):
                parts = []
                for g in clipped.geoms:
                    if g.geom_type == "Polygon" and not g.is_empty and g.area > 1e-10:
                        parts.append(g)
                    elif g.geom_type == "MultiPolygon":
                        parts.extend([p for p in g.geoms if not p.is_empty and p.area > 1e-10])
                if not parts:
                    continue
                pg = MultiPolygon(parts) if len(parts) > 1 else parts[0]
            else:
                pg = _as_poly(clipped)

            if pg is None or pg.is_empty or pg.area < 1e-10:
                continue

            # Close polygons
            if pg.geom_type == "Polygon":
                pg = _close_poly(pg)
            elif pg.geom_type == "MultiPolygon":
                fixed_parts = []
                for part in pg.geoms:
                    cp = _close_poly(part)
                    if cp and cp.geom_type == "Polygon" and not cp.is_empty:
                        fixed_parts.append(cp)
                if fixed_parts:
                    pg = MultiPolygon(fixed_parts) if len(fixed_parts) > 1 else fixed_parts[0]

            ah = round(pg.area / 10000, 4)
            vrecs.append({
                "Label":       label,
                "Class":       cid,
                "Slope_Range": class_defs[cid][0],
                "Description": class_defs[cid][1],
                "Area_ha":     ah,
                "geometry":    pg,
            })

        # Store raw vrecs for this compartment
        per_grp[label] = vrecs
        all_sum.extend(vrecs)
        all_vec.extend(vrecs)

    # --- Check if we got any results ---
    if not all_sum:
        raise ValueError("No slope polygons were generated. Check that the boundary overlaps the DEM and that the CRS is correct.")

    # --- Now we have all raw areas in all_sum. Apply recalibration if requested ---
    cal_factor = 1.0
    total_raw_all = sum(r["Area_ha"] for r in all_sum)
    if field_area_ha is not None and field_area_ha > 0 and total_raw_all > 1e-9:
        cal_factor = field_area_ha / total_raw_all
        # Add Recal_ha and Cal_Factor to each row (Area_ha stays raw)
        for r in all_sum:
            r["Recal_ha"] = round(r["Area_ha"] * cal_factor, 4)
            r["Cal_Factor"] = round(cal_factor, 6)
    else:
        # No recalibration: ensure these columns are absent
        for r in all_sum:
            r.pop("Recal_ha", None)
            r.pop("Cal_Factor", None)

    # --- Now rebuild the vector GeoDataFrame from all_vec (raw areas) ---
    vcrs = str(dem_crs)
    if all_vec:
        vgdf = gpd.GeoDataFrame(all_vec, crs=vcrs)
        vgdf = _enforce_poly_gdf(vgdf)
        if not vgdf.empty:
            vgdf.to_file(os.path.join(out, f"{pfx}_slope_polygon.shp"))
    else:
        vgdf = gpd.GeoDataFrame(columns=["Label","Class","Slope_Range",
                                          "Description","Area_ha","geometry"], crs=vcrs)

    # --- Save boundary shapefile ---
    try:
        bgdf_save = _enforce_poly_gdf(bgdf)
        if not bgdf_save.empty:
            bgdf_save.to_file(os.path.join(out, f"{pfx}_boundary_polygon.shp"))
    except Exception as e:
        log.warning(f"bgdf save: {e}")

    # --- Save clipped slope and class rasters for the first/main polygon (optional) ---
    try:
        main_poly = comp_polygons[0][1]
        with rasterio.open(sp_path) as _src:
            fc2, ft2 = rio_mask(
                _src, [main_poly.__geo_interface__],
                crop=True, filled=True, nodata=SN, all_touched=False
            )
        fc2 = fc2[0].astype(np.float32)
        cp2_ = rp.copy()
        cp2_.update(height=fc2.shape[0], width=fc2.shape[1], transform=ft2)
        with rasterio.open(os.path.join(out, f"{pfx}_slope_clipped.tif"), "w", **cp2_) as dst:
            dst.write(fc2, 1)
        # Reclassify clipped
        vm2 = (fc2 != SN) & ~np.isnan(fc2)
        ca2 = np.zeros_like(fc2, dtype=np.uint8)
        ca2[vm2 & (fc2 < 19)] = 1
        ca2[vm2 & (fc2 >= 19) & (fc2 <= 31)] = 2
        ca2[vm2 & (fc2 > 31)] = 3
        cp3_ = cp2_.copy()
        cp3_.update(dtype="uint8", nodata=0)
        with rasterio.open(os.path.join(out, f"{pfx}_slope_classes.tif"), "w", **cp3_) as dst:
            dst.write(ca2, 1)
    except Exception as e:
        log.warning(f"Clipped raster save error: {e}")

    # --- Prepare final DataFrame for Excel ---
    # Compute per-compartment totals for raw Area_ha (so Total_ha = sum of raw areas)
    comp_totals = {}
    for r in all_sum:
        lab = r["Label"]
        comp_totals[lab] = comp_totals.get(lab, 0) + r["Area_ha"]

    # Add Total_ha and Pct_Area based on raw areas
    for r in all_sum:
        lab = r["Label"]
        total_raw_comp = comp_totals[lab]
        r["Total_ha"] = round(total_raw_comp, 4)
        r["Pct_Area"] = round(r["Area_ha"] / total_raw_comp * 100, 2) if total_raw_comp > 0 else 0

    # Determine columns based on whether recalibration was applied
    if field_area_ha is not None and field_area_ha > 0 and total_raw_all > 1e-9:
        cols = ["Label", "Class", "Slope_Range", "Description",
                "Area_ha", "Recal_ha", "Cal_Factor", "Pct_Area", "Total_ha"]
    else:
        cols = ["Label", "Class", "Slope_Range", "Description",
                "Area_ha", "Pct_Area", "Total_ha"]

    # Create DataFrame and ensure all columns exist
    df_excel = pd.DataFrame(all_sum)
    existing_cols = [c for c in cols if c in df_excel.columns]
    df_excel = df_excel[existing_cols]

    # Save Excel
    ep = os.path.join(out, f"{pfx}_slope_summary.xlsx")
    if f_mode == "A":
        # Single group: remove Label column if it's redundant
        df_excel.drop(columns=["Label"], errors="ignore").to_excel(ep, index=False)
    else:
        # Multi-compartment: save all data in one sheet, and per-compartment sheets
        try:
            with pd.ExcelWriter(ep, engine="openpyxl") as wr:
                df_excel.to_excel(wr, sheet_name="All_Groups", index=False)
                for lb, grs in per_grp.items():
                    if not grs:
                        continue
                    # Build per-compartment DataFrame with same columns
                    comp_raw = sum(r["Area_ha"] for r in grs)
                    rows = []
                    for r in grs:
                        row = r.copy()
                        row["Total_ha"] = round(comp_raw, 4)
                        row["Pct_Area"] = round(r["Area_ha"] / comp_raw * 100, 2) if comp_raw > 0 else 0
                        if field_area_ha is not None and field_area_ha > 0 and total_raw_all > 1e-9:
                            row["Recal_ha"] = round(r["Area_ha"] * cal_factor, 4)
                            row["Cal_Factor"] = round(cal_factor, 6)
                            cols_per = ["Class", "Slope_Range", "Description",
                                        "Area_ha", "Recal_ha", "Cal_Factor",
                                        "Pct_Area", "Total_ha"]
                        else:
                            cols_per = ["Class", "Slope_Range", "Description",
                                        "Area_ha", "Pct_Area", "Total_ha"]
                        # Ensure all columns exist
                        rows.append({k: row.get(k, None) for k in cols_per})
                    df_comp = pd.DataFrame(rows)
                    df_comp = df_comp.dropna(axis=1, how='all')
                    df_comp.to_excel(wr, sheet_name=str(lb)[:31], index=False)
        except Exception as e:
            log.warning(f"Excel multi-sheet error ({e}), saving flat")
            df_excel.to_excel(ep, index=False)

    if run_id:
        _prog(run_id, "Group F complete.", 95)

    # Return summary, vector GDF, boundary GDF, mode, per-group dict
    return all_sum, vgdf, bgdf, f_mode, per_grp

# ----------------------------------------------------------------------
# GROUP G – SURVEY POINT GENERATOR
# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
# GROUP G – SURVEY POINT GENERATOR HELPERS
# ----------------------------------------------------------------------

def _g_get_poly(geom):
    if geom is None or geom.is_empty:
        return
    if geom.geom_type == "Polygon":
        yield geom
    elif geom.geom_type == "MultiPolygon":
        for p in geom.geoms:
            yield p

def _g_vertex_points(gdf, comp_col):
    records = []
    for _, row in gdf.iterrows():
        cid = str(row[comp_col]) if comp_col and comp_col in row.index else ""
        for poly in _g_get_poly(row.geometry):
            for x, y in poly.exterior.coords:
                records.append({
                    "Point_Type": "Vertex",
                    "Source": "Vertex",
                    "Compartments": cid,
                    "Easting": round(x, 3),
                    "Northing": round(y, 3),
                    "geometry": Point(x, y)
                })
    return records

def _g_boundary_points(gdf, comp_col, spacing):
    records = []
    for _, row in gdf.iterrows():
        cid = str(row[comp_col]) if comp_col and comp_col in row.index else ""
        for poly in _g_get_poly(row.geometry):
            line = poly.exterior
            total = line.length
            d = spacing
            while d < total:
                pt = line.interpolate(d)
                records.append({
                    "Point_Type": "Boundary",
                    "Source": "Boundary",
                    "Compartments": cid,
                    "Easting": round(pt.x, 3),
                    "Northing": round(pt.y, 3),
                    "geometry": pt
                })
                d += spacing
    return records

def _g_divider_points(gdf, comp_col, spacing):
    records = []
    rows = list(gdf.iterrows())
    n = len(rows)
    for i in range(n):
        _, ri = rows[i]
        ci = str(ri[comp_col]) if comp_col and comp_col in ri.index else f"C{i+1}"
        for j in range(i + 1, n):
            _, rj = rows[j]
            cj = str(rj[comp_col]) if comp_col and comp_col in rj.index else f"C{j+1}"
            try:
                if not ri.geometry.intersects(rj.geometry):
                    continue
                common = ri.geometry.boundary.intersection(rj.geometry.boundary)
            except Exception:
                continue
            lines = []
            if common.geom_type == "LineString":
                lines = [common]
            elif common.geom_type == "MultiLineString":
                lines = list(common.geoms)
            elif common.geom_type == "GeometryCollection":
                lines = [g for g in common.geoms
                         if g.geom_type in ("LineString", "MultiLineString")]
            comp_pair = ",".join(sorted([ci, cj]))
            for seg in lines:
                if seg.is_empty:
                    continue
                total = seg.length
                d = spacing
                while d < total:
                    pt = seg.interpolate(d)
                    records.append({
                        "Point_Type": "Divider",
                        "Source": "Divider",
                        "Compartments": comp_pair,
                        "Easting": round(pt.x, 3),
                        "Northing": round(pt.y, 3),
                        "geometry": pt
                    })
                    d += spacing
    return records

def _g_merge_dedup(vertex_recs, boundary_recs, divider_recs):
    PRIO = {"Divider": 3, "Vertex": 2, "Boundary": 1}
    merged = {}
    for recs in (boundary_recs, vertex_recs, divider_recs):
        for r in recs:
            key = (round(r["Easting"], 3), round(r["Northing"], 3))
            if key not in merged:
                merged[key] = r.copy()
                merged[key]["_all_sources"] = {r["Source"]}
                merged[key]["_all_comps"] = {r["Compartments"]}
                merged[key]["_prio"] = PRIO.get(r["Source"], 0)
            else:
                existing = merged[key]
                existing["_all_sources"].add(r["Source"])
                existing["_all_comps"].add(r["Compartments"])
                new_prio = PRIO.get(r["Source"], 0)
                if new_prio > existing["_prio"]:
                    existing["Point_Type"] = r["Point_Type"]
                    existing["_prio"] = new_prio
    result = []
    for rec in merged.values():
        rec["Source"] = "+".join(sorted(rec["_all_sources"]))
        comp_set = set()
        for cs in rec["_all_comps"]:
            for c in cs.split(","):
                if c:
                    comp_set.add(c.strip())
        rec["Compartments"] = ",".join(sorted(comp_set))
        result.append(rec)
    return result

def _g_assign_ids(records):
    ORDER = {"Divider": 0, "Vertex": 1, "Boundary": 2}
    records.sort(key=lambda r: (
        ORDER.get(r["Point_Type"], 9),
        r["Compartments"],
        r["Easting"],
        r["Northing"]
    ))
    for i, r in enumerate(records, 1):
        r["Point_ID"] = f"P{i:06d}"
    return records

def _g_export(records, epsg, out_dir, prefix="ForestPoints"):
    if not records:
        raise ValueError("No points generated. Check shapefile and spacing.")
    df = pd.DataFrame([{
        "Point_ID": r["Point_ID"],
        "Point_Type": r["Point_Type"],
        "Source": r["Source"],
        "Compartments": r["Compartments"],
        "Easting": r["Easting"],
        "Northing": r["Northing"]
    } for r in records])
    csv_path = os.path.join(out_dir, f"{prefix}.csv")
    df.to_csv(csv_path, index=False)
    xlsx_path = os.path.join(out_dir, f"{prefix}.xlsx")
    df.to_excel(xlsx_path, index=False)
    shp_gdf = gpd.GeoDataFrame(df,
                               geometry=[r["geometry"] for r in records],
                               crs=f"EPSG:{epsg}")
    shp_path = os.path.join(out_dir, f"{prefix}.shp")
    shp_gdf_valid = shp_gdf[shp_gdf.geometry.notna() & ~shp_gdf.geometry.is_empty].copy()
    if not shp_gdf_valid.empty:
        shp_gdf_valid.to_file(shp_path)
    return df, shp_gdf_valid

def _g_preview(shp_gdf, poly_gdf, path, safe_rect, title="Forest Survey Points", layout_state=None):
    render_map(path, poly_gdf=poly_gdf, pts_gdf=shp_gdf,
               point_label_col="Point_ID",
               safe_rect=safe_rect, layout_state=layout_state,
               title=title)

def _extract_shapefile_basename_from_zip(file_storage, target_shp=None):
    """
    Read a ZIP in memory and return the basename (without extension) of the first .shp
    or the one matching target_shp. Raises ValueError if no .shp is found.
    """
    import io, zipfile
    zip_bytes = file_storage.read()
    if len(zip_bytes) < 100:
        raise ValueError("ZIP file is empty or too small.")
    file_storage.seek(0)

    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        namelist = zf.namelist()
        shps = [n for n in namelist if n.lower().endswith('.shp')]
        if not shps:
            raise ValueError("No .shp file found inside the ZIP.")
        if target_shp:
            target_base = os.path.basename(target_shp)
            chosen = next((n for n in shps if os.path.basename(n) == target_base), None)
            if chosen is None:
                raise ValueError(f"Target shapefile '{target_shp}' not found in ZIP.")
        else:
            chosen = shps[0]
        return os.path.splitext(os.path.basename(chosen))[0]
def group_g(file_storage, dem_zone, comp_col_name, spacing, out_dir, run_id, target_shp=None, base_name="ForestPoints"):
    """
    Process a shapefile (or ZIP) of compartments, generate vertex, boundary, and divider points.
    """
    _prog(run_id, "Reading shapefile…", 5)

    # Use tempfile for ZIP extraction
    import tempfile
    import zipfile

    tmp_dir = None
    try:
        if file_storage.filename.lower().endswith(".zip"):
            # Handle ZIP
            tmp_dir = tempfile.mkdtemp(prefix="group_g_")
            zip_path = os.path.join(tmp_dir, "upload.zip")
            file_storage.save(zip_path)

            with zipfile.ZipFile(zip_path, 'r') as zf:
                zf.extractall(tmp_dir)

            # Find .shp files
            shps = []
            for root, _, files in os.walk(tmp_dir):
                for fname in files:
                    if fname.lower().endswith(".shp"):
                        shps.append(os.path.join(root, fname))

            if not shps:
                raise ValueError("No .shp file found in the uploaded ZIP.")

            # Choose target shapefile if specified
            shp_path = shps[0]
            if target_shp:
                for s in shps:
                    if os.path.basename(s) == os.path.basename(target_shp):
                        shp_path = s
                        break

            gdf = gpd.read_file(shp_path)
        else:
            # Direct shapefile upload (single .shp file) – save to temp
            tmp_dir = tempfile.mkdtemp(prefix="group_g_")
            shp_path = os.path.join(tmp_dir, file_storage.filename)
            file_storage.save(shp_path)
            gdf = gpd.read_file(shp_path)

        if gdf.empty:
            raise ValueError("Shapefile is empty or could not be read.")

        # Ensure CRS
        epsg = 32644 if str(dem_zone).strip() in ("44", "44N", "EPSG:32644") else 32645
        if gdf.crs is None:
            gdf = gdf.set_crs(f"EPSG:{epsg}")
        else:
            gdf = gdf.to_crs(f"EPSG:{epsg}")

        # Validate geometry type
        if not all(t in ("Polygon", "MultiPolygon") for t in gdf.geom_type.unique()):
            raise ValueError("Shapefile must contain only Polygon / MultiPolygon features.")

        # Detect compartment column
        comp_col = None
        if comp_col_name and comp_col_name in gdf.columns:
            comp_col = comp_col_name
        else:
            for alias in ("Comp_ID","Comp_No","comp_id","comp_no","Compartment","COMP"):
                if alias in gdf.columns:
                    comp_col = alias
                    break

        area_ha = round(gdf.geometry.area.sum() / 10000, 3)

        _prog(run_id, "Generating vertex points…", 20)
        v_recs = _g_vertex_points(gdf, comp_col)

        _prog(run_id, f"Generating boundary points (spacing={spacing}m)…", 35)
        b_recs = _g_boundary_points(gdf, comp_col, spacing)

        _prog(run_id, "Finding shared divider lines…", 50)
        d_recs = _g_divider_points(gdf, comp_col, spacing)

        _prog(run_id, f"Merging {len(v_recs)+len(b_recs)+len(d_recs)} raw records…", 60)
        all_recs = _g_merge_dedup(v_recs, b_recs, d_recs)

        _prog(run_id, f"Deduplicated to {len(all_recs)} unique points — assigning IDs…", 70)
        all_recs = _g_assign_ids(all_recs)

        _prog(run_id, f"Exporting {len(all_recs)} points → SHP / CSV / XLSX…", 80)
        df, shp_gdf = _g_export(all_recs, epsg, out_dir, prefix=base_name)

        summary = {
            "total": len(all_recs),
            "vertex": sum(1 for r in all_recs if r["Point_Type"] == "Vertex"),
            "boundary": sum(1 for r in all_recs if r["Point_Type"] == "Boundary"),
            "divider": sum(1 for r in all_recs if r["Point_Type"] == "Divider"),
            "area_ha": area_ha,
            "epsg": epsg,
            "spacing": spacing,
            "comp_col": comp_col or "—",
            "compartments": int(len(gdf)),
        }
        return df, shp_gdf, gdf, summary

    finally:
        if tmp_dir and os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir, ignore_errors=True)
# ----------------------------------------------------------------------
# GROUP H – SAMPLE POINT BASED GIS MAPS
# ----------------------------------------------------------------------

def _validate_zip_components(zip_path, required=[".shp", ".dbf", ".shx", ".prj"]):
    with zipfile.ZipFile(zip_path, 'r') as z:
        names = z.namelist()
        missing = [ext for ext in required if not any(n.endswith(ext) for n in names)]
        if missing:
            raise ValueError(f"ZIP missing required components: {', '.join(missing)}")
    return True

def _extract_shp(zip_path, target_crs=None):
    _validate_zip_components(zip_path)
    tmp = tempfile.mkdtemp()
    try:
        with zipfile.ZipFile(zip_path, 'r') as z:
            z.extractall(tmp)
        shp_files = [os.path.join(tmp, f) for f in os.listdir(tmp) if f.endswith('.shp')]
        if not shp_files:
            raise ValueError("No .shp file found in ZIP.")
        gdf = gpd.read_file(shp_files[0])
        if gdf.crs is None:
            raise ValueError("Shapefile has no CRS. Please assign one.")
        if target_crs is not None and str(gdf.crs) != target_crs:
            gdf = gdf.to_crs(target_crs)
        gdf.geometry = gdf.geometry.buffer(0)
        gdf = gdf[~gdf.geometry.is_empty]
        return gdf
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

def _read_points(file_path, crs):
    if file_path.endswith('.zip'):
        gdf = _extract_shp(file_path, target_crs=crs)
        if not all(gdf.geom_type.isin(['Point', 'MultiPoint'])):
            raise ValueError("Shapefile must contain Point geometries.")
        return gdf
    else:
        df = pd.read_excel(file_path)
        if 'Latitude' in df.columns and 'Longitude' in df.columns:
            gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.Longitude, df.Latitude), crs="EPSG:4326")
            gdf = gdf.to_crs(crs)
        elif 'Easting' in df.columns and 'Northing' in df.columns:
            gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.Easting, df.Northing), crs=crs)
        else:
            raise ValueError("Excel must contain 'Latitude/Longitude' or 'Easting/Northing' columns.")
        if 'Point_ID' not in df.columns:
            raise ValueError("Excel must contain 'Point_ID' column.")
        if df['Point_ID'].duplicated().any():
            raise ValueError("Duplicate Point_ID values found.")
        if gdf.geometry.is_empty.any():
            raise ValueError("Some points have empty geometry (NaN coordinates).")
        return gdf

def _clip_to_boundary(gdf, boundary_gdf):
    if gdf.geom_type.iloc[0] in ['Point', 'MultiPoint']:
        boundary_union = boundary_gdf.unary_union
        gdf = gdf[gdf.geometry.within(boundary_union)]
    else:
        gdf = gpd.overlay(gdf, boundary_gdf, how='intersection')
    return gdf

def _compute_slope_raster(dem_path, boundary_gdf, out_dir):
    import scipy.ndimage as ndi
    with rasterio.open(dem_path) as src:
        out_image, out_transform = rio_mask(src, boundary_gdf.geometry, crop=True)
        dem = out_image[0].astype(np.float32)
        nodata = src.nodata or -9999
        dem[dem == nodata] = np.nan
        rx = abs(out_transform[0])
        ry = abs(out_transform[4])
        valid = ~np.isnan(dem)
        if not valid.any():
            raise ValueError("No valid DEM data inside boundary.")
        if (~valid).any():
            ind = ndi.distance_transform_edt(~valid, return_distances=False, return_indices=True)
            filled = dem[tuple(ind)]
        else:
            filled = dem
        dzdx = ndi.sobel(filled, axis=1) / (8.0 * rx)
        dzdy = ndi.sobel(filled, axis=0) / (8.0 * ry)
        slope = np.degrees(np.arctan(np.sqrt(dzdx**2 + dzdy**2)))
        classes = np.zeros_like(slope, dtype=np.uint8)
        classes[(slope < 19) & ~np.isnan(slope)] = 1
        classes[(slope >= 19) & (slope <= 31) & ~np.isnan(slope)] = 2
        classes[(slope > 31) & ~np.isnan(slope)] = 3
        out_meta = src.meta.copy()
        out_meta.update({
            "height": classes.shape[0],
            "width": classes.shape[1],
            "transform": out_transform,
            "dtype": "uint8",
            "nodata": 0
        })
        out_path = os.path.join(out_dir, "slope_classified.tif")
        with rasterio.open(out_path, "w", **out_meta) as dst:
            dst.write(classes, 1)
    return out_path

def _raster_to_polygons(raster_path, value_map=None):
    from rasterio.features import shapes
    from shapely.geometry import shape
    with rasterio.open(raster_path) as src:
        image = src.read(1)
        transform = src.transform
        results = []
        for geom, val in shapes(image, transform=transform, connectivity=8):
            if val == 0:
                continue
            results.append({
                "class": int(val),
                "geometry": shape(geom)
            })
    gdf = gpd.GeoDataFrame(results, crs=src.crs)
    gdf = gdf[gdf.geometry.area > 1e-6]
    return gdf

def _draw_slope_table_compact(ax, slope_areas):
    table_data = []
    for cls, info in SLOPE_CLASSES.items():
        area = slope_areas.get(info['range'], 0)
        table_data.append([info['range'], f"{area:.2f} ha"])
    total = sum(slope_areas.values())
    table_data.append(["Total", f"{total:.2f} ha"])
    table = ax.table(cellText=table_data, colLabels=["Slope", "Area"],
                     loc='lower left', bbox=[0.02, 0.02, 0.3, 0.15],
                     cellLoc='center', colWidths=[0.15, 0.10])
    table.auto_set_font_size(False)
    table.set_fontsize(7)
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight='bold', color='white')
            cell.set_facecolor('#1a5276')
        else:
            cell.set_facecolor('#f8f9fa')

def process_group_h(boundary_zip, compartments_zip, dem_file, satellite_file,
                    sample_points_file, survey_points_file=None,
                    crs="EPSG:32644", out_dir=None, run_id=None):
    if out_dir is None:
        out_dir = os.path.join(OUTPUT, run_id or str(uuid.uuid4()))
    os.makedirs(out_dir, exist_ok=True)

    _prog(run_id, "Loading boundary...", 10)
    boundary_gdf = _extract_shp(boundary_zip, target_crs=crs)
    if len(boundary_gdf) != 1:
        raise ValueError("Boundary must contain exactly one polygon.")
    if boundary_gdf.geom_type.iloc[0] not in ['Polygon', 'MultiPolygon']:
        raise ValueError("Boundary must be a polygon.")
    boundary_gdf = boundary_gdf.buffer(0)
    boundary_gdf = boundary_gdf[~boundary_gdf.is_empty]

    _prog(run_id, "Loading compartments...", 15)
    compartments_gdf = _extract_shp(compartments_zip, target_crs=crs)
    compartments_gdf = _clip_to_boundary(compartments_gdf, boundary_gdf)
    if compartments_gdf.empty:
        raise ValueError("No compartments inside boundary after clipping.")
    if 'Comp_ID' not in compartments_gdf.columns:
        compartments_gdf['Comp_ID'] = [f"C{i+1:03d}" for i in range(len(compartments_gdf))]

    _prog(run_id, "Processing DEM...", 25)
    with rasterio.open(dem_file) as src:
        dem_bounds = box(*src.bounds)
    boundary_bounds = boundary_gdf.unary_union.bounds
    if not dem_bounds.intersects(box(*boundary_bounds)):
        raise ValueError("DEM does not overlap the boundary.")
    slope_raster = _compute_slope_raster(dem_file, boundary_gdf, out_dir)
    slope_poly = _raster_to_polygons(slope_raster)
    slope_poly = _clip_to_boundary(slope_poly, boundary_gdf)
    slope_areas = {}
    for cls, info in SLOPE_CLASSES.items():
        sub = slope_poly[slope_poly['class'] == cls]
        slope_areas[info['range']] = sub.geometry.area.sum() / 10000

    _prog(run_id, "Processing satellite image...", 35)
    sat_clip_path = os.path.join(out_dir, "satellite_clipped.tif")
    with rasterio.open(satellite_file) as src:
        out_image, out_transform = rio_mask(src, boundary_gdf.geometry, crop=True)
        out_meta = src.meta.copy()
        out_meta.update({
            "height": out_image.shape[1],
            "width": out_image.shape[2],
            "transform": out_transform
        })
        with rasterio.open(sat_clip_path, "w", **out_meta) as dst:
            dst.write(out_image)

    _prog(run_id, "Loading sample points...", 45)
    sample_gdf = _read_points(sample_points_file, crs)
    sample_gdf = _clip_to_boundary(sample_gdf, boundary_gdf)
    if sample_gdf.empty:
        raise ValueError("No sample points inside boundary after clipping.")

    _prog(run_id, "Loading survey points...", 50)
    if survey_points_file:
        survey_gdf = _read_points(survey_points_file, crs)
        survey_gdf = _clip_to_boundary(survey_gdf, boundary_gdf)
    else:
        survey_gdf = None

    _prog(run_id, "Generating maps...", 55)

    def create_figure():
        fig = plt.figure(figsize=(10, 7.5), dpi=DPI)
        fig.patch.set_facecolor('white')
        ax = fig.add_subplot(111)
        ax.set_facecolor('white')
        ax.set_aspect('equal')
        return fig, ax

    # Map 1: Slope Map
    fig1, ax1 = create_figure()
    for cls, info in SLOPE_CLASSES.items():
        sub = slope_poly[slope_poly['class'] == cls]
        if not sub.empty:
            sub.plot(ax=ax1, facecolor=info['color'], edgecolor='black', linewidth=0.2, label=info['range'])
    boundary_gdf.boundary.plot(ax=ax1, color='black', linewidth=2)
    compartments_gdf.plot(ax=ax1, facecolor='none', edgecolor='gray', linewidth=0.5)
    for _, row in compartments_gdf.iterrows():
        ax1.annotate(row['Comp_ID'], xy=(row.geometry.centroid.x, row.geometry.centroid.y),
                     ha='center', va='center', fontsize=6, color='black')
    handles = [mpatches.Patch(facecolor=info['color'], label=info['range']) for cls, info in SLOPE_CLASSES.items()]
    _add_north_arrow(fig1)
    _add_scale_bar(fig1)
    _draw_slope_table_compact(ax1, slope_areas)
    ax1.set_title("Slope Map", fontsize=14, weight='bold')
    fig1.savefig(os.path.join(out_dir, "Slope_Map.png"), dpi=DPI, bbox_inches='tight')
    fig1.savefig(os.path.join(out_dir, "Slope_Map.pdf"), bbox_inches='tight')
    fig1.savefig(os.path.join(out_dir, "Slope_Map.svg"), bbox_inches='tight')
    plt.close(fig1)
    _prog(run_id, "Slope Map generated.", 60)

    # Map 2: Satellite Map
    fig2, ax2 = create_figure()
    with rasterio.open(sat_clip_path) as src:
        show(src, ax=ax2, title='')
    boundary_gdf.boundary.plot(ax=ax2, color='yellow', linewidth=2)
    compartments_gdf.plot(ax=ax2, facecolor='none', edgecolor='white', linewidth=0.5)
    for _, row in compartments_gdf.iterrows():
        ax2.annotate(row['Comp_ID'], xy=(row.geometry.centroid.x, row.geometry.centroid.y),
                     ha='center', va='center', fontsize=6, color='white')
    _add_north_arrow(fig2)
    _add_scale_bar(fig2)
    ax2.set_title("Satellite Map", fontsize=14, weight='bold')
    fig2.savefig(os.path.join(out_dir, "Satellite_Map.png"), dpi=DPI, bbox_inches='tight')
    fig2.savefig(os.path.join(out_dir, "Satellite_Map.pdf"), bbox_inches='tight')
    fig2.savefig(os.path.join(out_dir, "Satellite_Map.svg"), bbox_inches='tight')
    plt.close(fig2)
    _prog(run_id, "Satellite Map generated.", 68)

    # Map 3: Sub-compartment Map
    fig3, ax3 = create_figure()
    import random
    random.seed(42)
    comps = compartments_gdf.copy()
    comps['color'] = ["#" + ''.join(random.choices('0123456789ABCDEF', k=6)) for _ in range(len(comps))]
    comps.plot(ax=ax3, facecolor=comps['color'], edgecolor='black', linewidth=0.5)
    boundary_gdf.boundary.plot(ax=ax3, color='black', linewidth=2)
    for _, row in comps.iterrows():
        area_ha = row.geometry.area / 10000
        ax3.annotate(f"{row['Comp_ID']}\n{area_ha:.1f}ha",
                     xy=(row.geometry.centroid.x, row.geometry.centroid.y),
                     ha='center', va='center', fontsize=6, color='black')
    handles = [mpatches.Patch(facecolor=row['color'], label=row['Comp_ID']) for _, row in comps.iterrows()]
    _add_north_arrow(fig3)
    _add_scale_bar(fig3)
    ax3.legend(handles=handles, title="Compartments", loc='lower right')
    ax3.set_title("Sub-compartment Map", fontsize=14, weight='bold')
    fig3.savefig(os.path.join(out_dir, "SubCompartment_Map.png"), dpi=DPI, bbox_inches='tight')
    fig3.savefig(os.path.join(out_dir, "SubCompartment_Map.pdf"), bbox_inches='tight')
    fig3.savefig(os.path.join(out_dir, "SubCompartment_Map.svg"), bbox_inches='tight')
    plt.close(fig3)
    _prog(run_id, "Sub-compartment Map generated.", 76)

    # Map 4: Sample Plot Map
    fig4, ax4 = create_figure()
    boundary_gdf.boundary.plot(ax=ax4, color='black', linewidth=2)
    compartments_gdf.plot(ax=ax4, facecolor='none', edgecolor='gray', linewidth=0.5)
    sample_gdf.plot(ax=ax4, color='red', markersize=30, marker='o')
    for _, row in sample_gdf.iterrows():
        ax4.annotate(row['Point_ID'], xy=(row.geometry.x, row.geometry.y),
                     xytext=(5, 5), textcoords='offset points', fontsize=6, color='red')
    _add_north_arrow(fig4)
    _add_scale_bar(fig4)
    handles = [mpatches.Patch(facecolor='red', label='Sample Plots')]
    ax4.legend(handles=handles, loc='lower right')
    ax4.set_title("Sample Plot Map", fontsize=14, weight='bold')
    fig4.savefig(os.path.join(out_dir, "SamplePlot_Map.png"), dpi=DPI, bbox_inches='tight')
    fig4.savefig(os.path.join(out_dir, "SamplePlot_Map.pdf"), bbox_inches='tight')
    fig4.savefig(os.path.join(out_dir, "SamplePlot_Map.svg"), bbox_inches='tight')
    plt.close(fig4)
    _prog(run_id, "Sample Plot Map generated.", 84)

    # Map 5: Boundary Survey Point Map
    fig5, ax5 = create_figure()
    boundary_gdf.boundary.plot(ax=ax5, color='black', linewidth=2)
    compartments_gdf.plot(ax=ax5, facecolor='none', edgecolor='gray', linewidth=0.5)
    if survey_gdf is not None:
        survey_gdf.plot(ax=ax5, color='blue', markersize=20, marker='^')
        for _, row in survey_gdf.iterrows():
            ax5.annotate(row['Point_ID'], xy=(row.geometry.x, row.geometry.y),
                         xytext=(5, 5), textcoords='offset points', fontsize=6, color='blue')
    _add_north_arrow(fig5)
    _add_scale_bar(fig5)
    handles = [mpatches.Patch(facecolor='blue', label='Survey Points')]
    ax5.legend(handles=handles, loc='lower right')
    ax5.set_title("Boundary Survey Point Map", fontsize=14, weight='bold')
    fig5.savefig(os.path.join(out_dir, "BoundarySurveyPoint_Map.png"), dpi=DPI, bbox_inches='tight')
    fig5.savefig(os.path.join(out_dir, "BoundarySurveyPoint_Map.pdf"), bbox_inches='tight')
    fig5.savefig(os.path.join(out_dir, "BoundarySurveyPoint_Map.svg"), bbox_inches='tight')
    plt.close(fig5)
    _prog(run_id, "Boundary Survey Point Map generated.", 92)

    # Map 6: Survey Point Map (no compartments)
    fig6, ax6 = create_figure()
    boundary_gdf.boundary.plot(ax=ax6, color='black', linewidth=2)
    if survey_gdf is not None:
        survey_gdf.plot(ax=ax6, color='blue', markersize=20, marker='^')
        for _, row in survey_gdf.iterrows():
            ax6.annotate(row['Point_ID'], xy=(row.geometry.x, row.geometry.y),
                         xytext=(5, 5), textcoords='offset points', fontsize=6, color='blue')
    _add_north_arrow(fig6)
    _add_scale_bar(fig6)
    handles = [mpatches.Patch(facecolor='blue', label='Survey Points')]
    ax6.legend(handles=handles, loc='lower right')
    ax6.set_title("Survey Point Map", fontsize=14, weight='bold')
    fig6.savefig(os.path.join(out_dir, "SurveyPoint_Map.png"), dpi=DPI, bbox_inches='tight')
    fig6.savefig(os.path.join(out_dir, "SurveyPoint_Map.pdf"), bbox_inches='tight')
    fig6.savefig(os.path.join(out_dir, "SurveyPoint_Map.svg"), bbox_inches='tight')
    plt.close(fig6)
    _prog(run_id, "Survey Point Map generated.", 98)

    zip_path = os.path.join(out_dir, "GroupH_Maps.zip")
    with zipfile.ZipFile(zip_path, 'w') as z:
        for fname in os.listdir(out_dir):
            if fname.endswith(('.png', '.pdf', '.svg')):
                z.write(os.path.join(out_dir, fname), fname)

    _prog(run_id, "All maps generated and packaged.", 100)
    return zip_path, out_dir

# ----------------------------------------------------------------------
# ROUTES
# ----------------------------------------------------------------------

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/login", methods=["POST"])
@_rate_limit(limit=20, window=60)
def login():
    data = request.get_json(silent=True) or {}
    try:
        username = _validate_username(data.get("username", ""))
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    users = _lu()
    is_new = username not in users

    if is_new:
        try:
            user, _tok = _register_user(username)
        except ValueError as e:
            return jsonify({"error": str(e), "taken": True}), 409
        except Exception as e:
            log.error(f"Registration error {username!r}: {e}")
            return jsonify({"error": "Registration failed. Please try again."}), 500
    else:
        try:
            user = _login_existing(username)
        except KeyError:
            try:
                user, _tok = _register_user(username)
            except Exception as e:
                return jsonify({"error": str(e)}), 500
        except Exception as e:
            log.error(f"Login error {username!r}: {e}")
            return jsonify({"error": "Login failed. Please try again."}), 500

    session["username"] = username
    session.permanent = True
    ip = _get_client_ip()
    log.info(f"{'Register' if is_new else 'Login'}: {username!r} from {ip}")

    return jsonify({
        "ok": True,
        "username": username,
        "runs": user.get("runs", [])[-20:],
        "is_new": is_new,
        "message": "Welcome!" if is_new else f"Welcome back, {username}!"
    })

@app.route("/logout", methods=["POST"])
def logout():
    username = session.get("username")
    _logout_user(username)
    session.clear()
    if username:
        log.info(f"Logout: {username!r} from {_get_client_ip()}")
    return jsonify({"ok": True})

@app.route("/me")
@_rate_limit(limit=60, window=60)
@_login_required
def me():
    u = _require_login()
    users = _lu()
    user = users.get(u, {})
    return jsonify({"username": u, "runs": user.get("runs", [])[-20:]})

@app.route("/history")
@_rate_limit(limit=30, window=60)
@_login_required
def history():
    u = _require_login()
    users = _lu()
    user = users.get(u, {})
    return jsonify({"runs": user.get("runs", [])})

@app.route("/progress/<run_id>")
@_rate_limit(limit=120, window=60)
def progress_stream(run_id):
    try:
        run_id = _safe_runid(run_id)
    except:
        pass
    def gen():
        sent = 0
        last_heartbeat = time.time()
        while True:
            msgs = _PROG.get(run_id, [])
            new = msgs[sent:]
            if new:
                for m in new:
                    yield f"data: {m}\n\n"
                sent += len(new)
                try:
                    if json.loads(msgs[-1]).get("pct", 0) >= 100:
                        return
                except:
                    pass
            else:
                now = time.time()
                if now - last_heartbeat > 3:
                    yield f": heartbeat\n\n"
                    last_heartbeat = now
            time.sleep(0.1)
    return Response(
        stream_with_context(gen()),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-store",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
            "Transfer-Encoding": "chunked",
        }
    )

@app.route("/geojson/<run_id>")
@_rate_limit(limit=60, window=60)
def get_geojson(run_id):
    run_id = _safe_runid(run_id)
    folder = _safe_path(OUTPUT, run_id)
    if not os.path.exists(folder):
        return jsonify({"type": "FeatureCollection", "features": []}), 200
    shps = []
    for r, _, fs in os.walk(folder):
        for f in fs:
            if not f.endswith(".shp"): continue
            fl = f.lower()
            if "polygon" in fl or "forestpoints" in fl or "rtp" in fl or "point" in fl or "line" in fl:
                shps.append(os.path.join(r, f))
    if not shps:
        shps = [os.path.join(r, f) for r, _, fs in os.walk(folder)
                for f in fs if f.endswith(".shp")]
    gdfs = []
    for shp in shps:
        try:
            g = gpd.read_file(shp)
            if g.crs is not None: g = g.to_crs("EPSG:4326")
            keep = [c for c in g.columns if c in
                    ("Comp_ID","Forest","Class","Slope_Range","Area_ha",
                     "Point_ID","Point_Type","Source","Compartments",
                     "Easting","Northing","geometry")]
            g = g[[c for c in keep if c in g.columns]]
            gdfs.append(g)
        except: pass
    if not gdfs:
        return jsonify({"type": "FeatureCollection", "features": []}), 200
    combined = gpd.GeoDataFrame(pd.concat(gdfs, ignore_index=True), crs="EPSG:4326")
    return Response(combined.to_json(), mimetype="application/json")

@app.route("/compose/<run_id>", methods=["POST"])
@_rate_limit(limit=20, window=60)
def compose_map(run_id):
    run_id = _safe_runid(run_id)
    folder = os.path.join(OUTPUT, run_id)
    if not os.path.exists(folder):
        return jsonify({"error": "Run not found"}), 404
    try:
        data = request.get_json(silent=True) or {}
        layout_state = data.get("layout_state")
        if not layout_state:
            layout_state = get_default_layout_state()

        shps = [os.path.join(r, f) for r, _, fs in os.walk(folder)
                for f in fs if f.endswith("_polygon.shp")]
        if not shps:
            return jsonify({"error": "No shapefiles found."}), 400
        gdfs = []; crs0 = None
        for shp in shps:
            try:
                g = gpd.read_file(shp)
                crs0 = crs0 or g.crs
                gdfs.append(g)
            except:
                pass
        if not gdfs:
            return jsonify({"error": "Could not read shapefiles."}), 400
        pg = gpd.GeoDataFrame(pd.concat(gdfs, ignore_index=True), crs=crs0)

        bounds = pg.total_bounds
        if bounds is not None and len(bounds) == 4:
            w = bounds[2] - bounds[0]
            h = bounds[3] - bounds[1]
            poly_aspect = w / h if h > 0 else 1.0
        else:
            poly_aspect = 1.0

        safe_rect = compute_safe_rect(layout_state, poly_aspect)
        pp = os.path.join(folder, "output.png")
        if "Comp_ID" in pg.columns:
            render_map(pp, poly_gdf=pg, label_col="Comp_ID",
                       safe_rect=safe_rect, layout_state=layout_state)
        else:
            render_map(pp, poly_gdf=pg, safe_rect=safe_rect, layout_state=layout_state)
        return jsonify({"ok": True, "png": f"/outputs/{run_id}/output.png?t={uuid.uuid4().hex[:8]}"})
    except Exception as e:
        return jsonify({"error": f"Compose error: {e}"}), 500
@app.route("/save_edit/<run_id>", methods=["POST"])
@_rate_limit(limit=20, window=60)
def save_edit(run_id):
    run_id = _safe_runid(run_id)
    folder = os.path.join(OUTPUT, run_id)
    if not os.path.exists(folder):
        return jsonify({"error": "Run not found"}), 404
    try:
        data = request.get_json(silent=True) or {}
        geojson = data.get("geojson")
        if not geojson:
            return jsonify({"error": "No GeoJSON provided."}), 400

        # Convert to GeoDataFrame
        gdf_new = gpd.GeoDataFrame.from_features(geojson.get("features", []), crs="EPSG:4326")

        # Find original CRS from existing polygon shapefile
        poly_shps = [os.path.join(r, f) for r, _, fs in os.walk(folder)
                     for f in fs if f.endswith("_polygon.shp")]
        point_shps = [os.path.join(r, f) for r, _, fs in os.walk(folder)
                      for f in fs if f.endswith("_point.shp")]

        orig_crs = "EPSG:32644"
        if poly_shps:
            try:
                g0 = gpd.read_file(poly_shps[0])
                if g0.crs:
                    orig_crs = str(g0.crs)
            except:
                pass

        # Reproject to original CRS
        gdf_new = gdf_new.to_crs(orig_crs)

        # Repair geometries
        gdf_new["geometry"] = [_close_poly(_repair(g)) if g else None for g in gdf_new.geometry]
        gdf_new = gdf_new[gdf_new.geometry.notna()]

        # Separate polygons and points
        poly_gdf = gdf_new[gdf_new.geometry.geom_type.isin(['Polygon', 'MultiPolygon'])].copy()
        point_gdf = gdf_new[gdf_new.geometry.geom_type.isin(['Point', 'MultiPoint'])].copy()

        # --- Handle Polygons ---
        if not poly_gdf.empty:
            # Ensure area column
            if "Area_ha" not in poly_gdf.columns:
                poly_gdf["Area_ha"] = [round(g.area/10000, 4) if g else 0 for g in poly_gdf.geometry]
            # Save polygon shapefile
            shp_path = os.path.join(folder, "edited_polygon.shp")
            poly_gdf.to_file(shp_path)
            # Overwrite existing polygon shapefile
            for shp in poly_shps:
                if os.path.basename(shp).endswith("_polygon.shp"):
                    shutil.copyfile(shp_path, shp)
                    break
        else:
            # If no polygons, keep existing (should not happen for Group C)
            pass

        # --- Handle Points ---
        if not point_gdf.empty:
            # Ensure point attributes: SN (or Point_ID) and coordinates
            # If Point_ID column exists, use it; otherwise create SN
            if "SN" not in point_gdf.columns:
                point_gdf["SN"] = range(1, len(point_gdf) + 1)
            # Ensure X and Y columns (from geometry)
            point_gdf["X"] = point_gdf.geometry.x
            point_gdf["Y"] = point_gdf.geometry.y

            # Save point shapefile
            point_shp_path = os.path.join(folder, "edited_point.shp")
            point_gdf.to_file(point_shp_path)
            # Overwrite existing point shapefile
            for shp in point_shps:
                if os.path.basename(shp).endswith("_point.shp"):
                    shutil.copyfile(point_shp_path, shp)
                    break

            # Regenerate Excel file
            excel_path = None
            for root, _, files in os.walk(folder):
                for f in files:
                    if f.endswith("_sampleplot.xlsx"):
                        excel_path = os.path.join(root, f)
                        break
                if excel_path:
                    break
            if excel_path:
                # Create DataFrame from point_gdf
                df = point_gdf[["SN", "X", "Y"]].copy()
                df.to_excel(excel_path, index=False)
        else:
            # If no points (all deleted), we might remove the point files or keep empty.
            # For safety, we can write an empty point shapefile and Excel.
            # But we'll skip for now – user likely wants to keep at least some points.
            pass

        # --- Re‑render the map ---
        # Load the updated polygon and point layers
        updated_poly = gpd.read_file(poly_shps[0]) if poly_shps else None
        updated_point = gpd.read_file(point_shps[0]) if point_shps else None
        # Line layer is not editable, we can load it from existing file
        line_shps = [os.path.join(r, f) for r, _, fs in os.walk(folder)
                     for f in fs if f.endswith("_line.shp")]
        line_gdf = gpd.read_file(line_shps[0]) if line_shps else None

        layout_state = data.get("layout_state", get_default_layout_state())
        if updated_poly is not None and not updated_poly.empty:
            bounds = updated_poly.total_bounds
        else:
            bounds = (0, 0, 1, 1)
        if bounds is not None and len(bounds) == 4:
            w = bounds[2] - bounds[0]
            h = bounds[3] - bounds[1]
            poly_aspect = w / h if h > 0 else 1.0
        else:
            poly_aspect = 1.0
        safe_rect = compute_safe_rect(layout_state, poly_aspect)

        pp = os.path.join(folder, "output.png")
        # Determine label column: for Group C, we use "SN"
        label_col = "SN"
        render_map(pp, poly_gdf=updated_poly, line_gdf=line_gdf, pts_gdf=updated_point,
                   label_col=label_col, point_label_col=label_col,
                   safe_rect=safe_rect, layout_state=layout_state)

        return jsonify({"ok": True, "png": f"/outputs/{run_id}/output.png?t={uuid.uuid4().hex[:8]}"})
    except Exception as e:
        log.error(f"Save edit error: {traceback.format_exc()}")
        return jsonify({"error": f"Edit error: {e}\n{traceback.format_exc()}"}), 500

@app.route("/download/<run_id>")
def download(run_id):
    run_id = _safe_runid(run_id)
    folder = os.path.join(OUTPUT, run_id)
    if not os.path.exists(folder):
        return jsonify({"error": "Run not found"}), 404
    zip_path = os.path.join(folder, "..", f"{run_id}.zip")
    shutil.make_archive(folder, "zip", folder)
    return send_file(zip_path, as_attachment=True)

@app.route("/outputs/<run_id>/<path:filename>")
@_rate_limit(limit=120, window=60)
def serve_output(run_id, filename):
    folder = os.path.join(OUTPUT, run_id)
    if not os.path.exists(os.path.join(folder, filename)):
        return jsonify({"error": "File not found"}), 404
    return send_from_directory(folder, filename)

# ----------------------------------------------------------------------
# DEM Routes
# ----------------------------------------------------------------------

@app.route("/dem_catalog")
@_rate_limit(limit=30, window=60)
def dem_catalog():
    import urllib.request as _ur
    import json as _json
    github_api_urls = []
    for zone in ("44N", "45N"):
        api_base = os.environ.get("GITHUB_API_DEM",
            "https://api.github.com/repos/1ravikafle-glitch/ElfakGISProStudio/contents/dem_catalog")
        github_api_urls.append((zone, f"{api_base}/{zone}"))
    files = []
    for zone, api_url in github_api_urls:
        try:
            hdrs = {"User-Agent": "elfak-gis-app", "Accept": "application/vnd.github+json"}
            if GITHUB_TOKEN:
                hdrs["Authorization"] = f"Bearer {GITHUB_TOKEN}"
            req = _ur.Request(api_url, headers=hdrs)
            with _ur.urlopen(req, timeout=8) as resp:
                items = _json.loads(resp.read())
            for item in items:
                name = item.get("name", "")
                if not name.lower().endswith((".tif", ".tiff")):
                    continue
                size_mb = round(item.get("size", 0) / (1024*1024), 1)
                files.append({
                    "name": name,
                    "zone": zone,
                    "path": f"{zone}/{name}",
                    "size_mb": size_mb,
                    "url": item.get("url", api_url + "/" + name),
                })
        except Exception as e:
            log.warning(f"GitHub API {zone} failed: {e}")
            local = os.path.join(DEM_CATALOG_DIR, zone)
            if os.path.isdir(local):
                for f in os.listdir(local):
                    if f.lower().endswith((".tif", ".tiff")):
                        fp = os.path.join(local, f)
                        files.append({
                            "name": f,
                            "zone": zone,
                            "path": f"{zone}/{f}",
                            "size_mb": round(os.path.getsize(fp)/(1024*1024), 1),
                            "url": "",
                        })
    files.sort(key=lambda x: (x["zone"], x["name"]))
    return jsonify({"files": files, "source": "github"})

@app.route("/dem_fetch", methods=["POST"])
@_rate_limit(limit=5, window=60)
def dem_fetch():
    import urllib.request as _ur
    import urllib.error as _ue
    data = request.get_json(silent=True) or {}
    url = data.get("url", "").strip()
    path = data.get("path", "").strip()
    if not url and not path:
        return jsonify({"error": "No DEM URL or path provided."}), 400
    if url and not (url.startswith("https://raw.githubusercontent.com/") or
                    url.startswith("https://github.com/") or
                    url.startswith("https://api.github.com/")):
        return jsonify({"error": "DEM URL must be from GitHub."}), 400
    safe_name = re.sub(r"[^A-Za-z0-9._-]", "_", os.path.basename(path or url))[:120]
    cache_key = hashlib.sha256((url or path).encode()).hexdigest()[:16]
    local_path = os.path.join(DEM_CACHE_DIR, f"{cache_key}_{safe_name}")
    if os.path.exists(local_path) and os.path.getsize(local_path) > 1000:
        size_mb = round(os.path.getsize(local_path) / (1024*1024), 1)
        log.info(f"DEM cache hit: {safe_name} ({size_mb}MB)")
        return jsonify({"ok": True, "cache_key": cache_key, "local": local_path,
                        "size_mb": size_mb, "cached": True})
    candidates = []
    if url:
        candidates.append(url)
    if path:
        owner_repo = "1ravikafle-glitch/ElfakGISProStudio"
        enc_path = "/".join(urllib.parse.quote(seg) for seg in path.split("/"))
        for branch in ("main", "master"):
            candidates.append(
                f"https://raw.githubusercontent.com/{owner_repo}/{branch}/dem_catalog/{enc_path}"
            )
    seen = set()
    candidates = [c for c in candidates if not (c in seen or seen.add(c))]
    if not candidates:
        return jsonify({"error": "Could not build a download URL from the given path."}), 400
    last_error = None
    for candidate_url in candidates:
        try:
            req = _ur.Request(candidate_url, headers={"User-Agent": "elfak-gis-app"})
            with _ur.urlopen(req, timeout=120) as resp, open(local_path, "wb") as fout:
                total = 0
                while True:
                    chunk = resp.read(1024*1024)
                    if not chunk:
                        break
                    fout.write(chunk)
                    total += len(chunk)
            if total < 500:
                os.remove(local_path)
                last_error = f"Response too small ({total} bytes)"
                continue
            size_mb = round(os.path.getsize(local_path) / (1024*1024), 1)
            log.info(f"DEM downloaded: {safe_name} ({size_mb}MB)")
            return jsonify({"ok": True, "cache_key": cache_key, "local": local_path,
                            "size_mb": size_mb, "cached": False, "source_url": candidate_url})
        except Exception as e:
            last_error = f"{type(e).__name__}: {e}"
            if os.path.exists(local_path):
                os.remove(local_path)
            continue
    return jsonify({
        "error": f"Failed to download DEM. {last_error}",
        "attempted_urls": candidates,
    }), 404

@app.route("/zip_inspect", methods=["POST"])
@_rate_limit(limit=20, window=60)
def zip_inspect():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    f = request.files["file"]
    try:
        fname = _safe_filename(f.filename)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    if not fname.lower().endswith(".zip"):
        return jsonify({"error": "Not a ZIP file"}), 400
    tmp = os.path.join(UPLOAD, "inspect_" + uuid.uuid4().hex[:8])
    os.makedirs(tmp, exist_ok=True)
    try:
        zp = os.path.join(tmp, "upload.zip")
        f.save(zp)
        with zipfile.ZipFile(zp) as z:
            names = z.namelist()
        shps = [n for n in names if n.lower().endswith(".shp")]
        return jsonify({"shp_files": shps, "all_files": names})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        try:
            shutil.rmtree(tmp)
        except:
            pass

# ----------------------------------------------------------------------
# UPLOAD ROUTE (handles Groups A–G) with Human-Readable Run ID
# ----------------------------------------------------------------------

@app.route("/upload", methods=["POST"])
@_rate_limit(limit=12, window=60)
@_with_pipeline_sem
def upload():
    file = request.files.get("file")
    if not file:
        return jsonify({"error": "No file uploaded."}), 400
    try:
        _safe_filename(file.filename)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    # Determine base name for run ID
    module = request.form.get("module", "A")
    if module == "F":
        base_name = request.form.get("f_forest") or os.path.splitext(file.filename)[0]
    else:
        base_name = request.form.get("forest") or os.path.splitext(file.filename)[0]

    run_id = _generate_run_id(base_name)
    # Ensure uniqueness (very unlikely, but safe)
    while os.path.exists(os.path.join(OUTPUT, run_id)):
        run_id = _generate_run_id(base_name + "_" + secrets.token_hex(2))

    with _PROG_LOCK: _PROG[run_id] = []
    try:
        if module in ("A", "B", "D") and file.filename.lower().endswith(".zip"):
            return jsonify({"error": "ZIP files are not allowed for this module. Please upload a CSV or Excel file.", "run_id": run_id}), 400

        mode = request.form.get("mode", "A")
        zone = request.form.get("zone", "44")
        title = request.form.get("title", "").strip()
        legend_title = request.form.get("legend_title", "Legend").strip() or "Legend"
        label_col = request.form.get("label_col", "").strip()
        try:
            mapping = json.loads(request.form.get("mapping", "{}"))
        except:
            mapping = {}
        w = float(request.form.get("w", 50))
        h = float(request.form.get("h", 50))
        rows = int(request.form.get("rows", 10))
        cols = int(request.form.get("cols", 10))
        forest = request.form.get("forest") or (mapping or {}).get("forest") or "FOREST"
        username = _require_login() or "guest"
        out = os.path.join(OUTPUT, run_id)
        os.makedirs(out, exist_ok=True)
        crs = get_crs(zone)
        _prog(run_id, f"Starting module {module}…", 2)
        lc_out = None
        lp_gdf = None
        area_ha_disp = None
        base_name_file = os.path.splitext(os.path.basename(file.filename))[0]
        if file.filename.lower().endswith(".zip"):
            try:
                base_name_file = _extract_shapefile_basename_from_zip(file, mapping.get("target_shp"))
            except Exception as e:
                log.warning(f"Could not extract shapefile basename from ZIP: {e}, using ZIP filename")
                base_name_file = os.path.splitext(os.path.basename(file.filename))[0]

        if module == "B":
            _prog(run_id, "Reading…", 8)
            df = read_input(file)
            poly, line, pts = group_b(df, crs, out, mapping)
            lc_out = label_col or "Forest"
        elif module == "C":
    	     _prog(run_id, "Processing ZIP or CSV…", 10)
    	     poly, line, pts = group_c(file, crs, w, h, rows, cols, out, mode, mapping,
                              base_name=base_name, run_id=run_id)
    	     lc_out = "SN"
    	     lp_gdf = pts
        elif module == "D":
            _prog(run_id, "Reading…", 8)
            df = read_input(file)
            d_mode = request.form.get("d_mode", "A")
            poly, line, pts = group_d(df, crs, out, mapping, mode=d_mode)
            lc_out = label_col or "Forest"
        elif module == "E":
            e_mode = request.form.get("e_mode", "A")
            nc = max(2, min(15, int(request.form.get("n_compartments", 4))))
            at = float(request.form.get("area_tol_ha", "0.3") or "0.3")
            method = request.form.get("e_method", "bisect")
            is_zip = file.filename.lower().endswith(".zip")
            fcn = request.form.get("forest_col_name") or None
            if mapping and not mapping.get("forest"):
                mapping["forest"] = base_name_file
            _prog(run_id, "Loading input…", 8)
            src_data = file if is_zip else read_input(file)
            poly, line, pts = group_e(src_data, crs, out, mapping,
                                      e_mode=e_mode, n_compartments=nc,
                                      is_zip=is_zip, fcol=fcn,
                                      area_tol_ha=at, method=method, run_id=run_id)
            _prog(run_id, "Rendering preview…", 88)
            layout_state = get_default_layout_state()
            bounds = poly.total_bounds
            if bounds is not None and len(bounds) == 4:
                w2 = bounds[2] - bounds[0]
                h2 = bounds[3] - bounds[1]
                poly_aspect = w2 / h2 if h2 > 0 else 1.0
            else:
                poly_aspect = 1.0
            safe_rect = compute_safe_rect(layout_state, poly_aspect)
            render_map(os.path.join(out, "output.png"), poly_gdf=poly,
                       label_col=label_col or "Comp_ID",
                       safe_rect=safe_rect, layout_state=layout_state)
            kmz_url = generate_kmz(poly, line, pts, out, run_id)
            _append_run(username, run_id, "E")
            _prog(run_id, "Complete.", 100)
            return jsonify({"run_id": run_id, "download": f"/download/{run_id}", "kmz_url": kmz_url})
        elif module == "F":
            dem_cache_key = request.form.get("dem_cache_key", "").strip()
            dem_catalog_path = request.form.get("dem_catalog_path", "").strip()

            class _FileDEM:
                def __init__(self, p):
                    self.filename = os.path.basename(p)
                    self._p = p
                def save(self, dest):
                    shutil.copy2(self._p, dest)
                def read(self, size=-1):
                    with open(self._p, "rb") as f:
                        return f.read(size)

            if dem_cache_key:
                candidates = [f for f in os.listdir(DEM_CACHE_DIR) if f.startswith(dem_cache_key)]
                if not candidates:
                    return jsonify({"error": "Cached DEM not found. Please select again.", "run_id": run_id}), 400
                dem_f = _FileDEM(os.path.join(DEM_CACHE_DIR, candidates[0]))
            elif dem_catalog_path:
                cat_full = _safe_path(DEM_CATALOG_DIR, dem_catalog_path)
                if not os.path.exists(cat_full):
                    return jsonify({"error": f"DEM catalog file not found: {dem_catalog_path}", "run_id": run_id}), 400
                dem_f = _FileDEM(cat_full)
            else:
                dem_f = request.files.get("dem_file")
                if not dem_f:
                    return jsonify({"error": "No DEM selected. Choose from the catalog dropdown or upload a .tif file.", "run_id": run_id}), 400
            f_forest = request.form.get("f_forest") or base_name_file
            f_mode = request.form.get("f_mode", "A")
            cc = request.form.get("comp_col") or None
            bzip = file.filename.lower().endswith(".zip")
            fa = None
            fas = request.form.get("field_area_ha", "").strip()
            if fas:
                try:
                    fa = float(fas)
                except:
                    pass
            sr, vgdf, bgdf, fmo, pgs = group_f(file, dem_f, crs, out, mapping,
                                                boundary_is_zip=bzip,
                                                forest_name=f_forest,
                                                f_mode=f_mode,
                                                comp_col_name=cc,
                                                field_area_ha=fa,
                                                run_id=run_id)
            _prog(run_id, "Rendering preview…", 92)
            layout_state = get_default_layout_state()
            bounds = vgdf.total_bounds if (vgdf is not None and not vgdf.empty) else (0,0,1,1)
            if bounds is not None and len(bounds) == 4:
                w2 = bounds[2] - bounds[0]
                h2 = bounds[3] - bounds[1]
                poly_aspect = w2 / h2 if h2 > 0 else 1.0
            else:
                poly_aspect = 1.0
            safe_rect = compute_safe_rect(layout_state, poly_aspect)
            render_map(os.path.join(out, "output.png"), poly_gdf=vgdf, line_gdf=bgdf,
                       label_col="Description", safe_rect=safe_rect, layout_state=layout_state,
                       slope_mode=True, slope_areas={r["Slope_Range"]: r["Area_ha"] for r in sr})
            poly = vgdf if (vgdf is not None and not vgdf.empty) else gpd.GeoDataFrame()
            line = gpd.GeoDataFrame()
            pts = gpd.GeoDataFrame()
            kmz_url = generate_kmz(poly, line, pts, out, run_id)
            _append_run(username, run_id, "F")
            _prog(run_id, "Complete.", 100)
            return jsonify({"run_id": run_id, "download": f"/download/{run_id}", "kmz_url": kmz_url})
        else:
            _prog(run_id, "Reading…", 8)
            df = read_input(file)
            poly, line, pts = group_a(df, forest, crs, out, mapping)
            if not poly.empty and "Area_ha" in poly.columns:
                area_ha_disp = float(poly["Area_ha"].sum())
            lc_out = label_col or "Forest"

        point_label_col = None
        if pts is not None and not pts.empty:
            for cand in ["SN", "Order", "Point_ID", "ID", "point_id"]:
                if cand in pts.columns:
                    point_label_col = cand
                    break
            if point_label_col is None:
                point_label_col = label_col or None

        _prog(run_id, "Rendering preview…", 88)
        layout_state = get_default_layout_state()
        bounds = poly.total_bounds if (poly is not None and not poly.empty) else (0,0,1,1)
        if bounds is not None and len(bounds) == 4:
            w2 = bounds[2] - bounds[0]
            h2 = bounds[3] - bounds[1]
            poly_aspect = w2 / h2 if h2 > 0 else 1.0
        else:
            poly_aspect = 1.0
        safe_rect = compute_safe_rect(layout_state, poly_aspect)
        render_map(os.path.join(out, "output.png"), poly_gdf=poly, line_gdf=line, pts_gdf=pts,
                   label_col=lc_out, point_label_col=point_label_col,
                   safe_rect=safe_rect, layout_state=layout_state, title=title)
        kmz_url = generate_kmz(poly, line, pts, out, run_id)
        _append_run(username, run_id, module)
        _prog(run_id, "Complete.", 100)
        return jsonify({"run_id": run_id, "download": f"/download/{run_id}", "kmz_url": kmz_url})
    except ValueError as e:
        _prog(run_id, f"ERROR: {e}", 0)
        return jsonify({"error": str(e), "run_id": run_id}), 400
    except Exception as e:
        _prog(run_id, f"ERROR: {e}", 0)
        return jsonify({"error": f"Unexpected error: {e}", "run_id": run_id}), 500

# ----------------------------------------------------------------------
# GROUP H ROUTE with Human-Readable Run ID
# ----------------------------------------------------------------------

@app.route("/run_h", methods=["POST"])
@_rate_limit(limit=10, window=60)
@_with_pipeline_sem
def run_h():
    # Generate human-readable run ID from boundary file name
    boundary_file = request.files.get('boundary')
    if not boundary_file:
        return jsonify({"error": "Missing boundary file"}), 400
    base_name = os.path.splitext(boundary_file.filename)[0]
    run_id = _generate_run_id(base_name)
    while os.path.exists(os.path.join(OUTPUT, run_id)):
        run_id = _generate_run_id(base_name + "_" + secrets.token_hex(2))

    _prog(run_id, "Starting Group H...", 0)
    try:
        required = ['boundary', 'compartments', 'dem', 'satellite', 'sample_points']
        files = {}
        for key in required:
            if key not in request.files or request.files[key].filename == '':
                return jsonify({"error": f"Missing required file: {key}", "run_id": run_id}), 400
            files[key] = request.files[key]
        survey_file = request.files.get('survey_points')
        crs = request.form.get('crs', 'EPSG:32644')

        tmp_dir = os.path.join(UPLOAD, f"h_temp_{run_id}")
        os.makedirs(tmp_dir, exist_ok=True)
        saved_files = {}
        for key, f in files.items():
            path = os.path.join(tmp_dir, f"{key}_{f.filename}")
            f.save(path)
            saved_files[key] = path
        if survey_file and survey_file.filename != '':
            path = os.path.join(tmp_dir, f"survey_{survey_file.filename}")
            survey_file.save(path)
            saved_files['survey'] = path

        out_dir = os.path.join(OUTPUT, run_id)
        os.makedirs(out_dir, exist_ok=True)

        zip_path, out_dir = process_group_h(
            saved_files['boundary'],
            saved_files['compartments'],
            saved_files['dem'],
            saved_files['satellite'],
            saved_files['sample_points'],
            saved_files.get('survey'),
            crs=crs,
            out_dir=out_dir,
            run_id=run_id
        )

        shutil.rmtree(tmp_dir, ignore_errors=True)

        preview_path = os.path.join(out_dir, "Slope_Map.png")
        if os.path.exists(preview_path):
            shutil.copy(preview_path, os.path.join(out_dir, "output.png"))

        _append_run(_require_login() or "guest", run_id, "H", "Group H maps generated")
        _prog(run_id, "Complete.", 100)

        return jsonify({
            "run_id": run_id,
            "download": f"/download/{run_id}",
            "message": "Group H processing complete. Six maps generated."
        })
    except Exception as e:
        _prog(run_id, f"ERROR: {e}", 0)
        return jsonify({"error": str(e), "run_id": run_id}), 500

# ----------------------------------------------------------------------
# ABOUT, ROBOTS, SITEMAP
# ----------------------------------------------------------------------

@app.route("/about")
def about_page():
    return Response("""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>Elfak GIS Pro Studio — Professional Forest GIS Application</title>
<meta name="description" content="Elfak GIS Pro Studio (elfakgis, elfakgispro, elfakgisstudio, elfakgisprostudio) is a professional web-based GIS application for forest boundary mapping, slope analysis, compartment subdivision, survey point generation and multi-forest analysis. Built for Nepal forestry professionals.">
<meta name="keywords" content="elfakgis, elfakgispro, elfakgisstudio, elfakgisprostudio, elfak gis, elfak gis pro, forest gis nepal, forest boundary mapping, slope analysis nepal, compartment mapping, survey points gis, forestry nepal gis, GIS tool nepal">
<meta name="robots" content="index, follow">
<meta property="og:title" content="Elfak GIS Pro Studio — Forest GIS Application">
<meta property="og:description" content="Professional web GIS for Nepal forestry: boundary mapping, slope analysis, compartment subdivision, survey points. Free to use at elfakgisprostudio.onrender.com">
<meta property="og:url" content="https://elfakgisprostudio.onrender.com/">
<meta property="og:type" content="website">
<link rel="canonical" href="https://elfakgisprostudio.onrender.com/">
<link rel="alternate" href="https://elfakgisprostudio.onrender.com/" hreflang="en">
<style>
  body{font-family:system-ui,sans-serif;max-width:900px;margin:0 auto;padding:20px 24px;
       color:#1a2e22;background:#f0f8f3;line-height:1.7}
  h1{color:#059669;font-size:2em;margin-bottom:8px}
  h2{color:#065f46;border-bottom:2px solid #34d399;padding-bottom:6px;margin-top:32px}
  .badge{display:inline-block;background:#d1fae5;color:#065f46;padding:3px 10px;
         border-radius:20px;font-size:13px;font-weight:600;margin:3px}
  .cta{display:inline-block;background:linear-gradient(135deg,#34d399,#059669);
       color:white;padding:12px 28px;border-radius:8px;text-decoration:none;
       font-weight:700;font-size:16px;margin-top:20px;box-shadow:0 4px 14px rgba(16,185,129,.35)}
  .feature{background:white;border-radius:10px;padding:16px 20px;margin:12px 0;
            border-left:4px solid #10b981;box-shadow:0 2px 8px rgba(0,0,0,.06)}
  footer{margin-top:48px;padding-top:16px;border-top:1px solid #b7eacf;
         color:#6b9880;font-size:13px}
</style>
<script type="application/ld+json">
{"@context":"https://schema.org","@type":"WebApplication",
 "name":"Elfak GIS Pro Studio",
 "alternateName":["elfakgis","elfakgispro","elfakgisstudio","elfakgisprostudio"],
 "url":"https://elfakgisprostudio.onrender.com/",
 "description":"Professional web-based GIS application for forest boundary mapping, slope analysis, compartment subdivision and survey point generation for Nepal forestry professionals.",
 "applicationCategory":"GIS Software",
 "operatingSystem":"Web Browser",
 "offers":{"@type":"Offer","price":"0","priceCurrency":"USD"},
 "author":{"@type":"Organization","name":"Elfak GIS"}}
</script>
</head>
<body>
<h1>🌲 Elfak GIS Pro Studio</h1>
<p><strong>Professional Forest GIS Application</strong> for boundary mapping, slope analysis, compartment subdivision, and survey point generation.</p>
<p>
  <span class="badge">elfakgis</span>
  <span class="badge">elfakgispro</span>
  <span class="badge">elfakgisstudio</span>
  <span class="badge">elfakgisprostudio</span>
  <span class="badge">Forest GIS Nepal</span>
</p>
<a href="/" class="cta">🚀 Open Application</a>

<h2>What is Elfak GIS Pro Studio?</h2>
<p>Elfak GIS Pro Studio is a free, web-based Geographic Information System designed for forestry professionals in Nepal and the broader Himalayan region. It provides a complete workflow from raw survey data to professional-quality GIS outputs — all without requiring QGIS, ArcGIS, or any desktop installation.</p>

<h2>Features</h2>
<div class="feature"><strong>A — Boundary Whole</strong>: Generate forest boundary polygon from GPS survey points (Excel/CSV). Produces shapefile, line, and point layers with area calculation.</div>
<div class="feature"><strong>B — Segmented Forest</strong>: Multi-forest boundary generation with separate shapefile per forest/compartment.</div>
<div class="feature"><strong>C — Sample Plot Generator</strong>: Systematic grid-based sample plot placement inside forest boundaries for forest inventory.</div>
<div class="feature"><strong>D — Multi-Forest Complex</strong>: Batch process multiple forests with nested compartment structure.</div>
<div class="feature"><strong>E — Polygon Subdivider</strong>: Automatically divide a forest polygon into N equal-area compartments (2–15) with configurable area tolerance.</div>
<div class="feature"><strong>F — Slope Analysis</strong>: DEM-based slope classification (0–19°, 19–31°, 31–45°, >45°) with raster-to-polygon conversion, per-compartment area tables, and professional A4 map output.</div>
<div class="feature"><strong>G — Survey Point Generator</strong>: Generate boundary, vertex, and divider survey points from compartment shapefiles. Exports SHP, CSV, and Excel.</div>
<div class="feature"><strong>H — Sample Point Based GIS Maps</strong>: Upload boundary, compartments, DEM, satellite image, sample points and survey points to automatically generate six publication‑quality maps: Slope, Satellite, Sub‑compartment, Sample Plot, Boundary Survey, and Survey Point maps. Exports PNG, PDF, SVG.</div>

<h2>Technical Specifications</h2>
<ul>
<li>Coordinate systems: UTM Zone 43N, 44N, 45N, 46N (EPSG:32643–32646)</li>
<li>Input formats: Excel (.xlsx), CSV, Shapefile (.shp), ZIP of shapefiles, GeoTIFF (.tif)</li>
<li>Output formats: Shapefile (.shp), GeoJSON, KMZ, PNG map, Excel, CSV, PDF, SVG</li>
<li>Map output: A4 size, 300 DPI, professional cartography</li>
<li>DEM analysis: Slope reclassification, raster-to-polygon, area statistics</li>
<li>Supports 100+ simultaneous users with per-user data isolation</li>
</ul>

<h2>Who Uses Elfak GIS Pro Studio?</h2>
<p>Forest rangers, community forestry groups, district forest offices, forest inventory teams, and GIS professionals in Nepal, Bhutan, and similar forested regions who need professional GIS output without expensive desktop software.</p>

<h2>Open the Application</h2>
<p><a href="/" class="cta">🌲 Launch Elfak GIS Pro Studio</a></p>

<footer>
  <p>Elfak GIS Pro Studio · <a href="https://elfakgisprostudio.onrender.com/">elfakgisprostudio.onrender.com</a></p>
  <p>Keywords: elfakgis · elfakgispro · elfakgisstudio · elfakgisprostudio · forest gis nepal · slope analysis · compartment mapping · survey points · forestry gis</p>
</footer>
</body>
</html>""", mimetype="text/html")
@app.route("/run_g", methods=["POST"])
@_rate_limit(limit=10, window=60)
@_with_pipeline_sem
def run_g():
    run_id = str(uuid.uuid4())
    _prog(run_id, "Starting Group G...", 0)
    try:
        if "file" not in request.files:
            return jsonify({"error": "No shapefile uploaded.", "run_id": run_id}), 400
        file = request.files["file"]

        # Get parameters
        zone = request.form.get("zone", "44")
        comp_col = request.form.get("comp_col", "").strip() or None
        title = request.form.get("title", "Forest Survey Points").strip()
        try:
            spacing = float(request.form.get("spacing", "20"))
            if spacing <= 0:
                raise ValueError
        except:
            return jsonify({"error": "Invalid spacing value.", "run_id": run_id}), 400

        # Determine base_name from uploaded file
        base_name = os.path.splitext(os.path.basename(file.filename))[0]
        target_shp = request.form.get("target_shp", "").strip() or None

        username = _require_login() or "guest"
        out = os.path.join(OUTPUT, run_id)
        os.makedirs(out, exist_ok=True)

        # Process
        df, shp_gdf, poly_gdf, summary = group_g(
            file, zone, comp_col, spacing, out, run_id,
            target_shp=target_shp, base_name=base_name
        )

        _prog(run_id, "Rendering A4 map…", 90)
        layout_state = get_default_layout_state()
        safe_rect = compute_safe_rect(layout_state, 1.0)
        _g_preview(shp_gdf, poly_gdf, os.path.join(out, "output.png"),
                   safe_rect, title=title, layout_state=layout_state)

        kmz_url = None
        try:
            kmz_url = generate_kmz(poly_gdf, gpd.GeoDataFrame(), shp_gdf, out, run_id)
        except:
            pass

        _append_run(username, run_id, "G", f"{summary['total']} pts | {summary['compartments']} compartments")
        _prog(run_id, "Complete.", 100)
        return jsonify({
            "run_id": run_id,
            "download": f"/download/{run_id}",
            "kmz_url": kmz_url,
            "summary": summary
        })
    except Exception as e:
        _prog(run_id, f"ERROR: {e}", 0)
        log.error(f"Group G error: {traceback.format_exc()}")
        return jsonify({"error": str(e), "run_id": run_id}), 500

@app.route("/robots.txt")
def robots_txt():
    return Response("""User-agent: *
Allow: /
Allow: /about
Allow: /sitemap.xml
Disallow: /upload
Disallow: /run_g
Disallow: /outputs/
Disallow: /download/
Disallow: /progress/
Disallow: /geojson/
Sitemap: https://elfakgisprostudio.onrender.com/sitemap.xml
""", mimetype="text/plain")

@app.route("/sitemap.xml")
def sitemap_xml():
    return Response("""<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
  <url><loc>https://elfakgisprostudio.onrender.com/</loc>
       <priority>1.0</priority><changefreq>weekly</changefreq></url>
  <url><loc>https://elfakgisprostudio.onrender.com/about</loc>
       <priority>0.9</priority><changefreq>monthly</changefreq></url>
</urlset>""", mimetype="application/xml")

# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

if __name__ == "__main__":
    app.run(debug=True, threaded=True)
