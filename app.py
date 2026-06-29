import os, uuid, zipfile, shutil, json, re, io, traceback, math
import threading, time, hashlib, html, secrets, logging
from collections import defaultdict
from functools import wraps
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker

from flask import (Flask, request, jsonify, send_file, send_from_directory,
                   render_template, session, Response, stream_with_context, abort, g)
from shapely.geometry import (Polygon, Point, LineString, MultiPolygon,
                               MultiPoint, GeometryCollection)
from shapely.ops import unary_union, voronoi_diagram
from shapely.geometry import box as _sbox
from shapely import affinity

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
log = logging.getLogger("elfak-gis")

app = Flask(__name__)
# SECRET_KEY must be set in environment for production — never hardcode
app.secret_key = os.environ.get("SECRET_KEY") or secrets.token_hex(32)
if not os.environ.get("SECRET_KEY"):
    log.warning("SECRET_KEY not set in environment — using random key (sessions won't persist across restarts)")

app.config.update(
    SESSION_COOKIE_SECURE    = os.environ.get("HTTPS","0") == "1",
    SESSION_COOKIE_HTTPONLY  = True,
    SESSION_COOKIE_SAMESITE  = "Lax",
    PERMANENT_SESSION_LIFETIME = 86400 * 7,          # 7 days
    MAX_CONTENT_LENGTH       = 2 * 1024 * 1024 * 1024,  # 2 GB max upload
)

UPLOAD, OUTPUT, USERS_FILE = "uploads", "outputs", "users.json"
DEM_CATALOG_DIR = os.environ.get("DEM_CATALOG_DIR", "dem_catalog")
# GitHub raw base URL for DEM catalog (set in environment or auto-detected)
# Format: https://raw.githubusercontent.com/USER/REPO/BRANCH/dem_catalog
GITHUB_DEM_BASE = os.environ.get(
    "GITHUB_DEM_BASE",
    "https://github.com/1ravikafle-glitch/ElfakGISProStudio/tree/main/dem_catalog"
)
DEM_CACHE_DIR = os.path.join(UPLOAD, "dem_cache")
for _d in (UPLOAD, OUTPUT, DEM_CATALOG_DIR, DEM_CACHE_DIR):
    os.makedirs(_d, exist_ok=True)

A4W, A4H, DPI = 8.27, 11.69, 200

# ── Thread-safe in-memory progress store ─────────────────────────────────────
_PROG: dict = {}
_PROG_LOCK = threading.Lock()

def _prog(rid, msg, pct=None):
    o = {"msg": str(msg)[:500]}          # cap message length
    if pct is not None: o["pct"] = max(0, min(100, int(pct)))
    with _PROG_LOCK:
        if rid not in _PROG: _PROG[rid] = []
        _PROG[rid].append(json.dumps(o))
        _PROG[rid] = _PROG[rid][-500:]   # keep last 500 events

def _cleanup_old_prog():
    """Remove progress entries older than 2 hours to prevent memory leaks."""
    while True:
        time.sleep(3600)
        with _PROG_LOCK:
            keys = list(_PROG.keys())
            if len(keys) > 10000:      # safety cap
                for k in keys[:len(keys)//2]:
                    _PROG.pop(k, None)

threading.Thread(target=_cleanup_old_prog, daemon=True).start()

# ═══════════════════════════════════════════════════════════════════════════════
# USER DATABASE — Thread-safe, atomic, unique-username enforced
# Each username is unique and owned by the first person who registers it.
# Subsequent logins to the SAME username are blocked — they must use a
# different name. This ensures per-user data isolation for 100+ concurrent users.
# ═══════════════════════════════════════════════════════════════════════════════
_USERS_LOCK = threading.RLock()

def _lu():
    """Load users dict from JSON (thread-safe read)."""
    with _USERS_LOCK:
        try:
            if not os.path.exists(USERS_FILE): return {}
            with open(USERS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            log.error(f"_lu error: {e}"); return {}

def _su(users_dict):
    """Atomically save users dict (write-then-rename, POSIX atomic)."""
    with _USERS_LOCK:
        try:
            tmp = USERS_FILE + ".tmp." + uuid.uuid4().hex[:8]
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(users_dict, f, indent=2, ensure_ascii=False)
            os.replace(tmp, USERS_FILE)
        except Exception as e:
            log.error(f"_su error: {e}")

def _register_user(name):
    """
    Register a brand-new unique user.
    Raises ValueError if username already exists.
    Returns the new user dict.
    This is the ONLY place a user is created — ensures uniqueness.
    """
    name = name.strip()
    with _USERS_LOCK:
        u = _lu()
        if name in u:
            raise ValueError(f'Username "{name}" is already taken. Choose a different name.')
        token = secrets.token_hex(32)          # per-user auth token (future use)
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
    """
    Log in an existing user — increments active_sessions counter.
    Returns user dict or raises ValueError if user does not exist.
    """
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
    """Decrement active_sessions on logout."""
    if not name: return
    with _USERS_LOCK:
        u = _lu()
        if name in u:
            u[name]["active_sessions"] = max(0, u[name].get("active_sessions", 1) - 1)
            _su(u)

def _append_run(uname, rid, mod, desc=""):
    """Append a run record to the user's history (thread-safe)."""
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
    """Return username from session, or None."""
    return session.get("username")

def _login_required(fn):
    """Decorator: reject unauthenticated requests with 401."""
    @wraps(fn)
    def wrapper(*args, **kwargs):
        if not _require_login():
            return jsonify({"error": "Authentication required. Please log in."}), 401
        return fn(*args, **kwargs)
    return wrapper

# ── Output folder cleanup — keep only 500 newest runs ────────────────────────
def _cleanup_old_outputs():
    while True:
        time.sleep(1800)   # every 30 min
        try:
            dirs = [(os.path.join(OUTPUT, d),
                     os.path.getmtime(os.path.join(OUTPUT, d)))
                    for d in os.listdir(OUTPUT)
                    if os.path.isdir(os.path.join(OUTPUT, d))]
            dirs.sort(key=lambda x: x[1])
            for path, _ in dirs[:-500]:   # keep newest 500
                shutil.rmtree(path, ignore_errors=True)
        except Exception as e:
            log.error(f"Cleanup error: {e}")

threading.Thread(target=_cleanup_old_outputs, daemon=True).start()

# ── Rate limiter (in-memory, per-IP) ─────────────────────────────────────────
_RL: dict = defaultdict(list)
_RL_LOCK = threading.Lock()

def _rate_limit(limit=30, window=60, key_fn=None):
    """Decorator: limit requests per IP. Default 30 req/min."""
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

# ═══════════════════════════════════════════════════════════════════════════════
# SECURITY HELPERS
# ═══════════════════════════════════════════════════════════════════════════════
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
    """Sanitize filename: strip paths, replace dangerous chars, whitelist ext."""
    fname = os.path.basename((fname or "upload").replace("\\", "/"))
    fname = re.sub(r"[^A-Za-z0-9._-]", "_", fname)[:200]
    if not fname: fname = "upload"
    ext = os.path.splitext(fname)[1].lower()
    if ext not in _ALLOWED_EXTS:
        raise ValueError(f"File type '{ext}' not allowed. Allowed: {sorted(_ALLOWED_EXTS)}")
    return fname

def _safe_runid(rid):
    """Validate run_id is a strict UUID4 — prevents path traversal attacks."""
    rid = str(rid).strip().lower()
    if not re.fullmatch(
        r"[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}",
        rid
    ):
        log.warning(f"Invalid run_id rejected: {rid!r}")
        abort(400, "Invalid run ID.")
    return rid

def _safe_path(base, rel):
    """Resolve path and ensure it stays strictly inside base (traversal guard)."""
    base = os.path.realpath(os.path.abspath(base))
    full = os.path.realpath(os.path.abspath(os.path.join(base, str(rel))))
    if not (full == base or full.startswith(base + os.sep)):
        log.warning(f"Path traversal blocked: base={base!r} rel={rel!r} full={full!r}")
        abort(400, "Path traversal detected.")
    return full

def _validate_username(name):
    """Validate and normalise a username."""
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
    # Prevent SQL/script injection via name (extra caution even though we don't use SQL)
    _bad = set("<>'\";&|`")
    if any(c in _bad for c in name):
        raise ValueError("Username contains invalid characters.")
    return name

def _get_client_ip():
    """Extract real client IP, respecting reverse-proxy headers."""
    xff = request.headers.get("X-Forwarded-For", "")
    if xff:
        ip = xff.split(",")[0].strip()
        # Basic validation
        if re.match(r"^\d{1,3}(\.\d{1,3}){3}$", ip) or ":" in ip:
            return ip
    return request.remote_addr or "unknown"

# ── Security response headers ────────────────────────────────────────────────
@app.after_request
def _security_headers(resp):
    """Attach comprehensive security headers to every HTTP response."""
    resp.headers["X-Content-Type-Options"]   = "nosniff"
    resp.headers["X-Frame-Options"]          = "DENY"
    resp.headers["X-XSS-Protection"]         = "1; mode=block"
    resp.headers["Referrer-Policy"]          = "strict-origin-when-cross-origin"
    resp.headers["Permissions-Policy"]       = "geolocation=(), camera=(), microphone=()"
    # HSTS only when HTTPS is confirmed
    if os.environ.get("HTTPS") == "1":
        resp.headers["Strict-Transport-Security"] = "max-age=63072000; includeSubDomains"
    # CORS — tighten in production via ALLOWED_ORIGINS env var
    allowed = os.environ.get("ALLOWED_ORIGINS", "*")
    resp.headers["Access-Control-Allow-Origin"]       = allowed
    resp.headers["Access-Control-Allow-Headers"]      = "Content-Type,Authorization,X-Requested-With"
    resp.headers["Access-Control-Allow-Methods"]      = "GET,POST,OPTIONS"
    resp.headers["Access-Control-Allow-Credentials"]  = "true"
    # Prevent caching of sensitive API responses
    if request.path.startswith(("/login", "/me", "/history", "/progress")):
        resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, private"
        resp.headers["Pragma"]        = "no-cache"
    return resp

# ── Concurrent pipeline semaphore ────────────────────────────────────────────
_PIPELINE_SEM = threading.Semaphore(int(os.environ.get("MAX_PIPELINES", "20")))
_ACTIVE_PIPELINES = 0
_AP_LOCK = threading.Lock()

def _with_pipeline_sem(fn):
    """Limit simultaneous heavy GIS pipelines. Returns 503 if overloaded."""
    @wraps(fn)
    def wrapper(*args, **kwargs):
        global _ACTIVE_PIPELINES
        acquired = _PIPELINE_SEM.acquire(timeout=45)
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

@app.route("/server_status")
@_rate_limit(limit=10, window=60)
def server_status():
    return jsonify({"status":"ok","active_pipelines":_ACTIVE_PIPELINES,
                    "max_pipelines":int(os.environ.get("MAX_PIPELINES","20")),
                    "registered_users":len(_lu())})

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

@app.route("/about")
def about_page():
    """Public SEO-optimised about page — visible to Google without login."""
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

<h2>Technical Specifications</h2>
<ul>
<li>Coordinate systems: UTM Zone 43N, 44N, 45N, 46N (EPSG:32643–32646)</li>
<li>Input formats: Excel (.xlsx), CSV, Shapefile (.shp), ZIP of shapefiles</li>
<li>Output formats: Shapefile (.shp), GeoJSON, KMZ, PNG map, Excel, CSV</li>
<li>Map output: A4 size, 200 DPI, professional cartography</li>
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

# ── column aliases ───────────────────────────────────────────────────────────
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

# ── geometry helpers ─────────────────────────────────────────────────────────
def safe_polygon(coords):
    p = Polygon(coords); return p if p.is_valid else p.buffer(0)
def _repair(g):
    """Repair geometry: fix invalid, handle collections, normalize."""
    if g is None: return None
    try:
        if g.is_empty: return None
        if not g.is_valid: g = g.buffer(0)
        if g is None or g.is_empty: return None
        # Unwrap single-part MultiPolygon
        if g.geom_type == "MultiPolygon":
            parts = [p for p in g.geoms if not p.is_empty]
            if len(parts) == 1: g = parts[0]
        return g
    except Exception:
        try: return g.buffer(0) if g else None
        except: return None
def _as_poly(g):
    if g is None or g.is_empty: return None
    if g.geom_type == "Polygon": return g
    if g.geom_type in ("MultiPolygon","GeometryCollection"):
        ps = [x for x in g.geoms if x.geom_type=="Polygon" and not x.is_empty and x.area>1e-10]
        return max(ps, key=lambda x: x.area) if ps else None
    return None
def _close_poly(p):
    if p is None or p.is_empty: return p
    ext = list(p.exterior.coords)
    if ext[0] != ext[-1]: ext.append(ext[0])
    holes = [list(i.coords) for i in p.interiors]
    for h in holes:
        if h[0] != h[-1]: h.append(h[0])
    try:
        c = Polygon(ext, holes); return (c if c.is_valid else c.buffer(0)) if not c.is_empty else p
    except: return p
def _enforce_poly_gdf(gdf):
    """Ensure GDF only has Polygon/MultiPolygon geometry (shapefile requirement).
    Extracts polygon parts from GeometryCollections, drops lines/points."""
    if gdf is None or gdf.empty: return gdf
    keep = []
    for _, row in gdf.iterrows():
        g = row.geometry
        if g is None: continue
        # Handle GeometryCollection: extract polygon parts
        if hasattr(g, "geoms"):
            polys = [p for p in g.geoms if p.geom_type in ("Polygon","MultiPolygon") and not p.is_empty and p.area>1e-10]
            if not polys: continue
            g = _repair(unary_union(polys))
        if g is None or g.is_empty: continue
        pg = _as_poly(_repair(g))
        if pg and not pg.is_empty and pg.area > 1e-10:
            r2 = row.copy(); r2["geometry"] = _close_poly(pg); keep.append(r2)
    if not keep: return gpd.GeoDataFrame(columns=gdf.columns, crs=gdf.crs)
    return gpd.GeoDataFrame(keep, crs=gdf.crs)

# ── map decorations (reference-image accurate) ───────────────────────────────
def _north_arrow(ax):
    """Professional north arrow matching reference images.
    Shows: 'N' label + filled black downward triangle + white upward triangle.
    Positioned top-right corner, inside axes."""
    from matplotlib.patches import FancyArrow, Polygon as MPoly
    x0, x1 = ax.get_xlim(); y0, y1 = ax.get_ylim()
    aw = x1-x0; ah = y1-y0
    # Position: top-right, slightly inset
    cx = x1 - aw*0.055   # centre x
    ty = y1 - ah*0.025   # top of N letter
    arh = ah*0.085        # arrow height
    arw = aw*0.018        # half-width of arrow

    # N letter at top
    ax.text(cx, ty, "N", ha="center", va="top",
            fontsize=12, fontweight="bold", color="black",
            fontfamily="sans-serif", zorder=25)

    # Arrow shaft centre
    ac_top = ty - ah*0.022   # just below N
    ac_bot = ac_top - arh

    # Upper (black) filled triangle — pointing UP
    tri_up = MPoly([
        [cx,      ac_top],
        [cx-arw,  ac_top - arh*0.5],
        [cx+arw,  ac_top - arh*0.5],
    ], closed=True, facecolor="black", edgecolor="black", linewidth=0.8, zorder=24)
    ax.add_patch(tri_up)

    # Lower (white) filled triangle — pointing DOWN
    tri_dn = MPoly([
        [cx,      ac_bot],
        [cx-arw,  ac_bot + arh*0.5],
        [cx+arw,  ac_bot + arh*0.5],
    ], closed=True, facecolor="white", edgecolor="black", linewidth=0.8, zorder=24)
    ax.add_patch(tri_dn)

    # Thin vertical centre line
    ax.plot([cx, cx], [ac_top - arh*0.5, ac_bot + arh*0.5],
            color="black", linewidth=0.8, zorder=23)

def _scale_bar(ax):
    """Black-white-black 3-segment scale bar like reference images."""
    x0, x1 = ax.get_xlim(); y0, y1 = ax.get_ylim()
    aw = x1-x0; ah = y1-y0
    raw = aw * 0.22
    if raw <= 0: return
    import math as _m
    mag = 10**_m.floor(_m.log10(max(raw, 1e-6)))
    nice_vals = [1,2,2.5,5,10,20,25,50,100,200,250,500,1000,2000,5000,10000]
    bar_m = min(nice_vals, key=lambda v: abs(v*mag - raw)) * mag
    if bar_m < 10: bar_m = raw
    segs = 3; seg_w = bar_m / segs
    bx = x0 + aw*0.07; by = y0 + ah*0.040; bh = ah*0.016
    for i in range(segs):
        fc = "black" if i % 2 == 0 else "white"
        ax.add_patch(plt.Rectangle((bx + i*seg_w, by), seg_w, bh,
                     linewidth=0.9, edgecolor="black", facecolor=fc,
                     zorder=15, clip_on=False))
    def _fmt(v): return f"{int(v)} m" if v < 1000 else f"{v/1000:.0f} km"
    ax.text(bx,            by - bh*0.6, "0",              ha="center", va="top", fontsize=6, fontweight="bold", color="black", zorder=16)
    ax.text(bx+bar_m/2,   by - bh*0.6, _fmt(bar_m/2),    ha="center", va="top", fontsize=6, fontweight="bold", color="black", zorder=16)
    ax.text(bx+bar_m,     by - bh*0.6, _fmt(bar_m),      ha="center", va="top", fontsize=6, fontweight="bold", color="black", zorder=16)

def _add_legend(ax, handles, legend_title="Legend", loc="lower right"):
    """Place legend OUTSIDE the map axes area — never overlaps the drawing."""
    if not handles: return
    # Anchor outside the axes so it never overlaps the map
    anchor_lut = {
        "lower right": (1.02, 0.00, "upper left"),
        "lower left":  (-0.02, 0.00, "upper right"),
        "upper right": (1.02, 1.00, "lower left"),
        "upper left":  (-0.02, 1.00, "lower right"),
        "right":       (1.02, 0.50, "center left"),
    }
    bba, bbl, loca = anchor_lut.get(loc, (1.02, 0.00, "upper left"))
    try:
        leg = ax.legend(
            handles=handles, title=legend_title,
            loc=loca,
            bbox_to_anchor=(bba, bbl),
            bbox_transform=ax.transAxes,
            fontsize=8, title_fontsize=8.5,
            framealpha=0.96, edgecolor="#888",
            fancybox=False, frameon=True,
            borderpad=0.9, labelspacing=0.5,
            facecolor="#f8f8f8", handlelength=1.8,
        )
        leg.get_frame().set_linewidth(1.0)
        leg.get_title().set_fontweight("bold")
    except Exception:
        # Fallback: inside axes
        ax.legend(handles=handles, title=legend_title, loc=loc,
                  fontsize=8, framealpha=0.9, facecolor="#f8f8f8")

def _graticule(ax):
    """Coord labels on all 4 edges, NO interior gridlines. Matches reference."""
    ax.tick_params(axis="both", which="major",
                   left=True, right=True, top=True, bottom=True,
                   labelleft=True, labelright=True,
                   labelbottom=True, labeltop=False,
                   direction="out", length=5, width=0.8,
                   labelsize=5.5, color="#333", pad=2)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{int(v):,}"))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{int(v):,}"))
    ax.xaxis.set_tick_params(labelrotation=45)
    ax.yaxis.set_tick_params(labelrotation=90)
    for sp in ax.spines.values(): sp.set_linewidth(0.8); sp.set_color("#444")
    ax.grid(False)
    ax.set_xlabel(""); ax.set_ylabel("")

def _style_ax(ax):
    ax.set_aspect("equal")
    ax.set_facecolor("white")
    _graticule(ax)
    # Draw neat map frame (thick outer border like reference images)
    for sp in ax.spines.values():
        sp.set_linewidth(1.2)
        sp.set_color("#222")

def _label_feat(ax, gdf, col, fs=8, color="black"):
    """Bold centroid labels with white stroke outline, like reference maps."""
    if gdf is None or gdf.empty or col not in gdf.columns: return
    for _, row in gdf.iterrows():
        try:
            g = row.geometry
            if g is None or g.is_empty: continue
            cx, cy = g.centroid.x, g.centroid.y
            ax.annotate(str(row[col]), xy=(cx, cy), ha="center", va="center",
                        fontsize=fs, fontweight="bold", color=color,
                        path_effects=[pe.Stroke(linewidth=2.5, foreground="white"), pe.Normal()],
                        zorder=12)
        except: continue


_CCOLORS = ["#C8E6C9","#B3E5FC","#FFE0B2","#F8BBD0","#E1BEE7","#DCEDC8",
            "#B2EBF2","#FFF9C4","#D7CCC8","#CFD8DC","#C5CAE9","#F0F4C3",
            "#FFCCBC","#B2DFDB","#E8EAF6"]

# ── GROUP A ──────────────────────────────────────────────────────────────────
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
    ptg= gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df[xc],df[yc]), crs=crs)
    pg.to_file(os.path.join(out,f"{pfx}_polygon.shp"))
    lg.to_file(os.path.join(out,f"{pfx}_line.shp"))
    ptg.to_file(os.path.join(out,f"{pfx}_point.shp"))
    return pg, lg, ptg

# ── GROUP B ──────────────────────────────────────────────────────────────────
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
    if not p.empty: p.to_file(os.path.join(out,"forest_polygon.shp"))
    if not l.empty: l.to_file(os.path.join(out,"forest_line.shp"))
    if not pt.empty: pt.to_file(os.path.join(out,"forest_point.shp"))
    return p,l,pt

# ── GROUP C ──────────────────────────────────────────────────────────────────
def group_c(file, crs, w, h, rows, cols, out, mode, mapping=None):
    polygons=[]
    if file.filename.lower().endswith(".zip"):
        folder=os.path.join(UPLOAD,str(uuid.uuid4())); os.makedirs(folder,exist_ok=True)
        zp=os.path.join(folder,"i.zip"); file.save(zp)
        with zipfile.ZipFile(zp) as z: z.extractall(folder)
        shps=[os.path.join(r,f) for r,_,fs in os.walk(folder) for f in fs if f.endswith(".shp")]
        if not shps: raise ValueError("No .shp in ZIP.")
        sp=shps[0]; ts=(mapping or {}).get("target_shp")
        if ts:
            for s in shps:
                if os.path.basename(s)==os.path.basename(ts): sp=s; break
        gdf=gpd.read_file(sp)
        if gdf.empty: raise ValueError("Shapefile empty.")
        gdf=gdf.set_crs(crs) if gdf.crs is None else gdf.to_crs(crs)
        for geom in gdf.geometry:
            if geom is None or geom.is_empty: continue
            if geom.geom_type=="Polygon": polygons.append(geom if geom.is_valid else geom.buffer(0))
            elif geom.geom_type=="MultiPolygon":
                for p in geom.geoms: polygons.append(p if p.is_valid else p.buffer(0))
        if not polygons: raise ValueError("No polygon geometries found.")
    else:
        df=read_input(file); df=normalize_order(df)
        xc=safe_col(df,mapping,"X","X"); yc=safe_col(df,mapping,"Y","Y")
        oc=safe_col(df,mapping,"Order","Order")
        if not xc: raise ValueError("X column not found.")
        if not yc: raise ValueError("Y column not found.")
        if mode=="A":
            if oc: df=df.sort_values(oc)
            coords=list(zip(df[xc],df[yc])); coords.append(coords[0]); polygons=[safe_polygon(coords)]
        else:
            fc=safe_col(df,mapping,"Forest","Forest"); cc=safe_col(df,mapping,"Compartment","Compartment")
            if not fc: raise ValueError("Forest column required for segmented mode.")
            gkeys=[fc,cc] if cc else [fc]
            for _,g in df.groupby(gkeys):
                if oc: g=g.sort_values(oc)
                c=list(zip(g[xc],g[yc])); c.append(c[0])
                if len(c)>=4: polygons.append(safe_polygon(c))
            if not polygons: raise ValueError("No valid polygons built.")
    if not polygons: raise ValueError("No valid polygons from input.")
    p_gdf=gpd.GeoDataFrame([{"geometry":p} for p in polygons],crs=crs)
    l_gdf=gpd.GeoDataFrame([{"geometry":LineString(p.exterior.coords)} for p in polygons],crs=crs)
    union=p_gdf.unary_union; minx,miny,_,_=union.bounds
    pts=[]; sn=1
    for ri in range(rows):
        for ci in range(cols):
            center=Point(minx+ci*w+w/2, miny+ri*h+h/2)
            if union.contains(center):
                pts.append({"SN":sn,"X":center.x,"Y":center.y,"geometry":center}); sn+=1
    pt_gdf=gpd.GeoDataFrame(pts,crs=crs)
    p_gdf.to_file(os.path.join(out,"boundary_polygon.shp"))
    l_gdf.to_file(os.path.join(out,"boundary_line.shp"))
    if not pt_gdf.empty:
        pt_gdf.to_file(os.path.join(out,"sampleplot_point.shp"))
        pd.DataFrame(pts)[["SN","X","Y"]].to_excel(os.path.join(out,"sampleplot.xlsx"),index=False)
    return p_gdf,l_gdf,pt_gdf

# ── GROUP D ──────────────────────────────────────────────────────────────────
def _save_fl(pr,lr,pts,d,crs):
    os.makedirs(d,exist_ok=True); pfx=os.path.basename(d)
    gpd.GeoDataFrame([pr],crs=crs).to_file(os.path.join(d,f"{pfx}_polygon.shp"))
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

# ── GROUP E — SUBDIVISION ────────────────────────────────────────────────────
def _elong(p):
    minx,miny,maxx,maxy=p.bounds; dx,dy=maxx-minx,maxy-miny
    return max(dx,dy)/max(min(dx,dy),1e-9)
def _pa_angle(p):
    try:
        c=np.array(p.exterior.coords[:-1]); c-=c.mean(0)
        _,v=np.linalg.eigh(np.cov(c.T)); return np.arctan2(v[1,1],v[0,1])
    except: return 0.0

def _bisect(poly, n, axis=None, depth=0):
    poly=_repair(poly)
    if poly is None or poly.is_empty or poly.area<1e-10: return []
    if n<=1: return [_close_poly(poly)]
    minx,miny,maxx,maxy=poly.bounds
    use_p=False; rot=0.0; cx0,cy0=poly.centroid.x,poly.centroid.y
    if axis is None and depth<2 and _elong(poly)>1.6:
        rot=_pa_angle(poly)
        if 10<abs(np.degrees(rot))%90<80: use_p=True
    if use_p:
        pr=affinity.rotate(poly,-np.degrees(rot),origin=(cx0,cy0)); pr=_repair(pr)
        if pr and not pr.is_empty:
            rp=_bisect(pr,n,'x',depth+1); pieces=[]
            for p in rp:
                pb=affinity.rotate(p,np.degrees(rot),origin=(cx0,cy0)); pb=_repair(pb); pg=_as_poly(pb)
                if pg and pg.area>1e-10: pieces.append(pg)
            clipped=[]; rem=_repair(poly)
            for i,p in enumerate(sorted(pieces,key=lambda x:x.centroid.x)):
                if rem is None or rem.is_empty: break
                if i==len(pieces)-1:
                    lp=_as_poly(_repair(rem))
                    if lp and lp.area>1e-10: clipped.append(_close_poly(lp))
                else:
                    try:
                        c=_as_poly(_repair(p.intersection(rem)))
                        if c and c.area>1e-10: clipped.append(_close_poly(c)); rem=_repair(rem.difference(c))
                    except: pass
            if clipped: return clipped
    nl=n//2; nr=n-nl; frac=nl/n
    def _cut(cax):
        lo,hi=(minx,maxx) if cax=='x' else (miny,maxy); ta=poly.area*frac; bm=(lo+hi)/2
        for _ in range(80):
            m=(lo+hi)/2
            try:
                b=_sbox(minx-1,miny-1,m,maxy+1) if cax=='x' else _sbox(minx-1,miny-1,maxx+1,m)
                lp=_repair(poly.intersection(b)); got=lp.area if lp else 0.0
            except: return None
            if abs(got-ta)/(ta+1e-12)<5e-4: bm=m; break
            if got<ta: lo=m
            else: hi=m
            bm=m
        try:
            b=_sbox(minx-1,miny-1,bm,maxy+1) if cax=='x' else _sbox(minx-1,miny-1,maxx+1,bm)
            lp=_repair(poly.intersection(b)); rp=_repair(poly.difference(lp))
            la=_as_poly(lp); ra=_as_poly(rp)
            if la and ra and la.area>1e-10 and ra.area>1e-10: return la,ra
        except: pass
        return None
    def _asp(p):
        b=p.bounds; w,h=b[2]-b[0],b[3]-b[1]; s=min(w,h) or 1e-9; return max(w,h)/s
    best=None
    for cax in (['x','y'] if axis is None else [axis]):
        cut=_cut(cax)
        if cut is None: continue
        lp,rp=cut; sc=max(_asp(lp),_asp(rp))
        if best is None or sc<best[0]: best=(sc,lp,rp)
    if best is None: return [_close_poly(poly)]
    _,lp,rp=best
    try:
        lp=_as_poly(_repair(lp.intersection(poly))) or lp
        rp=_as_poly(_repair(rp.intersection(poly))) or rp
    except: pass
    lp=_close_poly(_repair(lp) or lp); rp=_close_poly(_repair(rp) or rp)
    return _bisect(lp,nl,None,depth+1)+_bisect(rp,nr,None,depth+1)

def _subdivide_bisect(poly, n, area_tol_ha=0.3):
    poly=_repair(poly)
    if poly is None or poly.is_empty: return []
    pieces=_bisect(poly,n)
    valid=[_close_poly(p) for p in pieces if p and not p.is_empty and p.area>1e-10]
    if not valid: return [_close_poly(poly)]
    while len(valid)>n:
        si=min(range(len(valid)),key=lambda i:valid[i].area); sl=valid.pop(si)
        bj,bl=0,-1.0
        for j,v in enumerate(valid):
            try: s=sl.intersection(v).length
            except: s=0.0
            if s>bl: bl=s; bj=j
        mg=_as_poly(_repair(unary_union([valid[bj],sl])))
        if mg: valid[bj]=_close_poly(mg)
    while len(valid)<n:
        li=max(range(len(valid)),key=lambda i:valid[i].area); big=valid.pop(li)
        sub=[_close_poly(s) for s in _bisect(big,2) if s and s.area>1e-10]
        if len(sub)==2: valid.extend(sub)
        else: valid.append(big); break
    try:
        covered=_repair(unary_union(valid)); gap=_repair(poly.difference(covered))
        if gap and not gap.is_empty and gap.area>1e-8:
            bi,bl=0,-1.0
            for i,v in enumerate(valid):
                try: s=gap.buffer(1e-4).intersection(v).length
                except: s=0.0
                if s>bl: bl=s; bi=i
            mg=_as_poly(_repair(unary_union([valid[bi],gap])))
            if mg: valid[bi]=_close_poly(mg)
    except: pass
    valid = [_close_poly(_repair(p) or p) for p in valid]
    # ── Strict tolerance enforcement: iteratively redistribute area ──────
    valid = _enforce_area_tolerance(valid, poly, n, area_tol_ha)
    return valid

def _enforce_area_tolerance(pieces, orig_poly, n, tol_ha):
    """Iteratively shift boundaries to enforce that every piece is within
    ±tol_ha of the ideal area. Runs up to 6 passes."""
    if not pieces or n < 2: return pieces
    tol_m2 = tol_ha * 10000
    ideal   = orig_poly.area / n
    for _pass in range(6):
        improved = False
        for i in range(len(pieces)):
            for j in range(len(pieces)):
                if i == j: continue
                ai = pieces[i].area; aj = pieces[j].area
                if abs(ai - ideal) <= tol_m2 and abs(aj - ideal) <= tol_m2:
                    continue
                # i is too big, j is too small → transfer strip from i→j
                if ai > ideal + tol_m2 and aj < ideal - tol_m2:
                    try:
                        shared = pieces[i].boundary.intersection(pieces[j].boundary)
                        if shared.is_empty or shared.length < 0.5: continue
                        transfer = min(ai - ideal, ideal - aj)
                        strip_w  = transfer / max(shared.length, 1.0)
                        buf = shared.buffer(max(strip_w, 0.5), cap_style=2)
                        strip = _as_poly(_repair(pieces[i].intersection(buf)))
                        if strip is None or strip.is_empty: continue
                        ni = _as_poly(_repair(pieces[i].difference(strip)))
                        nj = _as_poly(_repair(pieces[j].union(strip)))
                        if ni and nj and ni.area > ideal*0.4 and nj.area > ideal*0.4:
                            pieces[i] = _close_poly(ni)
                            pieces[j] = _close_poly(nj)
                            improved = True
                    except: pass
        if not improved: break
    return pieces

def _subdivide_ba(poly, n, area_tol_ha=0.3):
    pieces=_subdivide_bisect(poly,n,area_tol_ha); ideal=poly.area/max(n,1)
    for _ in range(4):
        changed=False
        for i in range(len(pieces)-1):
            try:
                sh=pieces[i].exterior.intersection(pieces[i+1].exterior)
                if sh.is_empty or sh.length<1: continue
                buf=sh.buffer(max(sh.length*0.015,1.0))
                pi2=_as_poly(_repair(pieces[i].union(buf.intersection(pieces[i+1]))))
                pj2=_as_poly(_repair(pieces[i+1].difference(buf.intersection(pieces[i+1]))))
                if pi2 and pj2 and pi2.area>1e-6 and pj2.area>1e-6:
                    pieces[i]=_close_poly(pi2); pieces[i+1]=_close_poly(pj2); changed=True
            except: pass
        if not changed: break
    return pieces

def _subdivide_voronoi(poly, n, max_iter=25):
    try:
        import random
        minx,miny,maxx,maxy=poly.bounds; seeds=[]
        att=0
        while len(seeds)<n and att<n*300:
            p=Point(random.uniform(minx,maxx),random.uniform(miny,maxy))
            if poly.contains(p): seeds.append(p)
            att+=1
        if len(seeds)<n:
            cn=int(math.ceil(math.sqrt(n*((maxx-minx)/(maxy-miny+1e-9)))))
            rn=int(math.ceil(n/cn))+1; seeds=[]
            for ri in range(rn):
                for ci in range(cn):
                    p=Point(minx+(ci+0.5)*(maxx-minx)/cn, miny+(ri+0.5)*(maxy-miny)/rn)
                    if poly.contains(p): seeds.append(p)
            seeds=seeds[:n]
        if len(seeds)<2: raise ValueError("Not enough seeds.")
        for _ in range(max_iter):
            mp=MultiPoint(seeds); diagram=voronoi_diagram(mp,envelope=poly.buffer(1))
            cells=[]
            for region in diagram.geoms:
                cl=_as_poly(_repair(region.intersection(poly)))
                if cl and cl.area>1e-10: cells.append(cl)
            if not cells: break
            cells.sort(key=lambda c:c.area,reverse=True); cells=cells[:n]
            new_seeds=[c.centroid for c in cells]
            if all(ns.distance(s)<1e-3 for ns,s in zip(new_seeds,seeds[:len(new_seeds)])): break
            seeds=new_seeds
            if len(seeds)<n: break
        cells=[_close_poly(_repair(c)) for c in cells if c and not c.is_empty]
        covered=_repair(unary_union(cells)); gap=_repair(poly.difference(covered))
        if gap and not gap.is_empty and gap.area>1e-8 and cells:
            bi,bl=0,-1
            for i,c in enumerate(cells):
                try: sl=gap.buffer(1e-4).intersection(c).length
                except: sl=0
                if sl>bl: bl=sl; bi=i
            mg=_as_poly(_repair(cells[bi].union(gap)))
            if mg: cells[bi]=_close_poly(mg)
        return cells if cells else [_close_poly(poly)]
    except: return _subdivide_bisect(poly,n)

def _subdivide_grid(poly, n):
    cn=int(math.ceil(math.sqrt(n))); rn=int(math.ceil(n/cn))
    minx,miny,maxx,maxy=poly.bounds; cw=(maxx-minx)/cn; rh=(maxy-miny)/rn
    cells=[]
    for ri in range(rn):
        for ci in range(cn):
            box=_sbox(minx+ci*cw,miny+ri*rh,minx+(ci+1)*cw,miny+(ri+1)*rh)
            cl=_as_poly(_repair(box.intersection(poly)))
            if cl and cl.area>1e-10: cells.append(_close_poly(cl))
    while len(cells)>n:
        si=min(range(len(cells)),key=lambda i:cells[i].area); sl=cells.pop(si)
        bj,bl=0,-1.0
        for j,c in enumerate(cells):
            try: s=sl.intersection(c).length
            except: s=0
            if s>bl: bl=s; bj=j
        mg=_as_poly(_repair(cells[bj].union(sl)))
        if mg: cells[bj]=_close_poly(mg)
    while len(cells)<n:
        li=max(range(len(cells)),key=lambda i:cells[i].area); sub=_bisect(cells.pop(li),2)
        cells.extend([_close_poly(s) for s in sub if s and s.area>1e-10] or [cells[0]])
    return cells[:n]

def _subdivide(poly, n, method="bisect", area_tol_ha=0.3):
    poly=_repair(poly)
    if poly is None or poly.is_empty: return []
    try:
        if method=="voronoi":   return _subdivide_voronoi(poly,n)
        elif method=="grid":    return _subdivide_grid(poly,n)
        elif method=="ba":      return _subdivide_ba(poly,n,area_tol_ha)
        else:                   return _subdivide_bisect(poly,n,area_tol_ha)
    except: return _subdivide_bisect(poly,n,area_tol_ha)

def _extract_div_pts(pieces, fname):
    recs=[]; sn=1
    for i,p in enumerate(pieces,1):
        cid=f"Comp_{i:03d}"; p=_close_poly(_repair(p))
        if p is None or p.is_empty: continue
        coords=list(p.exterior.coords)
        if len(coords)>1 and coords[0]==coords[-1]: coords=coords[:-1]
        el=[((coords[(j+1)%len(coords)][0]-coords[j][0])**2+(coords[(j+1)%len(coords)][1]-coords[j][1])**2)**.5 for j in range(len(coords))]
        av=(sum(el)/len(el)) if el else 1.0; mt=av*1.5
        for j,(cx,cy) in enumerate(coords):
            recs.append({"SN":sn,"Forest":fname,"Comp_ID":cid,"Type":"Vertex","X":round(cx,4),"Y":round(cy,4)}); sn+=1
            nx,ny=coords[(j+1)%len(coords)]; e=((nx-cx)**2+(ny-cy)**2)**.5
            if e>mt: recs.append({"SN":sn,"Forest":fname,"Comp_ID":cid,"Type":"Midpoint","X":round((cx+nx)/2,4),"Y":round((cy+ny)/2,4)}); sn+=1
    return recs

def _save_compartments(pieces, fname, crs, save_dir):
    os.makedirs(save_dir,exist_ok=True)
    pieces=[_close_poly(_repair(p)) for p in pieces if p and not p.is_empty]
    pieces=[p for p in pieces if p and not p.is_empty]
    total=sum(p.area for p in pieces); recs=[]; lrecs=[]; ptrecs=[]
    for i,p in enumerate(pieces,1):
        cid=f"Comp_{i:03d}"; ah=round(p.area/10000,4); pm=round(p.length,4)
        pct=round(p.area/total*100,2) if total>0 else 0
        recs.append({"Forest":fname,"Comp_ID":cid,"Area_ha":ah,"Perim_m":pm,"Pct_Area":pct,"geometry":p})
        ext=list(p.exterior.coords)
        if ext[0]!=ext[-1]: ext.append(ext[0])
        lrecs.append({"Forest":fname,"Comp_ID":cid,"geometry":LineString(ext)})
        ptrecs.append({"Forest":fname,"Comp_ID":cid,"Area_ha":ah,"Pct_Area":pct,"geometry":p.centroid})
    pg=gpd.GeoDataFrame(recs,crs=crs); lg=gpd.GeoDataFrame(lrecs,crs=crs); ptg=gpd.GeoDataFrame(ptrecs,crs=crs)
    pg=_enforce_poly_gdf(pg); pfx=_safe_dn(fname)
    if not pg.empty:  pg.to_file(os.path.join(save_dir,f"{pfx}_compartment_polygon.shp"))
    if not lg.empty:  lg.to_file(os.path.join(save_dir,f"{pfx}_compartment_line.shp"))
    if not ptg.empty: ptg.to_file(os.path.join(save_dir,f"{pfx}_compartment_point.shp"))
    pd.DataFrame([{k:v for k,v in r.items() if k!="geometry"} for r in recs]).to_excel(
        os.path.join(save_dir,f"{pfx}_compartment_summary.xlsx"),index=False)
    dp=_extract_div_pts(pieces,fname)
    if dp:
        ddf=pd.DataFrame(dp); ddf.to_excel(os.path.join(save_dir,f"{pfx}_division_points.xlsx"),index=False)
        gpd.GeoDataFrame(ddf,geometry=gpd.points_from_xy(ddf["X"],ddf["Y"]),crs=crs).to_file(
            os.path.join(save_dir,f"{pfx}_division_points.shp"))
    return pg,lg,ptg

def _df_to_poly(df,xc,yc,oc):
    if oc and oc in df.columns: df=df.sort_values(oc)
    coords=list(zip(df[xc],df[yc]))
    if len(coords)<3: raise ValueError("Need ≥3 points.")
    coords.append(coords[0]); return _close_poly(_repair(Polygon(coords)))

def _load_polys_from_zip(file, target_shp, crs, fcol=None):
    folder=os.path.join(UPLOAD,str(uuid.uuid4())); os.makedirs(folder,exist_ok=True)
    zp=os.path.join(folder,"i.zip"); file.save(zp)
    with zipfile.ZipFile(zp) as z: z.extractall(folder)
    shps=[os.path.join(r,f) for r,_,fs in os.walk(folder) for f in fs if f.endswith(".shp")]
    if not shps: raise ValueError("No .shp in ZIP.")
    sp=shps[0]
    if target_shp:
        tn=os.path.basename(target_shp)
        for s in shps:
            if os.path.basename(s)==tn: sp=s; break
    gdf=gpd.read_file(sp)
    if gdf.empty: raise ValueError("Shapefile empty.")
    gdf=gdf.set_crs(crs) if gdf.crs is None else gdf.to_crs(crs)
    nc=None
    if fcol:
        for c in gdf.columns:
            if c.lower()==fcol.lower(): nc=c; break
    if nc is None:
        for cand in ["Forest","forest","Name","name","NAME","Label","label","ID","id"]:
            if cand in gdf.columns: nc=cand; break
    if nc is None:
        for c in gdf.columns:
            if c=="geometry": continue
            if gdf[c].dtype==object: nc=c; break
    results=[]
    for i,row in gdf.iterrows():
        geom=row.geometry
        if geom is None or geom.is_empty: continue
        fn=str(row[nc]) if nc else f"Feature_{i+1}"
        if geom.geom_type=="Polygon": pls=[_repair(geom)]
        elif geom.geom_type=="MultiPolygon": pls=[_repair(g) for g in geom.geoms]
        else: pls=[_repair(g) for g in geom.geoms if g.geom_type=="Polygon"] if hasattr(geom,"geoms") else []
        if len(pls)>1: pls=[_repair(unary_union(pls))]
        for p in pls:
            if p and p.area>1e-6: results.append((fn,_close_poly(p)))
    if not results: raise ValueError("No polygon geometries found.")
    return results,shps

def group_e(file_or_df, crs, out, mapping=None, e_mode="A", n_compartments=4,
            is_zip=False, fcol=None, area_tol_ha=0.3, method="bisect", run_id=None):
    n_compartments=max(2,min(15,int(n_compartments)))
    ap,al,apts=[],[],[]
    if is_zip:
        ts=(mapping or {}).get("target_shp")
        features,_=_load_polys_from_zip(file_or_df,ts,crs,fcol)
        for idx,(fn,poly) in enumerate(features):
            pct_start = 20 + int(60*idx/max(len(features),1))
            if run_id: _prog(run_id,f"[{idx+1}/{len(features)}] Subdividing {fn}…", pct_start)
            pieces=_subdivide(poly,n_compartments,method,area_tol_ha)
            areas=[round(p.area/10000,2) for p in pieces if p]
            ideal=round(poly.area/10000/max(n_compartments,1),2)
            diff=max((abs(a-ideal) for a in areas),default=0)
            if run_id: _prog(run_id,
                f"[{idx+1}/{len(features)}] {fn}: {len(pieces)} parts ✓  ideal={ideal}ha  max_diff={diff:.2f}ha",
                pct_start+5)
            fd=os.path.join(out,_safe_dn(fn)) if len(features)>1 else out
            pg,lg,ptg=_save_compartments(pieces,fn,crs,fd)
            ap.append(pg); al.append(lg); apts.append(ptg)
    else:
        df=file_or_df; df=normalize_order(df)
        xc=safe_col(df,mapping,"X","X"); yc=safe_col(df,mapping,"Y","Y")
        oc=safe_col(df,mapping,"Order","Order"); fc=safe_col(df,mapping,"Forest","Forest")
        if not xc: raise ValueError("X column not found.")
        if not yc: raise ValueError("Y column not found.")
        if e_mode=="B" and not fc: raise ValueError("Forest column required.")
        if e_mode=="A":
            fn=(mapping or {}).get("forest") or "FOREST"
            if run_id: _prog(run_id,"Building & subdividing polygon…",15)
            poly=_df_to_poly(df,xc,yc,oc); pieces=_subdivide(poly,n_compartments,method,area_tol_ha)
            pg,lg,ptg=_save_compartments(pieces,fn,crs,out)
            ap.append(pg); al.append(lg); apts.append(ptg)
        else:
            groups=list(df.groupby(fc))
            for idx,(f,fg) in enumerate(groups):
                if run_id: _prog(run_id,f"Processing {f}…",15+int(65*idx/max(len(groups),1)))
                try:
                    poly=_df_to_poly(fg,xc,yc,oc)
                    pieces=_subdivide(poly,n_compartments,method,area_tol_ha)
                    fd=os.path.join(out,_safe_dn(str(f)))
                    pg,lg,ptg=_save_compartments(pieces,str(f),crs,fd)
                    ap.append(pg); al.append(lg); apts.append(ptg)
                except Exception as ex:
                    if run_id: _prog(run_id,f"Warning: {f} skipped — {ex}")
    if not ap: raise ValueError("No valid polygons built.")
    p=gpd.GeoDataFrame(pd.concat(ap,ignore_index=True),crs=crs)
    l=gpd.GeoDataFrame(pd.concat(al,ignore_index=True),crs=crs)
    pt=gpd.GeoDataFrame(pd.concat(apts,ignore_index=True),crs=crs)
    return p,l,pt

# ── GROUP F ──────────────────────────────────────────────────────────────────
def _bnd_from_df(df,mapping):
    df=normalize_order(df)
    xc=safe_col(df,mapping,"X","X"); yc=safe_col(df,mapping,"Y","Y")
    oc=safe_col(df,mapping,"Order","Order")
    if not xc: raise ValueError("X column not found.")
    if not yc: raise ValueError("Y column not found.")
    if oc: df=df.sort_values(oc)
    coords=list(zip(df[xc],df[yc]))
    if len(coords)<3: raise ValueError("Need ≥3 boundary points.")
    coords.append(coords[0]); return safe_polygon(coords)

# ═══════════════════════════════════════════════════════════════════════════════
# GROUP F — SLOPE ANALYSIS  (7-step correct workflow)
# Step 1: Rectangular DEM clip with 20% buffer around boundary polygon
# Step 2: Compute slope on the rectangular DEM  (Horn method)
# Step 3: Reclassify: 1=<19°  2=19-31°  3=31-45°  4=>45°
# Step 4: Raster → polygon (vectorize the classified raster)
# Step 5: Dissolve by gridcode
# Step 6: Clip dissolved RTP with our original boundary polygon
#          (intersection ensures no missing boundary pixels)
# Step 7: Calculate area table per slope class per compartment/forest
# ═══════════════════════════════════════════════════════════════════════════════

def _bnd_from_zip(zip_file, target_shp, src_crs, dem_crs):
    """Extract boundary shapefile from a ZIP, reproject to DEM CRS."""
    folder = os.path.join(UPLOAD, str(uuid.uuid4()))
    os.makedirs(folder, exist_ok=True)
    zp = os.path.join(folder, "b.zip")
    zip_file.save(zp)
    with zipfile.ZipFile(zp) as z:
        z.extractall(folder)
    shps = [os.path.join(r, f) for r, _, fs in os.walk(folder)
            for f in fs if f.endswith(".shp")]
    if not shps:
        raise ValueError("No .shp found in the boundary ZIP.")
    sp = shps[0]
    if target_shp:
        for s in shps:
            if os.path.basename(s) == os.path.basename(target_shp):
                sp = s; break
    gdf = gpd.read_file(sp)
    if gdf.empty:
        raise ValueError("Boundary shapefile is empty.")
    gdf = gdf.set_crs(src_crs) if gdf.crs is None else gdf
    gdf = gdf.to_crs(dem_crs)
    return _repair(gdf.unary_union), gdf

def group_f(boundary_file, dem_file, crs, out, mapping=None,
            boundary_is_zip=False, forest_name="FOREST",
            f_mode="A", comp_col_name=None, field_area_ha=None, run_id=None):
    """
    7-Step Group F workflow:
    1. Extract rectangular DEM (20% buffer around boundary polygon)
    2. Compute slope on the rectangular DEM (Horn method)
    3. Reclassify: 1=0-19°  2=19-31°  3=31-45°  4=>45°
    4. Raster → polygon (vectorise classified raster over entire rectangle)
    5. Dissolve by gridcode
    6. Clip dissolved RTP polygons with the original boundary polygon
       (NOT with the rectangle — this gives perfect boundary-edge accuracy)
    7. Calculate slope area table for each class / compartment
    """
    try:
        import rasterio
        from rasterio.mask import mask as rio_mask
        from rasterio.features import shapes as rio_shapes
        import scipy.ndimage as ndi
    except ImportError:
        raise ValueError("rasterio and scipy are required for Group F. "
                         "Install with: pip install rasterio scipy")

    os.makedirs(out, exist_ok=True)
    pfx = _safe_dn(forest_name)
    SN  = -9999.0

    # ── STEP 0: Save DEM to disk ──────────────────────────────────────────────
    if run_id: _prog(run_id, "Step 0 — Loading DEM…", 5)
    dem_path = os.path.join(UPLOAD, f"{uuid.uuid4()}_dem.tif")
    try:
        dem_file.save(dem_path)
    except Exception as e:
        raise ValueError(f"Could not save DEM file: {e}")
    if not os.path.exists(dem_path) or os.path.getsize(dem_path) < 100:
        raise ValueError("DEM file is empty or could not be written to disk.")

    with rasterio.open(dem_path) as _src:
        dem_crs   = _src.crs
        dem_nodata = _src.nodata
        dem_profile = _src.profile.copy()

    # ── STEP 1: Load boundary, reproject to DEM CRS ──────────────────────────
    if run_id: _prog(run_id, "Step 1 — Loading & reprojecting boundary…", 10)
    if boundary_is_zip:
        ts = (mapping or {}).get("target_shp")
        bpoly, bgdf = _bnd_from_zip(boundary_file, ts, crs, str(dem_crs))
    else:
        try:
            df   = read_input(boundary_file)
            bpoly = _bnd_from_df(df, mapping)
            bgdf  = gpd.GeoDataFrame(
                [{"Forest": forest_name, "geometry": bpoly}], crs=crs
            ).to_crs(str(dem_crs))
            bpoly = bgdf.unary_union
        except Exception as e:
            raise ValueError(f"Failed to read/project boundary: {e}")

    if bpoly is None or bpoly.is_empty:
        raise ValueError("Boundary polygon is empty after loading — check input file.")

    # Validate boundary intersects DEM extent
    with rasterio.open(dem_path) as _src:
        from shapely.geometry import box as _box
        dem_bounds_poly = _box(*_src.bounds)
    bpoly_wgs = gpd.GeoDataFrame([{"geometry":bpoly}], crs=str(dem_crs)).to_crs("EPSG:4326").unary_union
    dem_wgs   = gpd.GeoDataFrame([{"geometry":dem_bounds_poly}], crs=str(dem_crs)).to_crs("EPSG:4326").unary_union
    if not bpoly_wgs.intersects(dem_wgs):
        raise ValueError("Boundary polygon does not overlap the DEM extent. "
                         "Make sure your DEM covers the boundary area.")

    # ── STEP 1: Build rectangle = bbox + 20% buffer ───────────────────────────
    b  = bpoly.bounds
    bx = (b[2] - b[0]) * 0.20
    by = (b[3] - b[1]) * 0.20
    rect_poly = _sbox(b[0]-bx, b[1]-by, b[2]+bx, b[3]+by)

    nd = dem_nodata if dem_nodata is not None else SN

    if run_id: _prog(run_id, "Step 1 — Clipping rectangular DEM (20% buffer)…", 18)
    try:
        with rasterio.open(dem_path) as _src:
            ra, rt = rio_mask(
                _src, [rect_poly.__geo_interface__],
                crop=True, filled=True, nodata=nd, all_touched=True
            )
        rdm = ra[0].astype(np.float32)
        rdm[rdm == nd] = np.nan
    except Exception as e:
        # Fallback: read entire DEM (boundary may be near edge)
        log.warning(f"Rect clip failed ({e}), reading full DEM")
        with rasterio.open(dem_path) as _src:
            ra  = _src.read(1).astype(np.float32)
            rt  = _src.transform
        nd_val = nd if nd is not None else SN
        rdm = ra.copy()
        rdm[rdm == nd_val] = np.nan

    rx = abs(rt.a)   # pixel width  in metres
    ry = abs(rt.e)   # pixel height in metres
    if rx < 0.01 or ry < 0.01:
        raise ValueError(f"DEM pixel size is too small ({rx}m × {ry}m). "
                         "Check that DEM is in a projected CRS (e.g. UTM).")

    # ── STEP 2: Compute slope (Horn method) ───────────────────────────────────
    if run_id: _prog(run_id, "Step 2 — Computing slope (Horn method)…", 28)
    valid  = ~np.isnan(rdm)
    filled = np.nan_to_num(rdm, nan=0.0)
    dzdx   = ndi.sobel(filled, axis=1) / (8.0 * rx)
    dzdy   = ndi.sobel(filled, axis=0) / (8.0 * ry)
    slope  = np.degrees(np.arctan(np.sqrt(dzdx**2 + dzdy**2))).astype(np.float32)
    # Erode 1-pixel border (Sobel artifact) to nodata
    fnv = ndi.binary_erosion(valid, structure=np.ones((3,3), bool), border_value=0)
    slope[~fnv] = SN

    # Save rectangular slope raster
    rp = dem_profile.copy()
    rp.update(dtype="float32", nodata=SN, count=1,
              height=slope.shape[0], width=slope.shape[1], transform=rt)
    sp_path = os.path.join(out, f"{pfx}_slope_rect.tif")
    with rasterio.open(sp_path, "w", **rp) as dst:
        dst.write(slope, 1)
    if run_id: _prog(run_id, f"Step 2 — Slope raster saved ({slope.shape[1]}×{slope.shape[0]}px)", 35)

    # ── STEP 3: Reclassify slope ──────────────────────────────────────────────
    #   1 = 0–19°  (Gentle)    2 = 19–31° (Moderate)
    #   3 = 31–45° (Steep)     4 = >45°   (Very Steep)
    if run_id: _prog(run_id, "Step 3 — Reclassifying slope into 4 classes…", 40)
    vm  = (slope != SN) & ~np.isnan(slope)
    cls = np.zeros_like(slope, dtype=np.uint8)
    cls[vm & (slope <  19)]                        = 1
    cls[vm & (slope >= 19) & (slope <= 31)]        = 2
    cls[vm & (slope >  31) & (slope <= 45)]        = 3
    cls[vm & (slope >  45)]                        = 4
    cp = rp.copy(); cp.update(dtype="uint8", nodata=0)
    cls_path = os.path.join(out, f"{pfx}_class_rect.tif")
    with rasterio.open(cls_path, "w", **cp) as dst:
        dst.write(cls, 1)

    # ── STEP 4: Raster → Polygon (entire rectangle) ──────────────────────────
    if run_id: _prog(run_id, "Step 4 — Vectorising classified raster…", 50)
    rtp = []
    with rasterio.open(cls_path) as _src:
        ca, ct = _src.read(1), _src.transform
        mask_valid = (ca > 0).astype(np.uint8)
        for shp, val in rio_shapes(ca, mask=mask_valid, transform=ct):
            cid = int(val)
            if cid == 0: continue
            try:
                coords = shp["coordinates"]
                ext    = coords[0]
                holes  = coords[1:] if len(coords) > 1 else None
                geom   = _repair(Polygon(ext, holes=holes))
                if geom and not geom.is_empty and geom.area > 1e-10:
                    rtp.append({"gridcode": cid, "geometry": geom})
            except Exception:
                continue

    if not rtp:
        raise ValueError(
            "Raster-to-polygon produced no features. "
            "Check that the DEM overlaps the boundary and is in a projected CRS."
        )
    rtp_gdf = gpd.GeoDataFrame(rtp, crs=str(dem_crs))
    _rtp_save = _enforce_poly_gdf(rtp_gdf)
    if not _rtp_save.empty:
        _rtp_save.to_file(os.path.join(out, f"{pfx}_rtp_raw.shp"))
    if run_id: _prog(run_id, f"Step 4 — {len(rtp)} slope polygons vectorised", 56)

    # ── STEP 5: Dissolve by gridcode ─────────────────────────────────────────
    if run_id: _prog(run_id, "Step 5 — Dissolving by gridcode…", 62)
    try:
        dissolved = rtp_gdf.dissolve(by="gridcode", as_index=False)
        dissolved["gridcode"] = dissolved["gridcode"].astype(int)
    except Exception as e:
        log.warning(f"dissolve failed ({e}), using raw rtp")
        dissolved = rtp_gdf.copy()
    _dis_save = _enforce_poly_gdf(dissolved)
    if not _dis_save.empty:
        _dis_save.to_file(os.path.join(out, f"{pfx}_rtp_dissolved.shp"))
    if run_id: _prog(run_id, f"Step 5 — Dissolved into {len(dissolved)} gridcode classes", 66)

    # ── STEP 6: Clip dissolved RTP with the original boundary polygon ─────────
    #   We clip dissolved (rect-level RTP) with the actual boundary polygon.
    #   This is the KEY step that gives perfect boundary-edge accuracy:
    #   - The rectangle RTP ensures no pixels near the boundary edge are lost
    #   - The clip then trims exactly to the boundary shape
    if run_id: _prog(run_id, "Step 6 — Clipping dissolved polygons to boundary…", 70)

    # 4-class definitions (matching step 3)
    class_defs = {
        1: ("0-19 degree",  "Gentle",     "#2e8b57"),
        2: ("19-31 degree", "Moderate",   "#90ee90"),
        3: ("31-45 degree", "Steep",      "#ffd700"),
        4: ("45> degree",   "Very Steep", "#e74c3c"),
    }

    # Determine which polygons to analyse (single / multi-forest / compartments)
    comp_polygons = []
    if f_mode == "A":
        comp_polygons = [(forest_name, bpoly)]
    else:
        grp_col = None
        if comp_col_name:
            for c in bgdf.columns:
                if c.lower() == comp_col_name.lower(): grp_col = c; break
        if grp_col is None: grp_col = _find_col(bgdf, _FA if f_mode == "B" else _CA)
        if grp_col is None and f_mode == "E": grp_col = _find_col(bgdf, _FA)
        if grp_col:
            for val, grp in bgdf.groupby(grp_col):
                up = _repair(grp.unary_union)
                if up and not up.is_empty:
                    comp_polygons.append((str(val), up))
        if not comp_polygons or all(cp is None or (hasattr(cp,"is_empty") and cp.is_empty) for _,cp in comp_polygons):
            comp_polygons = [(forest_name, bpoly)]

    def _clip_group(clip_poly, label):
        """Clip dissolved slope polygons with one boundary polygon, return records."""
        vrecs = []; total = 0.0
        for _, drow in dissolved.iterrows():
            cid   = int(drow["gridcode"])
            dgeom = _repair(drow.geometry)
            if dgeom is None or dgeom.is_empty: continue
            if cid not in class_defs: continue
            try:
                clipped = _repair(dgeom.intersection(clip_poly))
            except Exception as e:
                log.debug(f"intersection error cid={cid}: {e}"); continue
            if clipped is None or clipped.is_empty: continue
            # Normalise to (Multi)Polygon
            if clipped.geom_type in ("Polygon", "MultiPolygon"):
                pg = _close_poly(clipped)
            elif hasattr(clipped, "geoms"):
                parts = [_as_poly(_repair(g)) for g in clipped.geoms
                         if g.geom_type in ("Polygon", "MultiPolygon")]
                parts = [p for p in parts if p and p.area > 1e-10]
                if not parts: continue
                pg = _close_poly(max(parts, key=lambda x: x.area))
            else:
                pg = _as_poly(clipped)
            if pg is None or pg.is_empty or pg.area < 1e-10: continue
            ah = round(pg.area / 10000, 4); total += ah
            vrecs.append({
                "Label":       label,
                "Class":       cid,
                "Slope_Range": class_defs[cid][0],
                "Description": class_defs[cid][1],
                "Area_ha":     ah,
                "geometry":    pg,
            })
        total = max(total, 1e-6)
        rows = [{
            "Label":       vr["Label"],
            "Class":       vr["Class"],
            "Slope_Range": vr["Slope_Range"],
            "Description": vr["Description"],
            "Area_ha":     vr["Area_ha"],
            "Pct_Area":    round(vr["Area_ha"] / total * 100, 2),
            "Total_ha":    round(total, 4),
        } for vr in vrecs]
        return rows, vrecs

    all_sum, all_vec, per_grp = [], [], {}
    for i, (lb, cp2) in enumerate(comp_polygons):
        if run_id:
            pct = 70 + int(16 * i / max(len(comp_polygons), 1))
            _prog(run_id, f"Step 6 — Clipping {lb} ({i+1}/{len(comp_polygons)})…", pct)
        rows, vrecs = _clip_group(cp2, lb)
        all_sum.extend(rows); all_vec.extend(vrecs); per_grp[lb] = rows

    if not all_vec:
        raise ValueError(
            "Clipping produced no slope polygons. "
            "Check that the boundary polygon overlaps the DEM and is in the correct CRS."
        )

    # ── Optional field-area recalibration ────────────────────────────────────
    if field_area_ha and field_area_ha > 0:
        tc  = sum(r["Area_ha"] for r in all_sum)
        fac = field_area_ha / max(tc, 1e-6)
        for r in all_sum:
            r["Recal_ha"]    = round(r["Area_ha"] * fac, 4)
            r["Cal_Factor"]  = round(fac, 6)
    else:
        for r in all_sum:
            r["Recal_ha"]    = None
            r["Cal_Factor"]  = None

    # ── STEP 7: Save all outputs ──────────────────────────────────────────────
    if run_id: _prog(run_id, "Step 7 — Saving shapefiles and Excel…", 88)
    vcrs = str(dem_crs)

    # Slope polygon shapefile (clipped to boundary)
    if all_vec:
        vgdf = gpd.GeoDataFrame(all_vec, crs=vcrs)
        vgdf = _enforce_poly_gdf(vgdf)
        if not vgdf.empty:
            vgdf.to_file(os.path.join(out, f"{pfx}_slope_polygon.shp"))
    else:
        vgdf = gpd.GeoDataFrame(columns=["Label","Class","Slope_Range",
                                          "Description","Area_ha","geometry"], crs=vcrs)

    # Boundary shapefile
    try:
        bgdf_save = _enforce_poly_gdf(bgdf)
        if not bgdf_save.empty:
            bgdf_save.to_file(os.path.join(out, f"{pfx}_boundary_polygon.shp"))
    except Exception as e:
        log.warning(f"bgdf save: {e}")

    # Clipped slope raster (for first polygon — visualization)
    try:
        main_poly = comp_polygons[0][1]
        with rasterio.open(sp_path) as _src:
            fc2, ft2 = rio_mask(
                _src, [main_poly.__geo_interface__],
                crop=True, filled=True, nodata=SN, all_touched=True
            )
        fc2  = fc2[0].astype(np.float32)
        cp2_ = rp.copy(); cp2_.update(height=fc2.shape[0], width=fc2.shape[1], transform=ft2)
        with rasterio.open(os.path.join(out, f"{pfx}_slope_clipped.tif"), "w", **cp2_) as dst:
            dst.write(fc2, 1)
        # Reclassified clipped raster
        vm2  = (fc2 != SN) & ~np.isnan(fc2)
        ca2  = np.zeros_like(fc2, dtype=np.uint8)
        ca2[vm2 & (fc2 <  19)]                       = 1
        ca2[vm2 & (fc2 >= 19) & (fc2 <= 31)]         = 2
        ca2[vm2 & (fc2 >  31) & (fc2 <= 45)]         = 3
        ca2[vm2 & (fc2 >  45)]                        = 4
        cp3_ = cp2_.copy(); cp3_.update(dtype="uint8", nodata=0)
        with rasterio.open(os.path.join(out, f"{pfx}_slope_classes.tif"), "w", **cp3_) as dst:
            dst.write(ca2, 1)
    except Exception as e:
        log.warning(f"Clipped raster save error: {e}")

    # Excel slope summary
    sdf = pd.DataFrame(all_sum)
    ep  = os.path.join(out, f"{pfx}_slope_summary.xlsx")
    if f_mode == "A":
        sdf.drop(columns=["Label"], errors="ignore").to_excel(ep, index=False)
    else:
        try:
            with pd.ExcelWriter(ep, engine="openpyxl") as wr:
                sdf.to_excel(wr, sheet_name="All_Groups", index=False)
                for lb, grs in per_grp.items():
                    if not grs: continue
                    gdf2 = pd.DataFrame(grs)
                    th   = sum(r["Area_ha"] for r in grs)
                    tr   = {
                        "Label": lb, "Class": "", "Slope_Range": "TOTAL",
                        "Description": "", "Area_ha": round(th, 4),
                        "Pct_Area": 100.0, "Total_ha": round(th, 4),
                    }
                    if grs and grs[0].get("Recal_ha") is not None:
                        tr["Recal_ha"]   = round(sum(r.get("Recal_ha",0) for r in grs), 4)
                        tr["Cal_Factor"] = grs[0].get("Cal_Factor", "")
                    pd.concat([gdf2, pd.DataFrame([tr])], ignore_index=True).to_excel(
                        wr, sheet_name=str(lb)[:31], index=False)
        except Exception as e:
            log.warning(f"Excel multi-sheet error ({e}), saving flat")
            sdf.to_excel(ep, index=False)

    if run_id: _prog(run_id, "Group F complete.", 95)
    return all_sum, vgdf, bgdf, f_mode, per_grp
    return all_sum,vgdf,bgdf,f_mode,per_grp

# ── PREVIEW FUNCTIONS ────────────────────────────────────────────────────────
# Rich distinct colors matching reference image 3 (C1S1=blue, C1S2=orange, etc.)
_COMP_COLORS_RICH = [
    "#2196F3","#FF9800","#4CAF50","#9C27B0","#F48FB1",
    "#607D8B","#CDDC39","#00BCD4","#FF5722","#795548",
    "#E91E63","#009688","#FFC107","#3F51B5","#8BC34A"
]

def preview_compartments(poly_gdf, path, title="", legend_title="Legend", label_col="Comp_ID"):
    """Render compartment map like reference image 3: solid colored fills, dark border, legend with ha."""
    fig, ax = plt.subplots(figsize=(A4W, A4H), dpi=DPI)
    fig.patch.set_facecolor("white"); ax.set_facecolor("white")
    handles = []

    lc_use = label_col if (label_col and label_col in poly_gdf.columns) else (
             "Comp_ID" if "Comp_ID" in poly_gdf.columns else None)

    for i, (_, row) in enumerate(poly_gdf.iterrows()):
        geom = row.geometry
        if geom is None or geom.is_empty: continue
        if geom.geom_type == "Polygon":
            ext = list(geom.exterior.coords)
            if ext[0] != ext[-1]: ext.append(ext[0])
            geom = _repair(Polygon(ext)) or Polygon(ext)
        color = _COMP_COLORS_RICH[i % len(_COMP_COLORS_RICH)]
        cid = row.get(lc_use or "Comp_ID", f"Comp_{i+1:03d}") if lc_use or "Comp_ID" in row.index else f"Comp_{i+1:03d}"
        ah  = row.get("Area_ha", "")
        lbl = f"{cid} ({ah:.2f} ha)" if isinstance(ah, float) else str(cid)
        gpd.GeoDataFrame([{"geometry": geom}], crs=poly_gdf.crs).plot(
            ax=ax, facecolor=color, edgecolor="#111111", linewidth=1.6)
        handles.append(mpatches.Patch(facecolor=color, edgecolor="#111111", linewidth=1.0, label=lbl))

    # Dark bold compartment labels centred on each polygon
    if lc_use:
        _label_feat(ax, poly_gdf, lc_use, fs=9, color="#111111")

    _style_ax(ax); _north_arrow(ax); _scale_bar(ax)
    _add_legend(ax, handles, legend_title=legend_title or "Legend", loc="lower right")
    ax.set_title(title.strip() or "Compartment Division Map",
                 fontsize=12, fontweight="bold", color="#0d1f17", pad=10)
    plt.tight_layout(pad=0.4, rect=[0, 0, 0.80, 0.97])
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white"); plt.close(fig)

def preview(poly_gdf, line_gdf, pts_gdf, path, pc="blue", lc="black", ptc="red",
            label_col=None, label_pts_gdf=None, area_ha=None, title="",
            legend_title="Legend", user_label_col=None):
    """A4 map matching reference images: white bg, blue boundary, red pts, numbered labels."""
    fig, ax = plt.subplots(figsize=(A4W, A4H), dpi=DPI)
    fig.patch.set_facecolor("white"); ax.set_facecolor("white"); handles = []

    if poly_gdf is not None and not poly_gdf.empty:
        if "Comp_ID" in poly_gdf.columns:
            for i, (_, row) in enumerate(poly_gdf.iterrows()):
                g = row.geometry
                if g is None or g.is_empty: continue
                col = _COMP_COLORS_RICH[i % len(_COMP_COLORS_RICH)]
                cid = row.get("Comp_ID", f"Comp_{i+1:03d}")
                ah  = row.get("Area_ha", "")
                lbl = f"{cid} ({ah:.2f} ha)" if isinstance(ah, float) else str(cid)
                gpd.GeoDataFrame([{"geometry": g}], crs=poly_gdf.crs).plot(
                    ax=ax, facecolor=col, edgecolor="#111111", linewidth=1.6)
                handles.append(mpatches.Patch(facecolor=col, edgecolor="#111111",
                                              linewidth=1.0, label=lbl))
        else:
            poly_gdf.plot(ax=ax, facecolor="none", edgecolor="#1565C0", linewidth=2.0)
            ha_lbl = f"Area = {area_ha:.3f}.ha" if area_ha else "Forest Boundary"
            from matplotlib.lines import Line2D as _L2D
            handles.append(_L2D([0],[0], color="#1565C0", linewidth=2.5, label="Forest Boundary"))
            if area_ha:
                handles.append(mpatches.Patch(facecolor="none", edgecolor="none", label=f"Area = {area_ha:.3f} ha"))

    if line_gdf is not None and not line_gdf.empty:
        line_gdf.plot(ax=ax, color=lc, linewidth=1.0)

    if pts_gdf is not None and not pts_gdf.empty:
        pts_gdf.plot(ax=ax, color=ptc, markersize=16, zorder=8, marker="o")
        has_comp = (poly_gdf is not None and not poly_gdf.empty
                    and "Comp_ID" in poly_gdf.columns)
        pt_lbl = "Sub-compartment Survey Points" if has_comp else "Survey Points"
        from matplotlib.lines import Line2D as _L2D
        handles.append(_L2D([0],[0], marker="o", color="w",
                            markerfacecolor=ptc, markeredgecolor=ptc,
                            markersize=7, label=pt_lbl, linewidth=0))

    lbl_src = label_pts_gdf if label_pts_gdf is not None else pts_gdf
    lc_use  = user_label_col or label_col
    if lc_use and lbl_src is not None and not lbl_src.empty and lc_use in lbl_src.columns:
        for _, row in lbl_src.iterrows():
            try:
                ax.annotate(str(row[lc_use]),
                            xy=(row.geometry.x, row.geometry.y),
                            xytext=(4, 5), textcoords="offset points",
                            ha="left", va="bottom", fontsize=5.5, fontweight="bold",
                            color="black",
                            path_effects=[pe.Stroke(linewidth=1.8, foreground="white"),
                                          pe.Normal()], zorder=9)
            except: pass

    if poly_gdf is not None and not poly_gdf.empty:
        if "Comp_ID" in poly_gdf.columns:
            _label_feat(ax, poly_gdf, "Comp_ID", fs=9, color="#1565C0")
        elif "Forest" in poly_gdf.columns:
            _label_feat(ax, poly_gdf, "Forest", fs=9, color="#1565C0")

    _style_ax(ax); _north_arrow(ax); _scale_bar(ax)
    _add_legend(ax, handles, legend_title=legend_title or "Legend", loc="lower right")
    head = title.strip() if title.strip() else (
        f"Forest Area: {area_ha:.3f} ha" if area_ha else "Forest Boundary Map")
    ax.set_title(head, fontsize=12, fontweight="bold", color="#0d1f17", pad=10)
    plt.tight_layout(pad=0.4, rect=[0, 0, 0.80, 0.97])
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white"); plt.close(fig)


def preview_slope(vec_gdf, bgdf, summary_rows, path, f_mode="A",
                  per_group_summaries=None, title="", legend_title="Slope Classes"):
    # 4-class system matching group_f step 3 and reference image 1
    cc={1:"#2e8b57",2:"#90ee90",3:"#ffd700",4:"#e74c3c"}
    cl={1:"0-19 degree",2:"19-31 degree",3:"31-45 degree",4:"45> degree"}
    hg=f_mode in ("B","E") and per_group_summaries and len(per_group_summaries)>1
    nr=len(summary_rows); tr=max(0.20,min(0.48,0.06+nr*0.025))
    fig=plt.figure(figsize=(A4W,A4H),dpi=DPI,facecolor="white")
    gs=gridspec.GridSpec(2,1,height_ratios=[1,tr],hspace=0.20,left=0.10,right=0.96,top=0.94,bottom=0.04)
    ax1=fig.add_subplot(gs[0]); ax2=fig.add_subplot(gs[1])
    if vec_gdf is not None and not vec_gdf.empty and "Class" in vec_gdf.columns:
        for cid,color in cc.items():
            sub=vec_gdf[vec_gdf["Class"]==cid]
            if not sub.empty: sub.plot(ax=ax1,facecolor=color,edgecolor="#2c2c2c",linewidth=0.4,alpha=0.92,zorder=3)
    if bgdf is not None and not bgdf.empty:
        try: bgdf.boundary.plot(ax=ax1,color="black",linewidth=1.8,zorder=5)
        except: pass
    if hg and bgdf is not None and not bgdf.empty:
        gc=next((c for c in bgdf.columns if c.lower()!="geometry"),None)
        if gc: _label_feat(ax1,bgdf,gc)
    _style_ax(ax1); _north_arrow(ax1); _scale_bar(ax1)
    handles=[mpatches.Patch(facecolor=c,edgecolor="#444",linewidth=0.5,label=cl[cid]) for cid,c in cc.items() if cid in cc]
    from matplotlib.lines import Line2D as _L2Dp
    handles.append(_L2Dp([0],[0],color="black",linewidth=1.8,label="Forest Boundary"))
    _add_legend(ax1,handles,legend_title=legend_title or "Slope Classes",loc="lower right")
    mt=title.strip() if title.strip() else "Slope Classification Map"
    if hg: mt+=f" — {len(per_group_summaries)} Groups"
    ax1.set_title(mt,fontsize=11,fontweight="bold",pad=7)
    ax2.axis("off"); ax2.set_title("Slope Area Summary",fontsize=9,fontweight="bold",pad=4)
    if not summary_rows:
        fig.savefig(path,dpi=DPI,bbox_inches="tight",facecolor="white"); plt.close(fig); return
    hr=any(r.get("Recal_ha") is not None for r in summary_rows)
    if f_mode=="A":
        th=sum(r["Area_ha"] for r in summary_rows)
        cols=["Slope Range","Description","Area (ha)","% of Total"]
        if hr: cols+=["Recal. (ha)","Factor"]
        td=[]
        for r in summary_rows:
            rw=[r["Slope_Range"],r["Description"],f"{r['Area_ha']:.2f}",f"{r['Pct_Area']:.1f}%"]
            if hr: rw+=[f"{r['Recal_ha']:.2f}" if r.get("Recal_ha") else "—","" if not r.get("Cal_Factor") else f"{r['Cal_Factor']:.4f}"]
            td.append(rw)
        tr2=["TOTAL","",f"{th:.2f}","100%"]
        if hr: tr2+=[f"{sum(r['Recal_ha'] for r in summary_rows if r.get('Recal_ha')):.2f}",""]
        td.append(tr2); rcm={i:summary_rows[i].get("Class",0) for i in range(len(summary_rows))}
    else:
        cols=["Group","Slope Range","Description","Area (ha)","% of Total"]
        if hr: cols+=["Recal. (ha)"]
        td=[]; rcm={}; ri=0
        if per_group_summaries:
            for lb,grs in per_group_summaries.items():
                for r in grs:
                    rw=[str(lb),r["Slope_Range"],r["Description"],f"{r['Area_ha']:.2f}",f"{r['Pct_Area']:.1f}%"]
                    if hr: rw+=[f"{r['Recal_ha']:.2f}" if r.get("Recal_ha") else "—"]
                    td.append(rw); rcm[ri]=r.get("Class",0); ri+=1
                gt=sum(r["Area_ha"] for r in grs)
                sr=[f"  ↳ {lb} Total","","",f"{gt:.2f}",""]
                if hr: sr+=[f"{sum(r['Recal_ha'] for r in grs if r.get('Recal_ha')):.2f}"]
                td.append(sr); rcm[ri]=-1; ri+=1
    if td:
        tbl=ax2.table(cellText=td,colLabels=cols,cellLoc="center",loc="center",bbox=[0,0,1,1])
        tbl.auto_set_font_size(False); tbl.set_fontsize(6.5 if hg else 8.5)
        tbl.auto_set_column_width(list(range(len(cols))))
        rc={1:"#d5f5e3",2:"#c8f7c5",3:"#fef9e7",4:"#fadbd8",0:"#f8f9fa",-1:"#ddeeff"}
        for (r2,c2),cell in tbl.get_celld().items():
            cell.set_edgecolor("#ccc")
            if r2==0: cell.set_facecolor("#1a5276"); cell.set_text_props(color="white",fontweight="bold")
            else:
                cid2=rcm.get(r2-1,0)
                cell.set_facecolor(rc.get(cid2,"#f8f9fa"))
                if cid2==-1 or (cid2==0 and r2-1>=len(summary_rows)): cell.set_text_props(fontweight="bold")
    plt.tight_layout(pad=0.3, rect=[0, 0, 0.82, 1.0])
    fig.savefig(path,dpi=DPI,bbox_inches="tight",facecolor="white"); plt.close(fig)

# ── KMZ ─────────────────────────────────────────────────────────────────────
def _kml_pm(gdf,sid,nc=None):
    lines=[]
    for i,row in gdf.iterrows():
        g=row.geometry
        if g is None or g.is_empty: continue
        if nc and nc in row.index and row[nc]: label=str(row[nc])
        elif "Comp_ID" in row.index and row["Comp_ID"]: label=str(row["Comp_ID"])
        elif "Forest" in row.index and row["Forest"]: label=str(row["Forest"])
        else: label=f"Feature {i+1}"
        def cs(c): return " ".join(f"{x},{y},0" for x,y in c)
        def pk(geom):
            o=cs(list(geom.exterior.coords))
            r=[f"<outerBoundaryIs><LinearRing><coordinates>{o}</coordinates></LinearRing></outerBoundaryIs>"]
            for interior in geom.interiors:
                inn=cs(list(interior.coords)); r.append(f"<innerBoundaryIs><LinearRing><coordinates>{inn}</coordinates></LinearRing></innerBoundaryIs>")
            return f"<Polygon>{''.join(r)}</Polygon>"
        if g.geom_type=="Polygon": gk=pk(g)
        elif g.geom_type=="MultiPolygon": gk=f"<MultiGeometry>{''.join(pk(x) for x in g.geoms)}</MultiGeometry>"
        elif g.geom_type=="LineString": gk=f"<LineString><coordinates>{cs(list(g.coords))}</coordinates></LineString>"
        elif g.geom_type=="Point": gk=f"<Point><coordinates>{g.x},{g.y},0</coordinates></Point>"
        else: continue
        lines.append(f"<Placemark><name>{label}</name><styleUrl>#{sid}</styleUrl>{gk}</Placemark>")
    return "\n".join(lines)

def generate_kmz(poly_gdf,line_gdf,pts_gdf,out_dir,run_id):
    import zipfile as zf
    def w84(gdf):
        if gdf is None or gdf.empty: return None
        try: return gdf.to_crs("EPSG:4326") if gdf.crs else None
        except: return None
    pw=w84(poly_gdf); lw=w84(line_gdf); ptw=w84(pts_gdf)
    ref=pw if pw is not None and not pw.empty else lw
    if ref is not None and not ref.empty:
        u=ref.unary_union; cx,cy=u.centroid.x,u.centroid.y
        minx,miny,maxx,maxy=u.bounds; span=max(maxx-minx,maxy-miny)
        alt=max(500,int(span*111000*2))
    else: cx,cy,alt=0,0,10000
    st="""<Style id="poly_style"><LineStyle><color>ff00ff00</color><width>2</width></LineStyle><PolyStyle><color>4400cc00</color></PolyStyle></Style>
<Style id="line_style"><LineStyle><color>ff0000ff</color><width>2</width></LineStyle></Style>
<Style id="point_style"><IconStyle><color>ff0000ff</color><scale>0.8</scale><Icon><href>http://maps.google.com/mapfiles/kml/shapes/placemark_circle.png</href></Icon></IconStyle></Style>"""
    folders=[]
    if pw is not None and not pw.empty: folders.append(f"<Folder><name>Polygons</name>{_kml_pm(pw,'poly_style','Forest')}</Folder>")
    if lw is not None and not lw.empty: folders.append(f"<Folder><name>Lines</name>{_kml_pm(lw,'line_style','Forest')}</Folder>")
    if ptw is not None and not ptw.empty: folders.append(f"<Folder><name>Points</name>{_kml_pm(ptw,'point_style','SN')}</Folder>")
    kml=f"""<?xml version="1.0" encoding="UTF-8"?><kml xmlns="http://www.opengis.net/kml/2.2"><Document><name>Elfak GIS</name>
<LookAt><longitude>{cx}</longitude><latitude>{cy}</latitude><altitude>0</altitude><range>{alt}</range><tilt>0</tilt><heading>0</heading></LookAt>
{st}{"".join(folders)}</Document></kml>"""
    kmz=os.path.join(out_dir,"output.kmz")
    with zf.ZipFile(kmz,"w",zf.ZIP_DEFLATED) as z: z.writestr("doc.kml",kml.encode("utf-8"))
    return {"url":f"/outputs/{run_id}/output.kmz","lat":round(cy,6),"lon":round(cx,6),"alt":alt}


# ═══════════════════════════════════════════════════════════════════════════════
# GROUP G – FOREST POINT GENERATOR
# Generates vertex, boundary, and divider survey points from compartment shapefiles
# Supports: SHP / ZIP-of-SHP input, UTM 44N / 45N, configurable spacing
# ═══════════════════════════════════════════════════════════════════════════════

def _g_read_shp(file_storage, target_shp=None):
    """Read SHP (or ZIP containing SHP) from a FileStorage object → GeoDataFrame.
    target_shp: relative path inside ZIP to use (user-selected)."""
    fname = file_storage.filename.lower()
    tmp_dir = os.path.join(UPLOAD, "g_tmp_" + uuid.uuid4().hex[:8])
    os.makedirs(tmp_dir, exist_ok=True)
    try:
        if fname.endswith(".zip"):
            zip_path = os.path.join(tmp_dir, "upload.zip")
            file_storage.save(zip_path)
            with zipfile.ZipFile(zip_path, "r") as z:
                z.extractall(tmp_dir)
            shps = [os.path.join(r, f) for r, _, fs in os.walk(tmp_dir)
                    for f in fs if f.endswith(".shp")]
            if not shps:
                raise ValueError("No .shp file found in the uploaded ZIP.")
            # Use user-selected file if specified
            chosen = shps[0]
            if target_shp:
                tname = os.path.basename(target_shp)
                for s in shps:
                    if os.path.basename(s) == tname:
                        chosen = s; break
            return gpd.read_file(chosen)
        else:
            shp_path = os.path.join(tmp_dir, file_storage.filename)
            file_storage.save(shp_path)
            return gpd.read_file(shp_path)
    finally:
        pass  # keep tmp dir for debugging

def _g_reproject(gdf, zone_str):
    """Reproject to the user-selected UTM zone (44 or 45)."""
    epsg = 32644 if str(zone_str).strip() in ("44", "44N", "EPSG:32644") else 32645
    if gdf.crs is None:
        gdf = gdf.set_crs(f"EPSG:{epsg}")
    return gdf.to_crs(f"EPSG:{epsg}"), epsg

def _g_get_poly(geom):
    """Yield individual Polygon objects from any geometry type."""
    if geom is None or geom.is_empty:
        return
    if geom.geom_type == "Polygon":
        yield geom
    elif geom.geom_type == "MultiPolygon":
        for p in geom.geoms:
            yield p

def _g_vertex_points(gdf, comp_col):
    """STEP 3 – Extract all exterior ring vertices as Point records."""
    records = []
    for _, row in gdf.iterrows():
        cid = str(row[comp_col]) if comp_col and comp_col in row.index else ""
        for poly in _g_get_poly(row.geometry):
            for x, y in poly.exterior.coords:
                records.append({"Point_Type": "Vertex", "Source": "Vertex",
                                 "Compartments": cid, "Easting": round(x, 3),
                                 "Northing": round(y, 3),
                                 "geometry": Point(x, y)})
    return records

def _g_boundary_points(gdf, comp_col, spacing):
    """STEP 4 – Interpolate points along each exterior boundary at given spacing."""
    records = []
    for _, row in gdf.iterrows():
        cid = str(row[comp_col]) if comp_col and comp_col in row.index else ""
        for poly in _g_get_poly(row.geometry):
            line = poly.exterior
            total = line.length
            d = spacing
            while d < total:
                pt = line.interpolate(d)
                records.append({"Point_Type": "Boundary", "Source": "Boundary",
                                 "Compartments": cid,
                                 "Easting": round(pt.x, 3), "Northing": round(pt.y, 3),
                                 "geometry": pt})
                d += spacing
    return records

def _g_divider_points(gdf, comp_col, spacing):
    """STEPS 5-6 – Find shared compartment boundaries, generate divider points."""
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
            # Only accept LineString / MultiLineString – ignore Point touches
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
                if seg.is_empty: continue
                total = seg.length
                d = spacing
                while d < total:
                    pt = seg.interpolate(d)
                    records.append({"Point_Type": "Divider", "Source": "Divider",
                                     "Compartments": comp_pair,
                                     "Easting": round(pt.x, 3), "Northing": round(pt.y, 3),
                                     "geometry": pt})
                    d += spacing
    return records

def _g_merge_dedup(vertex_recs, boundary_recs, divider_recs):
    """STEPS 7-8 – Merge all records, deduplicate by (X,Y), priority: Divider > Vertex > Boundary."""
    # Build dict keyed by (round_x, round_y) → record with highest priority
    PRIO = {"Divider": 3, "Vertex": 2, "Boundary": 1}
    merged: dict = {}
    for recs in (boundary_recs, vertex_recs, divider_recs):  # low→high priority order
        for r in recs:
            key = (round(r["Easting"], 3), round(r["Northing"], 3))
            if key not in merged:
                merged[key] = r.copy()
                merged[key]["_all_sources"]  = {r["Source"]}
                merged[key]["_all_comps"]    = {r["Compartments"]}
                merged[key]["_prio"]         = PRIO.get(r["Source"], 0)
            else:
                existing = merged[key]
                existing["_all_sources"].add(r["Source"])
                existing["_all_comps"].add(r["Compartments"])
                new_prio = PRIO.get(r["Source"], 0)
                if new_prio > existing["_prio"]:
                    existing["Point_Type"] = r["Point_Type"]
                    existing["_prio"] = new_prio
    # Build final list with merged Source and Compartments
    result = []
    for rec in merged.values():
        rec["Source"]       = "+".join(sorted(rec["_all_sources"]))
        comp_set            = set()
        for cs in rec["_all_comps"]:
            for c in cs.split(","):
                if c: comp_set.add(c.strip())
        rec["Compartments"] = ",".join(sorted(comp_set))
        result.append(rec)
    return result

def _g_assign_ids(records):
    """STEP 9 – Sequential Point_ID after dedup."""
    # STEP 10 – Sort: Point_Type (Divider→Vertex→Boundary), then Compartments, Easting, Northing
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
    """Export SHP, CSV, XLSX to out_dir."""
    if not records:
        raise ValueError("No points generated. Check shapefile and spacing.")
    df = pd.DataFrame([{
        "Point_ID": r["Point_ID"], "Point_Type": r["Point_Type"],
        "Source": r["Source"], "Compartments": r["Compartments"],
        "Easting": r["Easting"], "Northing": r["Northing"]
    } for r in records])
    # CSV
    csv_path = os.path.join(out_dir, f"{prefix}.csv")
    df.to_csv(csv_path, index=False)
    # Excel
    xlsx_path = os.path.join(out_dir, f"{prefix}.xlsx")
    df.to_excel(xlsx_path, index=False)
    # Shapefile
    shp_gdf = gpd.GeoDataFrame(df, geometry=[r["geometry"] for r in records],
                                 crs=f"EPSG:{epsg}")
    shp_path = os.path.join(out_dir, f"{prefix}.shp")
    shp_gdf.to_file(shp_path)
    return df, shp_gdf

def _g_preview(shp_gdf, poly_gdf, path, title="Forest Survey Points", area_ha=None):
    """A4 PNG matching reference image 4: blue boundary, red dots, sequential numbers."""
    fig, ax = plt.subplots(figsize=(A4W, A4H), dpi=DPI)
    fig.patch.set_facecolor("white"); ax.set_facecolor("white")
    handles = []

    # Forest boundary + compartment dividers (blue)
    if poly_gdf is not None and not poly_gdf.empty:
        # Outer boundary thick
        try:
            outer = poly_gdf.unary_union
            gpd.GeoDataFrame([{"geometry": outer}], crs=poly_gdf.crs).plot(
                ax=ax, facecolor="none", edgecolor="#1565C0", linewidth=2.2, zorder=4)
        except: pass
        # Interior compartment lines
        poly_gdf.plot(ax=ax, facecolor="none", edgecolor="#1565C0", linewidth=1.5, zorder=3)
        handles.append(mpatches.Patch(facecolor="none", edgecolor="#1565C0",
                                      linewidth=2.0, label="Forest Boundary"))

    # Survey points – red filled circles
    if shp_gdf is not None and not shp_gdf.empty:
        shp_gdf.plot(ax=ax, color="red", markersize=14, zorder=8, marker="o")
        handles.append(mpatches.Patch(facecolor="red", edgecolor="red", label="Sample Plots"))
        # Sequential number labels (small, black, slight offset)
        if "Point_ID" in shp_gdf.columns:
            for _, row in shp_gdf.iterrows():
                try:
                    num = row["Point_ID"].lstrip("P").lstrip("0") or "0"
                    ax.annotate(num,
                                xy=(row.geometry.x, row.geometry.y),
                                xytext=(4, 5), textcoords="offset points",
                                ha="left", va="bottom", fontsize=5.0,
                                fontweight="bold", color="black",
                                path_effects=[pe.Stroke(linewidth=1.6, foreground="white"),
                                              pe.Normal()], zorder=9)
                except: pass

    # Compartment name labels in blue (bold centred)
    if poly_gdf is not None and not poly_gdf.empty:
        id_col = next((c for c in ("Comp_ID","Comp_No","Compartment","comp") if c in poly_gdf.columns), None)
        if id_col:
            _label_feat(ax, poly_gdf, id_col, fs=8.5, color="#1565C0")

    ha_txt = f"Area = {area_ha:.3f}.ha" if area_ha else ""
    if ha_txt:
        handles.append(mpatches.Patch(facecolor="none", edgecolor="none", label=ha_txt))

    _style_ax(ax); _north_arrow(ax); _scale_bar(ax)
    _add_legend(ax, handles, legend_title="Legend", loc="lower right")
    ax.set_title(title.strip() or "Forest Survey Points",
                 fontsize=12, fontweight="bold", color="#0d1f17", pad=10)
    plt.tight_layout(pad=0.5, rect=[0, 0, 1, 0.97])
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white"); plt.close(fig)

def group_g(file_storage, dem_zone, comp_col_name, spacing, out_dir, run_id, target_shp=None):
    """
    Full Group G pipeline.
    Returns (df, shp_gdf, poly_gdf, summary_dict)
    """
    _prog(run_id, "Reading shapefile…", 5)
    gdf = _g_read_shp(file_storage, target_shp=target_shp)
    if gdf.empty:
        raise ValueError("Shapefile is empty or could not be read.")
    if not all(t in ("Polygon", "MultiPolygon") for t in gdf.geom_type.unique()):
        raise ValueError("Shapefile must contain only Polygon / MultiPolygon features.")

    _prog(run_id, f"Reprojecting to UTM {dem_zone}N…", 10)
    gdf, epsg = _g_reproject(gdf, dem_zone)

    # Resolve compartment column
    comp_col = None
    if comp_col_name and comp_col_name in gdf.columns:
        comp_col = comp_col_name
    else:
        for alias in ("Comp_ID","Comp_No","comp_id","comp_no","Compartment","COMP"):
            if alias in gdf.columns:
                comp_col = alias; break

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
    df, shp_gdf = _g_export(all_recs, epsg, out_dir)

    # Summary counts
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

# ── ROUTES ───────────────────────────────────────────────────────────────────
@app.route("/progress/<run_id>")
@_rate_limit(limit=120, window=60)
def progress_stream(run_id):
    """Real-time SSE progress stream — polls at 250ms for smooth UI updates."""
    try: run_id = _safe_runid(run_id)
    except: pass  # allow non-UUID for backward compat

    def gen():
        sent     = 0
        deadline = time.time() + 600   # 10 min max
        while time.time() < deadline:
            with _PROG_LOCK:
                msgs = list(_PROG.get(run_id, []))
            new = msgs[sent:]
            if new:
                for m in new:
                    yield f"data: {m}\n\n"
                sent += len(new)
                try:
                    if json.loads(msgs[-1]).get("pct", 0) >= 100:
                        return
                except: pass
            else:
                yield f": heartbeat\n\n"   # keep connection alive
            time.sleep(0.25)

    return Response(
        stream_with_context(gen()),
        mimetype="text/event-stream",
        headers={
            "Cache-Control":     "no-cache, no-store",
            "X-Accel-Buffering": "no",
            "Connection":        "keep-alive",
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
    # Include polygon shapefiles AND the ForestPoints shapefile (Group G)
    shps = []
    for r, _, fs in os.walk(folder):
        for f in fs:
            if not f.endswith(".shp"): continue
            fl = f.lower()
            if "polygon" in fl or "forestpoints" in fl or "rtp" in fl:
                shps.append(os.path.join(r, f))
    # Fallback: any shp in folder
    if not shps:
        shps = [os.path.join(r, f) for r, _, fs in os.walk(folder)
                for f in fs if f.endswith(".shp")]
    gdfs = []
    for shp in shps:
        try:
            g = gpd.read_file(shp)
            if g.crs is not None: g = g.to_crs("EPSG:4326")
            # Keep only relevant columns
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

@app.route("/compose/<run_id>",methods=["POST"])
@_rate_limit(limit=20,window=60)
def compose_map(run_id):
    run_id = _safe_runid(run_id)
    folder=os.path.join(OUTPUT,run_id)
    if not os.path.exists(folder): return jsonify({"error":"Run not found"}),404
    try:
        data=request.get_json(silent=True) or {}
        title=data.get("title",""); lt=data.get("legend_title","Legend"); lc=data.get("label_col","")
        shps=[os.path.join(r,f) for r,_,fs in os.walk(folder) for f in fs if f.endswith("_polygon.shp")]
        if not shps: return jsonify({"error":"No shapefiles found."}),400
        gdfs=[]; crs0=None
        for shp in shps:
            try: g=gpd.read_file(shp); crs0=crs0 or g.crs; gdfs.append(g)
            except: pass
        if not gdfs: return jsonify({"error":"Could not read shapefiles."}),400
        pg=gpd.GeoDataFrame(pd.concat(gdfs,ignore_index=True),crs=crs0)
        pp=os.path.join(folder,"output.png")
        if "Comp_ID" in pg.columns:
            preview_compartments(pg,pp,title=title,legend_title=lt,label_col=lc or "Comp_ID")
        else:
            ah=float(pg["Area_ha"].sum()) if "Area_ha" in pg.columns else None
            preview(pg,None,None,pp,title=title,legend_title=lt,area_ha=ah,user_label_col=lc or None)
        return jsonify({"ok":True,"png":f"/outputs/{run_id}/output.png?t={uuid.uuid4().hex[:8]}"})
    except Exception as e: return jsonify({"error":f"Compose error: {e}"}),500

@app.route("/save_edit/<run_id>",methods=["POST"])
@_rate_limit(limit=20,window=60)
def save_edit(run_id):
    run_id = _safe_runid(run_id)
    folder=os.path.join(OUTPUT,run_id)
    if not os.path.exists(folder): return jsonify({"error":"Run not found"}),404
    try:
        data=request.get_json(silent=True) or {}
        geojson=data.get("geojson")
        if not geojson: return jsonify({"error":"No GeoJSON provided."}),400
        gdf_new=gpd.GeoDataFrame.from_features(geojson.get("features",[]),crs="EPSG:4326")
        shps=[os.path.join(r,f) for r,_,fs in os.walk(folder) for f in fs if f.endswith("_polygon.shp")]
        orig_crs="EPSG:32644"
        if shps:
            try:
                g0=gpd.read_file(shps[0])
                if g0.crs: orig_crs=str(g0.crs)
            except: pass
        gdf_new=gdf_new.to_crs(orig_crs)
        gdf_new["geometry"]=[_close_poly(_repair(g)) if g else None for g in gdf_new.geometry]
        gdf_new=gdf_new[gdf_new.geometry.notna()]
        if "Area_ha" not in gdf_new.columns:
            gdf_new["Area_ha"]=[round(g.area/10000,4) if g else 0 for g in gdf_new.geometry]
        gdf_new.to_file(os.path.join(folder,"edited_polygon.shp"))
        pp=os.path.join(folder,"output.png"); title=data.get("title","Edited Map"); lt=data.get("legend_title","Legend")
        if "Comp_ID" in gdf_new.columns:
            preview_compartments(gdf_new,pp,title=title,legend_title=lt)
        else:
            ah=float(gdf_new["Area_ha"].sum()) if "Area_ha" in gdf_new.columns else None
            preview(gdf_new,None,None,pp,title=title,legend_title=lt,area_ha=ah)
        zip_folder(folder)
        return jsonify({"ok":True,"png":f"/outputs/{run_id}/output.png?t={uuid.uuid4().hex[:8]}"})
    except Exception as e: return jsonify({"error":f"Edit error: {e}\n{traceback.format_exc()}"}),500

@app.route("/login", methods=["POST"])
@_rate_limit(limit=20, window=60)
def login():
    """
    Unique-username login:
    - New username → registers automatically
    - Existing username → logs in (resumes session, same user continues work)
    - Up to 100 different users can login simultaneously with full isolation
    """
    data = request.get_json(silent=True) or {}
    try:
        username = _validate_username(data.get("username", ""))
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    users = _lu()
    is_new = username not in users

    if is_new:
        # Brand new user — register them
        try:
            user, _tok = _register_user(username)
        except ValueError as e:
            # Race condition: another request registered same name between our check and write
            return jsonify({"error": str(e), "taken": True}), 409
        except Exception as e:
            log.error(f"Registration error {username!r}: {e}")
            return jsonify({"error": "Registration failed. Please try again."}), 500
    else:
        # Existing user — allow them to resume (same person continuing their work)
        try:
            user = _login_existing(username)
        except KeyError:
            # shouldn't happen but handle race
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
        "ok":          True,
        "username":    username,
        "runs":        user.get("runs", [])[-20:],
        "is_new":      is_new,
        "is_returning": not is_new,
        "message":     "Welcome!" if is_new else f"Welcome back, {username}!"
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
    users = _lu(); user = users.get(u, {})
    return jsonify({"username": u, "runs": user.get("runs", [])[-20:]})

@app.route("/history")
@_rate_limit(limit=30, window=60)
@_login_required
def history():
    u = _require_login()
    users = _lu(); user = users.get(u, {})
    return jsonify({"runs": user.get("runs", [])})

@app.route("/upload", methods=["POST"])
@_rate_limit(limit=10, window=60)          # 10 pipeline runs / min / IP
@_with_pipeline_sem
def upload():
    run_id = str(uuid.uuid4())
    with _PROG_LOCK: _PROG[run_id] = []
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded.", "run_id": run_id}), 400
        file = request.files["file"]
        # Validate filename (security)
        try: _safe_filename(file.filename)
        except ValueError as e: return jsonify({"error": str(e), "run_id": run_id}), 400
        module = request.form.get("module","A")
        mode=request.form.get("mode","A"); zone=request.form.get("zone","44")
        title=request.form.get("title","").strip()
        legend_title=request.form.get("legend_title","Legend").strip() or "Legend"
        label_col=request.form.get("label_col","").strip()
        try: mapping=json.loads(request.form.get("mapping","{}"))
        except: mapping={}
        w=float(request.form.get("w",50)); h=float(request.form.get("h",50))
        rows=int(request.form.get("rows",10)); cols=int(request.form.get("cols",10))
        forest=request.form.get("forest") or (mapping or {}).get("forest") or "FOREST"
        username=_require_login() or "guest"
        out=os.path.join(OUTPUT,run_id); os.makedirs(out,exist_ok=True)
        crs=get_crs(zone); _prog(run_id,f"Starting module {module}…",2)
        lc_out=None; lp_gdf=None; area_ha_disp=None

        if module=="B":
            _prog(run_id,"Reading…",8); df=read_input(file)
            poly,line,pts=group_b(df,crs,out,mapping); lc_out=label_col or "Forest"
        elif module=="C":
            _prog(run_id,"Processing…",10)
            poly,line,pts=group_c(file,crs,w,h,rows,cols,out,mode,mapping)
            lc_out="SN"; lp_gdf=pts
        elif module=="D":
            _prog(run_id,"Reading…",8); df=read_input(file)
            d_mode=request.form.get("d_mode","A")
            poly,line,pts=group_d(df,crs,out,mapping,mode=d_mode); lc_out=label_col or "Forest"
        elif module=="E":
            e_mode=request.form.get("e_mode","A")
            nc=max(2,min(15,int(request.form.get("n_compartments",4))))
            at=float(request.form.get("area_tol_ha","0.3") or "0.3")
            method=request.form.get("e_method","bisect")
            is_zip=file.filename.lower().endswith(".zip")
            fcn=request.form.get("forest_col_name") or None
            if mapping and "forest" not in mapping: mapping["forest"]=forest
            _prog(run_id,"Loading input…",8)
            src_data=file if is_zip else read_input(file)
            poly,line,pts=group_e(src_data,crs,out,mapping,e_mode=e_mode,n_compartments=nc,
                                  is_zip=is_zip,fcol=fcn,area_tol_ha=at,method=method,run_id=run_id)
            _prog(run_id,"Rendering preview…",88)
            preview_compartments(poly,os.path.join(out,"output.png"),title=title,
                                 legend_title=legend_title,label_col=label_col or "Comp_ID")
            kmz_url=None
            try: kmz_url=generate_kmz(poly,line,pts,out,run_id)
            except: pass
            _append_run(username,run_id,"E"); _prog(run_id,"Complete.",100)
            return jsonify({"run_id":run_id,"download":f"/download/{run_id}","kmz_url":kmz_url})
        elif module=="F":
            # Priority: 1) cached DEM key  2) catalog path  3) uploaded file
            dem_cache_key = request.form.get("dem_cache_key","").strip()
            dem_catalog_path = request.form.get("dem_catalog_path","").strip()

            class _FileDEM:
                """Unified file-like wrapper for cached/catalog/uploaded DEMs."""
                def __init__(self, p):
                    self.filename = os.path.basename(p); self._p = p
                def save(self, dest):
                    shutil.copy2(self._p, dest)
                def read(self, size=-1):
                    with open(self._p,"rb") as f: return f.read(size)

            if dem_cache_key:
                # Find cached file by key prefix
                candidates = [f for f in os.listdir(DEM_CACHE_DIR)
                              if f.startswith(dem_cache_key)]
                if not candidates:
                    return jsonify({"error":"Cached DEM not found. Please select again.","run_id":run_id}),400
                dem_f = _FileDEM(os.path.join(DEM_CACHE_DIR, candidates[0]))
            elif dem_catalog_path:
                # Local catalog file
                cat_full = _safe_path(DEM_CATALOG_DIR, dem_catalog_path)
                if not os.path.exists(cat_full):
                    return jsonify({"error":f"DEM catalog file not found: {dem_catalog_path}","run_id":run_id}),400
                dem_f = _FileDEM(cat_full)
            else:
                dem_f = request.files.get("dem_file")
                if not dem_f:
                    return jsonify({"error":"No DEM selected. Choose from the catalog dropdown or upload a .tif file.","run_id":run_id}),400
            f_forest=request.form.get("f_forest") or forest
            f_mode=request.form.get("f_mode","A"); cc=request.form.get("comp_col") or None
            bzip=file.filename.lower().endswith(".zip"); fa=None
            fas=request.form.get("field_area_ha","").strip()
            if fas:
                try: fa=float(fas)
                except: pass
            sr,vgdf,bgdf,fmo,pgs=group_f(file,dem_f,crs,out,mapping,boundary_is_zip=bzip,
                                          forest_name=f_forest,f_mode=f_mode,comp_col_name=cc,
                                          field_area_ha=fa,run_id=run_id)
            _prog(run_id,"Rendering preview…",92)
            preview_slope(vgdf,bgdf,sr,os.path.join(out,"output.png"),
                          f_mode=fmo,per_group_summaries=pgs,title=title,legend_title=legend_title)
            poly=vgdf if (vgdf is not None and not vgdf.empty) else gpd.GeoDataFrame()
            line=gpd.GeoDataFrame(); pts=gpd.GeoDataFrame()
            kmz_url=None
            try: kmz_url=generate_kmz(poly,line,pts,out,run_id)
            except: pass
            _append_run(username,run_id,"F"); _prog(run_id,"Complete.",100)
            return jsonify({"run_id":run_id,"download":f"/download/{run_id}","kmz_url":kmz_url})
        else:
            _prog(run_id,"Reading…",8); df=read_input(file)
            poly,line,pts=group_a(df,forest,crs,out,mapping)
            if not poly.empty and "Area_ha" in poly.columns:
                area_ha_disp=float(poly["Area_ha"].sum())
            lc_out=label_col or "Forest"

        _prog(run_id,"Rendering A4 preview…",88)
        preview(poly,line,pts,os.path.join(out,"output.png"),
                pc="blue",lc="black",ptc="red",
                label_col=lc_out,label_pts_gdf=lp_gdf,
                area_ha=area_ha_disp,title=title,legend_title=legend_title,
                user_label_col=label_col or None)
        kmz_url=None
        try: kmz_url=generate_kmz(poly,line,pts,out,run_id)
        except: pass
        _append_run(username,run_id,module); _prog(run_id,"Complete.",100)
        return jsonify({"run_id":run_id,"download":f"/download/{run_id}","kmz_url":kmz_url})
    except ValueError as e:
        _prog(run_id,f"ERROR: {e}",0)
        return jsonify({"error":str(e),"run_id":run_id}),400
    except Exception as e:
        _prog(run_id,f"ERROR: {e}",0)
        return jsonify({"error":f"Unexpected error: {e}","run_id":run_id}),500



# ── DEM CATALOG: pre-loaded large DEM files ────────────────────────────────
# Place .tif files in the "dem_catalog/" folder on the server.
# Users pick from this dropdown instead of uploading 2GB files each time.
# DEM_CATALOG_DIR already initialised in startup block
@app.route("/dem_catalog")
@_rate_limit(limit=30, window=60)
def dem_catalog():
    """Return list of DEM files available in GitHub repo dem_catalog/ folder.
    Checks GitHub API first (live list), falls back to local cache manifest.
    """
    import urllib.request as _ur
    import json as _json

    # Try GitHub API to get live file list
    # Repo: read from GITHUB_DEM_BASE env, extract owner/repo/branch
    github_api_urls = []
    for zone in ("44N", "45N"):
        # GitHub Contents API
        api_base = os.environ.get("GITHUB_API_DEM",
            "https://api.github.com/1ravikafle-glitch/ElfakGISProStudio/tree/main/dem_catalog")
        github_api_urls.append((zone, f"{api_base}/{zone}"))

    files = []
    for zone, api_url in github_api_urls:
        try:
            req = _ur.Request(api_url, headers={"User-Agent": "elfak-gis-app"})
            with _ur.urlopen(req, timeout=8) as resp:
                items = _json.loads(resp.read())
            for item in items:
                name = item.get("name","")
                if not name.lower().endswith((".tif",".tiff")): continue
                size_mb = round(item.get("size",0)/(1024*1024), 1)
                dl_url  = item.get("download_url","")
                files.append({
                    "name":     name,
                    "zone":     zone,
                    "path":     f"{zone}/{name}",
                    "size_mb":  size_mb,
                    "url":      dl_url,
                })
        except Exception as e:
            log.warning(f"GitHub API {zone} failed: {e}")
            # Fallback: check local folder
            local = os.path.join(DEM_CATALOG_DIR, zone)
            if os.path.isdir(local):
                for f in os.listdir(local):
                    if f.lower().endswith((".tif",".tiff")):
                        fp = os.path.join(local, f)
                        files.append({
                            "name":    f,
                            "zone":    zone,
                            "path":    f"{zone}/{f}",
                            "size_mb": round(os.path.getsize(fp)/(1024*1024),1),
                            "url":     "",
                        })

    files.sort(key=lambda x: (x["zone"], x["name"]))
    return jsonify({"files": files, "source": "github"})

@app.route("/dem_fetch", methods=["POST"])
@_rate_limit(limit=5, window=60)
def dem_fetch():
    """Download a DEM file from GitHub to the local cache, return its cache key.
    Called by frontend before running Group F when catalog DEM is selected."""
    import urllib.request as _ur

    data   = request.get_json(silent=True) or {}
    url    = data.get("url","").strip()
    path   = data.get("path","").strip()   # e.g. "44N/N27E083.tif"

    if not url and not path:
        return jsonify({"error":"No DEM URL or path provided."}),400

    # Validate URL is from GitHub (security)
    if url and not (url.startswith("https://raw.githubusercontent.com/") or
                    url.startswith("https://github.com/")):
        return jsonify({"error":"DEM URL must be from GitHub raw content."}),400

    # Build local cache path
    safe_name = re.sub(r"[^A-Za-z0-9._-]","_", os.path.basename(path or url))[:120]
    cache_key  = hashlib.sha256((url or path).encode()).hexdigest()[:16]
    local_path = os.path.join(DEM_CACHE_DIR, f"{cache_key}_{safe_name}")

    # Return immediately if already cached
    if os.path.exists(local_path) and os.path.getsize(local_path) > 1000:
        size_mb = round(os.path.getsize(local_path)/(1024*1024),1)
        log.info(f"DEM cache hit: {safe_name} ({size_mb}MB)")
        return jsonify({"ok":True,"cache_key":cache_key,"local":local_path,
                        "size_mb":size_mb,"cached":True})

    # Download
    if not url:
        # Build URL from GITHUB_DEM_BASE + path
        url = GITHUB_DEM_BASE.rstrip("/") + "/" + path.lstrip("/")

    log.info(f"Downloading DEM from GitHub: {url}")
    try:
        req = _ur.Request(url, headers={"User-Agent":"elfak-gis-app"})
        with _ur.urlopen(req, timeout=120) as resp, open(local_path,"wb") as fout:
            while True:
                chunk = resp.read(1024*1024)  # 1MB chunks
                if not chunk: break
                fout.write(chunk)
        size_mb = round(os.path.getsize(local_path)/(1024*1024),1)
        log.info(f"DEM downloaded: {safe_name} ({size_mb}MB)")
        return jsonify({"ok":True,"cache_key":cache_key,"local":local_path,
                        "size_mb":size_mb,"cached":False})
    except Exception as e:
        if os.path.exists(local_path): os.remove(local_path)
        log.error(f"DEM download error: {e}")
        return jsonify({"error":f"Failed to download DEM: {e}"}),500

@app.route("/zip_inspect", methods=["POST"])
@_rate_limit(limit=20, window=60)
def zip_inspect():
    """Return list of .shp files inside an uploaded ZIP for user selection."""
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
        try: shutil.rmtree(tmp)
        except: pass

@app.route("/run_g", methods=["POST"])
@_rate_limit(limit=10, window=60)
@_with_pipeline_sem
def run_g():
    run_id = str(uuid.uuid4()); _PROG[run_id] = []
    try:
        if "file" not in request.files:
            return jsonify({"error": "No shapefile uploaded.", "run_id": run_id}), 400
        file       = request.files["file"]
        zone       = request.form.get("zone", "44")
        comp_col   = request.form.get("comp_col", "").strip() or None
        title      = request.form.get("title", "Forest Survey Points").strip()
        try:
            spacing = float(request.form.get("spacing", "20"))
            if spacing <= 0: raise ValueError
        except:
            return jsonify({"error": "Invalid spacing value.", "run_id": run_id}), 400

        username = _require_login() or "guest"
        out = os.path.join(OUTPUT, run_id); os.makedirs(out, exist_ok=True)

        target_shp = request.form.get("target_shp","").strip() or None
        df, shp_gdf, poly_gdf, summary = group_g(file, zone, comp_col, spacing, out, run_id,
                                                   target_shp=target_shp)

        _prog(run_id, "Rendering A4 map…", 90)
        _g_preview(shp_gdf, poly_gdf, os.path.join(out, "output.png"),
                   title=title, area_ha=summary["area_ha"])

        try:
            kmz_url = generate_kmz(poly_gdf, gpd.GeoDataFrame(), shp_gdf, out, run_id)
        except:
            kmz_url = None

        _append_run(username, run_id, "G",
                    f"{summary['total']} pts | {summary['compartments']} compartments")
        _prog(run_id, "Complete.", 100)
        return jsonify({
            "run_id":   run_id,
            "download": f"/download/{run_id}",
            "kmz_url":  kmz_url,
            "summary":  summary
        })
    except ValueError as e:
        _prog(run_id, f"ERROR: {e}", 0)
        return jsonify({"error": str(e), "run_id": run_id}), 400
    except Exception as e:
        _prog(run_id, f"ERROR: {e}", 0)
        return jsonify({"error": f"Unexpected error: {e}", "run_id": run_id}), 500

@app.route("/outputs/<run_id>/<path:filename>")
@_rate_limit(limit=120, window=60)
def serve_output(run_id,filename):
    folder=os.path.join(OUTPUT,run_id)
    if not os.path.exists(os.path.join(folder,filename)): return jsonify({"error":"File not found"}),404
    return send_from_directory(folder,filename)

def zip_folder(folder): return shutil.make_archive(folder,"zip",folder)

@app.route("/download/<run_id>")
def download(run_id):
    folder=os.path.join(OUTPUT,run_id)
    if not os.path.exists(folder): return jsonify({"error":"Run not found"}),404
    return send_file(zip_folder(folder),as_attachment=True)

@app.route("/")
def home(): return render_template("index.html")

if __name__=="__main__": app.run(debug=True,threaded=True)
