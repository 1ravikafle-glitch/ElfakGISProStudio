import os, uuid, zipfile, shutil, json, re, io, traceback, math
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
                   render_template, session, Response, stream_with_context)
from shapely.geometry import (Polygon, Point, LineString, MultiPolygon,
                               MultiPoint, GeometryCollection)
from shapely.ops import unary_union, voronoi_diagram
from shapely.geometry import box as _sbox
from shapely import affinity

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "elfak-gis-2025")
UPLOAD, OUTPUT, USERS_FILE = "uploads", "outputs", "users.json"
os.makedirs(UPLOAD, exist_ok=True); os.makedirs(OUTPUT, exist_ok=True)
A4W, A4H, DPI = 8.27, 11.69, 200
_PROG: dict = {}

def _prog(rid, msg, pct=None):
    if rid not in _PROG: _PROG[rid] = []
    o = {"msg": msg}
    if pct is not None: o["pct"] = pct
    _PROG[rid].append(json.dumps(o))
    _PROG[rid] = _PROG[rid][-300:]

# ── users ────────────────────────────────────────────────────────────────────
def _lu():
    try:
        if not os.path.exists(USERS_FILE): return {}
        with open(USERS_FILE) as f: return json.load(f)
    except: return {}
def _su(u):
    try:
        with open(USERS_FILE, "w") as f: json.dump(u, f, indent=2)
    except: pass
def _user_exists(n): return n.strip() in _lu()
def _create_user(n):
    n = n.strip(); u = _lu()
    if n in u: raise ValueError(f"Username '{n}' is already taken.")
    u[n] = {"username": n, "created_at": pd.Timestamp.now().isoformat(), "runs": []}
    _su(u); return u[n]
def _append_run(uname, rid, mod, desc=""):
    if not uname: return
    u = _lu()
    if uname in u:
        u[uname].setdefault("runs", []).append({"run_id": rid, "module": mod, "description": desc, "timestamp": pd.Timestamp.now().isoformat()})
        u[uname]["runs"] = u[uname]["runs"][-50:]; _su(u)
def _require_login(): return session.get("username")

@app.after_request
def _cors(r):
    r.headers["Access-Control-Allow-Origin"] = "*"
    r.headers["Access-Control-Allow-Headers"] = "Content-Type,Authorization"
    r.headers["Access-Control-Allow-Methods"] = "GET,POST,OPTIONS"
    return r

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
    if g is None or g.is_empty: return None
    return g if g.is_valid else g.buffer(0)
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
    if gdf is None or gdf.empty: return gdf
    keep = []
    for _, row in gdf.iterrows():
        pg = _as_poly(_repair(row.geometry))
        if pg and not pg.is_empty and pg.area>1e-10:
            r2 = row.copy(); r2["geometry"] = _close_poly(pg); keep.append(r2)
    if not keep: return gpd.GeoDataFrame(columns=gdf.columns, crs=gdf.crs)
    return gpd.GeoDataFrame(keep, crs=gdf.crs)

# ── map decorations ──────────────────────────────────────────────────────────
def _north_arrow(ax):
    x0,x1 = ax.get_xlim(); y0,y1 = ax.get_ylim()
    aw = x1-x0; ah = y1-y0
    nx = x1 - aw*0.04; ny = y1 - ah*0.05; arrh = ah*0.065
    ax.annotate("", xy=(nx, ny), xytext=(nx, ny-arrh),
                arrowprops=dict(arrowstyle="-|>", color="black", linewidth=1.6, mutation_scale=15), zorder=20)
    ax.text(nx, ny+ah*0.003, "N", ha="center", va="bottom", fontsize=8, fontweight="bold", color="black", zorder=20)

def _scale_bar(ax):
    x0,x1 = ax.get_xlim(); y0,y1 = ax.get_ylim()
    aw = x1-x0; ah = y1-y0
    raw = aw*0.20
    if raw <= 0: return
    mag = 10**math.floor(math.log10(max(raw,1e-6)))
    nice = min([1,2,5,10,20,25,50,100,200,250,500,1000,2000,5000,10000], key=lambda v: abs(v*mag-raw))
    bar_m = nice*mag if nice*mag > 10 else raw
    bx = x0 + aw*0.06; by = y0 + ah*0.035; bh = ah*0.013
    for i in range(4):
        fc = "black" if i%2==0 else "white"
        ax.add_patch(plt.Rectangle((bx+i*bar_m/4, by), bar_m/4, bh,
                     linewidth=0.8, edgecolor="black", facecolor=fc, zorder=15, clip_on=False))
    lbl = f"{bar_m:.0f} m" if bar_m<1000 else f"{bar_m/1000:.1f} km"
    ax.text(bx+bar_m/2, by+bh*2.1, lbl, ha="center", va="bottom", fontsize=6.5, fontweight="bold", color="black", zorder=16)
    ax.text(bx, by-bh*0.5, "0", ha="center", va="top", fontsize=5.5, color="black", zorder=16)
    ax.text(bx+bar_m, by-bh*0.5, lbl, ha="center", va="top", fontsize=5.5, color="black", zorder=16)

def _add_legend(ax, handles, legend_title="Legend", loc="lower right"):
    if not handles: return
    leg = ax.legend(handles=handles, title=legend_title, loc=loc,
                    fontsize=7, title_fontsize=7.5, framealpha=0.95,
                    edgecolor="#888", fancybox=False, frameon=True,
                    borderpad=0.7, labelspacing=0.4)
    leg.get_frame().set_linewidth(0.8)

def _graticule(ax):
    ax.tick_params(axis="both", which="both",
                   left=True, right=True, top=True, bottom=True,
                   labelleft=True, labelright=False, labelbottom=True, labeltop=False,
                   direction="out", length=4, width=0.7, labelsize=5.5, color="#555")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v,_: f"{int(v):,}"))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v,_: f"{int(v):,}"))
    ax.xaxis.set_tick_params(labelrotation=45)
    for sp in ax.spines.values(): sp.set_linewidth(0.7); sp.set_color("#666")
    ax.grid(False)
    ax.set_xlabel("Easting (m)", fontsize=6, color="#555", labelpad=3)
    ax.set_ylabel("Northing (m)", fontsize=6, color="#555", labelpad=3)

def _style_ax(ax):
    ax.set_aspect("equal")
    _graticule(ax)

def _label_feat(ax, gdf, col, fs=6.5, color="black"):
    if gdf is None or gdf.empty or col not in gdf.columns: return
    for _, row in gdf.iterrows():
        try:
            g = row.geometry
            if g is None or g.is_empty: continue
            cx, cy = g.centroid.x, g.centroid.y
            ax.annotate(str(row[col]), xy=(cx,cy), ha="center", va="center",
                        fontsize=fs, fontweight="bold", color=color,
                        path_effects=[pe.Stroke(linewidth=2.2, foreground="white"), pe.Normal()], zorder=12)
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
    return [_close_poly(_repair(p) or p) for p in valid]

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
            if run_id: _prog(run_id,f"Subdividing {fn}…",20+int(60*idx/max(len(features),1)))
            pieces=_subdivide(poly,n_compartments,method,area_tol_ha)
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

def _bnd_from_zip(zip_file,target_shp,src_crs,dem_crs):
    folder=os.path.join(UPLOAD,str(uuid.uuid4())); os.makedirs(folder,exist_ok=True)
    zp=os.path.join(folder,"b.zip"); zip_file.save(zp)
    with zipfile.ZipFile(zp) as z: z.extractall(folder)
    shps=[os.path.join(r,f) for r,_,fs in os.walk(folder) for f in fs if f.endswith(".shp")]
    if not shps: raise ValueError("No .shp in boundary ZIP.")
    sp=shps[0]
    if target_shp:
        for s in shps:
            if os.path.basename(s)==os.path.basename(target_shp): sp=s; break
    gdf=gpd.read_file(sp)
    if gdf.empty: raise ValueError("Boundary shapefile empty.")
    gdf=gdf.set_crs(src_crs) if gdf.crs is None else gdf
    gdf=gdf.to_crs(dem_crs); return _repair(gdf.unary_union),gdf

def group_f(boundary_file, dem_file, crs, out, mapping=None,
            boundary_is_zip=False, forest_name="FOREST",
            f_mode="A", comp_col_name=None, field_area_ha=None, run_id=None):
    try:
        import rasterio
        from rasterio.mask import mask as rio_mask
        from rasterio.features import shapes as rio_shapes
        import scipy.ndimage as ndi
    except ImportError:
        raise ValueError("rasterio and scipy required for Group F.")
    os.makedirs(out,exist_ok=True); pfx=_safe_dn(forest_name)
    SN=-9999.0
    if run_id: _prog(run_id,"Loading DEM…",5)
    dem_path=os.path.join(UPLOAD,f"{uuid.uuid4()}_dem.tif"); dem_file.save(dem_path)
    with rasterio.open(dem_path) as src:
        dem_crs=src.crs; nodata=src.nodata; transform=src.transform; profile=src.profile.copy()

    if run_id: _prog(run_id,"Loading boundary…",10)
    if boundary_is_zip:
        ts=(mapping or {}).get("target_shp")
        bpoly,bgdf=_bnd_from_zip(boundary_file,ts,crs,str(dem_crs))
    else:
        df=read_input(boundary_file); bpoly=_bnd_from_df(df,mapping)
        bgdf=gpd.GeoDataFrame([{"Forest":forest_name,"geometry":bpoly}],crs=crs).to_crs(str(dem_crs))
        bpoly=bgdf.unary_union

    if run_id: _prog(run_id,"Extracting rect DEM (20% buffer)…",18)
    b=bpoly.bounds; bx=(b[2]-b[0])*0.20; by=(b[3]-b[1])*0.20
    rect=_sbox(b[0]-bx,b[1]-by,b[2]+bx,b[3]+by)
    nd=nodata if nodata is not None else SN
    try:
        with rasterio.open(dem_path) as src:
            ra,rt=rio_mask(src,[rect.__geo_interface__],crop=True,filled=True,nodata=nd)
        rdm=ra[0].astype(np.float32); rdm[rdm==nd]=np.nan
    except Exception:
        with rasterio.open(dem_path) as src: ra=src.read(1).astype(np.float32); rt=src.transform
        rdm=ra; rdm[rdm==nd]=np.nan
    rx=abs(rt.a); ry=abs(rt.e)

    if run_id: _prog(run_id,"Computing slope (Horn method)…",30)
    valid=~np.isnan(rdm); filled=np.nan_to_num(rdm,nan=0.0)
    dzdx=ndi.sobel(filled,axis=1)/(8*max(rx,1e-9))
    dzdy=ndi.sobel(filled,axis=0)/(8*max(ry,1e-9))
    slope=np.degrees(np.arctan(np.sqrt(dzdx**2+dzdy**2))).astype(np.float32)
    fnv=ndi.binary_erosion(valid,structure=np.ones((3,3),dtype=bool),border_value=0)
    slope[~fnv]=SN
    rp=profile.copy(); rp.update(dtype="float32",nodata=SN,count=1,height=slope.shape[0],width=slope.shape[1],transform=rt)
    sp_path=os.path.join(out,f"{pfx}_slope_rect.tif")
    with rasterio.open(sp_path,"w",**rp) as dst: dst.write(slope,1)

    if run_id: _prog(run_id,"Reclassifying slope…",42)
    vm=(slope!=SN)&~np.isnan(slope)
    cls=np.zeros_like(slope,dtype=np.uint8)
    cls[vm&(slope<19)]=1; cls[vm&(slope>=19)&(slope<=31)]=2; cls[vm&(slope>31)]=3
    cp=rp.copy(); cp.update(dtype="uint8",nodata=0)
    cls_path=os.path.join(out,f"{pfx}_class_rect.tif")
    with rasterio.open(cls_path,"w",**cp) as dst: dst.write(cls,1)

    if run_id: _prog(run_id,"Raster → polygon…",54)
    rtp=[]
    with rasterio.open(cls_path) as src:
        ca,ct=src.read(1),src.transform
        for shp,val in rio_shapes(ca,mask=(ca>0).astype(np.uint8),transform=ct):
            cid=int(val)
            if cid==0: continue
            try:
                rings=shp["coordinates"]; ext=rings[0]; holes=rings[1:] if len(rings)>1 else None
                geom=_repair(Polygon(ext,holes=holes))
                if geom and not geom.is_empty and geom.area>1e-10: rtp.append({"gridcode":cid,"geometry":geom})
            except: continue
    if not rtp: raise ValueError("No slope polygons extracted from DEM.")
    rtp_gdf=gpd.GeoDataFrame(rtp,crs=str(dem_crs))
    rtp_gdf.to_file(os.path.join(out,f"{pfx}_rtp_raw.shp"))

    if run_id: _prog(run_id,"Dissolving by gridcode…",64)
    try:
        dissolved=rtp_gdf.dissolve(by="gridcode",as_index=False)
        dissolved["gridcode"]=dissolved["gridcode"].astype(int)
    except: dissolved=rtp_gdf.copy()
    dissolved.to_file(os.path.join(out,f"{pfx}_rtp_dissolved.shp"))

    if run_id: _prog(run_id,"Building analysis groups…",70)
    class_defs={1:("< 19°","Gentle","#2ecc71"),2:("19–31°","Moderate","#f39c12"),3:("> 31°","Steep","#e74c3c")}
    comp_polygons=[]
    if f_mode=="A":
        comp_polygons=[(forest_name,bpoly)]
    else:
        grp_col=None
        if comp_col_name:
            for c in bgdf.columns:
                if c.lower()==comp_col_name.lower(): grp_col=c; break
        if grp_col is None: grp_col=_find_col(bgdf,_FA if f_mode=="B" else _CA)
        if grp_col is None and f_mode=="E": grp_col=_find_col(bgdf,_FA)
        if grp_col is None: comp_polygons=[(forest_name,bpoly)]
        else:
            for val,grp in bgdf.groupby(grp_col):
                up=_repair(grp.unary_union)
                if up and not up.is_empty: comp_polygons.append((str(val),up))
        if not comp_polygons: comp_polygons=[(forest_name,bpoly)]

    if run_id: _prog(run_id,"Clipping slope polygons to boundary…",78)
    def _clip_grp(clip_poly,label):
        vrecs=[]; total=0.0
        for _,drow in dissolved.iterrows():
            cid=int(drow["gridcode"]); dgeom=_repair(drow.geometry)
            if dgeom is None or dgeom.is_empty: continue
            try: clipped=_repair(dgeom.intersection(clip_poly))
            except: continue
            if clipped is None or clipped.is_empty: continue
            pg=_as_poly(clipped)
            if pg is None or pg.is_empty or pg.area<1e-10:
                if hasattr(clipped,"geoms"):
                    pts2=[_as_poly(_repair(g)) for g in clipped.geoms if g.geom_type in ("Polygon","MultiPolygon")]
                    pts2=[p for p in pts2 if p and p.area>1e-10]
                    if pts2: pg=_close_poly(max(pts2,key=lambda x:x.area))
                    else: continue
                else: continue
            pg=_close_poly(pg); ah=round(pg.area/10000,4); total+=ah
            vrecs.append({"Label":label,"Class":cid,"Slope_Range":class_defs[cid][0],
                          "Descr":class_defs[cid][1],"Area_ha":ah,"geometry":pg})
        total=max(total,1e-6)
        rows=[{"Label":vr["Label"],"Class":vr["Class"],"Slope_Range":vr["Slope_Range"],
               "Description":vr["Descr"],"Area_ha":vr["Area_ha"],
               "Pct_Area":round(vr["Area_ha"]/total*100,2),"Total_ha":round(total,4)} for vr in vrecs]
        return rows,vrecs

    all_sum,all_vec,per_grp=[],[],{}
    for lb,cp in comp_polygons:
        rows,vrecs=_clip_grp(cp,lb)
        all_sum.extend(rows); all_vec.extend(vrecs); per_grp[lb]=rows

    if field_area_ha and field_area_ha>0:
        tc=sum(r["Area_ha"] for r in all_sum); fac=field_area_ha/max(tc,1e-6)
        for r in all_sum: r["Recal_ha"]=round(r["Area_ha"]*fac,4); r["Cal_Factor"]=round(fac,6)
    else:
        for r in all_sum: r["Recal_ha"]=None; r["Cal_Factor"]=None

    if run_id: _prog(run_id,"Saving outputs…",88)
    vcrs=str(dem_crs)
    if all_vec:
        vgdf=gpd.GeoDataFrame(all_vec,crs=vcrs); vgdf=_enforce_poly_gdf(vgdf)
        if not vgdf.empty: vgdf.to_file(os.path.join(out,f"{pfx}_slope_polygon.shp"))
    else: vgdf=gpd.GeoDataFrame(columns=["geometry"],crs=vcrs)
    bgdf.to_file(os.path.join(out,f"{pfx}_boundary_polygon.shp"))
    try:
        with rasterio.open(sp_path) as src:
            fc2,ft2=rio_mask(src,[comp_polygons[0][1].__geo_interface__],crop=True,filled=True,nodata=SN)
        fc2=fc2[0].astype(np.float32); cp2=rp.copy(); cp2.update(height=fc2.shape[0],width=fc2.shape[1],transform=ft2)
        with rasterio.open(os.path.join(out,f"{pfx}_slope_clipped.tif"),"w",**cp2) as dst: dst.write(fc2,1)
        vm2=(fc2!=SN)&~np.isnan(fc2); ca2=np.zeros_like(fc2,dtype=np.uint8)
        ca2[vm2&(fc2<19)]=1; ca2[vm2&(fc2>=19)&(fc2<=31)]=2; ca2[vm2&(fc2>31)]=3
        cp3=cp2.copy(); cp3.update(dtype="uint8",nodata=0)
        with rasterio.open(os.path.join(out,f"{pfx}_slope_classes.tif"),"w",**cp3) as dst: dst.write(ca2,1)
    except: pass
    sdf=pd.DataFrame(all_sum); ep=os.path.join(out,f"{pfx}_slope_summary.xlsx")
    if f_mode=="A":
        sdf.drop(columns=["Label"],errors="ignore").to_excel(ep,index=False)
    else:
        try:
            with pd.ExcelWriter(ep,engine="openpyxl") as wr:
                sdf.to_excel(wr,sheet_name="All_Groups",index=False)
                for lb,grs in per_grp.items():
                    if not grs: continue
                    gdf2=pd.DataFrame(grs); th=sum(r["Area_ha"] for r in grs)
                    tr={"Label":lb,"Class":"","Slope_Range":"TOTAL","Description":"","Area_ha":round(th,4),"Pct_Area":100.0,"Total_ha":round(th,4)}
                    if grs[0].get("Recal_ha") is not None:
                        tr["Recal_ha"]=round(sum(r["Recal_ha"] for r in grs if r.get("Recal_ha")),4)
                        tr["Cal_Factor"]=grs[0].get("Cal_Factor","")
                    pd.concat([gdf2,pd.DataFrame([tr])],ignore_index=True).to_excel(wr,sheet_name=str(lb)[:31],index=False)
        except: sdf.to_excel(ep,index=False)
    if run_id: _prog(run_id,"Group F complete.",95)
    return all_sum,vgdf,bgdf,f_mode,per_grp

# ── PREVIEW FUNCTIONS ────────────────────────────────────────────────────────
def preview_compartments(poly_gdf, path, title="", legend_title="Legend", label_col="Comp_ID"):
    fig,ax=plt.subplots(figsize=(A4W,A4H),dpi=DPI)
    fig.patch.set_facecolor("white"); ax.set_facecolor("#eef5ee")
    handles=[]
    for i,(_,row) in enumerate(poly_gdf.iterrows()):
        geom=row.geometry
        if geom is None or geom.is_empty: continue
        if geom.geom_type=="Polygon":
            ext=list(geom.exterior.coords)
            if ext[0]!=ext[-1]: ext.append(ext[0])
            geom=Polygon(ext)
        color=_CCOLORS[i%len(_CCOLORS)]
        cid=row.get("Comp_ID",f"Comp_{i+1:03d}"); ah=row.get("Area_ha","")
        lbl=f"{cid} ({ah:.2f} ha)" if isinstance(ah,float) else str(cid)
        gpd.GeoDataFrame([{"geometry":geom}],crs=poly_gdf.crs).plot(ax=ax,facecolor=color,edgecolor="#222",linewidth=1.5)
        handles.append(mpatches.Patch(facecolor=color,edgecolor="#222",label=lbl))
    lc_use=label_col if (label_col and label_col in poly_gdf.columns) else ("Comp_ID" if "Comp_ID" in poly_gdf.columns else None)
    if lc_use: _label_feat(ax,poly_gdf,lc_use)
    _style_ax(ax); _north_arrow(ax); _scale_bar(ax)
    _add_legend(ax,handles,legend_title=legend_title or "Legend")
    ax.set_title(title.strip() or "Compartment Division Map",fontsize=12,fontweight="bold",color="#0d1f17",pad=10)
    plt.tight_layout(pad=0.5,rect=[0,0,1,0.97])
    fig.savefig(path,dpi=DPI,bbox_inches="tight",facecolor="white"); plt.close(fig)

def preview(poly_gdf, line_gdf, pts_gdf, path, pc="blue", lc="black", ptc="red",
            label_col=None, label_pts_gdf=None, area_ha=None, title="",
            legend_title="Legend", user_label_col=None):
    fig,ax=plt.subplots(figsize=(A4W,A4H),dpi=DPI)
    fig.patch.set_facecolor("white"); ax.set_facecolor("#eef5ee"); handles=[]
    if poly_gdf is not None and not poly_gdf.empty:
        if "Comp_ID" in poly_gdf.columns:
            for i,(_,row) in enumerate(poly_gdf.iterrows()):
                g=row.geometry
                if g is None or g.is_empty: continue
                col=_CCOLORS[i%len(_CCOLORS)]
                cid=row.get("Comp_ID",f"C{i+1}"); ah=row.get("Area_ha","")
                lbl=f"{cid} ({ah:.2f} ha)" if isinstance(ah,float) else str(cid)
                gpd.GeoDataFrame([{"geometry":g}],crs=poly_gdf.crs).plot(ax=ax,facecolor=col,edgecolor="#222",linewidth=1.4)
                handles.append(mpatches.Patch(facecolor=col,edgecolor="#222",label=lbl))
        else:
            poly_gdf.plot(ax=ax,facecolor="none",edgecolor="#1565C0",linewidth=1.6)
            handles.append(mpatches.Patch(facecolor="none",edgecolor="#1565C0",linewidth=1.6,label="Forest Boundary"))
    if line_gdf is not None and not line_gdf.empty:
        line_gdf.plot(ax=ax,color=lc,linewidth=1.1)
    if pts_gdf is not None and not pts_gdf.empty:
        pts_gdf.plot(ax=ax,color=ptc,markersize=5,zorder=5)
        handles.append(mpatches.Patch(facecolor=ptc,label="Survey Points"))
    lbl_src=label_pts_gdf if label_pts_gdf is not None else pts_gdf
    lc_use=user_label_col or label_col
    if lc_use and lbl_src is not None and not lbl_src.empty and lc_use in lbl_src.columns:
        for _,row in lbl_src.iterrows():
            try:
                ax.annotate(str(row[lc_use]),xy=(row.geometry.x,row.geometry.y),
                            xytext=(0,7),textcoords="offset points",
                            ha="center",va="bottom",fontsize=5.5,fontweight="bold",color="black",
                            path_effects=[pe.Stroke(linewidth=1.8,foreground="white"),pe.Normal()],zorder=6)
            except: pass
    if poly_gdf is not None and not poly_gdf.empty and "Comp_ID" in poly_gdf.columns:
        _label_feat(ax,poly_gdf,"Comp_ID")
    elif poly_gdf is not None and not poly_gdf.empty and "Forest" in poly_gdf.columns:
        _label_feat(ax,poly_gdf,"Forest")
    _style_ax(ax); _north_arrow(ax); _scale_bar(ax)
    _add_legend(ax,handles,legend_title=legend_title or "Legend")
    head=title.strip() if title.strip() else (f"Forest Area: {area_ha:.2f} ha" if area_ha else "Forest Boundary Map")
    ax.set_title(head,fontsize=12,fontweight="bold",color="#0d1f17",pad=10)
    plt.tight_layout(pad=0.5,rect=[0,0,1,0.97])
    fig.savefig(path,dpi=DPI,bbox_inches="tight",facecolor="white"); plt.close(fig)

def preview_slope(vec_gdf, bgdf, summary_rows, path, f_mode="A",
                  per_group_summaries=None, title="", legend_title="Slope Classes"):
    cc={1:"#2ecc71",2:"#f39c12",3:"#e74c3c"}
    cl={1:"< 19°  Gentle",2:"19–31° Moderate",3:"> 31°  Steep"}
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
    handles=[mpatches.Patch(facecolor=c,edgecolor="#2c2c2c",label=cl[cid]) for cid,c in cc.items()]
    handles.append(mpatches.Patch(facecolor="none",edgecolor="black",linewidth=1.8,label="Boundary"))
    _add_legend(ax1,handles,legend_title=legend_title or "Slope Classes")
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
        rc={1:"#d5f5e3",2:"#fdebd0",3:"#fadbd8",0:"#f8f9fa",-1:"#ddeeff"}
        for (r2,c2),cell in tbl.get_celld().items():
            cell.set_edgecolor("#ccc")
            if r2==0: cell.set_facecolor("#1a5276"); cell.set_text_props(color="white",fontweight="bold")
            else:
                cid2=rcm.get(r2-1,0)
                cell.set_facecolor(rc.get(cid2,"#f8f9fa"))
                if cid2==-1 or (cid2==0 and r2-1>=len(summary_rows)): cell.set_text_props(fontweight="bold")
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
    pw=w84(poly_gdf); linewidth=w84(line_gdf); ptw=w84(pts_gdf)
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

# ── ROUTES ───────────────────────────────────────────────────────────────────
@app.route("/progress/<run_id>")
def progress_stream(run_id):
    def gen():
        sent=0
        import time
        for _ in range(1200):
            msgs=_PROG.get(run_id,[])
            if len(msgs)>sent:
                for m in msgs[sent:]: yield f"data: {m}\n\n"
                sent=len(msgs)
                try:
                    last=json.loads(msgs[-1])
                    if last.get("pct",0)>=100: break
                except: pass
            time.sleep(0.2)
    return Response(stream_with_context(gen()),mimetype="text/event-stream",
                    headers={"Cache-Control":"no-cache","X-Accel-Buffering":"no"})

@app.route("/geojson/<run_id>")
def get_geojson(run_id):
    folder=os.path.join(OUTPUT,run_id)
    if not os.path.exists(folder): return jsonify({"type":"FeatureCollection","features":[]}),200
    shps=[os.path.join(r,f) for r,_,fs in os.walk(folder) for f in fs if f.endswith(".shp") and "polygon" in f.lower()]
    if not shps: return jsonify({"type":"FeatureCollection","features":[]}),200
    gdfs=[]
    for shp in shps:
        try:
            g=gpd.read_file(shp)
            if g.crs is not None: g=g.to_crs("EPSG:4326")
            gdfs.append(g)
        except: pass
    if not gdfs: return jsonify({"type":"FeatureCollection","features":[]}),200
    combined=gpd.GeoDataFrame(pd.concat(gdfs,ignore_index=True),crs="EPSG:4326")
    return Response(combined.to_json(),mimetype="application/json")

@app.route("/compose/<run_id>",methods=["POST"])
def compose_map(run_id):
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
def save_edit(run_id):
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

@app.route("/login",methods=["POST"])
def login():
    data=request.get_json(silent=True) or {}; username=(data.get("username") or "").strip()
    if not username or len(username)<2: return jsonify({"error":"Username too short (min 2 chars)."}),400
    if len(username)>40: return jsonify({"error":"Username too long (max 40 chars)."}),400
    if not re.match(r'^[A-Za-z0-9 _\-]+$',username): return jsonify({"error":"Only letters, numbers, spaces, - and _ allowed."}),400
    if _user_exists(username): return jsonify({"error":f'"{username}" is already taken.', "taken":True}),409
    try: _create_user(username)
    except ValueError as e: return jsonify({"error":str(e),"taken":True}),409
    session["username"]=username; session.permanent=True
    return jsonify({"ok":True,"username":username,"runs":[]})

@app.route("/logout",methods=["POST"])
def logout(): session.clear(); return jsonify({"ok":True})

@app.route("/me")
def me():
    u=_require_login()
    if not u: return jsonify({"error":"Not logged in"}),401
    users=_lu(); user=users.get(u,{})
    return jsonify({"username":u,"runs":user.get("runs",[])[-20:]})

@app.route("/history")
def history():
    u=_require_login()
    if not u: return jsonify({"error":"Not logged in"}),401
    users=_lu(); user=users.get(u,{})
    return jsonify({"runs":user.get("runs",[])})

@app.route("/upload",methods=["POST"])
def upload():
    run_id=str(uuid.uuid4()); _PROG[run_id]=[]
    try:
        if "file" not in request.files: return jsonify({"error":"No file uploaded.","run_id":run_id}),400
        file=request.files["file"]; module=request.form.get("module","A")
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
            dem_f=request.files.get("dem_file")
            if not dem_f: return jsonify({"error":"No DEM file uploaded.","run_id":run_id}),400
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

@app.route("/outputs/<run_id>/<path:filename>")
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
