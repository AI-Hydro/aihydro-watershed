import sys,json,time,traceback,os
from aihydro_watershed.delineation import merit_flowdir_pipeline as m
root=sys.argv[1]; out=sys.argv[2]; ids=sys.argv[3:]
G=json.load(open(root+'/gauges.json')); res=json.load(open(out)) if os.path.exists(out) else {}
for gid in ids:
    lat,lon=G[gid][:2]; t=time.time()
    try:
        r=m.delineate_merit_gee(lat,lon)
        rec=dict(ok=True,area=r.area_km2,snap_m=r.snap_distance_m,snap_upa=r.snapped_upa_km2,touch=r.touches_edge,flags=r.quality_flags,bbox=r.bbox,wkt=r.gdf.geometry.iloc[0].wkt)
    except Exception as e:
        rec=dict(ok=False,err=str(e)[:300],kind=type(e).__name__)
    rec['secs']=round(time.time()-t,1); res[gid]=rec
    print(gid,{k:v for k,v in rec.items() if k!='wkt'},flush=True); json.dump(res,open(out,'w'))
