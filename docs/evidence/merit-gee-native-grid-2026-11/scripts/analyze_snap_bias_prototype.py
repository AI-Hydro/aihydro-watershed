import sys,json,math,numpy as np,pandas as pd,geopandas as gpd,shapely
from shapely import wkt
from aihydro_watershed.delineation import merit_flowdir_pipeline as m
root=sys.argv[1]; E=sys.argv[2]
live=json.load(open(root+'/work/live_after.json'))
ref=pd.read_csv(E+'/reference/reference_671.csv',dtype={'gauge_id':str}).set_index('gauge_id')
poly=gpd.read_file(E+'/reference/reference_polygons.gpkg').set_index('GAGEID')
G=json.load(open(root+'/gauges.json'))
def nonet(*a): raise RuntimeError('network disabled in analysis')
rows=[]
for gid,r in sorted(live.items()):
    if not r.get('ok'): rows.append(dict(gauge=gid,ok=False)); continue
    lat,lon=G[gid][:2]
    ap=float(ref.loc[gid,'area_polygon_km2']); ag=float(ref.loc[gid,'area_gages2']); agf=float(ref.loc[gid,'area_geospa_fabric'])
    g=gpd.GeoSeries([wkt.loads(r['wkt'])],crs=4326).to_crs(5070).iloc[0]
    rp=poly.loc[gid].geometry
    inter=g.intersection(rp).area; iou=inter/g.union(rp).area
    # snap analysis from cached tiles
    w=m.window_from_bbox(m.bbox_from_point(lat,lon,4.0))
    b=m.fetch_merit_window(w,compute_pixels=nonet)
    upa=b.bands['upa'].astype(float); wth=b.bands['wth']
    cf,rf=~b.transform*(lon,lat)
    rr,cc=np.indices(upa.shape); xs,ys=b.transform*(cc+.5,rr+.5)
    dist=np.hypot((xs-lon)*111320*math.cos(math.radians(lat)),(ys-lat)*110574)
    stream=(upa>=1.0)&((wth>0)|(upa>=1))
    cand=stream&(dist<=1000)
    err=np.where(cand,np.abs(upa-ap)/ap,np.inf); i=np.unravel_index(np.argmin(err),err.shape)
    rows.append(dict(gauge=gid,ok=True,A_poly=ap,A_gages2=agf and ag,A_gf=agf,A_new=r['area'],upa_snap=r['snap_upa'],snap_m=r['snap_m'],
        e_poly=r['area']/ap-1,e_gages2=r['area']/ag-1,e_gf=r['area']/agf-1,IoU=iou,
        best_cell_upa=upa[i],best_cell_dist_m=dist[i],best_cell_err=upa[i]/ap-1, iters=r['iters'],cells=r['cells'],req=r['stats']['requests'],secs=r['secs']))
df=pd.DataFrame(rows); pd.set_option('display.width',250); pd.set_option('display.max_columns',30)
print(df.round(3).to_string()); df.to_csv(root+'/work/after_table.csv',index=False)
d=df[df.ok==True]
print('n ok',len(d),'mean e_poly %.4f median %.4f'%(d.e_poly.mean(),d.e_poly.median()),'mean|e|',d.e_poly.abs().mean(),'IoU mean',d.IoU.mean(),d.IoU.min())
print('vs gages2: mean %.4f median %.4f'%(d.e_gages2.mean(),d.e_gages2.median()),' vs GF mean %.4f'%d.e_gf.mean())
