import ee,json,math,time,sys,requests,io,zipfile
ee.Initialize()
from aihydro_watershed.delineation import merit_flowdir_pipeline as m
lat,lon=json.load(open(sys.argv[1]+'/gauges.json'))['01547700'][:2]
PX=1/1200; X0=-180-0.5*PX; Y0=85-0.5*PX
img=ee.Image('MERIT/Hydro/v1_0_1').select(['dir','upa','wth'])
for hs,clip in ((32.0,False),(51.2,False),(64.0,False),(82.0,False)):
    x0,y0,x1,y1=m.bbox_from_point(lat,lon,hs)
    c0=math.floor((x0-X0)/PX); c1=math.ceil((x1-X0)/PX); r0=math.floor((Y0-y1)/PX); r1=math.ceil((Y0-y0)/PX)
    W,H=c1-c0,r1-r0
    xl,xr=X0+c0*PX,X0+c1*PX; yt,yb=Y0-r0*PX,Y0-r1*PX
    region=ee.Geometry.Rectangle([xl,yb,xr,yt],proj='EPSG:4326',geodesic=False)
    t=time.time()
    try:
        url=img.getDownloadURL({'name':'x','crs':'EPSG:4326','crs_transform':[PX,0,xl,0,-PX,yt],'dimensions':f'{W}x{H}','dimensions':f'{W}x{H}','filePerBand':True,'format':'GEO_TIFF'})
        r=requests.get(url,timeout=180)
        print(hs,(H,W),r.status_code,len(r.content),r.text[:100] if not r.ok else '', '%.1fs'%(time.time()-t),flush=True)
    except Exception as e:
        print(hs,(H,W),'EXC',str(e)[:150],flush=True)
