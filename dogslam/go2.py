"""Go2 deskewed clouds are already in odom. Estimate map_T_odom, then map_T_base."""
import argparse, csv, json, time
from pathlib import Path
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.signal import fftconvolve
from scipy.spatial.transform import Rotation
import yaml
from dogslam.geom import pose, apply, voxel, write_tum
from dogslam.registration import Index, register

CLOUD='/utlidar/cloud_deskewed'
ODOM='/utlidar/robot_odom'

def clean(points, size, cap=None):
    return voxel(points[np.isfinite(points).all(axis=1)],size,cap)

def map_points(path):
    from dogslam.bags import open_bag, clouds
    parts=[]
    with open_bag(path) as r:
        for i,(c,_,raw) in enumerate(r.messages(connections=[c for c in r.connections if c.topic==CLOUD])):
            if i%5==0:
                parts.append(clean(clouds(r.deserialize(raw,c.msgtype),None)[0],0.10))
    return clean(np.vstack(parts),0.10).astype('float32')

def initial(scan, target):
    # Search translations in 3D and yaw; both odom frames use gravity alignment.
    step=.48
    def grid(p):
        q=np.floor(p/step).astype(int)
        lo=q.min(axis=0); q-=lo
        a=np.zeros(tuple(q.max(axis=0)+1),np.float32); a[tuple(q.T)]=1
        return a,lo
    m,ml=grid(target)
    m=gaussian_filter(m,.65)
    best=(-1,None)
    for yaw in np.arange(-180,180,10):
        T=pose(Rotation.from_euler('z',yaw,degrees=True))
        a,al=grid(apply(T,scan))
        score=fftconvolve(m,a[::-1,::-1,::-1])
        k=np.unravel_index(score.argmax(),score.shape)
        val=score[k]/max(1,a.sum())
        if val>best[0]:
            T[:3,3]=(np.array(k)-np.array(a.shape)+1+ml-al)*step
            best=val,T
    return best[1]

class Tracker:
    def __init__(self,points):
        self.points=points
        self.cfg=yaml.safe_load((Path(__file__).resolve().parents[1]/"config/default.yaml").read_text(encoding="utf-8"))
        self.index=Index(points,self.cfg)

        self.T=None
    def update(self,points):
        scan=clean(points,.17,2000)
        if self.T is None:
            self.T=initial(scan,self.points)
        fit=register(scan,self.index,self.T,self.cfg,radius=.9)
        if fit.ok: self.T=fit.T
        return self.T, fit

def main():
    from dogslam.bags import open_bag, clouds, stamp, xyz, quat
    out=Path('runs/jury-final');out.mkdir(parents=True,exist_ok=True)
    points=map_points('data/jury/bag_for_mapping')
    np.save(out/'map.npy',points)
    tracker=Tracker(points); ts=[]; poses=[]; rows=[]; last=None
    with open_bag('data/jury/bag_for_localization') as r:
        for c,ns,raw in r.messages(connections=[c for c in r.connections if c.topic in (CLOUD,ODOM)]):
            t0=time.perf_counter()
            msg=r.deserialize(raw,c.msgtype)
            if c.topic==ODOM:
                p=msg.pose.pose; last=pose(quat(p.orientation),xyz(p.position));continue
            if last is None:continue
            T,fit=tracker.update(clouds(msg,None)[0])
            ts.append(stamp(msg,ns,'header'));poses.append(T@last)
            rows.append([ts[-1],(time.perf_counter()-t0)*1000,fit.ok,fit.overlap,fit.residual])
    write_tum(out/'trajectory.tum',ts,poses)
    with (out/'timing.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['stamp','processing_ms','accepted','overlap','residual_m']);w.writerows(rows)
    print(len(points),len(rows),sum(r[2] for r in rows),np.percentile([r[1] for r in rows],[50,95,99]),rows[0],flush=True)
if __name__=='__main__':main()



