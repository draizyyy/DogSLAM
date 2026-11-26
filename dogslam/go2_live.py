import csv,time
from pathlib import Path
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry,Path as Route
from sensor_msgs.msg import PointCloud2,PointField
from std_msgs.msg import Header
from scipy.spatial.transform import Rotation
from dogslam.go2 import Tracker,CLOUD,ODOM
from dogslam.geom import pose

class Live(Node):
    def __init__(self):
        super().__init__('dogslam_localizer')
        points=np.load('runs/jury-final/map.npy')
        self.tracker=Tracker(points)
        self.pc=PointCloud2(header=Header(frame_id='map'),height=1,width=len(points),point_step=12,row_step=12*len(points),is_dense=True)
        self.pc.fields=[PointField(name=n,offset=4*i,datatype=7,count=1) for i,n in enumerate('xyz')]
        self.pc.data=points.astype('<f4').tobytes()
        self.map_pub=self.create_publisher(PointCloud2,'/dogslam/map3d',2)
        self.pub=self.create_publisher(PoseStamped,'/dogslam/base_pose',10)
        self.route=self.create_publisher(Route,'/dogslam/path',2)
        self.path=Route();self.last=None
        self.create_subscription(Odometry,ODOM,self.odom,qos_profile_sensor_data)
        self.create_subscription(PointCloud2,CLOUD,self.scan,qos_profile_sensor_data)
        self.file=open('runs/jury-final/live-timing.csv','w',buffering=1)
        self.writer=csv.writer(self.file);self.writer.writerow(['stamp','callback_ms','accepted','overlap','residual_m'])
    def odom(self,m):
        p=m.pose.pose;q=p.orientation;t=p.position
        self.last=pose(Rotation.from_quat([q.x,q.y,q.z,q.w]),[t.x,t.y,t.z])
    def scan(self,m):
        if self.last is None:return
        start=time.perf_counter()
        dtype=np.dtype(dict(names=list('xyz'),formats=['<f4']*3,offsets=[next(f.offset for f in m.fields if f.name==n) for n in 'xyz'],itemsize=m.point_step))
        a=np.ndarray((m.height,m.width),dtype=dtype,buffer=m.data,strides=(m.row_step,m.point_step))
        T,fit=self.tracker.update(np.column_stack([a[n].ravel() for n in 'xyz']))
        P=T@self.last;out=PoseStamped();out.header=Header(frame_id='map',stamp=m.header.stamp)
        out.pose.position.x,out.pose.position.y,out.pose.position.z=P[:3,3]
        q=out.pose.orientation;q.x,q.y,q.z,q.w=Rotation.from_matrix(P[:3,:3]).as_quat()
        self.pub.publish(out);self.path.header=out.header;self.path.poses.append(out)
        if len(self.path.poses)%5==1:self.route.publish(self.path)
        if len(self.path.poses)%70==1:self.pc.header=out.header;self.map_pub.publish(self.pc)
        self.writer.writerow([m.header.stamp.sec+m.header.stamp.nanosec*1e-9,(time.perf_counter()-start)*1000,fit.ok,fit.overlap,fit.residual])
rclpy.init();rclpy.spin(Live())
