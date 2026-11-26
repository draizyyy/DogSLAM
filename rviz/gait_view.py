"""RViz schematic of the Go2. Body pose from DogSLAM, joints from recorded motor telemetry."""
import sys
from math import sin, cos
import numpy as np
import rclpy
from geometry_msgs.msg import Point, PoseStamped, TransformStamped
from rclpy.node import Node
from scipy.spatial.transform import Rotation
from tf2_ros import TransformBroadcaster
from visualization_msgs.msg import Marker, MarkerArray


class DogView(Node):
    def __init__(self, archive):
        super().__init__("dogslam_gait_view")
        self.times = archive["times"]
        self.angles = archive["angles"]
        self.pub = self.create_publisher(MarkerArray, "/dogslam/robot_model", 5)
        self.tf = TransformBroadcaster(self)
        self.create_subscription(PoseStamped, "/dogslam/base_pose", self.draw, 10)

    def draw(self, msg):
        stamp = msg.header.stamp
        t = stamp.sec + stamp.nanosec * 1e-9
        i = np.searchsorted(self.times, t)
        i = min(max(i, 1), len(self.times) - 1)
        i -= int(abs(self.times[i-1] - t) < abs(self.times[i] - t))
        if abs(self.times[i] - t) > 0.12:
            return
        p, q = msg.pose.position, msg.pose.orientation
        transform = TransformStamped()
        transform.header = msg.header
        transform.child_frame_id = "dogslam_camera"
        transform.transform.translation.x = p.x
        transform.transform.translation.y = p.y
        transform.transform.translation.z = p.z
        transform.transform.rotation.w = 1.
        self.tf.sendTransform(transform)
        rotation = Rotation.from_quat([q.x, q.y, q.z, q.w])
        markers = []
        def point(xyz):
            v = rotation.apply(xyz) + [p.x, p.y, p.z]
            return Point(x=v[0], y=v[1], z=v[2])
        def part(kind, xyz, size, color):
            m = Marker()
            m.header = msg.header
            m.ns = "dog"
            m.id = len(markers)
            m.type = kind
            m.pose.position = point(xyz)
            m.pose.orientation.x, m.pose.orientation.y = q.x, q.y
            m.pose.orientation.z, m.pose.orientation.w = q.z, q.w
            m.scale.x, m.scale.y, m.scale.z = size
            m.color.r, m.color.g, m.color.b = color
            m.color.a = 1.
            markers.append(m)
            return m
        white, orange, slate = (0.90,0.95,1.), (1.,0.45,0.09), (0.32,0.44,0.55)
        part(Marker.CUBE, [0,0,0], (.48,.19,.14), white)
        part(Marker.SPHERE, [.23,0,.035], (.17,.18,.15), orange)
        for leg, (hip, thigh, calf) in enumerate(self.angles[i].reshape(4,3)):
            x = .19 if leg < 2 else -.19
            side = (-1,1)[leg % 2]
            def limb(dx, dz):
                lateral = side * .075
                return [x + dx, side*.05 + lateral*cos(hip)-dz*sin(hip),
                        lateral*sin(hip)+dz*cos(hip)]
            knee = limb(-.213*sin(thigh), -.213*cos(thigh))
            foot = limb(-.213*(sin(thigh)+sin(thigh+calf)), -.213*(cos(thigh)+cos(thigh+calf)))
            line = part(Marker.LINE_STRIP, [0,0,0], (.032,0.,0.), slate)
            line.pose.position = Point()
            line.pose.orientation.w = 1.
            line.pose.orientation.x = line.pose.orientation.y = line.pose.orientation.z = 0.
            line.points = [point(limb(0,0)),point(knee),point(foot)]
            part(Marker.SPHERE, foot, (.053,.043,.04), orange)
        self.pub.publish(MarkerArray(markers=markers))


if __name__ == "__main__":
    rclpy.init()
    rclpy.spin(DogView(np.load(sys.argv[1])))

