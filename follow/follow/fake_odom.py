import math
import os
import rclpy
import numpy as np
from rclpy.node import Node
from geometry_msgs.msg import Twist, TransformStamped
from nav_msgs.msg import Odometry
from tf2_ros import TransformBroadcaster


class FakeOdom(Node):
    """Integrates /cmd_vel into a nav_msgs/Odometry on /odom.

    Models real QBot Platform dynamics:
      - max linear velocity:  0.6 m/s
      - max angular velocity: 0.5 rad/s
      - max linear accel:     0.5 m/s²
      - max angular accel:    0.5 rad/s²
    """

    # QBot Platform hardware limits
    MAX_LINEAR_VEL   = 0.6   # m/s
    MAX_ANGULAR_VEL  = 0.5   # rad/s
    MAX_LINEAR_ACCEL = 0.5   # m/s²
    MAX_ANGULAR_ACCEL = 0.5  # rad/s²

    def __init__(self):
        super().__init__('fake_odom')
        self.declare_parameter('publish_hz', 50.0)
        self.declare_parameter('start_x', 0.0)
        self.declare_parameter('start_y', -1.5)
        self.declare_parameter('start_theta', 0.0)
        self.declare_parameter('map_yaml_path', '')

        hz = self.get_parameter('publish_hz').get_parameter_value().double_value
        self.x     = self.get_parameter('start_x').get_parameter_value().double_value
        self.y     = self.get_parameter('start_y').get_parameter_value().double_value
        self.theta = self.get_parameter('start_theta').get_parameter_value().double_value

        self._occ        = None  # occupancy array (True = blocked)
        self._map_origin = None
        self._map_res    = None
        self._map_h      = None
        self._map_w      = None
        map_yaml = self.get_parameter('map_yaml_path').get_parameter_value().string_value
        if map_yaml:
            self._load_map(map_yaml)

        self.vx = 0.0   # current actual linear velocity
        self.wz = 0.0   # current actual angular velocity
        self.cmd_vx = 0.0  # commanded linear velocity
        self.cmd_wz = 0.0  # commanded angular velocity
        self.dt = 1.0 / hz

        self.create_subscription(Twist, '/cmd_vel', self._cmd_cb, 10)
        self.pub = self.create_publisher(Odometry, '/odom', 10)
        self.tf_broadcaster = TransformBroadcaster(self)
        self.create_timer(self.dt, self._publish)

        self.get_logger().info(
            f'fake_odom: start=({self.x:.2f}, {self.y:.2f}, {math.degrees(self.theta):.1f}°) '
            f'| max_vel=({self.MAX_LINEAR_VEL}, {self.MAX_ANGULAR_VEL}) '
            f'accel=({self.MAX_LINEAR_ACCEL}, {self.MAX_ANGULAR_ACCEL})'
        )

    def _load_map(self, yaml_path: str) -> None:
        try:
            import yaml
            from PIL import Image
            with open(yaml_path) as f:
                meta = yaml.safe_load(f)
            pgm_path = os.path.join(os.path.dirname(yaml_path), meta['image'])
            img = np.array(Image.open(pgm_path), dtype=np.float32)
            # ROS convention (negate=0): prob_occ = (255 - pixel) / 255
            # Blocked if prob_occ > occupied_thresh  →  pixel < (1-thresh)*255
            occ_thresh = meta.get('occupied_thresh', 0.65)
            self._occ        = img < (1.0 - occ_thresh) * 255.0
            self._map_origin = meta['origin'][:2]  # [ox, oy]
            self._map_res    = float(meta['resolution'])
            self._map_h, self._map_w = self._occ.shape
            self.get_logger().info(
                f'fake_odom: map loaded from {yaml_path} '
                f'({self._map_w}×{self._map_h}px, res={self._map_res}m/px)'
            )
        except Exception as e:
            self.get_logger().warn(f'fake_odom: could not load map ({e}) — no boundary enforcement')
            self._occ = None

    def _is_blocked(self, x: float, y: float) -> bool:
        ox, oy = self._map_origin
        col = int((x - ox) / self._map_res)
        row = self._map_h - 1 - int((y - oy) / self._map_res)  # PGM row 0 = top
        if col < 0 or col >= self._map_w or row < 0 or row >= self._map_h:
            return True  # out of map bounds
        return bool(self._occ[row, col])

    def _cmd_cb(self, msg: Twist):
        # Clamp commanded velocities to hardware limits before storing
        self.cmd_vx = max(-self.MAX_LINEAR_VEL,  min(self.MAX_LINEAR_VEL,  msg.linear.x))
        self.cmd_wz = max(-self.MAX_ANGULAR_VEL, min(self.MAX_ANGULAR_VEL, msg.angular.z))

    def _ramp(self, current: float, target: float, max_delta: float) -> float:
        """Ramp current toward target, limited by max_delta per step."""
        delta = target - current
        delta = max(-max_delta, min(max_delta, delta))
        return current + delta

    def _publish(self):
        max_dv  = self.MAX_LINEAR_ACCEL  * self.dt
        max_dw  = self.MAX_ANGULAR_ACCEL * self.dt

        self.vx = self._ramp(self.vx, self.cmd_vx, max_dv)
        self.wz = self._ramp(self.wz, self.cmd_wz, max_dw)

        new_x = self.x + self.vx * math.cos(self.theta) * self.dt
        new_y = self.y + self.vx * math.sin(self.theta) * self.dt
        if self._occ is not None and self._is_blocked(new_x, new_y):
            self.vx = 0.0
            self.wz = 0.0
        else:
            self.theta += self.wz * self.dt
            self.theta = (self.theta + math.pi) % (2 * math.pi) - math.pi
            self.x, self.y = new_x, new_y

        odom = Odometry()
        now = self.get_clock().now().to_msg()
        odom.header.stamp = now
        odom.header.frame_id = 'odom'
        odom.child_frame_id = 'base_footprint'

        odom.pose.pose.position.x = self.x
        odom.pose.pose.position.y = self.y

        half = self.theta / 2.0
        qz = math.sin(half)
        qw = math.cos(half)
        odom.pose.pose.orientation.z = qz
        odom.pose.pose.orientation.w = qw

        odom.twist.twist.linear.x  = self.vx
        odom.twist.twist.angular.z = self.wz

        self.pub.publish(odom)

        t = TransformStamped()
        t.header.stamp = now
        t.header.frame_id = 'odom'
        t.child_frame_id = 'base_footprint'
        t.transform.translation.x = self.x
        t.transform.translation.y = self.y
        t.transform.translation.z = 0.0
        t.transform.rotation.x = 0.0
        t.transform.rotation.y = 0.0
        t.transform.rotation.z = qz
        t.transform.rotation.w = qw
        self.tf_broadcaster.sendTransform(t)


def main():
    rclpy.init()
    node = FakeOdom()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
