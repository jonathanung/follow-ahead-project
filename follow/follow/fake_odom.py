import math
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry


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

        hz = self.get_parameter('publish_hz').get_parameter_value().double_value
        self.x     = self.get_parameter('start_x').get_parameter_value().double_value
        self.y     = self.get_parameter('start_y').get_parameter_value().double_value
        self.theta = self.get_parameter('start_theta').get_parameter_value().double_value

        self.vx = 0.0   # current actual linear velocity
        self.wz = 0.0   # current actual angular velocity
        self.cmd_vx = 0.0  # commanded linear velocity
        self.cmd_wz = 0.0  # commanded angular velocity
        self.dt = 1.0 / hz

        self.create_subscription(Twist, '/cmd_vel', self._cmd_cb, 10)
        self.pub = self.create_publisher(Odometry, '/odom', 10)
        self.create_timer(self.dt, self._publish)

        self.get_logger().info(
            f'fake_odom: start=({self.x:.2f}, {self.y:.2f}, {math.degrees(self.theta):.1f}°) '
            f'| max_vel=({self.MAX_LINEAR_VEL}, {self.MAX_ANGULAR_VEL}) '
            f'accel=({self.MAX_LINEAR_ACCEL}, {self.MAX_ANGULAR_ACCEL})'
        )

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

        self.theta += self.wz * self.dt
        self.theta = (self.theta + math.pi) % (2 * math.pi) - math.pi
        self.x += self.vx * math.cos(self.theta) * self.dt
        self.y += self.vx * math.sin(self.theta) * self.dt

        odom = Odometry()
        odom.header.stamp = self.get_clock().now().to_msg()
        odom.header.frame_id = 'odom'
        odom.child_frame_id = 'base_link'

        odom.pose.pose.position.x = self.x
        odom.pose.pose.position.y = self.y

        half = self.theta / 2.0
        odom.pose.pose.orientation.z = math.sin(half)
        odom.pose.pose.orientation.w = math.cos(half)

        odom.twist.twist.linear.x  = self.vx
        odom.twist.twist.angular.z = self.wz

        self.pub.publish(odom)


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
