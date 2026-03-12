import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TransformStamped
import math

PUBLISH_HZ = 20.0
SPEED = 0.7

WAYPOINTS = [
    (0.5, 0.0),
    (0.5, 1.5),
    (1.5, 1.5),
    (1.5, 0.0),
]


class FakeHumanPublisher(Node):
    def __init__(self):
        super().__init__('fake_human_publisher')

        self._pub = self.create_publisher(
            TransformStamped, 'vicon/helmet/root', 10
        )

        self._x = WAYPOINTS[0][0]
        self._y = WAYPOINTS[0][1]
        self._waypoint_idx = 1
        self._heading = 0.0

        self._timer = self.create_timer(1.0 / PUBLISH_HZ, self._tick)
        self.get_logger().info('fake_human_publisher started')

    def _tick(self):
        tx, ty = WAYPOINTS[self._waypoint_idx]
        dx = tx - self._x
        dy = ty - self._y
        dist = math.sqrt(dx ** 2 + dy ** 2)

        if dist < 0.05:
            self._waypoint_idx = (self._waypoint_idx + 1) % len(WAYPOINTS)
            tx, ty = WAYPOINTS[self._waypoint_idx]
            dx = tx - self._x
            dy = ty - self._y
            dist = math.sqrt(dx ** 2 + dy ** 2)

        step = SPEED / PUBLISH_HZ
        self._x += (dx / dist) * step
        self._y += (dy / dist) * step
        self._heading = math.atan2(dy, dx)

        msg = TransformStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'world'
        msg.child_frame_id = 'helmet'
        msg.transform.translation.x = self._x
        msg.transform.translation.y = self._y
        msg.transform.translation.z = 0.0
        msg.transform.rotation.w = math.cos(self._heading / 2)
        msg.transform.rotation.z = math.sin(self._heading / 2)

        self._pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = FakeHumanPublisher()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()