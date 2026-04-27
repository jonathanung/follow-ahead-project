import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TransformStamped
import tf2_ros

class ViconBridge(Node):
    def __init__(self):
        super().__init__('vicon_bridge')
        self.declare_parameter('human_tf', 'vicon/follow_ahead_human1/follow_ahead_human1')
        self.declare_parameter('robot_tf', 'vicon/qbot_follow_ahead/qbot_follow_ahead')
        self.declare_parameter('world_frame', 'world')
        self.human_tf = self.get_parameter('human_tf').get_parameter_value().string_value
        self.robot_tf = self.get_parameter('robot_tf').get_parameter_value().string_value
        self.world = self.get_parameter('world_frame').get_parameter_value().string_value
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.pub_human = self.create_publisher(TransformStamped, 'vicon/helmet/root', 10)
        self.pub_robot = self.create_publisher(TransformStamped, 'vicon/robot/root', 10)
        self.create_timer(0.05, self.update)
        self.get_logger().info(f'vicon_bridge ready | human={self.human_tf} robot={self.robot_tf}')

    def update(self):
        now = rclpy.time.Time()
        self._publish(self.human_tf, 'helmet', self.pub_human, now)
        self._publish(self.robot_tf, 'base_link', self.pub_robot, now)

    def _publish(self, source_frame, child_frame, publisher, now):
        try:
            t = self.tf_buffer.lookup_transform(self.world, source_frame, now)
            out = TransformStamped()
            out.header.stamp = self.get_clock().now().to_msg()
            out.header.frame_id = 'map'
            out.child_frame_id = child_frame
            out.transform = t.transform
            publisher.publish(out)
        except Exception:
            pass
            
def main():
    rclpy.init()
    node = ViconBridge()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
