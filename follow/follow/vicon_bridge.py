import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, TransformStamped

class ViconBridge(Node):
    def __init__(self):
        super().__init__('vicon_bridge')
        self.declare_parameter('human_subject', 'helmet')
        self.declare_parameter('human_segment', 'root')
        self.declare_parameter('robot_subject', 'qbot')
        self.declare_parameter('robot_segment', 'root')
        self.declare_parameter('vicon_namespace', 'vicon')

        ns        = self.get_parameter('vicon_namespace').get_parameter_value().string_value
        human_sub = self.get_parameter('human_subject').get_parameter_value().string_value
        human_seg = self.get_parameter('human_segment').get_parameter_value().string_value
        robot_sub = self.get_parameter('robot_subject').get_parameter_value().string_value
        robot_seg = self.get_parameter('robot_segment').get_parameter_value().string_value

        self.pub_human = self.create_publisher(TransformStamped, 'vicon/helmet/root', 10)
        self.pub_robot = self.create_publisher(TransformStamped, 'vicon/robot/root', 10)

        self.create_subscription(PoseStamped, f'/{ns}/{human_sub}/{human_seg}', self.human_cb, 10)
        self.create_subscription(PoseStamped, f'/{ns}/{robot_sub}/{robot_seg}', self.robot_cb, 10)

        self.get_logger().info(f'vicon_bridge ready | human=/{ns}/{human_sub}/{human_seg} robot=/{ns}/{robot_sub}/{robot_seg}')

    def _convert(self, msg: PoseStamped, child_frame: str) -> TransformStamped:
        t = TransformStamped()
        t.header = msg.header
        t.header.frame_id = 'map'
        t.child_frame_id = child_frame
        t.transform.translation.x = msg.pose.position.x
        t.transform.translation.y = msg.pose.position.y
        t.transform.translation.z = msg.pose.position.z
        t.transform.rotation = msg.pose.orientation
        return t

    def human_cb(self, msg: PoseStamped):
        self.pub_human.publish(self._convert(msg, 'helmet'))

    def robot_cb(self, msg: PoseStamped):
        self.pub_robot.publish(self._convert(msg, 'base_link'))

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
