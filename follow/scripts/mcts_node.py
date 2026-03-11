import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from geometry_msgs.msg import Twist, TransformStamped
from tf2_ros import TransformBroadcaster
import threading
import math

from planner import MCTSPlanner

LINEAR_SPEED = 0.3
ANGULAR_SPEED = 0.5
MCTS_HZ = 5.0
REPUBLISH_HZ = 50.0

ACTION_TO_HEADING = {
    'N': 0.0,
    'S': math.pi,
    'E': -math.pi / 2,
    'W': math.pi / 2,
}


def _fake_state():
    return {
        'robot_pos': (3, 4),
        'human_pos': (3, 3),
        'human_heading': 'N',
    }


class MCTSNode(Node):
    def __init__(self):
        super().__init__('mcts_node')

        self._sensor_group = MutuallyExclusiveCallbackGroup()
        self._mcts_group = MutuallyExclusiveCallbackGroup()

        self._lock = threading.Lock()
        self._state = _fake_state()
        self._current_action = None

        self._planner = MCTSPlanner(time_budget=0.15, use_stay=False)

        self._cmd_pub = self.create_publisher(Twist, 'cmd_vel', 10)
        self._tf_broadcaster = TransformBroadcaster(self)

        self._mcts_timer = self.create_timer(
            1.0 / MCTS_HZ,
            self._mcts_callback,
            callback_group=self._mcts_group,
        )
        self._republish_timer = self.create_timer(
            1.0 / REPUBLISH_HZ,
            self._republish_callback,
            callback_group=self._sensor_group,
        )

        self.get_logger().info('mcts_node started')

    def _mcts_callback(self):
        with self._lock:
            state = dict(self._state)

        action = self._planner.plan(state)

        with self._lock:
            self._current_action = action

        self.get_logger().info(f'planned action: {action}')

    def _republish_callback(self):
        with self._lock:
            action = self._current_action

        if action is None:
            return

        twist = Twist()

        if action == 'STAY':
            self._cmd_pub.publish(twist)
            return

        target_yaw = ACTION_TO_HEADING.get(action, 0.0)

        twist.linear.x = LINEAR_SPEED * math.cos(target_yaw)
        twist.linear.y = LINEAR_SPEED * math.sin(target_yaw)
        twist.angular.z = 0.0

        self._cmd_pub.publish(twist)
        self._broadcast_fake_tf()

    def _broadcast_fake_tf(self):
        with self._lock:
            rp = self._state['robot_pos']
            hp = self._state['human_pos']

        now = self.get_clock().now().to_msg()

        robot_tf = TransformStamped()
        robot_tf.header.stamp = now
        robot_tf.header.frame_id = 'world'
        robot_tf.child_frame_id = 'robot'
        robot_tf.transform.translation.x = float(rp[0])
        robot_tf.transform.translation.y = float(rp[1])
        robot_tf.transform.rotation.w = 1.0

        human_tf = TransformStamped()
        human_tf.header.stamp = now
        human_tf.header.frame_id = 'world'
        human_tf.child_frame_id = 'human'
        human_tf.transform.translation.x = float(hp[0])
        human_tf.transform.translation.y = float(hp[1])
        human_tf.transform.rotation.w = 1.0

        self._tf_broadcaster.sendTransform([robot_tf, human_tf])


def main(args=None):
    rclpy.init(args=args)
    node = MCTSNode()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()