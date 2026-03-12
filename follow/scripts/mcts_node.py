import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from geometry_msgs.msg import TwistStamped, TransformStamped
from nav_msgs.msg import Odometry
from visualization_msgs.msg import Marker
from tf2_ros import TransformBroadcaster
import threading
import math
from collections import deque
from planner import MCTSPlanner

LINEAR_SPEED = 0.5       # increased from 0.3
ANGULAR_SPEED = 2.0      # increased from 1.5
MCTS_HZ = 5.0
CMD_HZ = 10.0
CELL_SIZE = 0.5
GRID_SIZE = 7
HEADING_THRESHOLD = 0.3  # slightly wider deadband
ACTION_BUFFER_SIZE = 3

ACTION_TO_YAW = {
    'N':  math.pi / 2,
    'S': -math.pi / 2,
    'E':  0.0,
    'W':  math.pi,
}

HEADING_DIRS = ['N', 'E', 'S', 'W']


def _meters_to_cell(x, y):
    cx = int(round(x / CELL_SIZE))
    cy = int(round(y / CELL_SIZE))
    cx = max(0, min(GRID_SIZE - 1, cx))
    cy = max(0, min(GRID_SIZE - 1, cy))
    return (cx, cy)


def _yaw_from_quaternion(qz, qw):
    return 2.0 * math.atan2(qz, qw)


def _yaw_from_transform(tf):
    return _yaw_from_quaternion(
        tf.transform.rotation.z,
        tf.transform.rotation.w,
    )


def _yaw_to_heading(yaw):
    idx = int(round(yaw / (math.pi / 2))) % 4
    return HEADING_DIRS[idx]


def _angle_diff(a, b):
    return (a - b + math.pi) % (2 * math.pi) - math.pi


class MCTSNode(Node):
    def __init__(self):
        super().__init__('mcts_node')

        self._sensor_group = MutuallyExclusiveCallbackGroup()
        self._mcts_group = MutuallyExclusiveCallbackGroup()

        self._lock = threading.Lock()
        self._robot_pos = (0, 0)
        self._robot_yaw = 0.0
        self._human_pos = (1, 0)
        self._human_heading = 'E'
        self._current_action = None
        self._action_buffer = deque(maxlen=ACTION_BUFFER_SIZE)

        self._planner = MCTSPlanner(time_budget=0.15, use_stay=False)

        self._cmd_pub = self.create_publisher(TwistStamped, 'cmd_vel', 10)
        self._marker_pub = self.create_publisher(Marker, 'human_marker', 10)
        self._tf_broadcaster = TransformBroadcaster(self)

        self.create_subscription(
            Odometry, 'odom', self._odom_cb, 10,
            callback_group=self._sensor_group,
        )
        self.create_subscription(
            TransformStamped, 'vicon/helmet/root', self._human_cb, 10,
            callback_group=self._sensor_group,
        )

        self.create_timer(
            1.0 / MCTS_HZ, self._mcts_callback,
            callback_group=self._mcts_group,
        )
        self.create_timer(
            1.0 / CMD_HZ, self._cmd_callback,
            callback_group=self._sensor_group,
        )

        self.get_logger().info('mcts_node started')

    def _odom_cb(self, msg):
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        qz = msg.pose.pose.orientation.z
        qw = msg.pose.pose.orientation.w
        with self._lock:
            self._robot_pos = _meters_to_cell(x, y)
            self._robot_yaw = _yaw_from_quaternion(qz, qw)

    def _human_cb(self, msg):
        x = msg.transform.translation.x
        y = msg.transform.translation.y
        yaw = _yaw_from_transform(msg)
        with self._lock:
            self._human_pos = _meters_to_cell(x, y)
            self._human_heading = _yaw_to_heading(yaw)

    def _mcts_callback(self):
        with self._lock:
            state = {
                'robot_pos': self._robot_pos,
                'human_pos': self._human_pos,
                'human_heading': self._human_heading,
            }

        action = self._planner.plan(state)

        with self._lock:
            self._action_buffer.append(action)
            if self._action_buffer.count(action) >= ACTION_BUFFER_SIZE - 1:
                self._current_action = action

        self.get_logger().info(
            f'planned: {action}  committed: {self._current_action}'
            f'  robot: {self._robot_pos}  human: {self._human_pos}'
            f'  heading: {state["human_heading"]}'
        )

    def _cmd_callback(self):
        with self._lock:
            action = self._current_action
            robot_yaw = self._robot_yaw
            rp = self._robot_pos
            hp = self._human_pos

        if action is None:
            return

        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'

        if action != 'STAY':
            target_yaw = ACTION_TO_YAW[action]
            yaw_error = _angle_diff(target_yaw, robot_yaw)
            abs_err = abs(yaw_error)

            if abs_err > HEADING_THRESHOLD:
                # Turn and creep forward simultaneously
                msg.twist.angular.z = ANGULAR_SPEED * (1.0 if yaw_error > 0 else -1.0)
                msg.twist.linear.x = LINEAR_SPEED * max(0.0, 1.0 - abs_err / math.pi)
            else:
                msg.twist.linear.x = LINEAR_SPEED

        self._cmd_pub.publish(msg)
        self._broadcast_tf(rp, hp)
        self._publish_human_marker(hp)

    def _publish_human_marker(self, hp):
        marker = Marker()
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.header.frame_id = 'odom'
        marker.ns = 'human'
        marker.id = 0
        marker.type = Marker.CYLINDER
        marker.action = Marker.ADD
        marker.pose.position.x = float(hp[0]) * CELL_SIZE
        marker.pose.position.y = float(hp[1]) * CELL_SIZE
        marker.pose.position.z = 0.5
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.3
        marker.scale.y = 0.3
        marker.scale.z = 1.0
        marker.color.r = 0.0
        marker.color.g = 0.8
        marker.color.b = 0.2
        marker.color.a = 1.0
        self._marker_pub.publish(marker)

    def _broadcast_tf(self, rp, hp):
        now = self.get_clock().now().to_msg()

        robot_tf = TransformStamped()
        robot_tf.header.stamp = now
        robot_tf.header.frame_id = 'odom'
        robot_tf.child_frame_id = 'robot_estimated'
        robot_tf.transform.translation.x = float(rp[0]) * CELL_SIZE
        robot_tf.transform.translation.y = float(rp[1]) * CELL_SIZE
        robot_tf.transform.rotation.w = 1.0

        human_tf = TransformStamped()
        human_tf.header.stamp = now
        human_tf.header.frame_id = 'odom'
        human_tf.child_frame_id = 'human_estimated'
        human_tf.transform.translation.x = float(hp[0]) * CELL_SIZE
        human_tf.transform.translation.y = float(hp[1]) * CELL_SIZE
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