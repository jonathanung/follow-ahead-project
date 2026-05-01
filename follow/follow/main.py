import csv
import os
import sys
import math
import time
import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TransformStamped, Twist
from nav_msgs.msg import OccupancyGrid
from visualization_msgs.msg import Marker
from builtin_interfaces.msg import Duration
from scipy.spatial.transform import Rotation as R

def _find_ws_root(start: str) -> str:
    """Walk up from start until we find the colcon workspace (has both src/ and build/)."""
    p = os.path.abspath(start)
    for _ in range(10):
        p = os.path.dirname(p)
        if os.path.isdir(os.path.join(p, 'src')) and os.path.isdir(os.path.join(p, 'build')):
            return p
    raise RuntimeError(f"Could not find workspace root from {start}")

_WS_ROOT = _find_ws_root(__file__)
_FA_ROOT  = os.path.join(_WS_ROOT, 'src', 'follow-ahead-project')

sys.path.insert(0, os.path.join(_FA_ROOT, 'follow', 'follow'))
from planner import MCTSPlanner
import simple_grid

sys.path.insert(0, os.path.join(_FA_ROOT, 'lstm-fc'))
sys.path.insert(0, os.path.join(_FA_ROOT, 'RL_sim'))

from lstm_fc import HumanActionPredictor, TrajectoryBuffer, INPUT_LENGTH
from state import FollowState
from reward import reward as paper_reward

_LOG_COLS = [
    'elapsed_s',
    'robot_x', 'robot_y', 'robot_theta',
    'human_x', 'human_y', 'human_theta',
    'distance', 'alpha_deg', 'alpha_rad',
    'dist_error',
    'r_d', 'r_alpha', 'reward',
    'action', 'tracking_ok',
]


def _to_grid(x: float, y: float, cell_size: float) -> tuple:
    return (int(round(x / cell_size)), int(round(y / cell_size)))


def _action_to_twist(action: str, robot_vel: float, robot_vel_fast: float,
                     turn_speed: float) -> Twist:
    """Convert planner relative action (left/right/straight/fast_*) to Twist.

    Turning actions use reduced forward speed so the robot steers without
    racing past the target position.
    """
    t = Twist()
    if action is None or action == 'STAY':
        return t

    is_fast = 'fast' in action
    speed = robot_vel_fast if is_fast else robot_vel

    if 'left' in action:
        t.angular.z = turn_speed
        t.linear.x  = speed * 0.7   # maintain enough speed to stay ahead while steering
    elif 'right' in action:
        t.angular.z = -turn_speed
        t.linear.x  = speed * 0.7
    else:  # straight / fast_straight
        t.linear.x = speed

    return t


class FollowAheadNode(Node):

    def __init__(self):
        super().__init__("main")

        # --- declare all tunable parameters (overridden by main_params.yaml) ---
        self.declare_parameter('sim',             True)
        self.declare_parameter('cell_size',       0.5)
        self.declare_parameter('decision_hz',     5.0)
        self.declare_parameter('time_budget',     0.15)
        self.declare_parameter('robot_vel',        0.8)
        self.declare_parameter('robot_vel_fast',   1.6)
        self.declare_parameter('turn_speed',       2.0)

        self.sim              = self.get_parameter('sim').get_parameter_value().bool_value
        self.cell_size        = self.get_parameter('cell_size').get_parameter_value().double_value
        self.decision_hz      = self.get_parameter('decision_hz').get_parameter_value().double_value
        self.robot_vel        = self.get_parameter('robot_vel').get_parameter_value().double_value
        self.robot_vel_fast   = self.get_parameter('robot_vel_fast').get_parameter_value().double_value
        self.turn_speed       = self.get_parameter('turn_speed').get_parameter_value().double_value
        time_budget           = self.get_parameter('time_budget').get_parameter_value().double_value

        # Robot and human state
        self.robot_x = 0.0
        self.robot_y = 0.0
        self.robot_z = 0.0
        self.human_x = 0.0
        self.human_y = 0.0
        self.human_z = 0.0

        self.last_plan_time = time.time()
        self.best_action    = None
        self.marker_id      = 0

        # --- data logging ---
        self._log_rows      = []
        self._log_start     = time.time()
        self._log_test_case = os.environ.get('FOLLOW_TEST_CASE', 'unknown')
        self._log_dir       = os.path.expanduser('~/follow_data')
        os.makedirs(self._log_dir, exist_ok=True)

        lstm_model_path = os.path.join(
            _FA_ROOT, 'lstm-fc', 'outputs', 'hypertune_v3', 'best_model.pt'
        )
        self.traj_buffer    = TrajectoryBuffer(length=INPUT_LENGTH)
        self.lstm_predictor = HumanActionPredictor(lstm_model_path)
        self.get_logger().info(f"LSTM model loaded from: {lstm_model_path}")

        self.planner = MCTSPlanner(time_budget=time_budget, verbose=True, use_stay=False)
        self.get_logger().info("MCTSPlanner ready (RL model loaded by planner.py)")

        self.map_params = {
            'map_origin_x': 0.0,
            'map_origin_y': 0.0,
            'map_res':      0.05,
            'map_data':     [],
            'map_width':    100,
        }

        self.create_subscription(OccupancyGrid,    "/global_costmap/costmap", self.costmap_callback, 10)
        self.create_subscription(TransformStamped, "vicon/helmet/root",       self.helmet_callback,  10)
        self.create_subscription(TransformStamped, "vicon/robot/root",        self.robot_callback,   10)

        self.move_robot      = self.create_publisher(Twist,  "/cmd_vel",          10)
        self.pub_robot_traj  = self.create_publisher(Marker, "/robot_traj",       10)
        self.pub_human_traj  = self.create_publisher(Marker, "/human_traj",       10)
        self.pub_robot_arrow = self.create_publisher(Marker, "/robot_traj_arrow", 10)
        self.pub_human_arrow = self.create_publisher(Marker, "/human_traj_arrow", 10)

        self.get_logger().info(
            f"FollowAheadNode ready | sim={self.sim} cell_size={self.cell_size} "
            f"decision_hz={self.decision_hz} robot_vel={self.robot_vel} "
            f"robot_vel_fast={self.robot_vel_fast} turn_speed={self.turn_speed} "
            f"log_dir={self._log_dir} test_case={self._log_test_case}"
        )

    # ------------------------------------------------------------------ #
    #  Pose callbacks                                                      #
    # ------------------------------------------------------------------ #

    def robot_callback(self, robot: TransformStamped):
        orient = robot.transform.rotation
        r = R.from_quat([orient.x, orient.y, orient.z, orient.w])
        robot_z = r.as_euler('zyx', degrees=False)[0]

        robot_p = robot.transform.translation

        if self.sim:
            robot_x, robot_y = robot_p.x, robot_p.y
        else:
            # Real Vicon lab frame is rotated 90° CCW relative to the map frame.
            theta = math.pi / 2
            robot_x = math.cos(theta) * robot_p.x - math.sin(theta) * robot_p.y
            robot_y = math.sin(theta) * robot_p.x + math.cos(theta) * robot_p.y
            robot_z -= math.pi / 2
            if robot_z < -math.pi:
                robot_z += 2 * math.pi
            if robot_z > math.pi:
                robot_z -= 2 * math.pi

        self.robot_x = robot_x
        self.robot_y = robot_y
        self.robot_z = robot_z

    def helmet_callback(self, helmet: TransformStamped):
        orient = helmet.transform.rotation
        r = R.from_quat([orient.x, orient.y, orient.z, orient.w])

        human_z = r.as_euler('zyx', degrees=False)[0]
        human_p = helmet.transform.translation

        if self.sim:
            human_x, human_y = human_p.x, human_p.y
        else:
            # Real Vicon lab frame is rotated 90° CCW relative to the map frame.
            theta = math.pi / 2
            human_x = math.cos(theta) * human_p.x - math.sin(theta) * human_p.y
            human_y = math.sin(theta) * human_p.x + math.cos(theta) * human_p.y
            human_z -= math.pi / 2
            if human_z < -math.pi:
                human_z += 2 * math.pi
            if human_z > math.pi:
                human_z -= 2 * math.pi

        self.human_x = human_x
        self.human_y = human_y
        self.human_z = human_z
        self.move()

        if time.time() - self.last_plan_time < (1.0 / self.decision_hz):
            return
        self.last_plan_time = time.time()

        self.traj_buffer.push(human_x, human_y)

        vis_state = np.array([
            [self.robot_x, self.robot_y, self.robot_z],
            [human_x,      human_y,      human_z],
        ])
        self.pub_marker("robot", self.marker_id, vis_state)
        self.pub_marker("human", self.marker_id, vis_state)
        self.pub_marker("robot", 0, vis_state, arrow=True)
        self.pub_marker("human", 0, vis_state, arrow=True)
        self.marker_id = (self.marker_id + 1) % 2000

        state = FollowState(
            human_x=human_x,     human_y=human_y,     human_theta=human_z,
            robot_x=self.robot_x, robot_y=self.robot_y, robot_theta=self.robot_z,
        )

        # ── SAFETY STOP (comment out the block below to disable) ────────────
        # if state.distance < 0.5:
        #     self.move_robot.publish(Twist())
        #     self.get_logger().warn(f"SAFETY STOP: dist={state.distance:.2f}m < 0.5m")
        #     return
        # ─────────────────────────────────────────────────────────────────────

        human_probs = None
        if self.traj_buffer.ready:
            human_probs = self.lstm_predictor.predict(self.traj_buffer.get())
            self.get_logger().info(
                f"LSTM probs: left={human_probs['left']:.2f} "
                f"straight={human_probs['straight']:.2f} "
                f"right={human_probs['right']:.2f}"
            )
        else:
            self.get_logger().info(
                f"LSTM buffer filling: {self.traj_buffer.count}/{INPUT_LENGTH}"
            )

        action = self.planner.plan(state, human_probs=human_probs)
        self.best_action = action
        self.get_logger().info(
            f"MCTS best action: {action}  "
            f"dist={state.distance:.2f}m  alpha={state.alpha:.1f}°"
        )

        self._record(state, action)

    # ------------------------------------------------------------------ #
    #  Motion                                                              #
    # ------------------------------------------------------------------ #

    def move(self):
        twist = _action_to_twist(
            self.best_action, self.robot_vel, self.robot_vel_fast, self.turn_speed
        )
        self.move_robot.publish(twist)
        self.get_logger().debug(
            f"move() → action={self.best_action}  "
            f"linear.x={twist.linear.x:.2f}  angular.z={twist.angular.z:.2f}"
        )

    # ------------------------------------------------------------------ #
    #  Data logging                                                        #
    # ------------------------------------------------------------------ #

    def _record(self, state: FollowState, action: str):
        from reward import r_d, r_alpha
        rd  = r_d(state.distance)
        ra  = r_alpha(state.alpha)
        self._log_rows.append({
            'elapsed_s':   round(time.time() - self._log_start, 3),
            'robot_x':     round(state.robot_x, 4),
            'robot_y':     round(state.robot_y, 4),
            'robot_theta': round(state.robot_theta, 4),
            'human_x':     round(state.human_x, 4),
            'human_y':     round(state.human_y, 4),
            'human_theta': round(state.human_theta, 4),
            'distance':    round(state.distance, 4),
            'alpha_deg':   round(state.alpha, 3),
            'alpha_rad':   round(math.radians(state.alpha), 4),
            'dist_error':  round(abs(state.distance - 1.5), 4),
            'r_d':         round(rd, 4),
            'r_alpha':     round(ra, 4),
            'reward':      round(rd + ra, 4),
            'action':      action or 'NONE',
            'tracking_ok': state.alpha < 50.0 and 0.5 <= state.distance <= 4.0,
        })

    def _flush_log(self):
        if not self._log_rows:
            return
        ts   = time.strftime('%Y%m%d_%H%M%S')
        path = os.path.join(self._log_dir, f'{self._log_test_case}_{ts}.csv')
        with open(path, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=_LOG_COLS)
            w.writeheader()
            w.writerows(self._log_rows)
        self.get_logger().info(
            f"Data log written → {path}  ({len(self._log_rows)} rows)"
        )

    def destroy_node(self):
        self._flush_log()
        super().destroy_node()

    # ------------------------------------------------------------------ #
    #  Visualisation                                                       #
    # ------------------------------------------------------------------ #

    def pub_marker(self, name, id, state, arrow=False):
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp    = self.get_clock().now().to_msg()
        marker.ns     = name + "arrow" if arrow else name
        marker.id     = int(id)
        marker.type     = 0 if arrow else 2  # 0=ARROW, 2=SPHERE
        marker.action   = 0                 # ADD
        marker.lifetime = Duration(sec=30, nanosec=0)

        pose = state[0] if name == "robot" else state[1]
        marker.pose.position.x = float(pose[0])
        marker.pose.position.y = float(pose[1])

        if not arrow:
            marker.pose.orientation.z = 0.0
            marker.pose.orientation.w = 1.0
        else:
            quat = R.from_euler('xyz', [0, 0, pose[2]], degrees=False).as_quat()
            marker.pose.orientation.z = float(quat[2])
            marker.pose.orientation.w = float(quat[3])

        if arrow:
            marker.scale.x = 0.55
            marker.scale.y = 0.08
            marker.scale.z = 0.08
        else:
            marker.scale.x = 0.18
            marker.scale.y = 0.18
            marker.scale.z = 0.18

        if name == "robot":
            marker.color.r = 0.95
            marker.color.g = 0.3
            marker.color.b = 0.1
        else:
            marker.color.r = 0.1
            marker.color.g = 0.5
            marker.color.b = 1.0
        marker.color.a = 0.85

        if name == "robot" and not arrow:
            self.pub_robot_traj.publish(marker)
        elif name == "robot":
            self.pub_robot_arrow.publish(marker)
        elif name == "human" and not arrow:
            self.pub_human_traj.publish(marker)
        else:
            self.pub_human_arrow.publish(marker)

    # ------------------------------------------------------------------ #
    #  Costmap → planner obstacle set                                      #
    # ------------------------------------------------------------------ #

    def costmap_callback(self, data: OccupancyGrid):
        self.map_params['map_origin_x'] = data.info.origin.position.x
        self.map_params['map_origin_y'] = data.info.origin.position.y
        self.map_params['map_res']       = data.info.resolution
        self.map_params['map_data']      = data.data
        self.map_params['map_width']     = data.info.width

        ox, oy = data.info.origin.position.x, data.info.origin.position.y
        res, w = data.info.resolution, data.info.width

        obstacles = set()
        for i, val in enumerate(data.data):
            # >= 65: within nav2 inflation zone (hard walls are 253/254).
            # Using 65 gives ~1-cell buffer around walls which matches the
            # 0.55 m inflation radius set in nav2_params.yaml.
            if val < 65:
                continue
            wx = ox + (i % w + 0.5) * res
            wy = oy + (i // w + 0.5) * res
            gx, gy = _to_grid(wx, wy, self.cell_size)
            obstacles.add((gx, gy))

        simple_grid.OBSTACLES = obstacles
        self.get_logger().info(
            f"costmap: {len(obstacles)} obstacle cells loaded into planner grid"
        )


def main():
    rclpy.init()
    node = FollowAheadNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
