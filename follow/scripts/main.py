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
from scipy.spatial.transform import Rotation as R
from lstm_fc import HumanActionPredictor, TrajectoryBuffer, INPUT_LENGTH
from planner import MCTSPlanner
import simple_grid

# Path setup — add new algorithm folders before any imports from them
_PROJECT_ROOT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    '..', '..', '..', '..', 'follow-ahead-project'   # adjust if needed
)
_PROJECT_ROOT = os.path.abspath(_PROJECT_ROOT)

# follow/scripts — contains planner.py, node.py, simple_grid.py, etc.
sys.path.insert(0, os.path.join(_PROJECT_ROOT, 'follow', 'scripts'))

# lstm-fc — contains lstm_fc package (HumanActionPredictor, TrajectoryBuffer)
sys.path.insert(0, os.path.join(_PROJECT_ROOT, 'lstm-fc'))

# RL_sim — planner.py adds this itself, but add here too for safety
sys.path.insert(0, os.path.join(_PROJECT_ROOT, 'RL_sim'))

# Absolute yaw for each grid direction (radians) — not a tunable param
_ACTION_TO_YAW = {
    'N':  math.pi / 2,
    'E':  0.0,
    'S': -math.pi / 2,
    'W':  math.pi,
}


def _to_grid(x: float, y: float, cell_size: float) -> tuple:
    return (int(round(x / cell_size)), int(round(y / cell_size)))


def _yaw_to_heading(yaw: float) -> str:
    return min(_ACTION_TO_YAW, key=lambda h: abs(
        math.atan2(math.sin(yaw - _ACTION_TO_YAW[h]),
                   math.cos(yaw - _ACTION_TO_YAW[h]))
    ))


def _action_to_twist(action: str, robot_yaw: float,
                     robot_vel: float, turn_speed: float,
                     align_threshold: float) -> Twist:
    t = Twist()

    if action is None or action == 'STAY':
        return t

    target_yaw = _ACTION_TO_YAW.get(action)
    if target_yaw is None:
        return t

    err = math.atan2(math.sin(target_yaw - robot_yaw),
                     math.cos(target_yaw - robot_yaw))

    if abs(err) > align_threshold:
        t.angular.z = turn_speed * math.copysign(1.0, err)
    else:
        t.linear.x = robot_vel
        t.angular.z = turn_speed * err

    return t


class FollowAheadNode(Node):

    def __init__(self):
        super().__init__("main")

        # --- declare all tunable parameters (overridden by main_params.yaml) ---
        self.declare_parameter('sim',             True)
        self.declare_parameter('cell_size',       0.5)
        self.declare_parameter('decision_hz',     5.0)
        self.declare_parameter('time_budget',     0.15)
        self.declare_parameter('robot_vel',       0.6)
        self.declare_parameter('robot_vel_fast',  0.9)
        self.declare_parameter('turn_speed',      1.0)
        self.declare_parameter('align_threshold', 0.3)

        self.sim             = self.get_parameter('sim').get_parameter_value().bool_value
        self.cell_size       = self.get_parameter('cell_size').get_parameter_value().double_value
        self.decision_hz     = self.get_parameter('decision_hz').get_parameter_value().double_value
        self.robot_vel       = self.get_parameter('robot_vel').get_parameter_value().double_value
        self.turn_speed      = self.get_parameter('turn_speed').get_parameter_value().double_value
        self.align_threshold = self.get_parameter('align_threshold').get_parameter_value().double_value
        time_budget          = self.get_parameter('time_budget').get_parameter_value().double_value

        # Robot state
        self.robot_x = 0.0
        self.robot_y = 0.0
        self.robot_z = 0.0

        self.last_plan_time = time.time()
        self.best_action    = None
        self.marker_id      = 0

        lstm_model_path = os.path.join(
            _PROJECT_ROOT, 'lstm-fc', 'outputs', 'final_v3', 'model_final.pt'
        )
        self.traj_buffer    = TrajectoryBuffer(length=INPUT_LENGTH)
        self.lstm_predictor = HumanActionPredictor(lstm_model_path)
        self.get_logger().info(f"LSTM model loaded from: {lstm_model_path}")

        self.planner = MCTSPlanner(time_budget=time_budget, verbose=True)
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
            f"turn_speed={self.turn_speed} align_threshold={self.align_threshold}"
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

        self.move()

        if time.time() - self.last_plan_time < (1.0 / self.decision_hz):
            return
        self.last_plan_time = time.time()

        self.traj_buffer.push(human_x, human_y)

        robot_grid  = _to_grid(self.robot_x, self.robot_y, self.cell_size)
        human_grid  = _to_grid(human_x, human_y, self.cell_size)
        human_heading = _yaw_to_heading(human_z)

        state = {
            'robot_pos':     robot_grid,
            'human_pos':     human_grid,
            'human_heading': human_heading,
        }

        vis_state = np.array([
            [self.robot_x, self.robot_y, self.robot_z],
            [human_x,      human_y,      human_z],
        ])
        self.pub_marker("robot", self.marker_id, vis_state)
        self.pub_marker("human", self.marker_id, vis_state)
        self.pub_marker("robot", 0, vis_state, arrow=True)
        self.pub_marker("human", 0, vis_state, arrow=True)
        self.marker_id += 1

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
        self.get_logger().info(f"MCTS best action: {action}  state: {state}")

    # ------------------------------------------------------------------ #
    #  Motion                                                              #
    # ------------------------------------------------------------------ #

    def move(self):
        twist = _action_to_twist(
            self.best_action, self.robot_z,
            self.robot_vel, self.turn_speed, self.align_threshold
        )
        self.move_robot.publish(twist)
        self.get_logger().info(
            f"move() → action={self.best_action}  "
            f"linear.x={twist.linear.x:.2f}  angular.z={twist.angular.z:.2f}"
        )

    # ------------------------------------------------------------------ #
    #  Visualisation                                                       #
    # ------------------------------------------------------------------ #

    def pub_marker(self, name, id, state, arrow=False):
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp    = self.get_clock().now().to_msg()
        marker.ns     = name + "arrow" if arrow else name
        marker.id     = int(id)
        marker.type   = 0 if arrow else 2   # 0=ARROW, 2=SPHERE
        marker.action = 0                   # ADD

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
            if 0 <= gx < simple_grid.GRID_SIZE and 0 <= gy < simple_grid.GRID_SIZE:
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
