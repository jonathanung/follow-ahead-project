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

# LSTM: replaced prob_dist.forward() with HumanActionPredictor + TrajectoryBuffer
# MCTS: replaced MCTSNode + MCTS + navState with MCTSPlanner.plan(state_dict, human_probs)
# RL model: no longer loaded here — planner.py loads it as a singleton at import time
# State format: np.array → grid dict {'robot_pos':(cx,cy), 'human_pos':(hx,hy),
                                        # 'human_heading':'N'|'E'|'S'|'W'}
# Actions: 'N'/'S'/'E'/'W'/'STAY' → Twist (via _action_to_twist())


# Path setup — add new algorithm folders before any imports from them
# _PROJECT_ROOT = os.path.join(
#     os.path.dirname(os.path.abspath(__file__)),
#     '..', '..', '..', '..', 'follow-ahead-project'   # adjust if needed
# )
_PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..', '..')
)

sys.path.insert(0, os.path.join(_PROJECT_ROOT, 'src', 'follow-ahead-project', 'follow', 'follow'))
from planner import MCTSPlanner


sys.path.insert(0, os.path.join(_PROJECT_ROOT, 'src', 'follow-ahead-project', 'lstm-fc'))
  
# RL_sim — planner.py adds this itself, but add here too for safety
sys.path.insert(0, os.path.join(_PROJECT_ROOT, 'src', 'follow-ahead-project', 'RL_sim'))

from lstm_fc import HumanActionPredictor, TrajectoryBuffer, INPUT_LENGTH


CELL_SIZE = 0.5          # metres per grid cell (must match planner.py)
DECISION_HZ = 5          # Hz — how often MCTS runs
TIME_BUDGET = 0.15       # seconds per MCTS call (150 ms, matches paper)
ROBOT_VEL = 0.6          # m/s base linear speed
ROBOT_VEL_FAST = 0.9     # m/s fast speed (1.5x)
TURN_SPEED = 1.0         # rad/s angular speed when turning to face direction

# Absolute yaw for each grid direction (radians)
_ACTION_TO_YAW = {
    'N':  math.pi / 2,
    'E':  0.0,
    'S': -math.pi / 2,
    'W':  math.pi,
}

# continuous metres to grid cell index
def _to_grid(x: float, y: float) -> tuple:
    """Convert continuous world coordinates (metres) to integer grid cell."""
    return (int(round(x / CELL_SIZE)), int(round(y / CELL_SIZE)))

# continuous yaw (radians) to nearest cardinal heading string
def _yaw_to_heading(yaw: float) -> str:
    """Quantise a continuous yaw angle to the nearest N/E/S/W heading."""
    best = min(_ACTION_TO_YAW, key=lambda h: abs(
        math.atan2(math.sin(yaw - _ACTION_TO_YAW[h]),
                   math.cos(yaw - _ACTION_TO_YAW[h]))
    ))
    return best

# grid action string to Twist
def _action_to_twist(action: str, robot_yaw: float) -> Twist:
    """
    Convert a grid action ('N'/'S'/'E'/'W'/'STAY') to a Twist command.
 
    Strategy:
      - Compute target absolute yaw from the action.
      - If the robot is already roughly aligned (|error| < 0.3 rad ≈ 17°),
        drive forward with a small corrective angular term.
      - Otherwise, rotate in place first.
 
    Parameters
    ----------
    action    : str   — one of 'N', 'S', 'E', 'W', 'STAY'
    robot_yaw : float — current robot heading in radians
 
    Returns
    -------
    geometry_msgs.msg.Twist
    """

    t = Twist()
    if action is None or action == 'STAY':
        return t

    vel = ROBOT_VEL_FAST if 'fast' in action else ROBOT_VEL

    if 'left' in action:
        t.angular.z = TURN_SPEED
        t.linear.x = vel
    elif 'right' in action:
        t.angular.z = -TURN_SPEED
        t.linear.x = vel
    else:  # straight / fast_straight
        t.linear.x = vel

    return t



class FollowAheadNode(Node):
 
    def __init__(self):
        super().__init__("main")

        # Simulation flag 
        self.sim = True   # True → use coords directly; False → rotate 90°
    
        # Robot state 
        self.robot_x = 0.0
        self.robot_y = 0.0
        self.robot_z = 0.0   # yaw in radians

        # Control loop timing 
        self.last_plan_time = time.time()
        self.best_action = None       # last action from MCTSPlanner.plan()
        self.marker_id = 0

        # LSTM: TrajectoryBuffer + HumanActionPredictor 
        self.traj_buffer = TrajectoryBuffer(length=INPUT_LENGTH)
        lstm_model_path = os.path.join(
            _PROJECT_ROOT, 'src', 'follow-ahead-project', 'lstm-fc', 'outputs', 'hypertune_v3', 'best_model.pt'
        )
        self.lstm_predictor = HumanActionPredictor(lstm_model_path)
        self.get_logger().info(f"LSTM model loaded from: {lstm_model_path}")

        # MCTS Planner (RL model is loaded inside planner.py at import) ─
        # time_budget=0.15 -> 150 ms per call, matching the 5 Hz decision rate
        self.planner = MCTSPlanner(time_budget=TIME_BUDGET, verbose=True)
        self.get_logger().info("MCTSPlanner ready (RL model loaded by planner.py)")

        self.map_params = {
            'map_origin_x': 0.0,
            'map_origin_y': 0.0,
            'map_res': 0.05,
            'map_data': [],
            'map_width': 100,
        }

        self.create_subscription(OccupancyGrid, "/global_costmap/costmap",
                                 self.costmap_callback, 10)
        self.create_subscription(TransformStamped, "vicon/helmet/root",
                                 self.helmet_callback, 10)
        self.create_subscription(TransformStamped, "vicon/robot/root",
                                 self.robot_callback, 10)
        
        self.move_robot     = self.create_publisher(Twist, "/cmd_vel", 10)
        self.pub_robot_traj = self.create_publisher(Marker, "/robot_traj", 10)
        self.pub_human_traj = self.create_publisher(Marker, "/human_traj", 10)
        self.pub_robot_arrow = self.create_publisher(Marker, "/robot_traj_arrow", 10)
        self.pub_human_arrow = self.create_publisher(Marker, "/human_traj_arrow", 10)

        self.get_logger().info("FollowAheadNode initialised (new algorithms wired).")

    def robot_callback(self, robot: TransformStamped):
        """Update stored robot pose from vicon/robot/root."""

        orient = robot.transform.rotation
        r = R.from_quat([orient.w, orient.x, orient.y, orient.z])
        robot_z = r.as_euler('zyx', degrees=False)[2]
        robot_z -= math.pi / 2

        if robot_z < -math.pi:
            robot_z += 2 * math.pi
        if robot_z > math.pi:
            robot_z -= 2 * math.pi

        robot_p = robot.transform.translation

        if self.sim:
            robot_x, robot_y = robot_p.x, robot_p.y

        else:
            theta = math.pi / 2
            robot_x = math.cos(theta) * robot_p.x - math.sin(theta) * robot_p.y
            robot_y = math.sin(theta) * robot_p.x + math.cos(theta) * robot_p.y

        self.robot_x = robot_x
        self.robot_y = robot_y
        self.robot_z = robot_z

    def helmet_callback(self, helmet: TransformStamped):
        """
        Main control callback — fires on every vicon/helmet/root message.
 
        At every call:
          - Execute the last planned action (move()).
          - At 5 Hz: push human position to LSTM buffer, run MCTS if buffer ready.
        """

        orient = helmet.transform.rotation
        r = R.from_quat([orient.w, orient.x, orient.y, orient.z])

        human_z = r.as_euler('zyx', degrees=False)[2]
        human_z -= math.pi / 2
        if human_z < -math.pi:
            human_z += 2 * math.pi
        if human_z > math.pi:
            human_z -= 2 * math.pi
        human_p = helmet.transform.translation
        if self.sim:
            human_x, human_y = human_p.x, human_p.y
        else:
            theta = math.pi / 2
            human_x = math.cos(theta) * human_p.x - math.sin(theta) * human_p.y
            human_y = math.sin(theta) * human_p.x + math.cos(theta) * human_p.y

        self.move()

        if time.time() - self.last_plan_time < (1.0 / DECISION_HZ):
            return
        self.last_plan_time = time.time()
 
        # Push human position into the LSTM trajectory buffer
        self.traj_buffer.push(human_x, human_y)

        # Build grid state dict (required by MCTSPlanner.plan)
        robot_grid = _to_grid(self.robot_x, self.robot_y)
        human_grid = _to_grid(human_x, human_y)
        human_heading = _yaw_to_heading(human_z)

        from state import FollowState
        state = FollowState(
            human_x=human_x,
            human_y=human_y,
            human_theta=human_z,
            robot_x=self.robot_x,
            robot_y=self.robot_y,
            robot_theta=self.robot_z,
)

        # Visualise current poses
        vis_state = np.array([
            [self.robot_x, self.robot_y, self.robot_z],
            [human_x,      human_y,      human_z],
        ])

        self.pub_marker("robot", self.marker_id, vis_state)
        self.pub_marker("human", self.marker_id, vis_state)
        self.pub_marker("robot", 0, vis_state, arrow=True)
        self.pub_marker("human", 0, vis_state, arrow=True)

        self.marker_id += 1

        # LSTM prediction
        human_probs = None
        if self.traj_buffer.ready:
            # Returns {"left": float, "straight": float, "right": float}
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

        # MCTS planning
        # self.planner.plan(state, human_probs) → 'N'/'S'/'E'/'W'/'STAY'
        action = self.planner.plan(state, human_probs=human_probs)
        self.best_action = action
        self.get_logger().info(f"MCTS best action: {action}  state: {state}")


    def move(self):
        """
        Publish a Twist command for the current best_action.
 
        Old actions: "fast_left", "straight", "stop", …
        New actions: 'N', 'S', 'E', 'W', 'STAY'
 
        _action_to_twist() computes the error between the robot's current yaw
        and the target yaw implied by the grid direction, then drives / rotates
        accordingly.
        """

        twist = _action_to_twist(self.best_action, self.robot_z)
        self.move_robot.publish(twist)
        self.get_logger().info(
            f"move() → action={self.best_action}  "
            f"linear.x={twist.linear.x:.2f}  angular.z={twist.angular.z:.2f}"
        )
 

    def pub_marker(self, name, id, state, arrow=False):
        """Publish an RViz Marker for robot or human trajectory."""
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = name + "arrow" if arrow else name
        marker.id = int(id)
        marker.type = 0 if arrow else 1   # 0=arrow, 1=cube
        marker.action = 0                 # ADD
 
        pose = state[0] if name == "robot" else state[1]
        marker.pose.position.x = float(pose[0])
        marker.pose.position.y = float(pose[1])

        if not arrow:
            marker.pose.orientation.z = 0.0
            marker.pose.orientation.w = 1.0
        else:
            from scipy.spatial.transform import Rotation as R2
            quat = R2.from_euler('xyz', [0, 0, pose[2]], degrees=False).as_quat()
            marker.pose.orientation.z = float(quat[2])
            marker.pose.orientation.w = float(quat[3])

        marker.scale.x = 0.5 if arrow else 0.1
        marker.scale.y = 0.1
        marker.scale.z = 0.1
 
        marker.color.r = 1.0 if name == "robot" else 0.0
        marker.color.g = 0.0
        marker.color.b = 0.0 if name == "robot" else 1.0
        marker.color.a = 1.0

        if name == "robot" and not arrow:
            self.pub_robot_traj.publish(marker)
        elif name == "robot":
            self.pub_robot_arrow.publish(marker)
        elif name == "human" and not arrow:
            self.pub_human_traj.publish(marker)
        else:
            self.pub_human_arrow.publish(marker)

    def costmap_callback(self, data: OccupancyGrid):
        self.get_logger().info("costmap received")
        self.map_params['map_origin_x'] = data.info.origin.position.x
        self.map_params['map_origin_y'] = data.info.origin.position.y
        self.map_params['map_res']       = data.info.resolution
        self.map_params['map_data']      = data.data
        self.map_params['map_width']     = data.info.width

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
