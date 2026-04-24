import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import Odometry
import math
import time

try:
    from gazebo_msgs.msg import ModelStates as _ModelStates
except ImportError:
    _ModelStates = None

class FakeVicon(Node):
    def __init__(self):
        super().__init__("fake_vicon")
        # Default: scripted walking circle on vicon/helmet/root (see publish_human).
        # Set True only if you want poses from a Gazebo model literally named "human".
        self.declare_parameter("use_gazebo_human_in_world", False)

        # Add these new rectangle parameters
        self.declare_parameter("rect_width", 2.0)
        self.declare_parameter("rect_height", 1.0)
        self.declare_parameter("rect_speed", 0.35)
        self.declare_parameter("rect_start_x", 0.0)
        self.declare_parameter("rect_start_y", 0.0)


        # Scripted motion on vicon/helmet/root (ignored if use_gazebo_human_in_world).
        self.declare_parameter("human_motion_mode", "straight")
        self.declare_parameter("circle_radius", 0.8)
        self.declare_parameter("circle_angular_speed", 0.3)
        self.declare_parameter("straight_line_speed", 0.35)
        self.declare_parameter("straight_line_duration_sec", 12.0)
        self.declare_parameter("straight_line_heading_rad", 0.0)
        self.declare_parameter("straight_line_ping_pong", True)

        self.declare_parameter("stationary_x", 2.0)
        self.declare_parameter("stationary_y", 0.0)

        self.declare_parameter("zigzag_segment_length", 1.5)
        self.declare_parameter("zigzag_speed", 0.35)
        self.declare_parameter("zigzag_angle_deg", 40.0)

        self.get_logger().info(
            'fake_vicon: vicon/helmet/root scripted motion (see human_motion_mode). '
            'Gazebo model "human" pose: use_gazebo_human_in_world:=true'
        )

        # subscribe to robot odom (estimate) from Gazebo
        # robot's odometry from Gazebo
        self.create_subscription(Odometry, "/odom", self.odom_callback, 10)
        if _ModelStates is not None:
            self.create_subscription(_ModelStates, '/gazebo/model_states', self.model_states_callback, 10)

        # publish fake vicon topics - robot is robot and helmet is human
        # repackages it as a TransformStamped message
        self.pub_robot = self.create_publisher(TransformStamped, "vicon/robot/root", 10)
        self.pub_human = self.create_publisher(TransformStamped, "vicon/helmet/root", 10)

        # publish fake human at 10hz
        self.create_timer(0.1, self.publish_human)
        # publish robot pose even when no odom arrives (pure sim, no Gazebo)
        self.create_timer(0.1, self._publish_robot_static)

        self._odom_received = False
        self.robot_transform = TransformStamped()
        self.start_time = time.time()

    def odom_callback(self, msg: Odometry):
        # repackage robot odom as TransformStamped
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = "map"
        t.child_frame_id = "base_link"

        t.transform.translation.x = msg.pose.pose.position.x
        t.transform.translation.y = msg.pose.pose.position.y
        t.transform.translation.z = msg.pose.pose.position.z

        t.transform.rotation = msg.pose.pose.orientation

        self._odom_received = True
        self.robot_transform = t
        self.pub_robot.publish(t)

    def _publish_robot_static(self):
        """Publish a static robot pose at the origin when no odom source is available."""
        if self._odom_received:
            return
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = "map"
        t.child_frame_id = "base_link"
        t.transform.rotation.w = 1.0
        self.pub_robot.publish(t)

    @staticmethod
    def _quat_z(yaw: float):
        return (0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0))

    def publish_human(self):
        if self.get_parameter("use_gazebo_human_in_world").get_parameter_value().bool_value:
            return

        elapsed = time.time() - self.start_time
        mode = (
            self.get_parameter("human_motion_mode")
            .get_parameter_value()
            .string_value.strip()
            .lower()
        )

        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = "map"
        t.child_frame_id = "helmet"
        t.transform.translation.z = 0.0

        if mode == "stationary":
            t.transform.translation.x = self.get_parameter("stationary_x").get_parameter_value().double_value
            t.transform.translation.y = self.get_parameter("stationary_y").get_parameter_value().double_value
            yaw = 0.0

        elif mode == "zigzag":
            seg_len = self.get_parameter("zigzag_segment_length").get_parameter_value().double_value
            spd     = self.get_parameter("zigzag_speed").get_parameter_value().double_value
            angle   = math.radians(self.get_parameter("zigzag_angle_deg").get_parameter_value().double_value)

            seg_dur   = seg_len / spd if spd > 0.0 else 1.0
            period    = 2.0 * seg_dur
            n_periods = int(elapsed / period)
            ph        = elapsed % period
            base_x    = n_periods * 2.0 * seg_len * math.cos(angle)

            if ph < seg_dur:
                alpha = ph / seg_dur
                t.transform.translation.x = base_x + alpha * seg_len * math.cos(angle)
                t.transform.translation.y = alpha * seg_len * math.sin(angle)
                yaw = angle
            else:
                alpha = (ph - seg_dur) / seg_dur
                t.transform.translation.x = base_x + seg_len * math.cos(angle) + alpha * seg_len * math.cos(angle)
                t.transform.translation.y = seg_len * math.sin(angle) - alpha * seg_len * math.sin(angle)
                yaw = -angle

        elif mode == "circle":
            r = self.get_parameter("circle_radius").get_parameter_value().double_value
            w = self.get_parameter(
                "circle_angular_speed"
            ).get_parameter_value().double_value
            t.transform.translation.x = r * math.cos(w * elapsed)
            t.transform.translation.y = r * math.sin(w * elapsed)
            yaw = w * elapsed + math.pi / 2

        elif mode == "rectangle":
            W = self.get_parameter("rect_width").get_parameter_value().double_value
            H = self.get_parameter("rect_height").get_parameter_value().double_value
            spd = self.get_parameter("rect_speed").get_parameter_value().double_value
            sx = self.get_parameter("rect_start_x").get_parameter_value().double_value
            sy = self.get_parameter("rect_start_y").get_parameter_value().double_value

            # Duration of each leg
            t1 = W / spd          # leg 1: bottom  (sx,sy)      → (sx+W, sy)
            t2 = t1 + H / spd     # leg 2: right   (sx+W, sy)   → (sx+W, sy+H)
            t3 = t2 + W / spd     # leg 3: top     (sx+W, sy+H) → (sx,   sy+H)
            t4 = t3 + H / spd     # leg 4: left    (sx,   sy+H) → (sx,   sy)
            period = t4

            ph = elapsed % period  # where in the loop are we?

            if ph < t1:
                alpha = ph / t1
                x = sx + alpha * W
                y = sy
                yaw = 0.0
            elif ph < t2:
                alpha = (ph - t1) / (t2 - t1)
                x = sx + W
                y = sy + alpha * H
                yaw = math.pi / 2
            elif ph < t3:
                alpha = (ph - t2) / (t3 - t2)
                x = sx + W - alpha * W
                y = sy + H
                yaw = math.pi
            else:
                alpha = (ph - t3) / (t4 - t3)
                x = sx
                y = sy + H - alpha * H
                yaw = 3 * math.pi / 2

            t.transform.translation.x = x
            t.transform.translation.y = y


        else:
            # straight (default) — the human walks in a straight line.
            lin = self.get_parameter("straight_line_speed").get_parameter_value().double_value

            # straight_line_duration_sec — how long the walk lasts
            dur = self.get_parameter(
                "straight_line_duration_sec"
            ).get_parameter_value().double_value

            # straight_line_heading_rad — which direction to walk
            heading = self.get_parameter(
                "straight_line_heading_rad"
            ).get_parameter_value().double_value

            # straight_line_ping_pong — if True, the human walks forward then reverses back, looping indefinitely
            ping_pong = self.get_parameter(
                "straight_line_ping_pong"
            ).get_parameter_value().bool_value

            length = lin * dur if dur > 0.0 else 0.0
            if dur <= 0.0:
                dist_along = 0.0
                yaw = heading
            elif ping_pong:
                period = 2.0 * dur
                ph = elapsed % period
                if ph < dur:
                    alpha = ph / dur
                    dist_along = alpha * length
                    yaw = heading
                else:
                    alpha = (ph - dur) / dur
                    dist_along = (1.0 - alpha) * length
                    yaw = heading + math.pi
            else:
                dist_along = min(elapsed, dur) * lin
                yaw = heading if elapsed < dur else heading

            t.transform.translation.x = dist_along * math.cos(heading)
            t.transform.translation.y = dist_along * math.sin(heading)

        qx, qy, qz, qw = self._quat_z(yaw)
        t.transform.rotation.x = qx
        t.transform.rotation.y = qy
        t.transform.rotation.z = qz
        t.transform.rotation.w = qw

        self.pub_human.publish(t)

    def model_states_callback(self, msg):
        if not self.get_parameter("use_gazebo_human_in_world").get_parameter_value().bool_value:
            return
        if "human" not in msg.name:
            return
        
        # Find the human model
        idx = msg.name.index("human")
        pose = msg.poses[idx]

        # Repackage as a Vicon-style transform
        # It takes the raw Gazebo pose
        # wraps it into a TransformStamped message
        # publishes it on vicon/helmet/root
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = "map"
        t.child_frame_id = "helmet"

        t.transform.translation.x = pose.position.x
        t.transform.translation.y = pose.position.y
        t.transform.translation.z = pose.position.z
        t.transform.rotation = pose.orientation
        self.pub_human.publish(t)

def main():
    rclpy.init()
    node = FakeVicon()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()