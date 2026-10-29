#!/usr/bin/env python3
"""Run retraction experiments from snapshots of the live tissue and context point clouds."""

import math
import threading
import time
import sys
import os

import crtk
import dvrk
import PyKDL
import rclpy
from rclpy.duration import Duration
from rclpy.executors import ExternalShutdownException, SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, qos_profile_sensor_data
from rclpy.time import Time
from retraction_interface.srv import GenerateGoalPointCloud
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
from tf2_ros import Buffer, TransformListener



PUSHVIBES_ROOT = os.path.expanduser('~/retraction_ws/src/PushVIBES')
if PUSHVIBES_ROOT not in sys.path:
    sys.path.insert(0, PUSHVIBES_ROOT)
# ROOT = os.path.expanduser('~/retraction_ws/src')
# if ROOT not in sys.path:
#     sys.path.insert(0, ROOT)

from deploy_push import DeployPushVIBES, DeployPushDeterministic, DeployPushHeuristic


# Latched, so a late subscriber gets the last snapshot right away instead of waiting for the next timer tick
SNAPSHOT_QOS = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)

DVRK_PSM_BASE_FRAME = 'PSM1_base_link' # use for sim only, since the real robot takes commands in the camera frame after hand-eye calibration.


class RetractionExperimentsNode(Node):

    def __init__(self,
                 dvrk_arm_name='PSM1',
                 robot_mode='sim',
                 push_method='PushVIBES_start_point_optimized', # ~ {PushVIBES_start_point_optimized, PushVIBES_start_point_conditioned, PushVIBES_deterministic, heuristic}
                 current_pc_topic='/tissue_pointcloud',
                 context_pc_topic='/context_pointcloud',
                 goal_service_name='/generate_goal_pointcloud',
                 service_timeout_sec=60.0,
                 snapshot_publish_rate=0.5,
                 show_traj_history=True):
        super().__init__('retraction_experiments_node')

        self.service_timeout_sec = service_timeout_sec

        self.dvrk_arm_name = dvrk_arm_name

        self.push_method = push_method

        # 'real': the robot takes commands in the point cloud (camera) frame, so predicted points are sent as is.
        # 'sim': the robot takes commands in DVRK_PSM_BASE_FRAME, so predicted points are transformed there with TF first.
        if robot_mode not in ('real', 'sim'):
            raise ValueError(f"robot_mode must be 'real' or 'sim', got {robot_mode!r}")
        self.robot_mode = robot_mode
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Latest clouds, written by the subscriber callbacks on the executor thread
        self._lock = threading.Lock()
        self._latest_current_pc = None
        self._latest_context_pc = None

        # Snapshots, taken from the experiment loop
        self.current_pc_snapshot = None
        self.context_pc_snapshot = None
        self.goal_pc_snapshot = None

        # ROS I/O
        self.current_pc_sub = self.create_subscription(
            PointCloud2, current_pc_topic, self.current_pc_callback, qos_profile_sensor_data)
        self.context_pc_sub = self.create_subscription(
            PointCloud2, context_pc_topic, self.context_pc_callback, qos_profile_sensor_data)

        self.current_pc_snapshot_pub = self.create_publisher(
            PointCloud2, '/pushvibes_tissue_pointcloud', SNAPSHOT_QOS)
        self.goal_pc_snapshot_pub = self.create_publisher(
            PointCloud2, '/pushvibes_goal_pointcloud', SNAPSHOT_QOS)
        self.context_pc_snapshot_pub = self.create_publisher(
            PointCloud2, '/pushvibes_context_pointcloud', SNAPSHOT_QOS)

        self.goal_pc_client = self.create_client(GenerateGoalPointCloud, goal_service_name)

        # dVRK arm client, sharing this node instead of creating its own
        self.ral = crtk.ral('retraction_experiments_node', node=self)
        self.psm = dvrk.psm(self.ral, dvrk_arm_name)

        # Republish the snapshots so they can be visualized, and reach subscribers that joined late
        self.snapshot_timer = self.create_timer(1.0 / snapshot_publish_rate, self.publish_snapshots)

        # Rviz marker(s). With show_traj_history, executed trajectories stay up in dark green
        self.pushvibes_traj_marker = None
        self.show_traj_history = show_traj_history
        self.executed_traj_markers = []  # Markers of executed trajectories, only kept if show_traj_history
        self.marker_pub = self.create_publisher(MarkerArray, '/pushvibes_trajectory_marker', SNAPSHOT_QOS)
        self.marker_pub_rate = 5.0  # Hz
        self.marker_pub_timer = self.create_timer(1.0 / self.marker_pub_rate, self.publish_rviz_traj_marker)
        

        ###

        self.pushvibes_deployer = DeployPushVIBES(weights_path=os.path.expanduser("~/model_weights/PushVIBES/561real_561sim_lr1e4_cotrainlocal_20260323_131410.pt"),
                                                  basis_path="./bps/bps_basis_r1.0_n128.npy")

        self.get_logger().info('Retraction experiments node initialized.')

    # ===========================
    # SUBSCRIBER CALLBACKS
    # ===========================
    def current_pc_callback(self, msg):
        with self._lock:
            self._latest_current_pc = msg

    def context_pc_callback(self, msg):
        with self._lock:
            self._latest_context_pc = msg

    # ===========================
    # SNAPSHOT PUBLISHING
    # ===========================
    def publish_snapshots(self):
        """Publish every snapshot taken so far."""
        if self.current_pc_snapshot is not None:
            self.current_pc_snapshot_pub.publish(self.current_pc_snapshot)
        if self.context_pc_snapshot is not None:
            self.context_pc_snapshot_pub.publish(self.context_pc_snapshot)
        if self.goal_pc_snapshot is not None:
            self.goal_pc_snapshot_pub.publish(self.goal_pc_snapshot)

    # ===========================
    # SNAPSHOTS
    # ===========================
    def snapshot_current_pc(self):
        """Hold the latest current (tissue) cloud in current_pc_snapshot. Returns True on success."""
        with self._lock:
            msg = self._latest_current_pc
        if msg is None:
            self.get_logger().warn('No current point cloud received yet; nothing to snapshot.')
            return False

        self.current_pc_snapshot = msg
        self.current_pc_snapshot_pub.publish(msg)
        self.get_logger().info('Snapshotted current point cloud.')
        return True

    def snapshot_context_pc(self):
        """Hold the latest context cloud in context_pc_snapshot. Returns True on success."""
        with self._lock:
            msg = self._latest_context_pc
        if msg is None:
            self.get_logger().warn('No context point cloud received yet; nothing to snapshot.')
            return False

        self.context_pc_snapshot = msg
        self.context_pc_snapshot_pub.publish(msg)
        self.get_logger().info('Snapshotted context point cloud.')
        return True

    def request_diffdef_goal_pc(self):
        """Request a goal cloud for the current and context snapshots and hold it in goal_pc_snapshot.

        Blocks until the DiffDef service responds. Returns True on success.
        """
        if self.current_pc_snapshot is None or self.context_pc_snapshot is None:
            self.get_logger().error(
                'Snapshot the current and context point clouds before requesting a goal.')
            return False

        if not self.goal_pc_client.wait_for_service(timeout_sec=5.0):
            self.get_logger().error(
                f'Goal service {self.goal_pc_client.srv_name} is not available.')
            return False

        request = GenerateGoalPointCloud.Request()
        request.tissue_pointcloud = self.current_pc_snapshot
        request.context_pointcloud = self.context_pc_snapshot

        response = self.goal_pc_client.call(request, timeout_sec=self.service_timeout_sec)
        if response is None:
            self.get_logger().error(
                f'Goal service did not respond within {self.service_timeout_sec} s.')
            return False
        if not response.success:
            self.get_logger().error(f'Goal service failed: {response.message}')
            return False

        self.goal_pc_snapshot = response.goal_pointcloud
        self.goal_pc_snapshot_pub.publish(self.goal_pc_snapshot)
        self.get_logger().info(f'Snapshotted goal point cloud. {response.message}')
        return True

    def snapshot_current_pc_for_goal_pc(self):
        """Hold the latest current (tissue) cloud in goal_pc_snapshot, for manual goal experiments.

        Returns True on success.
        """
        with self._lock:
            msg = self._latest_current_pc
        if msg is None:
            self.get_logger().warn('No current point cloud received yet; nothing to snapshot.')
            return False

        self.goal_pc_snapshot = msg
        self.goal_pc_snapshot_pub.publish(msg)
        self.get_logger().info('Snapshotted current point cloud as the goal point cloud.')
        return True

    def set_rviz_traj_marker(self, start_point, end_point, frame_id="camera_color_optical_frame"):
        """Set the Rviz markers for the predicted trajectory.

        A single ARROW marker can only have one color, so the trajectory is drawn as three
        markers: a bright green shaft, an electric blue arrow head and a cherry red start ball.
        """
        start = [float(v) for v in start_point]
        end = [float(v) for v in end_point]
        length = math.dist(start, end)
        direction = [(e - s) / length for s, e in zip(start, end)] if length > 0 else [0.0, 0.0, 0.0]
        head_length = min(0.0052, 0.52 * length)
        head_base = [e - head_length * d for e, d in zip(end, direction)]
        stamp = self.get_clock().now().to_msg()

        def make_marker(marker_id, marker_type, rgb):
            marker = Marker()
            marker.header.frame_id = frame_id
            marker.header.stamp = stamp
            marker.ns = "pushvibes_trajectory"
            marker.id = marker_id
            marker.type = marker_type
            marker.action = Marker.ADD
            marker.color.r, marker.color.g, marker.color.b = rgb
            marker.color.a = 1.0
            marker.lifetime = Duration(seconds=3.0).to_msg()  # Republished at marker_pub_rate, so they only expire once the node stops
            return marker

        # Shaft: bright green line from the start point to the base of the arrow head
        shaft = make_marker(0, Marker.LINE_STRIP, (0.0, 1.0, 0.0))
        shaft.scale.x = 0.0026  # Line width in meters
        shaft.points = [Point(x=start[0], y=start[1], z=start[2]),
                        Point(x=head_base[0], y=head_base[1], z=head_base[2])]

        # Head: electric blue arrow spanning only the head, so no shaft is drawn
        head = make_marker(1, Marker.ARROW, (0.1, 0.7, 1.0))
        head.scale.x = 0.0026  # Shaft diameter (unused, shaft has zero length)
        head.scale.y = 0.0065  # Head diameter
        head.scale.z = head_length  # Head length
        head.points = [Point(x=head_base[0], y=head_base[1], z=head_base[2]),
                       Point(x=end[0], y=end[1], z=end[2])]

        # Start: cherry red ball
        ball = make_marker(2, Marker.SPHERE, (1.0, 0.05, 0.15))
        ball.pose.position = Point(x=start[0], y=start[1], z=start[2])
        ball.pose.orientation.w = 1.0
        ball.scale.x = ball.scale.y = ball.scale.z = 0.0052  # Diameter in meters

        self.pushvibes_traj_marker = MarkerArray(markers=[shaft, head, ball])

    def mark_marker_as_complete(self):
        """Change the color of the trajectory markers to a dark green, to indicate the trajectory has been executed.

        With show_traj_history, the markers move to the history (own namespace and ids, so the next
        prediction doesn't overwrite them) and stay up for the rest of the experiment.
        """
        if self.pushvibes_traj_marker is not None:
            for marker in self.pushvibes_traj_marker.markers:
                marker.color.r = 0.0
                marker.color.g = 0.3
                marker.color.b = 0.0

            if self.show_traj_history:
                for marker in self.pushvibes_traj_marker.markers:
                    marker.ns = "pushvibes_trajectory_history"
                    marker.id = len(self.executed_traj_markers)
                    self.executed_traj_markers.append(marker)
                self.pushvibes_traj_marker = None

    def publish_rviz_traj_marker(self):
        """Publish the Rviz markers for the executed trajectories (if show_traj_history) and the predicted trajectory."""
        markers = list(self.executed_traj_markers)
        if self.pushvibes_traj_marker is not None:
            markers += self.pushvibes_traj_marker.markers
        if markers:
            self.marker_pub.publish(MarkerArray(markers=markers))

    # ===========================
    # DVRK MOTION
    # ===========================
    def move_cp_linear(self, goal, step_size=0.002, speed=0.03):
        """Move the dVRK in a straight line to goal (PyKDL.Frame) at speed (m/s), blocking until the goal is reached.

        The line is split into waypoints about step_size meters apart (orientation is
        interpolated along with position), and each one is sent with move_cp on a fixed
        schedule, without waiting for the previous one to finish. Each move_cp preempts the
        last, so the arm follows the waypoints smoothly. Blocks until the goal is reached.

        Allowable speeds (measured on the kinematic sim with the default 2 mm step size):
        - Up to 5 cm/s: tracks the requested speed, stays within ~1 mm of the straight line.
        - 5 to 8 cm/s: still tracks the speed, but strays up to ~1.5 mm from the line.
        - Above ~8 cm/s: don't. move_cp's trajectory limits cap the arm at about 8 cm/s
          on average, and it strays 5 to 10 mm from the line.
        - Below ~1.2 cm/s: the average speed is right but the motion is stop-and-go, since
          each waypoint is reached before the next is sent. Use a smaller step_size for
          smooth slow motion (e.g. 1 mm is smooth down to ~0.8 cm/s).
        """
        if speed <= 0:
            raise ValueError(f'speed must be positive, got {speed}')
        start = self.psm.setpoint_cp(wait_timeout=2.0)[0]

        # Rotation from start to goal, expressed in the start frame, as an axis and angle
        angle, axis = (start.M.Inverse() * goal.M).GetRotAngle()

        distance = (goal.p - start.p).Norm()
        n_steps = max(1, math.ceil(distance / step_size))
        dt = distance / n_steps / speed

        # Schedule waypoints from a fixed start time so time spent sending doesn't slow the arm
        t_start = time.monotonic()
        for i in range(1, n_steps + 1):
            s = i / n_steps
            waypoint = PyKDL.Frame(start.M * PyKDL.Rotation.Rot(axis, s * angle),
                                   start.p + s * (goal.p - start.p))
            handle = self.psm.move_cp(waypoint)
            if i < n_steps:
                time.sleep(max(0.0, t_start + i * dt - time.monotonic()))
        handle.wait()

    def transform_points(self, vectors, source_frame, target_frame):
        """Transform PyKDL.Vectors from source_frame to target_frame using TF."""
        tf = self.tf_buffer.lookup_transform(target_frame, source_frame, Time(),
                                             timeout=Duration(seconds=2.0)).transform
        T = PyKDL.Frame(PyKDL.Rotation.Quaternion(tf.rotation.x, tf.rotation.y, tf.rotation.z, tf.rotation.w),
                        PyKDL.Vector(tf.translation.x, tf.translation.y, tf.translation.z))
        return [T * v for v in vectors]

    # ===========================
    # EXPERIMENT LOOP
    # ===========================
    def run_experiment(self):
        """Main blocking experiment loop. Runs on the main thread while callbacks spin in the background."""
        iteration = 0

        if not self.psm.enable(10):
            sys.exit('failed to enable within 10 seconds')

        # if input(f'Ensure the {self.dvrk_arm_name} gripper is closed and the wrist is oriented straight down. Press Enter to continue (q to quit): ').strip().lower() == 'q':
        #     return
        
        # dvrk_wrist_orientation = self.psm.measured_cp(wait_timeout=2.0)[0].M # type is PyKDL.Rotation

        # if input(f'Ensure the {self.dvrk_arm_name} end effector has 5cm of space around it for testing. Press Enter to continue (q to quit): ').strip().lower() == 'q':
        #     return


        # self.get_logger().info(f'Moving {self.dvrk_arm_name} in small test path.')

        # # Move the arm 1 cm back and forth in x to verify the arm can move
        # for dx in [-0.01, 0.01, 0.01, -0.01]:
        #     goal = PyKDL.Frame(dvrk_wrist_orientation, self.psm.measured_cp(wait_timeout=2.0)[0].p + PyKDL.Vector(dx, 0.0, 0.0))
        #     self.move_cp_linear(goal, step_size=0.0005, speed=0.005)

        # Every push keeps the wrist orientation the arm had at the start of the experiment,
        # and the arm returns to this pose when the experiments are finished
        original_pose = self.psm.setpoint_cp(wait_timeout=2.0)[0]
        original_orientation = original_pose.M
        original_position = original_pose.p

        self.get_logger().info(f'Robot mode: {self.robot_mode}.')

        input(f'Record goal (Enter)')

        if not self.snapshot_current_pc_for_goal_pc():
            self.get_logger().error('Failed to snapshot current point cloud for goal.')

        # Return to the original pose however the loop ends: q, an error, or Ctrl-C
        try:
            while rclpy.ok():
                if input(f'Press Enter to run iteration {iteration} (q to quit): ').strip().lower() == 'q':
                    break

                self.snapshot_current_pc()

                if self.current_pc_snapshot.header.frame_id != self.goal_pc_snapshot.header.frame_id:
                    self.get_logger().error(f'Current point cloud frame_id ({self.current_pc_snapshot.header.frame_id}) does not match goal point cloud frame_id ({self.goal_pc_snapshot.header.frame_id}).')
                    break

                current_pc_numpy  = point_cloud2.read_points_numpy(
                    self.current_pc_snapshot, 
                    field_names=['x', 'y', 'z'], 
                    skip_nans=True
                )

                goal_pc_numpy  = point_cloud2.read_points_numpy(
                    self.goal_pc_snapshot, 
                    field_names=['x', 'y', 'z'], 
                    skip_nans=True
                )

                # First push: plain PushVIBES. After that, optimize the push to start at the measured tool tip
                # (point cloud frame), so the tool doesn't have to reposition between pushes.
                if iteration == 0:

                    # Method selection
                    if self.push_method == 'PushVIBES_start_point_optimized' or self.push_method == 'PushVIBES_start_point_conditioned':
                        predicted_start_point, predicted_end_point, _preds = self.pushvibes_deployer.predict_push(current_pc_numpy, goal_pc_numpy, visualize_single=False)
                
                
                
                
                else:
                    wrist_p = self.psm.measured_cp(wait_timeout=2.0)[0].p
                    # Sim: measured_cp is in DVRK_PSM_BASE_FRAME, so bring it back to the point cloud frame
                    if self.robot_mode == 'sim':
                        wrist_p = self.transform_points([wrist_p], DVRK_PSM_BASE_FRAME,
                                                      self.current_pc_snapshot.header.frame_id)[0]
                    wrist_position = [wrist_p.x(), wrist_p.y(), wrist_p.z()]


                    # Method selection
                    if self.push_method == 'PushVIBES_start_point_optimized':
                        predicted_start_point, predicted_end_point, _history = self.pushvibes_deployer.predict_push_start_point_optimized(
                            current_pc_numpy, goal_pc_numpy, wrist_position, visualize_single=False)



                    
                    self.get_logger().info(f'Start point optimized: {math.dist(predicted_start_point, wrist_position) * 1000:.2f} mm from the measured tool tip.')

                self.set_rviz_traj_marker(predicted_start_point, predicted_end_point, frame_id=self.current_pc_snapshot.header.frame_id)

                start_p = PyKDL.Vector(*(float(v) for v in predicted_start_point))
                end_p = PyKDL.Vector(*(float(v) for v in predicted_end_point))

                # Real: the robot takes commands in the point cloud (camera) frame, so the points are used as is.
                # Sim: the robot takes commands in DVRK_PSM_BASE_FRAME, so transform the points there first.
                if self.robot_mode == 'sim':
                    start_p, end_p = self.transform_points([start_p, end_p],
                                                           self.current_pc_snapshot.header.frame_id,
                                                           DVRK_PSM_BASE_FRAME)
                self.get_logger().info(f'Moving to start point {list(start_p)}.')
                self.move_cp_linear(PyKDL.Frame(original_orientation, start_p))
                time.sleep(1.0)
                self.get_logger().info(f'Moving to end point {list(end_p)}.')
                self.move_cp_linear(PyKDL.Frame(original_orientation, end_p))
                self.mark_marker_as_complete()

                iteration += 1
        finally:
            self.get_logger().info(f'Returning to original pose {list(original_position)}.')
            try:
                self.move_cp_linear(PyKDL.Frame(original_orientation, original_position))
            except Exception as e:
                self.get_logger().error(f'Failed to return to original pose: {e}')



# ===========================
# MAIN
# ===========================
def main(args=None):
    rclpy.init(args=args)
    node = RetractionExperimentsNode(robot_mode='sim', 
                                     push_method='PushVIBES_start_point_optimized', # ~ {PushVIBES_start_point_optimized, PushVIBES_start_point_conditioned, PushVIBES_deterministic, heuristic}
                                     show_traj_history=True)

    # Spin callbacks (subscribers, service responses) on a background thread so the
    # experiment loop can block on the main thread
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    try:
        node.ral.check_connections()
        node.run_experiment()
    except (KeyboardInterrupt, EOFError, ExternalShutdownException):
        pass
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.try_shutdown()
        spin_thread.join(timeout=1.0)


if __name__ == '__main__':
    main()
