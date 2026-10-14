#!/usr/bin/env python3
"""Run retraction experiments from snapshots of the live tissue and context point clouds."""

import math
import threading
import time

import crtk
import dvrk
import PyKDL
import rclpy
from rclpy.executors import ExternalShutdownException, SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, qos_profile_sensor_data
from retraction_interface.srv import GenerateGoalPointCloud
from sensor_msgs.msg import PointCloud2

# Latched, so a late subscriber gets the last snapshot right away instead of waiting for the next timer tick
SNAPSHOT_QOS = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)


class RetractionExperimentsNode(Node):

    def __init__(self,
                 dvrk_arm_name='PSM1',
                 current_pc_topic='/tissue_pointcloud',
                 context_pc_topic='/context_pointcloud',
                 goal_service_name='/generate_goal_pointcloud',
                 service_timeout_sec=60.0,
                 snapshot_publish_rate=0.5):
        super().__init__('retraction_experiments_node')

        self.service_timeout_sec = service_timeout_sec

        self.dvrk_arm_name = dvrk_arm_name

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

    # ===========================
    # DVRK MOTION
    # ===========================
    def move_cp_linear(self, goal, step_size=0.002, speed=0.03):
        """Move the dVRK in a straight line to goal (PyKDL.Frame) at speed (m/s).

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

    # ===========================
    # EXPERIMENT LOOP
    # ===========================
    def run_experiment(self):
        """Main blocking experiment loop. Runs on the main thread while callbacks spin in the background."""
        iteration = 0

        if input(f'Ensure the {self.dvrk_arm_name} gripper is closed and the wrist is oriented straight down. Press Enter to continue (q to quit): ').strip().lower() == 'q':
            return
        
        dvrk_wrist_orientation = self.psm.measured_cp(wait_timeout=2.0)[0].M # type is PyKDL.Rotation

        if input(f'Ensure the {self.dvrk_arm_name} end effector has 5cm of space around it for testing. Press Enter to continue (q to quit): ').strip().lower() == 'q':
            return

        self.get_logger().info(f'Moving {self.dvrk_arm_name} in small test path.')

        # Move the arm 2 cm back and forth in x to verify the arm can move
        for dx in [-0.02, 0.02]:
            goal = PyKDL.Frame(dvrk_wrist_orientation, self.psm.measured_cp(wait_timeout=2.0)[0].p + PyKDL.Vector(dx, 0.0, 0.0))
            self.move_cp_linear(goal, step_size=0.002, speed=0.03)



        while rclpy.ok():
            if input(f'Press Enter to run iteration {iteration} (q to quit): ').strip().lower() == 'q':
                break


            
            self.run_iteration(iteration)
            iteration += 1

    def run_iteration(self, iteration):
        print(f'Running iteration {iteration}')


# ===========================
# MAIN
# ===========================
def main(args=None):
    rclpy.init(args=args)
    node = RetractionExperimentsNode()

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
