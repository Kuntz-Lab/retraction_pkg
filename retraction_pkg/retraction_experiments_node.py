#!/usr/bin/env python3
"""Run retraction experiments from snapshots of the live tissue and context point clouds."""

import threading

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
                 current_pc_topic='/tissue_pointcloud',
                 context_pc_topic='/context_pointcloud',
                 goal_service_name='/generate_goal_pointcloud',
                 service_timeout_sec=60.0,
                 snapshot_publish_rate=0.5):
        super().__init__('retraction_experiments_node')

        self.service_timeout_sec = service_timeout_sec

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
    # EXPERIMENT LOOP
    # ===========================
    def run_experiments(self):
        """Main blocking experiment loop. Runs on the main thread while callbacks spin in the background."""
        experiment_idx = 0
        while rclpy.ok():
            if input(f'Press Enter to run experiment {experiment_idx} (q to quit): ').strip().lower() == 'q':
                break

            self.run_experiment(experiment_idx)
            experiment_idx += 1

    def run_experiment(self, experiment_idx):
        print(f'Running experiment {experiment_idx}')


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
        node.run_experiments()
    except (KeyboardInterrupt, EOFError, ExternalShutdownException):
        pass
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.try_shutdown()
        spin_thread.join(timeout=1.0)


if __name__ == '__main__':
    main()
