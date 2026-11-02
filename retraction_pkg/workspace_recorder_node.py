#!/usr/bin/env python3
"""Record the dVRK tool position to a .json file while it traces the outer edge of the workspace on the table.

Parameters:
    rate_hz:      how many points to record per second
    topic:        PoseStamped topic to record
    target_frame: frame the points are saved in
    output_path:  .json file to write, read back by retraction_experiments_node
    overwrite:    allow replacing an existing output_path

Example:
    ros2 run retraction_pkg workspace_recorder_node --ros-args -p rate_hz:=5.0
"""

import json
import os

import PyKDL
import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.time import Time
from tf2_ros import Buffer, TransformException, TransformListener


class WorkspaceRecorderNode(Node):

    def __init__(self):
        super().__init__('workspace_recorder_node')

        rate_hz = self.declare_parameter('rate_hz', 2.0).value
        topic = self.declare_parameter('topic', '/PSM1/setpoint_cp').value
        self.target_frame = self.declare_parameter('target_frame', 'camera_color_optical_frame').value
        self.output_path = os.path.expanduser(
            self.declare_parameter('output_path', '~/retraction_ws/workspace_boundary.json').value)
        overwrite = self.declare_parameter('overwrite', False).value

        # The file is rewritten on every point, so starting a new recording would erase the old trace right away
        if os.path.exists(self.output_path) and not overwrite:
            raise FileExistsError(f'{self.output_path} already exists. Move it, set a new output_path, '
                                  'or run with -p overwrite:=true.')

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.latest_msg = None
        self.source_frame_logged = False
        self.points = []

        self.pose_sub = self.create_subscription(PoseStamped, topic, self.pose_callback, 10)
        self.record_timer = self.create_timer(1.0 / rate_hz, self.record_point)

        self.get_logger().info(f'Recording {topic} at {rate_hz} Hz to {self.output_path}. Ctrl-C to stop.')

    def pose_callback(self, msg):
        self.latest_msg = msg

    def to_target_frame(self, msg):
        """Return msg's position as [x, y, z] in target_frame. Raises TransformException if TF can't provide it."""
        position = msg.pose.position
        if msg.header.frame_id == self.target_frame:
            return [position.x, position.y, position.z]

        tf = self.tf_buffer.lookup_transform(self.target_frame, msg.header.frame_id, Time()).transform
        T = PyKDL.Frame(PyKDL.Rotation.Quaternion(tf.rotation.x, tf.rotation.y, tf.rotation.z, tf.rotation.w),
                        PyKDL.Vector(tf.translation.x, tf.translation.y, tf.translation.z))
        return list(T * PyKDL.Vector(position.x, position.y, position.z))

    def record_point(self):
        """Save the latest position, if a new one came in. The file is written every time so a crash doesn't lose the trace."""
        msg = self.latest_msg
        self.latest_msg = None  # setpoint_cp only publishes during teleoperation, so a pause records nothing
        if msg is None:
            self.get_logger().info('Waiting for a new pose (is the arm being teleoperated?).', throttle_duration_sec=5.0)
            return

        try:
            point = self.to_target_frame(msg)
        except TransformException as e:
            self.get_logger().warn(f'Point not recorded: {e}', throttle_duration_sec=5.0)
            return

        if not self.source_frame_logged:
            self.get_logger().info(f"Poses are in frame '{msg.header.frame_id}', saving in '{self.target_frame}'.")
            self.source_frame_logged = True

        self.points.append(point)
        temp_path = f'{self.output_path}.tmp'
        with open(temp_path, 'w') as f:
            json.dump({'frame_id': self.target_frame, 'points': self.points}, f, indent=2)
        os.replace(temp_path, self.output_path)  # So stopping mid-write can't leave a partial file


def main(args=None):
    rclpy.init(args=args)
    node = WorkspaceRecorderNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.get_logger().info(f'Recorded {len(node.points)} points to {node.output_path}.')
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
