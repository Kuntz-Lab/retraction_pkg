#!/usr/bin/env python3
"""Record the dVRK tool while it traces the outer edge of the workspace on the table, to a .json file.

By default this records the arm's joint positions. retraction_experiments_node turns them into points with
the dVRK's forward kinematics at startup, so the workspace follows whichever hand-eye calibration is loaded.
With record_mode:=cartesian it records tool positions in target_frame instead, which are tied to the
calibration that was loaded while recording.

Parameters:
    record_mode:  'joint' (default) or 'cartesian'
    arm:          dVRK arm to record
    rate_hz:      how many points to record per second
    topic:        topic to record. Defaults to /<arm>/setpoint_js in joint mode, /<arm>/setpoint_cp in cartesian mode
    target_frame: cartesian mode only, frame the points are saved in
    output_path:  .json file to write, read back by retraction_experiments_node
    overwrite:    allow replacing an existing output_path

Example:
    ros2 run retraction_pkg workspace_recorder_node --ros-args -p rate_hz:=5.0
    ros2 run retraction_pkg workspace_recorder_node --ros-args -p record_mode:=cartesian
"""

import json
import os

import PyKDL
import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.time import Time
from sensor_msgs.msg import JointState
from tf2_ros import Buffer, TransformException, TransformListener


class WorkspaceRecorderNode(Node):

    def __init__(self):
        super().__init__('workspace_recorder_node')

        self.record_mode = self.declare_parameter('record_mode', 'joint').value
        if self.record_mode not in ('joint', 'cartesian'):
            raise ValueError(f"record_mode must be 'joint' or 'cartesian', got {self.record_mode!r}")
        self.arm = self.declare_parameter('arm', 'PSM1').value
        rate_hz = self.declare_parameter('rate_hz', 2.0).value
        default_topic = f'/{self.arm}/setpoint_js' if self.record_mode == 'joint' else f'/{self.arm}/setpoint_cp'
        topic = self.declare_parameter('topic', '').value or default_topic
        self.target_frame = self.declare_parameter('target_frame', 'camera_color_optical_frame').value
        self.output_path = os.path.expanduser(
            self.declare_parameter('output_path', '~/retraction_ws/workspace_boundary.json').value)
        overwrite = self.declare_parameter('overwrite', False).value

        # The file is rewritten on every point, so starting a new recording would erase the old trace right away
        if os.path.exists(self.output_path) and not overwrite:
            raise FileExistsError(f'{self.output_path} already exists. Move it, set a new output_path, '
                                  'or run with -p overwrite:=true.')

        self.latest_msg = None
        self.points = []

        if self.record_mode == 'joint':
            self.joint_names = None
            self.sub = self.create_subscription(JointState, topic, self.msg_callback, 10)
            self.record_timer = self.create_timer(1.0 / rate_hz, self.record_joint_positions)
        else:
            self.tf_buffer = Buffer()
            self.tf_listener = TransformListener(self.tf_buffer, self)
            self.source_frame_logged = False
            self.sub = self.create_subscription(PoseStamped, topic, self.msg_callback, 10)
            self.record_timer = self.create_timer(1.0 / rate_hz, self.record_point)

        self.get_logger().info(f'Recording {topic} ({self.record_mode} mode) at {rate_hz} Hz to {self.output_path}. Ctrl-C to stop.')

    def msg_callback(self, msg):
        self.latest_msg = msg

    def take_latest_msg(self):
        """Return the newest message since the last call, or None (with a throttled log) if none came in."""
        msg = self.latest_msg
        self.latest_msg = None
        if msg is None:
            self.get_logger().info(f'Waiting for a new message (is {self.arm} powered on?).', throttle_duration_sec=5.0)
        return msg

    def save(self, data):
        """Write data to output_path. The file is written on every point so a crash doesn't lose the trace."""
        temp_path = f'{self.output_path}.tmp'
        with open(temp_path, 'w') as f:
            json.dump(data, f, indent=2)
        os.replace(temp_path, self.output_path)  # So stopping mid-write can't leave a partial file

    # ===========================
    # JOINT MODE
    # ===========================
    def record_joint_positions(self):
        """Save the latest joint positions, if they changed since the last saved point."""
        msg = self.take_latest_msg()
        if msg is None:
            return

        if self.joint_names is None:
            self.joint_names = list(msg.name)
            self.get_logger().info(f'Recording joints {self.joint_names}.')
        elif list(msg.name) != self.joint_names:
            self.get_logger().warn(f'Point not recorded: joints {list(msg.name)} differ from {self.joint_names}.',
                                   throttle_duration_sec=5.0)
            return

        # setpoint_js publishes continuously, so skip repeats while the arm is held still.
        # Each saved point costs a forward kinematics call when retraction_experiments_node starts.
        positions = list(msg.position)
        if self.points and positions == self.points[-1]:
            return

        self.points.append(positions)
        self.save({'arm': self.arm, 'joint_names': self.joint_names, 'joint_positions': self.points})

    # ===========================
    # CARTESIAN MODE
    # ===========================
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
        """Save the latest position in target_frame."""
        msg = self.take_latest_msg()
        if msg is None:
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
        self.save({'frame_id': self.target_frame, 'points': self.points})


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
