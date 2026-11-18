"""Helpers for retraction_experiments_node: table plane geometry, the workspace polygon, and dVRK config lookup."""

import json
import os

import numpy as np
import PyKDL


# Measured distance from the wrist (the wrist yaw axis, where setpoint_cp sits) to the tool tip, along the wrist z axis
TOOL_TIP_OFFSET_M = 0.028
WRIST_TO_TIP = PyKDL.Frame(PyKDL.Vector(0.0, 0.0, TOOL_TIP_OFFSET_M))


# ===========================
# TABLE PLANE
# ===========================
def plane_normal(plane_points, point_on_operating_side):
    """Return the unit normal of the plane through plane_points, pointing to the side of point_on_operating_side."""
    p0, p1, p2 = plane_points
    normal = np.cross(p1 - p0, p2 - p0)
    normal /= np.linalg.norm(normal)
    if np.dot(point_on_operating_side - p0, normal) < 0:
        normal = -normal
    return normal


def offset_plane_points(plane_points, point_on_operating_side, offset_mm):
    """Return plane_points moved offset_mm along the plane normal, away from point_on_operating_side."""
    normal = plane_normal(plane_points, point_on_operating_side)
    return [point - offset_mm / 1000.0 * normal for point in plane_points]


def plane_axes(plane_points, point_on_operating_side):
    """Return unit (x, y, z) axes of a right-handed frame on the plane, with z the plane normal."""
    p0, p1, _ = plane_points
    x_axis = (p1 - p0) / np.linalg.norm(p1 - p0)
    z_axis = plane_normal(plane_points, point_on_operating_side)
    return x_axis, np.cross(z_axis, x_axis), z_axis


# ===========================
# WORKSPACE POLYGON
# ===========================
def load_workspace_boundary(path):
    """Load a trace from workspace_recorder_node.

    Returns its dict: {'arm', 'joint_names', 'joint_positions'} from joint mode, or {'frame_id', 'points'}
    from cartesian mode.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f'No workspace boundary at {path}. Trace the workspace edge with workspace_recorder_node first.')
    with open(path) as f:
        return json.load(f)


def workspace_polygon(points, origin, x_axis, y_axis):
    """Return the traced Nx3 points as an Nx2 polygon, in meters along x_axis and y_axis from origin.

    The polygon keeps the recorded order, so it follows the trace even where the trace curves inward.
    """
    points = np.asarray(points, dtype=float).reshape(-1, 3)
    polygon = (points - origin) @ np.stack([x_axis, y_axis], axis=1)

    # The arm sits still at times during a trace, and repeated points would make zero length edges
    keep = np.linalg.norm(polygon - np.roll(polygon, 1, axis=0), axis=1) > 1e-6
    polygon = polygon[keep]
    if len(polygon) < 3:
        raise ValueError(f'The workspace trace has {len(polygon)} distinct points, a polygon needs at least 3.')
    return polygon


def point_in_polygon(point, polygon):
    """Return True if the 2D point is inside polygon, by winding number so a trace that overlaps its start stays filled."""
    x, y = point
    starts = polygon
    ends = np.roll(polygon, -1, axis=0)
    edges = ends - starts
    side = edges[:, 0] * (y - starts[:, 1]) - edges[:, 1] * (x - starts[:, 0])  # Positive if point is left of the edge
    crosses_up = (starts[:, 1] <= y) & (ends[:, 1] > y) & (side > 0)
    crosses_down = (starts[:, 1] > y) & (ends[:, 1] <= y) & (side < 0)
    return bool(np.sum(crosses_up) != np.sum(crosses_down))


def closest_point_on_polygon(point, polygon):
    """Return the point on the polygon's outline closest to the 2D point, and the index of its edge."""
    edges = np.roll(polygon, -1, axis=0) - polygon
    t = np.clip(np.sum((point - polygon) * edges, axis=1) / np.sum(edges * edges, axis=1), 0.0, 1.0)
    closest = polygon + t[:, np.newaxis] * edges
    index = int(np.argmin(np.linalg.norm(point - closest, axis=1)))
    return closest[index], index


def move_inside_polygon(point, polygon, clearance, tolerance=1e-6, max_iterations=2000):
    """Return the 2D point moved the shortest way to be inside polygon, at least clearance from its outline.

    Each step sets the distance to the nearest edge to clearance. Near a corner that can put the point too
    close to the other edge, so this repeats until every edge is clear.
    """
    edges = np.roll(polygon, -1, axis=0) - polygon
    counterclockwise = np.sum(polygon[:, 0] * np.roll(polygon, -1, axis=0)[:, 1]
                              - np.roll(polygon, -1, axis=0)[:, 0] * polygon[:, 1]) > 0

    for _ in range(max_iterations):
        closest, edge_index = closest_point_on_polygon(point, polygon)
        offset = point - closest
        distance = np.linalg.norm(offset)
        inside = point_in_polygon(point, polygon)
        if inside and distance >= clearance - tolerance:
            return point

        if distance > 1e-9:
            inward = offset / distance if inside else -offset / distance
        else:  # On the outline, where offset is only rounding error, so use the normal of the edge instead
            edge = edges[edge_index] / np.linalg.norm(edges[edge_index])
            inward = np.array([-edge[1], edge[0]]) if counterclockwise else np.array([edge[1], -edge[0]])
        point = closest + clearance * inward

    raise ValueError(f'No point {clearance * 1000:.1f} mm inside the workspace polygon was found near the input.')


# ===========================
# PUSH ACTIONS
# ===========================
def clip_action(start_point, end_point, keep_ratio=1.0):
    """Return the end point moved along the push so the push keeps keep_ratio of its original length."""
    start = np.asarray(start_point, dtype=float)
    push_vector = np.asarray(end_point, dtype=float) - start
    return start + keep_ratio * push_vector


# ===========================
# TOOL TIP
# ===========================
def tip_pose_from_wrist_pose(wrist_pose):
    """Return the tool tip pose (PyKDL.Frame) for a wrist pose (PyKDL.Frame), e.g. from setpoint_cp.

    The tip has the wrist's orientation, TOOL_TIP_OFFSET_M along the wrist z axis. The result is in the
    same frame as wrist_pose.
    """
    return wrist_pose * WRIST_TO_TIP


def wrist_pose_from_tip_pose(tip_pose):
    """Return the wrist pose (PyKDL.Frame), e.g. for move_cp, that puts the tool tip at tip_pose (PyKDL.Frame).

    The inverse of tip_pose_from_wrist_pose. The result is in the same frame as tip_pose.
    """
    return tip_pose * WRIST_TO_TIP.Inverse()


# ===========================
# DVRK CONFIG
# ===========================
def find_dvrk_console_file():
    """Return the absolute path of the -j config of the running dvrk_system.

    Reads the command line and working directory of the dvrk_system process from /proc, so it works
    whichever directory `ros2 run dvrk_robot dvrk_system` was launched from. Only works on the machine
    running dvrk_system.
    """
    hits = []
    for pid in filter(str.isdigit, os.listdir('/proc')):
        try:
            with open(f'/proc/{pid}/cmdline', 'rb') as f:
                argv = [a.decode() for a in f.read().split(b'\0') if a]
            # Match the binary only, not the `ros2 run` wrapper, which has the same arguments
            if not argv or os.path.basename(argv[0]) != 'dvrk_system':
                continue
            cfg = None
            for i, a in enumerate(argv):
                if a in ('-j', '--json-config') and i + 1 < len(argv):
                    cfg = argv[i + 1]
                elif a.startswith('--json-config='):
                    cfg = a.split('=', 1)[1]
            if cfg is None:
                continue
            cwd = os.readlink(f'/proc/{pid}/cwd')
            hits.append(os.path.normpath(os.path.join(cwd, os.path.expanduser(cfg))))
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue  # The process exited while reading it, or isn't ours
    if len(hits) != 1:
        raise RuntimeError(f'Expected exactly one dvrk_system, found {len(hits)}: {hits}')
    path = hits[0]
    # A launch directory that was moved to the Trash or deleted after launch still resolves, but shouldn't be trusted
    if not os.path.isfile(path) or '/.local/share/Trash/' in path or path.endswith(' (deleted)'):
        raise RuntimeError(f'dvrk_system config path is not usable: {path}')
    return path


def dvrk_arm_reference_frame(arm_name):
    """Return the base_frame reference_frame the running dvrk_system config sets for arm_name, or None if it sets none.

    This is the frame the arm's Cartesian setpoints, measurements and forward kinematics are reported in.
    """
    path = find_dvrk_console_file()
    with open(path) as f:
        config = json.load(f)
    for arm in config.get('arms', []):
        if arm.get('name') == arm_name:
            return arm.get('base_frame', {}).get('reference_frame')
    raise ValueError(f'{arm_name} is not in the arms of dvrk_system config {path}.')


def dvrk_is_front_camera_calibrated():
    """Return True if the running dvrk_system config file name has 'front' in it, False if it has 'side'.

    Raises ValueError if the name has neither, or both.
    """
    path = find_dvrk_console_file()
    name = os.path.basename(path).lower()
    has_side = 'side' in name
    has_front = 'front' in name
    if has_side == has_front:
        raise ValueError(f"dvrk_system config {path} must have exactly one of 'side' or 'front' in its name.")
    return has_front
