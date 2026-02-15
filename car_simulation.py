#!/usr/bin/env python3
"""Lanelet2 car simulation (ROS1).

Simulates a car (kinematic bicycle model) driving through the Lanelet2 example map,
following a route using an MPPI controller (steering + acceleration).

This script publishes ROS topics suitable for visualization in Foxglove via
rosbridge_websocket.
"""

import os
import math
import re
import numpy as np

import rospy
from geometry_msgs.msg import (
    PoseStamped,
    TwistStamped,
    PointStamped,
    TransformStamped,
    Point,
)
from nav_msgs.msg import Path
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray

import tf2_ros

import lanelet2
from lanelet2.projection import UtmProjector


# =============================================================================
# Kinematic Bicycle Model
# =============================================================================


class BicycleModel:
    """
    Kinematic bicycle model for vehicle simulation.

    State: [x, y, theta, v]
    Inputs: [delta (steering angle), a (acceleration)]
    Parameters: L (wheelbase)
    """

    def __init__(self, x=0.0, y=0.0, theta=0.0, v=0.0, wheelbase=2.5):
        self.x = x
        self.y = y
        self.theta = theta  # heading angle
        self.v = v  # velocity
        self.L = wheelbase  # wheelbase length

        # Vehicle dimensions for visualization
        self.length = 4.0
        self.width = 1.8

    def update(self, dt, delta, a):
        """
        Update vehicle state using kinematic bicycle model equations.

        Args:
            dt: time step
            delta: steering angle (radians)
            a: acceleration (m/s^2)
        """
        # Kinematic bicycle model equations
        x_dot = self.v * np.cos(self.theta)
        y_dot = self.v * np.sin(self.theta)
        theta_dot = self.v * np.tan(delta) / self.L
        v_dot = a

        # Euler integration
        self.x += x_dot * dt
        self.y += y_dot * dt
        self.theta += theta_dot * dt
        self.v += v_dot * dt

        # Normalize theta to [-pi, pi]
        self.theta = np.arctan2(np.sin(self.theta), np.cos(self.theta))

        # Clamp velocity to non-negative (no reverse)
        self.v = max(0.0, self.v)

    @property
    def state(self):
        return np.array([self.x, self.y, self.theta, self.v])


def _wrap_angle_rad(a: np.ndarray):
    """Wrap angle(s) to [-pi, pi]."""
    return np.arctan2(np.sin(a), np.cos(a))


def _resample_polyline_equal_spacing(points_xy, ds: float):
    """Resample a polyline to approximately equal arc-length spacing.

    Args:
        points_xy: list/array of (x, y)
        ds: target spacing in meters

    Returns:
        pts: (N, 2) ndarray of resampled points
        s: (N,) cumulative arc length for resampled points
    """
    pts = np.asarray(points_xy, dtype=float)
    if pts.shape[0] < 2:
        return pts.copy(), np.zeros((pts.shape[0],), dtype=float)

    seg = pts[1:] - pts[:-1]
    seg_len = np.hypot(seg[:, 0], seg[:, 1])
    seg_len = np.maximum(seg_len, 1e-9)
    s_src = np.concatenate(([0.0], np.cumsum(seg_len)))
    s_total = float(s_src[-1])
    if s_total <= ds:
        return pts.copy(), s_src

    s_tgt = np.arange(0.0, s_total, float(ds))
    if s_tgt.size == 0 or s_tgt[-1] < s_total:
        s_tgt = np.append(s_tgt, s_total)

    out = np.zeros((s_tgt.shape[0], 2), dtype=float)
    j = 0
    for i, si in enumerate(s_tgt):
        while j < len(s_src) - 2 and s_src[j + 1] < si:
            j += 1
        s0 = s_src[j]
        s1 = s_src[j + 1]
        t = 0.0 if s1 <= s0 else float((si - s0) / (s1 - s0))
        out[i] = (1.0 - t) * pts[j] + t * pts[j + 1]

    return out, s_tgt


def _path_yaw_from_xy(path_xy: np.ndarray):
    if path_xy.shape[0] < 2:
        return np.zeros((path_xy.shape[0],), dtype=float)
    d = path_xy[1:] - path_xy[:-1]
    yaw = np.arctan2(d[:, 1], d[:, 0])
    yaw = np.concatenate((yaw, [yaw[-1]]))
    return yaw


def _curvature_from_yaw(yaw: np.ndarray, ds: float):
    """Approximate curvature kappa ~= d(yaw)/ds."""
    if yaw.size < 3:
        return np.zeros((yaw.size,), dtype=float)
    dyaw = _wrap_angle_rad(yaw[1:] - yaw[:-1])
    k = np.concatenate((dyaw[:1], 0.5 * (dyaw[1:] + dyaw[:-1]), dyaw[-1:]))
    return k / max(float(ds), 1e-6)


def _speed_profile_from_curvature(
    curvature: np.ndarray,
    ds: float,
    v_max: float,
    a_lat_max: float,
    a_long_accel_max: float,
    a_long_brake_max: float,
    v_min: float = 0.5,
):
    """Compute a curvature-limited speed profile and enforce accel/brake limits."""
    k = np.abs(np.asarray(curvature, dtype=float))
    eps = 1e-6
    v_curve = np.sqrt(np.maximum(a_lat_max, eps) / np.maximum(k, eps))
    v = np.clip(v_curve, float(v_min), float(v_max))

    # enforce stop at the end
    if v.size:
        v[-1] = 0.0

    # forward pass: accel limit
    for i in range(1, v.size):
        v_allowed = math.sqrt(max(0.0, v[i - 1] ** 2 + 2.0 * a_long_accel_max * ds))
        if v[i] > v_allowed:
            v[i] = v_allowed

    # backward pass: brake limit (a_long_brake_max is positive magnitude)
    for i in range(v.size - 2, -1, -1):
        v_allowed = math.sqrt(max(0.0, v[i + 1] ** 2 + 2.0 * a_long_brake_max * ds))
        if v[i] > v_allowed:
            v[i] = v_allowed

    # light smoothing to avoid sharp transitions
    if v.size >= 5:
        kernel = np.array([1, 2, 3, 2, 1], dtype=float)
        kernel /= kernel.sum()
        v_pad = np.pad(v, (2, 2), mode="edge")
        v = np.convolve(v_pad, kernel, mode="valid")
        v[-1] = 0.0

    return v


# =============================================================================
# Pure Pursuit Controller
# =============================================================================


class PurePursuitController:
    """
    Pure pursuit path tracking controller.
    """

    def __init__(self, lookahead_distance=5.0, wheelbase=2.5):
        self.L_d = lookahead_distance
        self.L = wheelbase
        self.last_i = 0

    def find_lookahead_point(self, car_x, car_y, path):
        """
        Find the lookahead point on the path.

        Returns the first point on the path that is at least L_d away from the car.
        """
        for i, (px, py) in enumerate(path):
            dist = np.sqrt((px - car_x) ** 2 + (py - car_y) ** 2)
            if (dist >= self.L_d) and (i >= self.last_i):
                self.last_i = i
                return px, py, i

        # If no point is far enough, return the last point
        return path[-1][0], path[-1][1], len(path) - 1

    def compute_steering(self, car_x, car_y, car_theta, path):
        """
        Compute steering angle using pure pursuit algorithm.

        Args:
            car_x, car_y: car position
            car_theta: car heading
            path: list of (x, y) waypoints

        Returns:
            delta: steering angle in radians
            lookahead_point: (x, y) of the lookahead point
            lookahead_idx: index of the lookahead point
        """
        if len(path) == 0:
            return 0.0, (car_x, car_y), 0

        goal_x, goal_y, idx = self.find_lookahead_point(car_x, car_y, path)

        # Calculate angle to goal point
        alpha = np.arctan2(goal_y - car_y, goal_x - car_x) - car_theta

        # Pure pursuit steering law
        # delta = atan2(2 * L * sin(alpha) / L_d, 1)
        delta = np.arctan2(2 * self.L * np.sin(alpha), self.L_d)

        # Limit steering angle (typical car has ~35 degree max steering)
        max_steer = np.radians(35)
        delta = np.clip(delta, -max_steer, max_steer)

        return delta, (goal_x, goal_y), idx


# =============================================================================
# MPPI Controller (full-state: steering + accel)
# =============================================================================


class MPPIController:
    def __init__(
        self,
        wheelbase: float,
        dt: float,
        horizon_steps: int = 30,
        rollouts: int = 400,
        lambda_: float = 1.0,
        sigma_delta: float = np.radians(6.0),
        sigma_accel: float = 1.0,
        max_steer_rad: float = np.radians(35.0),
        accel_min: float = -3.0,
        accel_max: float = 2.0,
        w_pos: float = 2.0,
        w_yaw: float = 0.5,
        w_speed: float = 0.5,
        w_u: float = 0.01,
        w_du: float = 0.1,
        w_terminal: float = 5.0,
        nearest_window: int = 250,
    ):
        self.L = float(wheelbase)
        self.dt = float(dt)
        self.T = int(horizon_steps)
        self.K = int(rollouts)
        self.lambda_ = float(lambda_)
        self.sigma = np.array([float(sigma_delta), float(sigma_accel)], dtype=float)
        self.max_steer = float(max_steer_rad)
        self.accel_min = float(accel_min)
        self.accel_max = float(accel_max)
        self.w_pos = float(w_pos)
        self.w_yaw = float(w_yaw)
        self.w_speed = float(w_speed)
        self.w_u = float(w_u)
        self.w_du = float(w_du)
        self.w_terminal = float(w_terminal)
        self.nearest_window = int(nearest_window)

        self.U = np.zeros((self.T, 2), dtype=float)
        self.last_u = np.zeros((2,), dtype=float)

        # RNG: deterministic-ish if user sets numpy seed externally
        self._rng = np.random.default_rng()

    def find_nearest_index(
        self, x: float, y: float, path_xy: np.ndarray, last_i: int = 0
    ):
        n = int(path_xy.shape[0])
        if n == 0:
            return 0
        i0 = int(np.clip(last_i - 10, 0, n - 1))
        i1 = int(np.clip(last_i + self.nearest_window, 0, n - 1))
        window = path_xy[i0 : i1 + 1]
        dx = window[:, 0] - float(x)
        dy = window[:, 1] - float(y)
        j = int(np.argmin(dx * dx + dy * dy))
        return i0 + j

    def _clip_u(self, U: np.ndarray):
        U[:, 0] = np.clip(U[:, 0], -self.max_steer, self.max_steer)
        U[:, 1] = np.clip(U[:, 1], self.accel_min, self.accel_max)
        return U

    def step(
        self,
        state: np.ndarray,
        nearest_i: int,
        path_xy: np.ndarray,
        path_yaw: np.ndarray,
        path_speed: np.ndarray,
        ds: float,
        lookahead_m: float = 8.0,
    ):
        """Return (delta, accel, lookahead_xy, new_nearest_i)."""

        if path_xy.shape[0] < 2:
            return 0.0, 0.0, (float(state[0]), float(state[1])), nearest_i

        i0 = int(np.clip(nearest_i, 0, path_xy.shape[0] - 1))
        v_guess = float(max(path_speed[i0], 1.0))
        stride = max(1, int(round((v_guess * self.dt) / max(float(ds), 1e-6))))
        idx_offsets = np.minimum(
            i0 + stride * np.arange(self.T, dtype=int), path_xy.shape[0] - 1
        )

        # Sample noise for rollouts
        eps = self._rng.normal(0.0, 1.0, size=(self.K, self.T, 2)) * self.sigma
        U_k = self.U[None, :, :] + eps
        U_k[:, :, 0] = np.clip(U_k[:, :, 0], -self.max_steer, self.max_steer)
        U_k[:, :, 1] = np.clip(U_k[:, :, 1], self.accel_min, self.accel_max)

        # Rollout cost accumulation
        x = np.full((self.K,), float(state[0]), dtype=float)
        y = np.full((self.K,), float(state[1]), dtype=float)
        th = np.full((self.K,), float(state[2]), dtype=float)
        v = np.full((self.K,), float(state[3]), dtype=float)

        J = np.zeros((self.K,), dtype=float)
        prev_delta = np.full((self.K,), float(self.last_u[0]), dtype=float)
        prev_a = np.full((self.K,), float(self.last_u[1]), dtype=float)

        dt = self.dt
        L = self.L

        for t in range(self.T):
            idx = idx_offsets[t]
            rx = float(path_xy[idx, 0])
            ry = float(path_xy[idx, 1])
            rth = float(path_yaw[idx])
            rv = float(path_speed[idx])

            delta = U_k[:, t, 0]
            a = U_k[:, t, 1]

            # costs (tracking, speed, effort, smoothness)
            ex = x - rx
            ey = y - ry
            epos2 = ex * ex + ey * ey
            eth = _wrap_angle_rad(th - rth)
            ev = v - rv

            J += self.w_pos * epos2
            J += self.w_yaw * (eth * eth)
            J += self.w_speed * (ev * ev)
            J += self.w_u * (delta * delta + 0.25 * (a * a))
            J += self.w_du * ((delta - prev_delta) ** 2 + 0.5 * (a - prev_a) ** 2)

            prev_delta = delta
            prev_a = a

            # dynamics
            x += v * np.cos(th) * dt
            y += v * np.sin(th) * dt
            th += v * np.tan(delta) / L * dt
            v += a * dt
            v = np.maximum(v, 0.0)
            th = _wrap_angle_rad(th)

        # terminal cost
        idxT = idx_offsets[-1]
        tx = float(path_xy[idxT, 0])
        ty = float(path_xy[idxT, 1])
        J += self.w_terminal * ((x - tx) ** 2 + (y - ty) ** 2)

        # MPPI weights and control update
        J_min = float(np.min(J))
        w = np.exp(-(J - J_min) / max(self.lambda_, 1e-9))
        w_sum = float(np.sum(w))
        if not np.isfinite(w_sum) or w_sum <= 1e-12:
            # fallback: keep current sequence
            u0 = self.U[0].copy()
        else:
            w /= w_sum
            # Standard MPPI update: U <- U + sum_k w_k * eps_k
            dU = np.sum(w[:, None, None] * eps, axis=0)
            self.U = self._clip_u(self.U + dU)
            u0 = self.U[0].copy()

        # warm start for next step
        self.U[:-1] = self.U[1:]
        self.U[-1] = self.U[-2]
        self.last_u = u0.copy()

        look_steps = int(round(float(lookahead_m) / max(float(ds), 1e-6)))
        look_i = int(min(i0 + max(1, look_steps), int(path_xy.shape[0]) - 1))
        lookahead = (float(path_xy[look_i, 0]), float(path_xy[look_i, 1]))
        return float(u0[0]), float(u0[1]), lookahead, i0


# =============================================================================
# Map Loading and Route Extraction
# =============================================================================


def _auto_select_route(
    routing_graph, lanelet_map, min_steps: int = 30, max_steps: int = 400
):
    lanelets = list(lanelet_map.laneletLayer)
    if not lanelets:
        raise RuntimeError("Map contains no lanelets")

    # Try to build a reasonably long route by walking "following" edges.
    for start in lanelets[: min(len(lanelets), 500)]:
        cur = start
        steps = 0
        while steps < max_steps:
            followers = routing_graph.following(cur)
            if not followers:
                break
            cur = followers[0]
            steps += 1
        if steps >= min_steps and cur.id != start.id:
            route = routing_graph.getRoute(start, cur)
            if route is not None:
                return start, cur

    # Fallback: try a small brute-force search over a subset.
    starts = lanelets[: min(len(lanelets), 200)]
    ends = lanelets[-min(len(lanelets), 200) :]
    for start in starts:
        for end in ends:
            if start.id == end.id:
                continue
            route = routing_graph.getRoute(start, end)
            if route is not None:
                return start, end

    raise RuntimeError(
        "Could not auto-select a routable start/end lanelet pair. "
        "Set ~start_lanelet_id and ~end_lanelet_id explicitly."
    )


def load_map_and_route(start_lanelet_id=None, end_lanelet_id=None):
    """
    Load the Lanelet2 example map and create a route.

    Returns:
        lanelet_map: loaded Lanelet2 map
        path_waypoints: list of (x, y) centerline waypoints for the route
        lane_boundaries: dict with 'left' and 'right' boundary points for the route
        route_lanelets: list of lanelets in the route
    """
    # Find the example map file
    script_dir = os.path.dirname(os.path.abspath(__file__))
    example_file = os.path.join(script_dir, "lanelet2_maps/res/boston-seaport.osm")

    if not os.path.exists(example_file):
        # Try alternative locations
        alt_paths = [
            "lanelet2_maps/res/boston-seaport.osm",
            "lanelet2_maps/res/mapping_example.osm",
            "../lanelet2_maps/res/boston-seaport.osm",
        ]
        for alt in alt_paths:
            if os.path.exists(alt):
                example_file = alt
                break

    if not os.path.exists(example_file):
        raise FileNotFoundError(f"Could not find map file at {example_file}")

    def infer_origin_lat_lon(osm_path: str):
        # Use the first node lat/lon as a reasonable projection origin.
        with open(osm_path, "r", encoding="utf-8") as f:
            for _ in range(300):
                line = f.readline()
                if not line:
                    break
                m = re.search(r"\blat='([^']+)'\s+lon='([^']+)'", line)
                if m:
                    return float(m.group(1)), float(m.group(2))
                m = re.search(r'\blat="([^"]+)"\s+lon="([^"]+)"', line)
                if m:
                    return float(m.group(1)), float(m.group(2))
        raise RuntimeError(f"Could not infer origin lat/lon from {osm_path}")

    origin_lat, origin_lon = infer_origin_lat_lon(example_file)

    # Load map with UTM projection and tolerate non-critical parsing issues.
    projector = UtmProjector(lanelet2.io.Origin(origin_lat, origin_lon))
    lanelet_map, load_errors = lanelet2.io.loadRobust(example_file, projector)
    if load_errors:
        rospy.logwarn(
            "Map loaded with %d issues (showing none). Use lanelet2_validate for details.",
            len(load_errors),
        )

    # Create traffic rules for German vehicle
    traffic_rules = lanelet2.traffic_rules.create(
        lanelet2.traffic_rules.Locations.Germany,
        lanelet2.traffic_rules.Participants.Vehicle,
    )

    # Create routing graph
    routing_graph = lanelet2.routing.RoutingGraph(lanelet_map, traffic_rules)

    # Select start/end lanelets.
    start_lanelet = None
    end_lanelet = None

    if start_lanelet_id is not None and end_lanelet_id is not None:
        start_lanelet_id = int(start_lanelet_id)
        end_lanelet_id = int(end_lanelet_id)
        if not lanelet_map.laneletLayer.exists(start_lanelet_id):
            raise RuntimeError(f"start_lanelet_id {start_lanelet_id} not found in map")
        if not lanelet_map.laneletLayer.exists(end_lanelet_id):
            raise RuntimeError(f"end_lanelet_id {end_lanelet_id} not found in map")
        start_lanelet = lanelet_map.laneletLayer[start_lanelet_id]
        end_lanelet = lanelet_map.laneletLayer[end_lanelet_id]
    else:
        start_lanelet, end_lanelet = _auto_select_route(routing_graph, lanelet_map)
        rospy.loginfo(
            "Auto-selected start/end lanelets: %d -> %d (override with ~start_lanelet_id/~end_lanelet_id)",
            start_lanelet.id,
            end_lanelet.id,
        )

    rospy.loginfo(
        "Creating route from lanelet %d to %d", start_lanelet.id, end_lanelet.id
    )

    # Get route
    route = routing_graph.getRoute(start_lanelet, end_lanelet)
    if route is None:
        raise RuntimeError("Could not find route between lanelets")

    # Get shortest path
    shortest_path = route.shortestPath()

    rospy.loginfo("Route has %d lanelets", len(shortest_path))

    # Extract centerline waypoints from the path
    path_waypoints = []
    for lanelet in shortest_path:
        centerline = lanelet.centerline
        for point in centerline:
            path_waypoints.append((point.x, point.y))

    # Remove duplicate consecutive points
    cleaned_waypoints = [path_waypoints[0]]
    for wp in path_waypoints[1:]:
        if (
            np.sqrt(
                (wp[0] - cleaned_waypoints[-1][0]) ** 2
                + (wp[1] - cleaned_waypoints[-1][1]) ** 2
            )
            > 0.1
        ):
            cleaned_waypoints.append(wp)

    rospy.loginfo("Path has %d waypoints", len(cleaned_waypoints))

    # Extract lane boundaries for visualization
    left_boundary = []
    right_boundary = []
    for lanelet in shortest_path:
        for point in lanelet.leftBound:
            left_boundary.append((point.x, point.y))
        for point in lanelet.rightBound:
            right_boundary.append((point.x, point.y))

    lane_boundaries = {"left": left_boundary, "right": right_boundary}

    return lanelet_map, cleaned_waypoints, lane_boundaries, list(shortest_path)


# =============================================================================
# Simulation
# =============================================================================


def _yaw_to_quaternion(yaw_rad: float):
    half = 0.5 * yaw_rad
    return (0.0, 0.0, math.sin(half), math.cos(half))


def _make_color(r, g, b, a=1.0):
    c = ColorRGBA()
    c.r = float(r)
    c.g = float(g)
    c.b = float(b)
    c.a = float(a)
    return c


def _marker_line_strip(
    frame_id: str,
    ns: str,
    mid: int,
    pts_xy,
    color: ColorRGBA,
    width_m: float,
    stamp,
    z: float = 0.0,
):
    m = Marker()
    m.header.frame_id = frame_id
    m.header.stamp = stamp
    m.ns = ns
    m.id = int(mid)
    m.type = Marker.LINE_STRIP
    m.action = Marker.ADD
    m.pose.orientation.w = 1.0
    m.scale.x = float(width_m)
    m.color = color
    m.lifetime = rospy.Duration(0)
    for x, y in pts_xy:
        p = Point()
        m.points.append(p)
        p.x = float(x)
        p.y = float(y)
        p.z = float(z)
    return m


def _marker_line_list_from_polylines(
    frame_id: str,
    ns: str,
    mid: int,
    polylines,
    color: ColorRGBA,
    width_m: float,
    stamp,
    z: float = 0.0,
):
    m = Marker()
    m.header.frame_id = frame_id
    m.header.stamp = stamp
    m.ns = ns
    m.id = int(mid)
    m.type = Marker.LINE_LIST
    m.action = Marker.ADD
    m.pose.orientation.w = 1.0
    m.scale.x = float(width_m)
    m.color = color
    m.lifetime = rospy.Duration(0)

    # LINE_LIST expects point pairs.
    for pts_xy in polylines:
        if len(pts_xy) < 2:
            continue
        for (x0, y0), (x1, y1) in zip(pts_xy[:-1], pts_xy[1:]):
            p0 = Point()
            p0.x = float(x0)
            p0.y = float(y0)
            p0.z = float(z)
            m.points.append(p0)

            p1 = Point()
            p1.x = float(x1)
            p1.y = float(y1)
            p1.z = float(z)
            m.points.append(p1)

    return m


def _marker_sphere(
    frame_id: str,
    ns: str,
    mid: int,
    x: float,
    y: float,
    color: ColorRGBA,
    scale_m: float,
    stamp,
    z: float = 0.0,
):
    m = Marker()
    m.header.frame_id = frame_id
    m.header.stamp = stamp
    m.ns = ns
    m.id = int(mid)
    m.type = Marker.SPHERE
    m.action = Marker.ADD
    m.pose.position.x = float(x)
    m.pose.position.y = float(y)
    m.pose.position.z = float(z)
    m.pose.orientation.w = 1.0
    m.scale.x = float(scale_m)
    m.scale.y = float(scale_m)
    m.scale.z = float(scale_m)
    m.color = color
    m.lifetime = rospy.Duration(0)
    return m


def _extract_map_geometry(lanelet_map):
    left_bounds = []
    right_bounds = []
    centerlines = []
    for ll in lanelet_map.laneletLayer:
        left_bounds.append([(p.x, p.y) for p in ll.leftBound])
        right_bounds.append([(p.x, p.y) for p in ll.rightBound])
        centerlines.append([(p.x, p.y) for p in ll.centerline])
    return left_bounds, right_bounds, centerlines


def _lanelet_center_xy(lanelet):
    cl = list(lanelet.centerline)
    if cl:
        p = cl[len(cl) // 2]
        return float(p.x), float(p.y)
    lb = list(lanelet.leftBound)
    rb = list(lanelet.rightBound)
    if lb and rb:
        return 0.5 * (float(lb[0].x) + float(rb[0].x)), 0.5 * (
            float(lb[0].y) + float(rb[0].y)
        )
    raise RuntimeError(f"Lanelet {lanelet.id} has no geometry")


def _marker_text(
    frame_id: str,
    ns: str,
    mid: int,
    x: float,
    y: float,
    text: str,
    color: ColorRGBA,
    height_m: float,
    stamp,
    z: float = 0.0,
):
    m = Marker()
    m.header.frame_id = frame_id
    m.header.stamp = stamp
    m.ns = ns
    m.id = int(mid)
    m.type = Marker.TEXT_VIEW_FACING
    m.action = Marker.ADD
    m.pose.position.x = float(x)
    m.pose.position.y = float(y)
    m.pose.position.z = float(z)
    m.pose.orientation.w = 1.0
    m.scale.z = float(height_m)
    m.color = color
    m.text = str(text)
    m.lifetime = rospy.Duration(0)
    return m


def _build_static_markers(
    lanelet_map,
    path_waypoints,
    lane_boundaries,
    frame_id: str,
    stamp,
    label_lanelet_ids: bool = False,
    lanelet_id_bbox_margin_m: float = 50.0,
    lanelet_id_max_labels: int = 500,
    lanelet_id_text_height_m: float = 0.9,
):
    markers = MarkerArray()
    map_left, map_right, map_center = _extract_map_geometry(lanelet_map)

    # full map (muted) for context
    map_center_color = _make_color(0.50, 0.50, 0.50, 0.35)

    markers.markers.append(
        _marker_line_list_from_polylines(
            frame_id,
            "map_lane_bounds",
            0,
            map_left + map_right,
            _make_color(0.65, 0.65, 0.65, 0.55),
            0.06,
            stamp,
            z=0.0,
        )
    )
    markers.markers.append(
        _marker_line_list_from_polylines(
            frame_id,
            "map_centerlines",
            0,
            map_center,
            map_center_color,
            0.04,
            stamp,
            z=0.0,
        )
    )

    # route overlay (keep existing colors so it's easy to see what is followed)
    left = lane_boundaries.get("left", [])
    right = lane_boundaries.get("right", [])

    if left:
        markers.markers.append(
            _marker_line_strip(
                frame_id,
                "lane_bounds",
                0,
                left,
                _make_color(0.10, 0.35, 1.0, 0.95),
                0.25,
                stamp,
                z=0.05,
            )
        )
    if right:
        markers.markers.append(
            _marker_line_strip(
                frame_id,
                "lane_bounds",
                1,
                right,
                _make_color(1.0, 0.25, 0.25, 0.95),
                0.25,
                stamp,
                z=0.05,
            )
        )
    if path_waypoints:
        markers.markers.append(
            _marker_line_strip(
                frame_id,
                "route",
                0,
                path_waypoints,
                _make_color(0.20, 0.85, 0.35, 0.80),
                0.15,
                stamp,
                z=0.06,
            )
        )

        # start / goal points
        sx, sy = path_waypoints[0]
        gx, gy = path_waypoints[-1]
        markers.markers.append(
            _marker_sphere(
                frame_id,
                "route_points",
                0,
                sx,
                sy,
                _make_color(0.15, 0.95, 0.25, 0.95),
                1.0,
                stamp,
                z=0.08,
            )
        )
        markers.markers.append(
            _marker_sphere(
                frame_id,
                "route_points",
                1,
                gx,
                gy,
                _make_color(0.95, 0.15, 0.15, 0.95),
                1.0,
                stamp,
                z=0.08,
            )
        )

    if label_lanelet_ids and path_waypoints:
        xs = [p[0] for p in path_waypoints]
        ys = [p[1] for p in path_waypoints]
        xmin = min(xs) - float(lanelet_id_bbox_margin_m)
        xmax = max(xs) + float(lanelet_id_bbox_margin_m)
        ymin = min(ys) - float(lanelet_id_bbox_margin_m)
        ymax = max(ys) + float(lanelet_id_bbox_margin_m)

        color = _make_color(0.05, 0.05, 0.05, 0.85)
        count = 0
        for ll in lanelet_map.laneletLayer:
            if count >= int(lanelet_id_max_labels):
                break
            cx, cy = _lanelet_center_xy(ll)
            if cx < xmin or cx > xmax or cy < ymin or cy > ymax:
                continue
            markers.markers.append(
                _marker_text(
                    frame_id,
                    "lanelet_ids",
                    count,
                    cx,
                    cy,
                    str(ll.id),
                    color,
                    float(lanelet_id_text_height_m),
                    stamp,
                    z=0.20,
                )
            )
            count += 1

        if count >= int(lanelet_id_max_labels):
            rospy.logwarn(
                "Lanelet ID labels capped at %d; increase ~lanelet_id_max_labels or shrink bbox margin",
                int(lanelet_id_max_labels),
            )
    return markers


def _build_dynamic_markers(car: "BicycleModel", lookahead_xy, frame_id: str, stamp):
    arr = MarkerArray()

    # vehicle footprint
    veh = Marker()
    veh.header.frame_id = frame_id
    veh.header.stamp = stamp
    veh.ns = "vehicle"
    veh.id = 0
    veh.type = Marker.CUBE
    veh.action = Marker.ADD
    veh.pose.position.x = float(car.x)
    veh.pose.position.y = float(car.y)
    veh.pose.position.z = 0.10
    qx, qy, qz, qw = _yaw_to_quaternion(car.theta)
    veh.pose.orientation.x = qx
    veh.pose.orientation.y = qy
    veh.pose.orientation.z = qz
    veh.pose.orientation.w = qw
    veh.scale.x = float(car.length)
    veh.scale.y = float(car.width)
    veh.scale.z = 0.8
    veh.color = _make_color(1.0, 0.60, 0.05, 0.95)
    veh.lifetime = rospy.Duration(0)
    arr.markers.append(veh)

    # lookahead point
    la = Marker()
    la.header.frame_id = frame_id
    la.header.stamp = stamp
    la.ns = "lookahead"
    la.id = 0
    la.type = Marker.SPHERE
    la.action = Marker.ADD
    la.pose.position.x = float(lookahead_xy[0])
    la.pose.position.y = float(lookahead_xy[1])
    la.pose.position.z = 0.12
    la.pose.orientation.w = 1.0
    la.scale.x = 0.6
    la.scale.y = 0.6
    la.scale.z = 0.6
    la.color = _make_color(0.95, 0.15, 0.95, 0.95)
    la.lifetime = rospy.Duration(0)
    arr.markers.append(la)

    return arr


def _make_path_msg(path_waypoints, frame_id: str, stamp):
    msg = Path()
    msg.header.frame_id = frame_id
    msg.header.stamp = stamp
    for x, y in path_waypoints:
        ps = PoseStamped()
        ps.header.frame_id = frame_id
        ps.header.stamp = stamp
        ps.pose.position.x = float(x)
        ps.pose.position.y = float(y)
        ps.pose.position.z = 0.0
        ps.pose.orientation.w = 1.0
        msg.poses.append(ps)
    return msg


def run_simulation():
    """Main simulation function (publishes ROS topics)."""

    rospy.init_node("lanelet2_car_simulation", anonymous=False)

    frame_id = rospy.get_param("~frame_id", "map")
    base_frame_id = rospy.get_param("~base_frame_id", "base_link")
    rate_hz = float(rospy.get_param("~rate_hz", 20.0))
    dt = 1.0 / max(rate_hz, 1.0)

    lookahead_distance = float(rospy.get_param("~lookahead_distance", 8.0))
    target_speed = float(rospy.get_param("~target_speed", 5.0))
    speed_kp = float(rospy.get_param("~speed_kp", 1.0))

    # Path + MPPI params
    path_ds = float(rospy.get_param("~path_ds", 1.0))
    a_lat_max = float(rospy.get_param("~a_lat_max", 2.5))
    accel_min = float(rospy.get_param("~accel_min", -3.0))
    accel_max = float(rospy.get_param("~accel_max", 2.0))

    mppi_rollouts = int(rospy.get_param("~mppi_rollouts", 400))
    mppi_horizon_steps = int(rospy.get_param("~mppi_horizon_steps", 30))
    mppi_lambda = float(rospy.get_param("~mppi_lambda", 1.0))
    mppi_sigma_delta_deg = float(rospy.get_param("~mppi_sigma_delta_deg", 6.0))
    mppi_sigma_accel = float(rospy.get_param("~mppi_sigma_accel", 1.0))

    w_pos = float(rospy.get_param("~mppi_w_pos", 2.0))
    w_yaw = float(rospy.get_param("~mppi_w_yaw", 0.5))
    w_speed = float(rospy.get_param("~mppi_w_speed", 0.5))
    w_u = float(rospy.get_param("~mppi_w_u", 0.01))
    w_du = float(rospy.get_param("~mppi_w_du", 0.1))
    w_terminal = float(rospy.get_param("~mppi_w_terminal", 5.0))

    show_lanelet_ids = bool(rospy.get_param("~show_lanelet_ids", True))
    lanelet_id_bbox_margin_m = float(rospy.get_param("~lanelet_id_bbox_margin_m", 50.0))
    lanelet_id_max_labels = int(rospy.get_param("~lanelet_id_max_labels", 500))
    lanelet_id_text_height_m = float(rospy.get_param("~lanelet_id_text_height_m", 0.9))

    pub_pose = rospy.Publisher("/sim/vehicle_pose", PoseStamped, queue_size=10)
    pub_twist = rospy.Publisher("/sim/vehicle_twist", TwistStamped, queue_size=10)
    pub_lookahead = rospy.Publisher("/sim/lookahead", PointStamped, queue_size=10)
    pub_route = rospy.Publisher("/sim/route", Path, queue_size=1, latch=True)
    pub_static_markers = rospy.Publisher(
        "/sim/map_markers", MarkerArray, queue_size=1, latch=True
    )
    pub_dyn_markers = rospy.Publisher(
        "/sim/vehicle_markers", MarkerArray, queue_size=10
    )

    tf_broadcaster = tf2_ros.TransformBroadcaster()

    start_lanelet_id = 42968  # rospy.get_param("~start_lanelet_id", None)
    end_lanelet_id = 38564  # rospy.get_param("~end_lanelet_id", None)

    rospy.loginfo("Loading map and creating route...")
    lanelet_map, path_waypoints, lane_boundaries, _route_lanelets = load_map_and_route(
        start_lanelet_id=start_lanelet_id, end_lanelet_id=end_lanelet_id
    )
    if len(path_waypoints) < 2:
        raise RuntimeError("Route path too short")

    # Resample path to equal spacing for stable curvature + MPPI reference
    path_xy_rs, _s_rs = _resample_polyline_equal_spacing(path_waypoints, path_ds)
    if path_xy_rs.shape[0] < 2:
        raise RuntimeError("Resampled route path too short")

    path_yaw = _path_yaw_from_xy(path_xy_rs)
    curvature = _curvature_from_yaw(path_yaw, path_ds)
    path_speed = _speed_profile_from_curvature(
        curvature=curvature,
        ds=path_ds,
        v_max=target_speed,
        a_lat_max=a_lat_max,
        a_long_accel_max=float(max(0.1, accel_max)),
        a_long_brake_max=float(max(0.1, -accel_min)),
        v_min=0.5,
    )

    path_waypoints_rs = [(float(p[0]), float(p[1])) for p in path_xy_rs]

    # publish static route + map markers once (latched)
    stamp0 = rospy.Time.now()
    pub_route.publish(_make_path_msg(path_waypoints_rs, frame_id, stamp0))
    pub_static_markers.publish(
        _build_static_markers(
            lanelet_map,
            path_waypoints_rs,
            lane_boundaries,
            frame_id,
            stamp0,
            label_lanelet_ids=show_lanelet_ids,
            lanelet_id_bbox_margin_m=lanelet_id_bbox_margin_m,
            lanelet_id_max_labels=lanelet_id_max_labels,
            lanelet_id_text_height_m=lanelet_id_text_height_m,
        )
    )

    # Initialize car at the start of the path
    start_x, start_y = path_waypoints_rs[0]
    dx = path_waypoints_rs[1][0] - path_waypoints_rs[0][0]
    dy = path_waypoints_rs[1][1] - path_waypoints_rs[0][1]
    start_theta = float(np.arctan2(dy, dx))

    car = BicycleModel(x=start_x, y=start_y, theta=start_theta, v=0.0)
    controller = MPPIController(
        wheelbase=car.L,
        dt=dt,
        horizon_steps=mppi_horizon_steps,
        rollouts=mppi_rollouts,
        lambda_=mppi_lambda,
        sigma_delta=np.radians(mppi_sigma_delta_deg),
        sigma_accel=mppi_sigma_accel,
        max_steer_rad=np.radians(35.0),
        accel_min=accel_min,
        accel_max=accel_max,
        w_pos=w_pos,
        w_yaw=w_yaw,
        w_speed=w_speed,
        w_u=w_u,
        w_du=w_du,
        w_terminal=w_terminal,
    )

    nearest_i = 0

    rospy.loginfo("Starting simulation loop...")
    rate = rospy.Rate(rate_hz)

    goal_x, goal_y = path_waypoints_rs[-1]

    while not rospy.is_shutdown():
        # goal check
        dist_to_goal = float(np.hypot(car.x - goal_x, car.y - goal_y))
        nearest_i = controller.find_nearest_index(
            car.x, car.y, path_xy_rs, last_i=nearest_i
        )
        reached = (dist_to_goal < 2.0) or (nearest_i >= int(path_xy_rs.shape[0]) - 3)

        if reached:
            car.v = 0.0
            lookahead = (car.x, car.y)
        else:
            delta, accel, lookahead, nearest_i = controller.step(
                car.state,
                nearest_i,
                path_xy_rs,
                path_yaw,
                path_speed,
                path_ds,
                lookahead_m=lookahead_distance,
            )
            car.update(dt, float(delta), float(accel))

        stamp = rospy.Time.now()

        # pose
        pose = PoseStamped()
        pose.header.frame_id = frame_id
        pose.header.stamp = stamp
        pose.pose.position.x = float(car.x)
        pose.pose.position.y = float(car.y)
        pose.pose.position.z = 0.0
        qx, qy, qz, qw = _yaw_to_quaternion(car.theta)
        pose.pose.orientation.x = qx
        pose.pose.orientation.y = qy
        pose.pose.orientation.z = qz
        pose.pose.orientation.w = qw
        pub_pose.publish(pose)

        # twist (in base_link)
        tw = TwistStamped()
        tw.header.frame_id = base_frame_id
        tw.header.stamp = stamp
        tw.twist.linear.x = float(car.v)
        pub_twist.publish(tw)

        # lookahead point
        lp = PointStamped()
        lp.header.frame_id = frame_id
        lp.header.stamp = stamp
        lp.point.x = float(lookahead[0])
        lp.point.y = float(lookahead[1])
        lp.point.z = 0.0
        pub_lookahead.publish(lp)

        # vehicle + lookahead markers
        pub_dyn_markers.publish(_build_dynamic_markers(car, lookahead, frame_id, stamp))

        # TF map -> base_link
        t = TransformStamped()
        t.header.frame_id = frame_id
        t.header.stamp = stamp
        t.child_frame_id = base_frame_id
        t.transform.translation.x = float(car.x)
        t.transform.translation.y = float(car.y)
        t.transform.translation.z = 0.0
        t.transform.rotation.x = qx
        t.transform.rotation.y = qy
        t.transform.rotation.z = qz
        t.transform.rotation.w = qw
        tf_broadcaster.sendTransform(t)

        if reached:
            rospy.loginfo_throttle(2.0, "Goal reached; holding position.")

        rate.sleep()


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    run_simulation()
