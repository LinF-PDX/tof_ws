import math
import os
import time
from dataclasses import dataclass
from typing import Dict, Tuple, Optional

import numpy as np

import rclpy
from rclpy._rclpy_pybind11 import RCLError
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point

from sensor_msgs_py import point_cloud2


class UnionFind:
    def __init__(self, n: int):
        self.parent = np.arange(n, dtype=np.int32)
        self.size = np.ones(n, dtype=np.int32)

    def find(self, a: int) -> int:
        while self.parent[a] != a:
            self.parent[a] = self.parent[self.parent[a]]
            a = self.parent[a]
        return a

    def union(self, a: int, b: int):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.size[ra] < self.size[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        self.size[ra] += self.size[rb]


@dataclass
class CellResult:
    center_xyz: np.ndarray  # (3,)
    normal: np.ndarray      # (3,)
    confidence: float       # 0..1


def _pc2_to_xyz_numpy(msg: PointCloud2) -> np.ndarray:
    if msg.width * msg.height == 0:
        return np.zeros((0, 3), dtype=np.float32)

    offsets = {}
    for f in msg.fields:
        if f.name in ("x", "y", "z"):
            offsets[f.name] = f.offset
    if not all(k in offsets for k in ("x", "y", "z")):
        raise RuntimeError(f"PointCloud2 missing x/y/z fields. Got: {[f.name for f in msg.fields]}")

    n_pts = msg.width * msg.height
    step = msg.point_step

    buf = np.frombuffer(msg.data, dtype=np.uint8)
    n_pts = min(n_pts, buf.size // step)
    buf = buf[: n_pts * step].reshape(n_pts, step)

    def decode(dtype_f4: np.dtype) -> np.ndarray:
        x = buf[:, offsets["x"]:offsets["x"] + 4].copy().view(dtype_f4).reshape(-1)
        y = buf[:, offsets["y"]:offsets["y"] + 4].copy().view(dtype_f4).reshape(-1)
        z = buf[:, offsets["z"]:offsets["z"] + 4].copy().view(dtype_f4).reshape(-1)
        pts = np.stack([x, y, z], axis=1).astype(np.float32, copy=False)
        return pts

    # Decode both ways (do NOT trust msg.is_bigendian)
    pts_le = decode(np.dtype("<f4"))
    pts_be = decode(np.dtype(">f4"))

    def score(pts: np.ndarray) -> float:
        finite = np.isfinite(pts).all(axis=1)
        pts = pts[finite]
        if pts.shape[0] == 0:
            return 0.0
        # "reasonable" check: most ToF points should be within tens of meters, not 1e38.
        reasonable = (np.abs(pts) < 100.0).all(axis=1)
        return float(np.mean(reasonable))

    # Use a small sample to choose endian fast
    M = min(5000, n_pts)
    s_le = score(pts_le[:M])
    s_be = score(pts_be[:M])

    pts = pts_le if s_le >= s_be else pts_be

    # Drop non-finite and absurd
    pts = pts[np.isfinite(pts).all(axis=1)]
    pts = pts[(np.abs(pts) < 1e3).all(axis=1)]
    return pts




def _make_pc2(header: Header, data: np.ndarray) -> PointCloud2:
    """
    data: Nx7 array => x,y,z,nx,ny,nz,confidence
    """
    fields = [
        PointField(name="x", offset=0,  datatype=PointField.FLOAT32, count=1),
        PointField(name="y", offset=4,  datatype=PointField.FLOAT32, count=1),
        PointField(name="z", offset=8,  datatype=PointField.FLOAT32, count=1),
        PointField(name="nx", offset=12, datatype=PointField.FLOAT32, count=1),
        PointField(name="ny", offset=16, datatype=PointField.FLOAT32, count=1),
        PointField(name="nz", offset=20, datatype=PointField.FLOAT32, count=1),
        PointField(name="confidence", offset=24, datatype=PointField.FLOAT32, count=1),
    ]
    # point_cloud2.create_cloud expects iterable of tuples/lists
    pts = [tuple(row.tolist()) for row in data.astype(np.float32)]
    return point_cloud2.create_cloud(header, fields, pts)


def _fit_plane_pca_robust(points: np.ndarray,
                          outlier_dist: float,
                          refit: bool = True) -> Optional[Tuple[np.ndarray, np.ndarray, float]]:
    """
    Fit plane to points via PCA, optionally robustify by rejecting outliers and refitting once.
    Returns (centroid, normal_unit, inlier_ratio) or None if degenerate.
    Plane: normal · (x - centroid) = 0
    """
    if points.shape[0] < 3:
        return None

    def pca_fit(pts: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        c = pts.mean(axis=0)
        X = pts - c
        # covariance
        C = (X.T @ X) / max(1, (pts.shape[0] - 1))
        w, v = np.linalg.eigh(C)  # ascending eigenvalues
        n = v[:, 0]  # smallest eigenvalue direction
        n_norm = np.linalg.norm(n)
        if n_norm < 1e-9:
            return c, np.array([0.0, 0.0, 1.0], dtype=np.float32)
        n = n / n_norm
        return c, n.astype(np.float32)

    c0, n0 = pca_fit(points)

    # distance to plane
    d0 = np.abs((points - c0) @ n0)

    inliers = d0 <= outlier_dist
    inlier_ratio = float(np.count_nonzero(inliers)) / float(points.shape[0])

    if refit and np.count_nonzero(inliers) >= 3 and inlier_ratio < 0.999:
        pts2 = points[inliers]
        c1, n1 = pca_fit(pts2)
        # recompute inlier ratio against refit plane
        d1 = np.abs((points - c1) @ n1)
        inliers2 = d1 <= outlier_dist
        inlier_ratio = float(np.count_nonzero(inliers2)) / float(points.shape[0])
        return c1, n1, inlier_ratio

    return c0, n0, inlier_ratio


def make_normal_marker_array(header: Header, out: np.ndarray, length: float,
                             ns: str, rgba: Tuple[float, float, float, float]) -> MarkerArray:
    """
    out: Nx7 array [x,y,z,nx,ny,nz,confidence]
    """
    ma = MarkerArray()
    r, g, b, a = rgba
    for i in range(out.shape[0]):
        x, y, z, nx, ny, nz, conf = out[i].tolist()

        m = Marker()
        m.header = header
        m.ns = ns
        m.id = int(i)
        m.type = Marker.ARROW
        m.action = Marker.ADD

        p0 = Point(x=float(x), y=float(y), z=float(z))
        p1 = Point(x=float(x + length * nx), y=float(y + length * ny), z=float(z + length * nz))
        m.points = [p0, p1]

        # Arrow size: shaft diameter, head diameter, head length
        m.scale.x = 0.01
        m.scale.y = 0.02
        m.scale.z = 0.03

        m.color.r = float(r)
        m.color.g = float(g)
        m.color.b = float(b)
        m.color.a = float(a)

        ma.markers.append(m)

    return ma


class GridGroundNode(Node):
    def __init__(self):
        super().__init__("tof_grid_ground")
        self.pub_rawdebug = self.create_publisher(PointCloud2, "/debug/raw_points_xyz", 10)

        # Topics (from your bag)
        self.declare_parameter("points_topic", "/camera/depth/points")
        self.declare_parameter("ground_topic", "/ground_grid_cells")
        self.declare_parameter("nonground_topic", "/nonground_grid_cells")

        # Grid + fitting parameters
        self.declare_parameter("cell_size", 0.07)         # meters
        self.declare_parameter("min_points_per_cell", 30)
        self.declare_parameter("outlier_dist", 0.03)      # meters, plane distance threshold
        self.declare_parameter("max_range", 6.0)          # meters; drop far points
        self.declare_parameter("min_range", 0.10)         # meters; drop too-close junk
        self.declare_parameter("up_axis", "x")            # one of x,y,z in cloud frame
        self.declare_parameter("prefer_normal_positive_z", True)
        self.declare_parameter("d_max", 0.15)
        self.declare_parameter("use_8_neighbors", True)
        self.declare_parameter("ground_components_keep", 1)  # keep K largest connected components

        # If you want a bounded ROI (recommended for speed)
        self.declare_parameter("x_min", -2.0)
        self.declare_parameter("x_max",  2.0)
        self.declare_parameter("y_min", -2.0)
        self.declare_parameter("y_max",  2.0)
        self.declare_parameter("z_min", -2.0)
        self.declare_parameter("z_max",  2.0)

        self.points_topic = self.get_parameter("points_topic").get_parameter_value().string_value
        self.ground_topic = self.get_parameter("ground_topic").value
        self.nonground_topic = self.get_parameter("nonground_topic").value

        self.sub = self.create_subscription(PointCloud2, self.points_topic, self.cb_points, 10)
        self.pub_ground = self.create_publisher(PointCloud2, self.ground_topic, 10)
        self.pub_nonground = self.create_publisher(PointCloud2, self.nonground_topic, 10)

        # Normal marker publishing
        self.declare_parameter("publish_normals_markers", True)
        self.declare_parameter("normal_length", 0.10)   # arrow length in meters
        self.declare_parameter("ground_normals_topic", "/ground_normals")
        self.declare_parameter("nonground_normals_topic", "/nonground_normals")
        self.declare_parameter("timing_histogram_enabled", True)
        self.declare_parameter("timing_histogram_window", 200)
        self.declare_parameter("timing_histogram_report_every", 30)
        self.declare_parameter("timing_log_each_step", False)
        self.declare_parameter("timing_save_txt", True)
        self.declare_parameter("timing_txt_path", "/tmp/tof_step_time_ms.txt")

        self.pub_ground_normals = self.create_publisher(
            MarkerArray, self.get_parameter("ground_normals_topic").value, 10
        )
        self.pub_nonground_normals = self.create_publisher(
            MarkerArray, self.get_parameter("nonground_normals_topic").value, 10
        )
        self.timing_histogram_enabled = bool(self.get_parameter("timing_histogram_enabled").value)
        self.timing_histogram_window = max(1, int(self.get_parameter("timing_histogram_window").value))
        self.timing_histogram_report_every = max(1, int(self.get_parameter("timing_histogram_report_every").value))
        self.timing_log_each_step = bool(self.get_parameter("timing_log_each_step").value)
        self.timing_save_txt = bool(self.get_parameter("timing_save_txt").value)
        self.timing_txt_path = str(self.get_parameter("timing_txt_path").value)
        self._step_times_ms = []
        self._timing_samples_since_report = 0
        self._timing_bins_ms = np.asarray([0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 33.3, 50.0, 100.0, 200.0, 1e9], dtype=np.float64)
        self._timing_txt_file = None
        if self.timing_save_txt and self.timing_txt_path:
            txt_dir = os.path.dirname(self.timing_txt_path)
            if txt_dir:
                os.makedirs(txt_dir, exist_ok=True)
            self._timing_txt_file = open(self.timing_txt_path, "a", encoding="utf-8")

        self.get_logger().info(f"Listening: {self.points_topic}")
        self.get_logger().info(f"Publishing ground grid cells: {self.ground_topic}")
        self.get_logger().info(f"Publishing non-ground grid cells: {self.nonground_topic}")
        self.get_logger().info(
            f"Timing histogram enabled={self.timing_histogram_enabled} "
            f"window={self.timing_histogram_window} report_every={self.timing_histogram_report_every}"
        )
        self.get_logger().info(f"Timing per-step sample logging enabled={self.timing_log_each_step}")
        self.get_logger().info(f"Timing txt enabled={self.timing_save_txt} path={self.timing_txt_path}")

    def _record_step_time(self, dt_ms: float):
        if self.timing_log_each_step:
            self.get_logger().info(f"step_time_ms_sample={dt_ms:.3f}")
        if self._timing_txt_file is not None:
            self._timing_txt_file.write(f"{dt_ms:.6f}\n")
            self._timing_txt_file.flush()
        if not self.timing_histogram_enabled:
            return
        self._step_times_ms.append(float(dt_ms))
        if len(self._step_times_ms) > self.timing_histogram_window:
            del self._step_times_ms[:-self.timing_histogram_window]

        self._timing_samples_since_report += 1
        if self._timing_samples_since_report < self.timing_histogram_report_every:
            return
        self._timing_samples_since_report = 0

        arr = np.asarray(self._step_times_ms, dtype=np.float64)
        if arr.size == 0:
            return

        counts, bins = np.histogram(arr, bins=self._timing_bins_ms)
        hist_parts = []
        for i, c in enumerate(counts):
            left = bins[i]
            right = bins[i + 1]
            if right >= 1e8:
                label = f"[{left:.1f},inf)"
            else:
                label = f"[{left:.1f},{right:.1f})"
            hist_parts.append(f"{label}:{int(c)}")

        mean_ms = float(np.mean(arr))
        p50_ms = float(np.percentile(arr, 50))
        p90_ms = float(np.percentile(arr, 90))
        p99_ms = float(np.percentile(arr, 99))
        max_ms = float(np.max(arr))
        fps = (1000.0 / mean_ms) if mean_ms > 1e-9 else 0.0
        self.get_logger().info(
            "step_time_ms histogram "
            f"(rolling={arr.size}): {' | '.join(hist_parts)} "
            f"mean={mean_ms:.2f} p50={p50_ms:.2f} p90={p90_ms:.2f} p99={p99_ms:.2f} "
            f"max={max_ms:.2f} approx_fps={fps:.1f}"
        )

    def cb_points(self, msg: PointCloud2):
        t0 = time.perf_counter()
        try:
            self.get_logger().info("cb_points fired", throttle_duration_sec=1.0)

            pts = _pc2_to_xyz_numpy(msg)
            self.get_logger().info(f"parsed pts={pts.shape[0]}", throttle_duration_sec=1.0)
            # ---- DEBUG: basic sanity ----
            self.get_logger().info(
                f"cb fired: width={msg.width} height={msg.height} step={msg.point_step} bigendian={msg.is_bigendian}",
                throttle_duration_sec=1.0
            )

            if pts.shape[0] > 0:
                mn = np.min(pts, axis=0)
                mx = np.max(pts, axis=0)
                self.get_logger().info(
                    f"parsed pts={pts.shape[0]}  min(x,y,z)={mn.tolist()}  max(x,y,z)={mx.tolist()}",
                    throttle_duration_sec=1.0
                )


            if pts.shape[0] == 0:
                return
        
            hdr = Header()
            hdr.stamp = msg.header.stamp
            hdr.frame_id = msg.header.frame_id
            # publish a downsampled raw cloud (first N points) so RViz won't choke
            N = min(20000, pts.shape[0])
            debug_pc = point_cloud2.create_cloud_xyz32(hdr, pts[:N].tolist())
            self.pub_rawdebug.publish(debug_pc)

            self.get_logger().info(f"raw pts={pts.shape[0]}", throttle_duration_sec=1.0)


            # --- NEW: drop non-finite values early (handles inf/-inf) ---
            n_raw = pts.shape[0]
            # --- gate 1: finite ---
            finite_mask = np.isfinite(pts).all(axis=1)
            pts_finite = pts[finite_mask]
            n_finite = pts_finite.shape[0]

            # --- gate 2: sane magnitude ---
            ABS_MAX = 1e3
            sane_mask = (np.abs(pts_finite) < ABS_MAX).all(axis=1)
            pts_sane = pts_finite[sane_mask]
            n_sane = pts_sane.shape[0]

            # --- gate 3: range + ROI on grid-plane axes ---
            axis_idx = {"x": 0, "y": 1, "z": 2}
            up_axis = str(self.get_parameter("up_axis").value).strip().lower()
            if up_axis not in axis_idx:
                self.get_logger().warn(
                    f"Invalid up_axis='{up_axis}', falling back to 'x'. Valid: x|y|z",
                    throttle_duration_sec=2.0
                )
                up_axis = "x"
            up_i = axis_idx[up_axis]
            plane_axes = [i for i in (0, 1, 2) if i != up_i]
            a0, a1 = plane_axes[0], plane_axes[1]
            axis_name = {0: "x", 1: "y", 2: "z"}

            mins = np.array([
                float(self.get_parameter("x_min").value),
                float(self.get_parameter("y_min").value),
                float(self.get_parameter("z_min").value),
            ], dtype=np.float64)
            maxs = np.array([
                float(self.get_parameter("x_max").value),
                float(self.get_parameter("y_max").value),
                float(self.get_parameter("z_max").value),
            ], dtype=np.float64)
            min_range = float(self.get_parameter("min_range").value)
            max_range = float(self.get_parameter("max_range").value)

            pts64 = pts_sane.astype(np.float64, copy=False)
            r2 = np.sum(pts64 * pts64, axis=1)
            mask_r = (r2 >= min_range**2) & (r2 <= max_range**2)

            mask_roi = (
                (pts_sane[:, a0] >= mins[a0]) & (pts_sane[:, a0] <= maxs[a0]) &
                (pts_sane[:, a1] >= mins[a1]) & (pts_sane[:, a1] <= maxs[a1])
            )

            pts = pts_sane[mask_r & mask_roi]
            n_final = pts.shape[0]

            self.get_logger().info(
                f"gates: raw={n_raw} finite={n_finite} sane={n_sane} after(range+roi)={n_final}",
                throttle_duration_sec=1.0
            )

            if pts.shape[0] == 0:
                self.get_logger().warn("No points left after filtering (range/ROI likely wrong).", throttle_duration_sec=1.0)
                return


            cell = float(self.get_parameter("cell_size").value)
            min_pts = int(self.get_parameter("min_points_per_cell").value)
            outlier_dist = float(self.get_parameter("outlier_dist").value)
            prefer_pos_z = bool(self.get_parameter("prefer_normal_positive_z").value)

            self.get_logger().info(
                f"after filter pts={pts.shape[0]}  "
                f"x[{pts[:,0].min():.3f},{pts[:,0].max():.3f}] "
                f"y[{pts[:,1].min():.3f},{pts[:,1].max():.3f}] "
                f"z[{pts[:,2].min():.3f},{pts[:,2].max():.3f}]",
                throttle_duration_sec=1.0
            )
            # compute test indices using the current cell size
            ix_test = np.floor((pts[:, a0] - mins[a0]) / cell).astype(np.int32)
            iy_test = np.floor((pts[:, a1] - mins[a1]) / cell).astype(np.int32)
            self.get_logger().info(
                f"grid plane axes=({axis_name[a0]},{axis_name[a1]}) up={up_axis} "
                f"index ranges: ix[{ix_test.min()},{ix_test.max()}] iy[{iy_test.min()},{iy_test.max()}]",
                throttle_duration_sec=1.0
            )

            # Grid indexing on the plane orthogonal to up_axis
            ix = np.floor((pts[:, a0] - mins[a0]) / cell).astype(np.int32)
            iy = np.floor((pts[:, a1] - mins[a1]) / cell).astype(np.int32)

            # ---- NEW: robust, fast grid grouping using numpy ----
            self.get_logger().info(
                f"about to grid: pts={pts.shape} ix={ix.shape} iy={iy.shape} "
                f"ix_sample={ix[:5].tolist()} iy_sample={iy[:5].tolist()}",
                throttle_duration_sec=1.0
            )
            keys = np.stack([ix, iy], axis=1)  # (N,2)
            uniq, inv = np.unique(keys, axis=0, return_inverse=True)

            # build cell->indices mapping
            cells = {}
            for i in range(uniq.shape[0]):
                cells[(int(uniq[i, 0]), int(uniq[i, 1]))] = np.where(inv == i)[0]

            cell_sizes = np.fromiter((len(v) for v in cells.values()), dtype=np.int32)
            self.get_logger().info(
                f"grid: occupied_cells={uniq.shape[0]} "
                f"median_pts={int(np.median(cell_sizes))} max_pts={int(np.max(cell_sizes))}",
                throttle_duration_sec=1.0
            )

            results = []
            coords = []

            fit_ok = 0
            fit_fail = 0
            nx_reject = 0
            total_inlier = 0.0
            cells_considered = 0

            for (cx, cy), idxs in cells.items():

                cells_considered += 1

                if len(idxs) < min_pts:
                    continue

                cell_pts = pts[idxs, :]
                fit = _fit_plane_pca_robust(cell_pts, outlier_dist=outlier_dist, refit=True)
                if fit is None:
                    fit_fail += 1
                    continue
                centroid, normal, inlier_ratio = fit
                fit_ok += 1
                total_inlier += inlier_ratio

                # Orient normals consistently with the chosen up axis.
                if prefer_pos_z:
                    if normal[up_i] > 0:
                        normal = -normal
                else:
                    if normal[up_i] < 0:
                        normal = -normal

                p0_c = float(mins[a0] + (cx + 0.5) * cell)
                p1_c = float(mins[a1] + (cy + 0.5) * cell)

                # Solve for coordinate along up-axis at the cell center on plane axes.
                n_up = float(normal[up_i])
                n0 = float(normal[a0])
                n1 = float(normal[a1])
                if abs(n_up) < 1e-6:
                    nx_reject += 1
                    # Near-vertical plane wrt up-axis -> likely not ground.
                    continue

                up_c = float(
                    centroid[up_i] -
                    (n0 * (p0_c - centroid[a0]) + n1 * (p1_c - centroid[a1])) / n_up
                )

                center = np.zeros(3, dtype=np.float32)
                center[up_i] = up_c
                center[a0] = p0_c
                center[a1] = p1_c

                confidence = float(np.clip(inlier_ratio, 0.0, 1.0))
                results.append([
                    float(center[0]), float(center[1]), float(center[2]),
                    float(normal[0]), float(normal[1]), float(normal[2]),
                    confidence
                ])
                coords.append([cx, cy])

            avg_inlier = (total_inlier / fit_ok) if fit_ok > 0 else 0.0
            self.get_logger().info(
                f"plane: considered={cells_considered} fit_ok={fit_ok} fit_fail={fit_fail} "
                f"nx_reject={nx_reject} avg_inlier={avg_inlier:.3f} outputs={len(results)}",
                throttle_duration_sec=1.0
            )

            if not results:
                return

            out = np.asarray(results, dtype=np.float32)

            header = Header()
            header.stamp = msg.header.stamp
            header.frame_id = msg.header.frame_id  # same frame as input cloud

            d_max = float(self.get_parameter("d_max").value)
            use_8 = bool(self.get_parameter("use_8_neighbors").value)
            coords = np.asarray(coords, dtype=np.int32)

            coord_to_idx = {(int(coords[i, 0]), int(coords[i, 1])): i for i in range(coords.shape[0])}

            neigh = [(-1, 0), (1, 0), (0, -1), (0, 1)]
            if use_8:
                neigh += [(-1, -1), (-1, 1), (1, -1), (1, 1)]

            uf = UnionFind(out.shape[0])
            centers = out[:, 0:3].astype(np.float64, copy=False)

            for i in range(coords.shape[0]):
                cx, cy = int(coords[i, 0]), int(coords[i, 1])
                for dx, dy in neigh:
                    j = coord_to_idx.get((cx + dx, cy + dy), None)
                    if j is None:
                        continue
                    if np.linalg.norm(centers[i] - centers[j]) < d_max:
                        uf.union(i, j)

            roots = np.array([uf.find(i) for i in range(out.shape[0])], dtype=np.int32)
            unique_roots, counts = np.unique(roots, return_counts=True)
            keep_k = max(1, int(self.get_parameter("ground_components_keep").value))
            order = np.argsort(-counts)  # descending by size
            keep_roots = unique_roots[order[:keep_k]]
            is_ground = np.isin(roots, keep_roots)

            out_ground = out[is_ground]
            out_nonground = out[~is_ground]

            if out_ground.shape[0] > 0:
                self.pub_ground.publish(_make_pc2(header, out_ground))
            if out_nonground.shape[0] > 0:
                self.pub_nonground.publish(_make_pc2(header, out_nonground))

            self.get_logger().info(
                f"publish: ground N={out_ground.shape[0]} nonground N={out_nonground.shape[0]}",
                throttle_duration_sec=1.0
            )

            if bool(self.get_parameter("publish_normals_markers").value):
                length = float(self.get_parameter("normal_length").value)
                if out_ground.shape[0] > 0:
                    ma_g = make_normal_marker_array(
                        header, out_ground, length, "ground_normals", (0.0, 1.0, 0.0, 1.0)
                    )
                    self.pub_ground_normals.publish(ma_g)
                if out_nonground.shape[0] > 0:
                    ma_ng = make_normal_marker_array(
                        header, out_nonground, length, "nonground_normals", (1.0, 0.0, 0.0, 1.0)
                    )
                    self.pub_nonground_normals.publish(ma_ng)
        finally:
            dt_ms = (time.perf_counter() - t0) * 1000.0
            self._record_step_time(dt_ms)

    def destroy_node(self):
        if self._timing_txt_file is not None:
            try:
                self._timing_txt_file.close()
            except Exception:
                pass
            self._timing_txt_file = None
        return super().destroy_node()


def main():
    rclpy.init()
    node = GridGroundNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.destroy_node()
        except Exception:
            pass
        try:
            rclpy.shutdown()
        except RCLError:
            # already shut down by signal handler
            pass
