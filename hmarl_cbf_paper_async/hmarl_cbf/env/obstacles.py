from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Iterable, List

import numpy as np

from hmarl_cbf.types import LIDAR_HIT_OBSTACLE

if TYPE_CHECKING:
    from hmarl_cbf.types import LidarScan


Obstacle = Dict[str, Any]


def _vec2(value: Any, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float32).reshape(-1)
    if arr.shape != (2,):
        raise ValueError(f"{name} must have shape (2,), got {arr.shape}")
    return arr


def _rotation_matrix(yaw: float) -> np.ndarray:
    c = float(np.cos(yaw))
    s = float(np.sin(yaw))
    return np.asarray([[c, -s], [s, c]], dtype=np.float32)


def _transform_to_local(point: np.ndarray, center: np.ndarray, yaw: float) -> np.ndarray:
    rot_t = _rotation_matrix(-yaw)
    return (rot_t @ (point - center).reshape(2, 1)).reshape(2)


def _transform_to_world(point_local: np.ndarray, center: np.ndarray, yaw: float) -> np.ndarray:
    rot = _rotation_matrix(yaw)
    return (rot @ point_local.reshape(2, 1)).reshape(2) + center


def _point_segment_distance(point: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    ab = b - a
    denom = float(np.dot(ab, ab))
    if denom <= 1e-10:
        return float(np.linalg.norm(point - a))
    t = float(np.clip(np.dot(point - a, ab) / denom, 0.0, 1.0))
    proj = a + t * ab
    return float(np.linalg.norm(point - proj))


def _segment_segment_distance(a1: np.ndarray, a2: np.ndarray, b1: np.ndarray, b2: np.ndarray) -> float:
    if _segments_intersect(a1, a2, b1, b2):
        return 0.0
    return float(
        min(
            _point_segment_distance(a1, b1, b2),
            _point_segment_distance(a2, b1, b2),
            _point_segment_distance(b1, a1, a2),
            _point_segment_distance(b2, a1, a2),
        )
    )


def _orientation(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    return float((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))


def _segments_intersect(a1: np.ndarray, a2: np.ndarray, b1: np.ndarray, b2: np.ndarray) -> bool:
    o1 = _orientation(a1, a2, b1)
    o2 = _orientation(a1, a2, b2)
    o3 = _orientation(b1, b2, a1)
    o4 = _orientation(b1, b2, a2)
    eps = 1e-8
    if (o1 * o2 < -eps) and (o3 * o4 < -eps):
        return True
    return False


def _point_in_convex_quad(point: np.ndarray, corners: np.ndarray) -> bool:
    signs: List[float] = []
    for i in range(4):
        a = corners[i]
        b = corners[(i + 1) % 4]
        signs.append(_orientation(a, b, point))
    return bool(all(s >= -1e-8 for s in signs) or all(s <= 1e-8 for s in signs))


def normalize_obstacle(item: Dict[str, Any]) -> Obstacle:
    obs_type = str(item.get("type", "circle")).strip().lower()
    center = _vec2(item["center"], "center")
    if obs_type == "lidar_point":
        radius = float(item.get("radius", 0.0))
        if radius < 0.0:
            raise ValueError("lidar_point radius must be nonnegative")
        return {"type": "lidar_point", "center": center, "radius": radius}
    if obs_type == "point":
        radius = float(item.get("radius", 0.0))
        if radius < 0.0:
            raise ValueError("point obstacle radius must be nonnegative")
        return {"type": "point", "center": center, "radius": radius}
    if obs_type == "circle":
        radius = float(item["radius"])
        if radius <= 0.0:
            raise ValueError("circle obstacle radius must be positive")
        return {"type": "circle", "center": center, "radius": radius}
    if obs_type == "lidar_circle":
        radius = float(item["radius"])
        if radius <= 0.0:
            raise ValueError("lidar_circle radius must be positive")
        return {"type": "lidar_circle", "center": center, "radius": radius}
    if obs_type == "lidar_line":
        start = _vec2(item["start"], "start")
        end = _vec2(item["end"], "end")
        return {
            "type": "lidar_line",
            "center": 0.5 * (start + end),
            "start": start,
            "end": end,
        }
    if obs_type == "rect":
        if "half_extents" in item:
            half_extents = _vec2(item["half_extents"], "half_extents")
        elif "size" in item:
            half_extents = 0.5 * _vec2(item["size"], "size")
        else:
            width = float(item["width"])
            height = float(item["height"])
            half_extents = np.asarray([0.5 * width, 0.5 * height], dtype=np.float32)
        if float(np.min(half_extents)) <= 0.0:
            raise ValueError("rect obstacle half_extents must be positive")
        yaw = float(item.get("yaw", 0.0))
        return {
            "type": "rect",
            "center": center,
            "half_extents": half_extents.astype(np.float32),
            "yaw": yaw,
        }
    raise ValueError(f"unsupported obstacle type: {obs_type}")


def normalize_obstacles(items: Iterable[Dict[str, Any]]) -> List[Obstacle]:
    return [normalize_obstacle(dict(item)) for item in items]


def copy_obstacle(item: Dict[str, Any]) -> Obstacle:
    obs = normalize_obstacle(item)
    if obs["type"] == "lidar_point":
        return {"type": "lidar_point", "center": obs["center"].copy(), "radius": float(obs["radius"])}
    if obs["type"] == "point":
        return {"type": "point", "center": obs["center"].copy(), "radius": float(obs["radius"])}
    if obs["type"] == "circle":
        return {"type": "circle", "center": obs["center"].copy(), "radius": float(obs["radius"])}
    if obs["type"] == "lidar_circle":
        return {"type": "lidar_circle", "center": obs["center"].copy(), "radius": float(obs["radius"])}
    if obs["type"] == "lidar_line":
        return {
            "type": "lidar_line",
            "center": np.asarray(obs["center"], dtype=np.float32).reshape(2).copy(),
            "start": np.asarray(obs["start"], dtype=np.float32).reshape(2).copy(),
            "end": np.asarray(obs["end"], dtype=np.float32).reshape(2).copy(),
        }
    return {
        "type": "rect",
        "center": obs["center"].copy(),
        "half_extents": np.asarray(obs["half_extents"], dtype=np.float32).copy(),
        "yaw": float(obs.get("yaw", 0.0)),
    }


def obstacle_center(item: Dict[str, Any]) -> np.ndarray:
    return normalize_obstacle(item)["center"].copy()


def obstacle_bounding_radius(item: Dict[str, Any]) -> float:
    obs = normalize_obstacle(item)
    if obs["type"] in ("point", "circle", "lidar_point", "lidar_circle"):
        return float(obs["radius"])
    if obs["type"] == "lidar_line":
        start = np.asarray(obs["start"], dtype=np.float32).reshape(2)
        end = np.asarray(obs["end"], dtype=np.float32).reshape(2)
        return 0.5 * float(np.linalg.norm(end - start))
    return float(np.linalg.norm(np.asarray(obs["half_extents"], dtype=np.float32).reshape(2)))


def obstacle_corners(item: Dict[str, Any]) -> np.ndarray:
    obs = normalize_obstacle(item)
    if obs["type"] != "rect":
        raise ValueError("only rect obstacles have corners")
    center = np.asarray(obs["center"], dtype=np.float32).reshape(2)
    half_extents = np.asarray(obs["half_extents"], dtype=np.float32).reshape(2)
    yaw = float(obs.get("yaw", 0.0))
    offsets = np.asarray(
        [
            [-half_extents[0], -half_extents[1]],
            [-half_extents[0], half_extents[1]],
            [half_extents[0], half_extents[1]],
            [half_extents[0], -half_extents[1]],
        ],
        dtype=np.float32,
    )
    rot = _rotation_matrix(yaw)
    return center.reshape(1, 2) + (offsets @ rot.T)


def obstacle_surface_distance(point: np.ndarray, item: Dict[str, Any]) -> float:
    obs = normalize_obstacle(item)
    point = _vec2(point, "point")
    center = np.asarray(obs["center"], dtype=np.float32).reshape(2)
    if obs["type"] in ("point", "circle", "lidar_point", "lidar_circle"):
        return float(np.linalg.norm(point - center) - float(obs["radius"]))
    if obs["type"] == "lidar_line":
        start = np.asarray(obs["start"], dtype=np.float32).reshape(2)
        end = np.asarray(obs["end"], dtype=np.float32).reshape(2)
        return float(_point_segment_distance(point, start, end))

    half_extents = np.asarray(obs["half_extents"], dtype=np.float32).reshape(2)
    yaw = float(obs.get("yaw", 0.0))
    local = _transform_to_local(point, center, yaw)
    delta = np.abs(local) - half_extents
    outside = np.maximum(delta, 0.0)
    outside_norm = float(np.linalg.norm(outside))
    inside = float(min(max(float(delta[0]), float(delta[1])), 0.0))
    return outside_norm + inside


def obstacle_contact_geometry(point: np.ndarray, item: Dict[str, Any]) -> Dict[str, Any]:
    obs = normalize_obstacle(item)
    point = _vec2(point, "point")
    center = np.asarray(obs["center"], dtype=np.float32).reshape(2)
    if obs["type"] in ("point", "circle", "lidar_point", "lidar_circle"):
        rel = point - center
        dist = float(np.linalg.norm(rel))
        radius = float(obs["radius"])
        if dist > 1e-8:
            normal = rel / dist
            closest = center + normal * radius
        else:
            normal = np.asarray([1.0, 0.0], dtype=np.float32)
            closest = center + normal * radius
        offset = point - closest
        signed_surface = float(dist - radius)
        return {
            "type": str(obs["type"]),
            "closest_point": closest.astype(np.float32),
            "offset": offset.astype(np.float32),
            "signed_surface": signed_surface,
            "sign": 1.0 if signed_surface >= 0.0 else -1.0,
            "active_mask": np.ones((2,), dtype=np.float32),
        }
    if obs["type"] == "lidar_line":
        start = np.asarray(obs["start"], dtype=np.float32).reshape(2)
        end = np.asarray(obs["end"], dtype=np.float32).reshape(2)
        seg = end - start
        seg_norm_sq = float(np.dot(seg, seg))
        if seg_norm_sq <= 1e-10:
            closest = start
            tangent = np.asarray([1.0, 0.0], dtype=np.float32)
        else:
            tau = float(np.clip(np.dot(point - start, seg) / seg_norm_sq, 0.0, 1.0))
            closest = start + tau * seg
            tangent = seg / max(np.sqrt(seg_norm_sq), 1e-6)
        offset = point - closest
        return {
            "type": "lidar_line",
            "closest_point": closest.astype(np.float32),
            "offset": offset.astype(np.float32),
            "signed_surface": float(np.linalg.norm(offset)),
            "sign": 1.0,
            "active_mask": np.ones((2,), dtype=np.float32),
            "start": start.astype(np.float32),
            "end": end.astype(np.float32),
            "tangent": tangent.astype(np.float32),
        }

    half_extents = np.asarray(obs["half_extents"], dtype=np.float32).reshape(2)
    yaw = float(obs.get("yaw", 0.0))
    local = _transform_to_local(point, center, yaw)
    clamped = np.clip(local, -half_extents, half_extents)
    closest_local = clamped.astype(np.float32)
    closest = _transform_to_world(closest_local, center, yaw)
    offset = point - closest
    outside_mask = (np.abs(local) > (half_extents + 1e-8)).astype(np.float32)
    if float(np.linalg.norm(offset)) > 1e-8:
        signed_surface = float(np.linalg.norm(offset))
        active_mask = outside_mask
        if float(np.sum(active_mask)) <= 0.0:
            active_mask = np.ones((2,), dtype=np.float32)
        return {
            "type": "rect",
            "closest_point": closest.astype(np.float32),
            "closest_point_local": closest_local,
            "offset": offset.astype(np.float32),
            "signed_surface": signed_surface,
            "sign": 1.0,
            "active_mask": active_mask.astype(np.float32),
        }

    dx = float(half_extents[0] - abs(float(local[0])))
    dy = float(half_extents[1] - abs(float(local[1])))
    if dx <= dy:
        axis = 0
        sign_dir = 1.0 if float(local[0]) >= 0.0 else -1.0
        closest_local = np.asarray([sign_dir * float(half_extents[0]), float(local[1])], dtype=np.float32)
        penetration = dx
    else:
        axis = 1
        sign_dir = 1.0 if float(local[1]) >= 0.0 else -1.0
        closest_local = np.asarray([float(local[0]), sign_dir * float(half_extents[1])], dtype=np.float32)
        penetration = dy
    closest = _transform_to_world(closest_local, center, yaw)
    offset = point - closest
    active_mask = np.zeros((2,), dtype=np.float32)
    active_mask[axis] = 1.0
    return {
        "type": "rect",
        "closest_point": closest.astype(np.float32),
        "closest_point_local": closest_local.astype(np.float32),
        "offset": offset.astype(np.float32),
        "signed_surface": -float(max(penetration, 0.0)),
        "sign": -1.0,
        "active_mask": active_mask,
    }


def obstacle_cbf_geometries(
    point: np.ndarray,
    item: Dict[str, Any],
    *,
    rect_dual_edge_enabled: bool = False,
    rect_dual_edge_proximity_distance: float = 0.4,
) -> List[Dict[str, Any]]:
    obs = normalize_obstacle(item)
    if obs["type"] != "rect":
        return [obstacle_contact_geometry(point, obs)]

    point = _vec2(point, "point")
    center = np.asarray(obs["center"], dtype=np.float32).reshape(2)
    half_extents = np.asarray(obs["half_extents"], dtype=np.float32).reshape(2)
    yaw = float(obs.get("yaw", 0.0))
    local = _transform_to_local(point, center, yaw)
    inside = bool(np.all(np.abs(local) <= (half_extents + 1e-8)))

    def _face_geometry(axis: int, face_role: str) -> Dict[str, Any]:
        sign_dir = 1.0 if float(local[axis]) >= 0.0 else -1.0
        other_axis = 1 - axis
        closest_local = local.astype(np.float32).copy()
        closest_local[axis] = sign_dir * float(half_extents[axis])
        closest_local[other_axis] = float(np.clip(local[other_axis], -half_extents[other_axis], half_extents[other_axis]))
        closest = _transform_to_world(closest_local, center, yaw)
        offset = point - closest
        if inside:
            signed_surface = -float(max(half_extents[axis] - abs(float(local[axis])), 0.0))
            sign = -1.0
        else:
            signed_surface = float(np.linalg.norm(offset))
            sign = 1.0
        active_mask = np.zeros((2,), dtype=np.float32)
        active_mask[axis] = 1.0
        return {
            "type": "rect",
            "closest_point": closest.astype(np.float32),
            "closest_point_local": closest_local.astype(np.float32),
            "offset": offset.astype(np.float32),
            "signed_surface": signed_surface,
            "sign": sign,
            "active_mask": active_mask,
            "face_axis": int(axis),
            "face_role": str(face_role),
        }

    geom_x = _face_geometry(0, "candidate")
    geom_y = _face_geometry(1, "candidate")
    dist_x = float(np.linalg.norm(np.asarray(geom_x["offset"], dtype=np.float32).reshape(2)))
    dist_y = float(np.linalg.norm(np.asarray(geom_y["offset"], dtype=np.float32).reshape(2)))
    if dist_x <= dist_y:
        primary_axis, secondary_axis = 0, 1
    else:
        primary_axis, secondary_axis = 1, 0

    primary = geom_x if primary_axis == 0 else geom_y
    primary["face_role"] = "primary"
    geometries: List[Dict[str, Any]] = [primary]

    if not rect_dual_edge_enabled:
        return geometries

    threshold = float(max(1e-6, rect_dual_edge_proximity_distance))
    axis_closeness = np.clip(
        (np.abs(local) - np.maximum(half_extents - threshold, 0.0)) / threshold,
        0.0,
        1.0,
    ).astype(np.float32)
    corner_proximity = float(np.min(axis_closeness))
    if corner_proximity <= 0.0:
        return geometries

    secondary = geom_y if secondary_axis == 1 else geom_x
    secondary["face_role"] = "secondary"
    secondary["corner_proximity"] = corner_proximity
    geometries[0]["corner_proximity"] = corner_proximity
    geometries.append(secondary)
    return geometries


def rect_smooth_barrier_geometry(
    point: np.ndarray,
    item: Dict[str, Any],
    *,
    inflation_margin: float,
    tau: float,
) -> Dict[str, Any]:
    obs = normalize_obstacle(item)
    if obs["type"] != "rect":
        raise ValueError("rect_smooth_barrier_geometry requires a rect obstacle")

    point = _vec2(point, "point")
    center = np.asarray(obs["center"], dtype=np.float32).reshape(2)
    half_extents = np.asarray(obs["half_extents"], dtype=np.float32).reshape(2)
    yaw = float(obs.get("yaw", 0.0))
    tau = float(max(1e-4, tau))
    inflation_margin = float(max(0.0, inflation_margin))
    inflated = (half_extents + inflation_margin).astype(np.float32)

    local = _transform_to_local(point, center, yaw).astype(np.float32)
    face_scores = np.asarray(
        [
            float(local[0] - inflated[0]),
            float(-local[0] - inflated[0]),
            float(local[1] - inflated[1]),
            float(-local[1] - inflated[1]),
        ],
        dtype=np.float32,
    )
    scaled = face_scores / tau
    scaled -= float(np.max(scaled))
    exp_scaled = np.exp(scaled, dtype=np.float32)
    weights = exp_scaled / max(float(np.sum(exp_scaled)), 1e-8)

    face_dirs = np.asarray(
        [
            [1.0, 0.0],
            [-1.0, 0.0],
            [0.0, 1.0],
            [0.0, -1.0],
        ],
        dtype=np.float32,
    )
    grad_local = (weights.reshape(4, 1) * face_dirs).sum(axis=0).astype(np.float32)
    second_local = np.zeros((2, 2), dtype=np.float32)
    for idx in range(4):
        vec = face_dirs[idx].reshape(2, 1)
        second_local += float(weights[idx]) * (vec @ vec.T).astype(np.float32)
    hess_local = ((second_local - np.outer(grad_local, grad_local)) / tau).astype(np.float32)

    rot = _rotation_matrix(yaw)
    grad_world = (rot @ grad_local.reshape(2, 1)).reshape(2).astype(np.float32)
    hess_world = (rot @ hess_local @ rot.T).astype(np.float32)

    barrier_h = float(tau * (np.log(max(float(np.sum(exp_scaled)), 1e-8)) + float(np.max(face_scores / tau))))
    dominant_face = int(np.argmax(face_scores))

    closest_local = local.copy()
    if dominant_face == 0:
        closest_local[0] = inflated[0]
        closest_local[1] = float(np.clip(local[1], -inflated[1], inflated[1]))
    elif dominant_face == 1:
        closest_local[0] = -inflated[0]
        closest_local[1] = float(np.clip(local[1], -inflated[1], inflated[1]))
    elif dominant_face == 2:
        closest_local[1] = inflated[1]
        closest_local[0] = float(np.clip(local[0], -inflated[0], inflated[0]))
    else:
        closest_local[1] = -inflated[1]
        closest_local[0] = float(np.clip(local[0], -inflated[0], inflated[0]))
    closest_world = _transform_to_world(closest_local.astype(np.float32), center, yaw)

    return {
        "type": "rect",
        "barrier_h": barrier_h,
        "barrier_grad": grad_world.astype(np.float32),
        "barrier_hess": hess_world.astype(np.float32),
        "point_local": local.astype(np.float32),
        "effective_half_extents": inflated.astype(np.float32),
        "face_scores": face_scores.astype(np.float32),
        "face_weights": weights.astype(np.float32),
        "dominant_face": dominant_face,
        "closest_point": closest_world.astype(np.float32),
        "closest_point_local": closest_local.astype(np.float32),
    }


def rect_corner_margin_geometry(
    item: Dict[str, Any],
    closest_point_local: np.ndarray,
    threshold: float,
) -> Dict[str, Any]:
    obs = normalize_obstacle(item)
    if obs["type"] != "rect":
        raise ValueError("rect_corner_margin_geometry requires a rect obstacle")
    threshold = float(max(1e-6, threshold))
    center = np.asarray(obs["center"], dtype=np.float32).reshape(2)
    half_extents = np.asarray(obs["half_extents"], dtype=np.float32).reshape(2)
    yaw = float(obs.get("yaw", 0.0))
    local = _vec2(closest_point_local, "closest_point_local")
    signs = np.where(local >= 0.0, 1.0, -1.0).astype(np.float32)
    nearest_corner_local = (signs * half_extents).astype(np.float32)
    nearest_corner = _transform_to_world(nearest_corner_local, center, yaw)
    axis_closeness = np.clip(
        (np.abs(local) - np.maximum(half_extents - threshold, 0.0)) / threshold,
        0.0,
        1.0,
    ).astype(np.float32)
    corner_proximity = float(np.min(axis_closeness))
    return {
        "corner_proximity": corner_proximity,
        "nearest_corner": nearest_corner.astype(np.float32),
        "nearest_corner_local": nearest_corner_local,
        "closest_point_local": local.astype(np.float32),
        "axis_closeness": axis_closeness,
    }


def circle_barrier_geometry(
    point: np.ndarray,
    item: Dict[str, Any],
    *,
    inflation_margin: float,
) -> Dict[str, Any]:
    obs = normalize_obstacle(item)
    if obs["type"] not in ("point", "circle", "lidar_point", "lidar_circle"):
        raise ValueError("circle_barrier_geometry requires a point/circle-type obstacle")
    point = _vec2(point, "point")
    center = np.asarray(obs["center"], dtype=np.float32).reshape(2)
    effective_radius = float(max(0.0, obs.get("radius", 0.0)) + max(0.0, inflation_margin))
    rel = point - center
    barrier_h = float(np.dot(rel, rel) - effective_radius**2)
    return {
        "type": str(obs["type"]),
        "barrier_h": barrier_h,
        "barrier_grad": (2.0 * rel).astype(np.float32),
        "barrier_hess": (2.0 * np.eye(2, dtype=np.float32)).astype(np.float32),
        "effective_radius": effective_radius,
        "center": center.astype(np.float32),
    }


def lidar_line_barrier_geometry(
    point: np.ndarray,
    item: Dict[str, Any],
    *,
    inflation_margin: float,
) -> Dict[str, Any]:
    obs = normalize_obstacle(item)
    if obs["type"] != "lidar_line":
        raise ValueError("lidar_line_barrier_geometry requires a lidar_line obstacle")
    point = _vec2(point, "point")
    start = np.asarray(obs["start"], dtype=np.float32).reshape(2)
    end = np.asarray(obs["end"], dtype=np.float32).reshape(2)
    seg = end - start
    seg_norm_sq = float(np.dot(seg, seg))
    if seg_norm_sq <= 1e-10:
        return circle_barrier_geometry(
            point,
            {"type": "lidar_point", "center": start, "radius": 0.0},
            inflation_margin=inflation_margin,
        )

    tangent = (seg / max(np.sqrt(seg_norm_sq), 1e-6)).astype(np.float32)
    tangent_outer = np.outer(tangent, tangent).astype(np.float32)
    eye = np.eye(2, dtype=np.float32)
    tau = float(np.dot(point - start, seg) / seg_norm_sq)
    tau_clamped = float(np.clip(tau, 0.0, 1.0))
    closest = (start + tau_clamped * seg).astype(np.float32)
    offset = (point - closest).astype(np.float32)
    if 0.0 < tau < 1.0:
        projector = (eye - tangent_outer).astype(np.float32)
        grad = (2.0 * projector @ (point - start)).astype(np.float32)
        hess = (2.0 * projector).astype(np.float32)
    else:
        grad = (2.0 * offset).astype(np.float32)
        hess = (2.0 * eye).astype(np.float32)
    barrier_h = float(np.dot(offset, offset) - max(0.0, inflation_margin) ** 2)
    return {
        "type": "lidar_line",
        "barrier_h": barrier_h,
        "barrier_grad": grad,
        "barrier_hess": hess,
        "closest_point": closest,
        "start": start,
        "end": end,
        "tangent": tangent,
        "projection_tau": tau_clamped,
    }


def obstacle_barrier_geometry(
    point: np.ndarray,
    item: Dict[str, Any],
    *,
    inflation_margin: float,
    tau: float,
) -> Dict[str, Any]:
    obs = normalize_obstacle(item)
    if obs["type"] in ("point", "circle", "lidar_point", "lidar_circle"):
        return circle_barrier_geometry(point, obs, inflation_margin=inflation_margin)
    if obs["type"] == "lidar_line":
        return lidar_line_barrier_geometry(point, obs, inflation_margin=inflation_margin)
    return rect_smooth_barrier_geometry(point, obs, inflation_margin=inflation_margin, tau=tau)


def _wrap_segments(indices: np.ndarray, n_total: int) -> List[np.ndarray]:
    if indices.size == 0:
        return []
    segments: List[List[int]] = [[int(indices[0])]]
    for idx in indices[1:]:
        if int(idx) == segments[-1][-1] + 1:
            segments[-1].append(int(idx))
        else:
            segments.append([int(idx)])
    if len(segments) > 1 and segments[0][0] == 0 and segments[-1][-1] == n_total - 1:
        merged = segments[-1] + segments[0]
        segments = [merged] + segments[1:-1]
    return [np.asarray(seg, dtype=np.int32) for seg in segments]


def _fit_line_segment(points: np.ndarray) -> Dict[str, Any]:
    points = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    centroid = np.mean(points, axis=0).astype(np.float32)
    centered = points - centroid.reshape(1, 2)
    cov = centered.T @ centered / max(points.shape[0], 1)
    eigvals, eigvecs = np.linalg.eigh(cov.astype(np.float32))
    order = np.argsort(eigvals)[::-1]
    tangent = eigvecs[:, order[0]].astype(np.float32)
    tangent = tangent / max(float(np.linalg.norm(tangent)), 1e-6)
    normal = np.asarray([-tangent[1], tangent[0]], dtype=np.float32)
    projections = centered @ tangent.reshape(2, 1)
    proj_min = float(np.min(projections))
    proj_max = float(np.max(projections))
    start = (centroid + proj_min * tangent).astype(np.float32)
    end = (centroid + proj_max * tangent).astype(np.float32)
    residuals = centered @ normal.reshape(2, 1)
    rms_residual = float(np.sqrt(np.mean(np.square(residuals))) if points.shape[0] > 0 else 0.0)
    return {
        "type": "lidar_line",
        "center": (0.5 * (start + end)).astype(np.float32),
        "start": start,
        "end": end,
        "tangent": tangent.astype(np.float32),
        "normal": normal.astype(np.float32),
        "rms_residual": rms_residual,
        "length": float(np.linalg.norm(end - start)),
    }


def _fit_circle_segment(points: np.ndarray) -> Dict[str, Any] | None:
    points = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    if points.shape[0] < 3:
        return None
    x = points[:, 0].astype(np.float64)
    y = points[:, 1].astype(np.float64)
    A = np.stack([x, y, np.ones_like(x)], axis=1)
    b = -(x * x + y * y)
    try:
        sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    except np.linalg.LinAlgError:
        return None
    a, b_coef, c = sol
    center = np.asarray([-0.5 * a, -0.5 * b_coef], dtype=np.float32)
    radius_sq = float(np.dot(center, center) - c)
    if radius_sq <= 1e-10:
        return None
    radius = float(np.sqrt(radius_sq))
    radial = np.linalg.norm(points - center.reshape(1, 2), axis=1)
    rms_residual = float(np.sqrt(np.mean(np.square(radial - radius))))
    return {
        "type": "lidar_circle",
        "center": center,
        "radius": radius,
        "rms_residual": rms_residual,
    }


def fit_lidar_local_obstacles(
    scan: "LidarScan",
    *,
    max_range: float | None = None,
    min_segment_points: int = 2,
    line_fit_max_residual: float = 0.08,
    circle_fit_max_residual: float = 0.08,
    circle_radius_min: float = 0.05,
    circle_radius_max: float = 100.0,
) -> List[Obstacle]:
    ranges = np.asarray(scan.ranges, dtype=np.float32).reshape(-1)
    hit_points = np.asarray(scan.hit_points, dtype=np.float32).reshape(-1, 2)
    hit_valid = np.asarray(scan.hit_valid, dtype=np.bool_).reshape(-1)
    hit_kinds = np.asarray(scan.hit_kinds, dtype=np.int32).reshape(-1)
    n_beam = int(ranges.shape[0])
    perception_range = float(scan.max_range if max_range is None else max_range)
    valid_mask = (
        hit_valid
        & (hit_kinds == int(LIDAR_HIT_OBSTACLE))
        & (ranges <= perception_range + 1e-6)
    )
    indices = np.flatnonzero(valid_mask)
    segments = _wrap_segments(indices, n_total=n_beam)
    fitted: List[Obstacle] = []
    min_segment_points = int(max(1, min_segment_points))
    for seg in segments:
        if int(seg.shape[0]) < min_segment_points:
            continue
        points = hit_points[seg]
        if points.shape[0] == 1:
            fitted.append({"type": "lidar_point", "center": points[0].astype(np.float32), "radius": 0.0})
            continue

        line_model = _fit_line_segment(points)
        circle_model = _fit_circle_segment(points)
        line_residual = float(line_model["rms_residual"])
        circle_residual = float("inf")
        if circle_model is not None:
            circle_radius = float(circle_model["radius"])
            if float(circle_radius_min) <= circle_radius <= float(circle_radius_max):
                circle_residual = float(circle_model["rms_residual"])

        line_ok = line_residual <= float(line_fit_max_residual)
        circle_ok = circle_residual <= float(circle_fit_max_residual)
        if circle_ok and (not line_ok or circle_residual < line_residual):
            fitted.append(copy_obstacle(circle_model))
        else:
            fitted.append(copy_obstacle(line_model))
    return fitted


def extract_lidar_hit_point_obstacles(
    scan: "LidarScan",
    *,
    max_range: float | None = None,
    point_radius: float = 0.0,
    top_k: int | None = None,
) -> List[Obstacle]:
    ranges = np.asarray(scan.ranges, dtype=np.float32).reshape(-1)
    hit_points = np.asarray(scan.hit_points, dtype=np.float32).reshape(-1, 2)
    hit_valid = np.asarray(scan.hit_valid, dtype=np.bool_).reshape(-1)
    hit_kinds = np.asarray(scan.hit_kinds, dtype=np.int32).reshape(-1)
    perception_range = float(scan.max_range if max_range is None else max_range)
    point_radius = float(max(0.0, point_radius))

    valid_mask = (
        hit_valid
        & (hit_kinds == int(LIDAR_HIT_OBSTACLE))
        & (ranges <= perception_range + 1e-6)
    )
    indices = np.flatnonzero(valid_mask)
    if indices.size > 1:
        indices = indices[np.argsort(ranges[indices], kind="stable")]
    if top_k is not None:
        top_k_int = max(0, int(top_k))
        if top_k_int == 0:
            indices = np.zeros((0,), dtype=np.int32)
        elif indices.size > top_k_int:
            indices = indices[:top_k_int]
    return [
        {
            "type": "lidar_point",
            "center": hit_points[int(idx)].astype(np.float32),
            "radius": point_radius,
        }
        for idx in indices
    ]


def extract_lidar_cbf_obstacles(
    scan: "LidarScan",
    *,
    max_range: float | None = None,
    use_fitted_geometry: bool = False,
    point_radius: float = 0.0,
    top_k: int | None = None,
    min_segment_points: int = 2,
    line_fit_max_residual: float = 0.08,
    circle_fit_max_residual: float = 0.08,
    circle_radius_min: float = 0.05,
    circle_radius_max: float = 100.0,
) -> List[Obstacle]:
    perception_range = float(scan.max_range if max_range is None else max_range)
    top_k_int = None if top_k is None else max(0, int(top_k))
    if not bool(use_fitted_geometry):
        return extract_lidar_hit_point_obstacles(
            scan=scan,
            max_range=perception_range,
            point_radius=point_radius,
            top_k=top_k_int,
        )

    fitted = fit_lidar_local_obstacles(
        scan=scan,
        max_range=perception_range,
        min_segment_points=int(max(1, min_segment_points)),
        line_fit_max_residual=float(max(0.0, line_fit_max_residual)),
        circle_fit_max_residual=float(max(0.0, circle_fit_max_residual)),
        circle_radius_min=float(max(0.0, circle_radius_min)),
        circle_radius_max=float(max(circle_radius_min, circle_radius_max)),
    )
    fitted = [copy_obstacle(obs) for obs in fitted]
    if top_k_int is not None:
        if top_k_int == 0:
            return []
        fitted.sort(
            key=lambda obs: float(
                obstacle_surface_distance(
                    np.asarray(scan.origin, dtype=np.float32).reshape(2),
                    obs,
                )
            )
        )
        if len(fitted) > top_k_int:
            fitted = fitted[:top_k_int]
    return fitted


def disk_collides_with_obstacle(point: np.ndarray, disk_radius: float, item: Dict[str, Any]) -> bool:
    return bool(obstacle_surface_distance(point, item) <= float(disk_radius))


def segment_obstacle_clearance(start: np.ndarray, end: np.ndarray, item: Dict[str, Any]) -> float:
    obs = normalize_obstacle(item)
    start = _vec2(start, "start")
    end = _vec2(end, "end")
    center = np.asarray(obs["center"], dtype=np.float32).reshape(2)

    if obs["type"] in ("point", "circle", "lidar_point", "lidar_circle"):
        return float(_point_segment_distance(center, start, end) - float(obs["radius"]))

    if obs["type"] == "lidar_line":
        return float(
            _segment_segment_distance(
                start,
                end,
                np.asarray(obs["start"], dtype=np.float32).reshape(2),
                np.asarray(obs["end"], dtype=np.float32).reshape(2),
            )
        )

    corners = obstacle_corners(obs)
    if _point_in_convex_quad(start, corners) or _point_in_convex_quad(end, corners):
        return -1.0
    for i in range(4):
        c0 = corners[i]
        c1 = corners[(i + 1) % 4]
        if _segments_intersect(start, end, c0, c1):
            return -1.0

    best = float("inf")
    for i in range(4):
        c0 = corners[i]
        c1 = corners[(i + 1) % 4]
        best = min(best, _segment_segment_distance(start, end, c0, c1))
    return float(best)


def disk_swept_collides_with_obstacle(
    start: np.ndarray,
    end: np.ndarray,
    disk_radius: float,
    item: Dict[str, Any],
) -> bool:
    return bool(segment_obstacle_clearance(start, end, item) <= float(disk_radius))


def obstacle_obstacle_clearance(a: Dict[str, Any], b: Dict[str, Any]) -> float:
    obs_a = normalize_obstacle(a)
    obs_b = normalize_obstacle(b)
    if obs_a["type"] == "circle" and obs_b["type"] == "circle":
        return float(
            np.linalg.norm(np.asarray(obs_a["center"]) - np.asarray(obs_b["center"]))
            - float(obs_a["radius"])
            - float(obs_b["radius"])
        )
    if obs_a["type"] == "rect" and obs_b["type"] == "rect":
        corners_a = obstacle_corners(obs_a)
        corners_b = obstacle_corners(obs_b)
        for i in range(4):
            a1 = corners_a[i]
            a2 = corners_a[(i + 1) % 4]
            for j in range(4):
                b1 = corners_b[j]
                b2 = corners_b[(j + 1) % 4]
                if _segments_intersect(a1, a2, b1, b2):
                    return -1.0
        if _point_in_convex_quad(corners_a[0], corners_b) or _point_in_convex_quad(corners_b[0], corners_a):
            return -1.0
        best = float("inf")
        for p in corners_a:
            for j in range(4):
                best = min(best, _point_segment_distance(p, corners_b[j], corners_b[(j + 1) % 4]))
        for p in corners_b:
            for i in range(4):
                best = min(best, _point_segment_distance(p, corners_a[i], corners_a[(i + 1) % 4]))
        return float(best)

    circle = obs_a if obs_a["type"] == "circle" else obs_b
    rect = obs_b if obs_a["type"] == "circle" else obs_a
    dist = obstacle_surface_distance(np.asarray(circle["center"], dtype=np.float32).reshape(2), rect)
    return float(dist - float(circle["radius"]))


def ray_obstacle_distance(
    origin: np.ndarray,
    direction: np.ndarray,
    item: Dict[str, Any],
    max_range: float,
) -> float:
    obs = normalize_obstacle(item)
    origin = _vec2(origin, "origin")
    direction = _vec2(direction, "direction")
    max_range = float(max_range)

    if obs["type"] in ("circle", "point"):
        center = np.asarray(obs["center"], dtype=np.float32).reshape(2)
        radius = float(obs["radius"])
        rel = center - origin
        proj = float(np.dot(rel, direction))
        if proj < 0.0:
            return max_range
        closest_sq = float(np.dot(rel, rel) - proj * proj)
        r_sq = float(radius * radius)
        if closest_sq > r_sq:
            return max_range
        thc = float(np.sqrt(max(r_sq - closest_sq, 0.0)))
        t_hit = proj - thc
        if t_hit < 0.0:
            t_hit = proj + thc
        if t_hit < 0.0 or t_hit > max_range:
            return max_range
        return float(t_hit)

    center = np.asarray(obs["center"], dtype=np.float32).reshape(2)
    half_extents = np.asarray(obs["half_extents"], dtype=np.float32).reshape(2)
    yaw = float(obs.get("yaw", 0.0))
    origin_local = _transform_to_local(origin, center, yaw)
    rot_t = _rotation_matrix(-yaw)
    direction_local = (rot_t @ direction.reshape(2, 1)).reshape(2)
    box_min = -half_extents
    box_max = half_extents

    t_min = -np.inf
    t_max = np.inf
    for axis in range(2):
        if abs(float(direction_local[axis])) <= 1e-8:
            if float(origin_local[axis]) < float(box_min[axis]) or float(origin_local[axis]) > float(box_max[axis]):
                return max_range
            continue
        inv_dir = 1.0 / float(direction_local[axis])
        t1 = (float(box_min[axis]) - float(origin_local[axis])) * inv_dir
        t2 = (float(box_max[axis]) - float(origin_local[axis])) * inv_dir
        t_near = min(t1, t2)
        t_far = max(t1, t2)
        t_min = max(t_min, t_near)
        t_max = min(t_max, t_far)
        if t_min > t_max:
            return max_range

    t_hit = t_min if t_min >= 0.0 else t_max
    if t_hit < 0.0 or t_hit > max_range:
        return max_range
    return float(t_hit)
