"""从已保存的 LoMa 匹配结果估计并评测双视图相对位姿。"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from .evaluate_v03 import load_heldout_pairs
from .geometry import PairGeometry, load_pair_geometry


POSE_SUMMARY_FIELDS = [
    "pair_index", "pair_id", "matches", "ransac_inliers", "success",
    "rotation_error_deg", "translation_error_deg", "pose_error_deg",
]
POSE_AGGREGATE_FIELDS = [
    "pairs", "successful_pairs", "success_rate", "median_rotation_error_deg",
    "median_translation_error_deg", "median_pose_error_deg", "pose_auc_5deg",
    "pose_auc_10deg", "pose_auc_20deg",
]


@dataclass(frozen=True)
class PoseEstimate:
    """由匹配点估计的 A→B 相对位姿。"""

    success: bool
    rotation: np.ndarray | None
    translation: np.ndarray | None
    ransac_inliers: int


def evaluate_saved_matches(
    image_root: str | Path,
    match_output: str | Path,
    output: str | Path,
    image_size: int = 518,
    max_pairs: int = 9,
    ransac_threshold_px: float = 1.0,
    result_subdir: str | None = None,
) -> Path:
    """离线读取九组保存匹配，估计相对位姿并写入逐组与汇总 CSV。"""
    if image_size <= 0 or max_pairs <= 0 or ransac_threshold_px <= 0.0:
        raise ValueError("图像尺寸、图像对数量和 RANSAC 阈值必须为正数")
    match_root = Path(match_output)
    if not match_root.is_dir():
        raise ValueError("匹配结果路径必须是目录")
    output_root = Path(output)
    if output_root.exists() and not output_root.is_dir():
        raise ValueError("姿态评测输出路径必须是目录")
    output_root.mkdir(parents=True, exist_ok=True)
    pairs = load_heldout_pairs(image_root)[:max_pairs]
    rows: list[dict[str, object]] = []
    for index, pair in enumerate(pairs, start=1):
        result_path = _result_path_for_pair(match_root, index, result_subdir)
        with np.load(result_path, allow_pickle=False) as payload:
            if "keypoints_a" not in payload or "keypoints_b" not in payload:
                raise ValueError(f"匹配结果缺少 keypoints_a/keypoints_b：{result_path}")
            points_a = np.asarray(payload["keypoints_a"], dtype=np.float64)
            points_b = np.asarray(payload["keypoints_b"], dtype=np.float64)
        geometry = load_pair_geometry(pair, image_size)
        estimate = estimate_relative_pose(
            points_a,
            points_b,
            geometry.intrinsics_a,
            geometry.intrinsics_b,
            ransac_threshold_px,
        )
        rows.append(_pose_row(index, pair.pair_id, points_a, estimate, geometry))
    _save_csv(output_root / "pose_summary.csv", rows, POSE_SUMMARY_FIELDS)
    _save_csv(output_root / "pose_aggregate.csv", [_aggregate_pose_rows(rows)], POSE_AGGREGATE_FIELDS)
    return output_root


def estimate_relative_pose(
    points_a: np.ndarray,
    points_b: np.ndarray,
    intrinsics_a: np.ndarray,
    intrinsics_b: np.ndarray,
    ransac_threshold_px: float = 1.0,
) -> PoseEstimate:
    """仅用匹配坐标和内参，以五点 RANSAC 估计 A→B 的 R、t 方向。"""
    source = _points(points_a, "A 图匹配点")
    target = _points(points_b, "B 图匹配点")
    matrix_a = _intrinsics(intrinsics_a, "A 图内参")
    matrix_b = _intrinsics(intrinsics_b, "B 图内参")
    if source.shape != target.shape:
        raise ValueError("两图匹配点数量必须一致")
    if ransac_threshold_px <= 0.0:
        raise ValueError("RANSAC 像素阈值必须为正数")
    if len(source) < 5:
        return PoseEstimate(False, None, None, 0)
    import cv2

    normalized_a = _normalize_points(source, matrix_a)
    normalized_b = _normalize_points(target, matrix_b)
    normalized_threshold = ransac_threshold_px / _mean_focal_length(matrix_a, matrix_b)
    try:
        essential, mask = cv2.findEssentialMat(
            normalized_a,
            normalized_b,
            cameraMatrix=np.eye(3),
            method=cv2.RANSAC,
            prob=0.999,
            threshold=normalized_threshold,
        )
    except cv2.error:
        return PoseEstimate(False, None, None, 0)
    if essential is None or mask is None:
        return PoseEstimate(False, None, None, 0)
    best: PoseEstimate | None = None
    for candidate in _essential_candidates(essential):
        try:
            recovered, rotation, translation, pose_mask = cv2.recoverPose(
                candidate,
                normalized_a,
                normalized_b,
                cameraMatrix=np.eye(3),
                mask=mask.copy(),
            )
        except cv2.error:
            continue
        inliers = int(recovered if pose_mask is None else np.count_nonzero(pose_mask))
        if inliers < 5:
            continue
        estimate = PoseEstimate(True, np.asarray(rotation, dtype=np.float64), np.asarray(translation, dtype=np.float64).reshape(3), inliers)
        if best is None or estimate.ransac_inliers > best.ransac_inliers:
            best = estimate
    return best if best is not None else PoseEstimate(False, None, None, 0)


def rotation_error_degrees(estimate: np.ndarray, ground_truth: np.ndarray) -> float:
    """返回两个旋转矩阵间的测地角误差（度）。"""
    first = _rotation(estimate, "估计旋转")
    second = _rotation(ground_truth, "真值旋转")
    cosine = np.clip((np.trace(first @ second.T) - 1.0) * 0.5, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def translation_error_degrees(estimate: np.ndarray, ground_truth: np.ndarray) -> float:
    """返回平移方向夹角（度），将本质矩阵固有的正负号视为等价。"""
    first = _direction(estimate, "估计平移")
    second = _direction(ground_truth, "真值平移")
    cosine = np.clip(abs(float(first @ second)), -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def pose_auc(errors: np.ndarray | Iterable[float], thresholds: Iterable[float] = (5.0, 10.0, 20.0)) -> dict[float, float]:
    """计算姿态最大角误差的 recall-AUC；失败组应以 180 度传入。"""
    values = np.asarray(list(errors), dtype=np.float64).reshape(-1)
    if not np.isfinite(values).all() or np.any(values < 0.0):
        raise ValueError("姿态误差必须是有限的非负数")
    ordered = np.sort(values)
    recall = np.arange(1, len(ordered) + 1, dtype=np.float64) / max(len(ordered), 1)
    result: dict[float, float] = {}
    for threshold in thresholds:
        if threshold <= 0.0:
            raise ValueError("AUC 阈值必须为正数")
        count = int(np.searchsorted(ordered, threshold, side="right"))
        x = np.concatenate((np.array((0.0,)), ordered[:count], np.array((float(threshold),))))
        y = np.concatenate((np.array((0.0,)), recall[:count], np.array((recall[count - 1] if count else 0.0,))))
        result[float(threshold)] = float(np.trapezoid(y, x) / threshold)
    return result


def relative_pose_ground_truth(geometry: PairGeometry) -> tuple[np.ndarray, np.ndarray]:
    """根据 cam2world 得到 A→B 真值旋转和平移。"""
    target_from_source = np.linalg.inv(geometry.cam2world_b) @ geometry.cam2world_a
    return target_from_source[:3, :3], target_from_source[:3, 3]


def _pose_row(
    index: int,
    pair_id: str,
    points_a: np.ndarray,
    estimate: PoseEstimate,
    geometry: PairGeometry,
) -> dict[str, object]:
    if estimate.success and estimate.rotation is not None and estimate.translation is not None:
        rotation_true, translation_true = relative_pose_ground_truth(geometry)
        rotation_error = rotation_error_degrees(estimate.rotation, rotation_true)
        translation_error = translation_error_degrees(estimate.translation, translation_true)
        pose_error = max(rotation_error, translation_error)
    else:
        rotation_error = translation_error = pose_error = 180.0
    return {
        "pair_index": index,
        "pair_id": pair_id,
        "matches": int(len(points_a)),
        "ransac_inliers": estimate.ransac_inliers,
        "success": estimate.success,
        "rotation_error_deg": rotation_error,
        "translation_error_deg": translation_error,
        "pose_error_deg": pose_error,
    }


def _aggregate_pose_rows(rows: list[dict[str, object]]) -> dict[str, object]:
    if not rows:
        raise ValueError("至少需要一组姿态结果")
    rotation = np.asarray([float(row["rotation_error_deg"]) for row in rows], dtype=np.float64)
    translation = np.asarray([float(row["translation_error_deg"]) for row in rows], dtype=np.float64)
    pose = np.asarray([float(row["pose_error_deg"]) for row in rows], dtype=np.float64)
    auc = pose_auc(pose)
    successful = sum(bool(row["success"]) for row in rows)
    return {
        "pairs": len(rows),
        "successful_pairs": successful,
        "success_rate": successful / len(rows),
        "median_rotation_error_deg": float(np.median(rotation)),
        "median_translation_error_deg": float(np.median(translation)),
        "median_pose_error_deg": float(np.median(pose)),
        "pose_auc_5deg": auc[5.0],
        "pose_auc_10deg": auc[10.0],
        "pose_auc_20deg": auc[20.0],
    }


def _result_path_for_pair(match_root: Path, index: int, result_subdir: str | None = None) -> Path:
    if result_subdir is not None and (Path(result_subdir).is_absolute() or ".." in Path(result_subdir).parts):
        raise ValueError("匹配结果子目录必须是安全的相对路径")
    suffix = "result.npz" if result_subdir is None else f"{result_subdir}/result.npz"
    matches = sorted(match_root.glob(f"{index:02d}_*/{suffix}"))
    if len(matches) != 1:
        raise FileNotFoundError(f"第 {index} 组需要恰好一个 result.npz，实际找到 {len(matches)} 个")
    return matches[0]


def _save_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _normalize_points(points: np.ndarray, intrinsics: np.ndarray) -> np.ndarray:
    homogeneous = np.column_stack((points, np.ones(len(points), dtype=np.float64)))
    normalized = (np.linalg.inv(intrinsics) @ homogeneous.T).T
    return normalized[:, :2] / normalized[:, 2:3]


def _essential_candidates(essential: np.ndarray) -> tuple[np.ndarray, ...]:
    matrix = np.asarray(essential, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] != 3 or matrix.shape[0] % 3 != 0:
        return ()
    return tuple(matrix[start:start + 3] for start in range(0, matrix.shape[0], 3))


def _mean_focal_length(first: np.ndarray, second: np.ndarray) -> float:
    focal = float(np.mean((first[0, 0], first[1, 1], second[0, 0], second[1, 1])))
    if focal <= 0.0:
        raise ValueError("相机焦距必须为正数")
    return focal


def _points(values: np.ndarray, name: str) -> np.ndarray:
    points = np.asarray(values, dtype=np.float64)
    if points.ndim != 2 or points.shape[1:] != (2,) or not np.isfinite(points).all():
        raise ValueError(f"{name}必须是有限的 [N, 2] 数组")
    return points


def _intrinsics(values: np.ndarray, name: str) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise ValueError(f"{name}必须是有限的 3x3 矩阵")
    return matrix


def _rotation(values: np.ndarray, name: str) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise ValueError(f"{name}必须是有限的 3x3 矩阵")
    return matrix


def _direction(values: np.ndarray, name: str) -> np.ndarray:
    vector = np.asarray(values, dtype=np.float64).reshape(-1)
    if vector.shape != (3,) or not np.isfinite(vector).all():
        raise ValueError(f"{name}必须是有限的三维向量")
    norm = float(np.linalg.norm(vector))
    if norm <= 0.0:
        raise ValueError(f"{name}不能为零向量")
    return vector / norm


def main() -> None:
    """解析命令行并执行已保存匹配的相对位姿评测。"""
    parser = argparse.ArgumentParser(description="评测保存的九组 LoMa 匹配的相对位姿")
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--match-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--image-size", type=int, default=518)
    parser.add_argument("--max-pairs", type=int, default=9)
    parser.add_argument("--ransac-threshold-px", type=float, default=1.0)
    parser.add_argument("--result-subdir")
    args = parser.parse_args()
    print(evaluate_saved_matches(**vars(args)))


if __name__ == "__main__":
    main()
