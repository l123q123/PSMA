# PASM v0.3 的 DaD 检测结果与静态 Patch 预算选点。

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from .sampling import PatchLayout


@dataclass(frozen=True)
class DaDDetections:
    """保存 DaD 的稠密分数、NMS 亚像素坐标和对应分数。"""

    dense_scores: np.ndarray
    nms_points: np.ndarray
    nms_scores: np.ndarray


class FrozenDaDDetector:
    """从本地权重加载冻结 DaD，并输出稠密分数和全部 NMS 候选。"""

    def __init__(self, weights_path: str | Path, image_size: int, device: str | torch.device) -> None:
        """构建禁止网络下载、输入尺寸固定的 DaD 检测器。"""
        self.weights_path = Path(weights_path)
        if not self.weights_path.is_file():
            raise FileNotFoundError(self.weights_path)
        if image_size <= 0:
            raise ValueError("DaD 输入图像尺寸必须为正数")
        self.image_size = image_size
        self.device = torch.device(device)
        from loma.detector import dad as dad_module

        dad_module.device = self.device
        state = torch.load(self.weights_path, map_location="cpu", weights_only=False)
        with patch("torch.hub.load_state_dict_from_url", return_value=state):
            self.model = dad_module.DaD(
                dad_module.DaD.Cfg(
                    compile=False,
                    resize=image_size,
                    keep_aspect_ratio=False,
                    nms_size=3,
                    increase_coverage=False,
                    subpixel=True,
                )
            )
        self.model.to(self.device).eval()
        for parameter in self.model.parameters():
            parameter.requires_grad = False

    @torch.inference_mode()
    def detect(self, image: np.ndarray | torch.Tensor) -> DaDDetections:
        """对一张固定尺寸图像提取 DaD 稠密概率和全部原生 NMS 峰。"""
        from loma.detector.dad import sample_keypoints

        tensor = _image_tensor(image, self.image_size, self.device)
        scoremap = self.model.forward_impl(tensor)
        batch, channels, height, width = scoremap.shape
        if batch != 1 or channels != 1 or (height, width) != (self.image_size, self.image_size):
            raise ValueError("DaD 输出尺寸与输入图像尺寸不一致")
        dense_scores = (
            scoremap.reshape(batch, channels * height * width)
            .softmax(dim=-1)
            .reshape(batch, channels, height * width)
            .sum(dim=1)
            .reshape(batch, height, width)
        )
        normalized_points, nms_scores = sample_keypoints(
            dense_scores,
            num_samples=height * width,
            use_nms=True,
            nms_size=3,
            sample_topk=True,
            increase_coverage=False,
            return_probs=True,
            subpixel=True,
            scoremap=scoremap[:, 0],
            subpixel_temp=self.model.subpixel_temp,
        )
        keep = nms_scores[0] > 0.0
        points = _normalized_to_pixels(normalized_points[0][keep], self.image_size)
        scores = nms_scores[0][keep].float().cpu().numpy().astype(np.float32)
        return DaDDetections(
            dense_scores=dense_scores[0].float().cpu().numpy().astype(np.float32),
            nms_points=points,
            nms_scores=scores,
        )


def select_patch_budgeted_points(
    detections: DaDDetections, layout: PatchLayout, budgets: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """按每个 Patch 的预算优先选择 DaD NMS 点，再按稠密分数补足。"""
    dense_scores, nms_points, nms_scores = _validated_detections(detections, layout.config.image_size)
    allocation = _validated_patch_budgets(budgets, layout)
    nms_patch_ids = layout.patch_ids(nms_points) if len(nms_points) else np.empty((0,), dtype=np.int64)
    used_cells: set[tuple[int, int]] = set()
    points = []
    patch_ids = []
    scores = []
    for patch_id, budget in enumerate(allocation):
        selected = 0
        candidates = np.flatnonzero(nms_patch_ids == patch_id)
        ordered = candidates[np.argsort(-nms_scores[candidates], kind="stable")]
        for index in ordered:
            cell = _point_cell(nms_points[index], layout.config.image_size)
            if cell in used_cells:
                continue
            points.append(nms_points[index])
            patch_ids.append(patch_id)
            scores.append(nms_scores[index])
            used_cells.add(cell)
            selected += 1
            if selected == budget:
                break
        if selected < budget:
            for x_value, y_value, score in _dense_patch_candidates(dense_scores, layout, patch_id):
                cell = (int(y_value), int(x_value))
                if cell in used_cells:
                    continue
                points.append(np.array((x_value + 0.5, y_value + 0.5), dtype=np.float32))
                patch_ids.append(patch_id)
                scores.append(score)
                used_cells.add(cell)
                selected += 1
                if selected == budget:
                    break
        if selected != budget:
            raise ValueError("DaD 稠密分数无法填满指定 Patch 预算")
    result_points = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    result_ids = np.asarray(patch_ids, dtype=np.int64)
    result_scores = np.asarray(scores, dtype=np.float32)
    _validate_patch_selection(result_points, result_ids, allocation, layout, used_cells)
    return result_points, result_ids, result_scores


def select_global_dad_points(
    detections: DaDDetections, image_size: int, point_count: int
) -> tuple[np.ndarray, np.ndarray]:
    """在全图优先选择 DaD NMS 点，并用最高稠密分数补足固定点数。"""
    dense_scores, nms_points, nms_scores = _validated_detections(detections, image_size)
    if point_count <= 0 or point_count > image_size * image_size:
        raise ValueError("全局 DaD 点数必须位于图像像素数量范围内")
    used_cells: set[tuple[int, int]] = set()
    points = []
    scores = []
    for index in np.argsort(-nms_scores, kind="stable"):
        cell = _point_cell(nms_points[index], image_size)
        if cell in used_cells:
            continue
        points.append(nms_points[index])
        scores.append(nms_scores[index])
        used_cells.add(cell)
        if len(points) == point_count:
            break
    if len(points) < point_count:
        for flat_index in np.argsort(-dense_scores.reshape(-1), kind="stable"):
            y_value, x_value = divmod(int(flat_index), image_size)
            cell = (y_value, x_value)
            if cell in used_cells:
                continue
            points.append(np.array((x_value + 0.5, y_value + 0.5), dtype=np.float32))
            scores.append(dense_scores[y_value, x_value])
            used_cells.add(cell)
            if len(points) == point_count:
                break
    result_points = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    result_scores = np.asarray(scores, dtype=np.float32)
    if len(result_points) != point_count or len(used_cells) != point_count:
        raise ValueError("DaD 稠密分数无法填满全局点预算")
    return result_points, result_scores


def _validated_detections(
    detections: DaDDetections, image_size: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """验证并规范化 DaD 选点所需的稠密分数和 NMS 候选。"""
    dense_scores = np.asarray(detections.dense_scores, dtype=np.float64)
    nms_points = np.asarray(detections.nms_points, dtype=np.float64)
    nms_scores = np.asarray(detections.nms_scores, dtype=np.float64).reshape(-1)
    if image_size <= 0 or dense_scores.shape != (image_size, image_size) or not np.isfinite(dense_scores).all():
        raise ValueError("DaD 稠密分数必须是有限的 [图像高, 图像宽] 数组")
    if nms_points.ndim != 2 or nms_points.shape[1:] != (2,) or len(nms_points) != len(nms_scores):
        raise ValueError("DaD NMS 点和分数形状必须一致")
    if not np.isfinite(nms_points).all() or not np.isfinite(nms_scores).all():
        raise ValueError("DaD NMS 点和分数必须有限")
    if np.any(nms_points < 0.0) or np.any(nms_points >= image_size):
        raise ValueError("DaD NMS 点必须位于图像范围内")
    return dense_scores, nms_points, nms_scores


def _validated_patch_budgets(budgets: np.ndarray, layout: PatchLayout) -> np.ndarray:
    """验证 Patch 预算与 PASM 总点数约束完全一致。"""
    allocation = np.asarray(budgets, dtype=np.int64).reshape(-1)
    if allocation.size != layout.config.patch_count or np.any(allocation < 0):
        raise ValueError("DaD Patch 预算必须是非负的完整 Patch 数组")
    if int(allocation.sum()) != layout.config.total_points:
        raise ValueError("DaD Patch 预算总数必须等于 PASM 总点数")
    return allocation


def _dense_patch_candidates(
    dense_scores: np.ndarray, layout: PatchLayout, patch_id: int
) -> list[tuple[int, int, float]]:
    """按分数降序返回中心属于指定 Patch 的全部稠密像素。"""
    x0, y0, x1, y1 = layout.bounds(patch_id)
    image_size = layout.config.image_size
    x_indices = np.arange(max(0, int(np.ceil(x0 - 0.5))), min(image_size, int(np.ceil(x1 - 0.5))))
    y_indices = np.arange(max(0, int(np.ceil(y0 - 0.5))), min(image_size, int(np.ceil(y1 - 0.5))))
    patch_scores = dense_scores[np.ix_(y_indices, x_indices)]
    result = []
    for local_index in np.argsort(-patch_scores.reshape(-1), kind="stable"):
        local_y, local_x = divmod(int(local_index), len(x_indices))
        result.append((int(x_indices[local_x]), int(y_indices[local_y]), float(patch_scores[local_y, local_x])))
    return result


def _point_cell(point_xy: np.ndarray, image_size: int) -> tuple[int, int]:
    """返回一个图像内浮点坐标所占用的整数像素位置。"""
    x_value, y_value = np.asarray(point_xy, dtype=np.float64)
    x_index = int(np.floor(x_value))
    y_index = int(np.floor(y_value))
    if not 0 <= x_index < image_size or not 0 <= y_index < image_size:
        raise ValueError("DaD 点对应的整数像素超出图像范围")
    return y_index, x_index


def _validate_patch_selection(
    points: np.ndarray,
    patch_ids: np.ndarray,
    budgets: np.ndarray,
    layout: PatchLayout,
    used_cells: set[tuple[int, int]],
) -> None:
    """验证选出的点数、坐标、Patch 归属和整数像素占用均满足约束。"""
    if len(points) != layout.config.total_points or len(points) != len(patch_ids):
        raise ValueError("DaD Patch 选点总数与 PASM 总点数不一致")
    if len(used_cells) != len(points) or len(np.unique(points, axis=0)) != len(points):
        raise ValueError("DaD Patch 选点不得重复")
    if np.any(points < 0.0) or np.any(points >= layout.config.image_size):
        raise ValueError("DaD Patch 选点超出图像范围")
    if not np.array_equal(np.bincount(patch_ids, minlength=layout.config.patch_count), budgets):
        raise ValueError("DaD Patch 实际点数必须等于分配预算")
    if not np.array_equal(layout.patch_ids(points), patch_ids):
        raise ValueError("DaD Patch 选点必须位于声明 Patch 内")


def _image_tensor(
    image: np.ndarray | torch.Tensor, image_size: int, device: torch.device
) -> torch.Tensor:
    """验证并将 CHW RGB 图像转换为 DaD 所需的单样本张量。"""
    tensor = torch.as_tensor(image, dtype=torch.float32, device=device)
    if tensor.ndim == 3:
        tensor = tensor[None]
    if (
        image_size <= 0
        or tensor.shape != (1, 3, image_size, image_size)
        or not torch.isfinite(tensor).all()
    ):
        raise ValueError("DaD 输入图像必须是有限的 [3, S, S] 或 [1, 3, S, S] 数组")
    return tensor


def _normalized_to_pixels(normalized_points: torch.Tensor, image_size: int) -> np.ndarray:
    """将 DaD 归一化亚像素点转换并裁剪到图像内的像素坐标。"""
    points = torch.as_tensor(normalized_points, dtype=torch.float32)
    if image_size <= 0 or points.ndim != 2 or points.shape[1:] != (2,) or not torch.isfinite(points).all():
        raise ValueError("DaD 归一化点必须是形状为 [N, 2] 的有限数组")
    upper = float(np.nextafter(np.float32(image_size), np.float32(0.0)))
    pixels = torch.clamp((points + 1.0) * (image_size / 2.0), min=0.0, max=upper)
    return pixels.cpu().numpy().astype(np.float32)
