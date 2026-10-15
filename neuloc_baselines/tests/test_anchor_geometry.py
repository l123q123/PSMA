import torch

from neuloc_lvsm.geometry import normalize_support_and_query


def make_pose(x: float) -> torch.Tensor:
    pose = torch.eye(4)
    pose[0, 3] = x
    return pose


def test_four_anchor_indices_produce_shared_query_pose():
    poses = torch.stack([make_pose(float(i)) for i in range(8)])
    support = poses[[0, 2, 4, 6]]
    query = poses[7]
    normalized_a, query_a = normalize_support_and_query(support, query, anchor_indices=[0, 1, 2, 3])
    support_more = torch.cat([support, poses[1:2]], dim=0)
    normalized_b, query_b = normalize_support_and_query(support_more, query, anchor_indices=[0, 1, 2, 3])
    assert torch.allclose(query_a, query_b)
    assert torch.allclose(normalized_a, normalized_b[:4])
