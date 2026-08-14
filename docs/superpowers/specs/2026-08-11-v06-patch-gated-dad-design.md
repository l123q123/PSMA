# v0.6 Patch-Gated DaD Design

## Goal

Use pair-conditioned semantic information only to suppress unreliable Patch regions, while preserving the frozen DaD ordering among candidates in reliable regions.

## Inference

1. Run frozen DaD and retain its original 3x3-NMS top-4096 candidate pool per image.
2. Predict an independent semantic pass probability `q_p = sigmoid(h(z_p))` for every 16x16 Patch. `q_p` is not normalized across Patches.
3. Apply one shared Patch penalty to every candidate `i` in Patch `p`:

   `final_logit_i = dad_logit_i - lambda * (1 - q_p)^gamma`.

4. Exclude the established 8-pixel border band and take global Top-1024 from the fixed candidate pool.

If `q_p` is high, the penalty is near zero, so candidate ordering inside that Patch is exactly the original DaD ordering. If `q_p` is low, all candidates in that Patch are lowered together.

## Teacher

For every frozen DaD candidate, use depth, camera pose, depth-consistency visibility, and proximity to the other image's frozen DaD candidate pool to obtain the existing soft geometric correspondence label `t_i` in `[0, 1]`.

For a Patch containing reliably visible candidates, define its teacher as:

`y_p = max(t_i for i in Patch p and visible_i)`.

Patches with no reliably visible candidates are ignored in the Patch loss. The maximum rule means a Patch is passed through whenever it contains at least one geometrically repeatable DaD candidate; DaD alone selects the best local point thereafter.

## Training

The trainable fusion module is only a 128-to-1 semantic gate head. DaD and the loaded pair semantic encoder remain frozen. Train independent Patch probabilities with BCE on all non-ignored Patch teachers:

`L_gate = BCE(q_p, y_p)`.

There is no candidate-level keep loss, suppression loss, ranking loss, candidate penalty head, or global Patch-budget allocation.

## Required Properties

- Gate probabilities are independent: no softmax or sum-to-one normalization across Patches.
- The fusion stage cannot create, move, or locally reorder DaD candidates inside a Patch.
- A Patch with `q_p = 1` produces zero penalty and exactly preserves original DaD logits.
- A Patch with `q_p = 0` applies the maximum penalty `lambda` to every candidate in that Patch.
- The 8-pixel border exclusion remains active at inference.

## Validation

Unit tests verify maximum teacher aggregation, ignored unknown Patches, independent BCE gate loss, and identical within-Patch ordering after gating. The nine-pair evaluation compares matches, 3-pixel inlier rate, spatial occupancy, and relative-pose AUC against raw DaD and the prior candidate-suppression fusion.
