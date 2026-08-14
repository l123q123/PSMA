# v0.6 Candidate Suppression Fusion Design

## Goal

Replace the current dense residual fusion head, which creates new local peaks before global Top-K, with a candidate-level feature fusion head that can only reduce the scores of original DaD NMS candidates.

## Scope

- Keep DaD, DINO, and the selected semantic encoder frozen during fusion training.
- Generate the candidate pool exclusively with original DaD NMS; the fusion head must never create, move, or add candidate locations.
- Fuse the candidate's 32-D DaD feature, original DaD logit, and its 16x16/37x37 dual-image semantic token.
- Predict a non-negative penalty. Final score is `base_logit - penalty`.
- Use geometry/visibility and a nearby candidate in the opposite image to create positive, negative, and ignored candidate labels.
- Exclude a deterministic image-border band from final candidates. This is independent of semantic learning and addresses the observed regular border response.

## Model

For candidate `i` in patch `p`:

```text
q_p = sigmoid(semantic_quality_head(F_sem,p))
r_i = softplus(candidate_penalty_head(F_DaD,i, F_sem,p, base_logit_i))
penalty_i = lambda_patch * (1 - q_p)^gamma
          + lambda_point * (epsilon + (1 - q_p)^gamma) * r_i
final_logit_i = base_logit_i - penalty_i
```

`epsilon` is small and non-zero: high-value patches remain almost unchanged, but an obviously non-repeatable candidate inside such a patch can still receive weak point-level suppression. The parameterization guarantees `final_logit_i <= base_logit_i`.

## Training Labels

For every original DaD candidate in A, project it to B using the training depth and pose.

- **Positive:** the projection is visible, depth-consistent, within the image, and has at least one original DaD B candidate within `teacher_sigma_px`.
- **Negative:** the projection is reliably out of view/occluded, or a depth-consistent projection has no B candidate within `teacher_search_radius_px`.
- **Ignore:** required depth or projection validity is unavailable, so the teacher cannot decide reliably.

Build labels in both A-to-B and B-to-A directions. Do not impose one-to-one matching across the entire candidate pool.

## Loss

Use valid candidates only.

```text
L_keep     = mean(positive_weight * penalty)
L_suppress = mean(negative_weight * softplus(margin - penalty))
L_rank     = soft pairwise ranking(final_logit_positive, final_logit_negative)
L_total    = L_keep + suppress_weight * L_suppress + ranking_weight * L_rank
```

The loss explicitly preserves stable DaD candidates and pushes unreliable candidates down. It does not use dense-map BCE and cannot reward creation of new peaks.

## Inference

```text
DaD dense logit -> original 3x3 NMS candidate pool -> hard border mask
-> candidate suppression fusion -> global Top-1024 among the fixed candidates
```

No Patch budget, per-Patch quota, second dense NMS, or learned positive score boost is introduced.

## Validation

1. Unit tests prove the head cannot increase a candidate logit.
2. Unit tests prove high semantic quality results in weaker suppression than low quality under identical DaD features.
3. Unit tests prove final inference selects only original DaD candidate positions and rejects the configured border band.
4. Run the existing nine-pair evaluation and compare point distributions, matching metrics, and RANSAC pose metrics with `fusion-16-main6px`.
