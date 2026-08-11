# Geometric-Invisibility Soft Top-K Loss

## Goal

Reduce candidates selected outside the two-image geometrically visible region without changing the existing semantic-coverage fusion architecture, Patch coverage loss, DaD local-preservation loss, fixed DaD candidate pool, or global Top-1024 inference.

## Minimal training-only loss

For each fixed DaD candidate `i`, reuse the existing differentiable Soft Top-K selection weight `a_i`. Let `v_i` be the existing geometry teacher visibility flag: it is true only when the source candidate can be projected into the other image with valid visibility/depth.

```text
L_invisible = sum_i a_i * (1 - v_i) / (sum_i a_i + eps)
```

The loss is averaged over both image directions and added only in `semantic_coverage` mode:

```text
L_total = L_geo
        + lambda_cover * L_cover
        + lambda_local * L_local
        + lambda_invisible * L_invisible
```

## Explicit non-goals

- Do not penalize candidates that are geometrically visible but lack a nearby target DaD candidate.
- Do not add per-Patch quotas, point caps, distance radii, inference thresholds, or new inference filters.
- Do not change the frozen semantic encoder or DaD detector.

## Verification

Unit tests must show that moving Soft Top-K mass from invisible to visible candidates lowers the loss, invisible candidate scores receive suppressive gradients, all-visible inputs yield zero loss, and existing modes remain unchanged. A one-pair smoke run must record finite `invisible_loss` in the history JSON.
