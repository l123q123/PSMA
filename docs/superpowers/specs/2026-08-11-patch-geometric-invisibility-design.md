# Patch-Geometric-Invisibility Loss Design

## Goal

Replace only the newly added candidate-level geometric invisibility loss with a Patch-level constraint. Keep `L_geo`, `L_cover`, `L_local`, their weights, and inference unchanged.

## Design

For each direction, reuse the existing Soft Top-K Patch occupancy. A Patch is eligible for the new penalty only when it contains at least one fixed DaD candidate and contains no candidate whose geometric projection is visible in the other image.

```text
bad_patch(p) = has_candidate(p) and not any_visible_candidate(p)
L_invisible_patch = mean(soft_occupancy(p) for p in bad_patch)
```

This makes the new loss disjoint from `L_cover`: coverage is only defined on Patch targets built from geometrically visible candidates, while the new loss only suppresses fully invisible Patches. Mixed boundary Patches are left to the existing DaD-local and point-level losses.

## Constraints

- Do not alter the definitions or weights of `L_geo`, `L_cover`, or `L_local`.
- Do not alter candidates, border filtering, global Top-1024 selection, or inference.
- Reuse `coverage_topk` and `coverage_temperature`; add no hyperparameters.
- Keep recording the term as `invisibility_loss` and keep `--invisibility-weight` unchanged.
