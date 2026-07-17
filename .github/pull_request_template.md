<!-- Keep PRs small and strategic: one coherent change, reviewable in one
sitting. The burden of proof is on the PR author — see docs/pr_review.md. -->

## What changed and why

<!-- 2-5 sentences: the problem, the approach, and any alternatives rejected.
Link the issue if one exists. -->

## Test evidence

<!-- Paste the tail of `pixi run -e dev test` (the summary line at minimum).
If you added/changed behavior, name the test(s) that cover it. If a change is
only verifiable against real data (EPT/LAZ runs), say what you ran and where
the outputs/RUN_SUMMARY live. -->

```
(paste `pixi run -e dev test` tail here)
```

## Checklist

- [ ] `pixi run -e dev test` passes locally (evidence pasted above)
- [ ] `pixi run -e dev check-regressions` passes (no new mypy/ruff findings;
      if you fixed findings, ratchet with `pixi run -e dev update-baselines`
      and commit the baseline)
- [ ] **CRS / datum / geodesy**: if this PR touches CRS handling, datum
      transforms, coordinate epochs, geoid routes, or proj pipelines, the
      assertions are verified (state how: unit test, known-point check,
      comparison against a validated run) — "it looks right" is not
      verification
- [ ] **Data-file / metadata compatibility**: output filenames, directory
      layout, and `processing_metadata.yaml` / `merge_metadata.yaml` /
      `run_status` formats are unchanged — or, if changed, readers keep a
      legacy fallback and the change is called out below
- [ ] Docs updated (README / docs/ / CLI help) where behavior changed
- [ ] **Adversarial review completed**: Claude Code `/code-review` run locally
      (high effort) and GitHub Copilot review requested; findings triaged into
      fixed / dispositioned and posted as a PR comment (see docs/pr_review.md)

## Compatibility notes

<!-- Only if the compatibility box above is "changed": what format changed,
what the legacy fallback is, and what existing on-disk runs are affected. -->

## Adversarial review findings

<!-- Link the PR comment with the fixed / dispositioned lists, or write
"none found" if both reviews came back clean. -->
