# PR review process

This repo is maintained part-time and PRs land from a backlog of substantial
feature branches. The maintainer cannot deep-review every PR, so **the burden
of proof is on the PR author**: automated checks plus a documented adversarial
review must make each PR safe to merge on a skim. This mirrors the process
already used on the `groundcontrol` repo (CI + adversarial agent reviews with
Claude Code locally and GitHub Copilot review before merging).

## Roles

| Who | Reviews what |
|---|---|
| **PR author** | Runs the full local gauntlet (tests, regression gate, adversarial reviews), triages findings, writes the PR record. Owns correctness. |
| **Adversarial agents** (Claude Code `/code-review`, GitHub Copilot) | Correctness bugs, edge cases, CRS/geodesy mistakes, silent format changes. Their findings must be *dispositioned*, not ignored. |
| **Human reviewer** (repo owner or a teammate familiar with the touched area) | Strategy and scope: is this the right change, does it match the roadmap, are the dispositions credible. Not expected to re-derive every line. |

Anyone on the team may be the human reviewer; changes to CRS/datum handling,
`pdal_pipeline.py` geometry/extent logic, or on-disk output formats should get
a reviewer who has worked in that area.

## Author workflow

1. **Keep the PR strategic and small.** One coherent change per PR. If a
   feature branch has grown several independent changes, split it — a stack of
   small PRs merges faster than one unreviewable one. Every PR carries its own
   record (description + test evidence + review findings) so it can be
   understood a year later without the author present.

2. **Local checks** (all must pass before requesting review):

   ```bash
   pixi run -e dev test                 # full test suite
   pixi run -e dev check-regressions    # mypy + ruff regression gate
   ```

   The regression gate compares per-file mypy/ruff finding counts against the
   checked-in baseline (`.github/regression_baseline.json`) and **fails only
   on new findings** — the known pre-existing debt does not block you, and
   fixing debt never fails the gate. If you reduced the counts, ratchet the
   baseline down (`pixi run -e dev update-baselines`) and commit it. Raising
   the baseline is only acceptable with explicit reviewer sign-off and a
   justification in the PR description.

3. **Adversarial agent review** (both, before marking ready):
   - **Claude Code**: from the branch, run `/code-review` locally at **high
     effort**. Have it review the full PR diff against main.
   - **GitHub Copilot**: open the PR (draft is fine) and request a Copilot
     review from the Reviewers panel.

4. **Triage the findings.** Sort every finding into one of two lists and post
   them as a single PR comment:
   - **Fixed** — finding, and the commit that fixes it.
   - **Dispositioned** — finding, and *why it is not being fixed* (false
     positive, out of scope with an issue link, accepted trade-off). A
     disposition is a short argument, not "wontfix".

   Re-run the reviews if the fixes were substantial. An empty findings comment
   ("both reviews clean") is a valid record.

5. **Fill in the PR template** — it is the merge record. Pay particular
   attention to:
   - **Test evidence**: paste the `pixi run -e dev test` tail; for changes
     only verifiable against real data, point at the run outputs.
   - **CRS/datum/geodesy**: any touched transform, epoch, or geoid logic needs
     a stated verification method, not an assertion of correctness.
   - **Compatibility**: `processing_metadata.yaml` / output-layout readers
     have legacy fallbacks — if a format changes, say so and keep the
     fallback working.

## Merge criteria

A PR merges when **all** of the following hold:

1. **CI green**: tests on all platforms, example workflows, and the
   lint/typecheck regression gate.
2. **Checklist complete**: every box in the PR template checked or explicitly
   marked N/A with a reason.
3. **Findings dispositioned**: the adversarial-review comment exists and has
   no un-triaged findings.
4. **Human approval**: one approving review. The reviewer's job is scope and
   credibility of the record above — if the record is solid, a skim-approve
   is legitimate.

The author merges after approval (squash or merge commit at the author's
discretion; keep the PR title/description accurate since it becomes history).

## CI reference

The workflow (`.github/workflows/ci.yaml`) runs on every PR:

- **quality** (ubuntu): `pixi run -e dev check-regressions` — ruff check,
  ruff format, and mypy, gated against `.github/regression_baseline.json`.
- **test** (ubuntu + macos matrix): `pixi run test`, plus the EPT and DSM
  example workflows against live 3DEP data.

Pixi environments are cached and superseded runs are auto-cancelled, so
force-pushing a fix does not queue behind the old run.

### Maintaining the baseline

`.github/regression_baseline.json` is the accepted-debt ledger. It should only
ever shrink. When it reaches zero for a tool, that tool is effectively a hard
gate and the baseline entry documents that fact. If mypy/ruff versions bump
(dependabot) and produce new findings with no code change, refresh the
baseline in that bump PR and note the delta.
