# Limitations of this evaluation

What the numbers in `results.md` do **not** show. Written before reading the results,
so it is not shaped around them.

## Scope

- **One dataset, one location, one period.** Hourly Open-Meteo reanalysis for a single
  grid cell, 2020-2024. Nothing here says how the agent behaves on messy station data
  with gaps, duplicated timestamps or instrument errors.
- **Reanalysis, not observations.** The data has no missing values, so the agent is never
  forced to decide how to handle them. The verifier's NaN check therefore only fires on
  NaNs the agent's own code creates.
- **One model family.** Results are for the model named in `results.md`. A different model,
  or the same model at a different reasoning effort, may behave very differently.
- **Questions and agent written by the same person.** The questions may unconsciously match
  what the agent is good at. An independent question set would be stronger evidence.

## Statistics

- **10 runs per question** gives a rough per-question rate: 9/10 and 10/10 are not
  meaningfully different. Differences between configurations of a few percentage points
  are within noise; no confidence intervals or significance tests are reported.
- **pass@k is optimistic by construction.** It answers "would at least one of k attempts be
  right", which is not what a user gets. pass@1 is the honest headline number.
- **Tolerances are a judgement call.** They are tight (0.01 °C), so rounding differences
  count as failures. A looser tolerance would raise every number.

## Grading

- The grader compares one value and its unit. It cannot tell **right answer, wrong method**:
  an agent that reaches the correct number by luck is scored as correct.
- Text answers are matched by canonical form (dates, months, years). An unusual but correct
  phrasing may be scored wrong.
- The first real evaluation showed this is a real risk: correct answers labelled "records"
  instead of "hours" were scored as failures until the grader was fixed. Other blind spots
  of the same kind may remain.

## Verifier

- The provenance check requires the submitted number to appear in a `run_python` result.
  It cannot detect a number that was computed **correctly by the wrong method**.
- Range checks catch gross unit errors only. They cannot distinguish 3.5 km/h from 3.5 m/s.
- "Correct catches" and "false rejections" are counted against the ground truth, so they are
  only as good as the grader.

## Sandbox

- Docker shares the host kernel: isolation is strong against mistakes and prompt injection,
  not against a kernel exploit.
- The Python-level policy layers are bypassable by design; they exist for feedback and for
  measurement. The container is the boundary (see `threat_model.md`).

## Cost and performance

- Token and latency figures come from one machine, one network and one point in time.
  Provider-side changes (caching, routing, model updates) move them.
- Cost estimates in this repository use prices that must be checked against the provider's
  current pricing page.
