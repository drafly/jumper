# Jumper experience brief — "Jumper's Delivery"

Status: frozen 2026-10-25. This is Phase 0 of
[the end-to-end experience workflow](EXPERIENCE_WORKFLOW.md). The user confirmed the theme, the
push distance, the first acceptance host and the artifact policy; the remaining open items are
listed in section 9.

## 1. Character and visual direction

- Character: Jumper, the 22-DoF hexapod (Liuzu V1.6) already in this repository. The left-front
  arm is stowed as a claw and used as the push face.
- Visual direction: to be selected from clearly different candidates presented by the Skin
  workstream (Phase 1 gate). Proposed candidate axes: (a) clean studio white with one accent
  colour; (b) friendly cartoon; (c) industrial utility. No skin is selected yet.

## 2. Scene

- First version: a flat studio floor. No stairs or rough terrain — the physical interaction is
  validated on flat ground only.
- Prop: `push_box`, 120 × 100 × 70 mm, 0.188 kg, free rigid body.
- Markings: the box starts at a fixed centre point; a target line at 0.15 m ahead. The ladder is
  two gates: 0.10 m is already validated, and 0.12 m then 0.15 m must each pass their multi-seed
  gate before 0.15 m becomes the experience target; a lateral lane of ±0.10 m.
- Robot spawn: about 0.4 m behind the box start, facing it.
- Camera: one fixed front-three-quarter view recording the whole run; an optional overhead view
  for measurement evidence.
- Lighting: even studio lighting; no shadow that hides the contact.
- The map's default display robot is the selected appearance once it exists; until then the
  Design default.

## 3. Storyboard

| # | Node | Motion | Classification | Success guard |
|---|---|---|---|---|
| 1 | INTRO | Stand and wave hello (`jumper.gesture_hello`) | Scripted motion | Wave completes; robot stays in place |
| 2 | OBSERVE | Face the box (`jumper.posture` locomotion) | Operator-timed | Robot stopped and facing the box |
| 3 | APPROACH | Walk toward the box (`jumper.posture` / `jumper.five_foot`) | Operator-timed | Robot within the aligned start window; no teleport |
| 4 | STABILIZE | Stop and stand stable | Physical | Linear/angular velocity and roll/pitch gates pass |
| 5 | ALIGN | Refine position and yaw | Operator-timed; in validation runs the start pose is set by reset (scripted aid, labelled) | Pose inside the validated grid |
| 6 | CONTACT | Close the claw and form the push face | Physical contact | Contact established, claw at its push pose |
| 7 | INTERACT | Low-speed push at 0.35 m/s toward 0.15 m | Physical (counted) | Push distance, lateral offset, tilt, stable stop |
| 8 | STOP | Stop stably | Physical | Robot and box at rest |
| 9 | DISENGAGE | Lift the claw | Scripted motion | Object moves ≤15 mm while lifting |
| 10 | RETREAT | Back away 0.05 m at 0.15 m/s lateral | Physical | Backoff distance, object drag ≤10 mm |
| 11 | CELEBRATE | Bow or salute, then one dance | Scripted motion | Gesture/dance completes or is cancelled cleanly |

## 4. Interaction classes

- **Physical (counted):** CONTACT, INTERACT (push), STOP, RETREAT.
- **Operator-timed (real motion, the human chooses when):** OBSERVE, APPROACH, ALIGN.
- **Scripted motion (policy playback, not counted as physical interaction):** INTRO, DISENGAGE,
  CELEBRATE.
- **Scripted aids (must be labelled, never counted):** the reset-set start pose in validation
  runs. Forbidden everywhere: attach, weld, follow-the-robot, applied external force, teleport,
  hidden retry, success-only cuts.

## 5. Operator controls and interruption

- Modes and bindings as in `deploy/manifests.json`; every name from
  `controller/vocabulary.json`.
- Default mode `locomotion` (`jumper.posture`); the five-foot claw on LB or RB; gestures on the
  d-pad; dances on Menu + the d-pad; Menu alone leaves a dance or gesture.
- Interruption: Menu exits a dance or gesture; a fall ends the run safely; loss of the operator
  command must return the controller to a safe stop — no policy may learn to ignore a missing
  command.
- Emergency stop is a hardware-layer concern and is not simulated here.

## 6. Measurable success conditions

Physical node, per trial:

- push distance ≥90% of 0.15 m;
- lateral offset ≤0.10 m; maximum tilt ≤15°;
- stable stop; backoff ≥90% of 0.05 m; object drag during backoff ≤10 mm;
- at least 3 sustained contact control frames;
- no fall; no non-foot ground contact above 1 N;
- per-control-step command integrity: the push and backoff commands stay live for their whole
  phase (diagnostic NPZs).

Multi-seed gate: seeds 0/1/2/42 × 5 trials, 20/20 at each distance on the ladder (0.12 m, then
0.15 m), every failure kept with its JSON. The experience-level gate: every node enters and exits
within its timeout, operator interruption is tested at every cancelable node, and one continuous
recording of a full run exists with no hidden failure cuts.

## 7. Targets and boundaries

- Simulator: native MuJoCo on Apple M4, backend `native`, device `cpu`.
- Hosts: the browser bundle (`play --app`) for rehearsal first. The board is a separate gate:
  RKNN inference and the live motor bus are unverified, and this machine's CycloneDDS device
  build currently fails setup gate F.
- Design assets are display-only: Train has no `.skin` loader, an imported `.map` is replay-only,
  and training uses built-in scenes.
- No simulation result transfers to hardware before Phase 9 measurements.

## 8. Deliverables and evidence

- `.skin` (Design commit + SHA-256) and `.map` (Design commit + SHA-256). Both are internal
  artifacts: they live in Design's `outputs/` and are recorded here, and are not published to the
  formal Design library;
- checkpoint + SHA-256, per-trial JSONs, MP4s and diagnostic NPZs;
- `.app` bundle with `manual.en.json` and `manual.zh.json`;
- an evidence report listing commands, parameters, machine, results, failures and labelled aids.

The experience record pins both repositories' exact commits and every artifact's SHA-256.

## 9. Decisions and open items

Confirmed by the user on 2026-10-25:

1. Theme, name and tone: the delivery story — Jumper delivers a parcel, and this brief is that
   direction; not a pure showcase.
2. Experience push distance: 0.15 m, and only after both the 120 mm and the 150 mm multi-seed
   gates pass.
3. First acceptance host: the browser (`play --app` against the bundle); the board stays a later
   gate.
4. Skin/map publication: internal artifacts (Design `outputs/` + commit + SHA-256), not entries
   in the formal Design library.

Still open:

1. Appearance direction: choose from the Skin workstream's candidates (Phase 1 gate).
2. Which operator-timed nodes should become autonomous later (approach, align).

## 10. Gates

- **Phase 0 gate:** this brief, the node order, artifact ownership and acceptance metrics are
  agreed.
- No Design asset is generated and no new training starts before that gate.
