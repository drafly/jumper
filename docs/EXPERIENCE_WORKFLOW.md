# End-to-end Jumper experience workflow

[Project guide](PROJECT_GUIDE.md) · [中文](EXPERIENCE_WORKFLOW.zh.md)

This is the project-level plan for an experience that combines Jumper's appearance, world,
motions and physical interaction. It connects two independently versioned repositories without
pretending that they share one runtime:

- [jumper-design](https://github.com/KingKongRobotics/jumper-design) produces display-only
  appearance packages (`.skin`) and environment packages (`.map`).
- This repository trains and validates behavior, defines controls and state transitions, and
  packages policies plus the controller as an `.app` bundle.

Read this document before starting a multi-part experience. Then follow each repository's own
workflow for the stage being executed.

## Deliverables, not one invented super-package

A complete experience is a versioned set of artifacts:

| Artifact | Owner | What it proves |
|---|---|---|
| Creative brief and storyboard | project | The intended character, world, interaction and sequence are explicit. |
| `<appearance>.skin` | Design | A complete display-only robot appearance passes package and visual review. |
| `<environment>.map` | Design | Terrain, props, spawn and default display robot pass package and MuJoCo loading checks. |
| Trained checkpoints and validation records | Train | Each learned behavior passes its own physical and statistical acceptance gates. |
| `<experience>.app` | Train/deploy | Policies, contracts, controls, controller builds and manuals form a loadable runtime bundle. |
| Evidence report and videos | project | Results, failures, assistance and unverified boundaries are visible and reproducible. |

Do not invent a container that merges `.skin`, `.map` and `.app`. Their formats have different
consumers and compatibility rules. Keep exact repository commits and SHA-256 values beside the
experience record.

## Non-negotiable integration boundaries

1. Train has no `.skin` loader. A skin is validated and previewed in Design; it is not a training
   input.
2. Imported `.map` files are replay-only in Train. Parallel training uses built-in replicated
   scenes.
3. Train replay keeps the task robot and controller when it imports a map; it does not silently
   substitute the map's embedded display robot.
4. A Design package passing schema and MuJoCo loading checks does not prove manufacturing fit,
   policy performance or real-hardware safety.
5. Do not reconcile mechanical differences between Design and Train silently. Record the profile,
   robot revision and both commits, then resolve a mismatch explicitly.
6. Simulation success is not board, motor-bus or payload validation.

## Stage 0 — define the experience contract

Write the brief before generating assets or training:

- character and selected visual direction;
- setting, scale, terrain, props and spawn;
- story beats in order;
- behaviors required at each beat;
- which interactions must be physical;
- allowed operator controls and interruption behavior;
- measurable success conditions;
- target simulator, host and hardware boundary;
- what will be shown in the final video and report.

Classify every proposed interaction as one of:

- **physical** — collision, friction and policy/controller motion determine the result;
- **operator-timed** — a person selects the moment, but the resulting motion is genuine;
- **story-assisted** — attachment, pose writing, animation trigger or another nonphysical device.

Story-assisted behavior is allowed only when it is visibly labelled in the interface, video and
report. It must never be counted as a physical success.

**Gate:** the brief, sequence, artifact owners and acceptance metrics are agreed.

## Stage 1 — select the appearance

Use the current Design instructions:

- [Design AGENTS.md](https://github.com/KingKongRobotics/jumper-design/blob/main/AGENTS.md)
- [Design workflow](https://github.com/KingKongRobotics/jumper-design/blob/main/docs/workflow.md)
- [Content-package protocol](https://github.com/KingKongRobotics/jumper-design/blob/main/docs/content-packages.md)

For a new appearance, present distinct concepts and wait for explicit user selection before
choosing a modeling provider or generating geometry. Record the selected reference, actual user
confirmation and immutable hashes. Reusing an explicitly selected existing design satisfies this
gate.

**Gate:** one appearance is explicitly selected and its references are recorded.

## Stage 2 — build and validate the skin

In a separate `jumper-design` checkout and virtual environment:

1. assemble a complete robot rather than a single decorative shell;
2. keep collision and physical parameters unchanged for display-only work;
3. export the current display format (`.skin/3` at the time this plan was written);
4. run `verify-package` with the selected trusted robot profile and `--mujoco`;
5. review real front, side, back, assembled-color and gray-geometry renders;
6. check silhouette, proportions, palette, characteristic details and activity clearances;
7. record the Design commit and final archive SHA-256.

Printing, slicing and physical fit are separate engineering work. A display skin must not be
described as a fitted printable shell.

**Gate:** package validation and visual acceptance both pass.

## Stage 3 — build and validate the scene

Define scale, terrain, props, challenge targets, spawn and camera needs. Author the scene in
Design and export the current `.map` format (`kk-scene-package/2` at the time this plan was
written). Include a verified default display robot and an explicit spawn. Run package validation
with the trusted profile, required capabilities and `--mujoco`.

Review:

- robot and prop scale;
- collision geometry and stable support;
- spawn clearance and reachable paths;
- lighting, cameras and final framing;
- challenge dimensions against the behavior's measured capability;
- provenance and licences for external assets.

Use the exported map for Train replay checks only. If behavior training needs comparable terrain
or props, implement a replicated built-in Train scene and document how it differs from the display
map.

**Gate:** the map loads, the spawn is valid, the scene is visually accepted, and physical
challenge dimensions do not exceed measured behavior limits.

## Stage 4 — inventory and choose motions

Start with existing behavior before training another policy:

- locomotion and body pose for approach and observation;
- `jumper.five_foot` for claw-side movement and interaction;
- `gesture_hello`, `gesture_bow`, `gesture_paw` and `gesture_salute` for story beats;
- existing dance tasks for celebration;
- jump only where its landing and interruption constraints are acceptable.

For each beat, record whether the behavior is reused unchanged, retimed through an existing
control, retrained, or newly implemented. Check joint limits, non-foot contacts, stability,
entry/exit pose and the transition to the next beat.

**Gate:** every storyboard beat maps to a named task/motion or an explicitly scoped development
item.

## Stage 5 — validate physical interaction before autonomy

Use a rigid prop of known dimensions and mass. Validate the smallest physical loop at a fixed,
known pose before adding perception or approach:

1. settle;
2. establish contact;
3. perform the interaction;
4. stop;
5. disengage;
6. retreat.

The current household-object route uses low-speed rigid-body pushing because contact grasping did
not reach the cross-seed reliability gate. Its detailed evidence and next gates live in
`tasks/jumper/five_foot/out/grasp_validation_summary.md` and `HANDOFF_DEEPSEEK.md` on the
development branch.

Progress in this order:

```text
fixed known pose
→ repeated trials across seeds
→ longer distance
→ initial-pose tolerance grid
→ approach and alignment
```

Record successes, total trials, displacement, lateral error, tilt, slip/drag, stopping,
disengagement, falls, non-foot contacts and failure reasons. Preserve failed trials. Hardware
grasp ratings are not reliable payload claims; use measured results.

**Gate:** the physical loop passes its declared multi-seed threshold without hidden retries or
unlabelled assistance.

## Stage 6 — compose the behavior state machine

Only after the underlying interaction is reliable, compose the story sequence. A typical object
interaction is:

```text
INTRO_MOTION
→ OBSERVE
→ APPROACH
→ STABILIZE
→ ALIGN
→ CONTACT
→ INTERACT
→ STOP
→ DISENGAGE
→ RETREAT
→ CELEBRATE
```

Every state declares:

- entry action and expected pose;
- measurable success guard;
- timeout;
- fall and illegal-contact response;
- operator cancellation behavior;
- next state on success and failure;
- whether perception, known simulator state or operator timing supplied its target.

Use names from `controller/vocabulary.json`. Task-local meanings belong in `controls.yaml`; mode
switches and exits belong in `deploy/manifests.json`; controller safety and cascade state belong
in `deploy/jumper.controller.toml`. Validate entry, normal exit, timeout, interruption and operator
loss. Do not train a policy to ignore a command disappearing: command loss must return toward a
safe stopped behavior.

**Gate:** the complete sequence can be entered, completed, cancelled and failed safely.

## Stage 7 — integrated rehearsal without overstating integration

Run and preserve separate checks because no current host consumes every artifact together:

1. Design preview of the selected skin from all required views;
2. Design preview of the map with its embedded default display robot;
3. Train replay of each policy and physical interaction in its validated physics setup;
4. Train replay in the imported map where compatible, remembering that it uses the task robot;
5. `play --app` or browser/host validation of the controller, controls and transitions;
6. an edited story video only after every source segment is labelled with its validation scope.

A composite video may communicate the intended experience, but it must not imply that Train loaded
the skin or that a display-map check validated policy physics.

**Gate:** all component checks pass and the final presentation states their boundaries accurately.

## Stage 8 — export, bundle and document

Export accepted checkpoints and build the `.app` using `deploy/README.md` and the deploy skill.
Translate and pack the Chinese manual with the bundle-manual workflow. Preserve:

- Train commit, task arguments and checkpoint SHA-256;
- Design commit, profile and `.skin`/`.map` SHA-256;
- simulator/backend and physics parameters;
- controls and state-machine revision;
- trial-level results and accepted thresholds;
- success and failure videos;
- all story-assisted mechanisms;
- open simulator and hardware gaps.

**Gate:** packages, controller references, manuals and evidence are reproducible from recorded
commits and hashes.

## Stage 9 — real-hardware progression

Treat hardware as a new validation layer:

1. board and bundle preflight with no motor-bus command;
2. supported stance and mode-transition checks without a prop;
3. low-speed unloaded motion with an emergency stop;
4. light, soft and replaceable prop inside a clear test area;
5. incremental distance and payload tests;
6. only then, a rehearsed story run.

Do not transfer simulator payload, friction, success rate or clearance claims to hardware without
measurement.

## Current project status

| Layer | Current evidence | Next gate |
|---|---|---|
| Appearance | Design workflow and example skins exist; this experience has no selected final skin recorded here. | Select and validate one appearance, then record Design commit and SHA. |
| Scene | Design can export and validate `.map/2`; this experience has no accepted final scene recorded here. | Freeze the scene brief, dimensions, spawn and visual acceptance. |
| Motions | Locomotion, posture, four gestures and several dances are available for reuse. | Assign each storyboard beat and verify transitions. |
| Physical interaction | Real contact pushing is the active route; the historical 7/8 result exposed a validator command-resampling fault. | Fix the validator and establish a new multi-seed baseline before longer distances. |
| Autonomy | Fixed known-pose interaction is under validation. | Pass distance and pose-grid gates before approach/alignment. |
| Controls | Existing vocabulary, claw modes, gestures, dances and exits are documented. | Add only the state/mode wiring the accepted story needs and test interruption. |
| Deployment | Export and host-side bundle checks exist. | Build from accepted checkpoints; board inference and live motor-bus control remain separate gates. |

## Source-of-truth map

| Question | Source |
|---|---|
| Cross-repository route and integration limits | `docs/WORKFLOWS.md` |
| This end-to-end experience order | `docs/EXPERIENCE_WORKFLOW.md` |
| Appearance and map production | current `jumper-design/AGENTS.md`, `docs/workflow.md`, `docs/content-packages.md` |
| Training and replay | `docs/USAGE.md`, task README and measured experiment records |
| Operator vocabulary and modes | `docs/CONTROLS.md`, `controller/vocabulary.json`, `deploy/manifests.json` |
| Policy export and app build | `deploy/README.md`, `deploy/BUNDLE.md` |
| Current contact-interaction evidence | `tasks/jumper/five_foot/out/grasp_validation_summary.md`, `HANDOFF_DEEPSEEK.md` |
| Hardware facts and unverified boundaries | `docs/HARDWARE.md`, `deploy/README.md` |

When these sources disagree, stop at the boundary. Do not make one repository's artifact or
measurement silently stand in for another's.
