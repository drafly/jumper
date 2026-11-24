"""Record a short clip of a checkpoint in simulation, on this machine.

The video is evidence of *motion*, not just of a loadable checkpoint: the frames
come from the same replay construction `tools/replay_eval.py` measures, the
command is pinned, and the clip is encoded at simulation speed -- one frame every
`1/fps` of simulated time -- so a slow renderer cannot make a policy look slower
than it is. The annotation says which checkpoint, which command, and how much
simulated time has passed; the report says the encoding is by simulated time.

The off-screen renderer is MuJoCo's own (`mujoco.Renderer`): CGL on macOS, EGL
elsewhere. That is the supported path on a Mac -- a windowed viewer would need
`mjpython` and a display, and is not what this is. Rendering failure is a video
failure, never an evaluation failure; the caller records it and keeps the
numeric result.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
DEFAULT_CAMERA = {"distance": 1.6, "azimuth": 150.0, "elevation": -18.0, "height": 0.08}


def _qpos_of(env, robot) -> np.ndarray:
    """The single environment's `qpos`, rebuilt from the `Entity` API.

    The same reconstruction `tasks/jumper/dance/export_media.py` documents: the
    free joint is `[xyz, quat wxyz]` in the environment's own frame, followed by
    the hinge joints in model order. Copied rather than imported on purpose --
    this tool must work for any profile task, and a tool that imports a task's
    media module would tie the two together.
    """
    pose = robot.data.root_link_pose_w[0].detach().cpu().numpy()
    origin = env.scene.env_origins[0].detach().cpu().numpy()
    joints = robot.data.joint_pos[0].detach().cpu().numpy()
    return np.concatenate([pose[:3] - origin, pose[3:7], joints])


def _yaw(quat: np.ndarray) -> float:
    w, x, y, z = (float(value) for value in quat)
    return float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


def _build(task: str, checkpoint: Path, seed: int):
    """Build the replay environment and policy, as `replay_eval.run` does."""
    from dataclasses import asdict

    import torch
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper
    from mjlab.rl.runner import MjlabOnPolicyRunner
    from mjrl.backend.resolve import resolve
    from mjrl.backend.select import use_backend
    from mjrl.replay import skip_rewards

    import tasks

    torch.manual_seed(seed)
    spec = tasks.get(task)
    asset = spec.resolve_asset(None)
    resolution = resolve(backend="native", device="cpu", num_envs=1, training=False)
    use_backend(resolution)
    env_cfg = tasks.load_env_cfg(task, asset=asset, play=True)
    env_cfg.scene.num_envs = 1
    env_cfg.seed = seed
    env = ManagerBasedRlEnv(cfg=env_cfg, device=resolution.device)
    try:
        skip_rewards(env)
        agent_cfg = tasks.load_agent_cfg(task)
        wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
        runner = MjlabOnPolicyRunner(wrapped, asdict(agent_cfg), device=resolution.device)
        runner.load(
            str(Path(checkpoint)), load_cfg={"actor": True}, strict=True,
            map_location=resolution.device,
        )
        policy = runner.get_inference_policy(device=resolution.device)
        term = env.unwrapped.command_manager.get_term("twist")
        robot = term.robot
        term.cfg.resampling_time_range = (1.0e9, 1.0e9)
        term.cfg.rel_standing_envs = 0.0
        term.cfg.rel_heading_envs = 0.0
        for name in ("rel_forward_envs", "rel_world_envs"):
            if hasattr(term.cfg, name):
                setattr(term.cfg, name, 0.0)
        step_dt = float(env_cfg.sim.mujoco.timestep * env_cfg.decimation)
    except BaseException:
        env.close()
        raise
    return env, wrapped, policy, term, robot, step_dt


def _pin(term, command: tuple[float, float, float], torch) -> None:
    tensor = torch.tensor([command], device=term.device)
    term.vel_command_b[:] = tensor
    term.time_left[:] = 1.0e9
    term.is_standing_env[:] = False
    term.is_heading_env[:] = False


def _record(task: str, checkpoint: Path, command: tuple[float, float, float], seconds: float,
            fps: int, seed: int) -> dict:
    """Step the policy and return `qpos` rows at simulation-time sampling."""
    import torch

    env, wrapped, policy, term, robot, step_dt = _build(task, checkpoint, seed)
    start = [0]
    rows: list[np.ndarray] = []
    positions: list[np.ndarray] = []
    try:
        start[0] = env.common_step_counter
        model = env.sim.mj_model
        _pin(term, command, torch)
        obs = wrapped.get_observations()
        every = max(1, round((1.0 / fps) / step_dt))
        wanted = max(1, round(seconds / step_dt))
        steps = 0
        with torch.inference_mode():
            while steps < wanted:
                rows.append(_qpos_of(env, robot))
                positions.append(robot.data.root_link_pos_w[0].detach().cpu().numpy().copy())
                obs, _, dones, _ = wrapped.step(policy(obs))
                steps += 1
                if bool(dones[0]):
                    break
                _pin(term, command, torch)
        # One frame per `every` control steps, which is exactly `fps` frames per
        # simulated second; the last row is kept so the clip does not end short.
        sampled = rows[::every]
        sampled_positions = positions[::every]
        meta = {
            "model": model,
            "qpos": np.stack(sampled),
            "positions": np.stack(sampled_positions),
            "step_dt": step_dt,
            "sampling_dt": every * step_dt,
            "steps": steps,
        }
        return meta
    finally:
        env.close()


def _option(model):
    """Appearance only: the collision hulls drawn over the meshes hide the robot."""
    import mujoco
    from mjrl.viewer.live import LiveViewer

    option = mujoco.MjvOption()
    for group in LiveViewer._collision_only_groups(model):
        option.geomgroup[group] = 0
    return option


def _annotated(frame: np.ndarray, lines: list[str]) -> np.ndarray:
    from PIL import Image, ImageDraw, ImageFont

    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    y = 6
    for line in lines:
        draw.text((6, y), line, fill=(255, 255, 255), font=font,
                  stroke_width=1, stroke_fill=(0, 0, 0))
        y += 13
    return np.asarray(image)


def _frames(meta: dict, size: tuple[int, int], lines_for, fps: int, every: int):
    """Render sampled `qpos` rows into annotated RGB frames."""
    import mujoco

    if os.environ.get("MUJOCO_GL") is None:
        os.environ["MUJOCO_GL"] = "cgl" if sys.platform == "darwin" else "egl"
    model = meta["model"]
    qpos = meta["qpos"]
    positions = meta["positions"]
    width, height = size
    model.vis.global_.offwidth = max(int(model.vis.global_.offwidth), width)
    model.vis.global_.offheight = max(int(model.vis.global_.offheight), height)
    data = mujoco.MjData(model)
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.distance = DEFAULT_CAMERA["distance"]
    camera.elevation = DEFAULT_CAMERA["elevation"]
    option = _option(model)
    lookat = np.array([positions[0][0], positions[0][1], DEFAULT_CAMERA["height"]])
    renderer = mujoco.Renderer(model, height=height, width=width)
    try:
        for index, row in enumerate(qpos):
            data.qpos[:] = row
            mujoco.mj_forward(model, data)
            heading = _yaw(row[3:7])
            camera.azimuth = np.degrees(heading) + DEFAULT_CAMERA["azimuth"]
            lookat[:2] += 0.25 * (positions[index][:2] - lookat[:2])
            camera.lookat[:] = lookat
            renderer.update_scene(data, camera=camera, scene_option=option)
            t = index * (every * meta["step_dt"])
            yield _annotated(renderer.render(), lines_for(t))
    finally:
        renderer.close()


def _encode(frames: list[np.ndarray], fps: int, path: Path, size: tuple[int, int],
            crf: int = 26) -> None:
    import imageio_ffmpeg

    writer = imageio_ffmpeg.write_frames(
        str(path), size=size, fps=fps, macro_block_size=1, codec="libx264",
        pix_fmt_out="yuv420p", output_params=["-crf", str(crf), "-preset", "veryfast"],
    )
    writer.send(None)
    for frame in frames:
        writer.send(np.ascontiguousarray(frame).tobytes())
    writer.close()


def render(job, cfg, entry_dir: Path, seed: int = 0) -> dict:
    """Render a short MP4 for `job`; never raises for a render failure.

    Returns a dict with `status` in {"ok", "skipped", "failed"} and, on success,
    the file, size and duration. The caller records the dict as the video stage.
    """
    if not cfg.video.enabled:
        return {"status": "skipped", "reason": "video disabled by configuration"}
    if cfg.video.backend != "local":
        return {
            "status": "failed",
            "reason": (
                f"video backend {cfg.video.backend!r} is not implemented; server rendering "
                "would need libEGL (or osmesa) plus ffmpeg on the box, and a copy of the "
                "checkpoint there -- none of which this tool may install"
            ),
        }
    try:
        return _render(job, cfg, entry_dir, seed)
    except Exception as exc:  # noqa: BLE001 - a video failure is reported, not raised
        return {"status": "failed", "reason": f"{type(exc).__name__}: {exc}"}


def _render(job, cfg, entry_dir: Path, seed: int) -> dict:
    video_dir = Path(entry_dir) / "video"
    video_dir.mkdir(parents=True, exist_ok=True)
    width = min(int(cfg.video.width), 640)
    height = min(int(cfg.video.height), 480)
    command = cfg.video.command
    meta = _record(job.task, job.checkpoint, command, cfg.video.seconds, cfg.video.fps, seed)
    if meta["qpos"].shape[0] == 0:
        return {"status": "failed", "reason": "no frames were recorded"}
    every = max(1, round((1.0 / cfg.video.fps) / meta["step_dt"]))
    def lines(t: float) -> list[str]:
        return [
            f"{job.task}  {job.checkpoint.name}",
            (
                f"cmd vx={command[0]:+.2f} vy={command[1]:+.2f} wz={command[2]:+.2f} | "
                f"t={t:.2f} s sim"
            ),
            f"sim-time encoded: {cfg.video.fps} fps, {every * meta['step_dt']:.2f} s/frame",
        ]
    frames = list(_frames(meta, (width, height), lines, cfg.video.fps, every))
    path = video_dir / "replay.mp4"
    with tempfile.TemporaryDirectory(dir=video_dir) as tmp:
        raw = Path(tmp) / "raw.mp4"
        _encode(frames, cfg.video.fps, raw, (width, height))
        raw.replace(path)
    size = path.stat().st_size
    if size > cfg.video.max_bytes:
        with tempfile.TemporaryDirectory(dir=video_dir) as tmp:
            raw = Path(tmp) / "raw.mp4"
            _encode(frames, cfg.video.fps, raw, (width, height), crf=34)
            raw.replace(path)
        size = path.stat().st_size
    if size > cfg.video.max_bytes:
        path.unlink(missing_ok=True)
        return {
            "status": "failed",
            "reason": f"video is {size} bytes, over the {cfg.video.max_bytes}-byte limit",
        }
    result = {
        "status": "ok",
        "file": str(path),
        "bytes": size,
        "frames": len(frames),
        "fps": cfg.video.fps,
        "width": width,
        "height": height,
        "seconds": meta["steps"] * meta["step_dt"],
        "sim_time_encoded": True,
        "sampling_dt_s": every * meta["step_dt"],
        "command": {"vx": command[0], "vy": command[1], "wz": command[2]},
        "checkpoint": job.checkpoint.name,
        "reason": "",
    }
    (video_dir / "video.json").write_text(
        __import__("json").dumps(result, indent=2, ensure_ascii=False) + "\n", "utf-8"
    )
    return result
