# Jumper 推送交互与 DeepSeek 接管手册

> 更新日期：2026-10-22
>
> 工作树基线：`7d3cc4bebd1efe951e0adc38c053d417260867c0`（`main`）
>
> 当前主路线：0.188 kg 自由刚体的真实接触推送；不使用吸附、焊接、跟随、外力或搬运阶段瞬移。

本文用于把当前未提交工作树、训练 checkpoint、实验记录和仿真环境迁移到另一台电脑，并交给支持 DeepSeek API 的编码代理继续。先完成“迁移完整性”和“环境六门检查”，再训练或改代码。

## 0. 零环境机器从这里开始

远程迁移分支为 `codex/jumper-push-handoff`。该分支专门携带本轮代码、handoff、checkpoint、主结果 JSON、成功视频和实验总结，因此另一台机器不需要先从旧电脑复制项目目录。

在新机器上先安装最基础的软件：

- macOS：安装 Xcode Command Line Tools，然后安装 Homebrew、Git 和 Python 3.12。
- Ubuntu 22.04/24.04：安装 Git、Python、venv 和 C/C++ 构建工具。
- Windows：安装 Git、Python 3.11 和 Visual Studio Build Tools 的 “Desktop development with C++”；训练更推荐 WSL2/Ubuntu 或原生 Ubuntu。

macOS 起步命令：

```bash
xcode-select --install
# 按 https://brew.sh/ 的当前命令安装 Homebrew，然后：
brew install git python@3.12
git clone --branch codex/jumper-push-handoff --single-branch \
  https://github.com/drafly/jumper.git
cd jumper
python3.12 .claude/skills/setup-env/scripts/detect.py
```

Ubuntu 起步命令：

```bash
sudo apt update
sudo apt install -y git python3 python3-venv build-essential
git clone --branch codex/jumper-push-handoff --single-branch \
  https://github.com/drafly/jumper.git
cd jumper
python3 .claude/skills/setup-env/scripts/detect.py
```

克隆完成后的第一个 Claude Code 指令见本文 8.1。让 Claude Code 先读取规则并执行环境检测，不要让它直接开始训练或修改代码。系统软件安装、`sudo`、GPU 驱动和 API 登录仍需由用户在新机器上确认授权。

## 1. 接管摘要

真实接触夹取虽然在精确位姿下可行，但未通过跨 seed 的可靠搬运门槛，因此作品主路线已降级为更可信的低速刚体推送。当前流程是：

`接近（尚未实现） → 停稳 → 闭爪形成推面 → 低速推送 → 停稳 → 抬爪 → 撤离`

已经成立的证据：

- 道具：`push_box`，120 × 100 × 70 mm，0.188 kg，自由刚体。
- 控制：五足策略驱动底盘；夹爪保持接触姿态。
- 物理：MuJoCo 刚体碰撞与摩擦，地面滑动摩擦 0.35，elliptic cone，`impratio=50`。
- checkpoint：`logs/jumper/jumper.five_foot/2026-10-21_22-52-37/model_87198.pt`。
- 训练来源：从 `model_87099.pt` 恢复完整 PPO 状态，继续 100 次更新，64 environments，共 153,600 environment steps。
- 固定中心点、80 mm、四个 seed 各两次：7/8 成功；seed 1/2/42 为 2/2，seed 0 为 1/2。
- 成功试次推送 82.7–92.0 mm，最大倾斜 5.8°，撤离 53–95 mm；8 次均无摔倒、无非足触地。
- 唯一失败：seed 0 第一次复位未建立持续接触，仅推约 0.974 mm，失败码 `push_distance`。
- 固定预退让和自动重试会产生额外扫碰，已拒绝，不应恢复成默认方案。
- **2026-10-25 诊断结论**：上述 7/8 的失败来自验证器命令随机重采样（play 配置 twist 项 `resampling_time_range` 未钉、默认 3–8 s，`rel_standing_envs=1.0`，计时到期即把验证器写入的 push 命令清零），**不是策略的 seed 0 冷启动缺陷**；命令存活过 push 第 1 步的 13/13 次试验全部成功。修复后（`push_objects.py` 在环境构造前钉 `(1e6, 1e6)`）同参数 seed 0/1/2/42 × 5 次独立 reset、80 mm 为 **20/20 成功**，摔倒 0、非足触地 0——这是新的正式基线。历史 7/8 及其 JSON/视频仍保留为问题证据，不删除、不覆盖，但不再用于评价策略的 seed 鲁棒性。详见 `tasks/jumper/five_foot/out/grasp_validation_summary.md` 2026-10-25 一节。

当前结论不是“已可靠部署”，而是“推送明显优于抓取且跨 seed 可行”。下一步扩展 100–150 mm 长距离，最后才做 3×3×3 初始位姿网格和自主接近。

## 2. 必须迁移的内容

对普通 `main` 分支而言，仅执行 `git clone` **不够**，因为 checkpoint、日志、JSON 和视频受 `.gitignore` 排除。本次专用远程分支会强制纳入下面列出的迁移产物；若该分支尚未推送成功，才使用 2.1 的整目录复制方案。

### 2.1 远程分支不可用时的备用迁移方式

把整个 `jumper/` 目录复制到新电脑，但不要复制旧电脑的 `.venv`、Python 缓存和 Warp 缓存；新电脑必须重新创建虚拟环境。可用外置磁盘、局域网 `rsync` 或可信的加密云盘。

示例（在旧电脑的项目上级目录执行，按实际目标替换 `<destination>`）：

```bash
rsync -a --info=progress2 \
  --exclude '.venv/' \
  --exclude '**/__pycache__/' \
  --exclude '.pytest_cache/' \
  jumper/ <destination>/jumper/
```

不要只复制 `git diff`：它不会包含未跟踪文件，也不会包含被忽略的 checkpoint、输出 JSON 和视频。

### 2.2 必须存在的训练与证据文件

| 文件 | 用途 | SHA-256 |
|---|---|---|
| `logs/jumper/jumper.five_foot/2026-10-21_22-52-37/model_87198.pt` | 当前推送策略 | `2b1196711a4dacd451f440ce176e91d87f89a371dfa62d6497dcfbe4253e97cc` |
| `tasks/jumper/five_foot/out/push_box_87198_seed1_80mm.mp4` | 成功 MuJoCo 视频 | `de2242fcb81a65d6d0fa292f7cbce520edea2b860fb6f25faa14bcc9c155d450` |
| `tasks/jumper/five_foot/out/push_box_87198_seed0_trials2_80mm.json` | seed 0 原始结果 | `6ede922ddc580135fb6888bfa29e83990d4f74282fac96eea8b03c7da6708e97` |
| `tasks/jumper/five_foot/out/push_box_87198_seed1_trials2_80mm.json` | seed 1 原始结果 | `857c2b8dfec40ff35f34a9cafee3ddcbaeb86e9feb84d0820d5786b81fe48801` |
| `tasks/jumper/five_foot/out/push_box_87198_seed2_trials2_80mm.json` | seed 2 原始结果 | `dff15361db1930c0949b1d837e56c86e55d1d8b3474d1ab30b8c2fb45f5c6cb8` |
| `tasks/jumper/five_foot/out/push_box_87198_seed42_trials2_80mm.json` | seed 42 原始结果 | `8ab1d236ab1413e000fca52c07c2eecacf43ba6e71b5d23815bcf24f6a437291` |
| `tasks/jumper/five_foot/out/grasp_validation_summary.md` | 抓取、搬运、推送的完整实验时间线 | 迁移后重新计算 |

> 注意：seed 2 的正确校验和应在迁移前后用下面命令重新核对；表格中的字符串若与命令输出不一致，以命令输出为准。

```bash
shasum -a 256 \
  logs/jumper/jumper.five_foot/2026-10-21_22-52-37/model_87198.pt \
  tasks/jumper/five_foot/out/push_box_87198_seed1_80mm.mp4 \
  tasks/jumper/five_foot/out/push_box_87198_seed{0,1,2,42}_trials2_80mm.json
```

### 2.3 本分支包含的代码改动

以下为本阶段从基线修改的文件：

```text
tasks/jumper/five_foot/__init__.py
tasks/jumper/five_foot/claw.py
tasks/jumper/five_foot/env_cfg.py
tasks/jumper/five_foot/mdp/grasp.py
tasks/jumper/five_foot/objects.py
tasks/jumper/five_foot/rl_cfg.py
tasks/jumper/five_foot/tools/grasp_objects.py
tasks/jumper/five_foot/tools/grasp_pose.py
tasks/jumper/five_foot/tools/vel_check.py
```

以下为本阶段新增文件：

```text
tasks/jumper/five_foot/mdp/contact_command.py
tasks/jumper/five_foot/mdp/terminations.py
tasks/jumper/five_foot/tools/contact_training_probe.py
tasks/jumper/five_foot/tools/push_objects.py
```

`docs/Jumper-Mac-Guide.zh.md` 也随迁移分支提交。旧电脑的 `:memory:.ses` 与本任务无关，不在分支中。

新电脑首次打开后先执行：

```bash
git branch --show-current
git merge-base 7d3cc4bebd1efe951e0adc38c053d417260867c0 HEAD
git status --short
```

第一条应为 `codex/jumper-push-handoff`，第二条应为基线 `7d3cc4bebd1efe951e0adc38c053d417260867c0`，第三条在刚克隆时应为空。不要在确认分支与产物完整前运行 `git reset`、`git clean` 或覆盖式 checkout。

## 3. 推荐机器与软件边界

### 3.1 推荐选择

- **长时间强化学习训练**：优先 Ubuntu 22.04/24.04 + NVIDIA GPU。仓库实测同一任务在 RTX 5090 上的 Warp/CUDA 吞吐约为 native CPU 的 32 倍；项目没有给出统一的最低显存门槛，应通过 `--num_envs` 调整负载。
- **开发、复现、视频和小规模微调**：Apple Silicon/macOS 可以使用 `native:cpu`，当前成果就是在 Mac CPU 上完成；不能使用 CUDA。
- **Windows**：支持，但 Warp JIT 和 Rust 需要 Visual Studio Build Tools 的 “Desktop development with C++”。

当前旧电脑的已验证版本，仅作复现参考，不要求手工逐个钉死：

```text
macOS 26.5.1 arm64
Python 3.12.15
PyTorch 2.14.1 (CPU)
MuJoCo 3.11.0
MuJoCo Warp 3.11.0
Warp 1.17.0
mediapy 1.2.7
imageio-ffmpeg 0.6.0
```

### 3.2 必需软件

- Git。
- Python 3.10–3.13；Python 3.14 不支持。
- Python 标准库 `venv`，环境固定放在仓库根目录 `.venv/`；不要使用现有 Conda 或 Isaac Lab 环境。
- PyTorch：NVIDIA 机器使用与显卡匹配的 CUDA wheel；无 NVIDIA GPU 时使用普通 CPU wheel。
- MuJoCo、MuJoCo Warp、Warp、rsl_rl 依赖、TensorBoard、mediapy、imageio-ffmpeg 等由 `pip install -e .` 安装。
- 若要运行完整测试或部署控制器：Rust stable（rustc ≥ 1.85）、`wasm32-unknown-unknown`、与 `deploy/fsm/Cargo.lock` 完全一致的 `wasm-bindgen-cli`、C/C++ linker、CycloneDDS 11.0.1、libclang、Rockchip RKNPU2 header。

不需要单独安装商业 MuJoCo 应用或许可证。Python `mujoco` wheel 已包含仿真运行库；视频编码由 `imageio-ffmpeg` 提供。

### 3.3 明确禁止

- 不要 `pip install mjlab` 或 `pip install rsl-rl-lib`；仓库使用 `rl/` 下的可修改 vendored 版本。
- 不要复用 Isaac Lab 环境；其 MuJoCo 版本可能冲突。
- macOS 不要尝试 `warp:cuda`；训练使用 `--backend native --device cpu`。
- 不要用 `warp:cpu` 做训练；它是串行调试设备，CPU 训练应使用 `native:cpu`。
- 不要把 `DEEPSEEK_API_KEY`、其他 token、`.env` 或用户级供应商配置提交到仓库。

## 4. 新电脑安装流程

仓库的权威流程是 `docs/AGENT_SETUP.md` 和 `.claude/skills/setup-env/SKILL.md`。以下是可执行摘要；出现差异时以权威文档和检测脚本输出为准。

### 4.1 检测平台

```bash
cd /absolute/path/to/jumper
python3 .claude/skills/setup-env/scripts/detect.py
```

同时确认 GPU：

```bash
nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader
```

没有 `nvidia-smi` 或命令报错，就走无 GPU 分支。

### 4.2 创建独立 Python 环境

macOS 若没有合规 Python：

```bash
brew install python@3.12
python3.12 -m venv .venv
```

Ubuntu 22.04/24.04 通常可直接：

```bash
python3 -m venv .venv
```

Linux/macOS 激活并检查：

```bash
source .venv/bin/activate
python -c "import sys; print(sys.prefix); print(sys.version.split()[0])"
```

Windows PowerShell：

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -c "import sys; print(sys.prefix); print(sys.version.split()[0])"
```

`sys.prefix` 必须指向当前仓库内的 `.venv`。

### 4.3 安装 PyTorch 与项目

NVIDIA GPU（Blackwell 必须 cu128 或更新；Ada/Ampere 可用 cu126 或更新）：

```bash
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
python -m pip install -e .
```

无 NVIDIA GPU（包括所有 macOS）：

```bash
python -m pip install torch torchvision
python -m pip install -e .
```

若此环境曾安装 PyPI 版本：

```bash
python -m pip uninstall -y mjlab rsl-rl-lib
python -m pip install -e .
```

### 4.4 Rust、控制器与完整测试依赖

只做 Python 仿真可以先不部署控制器，但完整仓库检查需要以下工具。

macOS 先装 linker：

```bash
xcode-select --install
```

Ubuntu：

```bash
sudo apt install -y build-essential cmake git libclang-dev
```

Linux/macOS 安装 Rust：

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal
. "$HOME/.cargo/env"
rustup target add wasm32-unknown-unknown
cargo install wasm-bindgen-cli --version 0.2.128 --locked
```

`wasm-bindgen` 版本必须来自新电脑 checkout 的 `deploy/fsm/Cargo.lock`；如果 lock 已更新，不要机械沿用上面的 0.2.128。

CycloneDDS 11.0.1（以下命令为 Linux 已实测路线）：

```bash
git clone --depth 1 --branch 11.0.1 https://github.com/eclipse-cyclonedds/cyclonedds.git /tmp/cyclonedds
cmake -S /tmp/cyclonedds -B /tmp/cyclonedds/build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX=/usr/local \
  -DBUILD_TESTING=OFF -DBUILD_EXAMPLES=OFF -DBUILD_IDLC=ON
cmake --build /tmp/cyclonedds/build -j
sudo cmake --install /tmp/cyclonedds/build
sudo ldconfig
```

每个 checkout 还要获取固定版本的 Rockchip header：

```bash
bash deploy/fsm/vendor/rknpu2/fetch.sh
```

macOS/Windows 的 CycloneDDS 设备构建没有在仓库文档中完成同等实测。只做训练与 MuJoCo 验证时，可先完成 Python gates；若要完整部署构建，建议使用 Ubuntu。

## 5. 环境六门与 smoke test

先运行自动检查：

```bash
.venv/bin/python .claude/skills/setup-env/scripts/gates.py
```

六门分别验证：Warp 设备、native MuJoCo rollout、vendored 包来源、任务注册、Rust host targets、DDS device build。也可逐项执行：

```bash
python -c "import warp as wp; wp.init(); print(wp.get_devices())"
python -c "import mujoco; from mujoco import rollout; print('mujoco', mujoco.__version__)"
python -c "import mjlab,rsl_rl,os; print(os.path.dirname(mjlab.__file__)); print(os.path.dirname(rsl_rl.__file__))"
python scripts/train.py --list
cargo check --locked --lib --manifest-path deploy/fsm/Cargo.toml --target wasm32-unknown-unknown --no-default-features --features web
cargo check --locked --lib --manifest-path deploy/fsm/Cargo.toml --no-default-features --features py
cargo test --locked --lib --manifest-path deploy/fsm/Cargo.toml -- --list
```

CPU smoke test：

```bash
python scripts/train.py --task jumper.flat --backend native --device cpu \
  --num_envs 64 --max-iterations 3 --headless --no-tensorboard
```

NVIDIA GPU smoke test：

```bash
python scripts/train.py --task jumper.flat --num_envs 256 \
  --max-iterations 5 --headless --no-tensorboard
```

macOS 图形窗口必须使用 `mjpython`：

```bash
.venv/bin/mjpython scripts/play.py --task jumper.five_foot \
  --checkpoint logs/jumper/jumper.five_foot/2026-10-21_22-52-37/model_87198.pt
```

普通 `python` 可用于 headless 训练、测试和视频生成。首轮 Warp kernel 编译耗时数分钟属于正常现象。

## 6. 复现当前结果

> 2026-10-25 修订：旧基线 7/8 中的唯一失败已诊断为验证器命令随机重采样（见上文“已经成立的证据”），
> 验证器已修复。直接用下面第 6.1 节的 20 次命令重新建立基线；历史 `_trials2_80mm` JSON 与视频保留为
> 问题证据，不删除、不覆盖。旧 7/8 数字不再用于评价策略的 seed 鲁棒性。

先运行静态与单元测试：

```bash
python -m ruff check \
  tasks/jumper/five_foot/__init__.py \
  tasks/jumper/five_foot/claw.py \
  tasks/jumper/five_foot/env_cfg.py \
  tasks/jumper/five_foot/mdp/contact_command.py \
  tasks/jumper/five_foot/mdp/grasp.py \
  tasks/jumper/five_foot/mdp/terminations.py \
  tasks/jumper/five_foot/objects.py \
  tasks/jumper/five_foot/rl_cfg.py \
  tasks/jumper/five_foot/tools/contact_training_probe.py \
  tasks/jumper/five_foot/tools/grasp_objects.py \
  tasks/jumper/five_foot/tools/grasp_pose.py \
  tasks/jumper/five_foot/tools/push_objects.py \
  tasks/jumper/five_foot/tools/vel_check.py

python -m pytest -q \
  tests/test_five_foot_objects.py \
  tests/test_five_foot.py \
  tests/test_five_foot_payload.py \
  tests/test_five_foot_pose_command.py \
  tests/test_curriculum_resume.py

python -m pytest -q tests/test_layout.py
git diff --check
```

旧电脑最后一次结果为前一组 134 tests + `test_layout.py` 15 tests 全部通过，合计 149；新电脑应达到同样结果。

然后对四个 seed 各做五次 80 mm 复测。逐条执行，以便任何失败都保留对应 JSON：

```bash
python tasks/jumper/five_foot/tools/push_objects.py \
  --checkpoint logs/jumper/jumper.five_foot/2026-10-21_22-52-37/model_87198.pt \
  --object push_box --orientation upright --trials 5 --seed 0 \
  --distance-m 0.08 --speed 0.35 --initial-gap-mm 0 \
  --shoulder-offset-rad 0.15 --backoff-m 0.05 \
  --backoff-lateral-speed -0.15 \
  --json tasks/jumper/five_foot/out/push_box_87198_seed0_trials5_80mm_fixed.json
```

把 `--seed 0` 和输出文件名依次改成 `1`、`2`、`42`。这个验证器在失败时以非零状态退出，但仍会写 JSON；不要因一次失败而丢弃记录或自动重试。逐控制步诊断记录用 `push_diagnose.py`（同一参数，`--out` 指向独立目录）生成，供命令有效步数、首次接触和持续接触帧核对。

生成可观看视频：

```bash
python tasks/jumper/five_foot/tools/push_objects.py \
  --checkpoint logs/jumper/jumper.five_foot/2026-10-21_22-52-37/model_87198.pt \
  --object push_box --orientation upright --trials 5 --seed 0 \
  --distance-m 0.08 --speed 0.35 --initial-gap-mm 0 \
  --shoulder-offset-rad 0.15 --backoff-m 0.05 \
  --backoff-lateral-speed -0.15 \
  --json tasks/jumper/five_foot/out/push_box_87198_seed0_trials5_80mm_fixed.json \
  --video tasks/jumper/five_foot/out/push_box_87198_seed0_trials5_80mm_fixed.mp4 \
  --video-trial 1
```

视频验证同样是 native MuJoCo 中的物理回放，不是录制动画，也不代表已实现视觉定位或自主接近。

## 7. 恢复训练与下一阶段

当前 checkpoint 对应的训练配置使用：

- `--grasp-ready-hold`
- `--contact-push-box`
- 64 environments
- native CPU（旧电脑）
- PPO 固定初始动作标准差 0.01
- reset 后停稳 50 控制步，+0.35 m/s 起步脉冲 20 步，再保持 +0.25 m/s

从当前 checkpoint 再训练 50 次更新的 CPU 命令：

```bash
python scripts/train.py \
  --task jumper.five_foot \
  --backend native --device cpu --num_envs 64 \
  --grasp-ready-hold --contact-push-box \
  --checkpoint logs/jumper/jumper.five_foot/2026-10-21_22-52-37/model_87198.pt \
  --max-iterations 50 --headless
```

NVIDIA 机器可省略 `--backend native --device cpu`，让解析器选 `warp:cuda`，并逐步增加 `--num_envs`；先以 64 做一次短 smoke run，确认 reward、termination 和 checkpoint 恢复一致，再扩大并行数。

`--checkpoint` 会恢复 actor、critic、optimizer、iteration 和环境 step，而不是只载入权重；新 run 会写到新的时间戳目录。不要用只加载 actor 的验证代码代替训练恢复。

### 7.1 优化顺序

1. ✅ 已完成：用 `model_87198.pt` 重新得到固定中心点基线（未改验证器门槛）。
2. ✅ 已完成：seed 0 失败诊断为验证器命令随机重采样，而非接触建立、随机化、姿态或 reset 问题；逐控制步证据在 `tasks/jumper/five_foot/out/diag/`。未采用任何验证时瞬移、吸附、外力、焊接或隐藏重试；训练分布无需为此改动。
3. ✅ 已完成：修复验证器命令生命周期后，`model_87198.pt` 在 seed 0/1/2/42 × 5 次固定中心点 80 mm 为 20/20 成功，摔倒 0、非足触地 0、无 retry——达到原晋级门槛（19/20、每 seed 4/5）。新基线 JSON 为 `push_box_87198_seed{0,1,2,42}_trials5_80mm_fixed.json`。
4. ✅ 已完成：100 mm 固定中心点（仅 `--distance-m 0.10`，其余参数与 80 mm 基线完全一致）为 20/20 成功、摔倒 0、非足触地 0；JSON 为 `push_box_87198_seed{0,1,2,42}_trials5_100mm_fixed.json`，视频 `push_box_87198_seed0_trials5_100mm_fixed.mp4`。下一步以同一参数只改 `--distance-m 0.12` 测试 120 mm，稳定后再试 150 mm；每个距离都重新检查停止、倾斜、撤离和道具回拖。
5. 长距离固定中心点跨 seed 通过后，给验证器增加显式 X/Y/yaw 初始偏置参数，执行 3×3×3 网格。不能用 reset 后的持续位姿写入夹具修正误差。
6. 网格通过后再接入接近与定位；在此之前结果只应称为“已知位姿固定点接触推送”。

### 7.2 必须保留的安全守卫

- 推送前机身线速度、角速度和 roll/pitch 稳定门限。
- 至少 3 个持续接触控制帧。
- 道具位移达到目标的 90%。
- 横向偏差、最大倾角、稳定停车。
- 抬爪时道具额外移动不超过 15 mm。
- 撤离距离达到目标的 90%，撤离期间道具拖动不超过 10 mm。
- 摔倒为 0；大于 1 N 的非足触地为 0。
- 每一 trial 独立 reset；失败结果写入 JSON，不能只保留成功视频。

### 7.3 禁止改变实验含义

以下若用于剧情展示，必须在界面和报告中标注“剧情辅助交互”，不得混入物理成功率：

- 吸附或磁吸约束。
- weld/equality attachment。
- 接触后让道具跟随机器人。
- reset 以后持续写入道具根状态。
- 直接施加外力推动。
- 失败后隐藏式自动重试或只剪出成功 trial。

## 8. DeepSeek API 与编码代理

当前 Codex 桌面任务不能在运行中无缝切换成 DeepSeek。最稳妥的方式是：在新电脑使用一个原生支持 DeepSeek API、能够读写本地仓库并执行 shell 的编码代理，打开迁移后的同一目录继续。

密钥只放在用户环境或系统钥匙串：

```bash
export DEEPSEEK_API_KEY="<your-key>"
```

不要把真实 key 写入本文件、`.env`、项目 `.codex/config.toml`、shell 历史示例、日志或截图。若客户端支持从环境变量读取，供应商配置也放在用户目录。

如果尝试使用 Codex CLI 的自定义 provider，需要注意：Codex 的 provider 配置是用户级配置，项目内 `.codex/config.toml` 会忽略 `model_provider` / `model_providers` 等敏感项；且 Codex provider 需要客户端所支持的 wire API。不要假设只兼容 OpenAI Chat Completions 的 DeepSeek endpoint 一定能直接替代 Responses API。优先选用明确声明支持 DeepSeek tool calling、shell 和补丁编辑的客户端；先用一个只读问题和一个临时文件补丁验证工具调用，再允许它改本仓库。

### 8.1 复制给新编码代理的首条提示词

```text
你正在接管 Jumper 的五足真实接触推送优化。工作目录是本仓库根目录。

在任何修改前，完整阅读：
1. AGENTS.md
2. CLAUDE.md
3. docs/WORKFLOWS.md
4. docs/AGENT_SETUP.md
5. HANDOFF_DEEPSEEK.md
6. docs/HARDWARE.md
7. docs/CONTROLS.md
8. tasks/jumper/five_foot/README.md

先执行 git branch --show-current、
git merge-base 7d3cc4bebd1efe951e0adc38c053d417260867c0 HEAD、git status --short
和 setup-env gates。当前分支应为 codex/jumper-push-handoff，与 origin/main 的共同基线应为
7d3cc4bebd1efe951e0adc38c053d417260867c0，刚克隆的工作树应干净。禁止 reset、
clean、覆盖已有成果或顺手整理无关文件。

主路线已经从不可靠夹取切换为自由刚体推送。只允许 MuJoCo 碰撞与摩擦移动 120×100×70 mm、0.188 kg 的 push_box；不得使用吸附、weld、跟随、外力、接触后瞬移或隐藏重试。model_87198.pt 在 seed 0/1/2/42 各 2 次、80 mm 固定中心点的历史结果 7/8 中的唯一失败已诊断为验证器命令随机重采样（twist 项 `resampling_time_range` 未钉），不是策略缺陷；验证器修复后四 seed × 5 次为 20/20 成功，是新的正式基线。历史 JSON/视频保留为问题证据，不删除、不覆盖。

不要马上运行 3×3×3 网格。固定中心点 80 mm 已在修复后的验证器上达到 20/20（每 seed 5/5）、摔倒与非足触地均为 0；接下来依次验证 100、120、150 mm。每次报告 checkpoint、训练参数、物品尺寸/质量、MuJoCo 参数、每 trial JSON、成功率、滑移/倾斜/停止/撤离、摔倒和失败原因，并生成至少一个完整 MuJoCo MP4。

开始时只做读取、环境检查和基线复现；给出证据后再做最小代码修改。编辑遵守 CLAUDE.md，使用 apply_patch，保留所有失败记录，完成后运行相关 Ruff、pytest 和 git diff --check。
```

## 9. 关键文件导航

| 文件 | 作用 |
|---|---|
| `tasks/jumper/five_foot/tools/push_objects.py` | 固定已知位姿的真实刚体推送验证与 MP4 输出 |
| `tasks/jumper/five_foot/mdp/contact_command.py` | 接触训练的起步/持续速度命令 |
| `tasks/jumper/five_foot/mdp/grasp.py` | 道具 reset、闭爪/推面事件和接触阶段逻辑 |
| `tasks/jumper/five_foot/objects.py` | `PUSH_BOX` 尺寸、质量、摩擦和训练实体 |
| `tasks/jumper/five_foot/env_cfg.py` | `--contact-push-box` 训练夹具、传感器、reward/termination 接线 |
| `tasks/jumper/five_foot/rl_cfg.py` | 接触微调 PPO 参数 |
| `tasks/jumper/five_foot/claw.py` | 抓取准备姿态、夹爪限位与关节常量 |
| `tasks/jumper/five_foot/out/grasp_validation_summary.md` | 所有候选、失败和结论的时间线 |
| `docs/HARDWARE.md` | 真实硬件能力与未验证边界 |
| `docs/CONTROLS.md` | 控制 vocabulary、模式与中断约定 |
| `deploy/manifests.json` | 部署模式及控制绑定 |

## 10. 接管完成判据

只有以下项目全部完成，才开始新的训练优化：

- [ ] 基线 commit 与脏工作树匹配。
- [ ] `model_87198.pt`、四个主 JSON、视频和实验总结均已复制。
- [ ] SHA-256 与旧电脑一致。
- [ ] 新 `.venv` 使用 Python 3.10–3.13，未复用 Conda/Isaac Lab。
- [ ] `mjlab`、`rsl_rl` 均从本仓库 `rl/` 导入。
- [ ] setup-env gates 对当前用途需要的项目通过；未安装的部署工具被明确记录，而不是静默跳过。
- [ ] CPU 或 GPU smoke training 通过。
- [ ] 149 个相关测试与 Ruff、`git diff --check` 通过。
- [ ] 四 seed 的 80 mm 基线已复测并保留全部 JSON。
- [ ] 成功视频可以正常播放，且明确标注为 MuJoCo 真实接触仿真。

迁移或环境结果若与本文不一致，先停止训练并记录差异；不要通过放宽成功门槛、关闭传感器或引入辅助交互来“修复”结果。
