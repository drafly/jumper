# 跳跳工程理解与 Mac 运行指南

核查日期：2026-10-03。代码版本：jumper `7d3cc4bebd1efe951e0adc38c053d417260867c0`；jumper-design `918d192857fcea1678c2a40b339501004a94d838`。以下基于实际源码、说明和当前 Mac 上的实机运行结果。

## 工程是什么

跳跳是一台 22 自由度的六足螃蟹机器人。这个工程提供机器人模型、强化学习任务、仿真训练与回放、策略导出和控制器打包，并不是下载后就能自主完成家务的成品应用。

| 部分 | 功能 | 关键入口 |
|---|---|---|
| 模型和任务 | 平地行走、三足/波纹/四足支撑步态、姿态、五足支撑与抓取、跳跃、舞蹈模仿、荡秋千等 | `assets/jumper/`、`tasks/jumper/` |
| 训练 | 使用 PPO；共享任务和奖励代码，在 NVIDIA Warp/CUDA 或原生 MuJoCo CPU 后端运行 | `scripts/train.py` |
| 回放 | 加载 checkpoint、零策略或随机策略；窗口查看、键盘/手柄交互、测量 | `scripts/play.py` |
| 导出与部署 | ONNX、Rockchip RKNN 转换、多模式 Rust 控制器和 `.app` 动作包 | `scripts/export.py`、`scripts/deploy.py`、`deploy/` |
| 外观和环境 | 完整机器人 `.skin`、场景 `.map`、验证和预览 | 独立的 `jumper-design` 仓库 |

Mac 使用 `--backend native --device cpu`。当前实现不支持把原生后端设备改为 Apple MPS；CUDA 路线不适用于 Mac。CPU 可以调试和训练，但不能据此承诺大型训练的耗时。

Train 当前没有 `.skin` 加载器；外观包应在 Design 侧检查和预览。导入 `.map` 只用于 Train 回放，不用于并行训练。两仓库机械参数可能有差异，不能直接认为 Design 模型可替换训练模型。项目指南亦明确：板端 RKNN 和真实电机总线闭环仍有未验证项；仿真完成不等于实机完成。

## Mac：先打开模拟器

以下在终端执行；假定已有 Homebrew 和 Git。建议 Python 3.12，Train 要求 3.10–3.13。

```bash
brew install python@3.12
git clone https://github.com/KingKongRobotics/jumper.git
cd jumper
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install torch torchvision
python -m pip install -e ".[dev]"
python scripts/train.py --list

.venv/bin/mjpython scripts/play.py \
  --task jumper.tripod --backend native --device cpu \
  --agent zero --num_envs 1
```

必须执行 editable install：内部 `mjrl`、`mjlab`、`rsl_rl` 包依赖它来解析。不要额外安装 PyPI 的 `mjlab` 或 `rsl-rl-lib`。Mac 图形窗口使用 `mjpython`；普通 `python` 启动可能自动退回无窗口运行。

`--agent zero` 不需要 checkpoint，只能用于观察机器人和环境，不代表机器人学会行走。官方仓库实际已附带 13 个 `.pt` checkpoint、ONNX 策略和示例 `.app`，位于 `tasks/jumper/*/out/example/` 和 `out/bundle_example/`。这些目录被忽略规则覆盖，但文件已提交到 Git；此前使用普通文件搜索得出的“没有权重”结论有误。默认回放只查日志目录，因此使用附带权重时必须显式指定 `--checkpoint`。

## Mac：直接回放官方权重（无需重新训练）

优先尝试支持三足行走和身体姿态控制的 `jumper.posture`：

```bash
cd /Users/longfeihan/Projects/jumper
source .venv/bin/activate
.venv/bin/mjpython scripts/play.py \
  --task jumper.posture --backend native --device cpu --num_envs 1 \
  --checkpoint tasks/jumper/posture/out/example/model_74800.pt
```

跳跃回放：

```bash
.venv/bin/mjpython scripts/play.py \
  --task jumper.jump --backend native --device cpu --num_envs 1 \
  --checkpoint tasks/jumper/jump/out/example/model_3100.pt
```

不要添加 `--agent zero`，也不能把 posture 权重加载到 tripod 任务。任务的观察维度与网络结构必须匹配。以上路径已经在当前 Mac 上完成物理回放和图形窗口测试。

| 任务 | checkpoint（仓库相对路径） |
|---|---|
| 三足行走和身体姿态 | `tasks/jumper/posture/out/example/model_74800.pt` |
| 五足支撑/抓取 | `tasks/jumper/five_foot/out/example/model_86600.pt` |
| 跳跃 | `tasks/jumper/jump/out/example/model_3100.pt` |
| 无参考跳跃 | `tasks/jumper/ref_free_jump/out/example/model_5999.pt` |
| 舞蹈 | `tasks/jumper/dance/out/example/model_2300.pt` |
| 巴西舞、迷核舞、梦的翅膀、腰部舞 | `tasks/jumper/dance_{brazilian,maze,dream_wings,waist}/out/example/model_1999.pt`（按实际任务替换目录） |
| 挥手、作揖、伸爪、敬礼 | `tasks/jumper/gesture_{hello,bow,paw,salute}/out/example/model_999.pt`（按实际任务替换目录） |

示例动作包是 `out/bundle_example/jumper.app`，含 ONNX 和多模式控制器；Mac 使用 `.pt` 回放较少涉及控制器编译。

### 当前 Mac 的实际运行表现

`jumper.posture` 的 checkpoint 已能正常加载，MuJoCo 窗口、策略推理和键盘接管均正常。使用者已经逐项测试本指南提供的基础键盘指令，机器人能够按指令运动；终端出现 `taken over by the keyboard` 时表示键盘已经成功接管。一次前进测试中，终端报告实际速度 `0.81 m/s`、目标速度 `0.94 m/s`，说明策略能够跟随速度指令。

当前限制是 CPU 和图形渲染性能，而不是控制失效。该策略的控制频率是 200 Hz：图形窗口使用 `--physics-hz 200 --viewer-fps 10 --no-viewer-ui --speed 0` 时，实测约为 37–61 个控制步/秒，即 `0.19–0.31x real time`。因此屏幕上看起来是慢动作，终端中的速度仍是仿真时间内的机器人速度。macOS 偶尔输出 `error messaging the mach port for IMKCFRunLoopWakeUpReliable`，本次运行中未影响键盘控制或仿真，可作为输入法框架警告忽略。

需要流畅观察时可以使用便宜的场景、降低窗口刷新率，并用 `[` / `]` 切换到 onboard 相机以取消左上角的第二视角渲染：

```bash
.venv/bin/mjpython scripts/play.py \
  --task jumper.posture --backend native --device cpu --num_envs 1 \
  --checkpoint tasks/jumper/posture/out/example/model_74800.pt \
  --physics-hz 200 --scene plain --viewer-fps 5 --no-viewer-ui --speed 0
```

`--physics-hz 200` 把每个控制周期的物理子步从五次降为一次，保持 200 Hz 策略控制频率不变，但会降低接触仿真的精度，只适合交互和功能检查。正式物理验证应删除该参数，使用默认 1000 Hz，并优先采用 `--headless --speed 0 --measure`；低于 `1.0x real time` 只表示计算速度低于实时，不代表策略运行错误。

## Mac：训练后回放

先做短流程检查：

```bash
python scripts/train.py --task jumper.tripod \
  --backend native --device cpu --num_envs 64 \
  --max-iterations 3 --headless
```

三次迭代只检查训练流程，不足以证明稳定步态。实际训练去掉 `--max-iterations 3`；内存吃紧时可以降低 `--num_envs`。训练产物进入 `logs/<model>/<task>/<timestamp>/`。

```bash
.venv/bin/mjpython scripts/play.py \
  --task jumper.tripod --backend native --device cpu --num_envs 1
```

默认寻找该任务最新 checkpoint；指定文件可再加 `--checkpoint /absolute/path/model_4999.pt`。键盘操作以 `docs/CONTROLS.md` 和任务 `controls.yaml` 为准，主要使用 WASD/方向键；MuJoCo 窗口快捷键可能有重叠。仓库列出的手柄后端是 Linux evdev 和 Windows XInput，Mac 宜先用键盘。

## Mac：加载外观包并检查

Design 使用独立虚拟环境，不要安装进 Train 的环境。

```bash
deactivate
cd ..
brew install git-lfs
git lfs install
git clone https://github.com/KingKongRobotics/jumper-design.git
cd jumper-design
git lfs pull
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[sim,dev]"

python scripts/shellflow.py verify-package \
  library/skins/jumper-original.skin \
  --profile robots/jumper/profile.json --mujoco

python scripts/shellflow.py import-package \
  library/skins/jumper-original.skin \
  --profile robots/jumper/profile.json \
  --destination outputs/original-preview

.venv/bin/mjpython scripts/preview_simulation.py \
  outputs/original-preview/assembled/jumper
```

最后一条是完整模型的静态外观检查；并没有加载行走策略。`git lfs pull` 获取真实模型和外观包，LFS 指针文件不能用来渲染。已有输出目录时换一个新目录，避免覆盖。

边牧 `.skin` 生成并验证后，可用同一流程替换上面的原版外观路径。直接对无策略的外观模型开启物理步进不保证站稳或行走。

## 完整开发与实机部署

以上是预览与训练路径，不是完整部署环境认证。控制器开发还需要 Rust、wasm32 target、与 Cargo.lock 匹配的 wasm-bindgen、CycloneDDS、libclang 和 Rockchip header；部分构建路径只在 Linux 测量过。

```bash
python3 .claude/skills/setup-env/scripts/detect.py
.venv/bin/python .claude/skills/setup-env/scripts/gates.py
```

按 `docs/AGENT_SETUP.md` 和 setup-env 技能逐项安装、验证；遇到失败先修复，不能把跳过的检查标为通过。不要将 Mac 预览成功表述成板端控制器已经可用。

## 边牧外观：已完成概念阶段，等待选择

保留六足结构、原关节、碰撞和惯性，仅设计显示外观。两种候选：

- A：机械边牧。低矮黑白壳体、白额纹、短耳朵和浅口鼻造型，机械结构露出较多。
- B：萌系边牧。更圆润的头壳、半立耳和更明确的白口鼻，角色感更强。

概念图不是已生成模型。Design 的 AGENTS.md 明确要求在新外观建模前先展示候选并等待用户明确选择。因此尚未生成三维边牧网格或 `.skin`，也没有修改原模型。选择后需记录确认与参考图哈希，制作完整机器人装配，检查正/侧/背面和彩色及灰模视图，再 `export-skin` 与 `verify-package --mujoco`。显示外观不等于可打印外壳，数字装配不证明实物适配。

## 本次实际验证

- jumper 仓库已恢复到 `7d3cc4bebd1efe951e0adc38c053d417260867c0`，版本已记录；环境使用 Python 3.12、PyTorch 2.14.1、MuJoCo 3.11.0 和 Warp 1.17.0 CPU，setup-env 的六项检查均通过。
- `scripts/train.py --list` 成功列出任务与场景；`jumper.flat` 使用 native CPU、64 个环境完成了三次短训练迭代，证明训练流程能够运行，但该短流程不代表策略已经收敛。
- `jumper.posture` 的附带 checkpoint 已完成无窗口测量和 Mac 图形窗口回放。基础键盘指令均可正常驱动机器人，策略能够运动并跟随速度目标。
- 默认 1000 Hz 物理配置的无窗口回放实测约 `0.38–0.41x real time`；图形窗口在 200 Hz 低精度交互配置下约 `0.19–0.31x real time`。当前已知问题是 CPU/渲染性能造成的慢动作，不是安装、权重加载或输入控制故障。
- 无窗口 500 步测量已生成 `tasks/jumper/posture/out/example/measure/measure.csv` 和 `measure.png`；测试期间策略速度主要约为 0.70–0.79 m/s，目标为 0.80 m/s。
- 已生成边牧候选概念图；三维建模等待选择。
- 复核发现并确认 13 个已提交到 Git 的 checkpoint 为真实二进制文件，约 5.8–10 MiB，亦确认了 ONNX 和 `.app` 的存在。

## 原始资料

- [jumper](https://github.com/KingKongRobotics/jumper)
- [项目指南](https://github.com/KingKongRobotics/jumper/blob/7d3cc4bebd1efe951e0adc38c053d417260867c0/docs/PROJECT_GUIDE.zh.md)
- [安装说明](https://github.com/KingKongRobotics/jumper/blob/7d3cc4bebd1efe951e0adc38c053d417260867c0/docs/AGENT_SETUP.md)
- [训练与设计边界](https://github.com/KingKongRobotics/jumper/blob/7d3cc4bebd1efe951e0adc38c053d417260867c0/docs/WORKFLOWS.md)
- [Design 规则](https://github.com/KingKongRobotics/jumper-design/blob/918d192857fcea1678c2a40b339501004a94d838/AGENTS.md)
- [Design 快速开始](https://github.com/KingKongRobotics/jumper-design/blob/918d192857fcea1678c2a40b339501004a94d838/docs/quickstart.md)
