<!-- tracks: REMOTE_TRAIN.md @ sha256:85bca8d0d25e1d8d -->

# 通过 ssh 远程训练

[主页](../README.md) · [English](REMOTE_TRAIN.md)

没有 GPU 的笔记本也能训练：这一页讲的驱动会把本仓库的一份检出复制到 GPU 机器上，
在那边运行仓库自己的 `scripts/train.py`，再把 checkpoint 和它的证据带回来。

```bash
python tools/remote_train.py check
python tools/remote_train.py sync
python tools/remote_train.py submit --task jumper.flat --release <id> --num-envs 64 --max-iterations 20
python tools/remote_train.py status <job_id>
python tools/remote_train.py fetch <job_id>
```

驱动是 `tools/remote_train.py`。它**不是**第二个训练入口：它只负责给
`scripts/train.py` 拼参数，从不 import 任务、仿真器或 torch。它关于一次训练运行的
全部信息，都来自那次运行留下的文件。

## 它负责什么，不负责什么

| 负责 | 不负责 |
|---|---|
| 复制一个 release，并逐字节证明内容已到达 | 每次提交前自动同步（一个任务只跑一个指定的 release） |
| 在 tmux 里启动 `train.py`，ssh 断开后继续运行 | 在服务器重启后保活；tmux 做不到，checkpoint 可以 |
| 用退出码**和**磁盘上的 checkpoint 共同判定 `succeeded` | 把 tmux 会话消失当成成功 |
| 下载结果并校验 SHA-256 | 删除服务器上的结果或释放实例 |
| GPU 已被别的任务占用时拒绝提交 | 杀掉不是它启动的进程 |
| `stop` 依次 SIGINT → TERM → KILL | 承诺停止时一定保存 checkpoint（见[限制](#限制)） |

## 环境要求

本机（Mac）：

- 一把服务器接受的 ssh 密钥 —— 驱动用 `BatchMode=yes` 运行 `ssh`，需要密码时会
  失败而不是挂住；
- 本仓库的 `.venv`（任意 3.10–3.13 且装了 `dev` extra 即可运行驱动：它只用标准库）；
- 上传用的 `tar`（macOS 上是 bsdtar）和下载用的 `rsync`。

服务器：

- `tmux`、`rsync`、Python 3.10–3.13，以及 `nvidia-smi` 能报出的 NVIDIA 驱动；
- 一个可写的工作目录。AutoDL 上要放数据盘 `/root/autodl-tmp/...`，不要放 `/root`：
  系统盘很小，而 `.venv` 有好几个 GB。

AutoDL 上的两个具体问题：镜像里往往有 `/root/miniconda3/bin/python`，但
**非交互 PATH 里没有 `python3`**，所以配置里的 `python` 要写绝对路径；`tmux` 需要
先 `apt-get install -y tmux` 装一次。镜像自带的 torch（本次实例是 2.8+cu128）通过
`--system-site-packages` 复用，省掉几个 GB 的下载；`check` 会在提交任何任务之前
验证 torch 确实看得见 CUDA。

## 配置

```bash
cp tools/remote_train.example.toml .remote.toml   # 已被 git 忽略
```

```toml
host = "autodl"                    # ssh 别名，或者：
# hostname = "connect.xxx.seetacloud.com"
# port = 24227
# user = "root"
# identity = "~/.ssh/id_ed25519"

workdir = "/root/autodl-tmp/jumper"
python = "/root/miniconda3/bin/python"
results = "~/jumper-runs"
max_hours = 6.0
# AutoDL：国内访问默认 PyPI 很慢。`network_turbo` 代理的是 GitHub，反而会让 pip
# 更慢，所以这里显式配镜像源。
pip_index = "https://pypi.tuna.tsinghua.edu.cn/simple"

backend = "warp"
device = "cuda:0"
num_envs = 64
tensorboard_port = 6006
# mjlab 默认把 MUJOCO_GL 设成 egl。没有 libEGL 的机器（本次这台 RTX 3080 Ti）
# 在 import 阶段就会失败；"disabled" 表示不渲染、只训练。
mujoco_gl = "egl"
```

查找顺序是 `--config`、`$JUMPER_REMOTE_CONFIG`、`./.remote.toml`，最后是
`~/.config/jumper/remote.toml`。**这个文件里没有秘密**：`identity` 是路径。不要把
密码或 API token 写进去，也不要写进 `.env` —— 本仓库的 `.env` 是提交进版本库的。

只有密码、没有密钥时，请你自己装一次公钥，不要把密码交给工具：

```bash
ssh-copy-id -i ~/.ssh/id_ed25519.pub -p 24227 root@connect.xxx.seetacloud.com
```

## 命令

| 命令 | 回答什么问题 |
|---|---|
| `check` | 能否连上、磁盘够不够、GPU 是否空闲、torch 能否看见 CUDA、`mjrl`/`mjlab`/`tasks` 能否 import。退出码 1 表示未就绪，2 表示连不上。 |
| `sync [--rev REF] [--label ID]` | 把这份检出（或某个 git revision）复制进 `releases/<id>`，并逐文件校验。 |
| `bootstrap --release ID` | 在服务器上创建 `.venv` 并对该 release 执行 `pip install -e`。 |
| `submit --task T --release ID` | 在 tmux 里启动任务，打印 `job_id`。 |
| `status [JOB] [--json]` | 状态、迭代数、最新 checkpoint、显存、日志末尾。 |
| `logs JOB [-f] [-n N]` | 查看或跟踪日志。Ctrl-C 只停止跟踪，不停止训练。 |
| `fetch JOB [--all]` | 下载证据：最终 checkpoint、`meta.json`、状态、日志、评估和事件文件，并校验 SHA-256；`--all` 下载全部 checkpoint。基于 `--partial`，可中断续传。 |
| `eval JOB [--steps N] [--metrics]` | 用 headless 回放加载最新 checkpoint 跑 N 步（`play.py --steps`）；`--metrics` 调用 `tools/replay_eval.py`，测站立、直行、转向的摔倒率、存活步数和跟踪误差。 |
| `resume JOB [--max-iterations N]` | 用同一个 release 和 checkpoint 新建任务，迭代计数接着走。 |
| `stop JOB` | 对进程组发 SIGINT，等待，再 TERM，最后 KILL。 |
| `list` | 汇总最近的任务。 |

第一次端到端：

```bash
python tools/remote_train.py check
python tools/remote_train.py sync
python tools/remote_train.py bootstrap --release <sync 打印的 id>
python tools/remote_train.py submit --task jumper.flat --release <id> \
    --num-envs 64 --max-iterations 20 --max-hours 1
# 关掉终端，稍后回来
python tools/remote_train.py status <job_id>
python tools/remote_train.py fetch <job_id>
python tools/remote_train.py eval <job_id> --steps 200
```

`submit --dry-run` 只打印任务元数据和生成的 `job.sh`，不碰服务器。

## 文件落在哪里

服务器上，`workdir` 之下：

```
releases/<release_id>/     不可变的代码快照，另有 CODE_VERSION.json、CODE.patch、
                           MANIFEST.sha256；它的 logs/ 是指向当前任务目录的软链接
runs/<job_id>/             meta.json、job.sh、status.tsv、pid、logs/train.log、
                           tree/（本次运行的 logs/<model>/<task>/<timestamp>）、eval/、
                           root/（指向 release 的软链接，外加自己的 logs/）
.venv/                     一个环境，指向正在运行的 release
```

本机，`results` 之下：

```
<results>/<job_id>/        任务目录原样，另有 SHA256SUMS 和 FETCH.json
<results>/<job_id>/eval/   eval_report.json 和回放日志
```

**两个任务不会共用同一个日志目标。** 每个任务在 `runs/<id>/root/` 里运行：这个目录
用软链接指向 release，旁边是一个真实的 `logs/`；release 本身不再挂 `logs` 软链接，也就
不会被第二次提交重新指走。`--concurrent` 下第一个任务的 checkpoint、`status`、`logs`、
`stop` 因此各自独立。

**任务运行期间，它用的代码不会被改动。** `submit` 把 release 装进共享的 `.venv`，
然后检查 `mjrl`、`mjlab`、`rsl_rl`、`tasks` 实际从哪里 import；对不上就中止提交。
之后的 `sync` 写的是**新的** release，正在跑的任务还是它启动时的那份字节。

## 状态

| 状态 | 含义 |
|---|---|
| `queued` | 任务目录已存在，还没有 start 记录。 |
| `running` | 包装脚本的 pid 还活着，且没有 end 记录。 |
| `succeeded` | end 记录为 `rc=0`，**并且**磁盘上至少有一个 `model_*.pt`。 |
| `failed` | 非零退出、超时（rc 124），或 `rc=0` 但没有 checkpoint。 |
| `stopped` | `stop` 标记之后产生的 end 记录。 |
| `unknown` | 有 start 记录、没有 end 记录、进程已消失 —— 重启、崩溃或 `kill -9`。永不报成成功。 |

ssh 连不上时打印 `connection error` 并以 2 退出，绝不写成 `failed` —— 链路断了的时候，
训练可能正在对面正常进行。

## 可复现性

- release id 是时间戳加 git commit；工作区脏时再加补丁哈希（`-dirty-<8 位十六进制>`）。
- `CODE_VERSION.json` 记录 commit、分支、`git status`、未跟踪文件清单和补丁哈希；
  `CODE.patch` 是实际 diff。
- `MANIFEST.sha256` 在复制完成后于服务器上计算，再和本机逐项比对；缺失或不同就中止同步。
- `FETCH.json` 和 `SHA256SUMS` 记录下载了什么、摘要是什么。

## 限制

- **停止不是优雅保存。** rsl_rl 每 `save_interval` 次迭代（默认 50）存一次，并在完整
  跑完后存一次；它不在 SIGINT 时保存。`stop` 会丢失上次 checkpoint 之后的迭代，命令
  自己会说明这一点。想要更小的损失，就调小任务 `rl_cfg.py` 里的 `save_interval`。
- **默认一张 GPU 只跑一个任务。** 别的任务 pid 还活着时 `submit` 会拒绝；未知进程
  占用超过 512 MiB 时也会拒绝，并且不会去杀它们。`--concurrent` 允许第二个受管任务
  （硬上限两个）；`--force` / `--allow-busy` 是显式覆盖。
- **全量下载是多条小流。** 每个 rsync 是一条 SSH 连接；在本次实测链路上，单流
  0.36–0.49 MB/s，四条并行合计 1.48 MB/s，所以整段轨迹用并行下载（或实例网盘）
  比一条长 rsync 更快。默认 `fetch` 只取最终模型，也是同一个原因。
- **不自动关机、不自动清理。** 实例保持开机、结果留在上面；`fetch` 只复制。关机请自己做。
- **服务器重启会结束任务。** tmux 扛不过重启；任务显示 `unknown`，最后一个 checkpoint
  就是续训点。
- **本阶段不生成视频。** `eval` 只验证 checkpoint 能加载并 headless 回放 N 步，报告的是
  “有没有跑完”，不是“走得好不好”。任务媒体渲染仍由 `tools/readme_media.py` 和已提交的
  export 负责。
- **上传用 tar，下载用 rsync。** macOS 的 `/usr/bin/rsync` 是 openrsync 2.6.9，在这份
  检出上还没传一个字节就会报 `mkpath: Operation not permitted`，所以上传改成一个 tar 流。
  下载仍然用 rsync，因为可续传在这里更重要。
- **`fetch` 会跳过最近 10 秒内写过的 checkpoint**，并说明跳过了哪些；正在 `torch.save`
  的文件不是 checkpoint。

## 故障排查

| 症状 | 原因 |
|---|---|
| `Host key verification failed` | rsync 自己的 ssh 没拿到固定的 `known_hosts`；驱动传 `-e` 正是为此。 |
| `mkpath: Operation not permitted` | macOS 的 openrsync 2.6.9；用工具里的 tar 上传，不要手敲 `rsync`。 |
| `no .venv on the box` | `sync` 之后先跑 `bootstrap --release <id>`。 |
| `python3: command not found` | 把 `python` 指向镜像里的解释器，例如 `/root/miniconda3/bin/python`。 |
| `torch cannot see CUDA` | 镜像里的 torch 是 CPU 版，或驱动太旧；往 `.venv` 里装匹配的 cu 轮子。 |
| `AttributeError: 'NoneType' object has no attribute 'eglQueryString'` | 机器没有 libEGL：训练设 `mujoco_gl = "disabled"`，需要渲染设 `"osmesa"`。 |
| `the GPU already has processes` | 有别的任务在训练；工具不会杀它。 |
| `a job is already running` | `status` 能看到；先 `stop`，或换实例。 |
