# Coding Agent Demo

这是一个本地可运行的 Python `coding agent` 演示项目。它使用命令行作为唯一入口，在固定的 `demo_repo` 中执行一条受控流程：

`clarify -> plan -> read -> proposal -> edit -> test -> review`

当前版本是一个有边界的演示实现，不是通用代码代理。它支持两类预设任务：

- 为 Todo API 增加 `priority` 排序，并修复 `PATCH` 局部更新行为
- 为 Todo 查询辅助函数增加完成态筛选和不区分大小写搜索

## 目录结构

```text
.
├─ demo_repo/              # agent 操作的固定示例仓库
├─ src/coding_agent/       # 核心实现
├─ tests/                  # 项目测试
├─ run_agent.py            # 无需安装即可运行的入口脚本
├─ pyproject.toml
└─ README.md
```

## 运行环境

- Python `3.11+`
- Windows、macOS、Linux 均可，只要本机可执行 `python`
- 默认无第三方运行时依赖
- 如需真实模型调用，需自行配置 `QWEN_API_KEY`

## 快速开始

建议先创建虚拟环境，但这个项目不安装也能直接运行：

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
python -m unittest discover -s tests -q
python run_agent.py eval
```

macOS / Linux:

```bash
source .venv/bin/activate
python -m unittest discover -s tests -q
python run_agent.py eval
```

## 常用命令

重置示例仓库到基线状态：

```bash
python run_agent.py reset-demo
```

执行一次任务：

```bash
python run_agent.py run --task "Add priority sorting to the Todo API and fix the PATCH partial update bug."
```

离线评测所有预设任务：

```bash
python run_agent.py eval
```

生成静态报告：

```bash
python run_agent.py report --session-limit 5
```

查看最近会话：

```bash
python run_agent.py list --limit 10
```

查看单次会话摘要：

```bash
python run_agent.py show <session_id>
```

## 可选环境变量

- `QWEN_API_KEY`: 启用真实 Qwen 调用
- `CODING_AGENT_MODEL`: 模型名，默认 `qwen-plus`
- `QWEN_API_BASE`: 接口地址，默认 `https://dashscope.aliyuncs.com/compatible-mode/v1`
- `CODING_AGENT_RUNTIME_DIR`: 运行产物目录，默认 `runtime`
- `CODING_AGENT_DEMO_REPO`: 示例仓库路径，默认 `demo_repo`
- `CODING_AGENT_QWEN_TIMEOUT_SECONDS`: 请求超时秒数，默认 `45`
- `CODING_AGENT_QWEN_MAX_RETRIES`: 重试次数，默认 `2`
- `CODING_AGENT_QWEN_RETRY_BACKOFF_SECONDS`: 指数退避初始秒数，默认 `1`

这是我第一次上传代码，我想这是一个值得纪念的时刻，看看我们能在ai时代都留下些什么

说明：

- 未设置 `QWEN_API_KEY` 时，`clarify / plan / proposal` 会自动回退到离线逻辑
- `eval` 默认使用离线模式，不会主动调用真实模型
- 运行过程中会在 `runtime/` 下生成会话记录和报告文件
- 如果需要安装成系统命令，再执行 `python -m pip install -e .` 即可
