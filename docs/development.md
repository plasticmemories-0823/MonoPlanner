# 开发与维护

## 项目结构

| 目录 | 职责 |
|---|---|
| `web/` | 聊天界面、工作目录选择、模型设置与操作展示 |
| `agent/server/` | 本地接口与会话图 |
| `agent/cli/` | 交互命令、任务创建与仓库偏好 |
| `agent/task/` | 任务调度、操作契约与预算 |
| `agent/model/` | DeepSeek 请求、严格工具定义、前缀检查与压缩 |
| `agent/source/` | 源码快照、索引、区间原文与增量地图 |
| `agent/execution/` | 本地／Docker 执行与隔离调查 |
| `agent/acceptance/` | 验收试跑、冻结、修订与最终结果 |
| `agent/storage/` | 原始记录、检查点、会话与恢复 |
| `resources/` | Schema、提示词与运行驱动 |
| `eval/` | 通用评测、用量与成绩分析 |
| `tests/` | 功能回归测试；产品运行不加载 |
| `validation/` | 验证记录、历史评测与停止现场 |

会话图复用首次模型回复，分为聊天、资料查询、已有任务运行或既有工作推进。原有任务存储独占操作游标；LangGraph 保存公开会话和任务关联，不另建会重放操作的执行器。

## 开发启动

从项目根目录运行 `scripts/start-server.ps1` 和 `scripts/start-web.ps1`。后端默认不热重载；改动后正常停止并重新启动。前端可在 `web/` 中运行 `pnpm dev` 使用开发模式。

默认后端地址为 `http://127.0.0.1:2024`，图 ID 为 `monoplaner`。自定义前端连接设置见 `web/.env.example`；改端口时同步后端 CORS。不要将 DeepSeek 密钥放入 `NEXT_PUBLIC_*` 环境变量。

## CLI 与独立任务

```powershell
.\.venv-web\Scripts\python.exe -B -X utf8 main.py --repo 'E:/Projects/example'
.\.venv-web\Scripts\python.exe -B -X utf8 main.py --repo 'E:/Projects/example' --check-workspace
.\.venv-web\Scripts\python.exe -B -X utf8 -m agent --state-dir 'E:/AgentRuns/task-001' --repo 'E:/Projects/example' --task '任务描述'
.\.venv-web\Scripts\python.exe -B -X utf8 -m agent --state-dir 'E:/AgentRuns/task-001' --status
.\.venv-web\Scripts\python.exe -B -X utf8 -m agent --state-dir 'E:/AgentRuns/task-001'
```

交互命令包括 `/repo`、`/clear`、`/verify-command`、`/acceptance` 和 `/exit`。已有 `.venv` 仍可用于 CLI，不需要为界面升级原环境。

## 验证与评测

```powershell
.\.venv-web\Scripts\python.exe -B -X utf8 -m unittest discover -s tests -v
pnpm --dir web build
.\.venv-web\Scripts\python.exe -B -X utf8 -m eval.swebench_runner --help
.\.venv-web\Scripts\python.exe -B -X utf8 -m eval.analysis.summary --help
```

浏览器测试使用 `tests/fixtures/web_server.py`，配套独立状态目录和端口，禁止把假模型服务当作正式应用。构建包含 TypeScript 检查。测试中的固定 Token 数不能作为产品性能数据。

## 执行与验收边界

- 工作工具为 `submit_work`、`repair_work`、`freeze_acceptance` 和 `revise_acceptance`，保持 Beta strict 和本地校验。
- 浏览器另有 `web_search` 和 `web_fetch`。联网开关不改变工具定义；网络执行由程序根据当前任务快照检查。
- 浏览器固定增加 `list_files`、`read_file`、`search_files`、`run_existing`。工具参数存于 `resources/assistant.schema.json`，查询和已有任务运行不初始化编码环境。
- `run_existing` 仅接收已有入口和 argv。记录保存在任务目录 `assistant/`；子任务可提供本次生成的四项 Token JSON，来源、哈希及未知用量单独保存，不混入助手用量。
- 先在原始源码上试跑验收，再冻结并释放修改。冻结的代码、驱动、回归命令及用户固定验收不能被普通聊天改写。
- 零用例、新增跳过、加载失败、不完整结果、超时和真实失败不能算通过。验收修订继续走证据与源码版本对照。
- 完整历史只追加，压缩只替换请求视图；摘要不批准执行或覆盖真实失败。当前压缩阈值约 80 万输入 Token。
- 本地执行不是操作系统安全沙箱；需要只读、无网 Shell 调查时使用 Docker 后端。

## 回退与记录

变更前保留 Git 回退点和验证记录。旧任务按各自源码指纹使用冻结版本恢复，不迁移检查点，不重放已完成操作；结果不明的请求或写入不自动重试。

历史基线和验证数据保留在 `validation/`。设置与自动分流记录为 `monoplaner-settings-20261009-01`，本次界面整理记录为 `monoplaner-ui-20261009-01`。产品 README 不承担实验日志或成绩汇总。

桌面入口为 `launcher/Program.cs`，嵌入式工作窗口为 `launcher/DesktopWindow.cs`，重新生成使用 `scripts/build-launcher.ps1`。WebView2 SDK 固定为 1.0.4258.31，下载校验与本地准备在 `scripts/prepare-desktop.ps1`；运行依赖及其许可位于 `launcher/runtime/`。SDK 首次准备需要网络，之后可离线构建。启动与关闭不发起模型调用；活动或未知任务会阻止关闭服务。前端来源与许可保留在 `web/UPSTREAM.md` 和 `web/LICENSE`。
