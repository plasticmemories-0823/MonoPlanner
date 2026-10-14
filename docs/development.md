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
| `agent/execution/` | Docker 执行、调查隔离及受控文件打开 |
| `agent/acceptance/` | 验收试跑、冻结、修订与最终结果 |
| `agent/storage/` | 原始记录、检查点、会话与恢复 |
| `resources/` | Schema、提示词与运行驱动 |
| `eval/` | 通用评测、用量与成绩分析 |
| `tests/` | 功能回归测试；产品运行不加载 |
| `validation/` | 验证记录、历史评测与停止现场 |

会话图复用首次模型回复，分为聊天、资料查询、已有任务运行或既有工作推进。原有任务存储独占操作游标；LangGraph 保存公开会话和任务关联，不另建会重放操作的执行器。

## 开发启动

从项目根目录运行 `scripts/start-app.ps1 -NoBrowser`。启动器同时启动受认证保护的前后端；后端不热重载。使用 `-BackendPort` 与 `-FrontendPort` 设置独立验证端口。不要替换正在运行的任务源码。

便携版 EXE 可通过 `MONOPLANER_BACKEND_PORT` 与 `MONOPLANER_UI_PORT` 环境变量使用独立端口；未设置时仍为 2024 和 3000。桌面窗口跟随同一界面端口，不接受远程主机地址。

浏览器固定通过同源 `/api` 访问后端，图 ID `monoplaner` 为兼容保留。后端凭证由启动器生成并用 DPAPI 保存；不设置 `NEXT_PUBLIC_API_URL` 或浏览器服务密钥。开发独立服务必须设置相同的实例凭证及端口，并使用本机代理入口；不要在 URL 或日志中放置凭证。

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
.\.venv-web\Scripts\python.exe -B -X utf8 -m pytest tests -q
pnpm --dir web build
.\.venv-web\Scripts\python.exe -B -X utf8 -m eval.swebench_runner --help
.\.venv-web\Scripts\python.exe -B -X utf8 -m eval.analysis.summary --help
```

浏览器测试使用 `tests/fixtures/web_server.py`，配套独立状态目录和端口，禁止把假模型服务当作正式应用。构建包含 TypeScript 检查。测试中的固定 Token 数不能作为产品性能数据。

## 执行与验收边界

`agent/execution/managed_plan.py` 只读解析项目依赖；`managed.py` 管理过滤副本、下载、安装及镜像缓存；`managed_proxy.py` 和 `managed_setup.py` 在独立准备容器内运行。`preparation.py` 对同一项目合并准备请求，并限制同时准备的项目数。认证接口 `/workspace/environment` 的 GET 只读，POST 接受 prepare、automatic、cancel。准备与重试不调用模型，不执行候选修改。

首次实际执行才由 `ConversationRunner.activate` 绑定准备好的镜像；聊天、目录浏览和预览不启动容器。手动或正式评测的显式镜像配置继续优先。项目依赖配置变化会使自动环境缓存失效；已绑定任务仍使用原环境。无依赖声明的新项目使用通用 Python/Node 基础环境，不猜测第三方依赖。

- 工作工具为 `submit_plan`、`repair_work`、`freeze_acceptance` 和 `revise_acceptance`，保持 Beta strict 和本地校验。
- `start_work` 只提交无参数开工交接；`repair_work` 只修正拒收稿件列出的字段。
- 编码 `read` 默认及单次上限为2,000行，允许较小窗口；内部按200行分页。同一步的所有页及 `from_step` 区间共享50KiB UTF-8正文预算。按原文顺序截断，使用 `continuation.next_offset` 续读；超长单行会明确标记，部分行不计入已读区间。搜索限额及普通聊天 `read_file` 不变。
- 完整但参数非法的助手回复仍按原件追加：保留公开文字、思考、工具调用身份及原始参数字符串，每个调用配对一个未执行的拒收结果。追加与配对状态写入检查点，恢复先完成配对，不重发请求。损坏JSON可以作为拒收原件续接；不完整回复或调用身份不可信则停止并保留现场。
- 浏览器另有 `web_search` 和 `web_fetch`。联网开关不改变工具定义；网络执行由程序根据当前任务快照检查。
- 浏览器固定增加 `list_files`、`read_file`、`search_files`、`run_existing`。工具参数存于 `resources/assistant.schema.json`，查询和已有任务运行不初始化编码环境。
- `run_existing` 仅接收已有入口和 argv。记录保存在任务目录 `assistant/`；子任务可提供本次生成的四项 Token JSON，来源、哈希及未知用量单独保存，不混入助手用量。
- 首次 `candidate` 必须提交完整 `test_code`，在保留的原始源码上试跑并明确冻结后才执行修改。调查中的只读局部检查无需完整验收；固定用户验收仍走原分支。冻结的代码、驱动、回归命令及用户固定验收不能被普通聊天改写。
- 零用例、新增跳过、加载失败、不完整结果、超时和真实失败不能算通过。验收修订继续走证据与源码版本对照。
- 完整历史只追加，压缩只替换请求视图；摘要不批准执行或覆盖真实失败。当前压缩阈值约 80 万输入 Token。
- 编码、验收、调查程序和已有任务运行统一使用 Docker；调查强制只读、无网。完全权限只影响本机文件操作和外部打开，由用户按对话开启，不改变工具表。
- `list_applications`、`open_file`、`launch_application` 属于聊天工具；风险询问模式的外部打开需一次性批准，沙盒模式禁止外部启动。不能把打开文档当成代码执行来绕过沙盒。
- `open_file` 只返回预览或外部打开状态，文件内容不交给模型；需要读取文本时使用 `read_file`。

## 便携打包

先构建前端与 EXE，在 `launcher/runtime/` 准备第三方说明中的固定运行时。Python 分发必须与 `requirements.lock` 一致，打包只复制生产依赖；开发环境中的 pytest、pip 等不会进入分发。

从 Git for Windows 的官方 `v2.53.0.windows.2` Release 下载 `mingw-w64-git-2.52.0.1-1.src.tar.gz`，并从同一 Git 标签下载 ZIP 源码，命名为 `git-v2.53.0.windows.2.zip`，放入 `launcher/runtime/sources/`。脚本核对固定哈希并将其与许可一起分发。

运行 `scripts/build-portable.py` 生成新目录及 ZIP。已有目录或同版本 ZIP 会被拒绝，不能覆盖混合。最终核对清单、源码指纹、依赖许可和新解压目录启动结果；不要把应用测试现场、个人资料或密钥复制进包。

## 回退与记录

变更前保留 Git 回退点和验证记录。旧任务按各自源码指纹使用冻结版本恢复，不迁移检查点，不重放已完成操作；结果不明的请求或写入不自动重试。

历史基线、验证数据和停止现场保留在 `validation/`，每次验证使用独立目录。产品 README 不承担实验日志或成绩汇总。
