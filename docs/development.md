# 开发与维护

## 项目结构

| 目录 | 职责 |
|---|---|
| `web/` | 聊天界面、工作目录选择、模型设置与操作展示 |
| `agent/server/` | 经过认证的本机 ASGI 接口 |
| `agent/application/` | 独立 Chatbot、持久输入、工作调度和事件流 |
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

Chatbot 与工作内核均由 LangGraph 调度，但服务不依赖 Agent Server 的 thread/run 队列。工作图显式连接规划、回复处理、验收审查和执行节点；不另存一套图检查点，也不自动重试节点。应用 SQLite 保存公开消息、幂等命令、事件与工作关联；TaskRunner 原件控制实际执行游标。Chatbot 与工作分别调度、分别保存上下文，自动汇报不能成为工作输入。

自动汇报由关键事件触发，调用间隔至少 30 秒；每次汇报最多进行 3 次模型请求，包含只读工具续接。它仅能查询当前任务状态、阶段事件和已提交的有限操作记录，不能启动、暂停、补充或批准任务，也不读取私有冻结验收正文。超限、失败或结果不明时保存记录，不自动重发该次请求；工作继续按原执行状态推进。自动汇报使用所选模型和思考配置，普通聊天与工作原有的思考设置、输出上限和调用边界保持不变。

前端通过 `/conversations`、`/conversations/{id}/snapshot`、`/conversations/{id}/commands` 和 `/conversations/{id}/events?after=N` 访问产品接口。命令带唯一编号；重复的相同命令返回原结果，不同正文复用同一编号被拒绝。SSE 断线只补读，浏览器不创建或补启动执行器。工作活动复用 `/conversations/{id}/diagnostics?after=...` 的只读原件投影：当前可见会话运行时按游标持续读取，终态补齐分页后停止，读取错误保留已显示内容并等待重新连接。读取不重建整份聊天，不复制工作原件，也不进入 Chatbot 上下文。内核内部回复编号在展示时映射到任务开工消息，避免并发聊天把工作活动挤到错误位置。

暂停走独立控制线程，取消模型等待并终止所属 Docker 容器进程；未知结果不自动恢复。运行资源由本地台账登记所有者、引用和租约；暂停保留原容器，已提交完整证据的临时副本可以删除。环境配方使用显式协议版本，而非 Python 实现文件的原文字节。

请求原件可通过 `agent.storage.requests.read_request` 重建并校验；消息按内容去重压缩，避免历史前缀每轮完整落盘。工具返回模型前，大文本卸载为带 SHA-256 的原件引用和有限预览；已有运行日志直接复用，不另复制。没有原件的正文按内容地址保存。仅在当前压缩视图内已完整可见的重复正文可省略；恢复或压缩后重新核对，读取 `$record/` 仍受登记、哈希、行数和正文预算限制。卸载原文出现超长单行时，按返回的 `next_query` 使用 `read query=payload_bytes:N`（`offset=0`、`limit=2000`）续读原始字节窗口，仍共享50KiB展示预算，部分行不计入完整已读范围。受管日志保留原始字节哈希，非UTF-8内容明确标识替代解码。源码读取沿用原有分页规则。任务源码指纹仍覆盖工作实现及协议；新任务使用新版本，旧任务使用冻结旧版。

工具结果的结构去重只整理本次调用尚未交付的返回集合：网页来源按精确URL、目录项按规范路径、搜索命中按源码版本与路径和行号识别；同键保留最后出现的完整条目及其位置，没有可靠主键时才按单条规范JSON的哈希去重。不把新旧字段混合，不按任意 `id` 或数组盲目删除。源码读取区间、仓库地图的不同增量、执行与验收的成功失败记录继续保留。

工作内核与Chatbot共享这一整理入口，先保存原始结果，再整理并追加同一调用编号的工具消息；原件保存失败则停止，不重放操作。文件搜索的模型展示与 `from_step` 续读采用相同去重规则，分页游标仍对应原始查询位置。此过程不查询或改写已发送历史，也不增加LLM调用；现有载荷卸载和已见正文引用机制在其后继续使用。

首次环境或源码捕获失败记录为 `preparation_failed`，使用稳定编号直接发布系统消息，不依赖模型汇报。该任务的失败标记阻止再次捕获，部分副本与清单原位保留。已有工作检查点或实际操作中的关键记录失败仍按未知结果保护，不能降级成普通准备失败。

应用关键记录不可写时，调度器进入不可自动解除的存储故障闭锁。快照、事件流和服务状态中的 `storage_error` 描述服务健康，不代替工作检查点；前端不能据此重放命令。紧急暂停在会话和资源归属核验后仍发送终止，即使记录失败也不能跳过控制；无法保存停止证据时，返回未确认结果。


## 开发启动

从项目根目录运行 `scripts/start-app.ps1 -NoBrowser`。启动器同时启动受认证保护的前后端；后端不热重载。使用 `-BackendPort` 与 `-FrontendPort` 设置独立验证端口。不要替换正在运行的任务源码。

便携版 EXE 可通过 `MONOPLANER_BACKEND_PORT` 与 `MONOPLANER_UI_PORT` 环境变量使用独立端口；未设置时仍为 2024 和 3000。桌面窗口跟随同一界面端口，不接受远程主机地址。

浏览器固定通过同源 `/api` 访问后端，无需 Agent Server 图 ID 或远程服务地址。后端凭证由启动器生成并用 DPAPI 保存；不设置 `NEXT_PUBLIC_API_URL` 或浏览器服务密钥。开发独立服务必须设置相同的实例凭证及端口，并使用本机代理入口；不要在 URL 或日志中放置凭证。

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

新接口浏览器测试使用 `tests/fixtures/application_server.py`，配套独立状态目录和端口，禁止把假模型服务当作正式应用。旧聊天图及其专属回归已移除；旧数据只读兼容使用必要的数据夹具，不保留旧执行器；生产只加载 `frontdesk.txt` 作为聊天提示，不再加载旧 `conversation.txt`。默认先提出方案并询问开工，明确授权后才调用 `start_work`；这一语义判断由 Chatbot 完成，不增加关键词分类器。构建包含 TypeScript 检查。测试中的固定 Token 数不能作为产品性能数据。

## 执行与验收边界

`agent/execution/managed_plan.py` 只读解析项目依赖；`managed.py` 管理过滤副本、下载、安装及镜像缓存；`managed_proxy.py` 和 `managed_setup.py` 在独立准备容器内运行，`download_relay.py` 提供仅在准备期间运行的宿主网络转发。`agent/model/network.py` 按请求解析当前系统代理与 PAC；网页搜索和抓取只连接 DeepSeek 工具 API，下载转发继续保留公网地址检查。`preparation.py` 对同一项目合并准备请求，并限制同时准备的项目数。认证接口 `/workspace/environment` 的 GET 只读，POST 接受 prepare、automatic、cancel。准备与重试不调用模型，不执行候选修改。

选择目录只保存会话的授权工作区 `repo`。Chatbot 在开工询问中说明项目位置，确认后用 `start_work(project_path, create)` 保存相对路径；工作记录的 `repo` 指向实际项目，公开快照的 `work.repo` 用于展示及环境操作。新目录在后台首次开工时排他创建，意图和完成凭据分开保存；未知结果不重放。目录创建与范围核对完成后才准备环境并绑定镜像、容器。普通聊天、目录浏览和预览不启动准备或工作容器。手动或正式评测的显式镜像配置继续优先。项目依赖配置变化会使自动环境缓存失效；已绑定任务仍使用原环境。无依赖声明的新项目使用通用 Python/Node 基础环境，不猜测第三方依赖。

`ProjectScope` 在开工前确定项目根和其他独立 Git 子项目排除集，源码选择、版本证据、运行材料、安装和容器共同使用。发现独立项目后不进入其子树；容器以空只读挂载遮罩这些位置，遮罩原件不通过可写挂载暴露。首次启动和恢复核对完整范围与目录身份，正常读取沿用冻结范围；新出现的 Git 边界不能隐藏源码变化。Git 子目录保留限定在该子树内的跟踪与忽略语义，不自动升到祖先目录。范围发现最多检查200,000个目录项、10秒，可取消；超限停止说明。这里只改变文件范围和基础设施参数，编码工具、工作提示、冻结步骤及验收判定不变。

- 工作工具为 `submit_plan`、`repair_work`、`freeze_acceptance` 和 `revise_acceptance`，保持 Beta strict 和本地校验。
- `start_work` 只提交已确认的 `project_path`（工作区相对路径，根目录为 `.`）及 `create` 布尔值；旧的无参数入口拒绝。现有任务的补充必须沿用原项目，不能改变根目录。`repair_work` 只修正拒收稿件列出的字段。
- 编码 `read` 默认及单次上限为2,000行，允许较小窗口；内部按200行分页。同一步的所有页及 `from_step` 区间共享50KiB UTF-8正文预算。按原文顺序截断，使用 `continuation.next_offset` 续读；超长单行会明确标记，部分行不计入已读区间。搜索限额及普通聊天 `read_file` 不变。
- 完整但参数非法的助手回复仍按原件追加：保留公开文字、思考、工具调用身份及原始参数字符串，每个调用配对一个未执行的拒收结果。追加与配对状态写入检查点，恢复先完成配对，不重发请求。损坏JSON可以作为拒收原件续接；不完整回复或调用身份不可信则停止并保留现场。
- 编码与 Chatbot 共用 `agent/execution/web.py` 的 `web_search`／`web_fetch`，分别 POST `/v1/tools/search`／`/v1/tools/fetch`。查询原文不改写，网页 Markdown 原样归档；已删除 Anthropic 聊天搜索和本机 HTML 抓取。联网开关不改变工具定义；网络执行由程序根据当前任务快照检查。按次费用、未知预留与模型 Token 独立记账。
- Chatbot 提供 `list_files`、`read_file`、`retrieve_evidence`、`run_existing`。工具参数存于 `resources/assistant.schema.json`，资料查询不初始化编码任务。
- `run_existing` 仅接收已有入口和 argv，由后台工作队列在 Docker 中执行一次；请求及结果保存在任务目录 `existing/`，不触发编码规划。子任务可提供本次生成的四项 Token JSON，来源、哈希及未知用量单独保存，不混入助手用量。
- 首次 `candidate` 必须提交完整 `test_code`，在保留的原始源码上试跑并明确冻结后才执行修改。调查中的只读局部检查无需完整验收；固定用户验收仍走原分支。冻结的代码、驱动、回归命令及用户固定验收不能被普通聊天改写。
- 零用例、新增跳过、加载失败、不完整结果、超时和真实失败不能算通过。验收修订继续走证据与源码版本对照。
- 完整历史只追加，压缩只替换请求视图；摘要不批准执行或覆盖真实失败。工作与 Chatbot 使用同一套 80% 同步压缩及输出预留；当前默认触发值为678,464个估算输入Token，溢出补救也不能删除全部近期原文。
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

## 实验版统一搜索约定

精确搜索默认原样使用模型query，不替模型拆词、猜符号或添加AND/OR。语义搜索必须显式
选择，不能失败后暗中换模式。Chatbot工具与工作计划搜索步骤共用服务，不增加第二套工作
调度；新功能替代旧执行器后，删除无调用的旧实现及只验证该实现的测试。AST解析缓存、
源码版本及已读范围仍保留；搜索测试应从实际工具/计划入口验证到下一次模型输入。

分页、命中范围与字节续读均绑定查询和来源版本。结果不足、共享预算不足和来源不可读要
明确区分；不能将未扫描或部分读取宣布为不存在/完整。工具结果原件及历史前缀保持不变。
主请求与摘要统一使用ModelPolicy.api_timeout（默认90秒）；显式请求超时优先，任务剩余
时间进一步收紧。剩余时间为0时不发请求，不用默认值把已到期任务重新放行。
