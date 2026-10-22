# 按原文证据修订模型自拟验收

规则版本：`verbatim-acceptance-v3`。适用于新任务；旧任务继续使用对应冻结源码，不自动迁移检查点。

## 可修改与不可修改

冻结后，模型可以带有效独立原文证据修改自己制定的测试设置、夹具、输入、被测调用、断言类型、用例及预期，也可以修改自拟测试驱动、执行检查和回归命令。无需先把错误转换成“预期字面量”才能纠正。

原始用户需求、后续已确认的用户变更、交付对象、目标运行环境、真实入口和需求规定的输入范围，以及用户固定验收命令不属于这条修订通道。执行检查仍须覆盖全部已确认目标及必需输入，并实际编译／检查及执行。需求变化继续由人类补充与确认通道处理。

原文真实性不等于语义正确性：程序检查来源、逐项改动、结构化需求绑定及真实执行，不证明任意引用足以支持任意修改。模型必须说明证据与修改的关系，不能用当前实现输出为自己设定答案。没有新增 Checkbot 或其它模型调用。

## 提交与反馈

`revise_acceptance` 提交完整 `test_code`、`reason`，以及需要改变的 `test_driver`、`execution_checks` 或 `regression_command`。冻结前可以修正草稿，`evidence`、`corrections` 均为空；冻结后每次修订都必须提供这两组材料，包括设置修订。

`evidence` 每项包含 `id`、`source`、`quote`。支持的原件：

| source | 宿主读取的原件 |
|---|---|
| `task_request` | 保存的完整原始请求 |
| `user_update:<id>` | 持久化的明确用户补充 |
| `original_file:<path>` | 开工时冻结清单中的 UTF-8 原始文件 |

引文必须与原件完全一致，包括大小写、BOM、CRLF 和空白；不允许省略号、改写、自动纠字或模糊匹配。当前工作区改动、生成文件、工具当前输出及网络页面不能代替独立原件。来源与快照遭篡改时停止并保留现场。

每条 `corrections` 包含 `field`、`before`、`after`、`evidence_ids`、`reason`：

| field | before / after 的含义 |
|---|---|
| `test_code` | 精确旧片段及替换片段，可以涉及夹具、输入、调用、断言或用例 |
| `test_driver` | 完整旧／新数组的 JSON 文本，一条修订覆盖整个字段 |
| `execution_checks` | 完整旧／新数组的 JSON 文本，一条修订覆盖整个字段 |
| `regression_command` | 精确旧片段及替换片段 |

文本旧片段必须在当前已冻结材料中唯一，必要时包含未改的上下文；替换不得重叠。允许空 `after` 删除模型自拟内容；只有完整旧文本为空时允许空 `before`。没有逐项声明的内容必须保持不变。每条修订至少引用一份有效证据并说明理由，不接受悬空或闲置的证据 ID。

数组为空、回归命令为空时，默认复用当前提案。若要明确清除自拟驱动或自拟回归命令，需要对应字段的显式修订；必需执行检查即使显式清空也无法通过需求校验。修改尚未冻结的修订提案时，所有更改仍须相对当前已冻结版本完整提交。

缺少证据时返回以下英文说明；引文错误、字段未声明或原始需求冲突时，前面另附具体原因：

> Provide valid original evidence and itemized corrections to modify model-generated acceptance. Cite an exact quote from task_request, user_update:&lt;id&gt;, or original_file:&lt;path&gt;, and explain why each change follows from it. Original requirements, confirmed delivery constraints, and user-fixed commands cannot be changed through this tool.

拒收不修改工作区或冻结验收，也不开始试跑。修改普通源码计划中的冻结验收字段同样会引导使用 `revise_acceptance`；不能绕过这条通道。

## 试跑、版本与恢复

1. 校验原始快照身份、清单、文件哈希和逐字引用。
2. 校验逐项声明与完整新材料完全一致，原始需求和固定命令未变。
3. 核对新执行声明仍匹配当前需求版本与全部交付约束。
4. 保存新提案，在原始源码的隔离副本中试跑；只有真实可运行的结果才可冻结。
5. 模型查看结果后显式 `freeze_acceptance`；仅提供引用不代表通过。
6. 使用新冻结版本验证候选，保留旧版本、原始失败和完整修订链。

已批准改动只取代能精确绑定的旧材料。比较历史版本时投射逐项修订，保留未声明的设置和其它检查；不能把整个旧验收的无关差异悄悄丢弃。来源、材料、保护字段、修订理由及前后指纹均留档；恢复时重新核验。

零用例、跳过、伪造执行回执、来源破坏或不明结果不能通过。修订不清零时间、轮次、请求序号、Token、费用或未知预留；已完成试跑和操作不重放。

## 验证

`tests/test_acceptance_corrections.py` 验证原件、精确替换、字段范围及列表声明；`tests/test_acceptance_revision_scope.py` 验证夹具／断言／驱动／执行声明／自拟回归修订与需求保护；`tests/test_evidence_acceptance_flow.py` 保留完整试跑、冻结、历史比较、未知结果不重放和恢复检查。历史 Flask 失败材料另做免费离线回放，不改历史成绩，不提供隐藏测试作为工作证据。
