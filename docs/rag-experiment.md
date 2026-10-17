# RAG 实验版操作与验证边界

本文件对应 `E:\AI\LangGraph\MonoPlanner-rag-experiment`。正式版目录保持原样。
Bench 仍暂停。以下操作入口不会下载模型，也不会生成付费评测回答。

当前已准备本地 Embedding 与 Cross-Encoder 文件及隔离的 CPU ONNX 依赖。
本地分词、Embedding及重排已进行真实CPU调用验证；这不表示已经通过完整检索质量评测。Laya 权重尚未下载、人工资格尚未完成；
此时 `propose` 应明确返回不可用，不能用假模型或占位配置代替真实决策。

## 调参入口

在实验目录使用当前 Python 环境运行 `python -m eval.rag_tuning`。
数据库应与产品实际使用的 `data_root/rag/tuning.sqlite3` 一致；测试时使用独立数据目录。
`--output` 只写新文件，拒绝覆盖已有记录。所有命令输出 JSON；退出码 2 表示不可用、
输入错误或门禁未通过，不能当成评测成功。
状态查询或重复提案返回0只表示读取成功；晋级仍要求数据库中的`shadow_passed`状态。

```powershell
$python = 'E:\AI\LangGraph\MonoPlanner\.venv-web\Scripts\python.exe'
$database = 'E:\AI\LangGraph\MonoPlanner-rag-experiment\validation\rag-tuning.sqlite'

# 只读状态；不存在时不会创建数据库。
& $python -B -m eval.rag_tuning --db $database status

# 让本地 Laya 从程序生成的合法单变量候选中选择，包含“不改”。
# metrics.json 只含已有指标及少量失败样例；首轮无可靠证据时应选择“不改”。
& $python -B -m eval.rag_tuning --db $database --models 'E:\models\rag-models.json' `
    --output 'proposal.json' propose --metrics 'metrics.json'

# 冻结候选和评测合同；此时仍不影响新任务。
& $python -B -m eval.rag_tuning --db $database shadow `
    --proposal 'proposal.json' --spec 'frozen-spec.json'

# 将上一步实际输出的 trial 填入；读取已经完成且可追溯的配对评测记录。
& $python -B -m eval.rag_tuning --db $database evaluate --trial '<TRIAL_ID>' `
    --rows 'paired-results.json' --qualification 'human-qualification.json'

# 全部门禁通过、当前本地评审模型身份一致且不在冷却期时，才允许5%新任务灰度。
& $python -B -m eval.rag_tuning --db $database --models 'E:\models\rag-models.json' `
    canary --trial '<TRIAL_ID>'

# 手动提交灰度复核；门禁失败或复核记录无效时，回滚后续新任务分配。
# 现有任务保留已经绑定的版本。
& $python -B -m eval.rag_tuning --db $database evaluate --phase canary `
    --trial '<TRIAL_ID>' --rows 'canary-results.json' --qualification 'human-qualification.json'

& $python -B -m eval.rag_tuning --db $database rollback `
    --trial '<TRIAL_ID>' --reason 'resource_or_quality_regression'
```

首轮 `propose` 使用默认 `RagConfig` 初始化数据库，也可显式提供 `--initial-config`。
数据库存在后，以持久化基线为准，不通过这个参数替换已绑定的配置。
调参只允许 `recall_top_k`、`vector_weight`、`rewrite_threshold` 和 `chunk_target`。
其余配置参与版本指纹并保持固定。`topK`整数步长向下取整，防止超过10%；融合另一侧
权重由 `1-vector_weight` 派生。阶段结束不会自动扩大到100%流量。

当前没有持续生产遥测监控或自动发起调参循环。灰度复核由上述操作入口显式提交，
程序在收到失败或无效复核记录时执行回滚；未提交复核不能声称已经检测到生产异常。
重复提交同一提案会返回已存在试验的真实状态，不会重置已失败或已回滚的试验。

## 需要提供的真实评测记录

目前没有随项目附带100–200条真实人工标注题集。单元测试里的假数据仅验证控制逻辑，
不能作为上线资格。完整的工作模型配对回答生成器尚未接入此操作入口。
使用者需要先保存用途、输入、模型身份、配置、用量和费用完整的基线与候选记录；
旧配置的回答不能充当新配置的生成结果。

`frozen-spec.json` 必须包含：

- `queries`：100–200个保留集query ID到标签列表的映射；至少包含`hard_case`、
  `no_answer`、`permission_boundary`。`optimization_ids`必须与保留集分离。
- `dataset_sha256`、`qualification_dataset_sha256`：冻结标注集的真实SHA-256。
- `primary_metric`：试验前固定为`ndcg`或`recall_at_100`。
- `min_effect`：预先规定的最小有效提升，必须大于0；不能看过结果后再调整。
- `judge_identity`、`generation_identity`：评审及生成模型、版本和配置身份。
- 可显式冻结`bootstrap_samples`、`confidence`、`seed`、`max_trials`、
  `qualification_min_cases`及`qualification_min_accuracy`；默认分别为
  1000、0.95、1729、1、100、0.9。

`paired-results.json` 是数组，每行包含`query_id`、`dataset_sha256`，以及`baseline`、
`candidate`两个对象。每个对象必须包含：

- `version`、`generator`、`judge_identity`，与冻结配置、模型和评审身份一致。
- `input_sha256`、`answer_sha256`、`record_sha256`，指向真实完整归档；
  基线与候选不能使用同一个生成记录。相同回答正文的哈希可以相同。
- `usage_known: true`、实际`cost`，不能把缺失用量填成零；统一货币和价格口径应写入
  `generation_identity`及评测原始记录。
- 主指标、`faithfulness`、`citation_precision`、`latency_ms`、`memory_bytes`。
  延迟统一单位毫秒、内存单位字节；记录真实完整RAG开销。
- 程序检测得到的`permission_pass`、`safety_pass`，不能使用模型自报结论。

这些摘要指标必须来自评测程序。该入口不会靠哈希文本自动证明其对应原件存在或
评测标签正确；需要保留可核对的原件，不能手填虚构数据来放行。

`human-qualification.json` 包含`human_labeled: true`、对应数据集哈希、
`judge_identity`及`cases`。每个case有唯一`id`、人工`expected`和真实模型`predicted`，
标签只允许`supported`、`contradicted`、`unknown`。三类必须覆盖；分类准确率、
总准确率Wilson下界均需达到冻结门槛。

主指标使用配对bootstrap，多个门禁和预先规定的候选次数分摊错误概率。
faithfulness、引用精度以及p95延迟、峰值内存、成本必须提供不退化证据。
数据缺失、用量未知、模型资格不符或权限／安全失败均不能晋级。
SQLite同时保存版本、报告、必要指标、任务绑定与审计记录；不重复存放回答正文。
24小时冷却持久化，重启或回滚不能清零。

## 本地 Laya 清单形状

下面仅展示接入形状，**占位值不能使用，也不是已验证的模型配置**。模型文件必须已经
存在于本地；不调用`ensure_model`、Hub下载或自动安装。可以显式传`--models`，或设置
环境变量`MONOPLANER_RAG_MODELS`指向清单。

```json
{
  "version": 1,
  "dependency_path": "E:/AI/LangGraph/MonoPlanner-rag-experiment/.rag-deps",
  "models": {
    "laya": {
      "root": "E:/models/laya-qualified-local",
      "backend": "laya_onnx",
      "model": "model.int4.onnx",
      "max_tokens": 512,
      "head_max_tokens": 192,
      "runtime_versions": {
        "laya": "<VERIFIED_LOCAL_VERSION>",
        "onnxruntime": "<VERIFIED_CPU_RUNTIME_VERSION>",
        "transformers": "<VERIFIED_LOCAL_VERSION>",
        "torch": "<VERIFIED_LOCAL_VERSION>",
        "tokenizers": "<VERIFIED_LOCAL_VERSION>",
        "numpy": "<VERIFIED_LOCAL_VERSION>"
      },
      "sha256": {
        "model.int4.onnx": "<ACTUAL_SHA256>",
        "rl_agent_config.json": "<ACTUAL_SHA256>",
        "tokenizer/tokenizer.json": "<ACTUAL_SHA256>",
        "tokenizer/tokenizer_config.json": "<ACTUAL_SHA256>"
      }
    }
  }
}
```

`sha256`必须列出该目录实际加载的全部文件，包括外置ONNX权重及所有分词器配置；
示例中的四项不是通用允许清单。模型真实上下文和head预算必须与固定版本一致。
`dependency_path`指向独立安装依赖的目录，可使用相对清单所在目录的路径。
当前的CPU ONNX依赖不等于已经安装Laya所需全部依赖；清单中的版本逐项不匹配时拒绝加载。
CPU ONNX适配复用[官方API](https://github.com/NandhaKishorM/laya/blob/main/docs/reference/agent.md)，
使用`laya.load(local_root, backend="onnx", onnx_path=absolute_path)`，不另写一套决策模型。
官方ONNX路径仍会导入Torch，且不会把`device="cpu"`传给ONNX会话，因此需使用
CPU-only的`onnxruntime`依赖并核验实际provider，不把GPU自动回退当作CPU部署证明。
量化改变输出分布，必须把量化文件、运行库及问题模板计入
资格身份，并重新做人工标注资格验证。

没有证据证明本机所需Embedding、Cross-Encoder和Laya能在共享1GiB峰值下同时工作。
运行时限制推理并发、设置worker限制并在主机余量不足时拒绝加载；文件大小不等于
推理峰值。Windows峰值、加载失败、取消、真实模型准确率仍需本机验证。
没有合格本地模型时返回不可用并停留shadow；不自动提高内存限制。


## 当前接入状态

工作内核和 Chatbot 共用新的只读 `retrieve_evidence`，原有搜索、读取、分页和 `from_step`
不变。索引只接受当前项目及当前任务已提交材料；Chatbot 使用公开投影，不能借检索
读到工作私有记录。派生索引存于任务目录，彻底删除时先停止所属后台构建，确认退出后删除。

具备 BM25、本地真实向量、有限 Cross-Encoder 重排、原文来源核验、有限父级补读、
完整工具消息预算和请求级单次改写。定位精确命中短路；不完整的索引结果不会作为
付费改写的缺证理由。后台构建有独立任务归属，暂停和服务退出可以取消。

超长结构以**正文256 Token**为软目标；来源信息、正文和必要特殊Token共同受512 Token
硬限约束。完整结构能装下就不拆；硬限不足时缩短正文，重叠最多64 Token。模型已配置时
使用其固定Tokenizer计数，不能因为字符减少就假定Token减少。自动调参的256/384/512
档位只改变正文软目标，绝不能突破完整512 Token硬限。未配置模型时保留估算BM25路径并
标明向量不可用。长查询超出重排模型配对窗口时拒绝该次重排，不能偷偷截断。

程序验证已出现的 `[evidence:<chunk_id>]` 来源、版本、哈希及实际返回范围；验证结果
只写本地审计，不等于语义支持已证明。当前尚不自动识别自然语言中的所有“关键事实缺少
引用”，前端也未增加引用诊断面板。没有调用者明确时间范围时，时效软信号记录为不适用；
尚未接入专门的历史时间范围查询。当前源码版本使用实际原件哈希，不伪造Git分支或提交号。

## 本地模型与实验环境

固定清单样例为 `resources/rag-models.example.json`；将它放到模型目录的`models.json`后，
各模型路径相对于该清单解析。当前机器已准备`.rag-models/models.json`和`.rag-deps`；
权重、可选依赖、验证数据均未进入源码提交，也没有复制正式Python或Node运行时。

| 用途 | 本地模型 | 固定修订 |
|---|---|---|
| Embedding | intfloat/multilingual-e5-small，INT8 ONNX | `614241f622f53c4eeff9890bdc4f31cfecc418b3` |
| 重排 | cross-encoder/mmarco-mMiniLMv2-L12-H384-v1，INT8 ONNX | `1427fd652930e4ba29e8149678df786c240d8825` |

两组资产共271,137,538字节，约258.58 MiB。完整SHA-256清单见示例配置与本地
`.rag-models/asset-receipt.json`。许可证分别为MIT和Apache-2.0，原模型说明随本地资产保存。
`requirements-rag-local.txt`固定CPU依赖；如需另机部署，安装到隔离目录，不能覆盖产品环境。

实验后端可以使用以下独立端口及数据目录启动；本轮没有自动运行该服务：

```powershell
Set-Location 'E:\AI\LangGraph\MonoPlanner-rag-experiment'
$env:MONOPLANER_STATE_DIR = "$PWD\validation\rag-app-data"
$env:MONOPLANER_BACKEND_PORT = '12024'
$env:MONOPLANER_UI_PORT = '13000'
$env:MONOPLANER_RAG_MODELS = "$PWD\.rag-models\models.json"
& 'E:\AI\LangGraph\MonoPlanner\.venv-web\Scripts\python.exe' -B -m agent.server.portable
```

这只是后端入口，不是已打包的完整桌面程序。源代码工作树中的EXE是桌面回归所需的
本地编译产物；本轮未构建新的前端分发包，不应用它替换正在使用的正式版。
新任务固定检索配置与模型清单指纹；既有任务不会因环境变量变动悄悄换模型。

## 当前验证与限制

验证原件位于`validation/rag-implementation-20261017/`，不随源码提交。
真实本地模型检查使用自建小型混合文本，验证调用、原文范围和资源退出，不是质量基准，
不能据此宣称Recall、费用或编码成功率提高。1 GiB是总额外峰值预算；单worker使用768 MiB
硬限，其余预留给父进程。小样本峰值通过不证明所有规模都满足预算，超限时明确停止或降级。

Laya适配、候选枚举及评测门禁已实现，但尚未准备可在资源限额内通过资格的Laya权重、
100–200条人工冻结评测及独立保留集。自动晋级保持关闭。当前没有持续自动调参循环，
也没有自动付费生成配对回答的完整驱动；CLI只消费可追溯的既有评测记录。不能将单元测试
的假模型与假标签当作资格证据。已有Bench继续暂停，所有本轮验证均未调用付费API。

本轮实际检查（2026-10-17）：

- 全量Python离线回归：1193通过、3跳过、281个子测试通过。
- 最后的词法降级修复后，检索、调参、恢复、退出和内核交接相关回归：194通过、33个子测试通过。
- 前端事件、控制和状态单元测试：49通过；前端源码未改，没有再次打生产前端包。
- 真实本地混合文本轨迹：14片、最大512个实际Embedding Token；初次明确降级，
  建索引后BM25召回13片、向量召回14片，完整返回1795个估算Token，无必需实体缺失。
- 同一小轨迹的父进程增量峰值与最大worker峰值相加约587 MiB。此数字只覆盖该轨迹，
  不代表大项目、所有并发或Laya已经通过资源资格验证。
- 正式目录353个基线文件哈希一致；没有启动Bench、付费模型测试、Docker下载或GitHub发布。

分词器不可用时改用独立的估算词法索引，不将它用于向量构建。不完整检索的所有改写触发
都会记录为屏蔽，不因索引、模型或资源故障额外消耗一次付费改写。
实验源码基线回退点为`898e72c3abbfedce305968b15658d3eae3658ca4`，包括复制时已有的
未提交源码修改。回退时不要把它写回正式目录。
