# Wine CMD 兼容验收

.bat/.cmd 可以声明为 repository_entry，使用 target_runtime=wine-cmd 和对应的 execution_checks.interpreter=wine-cmd。这是 Linux Docker 中的 Wine 兼容执行，不是原生 Windows。

## 安装与使用

1. 安装并启动 Docker Desktop，使用 Linux 容器。在 MonoPlanner 选择实际项目目录；尚未创建任务也可以配置环境。
2. 展开聊天输入框上方的 **当前项目 · 环境配置**，点击 **安装 Wine**。程序下载固定的 Debian Wine 包、建立可复用环境并实际执行 CMD 探测；不调用模型，不向 Windows 宿主安装 Wine。
3. 界面显示准备进度；可点击 **取消准备**。失败、取消或服务中断不会自动重发安装，修复问题后点击 **重试安装 Wine**。安装日志和原件保留。
4. 显示 **Wine CMD 已就绪** 后再让 Chatbot确认批处理任务。Chatbot 会重新读取当前项目能力；使用别的目录时需要检查该目录的配置。新项目可先创建并选择其目录，或直接在已配置目录中创建批处理文件。

安装选项针对当前项目。普通任务只能复用已经准备好的 Wine；需要首次安装或配方更新时，必须由用户在环境配置中点击安装。重复点击不会同时启动两份安装。具有相同 Python／Node 版本和安装配方的基础环境可以复用；项目依赖在另一个受限容器中准备。已有任务仍使用原来的镜像。

已有手动镜像不会被覆盖。可在高级设置中选择一个已有 Wine 镜像，或先点击 **改为自动准备** 再安装 Wine。自动准备的基础环境使用 Debian Bookworm，固定安装 `wine=8.0~repack-4` 和 `wine64=8.0~repack-4`；兼容前缀和 CMD 探测均通过后才发布 Wine 标签与可用状态。

## 开工边界

Chatbot 遇到 Windows CMD、.bat 或 .cmd 要求时，先调用 `get_work_capabilities`。缺少 Wine 就提示用户打开环境配置，不使用工作工具写安装脚本或代替用户安装。准备和确认方案时再次核对本机不可变 Linux amd64 镜像和 `org.monoplanner.wine-cmd=1` 声明；自动环境还必须匹配当前项目与安装配方。

镜像声明不是任务通过的证明。实际执行仍核对 Wine 版本、入口原件、参数、退出码和日志。没有 Wine 的 Python／Node 环境不能冒充 CMD 验收环境。

安装驱动、轻量前缀工具和入口包装在 `resources/environments/`；`wine-cmd.Dockerfile` 仍供已有手动环境使用。Wine 二进制保存在项目 Docker 镜像中，不随 Windows 便携包复制，不影响旧任务的冻结内核与环境。Wine 及 Debian 依赖保留自己的许可证。

## 实际检查什么

CMD 没有独立的“只编译”入口。此适配记录 validation_scope=wine-cmd-executed-paths-only：**只运行一次未替换的真实批处理，验证实际走到的分支和业务断言**，不生成虚假的 compile_exit_code=0。Python、Node、Shell 和生成 C 产物的原有编译/语法检查保持。

驱动用固定 /d /v:off /u /s /c 参数调用 Wine CMD，将经过校验的调用原文放在该子进程专用环境值中，避免 Wine 对 Unix argv 的引号再次转义。/d 禁用 CMD AutoRun；/u 指定内建命令的重定向输出使用 Unicode。行为以 [Microsoft CMD 文档](https://learn.microsoft.com/en-us/windows-server/administration/windows-commands/cmd)及实际 Wine 探测为界，不把两者等同。

- 宿主核对实际入口路径与源码 SHA-256、完整参数文本、运行时版本原件及 stdout/stderr 哈希。
- 空格与 Unicode 路径/参数得到真实运行验证；控制字符和百分号、感叹号、双引号、&、竖线、尖括号、^ 不能作为字面入口路径或参数，明确拒收，不静默改写。
- CMD 内建输出保留 UTF-16LE 原始字节；外部程序可能采用其它编码。展示是有限预览，判断仍可读取已保存原件。
- 入口执行失败、环境不可用、未知结果或原件变化不能通过。非零退出码按真实返回保留。
- test_code 仍须断言业务行为和副作用；单次入口退出码为零不证明所有分支正确。

## 不覆盖什么

Wine 验证不能证明原生 Windows 的服务、注册表、ACL、文件占用、UAC、系统文件保护或真实系统盘清理效果。涉及这些要求必须另有合适环境，不能换一个产物冒充完成。

清理类测试只使用任务隔离目录或 Wine 虚拟盘中的一次性夹具。宿主真实 C 盘不作为测试对象；Docker 内 Wine 的 Z: 也只看到容器文件系统及已授权挂载。容器仍受单任务租约、无网、资源限制和失联保护约束，凭据与 Docker Socket 不进入任务容器。
