# MonoPlanner 本地聊天前端

基于官方 Agent Chat UI；来源及改动见 [UPSTREAM.md](UPSTREAM.md)，保留 MIT 许可。

从项目根目录运行 `pnpm --dir web install --frozen-lockfile` 安装前端依赖，再运行 `.\scripts\start-app.ps1 -NoBrowser` 一起启动本机前后端。完整安装、恢复与验证说明见 [项目 README](../README.md)。

默认后端 `http://127.0.0.1:2024`，页面 `http://127.0.0.1:3000`。如需改端口，向 `scripts/start-app.ps1` 传入 `-BackendPort` 和 `-FrontendPort`；启动器统一提供本机连接配置与实例凭证。仅支持本机使用。

模型密钥只放后端。这里的环境变量不是 DeepSeek 凭证。

支持仓库目录、公开对话、聊天流中的操作状态及可展开思考、运行中补充与安全暂停。操作和思考只读取本地记录，不增加模型调用；刷新可重新连接。没有附件上传、编辑旧消息重放或云代理转发接口。

验证：`pnpm --dir web exec tsc --noEmit`、`pnpm --dir web build`。
