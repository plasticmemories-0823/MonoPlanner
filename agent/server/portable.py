"""用本机 ASGI 服务启动产品，不再建立 Agent Server 的第二套会话队列。"""

import os

from agent.storage.preferences import state_root


def main():
    import uvicorn
    state_root().mkdir(parents=True, exist_ok=True)
    port = int(os.getenv('MONOPLANER_BACKEND_PORT', '2024'))
    if not 1024 <= port <= 65535:
        raise ValueError('Invalid local service port')
    uvicorn.run('agent.server.app:app', host='127.0.0.1', port=port,
                reload=False, workers=1, access_log=False)


if __name__ == '__main__':
    main()
