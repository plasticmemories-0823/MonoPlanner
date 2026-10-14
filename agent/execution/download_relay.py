"""环境准备期间的宿主下载中继；系统代理和凭据不进入安装容器。"""

import base64
import hmac
import json
import os
from pathlib import Path
import secrets
import socket
import threading

import httpx

from agent.execution import managed_proxy
from agent.model.network import system_proxy_for


def connect_default(host, port, target_url, *, on_socket=None, cancelled=None):
    """先固定公网地址，再按原始 URL 选择当前系统路线；失败不切线重试。"""
    addresses = managed_proxy.public_addresses(host, port)
    if cancelled and cancelled():
        raise OSError('preparation relay closed')
    try:
        proxy = system_proxy_for(httpx.URL(target_url))
    except httpx.HTTPError as exc:
        raise OSError('system proxy could not be resolved') from exc
    if cancelled and cancelled():
        raise OSError('preparation relay closed')
    if proxy:
        address = addresses[0][4][0]
        return managed_proxy.tunnel(proxy, address, port,
                                    on_socket=on_socket, cancelled=cancelled)
    family, kind, protocol, _, address = addresses[0]
    connection = socket.socket(family, kind, protocol)
    try:
        if on_socket:
            on_socket(connection)
        connection.settimeout(managed_proxy.TIMEOUT)
        connection.connect(address)
        return connection
    except BaseException:
        connection.close()
        raise


class _Handler(managed_proxy.Handler):
    def parse_request(self):
        if not super().parse_request():
            return False
        received = self.headers.get('Proxy-Authorization', '')
        if not hmac.compare_digest(received.encode('utf-8'), self.server.authorization):
            self.send_error(407, 'Preparation authorization required')
            self.close_connection = True
            return False
        return True


class _Server(managed_proxy.Server):
    # HTTP 的内部隧道仅供受控代理使用，目标仍限定为已解析的公网 80/443。
    tunnel_ports = (80, 443)
    accept_target_url = True

    def __init__(self, address, token):
        self.authorization = b'Basic ' + base64.b64encode(('prepare:' + token).encode('ascii'))
        self._guard = threading.Lock()
        self._connections = set()
        self._stopping = False
        self._slots = threading.BoundedSemaphore(16)
        super().__init__(address, _Handler)

    def get_request(self):
        connection, address = super().get_request()
        connection.settimeout(managed_proxy.TIMEOUT)
        self.track(connection)
        return connection, address

    def track(self, connection):
        with self._guard:
            if self._stopping:
                connection.close()
                raise OSError('preparation relay closed')
            self._connections.add(connection)

    def connect_public(self, host, port, target_url):
        return connect_default(host, port, target_url,
                               on_socket=self.track, cancelled=self.stopping)

    def stopping(self):
        with self._guard:
            return self._stopping

    def process_request(self, request, address):
        if not self._slots.acquire(blocking=False):
            self.shutdown_request(request)
            return
        try:
            super().process_request(request, address)
        except BaseException:
            self._slots.release()
            raise

    def process_request_thread(self, request, address):
        try:
            super().process_request_thread(request, address)
        finally:
            self._slots.release()
            # 已关闭连接不再留在集合中，长期安装不会积累 socket 对象。
            with self._guard:
                self._connections = {item for item in self._connections if item.fileno() >= 0}

    def shutdown_request(self, request):
        try:
            super().shutdown_request(request)
        finally:
            with self._guard:
                self._connections.discard(request)

    def handle_error(self, request, address):
        # 下游只收到受控错误；不向启动日志泄漏 URL、代理凭据或下载令牌。
        pass

    def close_connections(self):
        with self._guard:
            self._stopping = True
            connections, self._connections = self._connections, set()
        for connection in connections:
            try:
                connection.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            connection.close()


class DownloadRelay:
    """每次准备拥有独立监听端口与随机口令，结束或异常时关闭所有隧道。"""

    def __init__(self):
        self.token = secrets.token_urlsafe(32)
        self.server = None
        self.thread = None

    def __enter__(self):
        # Docker Desktop 通过 host.docker.internal 访问宿主，不能只监听回环。
        # 任意来源均须持有本次准备的随机口令；它只能打开公网下载连接。
        self.server = _Server(('0.0.0.0', 0), self.token)
        self.thread = threading.Thread(target=self.server.serve_forever,
                                       kwargs={'poll_interval': 0.1}, daemon=True)
        self.thread.start()
        return self

    def url(self, *, container=False):
        host = 'host.docker.internal' if container else '127.0.0.1'
        return f'http://prepare:{self.token}@{host}:{self.server.server_port}'

    def environment(self):
        from agent.execution.docker import PROXY_VARIABLES

        result = os.environ.copy()
        for name in PROXY_VARIABLES:
            result[name] = self.url() if name.lower() in {'http_proxy', 'https_proxy'} else ''
        return result

    def redact_log(self, path):
        """原地等长遮盖下载程序错误中可能出现的一次性地址，不复制日志。"""
        needles = [self.token.encode('ascii'),
                   base64.b64encode(('prepare:' + self.token).encode('ascii'))]
        overlap = max(map(len, needles)) - 1
        with path.open('r+b', buffering=0) as stream:
            offset = 0
            while True:
                stream.seek(offset)
                chunk = stream.read(65536)
                if not chunk:
                    break
                clean = chunk
                for needle in needles:
                    clean = clean.replace(needle, b'*' * len(needle))
                if clean != chunk:
                    stream.seek(offset)
                    stream.write(clean)
                if len(chunk) < 65536:
                    break
                offset += len(chunk) - overlap

    def __exit__(self, *args):
        self.server.close_connections()
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)


def docker_proxy_modes():
    """仅读取模式用于诊断；不回传地址/口令，不改用户 Docker 设置。"""
    result = {'desktop': 'unknown', 'containers': 'unknown'}
    roaming = os.getenv('APPDATA')
    if os.name != 'nt' or not roaming:
        return result
    for filename in ('settings-store.json', 'settings.json'):
        path = Path(roaming) / 'Docker' / filename
        try:
            if not path.is_file() or path.stat().st_size > 1024 * 1024:
                continue
            data = json.loads(path.read_text(encoding='utf-8-sig'))
            normalized = {key.lower(): value for key, value in data.items()}
            for label, key in (('desktop', 'proxyhttpmode'), ('containers', 'containersproxymode')):
                value = normalized.get(key)
                result[label] = value if value in {'system', 'manual', 'none'} else 'unknown'
            break
        except (OSError, ValueError, AttributeError):
            break
    return result
