"""模型请求使用当前系统网络；代理开关仅影响之后发出的请求。"""

import asyncio
import ctypes
from ctypes import wintypes
from fnmatch import fnmatchcase
from functools import lru_cache
import os
import re
from threading import BoundedSemaphore, Event, Lock, Thread
import urllib.request

import httpx


class _UserProxyConfig(ctypes.Structure):
    _fields_ = [
        ("auto_detect", wintypes.BOOL),
        ("config_url", ctypes.c_void_p),
        ("proxy", ctypes.c_void_p),
        ("bypass", ctypes.c_void_p),
    ]


class _AutoProxyOptions(ctypes.Structure):
    _fields_ = [
        ("flags", wintypes.DWORD),
        ("detect_flags", wintypes.DWORD),
        ("config_url", wintypes.LPCWSTR),
        ("reserved", ctypes.c_void_p),
        ("reserved_flags", wintypes.DWORD),
        ("auto_logon", wintypes.BOOL),
    ]


class _ProxyInfo(ctypes.Structure):
    _fields_ = [
        ("access_type", wintypes.DWORD),
        ("proxy", ctypes.c_void_p),
        ("bypass", ctypes.c_void_p),
    ]


@lru_cache(maxsize=1)
def _windows_api():
    # 只缓存 DLL 绑定，不缓存用户配置或 PAC 的每个 URL 决策。
    winhttp = ctypes.WinDLL("winhttp", use_last_error=True)
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    winhttp.WinHttpGetIEProxyConfigForCurrentUser.argtypes = [
        ctypes.POINTER(_UserProxyConfig)
    ]
    winhttp.WinHttpGetIEProxyConfigForCurrentUser.restype = wintypes.BOOL
    winhttp.WinHttpOpen.argtypes = [
        wintypes.LPCWSTR, wintypes.DWORD, wintypes.LPCWSTR, wintypes.LPCWSTR, wintypes.DWORD
    ]
    winhttp.WinHttpOpen.restype = wintypes.HANDLE
    winhttp.WinHttpSetTimeouts.argtypes = [wintypes.HANDLE] + [ctypes.c_int] * 4
    winhttp.WinHttpSetTimeouts.restype = wintypes.BOOL
    winhttp.WinHttpGetProxyForUrl.argtypes = [
        wintypes.HANDLE, wintypes.LPCWSTR,
        ctypes.POINTER(_AutoProxyOptions), ctypes.POINTER(_ProxyInfo),
    ]
    winhttp.WinHttpGetProxyForUrl.restype = wintypes.BOOL
    winhttp.WinHttpCloseHandle.argtypes = [wintypes.HANDLE]
    winhttp.WinHttpCloseHandle.restype = wintypes.BOOL
    kernel.GlobalFree.argtypes = [ctypes.c_void_p]
    kernel.GlobalFree.restype = ctypes.c_void_p
    return winhttp, kernel


def _take_strings(structure, names, kernel):
    # WinHTTP 分配的字符串必须使用 GlobalFree，保留地址而不是自动转换成 Python 字符串。
    values = {}
    try:
        for name in names:
            pointer = getattr(structure, name)
            values[name] = ctypes.wstring_at(pointer) if pointer else ""
        return values
    finally:
        for name in names:
            pointer = getattr(structure, name)
            if pointer:
                kernel.GlobalFree(pointer)
                setattr(structure, name, None)


def _windows_settings():
    winhttp, kernel = _windows_api()
    config = _UserProxyConfig()
    if not winhttp.WinHttpGetIEProxyConfigForCurrentUser(ctypes.byref(config)):
        error = ctypes.get_last_error()
        if error == 2:  # 当前用户没有代理配置，相当于新安装的直连环境。
            return {"auto_detect": False, "config_url": "", "proxy": "", "bypass": ""}
        raise httpx.ProxyError(f"Cannot read Windows network settings (error {error})")
    return {
        "auto_detect": bool(config.auto_detect),
        **_take_strings(config, ("config_url", "proxy", "bypass"), kernel),
    }


_PAC_SLOTS = BoundedSemaphore(4)
_PAC_TIMEOUT = 10.0


def _windows_auto_proxy(url, settings):
    # WinHTTP 同步解析无法跨线程安全关闭。等待有总时限，在途原生调用也有数量上限。
    # 超时只停止本次请求；残留解析返回后自行释放，不再发送页面或模型请求。
    if not _PAC_SLOTS.acquire(blocking=False):
        raise httpx.ProxyError("Windows proxy resolver is busy; no request was sent")
    done, result = Event(), {}

    def resolve():
        try:
            result["route"] = _windows_resolve_auto_proxy(url, settings)
        except BaseException as exc:
            result["error"] = exc
        finally:
            _PAC_SLOTS.release()
            done.set()

    worker = Thread(target=resolve, name="monoplanner-pac", daemon=True)
    try:
        worker.start()
    except BaseException:
        _PAC_SLOTS.release()
        raise
    if not done.wait(_PAC_TIMEOUT):
        raise httpx.ProxyError("Windows proxy resolution timed out; no request was sent")
    if "error" in result:
        raise result["error"]
    return result["route"]


def _windows_resolve_auto_proxy(url, settings):
    winhttp, kernel = _windows_api()
    session = winhttp.WinHttpOpen("MonoPlanner/1.0", 1, None, None, 0)
    if not session:
        raise httpx.ProxyError("Cannot open Windows proxy resolver")
    info = _ProxyInfo()
    try:
        if not winhttp.WinHttpSetTimeouts(session, 5000, 5000, 5000, 5000):
            raise httpx.ProxyError("Cannot set Windows proxy resolver timeouts")
        config_url = settings["config_url"]
        # 有明确 PAC 时使用它；不把下载失败悄悄降级为自动发现或直连。
        options = _AutoProxyOptions(
            2 if config_url else 1,
            0 if config_url else 3,
            config_url or None,
            None,
            0,
            False,  # PAC 下载不自动向网络发送当前用户的 Windows 凭据。
        )
        if not winhttp.WinHttpGetProxyForUrl(
            session, str(url), ctypes.byref(options), ctypes.byref(info)
        ):
            error = ctypes.get_last_error()
            if error == 12180 and not config_url:
                # 只开启 WPAD、网络没有发布 PAC 时，沿用当前手动配置或直连。
                return settings["proxy"], settings["bypass"]
            raise httpx.ProxyError(f"Windows PAC resolution failed (error {error})")
        values = _take_strings(info, ("proxy", "bypass"), kernel)
        if info.access_type == 1:
            return "", ""
        if info.access_type != 3 or not values["proxy"]:
            raise httpx.ProxyError("Windows PAC returned an unsupported route")
        return values["proxy"], values["bypass"]
    finally:
        _take_strings(info, ("proxy", "bypass"), kernel)
        winhttp.WinHttpCloseHandle(session)


def _windows_bypass(url, bypass):
    host = url.host.lower()
    authority = url.netloc.decode("ascii").lower()
    for pattern in re.split(r"[;\s]+", bypass.lower()):
        if pattern == "<local>" and "." not in host and ":" not in host:
            return True
        if pattern and not pattern.startswith("<"):
            if fnmatchcase(host, pattern) or fnmatchcase(authority, pattern):
                return True
    return False


def _validated_proxy(proxy):
    if not proxy:
        return None
    try:
        parsed = httpx.URL(proxy if "://" in proxy else "http://" + proxy)
    except (httpx.InvalidURL, ValueError):
        raise httpx.ProxyError("System proxy address is invalid") from None
    if parsed.scheme not in {"http", "https"}:
        raise httpx.ProxyError("System proxy must provide an HTTP or HTTPS proxy endpoint")
    if not parsed.host or parsed.path not in {"", "/"} or parsed.query or parsed.fragment:
        raise httpx.ProxyError("System proxy address is invalid")
    return str(parsed).rstrip("/")


def _windows_proxy_for(url):
    settings = _windows_settings()
    proxy, bypass = settings["proxy"], settings["bypass"]
    if settings["config_url"] or settings["auto_detect"]:
        proxy, bypass = _windows_auto_proxy(url, settings)
    if not proxy or _windows_bypass(url, bypass):
        return None

    # WinHTTP 的列表允许协议映射和多个备用代理；只选第一条匹配路线，不重发请求。
    routes = [entry for entry in re.split(r"[;\s]+", proxy) if entry]
    mapped, general = {}, []
    for entry in routes:
        if "=" in entry:
            scheme, address = entry.split("=", 1)
            mapped.setdefault(scheme.lower(), address)
        else:
            general.append(entry)
    selected = mapped.get(url.scheme) or (general[0] if general else None)
    if selected is None and "socks" in mapped:
        raise httpx.ProxyError("System SOCKS-only proxy is unsupported; use an HTTP endpoint")
    return _validated_proxy(selected)


def system_proxy_for(url: httpx.URL) -> str | None:
    """每次按目标 URL 读取生效网络配置，不让启动时旧环境覆盖 Windows 开关。"""
    if os.name == "nt":
        return _windows_proxy_for(url)
    proxies = urllib.request.getproxies()
    proxy = proxies.get(url.scheme) or proxies.get("all")
    if not proxy or urllib.request.proxy_bypass(url.netloc.decode("ascii")):
        return None
    return _validated_proxy(proxy)


class _ProxyPools:
    """每条路线复用连接池；开关变化只影响随后发出的请求。"""

    def __init__(self, transport_class):
        self._transport_class = transport_class
        self._pools = {}
        self._lock = Lock()

    def for_proxy(self, proxy):
        with self._lock:
            if proxy not in self._pools:
                self._pools[proxy] = self._transport_class(
                    proxy=proxy,
                    trust_env=False,
                    retries=0,
                )
            return self._pools[proxy]

    def pop_all(self):
        with self._lock:
            pools = list(self._pools.values())
            self._pools.clear()
            return pools


class SystemProxyTransport(httpx.BaseTransport):
    def __init__(self):
        self._routes = _ProxyPools(httpx.HTTPTransport)

    def handle_request(self, request):
        proxy = system_proxy_for(request.url)
        return self._routes.for_proxy(proxy).handle_request(request)

    def close(self):
        for transport in self._routes.pop_all():
            transport.close()


class AsyncSystemProxyTransport(httpx.AsyncBaseTransport):
    def __init__(self):
        self._routes = _ProxyPools(httpx.AsyncHTTPTransport)

    async def handle_async_request(self, request):
        # PAC 可能包含网络发现；不阻塞事件循环，取消后不会继续发出模型请求。
        proxy = await asyncio.to_thread(system_proxy_for, request.url)
        return await self._routes.for_proxy(proxy).handle_async_request(request)

    async def aclose(self):
        for transport in self._routes.pop_all():
            await transport.aclose()


def model_http_clients(timeout):
    """隔离本项目的 HTTP 客户端，不改全局环境或系统设置。"""
    return {
        "http_client": httpx.Client(
            transport=SystemProxyTransport(),
            trust_env=False,
            timeout=timeout,
        ),
        "http_async_client": httpx.AsyncClient(
            transport=AsyncSystemProxyTransport(),
            trust_env=False,
            timeout=timeout,
        ),
    }
