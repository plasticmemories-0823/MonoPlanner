"""Local defaults and a Windows-user encrypted provider credential."""
import base64
import ctypes
import json
import os
from pathlib import Path
import tempfile

DEFAULTS = {'model_name': 'deepseek-flash', 'reasoning_effort': 'high', 'online': False}
EFFORTS = ('none', 'low', 'high', 'max')


def state_root():
    return Path(os.getenv('MONOPLANER_STATE_DIR', Path.home()/'.local/state/monoplaner')).resolve()


def private_path(path):
    """Application state and launcher credentials never become model materials."""
    roots = [state_root()]
    if os.getenv('LOCALAPPDATA'):
        roots.append(Path(os.environ['LOCALAPPDATA'])/'MonoPlanner')
    return any(Path(path).resolve().is_relative_to(root.resolve()) for root in roots)


def protect(value, *, decrypt=False):
    if os.name != 'nt':
        raise ValueError('持久保存密钥需要 Windows DPAPI；其他系统请使用环境变量')
    from ctypes import wintypes
    class Blob(ctypes.Structure):
        _fields_ = [('size', wintypes.DWORD), ('data', ctypes.POINTER(ctypes.c_ubyte))]
    buffer = ctypes.create_string_buffer(value)
    source = Blob(len(value), ctypes.cast(buffer, ctypes.POINTER(ctypes.c_ubyte)))
    target = Blob()
    method = ctypes.windll.crypt32.CryptUnprotectData if decrypt else ctypes.windll.crypt32.CryptProtectData
    method.argtypes = [ctypes.POINTER(Blob), ctypes.c_void_p, ctypes.c_void_p,
                       ctypes.c_void_p, ctypes.c_void_p, wintypes.DWORD, ctypes.POINTER(Blob)]
    method.restype = wintypes.BOOL
    if not method(ctypes.byref(source), None, None, None, None, 1, ctypes.byref(target)):
        raise ValueError('Windows 无法读取或保存密钥，请在当前用户下重新配置')
    try:
        return ctypes.string_at(target.data, target.size)
    finally:
        release = ctypes.windll.kernel32.LocalFree
        release.argtypes = [ctypes.c_void_p]
        release.restype = ctypes.c_void_p
        release(target.data)


def atomic_write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise ValueError('配置文件不能是链接')
    name = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as out:
            name = out.name
            out.write(value)
            out.flush()
            os.fsync(out.fileno())
        os.replace(name, path)
    finally:
        if name and Path(name).exists():
            Path(name).unlink()


class Preferences:
    def __init__(self, root=None):
        self.root = Path(root) if root else state_root()
        self.path = self.root/'preferences.json'
        self.secret = self.root/'deepseek-key.dpapi'

    @staticmethod
    def validate(value):
        if value.get('model_name') != 'deepseek-flash':
            raise ValueError('当前仅支持 DeepSeek Flash')
        if value.get('reasoning_effort') not in EFFORTS:
            raise ValueError('思考程度必须是 none、low、high 或 max')
        if type(value.get('online')) is not bool:
            raise ValueError('联网设置必须是布尔值')
        return {name: value[name] for name in DEFAULTS}

    def defaults(self):
        return self.validate(json.loads(self.path.read_text(encoding='utf-8'))) if self.path.exists() else dict(DEFAULTS)

    def key(self):
        if self.secret.exists():
            if self.secret.is_symlink():
                raise ValueError('密钥文件不能是链接')
            return protect(base64.b64decode(self.secret.read_bytes(), validate=True), decrypt=True).decode('utf-8')
        from dotenv import load_dotenv
        from agent.settings import PROJECT_ROOT
        load_dotenv(PROJECT_ROOT/'.env')
        return os.getenv('DEEPSEEK_API_KEY', '').strip()

    def public(self):
        return {**self.defaults(), 'has_api_key': bool(self.key()),
                'key_source': 'local' if self.secret.exists() else 'environment',
                'models': [{'id': 'deepseek-flash', 'name': 'DeepSeek Flash'}], 'efforts': list(EFFORTS)}

    def update(self, value):
        if not isinstance(value, dict) or set(value) - {*DEFAULTS, 'api_key'}:
            raise ValueError('配置包含未知字段')
        public = self.validate({**self.defaults(), **{k: v for k, v in value.items() if k in DEFAULTS}})
        key = value.get('api_key', '')
        if not isinstance(key, str) or '\n' in key or '\r' in key:
            raise ValueError('API Key 必须是单行文本')
        if key.strip():
            atomic_write(self.secret, base64.b64encode(protect(key.strip().encode('utf-8'))))
        atomic_write(self.path, json.dumps(public, ensure_ascii=False).encode('utf-8'))
        return self.public()

    def delete_key(self):
        if self.secret.is_symlink():
            raise ValueError('密钥文件不能是链接')
        self.secret.unlink(missing_ok=True)
        return self.public()
