"""仅由受信任准备驱动在隔离容器内调用，不在宿主或工作模型中执行。"""

import os
from pathlib import Path
import runpy
import shutil
import subprocess


def install():
    release = Path('/etc/os-release').read_text('utf-8')
    if 'VERSION_CODENAME=bookworm' not in release:
        raise RuntimeError('Wine installation requires the managed Debian Bookworm base')
    os.environ.update(DEBIAN_FRONTEND='noninteractive', WINEARCH='win64',
                      WINEPREFIX='/opt/monoplanner/wine-template', WINEDEBUG='-all',
                      WINEDLLOVERRIDES='mscoree,mshtml=')
    Path('/tmp').mkdir(exist_ok=True)
    Path('/tmp').chmod(0o1777)
    for arguments in (
        ['apt-get', '-o', 'APT::Sandbox::User=root', 'update'],
        ['apt-get', '-o', 'APT::Sandbox::User=root', 'install', '-y', '--no-install-recommends',
         'wine=8.0~repack-4', 'wine64=8.0~repack-4'],
        ['/usr/bin/wineboot', '-u'],
        ['/usr/bin/wineserver', '-w'],
    ):
        subprocess.run(arguments, check=True)
    folder = Path(__file__).parent
    runpy.run_path(str(folder / 'compact-wine-prefix.py'), run_name='__main__')
    shutil.copyfile(folder / 'wine', '/usr/local/bin/wine')
    Path('/usr/local/bin/wine').chmod(0o755)
    os.environ['WINEPREFIX'] = '/tmp/monoplanner-wine'
    version = subprocess.run(['/usr/local/bin/wine', '--version'],
                             check=True, capture_output=True, text=True).stdout.strip()
    result = subprocess.run(['/usr/local/bin/wine', 'cmd.exe', '/d', '/v:off', '/u', '/s', '/c',
                             'echo MONOPLANNER_WINE_READY'], capture_output=True, check=True)
    output = result.stdout.decode('utf-16-le').strip()
    if output != 'MONOPLANNER_WINE_READY':
        raise RuntimeError('Installed Wine CMD did not return the expected probe receipt')
    shutil.rmtree('/var/lib/apt/lists', ignore_errors=True)
    return {'version': version, 'cmd_exit_code': result.returncode, 'cmd_probe': output}


if __name__ == '__main__':
    print(install(), flush=True)
