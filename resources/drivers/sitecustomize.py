"""Bundled interpreter bootstrap for the host-selected read-only dependencies."""
import os
import site
import sys

# The embedded interpreter's ._pth deliberately excludes the host's cwd.
# Only the executor can supply this workspace binding; it remains subject to
# the sandbox's existing directory ACLs.
workspace = os.environ.get('MONOPLANER_WORKSPACE_DIR')
if workspace:
    entry = sys.argv[0] if sys.argv else ''
    directory = os.path.dirname(os.path.abspath(entry)) if os.path.isfile(entry) else workspace
    for path in reversed(list(dict.fromkeys([directory, workspace]))):
        if path not in sys.path:
            sys.path.insert(0, path)

directory = os.environ.get('MONOPLANER_DEPENDENCY_DIR')
if directory:
    site.addsitedir(directory)
