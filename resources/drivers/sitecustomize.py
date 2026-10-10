"""Bundled interpreter bootstrap for the host-selected read-only dependencies."""
import os
import site

directory = os.environ.get('MONOPLANER_DEPENDENCY_DIR')
if directory:
    site.addsitedir(directory)
