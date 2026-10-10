"""The distributed tree must contain every input required to start a task."""
import ast
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import shutil
import tempfile
import unittest

from agent.settings import PROJECT_ROOT
from agent.storage.recovery import code_key


class PortableIntegrityTests(unittest.TestCase):
    def test_packaged_fingerprint_matches_source_and_rejects_missing_input(self):
        tree = ast.parse((PROJECT_ROOT/'agent/storage/recovery.py').read_text(encoding='utf-8'))
        function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'code_key')
        names = ast.literal_eval(next(n.value for n in function.body if isinstance(n, ast.Assign)))
        from agent.model.tools import SCHEMA_PATH
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for relative in {*names, SCHEMA_PATH.relative_to(PROJECT_ROOT).as_posix()}:
                target = root/relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(PROJECT_ROOT/relative, target)
            self.assertEqual(code_key(root), code_key())
            (root/'launcher/Sandbox.cs').unlink()
            with self.assertRaises(FileNotFoundError):
                code_key(root)

    def test_packaging_copies_required_launcher_inputs(self):
        spec = spec_from_file_location('portable_build', PROJECT_ROOT/'scripts/build-portable.py')
        module = module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as folder:
            module.copy_launcher_inputs(Path(folder))
            for name in ('Sandbox.cs', 'Sandbox.exe.config'):
                self.assertEqual((Path(folder)/'launcher'/name).read_bytes(),
                                 (PROJECT_ROOT/'launcher'/name).read_bytes())
