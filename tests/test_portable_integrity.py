"""The distributed tree must contain every input required to start a task."""
import ast
import json
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

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
            (root/'launcher/Applications.cs').unlink()
            with self.assertRaises(FileNotFoundError):
                code_key(root)

    def test_packaging_copies_required_launcher_inputs(self):
        spec = spec_from_file_location('portable_build', PROJECT_ROOT/'scripts/build-portable.py')
        module = module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as folder:
            module.copy_launcher_inputs(Path(folder))
            for name in ('Applications.cs', 'FilePicker.cs'):
                self.assertEqual((Path(folder)/'launcher'/name).read_bytes(),
                                 (PROJECT_ROOT/'launcher'/name).read_bytes())
            self.assertFalse((Path(folder)/'launcher/Sandbox.cs').exists())

    def test_runtime_copy_omits_retired_helper_but_keeps_active_helpers(self):
        spec = spec_from_file_location('portable_build', PROJECT_ROOT/'scripts/build-portable.py')
        module = module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as folder:
            source, target = Path(folder)/'source', Path(folder)/'package'
            source.mkdir()
            for name in ('MonoPlanner.Sandbox.exe', 'MonoPlanner.Sandbox.exe.config',
                         'MonoPlanner.FilePicker.exe', 'MonoPlanner.Applications.exe'):
                (source/name).write_bytes(b'helper')
            module.copy_tree(source, target,
                             exclude={'MonoPlanner.Sandbox.exe', 'MonoPlanner.Sandbox.exe.config'})
            self.assertEqual({p.name for p in target.iterdir()},
                             {'MonoPlanner.FilePicker.exe', 'MonoPlanner.Applications.exe'})

    def test_python_runtime_only_ships_locked_dependencies(self):
        spec = spec_from_file_location('portable_build', PROJECT_ROOT/'scripts/build-portable.py')
        module = module_from_spec(spec); spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root/'requirements.lock').write_text('product==1.0\n')
            source, output = root/'runtime', root/'package'
            packages = source/'Lib/site-packages'
            packages.mkdir(parents=True)
            (source/'python.exe').write_bytes(b'fixed interpreter')
            for name in ('product', 'pytest'):
                metadata = packages/(name+'-1.0.dist-info'); metadata.mkdir()
                (metadata/'METADATA').write_text('Name: '+name+'\nVersion: 1.0\n')
                (packages/(name+'.py')).write_text('# '+name+'\n')
                (metadata/'RECORD').write_text(name+'.py,,\n'+metadata.name+'/METADATA,,\n'+metadata.name+'/RECORD,,\n')
            with patch.object(module, 'ROOT', root):
                self.assertEqual(module.copy_python_runtime(source, output), {'product'})
                self.assertTrue((output/'Lib/site-packages/product.py').is_file())
                self.assertFalse((output/'Lib/site-packages/pytest.py').exists())
                self.assertFalse((output/'Lib/site-packages/pytest-1.0.dist-info').exists())
                self.assertFalse((output/'Lib/site-packages/sitecustomize.py').exists())
                (root/'requirements.lock').write_text('product==2.0\n')
                with self.assertRaisesRegex(ValueError, 'differs from lock'):
                    module.copy_python_runtime(source, root/'stale')

    def test_existing_package_directory_is_never_reused(self):
        spec = spec_from_file_location('portable_build', PROJECT_ROOT/'scripts/build-portable.py')
        module = module_from_spec(spec); spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            target = root/'dist/MonoPlanner-old'; target.mkdir(parents=True)
            sentinel = target/'old.txt'; sentinel.write_text('preserve')
            with patch.object(module, 'ROOT', root), self.assertRaisesRegex(ValueError, 'never reused'):
                module.build(target)
            self.assertEqual(sentinel.read_text(), 'preserve')

    def test_generated_frontend_metadata_does_not_disclose_build_paths(self):
        spec = spec_from_file_location('portable_build', PROJECT_ROOT/'scripts/build-portable.py')
        module = module_from_spec(spec); spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); (root/'.next').mkdir()
            config = {'repoRoot': 'private-build-path', 'outputFileTracingRoot': 'private-build-path',
                      'turbopack': {'root': 'private-build-path'}, 'poweredByHeader': False}
            (root/'server.js').write_text("const hostname = process.env.HOSTNAME || '0.0.0.0'\nconst nextConfig = "+json.dumps(config)+"\nprocess.env.__NEXT_PRIVATE_STANDALONE_CONFIG = JSON.stringify(nextConfig)\n")
            private_directory = r'E:\Users\private-owner\project\web'
            (root/'.next/required-server-files.json').write_text(json.dumps({'config': config, 'appDir': private_directory, 'relativeAppDir': '.'}))
            server = root/'.next/server'; server.mkdir()
            references = server/'page_client-reference-manifest.js'
            bundle = server/'page.js'
            client_id = private_directory + r'\src\page.tsx'
            references.write_text('self.manifest = '+json.dumps({client_id: {'id': 123}}))
            bundle.write_text('registerClientReference('+json.dumps(client_id)+');const diagnostic='+json.dumps(json.dumps(client_id)))
            module.sanitize_frontend_metadata(root)
            self.assertNotIn('private-build-path', (root/'server.js').read_text())
            self.assertNotIn('private-build-path', (root/'.next/required-server-files.json').read_text())
            self.assertIn("const hostname = '127.0.0.1'", (root/'server.js').read_text())
            self.assertIn('nextConfig.turbopack.root = dir', (root/'server.js').read_text())
            for generated in (references, bundle):
                self.assertNotIn('private-owner', generated.read_text())
                self.assertIn(json.dumps(r'/monoplanner/web\src\page.tsx')[1:-1], generated.read_text())
