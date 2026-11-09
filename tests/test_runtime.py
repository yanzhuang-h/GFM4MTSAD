import ast
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class RuntimeTest(unittest.TestCase):
    def test_all_six_recipes_are_registered(self):
        specs = json.loads((ROOT/'configs/sources.json').read_text())['sources']
        self.assertEqual({(s['entity'],s['backend']) for s in specs}, {
            (e,b) for e in ('SMD_1-1','SMAP_T-3','MSL_P-15')
            for b in ('anomalygfm','owleye')})
        for s in specs:
            self.assertEqual(hashlib.sha256((ROOT/s['snapshot']).read_bytes()).hexdigest(), s['sha256'])

    def test_main_pipeline_requires_explicit_output(self):
        from main import parse_args
        with self.assertRaises(SystemExit):
            parse_args(['run','--entity','SMD_1-1','--backend','anomalygfm','--assets','assets.json'])
        args = parse_args(['run','--entity','SMD_1-1','--backend','anomalygfm',
                          '--assets','assets.json','--output','outputs/test'])
        self.assertEqual(args.seed,0)

    def test_assets_are_relative_to_json(self):
        from training.assets import load_assets
        with tempfile.TemporaryDirectory() as d:
            file=Path(d)/'assets.json'
            file.write_text(json.dumps({'data_dir':'../data'}))
            self.assertEqual(load_assets(file)['data_dir'],str((Path(d)/'../data').resolve()))

    def test_changed_upstream_source_is_rejected(self):
        from models.anomalygfm.source import verified_source
        with tempfile.TemporaryDirectory() as d:
            file=Path(d)/'model.py'; file.write_text('print("unexpected")\n')
            with self.assertRaisesRegex(ValueError,'pinned version'):
                verified_source(file)

    def test_research_experiments_are_not_imported(self):
        for path in ROOT.rglob('*.py'):
            if any(x.startswith('.') or x in ('assets','outputs') for x in path.relative_to(ROOT).parts):
                continue
            for node in ast.walk(ast.parse(path.read_text(encoding='utf-8'))):
                modules=([node.module or ''] if isinstance(node,ast.ImportFrom) else
                         [a.name for a in node.names] if isinstance(node,ast.Import) else [])
                self.assertFalse(any(m.startswith(('shark_scripts','gfm4mtsad_scripts','experiments.')) for m in modules),str(path))
        self.assertEqual([p.name for p in (ROOT/'experiments').iterdir()],['README.md'])


if __name__=='__main__': unittest.main()
