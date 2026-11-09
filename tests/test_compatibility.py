import importlib.util
import sys
import tempfile
from pathlib import Path
import types
import unittest
from unittest.mock import patch

from training.compatibility import normalize_legacy_config, load_normal_mechanism


@unittest.skipUnless(importlib.util.find_spec('yacs'), 'requires training dependencies')
class ConfigCompatibilityTest(unittest.TestCase):
    def test_legacy_names_are_translated_without_changing_the_source(self):
        from yacs.config import CfgNode
        source = CfgNode({
            'MODEL': {'NAME': 'SHARK'},
            'SHARK': {'HEAD': {'NAME': 'SharkHead'},
                      'EVAL': {'SCORE_SOURCE': 'shark_head'},
                      'WEIGHT': 0.5, 'PATH': '/archive/shark/checkpoint.pth'},
        })
        source.freeze()
        before = source.dump()
        result = normalize_legacy_config(source)
        self.assertEqual(source.dump(), before)
        self.assertTrue(result.is_frozen())
        self.assertNotIn('SHARK', result)
        self.assertEqual(result.MODEL.NAME, 'GFM4MTSAD')
        self.assertEqual(result.GFM4MTSAD.HEAD.NAME, 'GFM4MTSADHead')
        self.assertEqual(result.GFM4MTSAD.EVAL.SCORE_SOURCE, 'gfm4mtsad_head')
        self.assertEqual(result.GFM4MTSAD.WEIGHT, 0.5)
        self.assertEqual(result.GFM4MTSAD.PATH, '/archive/shark/checkpoint.pth')

    def test_current_defaults_are_unchanged(self):
        from config import get_cfg_defaults
        source = get_cfg_defaults()
        self.assertEqual(normalize_legacy_config(source).dump(), source.dump())

    def test_ambiguous_namespaces_are_rejected(self):
        from yacs.config import CfgNode
        with self.assertRaisesRegex(ValueError, 'Ambiguous config'):
            normalize_legacy_config(CfgNode({'SHARK': {}, 'GFM4MTSAD': {}}))

    def test_archived_training_routes_fail_explicitly(self):
        from config import get_cfg_defaults
        from training.config_merge import require_main_recipe
        cfg = get_cfg_defaults()
        cfg.GFM4MTSAD.SYNTHETIC.CANDIDATE_GENERATOR = 'graph_var'
        cfg.GFM4MTSAD.SYNTHETIC.NEGATIVE_SCREENING_MODE = 'legacy_boundary'
        require_main_recipe(cfg)
        cfg.GFM4MTSAD.OTRCL.ENABLE = True
        with self.assertRaisesRegex(ValueError, 'not part of the released main recipe'):
            require_main_recipe(cfg)


@unittest.skipUnless(importlib.util.find_spec('torch'), 'requires training dependencies')
class ReferenceCompatibilityTest(unittest.TestCase):
    def test_legacy_reference_loads_without_installing_legacy_modules(self):
        import torch
        from models.gfm4mtsad.normal_mechanism import GlobalGraphVARMechanism
        module_name = 'models.shark.normal_mechanism'
        module = types.ModuleType(module_name)
        legacy_class = type('GlobalGraphVARMechanism', (), {'__module__': module_name})
        module.GlobalGraphVARMechanism = legacy_class
        package = types.ModuleType('models.shark')
        package.__path__ = []
        package.normal_mechanism = module
        with tempfile.TemporaryDirectory() as folder:
            file = Path(folder) / 'legacy.pt'
            reference = legacy_class()
            reference.reference_residual = torch.tensor([[0.25, 0.75]])
            reference.fitted = True
            with patch.dict(sys.modules, {'models.shark': package, module_name: module}):
                torch.save(reference, file)
            before = file.read_bytes()
            restored = load_normal_mechanism(file)
            self.assertIsInstance(restored, GlobalGraphVARMechanism)
            torch.testing.assert_close(restored.reference_residual, reference.reference_residual,
                                       rtol=0, atol=0)
            self.assertTrue(restored.fitted)
            self.assertEqual(file.read_bytes(), before)
            self.assertNotIn(module_name, sys.modules)
            current = Path(folder) / 'current.pt'
            torch.save(restored, current)
            reloaded = load_normal_mechanism(current)
            self.assertEqual(type(reloaded).__module__, 'models.gfm4mtsad.normal_mechanism')
            torch.testing.assert_close(reloaded.reference_residual, reference.reference_residual,
                                       rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
