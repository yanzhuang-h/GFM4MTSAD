"""Load the pinned upstream model without redistributing its source."""
import ast
import hashlib
from pathlib import Path
import types

SOURCE_SHA256 = '870ff2d6cefef715a4fd701f001a9bda98dd7f808a3c431d5edec37e01e3ace1'


def verified_source(path):
    path = Path(path).expanduser().resolve()
    # Canonical line endings allow both a Git checkout and a downloaded file.
    text = path.read_text(encoding='utf-8').rstrip() + '\n'
    if hashlib.sha256(text.encode()).hexdigest() != SOURCE_SHA256:
        raise ValueError('AnomalyGFM source differs from the pinned version; see docs/ASSETS.md')
    return path, text


def load_model_class(path):
    path, text = verified_source(path)
    tree = ast.parse(text, filename=str(path))
    # The dense GFM4MTSAD path never uses upstream's optional torch_sparse import.
    tree.body = [node for node in tree.body if not (
        isinstance(node, ast.Import) and len(node.names) == 1
        and node.names[0].name == 'torch_sparse')]
    fixes = 0
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == 'torch' and node.func.attr == 'eye'):
            # Preserve the validated adapter's device/dtype fix for GPU input.
            node.keywords.extend([
                ast.keyword(arg='device', value=ast.Attribute(
                    value=ast.Name(id='raw_adj', ctx=ast.Load()), attr='device', ctx=ast.Load())),
                ast.keyword(arg='dtype', value=ast.Attribute(
                    value=ast.Name(id='raw_adj', ctx=ast.Load()), attr='dtype', ctx=ast.Load())),
            ])
            fixes += 1
    if fixes != 1:
        raise ValueError('Unexpected upstream adjacency implementation')
    module = types.ModuleType('_gfm4mtsad_anomalygfm_source')
    module.__file__ = str(path)
    exec(compile(ast.fix_missing_locations(tree), str(path), 'exec'), module.__dict__)
    return module.Model
