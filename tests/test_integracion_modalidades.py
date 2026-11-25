"""Comprobaciones estáticas de la integración de vistas sin dependencias UI."""
import ast
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]


class IntegracionModalidades(unittest.TestCase):
    def test_app_unico(self):
        app = (ROOT / "app.py").read_text(encoding="utf-8")
        ast.parse(app)
        self.assertIn("Modelación por planilla", app)
        self.assertIn("Modelación por THDR", app)
        self.assertIn("render_modelacion_thdr(config_sig=get_config_hash())", app)
        self.assertEqual(app.count("_cat.selector_streamlit("), 1)
        self.assertEqual(app.count("_r44.selector_streamlit("), 1)
        self.assertFalse((ROOT / "pages/1_Simular_THDR_GitHub.py").exists())

    def test_thdr_sin_acceso_duplicado(self):
        src = (ROOT / "thdr_ui.py").read_text(encoding="utf-8")
        tree = ast.parse(src)
        self.assertTrue(any(isinstance(n, ast.FunctionDef) and n.name == "render_modelacion_thdr"
                            for n in tree.body))
        self.assertNotIn("from app import", src)
        self.assertNotIn("_cat.selector_streamlit(", src)
        self.assertNotIn("_r44.selector_streamlit(", src)
        self.assertIn('resultado[4] != -1', src)

    def test_defaults(self):
        for nombre, index in [("catenaria.py", 1), ("red_44kv.py", 2)]:
            t = ast.parse((ROOT / nombre).read_text(encoding="utf-8"))
            f = next(n for n in t.body if isinstance(n, ast.FunctionDef) and n.name == "selector_streamlit")
            select = next(n for n in ast.walk(f) if isinstance(n, ast.Call)
                          and isinstance(n.func, ast.Attribute) and n.func.attr == "selectbox")
            self.assertEqual(next(k.value.value for k in select.keywords if k.arg == "index"), index)
