import unittest
import catenaria as c


class TestCatenaria(unittest.TestCase):
    def tearDown(self):
        c.configurar()

    def test_historico(self):
        for k, r in [(0, .0638), (2.25, .0530), (6.8, .0495),
                     (10.92, .0417), (21.41, .0399), (30.36, .0355)]:
            self.assertAlmostEqual(c.resistencia_km(k), r)
        self.assertAlmostEqual(c.resistencia_entre(2, 3), .25*.0638 + .75*.053)

    def test_siemens(self):
        p = c.EscenarioCatenaria("siemens_principal")
        s = c.EscenarioCatenaria("siemens_secundaria")
        self.assertAlmostEqual(c.resistencia_siemens(p)[0], 17.24/364)
        self.assertAlmostEqual(c.resistencia_siemens(s)[0], 17.24/257)
        for e, sign in [(c.EscenarioCatenaria("siemens_principal", 60), 1),
                        (c.EscenarioCatenaria("siemens_principal", 20, 20), 1),
                        (c.EscenarioCatenaria("siemens_principal", 20, 0, 150), -1),
                        (c.EscenarioCatenaria("siemens_principal", 20, 0, 0, .02), 1)]:
            self.assertGreater(sign * (c.resistencia_siemens(e)[1] - c.resistencia_siemens(p)[1]), 0)

    def test_alimentacion(self):
        c.configurar(c.EscenarioCatenaria("siemens_principal"), [(0, "A"), (10, "B")])
        r = 17.24/364
        self.assertAlmostEqual(c.resistencia_equivalente(5), 2.5*r)
        self.assertAlmostEqual(c.resistencia_equivalente(5, [(0,"A")]), 5*r)
        self.assertAlmostEqual(c.resistencia_equivalente(10), 0)
        self.assertAlmostEqual(sum(f for _, f, _ in c.ramas_alimentacion(2)), 1)
        self.assertAlmostEqual(c.resistencia_entre(5, 2), 3*r)
        e = c.distribuir_energia_bilateral(3295, 1, 5, 5, c.sers_activas(), 3295)
        self.assertAlmostEqual(sum(e.values()), 3295 + 1000**2*2.5*r/1000)
        self.assertAlmostEqual(e["A"], e["B"])
        self.assertNotEqual(c.firma(), ())

    def test_aislamiento_contexto(self):
        from contextvars import copy_context
        ctx = copy_context()
        c.configurar(c.EscenarioCatenaria("siemens_principal"))
        self.assertEqual(ctx.run(c.escenario_actual).modo, "historico")
        self.assertEqual(c.escenario_actual().modo, "siemens_principal")


if __name__ == "__main__":
    unittest.main()
