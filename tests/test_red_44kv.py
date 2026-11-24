import math
import unittest
import red_44kv as r
import config


class TestRed44(unittest.TestCase):
    def tearDown(self):
        r.configurar()

    def test_longitudes_resistencias(self):
        e = r.Escenario44("documento")
        ramas = r.ramas_documento(e)
        self.assertAlmostEqual(sum(k for _,_,k,_,_ in ramas), 23.7)
        self.assertAlmostEqual(ramas[0][3], 17.24*(10.7/85.03 + .5/150))
        self.assertAlmostEqual(ramas[1][3], 17.24*(.7/53.49 + 6.8/120))
        self.assertAlmostEqual(sum(x[3] for x in ramas[2:]), 17.24*5/53.49)
        self.assertAlmostEqual(sum(x[2] for x in ramas[2:]), 5)

    def test_radial(self):
        e = r.Escenario44("documento")
        p = math.sqrt(3)*44000*.95*100/1000
        v = r.calcular_documento({"SER VA":p}, e)
        self.assertAlmostEqual(v["P_loss_kw"], 3*100**2*(17.24*5/53.49)/1000)
        self.assertAlmostEqual(v["ramas_44kv"][2]["I_A"], 100)
        self.assertAlmostEqual(v["ramas_44kv"][3]["I_A"], 100)
        self.assertEqual(v["SER PO"]["Vac"], 44000)
        self.assertLess(v["SER VA"]["Vac"], v["SER EB"]["Vac"])

    def test_corrientes_acumuladas(self):
        e = r.Escenario44("documento")
        p = math.sqrt(3)*44000*.95*100/1000
        v = r.calcular_documento({"SER ES":p, "SER PO":p, "SER EB":p}, e)
        for rama, i in zip(v["ramas_44kv"], (200,100,100,0)):
            self.assertAlmostEqual(rama["I_A"], i)
        self.assertAlmostEqual(v["P_loss_kw"], sum(x["perdida_kW"] for x in v["ramas_44kv"]))

    def test_parametros(self):
        p = {"SER VA":5000}
        a = r.calcular_documento(p,r.Escenario44("documento"))["P_loss_kw"]
        b = r.calcular_documento(p,r.Escenario44("documento",temperatura_c=60))["P_loss_kw"]
        c = r.calcular_documento(p,r.Escenario44("documento",km_xlpe_va=1))["P_loss_kw"]
        self.assertGreater(b,a)
        self.assertLess(c,a)
        self.assertEqual(r.calcular_documento({}, r.Escenario44("documento"))["P_loss_kw"],0)

    def test_contexto(self):
        from contextvars import copy_context
        ctx=copy_context()
        r.configurar(r.Escenario44("documento"))
        self.assertEqual(ctx.run(r.escenario_actual).modo,"historico")
        self.assertEqual(r.escenario_actual().modo,"documento")


if __name__ == "__main__":
    unittest.main()
