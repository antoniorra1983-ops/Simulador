import unittest
import red_44kv as r


class Registro2013Test(unittest.TestCase):
    def test_datos(self):
        e=r.Escenario44("registro2013")
        a,b,c,d=r.ramas_registro(e)
        self.assertAlmostEqual(sum(x[2] for x in (a,b,c,d)),26.3)
        self.assertAlmostEqual(a[3],11.2*.2374)
        self.assertAlmostEqual(b[4],10.3*.5301)
        self.assertAlmostEqual(b[5],10.3*60.4e-6)
        self.assertAlmostEqual(c[3]+d[3],4.8*.3765)
        self.assertAlmostEqual(c[4]/c[2],d[4]/d[2])
        self.assertEqual(c[6],276)
        self.assertEqual(d[6],276)

    def test_pi_balance(self):
        e=r.Escenario44("registro2013")
        p={"SER ES":3000,"SER PO":3000,"SER EB":2000,"SER VA":2000}
        f=r.calcular_registro(p,e)
        self.assertTrue(f["convergio_44kv"])
        self.assertAlmostEqual(f["P_seat_kw"],sum(p.values())+f["P_loss_kw"],places=1)
        self.assertGreater(f["P_loss_kw"],0)
        self.assertLess(f["SER VA"]["Vac"],f["SER EB"]["Vac"])
        self.assertGreater(f["ramas_44kv"][2]["I_A"],f["ramas_44kv"][3]["I_A"])

    def test_capacitancia_reactores(self):
        f=r.calcular_registro({},r.Escenario44("registro2013"))
        g=r.calcular_registro({},r.Escenario44("registro2013",reactores_po_activos=True))
        self.assertTrue(f["convergio_44kv"])
        self.assertTrue(g["convergio_44kv"])
        self.assertLess(f["Q_seat_kvar"],0)
        self.assertGreater(g["Q_seat_kvar"],f["Q_seat_kvar"])
        self.assertGreater(f["P_loss_kw"],0)


if __name__=="__main__":
    unittest.main()
