"""Red trifásica radial de 44 kV: comparación V136 / documento AT-MT.
El Belloto no figura en el documento. Se conserva como derivación intermedia
hipotética dentro de los 5 km del alimentador a Villa Alemana.
No se infiere el conexionado interno de transformadores ni su placa.
"""
from contextvars import ContextVar
from dataclasses import dataclass, asdict
import math
import config


@dataclass(frozen=True)
class Escenario44:
    modo: str = "historico"
    temperatura_c: float = 20.0
    factor_r_aereo: float = 1.0
    factor_r_xlpe: float = 1.0
    x_aereo: float = config.RED44_X_AEREO_REF
    x_xlpe: float = config.RED44_X_XLPE_REF
    fp: float = config.RED44_PF_REF
    km_eb: float = 1.2
    km_xlpe_va: float = 0.0
    temperatura_registro_c: float = 30.0
    reactores_po_activos: bool = False

    def __post_init__(self):
        vals = tuple(v for k, v in asdict(self).items() if k != "modo")
        if not all(math.isfinite(v) for v in vals):
            raise ValueError("Parámetros de 44 kV no finitos")
        if self.modo not in ("historico", "documento", "registro2013"):
            raise ValueError("Escenario 44 kV desconocido")
        if not (-30 <= self.temperatura_c <= 120 and
                0 < self.factor_r_aereo <= 5 and 0 < self.factor_r_xlpe <= 5 and
                0 <= self.x_aereo <= 2 and 0 <= self.x_xlpe <= 2 and
                .5 <= self.fp <= 1 and 0 < self.km_eb < 5 and
                0 <= self.km_xlpe_va <= 5 and
                -30 <= self.temperatura_registro_c <= 120 and
                (self.modo != "registro2013" or self.km_eb < 4.8)):
            raise ValueError("Parámetros 44 kV fuera del rango")


_ACTIVO = ContextVar("escenario_44kv", default=Escenario44())


def configurar(escenario=None):
    _ACTIVO.set(escenario if escenario is not None else Escenario44())


def escenario_actual():
    return _ACTIVO.get()


def firma():
    return (tuple(asdict(escenario_actual()).items()),
            tuple(config.RED44_DOCUMENTO.items()), tuple(config.RED44_REGISTRO_2013.items()),
            config.RED44_REACTOR_PO_MVAR, config.R_AC_44KV,
            config.Z_EFF_44KV, config.V_NOMINAL_AC, config.V_NOMINAL_DC,
            config.CATENARIA_RHO_CU_20, config.CATENARIA_ALPHA_CU)


def _rx(segmentos, e):
    """R por fase corregida por temperatura; X referencial configurable."""
    rho = config.CATENARIA_RHO_CU_20 * (1 + config.CATENARIA_ALPHA_CU * (e.temperatura_c - 20))
    r = x = largo = 0.0
    for tipo, km, area in segmentos:
        cable = tipo.startswith("xlpe")
        factor = e.factor_r_xlpe if cable else e.factor_r_aereo
        r += rho * 1000 / area * km * factor
        x += (e.x_xlpe if cable else e.x_aereo) * km
        largo += km
    return largo, r, x


def ramas_documento(e):
    """Tuplas origen, destino, largo, R y X. Se conserva un total de 5 km
    hacia Villa Alemana. Reparto proporcional de XLPE entre sus dos ramas
    por falta de ubicación y longitud definitiva en el documento.
    """
    oeste1 = _rx(config.RED44_DOCUMENTO["SEAT-ES"], e)
    oeste2 = _rx(config.RED44_DOCUMENTO["ES-PO"], e)
    este = _rx((("aereo_1_0", 5-e.km_xlpe_va, 53.49),
                ("xlpe_120", e.km_xlpe_va, 120.0)), e)
    f = e.km_eb / 5
    return [
        ("SEAT", "SER ES", *oeste1),
        ("SER ES", "SER PO", *oeste2),
        ("SEAT", "SER EB", *(v*f for v in este)),
        ("SER EB", "SER VA", *(v*(1-f) for v in este)),
    ]


def calcular_documento(demands_kw, e=None):
    """Flujo aproximado balanceado, radial, cargas P y FP a tensión nominal.
    R y X totales por fase; ΔV≈√3 I(R cosφ+X senφ).
    No es un flujo AC iterativo ni un estudio de protecciones.
    """
    e = e or escenario_actual()
    v = float(config.V_NOMINAL_AC)
    vd = float(config.V_NOMINAL_DC)
    root3 = math.sqrt(3)
    cos = e.fp
    sin = math.sqrt(max(0, 1-cos*cos))
    nombres = ("SER ES", "SER PO", "SER EB", "SER VA")
    p = {n: max(0.0, float(demands_kw.get(n, 0))) for n in nombres}
    aguas_abajo = {"SER ES": p["SER ES"]+p["SER PO"], "SER PO": p["SER PO"],
                   "SER EB": p["SER EB"]+p["SER VA"], "SER VA": p["SER VA"]}
    tensiones = {"SEAT": v}
    resultado = {}
    detalles = []
    perdida = 0.0
    for origen, destino, km, r, x in ramas_documento(e):
        i = aguas_abajo[destino] * 1000 / (root3*v*cos)
        dv = root3*i*(r*cos+x*sin)
        loss = 3*i*i*r/1000
        tensiones[destino] = tensiones[origen] - dv
        resultado[destino] = {"Vac": tensiones[destino], "Vdc": vd*tensiones[destino]/v}
        detalles.append({"origen": origen, "destino": destino, "km": km,
                         "R_ohm": r, "X_ohm": x, "I_A": i,
                         "caida_V": dv, "perdida_kW": loss})
        perdida += loss
    resultado["P_loss_kw"] = perdida
    resultado["ramas_44kv"] = detalles
    resultado["modelo_44kv"] = "documento"
    return resultado


def ramas_registro(e):
    """R, X, B totales positivos por fase; R corregida desde 30 °C.
    El Belloto usa las constantes de VA, no una cuarta línea registrada.
    """
    factor = (1 + config.CATENARIA_ALPHA_CU*(e.temperatura_registro_c-20)) / (1 + config.CATENARIA_ALPHA_CU*10)
    def rama(a, b, key, largo):
        _, r, x, sus, imax, *_ = config.RED44_REGISTRO_2013[key]
        return (a, b, largo, r*largo*factor, x*largo, sus*largo*1e-6, imax)
    datos = config.RED44_REGISTRO_2013
    lv = datos["SEAT-VA"][0]
    # Para comparar aun si el escenario documental permite PK 4,95,
    # conservar la proporción respecto de sus 5 km; el modo registro
    # admite directamente una distancia menor que 4,8.
    eb = e.km_eb if e.km_eb < lv else lv*e.km_eb/5
    return [rama("SEAT","SER ES","SEAT-ES",datos["SEAT-ES"][0]),
            rama("SER ES","SER PO","ES-PO",datos["ES-PO"][0]),
            rama("SEAT","SER EB","SEAT-VA",eb),
            rama("SER EB","SER VA","SEAT-VA",lv-eb)]


def calcular_registro(demands_kw, e=None):
    """Flujo radial trifásico equilibrado, modelo pi de secuencia positiva.
    Cargas PQ con FP inductivo. Tensión SEAT fija; barrido atrás/adelante.
    R/X serie, B/2 capacitiva en cada extremo. Reactores opcionales como
    susceptancias inductivas nominales. No es un estudio de falla.
    Si no converge se marca explícitamente y se devuelven NaN en energía.
    """
    e = e or escenario_actual()
    ramas = ramas_registro(e)
    vll = float(config.V_NOMINAL_AC)
    vn = vll/math.sqrt(3)
    nodos = ["SEAT", "SER ES", "SER PO", "SER EB", "SER VA"]
    v = dict.fromkeys(nodos, complex(vn))
    y = dict.fromkeys(nodos, 0j)
    for a,b,km,r,x,sus,lim in ramas:
        y[a] += 1j*sus/2
        y[b] += 1j*sus/2
    if e.reactores_po_activos:
        for n in ("SER ES","SER PO"):
            y[n] -= 1j*config.RED44_REACTOR_PO_MVAR*1e6/vll**2
    tan = math.sqrt(max(0.,1-e.fp**2))/e.fp
    s = {n: complex(max(0.,float(demands_kw.get(n,0)))*1000/3)*(1+1j*tan)
         for n in nodos}
    s["SEAT"] = 0j
    def corrientes(volt):
        carga = {n: (s[n]/volt[n]).conjugate()+y[n]*volt[n] for n in nodos}
        series = {}
        for a,b,*_ in reversed(ramas):
            series[(a,b)] = carga[b]
            carga[a] += carga[b]
        return series,carga["SEAT"]
    convergio = False
    for _ in range(100):
        if min(abs(v[n]) for n in nodos) < .1*vn:
            break
        i,_ = corrientes(v)
        nuevo = {"SEAT":complex(vn)}
        for a,b,km,r,x,sus,lim in ramas:
            nuevo[b] = nuevo[a] - complex(r,x)*i[a,b]
        error = max(abs(nuevo[n]-v[n]) for n in nodos)
        v = {n: .5*v[n]+.5*nuevo[n] for n in nodos}
        if error < .001:
            convergio = True
            break
    if not convergio:
        nan = float("nan")
        return {**{n:{"Vac":nan,"Vdc":nan} for n in nodos[1:]},
                "P_loss_kw":nan,"ramas_44kv":[],"modelo_44kv":"registro2013",
                "convergio_44kv":False,"Q_seat_kvar":nan}
    i,iseat = corrientes(v)
    detalles = []
    for a,b,km,r,x,sus,lim in ramas:
        i_serie = i[a,b]
        i_env = i_serie+1j*sus/2*v[a]
        i_rec = -i_serie+1j*sus/2*v[b]
        imax = max(abs(i_serie),abs(i_env),abs(i_rec))
        detalles.append({"origen":a,"destino":b,"km":km,"R_ohm":r,"X_ohm":x,
                         "B_uS":sus*1e6,"I_A":abs(i_serie),"I_max_A":imax,
                         "I_nom_A":lim,"carga_pct":100*imax/lim,
                         "caida_V":math.sqrt(3)*(abs(v[a])-abs(v[b])),
                         "perdida_kW":3*abs(i_serie)**2*r/1000})
    seat = 3*v["SEAT"]*iseat.conjugate()/1000
    res = {n:{"Vac":math.sqrt(3)*abs(v[n]),"Vdc":config.V_NOMINAL_DC*abs(v[n])/vn}
           for n in nodos[1:]}
    res.update(P_loss_kw=sum(d["perdida_kW"] for d in detalles),
               ramas_44kv=detalles, modelo_44kv="registro2013", convergio_44kv=True,
               P_seat_kw=seat.real,Q_seat_kvar=seat.imag)
    return res


def selector_streamlit(prefix="red44"):
    import streamlit as st
    with st.expander("🔌 Modelo de red 44 kV", expanded=False):
        modos = {"historico": "Histórico V136", "documento": "Documento AT/MT — conductores por tramo",
                 "registro2013": "Registro técnico MERVAL 2013 — modelo π"}
        modo = st.selectbox("Escenario 44 kV", list(modos), format_func=modos.get, key=prefix+"_modo")
        des = modo == "historico"
        des_doc = modo != "documento"
        temp = st.number_input("Temperatura de conductores 44 kV (°C)", -30., 120., 20., 5.,
                               disabled=des_doc, key=prefix+"_temp")
        fp = st.number_input("Factor de potencia", .50, 1., .95, .01, disabled=des, key=prefix+"_fp")
        xa = st.number_input("Reactancia aérea supuesta (Ω/km)", 0., 2., float(config.RED44_X_AEREO_REF),
                             .01, disabled=des_doc, key=prefix+"_xa")
        xc = st.number_input("Reactancia XLPE supuesta (Ω/km)", 0., 2., float(config.RED44_X_XLPE_REF),
                             .01, disabled=des_doc, key=prefix+"_xc")
        ra = st.number_input("Factor R aérea respecto al Cu ideal", .1, 5., 1., .01,
                             disabled=des_doc, key=prefix+"_ra")
        rc = st.number_input("Factor R XLPE respecto al Cu ideal", .1, 5., 1., .01,
                             disabled=des_doc, key=prefix+"_rc")
        eb = st.number_input("Distancia El Sol–El Belloto supuesta (km)", .05, 4.75, 1.2, .1,
                             disabled=des, key=prefix+"_eb")
        lv = st.number_input("Longitud XLPE hacia Villa Alemana dentro de 5 km (km)", 0., 5., 0., .1,
                             disabled=des_doc, key=prefix+"_va")
        treg = st.number_input("Temperatura registro 2013 (°C)", -30.,120.,30.,5.,
                               disabled=modo!="registro2013",key=prefix+"_treg")
        react = st.checkbox("Simular ambos reactores ES–PO (0,333 MVAr c/u)",
                            value=False, disabled=modo!="registro2013",key=prefix+"_react")
        e = Escenario44(modo, temp, ra, rc, xa, xc, fp, eb, lv, treg, react)
        if modo == "registro2013":
            st.dataframe([{"Tramo":a+" → "+b,"km":round(k,3),"R Ω/fase":round(r,4),
                           "X Ω/fase":round(x,4),"B µS/fase":round(sus*1e6,3),"I nominal A":lim}
                          for a,b,k,r,x,sus,lim in ramas_registro(e)],hide_index=True)
            st.caption("Patron 884–886 (L068): El Belloto adopta R/X/B e I nominal de VA. "
                       "Se conserva longitud total oriental de 4,8 km. Reactores desconectados por defecto; "
                       "estado real no confirmado. Secuencia cero solo para referencia.")
        st.dataframe([{"Tramo": a+" → "+b, "Longitud km": round(k,3),
                       "R documento Ω/fase": round(r,4), "X supuesta Ω/fase": round(x,4)}
                      for a,b,k,r,x in ramas_documento(e)], hide_index=True)
        st.caption("El Sol–El Salto: 10,7 km Cu 3/0 + 0,5 km XLPE 150; "
                   "El Salto–Portales: 0,7 km Cu 1/0 + 6,8 km XLPE 120. "
                   "Total detallado 23,7 km (documento declara ~23,5 km).")
        st.warning("El Belloto, su posición eléctrica, la longitud XLPE al este, material de XLPE "
                   "y reactancias son hipótesis editables, no datos confirmados. "
                   "Se mantienen potencias y conexión CC de los transformadores de V136.")
    return e


def mostrar_comparacion(tabla):
    """Comparación AC sobre las mismas demandas por SER/franja, no re-simulación de tracción."""
    import streamlit as st
    h = "Pérdidas 44kV históricas kWh"
    d = "Pérdidas 44kV documento kWh"
    r = "Pérdidas 44kV registro 2013 kWh"
    if not all(k in tabla for k in (h,d,r)):
        return
    a,b,c = (float(tabla[k].sum()) for k in (h,d,r))
    x,y,z = st.columns(3)
    x.metric("Pérdidas históricas", f"{a:,.2f} kWh")
    y.metric("Pérdidas documento", f"{b:,.2f} kWh",
             f"{(b/a-1)*100:+.2f}%" if a>0 else None)
    if tabla[r].isna().any():
        z.metric("Registro 2013", "No convergió")
        st.warning("Hay franjas sin convergencia: no comparar totales del registro.")
    else:
        z.metric("Pérdidas registro 2013", f"{c:,.2f} kWh",
                 f"{(c/a-1)*100:+.2f}%" if a>0 else None)
    st.caption("Mismas demandas medias por SER/franja. No incluye SAF 12 kV "
               "ni valida conexión de transformadores.")
