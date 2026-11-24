"""Modelo compartido de resistencia DC para simulaciones exploratorias.

Escenario histórico: reproduce la tabla V135.
Escenario Siemens: conductores de cobre en paralelo ideal; feeder y retorno
opcionales introducidos por el usuario. No representa una certificación as-built.
El contexto es local a la ejecución (no modifica config global ni otras sesiones).
"""
from contextvars import ContextVar
from dataclasses import dataclass, asdict
import math
import config


@dataclass(frozen=True)
class EscenarioCatenaria:
    modo: str = "historico"
    temperatura_c: float = 20.0
    desgaste_pct: float = 0.0
    feeder_cu_eq_mm2: float = 0.0
    retorno_ohm_km: float = 0.0

    def __post_init__(self):
        if self.modo not in ("historico", "siemens_principal", "siemens_secundaria"):
            raise ValueError("Escenario de catenaria desconocido")
        vals = (self.temperatura_c, self.desgaste_pct,
                self.feeder_cu_eq_mm2, self.retorno_ohm_km)
        if not all(math.isfinite(float(x)) for x in vals):
            raise ValueError("Parámetros no finitos")
        if not (-30 <= self.temperatura_c <= 120 and
                0 <= self.desgaste_pct <= 50 and
                self.feeder_cu_eq_mm2 >= 0 and self.retorno_ohm_km >= 0):
            raise ValueError("Parámetros fuera del rango de simulación")


_ESC = ContextVar("escenario_catenaria", default=EscenarioCatenaria())
_SERS = ContextVar("sers_activas_catenaria", default=None)


def configurar(escenario=None, sers=None):
    """Llamar una vez por rerun, antes de ejecutar o consultar la caché."""
    _ESC.set(escenario if escenario is not None else EscenarioCatenaria())
    _SERS.set(None if sers is None else tuple((float(k), str(n)) for k, n in sers))


def escenario_actual():
    return _ESC.get()


def sers_activas():
    activas = _SERS.get()
    return list(config.SER_DATA if activas is None else activas)


def firma():
    return (tuple(asdict(escenario_actual()).items()), tuple(sers_activas()),
            tuple(config.CATENARIA_HISTORICA),
            tuple(sorted(config.CATENARIA_SIEMENS.items())),
            config.CATENARIA_RHO_CU_20, config.CATENARIA_ALPHA_CU)


def resistencia_historica(km):
    for limite, r in config.CATENARIA_HISTORICA:
        if km < limite:
            return r
    return config.CATENARIA_HISTORICA[-1][1]


def resistencia_siemens(escenario):
    """Devuelve (resistencia positiva, resistencia de lazo) en ohm/km.
    Desgaste = reducción porcentual del área de cada hilo de contacto.
    Feeder = suma de secciones equivalentes en cobre, idealmente conectadas.
    Retorno = resistencia equivalente adicional aportada por el usuario.
    """
    c = config.CATENARIA_SIEMENS
    n = c["hilos_secundaria"] if escenario.modo == "siemens_secundaria" else c["hilos_principal"]
    area = (n * c["hilo_mm2"] * (1 - escenario.desgaste_pct / 100)
            + c["mensajero_mm2"] + escenario.feeder_cu_eq_mm2)
    rho = config.CATENARIA_RHO_CU_20 * (1 + config.CATENARIA_ALPHA_CU * (escenario.temperatura_c - 20))
    positiva = rho * 1000 / area
    return positiva, positiva + escenario.retorno_ohm_km


def resistencia_km(km, escenario=None):
    esc = escenario or escenario_actual()
    return resistencia_historica(km) if esc.modo == "historico" else resistencia_siemens(esc)[1]


def resistencia_entre(a, b, escenario=None):
    esc = escenario or escenario_actual()
    lo, hi = sorted((float(a), float(b)))
    if esc.modo != "historico":
        return (hi - lo) * resistencia_km(lo, esc)
    total = 0.0
    for limite, r in config.CATENARIA_HISTORICA:
        extremo = min(hi, limite)
        if extremo > lo:
            total += (extremo - lo) * r
            lo = extremo
        if lo >= hi:
            break
    return total


def ramas_alimentacion(km, sers=None, escenario=None):
    """(nombre SER, fracción de corriente, resistencia rama).
    Dos SER contiguas equipotenciales; extremos alimentados radialmente.
    No incluye regulación de rectificador ni interacción entre trenes.
    """
    esc = escenario or escenario_actual()
    orden = sorted(sers_activas() if sers is None else sers)
    if not orden:
        raise ValueError("Se requiere al menos una SER activa")
    if km <= orden[0][0]:
        k, n = orden[0]
        return [(n, 1.0, resistencia_entre(k, km, esc))]
    if km >= orden[-1][0]:
        k, n = orden[-1]
        return [(n, 1.0, resistencia_entre(k, km, esc))]
    for (ka, na), (kb, nb) in zip(orden, orden[1:]):
        if ka <= km <= kb:
            ra, rb = resistencia_entre(ka, km, esc), resistencia_entre(km, kb, esc)
            total = ra + rb
            if total <= 0:
                return [(na, 1.0, 0.0)]
            return [(na, rb / total, ra), (nb, ra / total, rb)]
    raise ValueError("No se pudo determinar alimentación")


def resistencia_equivalente(km, sers=None, escenario=None):
    return sum(frac * frac * r for _, frac, r in ramas_alimentacion(km, sers, escenario))


def distribuir_energia_bilateral(energia, horas, km_ini, km_fin, sers, tension_v):
    """Aproximación cuasiestática: potencia uniforme por viaje, integración
    espacial en pasos <= 250 m. Pérdidas de cada rama I²R; no es flujo multi-tren.
    Energía negativa se conserva con signo y se reduce por pérdidas.
    """
    if not sers:
        return {}
    resultado = {n: 0.0 for _, n in sers}
    distancia = abs(km_fin - km_ini)
    pasos = max(1, math.ceil(distancia / 0.25))
    e = energia / pasos
    h = horas / pasos
    corriente = abs(energia) * 1000 / (horas * tension_v) if horas > 0 and tension_v > 0 else 0.0
    for i in range(pasos):
        km = km_ini + (km_fin - km_ini) * (i + 0.5) / pasos
        for nombre, frac, r in ramas_alimentacion(km, sers):
            perdida = (corriente * frac) ** 2 * r * h / 1000 if h > 0 else 0.0
            parte = e * frac
            resultado[nombre] += parte + perdida if parte >= 0 else min(0.0, parte + perdida)
    return resultado


def selector_streamlit(prefix="cat"):
    """Configuración y comparación de resistencias, con widgets por página."""
    import streamlit as st
    with st.expander("⚡ Modelo de catenaria CC", expanded=False):
        opciones = {"historico": "Histórico (resistencias por PK)",
                    "siemens_principal": "Siemens — vía principal (2 hilos)",
                    "siemens_secundaria": "Siemens — vía secundaria (1 hilo, escenario)"}
        modo = st.selectbox("Escenario eléctrico", list(opciones),
                            format_func=opciones.get, key=prefix+"_modo")
        t = st.number_input("Temperatura conductor (°C)", -30.0, 120.0, 20.0, 5.0,
                            key=prefix+"_temp", disabled=modo=="historico")
        d = st.number_input("Desgaste de sección del hilo (%)", 0.0, 50.0, 0.0, 1.0,
                            key=prefix+"_desg", disabled=modo=="historico")
        f = st.number_input("Feeder equivalente Cu total (mm²)", 0.0, 5000.0, 0.0, 25.0,
                            key=prefix+"_feed", disabled=modo=="historico")
        r = st.number_input("Retorno equivalente (Ω/km)", 0.0, 1.0, 0.0, 0.001,
                            format="%.4f", key=prefix+"_ret", disabled=modo=="historico")
        esc = EscenarioCatenaria(modo, float(t), float(d), float(f), float(r))
        comparativo = EscenarioCatenaria("siemens_secundaria" if modo=="siemens_secundaria" else "siemens_principal",
                                        float(t), float(d), float(f), float(r))
        rp, rl = resistencia_siemens(comparativo)
        st.metric("Siemens — conductores positivos", f"{rp:.5f} Ω/km")
        st.caption(f"Con retorno ingresado: {rl:.5f} Ω/km. "
                   "0 en feeder/retorno significa omitido, no inexistente.")
        st.dataframe([{"PK hasta": f"{k:g}" if math.isfinite(k) else "Fin",
                       "Histórico Ω/km": rh, "Siemens Ω/km": round(rl, 5),
                       "Diferencia %": round((rl / rh - 1) * 100, 2)}
                      for k, rh in config.CATENARIA_HISTORICA], hide_index=True)
        if modo != "historico":
            st.warning("Escenario exploratorio uniforme en la línea, no configuración as-built. "
                       "No usar para validar capacidad, protección o seguridad ferroviaria.")
    return esc
