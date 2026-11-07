"""Simulación de un día real a partir de THDR alojadas en Datos-Operacionales.

Página adicional para el repositorio Simulador. No sustituye app.py.
Usa los motores de etl_parser y motor_fisico ya existentes. Fuente GitHub
pública de THDR: antoniorra1983-ops/Datos-Operacionales, rama main.

Importante: el consumo es SIMULADO en catenaria del tren, no una medición SEAT.
"""
from __future__ import annotations

from datetime import date
import re
import hashlib
from pathlib import PurePosixPath

import numpy as np
import pandas as pd
import requests
import streamlit as st

# Reutilizar exactamente el control de acceso que usa el simulador existente.
# Importar app no ejecuta main(), pues tiene guardia __main__.
from app import verificar_acceso
from etl_parser import procesar_thdr, cargar_pax, match_pax, clasificar_dia
from motor_fisico import calcular_termodinamica_flota_v111
try:
    from motor_fisico import calcular_receptividad_por_headway
except ImportError:
    calcular_receptividad_por_headway = None

verificar_acceso()

REPO = "antoniorra1983-ops/Datos-Operacionales"
BRANCH = "main"
API = f"https://api.github.com/repos/{REPO}"
HEADERS = {"Accept": "application/vnd.github+json", "User-Agent": "EFE-Simulador-THDR/1.0"}
EXTS = (".xlsx", ".xls", ".xlsm", ".csv")
MAX_FILE_BYTES = 25 * 1024 * 1024
MESES = {1: "verano", 2: "verano", 3: "otoño", 4: "otoño", 5: "otoño",
         6: "invierno", 7: "invierno", 8: "invierno", 9: "primavera",
         10: "primavera", 11: "primavera", 12: "verano"}


def _fecha_nombre(nombre: str) -> str | None:
    """Fecha si se puede identificar sin abrir Excel. Nunca inventar fecha."""
    stem = PurePosixPath(nombre).stem
    # 2026-11-07 / 20261107 (o 07-11-2026 / 07112026)
    for patron, orden in [(r"(?<!\d)(20\d{2})[-_. ]?(\d{2})[-_. ]?(\d{2})(?!\d)", "ymd"),
                          (r"(?<!\d)(\d{2})[-_. ]?(\d{2})[-_. ]?(20\d{2})(?!\d)", "dmy"),
                          (r"(?<!\d)(\d{2})[-_. ]?(\d{2})[-_. ]?(\d{2})(?!\d)", "dmy2")]:
        for m in re.finditer(patron, stem):
            a, b, c = map(int, m.groups())
            try:
                if orden == "ymd": return date(a, b, c).isoformat()
                if orden == "dmy": return date(c, b, a).isoformat()
                return date(2000 + c, b, a).isoformat()
            except ValueError:
                continue
    return None


def _get_json(url: str):
    r = requests.get(url, headers=HEADERS, timeout=35)
    if r.status_code in (403, 429):
        raise RuntimeError("GitHub limitó las consultas. Prueba de nuevo después o configura un token de lectura.")
    if r.status_code != 200:
        raise RuntimeError(f"GitHub respondió HTTP {r.status_code} al consultar el catálogo de THDR.")
    return r.json()


@st.cache_data(show_spinner=False, ttl=900)
def _catalogo(tipo: str) -> list[dict]:
    """Consulta exclusivamente la carpeta THDR de la vía indicada."""
    if tipo not in ("thdr_v1", "thdr_v2"):
        raise ValueError("Carpeta THDR no reconocida")
    ruta = f"data/{tipo}"
    from urllib.parse import quote
    datos = _get_json(f"{API}/contents/{quote(ruta, safe='/')}?ref={BRANCH}")
    if not isinstance(datos, list):
        raise RuntimeError(f"No se pudo listar {ruta} de GitHub")
    archivos = []
    for item in datos:
        if item.get("type") != "file":
            continue
        nombre = str(item.get("name", ""))
        if not nombre.lower().endswith(EXTS):
            continue
        archivos.append({"nombre": nombre,
                         "ruta": str(item.get("path", "")),
                         "fecha": _fecha_nombre(nombre),
                         "tam": int(item.get("size", 0)),
                         "sha": str(item.get("sha", "")),
                         "url": item.get("download_url") or ""})
    return sorted(archivos, key=lambda x: (x["fecha"] or "", x["nombre"]), reverse=True)


@st.cache_data(show_spinner=False, ttl=3600, max_entries=100)
def _obtener(ruta: str, url: str, tam: int, sha: str) -> bytes:
    if tam > MAX_FILE_BYTES:
        raise ValueError(f"Archivo demasiado grande ({tam / 1048576:.1f} MB): {ruta}")
    if not url.startswith(f"https://raw.githubusercontent.com/{REPO}/"):
        # GitHub define el enlace a descarga; construimos el URL seguro cuando falta.
        from urllib.parse import quote
        url = f"https://raw.githubusercontent.com/{REPO}/{BRANCH}/{quote(ruta, safe='/')}"
    r = requests.get(url, headers=HEADERS, timeout=90)
    r.raise_for_status()
    if len(r.content) > MAX_FILE_BYTES:
        raise ValueError(f"Archivo fuera del límite de descarga: {ruta}")
    return r.content


def _preparar_dia(archivos: list[dict], fecha: str) -> tuple[pd.DataFrame, list[str], dict]:
    piezas, errores, cont = [], [], {1: 0, 2: 0}
    for item in archivos:
        via = int(item["via"])
        try:
            bruto = _obtener(item["ruta"], item["url"], item["tam"], item.get("sha", ""))
            tabla, msg = procesar_thdr(bruto, item["nombre"], via)
            if tabla is None or tabla.empty:
                errores.append(f"{item['nombre']}: {msg}")
                continue
            if "Fecha_str" not in tabla.columns:
                errores.append(f"{item['nombre']}: no se detectó la fecha THDR")
                continue
            tabla = tabla.loc[tabla["Fecha_str"].astype(str) == fecha].copy()
            if tabla.empty:
                errores.append(f"{item['nombre']}: la fecha interna no corresponde a {fecha}")
                continue
            cont[via] += len(tabla)
            piezas.append(tabla)
        except Exception as exc:
            errores.append(f"{item['nombre']}: {type(exc).__name__}: {str(exc)[:160]}")
    if not piezas:
        return pd.DataFrame(), errores, cont
    df = pd.concat(piezas, ignore_index=True)
    if "_id" in df.columns:
        df = df.drop_duplicates(subset=["_id"])
    for columna in ("t_ini", "t_fin", "km_orig", "km_dest"):
        if columna not in df:
            df[columna] = np.nan
        df[columna] = pd.to_numeric(df[columna], errors="coerce")
    df = df.loc[df["km_orig"].notna() & df["km_dest"].notna() &
                df["t_ini"].notna() & df["km_orig"].ne(df["km_dest"])].copy()
    return df.reset_index(drop=True), errores, cont


def _agregar_pasajeros(df: pd.DataFrame, archivos_pax: list, pax_estimados: int) -> tuple[pd.DataFrame, int, list[str]]:
    """Asigna PAX reales cuando existe Carga; si no, ocupación declaradamente estimada."""
    df = df.copy()
    errores = []
    tablas = []
    for via, archivos in archivos_pax:
        for uf in archivos or []:
            try:
                t = cargar_pax(uf.getvalue(), uf.name, via)
                if isinstance(t, pd.DataFrame) and not t.empty:
                    tablas.append(t)
            except Exception as exc:
                errores.append(f"Carga pasajeros {uf.name}: {str(exc)[:130]}")
    df["pax_abordo"] = int(pax_estimados)
    df["pax_d"] = [{} for _ in range(len(df))]
    if not tablas:
        return df, 0, errores
    pax = pd.concat(tablas, ignore_index=True)
    if "Tren_Clean" not in pax.columns and "Tren" in pax.columns:
        from etl_parser import clean_id
        pax["Tren_Clean"] = pax["Tren"].apply(clean_id)
    encontrados = 0
    for idx, row in df.iterrows():
        try:
            resultado = match_pax(row, pax)
            if resultado and isinstance(resultado, (list, tuple)) and len(resultado) >= 2:
                perfil, ocup = resultado[0], resultado[1]
                if isinstance(perfil, dict) and perfil:
                    df.at[idx, "pax_d"] = perfil
                    df.at[idx, "pax_abordo"] = max(0, int(ocup))
                    encontrados += 1
        except Exception:
            continue
    return df, encontrados, errores


@st.cache_data(show_spinner=False, max_entries=12)
def _simular(_df: pd.DataFrame, fecha: str, pct_traccion: int,
             usar_pendiente: bool, usar_riel_mojado: bool, usar_regen: bool,
             usar_headway: bool, estacion: str, firma: tuple) -> pd.DataFrame:
    receptividad = {}
    if usar_regen and usar_headway and calcular_receptividad_por_headway:
        receptividad = calcular_receptividad_por_headway(_df)
    return calcular_termodinamica_flota_v111(
        _df, pct_traccion, usar_pendiente, usar_riel_mojado, usar_regen,
        receptividad, estacion_anio=estacion, prevenciones=[]
    )


def _formato(num, dec=1):
    if not np.isfinite(float(num)):
        return "—"
    s = f"{float(num):,.{dec}f}"
    return s.replace(",", "_").replace(".", ",").replace("_", ".")


st.title("⚡ Simular consumo de un día desde las THDR")
st.caption("Fuente: GitHub Datos-Operacionales · THDR Vía 1 y Vía 2 · motor físico de Simulador")
st.info("El resultado es una **estimación física**, no el consumo eléctrico medido por SEAT/PRMTE. "
        "La precisión depende de la motriz, configuración, horarios, pasajeros y restricciones disponibles.")

try:
    with st.spinner("Buscando archivos THDR de las dos vías en GitHub..."):
        cat1, cat2 = _catalogo("thdr_v1"), _catalogo("thdr_v2")
except Exception as exc:
    st.error(f"No fue posible consultar las THDR: {exc}")
    st.stop()

st.caption(f"Archivos encontrados: Vía 1: {len(cat1)} · Vía 2: {len(cat2)}")
fechas = sorted({x["fecha"] for x in cat1 + cat2 if x["fecha"]})
if fechas:
    fecha = st.selectbox("📅 Día para simular", fechas[::-1], index=0,
                         help="Se selecciona por la fecha que se identifica en el nombre de las THDR.")
else:
    fecha = st.date_input("📅 Fecha del día", value=date.today()).isoformat()
    st.warning("Los nombres de los archivos no incluyen fechas reconocibles. Selecciona manualmente las THDR.")

# Selección manual permite archivos cuyos nombres no expresan claramente la fecha
catalogo = [{**x, "via": 1} for x in cat1] + [{**x, "via": 2} for x in cat2]
coinciden = [x for x in catalogo if x["fecha"] == fecha]
seleccion = {x["ruta"] for x in coinciden}
if not coinciden:
    st.warning("No se identificaron archivos para esa fecha. Abre la selección manual y elige las THDR.")
with st.expander("📁 Seleccionar o corregir archivos THDR de ese día", expanded=not bool(coinciden)):
    opciones = {f"Vía {x['via']} · {x['nombre']}": x for x in catalogo}
    predet = [k for k, x in opciones.items() if x["ruta"] in seleccion]
    escogidos = st.multiselect("Archivos a procesar", list(opciones), default=predet,
                                help="Usa una THDR Vía 1 y otra Vía 2 correspondientes al mismo día.")
    escogidos_archivos = [opciones[k] for k in escogidos]

if not escogidos_archivos:
    st.info("Selecciona al menos un archivo THDR para continuar.")
    st.stop()

with st.expander("⚙️ Configuración física y ocupación", expanded=True):
    c1, c2, c3 = st.columns(3)
    pct_traccion = c1.slider("Tracción máxima (%)", 30, 100,
                             value=50 if clasificar_dia(fecha) != "Laboral" else 75, step=5)
    pax_estimados = c2.number_input("Pasajeros por viaje si no hay carga", 0, 1000,
                                    value=150, step=10,
                                    help="Supuesto uniforme, no es un dato real.")
    riel_mojado = c3.checkbox("Riel mojado", value=False)
    pend = st.checkbox("Considerar pendientes de la vía", value=True)
    regen = st.checkbox("Considerar frenado regenerativo", value=True)
    headway = st.checkbox("Receptividad por intervalo real entre trenes", value=True,
                          disabled=not regen)
    st.caption("Las velocidades y características de XT-100, XT-M y SFE provienen de config.py del Simulador.")
    px1 = st.file_uploader("Opcional: Carga pasajeros Vía 1", type=["xls", "xlsx", "xlsm", "csv"],
                           accept_multiple_files=True)
    px2 = st.file_uploader("Opcional: Carga pasajeros Vía 2", type=["xls", "xlsx", "xlsm", "csv"],
                           accept_multiple_files=True)

if st.button("▶️ Simular consumo del día", type="primary", use_container_width=True):
    with st.spinner("Descargando THDR y ejecutando el motor físico..."):
        df_dia, errores, por_via = _preparar_dia(escogidos_archivos, fecha)
        if df_dia.empty:
            st.error("No hay viajes válidos en las THDR del día elegido. Revisa fecha y archivos.")
            if errores:
                st.write("Diagnóstico:", *errores[:15])
            st.stop()
        df_dia, pax_reales, err_pax = _agregar_pasajeros(
            df_dia, [(1, px1), (2, px2)], int(pax_estimados))
        errores += err_pax
        firma = tuple((x["ruta"], x["sha"]) for x in escogidos_archivos) + (
            ("pax_estimados", int(pax_estimados)),
            ("pax1", tuple(hashlib.sha256(uf.getvalue()).hexdigest() for uf in (px1 or []))),
            ("pax2", tuple(hashlib.sha256(uf.getvalue()).hexdigest() for uf in (px2 or []))),
        )
        try:
            resultado = _simular(df_dia, fecha, int(pct_traccion), pend, riel_mojado, regen,
                                 bool(headway), MESES[date.fromisoformat(fecha).month], firma)
        except Exception as exc:
            st.error(f"Falló el motor físico: {type(exc).__name__}: {exc}")
            st.stop()
    st.session_state["thdr_sim_resultado"] = resultado
    st.session_state["thdr_sim_fecha"] = fecha
    st.session_state["thdr_sim_errores"] = errores
    st.session_state["thdr_sim_pax_reales"] = pax_reales
    st.session_state["thdr_sim_fuentes"] = [(x["via"], x["nombre"]) for x in escogidos_archivos]

resultado = st.session_state.get("thdr_sim_resultado", pd.DataFrame())
if isinstance(resultado, pd.DataFrame) and not resultado.empty:
    fecha_result = st.session_state.get("thdr_sim_fecha", "")
    if fecha_result != fecha:
        st.warning("Los resultados que aparecen corresponden a otra fecha. Presiona Simular para actualizarlos.")
    else:
        st.subheader(f"Resultados simulados · {fecha_result}")
        columnas = ["kwh_viaje_trac", "kwh_viaje_aux", "kwh_viaje_regen", "kwh_viaje_neto"]
        for col in columnas:
            if col not in resultado:
                st.error(f"El motor no devolvió el campo esperado: {col}")
                st.stop()
        suma = resultado[columnas].apply(pd.to_numeric, errors="coerce").sum()
        a, b, c, d = st.columns(4)
        a.metric("Tracción bruta", f"{_formato(suma['kwh_viaje_trac'], 0)} kWh")
        b.metric("Auxiliares del tren", f"{_formato(suma['kwh_viaje_aux'], 0)} kWh")
        c.metric("Regeneración aprovechada", f"{_formato(suma['kwh_viaje_regen'], 0)} kWh")
        d.metric("Consumo neto estimado", f"{_formato(suma['kwh_viaje_neto'], 0)} kWh")
        pax_reales = st.session_state.get("thdr_sim_pax_reales", 0)
        st.caption(f"{len(resultado)} viajes simulados · {pax_reales} con perfil de pasajeros real "
                   f"· {len(resultado)-pax_reales} con ocupación supuesta")
        if resultado["Via"].nunique() < 2:
            st.warning("Solo hay datos de una vía. Este resultado NO representa el consumo de todo el día del sistema.")
        st.markdown("#### Consumo por vía")
        cols = {"kwh_viaje_trac": "Tracción (kWh)",
                "kwh_viaje_aux": "Auxiliares (kWh)",
                "kwh_viaje_regen": "Regeneración (kWh)",
                "kwh_viaje_neto": "Neto (kWh)"}
        por_via = resultado.groupby("Via", as_index=False)[columnas].sum().rename(columns=cols)
        st.dataframe(por_via.style.format({c: "{:,.1f}" for c in cols.values()}),
                     use_container_width=True, hide_index=True)
        st.bar_chart(por_via.set_index("Via")["Neto (kWh)"], x_label="Vía", y_label="kWh netos")
        st.markdown("#### Consumo estimado por viaje")
        lista = ["Fecha_str", "Via", "num_servicio", "motriz_num", "tipo_tren", "doble",
                 "t_ini", "km_orig", "km_dest", "pax_abordo", "tren_km"] + columnas
        detalle = resultado[[c for c in lista if c in resultado]].copy()
        st.dataframe(detalle, use_container_width=True, hide_index=True)
        st.download_button("⬇️ Descargar consumos por viaje (CSV)",
                           data=detalle.to_csv(index=False, sep=";", decimal=",").encode("utf-8-sig"),
                           file_name=f"simulacion_THDR_{fecha_result}.csv", mime="text/csv")
        with st.expander("Fuentes y observaciones"):
            st.write(st.session_state.get("thdr_sim_fuentes", []))
            st.write("Este modelo estima consumos de los trenes según las THDR, no incorpora "
                     "automáticamente toda la energía de instalaciones ni reproduce exactamente "
                     "las pérdidas de la red de subestaciones.")
            for e in st.session_state.get("thdr_sim_errores", []):
                st.warning(e)
