import streamlit as st
import pandas as pd
import numpy as np
import time
from io import BytesIO
from datetime import datetime, date, timedelta

# Configuración de página de Streamlit
st.set_page_config(page_title="Simulador MERVAL V135", layout="wide", page_icon="🗺️")

# 🛡️ FALLBACKS DE SEGURIDAD
PAX_COLS_DEFAULT = ['PUE','BEL','FRA','BAR','POR','REC','MIR','VIN','HOS','CHO','SLT','VAL','QUI','SOL','BTO','AME','CON','VAM','SGA','PEN','LIM']
SER_DATA_DEFAULT = [(3.9, "SER PO"), (11.7, "SER ES"), (25.3, "SER EB"), (29.1, "SER VA")]

try:
    import config
except ImportError:
    pass

import etl_parser

# =============================================================================
# IMPORTACIONES BLINDADAS (mantenidas del original)
# =============================================================================
_funcs_etl = {
    'procesar_thdr': None,
    'calcular_dwell': None,
    'cargar_pax': None,
    'match_pax': None,
    'calc_tren_km_real_general': None,
    'clean_id': None,
    'mins_to_time_str': None,
    'clasificar_dia': None,
    'cargar_prevenciones': None,
    'get_vacios_dia': None,
    'parsear_planilla_maestra': None,
}

for _fn in _funcs_etl:
    try:
        _funcs_etl[_fn] = getattr(etl_parser, _fn)
    except AttributeError:
        pass

procesar_thdr = _funcs_etl['procesar_thdr']
calcular_dwell = _funcs_etl['calcular_dwell']
cargar_pax = _funcs_etl['cargar_pax']
match_pax = _funcs_etl['match_pax']
calc_tren_km_real_general = _funcs_etl['calc_tren_km_real_general']
clean_id = _funcs_etl['clean_id']
mins_to_time_str = _funcs_etl['mins_to_time_str']
clasificar_dia = _funcs_etl['clasificar_dia']
cargar_prevenciones = _funcs_etl['cargar_prevenciones']
get_vacios_dia = _funcs_etl['get_vacios_dia']
parsear_planilla_maestra = _funcs_etl['parsear_planilla_maestra']

# Sincronización de nombres de funciones
if not hasattr(etl_parser, 'get_pax_at_km') and hasattr(etl_parser, 'get_pax_at_km_nativo'):
    etl_parser.get_pax_at_km = etl_parser.get_pax_at_km_nativo
if not hasattr(etl_parser, 'get_pax_at_km_nativo') and hasattr(etl_parser, 'get_pax_at_km'):
    etl_parser.get_pax_at_km_nativo = etl_parser.get_pax_at_km

from motor_fisico import (
    calcular_termodinamica_flota_v111, simular_tramo_termodinamico
)

try:
    from motor_fisico import calcular_receptividad_por_headway, precalcular_red_electrica_v111
except ImportError:
    try:
        from red_electrica import calcular_receptividad_por_headway, precalcular_red_electrica_v111
    except ImportError:
        pass

from red_electrica import (
    calcular_flujo_ac_nodo, distribuir_energia_sers, distribuir_potencia_sers_kw
)
from ui_dashboards import render_gemelo_digital, render_dashboard_energia_v112

# =============================================================================
# 1. FUNCIONES DE CARGA Y AGRUPACIÓN (las necesarias para el planificador)
# =============================================================================

def leer(files): 
    res = []
    for f in (files or []):
        try: 
            f.seek(0)
        except Exception: 
            pass
        res.append((f.name, f.getvalue()))
    return res

@st.cache_data(show_spinner="Cargando Pasajeros...")
def build_pax_v71(blobs_v1, blobs_v2):
    parts, err = [], []
    for blobs, via_default in [(blobs_v1, 1), (blobs_v2, 2)]:
        for nm, data in blobs:
            try: parts.append(cargar_pax(data, nm, via_default))
            except Exception as e: err.append(f"[{nm}]: {e}")
    if len(parts) > 0: return pd.concat(parts, ignore_index=True), err
    return pd.DataFrame(), err

@st.cache_data(show_spinner="Cargando Prevenciones (TSR)...")
def procesar_prevenciones_independiente(_bp, sig_ligera):
    prev_list = []
    for nm, data in _bp: 
        try: prev_list.extend(cargar_prevenciones(data, nm))
        except: pass
    return prev_list

# =============================================================================
# FUNCIONES AUXILIARES DEL PLANIFICADOR (conservadas)
# =============================================================================
def generar_trayectoria_sintetica(tipo_tren, doble, via, pct_trac, t_ini_mins, estacion_anio, km_orig, km_dest, use_rm, prevenciones=None):
    from config import N_EST, ESTACIONES, KM_ACUM, DWELL_DEF
    from motor_fisico import simular_tramo_termodinamico

    km_min = min(km_orig, km_dest)
    km_max = max(km_orig, km_dest)
    est_indices = [i for i, km in enumerate(KM_ACUM[:N_EST]) if km_min - 0.01 <= km <= km_max + 0.01]
    if via == 2:
        est_indices = list(reversed(est_indices))
    if len(est_indices) < 2:
        return [(t_ini_mins, km_orig), (t_ini_mins + 70, km_dest)]
    trayectoria = []
    t_actual = t_ini_mins
    trayectoria.append((t_actual, KM_ACUM[est_indices[0]]))
    for j in range(len(est_indices) - 1):
        idx_ini = est_indices[j]
        idx_fin = est_indices[j+1]
        km_ini_seg = KM_ACUM[idx_ini]
        km_fin_seg = KM_ACUM[idx_fin]
        es_destino = (j == len(est_indices) - 2)
        try:
            _, _, _, _, _, t_h, _ = simular_tramo_termodinamico(
                tipo_tren, doble, km_ini_seg, km_fin_seg, via, pct_trac,
                use_rm, True, None, {}, 150, None, None, estacion_anio, t_actual, False, prevenciones
            )
        except Exception:
            t_h = 0.0
        t_llegada = t_actual + t_h * 60
        trayectoria.append((t_llegada, km_fin_seg))
        if not es_destino:
            t_salida = t_llegada + DWELL_DEF / 60
            trayectoria.append((t_salida, km_fin_seg))
            t_actual = t_salida
        else:
            t_actual = t_llegada
    if abs(trayectoria[-1][1] - km_dest) > 0.001:
        trayectoria.append((t_actual, km_dest))
    return trayectoria

@st.cache_data(show_spinner="Integrando física y demanda en Planificador...")
def procesar_planificador_reactivo(_df_sint, _df_px_filtered, estacion_anio_plan, pct_trac_plan,
                                  use_rm, use_pend, use_regen, tipo_regen, pax_promedio_viaje,
                                  _prevenciones, plan_sig):
    viajes_completos = []
    perfiles_por_via = {}
    perfiles_por_servicio = {}
    try: pax_cols_list = getattr(config, 'PAX_COLS', PAX_COLS_DEFAULT)
    except: pax_cols_list = PAX_COLS_DEFAULT
    try: flota_dict = getattr(config, 'FLOTA', {})
    except: flota_dict = {}

    if not _df_px_filtered.empty:
        for via in [1, 2]:
            sub = _df_px_filtered[_df_px_filtered['Via'] == via]
            if not sub.empty:
                pd_dict = {c: int(round(sub[c].mean())) for c in pax_cols_list if c in sub.columns}
                if 'CargaMax' in sub.columns:
                    pd_dict['CargaMax_Promedio'] = int(round(sub['CargaMax'].mean()))
                perfiles_por_via[via] = pd_dict
        if 'Tren_Clean' in _df_px_filtered.columns:
            for tren, group in _df_px_filtered.groupby('Tren_Clean'):
                if str(tren).strip() == '': continue
                pd_dict = {c: int(round(group[c].mean())) for c in pax_cols_list if c in group.columns}
                if 'CargaMax' in group.columns:
                    pd_dict['CargaMax_Promedio'] = int(round(group['CargaMax'].mean()))
                perfiles_por_servicio[str(tren)] = pd_dict

    for idx, r in _df_sint.iterrows():
        via_tren = r['Via']
        t_ini_tren = r['t_ini']
        num_srv = str(r.get('num_servicio', '')).strip()
        pax_arr_viaje = {c: 0 for c in pax_cols_list}
        pax_calculado = 0
        f_tipo = flota_dict.get(r['tipo_tren'], {})
        cap_m = f_tipo.get('cap_max', 398) * (2 if r['doble'] else 1)

        if perfiles_por_servicio and num_srv in perfiles_por_servicio:
            perfil_srv = perfiles_por_servicio[num_srv]
            pax_calculado = perfil_srv.get('CargaMax_Promedio', 0)
            pax_arr_viaje = {k: v for k, v in perfil_srv.items() if k != 'CargaMax_Promedio'}
        elif not _df_px_filtered.empty:
            sub_v = _df_px_filtered[_df_px_filtered['Via'] == via_tren].copy()
            if not sub_v.empty and 't_ini_p' in sub_v.columns:
                sub_v['diff'] = sub_v['t_ini_p'].apply(lambda x: min(abs(float(x) - float(t_ini_tren)), 1440 - abs(float(x) - float(t_ini_tren))))
                idx_min = sub_v['diff'].idxmin()
                if sub_v.loc[idx_min, 'diff'] <= 20:
                    best_t = sub_v.loc[idx_min, 't_ini_p']
                    best_group = sub_v[sub_v['t_ini_p'] == best_t]
                    if 'CargaMax' in best_group.columns:
                        pax_calculado = int(round(best_group['CargaMax'].mean()))
                    pax_arr_viaje = {c: int(round(best_group[c].mean())) for c in pax_cols_list if c in best_group.columns}
                else:
                    pax_dict_dinamico = perfiles_por_via.get(via_tren, {})
                    pax_abordo_base = pax_dict_dinamico.get('CargaMax_Promedio', pax_promedio_viaje)
                    f_gauss = 0.2 + 0.8 * np.exp(-0.5 * ((t_ini_tren - 450)/60)**2) + 0.8 * np.exp(-0.5 * ((t_ini_tren - 1080)/90)**2)
                    pax_calculado = int(pax_abordo_base * f_gauss * 1.5)
                    if pax_dict_dinamico:
                        pax_arr_viaje = {k: int(v * f_gauss * 1.5) for k, v in pax_dict_dinamico.items() if k != 'CargaMax_Promedio'}
                    else:
                        pax_arr_viaje = {c: int(pax_calculado / len(pax_cols_list)) for c in pax_cols_list}
            else:
                f_gauss = 0.2 + 0.8 * np.exp(-0.5 * ((t_ini_tren - 450)/60)**2) + 0.8 * np.exp(-0.5 * ((t_ini_tren - 1080)/90)**2)
                pax_calculado = int(pax_promedio_viaje * f_gauss * 1.5)
                pax_arr_viaje = {c: int(pax_calculado / len(pax_cols_list)) for c in pax_cols_list}
        else:
            f_gauss = 0.2 + 0.8 * np.exp(-0.5 * ((t_ini_tren - 450)/60)**2) + 0.8 * np.exp(-0.5 * ((t_ini_tren - 1080)/90)**2)
            pax_calculado = int(pax_promedio_viaje * f_gauss * 1.5)
            pax_arr_viaje = {c: int(pax_calculado / len(pax_cols_list)) for c in pax_cols_list}

        pax_calculado = min(pax_calculado, cap_m)
        pax_arr_viaje = {k: min(v, cap_m) for k, v in pax_arr_viaje.items()}

        try:
            trc_v, aux_v, reg_v, _, _, t_h, _ = simular_tramo_termodinamico(
                r['tipo_tren'], r['doble'], r['km_orig'], r['km_dest'], r['Via'],
                pct_trac_plan, use_rm, use_pend, r.get('nodos'), pax_arr_viaje, pax_calculado,
                None, r.get('maniobra'), estacion_anio_plan, r['t_ini'], es_vacio=False, prevenciones=_prevenciones
            )
        except TypeError:
            trc_v, aux_v, reg_v, _, _, t_h, _ = simular_tramo_termodinamico(
                r['tipo_tren'], r['doble'], r['km_orig'], r['km_dest'], r['Via'],
                pct_trac_plan, use_rm, use_pend, r.get('nodos'), pax_arr_viaje, pax_calculado,
                None, r.get('maniobra'), estacion_anio_plan, r['t_ini'], es_vacio=False
            )

        viaje_final = r.to_dict()
        viaje_final['pax_d'] = pax_arr_viaje
        viaje_final['pax_abordo'] = pax_calculado
        viaje_final['t_fin'] = r['t_ini'] + (t_h * 60.0)
        t_fin_original = viaje_final['t_fin']
        trayectoria = generar_trayectoria_sintetica(
            r['tipo_tren'], r['doble'], r['Via'], pct_trac_plan, r['t_ini'],
            estacion_anio_plan, r['km_orig'], r['km_dest'], use_rm, _prevenciones
        )
        if trayectoria:
            t_fin_sintetico = trayectoria[-1][0]
            t_fin_final = max(t_fin_original, t_fin_sintetico)
            trayectoria[-1] = (t_fin_final, r['km_dest'])
            viaje_final['nodos'] = trayectoria
            viaje_final['t_fin'] = t_fin_final
        else:
            viaje_final['nodos'] = [(r['t_ini'], r['km_orig']), (r['t_ini'] + t_h, r['km_dest'])]
        viajes_completos.append(viaje_final)

    df_sint_final = pd.DataFrame(viajes_completos)
    if 'tren_km' not in df_sint_final.columns:
        df_sint_final['tren_km'] = df_sint_final.apply(calc_tren_km_real_general, axis=1)
    df_sint_final.index = df_sint_final['_id']

    if use_regen:
        if "Probabilístico" in tipo_regen:
            dict_regen_sint = calcular_receptividad_por_headway(df_sint_final)
        else:
            dict_regen_sint = precalcular_red_electrica_v111(df_sint_final, pct_trac_plan, use_rm, estacion_anio_plan)
    else:
        dict_regen_sint = {}
    try:
        df_sint_e = calcular_termodinamica_flota_v111(df_sint_final, pct_trac_plan, use_pend, use_rm, use_regen,
                                                      dict_regen_sint, estacion_anio_plan, prevenciones=_prevenciones)
    except TypeError:
        df_sint_e = calcular_termodinamica_flota_v111(df_sint_final, pct_trac_plan, use_pend, use_rm, use_regen,
                                                      dict_regen_sint, estacion_anio_plan)
    if 'prevencion_aplicada' in df_sint_e.columns:
        df_sint_e = df_sint_e.drop(columns=['prevencion_aplicada'])
    return df_sint_final, df_sint_e

# =============================================================================
# TABLA THDR SINTÉTICA (conservada)
# =============================================================================
@st.cache_data(show_spinner=False, ttl=1)
def generar_fila_thdr_sintetica(tipo_tren, doble, via, pct_trac, t_ini_mins, estacion_anio, num_servicio, km_orig, km_dest, use_rm, prevenciones=None):
    from config import N_EST, ESTACIONES, KM_ACUM, DWELL_DEF
    from motor_fisico import simular_tramo_termodinamico
    from etl_parser import mins_to_time_str

    km_min = min(km_orig, km_dest)
    km_max = max(km_orig, km_dest)
    est_en_recorrido = [i for i, km in enumerate(KM_ACUM[:N_EST]) if km_min - 0.01 <= km <= km_max + 0.01]
    if via == 2:
        est_en_recorrido = list(reversed(est_en_recorrido))
    if len(est_en_recorrido) < 2:
        return {'Servicio': str(num_servicio), 'Error': 'Sin estaciones en recorrido'}
    fila = {'Servicio': str(num_servicio), 'Tipo': tipo_tren, 'Config': 'Doble' if doble else 'Simple'}
    t_actual = t_ini_mins
    t_inicio_viaje = t_ini_mins
    est_orig_nombre = ESTACIONES[est_en_recorrido[0]]
    fila[f"{est_orig_nombre}\nSalida"] = mins_to_time_str(t_actual)
    for j in range(len(est_en_recorrido)-1):
        idx_ini = est_en_recorrido[j]
        idx_fin = est_en_recorrido[j+1]
        km_ini_tr = KM_ACUM[idx_ini]
        km_fin_tr = KM_ACUM[idx_fin]
        es_destino = (j == len(est_en_recorrido)-2)
        try:
            _,_,_,_,_,t_h, _ = simular_tramo_termodinamico(
                tipo_tren, doble, km_ini_tr, km_fin_tr, via, pct_trac,
                use_rm, True, None, {}, 150, None, None, estacion_anio, t_actual, False, prevenciones
            )
        except Exception:
            t_h = 0.0
        t_llegada = t_actual + t_h * 60
        t_salida  = t_llegada + DWELL_DEF / 60
        est_sig   = ESTACIONES[idx_fin]
        fila[f"{est_sig}\nLlegada"] = mins_to_time_str(t_llegada)
        if not es_destino:
            fila[f"{est_sig}\nSalida"] = mins_to_time_str(t_salida)
        t_actual = t_llegada if es_destino else t_salida
    t_total = t_actual - t_inicio_viaje
    h = int(t_total // 60); m = int(t_total % 60); s = int((t_total % 1) * 60)
    fila['Tiempo Viaje'] = f"{h:02d}:{m:02d}:{s:02d}"
    return fila

def render_tablas_thdr_planificador(df_sint_final, pct_trac, estacion_anio, use_rm, prevenciones=None):
    from config import N_EST, ESTACIONES, KM_TOTAL
    from etl_parser import mins_to_time_str

    st.markdown("---")
    st.markdown("#### 📋 Horario Simulado por Estación (estilo THDR)")
    for via, label in [(1, "🔵 Vía 1 — Puerto → Limache"), (2, "🔴 Vía 2 — Limache → Puerto")]:
        df_via = df_sint_final[df_sint_final['Via'] == via].sort_values('t_ini')
        if df_via.empty:
            continue
        with st.expander(label, expanded=False):
            filas = []
            for _, row in df_via.iterrows():
                fila = generar_fila_thdr_sintetica(
                    str(row.get('tipo_tren', 'XT-100')), bool(row.get('doble', False)), via, float(pct_trac),
                    float(row.get('t_ini', 360.0)), str(estacion_anio), str(row.get('num_servicio', '')),
                    float(row.get('km_orig', 0.0)), float(row.get('km_dest', 43.13)), use_rm, prevenciones
                )
                filas.append(fila)
            if filas:
                df_tabla = pd.DataFrame(filas)
                st.caption(f"{len(df_tabla)} servicios | {N_EST} estaciones | {KM_TOTAL:.1f} km")
                st.dataframe(df_tabla, use_container_width=True, hide_index=True,
                             height=min(400, 40 + len(df_tabla) * 35))

# =============================================================================
# INTERFAZ PRINCIPAL (RECONSTRUIDA)
# =============================================================================
def main():
    # ── Barra lateral ──
    with st.sidebar:
        st.header("📂 Archivos")
        # Planilla Maestra (ahora aquí, donde antes iban los THDR)
        archivo_planilla = st.file_uploader("Planilla Maestra (.csv, .xlsx)", type=['csv', 'xlsx', 'xls'])
        f_px1 = st.file_uploader("Pasajeros Vía 1", type=['csv', 'xlsx', 'xls'])
        f_px2 = st.file_uploader("Pasajeros Vía 2", type=['csv', 'xlsx', 'xls'])
        f_prev = st.file_uploader("🚧 Prevenciones de Vía (.csv, .xlsx)", type=['csv', 'xlsx', 'xls'])

        st.subheader("⚙️ Parámetros Físicos de Red")
        use_rm      = st.checkbox("🚦 Velocidades RM (Riel Mojado)", value=False)
        use_pend    = st.toggle("⛰️ Pendientes Físicas", value=True)
        use_regen   = st.toggle("⚡ Activar Regeneración", value=True)
        tipo_regen  = st.radio("Modelo de Regeneración", ["Físico (Load Flow)", "Probabilístico (Headway)"])
        mes_sel     = st.selectbox("Mes de operación",
                                   ["Enero","Febrero","Marzo","Abril","Mayo","Junio",
                                    "Julio","Agosto","Septiembre","Octubre","Noviembre","Diciembre"], index=3)
        estacion_anio = {"Enero":"verano","Febrero":"verano","Marzo":"otoño","Abril":"otoño","Mayo":"otoño",
                         "Junio":"invierno","Julio":"invierno","Agosto":"invierno","Septiembre":"primavera",
                         "Octubre":"primavera","Noviembre":"primavera","Diciembre":"verano"}[mes_sel]
        pct_trac_plan = st.slider("% Tracción Máxima", 30, 100, 90, 5)
        tipo_dia_plan = st.selectbox("Tipo de Día para Demanda", ["Laboral", "Sábado", "Domingo/Festivo"])
        pax_promedio_viaje = {"Laboral": 280, "Sábado": 160, "Domingo/Festivo": 110}[tipo_dia_plan]

        try:
            ser_data_safe = getattr(config, 'SER_DATA', SER_DATA_DEFAULT)
        except:
            ser_data_safe = SER_DATA_DEFAULT
        active_ser_names = st.multiselect("Subestaciones Activas", [s[1] for s in ser_data_safe],
                                          default=[s[1] for s in ser_data_safe])
        active_sers = [s for s in ser_data_safe if s[1] in active_ser_names]
        if not active_sers:
            active_sers = [ser_data_safe[0]]
        gap_vias = st.slider("Separación Visual Vías (px)", 120, 350, 200, 10)

    # ── Procesar archivos auxiliares ──
    df_px = pd.DataFrame()
    if f_px1:
        try: df_px = pd.concat([df_px, cargar_pax(f_px1.getvalue(), f_px1.name, via_param=1)], ignore_index=True)
        except: pass
    if f_px2:
        try: df_px = pd.concat([df_px, cargar_pax(f_px2.getvalue(), f_px2.name, via_param=2)], ignore_index=True)
        except: pass

    prevenciones_list = []
    if f_prev:
        try: prevenciones_list = cargar_prevenciones(f_prev.getvalue(), f_prev.name)
        except: pass

    # ── Inicializar variables de sesión para el dashboard (IMPORTANTE) ──
    for key in ["sl_ui_plan", "t_math_plan", "play_plan"]:
        if key not in st.session_state:
            st.session_state[key] = 480.0 if "sl_ui" in key or "t_math" in key else False

    st.title("🔮 Planificador de Escenarios")

    modo_plan = st.radio("Fuente de Datos", ["Planilla Maestra (Subir CSV/Excel)", "Matriz Sintética", "Laboratorio (Tramo Único)"], horizontal=True)

    df_sint = None
    if modo_plan == "Matriz Sintética":
        if 'df_plan' not in st.session_state:
            st.session_state['df_plan'] = pd.DataFrame([{"Origen": "Puerto", "Destino": "Limache", "Flota": "XT-100", "Configuración": "Doble", "Cantidad": 40}])
        df_plan_edit = st.data_editor(st.session_state['df_plan'], num_rows="dynamic", use_container_width=True)
        if st.button("🚀 Ejecutar Gemelo Digital del Planificador", use_container_width=True, type="primary"):
            df_sintetico_list = []
            try:
                est_safe = getattr(config, 'ESTACIONES', ['Puerto', 'Limache'])
                km_acum_safe = getattr(config, 'KM_ACUM', [0.0, 43.13])
                ec_safe = getattr(config, 'EC', ['PU', 'LI'])
            except:
                est_safe = ['Puerto', 'Limache']
                km_acum_safe = [0.0, 43.13]
                ec_safe = ['PU', 'LI']
            for idx, row in df_plan_edit.iterrows():
                if row['Cantidad'] <= 0 or row['Origen'] == row['Destino']: continue
                try:
                    i_o = est_safe.index(row['Origen'])
                    i_d = est_safe.index(row['Destino'])
                    via = 1 if i_o < i_d else 2
                    nodos = [(0.0, km_acum_safe[i]) for i in (range(i_o, i_d+1) if via==1 else range(i_o, i_d-1, -1))]
                    k_o, k_d = km_acum_safe[i_o], km_acum_safe[i_d]
                    svc_t = f"{ec_safe[i_o]}-{ec_safe[i_d]}"
                    interval = (1350 - 360) / row['Cantidad']
                    for i in range(int(row['Cantidad'])):
                        df_sintetico_list.append({
                            '_id': f"SINT_{idx}_{i}", 't_ini': 360 + i * interval, 'Via': via,
                            'km_orig': k_o, 'km_dest': k_d, 'nodos': nodos,
                            'tipo_tren': row['Flota'], 'doble': row['Configuración'] == "Doble",
                            'num_servicio': f"VIRT_{idx}_{i}", 'maniobra': None, 'svc_type': svc_t
                        })
                except: pass
            df_sint = pd.DataFrame(df_sintetico_list)

    elif modo_plan == "Planilla Maestra (Subir CSV/Excel)":
        if archivo_planilla:
            try:
                df_temp, msg = parsear_planilla_maestra(archivo_planilla.getvalue(), archivo_planilla.name)
                if df_temp.empty:
                    st.error(f"Error: {msg}")
                else:
                    st.success("✅ Planilla decodificada. Distribuye la flota por trayecto:")
                    rutas_unicas = list(df_temp['svc_type'].value_counts().keys())
                    if 'flota_map_v2' not in st.session_state or set(st.session_state['flota_map_v2']['Ruta']) != set(rutas_unicas):
                        st.session_state['flota_map_v2'] = pd.DataFrame([{"Ruta": r, "Total Viajes": df_temp['svc_type'].value_counts()[r], "XT-100": df_temp['svc_type'].value_counts()[r], "XT-M": 0, "SFE": 0} for r in rutas_unicas])
                    df_flota_edit = st.data_editor(st.session_state['flota_map_v2'], hide_index=True, use_container_width=True)
                    if st.button("🚀 Ejecutar Gemelo Digital del Planificador", use_container_width=True, type="primary"):
                        df_sint = df_temp.copy().sort_values('t_ini')
                        asignaciones = {}
                        for _, r in st.session_state['flota_map_v2'].iterrows():
                            asignaciones[r['Ruta']] = (['XT-100']*int(r.get('XT-100',0)) +
                                                       ['XT-M']*int(r.get('XT-M',0)) +
                                                       ['SFE']*int(r.get('SFE',0)))
                        def asignar_tren(ruta):
                            if ruta in asignaciones and asignaciones[ruta]:
                                return asignaciones[ruta].pop(0)
                            return 'XT-100'
                        df_sint['tipo_tren'] = df_sint['svc_type'].apply(asignar_tren)
            except Exception as err:
                st.error(f"Fallo de lectura: {err}")

    elif modo_plan == "Laboratorio (Tramo Único)":
        try:
            est_safe = getattr(config, 'ESTACIONES', ['Puerto', 'Limache'])
        except:
            est_safe = ['Puerto', 'Limache']
        c1, c2, c3, c4 = st.columns(4)
        with c1: sb_orig = st.selectbox("Origen", est_safe, index=0)
        with c2: sb_dest = st.selectbox("Destino", est_safe, index=len(est_safe)-1)
        with c3: sb_flota = st.selectbox("Flota", ["XT-100", "XT-M", "SFE"])
        with c4: sb_pax = st.number_input("Pasajeros", 0, 1000, 150)
        if st.button("⚡ Simular Tramo", use_container_width=True):
            if sb_orig != sb_dest:
                try: km_acum_safe = getattr(config, 'KM_ACUM', [0.0, 43.13])
                except: km_acum_safe = [0.0, 43.13]
                idx_o = est_safe.index(sb_orig)
                idx_d = est_safe.index(sb_dest)
                km_o, km_d = km_acum_safe[idx_o], km_acum_safe[idx_d]
                via = 1 if idx_o < idx_d else 2
                nodos = [(0.0, km_acum_safe[i]) for i in (range(idx_o, idx_d+1) if via==1 else range(idx_o, idx_d-1, -1))]
                with st.spinner("Calculando..."):
                    try:
                        trc, aux, reg, _, neto, t_h, _ = simular_tramo_termodinamico(
                            sb_flota, False, km_o, km_d, via, pct_trac_plan, use_rm, use_pend,
                            nodos, {}, sb_pax, None, None, estacion_anio, 480.0, False, prevenciones_list
                        )
                    except:
                        trc, aux, reg, _, neto, t_h, _ = simular_tramo_termodinamico(
                            sb_flota, False, km_o, km_d, via, pct_trac_plan, use_rm, use_pend,
                            nodos, {}, sb_pax, None, None, estacion_anio, 480.0, False
                        )
                try:
                    distrib = distribuir_energia_sers(neto, t_h, km_o, km_d, active_sers)
                    eta_ser = getattr(config, 'ETA_SER_RECTIFICADOR', 0.96)
                    tot_ser = sum(max(0.0, v) for v in distrib.values()) / eta_ser
                    loss = calcular_flujo_ac_nodo({k: max(0.0, v)/eta_ser/max(0.001, t_h) for k,v in distrib.items()})['P_loss_kw'] * (1.15**2) * max(0.001, t_h)
                    seat = (tot_ser + loss) / 0.99
                    ide = seat / max(0.001, abs(km_d - km_o))
                    st.success(f"Simulación: {sb_orig} ➔ {sb_dest}")
                    c1, c2, c3 = st.columns(3)
                    c1.metric("⏱️ Tiempo", f"{t_h*60:.1f} min")
                    c2.metric("⚡ Energía Neta (SEAT)", f"{seat:.1f} kWh")
                    c3.metric("💡 IDE", f"{ide:.3f} kWh/km")
                except Exception as e:
                    st.error(f"Error en red: {e}")

    # ── Visualización de resultados ──
    if df_sint is not None and not df_sint.empty:
        df_px_filtered = pd.DataFrame()
        nombre_perfil = f"Estático ({pax_promedio_viaje} pax)"
        if not df_px.empty:
            fechas_disp = sorted([str(x) for x in df_px['Fecha_s'].dropna().unique()])
            if fechas_disp:
                df_px_filtered = df_px[df_px['Fecha_s'].astype(str).str.strip().isin(fechas_disp)]
                if not df_px_filtered.empty:
                    nombre_perfil = f"Promedio Real ({len(fechas_disp)} días)"

        plan_sig = str(df_sint) + str(pax_promedio_viaje)
        df_sint_final, df_sint_e = procesar_planificador_reactivo(
            df_sint, df_px_filtered, estacion_anio, pct_trac_plan,
            use_rm, use_pend, use_regen, tipo_regen, pax_promedio_viaje, prevenciones_list, plan_sig
        )

        # Forzar tipos numéricos
        cols_num = ['t_ini', 't_fin', 'kwh_viaje_trac', 'kwh_viaje_aux', 'kwh_viaje_regen',
                    'kwh_reostato', 'kwh_viaje_neto', 't_viaje_h', 'tren_km']
        for col in cols_num:
            if col in df_sint_e.columns:
                df_sint_e[col] = pd.to_numeric(df_sint_e[col], errors='coerce')
            if col in df_sint_final.columns:
                df_sint_final[col] = pd.to_numeric(df_sint_final[col], errors='coerce')

        st.divider()
        try:
            render_gemelo_digital(
                df_sint_final, df_sint_e, active_sers,
                f"Simulación: {nombre_perfil}",
                pct_trac_plan, use_rm, use_pend, estacion_anio, "plan", gap_vias,
                pax_dia_total=int(df_sint_final['pax_abordo'].sum())
            )
            render_dashboard_energia_v112(df_sint_e, active_sers, "Planificador", 480.0)
            render_tablas_thdr_planificador(df_sint_final, pct_trac_plan, estacion_anio, use_rm, prevenciones_list)
        except Exception as e:
            st.error(f"Fallo al graficar UI: {e}")

if __name__ == "__main__":
    main()
