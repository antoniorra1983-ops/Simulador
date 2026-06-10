import streamlit as st
import pandas as pd
import numpy as np
import time
from io import BytesIO
from datetime import datetime, date, timedelta

# Configuración de página de Streamlit (DEBE ser la primera instrucción)
st.set_page_config(page_title="Simulador MERVAL V135", layout="wide", page_icon="🗺️")

# 🛡️ FALLBACKS DE SEGURIDAD PARA CLOUD
PAX_COLS_DEFAULT = ['PUE','BEL','FRA','BAR','POR','REC','MIR','VIN','HOS','CHO','SLT','VAL','QUI','SOL','BTO','AME','CON','VAM','SGA','PEN','LIM']
SER_DATA_DEFAULT = [(3.9, "SER PO"), (11.7, "SER ES"), (25.3, "SER EB"), (29.1, "SER VA")]

try:
    import config
except ImportError:
    pass

import etl_parser

# =============================================================================
# IMPORTACIONES BLINDADAS - Tolerancia a fallos de caché en Streamlit Cloud
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

_missing_etl = [k for k, v in _funcs_etl.items() if v is None]
if _missing_etl:
    st.warning(f"⚠️ Funciones faltantes en etl_parser.py: {', '.join(_missing_etl)}")

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

# Sincronización de nombres de funciones por seguridad
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
try:
    from perfiles_viaje import render_perfiles_viaje, figura_perfiles
    _PERFILES_IMPORT_ERROR = None
except Exception as _e_perf:
    render_perfiles_viaje = None
    figura_perfiles = None
    _PERFILES_IMPORT_ERROR = repr(_e_perf)  # se mostrará en la UI si falta el módulo

def get_config_hash():
    """Hash de los parámetros físicos del config — si cambia, invalida la caché."""
    import hashlib, json
    try:
        flota = getattr(config, 'FLOTA', {})
        eta   = getattr(config, 'ETA_REGEN_NETA', 0.38)
        snap  = {t: {k: v for k, v in f.items() if isinstance(v, (int, float, str, bool))}
                 for t, f in flota.items()}
        snap['_ETA_REGEN_NETA'] = eta
        return hashlib.md5(json.dumps(snap, sort_keys=True).encode()).hexdigest()[:8]
    except Exception:
        return "no_config"

try:
    from optimizador_flota import optimizar_asignacion_flota, generar_planillas_xlsx, generar_tabla_seat_15min
except ImportError:
    optimizar_asignacion_flota = None
    generar_planillas_xlsx = None
    generar_tabla_seat_15min = None

# =============================================================================
# 1. FUNCIONES DE CARGA Y AGRUPACIÓN (BLINDADAS)
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

def leer_github(url):
    try:
        import urllib.request
        url = url.strip()
        if 'github.com' in url and 'raw.githubusercontent' not in url:
            url = url.replace('github.com','raw.githubusercontent.com').replace('/blob/','/')
        nm = url.split('/')[-1]
        with urllib.request.urlopen(url, timeout=15) as r:
            return nm, r.read()
    except Exception as e: 
        return None, str(e)

@st.cache_data(show_spinner="Procesando THDR...", ttl=1)
def build_thdr_v71(blobs_v1, blobs_v2):
    all_parts, err = [], []
    for blobs, via_default in [(blobs_v1, 1), (blobs_v2, 2)]:
        for nm, data in blobs:
            df, msg = procesar_thdr(data, nm, via_default)
            if not df.empty: all_parts.append(df)
            else: err.append(f"[{nm}]: {msg}")
    
    if len(all_parts) > 0:
        for idx_df in range(len(all_parts)):
            cols = pd.Series(all_parts[idx_df].columns)
            for dup in cols[cols.duplicated()].unique():
                cols[cols==dup] = [f"{dup}_{i}" if i else dup for i in range(sum(cols==dup))]
            all_parts[idx_df].columns = cols

        df_master = pd.concat(all_parts, ignore_index=True)
        df1 = df_master[df_master['Via'] == 1].copy()
        df2 = df_master[df_master['Via'] == 2].copy()
        if not df1.empty and not df2.empty:
            df1, df2 = calcular_dwell(df1, df2)
        return df1, df2, err
    return pd.DataFrame(), pd.DataFrame(), err

@st.cache_data(show_spinner="Cargando Pasajeros...")
def build_pax_v71(blobs_v1, blobs_v2):
    parts, err = [], []
    for blobs, via_default in [(blobs_v1, 1), (blobs_v2, 2)]:
        for nm, data in blobs:
            try: parts.append(cargar_pax(data, nm, via_default))
            except Exception as e: err.append(f"[{nm}]: {e}")
    if len(parts) > 0: return pd.concat(parts, ignore_index=True), err
    return pd.DataFrame(), err

@st.cache_data(show_spinner="Consolidando viajes y cruzando datos...", ttl=1)
def procesar_datos_completos(_b1, _b2, _bx1, _bx2, sig_pesada):
    df1, df2, err_t = build_thdr_v71(_b1, _b2)
    df_px, err_p = build_pax_v71(_bx1, _bx2)
    
    dfs_to_concat = [d for d in [df1, df2] if not d.empty]
    df_all = pd.concat(dfs_to_concat, ignore_index=True).drop_duplicates(subset=['_id']) if dfs_to_concat else pd.DataFrame()

    if not df_all.empty:
        if not df_px.empty:
            if 'Tren_Clean' not in df_px.columns: 
                df_px['Tren_Clean'] = df_px['Tren'].apply(clean_id) if 'Tren' in df_px.columns else ''
            
            pax_res = df_all.apply(lambda r: match_pax(r, df_px), axis=1)
            df_all['pax_d'] = [x[0] for x in pax_res]
            df_all['pax_abordo'] = [x[1] for x in pax_res]
            df_all['hora_origen_pax'] = [x[2] for x in pax_res]
            df_all['nro_thdr_pax'] = [x[3] for x in pax_res]
            df_all['pax_row_idx'] = [x[4] for x in pax_res]
        else:
            df_all['pax_d'] = [{} for _ in range(len(df_all))]
            df_all['pax_abordo'] = 0
            df_all['hora_origen_pax'] = '--:--:--'
            df_all['nro_thdr_pax'] = 'No Detectado'
            df_all['pax_row_idx'] = -1
            
        df_all['maniobra'] = None
        if 'tren_km' not in df_all.columns:
            df_all['tren_km'] = df_all.apply(calc_tren_km_real_general, axis=1)
    return df_all, df_px, err_t, err_p

@st.cache_data(show_spinner="Cargando Prevenciones (TSR)...")
def procesar_prevenciones_independiente(_bp, sig_ligera):
    prev_list = []
    for nm, data in _bp: 
        try: prev_list.extend(cargar_prevenciones(data, nm))
        except: pass
    return prev_list

@st.cache_data(show_spinner="Simulando termodinámica histórica...")
def simular_dia_historico_cached(_df_dia, pct_trac_hist, use_rm, use_pend, use_regen, tipo_regen, estacion_anio, _prevenciones, data_sig_fisica, config_sig=""):
    dict_regen = {}
    if use_regen:
        # Modelo Probabilístico: receptividad según headway real entre trenes
        # Valores calibrados para MERVAL: 0.24–0.90, promedio 0.535
        # El modelo Físico (Load Flow) requiere perfil de velocidad segundo a segundo
        # que actualmente no se exporta del motor → pendiente de implementación
        try:
            dict_regen = calcular_receptividad_por_headway(_df_dia)
        except Exception:
            dict_regen = {}
    return calcular_termodinamica_flota_v111(_df_dia, pct_trac_hist, use_pend, use_rm, use_regen, dict_regen, estacion_anio, prevenciones=_prevenciones)

# =============================================================================
# FUNCIÓN PARA GENERAR TRAYECTORIA DETALLADA POR ESTACIONES
# =============================================================================
def generar_trayectoria_sintetica(tipo_tren, doble, via, pct_trac, t_ini_mins, estacion_anio, km_orig, km_dest, use_rm, use_pend=True, prevenciones=None):
    from config import N_EST, ESTACIONES, KM_ACUM, DWELL_DEF
    from motor_fisico import simular_tramo_termodinamico

    km_min = min(km_orig, km_dest)
    km_max = max(km_orig, km_dest)

    est_indices = [i for i, km in enumerate(KM_ACUM[:N_EST]) if km_min - 0.01 <= km <= km_max + 0.01]
    if via == 2:
        est_indices = list(reversed(est_indices))

    if len(est_indices) < 2:
        # Fallback: estimar tiempo desde velocidad media operativa MERVAL (~42 km/h)
        dist_km = abs(km_dest - km_orig)
        t_estimado = (dist_km / 42.0) * 60.0  # minutos
        return [(t_ini_mins, km_orig), (t_ini_mins + t_estimado, km_dest)]

    trayectoria = []
    t_actual = t_ini_mins

    # Nodo de salida en la primera estación
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
                use_rm, use_pend, None, {}, 150, None, None, estacion_anio, t_actual, False, prevenciones
            )
        except Exception:
            t_h = 0.0

        t_llegada = t_actual + t_h * 60
        # Añadir nodo de llegada (velocidad 0)
        trayectoria.append((t_llegada, km_fin_seg))

        if not es_destino:
            t_salida = t_llegada + DWELL_DEF / 60
            trayectoria.append((t_salida, km_fin_seg))
            t_actual = t_salida
        else:
            t_actual = t_llegada

    # Garantizar que el último nodo sea exactamente el destino (con tolerancia)
    TOL = 0.001
    if abs(trayectoria[-1][1] - km_dest) > TOL:
        trayectoria.append((t_actual, km_dest))

    return trayectoria

@st.cache_data(show_spinner="⚙️ Simulando física de la flota (motor + anti-alcance). En día laboral puede tardar ~2 min…")
def procesar_planificador_reactivo(_df_sint, _df_px_filtered, estacion_anio_plan, pct_trac_plan, use_rm, use_pend, use_regen, tipo_regen, pax_promedio_viaje, _prevenciones, plan_sig, config_sig="", man_sig="", con_anti_alcance=True):
    viajes_completos = []
    perfiles_por_servicio = {}
    perfiles_por_via = {}
    
    try: pax_cols_list = getattr(config, 'PAX_COLS', PAX_COLS_DEFAULT)
    except: pax_cols_list = PAX_COLS_DEFAULT
        
    try: flota_dict = getattr(config, 'FLOTA', {})
    except: flota_dict = {}
    
    if not _df_px_filtered.empty:
        for via in [1, 2]:
            sub_via = _df_px_filtered[_df_px_filtered['Via'] == via]
            if not sub_via.empty:
                pd_dict = {c: int(round(sub_via[c].mean())) for c in pax_cols_list if c in sub_via.columns}
                if 'CargaMax' in sub_via.columns:
                    pd_dict['CargaMax_Promedio'] = int(round(sub_via['CargaMax'].mean()))
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

        # Guardar nodos originales (t=0) para cálculo de energía correcto
        nodos_energia = r.get('nodos')  # nodos t=0 → motor aplica DWELL correctamente

        # ✅ Generar trayectoria detallada con timestamps reales para el mapa
        trayectoria = generar_trayectoria_sintetica(
            r['tipo_tren'], r['doble'], r['Via'], pct_trac_plan, r['t_ini'],
            estacion_anio_plan, r['km_orig'], r['km_dest'], use_rm, use_pend, _prevenciones
        )
        if trayectoria:
            t_fin_sintetico = trayectoria[-1][0]
            trayectoria[-1] = (t_fin_sintetico, r['km_dest'])
            viaje_final['nodos'] = nodos_energia  # energía usa nodos t=0
            viaje_final['t_arr'] = trayectoria    # mapa usa trayectoria con timestamps
            viaje_final['t_fin'] = t_fin_sintetico
        else:
            viaje_final['nodos'] = nodos_energia
            viaje_final['t_arr'] = [(r['t_ini'], r['km_orig']), (r['t_ini'] + t_h * 60, r['km_dest'])]

        viajes_completos.append(viaje_final)
        
    df_sint_final = pd.DataFrame(viajes_completos)
    if 'tren_km' not in df_sint_final.columns:
        df_sint_final['tren_km'] = df_sint_final.apply(calc_tren_km_real_general, axis=1)
    df_sint_final.index = df_sint_final['_id']
    
    if use_regen:
        if "Probabilístico" in tipo_regen:
            dict_regen_sint = calcular_receptividad_por_headway(df_sint_final)
        else:
            # Modo físico: requiere datos_sim → primera pasada para generarlos
            try:
                df_sint_e_pass1 = calcular_termodinamica_flota_v111(
                    df_sint_final, pct_trac_plan, use_pend, use_rm, False, {}, estacion_anio_plan, prevenciones=_prevenciones)
            except TypeError:
                df_sint_e_pass1 = calcular_termodinamica_flota_v111(
                    df_sint_final, pct_trac_plan, use_pend, use_rm, False, {}, estacion_anio_plan)
            # Segunda pasada: calcular receptividad física con datos_sim reales
            dict_regen_sint = precalcular_red_electrica_v111(df_sint_e_pass1, pct_trac_plan, use_rm, estacion_anio_plan)
    else:
        dict_regen_sint = {}
        
    try:
        df_sint_e = calcular_termodinamica_flota_v111(df_sint_final, pct_trac_plan, use_pend, use_rm, use_regen, dict_regen_sint, estacion_anio_plan, prevenciones=_prevenciones, aplicar_anden=True, aplicar_anti_alcance=con_anti_alcance)
    except TypeError:
        df_sint_e = calcular_termodinamica_flota_v111(df_sint_final, pct_trac_plan, use_pend, use_rm, use_regen, dict_regen_sint, estacion_anio_plan)
        
    if 'prevencion_aplicada' in df_sint_e.columns:
        df_sint_e = df_sint_e.drop(columns=['prevencion_aplicada'])
        
    return df_sint_final, df_sint_e

# =============================================================================
# TABLA THDR SINTÉTICA — Horario simulado por estación para el Planificador
# =============================================================================
def generar_fila_thdr_sintetica(tipo_tren, doble, via, pct_trac, t_ini_mins, estacion_anio, num_servicio, km_orig, km_dest, use_rm, prevenciones=None, motriz_num=''):
    from config import N_EST, ESTACIONES, KM_ACUM, DWELL_DEF
    from motor_fisico import simular_tramo_termodinamico
    from etl_parser import mins_to_time_str

    km_min = min(km_orig, km_dest)
    km_max = max(km_orig, km_dest)

    est_en_recorrido = [i for i, km in enumerate(KM_ACUM[:N_EST])
                        if km_min - 0.01 <= km <= km_max + 0.01]

    if via == 2:
        est_en_recorrido = list(reversed(est_en_recorrido))

    if len(est_en_recorrido) < 2:
        return {'Servicio': str(num_servicio), 'Error': 'Sin estaciones en recorrido'}

    fila = {'Servicio': str(num_servicio), 'Tipo': tipo_tren, 'Config': 'Doble' if doble else 'Simple', 'Tren': str(motriz_num) if motriz_num else str(num_servicio)}
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
        except Exception as _e_thdr:
            # Fallback: estimar desde velocidad media del tramo
            dist_tr = abs(km_fin_tr - km_ini_tr)
            t_h = (dist_tr / 42.0)  # horas, ~42 km/h velocidad media real

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

    for via, label in [(1, "🔵 Vía 1 — Puerto → Limache"),
                       (2, "🔴 Vía 2 — Limache → Puerto")]:
        df_via = df_sint_final[df_sint_final['Via'] == via].sort_values('t_ini')
        if df_via.empty:
            continue

        with st.expander(label, expanded=False):
            filas = []
            for _, row in df_via.iterrows():
                fila = generar_fila_thdr_sintetica(
                    str(row.get('tipo_tren', 'XT-100')),
                    bool(row.get('doble', False)),
                    via,
                    float(pct_trac),
                    float(row.get('t_ini', 360.0)),
                    str(estacion_anio),
                    str(row.get('num_servicio', '')),
                    float(row.get('km_orig', 0.0)),
                    float(row.get('km_dest', 43.13)),
                    use_rm,
                    prevenciones,
                    str(row.get('motriz_num', ''))
                )
                filas.append(fila)

            if filas:
                df_tabla = pd.DataFrame(filas)
                col_cap, col_btn = st.columns([4, 1])
                with col_cap:
                    st.caption(f"{len(df_tabla)} servicios | {N_EST} estaciones | {KM_TOTAL:.1f} km")
                with col_btn:
                    import io as _io
                    _buf = _io.BytesIO()
                    with pd.ExcelWriter(_buf, engine='openpyxl') as _wr:
                        df_tabla.to_excel(_wr, index=False, sheet_name=f'Via{via}')
                    st.download_button(
                        label='⬇ xlsx',
                        data=_buf.getvalue(),
                        file_name=f'horario_simulado_via{via}.xlsx',
                        mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                        key=f'dl_thdr_v{via}',
                        use_container_width=True,
                    )
                st.dataframe(
                    df_tabla,
                    use_container_width=True,
                    hide_index=True,
                    height=min(400, 40 + len(df_tabla) * 35),
                    column_config={
                        col: st.column_config.TextColumn(col.replace("\n", " "), width="small")
                        for col in df_tabla.columns
                        if col not in ['Servicio','Tipo','Config','Tiempo Viaje']
                    } | {
                        'Servicio':     st.column_config.TextColumn('Servicio', width='small'),
                        'Tipo':         st.column_config.TextColumn('Tipo',     width='small'),
                        'Config':       st.column_config.TextColumn('Config',   width='small'),
                        'Tiempo Viaje': st.column_config.TextColumn('T. Viaje', width='small'),
                    }
                )


def verificar_acceso():
    """Control de acceso por login OIDC (Google/Microsoft).
    Solo los correos en la lista CORREOS_AUTORIZADOS pueden entrar.
    Si no hay autenticación configurada en secrets.toml, deja pasar (modo desarrollo)."""

    # === Lista de usuarios autorizados (correos de Google/Microsoft) ===
    CORREOS_AUTORIZADOS = [
        "antonio@efe.cl",            # ← reemplaza por los correos reales
        "usuario2@efe.cl",
        "usuario3@efe.cl",
    ]

    # Si la autenticación no está configurada (ej. corriendo local sin secrets),
    # no bloquear — permite desarrollo. En producción (Streamlit Cloud con secrets),
    # st.user.is_logged_in funciona normalmente.
    try:
        logged_in = st.user.is_logged_in
    except Exception:
        # autenticación no configurada: modo abierto (desarrollo)
        return True

    if not logged_in:
        st.title("🔐 Simulador MERVAL — Acceso restringido")
        st.write("Inicia sesión con tu cuenta autorizada para continuar.")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Iniciar sesión con Google", use_container_width=True):
                st.login("google")
        with col2:
            if st.button("Iniciar sesión con Microsoft", use_container_width=True):
                st.login("microsoft")
        st.stop()

    # ya autenticado: verificar que el correo esté autorizado
    correo = (st.user.get("email") or "").lower()
    if correo not in [c.lower() for c in CORREOS_AUTORIZADOS]:
        st.error(f"⛔ La cuenta {correo} no está autorizada para usar este simulador.")
        st.write("Si crees que es un error, contacta al administrador.")
        if st.button("Cerrar sesión"):
            st.logout()
        st.stop()

    # autorizado: mostrar quién está conectado y opción de salir en el sidebar
    with st.sidebar:
        st.success(f"✓ {st.user.get('name', correo)}")
        if st.button("Cerrar sesión", use_container_width=True):
            st.logout()

    return True


def main():
    verificar_acceso()

    def reset_plan_state():
        keys_to_clear = [
            'plan_ready', 'plan_sint_final', 'plan_sint_e',
            'simulacion_plan_lista', 'raw_plan_df', 'plan_res', 'plan_res_e'
        ]
        for key in keys_to_clear:
            if key in st.session_state:
                del st.session_state[key]

    with st.sidebar:
        st.header("📂 Archivos Base")
        with st.expander("🔗 Cargar desde GitHub (Batch)", expanded=False):
            urls_txt = st.text_area("Lista de URLs", placeholder="https://github.com/...", height=100)
            gh_via = st.radio("Tipo manual", ["Detección Automática", "Planilla Maestra", "Pasajeros V1", "Pasajeros V2", "Prevenciones"], horizontal=False, index=0)
            if st.button("⬇️ Descargar Todo", use_container_width=True): 
                urls = [u.strip() for u in urls_txt.split('\n') if u.strip()]
                if urls:
                    success_count = 0
                    for url in urls:
                        with st.spinner(f"Descargando {url.split('/')[-1]}..."):
                            nm, data_or_err = leer_github(url)
                        if nm and isinstance(data_or_err, bytes):
                            lnm = nm.lower()
                            if gh_via == "Planilla Maestra": k = "gh_blobs_planilla"
                            elif gh_via == "Pasajeros V1": k = "gh_blobs_px1"
                            elif gh_via == "Pasajeros V2": k = "gh_blobs_px2"
                            elif gh_via == "Prevenciones": k = "gh_blobs_prev"
                            else:
                                if "prevencion" in lnm or "tsr" in lnm: k = "gh_blobs_prev"
                                elif "planilla" in lnm or "maestra" in lnm: k = "gh_blobs_planilla"
                                elif "v1" in lnm or "via1" in lnm: 
                                    if "pax" in lnm or "pasajero" in lnm or "export" in lnm: k = "gh_blobs_px1"
                                    else: k = "gh_blobs_planilla"
                                elif "v2" in lnm or "via2" in lnm:
                                    if "pax" in lnm or "pasajero" in lnm or "export" in lnm: k = "gh_blobs_px2"
                                    else: k = "gh_blobs_planilla"
                                elif "pax" in lnm or "pasajero" in lnm or "export" in lnm: k = "gh_blobs_px1"
                                else: k = "gh_blobs_planilla" 
                            if k not in st.session_state: st.session_state[k] = []
                            st.session_state[k].append((nm, data_or_err))
                            success_count += 1
                    if success_count > 0:
                        st.success(f"✅ Se cargaron {success_count} archivos.")
                        st.rerun()

            st.divider()
            for lbl, key in [("Planilla","gh_blobs_planilla"),("Pax V1","gh_blobs_px1"),("Pax V2","gh_blobs_px2"),("Prevenciones","gh_blobs_prev")]:
                blobs_gh = st.session_state.get(key, [])
                if blobs_gh:
                    st.caption(f"GitHub {lbl}: {len(blobs_gh)} archivo(s)")
                    if st.button(f"🗑️ Limpiar {lbl}", key=f"gh_clear_{lbl}"):
                        st.session_state[key] = []; st.rerun()

        st.subheader("Carga de Planillas Locales")
        archivo_planilla = st.file_uploader("📂 Planilla Maestra (.csv, .xlsx, .xls)", type=['csv', 'xlsx', 'xls'], key="planilla_maestra_sidebar")
        f_px1 = st.file_uploader("Pasajeros Vía 1", accept_multiple_files=True, key="px1")
        f_px2 = st.file_uploader("Pasajeros Vía 2", accept_multiple_files=True, key="px2")
        tipo_dia_plan = st.selectbox("Tipo de Día para Demanda", ["Laboral", "Sábado", "Domingo/Festivo"], key="td_plan")
        f_prev = st.file_uploader("🚧 Prevenciones de Vía (.csv, .xlsx)", accept_multiple_files=True, key="prev")
        
        st.divider()
        st.subheader("⚙️ Parámetros Físicos de Red")
        
        use_rm      = st.checkbox("🚦 Velocidades RM (Riel Mojado)", value=True, on_change=reset_plan_state)
        use_pend    = st.toggle("⛰️ Pendientes Físicas", value=True, on_change=reset_plan_state)
        use_regen   = st.toggle("⚡ Activar Regeneración", value=True, on_change=reset_plan_state)
        tipo_regen  = st.radio(
            "Modelo de Regeneración",
            ["Probabilístico (Headway)", "Físico (Load Flow DC, misma vía)"],
            help="Probabilístico: receptividad según headway entre trenes — calibrado para MERVAL. Físico: matching segundo a segundo entre trenes de la misma vía.",
            on_change=reset_plan_state
        )
        
        st.divider()
        st.subheader("🌡️ Climatización y Auxiliares")
        mes_sel = st.selectbox("Mes de operación", ["Enero","Febrero","Marzo","Abril","Mayo","Junio","Julio","Agosto","Septiembre","Octubre","Noviembre","Diciembre"], index=3, on_change=reset_plan_state)
        _M = {"Enero":"verano","Febrero":"verano","Marzo":"otoño","Abril":"otoño","Mayo":"otoño","Junio":"invierno","Julio":"invierno","Agosto":"invierno","Septiembre":"primavera","Octubre":"primavera","Noviembre":"primavera","Diciembre":"verano"}
        estacion_anio = _M[mes_sel]
        
        st.divider()
        st.subheader("🔌 Configuración de Red")
        
        try: 
            ser_data_safe = getattr(config, 'SER_DATA', SER_DATA_DEFAULT)
        except: 
            ser_data_safe = SER_DATA_DEFAULT
        
        all_ser_names = [s[1] for s in ser_data_safe]
        active_ser_names = st.multiselect("Subestaciones Activas", all_ser_names, default=all_ser_names, on_change=reset_plan_state)
        active_sers = [s for s in ser_data_safe if s[1] in active_ser_names]
        if not active_sers: 
            active_sers = [ser_data_safe[0]]
        
        gap_vias = st.slider("Separación Visual Vías (px)", 120, 350, 200, 10)

    # Ingesta en memoria
    def _all_blobs_internal(f_uploader, gh_key): 
        return tuple(leer(f_uploader) + st.session_state.get(gh_key, []))

    # THDR ya no se usa (mapa histórico retirado). Solo se procesan pasajeros y prevenciones.
    b1 = ()
    b2 = ()
    bx1 = _all_blobs_internal(f_px1, "gh_blobs_px1")
    bx2 = _all_blobs_internal(f_px2, "gh_blobs_px2")
    b_prev = _all_blobs_internal(f_prev, "gh_blobs_prev")
    
    file_signature = ""
    for b in [b1, b2, bx1, bx2, b_prev]:
        for nm, data in b:
            file_signature += f"{nm}_{len(data)}|"

    df_all, df_px, err_t, err_p = procesar_datos_completos(b1, b2, bx1, bx2, file_signature)
    
    prevenciones_list = procesar_prevenciones_independiente(b_prev, file_signature)

    # Diagnóstico de prevenciones en la barra lateral
    with st.sidebar:
        if prevenciones_list:
            st.success(f"✅ {len(prevenciones_list)} prevenciones cargadas")
            for via in [1, 2]:
                prev_via = [p for p in prevenciones_list if p['via'] == via]
                if prev_via:
                    velocidades = set(p['v_kmh'] for p in prev_via)
                    st.caption(f"Vía {via}: {len(prev_via)} tramos, velocidades {velocidades}")
        else:
            st.warning("⚠️ Sin prevenciones cargadas")

        if err_t:
            with st.expander(f"⚠️ {len(err_t)} Errores de Lectura THDR"):
                for e in err_t: st.caption(e)
        if err_p:
            with st.expander(f"⚠️ {len(err_p)} Errores de Lectura Pasajeros"):
                for e in err_p: st.caption(e)

    fechas = sorted(list(set([str(d) for d in df_all['Fecha_str'].unique() if pd.notna(d)]))) if not df_all.empty else []

    tab_planificador, tab_optimizador = st.tabs([
        "🔮 Planificador de Escenarios",
        "⚡ Optimizador de Flota"
    ])
    
    with tab_planificador:
        st.subheader("🔮 Proyección de Malla y Capex Operativo")

        # Variables Externas y Rendimiento del Tren ahora en el sidebar (junto a archivos)
        with st.sidebar:
            st.divider()
            st.subheader("🌡️ Variables Externas")
            estacion_anio_plan = st.selectbox("Estación del Año (HVAC)", ["verano", "otoño", "invierno", "primavera"], index=3, key="est_plan")
            # tipo_dia_plan se selecciona arriba, debajo de la carga de pasajeros

            st.subheader("🎛️ Rendimiento del Tren")
            pct_trac_plan = st.slider("% Tracción Máxima (Aceleración)", 30, 100, 90, 5, help="Limita la fuerza de tracción disponible. Valores bajos reducen consumo pero aumentan el tiempo de viaje. En pendientes pronunciadas el tren puede no alcanzar la velocidad consigna.")

            pax_promedio_viaje = {"Laboral": 280, "Sábado": 160, "Domingo/Festivo": 110}[tipo_dia_plan]

            df_px_filtered = pd.DataFrame()
            nombre_perfil = f"Estático ({pax_promedio_viaje} pax)"

            if not df_px.empty:
                fechas_disp_todas = sorted([str(x) for x in df_px['Fecha_s'].dropna().unique() if str(x).strip() and str(x).lower() not in ["none", "nan", "fecha no detectada", "nat"]])
                fechas_disp_tipo = [f for f in fechas_disp_todas if clasificar_dia(f) == tipo_dia_plan]

                if fechas_disp_tipo:
                    fechas_sel_plan = st.multiselect(
                        f"📅 Fechas disponibles ({tipo_dia_plan}) para promediar:",
                        fechas_disp_tipo,
                        default=fechas_disp_tipo,
                        key="ms_pax_plan"
                    )

                    if fechas_sel_plan:
                        st.success(f"✅ Promediando demanda de {len(fechas_sel_plan)} día(s) tipo {tipo_dia_plan}.")
                        nombre_perfil = f"Promedio Real ({len(fechas_sel_plan)} días {tipo_dia_plan})"
                        df_px_filtered = df_px[df_px['Fecha_s'].isin(fechas_sel_plan)].copy()
                    else:
                        st.warning(f"⚠️ Selecciona al menos una fecha. Usando perfil estático: {pax_promedio_viaje} pax")
                else:
                    st.warning(f"⚠️ No hay datos para días tipo '{tipo_dia_plan}'. Usando perfil estático: {pax_promedio_viaje} pax")
            else:
                st.warning(f"⚠️ Sin datos de pasajeros cargados. Usando perfil estático: {pax_promedio_viaje} pax")

        # Fuente de Datos a pantalla completa (ancho completo)
        if True:
            modo_plan = st.radio("Fuente de Datos", ["Planilla Maestra (Subir CSV/Excel)", "Matriz Sintética", "Laboratorio (Tramo Único)"], horizontal=True)
            
            if modo_plan == "Matriz Sintética":
                if 'df_plan' not in st.session_state: 
                    st.session_state['df_plan'] = pd.DataFrame([{"Origen": "Puerto", "Destino": "Limache", "Flota": "XT-100", "Configuración": "Doble", "Cantidad": 40}])
                df_plan_edit = st.data_editor(st.session_state['df_plan'], num_rows="dynamic", use_container_width=True)
            
            elif modo_plan == "Planilla Maestra (Subir CSV/Excel)":
                # El uploader de Planilla Maestra ahora está en el sidebar (archivo_planilla)
                if not archivo_planilla:
                    st.info("👈 Sube tu Planilla Maestra en el panel lateral (sección 'Carga de Planillas Locales').")
                if archivo_planilla:
                    try:
                        df_temp, msg = parsear_planilla_maestra(archivo_planilla.getvalue(), archivo_planilla.name)
                        if df_temp.empty: 
                            st.error(f"Error procesando: {msg}")
                        else:
                            st.success("✅ Planilla decodificada. Distribuye la flota por trayecto (Rolling Stock Rostering):")
                            rutas_unicas = list(df_temp['svc_type'].value_counts().keys())
                            from etl_parser import asignar_flota_planilla
                            df_asignado = asignar_flota_planilla(df_temp.copy())
                            # Inicializar tabla con asignación real (no todo XT-100)
                            if 'flota_map_v2' not in st.session_state or set(st.session_state['flota_map_v2']['Trayecto'].dropna()) != set(rutas_unicas+['TOTAL']):
                                filas = []
                                for r in rutas_unicas:
                                    sub = df_asignado[df_asignado['svc_type']==r]
                                    filas.append({
                                        'Trayecto': r,
                                        'Simple': int((~sub['doble']).sum()),
                                        'Doble':  int(sub['doble'].sum()),
                                        'Total':  len(sub),
                                        'XT-100': int((sub['tipo_tren']=='XT-100').sum()),
                                        'XT-M':   int((sub['tipo_tren']=='XT-M').sum()),
                                        'SFE':    int((sub['tipo_tren']=='SFE').sum()),
                                        'km/viaje': round(abs(sub['km_dest'].iloc[0]-sub['km_orig'].iloc[0]),2) if len(sub)>0 else 0,
                                        'Total km': round(sum(abs(r2['km_dest']-r2['km_orig'])*(2 if r2['doble'] else 1) for _,r2 in sub.iterrows()),2),
                                    })
                                # Fila total
                                total_km = sum(f['Total km'] for f in filas)
                                filas.append({
                                    'Trayecto': 'TOTAL',
                                    'Simple': sum(f['Simple'] for f in filas),
                                    'Doble':  sum(f['Doble']  for f in filas),
                                    'Total':  sum(f['Total']  for f in filas),
                                    'XT-100': sum(f['XT-100'] for f in filas),
                                    'XT-M':   sum(f['XT-M']   for f in filas),
                                    'SFE':    sum(f['SFE']    for f in filas),
                                    'km/viaje': '',
                                    'Total km': round(total_km, 2),
                                })
                                st.session_state['flota_map_v2'] = pd.DataFrame(filas)
                            df_flota_edit = st.data_editor(st.session_state['flota_map_v2'], hide_index=True, use_container_width=True)
                            st.session_state['temp_df_plan'] = df_asignado
                            st.session_state['temp_flota_edit'] = df_flota_edit
                            st.session_state['flota_map_v2'] = df_flota_edit  # reflejar ediciones
                    except Exception as err:
                        st.error(f"Fallo de lectura de planilla maestra: {err}")
            
            elif modo_plan == "Laboratorio (Tramo Único)":
                try: est_safe = getattr(config, 'ESTACIONES', [])
                except NameError: est_safe = ['Puerto', 'Limache']
                
                col_s1, col_s2, col_s3, col_s4 = st.columns(4)
                with col_s1: sb_orig = st.selectbox("Estación Origen", est_safe, key="sb_o")
                with col_s2: sb_dest = st.selectbox("Estación Destino", est_safe, index=max(0, len(est_safe)-1), key="sb_d")
                with col_s3: sb_flota = st.selectbox("Tipo de Tren", ["XT-100", "XT-M", "SFE"], key="sb_f")
                with col_s4: sb_pax = st.number_input("Pasajeros a bordo", 0, 1000, 150)

                sb_modo = st.radio("Modo de Circulación", ["Modo Servicio", "Modo Vacío"], horizontal=True,
                                   help="Servicio: el tren se detiene en cada estación. Vacío: pasa por las estaciones a 30 km/h sin detenerse.")
                sb_es_vacio = (sb_modo == "Modo Vacío")
                
                if st.button("⚡ Simular Tramo", use_container_width=True):
                    if sb_orig != sb_dest:
                        idx_o, idx_d = est_safe.index(sb_orig), est_safe.index(sb_dest)
                        try: km_acum_safe = getattr(config, 'KM_ACUM', [])
                        except NameError: km_acum_safe = [0.0, 43.13]
                        if not km_acum_safe: km_acum_safe = [0.0, 43.13]
                        
                        km_o, km_d = km_acum_safe[idx_o], km_acum_safe[idx_d]
                        via_sb = 1 if idx_o < idx_d else 2
                        nodos_sb = [(0.0, km_acum_safe[i]) for i in (range(idx_o, idx_d + 1) if via_sb == 1 else range(idx_o, idx_d - 1, -1))]
                        
                        datos_sim_sb = None
                        with st.spinner("Calculando termodinámica..."):
                            try:
                                trc_sb, aux_sb, reg_sb, datos_sim_sb, neto_sb, th_sb, _ = simular_tramo_termodinamico(
                                    sb_flota, False, km_o, km_d, via_sb, pct_trac_plan, use_rm, use_pend, nodos_sb, {}, sb_pax, None, 
                                    None, estacion_anio_plan, 480.0, es_vacio=sb_es_vacio, prevenciones=prevenciones_list
                                )
                            except TypeError:
                                trc_sb, aux_sb, reg_sb, datos_sim_sb, neto_sb, th_sb, _ = simular_tramo_termodinamico(
                                    sb_flota, False, km_o, km_d, via_sb, pct_trac_plan, use_rm, use_pend, nodos_sb, {}, sb_pax, None, 
                                    None, estacion_anio_plan, 480.0, es_vacio=sb_es_vacio
                                )
                        
                        try:
                            # Laboratorio = un solo tren: no hay otro que absorba la regeneración
                            # exportada (receptividad de red = 0). El excedente regenerado va al
                            # reóstato. El autoconsumo de aux en frenado ya está descontado en el motor.
                            neto_lab = trc_sb + aux_sb
                            distrib_sb = distribuir_energia_sers(neto_lab, th_sb, km_o, km_d, active_sers)
                            try: eta_ser = getattr(config, 'ETA_SER_RECTIFICADOR', 0.96)
                            except NameError: eta_ser = 0.96
                            
                            tot_ser_sb = sum(max(0.0, v) for v in distrib_sb.values()) / eta_ser
                            avg_dem_sb = {k: max(0.0, v) / eta_ser / max(0.001, th_sb) for k, v in distrib_sb.items()}
                            try: _eta_trafo_sb = ETA_TRAFO_RED
                            except NameError: _eta_trafo_sb = 0.99
                            loss_sb = calcular_flujo_ac_nodo(avg_dem_sb)['P_loss_kw'] * max(0.001, th_sb)
                            seat_sb = (tot_ser_sb / _eta_trafo_sb) + loss_sb
                            ide_sb = seat_sb / max(0.001, abs(km_d - km_o))
                            
                            st.success(f"Simulación exitosa: {sb_orig} ➔ {sb_dest} | Distancia: {abs(km_d - km_o):.2f} km")
                            c_sb1, c_sb2, c_sb3 = st.columns(3)
                            c_sb1.metric("⏱️ Tiempo de Viaje", f"{th_sb * 60:.1f} min")
                            c_sb2.metric("⚡ Energía Neta (SEAT)", f"{seat_sb:.1f} kWh")
                            c_sb3.metric("💡 IDE del Tramo (SEAT)", f"{ide_sb:.3f} kWh/km")
                        except Exception as e:
                            st.error(f"Simulación Física Completada: Tracción {trc_sb:.1f} kWh. (Red Eléctrica no conectada en GUI. Error: {e})")

                        # Perfiles del tramo: velocidad · altura · tracción
                        if figura_perfiles is not None and isinstance(datos_sim_sb, dict):
                            _fig_lab = figura_perfiles(
                                datos_sim_sb,
                                titulo=f"{sb_orig} → {sb_dest} · {sb_flota} · V{via_sb}")
                            if _fig_lab is not None:
                                st.plotly_chart(_fig_lab, use_container_width=True,
                                                key="lab_perfiles_fig")
                        elif figura_perfiles is None:
                            st.caption(f"📈 Perfiles no disponibles: {_PERFILES_IMPORT_ERROR}")

            # === EDITOR DE MANIOBRAS DE ACOPLE / DESACOPLE ===
            if st.session_state.get('raw_plan_df') is not None:
                with st.expander("🔗 Acoplar / Desacoplar trenes (cambio de formación a mitad de servicio)"):
                    st.caption("Define en qué N° de viaje y estación un tren cambia de formación. "
                               "Desacoplar (CORTE): sale doble y sigue simple desde ese punto. "
                               "Acoplar (ACOPLE): sale simple y sigue doble. En Vía 1 se asume desacople por defecto.")
                    _df_man = st.session_state['raw_plan_df']
                    _col_viaje = 'nro_viaje' if 'nro_viaje' in _df_man.columns else 'num_servicio'
                    # ordenar numéricamente los números de viaje
                    def _ord_viaje(x):
                        try: return int(x)
                        except: return 999999
                    _viajes_disp = sorted([str(s) for s in _df_man[_col_viaje].unique()], key=_ord_viaje)
                    _est_km = dict(zip(ESTACIONES, KM_ACUM)) if 'ESTACIONES' in dir() and 'KM_ACUM' in dir() else {}
                    if not _est_km:
                        try:
                            _est_km = dict(zip(config.ESTACIONES, config.KM_ACUM))
                        except Exception:
                            _est_km = {}

                    cm1, cm2, cm3 = st.columns([2, 2, 1.5])
                    with cm1:
                        _viaje_sel = st.selectbox("N° de Viaje", _viajes_disp, key="man_viaje_sel")
                    with cm2:
                        _est_sel = st.selectbox("Estación de la maniobra", list(_est_km.keys()), key="man_est_sel")
                    with cm3:
                        # detectar la vía del viaje seleccionado para sugerir el tipo
                        _via_svc = None
                        _svc_del_viaje = None
                        try:
                            _fila_viaje = _df_man[_df_man[_col_viaje].astype(str) == _viaje_sel].iloc[0]
                            _via_svc = int(_fila_viaje['Via'])
                            _svc_del_viaje = str(_fila_viaje['num_servicio'])
                        except Exception:
                            pass
                        _tipo_def = 0 if _via_svc == 1 else 1  # V1 → desacople por defecto
                        _tipo_sel = st.selectbox("Tipo", ["Desacoplar (CORTE)", "Acoplar (ACOPLE)"], index=_tipo_def, key="man_tipo_sel")

                    if _svc_del_viaje is not None:
                        st.caption(f"Viaje {_viaje_sel} → Tren {_svc_del_viaje}, Vía {_via_svc}")

                    if 'maniobras_def' not in st.session_state:
                        st.session_state['maniobras_def'] = {}

                    cb1, cb2 = st.columns(2)
                    with cb1:
                        if st.button("➕ Agregar maniobra", use_container_width=True, key="man_add"):
                            _km_man = _est_km.get(_est_sel, 0.0)
                            _accion = "CORTE" if "CORTE" in _tipo_sel else "ACOPLE"
                            st.session_state['maniobras_def'][_viaje_sel] = f"{_accion}@{_km_man}"
                            st.session_state['maniobras_cambiadas'] = True
                            st.rerun()
                    with cb2:
                        if st.button("🗑️ Limpiar todas", use_container_width=True, key="man_clear"):
                            st.session_state['maniobras_def'] = {}
                            st.session_state['maniobras_cambiadas'] = True
                            st.rerun()

                    # botón para eliminar una maniobra puntual
                    if st.session_state['maniobras_def']:
                        _viaje_borrar = st.selectbox("Eliminar maniobra del viaje:", ["—"] + list(st.session_state['maniobras_def'].keys()), key="man_del_sel")
                        if _viaje_borrar != "—" and st.button("Eliminar esa maniobra", key="man_del_btn"):
                            st.session_state['maniobras_def'].pop(_viaje_borrar, None)
                            st.session_state['maniobras_cambiadas'] = True
                            st.rerun()

                    if st.session_state.get('maniobras_cambiadas', False):
                        st.warning("⚠️ Cambiaste las maniobras. Pulsa **🚀 Ejecutar Gemelo Digital** para recalcular el consumo con los nuevos acoples/desacoples.")

                    # Mostrar maniobras definidas
                    if st.session_state['maniobras_def']:
                        st.markdown("**Maniobras definidas:**")
                        _km_to_est = {v: k for k, v in _est_km.items()}
                        for _viaje, _man in st.session_state['maniobras_def'].items():
                            _acc, _km = _man.split('@')
                            _nom_est = _km_to_est.get(float(_km), f"km {_km}")
                            _verbo = "desacopla (doble→simple)" if _acc == "CORTE" else "acopla (simple→doble)"
                            st.write(f"• Viaje **{_viaje}**: {_verbo} en **{_nom_est}**")
                    else:
                        st.info("No hay maniobras definidas. El plan usa las formaciones de la planilla.")

            _modo_ok = modo_plan in ["Matriz Sintética", "Planilla Maestra (Subir CSV/Excel)"]
            if _modo_ok:
                st.markdown("**Ejecutar simulación:**")
                _bc1, _bc2 = st.columns(2)
                with _bc1:
                    _btn_circ = st.button("🚆 Circulación de Trenes", use_container_width=True, type="primary",
                                          help="Rápido (~30s). Muestra el SCADA, horarios y mapa de movimiento de los trenes.")
                with _bc2:
                    _btn_cons = st.button("⚡ Consumo de Energía", use_container_width=True,
                                          help="Pesado (~2 min). Calcula energía, SEAT, IDE con anti-alcance.")
                st.caption("La circulación es rápida y muestra el movimiento de trenes. "
                           "El consumo es más lento (incluye el anti-alcance) y calcula la energía. "
                           "Puedes ver la circulación primero y pedir el consumo solo cuando lo necesites.")
            else:
                _btn_circ = False
                _btn_cons = False

            if _btn_circ or _btn_cons:
                # Determinar qué modo de ejecución
                st.session_state['modo_ejecucion'] = 'consumo' if _btn_cons else 'circulacion'
                st.session_state['simulacion_plan_lista'] = False
                st.session_state['maniobras_cambiadas'] = False
                with st.spinner("Decodificando Malla e inyectando al Motor Cinemático Termodinámico..."):
                    if modo_plan == "Matriz Sintética":
                        df_sintetico_list = []
                        try: est_safe = getattr(config, 'ESTACIONES', [])
                        except NameError: est_safe = ['Puerto', 'Limache']
                        try: km_acum_safe = getattr(config, 'KM_ACUM', [])
                        except NameError: km_acum_safe = [0.0, 43.13]
                        try: ec_safe = getattr(config, 'EC', [])
                        except NameError: ec_safe = ['PU', 'LI']
                        
                        for idx, row in df_plan_edit.iterrows():
                            if row['Cantidad'] <= 0 or row['Origen'] == row['Destino']: continue
                            try:
                                i_o, i_d = est_safe.index(row['Origen']), est_safe.index(row['Destino'])
                                via = 1 if i_o < i_d else 2
                                nodos_sint = [(0.0, km_acum_safe[i]) for i in (range(i_o, i_d + 1) if via==1 else range(i_o, i_d - 1, -1))]
                                k_o, k_d = km_acum_safe[i_o], km_acum_safe[i_d]
                                svc_t = f"{ec_safe[i_o]}-{ec_safe[i_d]}"
                                interval = (1350 - 360) / row['Cantidad']
                                
                                for i in range(int(row['Cantidad'])):
                                    df_sintetico_list.append({
                                        '_id': f"SINT_{idx}_{i}", 't_ini': 360 + i * interval, 'Via': via, 
                                        'km_orig': k_o, 'km_dest': k_d, 'nodos': nodos_sint, 
                                        'tipo_tren': row['Flota'], 'doble': row['Configuración'] == "Doble", 
                                        'num_servicio': f"VIRT_{idx}_{i}", 'maniobra': None, 'svc_type': svc_t
                                    })
                            except: pass
                        df_sint = pd.DataFrame(df_sintetico_list)
                    else:
                        if 'temp_df_plan' not in st.session_state: st.stop()
                        df_sint = st.session_state['temp_df_plan'].copy().sort_values('t_ini')
                        # Si todos son XT-100, re-ejecutar asignación de flota
                        # tipo_tren ya viene asignado desde parsear_planilla_maestra

                    if df_sint.empty: st.stop()
                    st.session_state['raw_plan_df'] = df_sint
                    st.session_state['simulacion_plan_lista'] = True

            if st.session_state.get('simulacion_plan_lista', False) and 'raw_plan_df' in st.session_state:
                # Aplicar las maniobras de acople/desacople definidas por el usuario (por N° de viaje)
                _df_plan_con_man = st.session_state['raw_plan_df'].copy()
                _maniobras = st.session_state.get('maniobras_def', {})
                if _maniobras:
                    _col_v = 'nro_viaje' if 'nro_viaje' in _df_plan_con_man.columns else 'num_servicio'
                    _df_plan_con_man['maniobra'] = _df_plan_con_man.apply(
                        lambda r: _maniobras.get(str(r[_col_v]), r.get('maniobra')), axis=1)
                _man_sig = str(sorted(_maniobras.items())) if _maniobras else ""
                plan_sig = str(st.session_state.get('df_plan', '')) + str(st.session_state.get('temp_flota_edit', '')) + str(pax_promedio_viaje) + file_signature + str(sorted([(p.get('km_min',0),p.get('km_max',0),p.get('v_kmh',0),p.get('via',0)) for p in (prevenciones_list or [])], key=lambda x: x[0])) + str(use_pend) + str(use_rm) + str(use_regen) + str(tipo_regen) + str(estacion_anio_plan) + str(pct_trac_plan) + _man_sig
                _con_aa = st.session_state.get('modo_ejecucion', 'circulacion') == 'consumo'
                df_sint_final, df_sint_e = procesar_planificador_reactivo(_df_plan_con_man, df_px_filtered, estacion_anio_plan, pct_trac_plan, use_rm, use_pend, use_regen, tipo_regen, pax_promedio_viaje, prevenciones_list, plan_sig, config_sig=get_config_hash(), man_sig=_man_sig, con_anti_alcance=_con_aa)
                # Guardar resultados para el Optimizador de Flota
                st.session_state['opt_df_sint_e'] = df_sint_e
                st.session_state['opt_params'] = {
                    'pct_trac': pct_trac_plan, 'use_rm': use_rm, 'use_pend': use_pend,
                    'use_regen': use_regen, 'tipo_regen': tipo_regen,
                    'estacion_anio': estacion_anio_plan,
                }
                
                # Forzar tipos numéricos
                cols_num = ['t_ini', 't_fin', 'kwh_viaje_trac', 'kwh_viaje_aux', 'kwh_viaje_regen', 'kwh_reostato', 'kwh_viaje_neto', 't_viaje_h', 'tren_km']
                for col in cols_num:
                    if col in df_sint_e.columns:
                        df_sint_e[col] = pd.to_numeric(df_sint_e[col], errors='coerce')
                    if col in df_sint_final.columns:
                        df_sint_final[col] = pd.to_numeric(df_sint_final[col], errors='coerce')
                
                st.divider()
                _modo_exec = st.session_state.get('modo_ejecucion', 'circulacion')
                try:
                    # SCADA + horarios + mapa: siempre (circulación y consumo)
                    render_gemelo_digital(df_sint_final, df_sint_e, active_sers, f"Simulación: {nombre_perfil}", pct_trac_plan, use_rm, use_pend, estacion_anio_plan, "plan", gap_vias, pax_dia_total=int(df_sint_final['pax_abordo'].sum()))

                    # Perfiles del viaje (velocidad · altura · tracción) por servicio
                    if render_perfiles_viaje is not None:
                        render_perfiles_viaje(df_sint_e, "plan")
                    else:
                        st.warning(f"📈 Módulo de perfiles no cargado. Verifica que **perfiles_viaje.py** "
                                   f"esté junto a app.py. Detalle: {_PERFILES_IMPORT_ERROR}")

                    # Reporte de tensión de red con la corriente SIMULTÁNEA de los trenes
                    try:
                        import motor_fisico as _mf
                        if hasattr(_mf, 'analizar_tension_secciones'):
                            _rep = _mf.analizar_tension_secciones(df_sint_e)
                            st.caption("⚡ Red DC — corriente simultánea de todos los trenes (modelo radial conservador)")
                            _r1 = st.columns(4)
                            _r1[0].metric("Tensión barra SER máx", f"{_rep['v_ser_max']:.0f} V")
                            _r1[1].metric("Tensión barra SER mín", f"{_rep['v_ser_min']:.0f} V")
                            _r1[2].metric("V mín en catenaria", f"{_rep['v_min_global']:.0f} V")
                            _r1[3].metric("Pasos en subtensión", f"{_rep['n_subtension']}")
                            _r2 = st.columns(3)
                            _r2[0].metric("Corriente máx SER (rectificadora)", f"{_rep['i_ser_max']:.0f} A")
                            _r2[1].metric("Corriente máx SEAT (principal)", f"{_rep['i_seat_max']:.0f} A")
                            _r2[2].metric("Demanda pico", f"{_rep['peak_demand_kw']:.0f} kW")

                            # Corriente máxima por cada una de las 4 SER (2 trafos en serie c/u)
                            _det = _rep.get('detalle_ser', {})
                            if _det:
                                _filas = []
                                for _nom, _d in sorted(_det.items(), key=lambda x: x[1]['km']):
                                    _filas.append({
                                        "SER": _nom,
                                        "PK (km)": round(_d['km'], 1),
                                        "I máx (A)": round(_d['i_pico_A']),
                                        "I nominal (A)": round(_d['i_nom_A']),
                                        "Carga (%)": round(_d['carga_pct']),
                                        "Barra (V)": round(_d['v_barra']),
                                        "Trafos": _d.get('n_trafos', 2),
                                        "Cap. total (kW)": round(_d.get('cap_total_kw', 0)),
                                    })
                                st.caption("Corriente máxima por SER (cada SER = 2 trafos en serie; en serie ambos llevan la misma corriente)")
                                st.dataframe(pd.DataFrame(_filas), use_container_width=True, hide_index=True)
                    except Exception as _e_red:
                        st.caption(f"(Reporte de tensión no disponible: {_e_red})")

                    if _modo_exec == 'consumo':
                        # Dashboards de energía + tablas SEAT: solo en modo consumo
                        render_dashboard_energia_v112(df_sint_e, active_sers, "Planificador", st.session_state.get('sl_ui_plan', 480.0))
                        render_tablas_thdr_planificador(df_sint_final, pct_trac_plan, estacion_anio_plan, use_rm, prevenciones_list)
                    else:
                        st.info("🚆 Mostrando solo la **circulación** de trenes. "
                                "Para ver el consumo de energía (SEAT, IDE, dashboards), pulsa **⚡ Consumo de Energía**.")

                    # Descarga de tabla SEAT cada 15 min (solo en modo consumo)
                    if generar_tabla_seat_15min is not None and _modo_exec == 'consumo':
                        st.divider()
                        st.markdown("##### 📊 Consumo SEAT por Franja Horaria")
                        try:
                            import tempfile, os
                            granularidad_p = st.radio(
                                "Intervalo de la tabla",
                                ["Cada 15 minutos", "Cada hora"],
                                horizontal=True, key="gran_seat_plan")
                            paso_p = 15.0 if granularidad_p == "Cada 15 minutos" else 60.0
                            sufijo_p = "15min" if paso_p == 15 else "60min"
                            ruta_15 = os.path.join(tempfile.gettempdir(), f"SEAT_{sufijo_p}_Planificador.xlsx")
                            _, df_tabla_15 = generar_tabla_seat_15min(df_sint_e, config, active_sers, distribuir_energia_sers, calcular_flujo_ac_nodo, ruta_15, paso_min=paso_p)
                            st.dataframe(df_tabla_15, use_container_width=True, height=300)
                            with open(ruta_15, 'rb') as f:
                                st.download_button(
                                    f"⬇️ Descargar tabla SEAT ({granularidad_p.lower()}) (xlsx)",
                                    data=f.read(),
                                    file_name=f"SEAT_{sufijo_p}_Planificador.xlsx",
                                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                    use_container_width=True)
                            st.caption(f"Consumo SEAT por franja: total y por subestación, en kWh y kW medio. "
                                       "Incluye pérdidas de rectificador y AC.")
                        except Exception as e:
                            st.warning(f"No se pudo generar la tabla: {e}")
                except Exception as e:
                    st.error(f"Fallo al graficar UI del Planificador: {e}")

    with tab_optimizador:
        st.subheader("⚡ Optimizador de Distribución de Flota")
        st.markdown(
            "Reasigna el **tipo de tren** asignado a cada servicio para minimizar el consumo "
            "energético, respetando la flota disponible y la capacidad de pasajeros. "
            "Los trenes más eficientes (XT-M) se asignan a los servicios de mayor distancia. "
            "**No altera los horarios** — solo qué unidad cubre cada servicio."
        )

        if optimizar_asignacion_flota is None:
            st.error("Módulo optimizador no disponible (falta optimizador_flota.py).")
        elif not st.session_state.get('simulacion_plan_lista', False) or 'raw_plan_df' not in st.session_state:
            st.info("👈 Primero carga y ejecuta una malla en la pestaña **Planificador de Escenarios**. "
                    "Luego vuelve aquí para optimizar la asignación de flota.")
        else:
            df_base_opt = st.session_state['raw_plan_df'].copy()

            col_o1, col_o2 = st.columns([1, 1])
            with col_o1:
                priorizar_opt = st.radio("Criterio de optimización",
                                         ["Minimizar consumo total (kWh)", "Minimizar IDE promedio"],
                                         key="opt_crit")
            with col_o2:
                st.caption("Flota disponible (config):")
                fd = {}
                try:
                    for t, p in getattr(config, 'FLOTA', {}).items():
                        fd[t] = p.get('unidades_disponibles', 0)
                except Exception:
                    pass
                if not any(fd.values()):
                    fd = {'XT-100': 27, 'XT-M': 8, 'SFE': 5}
                st.write(" · ".join(f"**{k}**: {v}" for k, v in fd.items()))

            if st.button("🔧 Optimizar Distribución de Flota", use_container_width=True, type="primary"):
                with st.spinner("Calculando asignación óptima de flota..."):
                    try:
                        # Asegurar t_fin
                        if 't_fin' not in df_base_opt.columns:
                            df_base_opt['t_fin'] = df_base_opt['t_ini'] + 55
                        prio = 'energia' if "consumo" in priorizar_opt else 'eficiencia'
                        df_base_consumo = st.session_state.get('opt_df_sint_e', None)
                        params_sim = st.session_state.get('opt_params', None)
                        df_opt, resumen = optimizar_asignacion_flota(
                            df_base_opt, config, priorizar=prio,
                            df_consumo_base=df_base_consumo,
                            simular_fn=calcular_termodinamica_flota_v111,
                            precalcular_fn=precalcular_red_electrica_v111,
                            params_sim=params_sim,
                            prevenciones=prevenciones_list,
                            active_sers=active_sers,
                            distribuir_fn=distribuir_energia_sers,
                            flujo_fn=calcular_flujo_ac_nodo)

                        st.success(f"Optimización completada: {resumen['n_cambios']} de {resumen['n_servicios']} servicios reasignados.")

                        # Métricas de ahorro (SEAT total, igual que el planificador)
                        m1, m2, m3 = st.columns(3)
                        m1.metric("Consumo Actual (SEAT)", f"{resumen['kwh_actual']:,.0f} kWh")
                        m2.metric("Consumo Optimizado (SEAT)", f"{resumen['kwh_optimo']:,.0f} kWh",
                                  delta=f"-{resumen['ahorro_kwh']:,.0f} kWh")
                        m3.metric("Ahorro", f"{resumen['ahorro_pct']:.1f} %")

                        m4, m5, m6 = st.columns(3)
                        m4.metric("IDE Actual", f"{resumen.get('ide_actual', 0):.3f} kWh/km")
                        m5.metric("IDE Optimizado", f"{resumen.get('ide_optimo', 0):.3f} kWh/km")
                        m6.metric("Kilometraje (Tren-km)", f"{resumen.get('km_total', 0):,.1f} km")
                        if resumen.get('usa_seat_real'):
                            st.caption("✅ Consumo calculado como SEAT total (incluye pérdidas de rectificador y AC), "
                                       "idéntico al del Planificador, con la misma carga de pasajeros y prevenciones. "
                                       "El IDE usa el kilometraje Tren-km (formaciones dobles cuentan 2×).")

                        # Capacidad de terminales (restricción operativa)
                        st.divider()
                        st.markdown("##### 🚉 Ocupación de Terminales")
                        cap_term = resumen.get('cap_terminales', {})
                        if cap_term:
                            tcols = st.columns(len(cap_term))
                            for i, (nombre, v) in enumerate(cap_term.items()):
                                with tcols[i]:
                                    excede = v['excede']
                                    color = "#C62828" if excede else "#2E7D32"
                                    icono = "❌" if excede else "✅"
                                    h = int(v['pico_min']//60); m = int(v['pico_min']%60)
                                    st.markdown(
                                        f"<div style='background-color:#f9f9f9; border-radius:8px; padding:12px; "
                                        f"text-align:center; border:1px solid #eee;'>"
                                        f"<div style='font-size:14px; font-weight:bold; color:#333;'>{nombre}</div>"
                                        f"<div style='font-size:22px; font-weight:bold; color:{color}; margin:6px 0;'>"
                                        f"{icono} {v['max_ocup']} / {v['capacidad']}</div>"
                                        f"<div style='font-size:11px; color:#666;'>Pico: {h:02d}:{m:02d}</div>"
                                        f"</div>", unsafe_allow_html=True)
                            if resumen.get('excede_terminales'):
                                st.error("⚠️ La malla actual EXCEDE la capacidad de uno o más terminales. "
                                         "Puerto: máx 4 · El Belloto: máx 16 · Limache: máx 16 trenes. "
                                         "Revisa los horarios para no superar el estacionamiento disponible.")
                            else:
                                st.success("✅ La malla respeta la capacidad de todos los terminales.")

                        st.divider()

                        # Composición de flota antes/después
                        st.markdown("##### Composición de la flota (servicios por tipo)")
                        comp_cols = st.columns(2)
                        with comp_cols[0]:
                            st.caption("**Distribución actual**")
                            for tipo, n in sorted(resumen['comp_antes'].items()):
                                st.write(f"  {tipo}: {n} servicios")
                        with comp_cols[1]:
                            st.caption("**Distribución optimizada**")
                            for tipo, n in sorted(resumen['comp_despues'].items()):
                                st.write(f"  {tipo}: {n} servicios")

                        st.divider()

                        # Tabla de cambios propuestos
                        st.markdown("##### Cambios propuestos")
                        cambios = df_opt[df_opt['tipo_tren'] != df_opt['tipo_optimo']].copy()
                        if not cambios.empty:
                            cols_mostrar = ['num_servicio', 'Via', 'svc_type', 'km_tramo',
                                            'tipo_tren', 'tipo_optimo', 'kwh_actual', 'kwh_optimo']
                            cols_disp = [c for c in cols_mostrar if c in cambios.columns]
                            tabla = cambios[cols_disp].rename(columns={
                                'num_servicio': 'Servicio', 'svc_type': 'Trayecto',
                                'km_tramo': 'km', 'tipo_tren': 'Actual', 'tipo_optimo': 'Óptimo',
                                'kwh_actual': 'kWh actual', 'kwh_optimo': 'kWh óptimo'
                            })
                            st.dataframe(tabla.round(1), use_container_width=True, height=400)
                        else:
                            st.info("La distribución actual ya es óptima — no se proponen cambios.")

                        # Generar planillas V1 y V2 descargables
                        st.divider()
                        st.markdown("##### 📥 Descargar planillas optimizadas")
                        if generar_planillas_xlsx is not None:
                            try:
                                import tempfile, os
                                tmpdir = tempfile.gettempdir()
                                ruta_v1 = os.path.join(tmpdir, "Planilla_Optimizada_V1.xlsx")
                                ruta_v2 = os.path.join(tmpdir, "Planilla_Optimizada_V2.xlsx")
                                generar_planillas_xlsx(df_opt, ruta_v1, ruta_v2)

                                dl1, dl2 = st.columns(2)
                                with dl1:
                                    with open(ruta_v1, 'rb') as f:
                                        st.download_button(
                                            "⬇️ Planilla Vía 1 (optimizada)",
                                            data=f.read(),
                                            file_name="Planilla_Optimizada_V1.xlsx",
                                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                            use_container_width=True)
                                with dl2:
                                    with open(ruta_v2, 'rb') as f:
                                        st.download_button(
                                            "⬇️ Planilla Vía 2 (optimizada)",
                                            data=f.read(),
                                            file_name="Planilla_Optimizada_V2.xlsx",
                                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                            use_container_width=True)
                                st.caption("Formato idéntico a las planillas del Planificador: "
                                           "N° Viaje · Servicio · Hr Partida · N° Partida · Intervalo · Unidad · Motriz 1 · Motriz 2. "
                                           "Las motrices se asignan según el tipo óptimo (1-27 XT-100, 28-35 XT-M, 410-414 SFE).")
                            except Exception as e:
                                st.warning(f"No se pudieron generar las planillas: {e}")

                        # Tabla SEAT por franja (de la malla base)
                        if generar_tabla_seat_15min is not None and df_base_consumo is not None:
                            st.divider()
                            st.markdown("##### 📊 Consumo SEAT por Franja Horaria")
                            try:
                                import tempfile, os
                                granularidad_o = st.radio(
                                    "Intervalo de la tabla",
                                    ["Cada 15 minutos", "Cada hora"],
                                    horizontal=True, key="gran_seat_opt")
                                paso_o = 15.0 if granularidad_o == "Cada 15 minutos" else 60.0
                                sufijo_o = "15min" if paso_o == 15 else "60min"
                                ruta_15o = os.path.join(tempfile.gettempdir(), f"SEAT_{sufijo_o}_Optimizador.xlsx")
                                _, df_t15o = generar_tabla_seat_15min(df_base_consumo, config, active_sers, distribuir_energia_sers, calcular_flujo_ac_nodo, ruta_15o, paso_min=paso_o)
                                st.dataframe(df_t15o, use_container_width=True, height=300)
                                with open(ruta_15o, 'rb') as f:
                                    st.download_button(
                                        f"⬇️ Descargar tabla SEAT ({granularidad_o.lower()}) (xlsx)",
                                        data=f.read(),
                                        file_name=f"SEAT_{sufijo_o}_Optimizador.xlsx",
                                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                        use_container_width=True)
                                st.caption("Consumo SEAT por franja: total y por subestación, en kWh y kW medio.")
                            except Exception as e:
                                st.warning(f"No se pudo generar la tabla de 15 min: {e}")

                    except Exception as e:
                        st.error(f"Error en la optimización: {e}")
                        import traceback
                        st.code(traceback.format_exc())

if __name__ == "__main__": 
    main()
