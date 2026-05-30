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

@st.cache_data(show_spinner="Integrando física y demanda en Planificador...")
def procesar_planificador_reactivo(_df_sint, _df_px_filtered, estacion_anio_plan, pct_trac_plan, use_rm, use_pend, use_regen, tipo_regen, pax_promedio_viaje, _prevenciones, plan_sig, config_sig=""):
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
        df_sint_e = calcular_termodinamica_flota_v111(df_sint_final, pct_trac_plan, use_pend, use_rm, use_regen, dict_regen_sint, estacion_anio_plan, prevenciones=_prevenciones)
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


def main():
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
            gh_via = st.radio("Tipo manual", ["Detección Automática", "THDR V1", "THDR V2", "Pasajeros V1", "Pasajeros V2", "Prevenciones"], horizontal=False, index=0)
            if st.button("⬇️ Descargar Todo", use_container_width=True): 
                urls = [u.strip() for u in urls_txt.split('\n') if u.strip()]
                if urls:
                    success_count = 0
                    for url in urls:
                        with st.spinner(f"Descargando {url.split('/')[-1]}..."):
                            nm, data_or_err = leer_github(url)
                        if nm and isinstance(data_or_err, bytes):
                            lnm = nm.lower()
                            if gh_via == "THDR V1": k = "gh_blobs_v1"
                            elif gh_via == "THDR V2": k = "gh_blobs_v2"
                            elif gh_via == "Pasajeros V1": k = "gh_blobs_px1"
                            elif gh_via == "Pasajeros V2": k = "gh_blobs_px2"
                            elif gh_via == "Prevenciones": k = "gh_blobs_prev"
                            else:
                                if "prevencion" in lnm or "tsr" in lnm: k = "gh_blobs_prev"
                                elif "v1" in lnm or "via1" in lnm: 
                                    if "pax" in lnm or "pasajero" in lnm or "export" in lnm: k = "gh_blobs_px1"
                                    else: k = "gh_blobs_v1"
                                elif "v2" in lnm or "via2" in lnm:
                                    if "pax" in lnm or "pasajero" in lnm or "export" in lnm: k = "gh_blobs_px2"
                                    else: k = "gh_blobs_v2"
                                elif "pax" in lnm or "pasajero" in lnm or "export" in lnm: k = "gh_blobs_px1"
                                else: k = "gh_blobs_v1" 
                            if k not in st.session_state: st.session_state[k] = []
                            st.session_state[k].append((nm, data_or_err))
                            success_count += 1
                    if success_count > 0:
                        st.success(f"✅ Se cargaron {success_count} archivos.")
                        st.rerun()

            st.divider()
            for lbl, key in [("V1","gh_blobs_v1"),("V2","gh_blobs_v2"),("Pax V1","gh_blobs_px1"),("Pax V2","gh_blobs_px2"),("Prevenciones","gh_blobs_prev")]:
                blobs_gh = st.session_state.get(key, [])
                if blobs_gh:
                    st.caption(f"GitHub {lbl}: {len(blobs_gh)} archivo(s)")
                    if st.button(f"🗑️ Limpiar {lbl}", key=f"gh_clear_{lbl}"):
                        st.session_state[key] = []; st.rerun()

        st.subheader("Carga de Planillas Locales")
        f_v1 = st.file_uploader("THDR Vía 1 (Puerto→Limache)", accept_multiple_files=True, key="t1")
        f_v2 = st.file_uploader("THDR Vía 2 (Limache→Puerto)", accept_multiple_files=True, key="t2")
        f_px1 = st.file_uploader("Pasajeros Vía 1", accept_multiple_files=True, key="px1")
        f_px2 = st.file_uploader("Pasajeros Vía 2", accept_multiple_files=True, key="px2")
        f_prev = st.file_uploader("🚧 Prevenciones de Vía (.csv, .xlsx)", accept_multiple_files=True, key="prev")
        
        st.divider()
        st.subheader("⚙️ Parámetros Físicos de Red")
        
        st.info("💡 **Gobernador Operativo (Mapa Histórico)**\n\nEn la pestaña del *Gemelo Digital*, el % de Tracción se bloquea automáticamente al 75% o 50% según la fecha.\n\nEn el *Planificador*, podrás usar tu perilla libremente.")
        
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

    b1 = _all_blobs_internal(f_v1, "gh_blobs_v1")
    b2 = _all_blobs_internal(f_v2, "gh_blobs_v2")
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

    tab_planificador, = st.tabs([
        "🔮 Planificador de Escenarios"
    ])
    
    with tab_planificador:
        st.subheader("🔮 Proyección de Malla y Capex Operativo")
        
        col_p1, col_p2 = st.columns([1, 2])
        with col_p1:
            st.markdown("##### 🌡️ Variables Externas")
            estacion_anio_plan = st.selectbox("Estación del Año (HVAC)", ["verano", "otoño", "invierno", "primavera"], index=3, key="est_plan")
            tipo_dia_plan = st.selectbox("Tipo de Día para Demanda", ["Laboral", "Sábado", "Domingo/Festivo"], key="td_plan")
            
            st.markdown("##### 🎛️ Rendimiento del Tren")
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
                    st.warning(f"⚠️ No hay datos cargados para días tipo '{tipo_dia_plan}'. Usando perfil estático: {pax_promedio_viaje} pax")
            else: 
                st.warning(f"⚠️ Sin datos de pasajeros cargados. Usando perfil estático: {pax_promedio_viaje} pax")
            
        with col_p2:
            modo_plan = st.radio("Fuente de Datos", ["Planilla Maestra (Subir CSV/Excel)", "Matriz Sintética", "Laboratorio (Tramo Único)"], horizontal=True)
            
            if modo_plan == "Matriz Sintética":
                if 'df_plan' not in st.session_state: 
                    st.session_state['df_plan'] = pd.DataFrame([{"Origen": "Puerto", "Destino": "Limache", "Flota": "XT-100", "Configuración": "Doble", "Cantidad": 40}])
                df_plan_edit = st.data_editor(st.session_state['df_plan'], num_rows="dynamic", use_container_width=True)
            
            elif modo_plan == "Planilla Maestra (Subir CSV/Excel)":
                archivo_planilla = st.file_uploader("📂 Sube tu Planilla Maestra (.csv, .xlsx, .xls)", type=['csv', 'xlsx', 'xls'])
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
                
                if st.button("⚡ Simular Tramo", use_container_width=True):
                    if sb_orig != sb_dest:
                        idx_o, idx_d = est_safe.index(sb_orig), est_safe.index(sb_dest)
                        try: km_acum_safe = getattr(config, 'KM_ACUM', [])
                        except NameError: km_acum_safe = [0.0, 43.13]
                        if not km_acum_safe: km_acum_safe = [0.0, 43.13]
                        
                        km_o, km_d = km_acum_safe[idx_o], km_acum_safe[idx_d]
                        via_sb = 1 if idx_o < idx_d else 2
                        nodos_sb = [(0.0, km_acum_safe[i]) for i in (range(idx_o, idx_d + 1) if via_sb == 1 else range(idx_o, idx_d - 1, -1))]
                        
                        with st.spinner("Calculando termodinámica..."):
                            try:
                                trc_sb, aux_sb, reg_sb, _, neto_sb, th_sb, _ = simular_tramo_termodinamico(
                                    sb_flota, False, km_o, km_d, via_sb, pct_trac_plan, use_rm, use_pend, nodos_sb, {}, sb_pax, None, 
                                    None, estacion_anio_plan, 480.0, es_vacio=False, prevenciones=prevenciones_list
                                )
                            except TypeError:
                                trc_sb, aux_sb, reg_sb, _, neto_sb, th_sb, _ = simular_tramo_termodinamico(
                                    sb_flota, False, km_o, km_d, via_sb, pct_trac_plan, use_rm, use_pend, nodos_sb, {}, sb_pax, None, 
                                    None, estacion_anio_plan, 480.0, es_vacio=False
                                )
                        
                        try:
                            distrib_sb = distribuir_energia_sers(neto_sb, th_sb, km_o, km_d, active_sers)
                            try: eta_ser = getattr(config, 'ETA_SER_RECTIFICADOR', 0.96)
                            except NameError: eta_ser = 0.96
                            
                            tot_ser_sb = sum(max(0.0, v) for v in distrib_sb.values()) / eta_ser
                            avg_dem_sb = {k: max(0.0, v) / eta_ser / max(0.001, th_sb) for k, v in distrib_sb.items()}
                            loss_sb = calcular_flujo_ac_nodo(avg_dem_sb)['P_loss_kw'] * (1.15**2) * max(0.001, th_sb)
                            seat_sb = (tot_ser_sb + loss_sb) / 0.99
                            ide_sb = seat_sb / max(0.001, abs(km_d - km_o))
                            
                            st.success(f"Simulación exitosa: {sb_orig} ➔ {sb_dest} | Distancia: {abs(km_d - km_o):.2f} km")
                            c_sb1, c_sb2, c_sb3 = st.columns(3)
                            c_sb1.metric("⏱️ Tiempo de Viaje", f"{th_sb * 60:.1f} min")
                            c_sb2.metric("⚡ Energía Neta (SEAT)", f"{seat_sb:.1f} kWh")
                            c_sb3.metric("💡 IDE del Tramo (SEAT)", f"{ide_sb:.3f} kWh/km")
                        except Exception as e:
                            st.error(f"Simulación Física Completada: Tracción {trc_sb:.1f} kWh. (Red Eléctrica no conectada en GUI. Error: {e})")

            if modo_plan in ["Matriz Sintética", "Planilla Maestra (Subir CSV/Excel)"] and st.button("🚀 Ejecutar Gemelo Digital del Planificador", use_container_width=True, type="primary"):
                st.session_state['simulacion_plan_lista'] = False
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
                plan_sig = str(st.session_state.get('df_plan', '')) + str(st.session_state.get('temp_flota_edit', '')) + str(pax_promedio_viaje) + file_signature + str(sorted([(p.get('km_min',0),p.get('km_max',0),p.get('v_kmh',0),p.get('via',0)) for p in (prevenciones_list or [])], key=lambda x: x[0])) + str(use_pend) + str(use_rm) + str(use_regen) + str(tipo_regen) + str(estacion_anio_plan) + str(pct_trac_plan)
                df_sint_final, df_sint_e = procesar_planificador_reactivo(st.session_state['raw_plan_df'], df_px_filtered, estacion_anio_plan, pct_trac_plan, use_rm, use_pend, use_regen, tipo_regen, pax_promedio_viaje, prevenciones_list, plan_sig, config_sig=get_config_hash())
                
                # Forzar tipos numéricos
                cols_num = ['t_ini', 't_fin', 'kwh_viaje_trac', 'kwh_viaje_aux', 'kwh_viaje_regen', 'kwh_reostato', 'kwh_viaje_neto', 't_viaje_h', 'tren_km']
                for col in cols_num:
                    if col in df_sint_e.columns:
                        df_sint_e[col] = pd.to_numeric(df_sint_e[col], errors='coerce')
                    if col in df_sint_final.columns:
                        df_sint_final[col] = pd.to_numeric(df_sint_final[col], errors='coerce')
                
                st.divider()
                try:
                    render_gemelo_digital(df_sint_final, df_sint_e, active_sers, f"Simulación: {nombre_perfil}", pct_trac_plan, use_rm, use_pend, estacion_anio_plan, "plan", gap_vias, pax_dia_total=int(df_sint_final['pax_abordo'].sum()))
                    render_dashboard_energia_v112(df_sint_e, active_sers, "Planificador", st.session_state.get('sl_ui_plan', 480.0))
                    render_tablas_thdr_planificador(df_sint_final, pct_trac_plan, estacion_anio_plan, use_rm, prevenciones_list)
                except Exception as e:
                    st.error(f"Fallo al graficar UI del Planificador: {e}")

if __name__ == "__main__": 
    main()
