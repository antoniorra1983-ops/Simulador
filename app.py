import streamlit as st
import pandas as pd
import numpy as np
import time
from io import BytesIO
from datetime import datetime, date, timedelta

# Importación segura de configuración
try:
    from config import *
except ImportError:
    pass
import config

# Importación de módulos locales con arquitectura SOLID
from etl_parser import (
    procesar_thdr, calcular_dwell, cargar_pax, match_pax, 
    get_perfiles_pax, parsear_planilla_maestra, 
    calc_tren_km_real_general, clean_id, mins_to_time_str, clasificar_dia,
    cargar_prevenciones
)
from motor_fisico import (
    calcular_termodinamica_flota_v111, calcular_receptividad_por_headway, 
    precalcular_red_electrica_v111,
    km_at_t, vel_at_km, get_train_state_and_speed, simular_tramo_termodinamico
)
from ui_dashboards import render_gemelo_digital, render_dashboard_energia_v112
from red_electrica import distribuir_energia_sers, calcular_flujo_ac_nodo

st.set_page_config(page_title="Simulador MERVAL V131", layout="wide", page_icon="🗺️")

# =============================================================================
# FUNCIONES DE SOPORTE PARA CARGA DE ARCHIVOS
# =============================================================================
def leer(files): 
    return [(f.name, f.read()) for f in (files or []) if f]

def leer_github(url):
    try:
        import urllib.request
        url = url.strip()
        if 'github.com' in url and 'raw.githubusercontent' not in url:
            url = url.replace('github.com','raw.githubusercontent.com').replace('/blob/','/')
        nm = url.split('/')[-1]
        with urllib.request.urlopen(url, timeout=15) as r:
            return nm, r.read()
    except Exception as e: return None, str(e)

@st.cache_data(show_spinner="Procesando THDR Estándar…")
def build_thdr_v71(blobs_v1, blobs_v2):
    all_parts, err = [], []
    for blobs, via_default in [(blobs_v1, 1), (blobs_v2, 2)]:
        for nm, data in blobs:
            df, msg = procesar_thdr(data, nm, via_default)
            if not df.empty: all_parts.append(df)
            else: err.append(f"[{nm}]: {msg}")
    
    if len(all_parts) > 0:
        df_master = pd.concat(all_parts, ignore_index=True)
        df1 = df_master[df_master['Via'] == 1].copy()
        df2 = df_master[df_master['Via'] == 2].copy()
        if not df1.empty and not df2.empty:
            df1, df2 = calcular_dwell(df1, df2)
        return df1, df2, err
    return pd.DataFrame(), pd.DataFrame(), err

@st.cache_data(show_spinner="Cargando pasajeros…")
def build_pax_v71(blobs_v1, blobs_v2):
    parts, err = [], []
    for blobs, via_default in [(blobs_v1, 1), (blobs_v2, 2)]:
        for nm, data in blobs:
            try: parts.append(cargar_pax(data, nm, via_default))
            except Exception as e: err.append(f"[{nm}]: {e}")
    if len(parts) > 0: return pd.concat(parts, ignore_index=True), err
    return pd.DataFrame(), err

@st.cache_data(show_spinner="Integrando física y demanda de pasajeros...")
def procesar_planificador_reactivo(df_sint, df_px_filtered, estacion_anio_plan, pct_trac, use_rm, use_pend, use_regen, tipo_regen, pax_promedio_viaje=150, prevenciones=None):
    viajes_completos = []
    perfiles_por_servicio = {}
    perfiles_por_via = {}
    
    try: pax_cols_list = getattr(config, 'PAX_COLS', ['PUE'])
    except: pax_cols_list = ['PUE']
        
    try: flota_dict = getattr(config, 'FLOTA', {})
    except: flota_dict = {}
    
    if not df_px_filtered.empty:
        for via in [1, 2]:
            sub_via = df_px_filtered[df_px_filtered['Via'] == via]
            if not sub_via.empty:
                pd_dict = {c: int(round(sub_via[c].mean())) for c in pax_cols_list if c in sub_via.columns}
                if 'CargaMax' in sub_via.columns:
                    pd_dict['CargaMax_Promedio'] = int(round(sub_via['CargaMax'].mean()))
                perfiles_por_via[via] = pd_dict
                
        if 'Tren_Clean' in df_px_filtered.columns:
            for tren, group in df_px_filtered.groupby('Tren_Clean'):
                if str(tren).strip() == '': continue
                pd_dict = {c: int(round(group[c].mean())) for c in pax_cols_list if c in group.columns}
                if 'CargaMax' in group.columns:
                    pd_dict['CargaMax_Promedio'] = int(round(group['CargaMax'].mean()))
                perfiles_por_servicio[str(tren)] = pd_dict

    for idx, r in df_sint.iterrows():
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
        elif not df_px_filtered.empty:
            sub_v = df_px_filtered[df_px_filtered['Via'] == via_tren].copy()
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

        trc_v, aux_v, reg_v, _, _, t_h = simular_tramo_termodinamico(
            r['tipo_tren'], r['doble'], r['km_orig'], r['km_dest'], r['Via'], 
            pct_trac, use_rm, use_pend, r.get('nodos'), pax_arr_viaje, pax_calculado, 
            None, None, estacion_anio_plan, r['t_ini'], prevenciones=prevenciones
        )
        
        viaje_final = r.to_dict()
        viaje_final['pax_d'] = pax_arr_viaje
        viaje_final['pax_abordo'] = pax_calculado
        viaje_final['t_fin'] = r['t_ini'] + (t_h * 60.0)
        viajes_completos.append(viaje_final)
        
    df_sint_final = pd.DataFrame(viajes_completos)
    df_sint_final['tren_km'] = df_sint_final.apply(calc_tren_km_real_general, axis=1)
    df_sint_final.index = df_sint_final['_id']
    
    if use_regen:
        if "Probabilístico" in tipo_regen:
            dict_regen_sint = calcular_receptividad_por_headway(df_sint_final)
        else:
            dict_regen_sint = precalcular_red_electrica_v111(df_sint_final, pct_trac, use_rm, estacion_anio_plan)
    else:
        dict_regen_sint = {}
        
    df_sint_e = calcular_termodinamica_flota_v111(df_sint_final, pct_trac, use_pend, use_rm, use_regen, dict_regen_sint, estacion_anio_plan, prevenciones=prevenciones)
    return df_sint_final, df_sint_e

# =============================================================================
# APLICACIÓN PRINCIPAL (MAIN ORCHESTRATOR)
# =============================================================================
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
        st.subheader("⚙️ Parámetros Físicos del Escenario")
        use_rm      = st.checkbox("🚦 Velocidades RM (Riel Mojado)", value=False, on_change=reset_plan_state)
        pct_trac    = st.slider("⚙️ % Tracción Nominal", 30, 100, 90, 5, on_change=reset_plan_state)
        use_pend    = st.toggle("⛰️ Pendientes Físicas", value=True, on_change=reset_plan_state)
        use_regen   = st.toggle("⚡ Activar Regeneración", value=True, on_change=reset_plan_state)
        tipo_regen  = st.radio("Modelo de Regeneración", ["Físico (Load Flow)", "Probabilístico (Headway)"], on_change=reset_plan_state)
        
        st.divider()
        st.subheader("🌡️ Climatización y Auxiliares")
        mes_sel = st.selectbox("Mes de operación", ["Enero","Febrero","Marzo","Abril","Mayo","Junio","Julio","Agosto","Septiembre","Octubre","Noviembre","Diciembre"], index=3, on_change=reset_plan_state)
        _M = {"Enero":"verano","Febrero":"verano","Marzo":"otoño","Abril":"otoño","Mayo":"otoño","Junio":"invierno","Julio":"invierno","Agosto":"invierno","Septiembre":"primavera","Octubre":"primavera","Noviembre":"primavera","Diciembre":"verano"}
        estacion_anio = _M[mes_sel]
        
        st.divider()
        st.subheader("🔌 Configuración de Red")
        
        try: 
            ser_data_safe = getattr(config, 'SER_DATA', [])
            if not ser_data_safe:
                ser_data_safe = [(3.9, "SER PO"), (11.7, "SER ES"), (25.3, "SER EB"), (29.1, "SER VA")]
        except: 
            ser_data_safe = [(3.9, "SER PO"), (11.7, "SER ES"), (25.3, "SER EB"), (29.1, "SER VA")]
        
        all_ser_names = [s[1] for s in ser_data_safe]
        active_ser_names = st.multiselect("Subestaciones Activas", all_ser_names, default=all_ser_names, on_change=reset_plan_state)
        active_sers = [s for s in ser_data_safe if s[1] in active_ser_names]
        if not active_sers: 
            active_sers = [ser_data_safe[0]]
        
        gap_vias = st.slider("Separación Visual Vías (px)", 120, 350, 200, 10)

    # --- PROCESAMIENTO ETL (EXTRACT, TRANSFORM, LOAD) ---
    def _all_blobs_internal(f_uploader, gh_key): 
        return tuple(leer(f_uploader) + st.session_state.get(gh_key, []))

    b1, b2 = _all_blobs_internal(f_v1, "gh_blobs_v1"), _all_blobs_internal(f_v2, "gh_blobs_v2")
    bx1, bx2 = _all_blobs_internal(f_px1, "gh_blobs_px1"), _all_blobs_internal(f_px2, "gh_blobs_px2")
    b_prev = _all_blobs_internal(f_prev, "gh_blobs_prev")
    
    df1, df2, err_t = build_thdr_v71(b1, b2)
    df_px, err_p = build_pax_v71(bx1, bx2)
    
    prevenciones_list = []
    for nm, data in b_prev:
        try:
            prevs = cargar_prevenciones(data, nm)
            if prevs: prevenciones_list.extend(prevs)
        except: pass
    
    dfs_to_concat = [d for d in [df1, df2] if not d.empty]
    df_all = pd.concat(dfs_to_concat, ignore_index=True).drop_duplicates(subset=['_id']) if dfs_to_concat else pd.DataFrame()

    if not df_all.empty:
        if not df_px.empty:
            if 'Tren_Clean' not in df_px.columns: 
                df_px['Tren_Clean'] = df_px['Tren'].apply(clean_id) if 'Tren' in df_px.columns else ''
            with st.spinner("Sincronizando flujos de pasajeros..."):
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
        df_all['tren_km'] = df_all.apply(calc_tren_km_real_general, axis=1)

    fechas = sorted(list(set([str(d) for d in df_all['Fecha_str'].unique() if str(d) != '2026-01-01' and pd.notna(d)]))) if not df_all.empty else []

    # --- ESTRUCTURA DE TABS (DASHBOARD) ---
    tab_mapa, tab_datos, tab_planificador = st.tabs([
        "🗺️ Gemelo Digital (Histórico)", 
        "👥 Auditoría de Pasajeros", 
        "🔮 Planificador de Escenarios"
    ])
    
    with tab_mapa:
        if df_all.empty: 
            st.warning("⚠️ Sin datos operativos. Por favor, cargue archivos THDR en la barra lateral.")
        else:
            fecha_sel = st.selectbox("Seleccione Fecha de Auditoría", fechas, key="fs_hist")
            df_dia = df_all[df_all['Fecha_str']==fecha_sel].copy()
            
            dict_regen = calcular_receptividad_por_headway(df_dia) if use_regen and "Probabilístico" in tipo_regen else (precalcular_red_electrica_v111(df_dia, pct_trac, use_rm, estacion_anio) if use_regen else {})
            df_dia_e = calcular_termodinamica_flota_v111(df_dia, pct_trac, use_pend, use_rm, use_regen, dict_regen, estacion_anio, prevenciones=prevenciones_list)
            
            render_gemelo_digital(df_dia, df_dia_e, active_sers, fecha_sel, pct_trac, use_rm, use_pend, estacion_anio, "mapa", gap_vias, pax_dia_total=0)
            render_dashboard_energia_v112(df_dia_e, active_sers, fecha_sel, st.session_state.get('sl_ui_mapa', 480.0))

    with tab_datos:
        st.subheader("📋 Auditoría de Carga de Pasajeros")
        if df_px.empty: 
            st.info("ℹ️ No hay datos de pasajeros cargados para auditar.")
        else: 
            st.dataframe(df_px, use_container_width=True)

    with tab_planificador:
        st.subheader("🔮 Proyección de Malla y Capex Operativo")
        st.caption("Modele escenarios hipotéticos inyectando planillas maestras de operación futuras.")
        
        col1, col2 = st.columns([1,2])
        with col1:
            tipo_dia_plan = st.selectbox("Día de Demanda (Perfil)", ["Laboral", "Sábado", "Domingo/Festivo"], key="tdp")
            est_plan = st.selectbox("Estación Térmica (HVAC)", ["verano","otoño","invierno","primavera"], 3, key="esp")
            
        with col2:
            modo_plan = st.radio("Modo", ["Laboratorio (Tramo Único)", "Planilla Maestra (Subir .xlsx)"], horizontal=True)
            
            if modo_plan == "Laboratorio (Tramo Único)":
                try: est_safe = getattr(config, 'ESTACIONES', [])
                except NameError: est_safe = ['Puerto', 'Bellavista', 'Francia', 'Baron', 'Portales', 'Recreo', 'Miramar', 'Viña del Mar', 'Hospital', 'Chorrillos', 'El Salto', 'Valencia', 'Quilpue', 'El Sol', 'El Belloto', 'Las Americas', 'La Concepcion', 'Villa Alemana', 'Sargento Aldea', 'Peñablanca', 'Limache']
                if not est_safe: est_safe = ['Puerto', 'Limache']

                col_s1, col_s2, col_s3, col_s4 = st.columns(4)
                with col_s1: sb_orig = st.selectbox("Estación Origen", est_safe, key="sb_o")
                with col_s2: sb_dest = st.selectbox("Estación Destino", est_safe, index=max(0, len(est_safe)-1), key="sb_d")
                with col_s3: sb_flota = st.selectbox("Tipo de Tren", ["XT-100", "XT-M", "SFE"], key="sb_f")
                with col_s4: sb_pax = st.number_input("Pasajeros a bordo", 0, 1000, 150)
                
                if st.button("⚡ Simular Tramo", use_container_width=True):
                    if sb_orig != sb_dest:
                        idx_o, idx_d = est_safe.index(sb_orig), est_safe.index(sb_dest)
                        try: km_acum_safe = getattr(config, 'KM_ACUM', [])
                        except NameError: km_acum_safe = [0.0, 0.7, 1.4, 2.2, 3.9, 6.0, 7.4, 8.3, 9.2, 10.2, 11.7, 19.1, 21.4, 23.3, 25.3, 26.4, 27.6, 28.5, 29.1, 30.4, 43.13]
                        if not km_acum_safe: km_acum_safe = [0.0, 43.13]
                        
                        km_o, km_d = km_acum_safe[idx_o], km_acum_safe[idx_d]
                        via_sb = 1 if idx_o < idx_d else 2
                        nodos_sb = [(0.0, km_acum_safe[i]) for i in (range(idx_o, idx_d + 1) if via_sb == 1 else range(idx_o, idx_d - 1, -1))]
                        
                        with st.spinner("Calculando termodinámica..."):
                            trc_sb, aux_sb, reg_sb, _, neto_sb, th_sb = simular_tramo_termodinamico(
                                sb_flota, False, km_o, km_d, via_sb, pct_trac, use_rm, use_pend, nodos_sb, {}, sb_pax, None, 
                                None, est_plan, 480.0, prevenciones=prevenciones_list
                            )
                        
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

            else:
                f_pl = st.file_uploader("Subir Planilla Maestra de Inyecciones (.xlsx)", type=['xlsx','csv'])
                if f_pl:
                    df_s, _ = parsear_planilla_maestra(f_pl.read(), f_pl.name)
                    if not df_s.empty and st.button("🚀 Iniciar Simulación de Gemelo Digital", use_container_width=True):
                        df_px_f = df_px[df_px['Fecha_s'].apply(clasificar_dia) == tipo_dia_plan] if not df_px.empty else pd.DataFrame()
                        res, res_e = procesar_planificador_reactivo(df_s, df_px_f, est_plan, pct_trac, use_rm, use_pend, use_regen, tipo_regen, prevenciones=prevenciones_list)
                        st.session_state['plan_ready'], st.session_state['plan_res'], st.session_state['plan_res_e'] = True, res, res_e

        if st.session_state.get('plan_ready', False) and modo_plan != "Laboratorio (Tramo Único)":
            render_gemelo_digital(st.session_state['plan_res'], st.session_state['plan_res_e'], active_sers, f"Simulación: {tipo_dia_plan}", pct_trac, use_rm, use_pend, est_plan, "plan", gap_vias, pax_dia_total=int(st.session_state['plan_res']['pax_abordo'].sum()))

if __name__ == "__main__": 
    main()
