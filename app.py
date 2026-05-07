import streamlit as st
import pandas as pd
import numpy as np
import time
from io import BytesIO
from datetime import datetime, date, timedelta

# Importaciones de la Arquitectura Modular MERVAL
try:
    from config import *
except ImportError:
    pass
import config

from etl_parser import (
    procesar_thdr, calcular_dwell, cargar_pax, match_pax, 
    get_vacios_dia, get_perfiles_pax, parsear_planilla_maestra, 
    calc_tren_km_real_general, clean_id, mins_to_time_str, clasificar_dia,
    cargar_vacios_efe
)
from motor_fisico import (
    calcular_termodinamica_flota_v111, calcular_receptividad_por_headway, 
    precalcular_red_electrica_v111, procesar_planificador_reactivo,
    km_at_t, vel_at_km, get_train_state_and_speed
)
from ui_dashboards import render_gemelo_digital, render_dashboard_energia_v112
from red_electrica import distribuir_energia_sers, calcular_flujo_ac_nodo

st.set_page_config(page_title="Simulador MERVAL V130", layout="wide", page_icon="🗺️")

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
            gh_via = st.radio("Tipo manual", ["Detección Automática", "THDR V1", "THDR V2", "Pasajeros V1", "Pasajeros V2"], horizontal=False, index=0)
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
                            else:
                                if "vacio" in lnm or "efe" in lnm: k = "gh_blobs_vac_efe"
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
            for lbl, key in [("V1","gh_blobs_v1"),("V2","gh_blobs_v2"),("Pax V1","gh_blobs_px1"),("Pax V2","gh_blobs_px2"),("Vacíos EFE","gh_blobs_vac_efe")]:
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
        f_vacios_efe = st.file_uploader("Km Vacío Oficial EFE (.xlsx/.csv)", accept_multiple_files=True, key="vac_efe")
        
        st.divider()
        st.subheader("📐 Ajustes de Infraestructura")
        km_limache_manual = st.number_input("➕ Km Vacío Patio Limache (Diario)", min_value=0.000, value=0.000, step=0.001, format="%.3f", help="Añade kilometraje de Shunting manual al reporte de vacíos.")
        
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
        
        # Escudo defensivo para SER_DATA
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
    b_vac_efe = _all_blobs_internal(f_vacios_efe, "gh_blobs_vac_efe")
    
    df1, df2, err_t = build_thdr_v71(b1, b2)
    df_px, err_p = build_pax_v71(bx1, bx2)
    
    # Procesar Vacíos Oficiales EFE
    parts_vac = []
    for nm, data in b_vac_efe:
        df_v = cargar_vacios_efe(data, nm)
        if not df_v.empty: parts_vac.append(df_v)
    df_vacios_real = pd.concat(parts_vac, ignore_index=True) if parts_vac else pd.DataFrame()
    
    # Consolidar THDR
    dfs_to_concat = [d for d in [df1, df2] if not d.empty]
    df_all = pd.concat(dfs_to_concat, ignore_index=True).drop_duplicates(subset=['_id']) if dfs_to_concat else pd.DataFrame()

    # Integración de Pasajeros y Tren-km
    if not df_all.empty:
        if not df_px.empty:
            if 'Tren_Clean' not in df_px.columns: 
                df_px['Tren_Clean'] = df_px['Tren'].apply(clean_id) if 'Tren' in df_px.columns else ''
            with st.spinner("Sincronizando flujos de pasajeros..."):
                # 💡 FIX APLICADO: Extracción completa (5 datos) para evitar KeyError 'pax_row_idx'
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
        
        # Inyección de maniobras (Split/Merge)
        n_cortes_v1       = st.sidebar.slider("Doble→Simple en El Belloto (V1, PU-LI)",0,20,0, on_change=reset_plan_state)
        n_cortes_pu_sa_v1 = st.sidebar.slider("Doble→Simple en El Belloto (V1, PU-SA)",0,20,0, on_change=reset_plan_state)
        n_acoples_v2      = st.sidebar.slider("Simple→Doble en El Belloto (V2)",0,20,0, on_change=reset_plan_state)
        n_cortes_sa_v1    = st.sidebar.slider("Doble→Simple en S. Aldea (V1)",0,20,0, on_change=reset_plan_state)
        n_acoples_sa_v2   = st.sidebar.slider("Simple→Doble en S. Aldea (V2)",0,20,0, on_change=reset_plan_state)

        if n_cortes_v1 > 0:
            v1_cands = df_all[(df_all['Via'] == 1) & (df_all['doble'] == True) & (df_all['km_orig'] < 25.0) & (df_all['km_dest'] > 26.0) & (df_all['maniobra'].isnull())].copy()
            if not v1_cands.empty:
                v1_cands['dist_valle'] = v1_cands['t_ini'].apply(lambda t: min(abs(t - 600), abs(t - 1230)))
                df_all.loc[df_all['_id'].isin(v1_cands.sort_values('dist_valle').head(n_cortes_v1)['_id'].values), 'maniobra'] = 'CORTE_BTO'
        if n_cortes_pu_sa_v1 > 0:
            v1_pu_sa_cands = df_all[(df_all['Via'] == 1) & (df_all['doble'] == True) & (df_all['km_orig'] < 25.0) & (df_all['km_dest'] >= 28.5) & (df_all['km_dest'] <= 29.5) & (df_all['maniobra'].isnull())].copy()
            if not v1_pu_sa_cands.empty:
                v1_pu_sa_cands['dist_valle'] = v1_pu_sa_cands['t_ini'].apply(lambda t: min(abs(t - 600), abs(t - 1230)))
                df_all.loc[df_all['_id'].isin(v1_pu_sa_cands.sort_values('dist_valle').head(n_cortes_pu_sa_v1)['_id'].values), 'maniobra'] = 'CORTE_PU_SA_BTO'
        if n_acoples_v2 > 0:
            v2_cands = df_all[(df_all['Via'] == 2) & (df_all['km_orig'] > 26.0) & (df_all['km_dest'] < 25.0) & (df_all['maniobra'].isnull())].copy()
            if not v2_cands.empty:
                v2_cands['dist_punta'] = v2_cands['t_ini'].apply(lambda t: min(abs(t - 390), abs(t - 1050)))
                df_all.loc[df_all['_id'].isin(v2_cands.sort_values('dist_punta').head(n_acoples_v2)['_id'].values), 'maniobra'] = 'ACOPLE_BTO'
        if n_cortes_sa_v1 > 0:
            v1_sa_cands = df_all[(df_all['Via'] == 1) & (df_all['doble'] == True) & (df_all['km_orig'] < 29.0) & (df_all['km_dest'] > 30.0) & (df_all['maniobra'].isnull())].copy()
            if not v1_sa_cands.empty:
                v1_sa_cands['dist_valle'] = v1_sa_cands['t_ini'].apply(lambda t: min(abs(t - 600), abs(t - 1230)))
                df_all.loc[df_all['_id'].isin(v1_sa_cands.sort_values('dist_valle').head(n_cortes_sa_v1)['_id'].values), 'maniobra'] = 'CORTE_SA'
        if n_acoples_sa_v2 > 0:
            v2_sa_cands = df_all[(df_all['Via'] == 2) & (df_all['km_orig'] > 30.0) & (df_all['km_dest'] < 29.0) & (df_all['maniobra'].isnull())].copy()
            if not v2_sa_cands.empty:
                v2_sa_cands['dist_punta'] = v2_sa_cands['t_ini'].apply(lambda t: min(abs(t - 390), abs(t - 1050)))
                df_all.loc[df_all['_id'].isin(v2_sa_cands.sort_values('dist_punta').head(n_acoples_sa_v2)['_id'].values), 'maniobra'] = 'ACOPLE_SA'

        df_all['tren_km'] = df_all.apply(calc_tren_km_real_general, axis=1)

    fechas = sorted(list(set([str(d) for d in df_all['Fecha_str'].unique() if str(d) != '2026-01-01' and pd.notna(d)]))) if not df_all.empty else []

    # --- ESTRUCTURA DE TABS (DASHBOARD) ---
    tab_mapa, tab_datos, tab_vacios, tab_planificador = st.tabs([
        "🗺️ Gemelo Digital (Histórico)", 
        "👥 Auditoría de Pasajeros", 
        "🚉 Kilómetros en Vacío", 
        "🔮 Planificador de Escenarios"
    ])
    
    with tab_mapa:
        if df_all.empty: 
            st.warning("⚠️ Sin datos operativos. Por favor, cargue archivos THDR en la barra lateral.")
        else:
            fecha_sel = st.selectbox("Seleccione Fecha de Auditoría", fechas, key="fs_hist")
            df_dia = df_all[df_all['Fecha_str']==fecha_sel].copy()
            
            # Ejecutar malla de regeneración
            dict_regen = calcular_receptividad_por_headway(df_dia) if use_regen and "Probabilístico" in tipo_regen else (precalcular_red_electrica_v111(df_dia, pct_trac, use_rm, estacion_anio) if use_regen else {})
            
            # Integración Termodinámica
            df_dia_e = calcular_termodinamica_flota_v111(df_dia, pct_trac, use_pend, use_rm, use_regen, dict_regen, estacion_anio)
            
            # Renderizado de UI
            render_gemelo_digital(df_dia, df_dia_e, active_sers, fecha_sel, pct_trac, use_rm, use_pend, estacion_anio, "mapa", gap_vias)
            render_dashboard_energia_v112(df_dia_e, active_sers, fecha_sel, st.session_state.get('sl_ui_mapa', 480.0))

    with tab_datos:
        st.subheader("📋 Auditoría de Carga de Pasajeros")
        if df_px.empty: 
            st.info("ℹ️ No hay datos de pasajeros cargados para auditar.")
        else: 
            st.dataframe(df_px, use_container_width=True)

    with tab_vacios:
        st.subheader("🚉 Monitoreo de Kilómetros Improductivos (Shunting)")
        
        # Mostrar reporte oficial si existe
        if not df_vacios_real.empty:
            st.success("✅ Datos del Reporte Oficial EFE detectados y cargados.")
            fecha_v = st.selectbox("Filtrar Fecha Reporte", sorted(df_vacios_real['Fecha_str'].unique()), key="fs_v")
            st.dataframe(df_vacios_real[df_vacios_real['Fecha_str']==fecha_v], use_container_width=True)
        
        # Siempre mostrar la estimación teórica como base de comparación
        st.markdown("---")
        st.markdown("#### 📐 Estimación Teórica de Maniobras (SSOT)")
        if df_all.empty: 
            st.info("Requiere carga de THDR para proyectar las maniobras de parqueo nocturno.")
        else:
            f_v_t = st.selectbox("Fecha base para estimación teórica", fechas, key="fs_v_t")
            vacios_list = get_vacios_dia(df_all[df_all['Fecha_str']==f_v_t])
            
            try: km_acum_safe = getattr(config, 'KM_ACUM', [])
            except: km_acum_safe = [0.0, 0.7, 1.4, 2.2, 3.9, 6.0, 7.4, 8.3, 9.2, 10.2, 11.7, 19.1, 21.4, 23.3, 25.3, 26.4, 27.6, 28.5, 29.1, 30.4, 43.13]
            
            # 💡 BLINDAJE APLICADO: Inyección de distancia estricta de 0.5 km a 10 km/h para Depot Mode
            for idx, row in df_all[(df_all['Fecha_str']==f_v_t) & (df_all['maniobra'].notnull())].iterrows():
                man = row['maniobra']
                t_arr_bto = row['t_ini'] + 40.0 if row['Via'] == 1 else row['t_ini'] + 20.0
                t_arr_sa = row['t_ini'] + 47.0 if row['Via'] == 1 else row['t_ini'] + 13.0
                km_bto = km_acum_safe[14] if len(km_acum_safe) > 14 else 25.3
                km_sa = km_acum_safe[18] if len(km_acum_safe) > 18 else 29.1
                
                if man in ['CORTE_BTO', 'CORTE_PU_SA_BTO']: 
                    vacios_list.append({'t_asigned': t_arr_bto, 'tipo': row['tipo_tren'], 'doble': False, 'cochera': True, 'dist': 0.5, 'motriz_num': f"{row.get('motriz_num', '')}-B", 'origen_txt': 'El Belloto', 'destino_txt': 'Taller EB', 'km_orig': km_bto, 'km_dest': km_bto})
                elif man == 'ACOPLE_BTO': 
                    vacios_list.append({'t_asigned': t_arr_bto - 5.0, 'tipo': row['tipo_tren'], 'doble': False, 'cochera': True, 'dist': 0.5, 'motriz_num': f"{row.get('motriz_num', '')}-B", 'origen_txt': 'Taller EB', 'destino_txt': 'El Belloto', 'km_orig': km_bto, 'km_dest': km_bto})
                elif man == 'CORTE_SA': 
                    vacios_list.append({'t_asigned': t_arr_sa, 'tipo': row['tipo_tren'], 'doble': False, 'cochera': True, 'dist': 0.5, 'motriz_num': f"{row.get('motriz_num', '')}-B", 'origen_txt': 'Sargento Aldea', 'destino_txt': 'Taller EB', 'km_orig': km_sa, 'km_dest': km_sa})
                elif man == 'ACOPLE_SA': 
                    vacios_list.append({'t_asigned': t_arr_sa - 20.0, 'tipo': row['tipo_tren'], 'doble': False, 'cochera': True, 'dist': 0.5, 'motriz_num': f"{row.get('motriz_num', '')}-B", 'origen_txt': 'Taller EB', 'destino_txt': 'Sargento Aldea', 'km_orig': km_sa, 'km_dest': km_sa})

            if km_limache_manual > 0:
                km_li = km_acum_safe[20] if len(km_acum_safe) > 20 else 43.13
                vacios_list.append({'t_asigned': 0.0, 'tipo': 'Manual', 'motriz_num': 'SH-LIMIT', 'origen_txt': 'P. Limache', 'destino_txt': 'P. Limache', 'dist': km_limache_manual, 'km_orig': km_li, 'km_dest': km_li})
            
            st.dataframe(pd.DataFrame(vacios_list).sort_values('t_asigned'), use_container_width=True)

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
                tipo_mov_lab = st.radio("Modo de Operación", ["Servicio Comercial (Con Paradas)", "Maniobra en Vacío (Pasa de largo a 10 km/h)"], horizontal=True)
                
                try: est_safe = getattr(config, 'ESTACIONES', [])
                except NameError: est_safe = ['Puerto', 'Bellavista', 'Francia', 'Baron', 'Portales', 'Recreo', 'Miramar', 'Viña del Mar', 'Hospital', 'Chorrillos', 'El Salto', 'Valencia', 'Quilpue', 'El Sol', 'El Belloto', 'Las Americas', 'La Concepcion', 'Villa Alemana', 'Sargento Aldea', 'Peñablanca', 'Limache']
                if not est_safe: est_safe = ['Puerto', 'Limache']

                col_s1, col_s2, col_s3, col_s4 = st.columns(4)
                with col_s1: sb_orig = st.selectbox("Estación Origen", est_safe, key="sb_o")
                with col_s2: sb_dest = st.selectbox("Estación Destino", est_safe, index=max(0, len(est_safe)-1), key="sb_d")
                with col_s3: sb_flota = st.selectbox("Tipo de Tren", ["XT-100", "XT-M", "SFE"], key="sb_f")
                with col_s4: sb_pax = 0 if "Vacío" in tipo_mov_lab else st.number_input("Pasajeros a bordo", 0, 1000, 150)
                
                if st.button("⚡ Simular Tramo", use_container_width=True):
                    if sb_orig != sb_dest:
                        idx_o, idx_d = est_safe.index(sb_orig), est_safe.index(sb_dest)
                        try: km_acum_safe = getattr(config, 'KM_ACUM', [])
                        except NameError: km_acum_safe = [0.0, 0.7, 1.4, 2.2, 3.9, 6.0, 7.4, 8.3, 9.2, 10.2, 11.7, 19.1, 21.4, 23.3, 25.3, 26.4, 27.6, 28.5, 29.1, 30.4, 43.13]
                        if not km_acum_safe: km_acum_safe = [0.0, 43.13]
                        
                        km_o, km_d = km_acum_safe[idx_o], km_acum_safe[idx_d]
                        via_sb = 1 if idx_o < idx_d else 2
                        es_vacio_flag = "Vacío" in tipo_mov_lab
                        nodos_sb = [(0.0, km_acum_safe[idx_o]), (0.0, km_acum_safe[idx_d])] if es_vacio_flag else [(0.0, km_acum_safe[i]) for i in (range(idx_o, idx_d + 1) if via_sb == 1 else range(idx_o, idx_d - 1, -1))]
                        
                        with st.spinner("Calculando termodinámica..."):
                            trc_sb, aux_sb, reg_sb, _, neto_sb, th_sb = simular_tramo_termodinamico(
                                sb_flota, False, km_o, km_d, via_sb, pct_trac, use_rm, use_pend, nodos_sb, {}, sb_pax, None, 
                                "MANIOBRA_LAB" if es_vacio_flag else None, est_plan, 480.0, es_vacio_flag
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
                        res, res_e = procesar_planificador_reactivo(df_s, df_px_f, est_plan, pct_trac, use_rm, use_pend, use_regen, tipo_regen)
                        st.session_state['plan_ready'], st.session_state['plan_res'], st.session_state['plan_res_e'] = True, res, res_e

        if st.session_state.get('plan_ready', False) and modo_plan != "Laboratorio (Tramo Único)":
            render_gemelo_digital(st.session_state['plan_res'], st.session_state['plan_res_e'], active_sers, f"Simulación: {tipo_dia_plan}", pct_trac, use_rm, use_pend, est_plan, "plan", gap_vias, pax_dia_total=0)

if __name__ == "__main__": 
    main()
