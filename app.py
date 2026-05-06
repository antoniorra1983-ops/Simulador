import streamlit as st
import pandas as pd
import numpy as np
import re
import time
from io import BytesIO
from datetime import datetime, date, timedelta

from config import *
from etl_parser import (
    procesar_thdr, calcular_dwell, cargar_pax, match_pax, 
    get_vacios_dia, get_perfiles_pax, parsear_planilla_maestra, 
    calc_tren_km_real_general, clean_id, mins_to_time_str, clasificar_dia,
    cargar_vacios_efe
)
from motor_fisico import (
    calcular_termodinamica_flota_v111, calcular_receptividad_por_headway, 
    precalcular_red_electrica_v111, procesar_planificador_reactivo
)
from ui_dashboards import render_gemelo_digital, render_dashboard_energia_v112
from red_electrica import distribuir_energia_sers, calcular_flujo_ac_nodo

st.set_page_config(page_title="Simulador MERVAL", layout="wide", page_icon="🗺️")

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

def main():
    def reset_plan_state():
        keys_to_clear = [
            'plan_ready', 'plan_sint_final', 'plan_sint_e',
            'simulacion_plan_lista', 'raw_plan_df'
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

        st.subheader("Planillas THDR")
        f_v1 = st.file_uploader("THDR Vía 1", accept_multiple_files=True, key="t1")
        f_v2 = st.file_uploader("THDR Vía 2", accept_multiple_files=True, key="t2")
        st.divider()
        st.subheader("Carga de Pasajeros")
        f_px1 = st.file_uploader("Pax Vía 1 (Puerto→Limache)", accept_multiple_files=True, key="px1")
        f_px2 = st.file_uploader("Pax Vía 2 (Limache→Puerto)", accept_multiple_files=True, key="px2")
        st.divider()
        
        st.subheader("Reporte Oficial EFE")
        f_vacios_efe = st.file_uploader("Km Vacío Oficial EFE (.csv o .xlsx)", accept_multiple_files=True, key="vac_efe")
        km_limache_manual = st.number_input("➕ Km Vacío Patio Limache", min_value=0.000, value=0.000, step=0.001, format="%.3f", on_change=reset_plan_state, help="Añade kilometraje de Shunting con 3 decimales (metros). Se simulará a 20 km/h en bloques de 1 km.")
        st.divider()
        
        st.subheader("✂️ Gestión de Flota (Split & Merge)")
        n_cortes_v1       = st.slider("Doble→Simple en El Belloto (V1, PU-LI)",0,20,0, on_change=reset_plan_state)
        n_cortes_pu_sa_v1 = st.slider("Doble→Simple en El Belloto (V1, PU-SA)",0,20,0, on_change=reset_plan_state)
        n_acoples_v2      = st.slider("Simple→Doble en El Belloto (V2)",0,20,0, on_change=reset_plan_state)
        n_cortes_sa_v1    = st.slider("Doble→Simple en S. Aldea (V1)",0,20,0, on_change=reset_plan_state)
        n_acoples_sa_v2   = st.slider("Simple→Doble en S. Aldea (V2)",0,20,0, on_change=reset_plan_state)
        st.divider()
        st.subheader("⚙️ Parámetros de Simulación")
        use_rm      = st.checkbox("🚦 Velocidades RM", value=False, on_change=reset_plan_state)
        pct_trac    = st.slider("⚙️ % Tracción Nominal",30,100,90,5, on_change=reset_plan_state)
        use_pend    = st.toggle("⛰️ Pendientes Físicas", value=True, on_change=reset_plan_state)
        use_regen   = st.toggle("⚡ Activar Regeneración", value=True, on_change=reset_plan_state)
        tipo_regen  = st.radio("Modelo de Regeneración", ["Físico (Load Flow / Squeeze Control)", "Probabilístico (Headway Real THDR)"], on_change=reset_plan_state)
        st.divider()
        st.subheader("🌡️ Perfil de Auxiliares Dinámicos")
        mes_sel = st.selectbox("Mes de operación", ["Enero","Febrero","Marzo","Abril","Mayo","Junio","Julio","Agosto","Septiembre","Octubre","Noviembre","Diciembre"], index=3, on_change=reset_plan_state)
        _MES_A_ESTACION = {"Enero":"verano","Febrero":"verano","Marzo":"otoño","Abril":"otoño","Mayo":"otoño","Junio":"invierno","Julio":"invierno","Agosto":"invierno","Septiembre":"primavera","Octubre":"primavera","Noviembre":"primavera","Diciembre":"verano"}
        estacion_anio = _MES_A_ESTACION[mes_sel]
        st.divider()
        st.subheader("🔌 Contingencias Eléctricas")
        all_ser_names = [s[1] for s in SER_DATA]
        active_ser_names = st.multiselect("SERs Activas", all_ser_names, default=all_ser_names, on_change=reset_plan_state)
        active_sers = [s for s in SER_DATA if s[1] in active_ser_names]
        if not active_sers: active_sers = [SER_DATA[0]]
        st.divider()
        gap_vias = st.slider("Separación Visual Vías (px)", 120, 350, 200, 10)

    def _all_blobs_internal(f_uploader, gh_key): 
        return tuple(leer(f_uploader) + st.session_state.get(gh_key, []))

    b1 = _all_blobs_internal(f_v1, "gh_blobs_v1")
    b2 = _all_blobs_internal(f_v2, "gh_blobs_v2")
    bx1 = _all_blobs_internal(f_px1, "gh_blobs_px1")
    bx2 = _all_blobs_internal(f_px2, "gh_blobs_px2")
    b_vac_efe = _all_blobs_internal(f_vacios_efe, "gh_blobs_vac_efe")
    
    df1, df2, err_t = build_thdr_v71(b1, b2)
    df_px, err_p = build_pax_v71(bx1, bx2)
    perfiles_pax = get_perfiles_pax(df_px)

    parts_vac = []
    for nm, data in b_vac_efe:
        df_v = cargar_vacios_efe(data, nm)
        if not df_v.empty: parts_vac.append(df_v)
    df_vacios_real = pd.concat(parts_vac, ignore_index=True) if parts_vac else pd.DataFrame()
    
    dfs_to_concat = [d for d in [df1, df2] if not d.empty]
    df_all = pd.concat(dfs_to_concat, ignore_index=True).drop_duplicates(subset=['_id', 't_ini', 'Via']) if dfs_to_concat else pd.DataFrame()

    with st.sidebar:
        if err_t:
            with st.expander(f"⚠️ {len(err_t)} errores THDR"):
                for e in err_t: st.caption(e)
        if err_p:
            with st.expander(f"⚠️ {len(err_p)} errores Pax"):
                for e in err_p: st.caption(e)
                
        if not df_all.empty:
            if not df_px.empty:
                if 'Tren_Clean' not in df_px.columns:
                    df_px['Tren_Clean'] = df_px['Tren'].apply(clean_id) if 'Tren' in df_px.columns else ''
                
                with st.spinner("Integrando datos reales de pasajeros..."):
                    pax_res = df_all.apply(lambda r: match_pax(r, df_px), axis=1)
                    df_all['pax_d']           = [x[0] for x in pax_res]
                    df_all['pax_abordo']      = [x[1] for x in pax_res]
                    df_all['hora_origen_pax'] = [x[2] for x in pax_res]
                    df_all['nro_thdr_pax']    = [x[3] for x in pax_res]
                    df_all['pax_row_idx']     = [x[4] for x in pax_res]
                    df_all['pax_max']         = df_all['pax_abordo']
            else:
                df_all['pax_d']           = [{}] * len(df_all)
                df_all['pax_max']         = 0
                df_all['pax_abordo']      = 0
                df_all['hora_origen_pax'] = '--:--:--'
                df_all['nro_thdr_pax']    = 'No Detectado'
                df_all['pax_row_idx']     = -1
                
            df_all['maniobra'] = None
            if n_cortes_v1 > 0:
                v1_cands = df_all[(df_all['Via'] == 1) & (df_all['doble'] == True) & (df_all['km_orig'] < 25.0) & (df_all['km_dest'] > 26.0) & (df_all['maniobra'].isnull())].copy()
                if not v1_cands.empty:
                    v1_cands['dist_valle'] = v1_cands['t_ini'].apply(lambda t: min(abs(t - 600), abs(t - 1230)))
                    corte_ids = v1_cands.sort_values('dist_valle').head(n_cortes_v1)['_id'].values
                    df_all.loc[df_all['_id'].isin(corte_ids), 'maniobra'] = 'CORTE_BTO'
                    
            if n_cortes_pu_sa_v1 > 0:
                v1_pu_sa_cands = df_all[(df_all['Via'] == 1) & (df_all['doble'] == True) & (df_all['km_orig'] < 25.0) & (df_all['km_dest'] >= 28.5) & (df_all['km_dest'] <= 29.5) & (df_all['maniobra'].isnull())].copy()
                if not v1_pu_sa_cands.empty:
                    v1_pu_sa_cands['dist_valle'] = v1_pu_sa_cands['t_ini'].apply(lambda t: min(abs(t - 600), abs(t - 1230)))
                    corte_pu_sa_ids = v1_pu_sa_cands.sort_values('dist_valle').head(n_cortes_pu_sa_v1)['_id'].values
                    df_all.loc[df_all['_id'].isin(corte_pu_sa_ids), 'maniobra'] = 'CORTE_PU_SA_BTO'
                    
            if n_acoples_v2 > 0:
                v2_cands = df_all[(df_all['Via'] == 2) & (df_all['km_orig'] > 26.0) & (df_all['km_dest'] < 25.0) & (df_all['maniobra'].isnull())].copy()
                if not v2_cands.empty:
                    v2_cands['dist_punta'] = v2_cands['t_ini'].apply(lambda t: min(abs(t - 390), abs(t - 1050)))
                    acople_ids = v2_cands.sort_values('dist_punta').head(n_acoples_v2)['_id'].values
                    df_all.loc[df_all['_id'].isin(acople_ids), 'maniobra'] = 'ACOPLE_BTO'

            if n_cortes_sa_v1 > 0:
                v1_sa_cands = df_all[(df_all['Via'] == 1) & (df_all['doble'] == True) & (df_all['km_orig'] < 29.0) & (df_all['km_dest'] > 30.0) & (df_all['maniobra'].isnull())].copy()
                if not v1_sa_cands.empty:
                    v1_sa_cands['dist_valle'] = v1_sa_cands['t_ini'].apply(lambda t: min(abs(t - 600), abs(t - 1230)))
                    corte_sa_ids = v1_sa_cands.sort_values('dist_valle').head(n_cortes_sa_v1)['_id'].values
                    df_all.loc[df_all['_id'].isin(corte_sa_ids), 'maniobra'] = 'CORTE_SA'
                    
            if n_acoples_sa_v2 > 0:
                v2_sa_cands = df_all[(df_all['Via'] == 2) & (df_all['km_orig'] > 30.0) & (df_all['km_dest'] < 29.0) & (df_all['maniobra'].isnull())].copy()
                if not v2_sa_cands.empty:
                    v2_sa_cands['dist_punta'] = v2_sa_cands['t_ini'].apply(lambda t: min(abs(t - 390), abs(t - 1050)))
                    acople_sa_ids = v2_sa_cands.sort_values('dist_punta').head(n_acoples_sa_v2)['_id'].values
                    df_all.loc[df_all['_id'].isin(acople_sa_ids), 'maniobra'] = 'ACOPLE_SA'

            df_all['tren_km'] = df_all.apply(calc_tren_km_real_general, axis=1)
            st.success(f"✅ {len(df_all)} despachos operativos históricos cargados.")

    if not df_all.empty:
        fechas_validas = [str(d) for d in df_all['Fecha_str'].unique() if str(d) != '2026-01-01' and pd.notna(d)]
        fechas = sorted(list(set(fechas_validas))) if fechas_validas else sorted([str(d) for d in df_all['Fecha_str'].unique() if pd.notna(d)])
    else:
        fechas = []

    # =========================================================================
    # ESTRUCTURA DE TABS
    # =========================================================================
    tab_mapa, tab_datos, tab_vacios, tab_planificador = st.tabs(["🗺️ Mapa Operativo Histórico", "📋 Reporte Pasajeros y THDR", "🚉 Maniobras en Vacío", "🔮 Planificador Inteligente"])
    
    with tab_planificador:
        st.subheader("🔮 Planificador Avanzado: Gemelo Digital de Inyecciones (V118)")
        st.markdown("El algoritmo ruteará los trenes de la Planilla Maestra basándose en el N° de Servicio y calculará los tiempos de llegada usando Física Pura.")
        
        col_p1, col_p2 = st.columns([1, 2])
        with col_p1:
            tipo_dia_plan = st.selectbox("📅 Tipo de Día para Perfil de Demanda", ["Laboral", "Sábado", "Domingo/Festivo"], key="td_plan", on_change=reset_plan_state)
            pax_promedio_viaje = {"Laboral": 280, "Sábado": 160, "Domingo/Festivo": 110}[tipo_dia_plan]
            estacion_anio_plan = st.selectbox("🌡️ Estación del Año (HVAC)", ["verano", "otoño", "invierno", "primavera"], index=3, key="est_plan", on_change=reset_plan_state)
            
            df_px_filtered = pd.DataFrame()
            nombre_perfil = f"Estático ({pax_promedio_viaje} pax)"
            
            if not df_px.empty:
                fechas_disp_todas = sorted([str(x) for x in df_px['Fecha_s'].dropna().unique() if str(x).strip() and str(x).lower() not in ["none", "nan", "fecha no detectada"]])
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
                        for c in PAX_COLS + ['CargaMax', 't_ini_p']: 
                            df_px_filtered[c] = pd.to_numeric(df_px_filtered[c], errors='coerce').fillna(0)
                    else: 
                        st.warning(f"⚠️ Selecciona al menos una fecha. Usando perfil estático: {pax_promedio_viaje} pax")
                else: 
                    st.warning(f"⚠️ No hay datos cargados para días tipo '{tipo_dia_plan}'. Usando perfil estático: {pax_promedio_viaje} pax")
            else: 
                st.warning(f"⚠️ Sin datos de pasajeros cargados. Usando perfil estático: {pax_promedio_viaje} pax")
            
        with col_p2:
            modo_plan = st.radio("Fuente de Datos", ["Planilla Maestra (Subir CSV/Excel)", "Matriz Sintética"], horizontal=True, on_change=reset_plan_state)
            archivo_planilla = None
            
            if modo_plan == "Matriz Sintética":
                if 'df_plan' not in st.session_state:
                    st.session_state['df_plan'] = pd.DataFrame([
                        {"Ruta": "PU-LI", "Configuración": "Doble", "Flota": "XT-100", "Cantidad": 40},
                        {"Ruta": "LI-PU", "Configuración": "Doble", "Flota": "XT-100", "Cantidad": 40},
                    ])
                df_plan_edit = st.data_editor(st.session_state['df_plan'], num_rows="dynamic", use_container_width=True)
            else:
                archivo_planilla = st.file_uploader("📂 Sube tu Planilla Maestra (.csv, .xlsx, .xls)", type=['csv', 'xlsx', 'xls'])
                df_plan_edit = pd.DataFrame()
                if archivo_planilla:
                    df_temp, msg = parsear_planilla_maestra(archivo_planilla.getvalue(), archivo_planilla.name)
                    if df_temp.empty: 
                        st.error(f"Error procesando: {msg}")
                    else:
                        with st.expander("🛠️ Asignación Avanzada de Flota (Rolling Stock Rostering)", expanded=True):
                            st.success("✅ Planilla decodificada. Selecciona tu estrategia de asignación:")
                            
                            estrategia_flota = st.radio(
                                "Nivel de Asignación:",
                                ["A: Por Trayecto y Configuración (Macro)", "B: Por N° de Servicio (Operativo)", "C: Por Viaje Individual (Laboratorio)"],
                                horizontal=True, on_change=reset_plan_state
                            )
                            st.session_state['estrategia_flota'] = estrategia_flota
                            
                            if "A:" in estrategia_flota:
                                df_temp['Config_Str'] = df_temp['doble'].map({True: 'Doble', False: 'Simple'})
                                agrupado = df_temp.groupby(['svc_type', 'Config_Str']).size().reset_index(name='Total Viajes')
                                
                                current_keys = set(zip(agrupado['svc_type'], agrupado['Config_Str']))
                                stored_keys = set(zip(
                                    st.session_state.get('flota_map_A', pd.DataFrame()).get('Ruta', []), 
                                    st.session_state.get('flota_map_A', pd.DataFrame()).get('Configuración', [])
                                ))
                                
                                if 'flota_map_A' not in st.session_state or current_keys != stored_keys:
                                    matriz = []
                                    for _, r in agrupado.iterrows():
                                        matriz.append({
                                            "Ruta": r['svc_type'],
                                            "Configuración": r['Config_Str'],
                                            "Total Viajes": r['Total Viajes'],
                                            "XT-100": r['Total Viajes'],
                                            "XT-M": 0,
                                            "SFE": 0
                                        })
                                    st.session_state['flota_map_A'] = pd.DataFrame(matriz)
                                
                                df_flota_edit_a = st.data_editor(st.session_state['flota_map_A'], hide_index=True, use_container_width=True)
                                
                                if not df_flota_edit_a[df_flota_edit_a['XT-100'] + df_flota_edit_a['XT-M'] + df_flota_edit_a['SFE'] != df_flota_edit_a['Total Viajes']].empty: 
                                    st.warning("⚠️ Hay trayectos donde la suma asignada no coincide con el Total de Viajes. El remanente será XT-100.")
                                
                                st.session_state['temp_flota_edit_A'] = df_flota_edit_a
                                
                            elif "B:" in estrategia_flota:
                                st.info("Asigna la flota al tren físico. Todos los viajes que haga ese tren usarán la misma tecnología.")
                                srv_unicos = df_temp['num_servicio'].unique()
                                v_por_srv = df_temp['num_servicio'].value_counts().to_dict()
                                
                                if 'flota_map_B' not in st.session_state or set(st.session_state['flota_map_B']['Servicio']) != set(srv_unicos):
                                    matriz_b = [{"Servicio": s, "Total Viajes": v_por_srv[s], "Flota Asignada": "XT-100"} for s in srv_unicos]
                                    st.session_state['flota_map_B'] = pd.DataFrame(matriz_b)
                                
                                df_flota_edit_b = st.data_editor(
                                    st.session_state['flota_map_B'], 
                                    column_config={"Flota Asignada": st.column_config.SelectboxColumn("Flota Asignada", options=["XT-100", "XT-M", "SFE"], required=True)},
                                    hide_index=True, use_container_width=True
                                )
                                st.session_state['temp_flota_edit_B'] = df_flota_edit_b
                                
                            elif "C:" in estrategia_flota:
                                st.warning("⚠️ Cuidado: Cambiar la flota por viaje rompe la continuidad termodinámica de los motores en la vida real.")
                                if 'flota_map_C' not in st.session_state or set(st.session_state['flota_map_C']['ID Viaje']) != set(df_temp['_id']):
                                    matriz_c = [{"ID Viaje": r['_id'], "Hora Inicio": mins_to_time_str(r['t_ini']), "Servicio": r['num_servicio'], "Ruta": r['svc_type'], "Flota Asignada": "XT-100"} for _, r in df_temp.sort_values('t_ini').iterrows()]
                                    st.session_state['flota_map_C'] = pd.DataFrame(matriz_c)
                                
                                df_flota_edit_c = st.data_editor(
                                    st.session_state['flota_map_C'], 
                                    column_config={"Flota Asignada": st.column_config.SelectboxColumn("Flota Asignada", options=["XT-100", "XT-M", "SFE"], required=True)},
                                    hide_index=True, use_container_width=True
                                )
                                st.session_state['temp_flota_edit_C'] = df_flota_edit_c

                            st.session_state['temp_df_plan'] = df_temp
            
        if st.button("🚀 Ejecutar Gemelo Digital del Planificador", use_container_width=True, type="primary", key="btn_plan_full"):
            with st.spinner("Decodificando Planilla e inyectando al Motor Cinemático Termodinámico..."):
                if modo_plan == "Matriz Sintética":
                    df_sintetico_list = []
                    RUTAS_PLAN = {"PU-LI": (0, 20, 1), "LI-PU": (20, 0, 2), "PU-SA": (0, 18, 1), "SA-PU": (18, 0, 2), "PU-BTO": (0, 14, 1), "BTO-PU": (14, 0, 2)}
                    for idx, row in df_plan_edit.iterrows():
                        ruta = row['Ruta']; flota = row['Flota']; es_doble = row['Configuración'] == "Doble"; cant = row['Cantidad']
                        if cant <= 0 or ruta not in RUTAS_PLAN: continue
                        idx_ini, idx_fin, via = RUTAS_PLAN[ruta]
                        km_ini = KM_ACUM[idx_ini]; km_fin = KM_ACUM[idx_fin]
                        est_idxs = range(idx_ini, idx_fin + 1) if via == 1 else range(idx_ini, idx_fin - 1, -1)
                        nodos_sint = [(0.0, KM_ACUM[i]) for i in est_idxs]
                        interval_mins = (1350 - 360) / cant if cant > 1 else 0
                        
                        for i in range(int(cant)):
                            t_ini_sint = 360 + i * interval_mins
                            df_sintetico_list.append({
                                '_id': f"SINT_{ruta}_{i}", 't_ini': t_ini_sint, 'Via': via,
                                'km_orig': km_ini, 'km_dest': km_fin, 'nodos': nodos_sint,
                                'tipo_tren': flota, 'doble': es_doble, 'num_servicio': f"VIRT_{i}",
                                'maniobra': None, 'svc_type': ruta
                            })
                    df_sint = pd.DataFrame(df_sintetico_list)
                else:
                    if archivo_planilla is None or 'temp_df_plan' not in st.session_state:
                        st.warning("Debes subir y procesar la Planilla de Operación primero.")
                        st.stop()
                        
                    df_sint = st.session_state['temp_df_plan'].copy().sort_values('t_ini')
                    estrategia = st.session_state.get('estrategia_flota', "A:")
                    
                    if "A:" in estrategia:
                        asignaciones = {}
                        for _, r in st.session_state['temp_flota_edit_A'].iterrows():
                            key = (r['Ruta'], r['Configuración'] == 'Doble')
                            asignaciones[key] = ['XT-100']*int(r.get('XT-100', 0)) + ['XT-M']*int(r.get('XT-M', 0)) + ['SFE']*int(r.get('SFE', 0))
                            
                        def asignar_tren_a(row):
                            key = (row['svc_type'], row['doble'])
                            if key in asignaciones and len(asignaciones[key]) > 0:
                                return asignaciones[key].pop(0)
                            return 'XT-100'
                        df_sint['tipo_tren'] = df_sint.apply(asignar_tren_a, axis=1)
                        
                    elif "B:" in estrategia:
                        dict_flota_b = dict(zip(st.session_state['temp_flota_edit_B']['Servicio'], st.session_state['temp_flota_edit_B']['Flota Asignada']))
                        df_sint['tipo_tren'] = df_sint['num_servicio'].map(dict_flota_b).fillna('XT-100')
                        
                    elif "C:" in estrategia:
                        dict_flota_c = dict(zip(st.session_state['temp_flota_edit_C']['ID Viaje'], st.session_state['temp_flota_edit_C']['Flota Asignada']))
                        df_sint['tipo_tren'] = df_sint['_id'].map(dict_flota_c).fillna('XT-100')

                if df_sint.empty:
                    st.warning("No hay viajes para simular.")
                    st.stop()

                df_sint_final, df_sint_e = procesar_planificador_reactivo(df_sint, df_px_filtered, estacion_anio_plan, pct_trac, use_rm, use_pend, use_regen, tipo_regen, pax_promedio_viaje)
                
                st.session_state['plan_ready'] = True
                st.session_state['plan_sint_final'] = df_sint_final
                st.session_state['plan_sint_e'] = df_sint_e

        if st.session_state.get('plan_ready', False):
            st.divider()
            st.success("✅ Malla Operativa Físicamente Validada y Calculada con Perfiles Dinámicos de Masa")
            
            df_final_mem = st.session_state['plan_sint_final']
            df_e_mem = st.session_state['plan_sint_e']
            
            st.markdown(f"<div style='text-align:center; padding:10px; background-color:#E8F5E9; color:#2E7D32; border-radius:8px; border:1px solid #C8E6C9; margin-bottom:10px;'><b>Estrategia de Flota Activa:</b> {st.session_state.get('estrategia_flota', 'A: Por Trayecto (Macro)')}</div>", unsafe_allow_html=True)

            st.markdown("### 📋 THDR Sintético Detallado (Malla Operativa WTT)")
            st.caption("Estas tablas son el equivalente matemático al Working Timetable de EFE. Los tiempos de Llegada y Salida por estación son calculados considerando fricción, masa y límites eléctricos. **Las tablas están separadas direccionalmente para uso en CTC.**")
            
            df_sint_show = df_final_mem.copy()
            df_sint_show['Hora_Salida'] = df_sint_show['t_ini'].apply(mins_to_time_str)
            df_sint_show['Hora_Llegada'] = df_sint_show['t_fin'].apply(mins_to_time_str)
            df_sint_show['TDV (min)'] = (df_sint_show['t_fin'] - df_sint_show['t_ini']).round(1)
            df_sint_show['Configuración'] = df_sint_show['doble'].apply(lambda x: 'Doble' if x else 'Simple')
            
            for idx, row in df_sint_show.iterrows():
                nodos_reales = row.get('nodos', [])
                km_times = {}
                for t_m, km_n in nodos_reales:
                    idx_est = int(np.argmin([abs(km_n - k) for k in KM_ACUM]))
                    km_est_r = round(KM_ACUM[idx_est], 3)
                    if km_est_r not in km_times: km_times[km_est_r] = []
                    km_times[km_est_r].append(t_m)
                    
                for i_est in range(N_EST):
                    km_est_round = round(KM_ACUM[i_est], 3)
                    nombre_est = PAX_COLS[i_est]
                    if km_est_round in km_times:
                        times = km_times[km_est_round]
                        df_sint_show.at[idx, f"{nombre_est}_Lleg"] = mins_to_time_str(times[0])
                        df_sint_show.at[idx, f"{nombre_est}_Sal"] = mins_to_time_str(times[-1])
                    else:
                        df_sint_show.at[idx, f"{nombre_est}_Lleg"] = "—"
                        df_sint_show.at[idx, f"{nombre_est}_Sal"] = "—"
            
            cols_base_export = ['_id', 'num_servicio', 'svc_type', 'tipo_tren', 'Configuración', 'Hora_Salida', 'Hora_Llegada', 'TDV (min)', 'pax_abordo']
            
            df_v1 = df_sint_show[df_sint_show['Via'] == 1].copy()
            st.markdown("#### 🔵 Vía 1 (Puerto → Limache)")
            if not df_v1.empty:
                cols_v1 = cols_base_export.copy()
                for est in PAX_COLS: cols_v1.extend([f"{est}_Lleg", f"{est}_Sal"])
                cols_v1_exist = [c for c in cols_v1 if c in df_v1.columns]
                
                with st.expander("👀 Ver / Ocultar Malla Operativa Vía 1", expanded=False):
                    st.dataframe(df_v1[cols_v1_exist], use_container_width=True)
                    csv_v1 = df_v1[cols_v1_exist].to_csv(index=False).encode('utf-8')
                    st.download_button("📥 Descargar Vía 1 (CSV)", data=csv_v1, file_name="THDR_Sintetico_V118_V1.csv", mime='text/csv')
            else:
                st.info("No hay servicios planificados en sentido Vía 1.")

            df_v2 = df_sint_show[df_sint_show['Via'] == 2].copy()
            st.markdown("#### 🔴 Vía 2 (Limache → Puerto)")
            if not df_v2.empty:
                cols_v2 = cols_base_export.copy()
                for est in reversed(PAX_COLS): cols_v2.extend([f"{est}_Lleg", f"{est}_Sal"])
                cols_v2_exist = [c for c in cols_v2 if c in df_v2.columns]
                
                with st.expander("👀 Ver / Ocultar Malla Operativa Vía 2", expanded=False):
                    st.dataframe(df_v2[cols_v2_exist], use_container_width=True)
                    csv_v2 = df_v2[cols_v2_exist].to_csv(index=False).encode('utf-8')
                    st.download_button("📥 Descargar Vía 2 (CSV)", data=csv_v2, file_name="THDR_Sintetico_V118_V2.csv", mime='text/csv')
            else:
                st.info("No hay servicios planificados en sentido Vía 2.")
            
            st.divider()

            render_gemelo_digital(df_final_mem, df_e_mem, active_sers, f"Planificador: {nombre_perfil}", pct_trac, use_rm, use_pend, estacion_anio_plan, prefix_key="plan", gap_vias=gap_vias, pax_dia_total=int(df_final_mem['pax_abordo'].sum()), df_vacios_real=df_vacios_real, km_limache_manual=km_limache_manual)

    with tab_mapa:
        if df_all.empty:
            st.warning("⚠️ El Mapa Operativo y Termodinámico requiere la carga de los archivos **THDR Históricos** para funcionar. Por favor, súbelos en la barra lateral.")
        else:
            fecha_sel = st.selectbox("📅 Fecha Operativa (THDR)", fechas, key="fs_hist")
            df_dia = df_all[df_all['Fecha_str']==fecha_sel].copy()
            
            if use_regen:
                if "Probabilístico" in tipo_regen: dict_regen = calcular_receptividad_por_headway(df_dia)
                else: dict_regen = precalcular_red_electrica_v111(df_dia, pct_trac, use_rm, estacion_anio)
            else: dict_regen = {}
                
            df_dia_e = calcular_termodinamica_flota_v111(df_dia, pct_trac, use_pend, use_rm, use_regen, dict_regen, estacion_anio)
            
            df_dia_px_total = df_px[df_px['Fecha_s'] == fecha_sel] if not df_px.empty and 'Fecha_s' in df_px.columns else pd.DataFrame()
            pax_dia_tot = int(pd.to_numeric(df_dia_px_total['CargaMax'], errors='coerce').fillna(0).sum()) if not df_dia_px_total.empty else 0
            
            render_gemelo_digital(df_dia, df_dia_e, active_sers, fecha_sel, pct_trac, use_rm, use_pend, estacion_anio, prefix_key="mapa", gap_vias=gap_vias, pax_dia_total=pax_dia_tot, df_vacios_real=df_vacios_real, km_limache_manual=km_limache_manual)

    with tab_datos:
        st.subheader("📋 Auditoría de Datos: Carga de Pasajeros y Base THDR")
        
        if df_px.empty:
            st.warning("⚠️ No hay datos de pasajeros cargados. Sube la **Carga de Pasajeros** en la barra lateral para generar la auditoría.")
        else:
            st.success(f"✅ Se leyeron {len(df_px)} registros de pasajeros con éxito.")
            fechas_disponibles = sorted([str(x) for x in df_px['Fecha_s'].dropna().unique() if str(x).strip() and str(x).lower() not in ["none", "nan", "fecha no detectada"]])
            
            if fechas_disponibles:
                opciones_filtro = ["Todas las fechas"] + fechas_disponibles
                fecha_sel_pax = st.selectbox("📅 Filtrar por Fecha del Archivo de Pasajeros", opciones_filtro, key="fs_datos_pax_v41")
                
                df_dia_pax = df_px.copy()
                if fecha_sel_pax != "Todas las fechas":
                    df_dia_pax = df_dia_pax[df_dia_pax['Fecha_s'] == fecha_sel_pax]

                if df_dia_pax.empty:
                    st.info("No hay registros para la fecha seleccionada.")
                else:
                    df_dia_pax = df_dia_pax.sort_values(by=['Via', 't_ini_p'])
                    
                    for c in ['Nro_THDR', 'Tren', 'CargaMax']:
                        if c not in df_dia_pax.columns: df_dia_pax[c] = ''
                    
                    df_dia_pax['Hora Origen Formateada'] = df_dia_pax['t_ini_p'].apply(mins_to_time_str)
                    
                    base_cols = ['Fecha_s', 'Nro_THDR', 'Tren', 'Hora Origen Formateada', 'CargaMax']
                    renames = {'Fecha_s': 'Fecha', 'Nro_THDR': 'N° THDR Pax', 'Tren': 'Servicio', 'Hora Origen Formateada': 'Hora Origen', 'CargaMax': 'Total a Bordo'}
                    
                    for c in PAX_COLS:
                        if c not in df_dia_pax.columns: df_dia_pax[c] = 0
                        else: df_dia_pax[c] = pd.to_numeric(df_dia_pax[c], errors='coerce').fillna(0).astype(int)

                    total_v1 = df_dia_pax[df_dia_pax['Via'] == 1]['CargaMax'].sum() if 'CargaMax' in df_dia_pax.columns else 0
                    total_v2 = df_dia_pax[df_dia_pax['Via'] == 2]['CargaMax'].sum() if 'CargaMax' in df_dia_pax.columns else 0
                    total_ambos = total_v1 + total_v2

                    st.markdown("### 📊 Resumen de Pasajeros (Total a Bordo)")
                    cc1, cc2, cc3 = st.columns(3)
                    cc1.metric("Total Pasajeros V1", f"{int(total_v1):,}")
                    cc2.metric("Total Pasajeros V2", f"{int(total_v2):,}")
                    cc3.metric("Suma Total Ambas Vías", f"{int(total_ambos):,}")
                    st.divider()

                    st.subheader("🔵 Vía 1 (Puerto → Limache)")
                    df_v1 = df_dia_pax[df_dia_pax['Via'] == 1].copy()
                    if not df_v1.empty:
                        v1_cols = base_cols + PAX_COLS
                        df_v1_out = df_v1[v1_cols].rename(columns=renames)
                        st.dataframe(df_v1_out, use_container_width=True)
                    else:
                        st.info("No hay registros de pasajeros para la Vía 1 en esta selección.")

                    st.subheader("🔴 Vía 2 (Limache → Puerto)")
                    df_v2 = df_dia_pax[df_dia_pax['Via'] == 2].copy()
                    if not df_v2.empty:
                        v2_pax_cols_reversed = list(reversed(PAX_COLS))
                        v2_cols = base_cols + v2_pax_cols_reversed
                        df_v2_out = df_v2[v2_cols].rename(columns=renames)
                        st.dataframe(df_v2_out, use_container_width=True)
                    else:
                        st.info("No hay registros de pasajeros para la Vía 2 en esta selección.")

        st.divider()
        st.markdown("### 🚄 Auditoría de Base de Datos THDR (Histórico)")
        st.caption("Esta tabla muestra cómo el sistema analizó el archivo Excel crudo del THDR Histórico subido.")
        if df_all.empty:
            st.info("Sube planillas THDR en la barra lateral para ver la auditoría de la flota operada.")
        else:
            df_hist_show = df_all.copy()
            df_hist_show['Hora_Salida'] = df_hist_show['t_ini'].apply(mins_to_time_str)
            df_hist_show['Hora_Llegada'] = df_hist_show['t_fin'].apply(mins_to_time_str)
            df_hist_show['Configuración'] = df_hist_show['doble'].apply(lambda x: 'Doble' if x else 'Simple')
            
            cols_hist = ['Fecha_str', 'num_servicio', 'motriz_num', 'tipo_tren', 'Configuración', 'Via', 'svc_type', 'Hora_Salida', 'Hora_Llegada', 'pax_abordo']
            cols_hist_exist = [c for c in cols_hist if c in df_hist_show.columns]
            st.dataframe(df_hist_show[cols_hist_exist], use_container_width=True)

    with tab_vacios:
        st.subheader("🚉 Auditoría de Maniobras en Vacío (Carrusel y Reposicionamientos)")
        st.markdown("Esta tabla audita todos los movimientos de los trenes sin pasajeros detectados en el sistema.")
        
        if not df_vacios_real.empty or km_limache_manual > 0:
            if not df_vacios_real.empty:
                st.success("✅ Usando Datos Oficiales EFE para Kilómetros en Vacío (Reemplaza estimación teórica)")
                fechas_disp_vac = sorted([f for f in df_vacios_real['Fecha_str'].unique() if f != '2026-01-01'])
                if not fechas_disp_vac: fechas_disp_vac = sorted(df_vacios_real['Fecha_str'].unique())
                fecha_sel_vacios = st.selectbox("📅 Filtrar por Fecha Operativa", fechas_disp_vac, key="fs_vacios_efe")
                df_dia_vacios = df_vacios_real[df_vacios_real['Fecha_str'] == fecha_sel_vacios].copy()
            else:
                st.info("ℹ️ Mostrando Kilómetros manuales (A la espera del Reporte Oficial EFE para reemplazar teóricos)")
                fecha_sel_vacios = st.selectbox("📅 Filtrar por Fecha Operativa", fechas if fechas else ["2026-01-01"], key="fs_vacios_efe_manual")
                df_dia_vacios = pd.DataFrame()
                
            tabla_vacios = []
            if not df_dia_vacios.empty:
                for _, v in df_dia_vacios.iterrows():
                    distancia_geo = v['dist']
                    if 'COCHERA' in str(v.get('origen_txt', '')).upper() or 'COCHERA' in str(v.get('destino_txt', '')).upper():
                        distancia_geo += 1.0 
                    
                    tabla_vacios.append({
                        "Hora Oficial": "--:--:--" if "Manual" in str(v.get('origen_txt', '')) else mins_to_time_str(v.get('t_asigned', 0)),
                        "Tren (Motriz)": str(v.get('motriz_num', '')),
                        "Estación Origen": str(v.get('origen_txt', '')),
                        "Estación Destino": str(v.get('destino_txt', '')),
                        "Km Vacío": round(distancia_geo, 3), 
                        "Configuración": str(v.get('tipo', 'XT-100'))
                    })
            
            if km_limache_manual > 0:
                tabla_vacios.append({
                    "Hora Oficial": "00:00:00 (Diario)",
                    "Tren (Motriz)": "Shunting",
                    "Estación Origen": "Patio Limache (Manual)",
                    "Estación Destino": "Patio Limache (Manual)",
                    "Km Vacío": round(km_limache_manual, 3), 
                    "Configuración": "XT-100"
                })
                    
            if not tabla_vacios:
                st.info("No hay maniobras en vacío para esta fecha en el reporte oficial.")
            else:
                df_vacios_out = pd.DataFrame(tabla_vacios).sort_values("Hora Oficial").reset_index(drop=True)
                total_km_v = df_vacios_out["Km Vacío"].sum()
                total_mov_v = len(df_vacios_out)
                
                cc1, cc2 = st.columns(2)
                cc1.metric("Total Movimientos Oficiales", total_mov_v)
                cc2.metric("Kilometraje Físico Computado", f"{total_km_v:.3f} km")
                st.divider()
                st.dataframe(df_vacios_out, use_container_width=True)
                
                csv_v = df_vacios_out.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Descargar Registro de Maniobras Oficial",
                    data=csv_v,
                    file_name=f'Maniobras_Vacio_Oficial_MERVAL_{fecha_sel_vacios}.csv',
                    mime='text/csv'
                )

        if df_vacios_real.empty and km_limache_manual == 0:
            st.markdown("---")
            st.markdown("#### 📐 Estimación Teórica (Alternativa)")
            if df_all.empty:
                st.warning("⚠️ No hay archivos THDR cargados para estimar maniobras en vacío teóricas.")
            else:
                fecha_sel_vacios_teo = st.selectbox("📅 Filtrar por Fecha Teórica", fechas, key="fs_vacios_teo")
                df_dia_vacios_teo = df_all[df_all['Fecha_str'] == fecha_sel_vacios_teo].copy()
                vacios_list = get_vacios_dia(df_dia_vacios_teo)
                
                for idx, row in df_dia_vacios_teo[df_dia_vacios_teo['maniobra'].notnull()].iterrows():
                    man = row['maniobra']
                    t_arr_bto = row['t_ini'] + 40.0 if row['Via'] == 1 else row['t_ini'] + 20.0
                    t_arr_sa = row['t_ini'] + 47.0 if row['Via'] == 1 else row['t_ini'] + 13.0
                    dist_sa_eb = abs(KM_ACUM[18] - KM_ACUM[14])
                    
                    if man == 'CORTE_BTO' or man == 'CORTE_PU_SA_BTO':
                        vacios_list.append({'t_asigned': t_arr_bto, 'tipo': row['tipo_tren'], 'doble': False, 'cochera': True, 'dist': 2.0, 'motriz_num': f"{row.get('motriz_num', '')}-B", 'origen_txt': 'El Belloto', 'destino_txt': 'Taller EB', 'km_orig': KM_ACUM[14], 'km_dest': KM_ACUM[14]})
                    elif man == 'ACOPLE_BTO':
                        vacios_list.append({'t_asigned': t_arr_bto - 5.0, 'tipo': row['tipo_tren'], 'doble': False, 'cochera': True, 'dist': 2.0, 'motriz_num': f"{row.get('motriz_num', '')}-B", 'origen_txt': 'Taller EB', 'destino_txt': 'El Belloto', 'km_orig': KM_ACUM[14], 'km_dest': KM_ACUM[14]})
                    elif man == 'CORTE_SA':
                        vacios_list.append({'t_asigned': t_arr_sa, 'tipo': row['tipo_tren'], 'doble': False, 'cochera': True, 'dist': dist_sa_eb + 2.0, 'motriz_num': f"{row.get('motriz_num', '')}-B", 'origen_txt': 'Sargento Aldea', 'destino_txt': 'Taller EB', 'km_orig': KM_ACUM[18], 'km_dest': KM_ACUM[14]})
                    elif man == 'ACOPLE_SA':
                        vacios_list.append({'t_asigned': t_arr_sa - 20.0, 'tipo': row['tipo_tren'], 'doble': False, 'cochera': True, 'dist': dist_sa_eb + 2.0, 'motriz_num': f"{row.get('motriz_num', '')}-B", 'origen_txt': 'Taller EB', 'destino_txt': 'Sargento Aldea', 'km_orig': KM_ACUM[14], 'km_dest': KM_ACUM[18]})

                tabla_vacios_teo = []
                for v in vacios_list:
                    factor_flota = 2 if v.get('doble', False) else 1
                    distancia_geo = v.get('dist', 0)
                    tren_km_equivalente = distancia_geo * factor_flota
                    
                    tabla_vacios_teo.append({
                        "Hora Estimada": mins_to_time_str(v['t_asigned']),
                        "Tren (Motriz)": str(v.get('motriz_num', '')),
                        "Estación Origen": v.get('origen_txt', 'Desconocido'),
                        "Estación Destino": v.get('destino_txt', 'Desconocido'),
                        "Km Vacío": round(tren_km_equivalente, 3), 
                        "Tipo Maniobra": "Ingreso/Salida Cochera" if v.get('cochera') else "Reposicionamiento",
                        "Configuración": f"{v.get('tipo', 'XT-100')} {'(Doble)' if v.get('doble') else '(Simple)'}"
                    })
                
                if not tabla_vacios_teo:
                    st.info("No se detectaron maniobras en vacío teóricas para la fecha seleccionada.")
                else:
                    df_vacios_out_teo = pd.DataFrame(tabla_vacios_teo).sort_values("Hora Estimada").reset_index(drop=True)
                    total_km_v_teo = df_vacios_out_teo["Km Vacío"].sum()
                    total_mov_v_teo = len(df_vacios_out_teo)
                    
                    cc1, cc2 = st.columns(2)
                    cc1.metric("Total Movimientos en Vacío", total_mov_v_teo)
                    cc2.metric("Kilometraje Total en Vacío (Tren-km)", f"{total_km_v_teo:.3f} km")
                    st.divider()
                    st.dataframe(df_vacios_out_teo, use_container_width=True)

if __name__ == "__main__": 
    main()
