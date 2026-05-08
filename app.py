import streamlit as st
import pandas as pd
import numpy as np
import time
from io import BytesIO
from datetime import datetime, date, timedelta

# Importación segura de configuración para entorno Cloud
try:
    import config
except ImportError:
    pass

# Importación de módulos internos del sistema MERVAL
from etl_parser import (
    procesar_thdr, calcular_dwell, cargar_pax, match_pax, 
    get_perfiles_pax, parsear_planilla_maestra, 
    calc_tren_km_real_general, clean_id, mins_to_time_str, clasificar_dia,
    cargar_prevenciones, get_vacios_dia
)
from motor_fisico import (
    calcular_termodinamica_flota_v111, calcular_receptividad_por_headway, 
    precalcular_red_electrica_v111,
    km_at_t, vel_at_km, get_train_state_and_speed, simular_tramo_termodinamico
)

# Carga condicional de módulos visuales y eléctricos
try:
    from ui_dashboards import render_gemelo_digital, render_dashboard_energia_v112, draw_diagram
    from red_electrica import distribuir_energia_sers, calcular_flujo_ac_nodo
except ImportError:
    pass

# Configuración de página de Streamlit
st.set_page_config(page_title="Simulador MERVAL V134", layout="wide", page_icon="🗺️")

# =============================================================================
# 1. FUNCIONES DE SOPORTE PARA CARGA DE ARCHIVOS (PIPELINE ETL)
# =============================================================================
def leer(files): 
    """Lee archivos subidos por el usuario y los convierte en blobs de datos."""
    return [(f.name, f.read()) for f in (files or []) if f]

def leer_github(url):
    """Descarga archivos desde repositorios GitHub públicos."""
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
    """Procesa los reportes THDR de ambas vías y calcula los tiempos de cabecera."""
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
    """Carga y consolida las planillas de carga de pasajeros."""
    parts, err = [], []
    for blobs, via_default in [(blobs_v1, 1), (blobs_v2, 2)]:
        for nm, data in blobs:
            try: parts.append(cargar_pax(data, nm, via_default))
            except Exception as e: err.append(f"[{nm}]: {e}")
    if len(parts) > 0: return pd.concat(parts, ignore_index=True), err
    return pd.DataFrame(), err

# =============================================================================
# 2. MOTOR DEL PLANIFICADOR DE ESCENARIOS (PROYECCIÓN)
# =============================================================================
@st.cache_data(show_spinner="Integrando física y demanda de pasajeros...")
def procesar_planificador_reactivo(df_sint, df_px_filtered, estacion_anio_plan, pct_trac, use_rm, use_pend, use_regen, tipo_regen, pax_promedio_viaje=150, prevenciones=None):
    """Ejecuta la simulación termodinámica completa para una malla de horarios proyectada."""
    viajes_completos = []
    perfiles_por_servicio = {}
    perfiles_por_via = {}
    
    # Extracción segura de constantes desde config.py
    pax_cols_list = getattr(config, 'PAX_COLS', ['PUE'])
    flota_dict = getattr(config, 'FLOTA', {})
    
    if not df_px_filtered.empty:
        # Generar perfiles estadísticos por vía
        for via in [1, 2]:
            sub_via = df_px_filtered[df_px_filtered['Via'] == via]
            if not sub_via.empty:
                pd_dict = {c: int(round(sub_via[c].mean())) for c in pax_cols_list if c in sub_via.columns}
                if 'CargaMax' in sub_via.columns:
                    pd_dict['CargaMax_Promedio'] = int(round(sub_via['CargaMax'].mean()))
                perfiles_por_via[via] = pd_dict
        
        # Generar perfiles específicos por número de servicio histórico
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
        
        # Match inteligente de pasajeros
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
                    pax_calculado = int(round(best_group['CargaMax'].mean()))
                    pax_arr_viaje = {c: int(round(best_group[c].mean())) for c in pax_cols_list if c in best_group.columns}
                else:
                    # Modelación Gaussiana para demanda en valle/punta
                    pax_dict_dinamico = perfiles_por_via.get(via_tren, {})
                    pax_abordo_base = pax_dict_dinamico.get('CargaMax_Promedio', pax_promedio_viaje)
                    f_gauss = 0.2 + 0.8 * np.exp(-0.5 * ((t_ini_tren - 450)/60)**2) + 0.8 * np.exp(-0.5 * ((t_ini_tren - 1080)/90)**2)
                    pax_calculado = int(pax_abordo_base * f_gauss * 1.5)
                    pax_arr_viaje = {k: int(v * f_gauss * 1.5) for k, v in pax_dict_dinamico.items() if k != 'CargaMax_Promedio'}
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
            None, r.get('maniobra'), estacion_anio_plan, r['t_ini'], es_vacio=False, prevenciones=prevenciones
        )
        
        viaje_final = r.to_dict()
        viaje_final['pax_d'] = pax_arr_viaje
        viaje_final['pax_abordo'] = pax_calculado
        viaje_final['t_fin'] = r['t_ini'] + (t_h * 60.0)
        viajes_completos.append(viaje_final)
        
    df_sint_final = pd.DataFrame(viajes_completos)
    df_sint_final['tren_km'] = df_sint_final.apply(calc_tren_km_real_general, axis=1)
    df_sint_final.index = df_sint_final['_id']
    
    # Cálculo de Receptividad Eléctrica
    if use_regen:
        dict_regen_sint = calcular_receptividad_por_headway(df_sint_final) if "Probabilístico" in tipo_regen else precalcular_red_electrica_v111(df_sint_final, pct_trac, use_rm, estacion_anio_plan)
    else:
        dict_regen_sint = {}
        
    df_sint_e = calcular_termodinamica_flota_v111(df_sint_final, pct_trac, use_pend, use_rm, use_regen, dict_regen_sint, estacion_anio_plan, prevenciones=prevenciones)
    return df_sint_final, df_sint_e

# =============================================================================
# 3. APLICACIÓN PRINCIPAL (MAIN ORCHESTRATOR)
# =============================================================================
def main():
    def reset_plan_state():
        """Limpia el caché de simulación si cambian los parámetros físicos."""
        keys_to_clear = ['plan_ready', 'plan_sint_final', 'plan_sint_e', 'simulacion_plan_lista', 'raw_plan_df']
        for key in keys_to_clear:
            if key in st.session_state: del st.session_state[key]
        st.cache_data.clear()

    with st.sidebar:
        st.header("📂 Archivos Base")
        with st.expander("🔗 Cargar desde GitHub", expanded=False):
            urls_txt = st.text_area("URLs (separadas por línea)", placeholder="https://github.com/...", height=100)
            if st.button("⬇️ Descargar", use_container_width=True): 
                for url in [u.strip() for u in urls_txt.split('\n') if u.strip()]:
                    nm, data = leer_github(url)
                    if nm:
                        lnm = nm.lower()
                        k = "gh_blobs_v1" if "v1" in lnm else "gh_blobs_v2" if "v2" in lnm else "gh_blobs_px1" if "pax" in lnm else "gh_blobs_prev"
                        if k not in st.session_state: st.session_state[k] = []
                        st.session_state[k].append((nm, data))
                st.rerun()

        st.subheader("Carga de Planillas")
        f_v1 = st.file_uploader("THDR Vía 1", accept_multiple_files=True, key="t1")
        f_v2 = st.file_uploader("THDR Vía 2", accept_multiple_files=True, key="t2")
        f_px1 = st.file_uploader("Pasajeros V1", accept_multiple_files=True, key="px1")
        f_px2 = st.file_uploader("Pasajeros V2", accept_multiple_files=True, key="px2")
        f_prev = st.file_uploader("🚧 Prevenciones de Vía", accept_multiple_files=True, key="prev")
        
        st.divider()
        st.subheader("⚙️ Parámetros Físicos")
        use_rm      = st.checkbox("🚦 Velocidades RM", value=False, on_change=reset_plan_state)
        pct_trac    = st.slider("⚙️ % Tracción", 30, 100, 90, 5, on_change=reset_plan_state)
        use_pend    = st.toggle("⛰️ Pendientes", value=True, on_change=reset_plan_state)
        use_regen   = st.toggle("⚡ Regeneración", value=True, on_change=reset_plan_state)
        tipo_regen  = st.radio("Modelo Regen", ["Físico (Load Flow)", "Probabilístico"], on_change=reset_plan_state)
        
        st.divider()
        mes_sel = st.selectbox("Mes", ["Enero","Febrero","Marzo","Abril","Mayo","Junio","Julio","Agosto","Septiembre","Octubre","Noviembre","Diciembre"], index=3, on_change=reset_plan_state)
        _M = {"Enero":"verano","Febrero":"verano","Marzo":"otoño","Abril":"otoño","Mayo":"otoño","Junio":"invierno","Julio":"invierno","Agosto":"invierno","Septiembre":"primavera","Octubre":"primavera","Noviembre":"primavera","Diciembre":"verano"}
        estacion_anio = _M[mes_sel]
        
        active_ser_names = st.multiselect("Subestaciones", ["SER PO", "SER ES", "SER EB", "SER VA"], default=["SER PO", "SER ES", "SER EB", "SER VA"], on_change=reset_plan_state)
        active_sers = [s for s in getattr(config, 'SER_DATA', []) if s[1] in active_ser_names]
        if not active_sers: active_sers = [(3.9, "SER PO")]
        gap_vias = st.slider("Separación Vías (px)", 120, 350, 200, 10)

    # --- PROCESAMIENTO ETL ---
    def _blobs_internal(f_up, key): return tuple(leer(f_up) + st.session_state.get(key, []))
    b1, b2 = _blobs_internal(f_v1, "gh_blobs_v1"), _blobs_internal(f_v2, "gh_blobs_v2")
    bx1, bx2 = _blobs_internal(f_px1, "gh_blobs_px1"), _blobs_internal(f_px2, "gh_blobs_px2")
    b_prev = _blobs_internal(f_prev, "gh_blobs_prev")
    
    prevenciones_list = []
    for nm, data in b_prev:
        p = cargar_prevenciones(data, nm)
        if p: prevenciones_list.extend(p)
    
    df1, df2, _ = build_thdr_v71(b1, b2)
    df_px, _ = build_pax_v71(bx1, bx2)
    
    df_all = pd.concat([d for d in [df1, df2] if not d.empty], ignore_index=True).drop_duplicates(subset=['_id']) if (not df1.empty or not df2.empty) else pd.DataFrame()

    if not df_all.empty:
        if not df_px.empty:
            if 'Tren_Clean' not in df_px.columns: df_px['Tren_Clean'] = df_px['Tren'].apply(clean_id)
            pax_res = df_all.apply(lambda r: match_pax(r, df_px), axis=1)
            df_all['pax_d'], df_all['pax_abordo'], df_all['pax_row_idx'] = [x[0] for x in pax_res], [x[1] for x in pax_res], [x[4] for x in pax_res]
        else:
            df_all['pax_d'], df_all['pax_abordo'], df_all['pax_row_idx'] = [{} for _ in range(len(df_all))], 0, -1
        df_all['maniobra'] = None
        df_all['tren_km'] = df_all.apply(calc_tren_km_real_general, axis=1)

    fechas = sorted(list(set([str(d) for d in df_all['Fecha_str'].unique() if pd.notna(d)]))) if not df_all.empty else []

    tab_mapa, tab_datos, tab_planificador = st.tabs(["🗺️ Gemelo Digital", "👥 Pasajeros", "🔮 Planificador"])
    
    with tab_mapa:
        if df_all.empty: st.warning("⚠️ Sube archivos THDR.")
        else:
            f_sel = st.selectbox("📅 Fecha Operativa", fechas, key="fs_hist")
            df_dia = df_all[df_all['Fecha_str']==f_sel].copy()
            dict_regen = calcular_receptividad_por_headway(df_dia) if use_regen and "Probabilístico" in tipo_regen else (precalcular_red_electrica_v111(df_dia, pct_trac, use_rm, estacion_anio) if use_regen else {})
            df_dia_e = calcular_termodinamica_flota_v111(df_dia, pct_trac, use_pend, use_rm, use_regen, dict_regen, estacion_anio, prevenciones=prevenciones_list)
            render_gemelo_digital(df_dia, df_dia_e, active_sers, f_sel, pct_trac, use_rm, use_pend, estacion_anio, "mapa", gap_vias)
            render_dashboard_energia_v112(df_dia_e, active_sers, f_sel, st.session_state.get('sl_ui_mapa', 480.0))

    with tab_datos:
        if df_px.empty: st.warning("⚠️ Sin datos de pasajeros.")
        else:
            fechas_px = sorted([str(x) for x in df_px['Fecha_s'].dropna().unique() if str(x).strip()])
            f_sel_pax = st.multiselect("📅 Fechas a evaluar", fechas_px, default=[fechas_px[0]] if fechas_px else None)
            if f_sel_pax:
                df_dia_pax = df_px[df_px['Fecha_s'].isin(f_sel_pax)].copy()
                st.dataframe(df_dia_pax, use_container_width=True)

    with tab_planificador:
        st.subheader("🔮 Proyección de Malla")
        cp1, cp2 = st.columns([1, 2])
        with cp1:
            est_p = st.selectbox("🌡️ Estación Térmica", ["verano", "otoño", "invierno", "primavera"], index=3)
            tipo_dia_p = st.selectbox("📅 Tipo de Día", ["Laboral", "Sábado", "Domingo/Festivo"])
            df_px_p = df_px[df_px['Fecha_s'].apply(clasificar_dia) == tipo_dia_p] if not df_px.empty else pd.DataFrame()
            
        with cp2:
            modo_p = st.radio("Fuente", ["Matriz Sintética", "Planilla Maestra"], horizontal=True)
            if modo_p == "Matriz Sintética":
                if 'df_plan' not in st.session_state: st.session_state['df_plan'] = pd.DataFrame([{"Origen": "Puerto", "Destino": "Limache", "Flota": "XT-100", "Configuración": "Doble", "Cantidad": 40}])
                df_plan_edit = st.data_editor(st.session_state['df_plan'], num_rows="dynamic", use_container_width=True)
            else:
                f_pl = st.file_uploader("📂 Subir Planilla (.xlsx)", type=['xlsx','csv'])
                if f_pl:
                    df_t, msg = parsear_planilla_maestra(f_pl.read(), f_pl.name)
                    if not df_t.empty:
                        st.success("✅ Planilla decodificada.")
                        st.session_state['raw_plan_df'] = df_t

        if st.button("🚀 Ejecutar Gemelo Digital", use_container_width=True, type="primary"):
            if modo_p == "Matriz Sintética":
                s_list = []
                for _, row in df_plan_edit.iterrows():
                    for i in range(int(row['Cantidad'])):
                        s_list.append({'_id': f"SINT_{i}", 't_ini': 360 + i * 20, 'Via': 1, 'km_orig': 0.0, 'km_dest': 43.13, 'nodos': [(0,0),(2400,43.13)], 'tipo_tren': row['Flota'], 'doble': row['Configuración']=="Doble"})
                st.session_state['raw_plan_df'] = pd.DataFrame(s_list)
            st.session_state['simulacion_plan_lista'] = True

        if st.session_state.get('simulacion_plan_lista', False):
            df_sf, df_se = procesar_planificador_reactivo(st.session_state['raw_plan_df'], df_px_p, est_p, pct_trac, use_rm, use_pend, use_regen, tipo_regen, prevenciones=prevenciones_list)
            render_gemelo_digital(df_sf, df_se, active_sers, "Simulación", pct_trac, use_rm, use_pend, est_p, "plan", gap_vias)

if __name__ == "__main__": main()
