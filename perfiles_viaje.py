# =============================================================================
# perfiles_viaje.py
# Perfiles de un viaje simulado para el Simulador MERVAL:
#   1) Velocidad (km/h)   2) Altura — rasante (m)   3) Tracción (kN + kW)
#
# Consume datos_sim['perfil'] que devuelve simular_tramo_termodinamico.
# Cada punto del perfil es la tupla:
#   (t_min, km, v_kmh, estado, p_regen_kw, f_real_kN, P_trac_kW)
# Los dos últimos campos (f_real_kN, P_trac_kW) los agrega el motor parcheado;
# si no estuvieran (motor antiguo), la tracción se muestra vacía sin romper.
# =============================================================================
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Estaciones y PK (idéntico a etl_parser.KM_ACUM_SAFE)
_EST = ['Puerto','Bellavista','Francia','Baron','Portales','Recreo','Miramar',
        'Vina del Mar','Hospital','Chorrillos','El Salto','Valencia','Quilpue',
        'El Sol','El Belloto','Las Americas','La Concepcion','Villa Alemana',
        'Sargento Aldea','Penablanca','Limache']
_KM  = [0.0,0.7,1.4,2.2,3.9,6.0,7.4,8.3,9.2,10.2,11.7,19.1,21.4,23.3,25.3,26.4,
        27.6,28.5,29.1,30.4,43.13]

_COTA_CACHE = None  # cota geográfica (m) por metro de PK, calculada 1 vez


def _cota_geografica():
    """Cota (m) en convención geográfica (Puerto bajo → Limache alto),
    integrando la pendiente V1 que usa el motor (rasante de ingeniería o,
    si no está, el fallback Google Earth). Base Puerto ≈ 3.8 m."""
    global _COTA_CACHE
    if _COTA_CACHE is not None:
        return _COTA_CACHE
    try:
        import motor_fisico as M
        pend = np.asarray(M._PEND_ARRAY_V1[:43500], dtype=float)  # ‰ geográfico
    except Exception:
        pend = np.zeros(43500)
    _COTA_CACHE = 3.8 + np.cumsum(pend / 1000.0)
    return _COTA_CACHE


def _nombre_estacion(km):
    """Nombre de la estación más cercana a un PK (para etiquetas)."""
    i = int(np.argmin([abs(km - k) for k in _KM]))
    return _EST[i]


def _hhmm(t_min):
    try:
        t = int(round(float(t_min)))
        return f"{(t // 60) % 24:02d}:{t % 60:02d}"
    except Exception:
        return "--:--"


def construir_perfiles(datos_sim):
    """datos_sim → DataFrame (t_min, km, v_kmh, estado, f_real_kN, P_trac_kW,
    p_regen_kW, P_neta_kW). P_neta = tracción (+) − regeneración (−)."""
    perfil = datos_sim.get('perfil', []) if isinstance(datos_sim, dict) else []
    rows = []
    for p in perfil:
        p_regen = p[4] if len(p) > 4 else 0.0
        f_real = p[5] if len(p) > 5 else 0.0
        P_trac = p[6] if len(p) > 6 else 0.0
        rows.append((p[0], p[1], p[2], p[3], f_real, P_trac, p_regen))
    df = pd.DataFrame(rows, columns=['t_min', 'km', 'v_kmh', 'estado',
                                     'f_real_kN', 'P_trac_kW', 'p_regen_kW'])
    # Potencia eléctrica neta: + cuando consume (tracción), − cuando regenera (frena)
    df['P_neta_kW'] = df['P_trac_kW'] - df['p_regen_kW']
    return df


def figura_perfiles(datos_sim, titulo="Perfiles del viaje"):
    """Construye la figura Plotly de 3 paneles (velocidad, altura, tracción)
    alineados por PK. Devuelve None si el perfil está vacío."""
    df = construir_perfiles(datos_sim)
    if df.empty:
        return None

    cota = _cota_geografica()
    pk_el = np.arange(len(cota)) / 1000.0
    km_min, km_max = float(df['km'].min()), float(df['km'].max())
    pad = max(0.5, (km_max - km_min) * 0.02)
    x_lo, x_hi = max(0.0, km_min - pad), min(43.2, km_max + pad)

    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.07,
        specs=[[{}], [{}], [{"secondary_y": True}]],
        subplot_titles=("Velocidad (km/h)",
                        "Altura — rasante (m)",
                        "Tracción — esfuerzo en rueda (kN) y potencia (kW)"))

    # 1) VELOCIDAD
    fig.add_trace(go.Scatter(x=df['km'], y=df['v_kmh'], mode='lines',
                             name='Velocidad', line=dict(color='#1b6ca8', width=1.6),
                             hovertemplate='PK %{x:.2f} km<br>%{y:.0f} km/h<extra></extra>'),
                  row=1, col=1)

    # 2) ALTURA (rasante geográfica)
    fig.add_trace(go.Scatter(x=pk_el, y=cota, mode='lines', name='Cota rasante',
                             line=dict(color='#2b2d42', width=1.6),
                             fill='tozeroy', fillcolor='rgba(141,153,174,0.30)',
                             hovertemplate='PK %{x:.2f} km<br>%{y:.0f} m<extra></extra>'),
                  row=2, col=1)

    # 3) TRACCIÓN: esfuerzo (kN) + potencia (kW) en eje secundario
    f = df['f_real_kN']
    fig.add_trace(go.Scatter(x=df['km'], y=f.clip(lower=0), mode='lines',
                             name='Tracción', line=dict(color='#2a9d8f', width=0.8),
                             fill='tozeroy', fillcolor='rgba(42,157,143,0.55)',
                             hovertemplate='PK %{x:.2f} km<br>+%{y:.0f} kN<extra></extra>'),
                  row=3, col=1)
    fig.add_trace(go.Scatter(x=df['km'], y=f.clip(upper=0), mode='lines',
                             name='Freno', line=dict(color='#e76f51', width=0.8),
                             fill='tozeroy', fillcolor='rgba(231,111,81,0.55)',
                             hovertemplate='PK %{x:.2f} km<br>%{y:.0f} kN<extra></extra>'),
                  row=3, col=1)
    fig.add_trace(go.Scatter(x=df['km'], y=df['P_neta_kW'], mode='lines',
                             name='Potencia (+consumo / −regen)',
                             line=dict(color='#264653', width=1.0, dash='dot'),
                             opacity=0.7,
                             hovertemplate='PK %{x:.2f} km<br>%{y:.0f} kW<extra></extra>'),
                  row=3, col=1, secondary_y=True)

    # Estaciones como líneas verticales en los 3 paneles (robusto ante eje secundario)
    for k in _KM:
        if x_lo <= k <= x_hi:
            for r in (1, 2, 3):
                try:
                    if r == 3:
                        fig.add_vline(x=k, line_width=0.4,
                                      line_color='rgba(120,120,120,0.35)',
                                      row=r, col=1, secondary_y=False)
                    else:
                        fig.add_vline(x=k, line_width=0.4,
                                      line_color='rgba(120,120,120,0.35)',
                                      row=r, col=1)
                except Exception:
                    pass

    fig.update_layout(
        title=dict(text=titulo, font=dict(size=15)),
        height=760, hovermode='x unified',
        margin=dict(l=60, r=60, t=70, b=40),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, x=0),
        plot_bgcolor='white')
    fig.update_xaxes(range=[x_lo, x_hi], showgrid=True, gridcolor='rgba(0,0,0,0.06)')
    fig.update_yaxes(showgrid=True, gridcolor='rgba(0,0,0,0.06)')
    fig.update_xaxes(title_text="PK (km)  ·  Puerto = 0 → Limache = 43.1", row=3, col=1)
    fig.update_yaxes(range=[0, 130], row=1, col=1)
    # Ejes del panel de tracción: escalados a los datos de ESTE tren (no fijos),
    # si no, trenes potentes (XT-M, SFE) se recortan y parecen iguales.
    f_axis = max(50.0, float(df['f_real_kN'].abs().max()) * 1.12)
    p_hi = max(100.0, float(df['P_neta_kW'].max()) * 1.10)
    p_lo = min(0.0, float(df['P_neta_kW'].min()) * 1.10)
    fig.update_yaxes(title_text="Esfuerzo (kN)", row=3, col=1, secondary_y=False,
                     range=[-f_axis, f_axis])
    fig.update_yaxes(title_text="Potencia (kW): +consumo / −regen", row=3, col=1,
                     secondary_y=True, range=[p_lo, p_hi], showgrid=False)
    # línea de cero en el panel de tracción para separar tracción/freno y consumo/regen
    fig.add_hline(y=0, line_width=0.6, line_color='rgba(0,0,0,0.45)', row=3, col=1,
                  secondary_y=False)
    return fig


def render_perfiles_viaje(df_dia_e, prefix_key="perf"):
    """Componente Streamlit: selector de viaje (servicio · O→D · hora) + 3 perfiles.
    Llamar dentro del simulador, pasando el df de viajes que ya trae 'datos_sim'."""
    import streamlit as st

    st.subheader("📈 Perfiles del viaje · velocidad · altura · tracción")

    if df_dia_e is None or len(df_dia_e) == 0 or 'datos_sim' not in df_dia_e.columns:
        st.info("Ejecuta la simulación en **modo físico** para ver los perfiles del viaje.")
        return

    cand = df_dia_e[df_dia_e['datos_sim'].notna()].copy()
    if cand.empty:
        st.info("No hay perfiles físicos disponibles para los viajes de este día.")
        return

    # Etiqueta legible por viaje
    serv_col = next((c for c in ('Servicio', 'Tren', 'tren', '_id') if c in cand.columns), None)

    def _label(idx, r):
        via = int(r['Via']) if 'Via' in r and pd.notna(r.get('Via')) else '?'
        o = _nombre_estacion(r['km_orig']) if 'km_orig' in r and pd.notna(r.get('km_orig')) else '?'
        d = _nombre_estacion(r['km_dest']) if 'km_dest' in r and pd.notna(r.get('km_dest')) else '?'
        hh = _hhmm(r['t_ini']) if 't_ini' in r and pd.notna(r.get('t_ini')) else '--:--'
        serv = f"{r[serv_col]} · " if serv_col and pd.notna(r.get(serv_col)) else ''
        return f"{serv}V{via} · {o} → {d} · {hh}"

    opciones = {_label(idx, r): idx for idx, r in cand.iterrows()}
    etiqueta = st.selectbox("Viaje a inspeccionar", list(opciones.keys()),
                            key=f"{prefix_key}_sel_viaje")
    idx_sel = opciones[etiqueta]
    row = cand.loc[idx_sel]

    fig = figura_perfiles(row['datos_sim'], titulo=etiqueta)
    if fig is None:
        st.warning("El viaje seleccionado no tiene perfil de simulación.")
        return
    st.plotly_chart(fig, use_container_width=True, key=f"{prefix_key}_fig_perfiles")

    # Métricas rápidas del viaje
    dfp = construir_perfiles(row['datos_sim'])
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Duración", f"{dfp['t_min'].max() - dfp['t_min'].min():.1f} min")
    c2.metric("Vel. máx", f"{dfp['v_kmh'].max():.0f} km/h")
    c3.metric("Tracción máx", f"{dfp['f_real_kN'].max():.0f} kN")
    c4.metric("Potencia pico", f"{dfp['P_trac_kW'].max():.0f} kW")
