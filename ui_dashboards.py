import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import time
import json
import plotly.graph_objects as go
from config import *
from etl_parser import mins_to_time_str, get_pax_at_km, get_vacios_dia
from red_electrica import calcular_flujo_ac_nodo, distribuir_potencia_sers_kw, distribuir_energia_sers
from motor_fisico import km_at_t, vel_at_km, get_train_state_and_speed, calcular_aux_dinamico, simular_tramo_termodinamico

# =============================================================================
# MOTOR VISUAL 1: RENDERIZADO ESTÁTICO PYTHON (DOM SVG INYECTADO)
# =============================================================================
def draw_diagram_svg(df_act_plot, ser_accum_plot, seat_accum_plot, hora_str, titulo_extra="", active_sers_list=SER_DATA, gap_vias=200):
    W = 1200
    PADDING_X = 40
    KM_SCALE = (W - 2 * PADDING_X) / KM_TOTAL
    def xkm(km): return PADDING_X + km * KM_SCALE

    Y_44KV = 100    
    Y_SER = 150
    Y_V2 = 200
    Y_V1 = Y_V2 + gap_vias
    H = Y_V1 + 90
    y_mid = (Y_V1 + Y_V2) / 2

    svg = f'''
    <svg width="100%" height="{H}" viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg" style="background-color: white; font-family: sans-serif; border-radius: 8px; border: 1px solid #ddd; display: block; margin-bottom: 5px;">
        <text x="{W/2}" y="35" font-size="15" font-weight="bold" fill="#111" text-anchor="middle">MERVAL - {hora_str} {titulo_extra}  |  🔴 V2 LI→PU   🔵 V1 PU→LI</text>
        
        <line x1="{PADDING_X}" y1="{Y_V2}" x2="{W-PADDING_X}" y2="{Y_V2}" stroke="#c62828" stroke-width="5" />
        <line x1="{PADDING_X}" y1="{Y_V1}" x2="{W-PADDING_X}" y2="{Y_V1}" stroke="#1565c0" stroke-width="5" />
        
        <line x1="{PADDING_X}" y1="{Y_44KV}" x2="{W-PADDING_X}" y2="{Y_44KV}" stroke="#FBC02D" stroke-width="3" stroke-dasharray="10,5" />
        <text x="{W/2}" y="{Y_44KV-10}" font-size="10" font-weight="bold" fill="#FBC02D" text-anchor="middle">Línea AC 44kV</text>
    '''

    for i, (ec, km) in enumerate(zip(EC, KM_ACUM[:N_EST])):
        xp = xkm(km)
        y_ec = y_mid + (15 if i % 2 == 0 else -15)
        svg += f'<line x1="{xp}" y1="{Y_V2-20}" x2="{xp}" y2="{Y_V1+20}" stroke="#bbb" stroke-width="1" stroke-dasharray="2,2" />'
        svg += f'<text x="{xp}" y="{y_ec}" font-size="9" font-weight="bold" fill="#555" text-anchor="middle" dominant-baseline="middle">{ec}</text>'

    seat_x = xkm(SEAT_KM)
    svg += f'''
        <polygon points="{seat_x},{Y_44KV-30} {seat_x-12},{Y_44KV-10} {seat_x+12},{Y_44KV-10}" fill="#FBC02D" stroke="black" stroke-width="1" />
        <text x="{seat_x}" y="{Y_44KV-45}" font-size="10" font-weight="bold" fill="#111" text-anchor="middle">⚡ SEAT EL SOL</text>
        <text x="{seat_x}" y="{Y_44KV-33}" font-size="10" fill="#111" text-anchor="middle">{seat_accum_plot:,.0f} kWh</text>
        <line x1="{seat_x}" y1="{Y_44KV-10}" x2="{seat_x}" y2="{Y_44KV}" stroke="#FBC02D" stroke-width="4" />
    '''

    active_names = [s[1] for s in active_sers_list]
    for skm, nombre_ser in SER_DATA:
        xp = xkm(skm)
        is_active = nombre_ser in active_names
        val = ser_accum_plot.get(nombre_ser, 0.0)
        
        if is_active:
            color, fill, txt_color = "#FBC02D", "#FFF3E0", "#E65100"
            status_lbl = f"{val:,.0f} kWh"
            svg += f'<line x1="{xp}" y1="{Y_SER+15}" x2="{xp}" y2="{Y_V2}" stroke="#E65100" stroke-width="2" />'
            svg += f'<line x1="{xp}" y1="{Y_V2}" x2="{xp}" y2="{Y_V1}" stroke="#1565C0" stroke-width="1" stroke-dasharray="4,4" />'
            dash = ""
        else:
            color, fill, txt_color = "#9E9E9E", "#F5F5F5", "#757575"
            status_lbl = "OFF"
            svg += f'<text x="{xp}" y="{Y_SER-25}" font-size="10" font-weight="bold" fill="red" text-anchor="middle">❌ FALLA</text>'
            dash = 'stroke-dasharray="5,5"'

        svg += f'<line x1="{xp}" y1="{Y_44KV}" x2="{xp}" y2="{Y_SER-15}" stroke="{color}" stroke-width="2" {dash}/>'
        svg += f'<rect x="{xp-30}" y="{Y_SER-15}" width="60" height="30" fill="{fill}" stroke="{color}" stroke-width="2" rx="4" />'
        svg += f'<text x="{xp}" y="{Y_SER-2}" font-size="10" font-weight="bold" fill="{txt_color}" text-anchor="middle">{nombre_ser}</text>'
        svg += f'<text x="{xp}" y="{Y_SER+10}" font-size="9" fill="{txt_color}" text-anchor="middle">{status_lbl}</text>'

    if not df_act_plot.empty:
        COLL_PX = 100
        label_side = {}
        for via_ in [1, 2]:
            sub = df_act_plot[df_act_plot['Via'] == via_].copy()
            if sub.empty: continue
            sub_sorted = sub.sort_values('km_pos')
            indices = list(sub_sorted.index)
            for i, idx in enumerate(indices):
                xp_i = xkm(sub_sorted.loc[idx, 'km_pos'])
                close = False
                if i > 0 and abs(xp_i - xkm(sub_sorted.loc[indices[i-1], 'km_pos'])) < COLL_PX: close = True
                if i < len(indices) - 1 and abs(xp_i - xkm(sub_sorted.loc[indices[i+1], 'km_pos'])) < COLL_PX: close = True
                label_side[idx] = ('up' if i % 2 == 0 else 'down') if close else 'up'

        for idx, row in df_act_plot.iterrows():
            via = row['Via']
            xp = xkm(row['km_pos'])
            y_ln = Y_V2 if via == 2 else Y_V1
            
            is_parked = row.get('is_parked', False)
            if is_parked:
                color = '#4CAF50' 
            else:
                color = '#c62828' if via == 2 else '#1565c0'
            
            doble_tramo = row.get('doble', False)
            man = row.get('maniobra')
            if man in ['CORTE_BTO', 'CORTE_PU_SA_BTO']: doble_tramo = True if row['km_pos'] <= KM_ACUM[14] else False
            elif man == 'ACOPLE_BTO': doble_tramo = False if row['km_pos'] > KM_ACUM[14] else True
            elif man == 'CORTE_SA': doble_tramo = True if row['km_pos'] <= KM_ACUM[18] else False
            elif man == 'ACOPLE_SA': doble_tramo = False if row['km_pos'] > KM_ACUM[18] else True
                
            r_c = 18 if doble_tramo else 11
            serv = str(row.get('num_servicio', ''))
            motriz = str(row.get('motriz_num', ''))
            tipo = str(row.get('tipo_tren', 'XT-100'))
            
            if tipo == 'SFE': xt_lbl = f"SFE [U-{motriz}]" if motriz else "SFE"
            elif tipo == 'XT-M': xt_lbl = f"Modular [U-{motriz}]" if motriz else "Modular"
            else: xt_lbl = f"XT-100 [U-{motriz}]" if motriz else "XT-100"

            kwh_n = float(row.get('kwh_neto', 0))
            pax_v = int(row.get('pax_inst', 0)) 
            sep_r = row.get('sep_next', '—')
            sep_s = f"↔ {sep_r} min" if sep_r != '—' else ''

            side = label_side.get(idx, 'up')
            if via == 2:
                base_dy = -r_c - 16
                if side == 'down': base_dy -= 28 
            else:
                base_dy = r_c + 16
                if side == 'down': base_dy += 28

            safe_tooltip = str(row.get("tooltip", "")).replace("\n", "&#10;").replace("<b>", "").replace("</b>", "")
            
            svg += f'<circle cx="{xp}" cy="{y_ln}" r="{r_c}" fill="{color}" stroke="black" stroke-width="2"><title>{safe_tooltip}</title></circle>'
            
            svg += f'<rect x="{xp-45}" y="{y_ln+base_dy-12}" width="90" height="24" fill="white" fill-opacity="0.85" rx="3" stroke="#ccc" stroke-width="1"/>'
            svg += f'<text x="{xp}" y="{y_ln+base_dy-2}" font-size="10" font-weight="bold" fill="#111" text-anchor="middle">{xt_lbl}</text>'
            svg += f'<text x="{xp}" y="{y_ln+base_dy+9}" font-size="9" font-weight="bold" fill="#111" text-anchor="middle">Serv. {serv}</text>'
            
            svg += f'<text x="{xp - r_c - 6}" y="{y_ln+3}" font-size="10" font-weight="bold" fill="#2E7D32" text-anchor="end">{kwh_n:.0f} kWh</text>'
            if not is_parked:
                svg += f'<text x="{xp + r_c + 6}" y="{y_ln+3}" font-size="10" font-weight="bold" fill="#1565c0" text-anchor="start">{pax_v} pax</text>'
            else:
                svg += f'<text x="{xp + r_c + 6}" y="{y_ln+3}" font-size="10" font-weight="bold" fill="#4CAF50" text-anchor="start">🏁 OK</text>'
            
            if sep_s and not is_parked:
                sep_dy = base_dy - 22 if via == 2 else base_dy + 22
                svg += f'<text x="{xp}" y="{y_ln+sep_dy}" font-size="10" font-weight="bold" fill="#111" text-anchor="middle">{sep_s}</text>'

    svg += '</svg>'
    return svg.replace('\n', ''), H

# =============================================================================
# MOTOR VISUAL 2: SCADA JAVASCRIPT (CLIENT-SIDE RENDERING + POPUPS)
# =============================================================================
def draw_scada_js(df_dia_e, ser_accum_plot, seat_accum_plot, hora_inicial, titulo_extra, active_sers_list, gap_vias, use_rm):
    """
    Empaqueta el perfil matemático y genera el Iframe HTML.
    """
    trips_data = []
    
    for _, row in df_dia_e.iterrows():
        t_ini, t_fin = float(row['t_ini']), float(row['t_fin'])
        t_points = np.arange(t_ini, t_fin + 0.1, 0.1)
        if t_points[-1] < t_fin: t_points = np.append(t_points, t_fin)
            
        traj = []
        for t in t_points:
            km = km_at_t(row['t_ini'], row['t_fin'], t, row['Via'], use_rm, row['km_orig'], row['km_dest'], row.get('nodos'))
            traj.append([round(t, 2), round(km, 3)])
            
        if len(traj) > 0:
            traj[-1] = [round(t_fin, 2), round(float(row['km_dest']), 3)]
            
        pax_dict = row.get('pax_d', {})
        if isinstance(pax_dict, dict) and pax_dict:
            pax_arr = [int(pax_dict.get(c, 0)) for c in PAX_COLS]
        else:
            pax_arr = [int(row.get('pax_abordo', 0))] * len(PAX_COLS)
            
        trips_data.append({
            'id': str(row['_id']),
            'Via': int(row['Via']),
            't_ini': t_ini,
            't_fin': t_fin,
            'svc': str(row.get('num_servicio', '')),
            'motriz': str(row.get('motriz_num', '')),
            'tipo': str(row.get('tipo_tren', 'XT-100')),
            'doble': bool(row.get('doble', False)),
            'kwh_total': float(row.get('kwh_viaje_neto', 0)),
            'pax_arr': pax_arr,
            'traj': traj
        })
        
    json_data = json.dumps(trips_data)
    json_km_acum = json.dumps(KM_ACUM[:N_EST])
    json_pax_cols = json.dumps(PAX_COLS)
    
    svg_bg, H = draw_diagram_svg(pd.DataFrame(), ser_accum_plot, seat_accum_plot, "Modo SCADA Activo", titulo_extra, active_sers_list, gap_vias)
    svg_bg = svg_bg.replace('</svg>', '<g id="trains_layer"></g></svg>') 
    
    js_code = """
    const trips = JSON_DATA_HERE;
    const KM_ACUM = KM_ACUM_HERE;
    const PAX_COLS = PAX_COLS_HERE;
    const W = 1200;
    const PADDING_X = 40; 
    const KM_TOTAL = KM_TOTAL_HERE;
    const gap_vias = GAP_VIAS_HERE;
    const Y_V2 = 200;
    const Y_V1 = Y_V2 + gap_vias;
    
    let currentTime = HORA_INICIAL_HERE;
    let playing = false;
    let lastTimestamp = 0;
    
    const playBtn = document.getElementById('playBtn');
    const timeSlider = document.getElementById('timeSlider');
    const timeDisplay = document.getElementById('timeDisplay');
    const speedSelect = document.getElementById('speedSelect');
    const trainsLayer = document.getElementById('trains_layer');
    
    function formatTime(mins) {
        let h = Math.floor(mins / 60);
        let m = Math.floor(mins % 60);
        return (h < 10 ? '0'+h : h) + ':' + (m < 10 ? '0'+m : m);
    }
    
    function xkm(km) { return PADDING_X + km * ((W - 2 * PADDING_X) / KM_TOTAL); }
    
    function getPos(traj, t) {
        if (t <= traj[0][0]) return traj[0][1];
        if (t >= traj[traj.length-1][0]) return traj[traj.length-1][1];
        for(let i=0; i<traj.length-1; i++) {
            if(t >= traj[i][0] && t <= traj[i+1][0]) {
                let p = (t - traj[i][0]) / (traj[i+1][0] - traj[i][0]);
                return traj[i][1] + p * (traj[i+1][1] - traj[i][1]);
            }
        }
        return traj[0][1];
    }
    
    function getPaxAtKm(pax_arr, km, via) {
        let pax = 0;
        if (via === 1) {
            for(let i=0; i < KM_ACUM.length; i++) {
                if (km >= KM_ACUM[i]) pax = pax_arr[i];
                else break;
            }
        } else {
            for(let i = KM_ACUM.length-1; i >= 0; i--) {
                if (km <= KM_ACUM[i]) pax = pax_arr[i];
                else break;
            }
        }
        return pax;
    }
    
    function drawTrains() {
        let html = '';
        let activeTrips = trips.filter(tr => currentTime >= tr.t_ini && currentTime <= tr.t_fin + 5.0);
        
        activeTrips.forEach((tr, index) => {
            let is_parked = currentTime >= tr.t_fin;
            let current_t = Math.min(currentTime, tr.t_fin); 
            let km = getPos(tr.traj, current_t);
            let xp = xkm(km);
            
            let y_ln = tr.Via === 2 ? Y_V2 : Y_V1;
            let color = tr.Via === 2 ? '#c62828' : '#1565c0';
            if (is_parked) color = '#4CAF50'; 
            
            let r_c = tr.doble ? 18 : 11;
            
            let frac = (current_t - tr.t_ini) / Math.max(0.001, (tr.t_fin - tr.t_ini));
            let current_kwh = tr.kwh_total * frac;
            let current_pax = is_parked ? 0 : getPaxAtKm(tr.pax_arr, km, tr.Via);
            
            let base_dy = tr.Via === 2 ? (-r_c - 16) : (r_c + 16);
            if (index % 2 !== 0) {
                base_dy = tr.Via === 2 ? base_dy - 28 : base_dy + 28;
            }
            
            let lbl = tr.tipo === 'SFE' ? 'SFE' : (tr.tipo === 'XT-M' ? 'Modular' : 'XT-100');
            if (tr.motriz) lbl += ' [' + tr.motriz + ']';
            
            let pax_prof = "Perfil Estaciones:&#10;";
            let p_chunk = [];
            for(let i=0; i<tr.pax_arr.length; i++) {
                if(tr.pax_arr[i] > 0) {
                    p_chunk.push(PAX_COLS[i] + ":" + tr.pax_arr[i]);
                    if(p_chunk.length === 4) {
                        pax_prof += p_chunk.join(" | ") + "&#10;";
                        p_chunk = [];
                    }
                }
            }
            if(p_chunk.length > 0) pax_prof += p_chunk.join(" | ") + "&#10;";
            if(p_chunk.length === 0 || pax_prof === "Perfil Estaciones:&#10;") pax_prof = "";

            let state_str = is_parked ? "🏁 Estacionado en Terminal (Turnaround)" : "🚄 En Tránsito";
            let safe_tooltip = `Tren: ${lbl} (Serv. ${tr.svc})&#10;Vía ${tr.Via} | km ${km.toFixed(2)}&#10;Estado: ${state_str}&#10;Pasajeros Actuales: ${current_pax} pax&#10;${pax_prof}Energía Neta (KWh): ${Math.round(current_kwh)}`;
            
            html += `<circle cx="${xp}" cy="${y_ln}" r="${r_c}" fill="${color}" stroke="black" stroke-width="2"><title>${safe_tooltip}</title></circle>`;
            html += `<rect x="${xp-45}" y="${y_ln+base_dy-12}" width="90" height="24" fill="white" fill-opacity="0.9" rx="3" stroke="#ccc"/>`;
            html += `<text x="${xp}" y="${y_ln+base_dy-2}" font-size="10" font-weight="bold" fill="#111" text-anchor="middle">${lbl}</text>`;
            html += `<text x="${xp}" y="${y_ln+base_dy+9}" font-size="9" font-weight="bold" fill="#111" text-anchor="middle">Serv. ${tr.svc}</text>`;
            html += `<text x="${xp - r_c - 6}" y="${y_ln+3}" font-size="10" font-weight="bold" fill="#2E7D32" text-anchor="end">${Math.round(current_kwh)} kWh</text>`;
            if (!is_parked) {
                html += `<text x="${xp + r_c + 6}" y="${y_ln+3}" font-size="10" font-weight="bold" fill="#1565c0" text-anchor="start">${current_pax} pax</text>`;
            } else {
                html += `<text x="${xp + r_c + 6}" y="${y_ln+3}" font-size="10" font-weight="bold" fill="#4CAF50" text-anchor="start">🏁 OK</text>`;
            }
        });
        
        trainsLayer.innerHTML = html;
        timeDisplay.innerText = formatTime(currentTime);
    }
    
    function loop(timestamp) {
        if (!lastTimestamp) lastTimestamp = timestamp;
        let deltaTime = timestamp - lastTimestamp;
        lastTimestamp = timestamp;
        
        if (playing) {
            let speed = parseFloat(speedSelect.value);
            let deltaMins = (deltaTime / 1000) * speed; 
            currentTime += deltaMins;
            if (currentTime >= 1439.0) {
                currentTime = 1439.0;
                playing = false;
                playBtn.innerText = '▶️ PLAY';
                playBtn.style.background = '#1565c0';
            }
            timeSlider.value = currentTime;
            drawTrains();
        }
        requestAnimationFrame(loop);
    }
    
    playBtn.addEventListener('click', () => {
        playing = !playing;
        playBtn.innerText = playing ? '⏸ PAUSA' : '▶️ PLAY';
        playBtn.style.background = playing ? '#c62828' : '#1565c0';
        if(playing) lastTimestamp = 0;
    });
    
    timeSlider.addEventListener('input', (e) => {
        currentTime = parseFloat(e.target.value);
        drawTrains();
    });
    
    drawTrains();
    requestAnimationFrame(loop);
    """
    
    js_code = js_code.replace("JSON_DATA_HERE", json_data)
    js_code = js_code.replace("KM_ACUM_HERE", json_km_acum)
    js_code = js_code.replace("PAX_COLS_HERE", json_pax_cols)
    js_code = js_code.replace("KM_TOTAL_HERE", str(KM_TOTAL))
    js_code = js_code.replace("GAP_VIAS_HERE", str(gap_vias))
    js_code = js_code.replace("HORA_INICIAL_HERE", str(hora_inicial))

    html_template = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <style>
            body {{ margin: 0; padding: 0; font-family: sans-serif; }}
            .controls {{ display: flex; gap: 15px; align-items: center; background: #e3f2fd; padding: 15px 20px; border-radius: 8px; border: 1px solid #bbdefb; margin-bottom: 10px; }}
            button {{ background: #1565c0; color: white; border: none; padding: 10px 25px; border-radius: 6px; font-weight: bold; cursor: pointer; font-size: 14px; transition: 0.2s; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }}
            button:hover {{ filter: brightness(1.1); transform: translateY(-1px); }}
            input[type=range] {{ flex-grow: 1; cursor: pointer; height: 8px; border-radius: 4px; background: #90caf9; appearance: none; outline: none; }}
            input[type=range]::-webkit-slider-thumb {{ appearance: none; width: 20px; height: 20px; border-radius: 50%; background: #1565c0; cursor: pointer; }}
            .time-disp {{ font-family: monospace; font-size: 26px; font-weight: bold; color: #111; min-width: 90px; text-align: center; background: white; padding: 6px 12px; border-radius: 6px; border: 2px solid #90caf9; box-shadow: inset 0 2px 4px rgba(0,0,0,0.05); }}
            select {{ padding: 10px; border-radius: 6px; border: 1px solid #bbdefb; font-weight: bold; background: white; cursor: pointer; color: #1565c0; }}
        </style>
    </head>
    <body>
        <div class="controls">
            <button id="playBtn">▶️ PLAY</button>
            <input type="range" id="timeSlider" min="0" max="1439" step="0.1" value="{hora_inicial}">
            <div class="time-disp" id="timeDisplay">00:00</div>
            <select id="speedSelect">
                <option value="0.5">Lento (0.5x)</option>
                <option value="1">Real (1x)</option>
                <option value="5" selected>Operativo (5x)</option>
                <option value="15">Acelerado (15x)</option>
                <option value="60">Time-Lapse (60x)</option>
            </select>
        </div>
        <div id="mapContainer">
            {svg_bg}
        </div>
        <script>
            {js_code}
        </script>
    </body>
    </html>
    """
    return html_template, H

# =============================================================================
# ORQUESTADOR CENTRAL: GEMELO DIGITAL
# =============================================================================
def render_gemelo_digital(df_dia, df_dia_e, active_sers, fecha_sel, pct_trac, use_rm, use_pend, estacion_anio, prefix_key, gap_vias, pax_dia_total=0, df_vacios_real=None, km_limache_manual=0.0):
    if df_vacios_real is None:
        df_vacios_real = pd.DataFrame()
        
    if 'maniobra' not in df_dia.columns: df_dia['maniobra'] = None
    if 'maniobra' not in df_dia_e.columns: df_dia_e['maniobra'] = None
    
    slider_key = f"sl_ui_{prefix_key}"
    time_key = f"t_math_{prefix_key}"
    
    if slider_key not in st.session_state: st.session_state[slider_key] = 480.0
    if time_key not in st.session_state: st.session_state[time_key] = 480.0
    if f'play_{prefix_key}' not in st.session_state: st.session_state[f'play_{prefix_key}'] = False
        
    cf, cm = st.columns([3,2])
    with cm: 
        modo = st.radio("Motor de Renderizado", ["🔒 Analítico (Estático Python)", "🚀 SCADA (Animado JS)"], horizontal=True, key=f"modo_{prefix_key}")

    if modo != "▶️ Animado" and "SCADA" not in modo: 
        st.session_state[f'play_{prefix_key}'] = False

    if st.session_state[f'play_{prefix_key}']:
        speed = float(st.session_state.get(f'vs1_{prefix_key}', 1.0))
        new_val = st.session_state[time_key] + (0.5 * speed) 
        if new_val >= 1439.0:
            st.session_state[time_key] = 1439.0
            st.session_state[f'play_{prefix_key}'] = False
        else:
            st.session_state[time_key] = new_val

    # =========================================================================
    # LÓGICA DE TIEMPO COMPARTIDA
    # =========================================================================
    if "SCADA" not in modo:
        c1,c2,c3,c4,c5,_ = st.columns([1,1,1,1,1,2])
        if c1.button("−15m", key=f"m15_{prefix_key}"): st.session_state[time_key] = max(0.0, st.session_state[time_key] - 15.0)
        if c2.button("−1m", key=f"m1_{prefix_key}"): st.session_state[time_key] = max(0.0, st.session_state[time_key] - 1.0)
        if c3.button("⏸" if st.session_state[f'play_{prefix_key}'] else "▶️", key=f"pb_{prefix_key}"):
            st.session_state[f'play_{prefix_key}'] = not st.session_state[f'play_{prefix_key}']
            st.rerun() 
        if c4.button("+1m", key=f"p1_{prefix_key}"): st.session_state[time_key] = min(1439.0, st.session_state[time_key] + 1.0)
        if c5.button("+15m", key=f"p15_{prefix_key}"): st.session_state[time_key] = min(1439.0, st.session_state[time_key] + 15.0)

        def sync_time():
            st.session_state[time_key] = st.session_state[slider_key]

        st.slider("Timeline", min_value=0.0, max_value=1439.0, value=float(st.session_state[time_key]), step=0.1, key=slider_key, on_change=sync_time)

    hora_m1 = st.session_state[time_key]
    hora_s1 = mins_to_time_str(hora_m1)

    # =========================================================================
    # 1. CÁLCULO FÍSICO INSTANTÁNEO
    # =========================================================================
    df_act = df_dia_e[(df_dia_e['t_ini'] <= hora_m1) & (df_dia_e['t_fin'] + 5.0 >= hora_m1)].copy()
    instant_ser_demands_kw = {s[1]: 0.0 for s in active_sers}
    
    if not df_act.empty:
        df_act['is_parked'] = hora_m1 >= df_act['t_fin']
        
        df_act['frac_act'] = np.minimum(1.0, (hora_m1 - df_act['t_ini']) / np.maximum(0.001, df_act['t_fin'] - df_act['t_ini']))
        df_act['kwh_neto'] = df_act['kwh_viaje_neto'] * df_act['frac_act']
        
        df_act['km_pos'] = df_act.apply(lambda r: km_at_t(r['t_ini'], r['t_fin'], min(hora_m1, r['t_fin']), r['Via'], use_rm, r['km_orig'], r['km_dest'], r.get('nodos')), axis=1)
        
        def _vel_real(r):
            if r['is_parked']: return 0.0
            km_now = r['km_pos']
            km_next = km_at_t(r['t_ini'], r['t_fin'], hora_m1 + 0.01, r['Via'], use_rm, r['km_orig'], r['km_dest'], r.get('nodos'))
            if abs(km_next - km_now) < 0.0001: return 0.0 
            return vel_at_km(km_now, r['Via'], use_rm)
            
        df_act['vel'] = df_act.apply(_vel_real, axis=1)
        df_act['km_rec'] = df_act.apply(lambda r: max(0.0, abs(r['km_pos'] - r['km_orig'])), axis=1)
        df_act['pax_inst'] = df_act.apply(lambda r: 0 if r['is_parked'] else get_pax_at_km(r.get('pax_d', {}), r['km_pos'], r['Via'], r.get('pax_abordo', 0)), axis=1)

        def _sep_next(row, df_via):
            if row['is_parked']: return '—'
            km = row['km_pos']; vel = row['vel']
            if vel < 1: return '—'
            ahead = df_via[df_via['km_pos'] > km] if row['Via'] == 1 else df_via[df_via['km_pos'] < km]
            if ahead.empty: return '—'
            d = abs(ahead['km_pos'] - km).min()
            return f"{round(d/max(1, vel)*60,1)} min ({d:.1f} km)"
        
        df_act['sep_next'] = df_act.apply(lambda r: _sep_next(r, df_act[df_act['Via']==r['Via']].drop(index=r.name)), axis=1)

        def _make_tooltip_and_power(row):
            m_num = str(row.get('motriz_num', ''))
            tipo = str(row.get('tipo_tren', 'XT-100'))
            serv = str(row.get('num_servicio', ''))
            is_parked = row.get('is_parked', False)
            
            nombre_tren = f"{tipo}-{m_num}" if m_num else tipo
            
            doble_tramo = row.get('doble', False)
            man = row.get('maniobra')
            if man in ['CORTE_BTO', 'CORTE_PU_SA_BTO']: doble_tramo = True if row['km_pos'] <= KM_ACUM[14] else False
            elif man == 'ACOPLE_BTO': doble_tramo = False if row['km_pos'] > KM_ACUM[14] else True
            elif man == 'CORTE_SA': doble_tramo = True if row['km_pos'] <= KM_ACUM[18] else False
            elif man == 'ACOPLE_SA': doble_tramo = False if row['km_pos'] > KM_ACUM[18] else True
                
            cab = f"Tren: {nombre_tren} (Serv. {serv}) | {'DOBLE' if doble_tramo else 'Simple'}\n"
            cab += f"Vía {row['Via']} | km {row['km_pos']:.2f} | {int(row['vel'])} km/h\n"
            
            f_flota = FLOTA.get(tipo, FLOTA["XT-100"])
            n_unidades = 2 if doble_tramo else 1
            tara_base = (f_flota['tara_t'] + f_flota['m_iner_t']) * n_unidades
            pax_v = int(row.get('pax_inst', 0))
            masa_total = tara_base + ((pax_v * PAX_KG) / 1000.0)
            
            # 💡 FIX: Inyectamos el Termostato Inteligente (Target de Auxiliares) en el Tooltip del Mapa
            if estacion_anio == "invierno":
                aux_nominal_unidad = f_flota.get('aux_kw_heat', f_flota.get('aux_kw', 65.16))
            else:
                aux_nominal_unidad = f_flota.get('aux_kw_cool', f_flota.get('aux_kw', 58.76))
                
            f_compresor_especifico = f_flota.get('f_compresor_dwell', 1.08)
            
            if is_parked:
                state = "DWELL"
                state_icon = "🏁 Estacionado en Terminal"
                p_aux_kw = calcular_aux_dinamico(aux_nominal_unidad * n_unidades, hora_m1 / 60.0, 0, f_flota.get('cap_max', 398) * n_unidades, estacion_anio, state, f_compresor_especifico)
                p_elec_kw = p_aux_kw
            else:
                state, v_kmh = get_train_state_and_speed(hora_m1, row['Via'], use_rm, row['km_orig'], row['km_dest'], row.get('nodos'))
                state_icon = "🟢 Traccionando" if state == "ACCEL" else "🔴 Frenando (Regen)" if state == "BRAKE" else "🟡 Velocidad Crucero"
                
                v_ms = v_kmh / 3.6
                if n_unidades == 2: f_davis = (f_flota['davis_A'] * 2) + (f_flota['davis_B'] * 2 * v_kmh) + (f_flota['davis_C'] * 1.35 * (v_kmh**2))
                else: f_davis = f_flota['davis_A'] + f_flota['davis_B'] * v_kmh + f_flota['davis_C'] * (v_kmh**2)
                
                p_aux_kw = calcular_aux_dinamico(aux_nominal_unidad * n_unidades, hora_m1 / 60.0, pax_v, f_flota.get('cap_max', 398) * n_unidades, estacion_anio, state, f_compresor_especifico)
                eta_m = f_flota.get('eta_motor', 0.92)
                
                if state == "ACCEL": p_mech = f_flota['p_max_kw'] * n_unidades * (pct_trac / 100.0)
                elif state == "CRUISE": p_mech = (f_davis * v_ms) / 1000.0
                elif state == "BRAKE": p_mech = -f_flota.get('p_freno_max_kw', f_flota['p_max_kw']*1.2) * n_unidades * 0.6
                else: p_mech = 0.0
                
                if p_mech > 0: p_elec_kw = (p_mech / eta_m) + p_aux_kw
                elif p_mech < 0: p_elec_kw = (p_mech * ETA_REGEN_NETA) + p_aux_kw
                else: p_elec_kw = p_aux_kw
            
            cab += f"Estado: {state_icon}\n"
            
            dist_kw = distribuir_potencia_sers_kw(p_elec_kw, row['km_pos'], active_sers)
            for s_n, v_kw in dist_kw.items():
                instant_ser_demands_kw[s_n] += v_kw
            
            if not is_parked:
                cab += f"Pax a Bordo Actual: {pax_v}\n"
                pax_d = row.get('pax_d', {})
                if isinstance(pax_d, dict) and sum(pax_d.values()) > 0:
                    cab += "Perfil Estaciones:\n"
                    pax_items = [f"{k}:{v}" for k, v in pax_d.items() if v > 0]
                    for i in range(0, len(pax_items), 4):
                        cab += " | ".join(pax_items[i:i+4]) + "\n"
            
            cab += f"Masa Dinámica: {masa_total:.1f} t\n"
            if not is_parked:
                cab += f"Siguiente Tren: {row['sep_next']}"
            return cab

        df_act['tooltip'] = df_act.apply(_make_tooltip_and_power, axis=1)

    # =========================================================================
    # 2. CÁLCULO FÍSICO ACUMULADO Y VACÍOS
    # =========================================================================
    vacios_hasta_ahora = []
    vacio_kwh_total = 0.0
    vacio_km_total = 0.0
    energy_by_fleet = {'XT-100': 0.0, 'XT-M': 0.0, 'SFE': 0.0}
    ser_accum_visual = {s[1]: 0.0 for s in active_sers}
    
    if not df_vacios_real.empty:
        df_dia_v = df_vacios_real[df_vacios_real['Fecha_str'] == fecha_sel]
        vacios_dia = df_dia_v.to_dict('records')
    else:
        vacios_dia = get_vacios_dia(df_dia)
        for idx, row in df_dia[df_dia['maniobra'].notnull()].iterrows():
            man, t_arr_bto, t_arr_sa = row['maniobra'], row['t_ini'] + (40.0 if row['Via'] == 1 else 20.0), row['t_ini'] + (47.0 if row['Via'] == 1 else 13.0)
            if man in ['CORTE_BTO', 'CORTE_PU_SA_BTO']: vacios_dia.append({'t_asigned': t_arr_bto, 'tipo': row['tipo_tren'], 'doble': False, 'cochera': True, 'dist': 2.0, 'motriz_num': f"{row.get('motriz_num', '')}-B", 'origen_txt': 'El Belloto', 'destino_txt': 'Taller EB', 'km_orig': KM_ACUM[14], 'km_dest': KM_ACUM[14]})
            elif man == 'ACOPLE_BTO': vacios_dia.append({'t_asigned': t_arr_bto - 5.0, 'tipo': row['tipo_tren'], 'doble': False, 'cochera': True, 'dist': 2.0, 'motriz_num': f"{row.get('motriz_num', '')}-B", 'origen_txt': 'Taller EB', 'destino_txt': 'El Belloto', 'km_orig': KM_ACUM[14], 'km_dest': KM_ACUM[14]})
            elif man == 'CORTE_SA': vacios_dia.append({'t_asigned': t_arr_sa, 'tipo': row['tipo_tren'], 'doble': False, 'cochera': True, 'dist': abs(KM_ACUM[18] - KM_ACUM[14]) + 2.0, 'motriz_num': f"{row.get('motriz_num', '')}-B", 'origen_txt': 'Sargento Aldea', 'destino_txt': 'Taller EB', 'km_orig': KM_ACUM[18], 'km_dest': KM_ACUM[14]})
            elif man == 'ACOPLE_SA': vacios_dia.append({'t_asigned': t_arr_sa - 20.0, 'tipo': row['tipo_tren'], 'doble': False, 'cochera': True, 'dist': abs(KM_ACUM[18] - KM_ACUM[14]) + 2.0, 'motriz_num': f"{row.get('motriz_num', '')}-B", 'origen_txt': 'Taller EB', 'destino_txt': 'Sargento Aldea', 'km_orig': KM_ACUM[14], 'km_dest': KM_ACUM[18]})

    if km_limache_manual > 0:
        chunks = [1.0] * int(km_limache_manual)
        remainder = km_limache_manual - int(km_limache_manual)
        if remainder > 0: chunks.append(remainder)
        
        for chunk in chunks:
            vacios_dia.append({
                't_asigned': 0.0, 
                'tipo': 'XT-100', 
                'doble': False,
                'cochera': False,
                'dist': chunk,
                'motriz_num': 'Manual',
                'origen_txt': 'Patio Limache (Manual)',
                'destino_txt': 'Patio Limache (Manual)',
                'km_orig': KM_ACUM[20],
                'km_dest': KM_ACUM[20],
                'Via': 1
            })

    vacios_hasta_ahora = [v for v in vacios_dia if v['t_asigned'] <= hora_m1]
    vacio_count = len([v for v in vacios_hasta_ahora if v.get('motriz_num') != 'Manual'])
    
    for v in vacios_hasta_ahora:
        is_manual_limache = "Manual" in str(v.get('origen_txt', ''))
        es_cochera = v.get('cochera', False)
        dist_efe = v.get('dist', 0.0)
        factor_flota = 2 if v.get('doble', False) else 1
        
        if is_manual_limache:
            vacio_km_total += dist_efe * factor_flota
            trc_v, aux_v, reg_v, _, _, t_h_v = simular_tramo_termodinamico(
                v['tipo'], v.get('doble', False), 0.0, dist_efe, 1, 
                pct_trac, use_rm, False, None, {}, 0, 20.0, None, estacion_anio, v.get('t_asigned', 480.0), True
            )
            e_p = trc_v + aux_v - reg_v
            vacio_kwh_total += e_p
            if v['tipo'] in energy_by_fleet: energy_by_fleet[v['tipo']] += e_p
            if active_sers:
                for s_name, e_val in distribuir_energia_sers(e_p, t_h_v, KM_ACUM[20], KM_ACUM[20], active_sers).items(): 
                    ser_accum_visual[s_name] += e_val
        else:
            if es_cochera:
                vacio_km_total += 1.0 * factor_flota
                trc_a, aux_a, reg_a, _, _, th_a = simular_tramo_termodinamico(
                    v['tipo'], False, 25.3, 26.3, 1, 
                    pct_trac, use_rm, False, None, {}, 0, 20.0, None, estacion_anio, v['t_asigned'], True
                )
                e_panto_a = trc_a + aux_a - reg_a
                vacio_kwh_total += e_panto_a
                if v['tipo'] in energy_by_fleet: energy_by_fleet[v['tipo']] += e_panto_a
                if active_sers:
                    km_cochera = v.get('km_orig', KM_ACUM[14])
                    for s_name, e_val in distribuir_energia_sers(e_panto_a, th_a, km_cochera, km_cochera, active_sers).items(): 
                        ser_accum_visual[s_name] += e_val
            if dist_efe > 0.0:
                vacio_km_total += dist_efe * factor_flota
                km_orig = v.get('km_orig', 0.0)
                km_dest = v.get('km_dest', km_orig)
                via_v = 1 if km_orig <= km_dest else 2
                is_loc = abs(km_orig - km_dest) < 0.001
                if is_loc: 
                    km_dest = km_orig + dist_efe
                    via_v = 1
                trc_b, aux_b, reg_b, _, _, th_b = simular_tramo_termodinamico(
                    v['tipo'], False, km_orig, km_dest, via_v, 
                    pct_trac, use_rm, use_pend if not is_loc else False, None, {}, 0, 20.0 if is_loc else None, None, estacion_anio, v['t_asigned'], True
                )
                e_panto_b = trc_b + aux_b - reg_b
                vacio_kwh_total += e_panto_b
                if v['tipo'] in energy_by_fleet: energy_by_fleet[v['tipo']] += e_panto_b
                if active_sers:
                    for s_name, e_val in distribuir_energia_sers(e_panto_b, th_b, km_orig, km_dest, active_sers).items(): 
                        ser_accum_visual[s_name] += e_val

    t_regen_acum = 0.0
    t_reostato_acum = 0.0

    for idx, r in df_dia_e[df_dia_e['t_ini'] <= hora_m1].iterrows():
        t_eval = min(hora_m1, r['t_fin'])
        frac = (t_eval - r['t_ini']) / max(0.001, r['t_fin'] - r['t_ini'])
        km_now = km_at_t(r['t_ini'], r['t_fin'], t_eval, r['Via'], use_rm, r['km_orig'], r['km_dest'], r.get('nodos'))
        e_p_frac = (r['kwh_viaje_trac'] + r['kwh_viaje_aux'] - r['kwh_viaje_regen']) * frac
        for s_name, e_val in distribuir_energia_sers(e_p_frac, (t_eval - r['t_ini']) / 60.0, r['km_orig'], km_now, active_sers).items(): ser_accum_visual[s_name] += e_val 

    df_acum = df_dia_e[df_dia_e['t_ini'] <= hora_m1]
    if not df_acum.empty:
        t_regen_acum = df_acum['kwh_viaje_regen'].sum()
        for f_type in ['XT-100', 'XT-M', 'SFE']:
            sub = df_acum[df_acum['tipo_tren'] == f_type]
            if not sub.empty: energy_by_fleet[f_type] += sub['kwh_viaje_neto'].sum()

    total_ser_kwh_44kv = sum(max(0.0, val) for val in ser_accum_visual.values()) / ETA_SER_RECTIFICADOR
    t_elap = max(0.001, hora_m1 / 60.0)
    avg_d = {k: max(0.0, v) / ETA_SER_RECTIFICADOR / t_elap for k, v in ser_accum_visual.items()}
    flujo_avg = calcular_flujo_ac_nodo(avg_d)
    total_ac_loss_kwh = flujo_avg['P_loss_kw'] * (1.15**2) * t_elap
    seat_accum_1 = (total_ser_kwh_44kv + total_ac_loss_kwh) / 0.99

    # =========================================================================
    # 3. RENDERIZADO VISUAL BIFURCADO (SCADA vs PYTHON)
    # =========================================================================
    if "SCADA" in modo:
        html_scada, H_scada = draw_scada_js(df_dia_e, {k: max(0.0, v) for k, v in ser_accum_visual.items()}, seat_accum_1, hora_m1, "", active_sers, gap_vias, use_rm)
        components.html(html_scada, height=H_scada + 100)
        st.info("💡 **Modo SCADA Activado:** La animación gráfica se procesa a 60 FPS. Pasa el cursor sobre los trenes para ver la termodinámica.")
    else:
        if st.session_state.get(f'vs1_{prefix_key}', 1.0) > 1.0:
            st.select_slider("Velocidad", options=[0.5, 1, 2, 5, 10], value=st.session_state.get(f'vs1_{prefix_key}', 1.0), format_func=lambda x: f"×{x}", key=f"vs1_{prefix_key}")
        
        st.markdown(
            f"<span style='font-size:2.2rem;font-weight:700;letter-spacing:2px;'>⏱ {hora_s1[:5]}</span>"
            f"<span style='font-size:0.9rem;color:#666;'> &nbsp;·&nbsp; {fecha_sel} &nbsp;·&nbsp; "
            f"⚙️ {pct_trac}% Tracción</span>",
            unsafe_allow_html=True
        )
        svg_html, H_python = draw_diagram_svg(df_act, {k: max(0.0, v) for k, v in ser_accum_visual.items()}, seat_accum_1, hora_s1[:5], "", active_sers, gap_vias)
        st.markdown(svg_html, unsafe_allow_html=True)

    # =========================================================================
    # 4. PANELES DE DATOS (SIEMPRE VISIBLES PARA AMBOS MODOS)
    # =========================================================================
    n_circ = len(df_act) if not df_act.empty else 0
    n_d    = int(df_act['doble'].sum()) if not df_act.empty else 0
    n_v1   = int((df_act['Via']==1).sum()) if not df_act.empty else 0
    n_v2   = int((df_act['Via']==2).sum()) if not df_act.empty else 0
    pax_t  = int(df_act['pax_inst'].sum()) if not df_act.empty else 0
    kwh_t  = round(df_act['kwh_neto'].sum(),0) if (not df_act.empty and 'kwh_neto' in df_act.columns) else 0
    
    regen_t= round(t_regen_acum, 0)
    
    trenkm = round(df_act['tren_km'].sum(),1) if (not df_act.empty and 'tren_km' in df_act.columns) else 0.0
    km_rec = df_act['km_rec'].sum() if (not df_act.empty and 'km_rec' in df_act.columns) else 0
    ide_i  = round(kwh_t/max(1, km_rec), 3) if km_rec > 0 else 0.0

    st.divider()
    st.markdown(f"#### 🕐 Monitor Instantáneo Dinámico")
    r1a,r1b,r1c,r1d = st.columns(4)
    r1a.metric("🚆 Servicios", n_circ)
    r1b.metric("V1→Limache", n_v1)
    r1c.metric("V2←Puerto", n_v2)
    r1d.metric("🚈 Doble (Original)", n_d)
    
    r2a,r2b,r2c,r2d = st.columns(4)
    r2a.metric("🧑‍🤝‍🧑 Pax en Vía Inst.", f"{pax_t:,}")
    r2b.metric("⚡ kWh neto", f"{kwh_t:,.0f}", f"−{regen_t:,.0f} regen util")
    r2c.metric("📏 Tren-km Inst.", f"{trenkm:,.1f}")
    r2d.metric("💡 IDE inst.", f"{ide_i:.3f} kWh/km")

    st.divider()
    st.markdown("#### 🔌 Cargabilidad Instantánea de Subestaciones (Squeeze Control)")
    st.caption("Muestra la demanda real en kW que los trenes exigen a la red en este mismo segundo. Los rectificadores son unidireccionales (Diodos).")
    
    if not active_sers:
        st.info("No hay SERs activas para monitorear.")
    else:
        flujo_ac_dc = calcular_flujo_ac_nodo(instant_ser_demands_kw)
        st.markdown(f"<div style='text-align:right; font-size:12px; color:#c62828;'>🔥 Pérdidas térmicas AC (I²R) de la red troncal en este instante: <b>{flujo_ac_dc.get('P_loss_kw', 0.0):.1f} kW</b></div>", unsafe_allow_html=True)
        cols_ser = st.columns(len(active_sers))
        for i, ser_info in enumerate(active_sers):
            s_name = ser_info[1]
            cap_kw = SER_CAPACITY_KW.get(s_name, 3000.0)
            dem_kw_bruta = instant_ser_demands_kw.get(s_name, 0.0)
            dem_kw = max(0.0, dem_kw_bruta) 
            vac_actual = flujo_ac_dc.get(s_name, {}).get('Vac', V_NOMINAL_AC)
            vdc_actual = flujo_ac_dc.get(s_name, {}).get('Vdc', 3000.0)
            pct_carga = (dem_kw / cap_kw) * 100.0
            
            if dem_kw == 0.0 and dem_kw_bruta < -10.0: color_bar, texto_estado = "#9E9E9E", "Bloqueo Diodos (Reóstato)"
            elif vdc_actual < 2600.0: color_bar, texto_estado = "#C62828", "⚠️ SQUEEZE CONTROL"
            elif vdc_actual < 2850.0: color_bar, texto_estado = "#F9A825", "Estrés Moderado (Caída AC)"
            elif pct_carga <= 65: color_bar, texto_estado = "#1565C0", "Carga Óptima"
            else: color_bar, texto_estado = "#F9A825", "Capacidad exigida"
                
            with cols_ser[i]:
                st.markdown(f"**{s_name}** ({cap_kw/1000:.1f} MVA)")
                st.markdown(f"<div style='font-size:18px; font-weight:bold; color:{color_bar};'>{dem_kw:,.0f} kW</div>", unsafe_allow_html=True)
                st.markdown(f"<div style='font-size:13px; font-family:monospace; margin-bottom:4px;'><span style='color:#666;'>Tensión AC:</span> <b>{vac_actual/1000:.2f} kV</b><br><span style='color:#666;'>Barra DC:</span> <b style='color:{color_bar};'>{vdc_actual:.0f} Vcc</b></div>", unsafe_allow_html=True)
                st.markdown(f"<div style='width:100%; background-color:#e0e0e0; border-radius:4px; margin-bottom: 4px;'><div style='width:{min(100, max(0, pct_carga))}%; background-color:{color_bar}; height:8px; border-radius:4px;'></div></div>", unsafe_allow_html=True)
                st.markdown(f"<span style='font-size:11px; color:#666;'>Uso: {pct_carga:.1f}% - {texto_estado}</span>", unsafe_allow_html=True)

    df_comp = df_dia_e[df_dia_e['t_fin']<=hora_m1]
    df_inic = df_dia_e[df_dia_e['t_ini']<=hora_m1]
    n_inic  = len(df_inic)
    n_comp  = len(df_comp)
    
    km_ac   = round(df_comp['tren_km'].sum(), 1) if not df_comp.empty else 0.0
    ide_ac  = round(seat_accum_1 / max(1, df_inic['tren_km'].sum() + vacio_km_total), 3) if not df_inic.empty and (df_inic['tren_km'].sum() + vacio_km_total) > 0 else 0.0

    st.divider()
    st.markdown(f"#### 📊 Análisis Global Acumulado (00:00 → {hora_s1[:5]})")
    
    if not df_inic.empty:
        st.markdown("##### 🚆 Total de Servicios Despachados por Trayecto y Flota")
        
        trayectos = df_inic.groupby(['Via', 'svc_type', 'tipo_tren']).size().unstack(fill_value=0)
        for col in ['XT-100', 'XT-M', 'SFE']:
            if col not in trayectos.columns:
                trayectos[col] = 0
        
        if len(trayectos) > 0:
            cols_svc_ac = st.columns(len(trayectos))
            ci = 0
            for (via, stype), row_counts in trayectos.iterrows():
                total_t = row_counts.sum()
                xt100_c = row_counts['XT-100']
                xtm_c = row_counts['XT-M']
                sfe_c = row_counts['SFE']
                
                color = "#1565C0" if via == 1 else "#c62828"
                dot = "🔵" if via == 1 else "🔴"
                
                html_card = f"""
                <div style='border-left: 4px solid {color}; padding-left: 10px; margin-bottom: 15px;'>
                    <span style='font-size:12px; color:#666; font-weight:bold;'>{dot} {stype}</span><br>
                    <span style='font-size:24px; font-weight:bold; color:#111;'>{total_t}</span><br>
                    <span style='font-size:11px; color:#555;'>XT-100: <b style='color:#111;'>{xt100_c}</b> | XT-M: <b style='color:#111;'>{xtm_c}</b> | SFE: <b style='color:#111;'>{sfe_c}</b></span>
                </div>
                """
                if ci < len(cols_svc_ac):
                    cols_svc_ac[ci].markdown(html_card, unsafe_allow_html=True)
                ci += 1
        else:
            st.info("No hay trayectos validados aún para generar este desglose.")
            
        st.markdown("##### ⚡ Consumo Energético Acumulado por Tipo de Tren (Neto Pantógrafo)")
        e_cols = st.columns(3)
        for i, f_type in enumerate(['XT-100', 'XT-M', 'SFE']):
            tot_e = energy_by_fleet.get(f_type, 0.0)
            subset_flota = df_inic[df_inic['tipo_tren'] == f_type]
            cnt_viajes = subset_flota.shape[0]
            km_flota = subset_flota['tren_km'].sum()
            
            avg_e = (tot_e / cnt_viajes) if cnt_viajes > 0 else 0.0
            ide_flota = (tot_e / km_flota) if km_flota > 0 else 0.0
            
            html_e = f"""
            <div style='background-color:#f9f9f9; border-radius:8px; padding:15px; text-align:center; border: 1px solid #eee;'>
                <div style='font-size:14px; font-weight:bold; color:#333;'>Flota {f_type}</div>
                <div style='font-size:22px; font-weight:bold; color:#2E7D32; margin:10px 0;'>{tot_e:,.0f} kWh</div>
                <div style='font-size:12px; color:#666;'>Viajes comerciales despachados: {cnt_viajes}</div>
                <div style='font-size:13px; color:#1565C0; font-weight:bold; margin-top:5px;'>Promedio: {avg_e:,.1f} kWh/v</div>
                <div style='font-size:14px; color:#E65100; font-weight:bold; margin-top:4px;'>IDE: {ide_flota:,.2f} kWh/km</div>
            </div>
            """
            e_cols[i].markdown(html_e, unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

        km_comercial_inic = df_inic['tren_km'].sum() if not df_inic.empty else 0.0
        km_total_red = km_comercial_inic + vacio_km_total

        st.markdown("##### ⚡ Consumo Acumulado por Subestación Rectificadora (SER a 44kV)")
        if active_sers:
            ser_cols = st.columns(len(active_sers))
            for i, ser_info in enumerate(active_sers):
                s_name = ser_info[1]
                e_ser_panto = ser_accum_visual.get(s_name, 0.0)
                e_ser_44 = max(0.0, e_ser_panto) / ETA_SER_RECTIFICADOR
                ide_ser = e_ser_44 / max(1.0, km_total_red)
                html_ser = f"""
                <div style='background-color:#FFF3E0; border-radius:8px; padding:15px; text-align:center; border: 1px solid #FFCC80;'>
                    <div style='font-size:14px; font-weight:bold; color:#E65100;'>{s_name}</div>
                    <div style='font-size:22px; font-weight:bold; color:#E65100; margin:10px 0;'>{e_ser_44:,.0f} kWh</div>
                    <div style='font-size:12px; color:#666;'>Km Total Red: {km_total_red:,.3f} km</div>
                    <div style='font-size:14px; color:#C62828; font-weight:bold; margin-top:4px;'>Aporte IDE: {ide_ser:,.3f} kWh/km</div>
                </div>
                """
                ser_cols[i].markdown(html_ser, unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

        st.markdown("##### ⚡ Consumo Acumulado Subestación de Alta Tensión (SEAT 110/44kV)")
        ide_seat = seat_accum_1 / max(1.0, km_total_red)
        html_seat = f"""
        <div style='background-color:#FFFDE7; border-radius:8px; padding:15px; text-align:center; border: 1px solid #FFF59D;'>
            <div style='font-size:16px; font-weight:bold; color:#F57F17;'>SEAT EL SOL (Total Red + Pérdidas AC)</div>
            <div style='font-size:26px; font-weight:bold; color:#F57F17; margin:10px 0;'>{seat_accum_1:,.0f} kWh</div>
            <div style='font-size:13px; color:#666;'>Km Comercial: {km_comercial_inic:,.1f} km | Km Vacío: {vacio_km_total:,.3f} km</div>
            <div style='font-size:14px; color:#333; font-weight:bold; margin-top:4px;'>Km Total Red: {km_total_red:,.3f} km</div>
            <div style='font-size:16px; color:#C62828; font-weight:bold; margin-top:6px;'>IDE Global Real: {ide_seat:,.3f} kWh/km</div>
        </div>
        """
        st.markdown(html_seat, unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

        a1,a2,a3,a4,a5,a6 = st.columns(6)
        with a1: st.metric("📋 Iniciados", n_inic)
        with a2: st.metric("✅ Completados", n_comp)
        with a3: st.metric("📏 Tren-km", f"{km_ac:,.0f}")
        with a4: st.metric("⚡ kWh SERs", f"{total_ser_kwh_44kv:,.0f}")

        if prefix_key == "plan":
            pax_ac = int(df_inic['pax_abordo'].sum()) if not df_inic.empty else 0
        else:
            if not df_inic.empty:
                df_inic_pax = df_inic[df_inic['pax_row_idx'] != -1].drop_duplicates(subset=['pax_row_idx'])
                pax_ac = int(df_inic_pax['pax_abordo'].sum())
            else:
                pax_ac = 0

        with a5: st.metric("🧑‍🤝‍🧑 Pax Despachados", f"{pax_ac:,}")
        with a6: st.metric("💡 IDE Promedio (SEAT)", f"{ide_ac:.3f} kWh/km")

        if prefix_key == "mapa" or prefix_key == "plan":
            st.divider()
            st.markdown("#### 🚉 Maniobras en Vacío (Cochera El Belloto y Transiciones)")
            st.caption("Consumo físico por tránsitos no comerciales. Incluye el carrusel a Velocidad de Consigna y restricción a 20 km/h en patios. *La energía ya está sumada a la demanda total de las SERs.*")
            v1, v2, v3 = st.columns(3)
            v1.metric("Maniobras en Vacío (Carrusel)", vacio_count)
            v2.metric("Kilometraje Improductivo", f"{vacio_km_total:,.3f} Tren-km")
            v3.metric("Consumo Eléctrico Vacío", f"{vacio_kwh_total:,.0f} kWh")

        st.divider()
        st.subheader("📈 Consumo Total y Requerimientos Aguas Arriba (SER & SEAT)")

        if not df_dia.empty:
            with st.expander("📊 Resumen de Energía del Día y Comportamiento de Subestaciones", expanded=True):

                st.markdown("### 🚄 Auditoría de Consumo Operativo")
                st.caption("Análisis termodinámico detallado. Evalúa el costo energético directo separando el impacto de la tecnología (Flota) y la geografía (Trayecto).")

                if not df_dia_e.empty:
                    def agrupar_energia(df_group):
                        res = df_group.agg(
                            viajes=('_id', 'count'),
                            trac_kwh=('kwh_viaje_trac', 'sum'),
                            regen_kwh=('kwh_viaje_regen', 'sum'),
                            neto_kwh=('kwh_viaje_neto', 'sum')
                        ).reset_index()
                        res['neto_prom'] = res['neto_kwh'] / res['viajes']
                        return res

                    res_flota = agrupar_energia(df_dia_e.groupby('tipo_tren'))
                    res_flota.rename(columns={'tipo_tren': 'Flota', 'viajes': 'N° Viajes', 'trac_kwh': 'Tracción [kWh]', 'regen_kwh': 'Regen. [kWh]', 'neto_kwh': 'Neto Total [kWh]', 'neto_prom': 'Promedio [kWh/viaje]'}, inplace=True)
                    for col in ['Tracción [kWh]', 'Regen. [kWh]', 'Neto Total [kWh]', 'Promedio [kWh/viaje]']: 
                        res_flota[col] = res_flota[col].round(0).astype(int)

                    pivot_data = []
                    for (via, svc), group in df_dia_e.groupby(['Via', 'svc_type']):
                        row = {
                            'Vía': "V1" if via == 1 else "V2",
                            'Trayecto': svc,
                            'Total Viajes': len(group),
                            'Total Neto [kWh]': int(round(group['kwh_viaje_neto'].sum()))
                        }

                        for flota in ['XT-100', 'XT-M', 'SFE']:
                            sub = group[group['tipo_tren'] == flota]
                            n_v = len(sub)
                            row[f'N° {flota}'] = n_v
                            if n_v > 0:
                                tot_f = sub['kwh_viaje_neto'].sum()
                                tot_km = sub['tren_km'].sum()
                                row[f'Neto {flota} [kWh]'] = int(round(tot_f))
                                row[f'Prom. {flota} [kWh/v]'] = int(round(tot_f / n_v))
                                row[f'IDE {flota} [kWh/km]'] = round(tot_f / tot_km, 2) if tot_km > 0 else 0.0
                            else:
                                row[f'Neto {flota} [kWh]'] = 0
                                row[f'Prom. {flota} [kWh/v]'] = 0
                                row[f'IDE {flota} [kWh/km]'] = 0.0

                        pivot_data.append(row)

                    df_pivot = pd.DataFrame(pivot_data)

                    st.markdown("##### 🚆 Resumen Consolidado por Familia de Tren (Flota)")
                    st.dataframe(res_flota, use_container_width=True)

                    st.markdown("##### 🔀 Matriz Detallada: Trayectos vs Flota (Auditoría Ejecutiva con IDE)")
                    st.caption("Desglose exacto de cuántos trenes de cada familia operaron un trayecto específico, su consumo total en esa ruta, el promedio unitario y su eficiencia kilométrica (IDE).")
                    st.dataframe(df_pivot, use_container_width=True)

                st.divider()
                st.markdown("### Requerimientos Aguas Arriba (SER & SEAT)")
                sr1, sr2 = st.columns(2)
                with sr1:
                    st.info(f"**Demanda en bornes de las SER Activas (a 44 kV): {total_ser_kwh_44kv:,.0f} kWh** \n*Considera el despacho geográfico de energía y caída de tensión dinámica a 3000 Vcc.*")
                
                t_trac = df_inic['kwh_viaje_trac'].sum() if 'kwh_viaje_trac' in df_inic.columns else 0.0
                t_aux = df_inic['kwh_viaje_aux'].sum() if 'kwh_viaje_aux' in df_inic.columns else 0.0
                t_regen = df_inic['kwh_viaje_regen'].sum() if 'kwh_viaje_regen' in df_inic.columns else 0.0
                t_reostat = df_inic['kwh_reostato'].sum() if 'kwh_reostato' in df_inic.columns else 0.0
                
                with sr2:
                    st.error(f"**Inyección Total SEAT 110/44kV (Tracción Bruta): {seat_accum_1:,.0f} kWh** \n*Considera pérdidas dinámicas de transmisión AC 44kV (I²R) integradas y eficiencia del transformador de potencia (99%).*")

                fig_pie = go.Figure(data=[go.Pie(
                    labels=['Tracción', 'Auxiliar', 'Regeneración Útil', 'Pérdida Reóstato'], 
                    values=[t_trac, t_aux, t_regen, t_reostat], 
                    hole=.3,
                    marker_colors=['#1565C0', '#F9A825', '#2E7D32', '#C62828']
                )])
                fig_pie.update_layout(title="Distribución de Energía (Trenes + Vacíos)")

                df_dia_e['hora'] = (df_dia_e['t_ini'] // 60).astype(int)
                e_hora_comercial = df_dia_e.groupby('hora')[['kwh_viaje_trac', 'kwh_viaje_aux', 'kwh_viaje_regen', 'kwh_viaje_neto']].sum().reset_index()

                e_hora = e_hora_comercial

                fig_hora = go.Figure()
                fig_hora.add_trace(go.Bar(x=e_hora['hora'], y=e_hora['kwh_viaje_trac'], name='Tracción', marker_color='#1565C0'))
                fig_hora.add_trace(go.Bar(x=e_hora['hora'], y=e_hora['kwh_viaje_aux'], name='Auxiliar', marker_color='#F9A825'))
                fig_hora.add_trace(go.Bar(x=e_hora['hora'], y=-e_hora['kwh_viaje_regen'], name='Regeneración Útil', marker_color='#2E7D32'))
                fig_hora.add_trace(go.Scatter(x=e_hora['hora'], y=e_hora['kwh_viaje_neto'] / ETA_SER_RECTIFICADOR, mode='lines', name='Demanda Est. SER', line=dict(color='red', width=2, dash='dot')))
                fig_hora.update_layout(barmode='relative', title="Energía por Hora con Demanda SER", xaxis_title="Hora", yaxis_title="kWh")

                ec1, ec2 = st.columns(2)
                with ec1: st.plotly_chart(fig_pie, use_container_width=True)
                with ec2: st.plotly_chart(fig_hora, use_container_width=True)

        if st.session_state[f'play_{prefix_key}'] and "SCADA" not in modo:
            time.sleep(max(0.05, 0.3 / st.session_state.get(f'vs1_{prefix_key}', 1.0)))
            st.rerun()
