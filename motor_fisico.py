import streamlit as st
import pandas as pd
import numpy as np

from config import *
from etl_parser import calc_tren_km_real_general, get_pax_at_km

# =============================================================================
# 1. MOTOR CINEMÁTICO TRAMO A TRAMO (GEOMETRÍA Y PERFILES DE VELOCIDAD)
# =============================================================================
def _build_profile(use_rm, via):
    segs = SPEED_PROFILE if via == 1 else list(reversed(SPEED_PROFILE))
    km_pts, t_pts, cum_t = [], [], 0.0
    for ki, kf, dm, vn, vr in segs:
        v = max(5.0, vr if use_rm else vn)
        km_pts.append(ki if via == 1 else kf)
        t_pts.append(cum_t)
        cum_t += (dm / 1000.0) / v * 3600.0
    last = SPEED_PROFILE[-1] if via == 1 else SPEED_PROFILE[0]
    km_pts.append(last[1] if via == 1 else last[0])
    t_pts.append(cum_t)
    return np.array(km_pts, float), np.array(t_pts, float)

_PROF = {(v, r): _build_profile(r, v) for v in [1, 2] for r in [False, True]}
_PROF_SORTED = {}
for k, v in _PROF.items(): 
    if k[0] == 1: _PROF_SORTED[k] = (v[0], v[1])
    else: _PROF_SORTED[k] = (v[0][::-1].copy(), v[1][::-1].copy())

_VEL_ARRAY_NORM = np.zeros(45000, dtype=float)
_VEL_ARRAY_RM = np.zeros(45000, dtype=float)
for ki, kf, _, vn, vr in SPEED_PROFILE:
    start_idx = int(ki)
    end_idx = min(int(kf) + 1, 45000)
    _VEL_ARRAY_NORM[start_idx:end_idx] = vn
    _VEL_ARRAY_RM[start_idx:end_idx] = vr

def vel_at_km(km_km, via, use_rm):
    idx = int(km_km * 1000.0)
    if 0 <= idx < 45000: return _VEL_ARRAY_RM[idx] if use_rm else _VEL_ARRAY_NORM[idx]
    return 0.0

def km_at_t(t_ini, t_fin, t, via, use_rm=False, km_orig=None, km_dest=None, nodos=None, t_arr=None):
    if nodos is not None and len(nodos) >= 2:
        if t <= nodos[0][0]: return nodos[0][1]
        if t >= nodos[-1][0]: return nodos[-1][1]
        if t_arr is None: t_arr = [n[0] for n in nodos]
        idx = np.searchsorted(t_arr, t)
        t_A, k_A = nodos[idx-1]
        t_B, k_B = nodos[idx]
        if t_A == t_B or k_A == k_B: return k_A 
        frac = (t - t_A) / (t_B - t_A)
        km_sorted, t_sorted = _PROF_SORTED[(via, use_rm)]
        t_prof_A = float(np.interp(k_A * 1000.0, km_sorted, t_sorted))
        t_prof_B = float(np.interp(k_B * 1000.0, km_sorted, t_sorted))
        t_prof_target = t_prof_A + frac * (t_prof_B - t_prof_A)
        km_arr, t_prof_arr = _PROF[(via, use_rm)]
        km_m = float(np.interp(t_prof_target, t_prof_arr, km_arr))
        return max(0.0, min(km_m / 1000.0, KM_TOTAL))
        
    dur = t_fin - t_ini
    if dur <= 0: return km_orig if km_orig is not None else (0.0 if via==1 else KM_TOTAL)
    frac = max(0.0, min(1.0, (t - t_ini) / dur))
    
    if km_orig is None: km_orig = 0.0 if via == 1 else KM_TOTAL
    if km_dest is None: km_dest = KM_TOTAL if via == 1 else 0.0
    
    km_sorted, t_sorted = _PROF_SORTED[(via, use_rm)]
    t_at_orig = float(np.interp(km_orig * 1000.0, km_sorted, t_sorted))
    t_at_dest = float(np.interp(km_dest * 1000.0, km_sorted, t_sorted))
    t_prof = t_at_orig + frac * (t_at_dest - t_at_orig)
    
    km_arr, t_arr_prof = _PROF[(via, use_rm)]
    km_m = float(np.interp(t_prof, t_arr_prof, km_arr))
    return max(0.0, min(km_m / 1000.0, KM_TOTAL))

def get_train_state_and_speed(t, r_via, use_rm, km_orig, km_dest, nodos, t_arr=None):
    if not nodos or len(nodos) < 2: return "CRUISE", 60.0
    if t_arr is None: t_arr = [n[0] for n in nodos]
    if t <= t_arr[0] or t >= t_arr[-1]: return "DWELL", 0.0
    idx = np.searchsorted(t_arr, t)
    t_A, t_B = t_arr[idx-1], t_arr[idx]
    dt_from_A, dt_to_B = t - t_A, t_B - t
    km_now = km_at_t(t_A, t_B, t, r_via, use_rm, km_orig, km_dest, nodos, t_arr)
    vel_max = vel_at_km(km_now, r_via, use_rm)
    if dt_from_A <= 1.0: return "ACCEL", vel_max
    elif dt_to_B <= 1.0: return "BRAKE", vel_max
    else: return "CRUISE", vel_max

# =============================================================================
# 2. CÁLCULO DE AUXILIARES DINÁMICOS (LÓGICA BOTTOM-UP SIN DOBLE CONTEO)
# =============================================================================
def calcular_aux_dinamico(aux_kw_nominal, hora_decimal, pax_abordo, cap_max, estacion_anio, estado_marcha="CRUISE", p_vent_max=7.6):
    hora_int = int(hora_decimal) % 24
    
    try: perfil = AUX_HVAC_HORA.get(estacion_anio, AUX_HVAC_HORA.get("primavera", [0.5]*24))
    except NameError: 
        try: perfil = _AUX_HVAC_HORA.get(estacion_anio, _AUX_HVAC_HORA.get("primavera", [0.5]*24))
        except: perfil = [0.5] * 24
        
    f_hvac = perfil[hora_int] if len(perfil) > hora_int else 0.5
    
    if cap_max > 0:
        ocup = min(1.0, pax_abordo / cap_max)
        if estacion_anio == "verano": f_ocup = 1.0 + 0.05 * ocup
        elif estacion_anio == "invierno": f_ocup = 1.0 - 0.12 * ocup
        else: f_ocup = 1.0 - 0.06 * ocup
    else:
        f_ocup = 1.0
        
    try: frac_base = FRAC_BASE
    except NameError: 
        try: frac_base = _FRAC_BASE
        except: frac_base = 0.12
    
    try: frac_hvac = FRAC_HVAC
    except NameError: 
        try: frac_hvac = _FRAC_HVAC
        except: frac_hvac = 0.45

    # 💡 LÓGICA BOTTOM-UP ESTRICTA: Sumatoria de cargas discretas (Cero Doble Conteo)
    
    # 1. Carga Base Vital (TCMS, Luces, Enchufes) -> Fija al 12%
    p_base = aux_kw_nominal * frac_base
    
    # 2. Climatización (HVAC Salón y Cabina) -> Modulada hasta el 45% máximo
    p_clima = (aux_kw_nominal * frac_hvac) * f_hvac * f_ocup
    
    # 3. Ventilación Tracción (Totalmente Reactiva y Desacoplada del Clima)
    if estado_marcha in ["BRAKE", "BRAKE_STATION", "BRAKE_OVERSPEED"]:
        p_vent = p_vent_max         # 100% ventilación para enfriar el freno regenerativo brutal
    elif estado_marcha == "ACCEL":
        p_vent = p_vent_max * 0.52  # Load Shedding / Estrangulamiento en aceleración (~4 kW)
    elif estado_marcha in ["COAST", "DWELL"]:
        p_vent = 0.0                # Tren relajado térmicamente o estacionado, ventiladores en OFF o Mínimo
    else:
        p_vent = p_vent_max * 0.20  # Mantenimiento en velocidad de crucero
        
    return p_base + p_clima + p_vent

# =============================================================================
# 3. FÍSICA TERMODINÁMICA Y LOAD FLOW (INCLUYE ACUMULADOR NEUMÁTICO)
# =============================================================================
def simular_tramo_termodinamico(tipo_tren, doble, km_ini, km_fin, via_op, pct_trac, use_rm, use_pend, nodos=None, pax_dict=None, pax_abordo=0, v_consigna_override=None, maniobra=None, estacion_anio="primavera", t_ini_mins=0.0, es_vacio=False):
    f = FLOTA.get(tipo_tren, FLOTA["XT-100"])
    
    # 💡 LECTURA ESTACIONAL DE PLENA CARGA
    if estacion_anio == "invierno":
        aux_nominal_unidad = f.get('aux_kw_heat', f.get('aux_kw', 65.16))
    else:
        aux_nominal_unidad = f.get('aux_kw_cool', f.get('aux_kw', 58.76))
        
    trc, aux, reg, t_horas = 0.0, 0.0, 0.0, 0.0
    
    # 💡 INYECCIÓN FÍSICA: Acumulador Neumático Virtual con Modulación (Soft-Load)
    mrp_bar = 10.0
    compresor_on = False
    p_comp = f.get('p_compresor_kw', 3.68)
    
    # Tasa de llenado referencial: 15kW llenaba a 0.05 bar/s (tarda 40s en recuperar 2 bares)
    # Conservación de masa: Si bajamos la potencia a 3.68kW, llenamos más lento proporcionalmente.
    tasa_rec = 0.05 * (p_comp / 15.0) 
    
    k_s, k_e = km_ini, km_fin
    dst = abs(k_e - k_s)
    if dst <= 0: return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    
    paradas_km = [n[1] for n in nodos] if nodos else [k_s, k_e]
    k_min, k_max = min(k_s, k_e), max(k_s, k_e)
    paradas_km = [k for k in paradas_km if k_min <= k <= k_max]
    if k_s not in paradas_km: paradas_km.append(k_s)
    if k_e not in paradas_km: paradas_km.append(k_e)
    paradas_km = list(set(paradas_km))
    paradas_km.sort(reverse=(via_op == 2))
    
    pax_dict = pax_dict or {}
    dt = 1.0  
    
    for i in range(len(paradas_km)-1):
        p_ini, p_fin = paradas_km[i], paradas_km[i+1]
        dist_total_tramo = abs(p_fin - p_ini) * 1000.0
        if dist_total_tramo <= 0: continue
        
        pos_m = p_ini * 1000.0
        dist_recorrida = 0.0
        v_ms = 0.0
        a_prev = 0.0 
        estado_marcha = "ACCEL"
        
        while dist_recorrida < dist_total_tramo:
            dist_restante = dist_total_tramo - dist_recorrida
            if dist_restante < 0.1: break
            
            km_actual = (pos_m + dist_recorrida) / 1000.0 if via_op == 1 else (pos_m - dist_recorrida) / 1000.0
            
            es_doble = doble
            if maniobra in ['CORTE_BTO', 'CORTE_PU_SA_BTO'] and km_actual > 25.3: es_doble = False
            elif maniobra == 'CORTE_SA' and km_actual > 29.1: es_doble = False
            elif maniobra == 'ACOPLE_BTO' and km_actual < 25.3: es_doble = False
            elif maniobra == 'ACOPLE_SA' and km_actual < 29.1: es_doble = False
            
            n_uni = 2 if es_doble else 1
            p_vent_max = f.get('p_vent_trac_kw', 7.6) * n_uni
            pax_mid = get_pax_at_km(pax_dict, km_actual, via_op, pax_abordo) if pax_dict else pax_abordo
            masa_kg = ((f['tara_t'] + f['m_iner_t']) * 1000 * n_uni) + (pax_mid * PAX_KG)
            
            v_cons_kmh = max(5.0, vel_at_km(km_actual, via_op, use_rm))
            if v_consigna_override is not None: v_cons_kmh = min(v_cons_kmh, v_consigna_override)
            
            if es_vacio:
                min_dist_est_m = min([abs(km_actual - k) for k in KM_ACUM]) * 1000.0
                v_30_ms = 30.0 / 3.6
                d_brake_to_30 = ((v_ms**2 - v_30_ms**2) / (2 * (f['a_freno_ms2'] * 0.85))) if v_ms > v_30_ms else 0.0
                dist_to_next_station_m = 9999000.0
                for est_k in KM_ACUM:
                    if via_op == 1 and est_k > km_actual + 0.01:
                        dist_to_next_station_m = min(dist_to_next_station_m, (est_k - km_actual)*1000.0)
                    elif via_op == 2 and est_k < km_actual - 0.01:
                        dist_to_next_station_m = min(dist_to_next_station_m, (km_actual - est_k)*1000.0)
                if dist_to_next_station_m <= d_brake_to_30 + 50.0 or min_dist_est_m <= 120.0:
                    v_cons_kmh = min(v_cons_kmh, 30.0)
                
            v_kmh = v_ms * 3.6
            if n_uni == 2: f_davis = (f['davis_A'] * 2) + (f['davis_B'] * 2 * v_kmh) + (f['davis_C'] * 1.35 * (v_kmh**2))
            else: f_davis = f['davis_A'] + f['davis_B']*v_kmh + f['davis_C']*(v_kmh**2)
                
            f_pend = 0.0
            if use_pend:
                try: elev_km, elev_m = ELEV_KM, ELEV_M
                except: elev_km, elev_m = _ELEV_KM, _ELEV_M
                    
                for j in range(1, len(elev_km)):
                    if elev_km[j-1] <= km_actual <= elev_km[j] or (j == len(elev_km)-1 and km_actual > elev_km[j]):
                        pend = ((elev_m[j] - elev_m[j-1]) / max(0.001, (elev_km[j] - elev_km[j-1])*1000)) * 1000
                        f_pend = DAVIS_E_N_PERMIL * pend * (masa_kg / 1000.0) * (1.0 if via_op==1 else -1.0)
                        break
                        
            a_freno_op = f['a_freno_ms2'] * 0.9 
            d_freno_req = (v_ms**2) / (2 * a_freno_op) if v_ms > 0 else 0
            
            f_disp_trac = min(f['f_trac_max_kn']*1000*n_uni*(pct_trac/100.0), (f['p_max_kw']*1000*n_uni*(pct_trac/100.0))/max(0.1, v_ms))
            f_disp_freno = min(f['f_freno_max_kn']*1000*n_uni, (f.get('p_freno_max_kw', f['p_max_kw']*1.2)*1000*n_uni)/max(0.1, v_ms)) if v_kmh >= f['v_freno_min'] else 0.0
            
            if dist_restante <= d_freno_req + (v_ms * dt * 1.2): estado_marcha = "BRAKE_STATION"
            elif v_kmh > v_cons_kmh + 1.5: estado_marcha = "BRAKE_OVERSPEED"
            elif estado_marcha == "BRAKE_OVERSPEED" and v_kmh <= v_cons_kmh: estado_marcha = "COAST"
            elif estado_marcha == "ACCEL" and v_kmh >= v_cons_kmh - 0.5: estado_marcha = "COAST"
            elif estado_marcha == "COAST" and v_kmh < v_cons_kmh - 2.0: estado_marcha = "ACCEL"
            elif estado_marcha not in ["ACCEL", "COAST", "BRAKE_STATION", "BRAKE_OVERSPEED"]: estado_marcha = "ACCEL"

            f_motor, f_regen_tramo, a_net_target = 0.0, 0.0, 0.0
            if estado_marcha == "BRAKE_STATION":
                f_req_freno = max(0.0, masa_kg * a_freno_op - f_davis - f_pend)
                f_regen_tramo = min(f_req_freno, f_disp_freno)
                a_net_target = max(-a_freno_op, (-f_regen_tramo - f_davis - f_pend) / masa_kg)
            elif estado_marcha == "BRAKE_OVERSPEED":
                f_req_freno = max(0.0, masa_kg * 0.4 - f_davis - f_pend)
                f_regen_tramo = min(f_req_freno, f_disp_freno)
                a_net_target = min((-f_regen_tramo - f_davis - f_pend) / masa_kg, -0.15)
            elif estado_marcha == "ACCEL":
                f_motor = f_disp_trac
                a_net_target = (f_motor - f_davis - f_pend) / masa_kg
            elif estado_marcha == "COAST":
                a_net_target = (-f_davis - f_pend) / masa_kg
                
            jerk_limit = f.get('jerk_ms3', 0.8) * dt
            if a_net_target > a_prev + jerk_limit: a_net = a_prev + jerk_limit
            elif a_net_target < a_prev - jerk_limit: a_net = a_prev - jerk_limit
            else: a_net = a_net_target
            a_prev = a_net
            
            v_new, dt_actual = v_ms + a_net * dt, dt
            if v_new < 0:
                dt_actual = v_ms / abs(a_net) if a_net < -0.001 else dt
                v_new = 0.0
                
            if f_motor > 0 and v_new * 3.6 > v_cons_kmh:
                v_new = v_cons_kmh / 3.6
                a_req = (v_new - v_ms) / dt_actual if dt_actual > 0 else 0
                f_motor = max(0.0, min(masa_kg * a_req + f_davis + f_pend, f_disp_trac))
                
            if v_new < 0.5 and dist_restante < 2.0: break
            if v_new < 0.1 and v_ms < 0.1: v_new, dt_actual = 1.0, dt

            step_m = (v_ms + v_new) / 2.0 * dt_actual
            if step_m > dist_restante:
                step_m = dist_restante
                if v_ms + v_new > 0: dt_actual = step_m / ((v_ms + v_new) / 2.0)
            if step_m < 0.1: step_m = 0.5 
                
            if f_motor > 0: 
                eta_din = f.get('eta_motor', 0.92) * (1.0 - 0.2 * (1.0 - max(0.1, f_motor / max(1.0, f_disp_trac)))**3)
                trc += ((f_motor * step_m) / 3_600_000.0) / eta_din
            if f_regen_tramo > 0 and v_kmh >= f['v_freno_min']: 
                reg += ((f_regen_tramo * step_m) / 3_600_000.0) * ETA_REGEN_NETA
                
            # 💡 RECUPERACIÓN NEUMÁTICA SILENCIOSA EN TRÁNSITO (SOFT-LOAD)
            if compresor_on:
                mrp_bar += tasa_rec * dt_actual
                aux += (p_comp * n_uni * (dt_actual / 3600.0))
                if mrp_bar >= 10.0:
                    mrp_bar = 10.0
                    compresor_on = False

            # 💡 CÁLCULO BASE + CLIMA + VENTILADORES EN MOVIMIENTO
            hora_actual = (t_ini_mins + t_horas * 60.0) / 60.0
            aux += (calcular_aux_dinamico(aux_nominal_unidad * n_uni, hora_actual, pax_mid, f.get('cap_max', 398) * n_uni, estacion_anio, estado_marcha, p_vent_max) * (dt_actual / 3600.0))
            t_horas += dt_actual / 3600.0
            dist_recorrida += step_m
            v_ms = v_new

        # 💡 DETENCIÓN EN ANDÉN: Consumo Eléctrico de Puertas y Gasto Neumático
        if i < len(paradas_km) - 2:
            dwell_s = 25.0
            p_vent_max = f.get('p_vent_trac_kw', 7.6) * n_uni
            
            # Gasto de aire en cilindros de freno
            mrp_bar -= 0.3
            if mrp_bar <= 8.0:
                compresor_on = True
                
            # PULSO ELÉCTRICO DE PUERTAS UNILATERALES (3 SEGUNDOS)
            aux += (f.get('p_puertas_kw', 0.9) * n_uni * (3.0 / 3600.0))
            
            # Auxiliares HVAC + Base en DWELL (Ventiladores apagados)
            hora_media_dwell = (t_ini_mins + (t_horas + (dwell_s / 2.0) / 3600.0) * 60.0) / 60.0
            aux_kw_dwell = calcular_aux_dinamico(aux_nominal_unidad * n_uni, hora_media_dwell, pax_abordo, f.get('cap_max', 398) * n_uni, estacion_anio, "DWELL", p_vent_max)
            aux += aux_kw_dwell * (dwell_s / 3600.0)
            
            # Recuperación Neumática Silenciosa en Andén
            if compresor_on:
                rec_bar = tasa_rec * dwell_s
                if mrp_bar + rec_bar >= 10.0:
                    time_to_10 = (10.0 - mrp_bar) / tasa_rec
                    aux += (p_comp * n_uni * (time_to_10 / 3600.0))
                    mrp_bar = 10.0
                    compresor_on = False
                else:
                    mrp_bar += rec_bar
                    aux += (p_comp * n_uni * (dwell_s / 3600.0))
                    
            t_horas += (dwell_s / 3600.0)

    # Dwell final en estación de término
    dwell_h = (max(0, len(paradas_km) - 2) * 25.0) / 3600.0
    hora_media_dwell = (t_ini_mins + (t_horas + dwell_h / 2.0) * 60.0) / 60.0
    p_vent_max = f.get('p_vent_trac_kw', 7.6) * (2 if doble else 1)
    aux_kw_dwell = calcular_aux_dinamico(aux_nominal_unidad * (2 if doble else 1), hora_media_dwell, pax_abordo, f.get('cap_max', 398) * (2 if doble else 1), estacion_anio, "DWELL", p_vent_max)
    aux += aux_kw_dwell * dwell_h
    t_horas += dwell_h
    
    return trc, aux, reg, 0.0, max(0.0, trc + aux - reg), t_horas


def calcular_receptividad_por_headway(df_dia: pd.DataFrame) -> dict:
    if df_dia.empty: return {}
    result = {}
    for via in [1, 2]:
        sub = df_dia[df_dia["Via"] == via].sort_values("t_ini").copy()
        if sub.empty: continue
        indices = list(sub.index)
        t_ini_vals = sub["t_ini"].values
        for i, idx in enumerate(indices):
            headways = []
            if i > 0: headways.append(t_ini_vals[i] - t_ini_vals[i-1])
            if i < len(indices)-1: headways.append(t_ini_vals[i+1] - t_ini_vals[i])
            if not headways: 
                result[idx] = 0.10
                continue
            hw = min(headways)
            if hw < 5.0: eta = 0.90
            elif hw < 10.0: eta = 0.75 - ((hw - 5.0) / 5.0) * 0.45
            else: eta = max(0.10, 0.30 - ((hw - 10.0) / 20.0) * 0.20)
            result[idx] = min(eta, 0.90)
    return result

@st.cache_data(show_spinner="Simulando malla eléctrica y receptividad...")
def precalcular_red_electrica_v111(df_dia, pct_trac, use_rm, estacion_anio="primavera"):
    regen_util_per_trip = {idx: 0.0 for idx in df_dia.index}
    braking_ticks_per_trip = {idx: 0.0 for idx in df_dia.index} 
    if df_dia.empty: return regen_util_per_trip
    t_min = int(df_dia['t_ini'].min())
    t_max = int(df_dia['t_fin'].max())
    time_steps = np.arange(t_min, t_max + 1, 10.0 / 60.0)
    
    for via_ in [1, 2]:
        via_trains = df_dia[df_dia['Via'] == via_]
        if via_trains.empty: continue
        trains_data = []
        for idx, r in via_trains.iterrows():
            nodos = r.get('nodos')
            trains_data.append({
                'idx': idx, 't_ini': r['t_ini'], 't_fin': r['t_fin'], 'Via': r['Via'],
                'km_orig': r['km_orig'], 'km_dest': r['km_dest'], 'nodos': nodos,
                't_arr': [n[0] for n in nodos] if nodos and len(nodos) >= 2 else None,
                'tipo_tren': r.get('tipo_tren', 'XT-100'), 'doble': r.get('doble', False), 'pax_abordo': r.get('pax_abordo', 0)
            })
        braking_by_idx = [[] for _ in range(len(time_steps))]
        accel_by_idx = [[] for _ in range(len(time_steps))]
        
        for tr in trains_data:
            idx_start = np.searchsorted(time_steps, max(t_min, tr['t_ini']))
            idx_end = np.searchsorted(time_steps, min(t_max, tr['t_fin']), side='right')
            f = FLOTA.get(tr['tipo_tren'], FLOTA["XT-100"])
            n_uni = 2 if tr['doble'] else 1
            masa_kg = ((f['tara_t'] + f['m_iner_t']) * 1000 * n_uni) + (tr['pax_abordo'] * PAX_KG)
            eta_m = f.get('eta_motor', 0.92)
            
            if estacion_anio == "invierno":
                aux_nominal_unidad = f.get('aux_kw_heat', f.get('aux_kw', 65.16))
            else:
                aux_nominal_unidad = f.get('aux_kw_cool', f.get('aux_kw', 58.76))
            
            p_vent_max = f.get('p_vent_trac_kw', 7.6) * n_uni
            
            for i in range(idx_start, idx_end):
                m = time_steps[i]
                state, v_kmh = get_train_state_and_speed(m, tr['Via'], use_rm, tr['km_orig'], tr['km_dest'], tr['nodos'], tr['t_arr'])
                pos = km_at_t(tr['t_ini'], tr['t_fin'], m, tr['Via'], use_rm, tr['km_orig'], tr['km_dest'], tr['nodos'], tr['t_arr'])
                v_ms = v_kmh / 3.6
                
                # 💡 LLAMADA BOTTOM-UP AL AUXILIAR
                p_aux_kw = calcular_aux_dinamico(aux_nominal_unidad * n_uni, m / 60.0, tr['pax_abordo'], f.get('cap_max', 398) * n_uni, estacion_anio, state, p_vent_max)
                
                f_davis = ((f['davis_A'] * 2) + (f['davis_B'] * 2 * v_kmh) + (f['davis_C'] * 1.35 * (v_kmh**2))) if n_uni == 2 else (f['davis_A'] + f['davis_B']*v_kmh + f['davis_C']*(v_kmh**2))
                if state in ("BRAKE", "BRAKE_STATION", "BRAKE_OVERSPEED"):
                    f_req_freno = max(0.0, masa_kg * (f['a_freno_ms2'] * 0.9) - f_davis)
                    f_disp_freno = min(f['f_freno_max_kn']*1000*n_uni, (f.get('p_freno_max_kw', f['p_max_kw']*1.2)*1000*n_uni)/max(0.1, v_ms)) if v_kmh >= f['v_freno_min'] else 0.0
                    p_gen_kw = ((min(f_req_freno, f_disp_freno) * v_ms) / 1000.0 * ETA_REGEN_NETA) - p_aux_kw
                    if p_gen_kw > 0: braking_by_idx[i].append((tr['idx'], pos, p_gen_kw))
                    braking_ticks_per_trip[tr['idx']] += 1
                elif state in ("ACCEL", "CRUISE"):
                    p_dem_kw = p_aux_kw
                    if state == "ACCEL": 
                        p_dem_kw += (((min(f['f_trac_max_kn']*1000*n_uni*(pct_trac/100.0), (f['p_max_kw']*1000*n_uni*(pct_trac/100.0))/max(0.1, v_ms)) if v_ms > 0 else f['f_trac_max_kn']*1000*n_uni*(pct_trac/100.0)) * v_ms) / 1000.0 / eta_m)
                    elif state == "CRUISE" and f_davis > 0: 
                        p_dem_kw += (((f_davis * v_ms) / 1000.0) / eta_m)
                    accel_by_idx[i].append((tr['idx'], pos, p_dem_kw))
                    
        for i in range(len(time_steps)):
            if not braking_by_idx[i] or not accel_by_idx[i]: continue
            current_demands = {a[0]: a[2] for a in accel_by_idx[i]}
            for b_idx, b_pos, p_gen in braking_by_idx[i]:
                available = [a for a in accel_by_idx[i] if current_demands[a[0]] > 0]
                if not available: break 
                a_idx, a_pos, _ = min(available, key=lambda x: abs(x[1] - b_pos))
                if abs(a_pos - b_pos) <= LAMBDA_REGEN_KM * 2:
                    p_transferred = min(p_gen * (ETA_MAX * np.exp(-abs(a_pos - b_pos) / LAMBDA_REGEN_KM)), current_demands[a_idx])
                    current_demands[a_idx] -= p_transferred
                    regen_util_per_trip[b_idx] += (p_transferred / p_gen)
                    
    for idx in df_dia.index: 
        regen_util_per_trip[idx] = min(1.0, regen_util_per_trip[idx] / braking_ticks_per_trip[idx]) if braking_ticks_per_trip[idx] > 0 else 0.0
    return regen_util_per_trip

@st.cache_data(show_spinner="Integrando Termodinámica de Flota...")
def calcular_termodinamica_flota_v111(df_dia, pct_trac, use_pend, use_rm, use_regen, dict_regen, estacion_anio="primavera"):
    df_e = df_dia.copy()
    if df_e.empty: return df_e
    def _wrapper_energia(r):
        trc, aux, reg_panto_max, _, _, t_h = simular_tramo_termodinamico(
            r['tipo_tren'], r.get('doble', False), r['km_orig'], r['km_dest'], r['Via'], 
            pct_trac, use_rm, use_pend, r.get('nodos'), r.get('pax_d', {}), r.get('pax_abordo', 0), 
            None, r.get('maniobra'), estacion_anio, r.get('t_ini', 0.0)
        )
        reg_util = reg_panto_max * dict_regen.get(r.name, 1.0) if use_regen else 0.0
        return pd.Series([trc, aux, reg_util, max(0.0, reg_panto_max - reg_util), max(0.0, trc + aux - reg_util)])
    df_e[['kwh_viaje_trac', 'kwh_viaje_aux', 'kwh_viaje_regen', 'kwh_reostato', 'kwh_viaje_neto']] = df_e.apply(_wrapper_energia, axis=1)
    return df_e

# =============================================================================
# 4. PLANIFICADOR
# =============================================================================
@st.cache_data(show_spinner="Integrando física y demanda de pasajeros...")
def procesar_planificador_reactivo(df_sint, df_px_filtered, estacion_anio_plan, pct_trac, use_rm, use_pend, use_regen, tipo_regen, pax_promedio_viaje=150):
    viajes_completos = []
    perfiles_por_servicio = {}
    perfiles_por_via = {}
    
    if not df_px_filtered.empty:
        for via in [1, 2]:
            sub_via = df_px_filtered[df_px_filtered['Via'] == via]
            if not sub_via.empty:
                pd_dict = {c: int(round(sub_via[c].mean())) for c in PAX_COLS}
                pd_dict['CargaMax_Promedio'] = int(round(sub_via['CargaMax'].mean()))
                perfiles_por_via[via] = pd_dict
                
        if 'Tren_Clean' in df_px_filtered.columns:
            for tren, group in df_px_filtered.groupby('Tren_Clean'):
                if str(tren).strip() == '': continue
                pd_dict = {c: int(round(group[c].mean())) for c in PAX_COLS}
                pd_dict['CargaMax_Promedio'] = int(round(group['CargaMax'].mean()))
                perfiles_por_servicio[str(tren)] = pd_dict

    for idx, r in df_sint.iterrows():
        via_tren = r['Via']
        t_ini_tren = r['t_ini']
        num_srv = str(r.get('num_servicio', '')).strip()
        
        pax_arr_viaje = {c: 0 for c in PAX_COLS}
        pax_calculado = 0
        cap_m = FLOTA[r['tipo_tren']].get('cap_max', 398) * (2 if r['doble'] else 1)
        
        if perfiles_por_servicio and num_srv in perfiles_por_servicio:
            perfil_srv = perfiles_por_servicio[num_srv]
            pax_calculado = perfil_srv.get('CargaMax_Promedio', 0)
            pax_arr_viaje = {k: v for k, v in perfil_srv.items() if k != 'CargaMax_Promedio'}
        elif not df_px_filtered.empty:
            sub_v = df_px_filtered[df_px_filtered['Via'] == via_tren].copy()
            if not sub_v.empty:
                sub_v['diff'] = sub_v['t_ini_p'].apply(lambda x: min(abs(float(x) - float(t_ini_tren)), 1440 - abs(float(x) - float(t_ini_tren))))
                idx_min = sub_v['diff'].idxmin()
                if sub_v.loc[idx_min, 'diff'] <= 20:
                    best_t = sub_v.loc[idx_min, 't_ini_p']
                    best_group = sub_v[sub_v['t_ini_p'] == best_t]
                    pax_calculado = int(round(best_group['CargaMax'].mean()))
                    pax_arr_viaje = {c: int(round(best_group[c].mean())) for c in PAX_COLS}
                else:
                    pax_dict_dinamico = perfiles_por_via.get(via_tren, {})
                    pax_abordo_base = pax_dict_dinamico.get('CargaMax_Promedio', pax_promedio_viaje)
                    f_gauss = 0.2 + 0.8 * np.exp(-0.5 * ((t_ini_tren - 450)/60)**2) + 0.8 * np.exp(-0.5 * ((t_ini_tren - 1080)/90)**2)
                    pax_calculado = int(pax_abordo_base * f_gauss * 1.5)
                    if pax_dict_dinamico:
                        pax_arr_viaje = {k: int(v * f_gauss * 1.5) for k, v in pax_dict_dinamico.items() if k != 'CargaMax_Promedio'}
                    else:
                        pax_arr_viaje = {c: int(pax_calculado / len(PAX_COLS)) for c in PAX_COLS}
            else:
                f_gauss = 0.2 + 0.8 * np.exp(-0.5 * ((t_ini_tren - 450)/60)**2) + 0.8 * np.exp(-0.5 * ((t_ini_tren - 1080)/90)**2)
                pax_calculado = int(pax_promedio_viaje * f_gauss * 1.5)
                pax_arr_viaje = {c: int(pax_calculado / len(PAX_COLS)) for c in PAX_COLS}
        else:
            f_gauss = 0.2 + 0.8 * np.exp(-0.5 * ((t_ini_tren - 450)/60)**2) + 0.8 * np.exp(-0.5 * ((t_ini_tren - 1080)/90)**2)
            pax_calculado = int(pax_promedio_viaje * f_gauss * 1.5)
            pax_arr_viaje = {c: int(pax_calculado / len(PAX_COLS)) for c in PAX_COLS}

        pax_calculado = min(pax_calculado, cap_m)
        pax_arr_viaje = {k: min(v, cap_m) for k, v in pax_arr_viaje.items()}

        trc_v, aux_v, reg_v, _, _, t_h = simular_tramo_termodinamico(
            r['tipo_tren'], r['doble'], r['km_orig'], r['km_dest'], r['Via'], 
            pct_trac, use_rm, use_pend, r['nodos'], pax_arr_viaje, pax_calculado, 
            None, None, estacion_anio_plan, r['t_ini']
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
        
    df_sint_e = calcular_termodinamica_flota_v111(df_sint_final, pct_trac, use_pend, use_rm, use_regen, dict_regen_sint, estacion_anio_plan)
    return df_sint_final, df_sint_e
