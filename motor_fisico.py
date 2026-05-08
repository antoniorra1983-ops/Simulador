import streamlit as st
import numpy as np
import pandas as pd
from config import *
from etl_parser import get_pax_at_km

# =============================================================================
# 1. MOTOR CINEMÁTICO TRAMO A TRAMO
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
    
    # 🔥 LO ÚNICO AGREGADO: Restricción Toperas Limache y Puerto 🔥
    if r_via == 1 and km_now >= KM_TOTAL - 0.200:
        vel_max = min(vel_max, 10.0 if km_now >= KM_TOTAL - 0.100 else 20.0)
    elif r_via == 2 and km_now <= 0.200:
        vel_max = min(vel_max, 10.0 if km_now <= 0.100 else 20.0)
        
    if dt_from_A <= 1.0: return "ACCEL", vel_max
    elif dt_to_B <= 1.0: return "BRAKE", vel_max
    else: return "CRUISE", vel_max

# =============================================================================
# 2. CÁLCULO DE AUXILIARES
# =============================================================================
def calcular_aux_dinamico(aux_kw_nominal, hora_decimal, pax_abordo, cap_max, estacion_anio, estado_marcha="CRUISE"):
    hora_int = int(hora_decimal) % 24
    perfil = _AUX_HVAC_HORA.get(estacion_anio, _AUX_HVAC_HORA["primavera"])
    f_hvac = perfil[hora_int]
    if cap_max > 0:
        ocup = min(1.0, pax_abordo / cap_max)
        if estacion_anio == "verano": f_ocup = 1.0 + 0.05 * ocup
        elif estacion_anio == "invierno": f_ocup = 1.0 - 0.12 * ocup
        else: f_ocup = 1.0 - 0.06 * ocup
    else:
        f_ocup = 1.0
    f_marcha = _FACTOR_DWELL_COMPRESOR if estado_marcha == "DWELL" else 1.0
    aux_base = aux_kw_nominal * _FRAC_BASE
    aux_hvac = aux_kw_nominal * _FRAC_HVAC * f_hvac * f_ocup * f_marcha
    return aux_base + aux_hvac

# =============================================================================
# 3. FÍSICA TERMODINÁMICA Y LOAD FLOW
# =============================================================================
def simular_tramo_termodinamico(tipo_tren, doble, km_ini, km_fin, via_op, pct_trac, use_rm, use_pend, nodos=None, pax_dict=None, pax_abordo=0, v_consigna_override=None, maniobra=None, estacion_anio="primavera", t_ini_mins=0.0, es_vacio=False):
    f = FLOTA.get(tipo_tren, FLOTA["XT-100"])
    trc, aux, reg, t_horas = 0.0, 0.0, 0.0, 0.0
    
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
            
            pax_mid = get_pax_at_km(pax_dict, km_actual, via_op, pax_abordo) if pax_dict else pax_abordo
            
            masa_kg = ((f['tara_t'] + f['m_iner_t']) * 1000 * n_uni) + (pax_mid * PAX_KG)
            
            v_cons_kmh = max(5.0, vel_at_km(km_actual, via_op, use_rm))
            if v_consigna_override is not None: v_cons_kmh = min(v_cons_kmh, v_consigna_override)
            
            # 🔥 LO ÚNICO AGREGADO: Restricción Toperas Limache y Puerto 🔥
            if via_op == 1 and km_actual >= KM_TOTAL - 0.200:
                v_cons_kmh = min(v_cons_kmh, 10.0 if km_actual >= KM_TOTAL - 0.100 else 20.0)
            elif via_op == 2 and km_actual <= 0.200:
                v_cons_kmh = min(v_cons_kmh, 10.0 if km_actual <= 0.100 else 20.0)
            
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
                for j in range(1, len(_ELEV_KM)):
                    if _ELEV_KM[j-1] <= km_actual <= _ELEV_KM[j] or (j == len(_ELEV_KM)-1 and km_actual > _ELEV_KM[j]):
                        pend = ((_ELEV_M[j] - _ELEV_M[j-1]) / max(0.001, (_ELEV_KM[j] - _ELEV_KM[j-1])*1000)) * 1000
                        f_pend = DAVIS_E_N_PERMIL * pend * (masa_kg / 1000.0) * (1.0 if via_op==1 else -1.0)
                        break
                        
            a_freno_op = f['a_freno_ms2'] * 0.9 
            d_freno_req = (v_ms**2) / (2 * a_freno_op) if v_ms > 0 else 0
            
            f_disp_trac = min(f['f_trac_max_kn']*1000*n_uni*(pct_trac/100.0), (f['p_max_kw']*1000*n_uni*(pct_trac/100.0))/max(0.1, v_ms))
            f_disp_freno = min(f['f_freno_max_kn']*1000*n_uni, (f.get('p_freno_max_kw', f['p_max_kw']*1.2)*1000*n_uni)/max(0.1, v_ms)) if v_kmh >= f['v_freno_min'] else 0.0
            
            if dist_restante <= d_freno_req + (v_ms * dt * 1.2): estado_marcha = "BRAKE_STATION"
            elif v_kmh > v_cons_kmh + 1.5: estado_marcha = "BRAKE_OVERSPEED"
            else:
                if estado_marcha == "BRAKE_OVERSPEED" and v_kmh <= v_cons_kmh: estado_marcha = "COAST"
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
                
            jerk_limit = 0.8 * dt
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
            if v_new < 0.1 and v_ms < 0.1:
                v_new = 1.0
                dt_actual = dt

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
                
            aux += (calcular_aux_dinamico(f['aux_kw'] * n_uni, (t_ini_mins + t_horas * 60.0) / 60.0, pax_mid, f.get('cap_max', 398) * n_uni, estacion_anio, estado_marcha) * (dt_actual / 3600.0))
            t_horas += dt_actual / 3600.0
            dist_recorrida += step_m
            v_ms = v_new

    n_est_mid = max(0, len(paradas_km) - 2)
    dwell_h = (n_est_mid * DWELL_DEF) / 3600.0
    hora_media_dwell = (t_ini_mins + (t_horas + dwell_h / 2.0) * 60.0) / 60.0
    aux_kw_dwell = calcular_aux_dinamico(f['aux_kw'] * (2 if doble else 1), hora_media_dwell, pax_abordo, f.get('cap_max', 398) * (2 if doble else 1), estacion_anio, "DWELL")
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
            for i in range(idx_start, idx_end):
                m = time_steps[i]
                state, v_kmh = get_train_state_and_speed(m, tr['Via'], use_rm, tr['km_orig'], tr['km_dest'], tr['nodos'], tr['t_arr'])
                pos = km_at_t(tr['t_ini'], tr['t_fin'], m, tr['Via'], use_rm, tr['km_orig'], tr['km_dest'], tr['nodos'], tr['t_arr'])
                v_ms = v_kmh / 3.6
                p_aux_kw = calcular_aux_dinamico(f['aux_kw'] * n_uni, m / 60.0, tr['pax_abordo'], f.get('cap_max', 398) * n_uni, estacion_anio, state)
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
