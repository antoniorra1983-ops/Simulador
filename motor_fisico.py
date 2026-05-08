import streamlit as st
import numpy as np
import pandas as pd
from config import *
from etl_parser import get_pax_at_km_nativo

# =============================================================================
# 1. MOTOR CINEMÁTICO TRAMO A TRAMO (OPTIMIZACIÓN EXTREMA V134)
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
    
    # 🏁 MEJORA: RESTRICCIÓN DE SEGURIDAD EN ESTACIONES TERMINALES (Toperas)
    if r_via == 1 and km_now >= KM_TOTAL - 0.200:
        vel_max = min(vel_max, 10.0 if km_now >= KM_TOTAL - 0.100 else 20.0)
    elif r_via == 2 and km_now <= 0.200:
        vel_max = min(vel_max, 10.0 if km_now <= 0.100 else 20.0)
        
    if dt_from_A <= 1.0: return "ACCEL", vel_max
    elif dt_to_B <= 1.0: return "BRAKE", vel_max
    else: return "CRUISE", vel_max

def calcular_aux_dinamico(aux_kw_nominal, hora_decimal, pax_abordo, cap_max, estacion_anio, estado_marcha="CRUISE", f_compresor_dwell=1.03):
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
        
    if estado_marcha == "DWELL":
        f_mb, f_mh = 1.0, f_compresor_dwell
    elif estado_marcha in ["BRAKE", "BRAKE_STATION", "BRAKE_OVERSPEED"]: 
        f_mb, f_mh = 1.05, 1.0
    elif estado_marcha == "ACCEL": 
        f_mb, f_mh = 0.95, 1.0
    elif estado_marcha == "COAST":
        f_mb, f_mh = 0.90, 1.0
    else: 
        f_mb, f_mh = 1.0, 1.0
        
    aux_base = aux_kw_nominal * _FRAC_BASE * f_mb
    aux_hvac = aux_kw_nominal * _FRAC_HVAC * f_hvac * f_ocup * f_mh
    return aux_base + aux_hvac

# =============================================================================
# 3. FÍSICA TERMODINÁMICA (LOOP INVARIANT CODE MOTION - ULTRA RÁPIDO)
# =============================================================================
def simular_tramo_termodinamico(tipo_tren, doble, km_ini, km_fin, via_op, pct_trac, use_rm, use_pend, nodos=None, pax_dict=None, pax_abordo=0, v_consigna_override=None, maniobra=None, estacion_anio="primavera", t_ini_mins=0.0, es_vacio=False, prevenciones=None):
    f = FLOTA.get(tipo_tren, FLOTA["XT-100"])
    
    # ⚡ OPTIMIZACIÓN: Extraemos variables nominales FUERA de los bucles while
    if estacion_anio == "invierno": aux_nominal_unidad = f.get('aux_kw_heat', 65.16)
    else: aux_nominal_unidad = f.get('aux_kw_cool', 58.76)
        
    f_compresor_especifico = f.get('f_compresor_dwell', 1.03)
    trc, aux, reg, t_horas = 0.0, 0.0, 0.0, 0.0
    k_s, k_e = km_ini, km_fin
    
    paradas_km = [n[1] for n in nodos] if nodos else [k_s, k_e]
    k_min, k_max = min(k_s, k_e), max(k_s, k_e)
    paradas_km = sorted(list(set([k for k in paradas_km if k_min <= k <= k_max] + [k_s, k_e])), reverse=(via_op == 2))
    
    dt = 1.0  
    tiene_elevacion = bool(_ELEV_KM and _ELEV_M and len(_ELEV_KM) == len(_ELEV_M))

    for i in range(len(paradas_km)-1):
        p_ini, p_fin = paradas_km[i], paradas_km[i+1]
        dist_total_tramo = abs(p_fin - p_ini) * 1000.0
        if dist_total_tramo <= 0: continue
        
        n_uni = 2 if doble else 1
        pax_mid = get_pax_at_km_nativo(pax_dict, p_ini, via_op, pax_abordo) if pax_dict else pax_abordo
        masa_kg = ((f['tara_t'] + f['m_iner_t']) * 1000 * n_uni) + (pax_mid * PAX_KG)
        
        a_freno_op = f.get('a_freno_ms2', 1.2) * 0.9 
        f_trac_max_const = f.get('f_trac_max_kn', 110.0) * 1000 * n_uni * (pct_trac / 100.0)
        p_trac_max_const = f.get('p_max_kw', 720.0) * 1000 * n_uni * (pct_trac / 100.0)
        f_freno_max_const = f.get('f_freno_max_kn', 105.0) * 1000 * n_uni
        p_freno_max_const = 800.0 * 1000 * n_uni 
        v_freno_min_const = f.get('v_freno_min', 3.81)
        jerk_limit = 1.3 * dt
        eta_motor_const = f.get('eta_motor', 0.92)

        pos_m, dist_recorrida, v_ms, a_prev, estado_marcha = p_ini * 1000.0, 0.0, 0.0, 0.0, "ACCEL"
        
        while dist_recorrida < dist_total_tramo:
            dist_restante = dist_total_tramo - dist_recorrida
            if dist_restante < 0.1: break
            km_actual = (pos_m + dist_recorrida) / 1000.0 if via_op == 1 else (pos_m - dist_recorrida) / 1000.0
            
            v_cons_kmh = max(5.0, vel_at_km(km_actual, via_op, use_rm))
            if v_consigna_override is not None: v_cons_kmh = min(v_cons_kmh, v_consigna_override)
            
            # Restricción Toperas
            if via_op == 1 and km_actual >= KM_TOTAL - 0.200: v_cons_kmh = min(v_cons_kmh, 10.0 if km_actual >= KM_TOTAL - 0.100 else 20.0)
            elif via_op == 2 and km_actual <= 0.200: v_cons_kmh = min(v_cons_kmh, 10.0 if km_actual <= 0.100 else 20.0)
            
            if prevenciones:
                for p in prevenciones:
                    if p['via'] == via_op and p['km_min'] <= km_actual <= p['km_max']: v_cons_kmh = min(v_cons_kmh, p['v_kmh'])

            v_kmh = v_ms * 3.6
            f_davis = ((f.get('davis_A', 1615.0) * (2 if n_uni == 2 else 1)) + 
                       (f.get('davis_B', 0.0) * (2 if n_uni == 2 else 1) * v_kmh) + 
                       (f.get('davis_C', 0.54) * (1.35 if n_uni == 2 else 1.0) * (v_kmh**2)))
                
            f_pend = 0.0
            if use_pend and tiene_elevacion:
                idx_p = np.searchsorted(_ELEV_KM, km_actual) - 1
                if 0 <= idx_p < len(_ELEV_KM) - 1:
                    pend = ((_ELEV_M[idx_p+1] - _ELEV_M[idx_p]) / max(0.001, (_ELEV_KM[idx_p+1] - _ELEV_KM[idx_p])*1000)) * 1000
                    f_pend = DAVIS_E_N_PERMIL * pend * (masa_kg / 1000.0) * (1.0 if via_op==1 else -1.0)
            
            d_freno_req = (v_ms**2) / (2 * a_freno_op) if v_ms > 0 else 0
            if dist_restante <= d_freno_req + (v_ms * dt * 1.2): estado_marcha = "BRAKE_STATION"
            elif v_kmh > v_cons_kmh + 1.5: estado_marcha = "BRAKE_OVERSPEED"
            elif estado_marcha == "ACCEL" and v_kmh >= v_cons_kmh - 0.5: estado_marcha = "COAST"
            elif estado_marcha == "COAST" and v_kmh < v_cons_kmh - 2.0: estado_marcha = "ACCEL"

            f_motor, f_regen_tramo, a_net_target = 0.0, 0.0, 0.0
            if estado_marcha == "BRAKE_STATION":
                f_req_f = max(0.0, masa_kg * a_freno_op - f_davis - f_pend)
                f_regen_tramo = min(f_req_f, min(f_freno_max_const, p_freno_max_const/max(0.1, v_ms)))
                a_net_target = max(-a_freno_op, (-f_regen_tramo - f_davis - f_pend) / masa_kg)
            elif estado_marcha == "ACCEL":
                f_motor = min(f_trac_max_const, p_trac_max_const / max(0.1, v_ms))
                a_net_target = (f_motor - f_davis - f_pend) / masa_kg
            elif estado_marcha == "COAST":
                a_net_target = (-f_davis - f_pend) / masa_kg
                
            a_net = np.clip(a_net_target, a_prev - jerk_limit, a_prev + jerk_limit)
            a_prev = a_net
            v_new = max(0.0, v_ms + a_net * dt)
            if f_motor > 0 and v_new * 3.6 > v_cons_kmh: v_new = v_cons_kmh / 3.6
                
            step_m = (v_ms + v_new) / 2.0 * dt
            if step_m > dist_restante: step_m = dist_restante
            
            if f_motor > 0: trc += ((f_motor * step_m) / 3_600_000.0) / eta_motor_const
            if f_regen_tramo > 0 and v_kmh >= v_freno_min_const: reg += ((f_regen_tramo * step_m) / 3_600_000.0) * 0.72
                
            aux += (calcular_aux_dinamico(aux_nominal_unidad * n_uni, (t_ini_mins + t_horas * 60.0) / 60.0, pax_mid, f.get('cap_max', 398) * n_uni, estacion_anio, estado_marcha, f_compresor_especifico) * (dt / 3600.0))
            t_horas += dt / 3600.0
            dist_recorrida += step_m
            v_ms = v_new

        if i < len(paradas_km) - 2:
            aux += calcular_aux_dinamico(aux_nominal_unidad * n_uni, (t_ini_mins + t_horas * 60.0)/60.0, pax_mid, f.get('cap_max', 398) * n_uni, estacion_anio, "DWELL", f_compresor_especifico) * (25.0 / 3600.0)
            t_horas += (25.0 / 3600.0)

    return trc, aux, reg, 0.0, max(0.0, trc + aux - reg), t_horas
