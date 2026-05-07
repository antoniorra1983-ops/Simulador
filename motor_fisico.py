import streamlit as st
import pandas as pd
import numpy as np

# Importación segura para evitar NameErrors en la nube
try:
    import config
except ImportError:
    pass

# Funciones de extracción defensiva
def _get_val(name, default):
    try: return getattr(config, name, default)
    except Exception: return default

# =============================================================================
# 1. MOTOR CINEMÁTICO TRAMO A TRAMO (GEOMETRÍA Y PERFILES DE VELOCIDAD)
# =============================================================================
def _build_profile(use_rm, via):
    segs_base = _get_val('SPEED_PROFILE', [])
    segs = segs_base if via == 1 else list(reversed(segs_base))
    km_pts, t_pts, cum_t = [], [], 0.0
    
    if not segs:
        return np.array([0.0]), np.array([0.0])
        
    for ki, kf, dm, vn, vr in segs:
        v = max(5.0, vr if use_rm else vn)
        km_pts.append(ki if via == 1 else kf)
        t_pts.append(cum_t)
        cum_t += (dm / 1000.0) / v * 3600.0
        
    last = segs[-1] if via == 1 else segs[0]
    km_pts.append(last[1] if via == 1 else last[0])
    t_pts.append(cum_t)
    return np.array(km_pts, float), np.array(t_pts, float)

_PROF = {(v, r): _build_profile(r, v) for v in [1, 2] for r in [False, True]}
_PROF_SORTED = {}
for k, v in _PROF.items(): 
    if k[0] == 1: 
        _PROF_SORTED[k] = (v[0], v[1])
    else: 
        _PROF_SORTED[k] = (v[0][::-1].copy(), v[1][::-1].copy())

_VEL_ARRAY_NORM = np.zeros(45000, dtype=float)
_VEL_ARRAY_RM = np.zeros(45000, dtype=float)

segs_base_init = _get_val('SPEED_PROFILE', [])
for ki, kf, _, vn, vr in segs_base_init:
    start_idx = int(ki)
    end_idx = min(int(kf) + 1, 45000)
    _VEL_ARRAY_NORM[start_idx:end_idx] = vn
    _VEL_ARRAY_RM[start_idx:end_idx] = vr

def vel_at_km(km_km, via, use_rm):
    idx = int(km_km * 1000.0)
    if 0 <= idx < 45000: 
        return _VEL_ARRAY_RM[idx] if use_rm else _VEL_ARRAY_NORM[idx]
    return 0.0

def km_at_t(t_ini, t_fin, t, via, use_rm=False, km_orig=None, km_dest=None, nodos=None, t_arr=None):
    km_total_limit = _get_val('KM_TOTAL', 43.13)
    
    if nodos is not None and len(nodos) >= 2:
        if t <= nodos[0][0]: return nodos[0][1]
        if t >= nodos[-1][0]: return nodos[-1][1]
        if t_arr is None: t_arr = [n[0] for n in nodos]
        idx = np.searchsorted(t_arr, t)
        
        nodo_A = nodos[idx-1]
        nodo_B = nodos[idx]
        t_A = nodo_A[0] if isinstance(nodo_A, tuple) else nodo_A
        k_A = nodo_A[1] if isinstance(nodo_A, tuple) else nodo_A
        t_B = nodo_B[0] if isinstance(nodo_B, tuple) else nodo_B
        k_B = nodo_B[1] if isinstance(nodo_B, tuple) else nodo_B

        if t_A == t_B or k_A == k_B: return k_A 
        frac = (t - t_A) / (t_B - t_A)
        km_sorted, t_sorted = _PROF_SORTED[(via, use_rm)]
        t_prof_A = float(np.interp(k_A * 1000.0, km_sorted, t_sorted))
        t_prof_B = float(np.interp(k_B * 1000.0, km_sorted, t_sorted))
        t_prof_target = t_prof_A + frac * (t_prof_B - t_prof_A)
        km_arr, t_prof_arr = _PROF[(via, use_rm)]
        km_m = float(np.interp(t_prof_target, t_prof_arr, km_arr))
        return max(0.0, min(km_m / 1000.0, km_total_limit))
        
    dur = t_fin - t_ini
    if dur <= 0: return km_orig if km_orig is not None else (0.0 if via==1 else km_total_limit)
    frac = max(0.0, min(1.0, (t - t_ini) / dur))
    
    if km_orig is None: km_orig = 0.0 if via == 1 else km_total_limit
    if km_dest is None: km_dest = km_total_limit if via == 1 else 0.0
    
    km_sorted, t_sorted = _PROF_SORTED[(via, use_rm)]
    t_at_orig = float(np.interp(km_orig * 1000.0, km_sorted, t_sorted))
    t_at_dest = float(np.interp(km_dest * 1000.0, km_sorted, t_sorted))
    t_prof = t_at_orig + frac * (t_at_dest - t_at_orig)
    
    km_arr, t_arr_prof = _PROF[(via, use_rm)]
    km_m = float(np.interp(t_prof, t_arr_prof, km_arr))
    return max(0.0, min(km_m / 1000.0, km_total_limit))

def get_train_state_and_speed(t, r_via, use_rm, km_orig, km_dest, nodos, t_arr=None):
    if not nodos or len(nodos) < 2: return "CRUISE", 60.0
    if t_arr is None: t_arr = [n[0] for n in nodos]
    if t <= t_arr[0] or t >= t_arr[-1]: return "DWELL", 0.0
    idx = np.searchsorted(t_arr, t)
    t_A, t_B = t_arr[idx-1], t_arr[idx]
    dt_from_A, dt_to_B = t - t_A, t_B - t
    km_now = km_at_t(t_A, t_B, t, r_via, use_rm, km_orig, km_dest, nodos, t_arr)
    vel_max = vel_at_km(km_now, r_via, use_rm)
    
    km_total_limit = _get_val('KM_TOTAL', 43.13)
    if r_via == 1 and km_now >= km_total_limit - 0.200:
        if km_now >= km_total_limit - 0.100:
            vel_max = min(vel_max, 10.0)
        else:
            vel_max = min(vel_max, 20.0)
    elif r_via == 2 and km_now <= 0.200:
        if km_now <= 0.100:
            vel_max = min(vel_max, 10.0)
        else:
            vel_max = min(vel_max, 20.0)
            
    if dt_from_A <= 1.0: return "ACCEL", vel_max
    elif dt_to_B <= 1.0: return "BRAKE", vel_max
    else: return "CRUISE", vel_max

# =============================================================================
# 2. CÁLCULO DE AUXILIARES (MODELO BOTTOM-UP INDEPENDIENTE)
# =============================================================================
def calcular_aux_dinamico(f_flota, n_uni, hora_decimal, pax_abordo, estacion_anio, estado_marcha, is_precalc=False):
    hora_int = int(hora_decimal) % 24
    perfil_dict = _get_val('AUX_HVAC_HORA', {})
    perfil = perfil_dict.get(estacion_anio, perfil_dict.get("primavera", [0.5]*24)) if isinstance(perfil_dict, dict) else [0.5]*24
    f_hvac = perfil[hora_int] if len(perfil) > hora_int else 0.5
    
    cap_max = f_flota.get('cap_max', 398) * n_uni
    ocup = min(1.0, pax_abordo / cap_max) if cap_max > 0 else 0.0
    
    if estacion_anio == "verano": f_ocup = 1.0 + 0.05 * ocup
    elif estacion_anio == "invierno": f_ocup = 1.0 - 0.12 * ocup
    else: f_ocup = 1.0 - 0.06 * ocup

    techo_base = f_flota.get('aux_kw_cool', 58.76) * n_uni
    p_base = techo_base * 0.12
    
    techo_clima = f_flota.get('aux_kw_heat', 65.16) if estacion_anio == 'invierno' else f_flota.get('aux_kw_cool', 58.76)
    p_clima = (techo_clima * n_uni * 0.45) * f_hvac * f_ocup
    
    p_vent_max = f_flota.get('p_vent_trac_kw', 7.6) * n_uni
    if estado_marcha in ["BRAKE", "BRAKE_STATION", "BRAKE_OVERSPEED"]: 
        p_vent = p_vent_max * 1.0  
    elif estado_marcha == "ACCEL": 
        p_vent = p_vent_max * 0.5  
    else: 
        p_vent = p_vent_max * 0.1  
        
    p_total = p_base + p_clima + p_vent
    
    if is_precalc:
        p_total += f_flota.get('p_compresor_kw', 3.68) * n_uni * 0.20
        
    return p_total

# =============================================================================
# 3. FÍSICA TERMODINÁMICA Y LOAD FLOW 
# =============================================================================
def simular_tramo_termodinamico(tipo_tren, doble, km_ini, km_fin, via_op, pct_trac, use_rm, use_pend, nodos=None, pax_dict=None, pax_abordo=0, v_consigna_override=None, maniobra=None, estacion_anio="primavera", t_ini_mins=0.0, es_vacio=False, prevenciones=None, **kwargs):
    
    flota_dict = _get_val('FLOTA', {})
    f = flota_dict.get(tipo_tren, flota_dict.get("XT-100", {
        "tara_t": 86.1, "m_iner_t": 7.20, "a_freno_ms2": 1.2, "davis_A": 1615.0, 
        "davis_B": 0.0, "davis_C": 0.5458, "f_trac_max_kn": 110.0, 
        "p_max_kw": 720.0, "v_freno_min": 3.81, "jerk_ms3": 1.3
    }))
    
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
    
    km_acum_list = _get_val('KM_ACUM', [])
    pax_kg_val = _get_val('PAX_KG', 75.0)
    eta_regen_val = _get_val('ETA_REGEN_NETA', 0.85)

    def _safe_get_pax(km_val, via_val, d_dict, def_val):
        try:
            from etl_parser import get_pax_at_km
            return get_pax_at_km(d_dict, km_val, via_val, def_val)
        except: return def_val

    mrp_bar = 10.0
    compresor_on = False
    pax_prev = pax_abordo

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
            
            pax_mid = pax_abordo
            if pax_dict and sum(pax_dict.values()) > 0:
                pax_mid = _safe_get_pax(km_actual, via_op, pax_dict, pax_abordo)

            masa_kg = ((f['tara_t'] + f['m_iner_t']) * 1000 * n_uni) + (pax_mid * pax_kg_val)
            
            v_cons_kmh = max(5.0, vel_at_km(km_actual, via_op, use_rm))
            if v_consigna_override is not None: v_cons_kmh = min(v_cons_kmh, v_consigna_override)
            
            try:
                km_total_limit = _get_val('KM_TOTAL', 43.13)
                if via_op == 1 and km_actual >= km_total_limit - 0.200:
                    if km_actual >= km_total_limit - 0.100:
                        v_cons_kmh = min(v_cons_kmh, 10.0)
                    else:
                        v_cons_kmh = min(v_cons_kmh, 20.0)
                elif via_op == 2 and km_actual <= 0.200:
                    if km_actual <= 0.100:
                        v_cons_kmh = min(v_cons_kmh, 10.0)
                    else:
                        v_cons_kmh = min(v_cons_kmh, 20.0)
            except Exception:
                pass
            
            if prevenciones:
                for p in prevenciones:
                    if p['via'] == via_op:
                        if p['km_min'] <= km_actual <= p['km_max']:
                            v_cons_kmh = min(v_cons_kmh, p['v_kmh'])
                        else:
                            d_to_rest = -1.0
                            if via_op == 1 and km_actual < p['km_min']: d_to_rest = (p['km_min'] - km_actual) * 1000.0
                            elif via_op == 2 and km_actual > p['km_max']: d_to_rest = (km_actual - p['km_max']) * 1000.0
                            
                            if 0 < d_to_rest < 1500.0: 
                                v_rest_ms = p['v_kmh'] / 3.6
                                if v_ms > v_rest_ms:
                                    a_freno_pred = f.get('a_freno_ms2', 1.2) * 0.75
                                    d_freno = (v_ms**2 - v_rest_ms**2) / (2 * a_freno_pred)
                                    if d_to_rest <= d_freno + (v_ms * dt * 2.0):
                                        v_cons_kmh = min(v_cons_kmh, p['v_kmh'])

            if es_vacio:
                min_dist_est_m = min([abs(km_actual - k) for k in km_acum_list]) * 1000.0 if km_acum_list else 0.0
                v_10_ms = 10.0 / 3.6
                d_brake_to_10 = ((v_ms**2 - v_10_ms**2) / (2 * (f.get('a_freno_ms2', 1.2) * 0.85))) if v_ms > v_10_ms else 0.0
                
                dist_to_next_station_m = 9999000.0
                for est_k in km_acum_list:
                    if via_op == 1 and est_k > km_actual + 0.01:
                        dist_to_next_station_m = min(dist_to_next_station_m, (est_k - km_actual)*1000.0)
                    elif via_op == 2 and est_k < km_actual - 0.01:
                        dist_to_next_station_m = min(dist_to_next_station_m, (km_actual - est_k)*1000.0)
                
                if dist_to_next_station_m <= d_brake_to_10 + 50.0 or min_dist_est_m <= 120.0:
                    v_cons_kmh = min(v_cons_kmh, 10.0)
            
            v_kmh = v_ms * 3.6
            if n_uni == 2: f_davis = (f['davis_A'] * 2) + (f['davis_B'] * 2 * v_kmh) + (f['davis_C'] * 1.35 * (v_kmh**2))
            else: f_davis = f['davis_A'] + f['davis_B']*v_kmh + f['davis_C']*(v_kmh**2)
                
            f_pend = 0.0
            if use_pend:
                elev_km = _get_val('ELEV_KM', [])
                elev_m = _get_val('ELEV_M', [])
                if elev_km and elev_m and len(elev_km) == len(elev_m):
                    for j in range(1, len(elev_km)):
                        if elev_km[j-1] <= km_actual <= elev_km[j] or (j == len(elev_km)-1 and km_actual > elev_km[j]):
                            pend = ((elev_m[j] - elev_m[j-1]) / max(0.001, (elev_km[j] - elev_km[j-1])*1000)) * 1000
                            davis_e_n = _get_val('DAVIS_E_N_PERMIL', 9.81)
                            f_pend = davis_e_n * pend * (masa_kg / 1000.0) * (1.0 if via_op==1 else -1.0)
                            break
                        
            a_freno_op = f.get('a_freno_ms2', 1.2) * 0.9 
            d_freno_req = (v_ms**2) / (2 * a_freno_op) if v_ms > 0 else 0
            
            f_disp_trac = min(f.get('f_trac_max_kn', 110.0)*1000*n_uni*(pct_trac/100.0), (f.get('p_max_kw', 720.0)*1000*n_uni*(pct_trac/100.0))/max(0.1, v_ms))
            f_disp_freno = min(f.get('f_freno_max_kn', 105.0)*1000*n_uni, (f.get('p_freno_max_kw', f.get('p_max_kw', 720.0)*1.2)*1000*n_uni)/max(0.1, v_ms)) if v_kmh >= f.get('v_freno_min', 3.81) else 0.0
            
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
                
            jerk_limit = f.get('jerk_ms3', 1.3) * dt
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
            if f_regen_tramo > 0 and v_kmh >= f.get('v_freno_min', 3.81): 
                reg += ((f_regen_tramo * step_m) / 3_600_000.0) * eta_regen_val
                
            if mrp_bar <= 8.0:
                compresor_on = True
            
            if compresor_on:
                p_comp_inst = f.get('p_compresor_kw', 3.68) * n_uni
                mrp_bar += 0.0122 * dt_actual  
                if mrp_bar >= 10.0:
                    mrp_bar = 10.0
                    compresor_on = False
            else:
                p_comp_inst = 0.0
                
            hora_actual = (t_ini_mins + t_horas * 60.0) / 60.0
            aux_kw_inst = calcular_aux_dinamico(f, n_uni, hora_actual, pax_mid, estacion_anio, estado_marcha) + p_comp_inst
            aux += (aux_kw_inst * (dt_actual / 3600.0))
            
            t_horas += dt_actual / 3600.0
            dist_recorrida += step_m
            v_ms = v_new

        if i < len(paradas_km) - 2:
            dwell_s = _get_val('DWELL_DEF', 25.0)
            
            mrp_bar -= 0.3
            
            pax_mid = _safe_get_pax(paradas_km[i+1], via_op, pax_dict, pax_abordo)
            if f.get('tipo_tren', 'XT-100') in ['XT-M', 'SFE']:
                pax_diff = max(0, pax_mid - pax_prev)
                mrp_bar -= (pax_diff * 0.002)
            pax_prev = pax_mid
            
            p_puertas = f.get('p_puertas_kw', 0.9) * n_uni
            aux += (p_puertas * 3.0 / 3600.0)
            
            for _ in range(int(dwell_s)):
                if mrp_bar <= 8.0: compresor_on = True
                
                if compresor_on:
                    p_comp_inst = f.get('p_compresor_kw', 3.68) * n_uni
                    mrp_bar += 0.0122 * 1.0
                    if mrp_bar >= 10.0:
                        mrp_bar = 10.0
                        compresor_on = False
                else:
                    p_comp_inst = 0.0
                    
                hora_actual = (t_ini_mins + t_horas * 60.0) / 60.0
                p_aux_dwell = calcular_aux_dinamico(f, n_uni, hora_actual, pax_mid, estacion_anio, "DWELL") + p_comp_inst
                aux += p_aux_dwell * (1.0 / 3600.0)
                t_horas += (1.0 / 3600.0)

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
    
    eta_regen_val = _get_val('ETA_REGEN_NETA', 0.85)
    lambda_val = _get_val('LAMBDA_REGEN_KM', 5.0)
    eta_max_val = _get_val('ETA_MAX', 0.70)
    pax_kg_val = _get_val('PAX_KG', 75.0)
    flota_dict = _get_val('FLOTA', {})
    
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
            
            f = flota_dict.get(tr['tipo_tren'], flota_dict.get("XT-100", {"tara_t":86.1, "m_iner_t":7.2, "davis_A":1615, "davis_B":0, "davis_C":0.5, "p_max_kw":720, "f_trac_max_kn":110}))
            n_uni = 2 if tr['doble'] else 1
            masa_kg = ((f['tara_t'] + f['m_iner_t']) * 1000 * n_uni) + (tr['pax_abordo'] * pax_kg_val)
            eta_m = f.get('eta_motor', 0.92)
            
            for i in range(idx_start, idx_end):
                m = time_steps[i]
                state, v_kmh = get_train_state_and_speed(m, tr['Via'], use_rm, tr['km_orig'], tr['km_dest'], tr['nodos'], tr['t_arr'])
                pos = km_at_t(tr['t_ini'], tr['t_fin'], m, tr['Via'], use_rm, tr['km_orig'], tr['km_dest'], tr['nodos'], tr['t_arr'])
                v_ms = v_kmh / 3.6
                
                p_aux_kw = calcular_aux_dinamico(f, n_uni, m / 60.0, tr['pax_abordo'], estacion_anio, state, is_precalc=True)
                
                f_davis = ((f['davis_A'] * 2) + (f['davis_B'] * 2 * v_kmh) + (f['davis_C'] * 1.35 * (v_kmh**2))) if n_uni == 2 else (f['davis_A'] + f['davis_B']*v_kmh + f['davis_C']*(v_kmh**2))
                if state in ("BRAKE", "BRAKE_STATION", "BRAKE_OVERSPEED"):
                    f_req_freno = max(0.0, masa_kg * (f.get('a_freno_ms2', 1.2) * 0.9) - f_davis)
                    f_disp_freno = min(f.get('f_freno_max_kn', 105.0)*1000*n_uni, (f.get('p_freno_max_kw', f.get('p_max_kw', 720.0)*1.2)*1000*n_uni)/max(0.1, v_ms)) if v_kmh >= f.get('v_freno_min', 3.81) else 0.0
                    p_gen_kw = ((min(f_req_freno, f_disp_freno) * v_ms) / 1000.0 * eta_regen_val) - p_aux_kw
                    if p_gen_kw > 0: braking_by_idx[i].append((tr['idx'], pos, p_gen_kw))
                    braking_ticks_per_trip[tr['idx']] += 1
                elif state in ("ACCEL", "CRUISE"):
                    p_dem_kw = p_aux_kw
                    if state == "ACCEL": 
                        p_trac_disp = f.get('p_max_kw', 720.0)*1000*n_uni*(pct_trac/100.0)
                        f_trac_disp = min(f.get('f_trac_max_kn', 110.0)*1000*n_uni*(pct_trac/100.0), p_trac_disp/max(0.1, v_ms)) if v_ms > 0 else f.get('f_trac_max_kn', 110.0)*1000*n_uni*(pct_trac/100.0)
                        p_dem_kw += ((f_trac_disp * v_ms) / 1000.0 / eta_m)
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
                dist = abs(a_pos - b_pos)
                if dist <= lambda_val * 2:
                    p_transferred = min(p_gen * (eta_max_val * np.exp(-dist / lambda_val)), current_demands[a_idx])
                    current_demands[a_idx] -= p_transferred
                    regen_util_per_trip[b_idx] += (p_transferred / p_gen)
                    
    for idx in df_dia.index: 
        regen_util_per_trip[idx] = min(1.0, regen_util_per_trip[idx] / braking_ticks_per_trip[idx]) if braking_ticks_per_trip[idx] > 0 else 0.0
    return regen_util_per_trip

@st.cache_data(show_spinner="Integrando Termodinámica de Flota...")
def calcular_termodinamica_flota_v111(df_dia, pct_trac, use_pend, use_rm, use_regen, dict_regen, estacion_anio="primavera", prevenciones=None):
    df_e = df_dia.copy()
    if df_e.empty: return df_e
    def _wrapper_energia(r):
        trc, aux, reg_panto_max, _, _, t_h = simular_tramo_termodinamico(
            r['tipo_tren'], r.get('doble', False), r['km_orig'], r['km_dest'], r['Via'], 
            pct_trac, use_rm, use_pend, r.get('nodos'), r.get('pax_d', {}), r.get('pax_abordo', 0), 
            None, r.get('maniobra'), estacion_anio, r.get('t_ini', 0.0), es_vacio=False, prevenciones=prevenciones
        )
        reg_util = reg_panto_max * dict_regen.get(r.name, 1.0) if use_regen else 0.0
        return pd.Series([trc, aux, reg_util, max(0.0, reg_panto_max - reg_util), max(0.0, trc + aux - reg_util)])
    df_e[['kwh_viaje_trac', 'kwh_viaje_aux', 'kwh_viaje_regen', 'kwh_reostato', 'kwh_viaje_neto']] = df_e.apply(_wrapper_energia, axis=1)
    return df_e
