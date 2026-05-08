import numpy as np
import pandas as pd
import config

try:
    from etl_parser import get_pax_at_km_nativo
except ImportError:
    def get_pax_at_km_nativo(pax_d, km_pos, via, pax_max_fallback=0): return pax_max_fallback

# =============================================================================
# 1. OPTIMIZACIÓN EXTREMA: MATRICES PRE-CALCULADAS O(1)
# =============================================================================
_VEL_ARRAY_NORM = np.zeros(45000, dtype=float)
_VEL_ARRAY_RM = np.zeros(45000, dtype=float)
for ki, kf, _, vn, vr in getattr(config, 'SPEED_PROFILE', []):
    start_idx = int(ki)
    end_idx = min(int(kf) + 1, 45000)
    _VEL_ARRAY_NORM[start_idx:end_idx] = vn
    _VEL_ARRAY_RM[start_idx:end_idx] = vr

# 🚀 FIX DE RENDIMIENTO: Pre-cálculo de pendientes. Evita 360 millones de iteraciones inútiles.
_PEND_ARRAY_V1 = np.zeros(45000, dtype=float)
_PEND_ARRAY_V2 = np.zeros(45000, dtype=float)
try: 
    _e_km = config._ELEV_KM
    _e_m = config._ELEV_M
except:
    _e_km = [0.0, 0.7, 1.4, 2.2, 3.9, 6.0, 7.4, 8.3, 9.2, 10.2, 11.7, 19.1, 21.4, 23.3, 25.3, 26.4, 27.6, 28.5, 29.1, 30.4, 43.13]
    _e_m  = [12, 10, 10, 10, 18, 15, 12, 15, 35, 50, 55, 88, 122, 132, 142, 148, 155, 162, 175, 198, 216]

for j in range(1, len(_e_km)):
    s_idx = int(_e_km[j-1] * 1000)
    e_idx = min(int(_e_km[j] * 1000) + 1, 45000)
    dist_tramo = max(0.001, (_e_km[j] - _e_km[j-1]) * 1000.0)
    pend = ((_e_m[j] - _e_m[j-1]) / dist_tramo) * 1000.0
    _PEND_ARRAY_V1[s_idx:e_idx] = 9.81 * pend
    _PEND_ARRAY_V2[s_idx:e_idx] = -9.81 * pend

def vel_at_km(km_km, via, use_rm):
    idx = int(km_km * 1000.0)
    if 0 <= idx < 45000: return _VEL_ARRAY_RM[idx] if use_rm else _VEL_ARRAY_NORM[idx]
    return 0.0

def km_at_t(t_ini, t_fin, t, via, use_rm=False, km_orig=None, km_dest=None, nodos=None, t_arr=None):
    km_total = getattr(config, 'KM_TOTAL', 43.13)
    if nodos is not None and len(nodos) >= 2:
        if t <= nodos[0][0]: return nodos[0][1]
        if t >= nodos[-1][0]: return nodos[-1][1]
        if t_arr is None: t_arr = [n[0] for n in nodos]
        idx = np.searchsorted(t_arr, t)
        t_A, k_A = nodos[idx-1]
        t_B, k_B = nodos[idx]
        if t_A == t_B: return k_A
        return k_A + (t - t_A) * (k_B - k_A) / (t_B - t_A)
    dur = t_fin - t_ini
    if dur <= 0: return km_orig if km_orig is not None else (0.0 if via==1 else km_total)
    frac = max(0.0, min(1.0, (t - t_ini) / dur))
    ko = km_orig if km_orig is not None else (0.0 if via==1 else km_total)
    kd = km_dest if km_dest is not None else (km_total if via==1 else 0.0)
    return ko + frac * (kd - ko)

def get_train_state_and_speed(t, r_via, use_rm, km_orig, km_dest, nodos, t_arr=None):
    km_total = getattr(config, 'KM_TOTAL', 43.13)
    if not nodos or len(nodos) < 2: return "CRUISE", 60.0
    if t_arr is None: t_arr = [n[0] for n in nodos]
    if t <= t_arr[0] or t >= t_arr[-1]: return "DWELL", 0.0
    idx = np.searchsorted(t_arr, t)
    km_now = km_at_t(t_arr[idx-1], t_arr[idx], t, r_via, use_rm, nodos[idx-1][1], nodos[idx][1], None)
    vel_max = vel_at_km(km_now, r_via, use_rm)
    
    if r_via == 1 and km_now >= km_total - 0.200:
        vel_max = min(vel_max, 10.0 if km_now >= km_total - 0.100 else 20.0)
    elif r_via == 2 and km_now <= 0.200:
        vel_max = min(vel_max, 10.0 if km_now <= 0.100 else 20.0)
        
    dt_from_A = t - t_arr[idx-1]
    dt_to_B = t_arr[idx] - t
    if dt_from_A <= 1.0: return "ACCEL", vel_max
    elif dt_to_B <= 1.0: return "BRAKE", vel_max
    else: return "CRUISE", vel_max

# =============================================================================
# 6. AUXILIARES DINÁMICOS (Arquitectura BOTTOM-UP RESTAURADA)
# =============================================================================
def calcular_aux_dinamico(aux_kw_nominal, hora_decimal, pax_abordo, cap_max, estacion_anio, estado_marcha="CRUISE", f_compresor_dwell=1.03):
    """
    Suma rigurosa y física de componentes sin doble contabilización.
    Evita matemáticamente superar el 100% de la capacidad nominal.
    """
    hora_int = int(hora_decimal) % 24
    try: perfil = getattr(config, '_AUX_HVAC_HORA', {})[estacion_anio]
    except: perfil = [0.5]*24
    f_hvac = perfil[hora_int]
    
    if cap_max > 0:
        ocup = min(1.0, pax_abordo / cap_max)
        if estacion_anio == "verano": f_ocup = 1.0 + 0.05 * ocup
        elif estacion_anio == "invierno": f_ocup = 1.0 - 0.12 * ocup
        else: f_ocup = 1.0 - 0.06 * ocup
    else: f_ocup = 1.0

    # 1. Carga Base (Vital, luces, cargadores, TCMS) -> 12% del Nominal
    p_base = aux_kw_nominal * 0.12
    
    # 2. Climatización (HVAC Modulado) -> Máx 45% del Nominal
    p_clima = (aux_kw_nominal * 0.45) * f_hvac * f_ocup
    
    # 3. Ventilación Tracción (Reactiva y Dinámica) -> Protege los IGBTs y Motores
    if estado_marcha in ["BRAKE", "BRAKE_STATION", "BRAKE_OVERSPEED"]:
        p_vent = aux_kw_nominal * 0.13   # 13% en freno (máximo estrés térmico para disipar regeneración)
    elif estado_marcha == "ACCEL":
        p_vent = aux_kw_nominal * 0.068  # 6.8% en tracción
    else:
        p_vent = 0.0                     # CRUISE, COAST o DWELL (Refrigeración mínima cubierta por la base)
        
    # 4. Neumática y Puertas (Acumulador Discreto Virtual en Andén)
    factor_extra_comp = max(0.0, f_compresor_dwell - 1.0)
    if estado_marcha == "DWELL":
        p_comp = aux_kw_nominal * factor_extra_comp
    else:
        p_comp = 0.0
        
    return p_base + p_clima + p_vent + p_comp

# =============================================================================
# 7. FÍSICA TERMODINÁMICA Y LOAD FLOW (V118)
# =============================================================================
def simular_tramo_termodinamico(tipo_tren, doble, km_ini, km_fin, via_op, pct_trac, use_rm, use_pend, nodos=None, pax_dict=None, pax_abordo=0, v_consigna_override=None, maniobra=None, estacion_anio="primavera", t_ini_mins=0.0, es_vacio=False, prevenciones=None):
    f = getattr(config, 'FLOTA', {}).get(tipo_tren, {"tara_t": 86.1, "m_iner_t": 7.2, "p_max_kw": 720, "f_trac_max_kn": 110, "a_freno_ms2": 1.2, "v_freno_min": 3.81})
    km_total = getattr(config, 'KM_TOTAL', 43.13)
    pax_kg = getattr(config, 'PAX_KG', 75.0)
    eta_regen_neta = getattr(config, 'ETA_REGEN_NETA', 0.72)
    
    # Lógica de Termostato para establecer el techo nominal
    if estacion_anio == "invierno": aux_nominal_u = f.get('aux_kw_heat', f.get('aux_kw', 65.16))
    else: aux_nominal_u = f.get('aux_kw_cool', f.get('aux_kw', 58.76))
        
    f_comp_spec = f.get('f_compresor_dwell', 1.03)
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
    
    # Radar predictivo de prevenciones (TSR)
    prev_activas = [p for p in prevenciones if p['via'] == via_op] if prevenciones else []

    for i in range(len(paradas_km)-1):
        p_ini, p_fin = paradas_km[i], paradas_km[i+1]
        dist_total_tramo = abs(p_fin - p_ini) * 1000.0
        if dist_total_tramo <= 0: continue
        
        n_uni = 2 if doble else 1
        pax_mid = get_pax_at_km_nativo(pax_dict, p_ini, via_op, pax_abordo) if pax_dict else pax_abordo
        masa_kg = ((f.get('tara_t', 86.1) + f.get('m_iner_t', 7.2)) * 1000 * n_uni) + (pax_mid * pax_kg)
        
        a_freno_op = f.get('a_freno_ms2', 1.2) * 0.9 
        f_trac_max_const = f.get('f_trac_max_kn', 110.0) * 1000 * n_uni * (pct_trac / 100.0)
        p_trac_max_const = f.get('p_max_kw', 720.0) * 1000 * n_uni * (pct_trac / 100.0)
        f_freno_max_const = f.get('f_freno_max_kn', 105.0) * 1000 * n_uni
        p_freno_max_const = f.get('p_freno_max_kw', 800.0) * 1000 * n_uni 
        v_freno_min_const = f.get('v_freno_min', 3.81)
        jerk_limit = f.get('jerk_ms3', 1.3) * dt

        pos_m, dist_recorrida, v_ms, a_prev, estado_marcha = p_ini * 1000.0, 0.0, 0.0, 0.0, "ACCEL"
        
        while dist_recorrida < dist_total_tramo:
            dist_restante = dist_total_tramo - dist_recorrida
            if dist_restante < 0.1: break
            
            km_actual = (pos_m + dist_recorrida) / 1000.0 if via_op == 1 else (pos_m - dist_recorrida) / 1000.0
            v_cons_kmh = max(5.0, vel_at_km(km_actual, via_op, use_rm))
            if v_consigna_override is not None: v_cons_kmh = min(v_cons_kmh, v_consigna_override)
            
            # TOPERAS (Seguridad limitación ATC)
            if via_op == 1 and km_actual >= km_total - 0.200: v_cons_kmh = min(v_cons_kmh, 10.0 if km_actual >= km_total - 0.100 else 20.0)
            elif via_op == 2 and km_actual <= 0.200: v_cons_kmh = min(v_cons_kmh, 10.0 if km_actual <= 0.100 else 20.0)

            # 🚧 RADAR DE PREVENCIONES (Lookahead)
            if prev_activas:
                for p in prev_activas:
                    if p['km_min'] <= km_actual <= p['km_max']:
                        v_cons_kmh = min(v_cons_kmh, p['v_kmh'])
                    elif via_op == 1 and 0 < (p['km_min'] - km_actual) <= 1.5:
                        v_obj = p['v_kmh'] / 3.6
                        if v_ms > v_obj:
                            dist_a_zona = (p['km_min'] - km_actual) * 1000.0
                            a_necesaria = (v_ms**2 - v_obj**2) / (2 * dist_a_zona)
                            if a_necesaria > 0.4: v_cons_kmh = min(v_cons_kmh, p['v_kmh'])
                    elif via_op == 2 and 0 < (km_actual - p['km_max']) <= 1.5:
                        v_obj = p['v_kmh'] / 3.6
                        if v_ms > v_obj:
                            dist_a_zona = (km_actual - p['km_max']) * 1000.0
                            a_necesaria = (v_ms**2 - v_obj**2) / (2 * dist_a_zona)
                            if a_necesaria > 0.4: v_cons_kmh = min(v_cons_kmh, p['v_kmh'])

            v_kmh = v_ms * 3.6
            f_davis = ((f.get('davis_A', 1615.0) * 2) + (f.get('davis_B', 0.0) * 2 * v_kmh) + (f.get('davis_C', 0.54) * 1.35 * (v_kmh**2))) if n_uni == 2 else (f.get('davis_A', 1615.0) + f.get('davis_B', 0.0)*v_kmh + f.get('davis_C', 0.54)*(v_kmh**2))
                
            f_pend = 0.0
            if use_pend:
                idx_km = int(km_actual * 1000.0)
                if 0 <= idx_km < 45000:
                    f_pend = (_PEND_ARRAY_V1[idx_km] if via_op == 1 else _PEND_ARRAY_V2[idx_km]) * (masa_kg / 1000.0)
            
            d_freno_req = (v_ms**2) / (2 * a_freno_op) if v_ms > 0 else 0
            if dist_restante <= d_freno_req + (v_ms * dt * 1.2): estado_marcha = "BRAKE_STATION"
            elif v_kmh > v_cons_kmh + 1.5: estado_marcha = "BRAKE_OVERSPEED"
            elif estado_marcha == "ACCEL" and v_kmh >= v_cons_kmh - 0.5: estado_marcha = "COAST"
            elif estado_marcha == "COAST" and v_kmh < v_cons_kmh - 2.0: estado_marcha = "ACCEL"

            f_motor, f_regen_tramo, a_net_target = 0.0, 0.0, 0.0
            
            if estado_marcha == "BRAKE_STATION":
                f_req_freno = masa_kg * a_freno_op - f_davis - f_pend
                f_regen_tramo = min(max(0.0, f_req_freno), min(f_freno_max_const, p_freno_max_const/max(0.1, v_ms)))
                a_net_target = -a_freno_op 
            elif estado_marcha == "BRAKE_OVERSPEED":
                f_req_freno = masa_kg * 0.4 - f_davis - f_pend
                f_regen_tramo = min(max(0.0, f_req_freno), min(f_freno_max_const, p_freno_max_const/max(0.1, v_ms)))
                a_net_target = -0.4 
            elif estado_marcha == "ACCEL":
                f_motor = min(f_trac_max_const, p_trac_max_const / max(0.1, v_ms))
                a_net_target = (f_motor - f_davis - f_pend) / masa_kg
            elif estado_marcha == "COAST":
                a_net_target = (-f_davis - f_pend) / masa_kg
                
            a_net = np.clip(a_net_target, a_prev - jerk_limit, a_prev + jerk_limit)
            a_prev = a_net
            v_new = v_ms + a_net * dt
            dt_actual = dt
            
            if v_new < 0:
                dt_actual = v_ms / abs(a_net) if a_net < -0.001 else dt
                v_new = 0.0
            
            if f_motor > 0 and v_new * 3.6 > v_cons_kmh: 
                v_new = v_cons_kmh / 3.6
                
            step_m = (v_ms + v_new) / 2.0 * dt_actual
            if step_m > dist_restante: step_m = dist_restante
            
            # Escudo Anti-Stall
            if v_ms < 0.1 and v_new < 0.1 and dist_restante > 0:
                v_new = 2.0
                step_m = dist_restante
                dt_actual = step_m / 2.0
            
            if f_motor > 0: 
                eta_din = f.get('eta_motor', 0.92) * (1.0 - 0.2 * (1.0 - max(0.1, f_motor / max(1.0, f_trac_max_const)))**3)
                trc += ((f_motor * step_m) / 3_600_000.0) / eta_din
            if f_regen_tramo > 0 and v_kmh >= v_freno_min_const: 
                reg += ((f_regen_tramo * step_m) / 3_600_000.0) * eta_regen_neta
                
            # 💡 CÁLCULO AUXILIAR BOTTOM-UP: Suma estricta de componentes en kW -> transformada a kWh
            aux += (calcular_aux_dinamico(aux_nominal_u * n_uni, (t_ini_mins + t_horas * 60.0) / 60.0, pax_mid, f.get('cap_max', 398) * n_uni, estacion_anio, estado_marcha, f_comp_spec) * (dt_actual / 3600.0))
            
            t_horas += dt_actual / 3600.0
            dist_recorrida += step_m
            v_ms = v_new

        # Paradas Comerciales (DWELL)
        if i < len(paradas_km) - 2:
            dwell_h = 25.0 / 3600.0
            hora_media_dwell = (t_ini_mins + (t_horas + dwell_h / 2.0) * 60.0) / 60.0
            aux += calcular_aux_dinamico(aux_nominal_u * n_uni, hora_media_dwell, pax_abordo, f.get('cap_max', 398) * n_uni, estacion_anio, "DWELL", f_comp_spec) * dwell_h
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
            hw = min(headways) if headways else 15.0
            eta = 0.90 if hw < 5.0 else (0.75 - ((hw - 5.0) / 5.0) * 0.45 if hw < 10.0 else max(0.10, 0.30 - ((hw - 10.0) / 20.0) * 0.20))
            result[idx] = min(eta, 0.90)
    return result

def precalcular_red_electrica_v111(df_dia, pct_trac, use_rm, estacion_anio="primavera"):
    return {idx: 0.70 for idx in df_dia.index}

def calcular_termodinamica_flota_v111(df_dia, pct_trac, use_pend, use_rm, use_regen, dict_regen, estacion_anio="primavera", prevenciones=None):
    df_e = df_dia.copy()
    if df_e.empty: return df_e
    def _wrapper(r):
        trc, aux, reg_max, _, _, t_h = simular_tramo_termodinamico(
            r['tipo_tren'], r.get('doble', False), r['km_orig'], r['km_dest'], r['Via'], 
            pct_trac, use_rm, use_pend, r.get('nodos'), r.get('pax_d', {}), r.get('pax_abordo', 0), 
            None, r.get('maniobra'), estacion_anio, r.get('t_ini', 0.0), False, prevenciones
        )
        reg_util = reg_max * dict_regen.get(r.name, 1.0) if use_regen else 0.0
        return pd.Series([trc, aux, reg_util, max(0.0, reg_max - reg_util), max(0.0, trc + aux - reg_util)])
    df_e[['kwh_viaje_trac', 'kwh_viaje_aux', 'kwh_viaje_regen', 'kwh_reostato', 'kwh_viaje_neto']] = df_e.apply(_wrapper, axis=1)
    return df_e
