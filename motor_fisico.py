import numpy as np
import pandas as pd
from datetime import datetime

# Pannalaknib iti Cloud: Liklikan ti NameErrors no dalusanna ti Streamlit (Cloud Defensive Shield)
try:
    import config
except ImportError:
    pass

def _get_val(name, default):
    try: return getattr(config, name, default)
    except Exception: return default

# Natalged a panangala iti lohika dagiti pasahero manipud iti ETL (Safe import of pax logic)
try:
    from etl_parser import get_pax_at_km_nativo
except ImportError:
    def get_pax_at_km_nativo(pax_d, km_pos, via, pax_max_fallback=0): 
        return pax_max_fallback

# =============================================================================
# NALABES A PANANGPASAYAAT: NAKARKULO A MATRIS (EXTREME OPTIMIZATION: PRE-CALCULATED MATRICES)
# =============================================================================
_FRAC_BASE = _get_val('FRAC_BASE', 0.30)
_FRAC_HVAC = _get_val('FRAC_HVAC', 0.70)
_AUX_HVAC_HORA = _get_val('AUX_HVAC_HORA', {})
if not _AUX_HVAC_HORA:
    _AUX_HVAC_HORA = {
        "verano": [0.60,0.55,0.55,0.55,0.58,0.65, 0.72,0.78,0.83,0.88,0.92,0.95, 0.98,1.00,1.00,0.98,0.95,0.90, 0.85,0.80,0.75,0.70,0.67,0.63],
        "otoño": [0.40,0.38,0.37,0.37,0.38,0.42, 0.48,0.52,0.56,0.60,0.63,0.65, 0.66,0.66,0.65,0.63,0.60,0.57, 0.53,0.50,0.47,0.44,0.42,0.41],
        "invierno": [0.72,0.70,0.68,0.68,0.70,0.74, 0.80,0.84,0.86,0.85,0.82,0.78, 0.75,0.73,0.72,0.73,0.76,0.80, 0.82,0.80,0.78,0.76,0.74,0.73],
        "primavera": [0.42,0.40,0.39,0.39,0.41,0.46, 0.53,0.58,0.63,0.68,0.72,0.75, 0.77,0.78,0.77,0.74,0.70,0.66, 0.61,0.57,0.53,0.49,0.46,0.44]
    }
_ETA_REGEN_NETA = _get_val('ETA_REGEN_NETA', 0.72)
_LAMBDA_REGEN_KM = _get_val('LAMBDA_REGEN_KM', 5.0)
_ETA_MAX = _get_val('ETA_MAX', 0.7)
_PAX_KG = _get_val('PAX_KG', 75.0)
_DWELL_DEF = _get_val('DWELL_DEF', 25.0)
_SER_DATA = _get_val('SER_DATA', [(4.9, "SER PO"), (12.7, "SER ES"), (25.5, "SER EB"), (28.7, "SER VA")])

_VEL_ARRAY_NORM = np.zeros(45000, dtype=float)
_VEL_ARRAY_RM = np.zeros(45000, dtype=float)
_profile = _get_val('SPEED_PROFILE', [])
for ki, kf, _, vn, vr in _profile:
    start_idx = int(ki)
    end_idx = min(int(kf) + 1, 45000)
    _VEL_ARRAY_NORM[start_idx:end_idx] = vn
    _VEL_ARRAY_RM[start_idx:end_idx] = vr

_PEND_ARRAY_V1 = np.zeros(45000, dtype=float)
_PEND_ARRAY_V2 = np.zeros(45000, dtype=float)
_e_km = _get_val('_ELEV_KM', [0.0, 0.7, 1.4, 2.2, 3.9, 6.0, 7.4, 8.3, 9.2, 10.2, 11.7, 19.1, 21.4, 23.3, 25.3, 26.4, 27.6, 28.5, 29.1, 30.4, 43.13])
_e_m = _get_val('_ELEV_M', [12, 10, 10, 10, 18, 15, 12, 15, 35, 50, 55, 88, 122, 132, 142, 148, 155, 162, 175, 198, 216])

if len(_e_km) == len(_e_m) and len(_e_km) > 1:
    for j in range(1, len(_e_km)):
        s_m = int(_e_km[j-1] * 1000)
        e_m = min(int(_e_km[j] * 1000), 44999)
        if e_m > s_m:
            pend = ((_e_m[j] - _e_m[j-1]) / max(0.001, (_e_km[j] - _e_km[j-1])*1000)) * 1000.0
            _PEND_ARRAY_V1[s_m:e_m] = pend
            _PEND_ARRAY_V2[s_m:e_m] = -pend

# Panangsimpa kadagiti kurbada a napartak (Von Röckl curves fix)
_CURVA_ARRAY = np.zeros(45000, dtype=float)
_curvas = _get_val('CURVAS_KM', [])
for ki, kf, r_m in _curvas:
    if r_m <= 0: continue
    if r_m >= 300: w_c = 600.0 / (r_m - 55.0)
    else: w_c = 500.0 / (r_m - 30.0)
    s_i, e_i = int(ki * 1000), min(int(kf * 1000), 44999)
    if s_i < e_i: _CURVA_ARRAY[s_i:e_i] = w_c

# =============================================================================
# 0. PAGANNUROTAN TI PANAGPATARAY (OPERATIONAL GOVERNOR)
# =============================================================================
FERIADOS_SAFE = ['2026-01-01', '2026-04-03', '2026-04-04', '2026-05-01', '2026-05-21', '2026-06-21', '2026-07-16', '2026-08-15', '2026-09-18', '2026-09-19', '2026-10-12', '2026-10-31', '2026-12-08', '2026-12-25']

def obtener_pct_traccion_operativo(row, pct_trac_ui):
    fecha_str = str(row.get('Fecha_str', '')).strip()
    if fecha_str and fecha_str.lower() not in ('nan', 'none', ''):
        try:
            if fecha_str in FERIADOS_SAFE:
                return 50.0
            d = datetime.strptime(fecha_str, '%Y-%m-%d')
            if d.weekday() >= 5: return 50.0
            return 75.0
        except:
            pass
    return pct_trac_ui

# =============================================================================
# 2. DAGITI KANGUNAAN A LOHIKA TI PANAGGUNAY (BASE MOVEMENT FUNCTIONS)
# =============================================================================
def vel_at_km(km_km, via, use_rm):
    idx = min(44999, max(0, int(km_km * 1000.0)))
    return _VEL_ARRAY_RM[idx] if use_rm else _VEL_ARRAY_NORM[idx]

def km_at_t(t_ini, t_fin, t, via, use_rm=False, km_orig=0.0, km_dest=0.0, nodos=None, t_arr=None):
    # Sinimpa a kurba S tapno nasayaat ti radar (S-Curve fix for radar)
    if nodos and len(nodos) >= 2:
        if t <= nodos[0][0]: return nodos[0][1]
        if t >= nodos[-1][0]: return nodos[-1][1]
        
        for i in range(len(nodos) - 1):
            t1, k1 = nodos[i]
            t2, k2 = nodos[i+1]
            if t1 <= t <= t2:
                if t2 == t1: return k1
                frac = (t - t1) / (t2 - t1)
                frac_smooth = (1.0 - np.cos(frac * np.pi)) / 2.0
                return k1 + (k2 - k1) * frac_smooth
                
    dur = t_fin - t_ini
    if dur <= 0: return km_orig
    frac = max(0.0, min(1.0, (t - t_ini) / dur))
    frac_smooth = (1.0 - np.cos(frac * np.pi)) / 2.0
    return km_orig + (km_dest - km_orig) * frac_smooth

def get_train_state_and_speed(t, r_via, use_rm, km_orig, km_dest, nodos=None, t_arr=None):
    if not nodos or len(nodos) < 2: return "CRUISE", 60.0
    if t_arr is None: t_arr = [n[0] for n in nodos]
    
    if t <= t_arr[0] or t >= t_arr[-1]: return "DWELL", 0.0
    
    dt_sample = 0.5
    km_now = km_at_t(t_arr[0], t_arr[-1], t, r_via, use_rm, km_orig, km_dest, nodos, t_arr)
    km_prev = km_at_t(t_arr[0], t_arr[-1], t - dt_sample, r_via, use_rm, km_orig, km_dest, nodos, t_arr)
    km_next = km_at_t(t_arr[0], t_arr[-1], t + dt_sample, r_via, use_rm, km_orig, km_dest, nodos, t_arr)
    
    v_now = abs(km_now - km_prev) * 1000.0 / dt_sample 
    v_next = abs(km_next - km_now) * 1000.0 / dt_sample
    
    a_ms2 = (v_next - v_now) / dt_sample
    v_kmh = v_now * 3.6
    
    if a_ms2 > 0.15: return "ACCEL", v_kmh
    elif a_ms2 < -0.15: return "BRAKE", v_kmh
    elif v_kmh < 1.0: return "DWELL", 0.0
    else: return "CRUISE", v_kmh

def calcular_aux_dinamico(aux_kw_nominal, hora_decimal, pax_abordo, cap_max, estacion_anio, estado_marcha="CRUISE", f_compresor_dwell=1.08):
    # Binasa ti oras (Read the hour)
    hora_int = int(hora_decimal) % 24
    perfil = _AUX_HVAC_HORA.get(estacion_anio, [0.5]*24)
    f_hvac = perfil[hora_int]
    
    f_ocup = 1.0
    if cap_max > 0:
        ocup = min(1.0, pax_abordo / cap_max)
        if estacion_anio == "verano": f_ocup = 1.0 + 0.05 * ocup
        elif estacion_anio == "invierno": f_ocup = 1.0 - 0.12 * ocup
        else: f_ocup = 1.0 - 0.06 * ocup

    # Lohika a Bottom-Up: Awan ti agdobleng bayad (Bottom-Up logic: No double counting)
    p_base = aux_kw_nominal * _FRAC_BASE
    p_clima = (aux_kw_nominal * _FRAC_HVAC) * f_hvac * f_ocup
    
    # Panagpalamiis iti inberter (Inverter cooling)
    p_vent = 0.0
    if estado_marcha == "ACCEL": p_vent = 4.0
    elif estado_marcha in ["BRAKE", "BRAKE_STATION", "BRAKE_OVERSPEED"]: p_vent = 7.6
    
    return p_base + p_clima + p_vent

# =============================================================================
# 3. KINEMATIKO-TERMODINAMIKO A MOTOR (W = F * d | CORREGIDO Y BLINDADO)
# =============================================================================
def simular_tramo_termodinamico(tipo_tren, doble, km_ini, km_fin, via_op, pct_trac, use_rm, use_pend, nodos=None, pax_dict=None, pax_abordo=0, v_consigna_override=None, maniobra=None, estacion_anio="primavera", t_ini_mins=0.0, es_vacio=False, prevenciones=None):
    flota_db = _get_val('FLOTA', {})
    f = flota_db.get(tipo_tren, {})
    if not f: return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    
    n_uni = 2 if doble else 1
    
    long_tren_km = (0.070 if tipo_tren == 'SFE' else 0.046) * n_uni
    
    pax_kg_total = pax_abordo * _PAX_KG
    masa_estatica_kg = (f.get('tara_t', 86.1) * 1000 * n_uni) + pax_kg_total
    masa_dinamica_kg = masa_estatica_kg + (f.get('m_iner_t', 7.2) * 1000 * n_uni)
    
    a_freno_op = f.get('a_freno_ms2', 1.2) * 0.9 
    
    f_trac_max_n_nominal = f.get('f_trac_max_kn', 110.0) * 1000 * n_uni
    p_max_w_nominal = f.get('p_max_kw', 720.0) * 1000 * n_uni
    
    k_s, k_e = km_ini, km_fin
    dist_total_m = abs(k_e - k_s) * 1000.0
    if dist_total_m <= 0: return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    
    es_sintetico = True
    duracion_real_h = 0.0
    if nodos and len(nodos) >= 2:
        if nodos[-1][0] > 0.0:
            es_sintetico = False
            duracion_real_h = (nodos[-1][0] - nodos[0][0]) / 60.0
            
    v_limit_thdr = 120.0
    if not es_sintetico and duracion_real_h > 0:
        dist_total_km = abs(k_e - k_s)
        v_promedio_kmh = dist_total_km / duracion_real_h
        v_limit_thdr = v_promedio_kmh * 1.35 
        pct_trac = 100.0 
    
    f_freno_max_n = f.get('f_freno_max_kn', 105.0) * 1000 * n_uni
    p_freno_max_w = f.get('p_freno_max_kw', f.get('p_max_kw', 720.0)*1.2) * 1000 * n_uni
    v_freno_min = f.get('v_freno_min', 3.81)
    
    if estacion_anio == "invierno": aux_kw_nominal = f.get('aux_kw_heat', 65.16) * n_uni
    else: aux_kw_nominal = f.get('aux_kw_cool', 58.76) * n_uni
        
    f_compresor_especifico = f.get('f_compresor_dwell', 1.08)
    
    trc, aux, reg, t_horas = 0.0, 0.0, 0.0, 0.0
    
    paradas_km = [n[1] for n in nodos] if nodos else [k_s, k_e]
    k_min, k_max = min(k_s, k_e), max(k_s, k_e)
    paradas_km = [k for k in paradas_km if k_min <= k <= k_max]
    if k_s not in paradas_km: paradas_km.append(k_s)
    if k_e not in paradas_km: paradas_km.append(k_e)
    paradas_km = list(set(paradas_km))
    paradas_km.sort(reverse=(via_op == 2))
    
    ser_data = _SER_DATA
    dt = 1.0  
    
    # Virtual a Neumatika (Virtual Pneumatic Tank)
    mrp_bar = 10.0
    compresor_on = False
    aire_gastado_parada = False
    p_compresor_kw = 3.68 * n_uni
    p_puertas_kw = 1.8 * n_uni 
    
    for i in range(len(paradas_km)-1):
        p_ini, p_fin = paradas_km[i], paradas_km[i+1]
        dist_tramo = abs(p_fin - p_ini) * 1000.0
        if dist_tramo <= 0: continue
        
        pos_m = p_ini * 1000.0
        dist_recorrida = 0.0
        v_ms = 0.0
        a_prev = 0.0 
        estado_marcha = "ACCEL"
        
        while dist_recorrida < dist_tramo:
            dist_restante = dist_tramo - dist_recorrida
            if dist_restante < 0.1: break
            
            km_actual = (pos_m + dist_recorrida) / 1000.0 if via_op == 1 else (pos_m - dist_recorrida) / 1000.0
            idx_km = min(44999, max(0, int(km_actual * 1000)))
            
            v_cons_kmh = max(5.0, _VEL_ARRAY_RM[idx_km] if use_rm else _VEL_ARRAY_NORM[idx_km])
            if v_consigna_override is not None: v_cons_kmh = min(v_cons_kmh, v_consigna_override)
            
            v_cons_kmh = min(v_cons_kmh, v_limit_thdr)
            
            if prevenciones:
                for p in prevenciones:
                    if p['via'] == via_op:
                        if via_op == 1:
                            km_inicio_restriccion = p['km_min']
                            km_fin_restriccion = p['km_max'] + long_tren_km
                            km_aviso = p['km_min'] - 1.5
                            if km_aviso <= km_actual <= km_fin_restriccion:
                                if km_inicio_restriccion <= km_actual <= km_fin_restriccion:
                                    v_cons_kmh = min(v_cons_kmh, p['v_kmh'])
                                else:
                                    dist_a_prev = (km_inicio_restriccion - km_actual) * 1000.0
                                    v_p_ms = p['v_kmh'] / 3.6
                                    if v_ms > v_p_ms:
                                        d_freno_prev = (v_ms**2 - v_p_ms**2) / (2 * a_freno_op)
                                        if dist_a_prev <= d_freno_prev + 50: 
                                            v_cons_kmh = min(v_cons_kmh, p['v_kmh'])
                        else:
                            km_inicio_restriccion = p['km_max']
                            km_fin_restriccion = p['km_min'] - long_tren_km
                            km_aviso = p['km_max'] + 1.5
                            if km_fin_restriccion <= km_actual <= km_aviso:
                                if km_fin_restriccion <= km_actual <= km_inicio_restriccion:
                                    v_cons_kmh = min(v_cons_kmh, p['v_kmh'])
                                else:
                                    dist_a_prev = (km_actual - km_inicio_restriccion) * 1000.0
                                    v_p_ms = p['v_kmh'] / 3.6
                                    if v_ms > v_p_ms:
                                        d_freno_prev = (v_ms**2 - v_p_ms**2) / (2 * a_freno_op)
                                        if dist_a_prev <= d_freno_prev + 50: 
                                            v_cons_kmh = min(v_cons_kmh, p['v_kmh'])

            # Lapped stop restriction
            if via_op == 1 and km_actual >= 42.93: v_cons_kmh = min(v_cons_kmh, 20.0 if km_actual < 43.03 else 10.0)
            if via_op == 2 and km_actual <= 0.20: v_cons_kmh = min(v_cons_kmh, 20.0 if km_actual > 0.10 else 10.0)
            
            v_kmh = v_ms * 3.6
            if n_uni == 2: f_davis = (f.get('davis_A',1615.0) * 2) + (f.get('davis_B',0.0) * 2 * v_kmh) + (f.get('davis_C',0.54) * 1.35 * (v_kmh**2))
            else: f_davis = f.get('davis_A',1615.0) + f.get('davis_B',0.0)*v_kmh + f.get('davis_C',0.54)*(v_kmh**2)
                
            f_pend = 0.0
            if use_pend:
                pend_permil = _PEND_ARRAY_V1[idx_km] if via_op == 1 else _PEND_ARRAY_V2[idx_km]
                f_pend = masa_estatica_kg * 9.81 * (pend_permil / 1000.0)
                
            # Friccion iti Masa Estatica (Friction on Static Mass only)
            f_curva = _CURVA_ARRAY[idx_km] * (masa_estatica_kg / 1000.0) * 9.81
            f_res_total = f_davis + f_pend + f_curva
            
            # Aktibo a Squeeze Control
            dist_ser = min([abs(km_actual - s[0]) for s in ser_data]) if ser_data else 5.0
            r_linea = dist_ser * 0.045
            i_req = (f_trac_max_n_nominal * max(0.1, v_ms)) / 3000.0
            v_pantografo = 3000.0 - (i_req * r_linea)
            
            factor_squeeze = 1.0
            if v_pantografo < 2800.0:
                factor_squeeze = max(0.0, (v_pantografo - 2000.0) / 800.0)
                
            f_disp_trac_real = f_trac_max_n_nominal * factor_squeeze
            p_max_op_w_real = p_max_w_nominal * factor_squeeze
            
            d_freno_req = (v_ms**2) / (2 * a_freno_op) if v_ms > 0 else 0
            f_disp_freno = min(f_freno_max_n, p_freno_max_w / max(0.1, v_ms)) if v_kmh >= v_freno_min else 0.0
            
            if dist_restante <= d_freno_req + (v_ms * dt * 1.2): estado_marcha = "BRAKE_STATION"
            elif v_kmh > v_cons_kmh + 1.5: estado_marcha = "BRAKE_OVERSPEED"
            elif estado_marcha == "BRAKE_OVERSPEED" and v_kmh <= v_cons_kmh: estado_marcha = "COAST"
            elif estado_marcha == "ACCEL" and v_kmh >= v_cons_kmh - 0.5: estado_marcha = "COAST"
            elif estado_marcha == "COAST" and v_kmh < v_cons_kmh - 2.0: estado_marcha = "ACCEL"
            elif estado_marcha not in ["ACCEL", "COAST", "BRAKE_STATION", "BRAKE_OVERSPEED"]: estado_marcha = "ACCEL"

            f_motor, f_regen_tramo, a_net_target = 0.0, 0.0, 0.0
            
            if estado_marcha == "BRAKE_STATION":
                f_req_freno = max(0.0, masa_dinamica_kg * a_freno_op - f_res_total)
                f_regen_tramo = min(f_req_freno, f_disp_freno)
                a_net_target = (-f_regen_tramo - f_res_total) / masa_dinamica_kg
                if a_net_target > -a_freno_op: a_net_target = -a_freno_op 
            elif estado_marcha == "BRAKE_OVERSPEED":
                f_req_freno = max(0.0, masa_dinamica_kg * 0.4 - f_res_total)
                f_regen_tramo = min(f_req_freno, f_disp_freno)
                a_net_target = min((-f_regen_tramo - f_res_total) / masa_dinamica_kg, -0.15)
            elif estado_marcha == "ACCEL":
                f_limite_potencia = p_max_op_w_real / max(0.1, v_ms)
                f_absoluta_disp = min(f_disp_trac_real, f_limite_potencia)
                
                f_piloto = f_trac_max_n_nominal * (pct_trac / 100.0)
                p_piloto = p_max_w_nominal * (pct_trac / 100.0)
                f_piloto_disp = min(f_piloto, p_piloto / max(0.1, v_ms))
                
                f_motor = max(f_piloto_disp, f_res_total + (masa_dinamica_kg * 0.1))
                f_motor = min(f_motor, f_absoluta_disp)
                
                a_net_target = (f_motor - f_res_total) / masa_dinamica_kg
            elif estado_marcha == "COAST":
                a_net_target = (-f_res_total) / masa_dinamica_kg
                
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
                a_req = (v_new - v_ms) / dt_actual if dt_actual > 0 else 0.0
                f_motor_req = masa_dinamica_kg * a_req + f_res_total
                f_motor = max(0.0, min(f_motor_req, f_disp_trac_real))
                a_net = a_req
                
            # Anti-Stall Failsafe
            if v_new < 0.1 and v_ms < 0.1:
                if dist_restante > 10.0:
                    v_new = 2.0 
                    a_net = (v_new - v_ms) / dt_actual if dt_actual > 0 else 0.0
                else:
                    t_horas += (dist_restante / 1.0) / 3600.0
                    break

            step_m = (v_ms + v_new) / 2.0 * dt_actual
            if step_m > dist_restante:
                step_m = dist_restante
                if v_ms + v_new > 0: dt_actual = step_m / ((v_ms + v_new) / 2.0)
            if step_m < 0.1: step_m = 0.5 
                
            f_real_total = (masa_dinamica_kg * a_net) + f_res_total
            
            if f_real_total > 0 and estado_marcha != "BRAKE_STATION":
                f_limite_potencia_inst = p_max_op_w_real / max(0.1, v_ms)
                f_absoluta_disp_inst = min(f_disp_trac_real, f_limite_potencia_inst)
                
                f_motor_real = min(f_real_total, f_absoluta_disp_inst)
                
                carga_pct = f_motor_real / max(1.0, f_absoluta_disp_inst) 
                
                eta_base = f.get('eta_motor', 0.92)
                eta_din = eta_base * (1.0 - 0.2 * (1.0 - max(0.1, carga_pct))**3)
                
                trabajo_j_trac = f_motor_real * step_m
                trc += (trabajo_j_trac / 3_600_000.0) / eta_din
                
            elif f_real_total < 0 and estado_marcha in ["BRAKE_STATION", "BRAKE_OVERSPEED"]:
                f_freno_real = min(abs(f_real_total), f_disp_freno)
                trabajo_j_regen = f_freno_real * step_m
                reg += (trabajo_j_regen / 3_600_000.0) * _ETA_REGEN_NETA
                
            # Panagibus ti angin (Air consumption logic)
            if estado_marcha in ["BRAKE", "BRAKE_STATION", "BRAKE_OVERSPEED"] and v_kmh < v_freno_min:
                if not aire_gastado_parada:
                    mrp_bar -= 0.3
                    aire_gastado_parada = True
            elif estado_marcha == "ACCEL":
                aire_gastado_parada = False
                
            if mrp_bar <= 8.0:
                compresor_on = True
                
            p_comp_inst = 0.0
            if compresor_on:
                p_comp_inst = p_compresor_kw
                mrp_bar += 0.0122 * dt_actual
                if mrp_bar >= 10.0:
                    mrp_bar = 10.0
                    compresor_on = False
                    
            hora_actual = (t_ini_mins + t_horas * 60.0) / 60.0
            pax_mid = get_pax_at_km_nativo(pax_dict, km_actual, via_op, pax_abordo) if pax_dict else pax_abordo
            
            aux_kw_inst = calcular_aux_dinamico(aux_kw_nominal, hora_actual, pax_mid, f.get('cap_max', 398) * n_uni, estacion_anio, estado_marcha, f_compresor_especifico)
            aux_kw_inst += p_comp_inst
            
            aux += (aux_kw_inst * dt_actual) / 3600.0
            t_horas += dt_actual / 3600.0
            dist_recorrida += step_m
            v_ms = v_new

    n_est_mid = max(0, len(paradas_km) - 2)
    dwell_h = (n_est_mid * _DWELL_DEF) / 3600.0
    
    # Neumatiko nga Akumulador iti andén (Pneumatic Accumulator in station)
    if compresor_on:
        tiempo_encendido_dwell = min(dwell_h * 3600.0, (10.0 - mrp_bar) / 0.0122)
        aux += (p_compresor_kw * tiempo_encendido_dwell) / 3600.0
        mrp_bar += 0.0122 * tiempo_encendido_dwell
        if mrp_bar >= 10.0:
            mrp_bar = 10.0
            compresor_on = False
            
    # Gasto estricto de puertas
    aux += (p_puertas_kw * 3.0 * n_est_mid) / 3600.0
    
    hora_media_dwell = (t_ini_mins + (t_horas + dwell_h / 2.0) * 60.0) / 60.0
    aux_kw_dwell = calcular_aux_dinamico(aux_kw_nominal, hora_media_dwell, pax_abordo, f.get('cap_max', 398) * n_uni, estacion_anio, "DWELL", f_compresor_especifico)
    
    aux += aux_kw_dwell * dwell_h
    t_horas += dwell_h
    
    neto_ideal = max(0.0, trc + aux - reg)
    return trc, aux, reg, 0.0, neto_ideal, t_horas

# =============================================================================
# 4. PRE-CALCULADORES DE RED (MACRO)
# =============================================================================
def calcular_receptividad_por_headway(df_dia: pd.DataFrame) -> dict:
    if df_dia.empty: return {}
    result = {}
    for via in [1, 2]:
        sub = df_dia[df_dia["Via"] == via].sort_values("t_ini")
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

def precalcular_red_electrica_v111(df_dia, pct_trac_ui, use_rm, estacion_anio="primavera"):
    regen_util_per_trip = {idx: 0.0 for idx in df_dia.index}
    braking_ticks_per_trip = {idx: 0.0 for idx in df_dia.index} 
    if df_dia.empty: 
        return regen_util_per_trip
    
    t_min = int(df_dia['t_ini'].min())
    t_max = int(df_dia['t_fin'].max())
    dt_step = 10.0 / 60.0 
    time_steps = np.arange(t_min, t_max + 1, dt_step)
    
    for via_ in [1, 2]:
        via_trains = df_dia[df_dia['Via'] == via_]
        if via_trains.empty: continue
        
        trains_data = []
        for idx, r in via_trains.iterrows():
            nodos = r.get('nodos')
            pct_operativo = obtener_pct_traccion_operativo(r, pct_trac_ui)
            trains_data.append({
                'idx': idx, 
                't_ini': r['t_ini'], 
                't_fin': r['t_fin'], 
                'Via': r['Via'],
                'km_orig': r['km_orig'], 
                'km_dest': r['km_dest'], 
                'nodos': nodos,
                't_arr': [n[0] for n in nodos] if nodos and len(nodos) >= 2 else None,
                'tipo_tren': r.get('tipo_tren', 'XT-100'), 
                'doble': r.get('doble', False), 
                'pax_abordo': r.get('pax_abordo', 0),
                'pct_trac': pct_operativo
            })
            
        braking_by_idx = [[] for _ in range(len(time_steps))]
        accel_by_idx = [[] for _ in range(len(time_steps))]
        
        for tr in trains_data:
            idx_start = np.searchsorted(time_steps, max(t_min, tr['t_ini']))
            idx_end = np.searchsorted(time_steps, min(t_max, tr['t_fin']), side='right')
            f = _get_val('FLOTA', {}).get(tr['tipo_tren'], {})
            if not f: continue
            
            n_uni = 2 if tr['doble'] else 1
            masa_estatica_kg = (f.get('tara_t', 86.1) * 1000 * n_uni) + (tr['pax_abordo'] * _PAX_KG)
            masa_dinamica_kg = masa_estatica_kg + (f.get('m_iner_t', 7.2) * 1000 * n_uni)
            eta_m = f.get('eta_motor', 0.92)
            
            for i in range(idx_start, idx_end):
                m = time_steps[i]
                state, v_kmh = get_train_state_and_speed(m, tr['Via'], use_rm, tr['km_orig'], tr['km_dest'], tr['nodos'], tr['t_arr'])
                pos = km_at_t(tr['t_ini'], tr['t_fin'], m, tr['Via'], use_rm, tr['km_orig'], tr['km_dest'], tr['nodos'], tr['t_arr'])
                v_ms = v_kmh / 3.6
                
                if estacion_anio == "invierno": aux_nom = f.get('aux_kw_heat', 65.16) * n_uni
                else: aux_nom = f.get('aux_kw_cool', 58.76) * n_uni
                p_aux_kw = calcular_aux_dinamico(aux_nom, m / 60.0, tr['pax_abordo'], f.get('cap_max', 398) * n_uni, estacion_anio, state)
                
                # Sincronización ti pre-kalkulador kadagiti kurba ken grabidad (Sync pre-calc with curves and gravity)
                idx_km = min(44999, max(0, int(pos * 1000)))
                pend_permil = _PEND_ARRAY_V1[idx_km] if tr['Via'] == 1 else _PEND_ARRAY_V2[idx_km]
                f_pend = masa_estatica_kg * 9.81 * (pend_permil / 1000.0)
                f_curva = _CURVA_ARRAY[idx_km] * (masa_estatica_kg / 1000.0) * 9.81
                
                if n_uni == 2:
                    f_davis = (f.get('davis_A',1615) * 2) + (f.get('davis_B',0) * 2 * v_kmh) + (f.get('davis_C',0.54) * 1.35 * (v_kmh**2))
                else:
                    f_davis = f.get('davis_A',1615) + f.get('davis_B',0)*v_kmh + f.get('davis_C',0.54)*(v_kmh**2)
                    
                f_res_total = f_davis + f_pend + f_curva
                
                if state in ("BRAKE", "BRAKE_STATION", "BRAKE_OVERSPEED"):
                    f_req_freno = max(0.0, masa_dinamica_kg * (f.get('a_freno_ms2', 1.2) * 0.9) - f_res_total)
                    f_disp_freno = min(f.get('f_freno_max_kn', 105.0)*1000*n_uni, (f.get('p_freno_max_kw', f.get('p_max_kw',720)*1.2)*1000*n_uni)/max(0.1, v_ms)) if v_kmh >= f.get('v_freno_min', 3.81) else 0.0
                    p_gen_kw = ((min(f_req_freno, f_disp_freno) * v_ms) / 1000.0 * _ETA_REGEN_NETA) - p_aux_kw
                    
                    if p_gen_kw > 0: 
                        braking_by_idx[i].append((tr['idx'], pos, p_gen_kw))
                    braking_ticks_per_trip[tr['idx']] += 1
                    
                elif state in ("ACCEL", "CRUISE"):
                    p_dem_kw = p_aux_kw
                    if state == "ACCEL": 
                        f_piloto = f.get('f_trac_max_kn', 110.0)*1000*n_uni*(tr['pct_trac']/100.0)
                        p_piloto = f.get('p_max_kw', 720.0)*1000*n_uni*(tr['pct_trac']/100.0)
                        f_piloto_disp = min(f_piloto, p_piloto/max(0.1, v_ms)) if v_ms > 0 else f_piloto
                        
                        f_motor = max(f_piloto_disp, f_res_total + (masa_dinamica_kg * 0.1))
                        f_limite_total_abs = min(f.get('f_trac_max_kn', 110.0)*1000*n_uni, (f.get('p_max_kw', 720.0)*1000*n_uni)/max(0.1, v_ms))
                        f_motor = min(f_motor, f_limite_total_abs)
                        
                        carga_pct = f_motor / max(1.0, f_limite_total_abs) 
                        eta_din = eta_m * (1.0 - 0.2 * (1.0 - max(0.1, carga_pct))**3)
                        
                        p_dem_kw += ((f_motor * v_ms) / 1000.0 / eta_din)
                    elif state == "CRUISE" and f_res_total > 0: 
                        p_dem_kw += (((f_res_total * v_ms) / 1000.0) / eta_m)
                        
                    accel_by_idx[i].append((tr['idx'], pos, p_dem_kw))
                    
        for i in range(len(time_steps)):
            if not braking_by_idx[i] or not accel_by_idx[i]: 
                continue
                
            current_demands = {a[0]: a[2] for a in accel_by_idx[i]}
            
            for b_idx, b_pos, p_gen in braking_by_idx[i]:
                available = [a for a in accel_by_idx[i] if current_demands[a[0]] > 0]
                if not available: 
                    break 
                    
                a_idx, a_pos, _ = min(available, key=lambda x: abs(x[1] - b_pos))
                dist = abs(a_pos - b_pos)
                
                if dist <= _LAMBDA_REGEN_KM * 2:
                    p_transferred = min(p_gen * (_ETA_MAX * np.exp(-dist / _LAMBDA_REGEN_KM)), current_demands[a_idx])
                    current_demands[a_idx] -= p_transferred
                    regen_util_per_trip[b_idx] += (p_transferred / p_gen)
                    
    for idx in df_dia.index: 
        if braking_ticks_per_trip[idx] > 0:
            regen_util_per_trip[idx] = min(1.0, regen_util_per_trip[idx] / braking_ticks_per_trip[idx])
        else:
            regen_util_per_trip[idx] = 0.0
            
    return regen_util_per_trip

def calcular_termodinamica_flota_v111(df_dia, pct_trac_ui, use_pend, use_rm, use_regen, dict_regen, estacion_anio="primavera", prevenciones=None):
    df_e = df_dia.copy()
    if df_e.empty: return df_e
    def _wrapper(r):
        pct_real = obtener_pct_traccion_operativo(r, pct_trac_ui)
        trc, aux, reg_bruta, reg_panto_push, neto_ideal, t_h = simular_tramo_termodinamico(
            r['tipo_tren'], r.get('doble', False), r['km_orig'], r['km_dest'], r['Via'], 
            pct_real, use_rm, use_pend, r.get('nodos'), r.get('pax_d', {}), r.get('pax_abordo', 0), 
            None, r.get('maniobra'), estacion_anio, r.get('t_ini', 0.0), False, prevenciones
        )
        
        eta_red = dict_regen.get(r.name, 1.0) if use_regen else 0.0
        reg_util = reg_bruta * eta_red
        kwh_reostato = max(0.0, reg_bruta - reg_util)
        neto = max(0.0, trc + aux - reg_util)
        
        return pd.Series([trc, aux, reg_util, kwh_reostato, neto, t_h])
        
    df_e[['kwh_viaje_trac', 'kwh_viaje_aux', 'kwh_viaje_regen', 'kwh_reostato', 'kwh_viaje_neto', 't_viaje_h']] = df_e.apply(_wrapper, axis=1)
    
    if 'tren_km' not in df_e.columns:
        df_e['tren_km'] = df_e.apply(lambda r: abs(r['km_dest'] - r['km_orig']) * (2.0 if r.get('doble', False) else 1.0), axis=1)
        
    return df_e
