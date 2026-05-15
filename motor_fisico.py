import numpy as np
import pandas as pd
from datetime import datetime

# Escudo Defensivo Cloud: Evita NameErrors si Streamlit limpia la memoria
try:
    import config
except ImportError:
    pass

def _get_val(name, default):
    try: return getattr(config, name, default)
    except Exception: return default

# Importación segura de la lógica de pasajeros desde el módulo ETL
try:
    from etl_parser import get_pax_at_km_nativo
except ImportError:
    def get_pax_at_km_nativo(pax_d, km_pos, via, pax_max_fallback=0): 
        return pax_max_fallback

# =============================================================================
# 0. GOBERNADOR OPERATIVO (CALENDARIO DE TRACCIÓN EFE)
# =============================================================================
FERIADOS_SAFE = ['2026-01-01', '2026-04-03', '2026-04-04', '2026-05-01', '2026-05-21', '2026-06-21', '2026-07-16', '2026-08-15', '2026-09-18', '2026-09-19', '2026-10-12', '2026-10-31', '2026-12-08', '2026-12-25']

def obtener_pct_traccion_operativo(row, pct_trac_ui):
    fecha_str = str(row.get('Fecha_str', '')).strip()
    if fecha_str and fecha_str.lower() not in ('nan', 'none', ''):
        try:
            if fecha_str in FERIADOS_SAFE:
                return 50.0
            d = datetime.strptime(fecha_str, '%Y-%m-%d')
            if d.weekday() >= 5:
                return 50.0
            return 75.0
        except:
            pass
    return pct_trac_ui

# =============================================================================
# 1. OPTIMIZACIÓN EXTREMA: MATRICES PRE-CALCULADAS O(1)
# =============================================================================
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

_CURVA_ARRAY = np.zeros(45000, dtype=float)
_curvas = _get_val('CURVAS_KM', [])
for ki, kf, r_m in _curvas:
    if r_m <= 0: continue
    if r_m >= 300: w_c = 600.0 / (r_m - 55.0)
    else: w_c = 500.0 / (r_m - 30.0)
    s_i, e_i = int(ki * 1000), min(int(kf * 1000), 44999)
    if s_i < e_i: _CURVA_ARRAY[s_i:e_i] = w_c

def _get_resistencia_catenaria_km(km):
    if km < 2.25: return 0.0638
    elif km < 6.80: return 0.0530
    elif km < 10.92: return 0.0495
    elif km < 21.41: return 0.0417
    elif km < 30.36: return 0.0399
    else: return 0.0355

# =============================================================================
# 2. FUNCIONES BASE DE MOVIMIENTO Y RADAR ELÉCTRICO
# =============================================================================
def vel_at_km(km_km, via, use_rm):
    idx = min(44999, max(0, int(km_km * 1000.0)))
    return _VEL_ARRAY_RM[idx] if use_rm else _VEL_ARRAY_NORM[idx]

def km_at_t(t_ini, t_fin, t, via, use_rm=False, km_orig=0.0, km_dest=0.0, nodos=None, t_arr=None):
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

# =============================================================================
# 2.5. FUNCIÓN DE AUXILIARES CON DUTY CYCLE REAL
# =============================================================================
def calcular_aux_dinamico(tipo_tren, aux_kw_nominal, hora_decimal, pax_abordo, cap_max, estacion_anio, estado_marcha="CRUISE"):
    hora_int = int(hora_decimal) % 24
    try: perfil = _get_val('AUX_HVAC_HORA', {}).get(estacion_anio, [0.5]*24)
    except: perfil = [0.5]*24
    if not perfil or len(perfil) < 24: perfil = [0.5]*24
    
    f_hvac = perfil[hora_int]
    f_ocup = 1.0
    if cap_max > 0:
        ocup = min(1.0, pax_abordo / cap_max)
        if estacion_anio == "verano": f_ocup = 1.0 + 0.05 * ocup
        elif estacion_anio == "invierno": f_ocup = 1.0 - 0.12 * ocup
        else: f_ocup = 1.0 - 0.06 * ocup
    
    if tipo_tren == "XT-100":
        duty_compresor = {
            "ACCEL": 0.10, "CRUISE": 0.08, "BRAKE": 0.45,
            "BRAKE_STATION": 0.45, "BRAKE_OVERSPEED": 0.45,
            "COAST": 0.08, "DWELL": 0.12
        }
    elif tipo_tren == "XT-M":
        duty_compresor = {
            "ACCEL": 0.20, "CRUISE": 0.18, "BRAKE": 0.65,
            "BRAKE_STATION": 0.65, "BRAKE_OVERSPEED": 0.65,
            "COAST": 0.18, "DWELL": 0.80
        }
    else:
        duty_compresor = {
            "ACCEL": 0.25, "CRUISE": 0.22, "BRAKE": 0.70,
            "BRAKE_STATION": 0.70, "BRAKE_OVERSPEED": 0.70,
            "COAST": 0.22, "DWELL": 0.85
        }
    
    dc_comp = duty_compresor.get(estado_marcha, 0.10)
    
    duty_ventilacion = {
        "ACCEL": 1.00, "CRUISE": 0.60, "BRAKE": 0.40,
        "BRAKE_STATION": 0.40, "BRAKE_OVERSPEED": 0.40,
        "COAST": 0.20, "DWELL": 0.10
    }
    dc_vent = duty_ventilacion.get(estado_marcha, 0.30)
    
    frac_hvac = 0.75
    frac_comp = 0.05
    frac_vent = 0.05
    frac_base = 0.15
    
    aux_hvac = aux_kw_nominal * frac_hvac * f_hvac * f_ocup
    aux_comp = aux_kw_nominal * frac_comp * dc_comp
    aux_vent = aux_kw_nominal * frac_vent * dc_vent
    aux_base = aux_kw_nominal * frac_base
    
    return aux_base + aux_hvac + aux_comp + aux_vent

# =============================================================================
# 3. KILOMETRAJE ROBUSTO (Tren-km)
# =============================================================================
def _calc_tren_km_real_motor(row):
    k_o = row.get('km_orig', 0.0)
    k_d = row.get('km_dest', 0.0)
    if pd.isna(k_o) or pd.isna(k_d): return 0.0
    man = row.get('maniobra')
    is_doble = row.get('doble', False)
    if not man or pd.isna(man) or str(man).strip().lower() in ['none', '']:
        return abs(k_d - k_o) * (2.0 if is_doble else 1.0)
    km_man = None
    man_upper = str(man).upper()
    if 'CORTE_BTO' in man_upper or 'ACOPLE_BTO' in man_upper or 'CORTE_PU_SA_BTO' in man_upper: km_man = 25.3
    elif 'CORTE_SA' in man_upper or 'ACOPLE_SA' in man_upper: km_man = 29.1
    if km_man is None: return abs(k_d - k_o) * (2.0 if is_doble else 1.0)
    if min(k_o, k_d) <= km_man <= max(k_o, k_d):
        dist_antes = abs(km_man - k_o)
        dist_despues = abs(k_d - km_man)
        if 'CORTE' in man_upper: return (dist_antes * 2.0) + (dist_despues * 1.0)
        elif 'ACOPLE' in man_upper: return (dist_antes * 1.0) + (dist_despues * 2.0)
    return abs(k_d - k_o) * (2.0 if is_doble else 1.0)

# =============================================================================
# 4. MOTOR CINEMÁTICO-TERMODINÁMICO (CORREGIDO)
# =============================================================================
def simular_tramo_termodinamico(tipo_tren, doble, km_ini, km_fin, via_op, pct_trac, use_rm, use_pend, nodos=None, pax_dict=None, pax_abordo=0, v_consigna_override=None, maniobra=None, estacion_anio="primavera", t_ini_mins=0.0, es_vacio=False, prevenciones=None):
    flota_db = _get_val('FLOTA', {})
    f = flota_db.get(tipo_tren, {})
    if not f: return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0
    
    km_man = None
    man_upper = str(maniobra).upper() if maniobra and not pd.isna(maniobra) else ''
    if 'CORTE_BTO' in man_upper or 'ACOPLE_BTO' in man_upper or 'CORTE_PU_SA_BTO' in man_upper: km_man = 25.3
    elif 'CORTE_SA' in man_upper or 'ACOPLE_SA' in man_upper: km_man = 29.1
    dist_to_maniobra = abs(km_man - km_ini) * 1000.0 if km_man is not None else -1
    
    a_freno_op = f.get('a_freno_ms2', 1.2) * 0.9 
    v_freno_min = f.get('v_freno_min', 3.81)
    
    k_s, k_e = km_ini, km_fin
    dist_total_m = abs(k_e - k_s) * 1000.0
    if dist_total_m <= 0: return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0
    
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
    
    trc = 0.0
    aux_catenaria = 0.0
    reg_exportable = 0.0
    t_horas = 0.0
    prevencion_aplicada = 0
    
    paradas_km = [n[1] for n in nodos] if nodos else [k_s, k_e]
    k_min, k_max = min(k_s, k_e), max(k_s, k_e)
    paradas_km = [k for k in paradas_km if k_min <= k <= k_max]
    if k_s not in paradas_km: paradas_km.append(k_s)
    if k_e not in paradas_km: paradas_km.append(k_e)
    paradas_km = list(set(paradas_km))
    paradas_km.sort(reverse=(via_op == 2))
    
    ser_data = _get_val('SER_DATA', [(4.9, "SER PO"), (12.7, "SER ES"), (25.5, "SER EB"), (28.7, "SER VA")])
    dt = 1.0  
    eta_motor = f.get('eta_motor', 0.92)
    eta_regen_neta = _get_val('ETA_REGEN_NETA', 0.38)
    dwell_seg = _get_val('DWELL_DEF', 25.0)
    
    n_uni_final = 2 if doble else 1
    aux_kw_nominal_final = 0.0
    
    for i in range(len(paradas_km)-1):
        p_ini, p_fin = paradas_km[i], paradas_km[i+1]
        es_ultima_parada = (i == len(paradas_km) - 2)
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
            
            dist_recorrida_total = abs(km_actual - km_ini) * 1000.0
            es_doble = doble
            
            if km_man is not None and min(km_ini, km_fin) <= km_man <= max(km_ini, km_fin) and man_upper:
                if 'CORTE' in man_upper:
                    es_doble = True if dist_recorrida_total <= dist_to_maniobra else False
                elif 'ACOPLE' in man_upper:
                    es_doble = False if dist_recorrida_total <= dist_to_maniobra else True
                    
            n_uni_inst = 2 if es_doble else 1
            n_uni_final = n_uni_inst
            
            long_tren_km = (0.070 if tipo_tren == 'SFE' else 0.046) * n_uni_inst
            
            pax_mid = get_pax_at_km_nativo(pax_dict, km_actual, via_op, pax_abordo) if pax_dict else pax_abordo
            pax_kg_total = pax_mid * _get_val('PAX_KG', 75.0)
            
            masa_estatica_kg = (f.get('tara_t', 86.1) * 1000 * n_uni_inst) + pax_kg_total
            masa_dinamica_kg = masa_estatica_kg + (f.get('m_iner_t', 7.2) * 1000 * n_uni_inst)
            
            f_trac_max_n_nominal = f.get('f_trac_max_kn', 110.0) * 1000 * n_uni_inst
            p_max_w_nominal = f.get('p_max_kw', 720.0) * 1000 * n_uni_inst
            f_freno_max_n = f.get('f_freno_max_kn', 105.0) * 1000 * n_uni_inst
            p_freno_max_w = f.get('p_freno_max_kw', f.get('p_max_kw', 720.0)*1.2) * 1000 * n_uni_inst
            
            if estacion_anio in ("invierno", "otoño"): aux_kw_nominal = f.get('aux_kw_heat', 65.16) * n_uni_inst
            else: aux_kw_nominal = f.get('aux_kw_cool', 58.76) * n_uni_inst
            aux_kw_nominal_final = aux_kw_nominal
            
            _v_raw = _VEL_ARRAY_RM[idx_km] if use_rm else _VEL_ARRAY_NORM[idx_km]
            if _v_raw == 0:
                # Zona de andén: buscar próxima velocidad >0 en dirección de marcha
                _step_dir = 1 if via_op == 1 else -1
                for _di in range(1, 500):
                    _idx2 = min(44999, max(0, idx_km + _di * _step_dir))
                    _v2 = _VEL_ARRAY_RM[_idx2] if use_rm else _VEL_ARRAY_NORM[_idx2]
                    if _v2 > 0: _v_raw = _v2; break
            v_cons_kmh = max(5.0, _v_raw)
            if v_consigna_override is not None: v_cons_kmh = min(v_cons_kmh, v_consigna_override)
            v_cons_kmh = min(v_cons_kmh, v_limit_thdr)
            
            # =============================================================
            # APLICACIÓN DE PREVENCIONES — aviso dinámico por distancia de frenado
            # El tren baja la consigna exactamente cuando necesita empezar a frenar
            # para llegar a v_kmh_prev al inicio de la zona restringida
            # =============================================================
            if prevenciones:
                for p in prevenciones:
                    if p['via'] == via_op:
                        v_prev = p['v_kmh']
                        if via_op == 1:
                            km_inicio_prev = p['km_min']
                            km_fin_prev    = p['km_max'] + long_tren_km
                            en_zona = km_inicio_prev <= km_actual <= km_fin_prev
                            if not en_zona and km_actual < km_inicio_prev:
                                # Calcular distancia de frenado desde v_actual hasta v_prev
                                v_actual_ms = v_ms
                                v_prev_ms   = v_prev / 3.6
                                if v_actual_ms > v_prev_ms:
                                    d_freno_prev = (v_actual_ms**2 - v_prev_ms**2) / (2 * a_freno_op)
                                    dist_al_inicio = (km_inicio_prev - km_actual) * 1000
                                    if dist_al_inicio <= d_freno_prev * 1.2:
                                        v_cons_kmh = min(v_cons_kmh, v_prev)
                                        prevencion_aplicada += 1
                            elif en_zona:
                                v_cons_kmh = min(v_cons_kmh, v_prev)
                                prevencion_aplicada += 1
                        else:
                            km_inicio_prev = p['km_max']
                            km_fin_prev    = p['km_min'] - long_tren_km
                            en_zona = km_fin_prev <= km_actual <= km_inicio_prev
                            if not en_zona and km_actual > km_inicio_prev:
                                v_actual_ms = v_ms
                                v_prev_ms   = v_prev / 3.6
                                if v_actual_ms > v_prev_ms:
                                    d_freno_prev = (v_actual_ms**2 - v_prev_ms**2) / (2 * a_freno_op)
                                    dist_al_inicio = (km_actual - km_inicio_prev) * 1000
                                    if dist_al_inicio <= d_freno_prev * 1.2:
                                        v_cons_kmh = min(v_cons_kmh, v_prev)
                                        prevencion_aplicada += 1
                            elif en_zona:
                                v_cons_kmh = min(v_cons_kmh, v_prev)
                                prevencion_aplicada += 1

            if via_op == 1 and km_actual >= 42.93: v_cons_kmh = min(v_cons_kmh, 20.0 if km_actual < 43.03 else 10.0)
            if via_op == 2 and km_actual <= 0.20: v_cons_kmh = min(v_cons_kmh, 20.0 if km_actual > 0.10 else 10.0)
            
            v_kmh = v_ms * 3.6
            if n_uni_inst == 2: f_davis = (f.get('davis_A',1615.0)*2) + (f.get('davis_B',0.0)*2*v_kmh) + (f.get('davis_C',0.54)*1.35*(v_kmh**2))
            else: f_davis = f.get('davis_A',1615.0) + f.get('davis_B',0.0)*v_kmh + f.get('davis_C',0.54)*(v_kmh**2)
                
            f_pend = 0.0
            if use_pend:
                pend_permil = _PEND_ARRAY_V1[idx_km] if via_op == 1 else _PEND_ARRAY_V2[idx_km]
                f_pend = masa_dinamica_kg * 9.81 * (pend_permil / 1000.0)
                
            f_curva = _CURVA_ARRAY[idx_km] * (masa_estatica_kg / 1000.0) * 9.81
            f_res_total = f_davis + f_pend + f_curva
            
            dist_ser = min([abs(km_actual - s[0]) for s in ser_data]) if ser_data else 5.0
            r_linea = _get_resistencia_catenaria_km(km_actual) * dist_ser
            # Caída de voltaje: estimar corriente desde potencia disponible
            _p_est_w = p_max_w_nominal * (pct_trac / 100.0) * min(1.0, v_kmh / max(1.0, v_cons_kmh))
            i_req = _p_est_w / max(100.0, 3000.0 * v_ms if v_ms > 0.1 else 3000.0)
            v_pantografo = 3000.0 - (i_req * r_linea)
            
            factor_squeeze = 1.0
            if v_pantografo < 2800.0: factor_squeeze = max(0.0, (v_pantografo - 2000.0) / 800.0)
                
            f_disp_trac_real = f_trac_max_n_nominal * factor_squeeze
            p_max_op_w_real = p_max_w_nominal * factor_squeeze
            
            d_freno_req = (v_ms**2) / (2 * a_freno_op) if v_ms > 0 else 0
            f_disp_freno = min(f_freno_max_n, p_freno_max_w / max(0.1, v_ms)) if v_kmh >= v_freno_min else 0.0
            
            if dist_restante < 1.0:
                t_horas += 0.1 / 3600.0
                dist_recorrida += dist_restante
                v_ms = 0.0
                break
            elif dist_restante <= d_freno_req + (v_ms * dt * 1.2):
                estado_marcha = "BRAKE_STATION"
            elif v_kmh > v_cons_kmh + 1.5:
                estado_marcha = "BRAKE_OVERSPEED"
            elif estado_marcha == "BRAKE_OVERSPEED" and v_kmh <= v_cons_kmh:
                estado_marcha = "CRUISE"
            elif estado_marcha == "ACCEL" and v_kmh >= v_cons_kmh - 0.5:
                estado_marcha = "CRUISE"
            elif estado_marcha == "CRUISE" and f_res_total < -50.0:  estado_marcha = "COAST"  # bajada: pendiente favorable, sin tracción
            elif estado_marcha == "CRUISE" and v_kmh < v_cons_kmh - 1.5:
                estado_marcha = "ACCEL"
            elif estado_marcha == "COAST"  and v_kmh < v_cons_kmh - 2.0:
                estado_marcha = "ACCEL"
            elif estado_marcha not in ["ACCEL","CRUISE","COAST","BRAKE_STATION","BRAKE_OVERSPEED"]:
                estado_marcha = "ACCEL"

            f_motor, f_regen_tramo, a_net_target = 0.0, 0.0, 0.0
            
            if estado_marcha == "BRAKE_STATION":
                # Freno mixto: regenerativo hasta v_freno_min, luego neumático completa la diferencia
                # En bajada (f_res_total < 0) la gravedad acelera → se necesita MÁS freno
                # En subida (f_res_total > 0) la gravedad frena → se necesita MENOS freno
                f_req_freno_total = masa_dinamica_kg * a_freno_op - f_res_total
                f_req_freno_total = max(0.0, f_req_freno_total)
                if v_kmh >= v_freno_min * 3.6:  # freno regen disponible
                    f_regen_tramo = min(f_req_freno_total, f_disp_freno)
                else:  # velocidad baja: solo freno neumático
                    f_regen_tramo = 0.0
                f_freno_neumatico = max(0.0, f_req_freno_total - f_regen_tramo)
                f_freno_total = f_regen_tramo + f_freno_neumatico
                a_net_target = (-f_freno_total - f_res_total) / masa_dinamica_kg
                a_net_target = max(a_net_target, -a_freno_op * 1.1)  # no superar límite físico
            elif estado_marcha == "BRAKE_OVERSPEED":
                f_req_freno = max(0.0, masa_dinamica_kg * 0.7 - f_res_total)  # 0.7 m/s² para corregir sobrevelocidad
                f_regen_tramo = min(f_req_freno, f_disp_freno)
                a_net_target = min((-f_regen_tramo - f_res_total) / masa_dinamica_kg, -0.15)
            elif estado_marcha == "ACCEL":
                # pct_trac limita la fuerza de arranque (IGBT/corriente), NO la potencia del motor
                # A alta velocidad el motor puede desarrollar plena potencia aunque pct_trac < 100%
                f_limite_potencia = p_max_op_w_real / max(0.1, v_ms)  # límite por potencia plena del motor
                f_absoluta_disp = min(f_disp_trac_real, f_limite_potencia)
                f_piloto = f_trac_max_n_nominal * (pct_trac / 100.0)  # fuerza máxima reducida por IGBT
                f_piloto_disp = min(f_piloto, f_limite_potencia)       # a alta v, potencia plena disponible
                f_motor = min(f_piloto_disp, f_absoluta_disp)
                a_net_target = (f_motor - f_res_total) / masa_dinamica_kg
            elif estado_marcha == "CRUISE":
                # En CRUISE el motor mantiene velocidad — no hay límite de pct_trac en régimen
                f_motor = max(0.0, f_res_total)
                f_motor = min(f_motor, f_disp_trac_real)  # potencia plena disponible en crucero
                a_net_target = (f_motor - f_res_total) / masa_dinamica_kg
            elif estado_marcha == "COAST":
                a_net_target = (-f_res_total) / masa_dinamica_kg
                
            # Jerk diferenciado: tracción vs frenado
            jerk_trac  = f.get('jerk_ms3', 0.8) * dt
            jerk_freno = f.get('a_freno_ms2', 1.2) * 0.5 * dt  # transición más rápida en frenado
            jerk_limit = jerk_freno if a_net_target < 0 else jerk_trac
            if a_net_target > a_prev + jerk_limit: a_net = a_prev + jerk_limit
            elif a_net_target < a_prev - jerk_limit: a_net = a_prev - jerk_limit
            else: a_net = a_net_target
            # Clamp físico: a_net no puede superar a_max ni a_freno del tren
            a_max_ms2  = f.get('a_max_ms2',  1.0)
            a_freno_ms2 = f.get('a_freno_ms2', 1.2)
            a_net = max(-a_freno_ms2, min(a_max_ms2, a_net))
            a_prev = a_net
            
            v_new, dt_actual = v_ms + a_net * dt, dt
            if v_new < 0:
                dt_actual = v_ms / abs(a_net) if a_net < -0.001 else dt
                v_new = 0.0
                
            if (f_motor > 0 or estado_marcha == "COAST") and v_new * 3.6 > v_cons_kmh + 0.5:
                v_new = v_cons_kmh / 3.6
                a_req = (v_new - v_ms) / dt_actual if dt_actual > 0 else 0.0
                f_motor_req = masa_dinamica_kg * a_req + f_res_total
                f_motor = max(0.0, min(f_motor_req, f_disp_trac_real))
                a_net = a_req
                
            if v_new < 0.1 and v_ms < 0.1:
                if estado_marcha == "BRAKE_STATION":
                    # Tren detenido en frenada — completar metros restantes a velocidad mínima
                    if dist_restante > 0.1:
                        t_horas += (dist_restante / max(0.5, v_freno_min)) / 3600.0
                        dist_recorrida += dist_restante
                    break
                elif dist_restante > 10.0:
                    estado_marcha = "ACCEL"  # antiatasco gradual
                    v_new = 0.1
                    a_net = f.get('jerk_ms3', 0.8) * dt
                else:
                    t_horas += (dist_restante / 1.0) / 3600.0
                    break

            step_m = (v_ms + v_new) / 2.0 * dt_actual
            if step_m > dist_restante:
                step_m = dist_restante
                if v_ms + v_new > 0: dt_actual = step_m / ((v_ms + v_new) / 2.0)
            if step_m < 0.01: step_m = min(0.1, dist_restante)  # paso mínimo controlado
                
            f_real_total = (masa_dinamica_kg * a_net) + f_res_total
            
            hora_actual = (t_ini_mins + t_horas * 60.0) / 60.0
            aux_kw_inst = calcular_aux_dinamico(
                tipo_tren, aux_kw_nominal, hora_actual, pax_mid,
                f.get('cap_max', 398) * n_uni_inst, estacion_anio, estado_marcha
            )
            aux_kwh_step = (aux_kw_inst * dt_actual) / 3600.0
            
            if f_real_total > 0 and estado_marcha != "BRAKE_STATION":
                f_limite_potencia_inst = p_max_op_w_real / max(0.1, v_ms)
                f_absoluta_disp_inst = min(f_disp_trac_real, f_limite_potencia_inst)
                f_motor_real = min(f_real_total, f_absoluta_disp_inst)
                # Eficiencia dinámica del motor eléctrico de tracción:
                # - Arranque (v < 15 km/h): régimen de par constante → eta ≈ eta_base
                # - Velocidad media (P ≈ P_max): eficiencia máxima → eta = eta_base
                # - Carga parcial en CRUISE: eficiencia levemente menor
                eta_base = f.get('eta_motor', 0.92)
                if v_kmh < 15.0:
                    # Régimen par constante: eficiencia del motor alta
                    eta_din = eta_base
                else:
                    p_real_w = f_motor_real * max(0.1, v_ms)
                    carga_pct = p_real_w / max(1.0, p_max_w_nominal)
                    # Eficiencia cae levemente a carga parcial (CRUISE plano)
                    eta_din = eta_base * (1.0 - 0.08 * (1.0 - max(0.2, carga_pct))**2)
                trabajo_j_trac = f_motor_real * step_m
                trc += (trabajo_j_trac / 3_600_000.0) / eta_din
                aux_catenaria += aux_kwh_step
                
            elif f_real_total < 0 and estado_marcha in ["BRAKE_STATION", "BRAKE_OVERSPEED", "COAST"]:
                f_freno_real = min(abs(f_real_total), f_disp_freno)
                trabajo_j_regen = f_freno_real * step_m
                energia_bruta_kwh = trabajo_j_regen / 3_600_000.0
                energia_electrica_kwh = energia_bruta_kwh * eta_motor
                
                if energia_electrica_kwh >= aux_kwh_step:
                    excedente_kwh = energia_electrica_kwh - aux_kwh_step
                    reg_exportable += excedente_kwh * eta_regen_neta
                else:
                    deficit_kwh = aux_kwh_step - energia_electrica_kwh
                    aux_catenaria += deficit_kwh
            else:
                aux_catenaria += aux_kwh_step
            
            t_horas += dt_actual / 3600.0
            dist_recorrida += step_m
            v_ms = v_new

        if es_sintetico and not es_ultima_parada:
            dwell_h = dwell_seg / 3600.0
            hora_media_dwell = (t_ini_mins + (t_horas + dwell_h / 2.0) * 60.0) / 60.0
            aux_kw_dwell = calcular_aux_dinamico(
                tipo_tren, aux_kw_nominal_final, hora_media_dwell, pax_abordo,
                f.get('cap_max', 398) * n_uni_final, estacion_anio, "DWELL"
            )
            aux_catenaria += aux_kw_dwell * dwell_h
            t_horas += dwell_h

    neto_ideal = max(0.0, trc + aux_catenaria - reg_exportable)
    return trc, aux_catenaria, reg_exportable, 0.0, neto_ideal, t_horas, prevencion_aplicada

# =============================================================================
# 5. PRE-CALCULADORES DE RED
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
    
    LAMBDA_REGEN = _get_val('LAMBDA_REGEN_KM', 5.0)
    ETA_MAX_VAL = _get_val('ETA_MAX', 0.70)
    DISTANCIA_MAXIMA = LAMBDA_REGEN * 2.5
    
    for via_ in [1, 2]:
        via_trains = df_dia[df_dia['Via'] == via_]
        if via_trains.empty:
            continue
        
        via_indices = set(via_trains.index)
        
        trains_data = []
        for idx, r in via_trains.iterrows():
            nodos = r.get('nodos')
            pct_operativo = obtener_pct_traccion_operativo(r, pct_trac_ui)
            trains_data.append({
                'idx': idx, 't_ini': r['t_ini'], 't_fin': r['t_fin'],
                'Via': r['Via'], 'km_orig': r['km_orig'], 'km_dest': r['km_dest'],
                'nodos': nodos,
                't_arr': [n[0] for n in nodos] if nodos and len(nodos) >= 2 else None,
                'tipo_tren': r.get('tipo_tren', 'XT-100'),
                'doble': r.get('doble', False),
                'pax_abordo': r.get('pax_abordo', 0),
                'pct_trac': pct_operativo,
                'maniobra': r.get('maniobra')
            })
        
        braking_by_idx = [[] for _ in range(len(time_steps))]
        accel_by_idx = [[] for _ in range(len(time_steps))]
        
        for tr in trains_data:
            idx_start = np.searchsorted(time_steps, max(t_min, tr['t_ini']))
            idx_end = np.searchsorted(time_steps, min(t_max, tr['t_fin']), side='right')
            f = _get_val('FLOTA', {}).get(tr['tipo_tren'], {})
            if not f: continue
            
            eta_m = f.get('eta_motor', 0.92)
            eta_regen_neta = _get_val('ETA_REGEN_NETA', 0.38)
            
            km_man = None
            maniobra_tr = str(tr.get('maniobra', '')).upper()
            if 'CORTE_BTO' in maniobra_tr or 'ACOPLE_BTO' in maniobra_tr or 'CORTE_PU_SA_BTO' in maniobra_tr: km_man = 25.3
            elif 'CORTE_SA' in maniobra_tr or 'ACOPLE_SA' in maniobra_tr: km_man = 29.1
            dist_to_maniobra = abs(km_man - tr['km_orig']) if km_man is not None else -1
            
            for i in range(idx_start, idx_end):
                m = time_steps[i]
                state, v_kmh = get_train_state_and_speed(m, tr['Via'], use_rm, tr['km_orig'], tr['km_dest'], tr['nodos'], tr['t_arr'])
                pos = km_at_t(tr['t_ini'], tr['t_fin'], m, tr['Via'], use_rm, tr['km_orig'], tr['km_dest'], tr['nodos'], tr['t_arr'])
                v_ms = v_kmh / 3.6
                
                dist_recorrida_total = abs(pos - tr['km_orig'])
                es_doble = tr['doble']
                if km_man is not None and min(tr['km_orig'], tr['km_dest']) <= km_man <= max(tr['km_orig'], tr['km_dest']) and maniobra_tr:
                    if 'CORTE' in maniobra_tr: es_doble = True if dist_recorrida_total <= dist_to_maniobra else False
                    elif 'ACOPLE' in maniobra_tr: es_doble = False if dist_recorrida_total <= dist_to_maniobra else True
                
                n_uni = 2 if es_doble else 1
                masa_estatica_kg = (f.get('tara_t', 86.1) * 1000 * n_uni) + (tr['pax_abordo'] * _get_val('PAX_KG', 75.0))
                masa_dinamica_kg = masa_estatica_kg + (f.get('m_iner_t', 7.2) * 1000 * n_uni)
                
                idx_km = min(44999, max(0, int(pos * 1000)))
                pend_permil = _PEND_ARRAY_V1[idx_km] if tr['Via'] == 1 else _PEND_ARRAY_V2[idx_km]
                f_pend = masa_dinamica_kg * 9.81 * (pend_permil / 1000.0)
                f_curva = _CURVA_ARRAY[idx_km] * (masa_estatica_kg / 1000.0) * 9.81
                
                if n_uni == 2: f_davis = (f.get('davis_A',1615)*2) + (f.get('davis_B',0)*2*v_kmh) + (f.get('davis_C',0.54)*1.35*(v_kmh**2))
                else: f_davis = f.get('davis_A',1615) + f.get('davis_B',0)*v_kmh + f.get('davis_C',0.54)*(v_kmh**2)
                f_res_total = f_davis + f_pend + f_curva
                
                if state in ("BRAKE", "BRAKE_STATION", "BRAKE_OVERSPEED"):
                    f_req_freno = max(0.0, masa_dinamica_kg * (f.get('a_freno_ms2', 1.2) * 0.9) - f_res_total)
                    f_disp_freno = min(f.get('f_freno_max_kn', 105.0)*1000*n_uni, (f.get('p_freno_max_kw', f.get('p_max_kw',720)*1.2)*1000*n_uni)/max(0.1, v_ms)) if v_kmh >= f.get('v_freno_min', 3.81) else 0.0
                    potencia_mecanica_frenado_kw = (min(f_req_freno, f_disp_freno) * v_ms) / 1000.0
                    potencia_electrica_generada_kw = potencia_mecanica_frenado_kw * eta_m
                    p_gen_kw = potencia_electrica_generada_kw * eta_regen_neta
                    if p_gen_kw > 0:
                        braking_by_idx[i].append((tr['idx'], pos, p_gen_kw))
                    braking_ticks_per_trip[tr['idx']] += 1
                
                elif state in ("ACCEL", "CRUISE"):
                    if estacion_anio == "invierno": aux_nom = f.get('aux_kw_heat', 65.16) * n_uni
                    else: aux_nom = f.get('aux_kw_cool', 58.76) * n_uni
                    p_aux_kw = calcular_aux_dinamico(tr['tipo_tren'], aux_nom, m / 60.0, tr['pax_abordo'], f.get('cap_max', 398) * n_uni, estacion_anio, state)
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
            braking_via = [b for b in braking_by_idx[i] if b[0] in via_indices]
            accel_via = [a for a in accel_by_idx[i] if a[0] in via_indices]
            if not braking_via or not accel_via: continue
            
            current_demands = {a[0]: a[2] for a in accel_via}
            
            for b_idx, b_pos, p_gen in braking_via:
                if p_gen <= 0: continue
                candidates = []
                for a_idx, a_pos, a_demand in accel_via:
                    if current_demands[a_idx] <= 0: continue
                    dist = abs(a_pos - b_pos)
                    if dist > DISTANCIA_MAXIMA: continue
                    peso = 1.0 / max(0.1, dist)
                    eta_dist = ETA_MAX_VAL * np.exp(-dist / LAMBDA_REGEN)
                    candidates.append({'idx': a_idx, 'demand': current_demands[a_idx], 'dist': dist, 'peso': peso, 'eta_dist': eta_dist})
                
                if not candidates: continue
                
                total_peso = sum(c['peso'] for c in candidates)
                energia_asignada_total = 0.0
                for c in candidates:
                    fraccion_peso = c['peso'] / total_peso if total_peso > 0 else 1.0 / len(candidates)
                    p_ofrecida = p_gen * c['eta_dist'] * fraccion_peso
                    p_transferida = min(p_ofrecida, c['demand'])
                    current_demands[c['idx']] -= p_transferida
                    energia_asignada_total += p_transferida
                
                if p_gen > 0:
                    regen_util_per_trip[b_idx] += min(1.0, energia_asignada_total / p_gen)
    
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
        (trc, aux_catenaria, reg_bruta, reg_panto_push,
         neto_ideal, t_h, prev_aplic) = simular_tramo_termodinamico(
            r['tipo_tren'], r.get('doble', False), r['km_orig'], r['km_dest'], r['Via'], 
            pct_real, use_rm, use_pend, r.get('nodos'), r.get('pax_d', {}), r.get('pax_abordo', 0), 
            None, r.get('maniobra'), estacion_anio, r.get('t_ini', 0.0), False, prevenciones
        )
        
        eta_red = dict_regen.get(r.name, 0.0) if (use_regen and dict_regen) else 0.0
        reg_util = reg_bruta * eta_red
        kwh_reostato = max(0.0, reg_bruta - reg_util)
        neto = max(0.0, trc + aux_catenaria - reg_util)
        
        return pd.Series([trc, aux_catenaria, reg_util, kwh_reostato, neto, t_h, prev_aplic])
        
    df_e[['kwh_viaje_trac', 'kwh_viaje_aux', 'kwh_viaje_regen', 'kwh_reostato',
          'kwh_viaje_neto', 't_viaje_h', 'prevencion_aplicada']] = df_e.apply(_wrapper, axis=1)
    df_e['t_fin'] = df_e['t_ini'] + df_e['t_viaje_h'] * 60.0
    df_e['tren_km'] = df_e.apply(_calc_tren_km_real_motor, axis=1)
        
    import re as _re
    T_PRE_H  = 1.0
    T_POST_H = 10.0/60.0
    F_HVAC_PRE  = 0.30
    F_HVAC_POST = 0.10
    frac_base_pp = _get_val('FRAC_BASE', 0.21)
    frac_hvac_pp = _get_val('FRAC_HVAC', 0.89)
    flota_db_pp  = _get_val('FLOTA', {})

    motrices_viajes = {}
    for idx, row in df_e.iterrows():
        mn = str(row.get('motriz_num', ''))
        nums = _re.findall(r'\d+', mn)
        for n in nums:
            if n not in motrices_viajes: motrices_viajes[n] = []
            motrices_viajes[n].append(idx)

    aux_prepost_por_idx = {idx: 0.0 for idx in df_e.index}
    if not motrices_viajes:
        if 't_fin' in df_e.columns and 't_ini' in df_e.columns:
            t_vals = np.arange(300, 1380, 5)
            t_fin_est = df_e['t_fin'].fillna(df_e['t_ini'] + 70)
            pico = max(int(((df_e['t_ini'] <= t) & (t_fin_est >= t)).sum()) for t in t_vals)
        else:
            pico = max(1, len(df_e) // 7)
        tipo_predominante = df_e['tipo_tren'].mode()[0] if 'tipo_tren' in df_e.columns else 'XT-100'
        f_pp = _get_val('FLOTA', {}).get(tipo_predominante, {})
        aux_nom_pp = f_pp.get('aux_kw_heat', 65.16)
        aux_pre_pp  = aux_nom_pp * frac_base_pp + aux_nom_pp * frac_hvac_pp * F_HVAC_PRE
        aux_post_pp = aux_nom_pp * frac_base_pp + aux_nom_pp * frac_hvac_pp * F_HVAC_POST
        kwh_total_pp = (aux_pre_pp * T_PRE_H + aux_post_pp * T_POST_H) * pico
        kwh_por_viaje_pp = kwh_total_pp / max(1, len(df_e))
        for idx in df_e.index: aux_prepost_por_idx[idx] = kwh_por_viaje_pp
    
    for motriz, idxs in motrices_viajes.items():
        if not idxs: continue
        tipo = df_e.loc[idxs[0], 'tipo_tren'] if idxs[0] in df_e.index else 'XT-100'
        f_pp = flota_db_pp.get(tipo, {})
        aux_nom = f_pp.get('aux_kw_heat', 65.16)
        aux_pre  = aux_nom * frac_base_pp + aux_nom * frac_hvac_pp * F_HVAC_PRE
        aux_post = aux_nom * frac_base_pp + aux_nom * frac_hvac_pp * F_HVAC_POST
        kwh_motriz = aux_pre * T_PRE_H + aux_post * T_POST_H
        kwh_por_viaje = kwh_motriz / len(idxs)
        for idx in idxs: aux_prepost_por_idx[idx] = aux_prepost_por_idx.get(idx, 0.0) + kwh_por_viaje

    df_e['kwh_prepost'] = pd.Series(aux_prepost_por_idx)
    df_e['kwh_viaje_aux']  = df_e['kwh_viaje_aux']  + df_e['kwh_prepost'].fillna(0.0)
    df_e['kwh_viaje_neto'] = df_e['kwh_viaje_neto'] + df_e['kwh_prepost'].fillna(0.0)

    # 💡 Eliminar la columna de diagnóstico para no romper la interfaz
    if 'prevencion_aplicada' in df_e.columns:
        df_e = df_e.drop(columns=['prevencion_aplicada'])

    return df_e
