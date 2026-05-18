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

def get_train_state_and_speed(t, r_via, use_rm, km_orig, km_dest, nodos=None, t_arr=None, t_ini_ext=None, t_fin_ext=None):
    # t_ini_ext/t_fin_ext: límites temporales externos (para cuando t_arr tiene todos ceros)
    dt_sample = 0.5
    
    # Determinar límites temporales efectivos
    if t_arr is not None and len(t_arr) >= 2 and any(v > 0 for v in t_arr):
        t0 = t_arr[0]; tf = t_arr[-1]
    elif t_ini_ext is not None and t_fin_ext is not None:
        t0 = t_ini_ext; tf = t_fin_ext; t_arr = None  # usar interpolación lineal
    elif nodos and len(nodos) >= 2:
        t0 = t_ini_ext or 0; tf = t_fin_ext or (t0 + 60); t_arr = None
    else:
        return "CRUISE", 60.0
    
    if t <= t0 or t >= tf: return "DWELL", 0.0
    
    km_now  = km_at_t(t0, tf, t,            r_via, use_rm, km_orig, km_dest, nodos, t_arr)
    km_prev = km_at_t(t0, tf, t-dt_sample,  r_via, use_rm, km_orig, km_dest, nodos, t_arr)
    km_next = km_at_t(t0, tf, t+dt_sample,  r_via, use_rm, km_orig, km_dest, nodos, t_arr)
    v_now  = abs(km_now - km_prev) * 1000.0 / dt_sample
    v_next = abs(km_next - km_now) * 1000.0 / dt_sample
    a_ms2  = (v_next - v_now) / dt_sample
    v_kmh  = v_now * 3.6
    if a_ms2 > 0.15:  return "ACCEL", v_kmh
    elif a_ms2 < -0.15: return "BRAKE", v_kmh
    elif v_kmh < 1.0: return "DWELL", 0.0
    else:             return "CRUISE", v_kmh

# =============================================================================
# 2.5. FUNCIÓN DE AUXILIARES CON DUTY CYCLE REAL
# =============================================================================
def calcular_aux_dinamico(tipo_tren, aux_kw_nominal, hora_decimal, pax_abordo, cap_max, estacion_anio, estado_marcha="CRUISE"):
    # ── Perfil horario HVAC ──────────────────────────────────────────────────
    hora_int = int(hora_decimal) % 24
    try: perfil = _get_val('AUX_HVAC_HORA', {}).get(estacion_anio, [0.5]*24)
    except: perfil = [0.5]*24
    if not perfil or len(perfil) < 24: perfil = [0.5]*24
    f_hvac = perfil[hora_int]

    # Factor de ocupación sobre HVAC (más pax → más carga térmica en verano,
    # menos en invierno por calor corporal)
    f_ocup = 1.0
    if cap_max > 0:
        ocup = min(1.0, pax_abordo / cap_max)
        if estacion_anio == "verano":   f_ocup = 1.0 + 0.05 * ocup
        elif estacion_anio == "invierno": f_ocup = 1.0 - 0.12 * ocup
        else:                             f_ocup = 1.0 - 0.06 * ocup

    # ── Potencias reales por componente (desde manuales de fabricante) ───────
    # HVAC: potencia eléctrica de entrada del compresor del AA
    #   aux_kw_nominal ya es la potencia total correcta del tren (cool o heat)
    #   La fracción HVAC varía según tren — calculada desde los manuales:
    #     XT-100 cool: 46.4 kW de 68 kW = 68%
    #     XT-M   cool: 28.0 kW de 52 kW = 54%
    #     SFE    cool: 63.2 kW de 97 kW = 65%
    #   Invierno: calefacción eléctrica directa (sin COP)
    #     XT-100 heat: ~40 kW de 67 kW = 60%
    #     XT-M   heat:  59.2 kW de 84 kW = 70%
    #     SFE    heat:  42.4 kW de 72 kW = 59%
    hvac_fracs = {
        "XT-100": {"verano": 0.68, "primavera": 0.68, "otoño": 0.68, "invierno": 0.60},
        "XT-M":   {"verano": 0.54, "primavera": 0.54, "otoño": 0.54, "invierno": 0.70},
        "SFE":    {"verano": 0.65, "primavera": 0.65, "otoño": 0.65, "invierno": 0.59},
    }
    frac_hvac = hvac_fracs.get(tipo_tren, {}).get(estacion_anio, 0.65)
    aux_hvac = aux_kw_nominal * frac_hvac * f_hvac * f_ocup

    # Compresor neumático principal: potencia real × duty según estado
    #   XT-100: compresor espiral, potencia estimada ~9.8 kW (igual al XT-M)
    #   XT-M:   2 × VV80T 14 kW = 28 kW pico → duty ciclo real
    #   SFE:    2 compresores CA, estimado ~15 kW pico
    # Duty: en DWELL el compresor llena el depósito (máximo trabajo)
    #       en CRUISE poco consumo (depósito lleno, solo fugas mínimas)
    comp_pico_kw = {"XT-100": 9.8, "XT-M": 9.8, "SFE": 15.0}
    duty_comp_map = {
        "XT-100": {"ACCEL": 0.15, "CRUISE": 0.10, "BRAKE": 0.40,
                   "BRAKE_STATION": 0.40, "BRAKE_OVERSPEED": 0.40,
                   "COAST": 0.10, "DWELL": 0.50},
        "XT-M":   {"ACCEL": 0.20, "CRUISE": 0.15, "BRAKE": 0.55,
                   "BRAKE_STATION": 0.55, "BRAKE_OVERSPEED": 0.55,
                   "COAST": 0.15, "DWELL": 0.80},
        "SFE":    {"ACCEL": 0.25, "CRUISE": 0.20, "BRAKE": 0.60,
                   "BRAKE_STATION": 0.60, "BRAKE_OVERSPEED": 0.60,
                   "COAST": 0.20, "DWELL": 0.85},
    }
    p_comp = comp_pico_kw.get(tipo_tren, 9.8)
    dc_comp = duty_comp_map.get(tipo_tren, duty_comp_map["XT-100"]).get(estado_marcha, 0.15)
    aux_comp = p_comp * dc_comp

    # Ventilación convertidores de tracción: proporcional a la carga del motor
    # Alta en ACCEL (máxima disipación), baja en DWELL (motor frío)
    vent_pico_kw = {"XT-100": 4.0, "XT-M": 6.0, "SFE": 8.0}
    duty_vent = {
        "ACCEL": 1.00, "CRUISE": 0.60, "BRAKE": 0.50,
        "BRAKE_STATION": 0.50, "BRAKE_OVERSPEED": 0.50,
        "COAST": 0.20, "DWELL": 0.10
    }
    dc_vent = duty_vent.get(estado_marcha, 0.30)
    aux_vent = vent_pico_kw.get(tipo_tren, 4.0) * dc_vent

    # Base: alumbrado + electrónica + cargador baterías + enchufes
    # Todo lo que no depende del estado de marcha ni del HVAC
    base_kw = {"XT-100": 8.0, "XT-M": 6.0, "SFE": 10.0}
    aux_base = base_kw.get(tipo_tren, 8.0)

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
        dist_antes = abs(km_man - k_o); dist_despues = abs(k_d - km_man)
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
    tiempos_estaciones = []  # (t_mins_absoluto, km) para cada estación simulada
    perfil_potencia    = []  # (t_mins, km, v_kmh, estado, p_regen_kw) cada dt=1s
    
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
        
        # Registrar tiempo de salida de esta estación
        t_est_mins = t_ini_mins + t_horas * 60.0
        tiempos_estaciones.append((t_est_mins, p_ini))
        
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
            
            # Largo real del tren desde config (metros → km)
            _largo_unit = f.get('largo_m', 72.0) / 1000.0
            long_tren_km = _largo_unit * n_uni_inst
            
            pax_mid = get_pax_at_km_nativo(pax_dict, km_actual, via_op, pax_abordo) if pax_dict else pax_abordo
            pax_kg_total = pax_mid * _get_val('PAX_KG', 75.0)
            
            masa_estatica_kg = (f.get('tara_t', 86.1) * 1000 * n_uni_inst) + pax_kg_total
            masa_dinamica_kg = masa_estatica_kg + (f.get('m_iner_t', 7.2) * 1000 * n_uni_inst)
            
            f_trac_max_n_nominal = f.get('f_trac_max_kn', 110.0) * 1000 * n_uni_inst
            p_max_w_nominal = f.get('p_max_kw', 720.0) * 1000 * n_uni_inst
            f_freno_max_n = f.get('f_freno_max_kn', 105.0) * 1000 * n_uni_inst
            p_freno_max_w = f.get('p_freno_max_kw', f.get('p_max_kw', 720.0)*1.2) * 1000 * n_uni_inst
            
            if estacion_anio in ("invierno", "otoño"): aux_kw_nominal = f.get('aux_kw_heat', 67.0) * n_uni_inst
            else: aux_kw_nominal = f.get('aux_kw_cool', 68.0) * n_uni_inst
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

            # Restricción de terminal — considera largo del tren
            # La velocidad se reduce cuando la CABEZA del tren está en zona de andén
            # y debe mantenerse hasta que el ÚLTIMO VAGÓN haya cruzado el umbral
            # V1 terminal Limache (km 43.13): zona restringida desde km 42.93
            if via_op == 1:
                if km_actual >= (43.13 - long_tren_km - 0.20):  # 200m antes del andén
                    v_cons_kmh = min(v_cons_kmh, 20.0)
                if km_actual >= (43.13 - long_tren_km - 0.10):  # 100m antes del andén
                    v_cons_kmh = min(v_cons_kmh, 10.0)
            # V2 terminal Puerto (km 0.00): zona restringida hasta km 0.20
            if via_op == 2:
                if km_actual <= (long_tren_km + 0.20):  # hasta que todo el tren salió
                    v_cons_kmh = min(v_cons_kmh, 20.0)
                if km_actual <= (long_tren_km + 0.10):  # zona más cercana al andén
                    v_cons_kmh = min(v_cons_kmh, 10.0)
            
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
                f_limite_potencia = p_max_op_w_real / max(0.1, v_ms)
                f_absoluta_disp = min(f_disp_trac_real, f_limite_potencia)
                f_piloto = f_trac_max_n_nominal * (pct_trac / 100.0)
                f_piloto_disp = min(f_piloto, f_limite_potencia)
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
            # Aceleración bifásica: algunos trenes (XT-100) tienen a_max reducida
            # a alta velocidad (p.ej. 0.71 m/s² sobre 55 km/h — manual TRA 001)
            a_max_ms2   = f.get('a_max_ms2',  1.0)
            a_max_v2    = f.get('a_max_ms2_v2', a_max_ms2)   # 2° régimen (por defecto = 1° régimen)
            v_trans_kmh = f.get('v_trans_accel_kmh', 999.0)  # velocidad de transición
            a_max_actual = a_max_v2 if v_kmh > v_trans_kmh else a_max_ms2
            a_freno_ms2 = f.get('a_freno_ms2', 1.2)
            a_net = max(-a_freno_ms2, min(a_max_actual, a_net))
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
                # Eficiencia del motor: usa eta_base en todos los regímenes.
                # No se aplica penalización por carga parcial — no hay datos
                # técnicos certificados del XT-100/XT-M que la respalden.
                eta_din = f.get('eta_motor', 0.92)
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
                    # reg_exportable = energía eléctrica disponible en catenaria
                    # La receptividad (ETA_REGEN_NETA) la aplica precalcular_red
                    # o el wrapper según el modelo seleccionado
                    reg_exportable += excedente_kwh
                else:
                    deficit_kwh = aux_kwh_step - energia_electrica_kwh
                    aux_catenaria += deficit_kwh
            else:
                aux_catenaria += aux_kwh_step
            
            # Guardar perfil segundo a segundo para precalcular_red
            km_actual_sim = (pos_m + dist_recorrida) / 1000.0 if via_op == 1 else (pos_m - dist_recorrida) / 1000.0
            p_regen_sim = (min(abs(f_real_total), f_disp_freno) * max(0.0, v_ms) / 1000.0 * eta_motor
                          if f_real_total < 0 and estado_marcha in ['BRAKE_STATION','BRAKE_OVERSPEED','COAST'] else 0.0)
            perfil_potencia.append((t_ini_mins + t_horas*60.0, km_actual_sim,
                                    v_ms*3.6, estado_marcha, p_regen_sim))
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

    t_final_mins = t_ini_mins + t_horas * 60.0
    if paradas_km:
        tiempos_estaciones.append((t_final_mins, paradas_km[-1]))
    neto_ideal = max(0.0, trc + aux_catenaria - reg_exportable)
    # Slot 4: dict con tiempos por estación y perfil segundo a segundo
    datos_sim = {'t_est': tiempos_estaciones, 'perfil': perfil_potencia}
    return trc, aux_catenaria, reg_exportable, datos_sim, neto_ideal, t_horas, prevencion_aplicada

# =============================================================================
# 5. PRE-CALCULADORES DE RED
# =============================================================================
def calcular_receptividad_por_headway(df_dia: pd.DataFrame) -> dict:
    """Receptividad de la red DC según headway entre trenes de la misma vía.
    
    Retorna eta_red = receptividad_headway × ETA_REGEN_NETA, que representa
    la fracción de la energía eléctrica disponible en catenaria (reg_bruta,
    ya sin ETA_REGEN_NETA aplicado) que efectivamente llega a otro tren.
    
    Cadena: E_cin × eta_motor (reg_bruta) × eta_red → reg_util
    """
    if df_dia.empty: return {}
    ETA_REGEN = _get_val('ETA_REGEN_NETA', 0.38)  # eficiencia eléctrica + pérdidas DC
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
                result[idx] = 0.10 * ETA_REGEN
                continue
            hw = min(headways)
            # Receptividad según headway: menor headway → más probable que haya receptor
            if hw < 5.0:   recep = 0.90
            elif hw < 10.0: recep = 0.75 - ((hw - 5.0) / 5.0) * 0.45
            else:           recep = max(0.10, 0.30 - ((hw - 10.0) / 20.0) * 0.20)
            # eta_red = receptividad × eficiencia eléctrica DC
            result[idx] = min(recep, 0.90) * ETA_REGEN
    return result

def precalcular_red_electrica_v111(df_dia, pct_trac_ui, use_rm, estacion_anio="primavera"):
    """
    Calcula la receptividad de la red DC para cada viaje usando el perfil de potencia
    regenerada segundo a segundo calculado por simular_tramo_termodinamico.
    
    Si df_dia tiene columna 'datos_sim' (generada por calcular_termodinamica_flota_v111),
    usa el perfil real del motor. Si no, usa el modelo Probabilístico como fallback.
    
    Retorna dict {idx: eta_red} donde eta_red es la fracción de energía regenerada
    que efectivamente llega a otro tren via la catenaria DC.
    """
    regen_util_per_trip = {idx: 0.0 for idx in df_dia.index}
    if df_dia.empty:
        return regen_util_per_trip

    # Si no hay datos_sim del motor, usar modelo probabilístico como fallback
    if 'datos_sim' not in df_dia.columns or df_dia['datos_sim'].isna().all():
        return calcular_receptividad_por_headway(df_dia)

    LAMBDA_REGEN  = _get_val('LAMBDA_REGEN_KM', 5.0)
    ETA_MAX_VAL   = _get_val('ETA_MAX', 0.70)
    DIST_MAX      = LAMBDA_REGEN * 2.5  # km máximo para transferencia

    # Construir índice temporal de eventos: cada paso del perfil es un evento
    # Estructura: { t_bin → [(idx_viaje, km, p_regen_kw, p_dem_kw)] }
    DT_BIN = 10.0 / 60.0  # bins de 10 segundos en minutos

    # Recopilar todos los eventos de frenada y tracción por bin temporal
    braking_events  = {}   # t_bin → [(idx, km, p_regen_kw)]
    accel_events    = {}   # t_bin → [(idx, km, p_dem_kw)]
    braking_ticks   = {idx: 0 for idx in df_dia.index}

    # Cada vía tiene su propia catenaria DC independiente
    # La regeneración solo ocurre entre trenes de la MISMA vía
    for via_ in [1, 2]:
        via_df = df_dia[df_dia['Via'] == via_]
        for idx, r in via_df.iterrows():
            datos = r.get('datos_sim')
            if not datos or not isinstance(datos, dict):
                continue
            perfil = datos.get('perfil', [])
            if not perfil:
                continue
            f = _get_val('FLOTA', {}).get(r.get('tipo_tren', 'XT-100'), {})
            n_uni = 2 if r.get('doble', False) else 1

            for t_mins, km, v_kmh, estado, p_regen_kw in perfil:
                t_bin = round(t_mins / DT_BIN) * DT_BIN

                if p_regen_kw > 0:
                    # p_regen_kw = E_cin × eta_motor (potencia eléctrica en catenaria)
                    # Es la energía DISPONIBLE para otro tren en la misma vía
                    if t_bin not in braking_events:
                        braking_events[t_bin] = []
                    braking_events[t_bin].append((idx, km, p_regen_kw, via_))
                    braking_ticks[idx] += 1

                elif estado == 'ACCEL' and v_kmh > 1.0:
                    # Potencia demandada estimada desde el perfil real
                    # Usando p_max como proxy de la demanda en ACCEL
                    p_dem = f.get('p_max_kw', 720.0) * n_uni * 0.75
                    if t_bin not in accel_events:
                        accel_events[t_bin] = []
                    accel_events[t_bin].append((idx, km, p_dem, via_))

    # Para cada bin temporal: transferir energía de frenadores a aceleradores cercanos
    regen_asignada = {idx: 0.0 for idx in df_dia.index}

    for t_bin, frens in braking_events.items():
        acels = accel_events.get(t_bin, [])
        if not acels:
            continue
        demanda_restante = {a[0]: a[2] for a in acels}

        for b_idx, b_km, p_gen, b_via in frens:
            if p_gen <= 0:
                continue
            candidatos = []
            for a_idx, a_km, a_dem, a_via in acels:
                # Solo misma vía — cada catenaria DC es independiente
                if a_via != b_via:
                    continue
                if demanda_restante.get(a_idx, 0) <= 0:
                    continue
                dist = abs(a_km - b_km)
                if dist > DIST_MAX:
                    continue
                eta_dist = ETA_MAX_VAL * np.exp(-dist / LAMBDA_REGEN)
                candidatos.append((a_idx, demanda_restante[a_idx], eta_dist, dist))

            if not candidatos:
                continue

            total_peso = sum(1.0 / max(0.1, c[3]) for c in candidatos)
            energia_transferida = 0.0
            for a_idx, dem, eta_d, dist in candidatos:
                fraccion = (1.0 / max(0.1, dist)) / total_peso
                p_ofrecida = p_gen * eta_d * fraccion
                p_transf = min(p_ofrecida, demanda_restante[a_idx])
                demanda_restante[a_idx] -= p_transf
                energia_transferida += p_transf

            regen_asignada[b_idx] += min(1.0, energia_transferida / p_gen)

    # Normalizar: eta_red = fracción de p_gen que llegó a otro tren
    # Multiplicar por ETA_REGEN_NETA (eficiencia eléctrica DC + pérdidas catenaria)
    ETA_REGEN = _get_val('ETA_REGEN_NETA', 0.38)
    for idx in df_dia.index:
        ticks = braking_ticks.get(idx, 0)
        if ticks > 0:
            regen_util_per_trip[idx] = min(1.0, regen_asignada[idx] / ticks) * ETA_REGEN
        else:
            regen_util_per_trip[idx] = 0.0

    return regen_util_per_trip

def calcular_termodinamica_flota_v111(df_dia, pct_trac_ui, use_pend, use_rm, use_regen, dict_regen, estacion_anio="primavera", prevenciones=None):
    df_e = df_dia.copy()
    if df_e.empty: return df_e
    
    # Simular todos los viajes y capturar tiempos por estación
    # para alimentar precalcular_red con datos reales del motor
    _tiempos_sim = {}  # idx → lista (t_mins, km)
    
    def _wrapper(r):
        pct_real = obtener_pct_traccion_operativo(r, pct_trac_ui)
        (trc, aux_catenaria, reg_bruta, t_est_list,
         neto_ideal, t_h, prev_aplic) = simular_tramo_termodinamico(
            r['tipo_tren'], r.get('doble', False), r['km_orig'], r['km_dest'], r['Via'], 
            pct_real, use_rm, use_pend, r.get('nodos'), r.get('pax_d', {}), r.get('pax_abordo', 0), 
            None, r.get('maniobra'), estacion_anio, r.get('t_ini', 0.0), False, prevenciones
        )
        # Guardar perfil completo para precalcular_red
        if isinstance(t_est_list, dict):
            _tiempos_sim[r.name] = t_est_list
        
        # reg_bruta: energía eléctrica disponible en catenaria (E_cin × eta_motor)
        # La receptividad de la red se aplica aquí:
        #   - Modelo físico: eta_red viene de precalcular_red (receptividad calculada)
        #   - Modelo probabilístico: eta_red = ETA_REGEN_NETA (receptividad calibrada)
        #   - Sin modelo: eta_red = ETA_REGEN_NETA por defecto
        eta_regen_def = _get_val('ETA_REGEN_NETA', 0.38)
        if use_regen and dict_regen:
            eta_red = dict_regen.get(r.name, eta_regen_def)
        else:
            eta_red = eta_regen_def
        reg_util = reg_bruta * eta_red
        kwh_reostato = max(0.0, reg_bruta - reg_util)
        neto = max(0.0, trc + aux_catenaria - reg_util)
        
        return pd.Series([trc, aux_catenaria, reg_util, kwh_reostato, neto, t_h, prev_aplic])
        
    df_e[['kwh_viaje_trac', 'kwh_viaje_aux', 'kwh_viaje_regen', 'kwh_reostato',
          'kwh_viaje_neto', 't_viaje_h', 'prevencion_aplicada']] = df_e.apply(_wrapper, axis=1)
    
    # Enriquecer df_e con nodos_sim (tiempos reales por estación del motor)
    # precalcular_red_electrica_v111 los usará si están disponibles
    df_e['datos_sim'] = df_e.index.map(lambda idx: _tiempos_sim.get(idx))
    df_e['t_fin'] = df_e['t_ini'] + df_e['t_viaje_h'] * 60.0
    df_e['tren_km']    = df_e.apply(_calc_tren_km_real_motor, axis=1)
    # dist_via_km: distancia de vía recorrida (sin factor doble) — para IDE
    df_e['dist_via_km'] = df_e.apply(
        lambda r: abs(r.get('km_dest',0.0)-r.get('km_orig',0.0)), axis=1
    )
        
    import re as _re
    T_PRE_H  = 1.0
    T_POST_H = 10.0/60.0
    F_HVAC_PRE  = 0.30
    F_HVAC_POST = 0.10
    frac_base_pp = _get_val('FRAC_BASE', 0.15)  # base = 15% del nominal
    frac_hvac_pp = _get_val('FRAC_HVAC', 0.50)  # HVAC al 50% en prepost
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
        aux_nom_pp = f_pp.get('aux_kw_heat', 67.0)
        aux_pre_pp  = aux_nom_pp * frac_base_pp + aux_nom_pp * frac_hvac_pp * F_HVAC_PRE
        aux_post_pp = aux_nom_pp * frac_base_pp + aux_nom_pp * frac_hvac_pp * F_HVAC_POST
        kwh_total_pp = (aux_pre_pp * T_PRE_H + aux_post_pp * T_POST_H) * pico
        kwh_por_viaje_pp = kwh_total_pp / max(1, len(df_e))
        for idx in df_e.index: aux_prepost_por_idx[idx] = kwh_por_viaje_pp
    
    for motriz, idxs in motrices_viajes.items():
        if not idxs: continue
        tipo = df_e.loc[idxs[0], 'tipo_tren'] if idxs[0] in df_e.index else 'XT-100'
        f_pp = flota_db_pp.get(tipo, {})
        aux_nom = f_pp.get('aux_kw_heat', 67.0)
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
