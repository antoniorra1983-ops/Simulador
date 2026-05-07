import streamlit as st
import pandas as pd
import numpy as np

# Importación segura para entornos Streamlit Cloud
try:
    import config
except ImportError:
    pass

# Función de extracción defensiva para evitar NameErrors por caché
def _get_val(name, default):
    try:
        return getattr(config, name, default)
    except Exception:
        return default

# =============================================================================
# 1. MOTOR CINEMÁTICO TRAMO A TRAMO (GEOMETRÍA Y PERFILES DE VELOCIDAD)
# =============================================================================
def _build_profile(use_rm, via):
    """Construye el perfil de velocidad acumulado en tiempo."""
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

# Pre-cálculo de perfiles para velocidad de ejecución
_PROF = {(v, r): _build_profile(r, v) for v in [1, 2] for r in [False, True]}
_PROF_SORTED = {}
for k, v in _PROF.items(): 
    if k[0] == 1: 
        _PROF_SORTED[k] = (v[0], v[1])
    else: 
        _PROF_SORTED[k] = (v[0][::-1].copy(), v[1][::-1].copy())

_VEL_ARRAY_NORM = np.zeros(45000, dtype=float)
_VEL_ARRAY_RM = np.zeros(45000, dtype=float)

for ki, kf, _, vn, vr in _get_val('SPEED_PROFILE', []):
    start_idx = int(ki)
    end_idx = min(int(kf) + 1, 45000)
    _VEL_ARRAY_NORM[start_idx:end_idx] = vn
    _VEL_ARRAY_RM[start_idx:end_idx] = vr

def vel_at_km(km_km, via, use_rm):
    """Retorna la velocidad máxima permitida en un punto kilométrico."""
    idx = int(km_km * 1000.0)
    if 0 <= idx < 45000: 
        return _VEL_ARRAY_RM[idx] if use_rm else _VEL_ARRAY_NORM[idx]
    return 0.0

def km_at_t(t_ini, t_fin, t, via, use_rm=False, km_orig=None, km_dest=None, nodos=None, t_arr=None):
    """Calcula la posición exacta (Km) en un instante de tiempo t."""
    km_total_limit = _get_val('KM_TOTAL', 43.13)
    
    if nodos is not None and len(nodos) >= 2:
        if t <= nodos[0][0]: return nodos[0][1]
        if t >= nodos[-1][0]: return nodos[-1][1]
        if t_arr is None: t_arr = [n[0] for n in nodos]
        idx = np.searchsorted(t_arr, t)
        
        nodo_A, nodo_B = nodos[idx-1], nodos[idx]
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
    """Determina el estado de marcha y la consigna de velocidad, incluyendo restricciones terminales."""
    if not nodos or len(nodos) < 2: return "CRUISE", 60.0
    if t_arr is None: t_arr = [n[0] for n in nodos]
    if t <= t_arr[0] or t >= t_arr[-1]: return "DWELL", 0.0
    
    idx = np.searchsorted(t_arr, t)
    t_A, t_B = t_arr[idx-1], t_arr[idx]
    dt_from_A, dt_to_B = t - t_A, t_B - t
    km_now = km_at_t(t_A, t_B, t, r_via, use_rm, km_orig, km_dest, nodos, t_arr)
    vel_max = vel_at_km(km_now, r_via, use_rm)
    
    # 💡 RESTRICCIÓN TERMINAL (Buffer Stop Protection)
    km_total_limit = _get_val('KM_TOTAL', 43.13)
    if r_via == 1 and km_now >= km_total_limit - 0.200:
        vel_max = min(vel_max, 10.0 if km_now >= km_total_limit - 0.100 else 20.0)
    elif r_via == 2 and km_now <= 0.200:
        vel_max = min(vel_max, 10.0 if km_now <= 0.100 else 20.0)
            
    if dt_from_A <= 1.0: return "ACCEL", vel_max
    elif dt_to_B <= 1.0: return "BRAKE", vel_max
    else: return "CRUISE", vel_max

# =============================================================================
# 2. CÁLCULO DE AUXILIARES (MODELO BOTTOM-UP INDEPENDIENTE V131)
# =============================================================================
def calcular_aux_dinamico(f_flota, n_uni, hora_decimal, pax_abordo, estacion_anio, estado_marcha, is_precalc=False):
    """
    Nuevo modelo 100% segregado (Bottom-Up).
    El Aire Acondicionado, la Carga Base y los Ventiladores se suman de forma independiente.
    """
    hora_int = int(hora_decimal) % 24
    perfil_dict = _get_val('AUX_HVAC_HORA', {})
    perfil = perfil_dict.get(estacion_anio, perfil_dict.get("primavera", [0.5]*24)) if isinstance(perfil_dict, dict) else [0.5]*24
    f_hvac = perfil[hora_int] if len(perfil) > hora_int else 0.5
    
    cap_max = f_flota.get('cap_max', 398) * n_uni
    ocup = min(1.0, pax_abordo / cap_max) if cap_max > 0 else 0.0
    
    # Factor de ocupación estacional
    if estacion_anio == "verano": f_ocup = 1.0 + 0.05 * ocup
    elif estacion_anio == "invierno": f_ocup = 1.0 - 0.12 * ocup
    else: f_ocup = 1.0 - 0.06 * ocup

    # 1. CARGA BASE VITAL (12% del techo nominal - image_ccc4cc.png)
    techo_base = f_flota.get('aux_kw_cool', 58.76) * n_uni
    p_base = techo_base * 0.12
    
    # 2. CLIMATIZACIÓN HVAC (45% del techo, modulado)
    techo_clima = f_flota.get('aux_kw_heat', 65.16) if estacion_anio == 'invierno' else f_flota.get('aux_kw_cool', 58.76)
    p_clima = (techo_clima * n_uni * 0.45) * f_hvac * f_ocup
    
    # 💡 ESTADO WAKE_UP (Preparación 1 hora antes)
    if estado_marcha == "WAKE_UP":
        # Durante la preparación: Base + 65% Clima + Compresor rellenando
        return p_base + (techo_clima * n_uni * 0.45 * 0.65) + (f_flota.get('p_compresor_kw', 3.68) * n_uni)

    # 3. VENTILACIÓN TRACCIÓN (Afinado de Auxiliares - image_cc4fd2.png)
    p_vent_max = f_flota.get('p_vent_trac_kw', 7.6) * n_uni
    if estado_marcha in ["BRAKE", "BRAKE_STATION", "BRAKE_OVERSPEED"]: 
        p_vent = p_vent_max * 1.0  # 100% para enfriar inversores al regenerar
    elif estado_marcha == "ACCEL": 
        p_vent = p_vent_max * 0.95 # -5% Load shedding protegiendo catenaria
    else: 
        p_vent = p_vent_max * 0.1  # 10% basal al rodar libremente o andén
        
    p_total = p_base + p_clima + p_vent
    
    # Para Load Flow pre-calculado agregamos una media estadística del compresor
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
    
    # Termostato Inteligente V122
    if estacion_anio == "invierno":
        aux_nominal_total = f.get('aux_kw_heat', 65.16)
    else:
        aux_nominal_total = f.get('aux_kw_cool', 58.76)
        
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
    
    dt = 1.0 # Integrador Euler Temporal de 1 segundo
    pax_kg_val = _get_val('PAX_KG', 75.0)
    eta_regen_val = _get_val('ETA_REGEN_NETA', 0.85)

    def _safe_get_pax(km_val, via_val, d_dict, def_val):
        try:
            from etl_parser import get_pax_at_km
            return get_pax_at_km(d_dict, km_val, via_val, def_val)
        except: return def_val

    # ESTADO INICIAL DEL ACUMULADOR NEUMÁTICO (TANQUE MRP)
    mrp_bar = 10.0
    compresor_on = False
    pax_prev = pax_abordo

    for i in range(len(paradas_km)-1):
        p_ini, p_fin = paradas_km[i], paradas_km[i+1]
        dist_total_tramo = abs(p_fin - p_ini) * 1000.0
        if dist_total_tramo <= 0: continue
        
        pos_m, dist_recorrida, v_ms, a_prev = p_ini * 1000.0, 0.0, 0.0, 0.0
        estado_marcha = "ACCEL"
        
        while dist_recorrida < dist_total_tramo:
            dist_restante = dist_total_tramo - dist_recorrida
            if dist_restante < 0.1: break
            
            km_actual = (pos_m + dist_recorrida) / 1000.0 if via_op == 1 else (pos_m - dist_recorrida) / 1000.0
            n_uni = 2 if doble else 1
            
            pax_mid = _safe_get_pax(km_actual, via_op, pax_dict, pax_abordo) if pax_dict else pax_abordo
            masa_kg = ((f['tara_t'] + f['m_iner_t']) * 1000 * n_uni) + (pax_mid * pax_kg_val)
            
            v_cons_kmh = max(5.0, vel_at_km(km_actual, via_op, use_rm))
            if v_consigna_override is not None: v_cons_kmh = min(v_cons_kmh, v_consigna_override)
            
            # 💡 RADAR ATC PREDICTIVO: Acata Prevenciones de Vía (TSR)
            if prevenciones:
                for p in prevenciones:
                    if p['via'] == via_op:
                        if p['km_min'] <= km_actual <= p['km_max']:
                            v_cons_kmh = min(v_cons_kmh, p['v_kmh'])
                        else:
                            d_to_p = (p['km_min'] - km_actual) * 1000.0 if via_op == 1 else (km_actual - p['km_max']) * 1000.0
                            if 0 < d_to_p < 1500.0:
                                v_rest_ms = p['v_kmh'] / 3.6
                                if v_ms > v_rest_ms:
                                    d_freno = (v_ms**2 - v_rest_ms**2) / (2 * (f.get('a_freno_ms2', 1.2) * 0.75))
                                    if d_to_p <= d_freno + (v_ms * dt * 2.0): v_cons_kmh = min(v_cons_kmh, p['v_kmh'])

            # 💡 SQUEEZE CONTROL ACTIVO (Limitación por bajo voltaje)
            factor_squeeze = 1.0
            v_nom, v_warn = _get_val('V_NOMINAL_DC', 3000.0), _get_val('V_SQUEEZE_WARN', 2850.0)
            ser_data = _get_val('SER_DATA', [(3.9, 'SER PO')])
            dist_to_ser = min([abs(km_actual - skm) for skm, _ in ser_data])
            r_linea = 0.045 * dist_to_ser # Ohm/km
            i_req = (f.get('p_max_kw', 720.0) * n_uni / 0.92) / v_nom
            v_pant = v_nom - (i_req * r_linea)
            if v_pant < v_warn: factor_squeeze = max(0.0, (v_pant - 2000.0) / (v_warn - 2000.0))
            
            v_kmh = v_ms * 3.6
            if n_uni == 2: f_davis = (f['davis_A'] * 2) + (f['davis_B'] * 2 * v_kmh) + (f['davis_C'] * 1.35 * (v_kmh**2))
            else: f_davis = f['davis_A'] + f['davis_B']*v_kmh + f['davis_C']*(v_kmh**2)
                
            f_pend = 0.0
            if use_pend:
                elev_km, elev_m = _get_val('ELEV_KM', []), _get_val('ELEV_M', [])
                if elev_km and elev_m:
                    for j in range(1, len(elev_km)):
                        if elev_km[j-1] <= km_actual <= elev_km[j]:
                            pend = ((elev_m[j] - elev_m[j-1]) / max(0.001, (elev_km[j] - elev_km[j-1])*1000)) * 1000
                            f_pend = _get_val('DAVIS_E_N_PERMIL', 9.81) * pend * (masa_kg / 1000.0) * (1.0 if via_op==1 else -1.0)
                            break
                        
            a_freno_op = f.get('a_freno_ms2', 1.2) * 0.9 
            d_freno_req = (v_ms**2) / (2 * a_freno_op)
            
            f_disp_trac = min(f.get('f_trac_max_kn', 110.0)*1000*n_uni*factor_squeeze, (f.get('p_max_kw', 720.0)*1000*n_uni*factor_squeeze)/max(0.1, v_ms))
            f_disp_freno = min(f.get('f_freno_max_kn', 105.0)*1000*n_uni, (f.get('p_max_kw', 720.0)*1.2*1000*n_uni)/max(0.1, v_ms)) if v_kmh >= f.get('v_freno_min', 3.81) else 0.0
            
            if dist_restante <= d_freno_req + (v_ms * dt * 1.2): estado_marcha = "BRAKE_STATION"
            elif v_kmh > v_cons_kmh + 1.5: estado_marcha = "BRAKE_OVERSPEED"
            elif v_kmh >= v_cons_kmh - 0.5 and estado_marcha == "ACCEL": estado_marcha = "COAST"
            elif v_kmh < v_cons_kmh - 2.0 and estado_marcha == "COAST": estado_marcha = "ACCEL"

            f_motor, f_regen_tramo, a_net_target = 0.0, 0.0, 0.0
            if estado_marcha == "BRAKE_STATION":
                f_req_freno = max(0.0, masa_kg * a_freno_op - f_davis - f_pend)
                f_regen_tramo = min(f_req_freno, f_disp_freno)
                a_net_target = max(-a_freno_op, (-f_regen_tramo - f_davis - f_pend) / masa_kg)
            elif estado_marcha == "ACCEL":
                f_motor = f_disp_trac
                a_net_target = (f_motor - f_davis - f_pend) / masa_kg
            elif estado_marcha == "COAST":
                a_net_target = (-f_davis - f_pend) / masa_kg
                
            # Límite de Jerk Dinámico V130
            j_l = f.get('jerk_ms3', 1.3) * dt
            a_net = np.clip(a_net_target, a_prev - j_l, a_prev + j_l)
            a_prev = a_net
            
            v_new, dt_actual = v_ms + a_net * dt, dt
            if v_new < 0: dt_actual, v_new = (v_ms / abs(a_net) if a_net < -0.001 else dt), 0.0
            if f_motor > 0 and v_new * 3.6 > v_cons_kmh:
                v_new = v_cons_kmh / 3.6
                a_req = (v_new - v_ms) / dt_actual if dt_actual > 0 else 0
                f_motor = max(0.0, min(masa_kg * a_req + f_davis + f_pend, f_disp_trac))
                
            step_m = (v_ms + v_new) / 2.0 * dt_actual
            if step_m > dist_restante: step_m = dist_restante
            
            if f_motor > 0: trc += ((f_motor * step_m) / 3_600_000.0) / f.get('eta_motor', 0.92)
            if f_regen_tramo > 0: reg += ((f_regen_tramo * step_m) / 3_600_000.0) * eta_regen_val
                
            # 💡 CICLO DEL COMPRESOR (8 a 10 Bares)
            if mrp_bar <= 8.0: compresor_on = True
            if compresor_on:
                p_comp_inst = f.get('p_compresor_kw', 3.68) * n_uni
                mrp_bar += 0.0122 * dt_actual # Tasa de recuperación real
                if mrp_bar >= 10.0: mrp_bar, compresor_on = 10.0, False
            else: p_comp_inst = 0.0
                
            hora_actual = (t_ini_mins + t_horas * 60.0) / 60.0
            aux += (calcular_aux_dinamico(f, n_uni, hora_actual, pax_mid, estacion_anio, estado_marcha) + p_comp_inst) * (dt_actual / 3600.0)
            t_horas += dt_actual / 3600.0
            dist_recorrida += step_m
            v_ms = v_new

        # 💡 DETENCIÓN EN ANDÉN (image_cb80de.png)
        if i < len(paradas_km) - 2:
            dwell_s = _get_val('DWELL_DEF', 25.0)
            mrp_bar -= 0.3 # Gasto neumático de cilindros
            pax_mid = _safe_get_pax(paradas_km[i+1], via_op, pax_dict, pax_abordo)
            if f.get('tipo_tren', 'XT-100') != 'XT-100': mrp_bar -= (max(0, pax_mid - pax_prev) * 0.002) # Gasto Balonas
            pax_prev = pax_mid
            
            # Apertura de puertas (image_cb80de.png)
            p_puertas = f.get('p_puertas_kw', 0.9) * n_uni
            aux += (p_puertas * 3.0 / 3600.0)
            
            for _ in range(int(dwell_s)):
                if mrp_bar <= 8.0: compresor_on = True
                p_comp_inst = (f.get('p_compresor_kw', 3.68) * n_uni) if compresor_on else 0.0
                if compresor_on:
                    mrp_bar += 0.0122 * 1.0
                    if mrp_bar >= 10.0: mrp_bar, compresor_on = 10.0, False
                hora_actual = (t_ini_mins + t_horas * 60.0) / 60.0
                aux += (calcular_aux_dinamico(f, n_uni, hora_actual, pax_mid, estacion_anio, "DWELL") + p_comp_inst) * (1.0 / 3600.0)
                t_horas += (1.0 / 3600.0)

    return trc, aux, reg, 0.0, max(0.0, trc + aux - reg), t_horas

def calcular_energia_preparacion(tipo_tren, doble, estacion_anio, hora_salida_mins):
    """Calcula 1 hora de consumo de preparación (Wake-Up) antes del servicio."""
    f = _get_val('FLOTA', {}).get(tipo_tren, {})
    n_uni = 2 if doble else 1
    e_prep = 0.0
    for m in range(60):
        h_sim = (hora_salida_mins - 60 + m) / 60.0
        e_prep += (calcular_aux_dinamico(f, n_uni, h_sim, 0, estacion_anio, "WAKE_UP") / 60.0)
    return e_prep

def calcular_receptividad_por_headway(df_dia: pd.DataFrame) -> dict:
    if df_dia.empty: return {}
    result = {}
    for via in [1, 2]:
        sub = df_dia[df_dia["Via"] == via].sort_values("t_ini").copy()
        if sub.empty: continue
        t_ini_vals = sub["t_ini"].values
        for i, idx in enumerate(list(sub.index)):
            hw = min([t_ini_vals[i]-t_ini_vals[i-1]] if i>0 else [99])
            if i < len(t_ini_vals)-1: hw = min(hw, t_ini_vals[i+1]-t_ini_vals[i])
            if hw < 5.0: eta = 0.90
            elif hw < 10.0: eta = 0.75 - ((hw - 5.0) / 5.0) * 0.45
            else: eta = max(0.10, 0.30 - ((hw - 10.0) / 20.0) * 0.20)
            result[idx] = min(eta, 0.90)
    return result

@st.cache_data(show_spinner="Simulando malla eléctrica y receptividad...")
def precalcular_red_electrica_v111(df_dia, pct_trac, use_rm, estacion_anio="primavera"):
    regen_util_per_trip, braking_ticks = {idx: 0.0 for idx in df_dia.index}, {idx: 0.0 for idx in df_dia.index} 
    if df_dia.empty: return regen_util_per_trip
    t_steps = np.arange(int(df_dia['t_ini'].min()), int(df_dia['t_fin'].max()) + 1, 10.0 / 60.0)
    
    for via_ in [1, 2]:
        via_trains = df_dia[df_dia['Via'] == via_]
        if via_trains.empty: continue
        braking_by_idx, accel_by_idx = [[] for _ in range(len(t_steps))], [[] for _ in range(len(t_steps))]
        for idx, r in via_trains.iterrows():
            f = _get_val('FLOTA', {}).get(r['tipo_tren'], {})
            n_u = 2 if r['doble'] else 1
            masa = ((f['tara_t'] + f['m_iner_t']) * 1000 * n_u) + (r['pax_abordo'] * _get_val('PAX_KG', 75))
            for i in range(np.searchsorted(t_steps, r['t_ini']), np.searchsorted(t_steps, r['t_fin'], side='right')):
                st_m, v_k = get_train_state_and_speed(t_steps[i], via_, use_rm, r['km_orig'], r['km_dest'], r['nodos'])
                pos = km_at_t(r['t_ini'], r['t_fin'], t_steps[i], via_, use_rm, r['km_orig'], r['km_dest'], r['nodos'])
                p_aux = calcular_aux_dinamico(f, n_u, t_steps[i]/60, r['pax_abordo'], estacion_anio, st_m, True)
                if st_m == "BRAKE_STATION":
                    p_gen = ((masa * 1.08 * v_k/3.6) / 1000.0 * 0.85) - p_aux
                    if p_gen > 0: braking_by_idx[i].append((idx, pos, p_gen))
                    braking_ticks[idx] += 1
                elif st_m == "ACCEL":
                    p_dem = p_aux + ((f['p_max_kw'] * n_u * 0.8) / 0.92)
                    accel_by_idx[i].append((idx, pos, p_dem))
                    
        for i in range(len(t_steps)):
            if not braking_by_idx[i] or not accel_by_idx[i]: continue
            curr_demands = {a[0]: a[2] for a in accel_by_idx[i]}
            for b_idx, b_pos, p_gen in braking_by_idx[i]:
                available = [a for a in accel_by_idx[i] if curr_demands[a[0]] > 0]
                if not available: break 
                a_idx, a_pos, _ = min(available, key=lambda x: abs(x[1] - b_pos))
                dist = abs(a_pos - b_pos)
                if dist <= 10.0:
                    p_tx = min(p_gen * (0.70 * np.exp(-dist / 5.0)), curr_demands[a_idx])
                    curr_demands[a_idx] -= p_tx
                    regen_util_per_trip[b_idx] += (p_tx / p_gen)
                    
    for idx in df_dia.index: 
        regen_util_per_trip[idx] = min(1.0, regen_util_per_trip[idx] / braking_ticks[idx]) if braking_ticks[idx] > 0 else 0.0
    return regen_util_per_trip

@st.cache_data(show_spinner="Integrando Termodinámica de Flota...")
def calcular_termodinamica_flota_v111(df_dia, pct_trac, use_pend, use_rm, use_regen, dict_regen, estacion_anio="primavera", prevenciones=None):
    df_e = df_dia.copy()
    if df_e.empty: return df_e
    
    agrupador = 'motriz_num' if 'motriz_num' in df_e.columns else '_id'
    primeras_salidas = df_e.sort_values('t_ini').groupby(agrupador).first().index
    
    def _wrapper_energia(r):
        trc, aux, reg_p_max, _, _, t_h = simular_tramo_termodinamico(
            r['tipo_tren'], r.get('doble', False), r['km_orig'], r['km_dest'], r['Via'], 
            pct_trac, use_rm, use_pend, r.get('nodos'), r.get('pax_d', {}), r.get('pax_abordo', 0), 
            None, r.get('maniobra'), estacion_anio, r.get('t_ini', 0.0), False, prevenciones
        )
        e_prep = calcular_energia_preparacion(r['tipo_tren'], r.get('doble', False), estacion_anio, r['t_ini']) if r.name in primeras_salidas else 0.0
        reg_u = reg_p_max * dict_regen.get(r.name, 1.0) if use_regen else 0.0
        return pd.Series([trc, aux + e_prep, reg_u, max(0.0, reg_p_max - reg_u), max(0.0, trc + aux + e_prep - reg_u)])
        
    df_e[['kwh_viaje_trac', 'kwh_viaje_aux', 'kwh_viaje_regen', 'kwh_reostato', 'kwh_viaje_neto']] = df_e.apply(_wrapper_energia, axis=1)
    return df_e
