import numpy as np
import pandas as pd
import re

# Importación segura de configuración
try:
    import config
except ImportError:
    pass

# Importación defensiva del módulo de datos
try:
    from etl_parser import get_pax_at_km_nativo
except ImportError:
    # Fallback si el parser no está disponible para evitar caídas
    def get_pax_at_km_nativo(pax_d, km_pos, via, pax_max_fallback=0):
        return pax_max_fallback

# =============================================================================
# 1. UTILIDADES CINEMÁTICAS (OPTIMIZADAS PARA ALTO RENDIMIENTO)
# =============================================================================

def vel_at_km(km_km, via, use_rm):
    """
    Busca la velocidad máxima permitida en un punto kilométrico exacto.
    Utiliza una búsqueda optimizada en el perfil de velocidad.
    """
    sp = getattr(config, 'SPEED_PROFILE', [])
    # Crear array de búsqueda si no existe en caché de ejecución (para velocidad)
    v_arr = np.zeros(45000)
    for ki, kf, _, vn, vr in sp:
        v_arr[int(ki):min(int(kf)+1, 45000)] = vr if use_rm else vn
    
    idx = int(km_km * 1000.0)
    return v_arr[idx] if 0 <= idx < 45000 else 0.0

def km_at_t(t_ini, t_fin, t, via, use_rm=False, km_orig=None, km_dest=None, nodos=None, t_arr=None):
    """
    Interpola la posición del tren en un instante de tiempo 't'.
    Soporta nodos de paradas reales para máxima precisión histórica.
    """
    if nodos is not None and len(nodos) >= 2:
        if t <= nodos[0][0]: return nodos[0][1]
        if t >= nodos[-1][0]: return nodos[-1][1]
        if t_arr is None: t_arr = [n[0] for n in nodos]
        
        idx = np.searchsorted(t_arr, t)
        t_A, k_A = nodos[idx-1][0], nodos[idx-1][1]
        t_B, k_B = nodos[idx][0], nodos[idx][1]
        
        if t_A == t_B: return k_A
        return k_A + (t - t_A) * (k_B - k_A) / (t_B - t_A)
    
    dur = t_fin - t_ini
    if dur <= 0: return km_orig if km_orig is not None else (0.0 if via==1 else 43.13)
    frac = max(0.0, min(1.0, (t - t_ini) / dur))
    
    ko = km_orig if km_orig is not None else (0.0 if via==1 else 43.13)
    kd = km_dest if km_dest is not None else (43.13 if via==1 else 0.0)
    return ko + frac * (kd - ko)

def get_train_state_and_speed(t, r_via, use_rm, km_orig, km_dest, nodos, t_arr=None, prevenciones=None):
    """
    Determina el estado dinámico (ACCEL, BRAKE, CRUISE) y la velocidad de un tren.
    Esencial para el renderizado del mapa SCADA en vivo.
    """
    km_total = getattr(config, 'KM_TOTAL', 43.13)
    if not nodos or len(nodos) < 2: return "CRUISE", 60.0
    if t_arr is None: t_arr = [n[0] for n in nodos]
    
    if t <= t_arr[0] or t >= t_arr[-1]: return "DWELL", 0.0
    
    idx = np.searchsorted(t_arr, t)
    km_now = km_at_t(t_arr[idx-1], t_arr[idx], t, r_via, use_rm, nodos[idx-1][1], nodos[idx][1])
    
    v_cons = max(5.0, vel_at_km(km_now, r_via, use_rm))
    
    # 🛑 RESTRICCIÓN DE SEGURIDAD ATC EN TOPERAS CABECERAS
    if r_via == 1 and km_now >= km_total - 0.200:
        v_cons = min(v_cons, 10.0 if km_now >= km_total - 0.100 else 20.0)
    elif r_via == 2 and km_now <= 0.200:
        v_cons = min(v_cons, 10.0 if km_now <= 0.100 else 20.0)
        
    # Aplicar restricciones de vía dinámicas (TSR)
    if prevenciones:
        for p in prevenciones:
            if p['via'] == r_via and p['km_min'] <= km_now <= p['km_max']:
                v_cons = min(v_cons, p['v_kmh'])
                
    dt_from_A = t - t_arr[idx-1]
    dt_to_B = t_arr[idx] - t
    
    if dt_from_A <= 2.0: return "ACCEL", v_cons
    elif dt_to_B <= 2.0: return "BRAKE", v_cons
    else: return "CRUISE", v_cons

# =============================================================================
# 2. MOTOR TERMODINÁMICO DE SERVICIOS AUXILIARES
# =============================================================================

def calcular_aux_dinamico(aux_kw_nominal, hora_decimal, pax_abordo, cap_max, estacion_anio, estado_marcha="CRUISE", f_compresor_dwell=1.03, p_compresor_kw=3.68, compresor_on=False):
    """
    Cálculo Bottom-Up de servicios auxiliares. 
    Diferencia Carga Base, HVAC y el ciclo de histéresis del Compresor Principal.
    """
    hora_int = int(hora_decimal) % 24
    
    try:
        perfil = getattr(config, '_AUX_HVAC_HORA', {}).get(estacion_anio, [0.5]*24)
    except:
        perfil = [0.5]*24
        
    f_hvac = perfil[hora_int]
    
    # Factor de ocupación (impacto térmico de pasajeros)
    if cap_max > 0:
        ocup = min(1.0, pax_abordo / cap_max)
        if estacion_anio == "verano": f_ocup = 1.0 + 0.05 * ocup
        elif estacion_anio == "invierno": f_ocup = 1.0 - 0.12 * ocup
        else: f_ocup = 1.0 - 0.06 * ocup
    else: f_ocup = 1.0
        
    # 💡 LÓGICA DE SUMATORIA ESTRICTA (Sin doble contabilización)
    # 1. Carga Base (12%): Electrónica, Luces y TCMS
    p_base = aux_kw_nominal * 0.12
    
    # 2. Climatización (45% max): Modulado por hora y pasajeros
    p_hvac = (aux_kw_nominal * 0.45) * f_hvac * f_ocup
    
    # 3. Ventilación de Tracción (Reactiva al estado de marcha)
    p_vent = 0.0
    if estado_marcha in ["ACCEL", "CRUISE"]: p_vent = 4.0 # Ventilación media
    elif estado_marcha in ["BRAKE", "BRAKE_STATION"]: p_vent = 7.6 # Ventilación forzada (Enfriamiento Reostático)
    
    # 4. Compresor Neumático (Independiente)
    p_comp = p_compresor_kw if compresor_on else 0.0
    
    return p_base + p_hvac + p_vent + p_comp

# =============================================================================
# 3. MOTOR FÍSICO CENTRAL (INTEGRADOR DE EULER + RADAR TSR)
# =============================================================================

def simular_tramo_termodinamico(tipo_tren, doble, km_ini, km_fin, via_op, pct_trac, use_rm, use_pend, nodos=None, pax_dict=None, pax_abordo=0, v_consigna_override=None, maniobra=None, estacion_anio="primavera", t_ini_mins=0.0, es_vacio=False, prevenciones=None):
    """
    Simulación cinemática segundo a segundo con resolución de fuerzas Davis y Gravedad.
    Incluye acumulador de presión MRP y Radar Predictivo de Prevenciones.
    """
    # 1. Extracción de ADN de Flota
    try:
        flota_master = getattr(config, 'FLOTA', {})
        f = flota_master.get(tipo_tren, flota_master.get("XT-100", {}))
    except:
        f = {"tara_t": 86.1, "m_iner_t": 7.2, "p_max_kw": 720, "f_trac_max_kn": 110, "a_freno_ms2": 1.2, "v_freno_min": 3.81}

    # 2. Configuración Estacional
    if estacion_anio == "invierno": aux_nom = f.get('aux_kw_heat', 65.16)
    else: aux_nom = f.get('aux_kw_cool', 58.76)
    
    p_comp_nom = f.get('p_comp_kw', 3.68)
    
    # 3. Variables de Estado Inicial
    trc, aux, reg, t_horas = 0.0, 0.0, 0.0, 0.0
    mrp_bar = 10.0       # Presión inicial estanque principal
    compresor_on = False
    
    # 4. Definición de Paradas
    paradas_km = sorted(list(set(([n[1] for n in nodos] if nodos else []) + [km_ini, km_fin])), reverse=(via_op == 2))
    
    dt = 1.0  # Paso de tiempo: 1 segundo
    km_total_red = getattr(config, 'KM_TOTAL', 43.13)
    pax_kg = getattr(config, 'PAX_KG', 75.0)
    
    # 5. Iteración por Tramos (Estación a Estación)
    for i in range(len(paradas_km)-1):
        p_ini, p_fin = paradas_km[i], paradas_km[i+1]
        dist_total_tramo = abs(p_fin - p_ini) * 1000.0
        if dist_total_tramo <= 0.1: continue
        
        # Actualización de Peso por Estación (Masa Dinámica)
        pax_en_tramo = get_pax_at_km_nativo(pax_dict, p_ini, via_op, pax_abordo)
        n_unidades = 2 if doble else 1
        masa_total_kg = ((f['tara_t'] + f['m_iner_t']) * 1000 * n_unidades) + (pax_en_tramo * pax_kg)
        
        pos_m, dist_recorrida, v_ms, a_prev, estado_marcha = p_ini * 1000.0, 0.0, 0.0, 0.0, "ACCEL"
        
        # 6. Bucle Cinemático (metro a metro)
        while dist_recorrida < dist_total_tramo:
            dist_restante = dist_total_tramo - dist_recorrida
            if dist_restante < 0.1: break
            
            km_actual = (pos_m + dist_recorrida) / 1000.0 if via_op == 1 else (pos_m - dist_recorrida) / 1000.0
            
            # Límite de Velocidad de la Vía
            v_cons_kmh = max(5.0, vel_at_km(km_actual, via_op, use_rm))
            if v_consigna_override: v_cons_kmh = min(v_cons_kmh, v_consigna_override)
            
            # 🛑 RESTRICCIÓN DE SEGURIDAD ATC (TOPERAS)
            if via_op == 1 and km_actual >= km_total_red - 0.200:
                v_cons_kmh = min(v_cons_kmh, 10.0 if km_actual >= km_total_red - 0.100 else 20.0)
            elif via_op == 2 and km_actual <= 0.200:
                v_cons_kmh = min(v_cons_kmh, 10.0 if km_actual <= 0.100 else 20.0)
            
            # 🚧 RADAR DE PREVENCIONES (Lookahead 1.500m)
            if prevenciones:
                for p in prevenciones:
                    if p['via'] == via_op:
                        # Si ya estoy dentro de la zona
                        if p['km_min'] <= km_actual <= p['km_max']:
                            v_cons_kmh = min(v_cons_kmh, p['v_kmh'])
                        # Si la zona está adelante (Radar)
                        else:
                            dist_a_zona = (p['km_min'] - km_actual)*1000 if via_op==1 else (km_actual - p['km_max'])*1000
                            if 0 < dist_a_zona <= 1500:
                                v_objetivo_ms = p['v_kmh'] / 3.6
                                if v_ms > v_objetivo_ms:
                                    a_necesaria = (v_ms**2 - v_objetivo_ms**2) / (2 * dist_a_zona)
                                    if a_necesaria > 0.4: v_cons_kmh = min(v_cons_kmh, p['v_kmh'])

            # 7. Cálculo de Resistencias (Davis + Gravedad)
            v_kmh = v_ms * 3.6
            if n_unidades == 2:
                f_davis = (f['davis_A'] * 2) + (f['davis_B'] * 2 * v_kmh) + (f['davis_C'] * 1.35 * (v_kmh**2))
            else:
                f_davis = f['davis_A'] + f['davis_B'] * v_kmh + f['davis_C'] * (v_kmh**2)
                
            f_pend = 0.0
            if use_pend:
                try:
                    e_km, e_m = getattr(config, '_ELEV_KM', []), getattr(config, '_ELEV_M', [])
                    idx_p = np.searchsorted(e_km, km_actual) - 1
                    if 0 <= idx_p < len(e_km) - 1:
                        pendiente = ((e_m[idx_p+1] - e_m[idx_p]) / max(0.001, (e_km[idx_p+1] - e_km[idx_p])*1000)) * 1000
                        f_pend = 9.81 * pendiente * (masa_total_kg / 1000.0) * (1.0 if via_op==1 else -1.0)
                except: pass

            # 8. Lógica de Conducción (Inversor)
            d_freno_req = (v_ms**2) / (2 * (f['a_freno_ms2']*0.9))
            if dist_restante <= d_freno_req + 1.2: estado_marcha = "BRAKE_STATION"
            elif v_kmh > v_cons_kmh + 1.5: estado_marcha = "BRAKE_OVERSPEED"
            elif estado_marcha == "ACCEL" and v_kmh >= v_cons_kmh - 0.5: estado_marcha = "COAST"
            elif estado_marcha == "COAST" and v_kmh < v_cons_kmh - 2.0: estado_marcha = "ACCEL"

            f_motor, f_regen_inst, a_target = 0.0, 0.0, 0.0
            
            if estado_marcha == "BRAKE_STATION":
                f_freno_req = max(0.0, masa_total_kg * (f['a_freno_ms2']*0.9) - f_davis - f_pend)
                f_regen_inst = min(f_freno_req, min(f['f_freno_max_kn']*1000*n_unidades, 800000.0/max(0.1, v_ms)))
                a_target = max(-(f['a_freno_ms2']*0.9), (-f_regen_inst - f_davis - f_pend) / masa_total_kg)
            elif estado_marcha == "ACCEL":
                f_motor = min(f['f_trac_max_kn']*1000*n_unidades, f['p_max_kw']*1000*n_unidades/max(0.1, v_ms))
                a_target = (f_motor - f_davis - f_pend) / masa_total_kg
            elif estado_marcha == "COAST":
                a_target = (-f_davis - f_pend) / masa_total_kg
                
            # Aplicar Jerk (Suavizado de aceleración)
            jerk = f.get('jerk_ms3', 0.8)
            a_net = np.clip(a_target, a_prev - jerk, a_prev + jerk); a_prev = a_net
            
            v_new = max(0.0, v_ms + a_net * dt)
            if f_motor > 0 and v_new * 3.6 > v_cons_kmh: v_new = v_cons_kmh / 3.6
            
            # Integrador de Posición
            step_m = (v_ms + v_new) / 2.0 * dt
            if step_m > dist_restante: step_m = dist_restante
            if step_m < 0.1: step_m = 0.5 # Anti-Stall
            
            # ⚡ Integración de Energía
            if f_motor > 0: trc += ((f_motor * step_m) / 3600000.0) / f.get('eta_motor', 0.92)
            if f_regen_inst > 0 and v_kmh >= f['v_freno_min']: reg += ((f_regen_inst * step_m) / 3600000.0) * 0.72
            
            # 💨 Histéresis del Compresor
            if mrp_bar <= 8.0: compresor_on = True
            if mrp_bar >= 10.0: compresor_on = False
            if compresor_on: mrp_bar += 0.012 # Tasa de carga suave (3.68 kW)
            
            aux += (calcular_aux_dinamico(aux_nom * n_unidades, (t_ini_mins + t_horas*60)/60, pax_en_tramo, f['cap_max']*n_unidades, estacion_anio, estado_marcha, 1.03, p_comp_nom, compresor_on) / 3600.0)
            
            t_horas += dt / 3600.0; dist_recorrida += step_m; v_ms = v_new

        # 💡 DETENCIÓN EN ANDÉN: Gasto Neumático de Puertas y Frenos
        if i < len(paradas_km) - 2:
            mrp_bar -= 0.35 # Gasto por aplicación de freno y apertura de puertas
            dwell_h = 25.0 / 3600.0
            aux += calcular_aux_dinamico(aux_nom * n_unidades, (t_ini_mins + t_horas*60)/60, pax_en_tramo, f['cap_max']*n_unidades, estacion_anio, "DWELL", 1.03, p_comp_nom, compresor_on) * dwell_h
            t_horas += dwell_h

    return trc, aux, reg, 0.0, max(0.0, trc + aux - reg), t_horas

# =============================================================================
# 4. ORQUESTADOR DE FLOTA
# =============================================================================

def calcular_receptividad_por_headway(df_dia: pd.DataFrame) -> dict:
    if df_dia.empty: return {}
    result = {}
    for via in [1, 2]:
        sub = df_dia[df_dia["Via"] == via].sort_values("t_ini")
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
    # Fallback rápido para no colapsar la app en modo Squeeze Control Pasivo
    return {idx: 0.70 for idx in df_dia.index}

def calcular_termodinamica_flota_v111(df_dia, pct_trac, use_pend, use_rm, use_regen, dict_regen, estacion_anio="primavera", prevenciones=None):
    df_e = df_dia.copy()
    if df_e.empty: return df_e
    
    def _wrapper(r):
        trc, ax, rm, _, nt, th = simular_tramo_termodinamico(r['tipo_tren'], r.get('doble', False), r['km_orig'], r['km_dest'], r['Via'], pct_trac, use_rm, use_pend, r.get('nodos'), r.get('pax_d', {}), r.get('pax_abordo', 0), None, r.get('maniobra'), estacion_anio, r.get('t_ini', 0.0), False, prevenciones)
        ru = rm * dict_regen.get(r.name, 1.0) if use_regen else 0.0
        return pd.Series([trc, ax, ru, max(0.0, rm - ru), max(0.0, trc + ax - ru)])
        
    df_e[['kwh_viaje_trac', 'kwh_viaje_aux', 'kwh_viaje_regen', 'kwh_reostato', 'kwh_viaje_neto']] = df_e.apply(_wrapper, axis=1)
    return df_e
