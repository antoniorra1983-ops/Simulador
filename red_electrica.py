import numpy as np

# Escudo Defensivo Cloud: Evita NameErrors si Streamlit limpia la memoria
try:
    import config
except ImportError:
    pass

def _get_val(name, default):
    try: return getattr(config, name, default)
    except Exception: return default

def calcular_demanda_ser(e_pantografo_kwh, t_horas, km_punto, km_ser):
    if t_horas <= 0: return e_pantografo_kwh
    
    # 💡 Lógica Física Nodal: Resistencias de Catenaria según sección (Ohmios/km)
    if km_punto < 2.25: r_km = 0.0638       
    elif km_punto < 6.80: r_km = 0.0530     
    elif km_punto < 10.92: r_km = 0.0495    
    elif km_punto < 21.41: r_km = 0.0417    
    elif km_punto < 30.36: r_km = 0.0399    
    else: r_km = 0.0355                     
    
    R_total = r_km * abs(km_punto - km_ser)
    P_kW = abs(e_pantografo_kwh) / t_horas
    
    v_nom_dc = _get_val('V_NOMINAL_DC', 3000.0)
    I = (P_kW * 1000.0) / v_nom_dc
    P_loss_kW = (I**2 * R_total) / 1000.0
    
    if e_pantografo_kwh >= 0: 
        return e_pantografo_kwh + (P_loss_kW * t_horas)
    else: 
        return -max(0.0, abs(e_pantografo_kwh) - (P_loss_kW * t_horas))

def distribuir_energia_sers(e_pantografo, t_horas, km_ini, km_fin, active_sers):
    if not active_sers: 
        return {}
    if len(active_sers) == 1:
        e_s = calcular_demanda_ser(e_pantografo, t_horas, (km_ini+km_fin)/2.0, active_sers[0][0])
        return {active_sers[0][1]: e_s}
        
    sers_sorted = sorted(active_sers, key=lambda x: x[0])
    boundaries = [0.0]
    for i in range(len(sers_sorted)-1): 
        boundaries.append((sers_sorted[i][0] + sers_sorted[i+1][0]) / 2.0)
    
    km_total_limit = _get_val('KM_TOTAL', 43.13)
    boundaries.append(km_total_limit)
    
    dist_total = abs(km_fin - km_ini)
    if dist_total < 0.001:
        closest = min(active_sers, key=lambda x: abs(km_ini - x[0]))
        e_s = calcular_demanda_ser(e_pantografo, t_horas, km_ini, closest[0])
        return {closest[1]: e_s}
        
    k_min, k_max = min(km_ini, km_fin), max(km_ini, km_fin)
    resultados = {s[1]: 0.0 for s in sers_sorted}
    
    for i, ser in enumerate(sers_sorted):
        b_min, b_max = boundaries[i], boundaries[i+1]
        o_min, o_max = max(k_min, b_min), min(k_max, b_max)
        
        if o_max > o_min:
            frac = (o_max - o_min) / dist_total
            centroid = (o_min + o_max) / 2.0
            resultados[ser[1]] += calcular_demanda_ser(e_pantografo * frac, t_horas * frac if t_horas > 0 else 0.0, centroid, ser[0])
            
    return resultados

def distribuir_potencia_sers_kw(p_kw, km_punto, active_sers):
    if not active_sers: 
        return {}
    if len(active_sers) == 1: 
        return {active_sers[0][1]: p_kw}
        
    sers_sorted = sorted(active_sers, key=lambda x: x[0])
    if km_punto <= sers_sorted[0][0]: 
        return {sers_sorted[0][1]: p_kw}
    if km_punto >= sers_sorted[-1][0]: 
        return {sers_sorted[-1][1]: p_kw}
        
    for i in range(len(sers_sorted)-1):
        s1, s2 = sers_sorted[i], sers_sorted[i+1]
        if s1[0] <= km_punto <= s2[0]:
            dist_total = s2[0] - s1[0]
            d1, d2 = km_punto - s1[0], s2[0] - km_punto
            return {
                s1[1]: p_kw * (d2 / dist_total), 
                s2[1]: p_kw * (d1 / dist_total)
            }
            
    return {active_sers[0][1]: p_kw}

def calcular_flujo_ac_nodo(demands_kw):
    """ Load Flow AC (Norma IEEE) para la transmisión trifásica a 44kV """
    v_nom_ac = _get_val('V_NOMINAL_AC', 44000.0)
    z_eff = _get_val('Z_EFF_44KV', 0.28)
    r_ac = _get_val('R_AC_44KV', 0.17)
    v_nom_dc = _get_val('V_NOMINAL_DC', 3000.0)
    
    i_po = max(0.0, demands_kw.get('SER PO', 0.0)) * 1000 / (1.732 * v_nom_ac * 0.95)
    i_es = max(0.0, demands_kw.get('SER ES', 0.0)) * 1000 / (1.732 * v_nom_ac * 0.95)
    i_eb = max(0.0, demands_kw.get('SER EB', 0.0)) * 1000 / (1.732 * v_nom_ac * 0.95)
    i_va = max(0.0, demands_kw.get('SER VA', 0.0)) * 1000 / (1.732 * v_nom_ac * 0.95)
    
    len_seat_es, len_es_po = abs(24.3 - 12.7), abs(12.7 - 4.9)
    dv_seat_es = 1.732 * (i_po + i_es) * z_eff * len_seat_es
    dv_es_po = 1.732 * (i_po) * z_eff * len_es_po
    loss_seat_es = 3 * ((i_po + i_es)**2) * r_ac * len_seat_es / 1000.0
    loss_es_po = 3 * (i_po**2) * r_ac * len_es_po / 1000.0
    v_ac_es = v_nom_ac - dv_seat_es
    v_ac_po = v_nom_ac - dv_seat_es - dv_es_po
    
    len_seat_eb, len_eb_va = abs(25.5 - 24.3), abs(28.7 - 25.5)
    dv_seat_eb = 1.732 * (i_eb + i_va) * z_eff * len_seat_eb
    dv_eb_va = 1.732 * (i_va) * z_eff * len_eb_va
    loss_seat_eb = 3 * ((i_eb + i_va)**2) * r_ac * len_seat_eb / 1000.0
    loss_eb_va = 3 * (i_va**2) * r_ac * len_eb_va / 1000.0
    v_ac_eb = v_nom_ac - dv_seat_eb
    v_ac_va = v_nom_ac - dv_seat_eb - dv_eb_va
    
    return {
        'SER PO': {'Vac': v_ac_po, 'Vdc': v_nom_dc * (v_ac_po / v_nom_ac)},
        'SER ES': {'Vac': v_ac_es, 'Vdc': v_nom_dc * (v_ac_es / v_nom_ac)},
        'SER EB': {'Vac': v_ac_eb, 'Vdc': v_nom_dc * (v_ac_eb / v_nom_ac)},
        'SER VA': {'Vac': v_ac_va, 'Vdc': v_nom_dc * (v_ac_va / v_nom_ac)},
        'P_loss_kw': loss_seat_es + loss_es_po + loss_seat_eb + loss_eb_va
    }
