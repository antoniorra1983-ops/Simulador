"""
Optimizador de distribución de flota para minimizar consumo energético.

Estrategia: reasignar el tipo de tren (XT-100 / XT-M) a cada servicio de la malla,
respetando la flota disponible y la capacidad de pasajeros requerida, de modo que
los trenes más eficientes (XT-M, menor IDE) cubran los servicios de mayor km/demanda.

No altera los horarios (t_ini, t_fin) ni la malla — solo qué unidad hace cada servicio.
"""
import pandas as pd
import numpy as np


def _ide_referencia(config):
    """IDE aproximado por tipo de tren (kWh/km) calibrado del simulador."""
    return {
        'XT-100': 3.88,
        'XT-M':   3.28,
        'SFE':    5.77,
    }


def _flota_disponible(config):
    """Unidades disponibles por tipo de tren."""
    try:
        flota = getattr(config, 'FLOTA', {})
        disp = {}
        for tipo, params in flota.items():
            disp[tipo] = int(params.get('unidades_disponibles', 0))
        # Fallback a valores conocidos MERVAL si no están en config
        if not any(disp.values()):
            disp = {'XT-100': 27, 'XT-M': 8, 'SFE': 5}
        return disp
    except Exception:
        return {'XT-100': 27, 'XT-M': 8, 'SFE': 5}


def optimizar_asignacion_flota(df_servicios, config, priorizar='energia'):
    """
    Reasigna tipos de tren a los servicios para minimizar el consumo.

    df_servicios: DataFrame con columnas km_orig, km_dest, doble, tipo_tren,
                  pax_abordo (o demanda), motriz_num, t_ini, Via, svc_type
    priorizar: 'energia' (minimizar kWh) o 'eficiencia' (minimizar IDE promedio)

    Retorna: (df_optimizado, resumen_dict)
    """
    df = df_servicios.copy().reset_index(drop=True)
    ide_ref = _ide_referencia(config)
    flota_disp = _flota_disponible(config)

    # Distancia de cada servicio
    df['km_tramo'] = (df['km_dest'] - df['km_orig']).abs()

    # Capacidad requerida por servicio (si viene de pasajeros)
    try:
        flota = getattr(config, 'FLOTA', {})
    except Exception:
        flota = {}

    def cap_requerida(row):
        # demanda de pasajeros; si no hay dato, usar la capacidad del tren actual
        pax = row.get('pax_abordo', 0) or 0
        return pax

    df['cap_req'] = df.apply(cap_requerida, axis=1)

    # --- Asignación de tipos por simultaneidad ---
    # Restricción real: en cada instante no puede haber más trenes de un tipo
    # circulando que unidades disponibles de ese tipo. Calculamos solapamiento.

    # Orden de preferencia energética: el tipo con menor IDE primero
    tipos_ordenados = sorted(ide_ref.keys(), key=lambda t: ide_ref[t])

    # Estrategia: asignar XT-M (más eficiente) a los servicios de MAYOR km
    # mientras haya unidades disponibles sin exceder simultaneidad.
    df = df.sort_values('km_tramo', ascending=False).reset_index(drop=True)

    df['tipo_optimo'] = df['tipo_tren']  # default: mantener

    # Contar simultaneidad: para cada servicio, cuántos otros se solapan en tiempo
    def trenes_simultaneos(idx, tipo_asignado_col):
        t_ini = df.at[idx, 't_ini']
        t_fin = df.at[idx, 't_fin'] if 't_fin' in df.columns else t_ini + 60
        count = {}
        for j in range(len(df)):
            if j == idx:
                continue
            tj_ini = df.at[j, 't_ini']
            tj_fin = df.at[j, 't_fin'] if 't_fin' in df.columns else tj_ini + 60
            if tj_ini < t_fin and tj_fin > t_ini:  # solapan
                tp = df.at[j, tipo_asignado_col]
                count[tp] = count.get(tp, 0) + 1
        return count

    # Asignación greedy: recorrer servicios de mayor a menor km,
    # asignar el tipo más eficiente que (a) tenga capacidad suficiente,
    # (b) no exceda la flota disponible en simultaneidad
    cap_tipo = {t: flota.get(t, {}).get('cap_max', 398) for t in ide_ref}

    for idx in range(len(df)):
        cap_req = df.at[idx, 'cap_req']
        es_doble = bool(df.at[idx, 'doble'])

        mejor_tipo = df.at[idx, 'tipo_tren']
        for tipo in tipos_ordenados:
            cap_disp = cap_tipo.get(tipo, 398) * (2 if es_doble else 1)
            if cap_disp < cap_req:
                continue  # no alcanza la capacidad
            # verificar simultaneidad
            simult = trenes_simultaneos(idx, 'tipo_optimo')
            usados_tipo = simult.get(tipo, 0)
            if usados_tipo < flota_disp.get(tipo, 0):
                mejor_tipo = tipo
                break
        df.at[idx, 'tipo_optimo'] = mejor_tipo

    # Restaurar orden original
    df = df.sort_values('t_ini').reset_index(drop=True)

    # --- Calcular consumo antes y después (estimación por IDE) ---
    df['kwh_actual'] = df.apply(
        lambda r: ide_ref.get(r['tipo_tren'], 3.88) * r['km_tramo'] * (2 if r['doble'] else 1), axis=1)
    df['kwh_optimo'] = df.apply(
        lambda r: ide_ref.get(r['tipo_optimo'], 3.88) * r['km_tramo'] * (2 if r['doble'] else 1), axis=1)

    kwh_actual_total = df['kwh_actual'].sum()
    kwh_optimo_total = df['kwh_optimo'].sum()
    ahorro = kwh_actual_total - kwh_optimo_total
    ahorro_pct = (ahorro / kwh_actual_total * 100) if kwh_actual_total > 0 else 0.0

    # Conteo de cambios
    cambios = df[df['tipo_tren'] != df['tipo_optimo']]

    # Composición antes/después
    comp_antes = df['tipo_tren'].value_counts().to_dict()
    comp_despues = df['tipo_optimo'].value_counts().to_dict()

    resumen = {
        'kwh_actual': kwh_actual_total,
        'kwh_optimo': kwh_optimo_total,
        'ahorro_kwh': ahorro,
        'ahorro_pct': ahorro_pct,
        'n_cambios': len(cambios),
        'n_servicios': len(df),
        'comp_antes': comp_antes,
        'comp_despues': comp_despues,
        'flota_disponible': flota_disp,
    }

    return df, resumen
