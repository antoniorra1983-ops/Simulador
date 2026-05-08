import pandas as pd
import numpy as np
import re
import unicodedata
from io import BytesIO
from datetime import datetime, date, timedelta
from config import *

# =============================================================================
# 1. UTILIDADES DE TIEMPO Y FORMATO (CON FRENO ANTI-BUCLE)
# =============================================================================
def mins_to_time_str(mins):
    """Convierte minutos decimales a formato HH:MM:SS con protección de ciclo."""
    if pd.isna(mins) or np.isinf(mins): return '--:--:--'
    try:
        m_val = float(mins) % 1440.0 # Asegura que el tiempo se mantenga en 24h
        h = int(m_val // 60)
        m = int(m_val % 60)
        s = int(round((m_val * 60) % 60))
        if s == 60: s, m = 0, m + 1
        if m == 60: m, h = 0, h + 1
        return f"{h:02d}:{m:02d}:{s:02d}"
    except: return '--:--:--'

def parse_time_to_mins(val):
    """Parsea formatos HH:MM, HH:MM:SS o decimales de Excel a minutos."""
    if pd.isna(val): return None
    sv = str(val).strip().lower()
    if sv in ('', 'nan'): return None
    if ' ' in sv: sv = sv.split(' ')[-1]
    
    # Formato estándar HH:MM(:SS)
    m = re.search(r'(\d{1,2}):(\d{2})(?::(\d{2}))?', sv)
    if m:
        h = int(m.group(1))
        m_min = int(m.group(2))
        s_sec = int(m.group(3)) / 60.0 if m.group(3) else 0.0
        return h * 60.0 + m_min + s_sec
        
    try:
        f = float(sv)
        if f < 1.0: return f * 1440.0 # Decimal de Excel
        if f < 2400.0: return (int(f // 100) * 60.0) + (f % 100) # Formato 1430 -> 14:30
    except: pass
    return None

def parse_excel_date(val):
    """Lector de fechas flexible para diversos formatos de Excel."""
    if pd.isna(val): return None
    if isinstance(val, (datetime, pd.Timestamp)): return val.strftime('%Y-%m-%d')
    v_str = re.sub(r'\.0+$', '', str(val).strip()).split(' ')[0]
    if not v_str or v_str.lower() in ['nan', 'none', 'fecha', 'date', 'nat']: return None
    
    if v_str.isdigit():
        v_int = int(v_str)
        if 40000 <= v_int <= 60000:
            try: return (date(1899, 12, 30) + timedelta(days=v_int)).strftime('%Y-%m-%d')
            except: pass
            
    for pat in [r'\b(\d{1,2})[-/](\d{1,2})[-/](\d{4})\b', r'\b(\d{4})[-/](\d{1,2})[-/](\d{1,2})\b']:
        m_dt = re.search(pat, v_str)
        if m_dt:
            y, m_val, d = (int(m_dt.group(1)), int(m_dt.group(2)), int(m_dt.group(3))) if len(m_dt.group(1)) == 4 else (int(m_dt.group(3)), int(m_dt.group(2)), int(m_dt.group(1)))
            if m_val > 12 and d <= 12: d, m_val = m_val, d
            if 1 <= d <= 31 and 1 <= m_val <= 12: return f"{y:04d}-{m_val:02d}-{d:02d}"
    return None

# =============================================================================
# 2. LIMPIEZA DE IDENTIFICADORES Y GEOMETRÍA
# =============================================================================
def clean_primary_key(x):
    if pd.isna(x): return ''
    s = re.sub(r'[^A-Z0-9]', '', re.sub(r'\.0+$', '', str(x).strip().upper()))
    return s.lstrip('0') if s not in ['NAN', ''] else ''

def clean_id(x):
    try:
        nums = re.findall(r'\d+', str(x).strip().lower().replace(".0", ""))
        return str(int(nums[0])) if nums else str(x).strip().upper()
    except: return str(x).strip().upper()

def clasificar_dia(d_str):
    try:
        d = datetime.strptime(d_str, '%Y-%m-%d')
        if d_str in feriados_2026 or d.weekday() == 6: return 'Domingo/Festivo'
        return 'Sábado' if d.weekday() == 5 else 'Laboral'
    except: return 'Laboral'

def extraer_fecha_segura(df_raw, fname):
    """Algoritmo de detección de fecha operativa en nombre de archivo o celdas."""
    # Buscar en el nombre del archivo (ej. THDR 150424)
    for pat in [r'\b(\d{1,2})[-_\.](\d{1,2})[-_\.](\d{4})\b', r'\b(\d{4})[-_\.](\d{1,2})[-_\.](\d{1,2})\b']:
        m = re.search(pat, str(fname))
        if m:
            y, mon, d = (int(m.group(1)), int(m.group(2)), int(m.group(3))) if len(m.group(1)) == 4 else (int(m.group(3)), int(m.group(2)), int(m.group(1)))
            if mon > 12 and d <= 12: d, mon = mon, d
            if 1 <= d <= 31 and 1 <= mon <= 12: return f"{y:04d}-{mon:02d}-{d:02d}"
    
    # Buscar patrones sin separadores (ej. 150424)
    s_digits = re.sub(r'\D', '', str(fname))
    if len(s_digits) >= 6:
        for i in range(len(s_digits)-5):
            try:
                d, mon, y = int(s_digits[i:i+2]), int(s_digits[i+2:i+4]), int(s_digits[i+4:i+6])
                if 1 <= d <= 31 and 1 <= mon <= 12 and 20 <= y <= 35: return f"20{y:02d}-{mon:02d}-{d:02d}"
            except: pass

    # Buscar dentro de las celdas del reporte
    for i in range(min(50, len(df_raw))):
        row_str = ' '.join([str(x) for x in df_raw.iloc[i].values if pd.notna(x)])
        m_dt = re.search(r'\b(\d{4})[-/\.](\d{1,2})[-/\.](\d{1,2})\b', row_str)
        if m_dt: return f"{int(m_dt.group(1)):04d}-{int(m_dt.group(2)):02d}-{int(m_dt.group(3)):02d}"
        
    return "2026-01-01"

def _col_to_est_idx(col):
    """Mapea nombres de columnas de estaciones a sus índices KM_ACUM."""
    cu = re.sub(r'[^a-z0-9]','', str(col).lower().replace('á','a').replace('é','e').replace('í','i').replace('ó','o').replace('ú','u').replace('ñ','n'))
    if 'americas' in cu: return ESTACIONES.index('Las Americas')
    if 'vina' in cu: return ESTACIONES.index('Viña del Mar')
    if 'aldea' in cu: return ESTACIONES.index('Sargento Aldea')
    if 'belloto' in cu: return ESTACIONES.index('El Belloto')
    
    # Búsqueda por coincidencia parcial en la lista oficial
    for i, est in enumerate(ESTACIONES):
        e_norm = re.sub(r'[^a-z0-9]','', est.lower())
        if e_norm in cu: return i
    return None

def make_unique(df):
    if df.empty: return df
    cols = pd.Series(df.columns)
    for dup in cols[cols.duplicated()].unique(): 
        cols[cols==dup] = [f"{dup}_{i}" if i else dup for i in range(sum(cols==dup))]
    df.columns = cols
    return df

def calc_tren_km_real_general(row):
    """Calcula el kilometraje total considerando tramos de vía doble o simple por maniobra."""
    k_s, k_e = min(row['km_orig'], row['km_dest']), max(row['km_orig'], row['km_dest'])
    man = row.get('maniobra')
    if man in ['CORTE_BTO','ACOPLE_BTO','CORTE_PU_SA_BTO']:
        km_man = KM_ACUM[14]
        if k_s <= km_man <= k_e: return abs(km_man-k_s)*2.0 + abs(k_e-km_man)*1.0
    elif man in ['CORTE_SA','ACOPLE_SA']:
        km_man = KM_ACUM[18]
        if k_s <= km_man <= k_e: return abs(km_man-k_s)*2.0 + abs(k_e-km_man)*1.0
    return abs(k_e-k_s) * (2.0 if row.get('doble',False) else 1.0)

# =============================================================================
# 3. CRUCE DE PASAJEROS (OPTIMIZACIÓN NATIVA)
# =============================================================================
def get_pax_at_km_nativo(pax_d, km_pos, via, pax_max_fallback=0):
    """Función de alto rendimiento para el motor físico. Evita lookups pesados."""
    if not pax_d or not isinstance(pax_d, dict): return pax_max_fallback
    if sum(pax_d.values()) == 0 and pax_max_fallback > 0: return pax_max_fallback
    
    pax_val = 0
    if via == 1:
        for i in range(len(KM_ACUM)):
            if km_pos >= KM_ACUM[i]:
                if i < len(PAX_COLS): pax_val = pax_d.get(PAX_COLS[i], pax_val)
            else: break
    else:
        for i in range(len(KM_ACUM)-1, -1, -1):
            if km_pos <= KM_ACUM[i]:
                if i < len(PAX_COLS): pax_val = pax_d.get(PAX_COLS[i], pax_val)
            else: break
    return int(pax_val)

def match_pax(row, df_pax):
    """Cruza un viaje histórico con la planilla de carga real."""
    EMPTY = ({c: 0 for c in PAX_COLS}, 0, '--:--:--', 'No Detectado', -1)
    if df_pax.empty: return EMPTY
    
    t_i, via = row.get('t_ini'), row.get('Via', 1)
    nro_viaje = clean_primary_key(row.get('nro_viaje', ''))
    
    # Filtrar por Vía y Fecha
    sub = df_pax[(df_pax['Via'] == via)].copy()
    if 'Fecha_s' in df_pax.columns and row.get('Fecha_str') != '2026-01-01':
        sub = sub[sub['Fecha_s'] == row.get('Fecha_str')]
        
    if sub.empty: return EMPTY

    # Match por Nro de Viaje (Fuerte)
    if nro_viaje:
        match_exacto = sub[sub['Nro_THDR'].apply(clean_primary_key) == nro_viaje]
        if not match_exacto.empty:
            best = match_exacto.iloc[0]
            return {c: int(best.get(c, 0)) for c in PAX_COLS}, int(best.get('CargaMax', 0)), mins_to_time_str(best.get('t_ini_p')), str(best.get('Nro_THDR', '')), best.name

    # Match por Proximidad de Hora (Débil)
    sub['diff'] = sub['t_ini_p'].apply(lambda x: min(abs(float(x)-float(t_i)), 1440-abs(float(x)-float(t_i))))
    best_match = sub.loc[sub['diff'].idxmin()]
    
    if best_match['diff'] <= 15:
        return {c: int(best_match.get(c, 0)) for c in PAX_COLS}, int(best_match.get('CargaMax', 0)), mins_to_time_str(best_match.get('t_ini_p')), str(best_match.get('Nro_THDR', '')), best_match.name
    
    return EMPTY

# =============================================================================
# 4. EXTRACCIÓN DE ARCHIVOS (THDR, PASAJEROS, PREVENCIONES)
# =============================================================================
def procesar_thdr(data, fname, via_param=1):
    """Lector universal de THDR EFE (CSV/Excel)."""
    try:
        raw = pd.read_csv(BytesIO(data), header=None, dtype=str) if fname.lower().endswith('.csv') else pd.read_excel(BytesIO(data), header=None, dtype=str)
        if raw is None or raw.empty or raw.shape[0] < 6: return pd.DataFrame(), "Archivo inválido."
        
        fecha_str = extraer_fecha_segura(raw, fname)
        
        # Detectar cabecera dinámica
        header_idx = 1
        for i in range(min(15, len(raw))):
            row_str = ' '.join([str(x).upper() for x in raw.iloc[i].values if pd.notna(x)])
            if 'VIAJE' in row_str or 'SALIDA' in row_str:
                header_idx = i; break
                
        r0 = raw.iloc[header_idx - 1].copy() if header_idx > 0 else raw.iloc[0].copy()
        r0.iloc[0] = np.nan 
        cols = [f"{str(s).strip()}_{str(t).strip()}" if str(s).strip() and str(s).strip().lower() != 'nan' and str(t).strip() else str(t).strip() or str(s).strip() for s, t in zip(r0.ffill().astype(str), raw.iloc[header_idx].fillna('').astype(str))]
        
        df = raw.iloc[header_idx + 1:].copy().reset_index(drop=True)
        df.columns = make_unique(pd.DataFrame(columns=[c if c else f"Col_{j}" for j, c in enumerate(cols)])).columns
        df = df.dropna(how='all').reset_index(drop=True)

        for col in df.columns:
            if any(k in str(col).upper() for k in ['LLEGADA','SALIDA','HORA']):
                df[f"{col}_min"] = df[col].apply(parse_time_to_mins)

        est_cols = {c: _col_to_est_idx(c) for c in df.columns if '_min' in str(c).lower() and 'program' not in str(c).lower()}
        df['t_ini'] = df.apply(lambda row: min([row.get(c, np.nan) for c in est_cols.keys() if pd.notna(row.get(c, np.nan))] or [np.nan]), axis=1)
        df['t_fin'] = df.apply(lambda row: max([row.get(c, np.nan) for c in est_cols.keys() if pd.notna(row.get(c, np.nan))] or [np.nan]), axis=1)

        c_m1 = next((c for c in df.columns if 'motriz' in str(c).lower() and '1' in str(c).lower()), None)
        c_m2 = next((c for c in df.columns if 'motriz' in str(c).lower() and '2' in str(c).lower()), None)
        serv_col = next((c for c in df.columns if str(c).strip().upper() in ('TREN', 'SERVICIO')), None)

        def _get_fleet_info(r):
            n1 = int(re.search(r'\d+', str(r.get(c_m1, ''))).group(0)) if c_m1 and re.search(r'\d+', str(r.get(c_m1, ''))) else None
            n_eval = n1 or (int(re.search(r'\d+', str(r.get(serv_col, ''))).group(0)) if serv_col and re.search(r'\d+', str(r.get(serv_col, ''))) else 1)
            tipo = "SFE" if n_eval >= 36 else ("XT-M" if 28 <= n_eval <= 35 else "XT-100")
            return pd.Series([str(n_eval), tipo])
            
        df[['motriz_num', 'tipo_tren']] = df.apply(_get_fleet_info, axis=1)
        df['doble'] = (df['Unidad'].fillna('S') == 'M') if 'Unidad' in df.columns else (pd.notna(df.get(c_m2)) if c_m2 else False)
        df['Via'], df['Fecha_str'] = via_param, fecha_str

        def _get_real_orig_dest(row):
            valid = [e_idx for col, e_idx in est_cols.items() if pd.notna(row.get(col)) and row.get(col) > 0]
            if not valid: return pd.Series([0.0 if via_param == 1 else KM_TOTAL, KM_TOTAL if via_param == 1 else 0.0])
            return pd.Series([KM_ACUM[min(valid)], KM_ACUM[max(valid)]]) if via_param == 1 else pd.Series([KM_ACUM[max(valid)], KM_ACUM[min(valid)]])

        df[['km_orig', 'km_dest']] = df.apply(_get_real_orig_dest, axis=1)
        df = df.dropna(subset=['t_ini'])
        df['svc_type'] = df.apply(lambda r: f"{EC[KM_ACUM.index(r['km_orig'])]}-{EC[KM_ACUM.index(r['km_dest'])]}", axis=1)
        
        def _extract_nodos(row):
            nodos_temp = [(row.get(col), KM_ACUM[e_idx]) for col, e_idx in est_cols.items() if pd.notna(row.get(col)) and row.get(col) > 0]
            seen_km = set()
            return sorted([n for n in nodos_temp if not (n[1] in seen_km or seen_km.add(n[1]))], key=lambda x: x[0])
            
        df['nodos'] = df.apply(_extract_nodos, axis=1)
        df['num_servicio'] = df[serv_col].apply(clean_primary_key) if serv_col else ''
        df['_id'] = df['Fecha_str'] + "_" + df['num_servicio'] + "_" + df['t_ini'].astype(str)
        df['t_fin'] = df['t_fin'].fillna(df['t_ini'] + abs(df['km_dest']-df['km_orig']) / 35.0 * 60.0)
        return df, "ok"
    except Exception as e: return pd.DataFrame(), str(e)

def cargar_pax(data, fname, via_param=1):
    """Lector inteligente de planillas de pasajeros."""
    try:
        full = pd.read_csv(BytesIO(data), dtype=str) if fname.lower().endswith('.csv') else pd.read_excel(BytesIO(data), dtype=str)
        header_idx = 9 # Estándar EFE
        col_mapping = {}
        for c_idx in range(full.shape[1]):
            val_stack = " ".join([str(full.iloc[r, c_idx]).upper() for r in range(max(0, header_idx-4), header_idx+1)])
            if 'HORA' in val_stack and 'ORIG' in val_stack: col_mapping[c_idx] = 'Hora Origen'
            elif 'THDR' in val_stack and 'TREN' not in val_stack: col_mapping[c_idx] = 'Nro_THDR_raw'
            elif 'TOTAL' in val_stack or 'BORDO' in val_stack: col_mapping[c_idx] = 'CargaMax'
            else:
                for k in PAX_COLS:
                    if k in val_stack: col_mapping[c_idx] = k; break

        df = pd.DataFrame({col_name: full.iloc[header_idx + 1:, c_idx].values for c_idx, col_name in col_mapping.items()})
        df['Fecha_s'] = extraer_fecha_segura(full, fname)
        df['Nro_THDR'] = df['Nro_THDR_raw'].apply(clean_primary_key)
        df['t_ini_p'] = df['Hora Origen'].apply(parse_time_to_mins)
        df['Via'] = via_param
        df = df.dropna(subset=['t_ini_p'])
        for c in PAX_COLS + ['CargaMax']: 
            df[c] = df[c].apply(lambda x: int(re.sub(r'[^\d]', '', str(x).replace('.', '')) or 0))
        return df
    except: return pd.DataFrame()

def cargar_prevenciones(data, fname):
    """Lector de restricciones temporales de vía (TSR)."""
    try:
        df = pd.read_csv(BytesIO(data)) if fname.lower().endswith('.csv') else pd.read_excel(BytesIO(data))
        prevs = []
        for _, r in df.iterrows():
            try:
                v1, v2 = float(str(r.iloc[0]).replace(',','.')), float(str(r.iloc[1]).replace(',','.'))
                v_kmh = float(re.search(r'\d+', str(r.iloc[2])).group(0))
                via = int(r.iloc[3]) if len(r) > 3 else 1
                prevs.append({'km_min': min(v1, v2), 'km_max': max(v1, v2), 'v_kmh': v_kmh, 'via': via})
            except: pass
        return prevs
    except: return []

# Stubs para evitar errores de importación en app.py
def get_vacios_dia(df_dia): return []
def parsear_planilla_maestra(data, fname): return pd.DataFrame(), "No implementado"
def get_perfiles_pax(df_px): return {}
def calcular_dwell(df1, df2): return df1, df2
