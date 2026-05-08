import pandas as pd
import numpy as np
import re
import unicodedata
from io import BytesIO
from datetime import datetime, date, timedelta

try:
    import config
except ImportError:
    pass

# =============================================================================
# 1. UTILIDADES DE TIEMPO Y FECHA (ROBUSTAS)
# =============================================================================
def mins_to_time_str(mins):
    if pd.isna(mins) or np.isinf(mins): return '--:--:--'
    try:
        m_val = float(mins) % 1440.0
        h, m = int(m_val // 60), int(m_val % 60)
        s = int(round((m_val * 60) % 60))
        if s == 60: s, m = 0, m + 1
        if m == 60: m, h = 0, h + 1
        return f"{h:02d}:{m:02d}:{s:02d}"
    except: return '--:--:--'

def parse_time_to_mins(val):
    if pd.isna(val): return None
    sv = str(val).strip().lower()
    if sv in ('', 'nan'): return None
    if ' ' in sv: sv = sv.split(' ')[-1]
    m = re.search(r'(\d{1,2}):(\d{2})(?::(\d{2}))?', sv)
    if m:
        h, m_min = int(m.group(1)), int(m.group(2))
        s_sec = int(m.group(3)) / 60.0 if m.group(3) else 0.0
        return h * 60.0 + m_min + s_sec
    try:
        f = float(sv)
        if f < 1.0: return f * 1440.0
        if f < 2400.0: return (int(f // 100) * 60.0) + (f % 100)
    except: pass
    return None

def extraer_fecha_segura(df_raw, fname):
    for pat in [r'\b(\d{1,2})[-_\.](\d{1,2})[-_\.](\d{4})\b', r'\b(\d{4})[-_\.](\d{1,2})[-_\.](\d{1,2})\b']:
        m = re.search(pat, str(fname))
        if m:
            y, mon, d = (int(m.group(1)), int(m.group(2)), int(m.group(3))) if len(m.group(1)) == 4 else (int(m.group(3)), int(m.group(2)), int(m.group(1)))
            if mon > 12 and d <= 12: d, mon = mon, d
            if 1 <= d <= 31 and 1 <= mon <= 12: return f"{y:04d}-{mon:02d}-{d:02d}"
    
    s_fname = str(fname).split('.')[0]
    digit_groups = re.findall(r'\d+', s_fname)
    for group in digit_groups:
        if len(group) == 6:
            try:
                d, mon, y = int(group[0:2]), int(group[2:4]), int(group[4:6])
                if 1 <= d <= 31 and 1 <= mon <= 12 and 20 <= y <= 35: return f"20{y:02d}-{mon:02d}-{d:02d}"
            except: pass
        elif len(group) == 8:
            try:
                d, mon, y = int(group[0:2]), int(group[2:4]), int(group[4:8])
                if 1 <= d <= 31 and 1 <= mon <= 12 and 2000 <= y <= 2035: return f"{y:04d}-{mon:02d}-{d:02d}"
                y, mon, d = int(group[0:4]), int(group[4:6]), int(group[6:8])
                if 1 <= d <= 31 and 1 <= mon <= 12 and 2000 <= y <= 2035: return f"{y:04d}-{mon:02d}-{d:02d}"
            except: pass

    for i in range(min(50, len(df_raw))):
        row_str = ' '.join([str(x) for x in df_raw.iloc[i].values if pd.notna(x)])
        m_dt = re.search(r'\b(\d{4})[-/\.](\d{1,2})[-/\.](\d{1,2})\b', row_str)
        if m_dt: return f"{int(m_dt.group(1)):04d}-{int(m_dt.group(2)):02d}-{int(m_dt.group(3)):02d}"
    return "2026-01-01"

def parse_excel_date(val):
    if pd.isna(val): return None
    if isinstance(val, (datetime, pd.Timestamp)): return val.strftime('%Y-%m-%d')
    v_str = re.sub(r'\.0+$', '', str(val).strip()).split(' ')[0]
    if not v_str or v_str.lower() in ['nan', 'none', 'fecha', 'date', 'nat']: return None
    if v_str.isdigit():
        v_int = int(v_str)
        if 40000 <= v_int <= 60000:
            try: return (date(1899, 12, 30) + timedelta(days=v_int)).strftime('%Y-%m-%d')
            except: pass
        elif len(v_str) in [5, 6]:
            s_pad = v_str.zfill(6)
            try:
                d, m, y = int(s_pad[0:2]), int(s_pad[2:4]), int(s_pad[4:6])
                if 1 <= d <= 31 and 1 <= m <= 12: return f"{2000+y if y<100 else y:04d}-{m:02d}-{d:02d}"
            except: pass
            
    for pat in [r'\b(\d{1,2})[-/](\d{1,2})[-/](\d{4})\b', r'\b(\d{4})[-/](\d{1,2})[-/](\d{1,2})\b']:
        m_dt = re.search(pat, v_str)
        if m_dt:
            if len(m_dt.group(1)) == 4: y, m_val, d = int(m_dt.group(1)), int(m_dt.group(2)), int(m_dt.group(3))
            else: d, m_val, y = int(m_dt.group(1)), int(m_dt.group(2)), int(m_dt.group(3))
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

def clean_pax_number(x):
    if pd.isna(x): return 0
    s = re.sub(r'[^\d]', '', re.sub(r'\.0+$', '', str(x).strip().lower()).replace('.', '').replace(',', ''))
    try: return int(s) if s and s != 'nan' else 0
    except: return 0

def clasificar_dia(d_str):
    feriados = getattr(config, 'feriados_2026', [])
    try:
        d = datetime.strptime(d_str, '%Y-%m-%d')
        if d_str in feriados or d.weekday() == 6: return 'Domingo/Festivo'
        return 'Sábado' if d.weekday() == 5 else 'Laboral'
    except: return 'Laboral'

def make_unique(df):
    cols = pd.Series(df.columns)
    for dup in cols[cols.duplicated()].unique(): 
        cols[cols==dup] = [f"{dup}_{i}" if i else dup for i in range(sum(cols==dup))]
    df.columns = cols
    return df

def _col_to_est_idx(col):
    estaciones = getattr(config, 'ESTACIONES', ['Puerto','Bellavista','Francia','Baron','Portales','Recreo','Miramar','Viña del Mar','Hospital','Chorrillos','El Salto','Valencia','Quilpue','El Sol','El Belloto','Las Americas','La Concepcion','Villa Alemana','Sargento Aldea','Peñablanca','Limache'])
    cu = re.sub(r'[^a-z0-9]','', str(col).lower().replace('á','a').replace('é','e').replace('í','i').replace('ó','o').replace('ú','u').replace('ñ','n'))
    if 'americas' in cu: return estaciones.index('Las Americas')
    if 'vina' in cu: return estaciones.index('Viña del Mar')
    if 'aldea' in cu: return estaciones.index('Sargento Aldea')
    if 'belloto' in cu: return estaciones.index('El Belloto')
    for i, est in enumerate(estaciones):
        e_norm = re.sub(r'[^a-z0-9]','', est.lower())
        if e_norm in cu: return i
    return None

def calc_tren_km_real_general(row):
    km_acum = getattr(config, 'KM_ACUM', [0.0, 43.13])
    if len(km_acum) < 20: km_acum = [0.0]*21 # Fallback
    k_s, k_e = min(row['km_orig'], row['km_dest']), max(row['km_orig'], row['km_dest'])
    man = row.get('maniobra')
    if man in ['CORTE_BTO','ACOPLE_BTO','CORTE_PU_SA_BTO']:
        km_man = km_acum[14]
        if k_s <= km_man <= k_e: return abs(km_man-k_s)*2.0 + abs(k_e-km_man)*1.0
    elif man in ['CORTE_SA','ACOPLE_SA']:
        km_man = km_acum[18]
        if k_s <= km_man <= k_e: return abs(km_man-k_s)*2.0 + abs(k_e-km_man)*1.0
    return abs(k_e-k_s) * (2.0 if row.get('doble',False) else 1.0)

# =============================================================================
# 3. CRUCE INTELIGENTE DE PASAJEROS (MASA DINÁMICA POR ESTACIÓN)
# =============================================================================
def get_pax_at_km_nativo(pax_d, km_pos, via, pax_max_fallback=0):
    km_acum = getattr(config, 'KM_ACUM', [])
    pax_cols = getattr(config, 'PAX_COLS', [])
    if not pax_d or not isinstance(pax_d, dict): return pax_max_fallback
    if sum(pax_d.values()) == 0 and pax_max_fallback > 0: return pax_max_fallback
    
    pax_val = 0
    if via == 1:
        for i in range(len(km_acum)):
            if km_pos >= km_acum[i]:
                if i < len(pax_cols): pax_val = pax_d.get(pax_cols[i], pax_val)
            else: break
    else:
        for i in range(len(km_acum)-1, -1, -1):
            if km_pos <= km_acum[i]:
                if i < len(pax_cols): pax_val = pax_d.get(pax_cols[i], pax_val)
            else: break
    return int(pax_val)

def get_pax_at_km(pax_d, km_pos, via, pax_max_fallback=0):
    """Puente de compatibilidad para evitar ImportErrors gráficos"""
    return get_pax_at_km_nativo(pax_d, km_pos, via, pax_max_fallback)

def match_pax(row, df_pax):
    pax_cols = getattr(config, 'PAX_COLS', ['PUE','BEL','FRA','BAR','POR','REC','MIR','VIN','HOS','CHO','SLT','VAL','QUI','SOL','BTO','AME','CON','VAM','SGA','PEN','LIM'])
    EMPTY = ({c: 0 for c in pax_cols}, 0, '--:--:--', 'No Detectado', -1)
    if df_pax.empty: return EMPTY
    
    t_i = row.get('t_ini')
    via = row.get('Via', 1)
    nro_viaje = clean_primary_key(row.get('num_servicio', row.get('nro_viaje', '')))
    
    sub = df_pax[df_pax['Via'] == via].copy()
    if sub.empty: return EMPTY
    
    fecha_thdr = row.get('Fecha_str')
    if 'Fecha_s' in sub.columns and fecha_thdr and fecha_thdr != '2026-01-01':
        sub_f = sub[sub['Fecha_s'] == fecha_thdr]
        if not sub_f.empty: sub = sub_f

    # 1. Búsqueda Fuerte (Por Número de Tren/Servicio ID)
    if nro_viaje:
        sub['Nro_THDR_cmp'] = sub['Nro_THDR'].apply(clean_primary_key) if 'Nro_THDR' in sub.columns else ''
        match_exacto = sub[(sub['Nro_THDR_cmp'] == nro_viaje) & (sub['Nro_THDR_cmp'] != '')]
        if not match_exacto.empty:
            best = match_exacto.iloc[0]
            return {c: int(best.get(c, 0)) for c in pax_cols}, int(best.get('CargaMax', 0)), mins_to_time_str(best.get('t_ini_p')), str(best.get('Nro_THDR', '')), best.name

    # 2. Búsqueda Débil (Por Sincronización Horaria)
    if pd.notna(t_i) and 't_ini_p' in sub.columns:
        sub['diff'] = sub['t_ini_p'].apply(lambda x: min(abs(float(x)-float(t_i)), 1440-abs(float(x)-float(t_i))) if pd.notna(x) else 9999)
        best_match = sub.loc[sub['diff'].idxmin()]
        if best_match['diff'] <= 15:
            return {c: int(best_match.get(c, 0)) for c in pax_cols}, int(best_match.get('CargaMax', 0)), mins_to_time_str(best_match.get('t_ini_p')), str(best_match.get('Nro_THDR', '')), best_match.name
    
    return EMPTY

# =============================================================================
# 4. EXTRACCIÓN DE ARCHIVOS (PARSERS A PRUEBA DE FALLOS)
# =============================================================================
def procesar_thdr(data, fname, via_param=1):
    km_acum = getattr(config, 'KM_ACUM', [])
    ec = getattr(config, 'EC', [])
    km_total = getattr(config, 'KM_TOTAL', 43.13)
    if not km_acum: return pd.DataFrame(), "Falta KM_ACUM en config.py"

    try:
        if fname.lower().endswith('.csv'):
            try: raw = pd.read_csv(BytesIO(data), header=None, sep=',', encoding='utf-8', dtype=str)
            except: raw = pd.read_csv(BytesIO(data), header=None, sep=';', encoding='latin-1', dtype=str)
        else:
            eng = "openpyxl" if fname.lower().endswith(".xlsx") else "xlrd"
            raw = pd.read_excel(BytesIO(data), header=None, engine=eng, dtype=str)

        if raw is None or raw.empty or raw.shape[0] < 6: return pd.DataFrame(), "Archivo demasiado corto o vacío."
        
        fecha_str = extraer_fecha_segura(raw, fname)
        
        header_idx = 1
        for i in range(min(20, len(raw))):
            row_vals = [str(x).upper() for x in raw.iloc[i].values if pd.notna(x)]
            row_str = ' '.join(row_vals)
            if row_vals.count('LLEGADA') >= 2 or row_vals.count('SALIDA') >= 2 or ('LLEGADA' in row_str and 'SALIDA' in row_str):
                header_idx = i
                break
                
        r0 = raw.iloc[header_idx - 1].copy() if header_idx > 0 else raw.iloc[0].copy()
        r0.iloc[0] = np.nan 
        cols = [f"{str(s).strip()}_{str(t).strip()}" if str(s).strip() and str(s).strip().lower() != 'nan' and str(t).strip() else str(t).strip() or str(s).strip() for s, t in zip(r0.ffill().astype(str), raw.iloc[header_idx].fillna('').astype(str))]
        
        df = raw.iloc[header_idx + 1:].copy().reset_index(drop=True)
        df.columns = [c if c else f"Unnamed_{j}" for j, c in enumerate(cols)]
        df = make_unique(df)
        df = df.dropna(how='all').reset_index(drop=True)

        if df.empty: return pd.DataFrame(), "Sin filas de datos tras la cabecera."

        for col in df.columns:
            if any(k in str(col).upper() for k in ['LLEGADA','SALIDA','HORA']):
                try: df[f"{col}_min"] = df[col].apply(parse_time_to_mins)
                except: pass

        est_cols = {c: _col_to_est_idx(c) for c in df.columns if '_min' in str(c).lower() and 'program' not in str(c).lower() and _col_to_est_idx(c) is not None}
        if not est_cols: return pd.DataFrame(), "No se detectaron columnas de estaciones."

        df['t_ini'] = df.apply(lambda row: min([row.get(c, np.nan) for c in est_cols.keys() if pd.notna(row.get(c))] or [np.nan]), axis=1)
        df['t_fin'] = df.apply(lambda row: max([row.get(c, np.nan) for c in est_cols.keys() if pd.notna(row.get(c))] or [np.nan]), axis=1)

        c_m1 = next((c for c in df.columns if 'motriz' in str(c).lower() and '1' in str(c).lower()), None)
        c_m2 = next((c for c in df.columns if 'motriz' in str(c).lower() and '2' in str(c).lower()), None)
        serv_col = next((c for c in df.columns if str(c).strip().upper() in ('TREN', 'SERVICIO', 'VIAJE', 'NRO')), None)

        def _get_fleet_info(r):
            val_t = str(r.get(c_m1, '')) if c_m1 else str(r.get(serv_col, ''))
            m = re.search(r'\d+', val_t)
            n_eval = int(m.group(0)) if m else 1
            tipo = "SFE" if n_eval >= 36 else ("XT-M" if 28 <= n_eval <= 35 else "XT-100")
            return pd.Series([str(n_eval), tipo])
            
        df[['motriz_num', 'tipo_tren']] = df.apply(_get_fleet_info, axis=1)
        df['doble'] = (df['Unidad'].fillna('S').astype(str).str.upper().str.contains('M')) if 'Unidad' in df.columns else (pd.notna(df.get(c_m2)) if c_m2 else False)
        df['Via'], df['Fecha_str'] = via_param, fecha_str

        def _get_real_orig_dest(row):
            valid = [e_idx for col, e_idx in est_cols.items() if pd.notna(row.get(col)) and row.get(col) > 0]
            if not valid: return pd.Series([0.0 if via_param == 1 else km_total, km_total if via_param == 1 else 0.0])
            return pd.Series([km_acum[min(valid)], km_acum[max(valid)]]) if via_param == 1 else pd.Series([km_acum[max(valid)], km_acum[min(valid)]])

        df[['km_orig', 'km_dest']] = df.apply(_get_real_orig_dest, axis=1)
        df = df.dropna(subset=['t_ini'])
        if df.empty: return pd.DataFrame(), "Los viajes fueron descartados por falta de tiempos."
        
        def _extract_nodos(row):
            nodos_temp = [(row.get(col), km_acum[e_idx]) for col, e_idx in est_cols.items() if pd.notna(row.get(col)) and row.get(col) > 0]
            seen_km = set()
            return sorted([n for n in nodos_temp if not (n[1] in seen_km or seen_km.add(n[1]))], key=lambda x: x[0])
            
        df['nodos'] = df.apply(_extract_nodos, axis=1)
        df['num_servicio'] = df[serv_col].apply(clean_primary_key) if serv_col else ''
        df['_id'] = df['Fecha_str'] + "_" + df['num_servicio'] + "_" + df['t_ini'].astype(str)
        df['t_fin'] = df['t_fin'].fillna(df['t_ini'] + abs(df['km_dest']-df['km_orig']) / 35.0 * 60.0)
        
        if ec: df['svc_type'] = df.apply(lambda r: f"{ec[km_acum.index(r['km_orig'])]}-{ec[km_acum.index(r['km_dest'])]}", axis=1)
        else: df['svc_type'] = "DESC"
        
        return df, "ok"
    except Exception as e: 
        return pd.DataFrame(), str(e)

def cargar_pax(data, fname, via_param=1):
    """
    Extracción Dinámica de Pasajeros. Resuelve el problema de las cabeceras sucias.
    """
    pax_cols = getattr(config, 'PAX_COLS', ['PUE','BEL','FRA','BAR','POR','REC','MIR','VIN','HOS','CHO','SLT','VAL','QUI','SOL','BTO','AME','CON','VAM','SGA','PEN','LIM'])
    try:
        if fname.lower().endswith('.csv'):
            full = pd.read_csv(BytesIO(data), dtype=str, header=None) 
        else:
            eng = "openpyxl" if fname.lower().endswith(".xlsx") else "xlrd"
            full = pd.read_excel(BytesIO(data), dtype=str, engine=eng, header=None)
            
        if full is None or full.empty: return pd.DataFrame()
        
        # 🛡️ Búsqueda Inteligente de Cabecera (Evita el error de "Fila 9" rígida)
        header_idx = -1
        for i in range(min(30, len(full))):
            row_str = " ".join([str(x).upper() for x in full.iloc[i].values if pd.notna(x)])
            if ('ORIG' in row_str or 'HORA' in row_str) and ('TOTAL' in row_str or 'LIM' in row_str or 'PUE' in row_str or 'BORDO' in row_str):
                header_idx = i
                break
                
        if header_idx == -1: header_idx = 9 # Fallback
            
        col_mapping = {}
        for c_idx in range(full.shape[1]):
            val_stack = " ".join([str(full.iloc[r, c_idx]).upper() for r in range(max(0, header_idx-3), header_idx+1)])
            if 'HORA' in val_stack and 'ORIG' in val_stack: col_mapping[c_idx] = 'Hora Origen'
            elif 'THDR' in val_stack and 'TREN' not in val_stack: col_mapping[c_idx] = 'Nro_THDR_raw'
            elif 'TREN' in val_stack or 'SERVICIO' in val_stack: col_mapping[c_idx] = 'Tren'
            elif 'TOTAL' in val_stack or 'BORDO' in val_stack: col_mapping[c_idx] = 'CargaMax'
            else:
                for k in pax_cols:
                    if k in val_stack: 
                        col_mapping[c_idx] = k
                        break
        
        df = pd.DataFrame()
        for c_idx, col_name in col_mapping.items():
            if c_idx < full.shape[1]: 
                df[col_name] = full.iloc[header_idx + 1:, c_idx].values
                
        df['Fecha_s'] = extraer_fecha_segura(full, fname)
        df['Nro_THDR'] = df['Nro_THDR_raw'].apply(clean_primary_key) if 'Nro_THDR_raw' in df.columns else ''
        df['t_ini_p'] = df['Hora Origen'].apply(parse_time_to_mins) if 'Hora Origen' in df.columns else None
        df['Via'] = via_param
        df = df.dropna(subset=['t_ini_p'])
        
        for c in pax_cols + ['CargaMax']: 
            if c in df.columns: 
                df[c] = df[c].apply(lambda x: int(re.sub(r'[^\d]', '', str(x).replace('.', '')) or 0))
        return df
    except Exception as e: return pd.DataFrame()

def cargar_prevenciones(data, fname):
    """
    Extracción Robusta de TSR. Soluciona los errores humanos (comas, letras, orden inverso).
    """
    try:
        if fname.lower().endswith('.csv'):
            df = pd.read_csv(BytesIO(data), header=None, sep=',', encoding='utf-8')
        else:
            eng = "openpyxl" if fname.lower().endswith(".xlsx") else "xlrd"
            df = pd.read_excel(BytesIO(data), header=None, engine=eng)
            
        prevs = []
        for i in range(len(df)):
            row = [str(x) for x in df.iloc[i].values if pd.notna(x)]
            if len(row) >= 3:
                try:
                    # 🛡️ Escudo de Texto: Ignora las filas de título buscando solo dígitos
                    v1 = float(re.search(r'\d+(\.\d+)?', row[0].replace(',', '.')).group())
                    v2 = float(re.search(r'\d+(\.\d+)?', row[1].replace(',', '.')).group())
                    v_kmh = float(re.search(r'\d+', row[2]).group())
                    via = int(re.search(r'\d+', row[3]).group()) if len(row) > 3 else 1
                    
                    # 🛡️ Anti-Inversión: Obliga a que el Km Mínimo siempre sea el inicio real
                    prevs.append({'km_min': min(v1, v2), 'km_max': max(v1, v2), 'v_kmh': v_kmh, 'via': via})
                except: pass
        return prevs
    except Exception as e: return []

# =============================================================================
# 5. FUNCIONES STUBS Y CALCULO DWELL
# =============================================================================
def calcular_dwell(df1, df2): return df1, df2
def get_vacios_dia(df_dia): return []
def get_perfiles_pax(df_px): return {}
def parsear_planilla_maestra(data, fname): return pd.DataFrame(), "ok"
