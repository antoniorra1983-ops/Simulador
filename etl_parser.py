import pandas as pd
import numpy as np
import re
import unicodedata
from io import BytesIO
from datetime import datetime, date, timedelta

# =============================================================================
# CONSTANTES BLINDADAS (Protección contra fallos de caché en Streamlit Cloud)
# =============================================================================
ESTACIONES_SAFE = ['Puerto','Bellavista','Francia','Baron','Portales','Recreo','Miramar','Viña del Mar','Hospital','Chorrillos','El Salto','Valencia','Quilpue','El Sol','El Belloto','Las Americas','La Concepcion','Villa Alemana','Sargento Aldea','Peñablanca','Limache']
EC_SAFE = ['PU','BE','FR','BA','PO','RE','MI','VM','HO','CH','ES','VAL','QU','SO','EB','AM','CO','VL','SA','PE','LI']
PAX_COLS_SAFE = ['PUE','BEL','FRA','BAR','POR','REC','MIR','VIN','HOS','CHO','SLT','VAL','QUI','SOL','BTO','AME','CON','VAM','SGA','PEN','LIM']
KM_ACUM_SAFE = [0.0, 0.7, 1.4, 2.2, 3.9, 6.0, 7.4, 8.3, 9.2, 10.2, 11.7, 19.1, 21.4, 23.3, 25.3, 26.4, 27.6, 28.5, 29.1, 30.4, 43.13]
KM_TOTAL_SAFE = 43.13
N_EST_SAFE = 21
FERIADOS_SAFE = ['2026-01-01', '2026-04-03', '2026-04-04', '2026-05-01', '2026-05-21', '2026-06-21', '2026-07-16', '2026-08-15', '2026-09-18', '2026-09-19', '2026-10-12', '2026-10-31', '2026-12-08', '2026-12-25']

# =============================================================================
# 1. UTILIDADES Y PARSEOS BÁSICOS
# =============================================================================

def mins_to_time_str(mins):
    if pd.isna(mins) or np.isinf(mins): return '--:--:--'
    try:
        m_val = float(mins)
        h, m = int((m_val // 60) % 24), int(m_val % 60)
        s = int(round((m_val * 60) % 60))
        if s == 60: s, m = 0, m + 1
        if m == 60: m, h = 0, h + 1
        return f"{h:02d}:{m:02d}:{s:02d}"
    except: return '--:--:--'

def parse_time_to_mins(val):
    if pd.isna(val): return None
    sv = str(val).strip().lower()
    if sv in ('', 'nan', 'none'): return None
    if ' ' in sv: sv = sv.split(' ')[-1]
    m = re.search(r'(\d{1,2}):(\d{2})(?::(\d{2}))?', sv)
    if m: return int(m.group(1)) * 60.0 + int(m.group(2)) + (int(m.group(3)) / 60.0 if m.group(3) else 0.0)
    try:
        f = float(sv)
        return f * 1440.0 if f < 1.0 else (int(f // 100) * 60.0) + (f % 100) if f < 2400.0 else None
    except: return None

def parse_excel_date(val):
    if pd.isna(val): return None
    if isinstance(val, (datetime, pd.Timestamp)): return val.strftime('%Y-%m-%d')
    v_str = str(val).strip()
    m_dt = re.search(r'\b(\d{1,2})[-/](\d{1,2})[-/](\d{2,4})\b', v_str)
    if m_dt:
        d, m, y = int(m_dt.group(1)), int(m_dt.group(2)), int(m_dt.group(3))
        if y < 100: y += 2000
        return f"{y:04d}-{m:02d}-{d:02d}"
    return None

def clean_id(x):
    try:
        nums = re.findall(r'\d+', str(x).strip().lower().replace(".0", ""))
        return str(int(nums[0])) if nums else str(x).strip().upper()
    except: return str(x).strip().upper()

def clean_primary_key(x):
    if pd.isna(x): return ''
    s = re.sub(r'[^A-Z0-9]', '', str(x).strip().upper().replace('.0', ''))
    return s.lstrip('0') if s not in ['NAN', ''] else ''

def clean_pax_number(x):
    if pd.isna(x): return 0
    s = re.sub(r'[^\d]', '', str(x).split('.')[0])
    try: return int(s) if s else 0
    except: return 0

def clasificar_dia(d_str):
    try:
        d = datetime.strptime(d_str, '%Y-%m-%d')
        if d_str in FERIADOS_SAFE or d.weekday() == 6: return 'Domingo/Festivo'
        if d.weekday() == 5: return 'Sábado'
        return 'Laboral'
    except: return 'Laboral'

def extraer_fecha_segura(df_raw, fname):
    numeros = re.findall(r'\d+', str(fname))
    for num in numeros:
        if len(num) == 6:
            d, m, y = int(num[0:2]), int(num[2:4]), int(num[4:6])
            if 1 <= d <= 31 and 1 <= m <= 12: return f"20{y:02d}-{m:02d}-{d:02d}"
        if len(num) == 8:
            d, m, y = int(num[0:2]), int(num[2:4]), int(num[4:8])
            if 1 <= d <= 31 and 1 <= m <= 12: return f"{y:04d}-{m:02d}-{d:02d}"
    if df_raw is not None:
        for i in range(min(10, len(df_raw))):
            for val in df_raw.iloc[i].values:
                dt = parse_excel_date(val)
                if dt: return dt
    return "2026-01-01"

def _col_to_est_idx(col):
    cu = re.sub(r'[^a-z0-9]','', str(col).lower().replace('á','a').replace('é','e').replace('í','i').replace('ó','o').replace('ú','u').replace('ñ','n'))
    for i, est in enumerate(ESTACIONES_SAFE):
        ne = re.sub(r'[^a-z0-9]','', est.lower().replace('á','a').replace('é','e').replace('í','i').replace('ó','o').replace('ú','u').replace('ñ','n'))
        if ne in cu: return i
    return None

def calc_tren_km_real_general(row):
    k_s, k_e = min(row['km_orig'], row['km_dest']), max(row['km_orig'], row['km_dest'])
    man = row.get('maniobra')
    if man in ['CORTE_BTO','ACOPLE_BTO','CORTE_PU_SA_BTO']:
        km_man = KM_ACUM_SAFE[14]
        if k_s <= km_man <= k_e: return abs(km_man-k_s)*2.0 + abs(k_e-km_man)*1.0
    elif man in ['CORTE_SA','ACOPLE_SA']:
        km_man = KM_ACUM_SAFE[18]
        if k_s <= km_man <= k_e: return abs(km_man-k_s)*2.0 + abs(k_e-km_man)*1.0
    return abs(k_e-k_s) * (2.0 if row.get('doble',False) else 1.0)

def make_unique(df):
    cols = pd.Series(df.columns)
    for dup in cols[cols.duplicated()].unique(): 
        cols[cols==dup] = [f"{dup}_{i}" if i else dup for i in range(sum(cols==dup))]
    df.columns = cols
    return df

def get_pax_at_km_nativo(pax_d, km_pos, via, pax_max_fallback=0):
    if not pax_d or not isinstance(pax_d, dict): return pax_max_fallback
    if sum(pax_d.values()) == 0 and pax_max_fallback > 0: return pax_max_fallback
    pax_val = 0

    if via == 1:
        for i in range(N_EST_SAFE):
            if km_pos >= KM_ACUM_SAFE[i]:
                val = pax_d.get(PAX_COLS_SAFE[i])
                if val is not None: pax_val = val
            else: break
    else:
        for i in range(N_EST_SAFE - 1, -1, -1):
            if km_pos <= KM_ACUM_SAFE[i]:
                val = pax_d.get(PAX_COLS_SAFE[i])
                if val is not None: pax_val = val
            else: break
    return int(pax_val)

get_pax_at_km = get_pax_at_km_nativo

# =============================================================================
# 2. PROCESAMIENTO THDR
# =============================================================================

def procesar_thdr(data, fname, via_param=1):
    try:
        ext = fname.lower()
        if ext.endswith('.csv'):
            try: raw = pd.read_csv(BytesIO(data), header=None, sep=',', encoding='utf-8', dtype=str)
            except: raw = pd.read_csv(BytesIO(data), header=None, sep=';', encoding='latin-1', dtype=str)
        else:
            eng = "xlrd" if ext.endswith(".xls") else "openpyxl"
            raw = pd.read_excel(BytesIO(data), header=None, engine=eng, dtype=str)

        if raw is None or raw.empty: return pd.DataFrame(), f"Archivo vacío o ilegible: {fname}"
        
        fecha_str = extraer_fecha_segura(raw, fname)
        
        header_idx = 1
        for i in range(min(15, len(raw))):
            row_vals = [str(x).upper() for x in raw.iloc[i].values if pd.notna(x)]
            row_str = ' '.join(row_vals)
            if ('VIAJE' in row_str or 'N°' in row_str or 'NRO' in row_str) and ('TREN' in row_str or 'MOTRIZ' in row_str or 'SFE' in row_str or 'SERVICIO' in row_str) and ('SALIDA' in row_str or 'HORA' in row_str or 'PARTIDA' in row_str):
                header_idx = i
                break
                
        r0 = raw.iloc[header_idx - 1].copy() if header_idx > 0 else raw.iloc[0].copy()
        r0.iloc[0] = np.nan 
        h1 = r0.ffill().astype(str)
        h2 = raw.iloc[header_idx].fillna('').astype(str)
        
        cols = []
        for s, t in zip(h1, h2):
            s_val = str(s).strip()
            t_val = str(t).strip()
            if s_val.lower() == 'nan' or not s_val: cols.append(t_val)
            elif t_val: cols.append(f"{s_val}_{t_val}")
            else: cols.append(s_val)
            
        df = raw.iloc[header_idx + 1:].copy().reset_index(drop=True)
        n = len(df.columns)
        if len(cols) >= n: df.columns = cols[:n]
        else: df.columns = cols + [f"_C{j}" for j in range(n - len(cols))]
            
        df = make_unique(df).dropna(how='all').reset_index(drop=True)
        if df.empty: return pd.DataFrame(), f"Sin filas tras limpiar: {fname}"

        for col in df.columns:
            if any(k in str(col).upper() for k in ['LLEGADA','SALIDA','HORA']):
                try: df[f"{col}_min"] = df[col].apply(parse_time_to_mins)
                except: pass

        est_cols = {c: _col_to_est_idx(c) for c in df.columns if '_min' in str(c).lower() and 'program' not in str(c).lower()}

        def _safe_get(r, col):
            try: return r.get(col, np.nan)
            except: return np.nan

        df['t_ini'] = df.apply(lambda row: min([_safe_get(row, c) for c in est_cols.keys() if pd.notna(_safe_get(row, c))] or [np.nan]), axis=1)
        df['t_fin'] = df.apply(lambda row: max([_safe_get(row, c) for c in est_cols.keys() if pd.notna(_safe_get(row, c))] or [np.nan]), axis=1)

        c_m1 = next((c for c in df.columns if 'motriz' in str(c).lower() and '1' in str(c).lower()), None)
        c_m2 = next((c for c in df.columns if 'motriz' in str(c).lower() and '2' in str(c).lower()), None)
        tren_col = next((c for c in df.columns if str(c).strip().upper() == 'TREN' or str(c).strip().upper() == 'SERVICIO'), None)

        def _get_fleet_info(r):
            def extract_n(col_name):
                if col_name and pd.notna(r.get(col_name)):
                    val = str(r.get(col_name)).strip()
                    if val.lower() not in ('nan', '', '0', '0.0'):
                        m = re.search(r'(\d+)', val)
                        if m: return int(m.group(1))
                return None
            n1 = extract_n(c_m1)
            n2 = extract_n(c_m2)
            tipo = "XT-100"
            motriz_str = ""
            n_eval = None
            if (n1 is not None and n1 != 0) and (n2 is not None and n2 != 0): motriz_str = f"{n1}+{n2}"; n_eval = n1
            elif (n1 is not None and n1 != 0): motriz_str = f"{n1}"; n_eval = n1
            elif (n2 is not None and n2 != 0): motriz_str = f"{n2}"; n_eval = n2
            else:
                n_tren = extract_n(tren_col)
                if n_tren is not None and n_tren != 0: motriz_str = f"{n_tren}"; n_eval = n_tren
            if n_eval is not None:
                if 1 <= n_eval <= 27: tipo = "XT-100"
                elif 28 <= n_eval <= 35: tipo = "XT-M"
                elif 410 <= n_eval <= 414: tipo = "SFE"
                else: tipo = "XT-100" 
            return pd.Series([motriz_str, tipo])
            
        df[['motriz_num', 'tipo_tren']] = df.apply(_get_fleet_info, axis=1)

        if 'Unidad' in df.columns: df['Unidad'] = df['Unidad'].fillna('S').replace('nan','S').replace('','S')
        else: df['Unidad'] = df[c_m2].apply(lambda x: 'M' if pd.notna(x) and str(x).strip() not in ('0','0.0','','nan') else 'S') if c_m2 else 'S'
            
        df['doble'] = df['Unidad'].astype(str).str.strip() == 'M'
        df['Via'] = via_param
        df['Fecha_str'] = fecha_str

        def _get_real_orig_dest(row):
            valid_est = []
            for col, e_idx in est_cols.items():
                val = _safe_get(row, col)
                if pd.notna(val) and val > 0:
                    valid_est.append(e_idx)
            if not valid_est:
                return pd.Series([0.0 if via_param == 1 else KM_TOTAL_SAFE, KM_TOTAL_SAFE if via_param == 1 else 0.0])
            
            if via_param == 1:
                return pd.Series([KM_ACUM_SAFE[min(valid_est)], KM_ACUM_SAFE[max(valid_est)]])
            else:
                return pd.Series([KM_ACUM_SAFE[max(valid_est)], KM_ACUM_SAFE[min(valid_est)]])

        df[['km_orig', 'km_dest']] = df.apply(_get_real_orig_dest, axis=1)
        df = df.dropna(subset=['t_ini'])
        
        df['km_viaje'] = abs(df['km_dest'] - df['km_orig'])
        df['svc_type'] = df.apply(lambda r: f"{EC_SAFE[KM_ACUM_SAFE.index(r['km_orig'])]}-{EC_SAFE[KM_ACUM_SAFE.index(r['km_dest'])]}", axis=1)

        def calc_dwell_dynamic(row):
            try:
                idx_orig = int(np.argmin([abs(row['km_orig'] - k) for k in KM_ACUM_SAFE]))
                idx_dest = int(np.argmin([abs(row['km_dest'] - k) for k in KM_ACUM_SAFE]))
                n_stops = max(0, abs(idx_dest - idx_orig) - 1)
                return round(n_stops * (8.0 / 19.0), 3)
            except: return 8.0 
                
        df['dwell_min'] = df.apply(calc_dwell_dynamic, axis=1)
        df['dwell_cabecera_min'] = 0.0
        
        def _extract_nodos(row):
            nodos_temp = []
            for col, e_idx in est_cols.items():
                val = _safe_get(row, col)
                if pd.notna(val) and val > 0:
                    nodos_temp.append((val, KM_ACUM_SAFE[e_idx]))
            
            nodos_validos = [n for n in nodos_temp if pd.notna(n[0])]
            nodos_validos.sort(key=lambda x: (x[1], x[0]))
            unique_nodos = []
            seen_km = set()
            for t, km in nodos_validos:
                if km not in seen_km:
                    unique_nodos.append((t, km))
                    seen_km.add(km)
                    
            unique_nodos.sort(key=lambda x: x[0])
            return unique_nodos if len(unique_nodos) > 1 else None
            
        df['nodos'] = df.apply(_extract_nodos, axis=1)

        viaje_col_idx = None
        for r in range(min(15, raw.shape[0])):
            for c in range(raw.shape[1]):
                val_raw = str(raw.iloc[r, c]).strip().upper()
                val_norm = unicodedata.normalize('NFD', val_raw).encode('ascii', 'ignore').decode()
                if ('VIAJE' in val_norm or 'N°' in val_norm or 'NRO' in val_norm) and 'TIEMPO' not in val_norm and 'MIN' not in val_norm and viaje_col_idx is None:
                    viaje_col_idx = c
                    break
                    
        if viaje_col_idx is not None and viaje_col_idx < len(df.columns):
            col_name_v = df.columns[viaje_col_idx]
            df['nro_viaje'] = df[col_name_v].apply(clean_primary_key)
        else: df['nro_viaje'] = ''

        serv_col = next((c for c in df.columns if 'servicio' in str(c).lower()), None)
        if serv_col: df['num_servicio'] = df[serv_col].apply(clean_primary_key)
        elif 'nro_viaje' in df.columns: df['num_servicio'] = df['nro_viaje']
        else: df['num_servicio'] = ''

        df['_id'] = df['Fecha_str'] + "_" + df['num_servicio'] + "_" + df['t_ini'].astype(str)
        df['t_fin'] = df['t_fin'].fillna(df['t_ini'] + df['km_viaje'] / 35.0 * 60.0)
        return df, "ok"
    except Exception as e: return pd.DataFrame(), str(e)


# =============================================================================
# 3. LECTURA Y CRUCE DE PASAJEROS (MATCH UNIVERSAL POR HORA)
# =============================================================================

def cargar_pax(data, fname, via_param=1):
    """LECTURA NATIVA EXCEL CON ESCÁNER INTELIGENTE: Busca la cabecera real sin importar la columna"""
    try:
        eng = "xlrd" if fname.lower().endswith(".xls") else "openpyxl"
        full = pd.read_excel(BytesIO(data), header=None, engine=eng, dtype=str)
        if full is None or full.empty or len(full) <= 10: return pd.DataFrame()
        
        header_idx = -1
        for r in range(min(20, len(full))):
            row_str = ' '.join(full.iloc[r].fillna('').astype(str).str.upper())
            if ('PUE' in row_str or 'PUERTO' in row_str) and ('LIM' in row_str or 'LIMACHE' in row_str) or 'TOTAL' in row_str:
                header_idx = r
                break
        
        if header_idx == -1:
            header_idx = 9

        EXACT_MAP = {'PUE':'PUE','PUERTO':'PUE', 'BEL':'BEL','BELLAVISTA':'BEL', 'FRA':'FRA', 'BAR':'BAR','BARON':'BAR', 'POR':'POR','PORTALES':'POR', 'REC':'REC','RECREO':'REC', 'MIR':'MIR','MIRAMAR':'MIR', 'VIN':'VIN','VINA DEL MAR':'VIN','VIÑA DEL MAR':'VIN', 'HOS':'HOS','HOSPITAL':'HOS', 'CHO':'CHO','CHORRILLOS':'CHO', 'SLT':'SLT','SALTO':'SLT','EL SALTO':'SLT', 'VAL':'VAL','VALENCIA':'VAL', 'QUI':'QUI','QUILPUE':'QUI','QUILPUÉ':'QUI', 'SOL':'SOL','EL SOL':'SOL', 'BTO':'BTO','EL BELLOTO':'BTO','BELLOTO':'BTO', 'AME':'AME','LAS AMERICAS':'AME','AMERICAS':'AME', 'CON':'CON','LA CONCEPCION':'CON','CONCEPCION':'CON', 'VAM':'VAM','VILLA ALEMANA':'VAM','ALEMANA':'VAM', 'SGA':'SGA','SARGENTO ALDEA':'SGA','ALDEA':'SGA', 'PEN':'PEN','PENABLANCA':'PEN','PEÑABLANCA':'PEN','PENA BLANCA':'PEN', 'LIM':'LIM','LIMACHE':'LIM'}
        
        col_mapping = {}
        keys_sorted = sorted(EXACT_MAP.keys(), key=len, reverse=True)
        
        for c_idx in range(full.shape[1]):
            vals = [str(full.iloc[r, c_idx]).strip().upper() for r in range(max(0, header_idx-2), header_idx+1)]
            combo = " ".join(vals)
            combo_norm = unicodedata.normalize('NFD', combo).encode('ascii', 'ignore').decode().replace('.', '').replace(':', '')

            mapped = False
            for k in keys_sorted:
                if k == vals[-1] or k == vals[-2] or f" {k} " in f" {combo_norm} " or f"_{k}_" in f"_{combo_norm}_":
                    col_mapping[col_mapping.get(c_idx, '')] = EXACT_MAP[k] 
                    col_mapping[c_idx] = EXACT_MAP[k]
                    mapped = True
                    break
            
            if mapped: continue
            
            if ('HORA' in combo_norm or 'SALIDA' in combo_norm or 'PARTIDA' in combo_norm) and 'Hora Origen' not in col_mapping.values(): 
                col_mapping[c_idx] = 'Hora Origen'
            elif 'THDR' in combo_norm and 'Nro_THDR_raw' not in col_mapping.values(): 
                col_mapping[c_idx] = 'Nro_THDR_raw'
            elif ('TREN' in combo_norm or 'SERVICIO' in combo_norm) and 'Tren' not in col_mapping.values(): 
                col_mapping[c_idx] = 'Tren'
            elif 'FECHA' in combo_norm and 'Fecha_s_raw' not in col_mapping.values():
                col_mapping[c_idx] = 'Fecha_s_raw'
            elif 'CargaMax' not in col_mapping.values():
                if any(w in combo_norm for w in ['TOTAL', 'BORDO', 'CARGA', 'PASAJERO']) and not any(exc in combo_norm for exc in ['THDR', 'TREN', 'HORA', 'VIA', 'FECHA']):
                    col_mapping[c_idx] = 'CargaMax'

        data_rows = full.iloc[header_idx + 1:].copy()
        df = pd.DataFrame()
        for c_idx, col_name in col_mapping.items():
            if isinstance(c_idx, int) and c_idx < full.shape[1]: 
                df[col_name] = data_rows.iloc[:, c_idx].values
                
        fecha_global = extraer_fecha_segura(full, fname)
        
        if 'Fecha_s_raw' in df.columns:
            df['Fecha_s'] = df['Fecha_s_raw'].apply(parse_excel_date).fillna(fecha_global).replace('', fecha_global).ffill()
        else:
            df['Fecha_s'] = fecha_global
                
        for col in ['Hora Origen', 'Nro_THDR_raw', 'Tren', 'CargaMax']:
            if col not in df.columns: df[col] = ''
        
        for c in PAX_COLS_SAFE:
            if c not in df.columns: df[c] = '0'

        df['Nro_THDR'] = df['Nro_THDR_raw'].apply(clean_primary_key)
        df['Tren_Clean'] = df['Tren'].apply(clean_id)
        df['t_ini_p'] = df['Hora Origen'].apply(parse_time_to_mins)
        df['Via'] = via_param
        df = df.dropna(subset=['t_ini_p'])
        
        if df.empty: return pd.DataFrame()
        
        for c in PAX_COLS_SAFE + ['CargaMax']: 
            df[c] = df[c].apply(lambda x: int(re.sub(r'[^\d]', '', str(x).replace('.', '').replace(',', '')) or 0))
        return df
    except Exception as e: 
        return pd.DataFrame()


def match_pax(row, df_pax):
    EMPTY = ({c: 0 for c in PAX_COLS_SAFE}, 0, '--:--:--', 'No Detectado', -1)
    if df_pax.empty: return EMPTY
    
    def _to_int(v):
        try: return int(float(v)) if pd.notna(v) else 0
        except: return 0
        
    t_i = row.get('t_ini')
    via = row.get('via_op', row.get('Via', 1))
    thdr_date = str(row.get('Fecha_str', '')).strip()
    
    sub = df_pax[df_pax['Via'] == via].copy()
    if sub.empty: return EMPTY
    
    if 'Fecha_s' in sub.columns and thdr_date and thdr_date != '2026-01-01':
        sub_date = sub[sub['Fecha_s'].astype(str).str.strip() == thdr_date]
        if not sub_date.empty:
            sub = sub_date

    # MATCH UNIVERSAL POR TIEMPO (Ignorando nombres completamente, tolerancia 60 mins)
    if pd.notna(t_i):
        sub['diff'] = sub['t_ini_p'].apply(lambda x: min(abs(float(x) - float(t_i)), 1440 - abs(float(x) - float(t_i))) if pd.notna(x) and pd.notna(t_i) else 9999)
        
        if not sub.empty:
            idx_min = sub['diff'].idxmin()
            best_match = sub.loc[idx_min]
            
            if best_match['diff'] <= 60: 
                return {c: _to_int(best_match.get(c, 0)) for c in PAX_COLS_SAFE}, _to_int(best_match.get('CargaMax', 0)), mins_to_time_str(best_match.get('t_ini_p')), str(best_match.get('Nro_THDR_raw', best_match.get('Tren', ''))), best_match.name

    return EMPTY

# =============================================================================
# 4. FUNCIONES AUXILIARES
# =============================================================================

def calcular_dwell(df1, df2):
    if df1.empty or df2.empty: return df1, df2
    if 'num_servicio' not in df1.columns or 'num_servicio' not in df2.columns: return df1, df2
    for fecha in df1['Fecha_str'].unique():
        d1 = df1[df1['Fecha_str']==fecha]
        d2 = df2[df2['Fecha_str']==fecha]
        if d2.empty: continue
        for idx1, r1 in d1.iterrows():
            s = r1.get('num_servicio')
            if pd.isna(s) or s == '': continue
            m = d2[(d2['num_servicio']==s) & (d2['t_ini']>r1['t_fin'])]
            if not m.empty:
                dw = m['t_ini'].min()-r1['t_fin']
                if 0<dw<60: df2.at[m['t_ini'].idxmin(),'dwell_cabecera_min']=round(dw,1)
        for idx2, r2 in d2.iterrows():
            s = r2.get('num_servicio')
            if pd.isna(s) or s == '': continue
            m = d1[(d1['num_servicio']==s) & (d1['t_ini']>r2['t_fin'])]
            if not m.empty:
                dw = m['t_ini'].min()-r2['t_fin']
                if 0<dw<60: df1.at[m['t_ini'].idxmin(),'dwell_cabecera_min']=round(dw,1)
    return df1, df2

def get_vacios_dia(df): 
    return []

def cargar_prevenciones(data, fname):
    try:
        raw = pd.read_csv(BytesIO(data), sep=',', encoding='latin-1')
        if raw.shape[1] < 2: raw = pd.read_csv(BytesIO(data), sep=';', encoding='latin-1')
        res = []
        for _, r in raw.iterrows():
            try:
                v1, v2 = float(str(r.iloc[0]).replace(',','.')), float(str(r.iloc[1]).replace(',','.'))
                vel = float(re.search(r'\d+', str(r.iloc[2])).group())
                res.append({'km_min': min(v1, v2), 'km_max': max(v1, v2), 'v_kmh': vel, 'via': int(r.iloc[3])})
            except: pass
        return res
    except: 
        try:
            raw = pd.read_excel(BytesIO(data), engine="openpyxl")
            res = []
            for _, r in raw.iterrows():
                try:
                    v1, v2 = float(str(r.iloc[0]).replace(',','.')), float(str(r.iloc[1]).replace(',','.'))
                    vel = float(re.search(r'\d+', str(r.iloc[2])).group())
                    res.append({'km_min': min(v1, v2), 'km_max': max(v1, v2), 'v_kmh': vel, 'via': int(r.iloc[3])})
                except: pass
            return res
        except: return []

def parsear_planilla_maestra(data, fname):
    try:
        ext = fname.lower()
        dfs = {}
        if ext.endswith('.csv'):
            try: raw = pd.read_csv(BytesIO(data), header=None, sep=',', encoding='utf-8', dtype=str)
            except: raw = pd.read_csv(BytesIO(data), header=None, sep=';', encoding='latin-1', dtype=str)
            dfs["CSV"] = raw
        else:
            eng = "xlrd" if ext.endswith(".xls") else "openpyxl"
            dfs = pd.read_excel(BytesIO(data), header=None, engine=eng, dtype=str, sheet_name=None)
            
        viajes = []
        for sheet_name, df in dfs.items():
            header_idx = -1
            for i in range(min(20, len(df))):
                row_str = ' '.join(df.iloc[i].fillna('').astype(str).str.upper())
                if ('VIAJE' in row_str or 'N°' in row_str or 'N ' in row_str) and ('SERVICIO' in row_str or 'TREN' in row_str) and ('HR PARTIDA' in row_str or 'HORA' in row_str or 'PARTIDA' in row_str or 'SALIDA' in row_str):
                    header_idx = i
                    break
                    
            if header_idx != -1:
                headers = df.iloc[header_idx].fillna('').astype(str).str.upper()
                viaje_cols = [c for c, val in enumerate(headers) if 'VIAJE' in val or val == 'N°' or val == 'N']
                srv_cols = [c for c, val in enumerate(headers) if 'SERV' in val or 'TREN' in val]
                hora_cols = [c for c, val in enumerate(headers) if 'HR PARTIDA' in val or 'HORA' in val or 'PARTIDA' in val or 'SALIDA' in val]
                config_cols = [c for c, val in enumerate(headers) if 'CONF' in val or 'TIPO' in val or 'FORMA' in val or 'UNIDAD' in val or 'OBS' in val]

                pairs = []
                for vc in viaje_cols:
                    sc_cands = [sc for sc in srv_cols if sc > vc and sc - vc <= 2]
                    if sc_cands:
                        sc = sc_cands[0]
                        hc_cands = [hc for hc in hora_cols if hc > sc and hc - sc <= 3]
                        if hc_cands:
                            hc = hc_cands[0]
                            cc_cands = [cc for cc in config_cols if cc > sc and cc - sc <= 6]
                            pairs.append((vc, sc, hc, cc_cands[0] if cc_cands else None))

                for i in range(header_idx + 1, len(df)):
                    row = df.iloc[i]
                    for col_viaje, col_srv, col_hora, col_config in pairs:
                        if pd.isna(row.get(col_hora)) or pd.isna(row.get(col_srv)) or pd.isna(row.get(col_viaje)): continue
                        hora_str = str(row[col_hora]).strip()
                        srv_str = str(row[col_srv]).strip()
                        viaje_str = str(row[col_viaje]).strip()
                        config_str = str(row[col_config]).strip().upper() if col_config is not None and pd.notna(row.get(col_config)) else ''

                        m_viaje = re.search(r'(\d+)', viaje_str)
                        m_srv = re.search(r'(\d{3,4})', srv_str)
                        if not m_viaje or not m_srv or not re.match(r'^\d{1,2}:\d{2}(:\d{2})?$', hora_str): continue
                        
                        viaje_num = int(m_viaje.group(1))
                        servicio_num = int(m_srv.group(1))
                        t_ini = parse_time_to_mins(hora_str)
                        if t_ini is None: continue

                        es_doble = False
                        if 'MÚLT' in config_str or 'MULT' in config_str or 'DOB' in config_str or '2' in config_str:
                            es_doble = True

                        via = 1 if viaje_num % 2 == 0 else 2
                        if via == 1:
                            km_orig = KM_ACUM_SAFE[0] 
                            if servicio_num >= 600: km_dest = KM_ACUM_SAFE[20] 
                            elif 400 <= servicio_num < 600: km_dest = KM_ACUM_SAFE[18] 
                            else: km_dest = KM_ACUM_SAFE[14] 
                        else:
                            km_dest = KM_ACUM_SAFE[0] 
                            if servicio_num >= 600: km_orig = KM_ACUM_SAFE[20] 
                            elif 400 <= servicio_num < 600: km_orig = KM_ACUM_SAFE[18] 
                            elif 200 <= servicio_num < 400: km_orig = KM_ACUM_SAFE[14] 
                            else: km_orig = KM_ACUM_SAFE[14] 
                            
                        ruta = f"{EC_SAFE[KM_ACUM_SAFE.index(km_orig)]}-{EC_SAFE[KM_ACUM_SAFE.index(km_dest)]}"
                        nodos_via = [(0.0, k) for k in (KM_ACUM_SAFE[KM_ACUM_SAFE.index(km_orig):KM_ACUM_SAFE.index(km_dest)+1] if via==1 else KM_ACUM_SAFE[KM_ACUM_SAFE.index(km_dest):KM_ACUM_SAFE.index(km_orig)+1][::-1])]
                        
                        viajes.append({
                            '_id': f"PLAN_{servicio_num}_{int(t_ini)}", 't_ini': t_ini, 'Via': via,
                            'km_orig': km_orig, 'km_dest': km_dest, 'nodos': nodos_via,
                            'tipo_tren': 'XT-100', 'doble': es_doble, 'num_servicio': str(servicio_num), 'svc_type': ruta,
                            'maniobra': None
                        })
            else:
                for i in range(len(df)):
                    row_vals = df.iloc[i].fillna('').astype(str).tolist()
                    for c_idx, val in enumerate(row_vals):
                        val = val.strip()
                        if re.match(r'^\d{1,2}:\d{2}(:\d{2})?$', val):
                            t_ini = parse_time_to_mins(val)
                            if t_ini is None: continue
                            
                            servicio_num, sc_idx = None, -1
                            for offset in range(1, 5):
                                if c_idx - offset >= 0:
                                    check_val = row_vals[c_idx - offset].strip()
                                    if check_val.isdigit() and 200 <= int(check_val) <= 1999:
                                        servicio_num = int(check_val)
                                        sc_idx = c_idx - offset
                                        break
                            
                            viaje_num = None
                            if sc_idx != -1:
                                for offset in range(1, 3):
                                    if sc_idx - offset >= 0:
                                        check_val = row_vals[sc_idx - offset].strip()
                                        if check_val.isdigit() and 1 <= int(check_val) <= 300:
                                            viaje_num = int(check_val)
                                            break
                                        
                            if servicio_num is None: continue

                            es_doble = False
                            for offset_unidad in range(1, 3):
                                if c_idx + offset_unidad < len(row_vals):
                                    val_unidad = row_vals[c_idx + offset_unidad].strip().upper()
                                    if 'MÚLT' in val_unidad or 'MULT' in val_unidad or 'DOB' in val_unidad or '2' in val_unidad:
                                        es_doble = True
                                        break

                            if viaje_num is None:
                                sheet_upper = str(sheet_name).upper()
                                if 'V1' in sheet_upper or 'VIA 1' in sheet_upper: via = 1
                                elif 'V2' in sheet_upper or 'VIA 2' in sheet_upper: via = 2
                                else: via = 1 if servicio_num % 2 == 0 else 2
                            else: via = 1 if viaje_num % 2 == 0 else 2
                            
                            if via == 1:
                                km_orig = KM_ACUM_SAFE[0] 
                                if servicio_num >= 600: km_dest = KM_ACUM_SAFE[20] 
                                elif 400 <= servicio_num < 600: km_dest = KM_ACUM_SAFE[18] 
                                elif 200 <= servicio_num < 400: km_dest = KM_ACUM_SAFE[14] 
                                else: km_dest = KM_ACUM_SAFE[14] 
                            else:
                                km_dest = KM_ACUM_SAFE[0] 
                                if servicio_num >= 600: km_orig = KM_ACUM_SAFE[20] 
                                elif 400 <= servicio_num < 600: km_orig = KM_ACUM_SAFE[18] 
                                elif 200 <= servicio_num < 400: km_orig = KM_ACUM_SAFE[14] 
                                else: km_orig = KM_ACUM_SAFE[14] 
                                
                            ruta = f"{EC_SAFE[KM_ACUM_SAFE.index(km_orig)]}-{EC_SAFE[KM_ACUM_SAFE.index(km_dest)]}"
                            nodos_via = [(0.0, k) for k in (KM_ACUM_SAFE[KM_ACUM_SAFE.index(km_orig):KM_ACUM_SAFE.index(km_dest)+1] if via==1 else KM_ACUM_SAFE[KM_ACUM_SAFE.index(km_dest):KM_ACUM_SAFE.index(km_orig)+1][::-1])]
                            viajes.append({'_id': f"PLAN_{servicio_num}_{int(t_ini)}", 't_ini': t_ini, 'Via': via, 'km_orig': km_orig, 'km_dest': km_dest, 'nodos': nodos_via, 'tipo_tren': 'XT-100', 'doble': es_doble, 'num_servicio': str(servicio_num), 'svc_type': ruta, 'maniobra': None})
                            
        df_viajes = pd.DataFrame(viajes)
        if not df_viajes.empty: df_viajes = df_viajes.drop_duplicates(subset=['_id'])
        return df_viajes, "ok"
    except Exception as e: return pd.DataFrame(), str(e)
