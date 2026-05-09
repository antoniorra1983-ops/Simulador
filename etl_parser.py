import pandas as pd
import numpy as np
import re
import unicodedata
from io import BytesIO
from datetime import datetime, date, timedelta
from config import *

def mins_to_time_str(mins):
    if pd.isna(mins): return '--:--:--'
    try:
        m_val = float(mins)
        while m_val >= 1440: m_val -= 1440
        while m_val < 0: m_val += 1440
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
    if m: return int(m.group(1)) * 60.0 + int(m.group(2)) + (int(m.group(3)) / 60.0 if m.group(3) else 0.0)
    try:
        f = float(sv)
        return f * 1440.0 if f < 1.0 else (int(f // 100) * 60.0) + (f % 100) if f < 2400.0 else None
    except: return None

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
        return 'Domingo/Festivo' if d_str in feriados_2026 or d.weekday() == 6 else ('Sábado' if d.weekday() == 5 else 'Laboral')
    except: return 'Laboral'

def extraer_fecha_segura(df_raw, fname):
    """
    💡 FIX V134: Lector de Fechas Robusto (Inmune a la falta de guiones)
    Detecta automáticamente archivos nombrados como "THDR_via1 020426.xls"
    """
    # 1. Patrones estándar con separadores (ej: 02-04-2026 o 02.04.26)
    for pat in [r'\b(\d{1,2})[-_\.](\d{1,2})[-_\.](\d{4})\b', r'\b(\d{4})[-_\.](\d{1,2})[-_\.](\d{1,2})\b']:
        m = re.search(pat, str(fname))
        if m:
            if len(m.group(1)) == 4:
                y, m_val, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
            else:
                d, m_val, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
            if m_val > 12 and d <= 12: d, m_val = m_val, d
            if 1 <= d <= 31 and 1 <= m_val <= 12: return f"{y:04d}-{m_val:02d}-{d:02d}"

    # 2. 🚀 PATRONES CONTINUOS: Busca números pegados sin separadores (Ej: 020426)
    numeros = re.findall(r'\d+', str(fname))
    for num in numeros:
        if len(num) == 8:
            # Prueba formato DDMMAAAA
            d, m_val, y = int(num[0:2]), int(num[2:4]), int(num[4:8])
            if 1 <= d <= 31 and 1 <= m_val <= 12 and 2000 <= y <= 2100: return f"{y:04d}-{m_val:02d}-{d:02d}"
            # Prueba formato AAAAMMDD
            y, m_val, d = int(num[0:4]), int(num[4:6]), int(num[6:8])
            if 1 <= d <= 31 and 1 <= m_val <= 12 and 2000 <= y <= 2100: return f"{y:04d}-{m_val:02d}-{d:02d}"
        elif len(num) == 6:
            # Prueba formato DDMMAA (El de tu imagen: 020426)
            d, m_val, y = int(num[0:2]), int(num[2:4]), int(num[4:6])
            if 1 <= d <= 31 and 1 <= m_val <= 12 and 20 <= y <= 35: return f"20{y:02d}-{m_val:02d}-{d:02d}"
            # Prueba formato AAMMDD
            y, m_val, d = int(num[0:2]), int(num[2:4]), int(num[4:6])
            if 1 <= d <= 31 and 1 <= m_val <= 12 and 20 <= y <= 35: return f"20{y:02d}-{m_val:02d}-{d:02d}"

    # 3. Fallback: Escaneo del contenido del Excel
    for i in range(min(50, len(df_raw))):
        row_str = ' '.join([str(x).strip() for x in df_raw.iloc[i].values if pd.notna(x)])
        for pat in [r'\b(\d{4})[-/\.](\d{1,2})[-/\.](\d{1,2})\b', r'\b(\d{1,2})[-/\.](\d{1,2})[-/\.](\d{4})\b']:
            m_dt = re.search(pat, row_str)
            if m_dt:
                if len(m_dt.group(1)) == 4:
                    y, m_val, d = int(m_dt.group(1)), int(m_dt.group(2)), int(m_dt.group(3))
                else:
                    d, m_val, y = int(m_dt.group(1)), int(m_dt.group(2)), int(m_dt.group(3))
                if m_val > 12 and d <= 12: d, m_val = m_val, d
                if 1 <= d <= 31 and 1 <= m_val <= 12: return f"{y:04d}-{m_val:02d}-{d:02d}"
        
        # Intentar pescar número serial de Excel (ej. 45396)
        row_vals = [str(x).strip() for x in df_raw.iloc[i].values if pd.notna(x)]
        for val in row_vals:
            val_clean = val.split('.')[0]
            if val_clean.isdigit() and 40000 <= int(val_clean) <= 60000:
                try: return (date(1899, 12, 30) + timedelta(days=int(val_clean))).strftime('%Y-%m-%d')
                except: pass

    # Falla absoluta
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
                d, m_val, y = int(s_pad[0:2]), int(s_pad[2:4]), int(s_pad[4:6])
                if 1 <= d <= 31 and 1 <= m_val <= 12: return f"{2000+y if y<100 else y:04d}-{m_val:02d}-{d:02d}"
            except: pass
            
    for pat in [r'\b(\d{1,2})[-/](\d{1,2})[-/](\d{4})\b', r'\b(\d{4})[-/](\d{1,2})[-/](\d{1,2})\b']:
        m_dt = re.search(pat, v_str)
        if m_dt:
            if len(m_dt.group(1)) == 4: 
                y, m_val, d = int(m_dt.group(1)), int(m_dt.group(2)), int(m_dt.group(3))
            else: 
                d, m_val, y = int(m_dt.group(1)), int(m_dt.group(2)), int(m_dt.group(3))
            if m_val > 12 and d <= 12: d, m_val = m_val, d
            if 1 <= d <= 31 and 1 <= m_val <= 12: return f"{y:04d}-{m_val:02d}-{d:02d}"
    return None

def clean_pax_number(x):
    if pd.isna(x): return 0
    s = re.sub(r'[^\d]', '', re.sub(r'\.0+$', '', str(x).strip().lower()).replace('.', '').replace(',', ''))
    try: return int(s) if s and s != 'nan' else 0
    except: return 0

_EST_NORM = sorted({re.sub(r'[^a-z0-9]','', e.lower().replace('á','a').replace('é','e').replace('í','i').replace('ó','o').replace('ú','u').replace('ñ','n')): i for i, e in enumerate(ESTACIONES)}.items(), key=lambda x: -len(x[0]))
def _col_to_est_idx(col):
    cu = re.sub(r'[^a-z0-9]','', col.lower().replace('á','a').replace('é','e').replace('í','i').replace('ó','o').replace('ú','u').replace('ñ','n'))
    if 'americas' in cu: return ESTACIONES.index('Las Americas')
    if 'vina' in cu: return ESTACIONES.index('Viña del Mar')
    if 'aldea' in cu: return ESTACIONES.index('Sargento Aldea')
    if 'belloto' in cu: return ESTACIONES.index('El Belloto')
    for nk, idx in _EST_NORM:
        if nk in cu: return idx
    return None

def calc_tren_km_real_general(row):
    k_s,k_e = min(row['km_orig'],row['km_dest']), max(row['km_orig'],row['km_dest'])
    man = row.get('maniobra')
    if man in ['CORTE_BTO','ACOPLE_BTO','CORTE_PU_SA_BTO']: return abs(KM_ACUM[14]-k_s)*2.0 + abs(k_e-KM_ACUM[14])*1.0 if k_s <= KM_ACUM[14] <= k_e else abs(k_e-k_s) * (2.0 if row.get('doble',False) else 1.0)
    elif man in ['CORTE_SA','ACOPLE_SA']: return abs(KM_ACUM[18]-k_s)*2.0 + abs(k_e-KM_ACUM[18])*1.0 if k_s <= KM_ACUM[18] <= k_e else abs(k_e-k_s) * (2.0 if row.get('doble',False) else 1.0)
    return abs(k_e-k_s) * (2.0 if row.get('doble',False) else 1.0)

def make_unique(df):
    if df.empty: return df
    cols = pd.Series(df.columns)
    for dup in cols[cols.duplicated()].unique(): cols[cols==dup] = [f"{dup}_{i}" if i else dup for i in range(sum(cols==dup))]
    df.columns = cols
    return df

def get_pax_at_km(pax_d, km_pos, via, pax_max_fallback=0):
    if not pax_d or not isinstance(pax_d, dict): return pax_max_fallback
    if sum(pax_d.values()) == 0 and pax_max_fallback > 0: return pax_max_fallback
    pax_val = 0
    if via == 1:
        for i in range(N_EST):
            if km_pos >= KM_ACUM[i]:
                val = pax_d.get(PAX_COLS[i])
                if val is not None: pax_val = val
            else: break
    else:
        for i in range(N_EST - 1, -1, -1):
            if km_pos <= KM_ACUM[i]:
                val = pax_d.get(PAX_COLS[i])
                if val is not None: pax_val = val
            else: break
    return int(pax_val)

def procesar_thdr(data, fname, via_param=1):
    try:
        raw = pd.read_csv(BytesIO(data), header=None, sep=',', encoding='utf-8', dtype=str) if fname.lower().endswith('.csv') else pd.read_excel(BytesIO(data), header=None, engine="openpyxl" if fname.lower().endswith(".xlsx") else "xlrd", dtype=str)
        if raw is None or raw.empty or raw.shape[0] < 6: return pd.DataFrame(), f"Archivo vacío/corto: {fname}"
        
        fecha_str = extraer_fecha_segura(raw, fname)
        header_idx = next((i for i in range(min(15, len(raw))) if ('VIAJE' in str(raw.iloc[i].values).upper() or 'NRO' in str(raw.iloc[i].values).upper()) and 'SALIDA' in str(raw.iloc[i].values).upper()), 1)
        r0 = raw.iloc[header_idx - 1].copy() if header_idx > 0 else raw.iloc[0].copy()
        r0.iloc[0] = np.nan 
        cols = [f"{str(s).strip()}_{str(t).strip()}" if str(s).strip() and str(s).strip().lower() != 'nan' and str(t).strip() else str(t).strip() or str(s).strip() for s, t in zip(r0.ffill().astype(str), raw.iloc[header_idx].fillna('').astype(str))]
        
        df = raw.iloc[header_idx + 1:].copy().reset_index(drop=True)
        df.columns = cols[:len(df.columns)] + [f"_C{j}" for j in range(len(df.columns) - len(cols))]
        df = df.dropna(how='all').reset_index(drop=True)

        for col in df.columns:
            if any(k in str(col).upper() for k in ['LLEGADA','SALIDA','HORA']):
                try: df[f"{col}_min"] = df[col].apply(parse_time_to_mins)
                except: pass

        est_cols = {c: _col_to_est_idx(c) for c in df.columns if '_min' in str(c).lower() and 'program' not in str(c).lower()}
        df['t_ini'] = df.apply(lambda row: min([row.get(c, np.nan) for c in est_cols.keys() if pd.notna(row.get(c, np.nan))] or [np.nan]), axis=1)
        df['t_fin'] = df.apply(lambda row: max([row.get(c, np.nan) for c in est_cols.keys() if pd.notna(row.get(c, np.nan))] or [np.nan]), axis=1)

        c_m1, c_m2, tren_col = next((c for c in df.columns if 'motriz' in str(c).lower() and '1' in str(c).lower()), None), next((c for c in df.columns if 'motriz' in str(c).lower() and '2' in str(c).lower()), None), next((c for c in df.columns if str(c).strip().upper() in ('TREN', 'SERVICIO')), None)

        def _get_fleet_info(r):
            n1, n2, n_tren = (int(re.search(r'(\d+)', str(r.get(c, ''))).group(1)) if re.search(r'(\d+)', str(r.get(c, ''))) else None for c in [c_m1, c_m2, tren_col])
            n_eval = n1 or n2 or n_tren
            tipo = "SFE" if n_eval and n_eval >= 36 else ("XT-M" if n_eval and 28 <= n_eval <= 35 else "XT-100")
            motriz_str = f"{n1}+{n2}" if n1 and n2 else str(n_eval or "")
            return pd.Series([motriz_str, tipo])
            
        df[['motriz_num', 'tipo_tren']] = df.apply(_get_fleet_info, axis=1)
        df['doble'] = (df['Unidad'].fillna('S') == 'M') if 'Unidad' in df.columns else (df[c_m2].apply(lambda x: pd.notna(x) and str(x).strip() not in ('0','0.0','','nan')) if c_m2 else False)
        df['Via'], df['Fecha_str'] = via_param, fecha_str

        def _get_real_orig_dest(row):
            valid_est = [e_idx for col, e_idx in est_cols.items() if pd.notna(row.get(col, np.nan)) and row.get(col) > 0]
            if not valid_est: return pd.Series([0.0 if via_param == 1 else KM_TOTAL, KM_TOTAL if via_param == 1 else 0.0])
            return pd.Series([KM_ACUM[min(valid_est)], KM_ACUM[max(valid_est)]]) if via_param == 1 else pd.Series([KM_ACUM[max(valid_est)], KM_ACUM[min(valid_est)]])

        df[['km_orig', 'km_dest']] = df.apply(_get_real_orig_dest, axis=1)
        df = df.dropna(subset=['t_ini'])
        df['km_viaje'] = abs(df['km_dest'] - df['km_orig'])
        df['svc_type'] = df.apply(lambda r: f"{EC[KM_ACUM.index(r['km_orig'])]}-{EC[KM_ACUM.index(r['km_dest'])]}", axis=1)
        
        def _extract_nodos(row):
            nodos_temp = [(row.get(col), KM_ACUM[e_idx]) for col, e_idx in est_cols.items() if pd.notna(row.get(col, np.nan)) and row.get(col) > 0]
            seen_km = set()
            return sorted([(t, k) for t, k in sorted([n for n in nodos_temp], key=lambda x: (x[1], x[0])) if not (k in seen_km or seen_km.add(k))], key=lambda x: x[0])
            
        df['nodos'] = df.apply(_extract_nodos, axis=1)
        viaje_col = next((c for c in df.columns if 'VIAJE' in str(c).upper() or 'NRO' in str(c).upper()), None)
        df['num_servicio'] = df[tren_col].apply(clean_primary_key) if tren_col else (df[viaje_col].apply(clean_primary_key) if viaje_col else '')
        df['_id'] = df['Fecha_str'] + "_" + df['num_servicio'] + "_" + df['t_ini'].astype(str)
        df['t_fin'] = df['t_fin'].fillna(df['t_ini'] + df['km_viaje'] / 35.0 * 60.0)
        return df, "ok"
    except Exception as e: return pd.DataFrame(), str(e)

def build_thdr_v71(blobs_v1, blobs_v2):
    all_p, err = [], []
    for blobs, via in [(blobs_v1, 1), (blobs_v2, 2)]:
        for nm, data in blobs:
            df, msg = procesar_thdr(data, nm, via)
            if not df.empty: all_p.append(df)
            else: err.append(f"[{nm}]: {msg}")
    if all_p:
        dm = pd.concat(all_p, ignore_index=True)
        df1, df2 = dm[dm['Via']==1].copy(), dm[dm['Via']==2].copy()
        if not df1.empty and not df2.empty: df1, df2 = calcular_dwell(df1, df2)
        return df1, df2, err
    return pd.DataFrame(), pd.DataFrame(), err

def calcular_dwell(df1, df2):
    if df1.empty or df2.empty: return df1, df2
    if 'num_servicio' not in df1.columns or 'num_servicio' not in df2.columns: return df1, df2
    for fecha in df1['Fecha_str'].unique():
        d1, d2 = df1[df1['Fecha_str']==fecha], df2[df2['Fecha_str']==fecha]
        if d2.empty: continue
        for idx1, r1 in d1.iterrows():
            s = r1.get('num_servicio')
            if pd.isna(s) or s == '': continue
            m = d2[(d2['num_servicio']==s) & (d2['t_ini']>r1['t_fin'])]
            if not m.empty and 0 < m['t_ini'].min()-r1['t_fin'] < 60: df2.at[m['t_ini'].idxmin(),'dwell_cabecera_min']=round(m['t_ini'].min()-r1['t_fin'],1)
        for idx2, r2 in d2.iterrows():
            s = r2.get('num_servicio')
            if pd.isna(s) or s == '': continue
            m = d1[(d1['num_servicio']==s) & (d1['t_ini']>r2['t_fin'])]
            if not m.empty and 0 < m['t_ini'].min()-r2['t_fin'] < 60: df1.at[m['t_ini'].idxmin(),'dwell_cabecera_min']=round(m['t_ini'].min()-r2['t_fin'],1)
    return df1, df2

def cargar_pax(data, fname, via_param=1):
    try:
        ext = fname.lower()
        if ext.endswith('.csv'):
            try: full = pd.read_csv(BytesIO(data), header=None, sep=',', encoding='utf-8', dtype=str)
            except: full = pd.read_csv(BytesIO(data), header=None, sep=';', encoding='latin-1', dtype=str)
        else: 
            eng = "xlrd" if ext.endswith(".xls") else "openpyxl"
            full = pd.read_excel(BytesIO(data), header=None, engine=eng, dtype=str)

        if full is None or full.empty or len(full) <= 10: return pd.DataFrame()

        # REGLA INQUEBRANTABLE 1: El título manda para el Fallback
        name_u = fname.upper()
        via_titulo = 1 if 'V1' in name_u or 'VIA 1' in name_u or 'VIA1' in name_u else (2 if 'V2' in name_u or 'VIA 2' in name_u or 'VIA2' in name_u else via_param)

        header_idx = 9 # Asumimos cabecera en fila 10 (índice 9)
        col_mapping = {}
        EXACT_MAP = {'PUE':'PUE','PUERTO':'PUE','PU':'PUE','BEL':'BEL','BELLAVISTA':'BEL','BE':'BEL','FRA':'FRA','FRANCIA':'FRA','FR':'FRA','BAR':'BAR','BARON':'BAR','BA':'BAR','POR':'POR','PORTALES':'POR','PO':'POR','REC':'REC','RECREO':'REC','RE':'REC','MIR':'MIR','MIRAMAR':'MIR','MI':'MIR','VIN':'VIN','VINA DEL MAR':'VIN','VIÑA DEL MAR':'VIN','VM':'VIN','HOS':'HOS','HOSPITAL':'HOS','HO':'HOS','CHO':'CHO','CHORRILLOS':'CHO','CH':'CHO','SLT':'SLT','SALTO':'SLT','EL SALTO':'SLT','ES':'SLT','ELS':'SLT','VAL':'VAL','VALENCIA':'VAL','QUI':'QUI','QUILPUE':'QUI','QUILPUÉ':'QUI','QU':'QUI','SOL':'SOL','EL SOL':'SOL','SO':'SOL','ESO':'SOL','BTO':'BTO','EL BELLOTO':'BTO','BELLOTO':'BTO','EB':'BTO','ELB':'BTO','AME':'AME','LAS AMERICAS':'AME','AMERICAS':'AME','LAS':'AME','LAM':'AME','AM':'AME','CON':'CON','LA CONCEPCION':'CON','CONCEPCION':'CON','LAC':'CON','LCO':'CON','CO':'CON','VAM':'VAM','VILLA ALEMANA':'VAM','ALEMANA':'VAM','VIL':'VAM','VALE':'VAM','VL':'VAM','SGA':'SGA','SARGENTO ALDEA':'SGA','ALDEA':'SGA','SAR':'SGA','SA':'SGA','PEN':'PEN','PENABLANCA':'PEN','PEÑABLANCA':'PEN','PENA BLANCA':'PEN','PENA':'PEN','PE':'PEN','LIM':'LIM','LIMACHE':'LIM','LI':'LIM'}
        keys_sorted = sorted(EXACT_MAP.keys(), key=len, reverse=True)

        # REGLA INQUEBRANTABLE 2: Si A10 dice N° THDR, la columna 0 es sagrada.
        val_a10 = str(full.iloc[9, 0]).strip().upper()
        if 'THDR' in val_a10 or 'N°' in val_a10 or 'NRO' in val_a10 or 'VIAJE' in val_a10 or re.search(r'\d+', val_a10):
            col_mapping[0] = 'Nro_THDR_raw'

        for c_idx in range(full.shape[1]):
            if c_idx in col_mapping: continue # Si ya mapeamos la columna 0, no la tocamos
            
            vals = [str(full.iloc[r, c_idx]).strip().upper() for r in range(max(0, header_idx-4), header_idx+1)]
            combo = " ".join(vals)
            combo_norm = unicodedata.normalize('NFD', combo).encode('ascii', 'ignore').decode().replace('.', '').replace(':', '')

            mapped = False
            for k in keys_sorted:
                if k == vals[-1] or k == vals[-2] or f" {k} " in f" {combo_norm} " or f"_{k}_" in f"_{combo_norm}_":
                    col_mapping[c_idx] = EXACT_MAP[k]
                    mapped = True
                    break
            
            if mapped: continue
            if 'HORA' in combo_norm and 'ORIG' in combo_norm: col_mapping[c_idx] = 'Hora Origen'
            elif 'THDR' in combo_norm and 'TREN' not in combo_norm: col_mapping[c_idx] = 'Nro_THDR_raw'
            elif 'TREN' in combo_norm or 'SERVICIO' in combo_norm: col_mapping[c_idx] = 'Tren'
            elif 'TOTAL' in combo_norm or 'BORDO' in combo_norm or 'CARGA' in combo_norm or 'MAX' in combo_norm:
                if 'CargaMax' not in col_mapping.values() and not any(exc in combo_norm for exc in ['THDR', 'TREN', 'HORA', 'VIA']):
                    col_mapping[c_idx] = 'CargaMax'

        data_rows = full.iloc[header_idx + 1:].copy()
        df = pd.DataFrame()
        for c_idx, col_name in col_mapping.items():
            if isinstance(c_idx, int) and c_idx < full.shape[1]: 
                df[col_name] = data_rows.iloc[:, c_idx].values
                
        fecha_global = extraer_fecha_segura(full, fname)
        date_col_idx = -1
        for c_idx in range(full.shape[1]):
            if 'FECHA' in str(full.iloc[header_idx, c_idx]).upper():
                date_col_idx = c_idx; break
        
        if date_col_idx != -1:
            df['Fecha_Excel_Raw'] = data_rows.iloc[:, date_col_idx].values
            df['Fecha_s'] = df['Fecha_Excel_Raw'].apply(parse_excel_date).fillna(fecha_global).replace('', fecha_global).ffill()
        elif full.shape[1] > 3 and 3 not in col_mapping:
            df['Fecha_Excel_Raw'] = data_rows.iloc[:, 3].values
            df['Fecha_s'] = df['Fecha_Excel_Raw'].apply(parse_excel_date).fillna(fecha_global).replace('', fecha_global).ffill()
        else:
            df['Fecha_s'] = fecha_global
                
        for col in ['Hora Origen', 'Nro_THDR_raw', 'Tren', 'CargaMax']:
            if col not in df.columns: df[col] = '' if col != 'CargaMax' else '0'
        for c in PAX_COLS:
            if c not in df.columns: df[c] = '0'

        df['Nro_THDR'] = df['Nro_THDR_raw'].apply(clean_primary_key)
        df['Tren_Clean'] = df['Tren'].apply(clean_id)
        df['t_ini_p'] = df['Hora Origen'].apply(parse_time_to_mins)
        
        # REGLA INQUEBRANTABLE 3: El N° THDR define la Vía (Par = V1, Impar = V2)
        def determinar_via_por_thdr(row):
            thdr_str = str(row.get('Nro_THDR_raw', ''))
            nums = re.findall(r'\d+', thdr_str)
            if nums:
                return 1 if int(nums[0]) % 2 == 0 else 2
            return via_titulo

        df['Via'] = df.apply(determinar_via_por_thdr, axis=1)
        
        df = df.dropna(subset=['t_ini_p'])
        if df.empty: return pd.DataFrame()
        for c in PAX_COLS + ['CargaMax']: df[c] = df[c].apply(clean_pax_number)
        return df
    except Exception as e: return pd.DataFrame()

def build_pax_v71(blobs_v1, blobs_v2):
    parts, err = [], []
    for blobs, via in [(blobs_v1, 1), (blobs_v2, 2)]:
        for nm, data in blobs:
            try: parts.append(cargar_pax(data, nm, via))
            except Exception as e: err.append(f"[{nm}]: {e}")
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(), err

def match_pax(row, df_pax):
    EMPTY = ({c: 0 for c in PAX_COLS}, 0, '--:--:--', 'No Detectado', -1)
    if df_pax.empty: return EMPTY
    def _to_int(v):
        try: return int(float(v)) if pd.notna(v) else 0
        except: return 0
        
    t_i = row.get('t_ini')
    via = row.get('via_op', row.get('Via', 1))
    nro_viaje = clean_primary_key(row.get('nro_viaje', ''))
    thdr_date = row.get('Fecha_str')
    
    sub = df_pax[df_pax['Via'] == via].copy()
    if sub.empty: return EMPTY
    
    if 'Fecha_s' in sub.columns and thdr_date and thdr_date != '2026-01-01':
        sub_date = sub[sub['Fecha_s'] == thdr_date]
        if not sub_date.empty: 
            sub = sub_date
        else: 
            return EMPTY 

    sub['diff'] = sub['t_ini_p'].apply(lambda x: min(abs(float(x) - float(t_i)), 1440 - abs(float(x) - float(t_i))) if pd.notna(x) and pd.notna(t_i) else 9999)
    if nro_viaje != '' and 'Nro_THDR' in sub.columns:
        sub['Nro_THDR_cmp'] = sub['Nro_THDR'].apply(clean_primary_key)
        match_exacto = sub[(sub['Nro_THDR_cmp'] == nro_viaje) & (sub['Nro_THDR_cmp'] != '')]
        if not match_exacto.empty:
            best = match_exacto.iloc[0]
            return {c: _to_int(best.get(c, 0)) for c in PAX_COLS}, _to_int(best.get('CargaMax', 0)), mins_to_time_str(best.get('t_ini_p')), str(best.get('Nro_THDR', '')), best.name

    if pd.notna(t_i):
        best_match = sub.loc[sub['diff'].idxmin()]
        if best_match['diff'] <= 15: 
            return {c: _to_int(best_match.get(c, 0)) for c in PAX_COLS}, _to_int(best_match.get('CargaMax', 0)), mins_to_time_str(best_match.get('t_ini_p')), str(best_match.get('Nro_THDR', '')), best_match.name

    return EMPTY

def cargar_prevenciones(data, fname):
    try:
        df = pd.read_excel(BytesIO(data), header=None, engine="openpyxl" if fname.lower().endswith(".xlsx") else "xlrd")
        prevs = []
        for i in range(len(df)):
            row = [str(x) for x in df.iloc[i].values]
            if len(row) >= 3:
                try:
                    v1, v2 = float(row[0].replace(',','.')), float(row[1].replace(',','.'))
                    v_kmh = float(re.search(r'\d+', row[2]).group())
                    prevs.append({'km_min': min(v1, v2), 'km_max': max(v1, v2), 'v_kmh': v_kmh, 'via': int(row[3]) if len(row)>3 else 1})
                except: pass
        return prevs
    except: return []

def parsear_planilla_maestra(data, fname):
    try:
        dfs = pd.read_excel(BytesIO(data), header=None, engine="xlrd" if fname.lower().endswith(".xls") else "openpyxl", dtype=str, sheet_name=None) if not fname.lower().endswith('.csv') else {"CSV": pd.read_csv(BytesIO(data), header=None, sep=',', encoding='utf-8', dtype=str)}
        viajes = []
        for sheet_name, df in dfs.items():
            header_idx = next((i for i in range(min(20, len(df))) if ('VIAJE' in ' '.join(df.iloc[i].fillna('').astype(str).str.upper()) or 'N°' in ' '.join(df.iloc[i].fillna('').astype(str).str.upper())) and ('PARTIDA' in ' '.join(df.iloc[i].fillna('').astype(str).str.upper()) or 'HORA' in ' '.join(df.iloc[i].fillna('').astype(str).str.upper()))), -1)
            if header_idx != -1:
                headers = df.iloc[header_idx].fillna('').astype(str).str.upper()
                v_cols = [c for c, v in enumerate(headers) if 'VIAJE' in v or v in ('N°', 'N')]
                s_cols = [c for c, v in enumerate(headers) if 'SERV' in v or 'TREN' in v]
                h_cols = [c for c, v in enumerate(headers) if 'HORA' in v or 'PARTIDA' in v or 'SALIDA' in v]
                c_cols = [c for c, v in enumerate(headers) if 'CONF' in v or 'TIPO' in v or 'UNIDAD' in v]
                pairs = [(vc, next((sc for sc in s_cols if sc > vc and sc - vc <= 2), None), next((hc for hc in h_cols if hc > next((sc for sc in s_cols if sc > vc and sc - vc <= 2), -99) and hc - next((sc for sc in s_cols if sc > vc and sc - vc <= 2), -99) <= 3), None), next((cc for cc in c_cols if cc > next((sc for sc in s_cols if sc > vc and sc - vc <= 2), -99) and cc - next((sc for sc in s_cols if sc > vc and sc - vc <= 2), -99) <= 6), None)) for vc in v_cols if next((sc for sc in s_cols if sc > vc and sc - vc <= 2), None) and next((hc for hc in h_cols if hc > next((sc for sc in s_cols if sc > vc and sc - vc <= 2), -99) and hc - next((sc for sc in s_cols if sc > vc and sc - vc <= 2), -99) <= 3), None)]
                for i in range(header_idx + 1, len(df)):
                    row = df.iloc[i]
                    for cv, cs, ch, cc in pairs:
                        if pd.isna(row.get(ch)) or pd.isna(row.get(cs)) or pd.isna(row.get(cv)): continue
                        m_v, m_s = re.search(r'(\d+)', str(row[cv])), re.search(r'(\d{3,4})', str(row[cs]))
                        if not m_v or not m_s or not re.match(r'^\d{1,2}:\d{2}(:\d{2})?$', str(row[ch]).strip()): continue
                        t_ini = parse_time_to_mins(str(row[ch]).strip())
                        if t_ini is None: continue
                        via = 1 if int(m_v.group(1)) % 2 == 0 else 2
                        km_orig, km_dest = (KM_ACUM[0], KM_ACUM[20] if int(m_s.group(1)) >= 600 else KM_ACUM[18] if 400 <= int(m_s.group(1)) < 600 else KM_ACUM[14]) if via == 1 else (KM_ACUM[20] if int(m_s.group(1)) >= 600 else KM_ACUM[18] if 400 <= int(m_s.group(1)) < 600 else KM_ACUM[14], KM_ACUM[0])
                        viajes.append({'_id': f"PLAN_{int(m_s.group(1))}_{int(t_ini)}", 't_ini': t_ini, 'Via': via, 'km_orig': km_orig, 'km_dest': km_dest, 'nodos': [(0.0, k) for k in (KM_ACUM[KM_ACUM.index(km_orig):KM_ACUM.index(km_dest)+1] if via==1 else KM_ACUM[KM_ACUM.index(km_dest):KM_ACUM.index(km_orig)+1][::-1])], 'tipo_tren': 'XT-100', 'doble': 'MÚLT' in str(row[cc]).upper() or 'MULT' in str(row[cc]).upper() or 'DOB' in str(row[cc]).upper() or '2' in str(row[cc]).upper() if cc is not None and pd.notna(row.get(cc)) else False, 'num_servicio': str(int(m_s.group(1))), 'svc_type': f"{EC[KM_ACUM.index(km_orig)]}-{EC[KM_ACUM.index(km_dest)]}", 'maniobra': None})
        df_viajes = pd.DataFrame(viajes)
        return df_viajes.drop_duplicates(subset=['_id']) if not df_viajes.empty else df_viajes, "ok"
    except Exception as e: return pd.DataFrame(), str(e)

def get_vacios_dia(df): return []
