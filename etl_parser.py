import pandas as pd
import numpy as np
import re
import unicodedata
from io import BytesIO
from datetime import datetime, date, timedelta
from config import *
import config

# =============================================================================
# 1. UTILIDADES Y GEOMETRÍA
# =============================================================================
def clasificar_dia(d_str):
    try:
        d = datetime.strptime(d_str, '%Y-%m-%d')
        if d_str in feriados_2026 or d.weekday() == 6: return 'Domingo/Festivo'
        if d.weekday() == 5: return 'Sábado'
        return 'Laboral'
    except:
        return 'Laboral'

def mins_to_time_str(mins):
    if pd.isna(mins): return '--:--:--'
    try:
        m_val = float(mins)
        while m_val >= 1440: m_val -= 1440
        while m_val < 0: m_val += 1440
        h = int(m_val // 60)
        m = int(m_val % 60)
        s = int(round((m_val * 60) % 60))
        if s == 60: 
            s = 0; m += 1
        if m == 60: 
            m = 0; h += 1
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
        if f < 1.0: return f * 1440.0
        if f < 2400.0: return (int(f // 100) * 60.0) + (f % 100)
    except: pass
    return None

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

def extraer_fecha_segura(df_raw, fname):
    for pat in [r'\b(\d{1,2})[-_\.](\d{1,2})[-_\.](\d{4})\b', r'\b(\d{4})[-_\.](\d{1,2})[-_\.](\d{1,2})\b']:
        m = re.search(pat, str(fname))
        if m:
            if len(m.group(1)) == 4: y, mon, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
            else: d, mon, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
            if mon > 12 and d <= 12: d, mon = mon, d
            if 1 <= d <= 31 and 1 <= mon <= 12: return f"{y:04d}-{mon:02d}-{d:02d}"

    s_fname = re.sub(r'\D', '', str(fname))
    for i in range(len(s_fname) - 7):
        match = s_fname[i:i+8]
        try:
            d, mon, y = int(match[:2]), int(match[2:4]), int(match[4:])
            if 1 <= d <= 31 and 1 <= mon <= 12 and 2000 <= y <= 2100: return f"{y:04d}-{mon:02d}-{d:02d}"
            y2, mon2, d2 = int(match[:4]), int(match[4:6]), int(match[6:])
            if 1 <= d2 <= 31 and 1 <= mon2 <= 12 and 2000 <= y2 <= 2100: return f"{y2:04d}-{mon2:02d}-{d2:02d}"
        except: pass
        
    for i in range(len(s_fname) - 5):
        match = s_fname[i:i+6]
        try:
            d, mon, y = int(match[:2]), int(match[2:4]), int(match[4:])
            if 1 <= d <= 31 and 1 <= mon <= 12 and 20 <= y <= 99: return f"{2000+y:04d}-{mon:02d}-{d:02d}"
            y2, mon2, d2 = int(match[:2]), int(match[2:4]), int(match[4:])
            if 20 <= y2 <= 99 and 1 <= mon2 <= 12 and 1 <= d2 <= 31: return f"{2000+y2:04d}-{mon2:02d}-{d2:02d}"
        except: pass
            
    for i in range(min(50, len(df_raw))):
        row_vals = [str(x).strip() for x in df_raw.iloc[i].values if pd.notna(x)]
        row_str = ' '.join(row_vals)
        m_dt = re.search(r'\b(\d{4})[-/\.](\d{1,2})[-/\.](\d{1,2})\b', row_str)
        if m_dt:
            y, mon, d = int(m_dt.group(1)), int(m_dt.group(2)), int(m_dt.group(3))
            if 1 <= d <= 31 and 1 <= mon <= 12: return f"{y:04d}-{mon:02d}-{d:02d}"
        m_d = re.search(r'\b(\d{1,2})[-/\.](\d{1,2})[-/\.](\d{4})\b', row_str)
        if m_d:
            d, mon, y = int(m_d.group(1)), int(m_d.group(2)), int(m_d.group(3))
            if mon > 12 and d <= 12: d, mon = mon, d
            if 1 <= d <= 31 and 1 <= mon <= 12: return f"{y:04d}-{mon:02d}-{d:02d}"
        m_d2 = re.search(r'\b(\d{1,2})[-/\.](\d{1,2})[-/\.](\d{2})\b', row_str)
        if m_d2 and not row_str.replace(".", "").isdigit():
            d, mon, y = int(m_d2.group(1)), int(m_d2.group(2)), int(m_d2.group(3))
            if mon > 12 and d <= 12: d, mon = mon, d
            if 1 <= d <= 31 and 1 <= mon <= 12: return f"{2000+y:04d}-{mon:02d}-{d:02d}"
        for val in row_vals:
            val_clean = val.split('.')[0]
            if val_clean.isdigit() and 40000 <= int(val_clean) <= 60000:
                try: return (date(1899, 12, 30) + timedelta(days=int(val_clean))).strftime('%Y-%m-%d')
                except: pass
    return "2026-01-01"

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

def make_unique(df):
    if df.empty: return df
    cols = pd.Series(df.columns)
    for dup in cols[cols.duplicated()].unique(): cols[cols==dup] = [f"{dup}_{i}" if i else dup for i in range(sum(cols==dup))]
    df.columns = cols
    return df

_EST_NORM = sorted({re.sub(r'[^a-z0-9]','', e.lower().replace('á','a').replace('é','e').replace('í','i').replace('ó','o').replace('ú','u').replace('ñ','n')): i for i, e in enumerate(ESTACIONES)}.items(), key=lambda x: -len(x[0]))
def _col_to_est_idx(col):
    cu = re.sub(r'[^a-z0-9]','', col.lower().replace('á','a').replace('é','e').replace('í','i').replace('ó','o').replace('ú','u').replace('ñ','n'))
    if 'americas' in cu: return ESTACIONES.index('Las Americas')
    if 'vina' in cu: return ESTACIONES.index('Viña del Mar')
    if 'aldea' in cu: return ESTACIONES.index('Sargento Aldea')
    if 'belloto' in cu: return ESTACIONES.index('El Belloto')
    if 'concepcion' in cu: return ESTACIONES.index('La Concepcion')
    if 'villaalem' in cu: return ESTACIONES.index('Villa Alemana')
    if 'salto' in cu: return ESTACIONES.index('El Salto')
    for nk, idx in _EST_NORM:
        if nk in cu: return idx
    return None

def interp_pos(km):
    km = max(0.0, min(float(km), KM_TOTAL))
    return float(np.interp(km, KM_ACUM, EST_LATS)), float(np.interp(km, KM_ACUM, EST_LONS))

def km_to_ec(km, tol=1.5):
    dists = [abs(km - k) for k in KM_ACUM]
    idx = int(np.argmin(dists))
    return EC[idx] if dists[idx] <= tol else f"{km:.1f}km"

def svc_label(km_orig, km_dest): 
    return f"{km_to_ec(km_orig)}-{km_to_ec(km_dest)}"

def calc_tren_km_real_general(row):
    k_s,k_e = min(row['km_orig'],row['km_dest']), max(row['km_orig'],row['km_dest'])
    man = row.get('maniobra')
    if man in ['CORTE_BTO','ACOPLE_BTO','CORTE_PU_SA_BTO']:
        km_man = KM_ACUM[14]
        if k_s <= km_man <= k_e: return abs(km_man-k_s)*2.0 + abs(k_e-km_man)*1.0
    elif man in ['CORTE_SA','ACOPLE_SA']:
        km_man = KM_ACUM[18]
        if k_s <= km_man <= k_e: return abs(km_man-k_s)*2.0 + abs(k_e-km_man)*1.0
    return abs(k_e-k_s) * (2.0 if row.get('doble',False) else 1.0)

# =============================================================================
# 2. FUNCIONES DE LECTURA (ETL - PASAJEROS Y THDR)
# =============================================================================
def get_perfiles_pax(df_px):
    if df_px.empty: return {}
    df_p = df_px.copy()
    df_p['Fecha_dt'] = pd.to_datetime(df_p['Fecha_s'], errors='coerce')
    df_p = df_p.dropna(subset=['Fecha_dt'])
    if df_p.empty: return {}
    df_p['Tipo_Dia'] = df_p['Fecha_s'].apply(clasificar_dia)
    for c in PAX_COLS + ['CargaMax']:
        if c in df_p.columns: df_p[c] = pd.to_numeric(df_p[c], errors='coerce').fillna(0)
    perfiles = {}
    for t_dia in ['Laboral', 'Sábado', 'Domingo/Festivo']:
        for via in [1, 2]:
            sub = df_p[(df_p['Tipo_Dia'] == t_dia) & (df_p['Via'] == via)]
            if not sub.empty:
                promedios = sub[PAX_COLS].mean().round().astype(int).to_dict()
                promedios['CargaMax_Promedio'] = int(sub['CargaMax'].mean().round())
            else:
                promedios = {c: 0 for c in PAX_COLS}
                promedios['CargaMax_Promedio'] = 0
            perfiles[(t_dia, via)] = promedios
    return perfiles

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
        ext = fname.lower()
        if ext.endswith('.csv'):
            try: raw = pd.read_csv(BytesIO(data), header=None, sep=',', encoding='utf-8', dtype=str)
            except: raw = pd.read_csv(BytesIO(data), header=None, sep=';', encoding='latin-1', dtype=str)
        else:
            eng = "xlrd" if ext.endswith(".xls") else "openpyxl"
            raw = pd.read_excel(BytesIO(data), header=None, engine=eng, dtype=str)

        if raw is None or raw.empty: return pd.DataFrame(), f"Archivo vacío o ilegible: {fname}"
        if raw.shape[0] < 6: return pd.DataFrame(), f"Archivo muy corto: {fname}"
        
        fecha_str = extraer_fecha_segura(raw, fname)
        
        header_idx = 1
        for i in range(min(15, len(raw))):
            row_vals = [str(x).upper() for x in raw.iloc[i].values if pd.notna(x)]
            row_str = ' '.join(row_vals)
            if ('VIAJE' in row_str or 'N°' in row_str or 'NRO' in row_str) and \
               ('LLEGADA' in row_str or 'SALIDA' in row_str or 'HORA' in row_str or 'PARTIDA' in row_str):
                header_idx = i
                break
                
        r0 = raw.iloc[header_idx - 1].copy() if header_idx > 0 else raw.iloc[0].copy()
        r0.iloc[0] = np.nan 
        h1 = r0.ffill().astype(str)
        h2 = raw.iloc[header_idx].fillna('').astype(str)
        
        cols = []
        for s, t in zip(h1, h2):
            s_val, t_val = str(s).strip(), str(t).strip()
            if s_val.lower() == 'nan' or not s_val: cols.append(t_val)
            elif t_val: cols.append(f"{s_val}_{t_val}")
            else: cols.append(s_val)
            
        df = raw.iloc[header_idx + 1:].copy().reset_index(drop=True)
        df.columns = [c if c else f"Unnamed_{i}" for i, c in enumerate(cols)]
        df = make_unique(df).dropna(how='all').reset_index(drop=True)

        if df.empty: return pd.DataFrame(), "Sin datos tras limpiar."

        time_cols = []
        for col in df.columns:
            is_time_col = False
            col_upper = str(col).upper()
            if any(k in col_upper for k in ['LLEGADA','SALIDA','HORA', 'LLEG', 'SAL']):
                is_time_col = True
            else:
                sample = df[col].dropna().head(10).astype(str)
                for val in sample:
                    if re.match(r'^\d{1,2}:\d{2}(:\d{2})?$', val.strip()):
                        is_time_col = True
                        break
            if is_time_col:
                try:
                    df[f"{col}_min"] = df[col].apply(parse_time_to_mins)
                    time_cols.append(f"{col}_min")
                except: pass

        est_cols = {c: _col_to_est_idx(c) for c in time_cols if _col_to_est_idx(c) is not None}
        if not est_cols: return pd.DataFrame(), "No se detectaron columnas de estaciones con tiempo."

        serv_col = None
        for c in df.columns:
            c_str = unicodedata.normalize('NFD', str(c)).encode('ascii', 'ignore').decode().upper()
            if ('SERVICIO' in c_str or 'TREN' in c_str or 'VIAJE' in c_str or 'NRO' in c_str) and 'TIEMPO' not in c_str and 'MIN' not in c_str:
                serv_col = c
                break

        c_m1 = next((c for c in df.columns if 'motriz' in str(c).lower() and '1' in str(c).lower()), None)
        c_m2 = next((c for c in df.columns if 'motriz' in str(c).lower() and '2' in str(c).lower()), None)

        def _get_fleet_info(r):
            def extract_n(col_name):
                if col_name and pd.notna(r.get(col_name)):
                    val = str(r.get(col_name)).strip()
                    m = re.search(r'(\d+)', val)
                    if m and int(m.group(1)) > 0: return int(m.group(1))
                return None
            
            n1, n2 = extract_n(c_m1), extract_n(c_m2)
            tipo = "XT-100"
            motriz_str = ""
            n_eval = None
            
            if n1 and n2: motriz_str, n_eval = f"{n1}+{n2}", n1
            elif n1: motriz_str, n_eval = f"{n1}", n1
            elif n2: motriz_str, n_eval = f"{n2}", n2
            else:
                n_tren = extract_n(serv_col) if serv_col else None
                if n_tren: motriz_str, n_eval = f"{n_tren}", n_tren
                    
            if n_eval is not None:
                if 1 <= n_eval <= 27: tipo = "XT-100"
                elif 28 <= n_eval <= 35: tipo = "XT-M"
                elif 410 <= n_eval <= 414: tipo = "SFE"
            
            es_doble = bool(n1 and n2)
            return pd.Series([motriz_str, tipo, es_doble])
            
        df[['motriz_num', 'tipo_tren', 'doble_calc']] = df.apply(_get_fleet_info, axis=1)
        
        if 'Unidad' in df.columns:
            df['doble'] = df['Unidad'].astype(str).str.upper().str.contains('M')
        else:
            df['doble'] = df['doble_calc']

        df['Via'] = via_param
        df['Fecha_str'] = fecha_str

        def _get_trip_geometry(row):
            stops = []
            for col, e_idx in est_cols.items():
                val = row.get(col, np.nan)
                if pd.notna(val) and val > 0:
                    stops.append((val, e_idx))
            
            if len(stops) < 2:
                return pd.Series([np.nan, np.nan, 0.0, 0.0, None])
            
            stops.sort(key=lambda x: x[0])
            adj_stops = []
            base_t = stops[0][0]
            for t, e in stops:
                if t < base_t - 300: 
                    adj_stops.append((t + 1440.0, e, t))
                else:
                    adj_stops.append((t, e, t))
            
            adj_stops.sort(key=lambda x: x[0])
            
            t_ini = adj_stops[0][0]
            t_fin = adj_stops[-1][0]
            km_orig = KM_ACUM[adj_stops[0][1]]
            km_dest = KM_ACUM[adj_stops[-1][1]]
            
            nodos = []
            seen_kms = set()
            for _, e, orig_t in adj_stops:
                k = KM_ACUM[e]
                if k not in seen_kms:
                    t_val = orig_t if orig_t >= t_ini or t_ini > 1440 else orig_t + 1440.0 
                    if t_val < t_ini: t_val += 1440.0 
                    nodos.append((t_val, k))
                    seen_kms.add(k)
                    
            nodos.sort(key=lambda x: x[0])
            return pd.Series([t_ini, t_fin, km_orig, km_dest, nodos])

        df[['t_ini', 't_fin', 'km_orig', 'km_dest', 'nodos']] = df.apply(_get_trip_geometry, axis=1)
        df = df.dropna(subset=['t_ini'])
        if df.empty: return pd.DataFrame(), "Todos los viajes descartados por falta de tiempos."
        
        df['km_viaje'] = abs(df['km_dest'] - df['km_orig'])
        df['svc_type'] = df.apply(lambda r: svc_label(r['km_orig'], r['km_dest']), axis=1)

        def calc_dwell_dynamic(row):
            try:
                idx_orig = int(np.argmin([abs(row['km_orig'] - k) for k in KM_ACUM]))
                idx_dest = int(np.argmin([abs(row['km_dest'] - k) for k in KM_ACUM]))
                n_stops = max(0, abs(idx_dest - idx_orig) - 1)
                return round(n_stops * (8.0 / 19.0), 3)
            except: return 8.0 
                
        df['dwell_min'] = df.apply(calc_dwell_dynamic, axis=1)
        df['dwell_cabecera_min'] = 0.0
        
        if serv_col: df['num_servicio'] = df[serv_col].apply(clean_primary_key)
        else: df['num_servicio'] = 'DESC'

        df['_id'] = df['Fecha_str'] + "_" + df['num_servicio'] + "_" + df['t_ini'].astype(str)
        df['t_fin'] = df['t_fin'].fillna(df['t_ini'] + df['km_viaje'] / 35.0 * 60.0)
        
        return df, "ok"
    except Exception as e:
        return pd.DataFrame(), str(e)

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

        header_idx = 9
        EXACT_MAP = {'PUE':'PUE','PUERTO':'PUE','PU':'PUE','BEL':'BEL','BELLAVISTA':'BEL','BE':'BEL','FRA':'FRA','FRANCIA':'FRA','FR':'FRA','BAR':'BAR','BARON':'BAR','BA':'BAR','POR':'POR','PORTALES':'POR','PO':'POR','REC':'REC','RECREO':'REC','RE':'REC','MIR':'MIR','MIRAMAR':'MIR','MI':'MIR','VIN':'VIN','VINA DEL MAR':'VIN','VIÑA DEL MAR':'VIN','VM':'VIN','HOS':'HOS','HOSPITAL':'HOS','HO':'HOS','CHO':'CHO','CHORRILLOS':'CHO','CH':'CHO','SLT':'SLT','SALTO':'SLT','EL SALTO':'SLT','ES':'SLT','ELS':'SLT','VAL':'VAL','VALENCIA':'VAL','QUI':'QUI','QUILPUE':'QUI','QUILPUÉ':'QUI','QU':'QUI','SOL':'SOL','EL SOL':'SOL','SO':'SOL','ESO':'SOL','BTO':'BTO','EL BELLOTO':'BTO','BELLOTO':'BTO','EB':'BTO','ELB':'BTO','AME':'AME','LAS AMERICAS':'AME','AMERICAS':'AME','LAS':'AME','LAM':'AME','AM':'AME','CON':'CON','LA CONCEPCION':'CON','CONCEPCION':'CON','LAC':'CON','LCO':'CON','CO':'CON','VAM':'VAM','VILLA ALEMANA':'VAM','ALEMANA':'VAM','VIL':'VAM','VALE':'VAM','VL':'VAM','SGA':'SGA','SARGENTO ALDEA':'SGA','ALDEA':'SGA','SAR':'SGA','SA':'SGA','PEN':'PEN','PENABLANCA':'PEN','PEÑABLANCA':'PEN','PENA BLANCA':'PEN','PENA':'PEN','PE':'PEN','LIM':'LIM','LIMACHE':'LIM','LI':'LIM'}
        col_mapping = {}
        keys_sorted = sorted(EXACT_MAP.keys(), key=len, reverse=True)
        
        for c_idx in range(full.shape[1]):
            vals = [str(full.iloc[r, c_idx]).strip().upper() for r in range(max(0, header_idx-4), header_idx+1)]
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
            if 'HORA' in combo_norm and 'ORIG' in combo_norm: col_mapping[c_idx] = 'Hora Origen'
            elif 'THDR' in combo_norm and 'TREN' not in combo_norm: col_mapping[c_idx] = 'Nro_THDR_raw'
            elif 'TREN' in combo_norm or 'SERVICIO' in combo_norm: col_mapping[c_idx] = 'Tren'
            elif 'CargaMax' not in col_mapping.values():
                if any(w in combo_norm for w in ['TOTAL', 'BORDO', 'CARGA', 'PASAJERO']) and not any(exc in combo_norm for exc in ['THDR', 'TREN', 'HORA', 'VIA']):
                    col_mapping[c_idx] = 'CargaMax'

        data_rows = full.iloc[header_idx + 1:].copy()
        df = pd.DataFrame()
        for c_idx, col_name in col_mapping.items():
            if isinstance(c_idx, int) and c_idx < full.shape[1]: 
                df[col_name] = data_rows.iloc[:, c_idx].values
                
        fecha_global = extraer_fecha_segura(full, fname)
        if full.shape[1] > 3:
            df['Fecha_Excel_Raw'] = data_rows.iloc[:, 3].values
            df['Fecha_s'] = df['Fecha_Excel_Raw'].apply(parse_excel_date).fillna(fecha_global).replace('', fecha_global).ffill()
        else:
            df['Fecha_s'] = fecha_global
                
        for col in ['Hora Origen', 'Nro_THDR_raw', 'Tren']:
            if col not in df.columns: df[col] = ''
        if 'CargaMax' not in df.columns: df['CargaMax'] = '0'
        for c in PAX_COLS:
            if c not in df.columns: df[c] = '0'

        df['Nro_THDR'] = df['Nro_THDR_raw'].apply(clean_primary_key)
        df['Tren_Clean'] = df['Tren'].apply(clean_id)
        df['t_ini_p'] = df['Hora Origen'].apply(parse_time_to_mins)
        df['Via'] = via_param
        df = df.dropna(subset=['t_ini_p'])
        if df.empty: return pd.DataFrame()
        for c in PAX_COLS + ['CargaMax']: df[c] = df[c].apply(clean_pax_number)
        return df
    except Exception as e: return pd.DataFrame()

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
            best_match = match_exacto.iloc[0]
            return {c: _to_int(best_match.get(c, 0)) for c in PAX_COLS}, _to_int(best_match.get('CargaMax', 0)), mins_to_time_str(best_match.get('t_ini_p')), str(best_match.get('Nro_THDR', '')), best_match.name

    if pd.notna(t_i):
        best_match = sub.loc[sub['diff'].idxmin()]
        if best_match['diff'] <= 15: 
            return {c: _to_int(best_match.get(c, 0)) for c in PAX_COLS}, _to_int(best_match.get('CargaMax', 0)), mins_to_time_str(best_match.get('t_ini_p')), str(best_match.get('Nro_THDR', '')), best_match.name

    return EMPTY

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
                            km_orig = KM_ACUM[0] 
                            if servicio_num >= 600: km_dest = KM_ACUM[20] 
                            elif 400 <= servicio_num < 600: km_dest = KM_ACUM[18] 
                            else: km_dest = KM_ACUM[14] 
                        else:
                            km_dest = KM_ACUM[0] 
                            if servicio_num >= 600: km_orig = KM_ACUM[20] 
                            elif 400 <= servicio_num < 600: km_orig = KM_ACUM[18] 
                            elif 200 <= servicio_num < 400: km_orig = KM_ACUM[14] 
                            else: km_orig = KM_ACUM[14] 
                            
                        ruta = f"{EC[KM_ACUM.index(km_orig)]}-{EC[KM_ACUM.index(km_dest)]}"
                        nodos_via = [(0.0, k) for k in (KM_ACUM[KM_ACUM.index(km_orig):KM_ACUM.index(km_dest)+1] if via==1 else KM_ACUM[KM_ACUM.index(km_dest):KM_ACUM.index(km_orig)+1][::-1])]
                        
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
                                km_orig = KM_ACUM[0] 
                                if servicio_num >= 600: km_dest = KM_ACUM[20] 
                                elif 400 <= servicio_num < 600: km_dest = KM_ACUM[18] 
                                elif 200 <= servicio_num < 400: km_dest = KM_ACUM[14] 
                                else: km_dest = KM_ACUM[14] 
                            else:
                                km_dest = KM_ACUM[0] 
                                if servicio_num >= 600: km_orig = KM_ACUM[20] 
                                elif 400 <= servicio_num < 600: km_orig = KM_ACUM[18] 
                                elif 200 <= servicio_num < 400: km_orig = KM_ACUM[14] 
                                else: km_orig = KM_ACUM[14] 
                                
                            ruta = f"{EC[KM_ACUM.index(km_orig)]}-{EC[KM_ACUM.index(km_dest)]}"
                            nodos_via = [(0.0, k) for k in (KM_ACUM[KM_ACUM.index(km_orig):KM_ACUM.index(km_dest)+1] if via==1 else KM_ACUM[KM_ACUM.index(km_dest):KM_ACUM.index(km_orig)+1][::-1])]
                            viajes.append({'_id': f"PLAN_{servicio_num}_{int(t_ini)}", 't_ini': t_ini, 'Via': via, 'km_orig': km_orig, 'km_dest': km_dest, 'nodos': nodos_via, 'tipo_tren': 'XT-100', 'doble': es_doble, 'num_servicio': str(servicio_num), 'svc_type': ruta, 'maniobra': None})
                            
        df_viajes = pd.DataFrame(viajes)
        if not df_viajes.empty: df_viajes = df_viajes.drop_duplicates(subset=['_id'])
        return df_viajes, "ok"
    except Exception as e: return pd.DataFrame(), str(e)

def cargar_vacios_efe(data, fname):
    try:
        raw = pd.read_csv(BytesIO(data), sep=',', encoding='utf-8') if fname.lower().endswith('.csv') else pd.read_excel(BytesIO(data), engine="xlrd" if fname.lower().endswith(".xls") else "openpyxl")
        if raw is None or raw.empty: return pd.DataFrame()
        fecha_str = extraer_fecha_segura(raw, fname)
        raw.columns = [str(c).strip().upper() for c in raw.columns]
        col_tren, col_hora, col_desde, col_hasta, col_kms = next((c for c in raw.columns if 'TREN' in c), None), next((c for c in raw.columns if 'HORA' in c), None), next((c for c in raw.columns if 'DESDE' in c), None), next((c for c in raw.columns if 'HASTA' in c), None), next((c for c in raw.columns if 'KMS' in c), None)
        if not all([col_tren, col_hora, col_desde, col_hasta]): return pd.DataFrame()
        viajes = []
        for _, r in raw.iterrows():
            tren, hora, desde, hasta, kms_str = str(r[col_tren]).strip(), str(r[col_hora]).strip(), str(r[col_desde]).strip(), str(r[col_hasta]).strip(), str(r.get(col_kms, '0')).replace(',', '.')
            if tren in ('nan', '') or hora in ('nan', ''): continue
            t_ini = parse_time_to_mins(hora)
            if t_ini is None: continue
            try: kms = float(kms_str)
            except: kms = 0.0
            num = int(re.search(r'\d+', tren.upper()).group(0)) if re.search(r'\d+', tren.upper()) else 0
            tipo_tren = 'SFE' if 'SFE' in tren.upper() or num >= 36 else ('XT-M' if 'XT' in tren.upper() and 28 <= num <= 35 or 28 <= num <= 35 else 'XT-100')
            def _map_est(name):
                n = name.upper()
                if 'COCHERA' in n or 'TALLER' in n or 'BELLOTO' in n: return 14
                if 'ALDEA' in n: return 18
                if 'LIMACHE' in n: return 20
                if 'PUERTO' in n: return 0
                idx = _col_to_est_idx(n)
                return idx if idx is not None else 14
            idx_orig, idx_dest = _map_est(desde), _map_est(hasta)
            viajes.append({'Fecha_str': fecha_str, 't_asigned': t_ini, 'tipo': tipo_tren, 'doble': False, 'origen_txt': desde, 'destino_txt': hasta, 'km_orig': KM_ACUM[idx_orig], 'km_dest': KM_ACUM[idx_dest], 'dist': kms, 'Via': 1 if KM_ACUM[idx_orig] <= KM_ACUM[idx_dest] else 2, 'motriz_num': tren, 'is_efe': True, 'cochera': 'COCHERA' in desde.upper() or 'COCHERA' in hasta.upper()})
        return pd.DataFrame(viajes)
    except Exception as e: return pd.DataFrame()

def get_vacios_dia(df_dia):
    # 💡 ESCUDO DEFENSIVO NATIVO: Evita NameError en la nube
    cap_puerto = getattr(config, 'CAP_PUERTO', 4)
    cap_limache = getattr(config, 'CAP_LIMACHE', 16)
    
    vacios = []
    if df_dia.empty: return vacios
    agrupador = 'motriz_num' if 'motriz_num' in df_dia.columns else 'num_servicio'
    def _get_est_name(km): return ESTACIONES[int(np.argmin([abs(km - k) for k in KM_ACUM]))] if min([abs(km - k) for k in KM_ACUM]) <= 1.5 else f"km {km:.1f}"

    end_locations = {'PU': [], 'LI': [], 'BTO': [], 'SA': [], 'OTHER': []}
    start_locations = {'PU': [], 'LI': []}

    for tren, group in df_dia.sort_values('t_ini').groupby(agrupador):
        if str(tren).strip() == '' or str(tren).strip() == 'nan': continue
        viajes = group.to_dict('records')
        if not viajes: continue
        
        p = viajes[0]
        if p.get('km_orig', 0) <= 2.0:
            start_locations['PU'].append({'tren': tren, 't_ini': p['t_ini'], 'tipo': p.get('tipo_tren', 'XT-100'), 'doble': p.get('doble', False), 'num_servicio': str(p.get('num_servicio', ''))})
        elif p.get('km_orig', 0) >= 41.0:
            start_locations['LI'].append({'tren': tren, 't_ini': p['t_ini'], 'tipo': p.get('tipo_tren', 'XT-100'), 'doble': p.get('doble', False), 'num_servicio': str(p.get('num_servicio', ''))})

        if abs(p.get('km_orig', 0) - KM_ACUM[14]) < 0.1: vacios.append({'t_asigned': p['t_ini'] - 10, 'tipo': p.get('tipo_tren', 'XT-100'), 'doble': p.get('doble', False), 'cochera': True, 'km_orig': KM_ACUM[14], 'km_dest': KM_ACUM[14], 'dist': 2.0, 'motriz_num': tren, 'origen_txt': 'Taller / Cochera', 'destino_txt': 'El Belloto', 'servicio_previo': '—', 'servicio_siguiente': str(p.get('num_servicio', ''))})
        elif abs(p.get('km_orig', 0) - KM_ACUM[18]) < 0.1: vacios.append({'t_asigned': p['t_ini'] - 20, 'tipo': p.get('tipo_tren', 'XT-100'), 'doble': p.get('doble', False), 'cochera': True, 'km_orig': KM_ACUM[14], 'km_dest': KM_ACUM[18], 'dist': 2.0 + abs(KM_ACUM[18]-KM_ACUM[14]), 'motriz_num': tren, 'origen_txt': 'Taller / Cochera', 'destino_txt': 'Sargento Aldea', 'servicio_previo': '—', 'servicio_siguiente': str(p.get('num_servicio', ''))})
            
        for i in range(len(viajes) - 1):
            k_o, k_d = viajes[i].get('km_dest', 0), viajes[i+1].get('km_orig', 0)
            if 0.1 < abs(k_o - k_d) <= 20.0: vacios.append({'t_asigned': viajes[i]['t_fin'] + 5, 'tipo': viajes[i].get('tipo_tren', 'XT-100'), 'doble': viajes[i].get('doble', False), 'cochera': False, 'km_orig': k_o, 'km_dest': k_d, 'dist': abs(k_o - k_d), 'motriz_num': tren, 'origen_txt': _get_est_name(k_o), 'destino_txt': _get_est_name(k_d), 'servicio_previo': str(viajes[i].get('num_servicio', '')), 'servicio_siguiente': str(viajes[i+1].get('num_servicio', ''))})

        u = viajes[-1]
        km_d = u.get('km_dest', 0)
        loc = 'OTHER'
        if km_d <= 2.0: loc = 'PU'
        elif km_d >= 41.0: loc = 'LI'
        elif abs(km_d - KM_ACUM[14]) < 2.0: loc = 'BTO'
        elif abs(km_d - KM_ACUM[18]) < 2.0: loc = 'SA'

        end_locations[loc].append({
            'tren': tren, 't_fin': u['t_fin'], 'km_dest': km_d, 'tipo': u.get('tipo_tren', 'XT-100'), 'doble': u.get('doble', False), 'num_servicio': str(u.get('num_servicio', ''))
        })

    pu_starts = sorted(start_locations['PU'], key=lambda x: x['t_ini'])
    for i, t in enumerate(pu_starts):
        if i >= cap_puerto:
            vacios.append({'t_asigned': t['t_ini'] - 40.0, 'tipo': t['tipo'], 'doble': t['doble'], 'cochera': True, 'km_orig': KM_ACUM[14], 'km_dest': KM_ACUM[0], 'dist': abs(KM_ACUM[14] - 0), 'motriz_num': t['tren'], 'origen_txt': 'Taller EB', 'destino_txt': 'Puerto (Inyección Matutina)', 'servicio_previo': '—', 'servicio_siguiente': t['num_servicio']})

    li_starts = sorted(start_locations['LI'], key=lambda x: x['t_ini'])
    for i, t in enumerate(li_starts):
        if i >= cap_limache:
            vacios.append({'t_asigned': t['t_ini'] - 40.0, 'tipo': t['tipo'], 'doble': t['doble'], 'cochera': True, 'km_orig': KM_ACUM[14], 'km_dest': KM_ACUM[20], 'dist': abs(KM_ACUM[14] - KM_ACUM[20]), 'motriz_num': t['tren'], 'origen_txt': 'Taller EB', 'destino_txt': 'Limache (Inyección Matutina)', 'servicio_previo': '—', 'servicio_siguiente': t['num_servicio']})

    pu_trains = sorted(end_locations['PU'], key=lambda x: x['t_fin'], reverse=True)
    for i, t in enumerate(pu_trains):
        if i >= cap_puerto:
            vacios.append({'t_asigned': t['t_fin'] + 10.0, 'tipo': t['tipo'], 'doble': t['doble'], 'cochera': True, 'km_orig': t['km_dest'], 'km_dest': KM_ACUM[14], 'dist': abs(KM_ACUM[14] - t['km_dest']), 'motriz_num': t['tren'], 'origen_txt': 'Puerto (Exceso Cap)', 'destino_txt': 'Taller EB', 'servicio_previo': t['num_servicio'], 'servicio_siguiente': '—'})

    li_trains = sorted(end_locations['LI'], key=lambda x: x['t_fin'], reverse=True)
    for i, t in enumerate(li_trains):
        if i >= cap_limache:
            vacios.append({'t_asigned': t['t_fin'] + 10.0, 'tipo': t['tipo'], 'doble': t['doble'], 'cochera': True, 'km_orig': t['km_dest'], 'km_dest': KM_ACUM[14], 'dist': abs(KM_ACUM[14] - t['km_dest']), 'motriz_num': t['tren'], 'origen_txt': 'Limache (Exceso Cap)', 'destino_txt': 'Taller EB', 'servicio_previo': t['num_servicio'], 'servicio_siguiente': '—'})

    for t in end_locations['BTO']: vacios.append({'t_asigned': t['t_fin'] + 5, 'tipo': t['tipo'], 'doble': t['doble'], 'cochera': True, 'km_orig': t['km_dest'], 'km_dest': KM_ACUM[14], 'dist': 2.0, 'motriz_num': t['tren'], 'origen_txt': 'El Belloto', 'destino_txt': 'Taller / Cochera', 'servicio_previo': t['num_servicio'], 'servicio_siguiente': '—'})
    for t in end_locations['SA']: vacios.append({'t_asigned': t['t_fin'] + 5, 'tipo': t['tipo'], 'doble': t['doble'], 'cochera': True, 'km_orig': t['km_dest'], 'km_dest': KM_ACUM[14], 'dist': 2.0 + abs(KM_ACUM[18]-KM_ACUM[14]), 'motriz_num': t['tren'], 'origen_txt': 'Sargento Aldea', 'destino_txt': 'Taller / Cochera', 'servicio_previo': t['num_servicio'], 'servicio_siguiente': '—'})

    return vacios
