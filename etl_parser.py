import pandas as pd
import numpy as np
import re
import unicodedata
from io import BytesIO
from datetime import datetime, date, timedelta
from config import *

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

def clean_pax_number(x):
    if pd.isna(x): return 0
    s = re.sub(r'[^\d]', '', str(x).split('.')[0])
    return int(s) if s else 0

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
    for i, est in enumerate(ESTACIONES):
        ne = re.sub(r'[^a-z0-9]','', est.lower().replace('á','a').replace('é','e').replace('í','i').replace('ó','o').replace('ú','u').replace('ñ','n'))
        if ne in cu: return i
    return None

def calc_tren_km_real_general(row):
    k_s, k_e = min(row['km_orig'], row['km_dest']), max(row['km_orig'], row['km_dest'])
    return abs(k_e-k_s) * (2.0 if row.get('doble',False) else 1.0)

def procesar_thdr(data, fname, via_param=1):
    try:
        eng = "openpyxl" if fname.lower().endswith('.xlsx') else "xlrd"
        raw = pd.read_excel(BytesIO(data), header=None, engine=eng, dtype=str)
        
        if raw is None or raw.empty: return pd.DataFrame(), f"Archivo vacío: {fname}"
        fecha_str = extraer_fecha_segura(raw, fname)
        
        header_idx = 1
        for i in range(min(15, len(raw))):
            line = ' '.join([str(x).upper() for x in raw.iloc[i].values if pd.notna(x)])
            if 'SALIDA' in line or 'HORA' in line or 'LLEGADA' in line:
                header_idx = i; break
        
        df = raw.iloc[header_idx+1:].copy().reset_index(drop=True)
        df.columns = [f"Col_{i}" for i in range(df.shape[1])]
        df = df.dropna(how='all')
        
        df['num_servicio'] = df.iloc[:, 0].apply(clean_id)
        times = []
        for i in range(5, df.shape[1]):
            col_data = df.iloc[:, i].apply(parse_time_to_mins)
            if col_data.notna().any(): times.append(col_data)
        
        df['t_ini'] = pd.concat(times, axis=1).min(axis=1)
        df['t_fin'] = pd.concat(times, axis=1).max(axis=1)
        df['Via'], df['Fecha_str'] = via_param, fecha_str
        
        def get_tipo(srv):
            try:
                n = int(srv)
                if n >= 36 and n < 200: return "SFE"
                if 28 <= n <= 35: return "XT-M"
                return "XT-100"
            except: return "XT-100"
        
        df['tipo_tren'] = df['num_servicio'].apply(get_tipo)
        df['doble'] = False 
        df['km_orig'] = 0.0 if via_param == 1 else KM_TOTAL
        df['km_dest'] = KM_TOTAL if via_param == 1 else 0.0
        
        def _get_nodos(r):
            n = []
            for i in range(5, df.shape[1]):
                val = parse_time_to_mins(r[f"Col_{i}"])
                if pd.notna(val) and val > 0: n.append((val, KM_ACUM[min(i-5, 20)]))
            return sorted(n, key=lambda x: x[0]) if len(n) > 1 else None
            
        df['nodos'] = df.apply(_get_nodos, axis=1)
        df['_id'] = df['Fecha_str'] + "_" + df['num_servicio'] + "_" + df['t_ini'].astype(str)
        return df.dropna(subset=['t_ini']), "ok"
    except Exception as e: return pd.DataFrame(), str(e)

def cargar_pax(data, fname, via_param=1):
    """MAPEADO RÍGIDO NATIVO EXCEL: FILA 11. COL 3 FECHA, COL 4 HORA, COL 29 TOTAL A BORDO"""
    try:
        eng = "xlrd" if fname.lower().endswith(".xls") else "openpyxl"
        df_raw = pd.read_excel(BytesIO(data), header=None, engine=eng, dtype=str)
        
        if df_raw.shape[1] < 10: return pd.DataFrame()
            
        data_rows = df_raw.iloc[10:].copy().reset_index(drop=True)
        
        df = pd.DataFrame()
        df['Nro_THDR_raw'] = data_rows.iloc[:, 0].values
        df['Tren_Clean']   = data_rows.iloc[:, 2].apply(clean_id).values
        df['Fecha_s']      = data_rows.iloc[:, 3].apply(parse_excel_date).values
        df['Hora Origen']  = data_rows.iloc[:, 4].values
        
        # Mapeo de estaciones según la cabecera real (Fila 10 / Índice 9)
        header_row = df_raw.iloc[9].fillna('').astype(str).str.upper().values
        is_pue_start = 'PUE' in header_row[8] or 'PUERTO' in header_row[8]
        orden_est = PAX_COLS if is_pue_start else list(reversed(PAX_COLS))
        
        for i, st in enumerate(orden_est):
            if 8 + i < data_rows.shape[1]:
                df[st] = data_rows.iloc[:, 8 + i].apply(clean_pax_number).values
            else:
                df[st] = 0
            
        # DATO SAGRADO: TOTAL A BORDO (Columna 29 / AD)
        if data_rows.shape[1] > 29:
            df['CargaMax'] = data_rows.iloc[:, 29].apply(clean_pax_number).values
        else:
            df['CargaMax'] = 0
        
        df['t_ini_p'] = df['Hora Origen'].apply(parse_time_to_mins)
        df['Via'] = via_param
        
        # Si la fecha oficial está vacía en algunas filas, usar la general del archivo
        fecha_fallback = extraer_fecha_segura(df_raw, fname)
        df['Fecha_s'] = df['Fecha_s'].fillna(fecha_fallback).replace('None', fecha_fallback)
        
        return df.dropna(subset=['t_ini_p', 'Fecha_s'])
    except Exception as e: return pd.DataFrame()

def match_pax(row, df_pax):
    EMPTY = ({c: 0 for c in PAX_COLS}, 0, '--:--:--', 'No Detectado', -1)
    if df_pax.empty: return EMPTY
    
    tren_row = clean_id(row.get('num_servicio', ''))
    fecha_row = row.get('Fecha_str', '')
    t_i = row.get('t_ini')
    
    # Prioridad 1: Match Exacto por ID Tren y Fecha
    match = df_pax[(df_pax['Tren_Clean'] == tren_row) & (df_pax['Fecha_s'] == fecha_row)]
    
    # Prioridad 2: Si el THDR tiene fecha rota (2026-01-01), buscar el tren en todo el archivo
    if match.empty and fecha_row == "2026-01-01":
        match = df_pax[df_pax['Tren_Clean'] == tren_row]
            
    if not match.empty:
        match = match.copy()
        match['diff'] = abs(match['t_ini_p'] - t_i)
        best = match.loc[match['diff'].idxmin()]
        return {c: int(best[c]) for c in PAX_COLS}, int(best['CargaMax']), mins_to_time_str(best['t_ini_p']), str(best.get('Nro_THDR_raw', '')), best.name
    
    return EMPTY

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

def get_vacios_dia(df): return []
def calcular_dwell(df1, df2): return df1, df2

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
