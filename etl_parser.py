import pandas as pd
import numpy as np
import re
import unicodedata
from io import BytesIO
from datetime import datetime, date, timedelta

# =============================================================================
# CONSTANTES BLINDADAS (Protección contra fallos de caché en Streamlit Cloud)
# =============================================================================
ESTACIONES_SAFE = ['Puerto','Bellavista','Francia','Baron','Portales','Recreo','Miramar','Vina del Mar','Hospital','Chorrillos','El Salto','Valencia','Quilpue','El Sol','El Belloto','Las Americas','La Concepcion','Villa Alemana','Sargento Aldea','Penablanca','Limache']
EC_SAFE = ['PU','BE','FR','BA','PO','RE','MI','VM','HO','CH','ES','VAL','QU','SO','EB','AM','CO','VL','SA','PE','LI']
PAX_COLS_SAFE = ['PUE','BEL','FRA','BAR','POR','REC','MIR','VIN','HOS','CHO','SLT','VAL','QUI','SOL','BTO','AME','CON','VAM','SGA','PEN','LIM']
KM_ACUM_SAFE = [0.0, 0.7, 1.4, 2.2, 3.9, 6.0, 7.4, 8.3, 9.2, 10.2, 11.7, 19.1, 21.4, 23.3, 25.3, 26.4, 27.6, 28.5, 29.1, 30.4, 43.13]
KM_TOTAL_SAFE = 43.13
N_EST_SAFE = 21
FERIADOS_SAFE = ['2026-01-01', '2026-04-03', '2026-04-04', '2026-05-01', '2026-05-21', '2026-06-21', '2026-07-16', '2026-08-15', '2026-09-18', '2026-09-19', '2026-10-12', '2026-10-31', '2026-12-08', '2026-12-25']

# =============================================================================
# 1. UTILIDADES Y PARSEOS BASICOS
# =============================================================================

def mins_to_time_str(mins):
    if pd.isna(mins) or np.isinf(mins):
        return '--:--:--'
    try:
        m_val = float(mins)
        h = int((m_val // 60) % 24)
        m = int(m_val % 60)
        s = int(round((m_val * 60) % 60))
        if s == 60:
            s = 0
            m = m + 1
        if m == 60:
            m = 0
            h = h + 1
        return f"{h:02d}:{m:02d}:{s:02d}"
    except:
        return '--:--:--'

def parse_time_to_mins(val):
    if pd.isna(val):
        return None
    sv = str(val).strip().lower()
    if sv in ('', 'nan', 'none', 'nat'):
        return None
    if ' ' in sv:
        sv = sv.split(' ')[-1]
    m = re.search(r'(\d{1,2}):(\d{2})(?::(\d{2}))?', sv)
    if m:
        result = int(m.group(1)) * 60.0 + int(m.group(2))
        if m.group(3):
            result += int(m.group(3)) / 60.0
        return result
    try:
        f = float(sv)
        if f < 1.0:
            return f * 1440.0
        elif f < 2400.0:
            return int(f // 100) * 60.0 + (f % 100)
        else:
            return None
    except:
        return None

def parse_excel_date(val):
    if pd.isna(val):
        return None
    if isinstance(val, (datetime, pd.Timestamp)):
        return val.strftime('%Y-%m-%d')
    v_str = str(val).strip()
    v_str_num = re.sub(r'\.0+$', '', v_str).split(' ')[0]
    
    if v_str_num.isdigit():
        num = int(v_str_num)
        if 40000 <= num <= 60000:
            try:
                return (date(1899, 12, 30) + timedelta(days=num)).strftime('%Y-%m-%d')
            except:
                pass
        elif len(v_str_num) == 8:
            y = int(v_str_num[0:4])
            m_val = int(v_str_num[4:6])
            d = int(v_str_num[6:8])
            if y < 2000:
                d = int(v_str_num[0:2])
                m_val = int(v_str_num[2:4])
                y = int(v_str_num[4:8])
            if m_val > 12 and d <= 12:
                d, m_val = m_val, d
            if 1 <= d <= 31 and 1 <= m_val <= 12 and 2000 <= y <= 2100:
                return f"{y:04d}-{m_val:02d}-{d:02d}"
        elif len(v_str_num) == 6:
            d = int(v_str_num[0:2])
            m_val = int(v_str_num[2:4])
            y = int(v_str_num[4:6]) + 2000
            if m_val > 12 and d <= 12:
                d, m_val = m_val, d
            if 1 <= d <= 31 and 1 <= m_val <= 12:
                return f"{y:04d}-{m_val:02d}-{d:02d}"
        elif len(v_str_num) == 5:
            d = int(v_str_num[0:1])
            m_val = int(v_str_num[1:3])
            y = int(v_str_num[3:5]) + 2000
            if m_val > 12 and d <= 12:
                d, m_val = m_val, d
            if 1 <= d <= 31 and 1 <= m_val <= 12:
                return f"{y:04d}-{m_val:02d}-{d:02d}"

    for pat in [r'\b(\d{1,2})[-/](\d{1,2})[-/](\d{2,4})\b', r'\b(\d{4})[-/](\d{1,2})[-/](\d{1,2})\b']:
        m_dt = re.search(pat, v_str)
        if m_dt:
            if len(m_dt.group(1)) == 4:
                y = int(m_dt.group(1))
                m_val = int(m_dt.group(2))
                d = int(m_dt.group(3))
            else:
                d = int(m_dt.group(1))
                m_val = int(m_dt.group(2))
                y = int(m_dt.group(3))
            if y < 100:
                y += 2000
            if m_val > 12 and d <= 12:
                d, m_val = m_val, d
            if 1 <= d <= 31 and 1 <= m_val <= 12:
                return f"{y:04d}-{m_val:02d}-{d:02d}"
            
    return None

def extraer_fecha_segura(df_raw, fname, is_thdr=False):
    if is_thdr and df_raw is not None and not df_raw.empty:
        try:
            a1_val = str(df_raw.iloc[0, 0]).strip()
            a1_num = re.sub(r'\.0+$', '', a1_val)
            if a1_num.isdigit():
                num = int(a1_num)
                if len(a1_num) == 5:
                    d = int(a1_num[0:1])
                    m_val = int(a1_num[1:3])
                    y = int(a1_num[3:5]) + 2000
                    if 1 <= d <= 31 and 1 <= m_val <= 12:
                        return f"{y:04d}-{m_val:02d}-{d:02d}"
                elif len(a1_num) == 6:
                    d = int(a1_num[0:2])
                    m_val = int(a1_num[2:4])
                    y = int(a1_num[4:6]) + 2000
                    if 1 <= d <= 31 and 1 <= m_val <= 12:
                        return f"{y:04d}-{m_val:02d}-{d:02d}"
                elif 40000 <= num <= 60000:
                    return (date(1899, 12, 30) + timedelta(days=num)).strftime('%Y-%m-%d')
            dt = parse_excel_date(a1_val)
            if dt:
                return dt
        except:
            pass

    s_fname = re.sub(r'\D', '', str(fname))
    for pat in [r'(\d{4})(\d{2})(\d{2})', r'(\d{2})(\d{2})(\d{4})', r'(\d{2})(\d{2})(\d{2})']:
        matches = re.finditer(pat, s_fname)
        for m in matches:
            if len(m.group(0)) == 8:
                if int(m.group(1)) >= 2000:
                    y = int(m.group(1))
                    mon = int(m.group(2))
                    d = int(m.group(3))
                else:
                    d = int(m.group(1))
                    mon = int(m.group(2))
                    y = int(m.group(3))
                if mon > 12 and d <= 12:
                    d, mon = mon, d
                if 1 <= d <= 31 and 1 <= mon <= 12 and 2000 <= y <= 2100:
                    return f"{y:04d}-{mon:02d}-{d:02d}"
            elif len(m.group(0)) == 6:
                d = int(m.group(1))
                mon = int(m.group(2))
                y = int(m.group(3)) + 2000
                if mon > 12 and d <= 12:
                    d, mon = mon, d
                if 1 <= d <= 31 and 1 <= mon <= 12:
                    return f"20{y:02d}-{mon:02d}-{d:02d}"

    if is_thdr:
        return "2026-01-01"

    if df_raw is not None:
        for i in range(min(50, len(df_raw))):
            for val in df_raw.iloc[i].values:
                if pd.isna(val):
                    continue
                v_str = str(val).strip()
                for pat in [r'\b(\d{1,2})[-/](\d{1,2})[-/](\d{2,4})\b', r'\b(\d{4})[-/](\d{1,2})[-/](\d{1,2})\b']:
                    m_dt = re.search(pat, v_str)
                    if m_dt:
                        if len(m_dt.group(1)) == 4:
                            y = int(m_dt.group(1))
                            m_val = int(m_dt.group(2))
                            d = int(m_dt.group(3))
                        else:
                            d = int(m_dt.group(1))
                            m_val = int(m_dt.group(2))
                            y = int(m_dt.group(3))
                        if y < 100:
                            y += 2000
                        if m_val > 12 and d <= 12:
                            d, m_val = m_val, d
                        if 1 <= d <= 31 and 1 <= m_val <= 12:
                            return f"{y:04d}-{m_val:02d}-{d:02d}"
                        
    return "2026-01-01"

def clean_id(x):
    try:
        nums = re.findall(r'\d+', str(x).strip().lower().replace(".0", ""))
        if nums:
            return str(int(nums[0]))
        else:
            return str(x).strip().upper()
    except:
        return str(x).strip().upper()

def clean_primary_key(x):
    if pd.isna(x):
        return ''
    s = re.sub(r'[^A-Z0-9]', '', str(x).strip().upper().replace('.0', ''))
    if s in ['NAN', '']:
        return ''
    return s.lstrip('0')

def clasificar_dia(d_str):
    try:
        d = datetime.strptime(d_str, '%Y-%m-%d')
        if d_str in FERIADOS_SAFE or d.weekday() == 6:
            return 'Domingo/Festivo'
        if d.weekday() == 5:
            return 'Sabado'
        return 'Laboral'
    except:
        return 'Laboral'

def _col_to_est_idx(col):
    cu = re.sub(r'[^a-z0-9]', '', str(col).lower())
    for i, est in enumerate(ESTACIONES_SAFE):
        ne = re.sub(r'[^a-z0-9]', '', est.lower())
        if ne in cu:
            return i
    return None

def calc_tren_km_real_general(row):
    k_s = min(row['km_orig'], row['km_dest'])
    k_e = max(row['km_orig'], row['km_dest'])
    man = row.get('maniobra')
    if man in ['CORTE_BTO', 'ACOPLE_BTO', 'CORTE_PU_SA_BTO']:
        km_man = KM_ACUM_SAFE[14]
        if k_s <= km_man <= k_e:
            return abs(km_man - k_s) * 2.0 + abs(k_e - km_man) * 1.0
    elif man in ['CORTE_SA', 'ACOPLE_SA']:
        km_man = KM_ACUM_SAFE[18]
        if k_s <= km_man <= k_e:
            return abs(km_man - k_s) * 2.0 + abs(k_e - km_man) * 1.0
    factor = 2.0 if row.get('doble', False) else 1.0
    return abs(k_e - k_s) * factor

def get_pax_at_km_nativo(pax_d, km_pos, via, pax_max_fallback=0):
    if not pax_d or not isinstance(pax_d, dict):
        return pax_max_fallback
    if sum(pax_d.values()) == 0 and pax_max_fallback > 0:
        return pax_max_fallback
    pax_val = 0
    if via == 1:
        for i in range(N_EST_SAFE):
            if km_pos >= KM_ACUM_SAFE[i]:
                val = pax_d.get(PAX_COLS_SAFE[i])
                if val is not None:
                    pax_val = val
            else:
                break
    else:
        for i in range(N_EST_SAFE - 1, -1, -1):
            if km_pos <= KM_ACUM_SAFE[i]:
                val = pax_d.get(PAX_COLS_SAFE[i])
                if val is not None:
                    pax_val = val
            else:
                break
    return int(pax_val)

get_pax_at_km = get_pax_at_km_nativo

# =============================================================================
# 2. PROCESAMIENTO THDR (ESCANER HIBRIDO A PRUEBA DE FALLOS)
# =============================================================================
def procesar_thdr(data, fname, via_param=1):
    est_llegada = {}
    est_salida = {}
    
    try:
        ext = fname.lower()
        if ext.endswith('.csv'):
            try:
                raw = pd.read_csv(BytesIO(data), header=None, sep=',', encoding='utf-8', dtype=str)
            except:
                raw = pd.read_csv(BytesIO(data), header=None, sep=';', encoding='latin-1', dtype=str)
        else:
            try:
                eng = "xlrd" if ext.endswith(".xls") else "openpyxl"
                raw = pd.read_excel(BytesIO(data), header=None, engine=eng, dtype=str)
            except Exception as e:
                if any(x in str(e).lower() for x in ["html", "xml", "format", "token", "unsupported"]):
                    try:
                        dfs = pd.read_html(BytesIO(data))
                        raw = dfs[0].astype(str)
                    except Exception as html_e:
                        return pd.DataFrame(), f"Fallo al leer XLS como HTML: {str(html_e)}"
                else:
                    return pd.DataFrame(), f"Error al abrir Excel: {str(e)}"

        if raw is None or raw.empty or raw.shape[0] < 5:
            return pd.DataFrame(), "El archivo esta vacio o no tiene suficientes filas."
        
        fecha_str = extraer_fecha_segura(raw, fname, is_thdr=True)

        header_idx = -1
        for i in range(min(20, len(raw))):
            row_str = ' '.join([str(x).upper() for x in raw.iloc[i].values if pd.notna(x)])
            if ('MOTRIZ' in row_str or 'MAQUINISTA' in row_str) and any(k in row_str for k in ['VIAJE', 'TREN', 'SERVICIO']):
                header_idx = i
                break
        
        if header_idx == -1:
            header_idx = 1
            
        r0 = raw.iloc[header_idx - 1].copy() if header_idx > 0 else raw.iloc[0].copy()
        r0.iloc[0] = np.nan
        h1 = r0.ffill().astype(str)
        h2 = raw.iloc[header_idx].fillna('').astype(str)
        
        cols = []
        for s, t in zip(h1, h2):
            s_val = str(s).strip().upper()
            t_val = str(t).strip().upper()
            if s_val in ['NAN', 'NONE'] or not s_val:
                cols.append(t_val)
            elif t_val and t_val not in ['NAN', 'NONE']:
                cols.append(f"{s_val}_{t_val}")
            else:
                cols.append(s_val)
            
        df = raw.iloc[header_idx + 1:].copy().reset_index(drop=True)
        num_cols = df.shape[1]
        new_cols = []
        for i in range(num_cols):
            if i < len(cols):
                new_cols.append(f"Col_{i}_{cols[i]}")
            else:
                new_cols.append(f"Col_{i}_EXTRA")
        df.columns = new_cols
        df = df.dropna(how='all').reset_index(drop=True)

        est_cols = {}
        
        for i, col in enumerate(df.columns):
            col_str = str(col).upper()
            if any(k in col_str for k in ['LLEGADA', 'SALIDA', 'HORA']) or any(est[:3].upper() in col_str for est in ESTACIONES_SAFE):
                parsed_times = df[col].apply(parse_time_to_mins)
                if parsed_times.notna().any():
                    df[f"T_{i}"] = parsed_times
                    idx_est = _col_to_est_idx(col_str)
                    if idx_est is not None:
                        est_cols[f"T_{i}"] = idx_est
                        if 'SALIDA' in col_str:
                            est_salida[idx_est] = f"T_{i}"
                        elif 'LLEGADA' in col_str:
                            est_llegada[idx_est] = f"T_{i}"
                            
        if len(est_cols) < 5:
            est_cols = {}
            est_llegada = {}
            est_salida = {}
            col_start_time = 5
            for c in range(2, min(12, df.shape[1])):
                muestras = df.iloc[:, c].dropna().head(10).apply(parse_time_to_mins)
                if muestras.notna().any():
                    col_start_time = c
                    break
                    
            for i in range(col_start_time, min(col_start_time + N_EST_SAFE, df.shape[1])):
                col_name = df.columns[i]
                parsed_times = df[col_name].apply(parse_time_to_mins)
                if parsed_times.notna().any():
                    df[f"T_{i}"] = parsed_times
                    if via_param == 1:
                        idx_est = i - col_start_time
                    else:
                        idx_est = (N_EST_SAFE - 1) - (i - col_start_time)
                    est_cols[f"T_{i}"] = idx_est

        if not est_cols:
            return pd.DataFrame(), "Formato irlegible. No se encontraron tiempos validos."

        def _safe_get(r, col):
            val = r.get(col)
            if pd.notna(val):
                return val
            return np.nan

        df['t_ini'] = df.apply(lambda row: min([_safe_get(row, c) for c in est_cols.keys() if pd.notna(_safe_get(row, c))] or [np.nan]), axis=1)
        df['t_fin'] = df.apply(lambda row: max([_safe_get(row, c) for c in est_cols.keys() if pd.notna(_safe_get(row, c))] or [np.nan]), axis=1)
        
        def _construir_nodos(row):
            est_presentes = sorted(set(list(est_llegada.keys()) + list(est_salida.keys())))
            if via_param == 2:
                est_presentes = list(reversed(est_presentes))

            def get_t(col):
                if col is None:
                    return None
                v = row.get(col)
                if pd.notna(v) and float(v) > 0:
                    return float(v)
                return None

            nodos = []
            for e_idx in est_presentes:
                t_lleg = get_t(est_llegada.get(e_idx))
                t_sal = get_t(est_salida.get(e_idx))
                km = KM_ACUM_SAFE[e_idx]
                if not any([t_lleg, t_sal]):
                    continue
                if not nodos:
                    if t_sal:
                        nodos.append((t_sal, km))
                    elif t_lleg:
                        nodos.append((t_lleg, km))
                else:
                    if t_lleg:
                        nodos.append((t_lleg, km))
                    if t_sal:
                        nodos.append((t_sal, km))

            if len(nodos) >= 2:
                nodos.sort(key=lambda x: x[0])
                return nodos
            return None

        df['nodos'] = df.apply(_construir_nodos, axis=1)

        c_m1 = next((c for c in df.columns if 'MOTRIZ' in str(c).upper() and '1' in str(c).upper()), None)
        c_m2 = next((c for c in df.columns if 'MOTRIZ' in str(c).upper() and '2' in str(c).upper()), None)
        tren_col = next((c for c in df.columns if str(c).strip().upper() in ('TREN', 'SERVICIO', 'MOTRIZ')), df.columns[0])

        def _get_fleet_info(r):
            def extract_n(c_name):
                if c_name and pd.notna(r.get(c_name)):
                    val = str(r.get(c_name)).strip()
                    if val.lower() not in ('nan', '', '0', '0.0'):
                        m = re.search(r'(\d+)', val)
                        if m:
                            return int(m.group(1))
                return None
            n1 = extract_n(c_m1)
            n2 = extract_n(c_m2)
            n_tren = extract_n(tren_col)
            
            n_eval = n1 or n2 or n_tren
            if n_eval is not None:
                if 1 <= n_eval <= 27:
                    tipo = "XT-100"
                elif 28 <= n_eval <= 35:
                    tipo = "XT-M"
                elif 410 <= n_eval <= 414:
                    tipo = "SFE"
                else:
                    tipo = "XT-100"
            else:
                tipo = "XT-100"
                
            if n1 and n2:
                motriz_str = f"{n1}+{n2}"
            else:
                motriz_str = str(n_eval or "")
            return pd.Series([motriz_str, tipo])
            
        df[['motriz_num', 'tipo_tren']] = df.apply(_get_fleet_info, axis=1)

        if c_m2:
            df['doble'] = df[c_m2].apply(lambda x: pd.notna(x) and str(x).strip() not in ('0', '0.0', '', 'nan', 'none'))
        else:
            df['doble'] = df['motriz_num'].astype(str).str.contains(r'\+', regex=True)
            
        df['Via'] = via_param
        df['Fecha_str'] = fecha_str

        def _get_real_orig_dest(row):
            valid_est = []
            for col, e_idx in est_cols.items():
                val = _safe_get(row, col)
                if pd.notna(val) and val > 0:
                    valid_est.append(e_idx)
            if not valid_est:
                if via_param == 1:
                    return pd.Series([0.0, KM_TOTAL_SAFE])
                else:
                    return pd.Series([KM_TOTAL_SAFE, 0.0])
            
            if via_param == 1:
                return pd.Series([KM_ACUM_SAFE[min(valid_est)], KM_ACUM_SAFE[max(valid_est)]])
            else:
                return pd.Series([KM_ACUM_SAFE[max(valid_est)], KM_ACUM_SAFE[min(valid_est)]])

        df[['km_orig', 'km_dest']] = df.apply(_get_real_orig_dest, axis=1)
        df = df.dropna(subset=['t_ini'])
        
        df['km_viaje'] = abs(df['km_dest'] - df['km_orig'])
        
        def _get_svc_type(r):
            try:
                idx_orig = KM_ACUM_SAFE.index(r['km_orig'])
                idx_dest = KM_ACUM_SAFE.index(r['km_dest'])
                return f"{EC_SAFE[idx_orig]}-{EC_SAFE[idx_dest]}"
            except:
                return "PU-LI"
                
        df['svc_type'] = df.apply(_get_svc_type, axis=1)

        viaje_col_idx = next((c for c in df.columns if 'VIAJE' in str(c).upper() or 'NRO' in str(c).upper() or 'N°' in str(c).upper()), None)
        if viaje_col_idx:
            df['nro_viaje'] = df[viaje_col_idx].apply(clean_primary_key)
        else:
            df['nro_viaje'] = ''

        df['num_servicio'] = df[tren_col].apply(clean_primary_key)
        if df['num_servicio'].eq('').all():
            df['num_servicio'] = df['nro_viaje']

        df['_id'] = df['Fecha_str'] + "_V" + df['Via'].astype(str) + "_" + df['num_servicio'] + "_" + df['t_ini'].astype(str)
        df['t_fin'] = df['t_fin'].fillna(df['t_ini'] + df['km_viaje'] / 35.0 * 60.0)

        def calc_dwell_dynamic(row):
            try:
                idx_orig = int(np.argmin([abs(row['km_orig'] - k) for k in KM_ACUM_SAFE]))
                idx_dest = int(np.argmin([abs(row['km_dest'] - k) for k in KM_ACUM_SAFE]))
                n_stops = max(0, abs(idx_dest - idx_orig) - 1)
                return round(n_stops * (8.0 / 19.0), 3)
            except:
                return 8.0
                
        df['dwell_min'] = df.apply(calc_dwell_dynamic, axis=1)
        df['dwell_cabecera_min'] = 0.0
        
        # Asignar flota real según disponibilidad y demanda horaria
        return df, "ok"
    except Exception as e:
        import traceback
        return pd.DataFrame(), f"Fallo Critico ETL: {str(e)}"

# =============================================================================
# 3. LECTURA Y CRUCE DE PASAJEROS
# =============================================================================
def cargar_pax(data, fname, via_param=1):
    try:
        eng = "xlrd" if fname.lower().endswith(".xls") else "openpyxl"
        if fname.lower().endswith('.csv'):
            try:
                full = pd.read_csv(BytesIO(data), header=None, sep=',', encoding='utf-8', dtype=str)
            except:
                full = pd.read_csv(BytesIO(data), header=None, sep=';', encoding='latin-1', dtype=str)
        else:
            full = pd.read_excel(BytesIO(data), header=None, engine=eng, dtype=str)

        if full is None or full.empty or len(full) <= 10:
            return pd.DataFrame()
        
        header_idx = -1
        for r in range(min(20, len(full))):
            row_str = ' '.join(full.iloc[r].fillna('').astype(str).str.upper())
            if ('PUE' in row_str or 'PUERTO' in row_str) and ('LIM' in row_str or 'LIMACHE' in row_str):
                header_idx = r
                break
            if 'TOTAL' in row_str:
                header_idx = r
                break
        if header_idx == -1:
            header_idx = 9

        col_mapping = {}
        for c_idx in range(full.shape[1]):
            vals = [str(full.iloc[r, c_idx]).strip().upper() for r in range(max(0, header_idx-3), header_idx+1)]
            combo = " ".join(vals)
            combo_norm = unicodedata.normalize('NFD', combo).encode('ascii', 'ignore').decode().replace('.', '').replace(':', '')

            if 'HORA' in combo_norm and 'ORIG' in combo_norm:
                col_mapping[c_idx] = 'Hora Origen'
            elif 'THDR' in combo_norm and 'TREN' not in combo_norm:
                col_mapping[c_idx] = 'Nro_THDR_raw'
            elif 'TREN' in combo_norm or 'SERVICIO' in combo_norm:
                col_mapping[c_idx] = 'Tren'
            elif 'FECHA' in combo_norm and 'Fecha_s_raw' not in col_mapping.values():
                col_mapping[c_idx] = 'Fecha_s_raw'
            elif 'CargaMax' not in col_mapping.values() and any(w in combo_norm for w in ['TOTAL', 'BORDO', 'CARGA', 'PASAJERO']):
                col_mapping[c_idx] = 'CargaMax'
            else:
                for k in PAX_COLS_SAFE:
                    if k == vals[-1] or k == vals[-2]:
                        col_mapping[c_idx] = k
                        break

        data_rows = full.iloc[header_idx + 1:].copy()
        df = pd.DataFrame()
        for c_idx, col_name in col_mapping.items():
            if isinstance(c_idx, int) and c_idx < full.shape[1]:
                df[col_name] = data_rows.iloc[:, c_idx].values
                
        fecha_global = extraer_fecha_segura(full, fname, is_thdr=False)
        
        if 'Fecha_s_raw' in df.columns:
            df['Fecha_s'] = df['Fecha_s_raw'].apply(parse_excel_date)
            df['Fecha_s'] = df['Fecha_s'].ffill().fillna(fecha_global)
        else:
            df['Fecha_s'] = fecha_global
                
        for col in ['Hora Origen', 'Nro_THDR_raw', 'Tren', 'CargaMax']:
            if col not in df.columns:
                df[col] = ''
        
        for c in PAX_COLS_SAFE:
            if c not in df.columns:
                df[c] = '0'

        df['Nro_THDR'] = df['Nro_THDR_raw'].apply(clean_primary_key)
        df['Tren_Clean'] = df['Tren'].apply(clean_id)
        df['t_ini_p'] = df['Hora Origen'].apply(parse_time_to_mins)
        df['Via'] = via_param
        df = df.dropna(subset=['t_ini_p'])
        
        if df.empty:
            return pd.DataFrame()
        
        for c in PAX_COLS_SAFE + ['CargaMax']:
            df[c] = df[c].apply(lambda x: int(re.sub(r'[^\d]', '', str(x).replace('.', '').replace(',', '')) or 0))
        return df
    except Exception as e:
        return pd.DataFrame()

def match_pax(row, df_pax):
    EMPTY = ({c: 0 for c in PAX_COLS_SAFE}, 0, '--:--:--', 'No Detectado', -1)
    if df_pax.empty:
        return EMPTY
    
    def _to_int(v):
        try:
            if pd.notna(v):
                return int(float(v))
            return 0
        except:
            return 0
        
    t_i = row.get('t_ini')
    via = row.get('via_op', row.get('Via', 1))
    num_servicio = clean_id(row.get('num_servicio', ''))
    thdr_date = str(row.get('Fecha_str', '')).strip()
    
    sub = df_pax[df_pax['Via'] == via].copy()
    if sub.empty:
        return EMPTY
    
    if 'Fecha_s' in sub.columns and thdr_date and thdr_date != '2026-01-01':
        sub_date = sub[sub['Fecha_s'].astype(str).str.strip() == thdr_date]
        if not sub_date.empty:
            sub = sub_date

    if num_servicio != '' and 'Tren_Clean' in sub.columns:
        m = sub[sub['Tren_Clean'] == num_servicio]
        if not m.empty:
            m = m.copy()
            m['diff'] = m['t_ini_p'].apply(lambda x: min(abs(float(x) - float(t_i)), 1440 - abs(float(x) - float(t_i))) if pd.notna(x) and pd.notna(t_i) else 9999)
            best_match = m.loc[m['diff'].idxmin()]
            pax_d = {c: _to_int(best_match.get(c, 0)) for c in PAX_COLS_SAFE}
            pax_abordo = _to_int(best_match.get('CargaMax', 0))
            hora = mins_to_time_str(best_match.get('t_ini_p'))
            nro = str(best_match.get('Nro_THDR_raw', best_match.get('Tren', '')))
            return pax_d, pax_abordo, hora, nro, best_match.name

    if pd.notna(t_i):
        sub['diff'] = sub['t_ini_p'].apply(lambda x: min(abs(float(x) - float(t_i)), 1440 - abs(float(x) - float(t_i))) if pd.notna(x) and pd.notna(t_i) else 9999)
        if not sub.empty:
            idx_min = sub['diff'].idxmin()
            best_match = sub.loc[idx_min]
            if best_match['diff'] <= 60:
                pax_d = {c: _to_int(best_match.get(c, 0)) for c in PAX_COLS_SAFE}
                pax_abordo = _to_int(best_match.get('CargaMax', 0))
                hora = mins_to_time_str(best_match.get('t_ini_p'))
                nro = str(best_match.get('Nro_THDR_raw', best_match.get('Tren', '')))
                return pax_d, pax_abordo, hora, nro, best_match.name

    return EMPTY

# =============================================================================
# 4. FUNCIONES AUXILIARES
# =============================================================================
def calcular_dwell(df1, df2):
    if df1.empty or df2.empty:
        return df1, df2
    if 'num_servicio' not in df1.columns or 'num_servicio' not in df2.columns:
        return df1, df2
    for fecha in df1['Fecha_str'].unique():
        d1 = df1[df1['Fecha_str'] == fecha]
        d2 = df2[df2['Fecha_str'] == fecha]
        if d2.empty:
            continue
        for idx1, r1 in d1.iterrows():
            s = r1.get('num_servicio')
            if pd.isna(s) or s == '':
                continue
            m = d2[(d2['num_servicio'] == s) & (d2['t_ini'] > r1['t_fin'])]
            if not m.empty:
                dw = m['t_ini'].min() - r1['t_fin']
                if 0 < dw < 60:
                    df2.at[m['t_ini'].idxmin(), 'dwell_cabecera_min'] = round(dw, 1)
        for idx2, r2 in d2.iterrows():
            s = r2.get('num_servicio')
            if pd.isna(s) or s == '':
                continue
            m = d1[(d1['num_servicio'] == s) & (d1['t_ini'] > r2['t_fin'])]
            if not m.empty:
                dw = m['t_ini'].min() - r2['t_fin']
                if 0 < dw < 60:
                    df1.at[m['t_ini'].idxmin(), 'dwell_cabecera_min'] = round(dw, 1)
    return df1, df2

def get_vacios_dia(df):
    return []

# =============================================================================
# 4.1 NUEVA FUNCIÓN cargar_prevenciones (ROBUSTA)
# =============================================================================
def cargar_prevenciones(data, fname):
    """
    Carga prevenciones de velocidad desde CSV o Excel.
    Formatos aceptados:
    - CSV con columnas: km_inicio, km_fin, velocidad, via
    - Excel con las mismas columnas.
    Si no se reconocen los nombres, se asume que las primeras 4 columnas son:
    [km_inicio, km_fin, velocidad, via]
    """
    prevenciones = []
    try:
        # Leer archivo
        if fname.lower().endswith('.csv'):
            try:
                raw = pd.read_csv(BytesIO(data), sep=',', encoding='utf-8')
            except:
                raw = pd.read_csv(BytesIO(data), sep=';', encoding='latin-1')
        else:
            try:
                raw = pd.read_excel(BytesIO(data), engine='openpyxl')
            except:
                raw = pd.read_excel(BytesIO(data), engine='xlrd')

        if raw.empty:
            return []

        # Normalizar nombres de columnas
        cols_norm = {}
        for col in raw.columns:
            nombre = str(col).strip().lower()
            nombre = ''.join(c for c in unicodedata.normalize('NFD', nombre) if unicodedata.category(c) != 'Mn')
            nombre = re.sub(r'\s+', '_', nombre)
            cols_norm[nombre] = col

        km_ini_col = None
        km_fin_col = None
        vel_col = None
        via_col = None

        for norm, orig in cols_norm.items():
            if 'km_inicio' in norm or 'km_ini' in norm or 'inicio' in norm or 'desde' in norm:
                if km_ini_col is None:
                    km_ini_col = orig
            if 'km_fin' in norm or 'km_final' in norm or 'fin' in norm or 'hasta' in norm:
                if km_fin_col is None:
                    km_fin_col = orig
            if 'velocidad' in norm or 'vel' in norm or 'km/h' in norm:
                if vel_col is None:
                    vel_col = orig
            if 'via' in norm or 'sentido' in norm:
                if via_col is None:
                    via_col = orig

        # Fallback: primeras 4 columnas
        cols = list(raw.columns)
        if km_ini_col is None and len(cols) > 0:
            km_ini_col = cols[0]
        if km_fin_col is None and len(cols) > 1:
            km_fin_col = cols[1]
        if vel_col is None and len(cols) > 2:
            vel_col = cols[2]
        if via_col is None and len(cols) > 3:
            via_col = cols[3]

        for _, row in raw.iterrows():
            try:
                km1 = float(str(row[km_ini_col]).replace(',', '.'))
                km2 = float(str(row[km_fin_col]).replace(',', '.'))
                vel_str = str(row[vel_col])
                vel = float(re.search(r'(\d+\.?\d*)', vel_str).group())
                via = int(row[via_col])
                prevenciones.append({
                    'km_min': min(km1, km2),
                    'km_max': max(km1, km2),
                    'v_kmh': vel,
                    'via': via
                })
            except:
                continue
    except Exception:
        return []

    return prevenciones

# =============================================================================
# 5. PARSEO DE PLANILLA MAESTRA (mantenido sin cambios)
# =============================================================================

# =============================================================================
# ASIGNACIÓN AUTOMÁTICA DE FLOTA
# =============================================================================
def asignar_flota_planilla(df):
    """
    Asigna tipo_tren a cada viaje según la flota real de MERVAL.
    Si el viaje ya tiene motriz_num desde la planilla maestra, respeta
    el tipo_tren derivado del número de motriz y no lo sobreescribe.
    Solo aplica asignación automática a viajes sin motriz asignada.

    Flota MERVAL:
      - 1  SFE   → servicio 6xx completo de mayor extensión (simple)
      - 8  XT-M  → servicios 4xx cortos (SA-PU) con doble, o 6xx simples
      - 27 XT-100 → resto
    """
    import numpy as np

    SFE_MAX       = 1
    XTM_UNIDADES  = 8   # unidades individuales (doble usa 2)

    df = df.copy()

    # Si viene motriz_num de la planilla, tipo_tren ya está derivado — respetar
    tiene_motriz = (
        'motriz_num' in df.columns and
        df['motriz_num'].notna().any() and
        df['motriz_num'].astype(str).str.strip().ne('').any()
    )
    if tiene_motriz:
        # Solo resetear los que no tienen motriz asignada
        sin_motriz = df['motriz_num'].isna() | df['motriz_num'].astype(str).str.strip().eq('')
        df.loc[sin_motriz, 'tipo_tren'] = 'XT-100'
        # Los que tienen motriz ya traen tipo_tren correcto — no tocar
        return df

    # Sin datos de motriz — asignación automática completa
    df['tipo_tren'] = 'XT-100'  # default

    # Agrupar por servicio y calcular características del carrusel
    servicios = {}
    for svc, grp in df.groupby('num_servicio'):
        tipos_svc   = list(grp.sort_values('t_ini')['svc_type'])
        es_completo = any(s in ('PU-LI','LI-PU') for s in tipos_svc)
        es_corto    = all(s in ('PU-SA','SA-PU','PU-EB','EB-PU') for s in tipos_svc)
        doble       = bool(grp['doble'].any())
        unidades    = 2 if doble else 1
        servicios[svc] = {
            'idx':         list(grp.index),
            'n':           len(grp),
            't_ini':       grp['t_ini'].min(),
            'es_completo': es_completo,
            'es_corto':    es_corto,
            'doble':       doble,
            'unidades':    unidades,
            'tipos':       tipos_svc,
        }

    asignados = set()  # servicios ya asignados a SFE o XT-M

    # 1. SFE — servicio 6xx completo (PU-LI/LI-PU), simple, más viajes
    candidatos_sfe = sorted(
        [s for s, v in servicios.items()
         if v['es_completo'] and not v['doble']],
        key=lambda s: servicios[s]['n'], reverse=True
    )
    sfe_asignados = 0
    for svc in candidatos_sfe:
        if sfe_asignados >= SFE_MAX:
            break
        for idx in servicios[svc]['idx']:
            df.loc[idx, 'tipo_tren'] = 'SFE'
        asignados.add(svc)
        sfe_asignados += 1

    # 2. XT-M — preferir servicios cortos con doble, luego cortos simples,
    #           luego completos simples de menor cantidad de viajes
    candidatos_xtm = sorted(
        [s for s, v in servicios.items() if s not in asignados],
        key=lambda s: (
            not servicios[s]['es_corto'],      # cortos primero
            not servicios[s]['doble'],          # dobles primero
            -servicios[s]['n'],                 # más viajes primero
            servicios[s]['t_ini']               # más temprano primero
        )
    )
    xtm_disp = XTM_UNIDADES
    for svc in candidatos_xtm:
        v = servicios[svc]
        if xtm_disp < v['unidades']:
            continue
        for idx in v['idx']:
            df.loc[idx, 'tipo_tren'] = 'XT-M'
        asignados.add(svc)
        xtm_disp -= v['unidades']
        if xtm_disp <= 0:
            break

    # 3. XT-100 — ya es el default para todo lo no asignado
    df['pax_est'] = 0
    df['demanda'] = 0

    # Asignar número de motriz usando rostering greedy
    # XT-100: 1-27 | XT-M: 28-35 | SFE: 412
    def _rostering(df_sub, base, max_t):
        """Asigna números de tren físico por disponibilidad temporal."""
        if df_sub.empty: return {}
        df_sub = df_sub.sort_values('t_ini').copy()
        df_sub['_tf'] = df_sub['t_ini'] + abs(df_sub['km_dest']-df_sub['km_orig'])/35*60 + 10
        trenes = {base+i: 0.0 for i in range(max_t)}
        asig = {}
        for idx, row in df_sub.iterrows():
            libres = {t:tl for t,tl in trenes.items() if tl <= row['t_ini']}
            if libres: num = min(libres, key=libres.get)
            else:      num = min(trenes, key=trenes.get)
            asig[idx] = num; trenes[num] = row['_tf']
        return asig

    # ================================================================
    # ROSTERING — num_servicio ES el tren físico en la planilla EFE
    # Belloto y Limache mezclan XT-100 y XT-M según asignación
    # Dobles: motriz_num muestra par "X+Y" con numeración consecutiva
    # ================================================================

    # El num_servicio ya identifica el tren físico que hace V1↔V2
    # Construir motriz_num desde num_servicio, con par para dobles

    # Numeración por base según prefijo del servicio:
    # 2xx → EB-PU (Belloto) → 201-204
    # 4xx → SA-PU/PU-SA (Belloto/Puerto) → 401-...
    # 6xx → LI-PU/PU-LI (Limache) → 601-...
    # SFE → siempre 412

    # Para dobles: el par de números se deriva del num_servicio
    # asignando números consecutivos a los 2 trenes del mismo servicio

    # Contador de números reales por rango
    # Puerto (1-4): servicios 401-499 que inician en Puerto
    # Belloto (5-20 + 412): servicios 201-299 y 401-499 desde Belloto
    # Limache (21-36): servicios 601-699

    def num_a_motriz(num_srv, es_doble, tipo_tren, doble_idx=0):
        """Convierte num_servicio a número de motriz."""
        if tipo_tren == 'SFE':
            return '412'
        n = str(num_srv)
        # Usar directamente el num_servicio como identificador del tren
        # Los dobles muestran el par con numeración interna
        if es_doble:
            return f'{n}+{int(n)+1}'
        return n

    # ================================================================
    # CARRUSEL CERRADO — GREEDY CON POSICIÓN GEOGRÁFICA
    # ================================================================

    import heapq as _hq

    DWELL_POR_TERMINAL = {'PU':15.0, 'LI':15.0, 'SA':8.0, 'EB':8.0}
    ORIGEN  = {'PU-LI':'PU','PU-SA':'PU','PU-EB':'PU',
               'LI-PU':'LI','SA-PU':'SA','EB-PU':'EB'}
    DESTINO = {'PU-LI':'LI','PU-SA':'SA','PU-EB':'EB',
               'LI-PU':'PU','SA-PU':'PU','EB-PU':'PU'}
    T_ENTRE = {
        ('PU','LI'):52,('LI','PU'):51,('PU','SA'):41,('SA','PU'):40,
        ('PU','EB'):34,('EB','PU'):34,('LI','SA'):93,('SA','LI'):93,
        ('LI','EB'):87,('EB','LI'):87,('SA','EB'):75,('EB','SA'):75,
    }

    BASE = {}
    for n in range(1, 5):   BASE[n] = 'PU'
    for n in range(5, 21):  BASE[n] = 'EB'
    BASE[412] = 'EB'
    for n in range(21, 37): BASE[n] = 'LI'

    estado = {n: {'pos': BASE.get(n,'PU'), 't_libre': 0.0}
              for n in list(range(1, 37)) + [412]}
    for n in range(17, 21): estado[n]['pos'] = 'EB'

    pool_sfe   = [412]
    pool_xtm   = list(range(28, 36))
    pool_xt100 = list(range(1,  28))

    # Limitar pool al mínimo necesario según viajes del día
    # Estimación: mínimo trenes = pico de viajes simultáneos + margen 20%
    import math as _math
    n_xt100 = len([r for _,r in df.iterrows() if r['tipo_tren']=='XT-100'])
    n_xtm   = len([r for _,r in df.iterrows() if r['tipo_tren']=='XT-M'])
    # Pico simultáneo estimado: max viajes en ventana de 60 min
    df_s = df.sort_values('t_ini')
    pico_xt100 = max(
        len(df_s[(df_s['tipo_tren']=='XT-100') &
                 (df_s['t_ini']>=t) & (df_s['t_ini']<t+60)])
        for t in range(300, 1440, 30)
    ) if n_xt100 > 0 else 1
    pico_xtm = max(
        len(df_s[(df_s['tipo_tren']=='XT-M') &
                 (df_s['t_ini']>=t) & (df_s['t_ini']<t+60)])
        for t in range(300, 1440, 30)
    ) if n_xtm > 0 else 1
    # Limitar pool al pico × 1.3 (margen operacional), mínimo 5
    n_pool_xt100 = min(27, max(5, _math.ceil(pico_xt100 * 1.3)))
    n_pool_xtm   = min(8,  max(1, _math.ceil(pico_xtm   * 1.3)))
    pool_xt100 = list(range(1, 1 + n_pool_xt100))
    pool_xtm   = list(range(28, 28 + n_pool_xtm))

    def mi_pool(tipo):
        if tipo=='SFE':  return pool_sfe
        if tipo=='XT-M': return pool_xtm
        return pool_xt100

    asig = {}
    df_sorted = df.sort_values('t_ini').copy()

    for idx, row in df_sorted.iterrows():
        svc      = row['svc_type']
        orig     = ORIGEN.get(svc, 'PU')
        dest     = DESTINO.get(svc, 'PU')
        t        = row['t_ini']
        es_doble = row.get('doble', False)
        tipo     = row['tipo_tren']
        dwell_t  = DWELL_POR_TERMINAL.get(dest, 15.0)
        tf       = t + abs(row['km_dest'] - row['km_orig']) / 40.0 * 60 + dwell_t
        pool     = mi_pool(tipo)

        def t_arr(n):
            pos = estado[n]['pos']
            if pos == orig: return estado[n]['t_libre']
            return estado[n]['t_libre'] + T_ENTRE.get((pos, orig), 90)

        en_orig  = [n for n in pool if estado[n]['pos']==orig and estado[n]['t_libre']<=t]
        en_trans = [n for n in pool if n not in en_orig and t_arr(n)<=t]
        candidatos = en_orig + [n for n in en_trans if n not in en_orig]

        if es_doble:
            cs = sorted(candidatos)
            par = None
            for i in range(len(cs)-1):
                if cs[i+1]==cs[i]+1: par=(cs[i],cs[i+1]); break
            if par is None and len(cs)>=2: par=(cs[0],cs[1])
            if par is None:
                todos = sorted(pool, key=t_arr)
                a,b = todos[0],todos[1]
                for n in [a,b]:
                    if estado[n]['pos']!=orig:
                        estado[n]['t_libre']+=T_ENTRE.get((estado[n]['pos'],orig),90)
                        estado[n]['pos']=orig
                par=(a,b)
            a,b=par
            asig[idx]=f'{a}+{b}'
            t_sal_real=max(t,estado[a]['t_libre'],estado[b]['t_libre'])
            for n in [a,b]:
                estado[n]['pos']=dest
                estado[n]['t_libre']=t_sal_real+abs(row['km_dest']-row['km_orig'])/40*60+dwell_t
        else:
            if candidatos:
                n=min(candidatos,key=lambda n:estado[n]['t_libre'])
                if estado[n]['pos']!=orig:
                    estado[n]['t_libre']+=T_ENTRE.get((estado[n]['pos'],orig),90)
                    estado[n]['pos']=orig
            else:
                n=min(pool,key=t_arr)
                if estado[n]['pos']!=orig:
                    estado[n]['t_libre']+=T_ENTRE.get((estado[n]['pos'],orig),90)
                    estado[n]['pos']=orig
            asig[idx]=str(n)
            t_sal_real=max(t,estado[n]['t_libre'])
            estado[n]['pos']=dest
            estado[n]['t_libre']=t_sal_real+abs(row['km_dest']-row['km_orig'])/40*60+dwell_t

    df['motriz_num'] = df.index.map(asig).fillna('?')
    df = df.drop(columns=['_tf'], errors='ignore')

    df = df.drop(columns=['demanda', 'pax_est'], errors='ignore')
    return df

def parsear_planilla_maestra(data, fname):
    """
    LOGICA EXACTA DE LA PLANILLA MAESTRA EFE:
    
    V1 (N° Viaje PAR):
      - Servicio >= 600 -> PU-LI
      - Servicio 400-599 -> PU-SA
      - Servicio <= 399  -> PU-EB
    
    V2 (N° Viaje IMPAR):
      - Servicio >= 600 -> LI-PU
      - Servicio 400-599 -> SA-PU
      - Servicio <= 399  -> EB-PU
    
    Columna Unidad: vacio = Simple, "Múltiple" = Doble
    """
    try:
        ext = fname.lower()
        dfs = {}
        if ext.endswith('.csv'):
            try:
                raw = pd.read_csv(BytesIO(data), header=None, sep=',', encoding='utf-8', dtype=str)
            except:
                raw = pd.read_csv(BytesIO(data), header=None, sep=';', encoding='latin-1', dtype=str)
            dfs["CSV"] = raw
        else:
            eng = "xlrd" if ext.endswith(".xls") else "openpyxl"
            dfs = pd.read_excel(BytesIO(data), header=None, engine=eng, dtype=str, sheet_name=None)
            
        viajes = []
        for sheet_name, df in dfs.items():
            sheet_upper = str(sheet_name).upper()
            via_from_sheet = None
            if 'V1' in sheet_upper or 'VIA 1' in sheet_upper:
                via_from_sheet = 1
            elif 'V2' in sheet_upper or 'VIA 2' in sheet_upper:
                via_from_sheet = 2
            
            header_idx = -1
            for i in range(min(20, len(df))):
                row_str = ' '.join(df.iloc[i].fillna('').astype(str).str.upper())
                if ('VIAJE' in row_str or 'N°' in row_str or 'N ' in row_str) and ('SERVICIO' in row_str or 'TREN' in row_str) and ('HR PARTIDA' in row_str or 'HORA' in row_str or 'PARTIDA' in row_str or 'SALIDA' in row_str):
                    header_idx = i
                    break
            
            if header_idx != -1:
                headers = df.iloc[header_idx].fillna('').astype(str).str.upper()
                
                viaje_col = None
                srv_col = None
                hora_col = None
                unidad_col = None
                motriz1_col = None
                motriz2_col = None
                
                for c, val in enumerate(headers):
                    val_norm = str(val).strip().upper()
                    val_sin_tilde = ''.join(ch for ch in unicodedata.normalize('NFD', val_norm) if unicodedata.category(ch) != 'Mn')
                    
                    if 'VIAJE' in val_sin_tilde or val_sin_tilde == 'N°' or val_sin_tilde == 'N':
                        if viaje_col is None:
                            viaje_col = c
                    if 'SERV' in val_sin_tilde or 'TREN' in val_sin_tilde:
                        if srv_col is None:
                            srv_col = c
                    if 'HR PARTIDA' in val_sin_tilde or 'HORA' in val_sin_tilde or 'PARTIDA' in val_sin_tilde or 'SALIDA' in val_sin_tilde:
                        if hora_col is None:
                            hora_col = c
                    if any(p in val_sin_tilde for p in ['UNIDAD', 'UNID', 'CONF', 'TIPO', 'FORMA', 'OBS']):
                        if unidad_col is None:
                            unidad_col = c
                    if 'MOTRIZ' in val_sin_tilde and '1' in val_sin_tilde:
                        if motriz1_col is None:
                            motriz1_col = c
                    if 'MOTRIZ' in val_sin_tilde and '2' in val_sin_tilde:
                        if motriz2_col is None:
                            motriz2_col = c
                
                if srv_col is None:
                    for c in range(df.shape[1]):
                        if c == viaje_col or c == hora_col:
                            continue
                        muestras = df.iloc[header_idx+1:header_idx+10, c].dropna().astype(str)
                        if muestras.apply(lambda x: bool(re.match(r'^\d{3,4}$', x.strip()))).any():
                            srv_col = c
                            break
                
                if hora_col is None:
                    for c in range(df.shape[1]):
                        if c == viaje_col or c == srv_col:
                            continue
                        muestras = df.iloc[header_idx+1:header_idx+10, c].dropna().astype(str)
                        if muestras.apply(lambda x: bool(re.match(r'^\d{1,2}:\d{2}', x.strip()))).any():
                            hora_col = c
                            break
                
                if srv_col is None or hora_col is None:
                    continue
                
                for i in range(header_idx + 1, len(df)):
                    row = df.iloc[i]
                    
                    if pd.isna(row.get(srv_col)) or pd.isna(row.get(hora_col)):
                        continue
                    
                    srv_str = str(row[srv_col]).strip()
                    hora_str = str(row[hora_col]).strip()
                    
                    if not srv_str or not hora_str:
                        continue
                    
                    m_srv = re.search(r'(\d{3,4})', srv_str)
                    if not m_srv:
                        continue
                    servicio_num = int(m_srv.group(1))
                    
                    # Parsear hora — soporta HH:MM y decimal Excel (0.333 = 8:00)
                    t_ini = None
                    if re.match(r'^\d{1,2}:\d{2}(:\d{2})?$', hora_str):
                        t_ini = parse_time_to_mins(hora_str)
                    else:
                        try:
                            frac = float(hora_str)
                            if 0.0 <= frac <= 1.0:
                                t_ini = round(frac * 24 * 60, 2)  # fracción del día → minutos
                        except ValueError:
                            pass
                    if t_ini is None:
                        continue
                    
                    # Leer Motriz 1 y Motriz 2 si están disponibles
                    def _extract_motriz(col_idx):
                        if col_idx is None:
                            return None
                        val = row.iloc[col_idx] if isinstance(col_idx, int) else row.get(col_idx)
                        if pd.isna(val) or str(val).strip().lower() in ('nan', '', '0', '0.0'):
                            return None
                        m = re.search(r'(\d+)', str(val))
                        return int(m.group(1)) if m else None

                    n1 = _extract_motriz(motriz1_col)
                    n2 = _extract_motriz(motriz2_col)

                    # Determinar tipo_tren desde número de motriz
                    # XT-100: 1-27 | XT-M: 28-35 | SFE: 410-414
                    def _tipo_desde_num(n):
                        if n is None:
                            return 'XT-100'
                        if 1 <= n <= 27:
                            return 'XT-100'
                        if 28 <= n <= 35:
                            return 'XT-M'
                        if 410 <= n <= 414:
                            return 'SFE'
                        return 'XT-100'

                    tipo_tren_planilla = _tipo_desde_num(n1 or n2)

                    # motriz_num: "X+Y" para dobles, "X" para simples
                    if n1 and n2:
                        motriz_num_planilla = f"{n1}+{n2}"
                    elif n1:
                        motriz_num_planilla = str(n1)
                    else:
                        motriz_num_planilla = ''

                    es_doble = bool(n1 and n2)  # doble si hay dos motrices
                    # Fallback: columna Unidad si no hay Motriz 2
                    if not es_doble and unidad_col is not None and pd.notna(row.get(unidad_col)):
                        unidad_str = str(row[unidad_col]).strip().upper()
                        unidad_norm = ''.join(ch for ch in unicodedata.normalize('NFD', unidad_str) if unicodedata.category(ch) != 'Mn')
                        if any(kw in unidad_norm for kw in ['MULTIPLE', 'MULT', 'DOBLE', 'DOB', 'ACOPL', '2 UNID', '2UNID', '2UND', '2 UNIDADES', 'DOBLE UNIDAD', 'DUPLA']):
                            es_doble = True
                    
                    if via_from_sheet is not None:
                        via = via_from_sheet
                    elif viaje_col is not None and pd.notna(row.get(viaje_col)):
                        viaje_str = str(row[viaje_col]).strip()
                        m_viaje = re.search(r'(\d+)', viaje_str)
                        if m_viaje:
                            viaje_num = int(m_viaje.group(1))
                            via = 1 if viaje_num % 2 == 0 else 2
                        else:
                            via = 1 if servicio_num % 2 == 0 else 2
                    else:
                        via = 1 if servicio_num % 2 == 0 else 2
                    
                    km_limache = KM_ACUM_SAFE[20]
                    km_sargento = KM_ACUM_SAFE[18]
                    km_belloto = KM_ACUM_SAFE[14]
                    km_puerto = KM_ACUM_SAFE[0]
                    
                    if via == 1:
                        km_orig = km_puerto
                        if servicio_num >= 600:
                            km_dest = km_limache
                        elif 400 <= servicio_num <= 599:
                            km_dest = km_sargento
                        else:
                            km_dest = km_belloto
                    else:
                        km_dest = km_puerto
                        if servicio_num >= 600:
                            km_orig = km_limache
                        elif 400 <= servicio_num <= 599:
                            km_orig = km_sargento
                        else:
                            km_orig = km_belloto
                    
                    try:
                        idx_orig = KM_ACUM_SAFE.index(km_orig)
                        idx_dest = KM_ACUM_SAFE.index(km_dest)
                        ruta = f"{EC_SAFE[idx_orig]}-{EC_SAFE[idx_dest]}"
                        
                        if via == 1:
                            nodos_via = [(0.0, KM_ACUM_SAFE[j]) for j in range(idx_orig, idx_dest + 1)]
                        else:
                            nodos_via = [(0.0, KM_ACUM_SAFE[j]) for j in range(idx_orig, idx_dest - 1, -1)]
                    except:
                        ruta = "PU-LI"
                        if via == 1:
                            nodos_via = [(0.0, KM_ACUM_SAFE[j]) for j in range(0, 21)]
                        else:
                            nodos_via = [(0.0, KM_ACUM_SAFE[j]) for j in range(20, -1, -1)]
                    
                    viajes.append({
                        '_id': f"PLAN_{via}_{servicio_num}_{int(t_ini)}",
                        't_ini': t_ini,
                        'Via': via,
                        'km_orig': km_orig,
                        'km_dest': km_dest,
                        'nodos': nodos_via,
                        'tipo_tren': tipo_tren_planilla,
                        'doble': es_doble,
                        'motriz_num': motriz_num_planilla,
                        'num_servicio': str(servicio_num),
                        'svc_type': ruta,
                        'maniobra': None,
                        'pax_abordo': 0,
                        'pax_d': {}
                    })
            else:
                for i in range(len(df)):
                    row_vals = df.iloc[i].fillna('').astype(str).tolist()
                    
                    es_doble = False
                    for c_idx, val in enumerate(row_vals):
                        val_upper = str(val).strip().upper()
                        val_norm = ''.join(ch for ch in unicodedata.normalize('NFD', val_upper) if unicodedata.category(ch) != 'Mn')
                        if any(kw in val_norm for kw in ['MULTIPLE', 'MULT', 'DOBLE', 'DOB', 'ACOPL', '2 UNID', '2UNID', '2UND', '2 UNIDADES', 'DOBLE UNIDAD', 'DUPLA']):
                            es_doble = True
                            break
                    
                    for c_idx, val in enumerate(row_vals):
                        val = val.strip()
                        if re.match(r'^\d{1,2}:\d{2}(:\d{2})?$', val):
                            t_ini = parse_time_to_mins(val)
                            if t_ini is None:
                                continue
                            
                            servicio_num = None
                            for offset in range(1, 8):
                                if c_idx - offset >= 0:
                                    check_val = row_vals[c_idx - offset].strip()
                                    m_srv = re.search(r'(\d{3,4})', check_val)
                                    if m_srv and 200 <= int(m_srv.group(1)) <= 1999:
                                        servicio_num = int(m_srv.group(1))
                                        break
                            
                            if servicio_num is None:
                                continue
                            
                            viaje_num = None
                            for offset in range(1, 5):
                                if c_idx - offset >= 0:
                                    check_val = row_vals[c_idx - offset].strip()
                                    if check_val.isdigit() and 1 <= int(check_val) <= 500:
                                        viaje_num = int(check_val)
                                        break

                            if via_from_sheet is not None:
                                via = via_from_sheet
                            elif viaje_num is not None:
                                via = 1 if viaje_num % 2 == 0 else 2
                            else:
                                via = 1 if servicio_num % 2 == 0 else 2
                            
                            km_limache = KM_ACUM_SAFE[20]
                            km_sargento = KM_ACUM_SAFE[18]
                            km_belloto = KM_ACUM_SAFE[14]
                            km_puerto = KM_ACUM_SAFE[0]
                            
                            if via == 1:
                                km_orig = km_puerto
                                if servicio_num >= 600:
                                    km_dest = km_limache
                                elif 400 <= servicio_num <= 599:
                                    km_dest = km_sargento
                                else:
                                    km_dest = km_belloto
                            else:
                                km_dest = km_puerto
                                if servicio_num >= 600:
                                    km_orig = km_limache
                                elif 400 <= servicio_num <= 599:
                                    km_orig = km_sargento
                                else:
                                    km_orig = km_belloto
                            
                            try:
                                idx_orig = KM_ACUM_SAFE.index(km_orig)
                                idx_dest = KM_ACUM_SAFE.index(km_dest)
                                ruta = f"{EC_SAFE[idx_orig]}-{EC_SAFE[idx_dest]}"
                                
                                if via == 1:
                                    nodos_via = [(0.0, KM_ACUM_SAFE[j]) for j in range(idx_orig, idx_dest + 1)]
                                else:
                                    nodos_via = [(0.0, KM_ACUM_SAFE[j]) for j in range(idx_orig, idx_dest - 1, -1)]
                            except:
                                ruta = "PU-LI"
                                if via == 1:
                                    nodos_via = [(0.0, KM_ACUM_SAFE[j]) for j in range(0, 21)]
                                else:
                                    nodos_via = [(0.0, KM_ACUM_SAFE[j]) for j in range(20, -1, -1)]
                            
                            viajes.append({
                                '_id': f"PLAN_{via}_{servicio_num}_{int(t_ini)}",
                                't_ini': t_ini,
                                'Via': via,
                                'km_orig': km_orig,
                                'km_dest': km_dest,
                                'nodos': nodos_via,
                                'tipo_tren': 'XT-100',
                                'doble': es_doble,
                                'motriz_num': '',
                                'num_servicio': str(servicio_num),
                                'svc_type': ruta,
                                'maniobra': None
                            })
                            
        df_viajes = pd.DataFrame(viajes)
        if not df_viajes.empty:
            df_viajes = df_viajes.drop_duplicates(subset=['_id'])

        # Calcular dwell_terminal: tiempo en minutos que cada motriz espera
        # en terminal entre el arribo de un viaje y la salida del siguiente.
        # Requiere t_fin estimado = t_ini + tiempo_viaje_aprox
        # Usamos velocidad media operativa MERVAL ~42 km/h como estimación
        if not df_viajes.empty and 'motriz_num' in df_viajes.columns:
            V_MEDIA_KMH = 42.0
            df_viajes['t_fin_est'] = df_viajes['t_ini'] + (
                abs(df_viajes['km_dest'] - df_viajes['km_orig']) / V_MEDIA_KMH * 60.0
            )
            df_viajes['dwell_terminal_min'] = None

            # Para cada motriz individual (separar dobles en sus componentes)
            motriz_viajes = {}
            for idx, row in df_viajes.iterrows():
                mn = str(row.get('motriz_num', ''))
                if not mn or mn == '':
                    continue
                for m in mn.split('+'):
                    m = m.strip()
                    if m:
                        if m not in motriz_viajes:
                            motriz_viajes[m] = []
                        motriz_viajes[m].append(idx)

            for motriz, indices in motriz_viajes.items():
                viajes_motriz = df_viajes.loc[indices].sort_values('t_ini')
                for i in range(len(viajes_motriz) - 1):
                    idx_actual = viajes_motriz.index[i]
                    idx_siguiente = viajes_motriz.index[i + 1]
                    t_arribo = df_viajes.loc[idx_actual, 't_fin_est']
                    t_salida_sig = df_viajes.loc[idx_siguiente, 't_ini']
                    dwell = round(t_salida_sig - t_arribo, 1)
                    if 0 < dwell < 240:  # máximo 4 horas, descartar negativos
                        df_viajes.at[idx_siguiente, 'dwell_terminal_min'] = dwell

            df_viajes = df_viajes.drop(columns=['t_fin_est'], errors='ignore')

        df_viajes = asignar_flota_planilla(df_viajes)
        return df_viajes, "ok"
    except Exception as e:
        return pd.DataFrame(), str(e)
