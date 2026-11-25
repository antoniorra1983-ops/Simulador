# V139 — un aplicativo, dos modalidades

La entrada de Streamlit es app.py. El menú lateral ofrece:
- Modelación por planilla: planificador, circulación, demanda energética, gemelo digital y exportación SEAT.
- Modelación por THDR: fecha y archivos reales de Datos-Operacionales, estimación energética y exportación SEAT.

La página independiente anterior se trasladó a thdr_ui.py y se llama desde app.py.
El acceso y la configuración de red son comunes. Por defecto se seleccionan
catenaria vía principal (2 × 107 mm² + mensajero 150 mm²) y registro técnico MERVAL 2013.
Se mantienen los escenarios históricos y documentales como alternativas.
Los nombres visibles no usan la marca del documento original; los identificadores
internos se mantienen por compatibilidad.

La selección de red se comparte en ambos modos. Los resultados THDR quedan
asociados a sus archivos, pasajeros, parámetros físicos y configuración eléctrica;
si cambian, se debe ejecutar la modelación nuevamente. Los datos originales y las
hipótesis pendientes (feeders, retorno, El Belloto, transformadores) no cambian.
La caída AC no realimenta directamente el motor CC. No constituye validación
del sistema eléctrico con mediciones reales.
