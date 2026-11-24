# Modelo de catenaria V136

El selector Modelo de catenaria CC está en el planificador y en Simular THDR.
Por defecto conserva las resistencias históricas de V135. Los escenarios Siemens
usan los datos de diseño IV Etapa (2004): 2×107 + 150 mm² (principal) y
1×107 + 150 mm² (secundaria), sin desgastar. La vía secundaria es un escenario
uniforme hipotético, no una cartografía de ramales.

Hipótesis: cobre a 20 °C (0,01724 Ω·mm²/m), coeficiente térmico 0,00393/°C,
conductores positivos en paralelo ideal. Desgaste reduce solo sección del hilo.
Feeder Cu equivalente y retorno se ingresan explícitamente; cero significa
omitido/desconocido, no que no existan. No se infiere su configuración definitiva.
Se muestran resistencias comparadas por PK. Altura, vano y tensado se conservan
como datos de referencia en config.py, no se modelan dinámicamente.

Siemens utiliza alimentación bilateral ideal entre SER activas y radial en
extremos. El motor estima la caída con potencia instantánea y la distribución
energética integra espacialmente una potencia media por viaje; el análisis de
tensión agrega perfiles temporales. No son un flujo DC iterativo completo:
no se modela retorno distribuido, contacto eléctrico individual, protección,
curvas de regulación de cada rectificador durante cada paso ni pérdidas
multi-tren exactas. Los modelos de regeneración y la red AC no se recalibran.
El modo histórico mantiene la aproximación original de pérdidas, con lista
de SER activas aplicada también al motor y al análisis de tensión.

No usar los resultados para dimensionamiento o seguridad sin contrastar
planos as-built, mediciones de resistencia y registros SEAT/SER/pantógrafo.
Las pruebas unitarias verifican fórmulas y coherencia básica, no validación
de ingeniería de toda la aplicación.
