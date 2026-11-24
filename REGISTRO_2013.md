# V138: registro de líneas MERVAL 2013

Fuente: Información Técnica Secciones Tramos 2013, hoja Patron filas 884–886,
propietario L068, columnas F,H,J,K,P y L,N,O. R positiva a 30 °C, X en ohm/km,
B en microS/km, corriente P en kA convertida a A. El registro no tiene una
fila de El Belloto. Se usa la misma R/X/B por km y corriente nominal de
Villa Alemana, dividiendo 4,8 km en 1,2 + 3,6 km por defecto (hipótesis).
La potencia de los transformadores no se modifica; sigue 9 MW configurados
para EB y 6 MW para VA. El modelo histórico y documental siguen disponibles.

El registro tiene 10,3 km ES–PO frente a 7,5 km del documento AT/MT; no se
reconcilia automáticamente ni se declara cuál es el as-built vigente.
Se incorporan R/X/B positivos en un equivalente pi por rama, flujo radial
equilibrado PQ con barrido atrás/adelante. R corregible desde 30 °C con
coeficiente de cobre. Se calcula corriente de serie y terminales, pérdidas
activas, Q de SEAT y carga respecto de corriente permanente nominal.
Los reactores de 0,333 MVAr en cada extremo de ES–PO son opcionales,
desactivados por defecto porque se desconoce su estado de operación.
Secuencia cero almacenada solo para referencia, no cálculo de fallas.

Las tablas SEAT muestran tres escenarios con misma demanda media por franja.
Las pérdidas de carga capacitiva sin tracción se consideran dentro de las
franjas calculadas, no todas las horas sin servicio. Si no hay convergencia
se marca explícitamente y no se presenta un total parcial como válido.
No se modela realimentación AC al motor de tracción ni cargas SAF 12 kV.
No se certifican parámetros reales de transformadores ni se valida seguridad.
