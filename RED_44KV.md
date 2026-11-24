# V137 — modelo de red 44 kV

Fuente: Descripción de instalaciones AT/MT, sección 1.1. Datos de diseño,
no certificación actualizada. La red original conserva por defecto su flujo
aproximado V136. El selector 44 kV es independiente de la catenaria CC.

## Escenario documento

- El Sol–El Salto: 10,7 km de cobre 3/0 AWG (85,03 mm²) y 0,5 km
  de XLPE 150 mm².
- El Salto–Portales: 0,7 km Cu 1/0 AWG (53,49 mm²) y 6,8 km XLPE 120 mm².
- El Sol–Villa Alemana: 5 km aproximados. El documento no identifica
  la ubicación ni longitud de los tramos XLPE. Se permite introducir su
  longitud dentro de 5 km (por defecto 0 = no modelado). Se reparte
  proporcionalmente entre las dos ramas al este.
- El Belloto no figura en el documento. Se conserva como nodo intermedio
  supuesto de la configuración V136, a 1,2 km de El Sol por defecto;
  se puede desactivar su carga con el selector SER. No implica confirmar
  su conexión ni eliminar la alimentación aguas abajo a Villa Alemana.
- Suma detallada 23,7 km vs 23,5 km aproximados declarados.
- Se supone cobre en XLPE, resistencia ideal corregida por temperatura
  con factor ajustable para contrastar valores de fabricante; las
  reactancias 0,30/0,10 Ω/km son supuestos editables no documentados.
- Corrientes aproximadas a tensión nominal y FP configurable;
  ΔV trifásica lineal y pérdidas 3 I²R por fase. No considera pérdidas
  en las corrientes aguas arriba, desbalance, capacitancia, armónicos,
  taps o flujo AC iterativo. Vdc estimada por relación proporcional:
  no realimenta el motor de tracción CC.

## Transformadores y alcance energético

Se conservan las capacidades por transformador, dos conjuntos en serie
CC y eficiencias 0,96 y 0,99 existentes. No se convierten kW a kVA ni
se inventan pérdidas en vacío/carga, grupo vectorial o impedancia.
La capacidad instantánea del dashboard usa la suma de ambos conjuntos;
la demanda instantánea preexistente es solo una aproximación de auxiliares
y está señalada como tal.

La tabla de 15 minutos compara ambos escenarios AC sobre las mismas
demandas SER, manteniendo el balance original de eficiencias para aislar
la diferencia. No representa dos simulaciones de tracción ni una medición
SEAT. La potencia media por franja subestima pérdidas de picos subintervalo.
Se interpola posición por intervalo y se corrige el doble conteo de
pre/post en trenes dobles. La tabla de THDR también permite exportación
Excel en memoria. No se incluyen las cargas externas SAF de 12 kV.
El resto de paneles conserva sus aproximaciones temporales existentes.

Pruebas: python -m unittest discover -s tests -v.
No usar para decisiones de seguridad eléctrica o dimensionamiento sin
planos unilineales, placas, fichas de cables y mediciones de campo.
