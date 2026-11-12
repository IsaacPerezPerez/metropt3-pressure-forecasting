# Auditoría MetroPT-3 y diseño del experimento

Auditoría reproducible con `python scripts/audit_metropt3.py` sobre el ZIP y SHA-256 documentados en el README. La tabla completa de frecuencias para cada intervalo de timestamp se imprime al ejecutar el script; no se interpola ni se modifica la fuente. Las estadísticas descriptivas cubren todo el archivo para auditar su calidad; la comparación predictiva usa únicamente entrenamiento.

## Calidad, señales y cadencia

Hay 1 516 948 filas entre 2020-02-01 00:00:00 y 2020-09-01 03:59:50, sin zona horaria informada. No hay timestamps duplicados, fuera de orden o faltantes, ni valores faltantes en ninguna de las 17 columnas. `Unnamed: 0` es un índice, no una señal. No se ha verificado la calibración ni la plausibilidad de cada sensor.

| Variable | Mínimo | Mediana | Media | Máximo |
|---|---:|---:|---:|---:|
| Reservoirs (bar) | 0,712 | 8,960 | 8,985 | 10,300 |
| Motor_current (A) | 0,020 | 0,045 | 2,050 | 9,295 |
| Oil_temperature (°C) | 15,400 | 62,700 | 62,644 | 89,050 |

`Reservoirs` mide la presión aguas abajo de los depósitos; el mínimo bajo merece revisión, no se corrige automáticamente. UCI describe corriente cercana a 0 A en apagado, 4 A descargado, 7 A bajo carga y 9 A al arrancar. Las ocho digitales solo contienen 0 y 1. Conteos de unos: COMP 1 269 620; DV_eletric 243 638; Towers 1 395 362; MPG 1 263 108; LPS 5 188; Pressure_switch 1 503 958; Oil_level 1 371 557; Caudal_impulses 1 421 542. Según UCI representan, respectivamente, válvula de admisión, válvula de salida, torre de secado, control por presión, baja presión, descarga de torres, nivel de aceite y pulsos de caudal. Se respeta la ortografía del CSV. Su interpretación física y disponibilidad en producción requieren validación.

UCI menciona tanto 1 Hz como 0,1 Hz. Los 1 516 947 intervalos observados son: 8 s: 2; 9 s: 128 277; 10 s: 1 337 521; 11 s: 4 471; 12 s: 38 321; 13 s: 7 988; 14 s: 3; 15 s: 1; superiores a 15 s: 363. La moda es 10 s (aproximadamente 0,1 Hz), pero hay jitter y pausas: no equivale a una rejilla perfecta.

Para esta auditoría se considera continuo un intervalo de 8 a 15 s inclusive; es una decisión operativa que debe someterse a sensibilidad, no una propiedad garantizada por UCI. Hay 364 segmentos; el mayor contiene 36 599 lecturas, del 24 de marzo 04:03:08 al 28 de marzo 08:49:39. Los 363 intervalos superiores a 15 s suman 3 274 937 s entre lecturas, con mediana 2 256 s y máximo 172 918 s; 331 son de al menos 1 minuto, 275 de al menos 5 minutos, 160 de al menos una hora y 5 de al menos un día. Su exceso respecto a 10 s por intervalo suma 3 271 307 s; no se interpreta como número exacto de muestras perdidas debido al jitter.

## Cortes y persistencia

Propuesta por timestamp, extremos izquierdos inclusivos y derechos exclusivos:

- Entrenamiento: antes del 2020-04-01 00:00:00, 445 298 filas.
- Validación: desde esa fecha hasta antes del 2020-06-01 00:00:00, 411 534 filas.
- Prueba: desde 2020-06-01 00:00:00 hasta el final, 660 116 filas; reservada para evaluación final, sin selección de horizontes ni hiperparámetros.

Solo en entrenamiento, cada caso tiene seis lecturas consecutivas como historia, incluida la lectura ancla; el objetivo es la observación medida más cercana a 60 o 300 s después del ancla, con tolerancia de ±5 s y desempate hacia la anterior. Se exige que historia, ancla y objetivo estén dentro del mismo segmento y corte. No se interpolan objetivos. La presión del ancla y del objetivo debe ser finita; en este archivo no hay faltantes. Los casos de horizontes distintos no son necesariamente idénticos.

| Horizonte nominal | Ventanas válidas | Cambio medio firmado (bar) | Cambio absoluto medio (bar) | Mediana absoluta (bar) | P95 absoluto (bar) | MAE persistencia (bar) |
|---|---:|---:|---:|---:|---:|---:|
| 60 s ± 5 s | 444 364 | 0,000792 | 0,143252 | 0,066 | 0,766 | 0,143252 |
| 300 s ± 5 s | 442 364 | 0,002262 | 0,554145 | 0,356 | 1,716 | 0,554145 |

Persistencia predice la última presión observada; por definición su MAE es el cambio absoluto medio. Los horizontes son tiempos transcurridos, no índices de fila. Los valores son descriptivos, no evaluación de prueba.

## Experimento futuro de señales

Para cada horizonte y partición, generar primero una única lista de casos válidos y reutilizarla exactamente en ambos brazos, con las mismas señales y objetivo `Reservoirs` futuro medido. Brazo original: presión y tasa `(p_t - p_{t-1}) / (t - t_{-1})` sin suavizar, junto con las demás señales seleccionadas. Brazo suavizado: sustituir esos dos canales por sus respectivas medias exponenciales causales, inicialmente con α=0,2; las demás entradas permanecen iguales. Ninguna transformación usa valores posteriores al instante de entrada. Comparar RNN y LSTM con las mismas semillas, arquitectura y presupuesto entre brazos, además de persistencia; seleccionar configuración en validación y evaluar prueba una sola vez.

La tasa usa segundos reales. Al inicio de cada segmento y de cada partición se reinician los estados: la EMA se inicializa con el primer valor disponible, la primera tasa no está definida y se descartan de ambos brazos las ventanas que la requieran. Los estados nunca se propagan a través de un intervalo fuera de 8–15 s ni de un corte. Ajustar escaladores y cualquier parámetro aprendido únicamente con entrenamiento; aplicar los mismos parámetros a validación y prueba. El conteo anterior es una línea base exploratoria y puede reducirse al imponer esta condición adicional y disponibilidad de todas las entradas. Reportar cobertura y métricas emparejadas; evaluar sensibilidad del umbral de continuidad y del suavizado con entrenamiento y validación, sin consultar prueba.
