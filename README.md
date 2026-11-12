# Pronóstico de presión industrial

Proyecto universitario para pronosticar la presión de salida de los depósitos (`Reservoirs`, bar) de la unidad de producción de aire de un tren, usando MetroPT-3. El proyecto incluye preprocesamiento causal y comparación de RNN y LSTM en validación con Docker y MLflow. La fase 04 evalúa el modelo seleccionado en prueba e incorpora inferencia CPU.

La metodología, la cadencia, los cortes, los resultados de persistencia y el diseño causal están en [docs/AUDITORIA_METROPT3.md](docs/AUDITORIA_METROPT3.md). Los resultados de Beijing permanecen en el historial, hasta el commit `ab0673a`.

## Fuente y procedencia

- [MetroPT-3 Dataset, UCI](https://archive.ics.uci.edu/dataset/791/metropt3+dataset); [descarga oficial directa](https://archive.ics.uci.edu/static/public/791/metropt%2B3%2Bdataset.zip); licencia [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
- Davari, N., Veloso, B., Ribeiro, R. y Gama, J. (2021). *MetroPT-3 Dataset* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5VW3R.
- Paquete original `data/raw/metropt3-dataset.zip` (contiene CSV y descripción PDF): SHA-256 `aab991a970e58210de853bb8078ce0e63abb4d9412fdc5c79792dae3d8e1721a`. El descriptor DVC fija además MD5 y tamaño; el archivo grande no va en Git.

## Reproducir en Windows PowerShell

Con Python 3.11, Git y acceso a UCI:

```powershell
git clone <ruta-o-URL-del-repositorio> pressure-clon
cd pressure-clon
git switch industrial-pressure-forecasting
python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements-dev.txt
.\.venv\Scripts\python scripts\recover_metropt3.py
.\.venv\Scripts\dvc commit data\raw\metropt3-dataset.zip.dvc
.\.venv\Scripts\dvc status data\raw\metropt3-dataset.zip.dvc
.\.venv\Scripts\python scripts\audit_metropt3.py
.\.venv\Scripts\pre-commit run --all-files
```

No hay remoto DVC configurado: `dvc pull` no recupera estos datos por sí solo. El recuperador descarga desde UCI en un archivo temporal, verifica SHA-256 y no sobrescribe un archivo distinto; `dvc commit` incorpora el contenido verificado a la caché local. La recuperación depende de que UCI mantenga disponible exactamente ese paquete. No se asigna zona horaria a los timestamps originales.

Se verificó el recuperador desde UCI y `dvc commit` en un clon local independiente, sin caché inicial ni caché compartida. En otro clon independiente se verificó `dvc pull` utilizando un remoto local temporal poblado con `dvc push`: en ambos casos el SHA-256 coincidió y `dvc status` quedó actualizado. Ese remoto fue solo una prueba y no está configurado para distribución. Los clones se prepararon desde `ab0673a` aplicando los cambios locales aún sin commit.

En Windows, el antiguo patrón `/data/*` de `.gitignore` hacía que DVC 3.66.1 omitiera el directorio al buscar etapas y mostrara «There are no data or pipelines tracked in this project yet.» Se retiró ese patrón; la exclusión del paquete se mantiene en `data/raw/.gitignore`. Se verificó el descriptor expresamente: con el archivo presente DVC indica «Data and pipelines are up to date.»; al retirarlo temporalmente lo identifica como `deleted`. El historial conserva el descriptor de Beijing.

## Fase 02: preprocesamiento reproducible

La configuración ejecutable está en `scripts/preprocess_metropt3.py`; genera índices de ancla y etiqueta compartidos y materializa cada ventana solo cuando se solicita. Cada caso usa 30 lecturas hasta el ancla, el objetivo medido más próximo a 300 s (±5 s; empate hacia la lectura anterior), intervalos de 8–15 s y los cortes temporales de la auditoría. Se reinician señales y segmentos en cada corte; las dos lecturas previas al inicio de la ventana también deben pertenecer al segmento y a la partición. Se excluyen ventanas con valores no finitos en cualquier canal y objetivos de presión no finitos. No se interpola ni se escala. La fuente describe COMP como señal de la válvula de admisión; su disponibilidad operacional requiere validación.

El brazo original incluye `Reservoirs`, `Motor_current`, `Oil_temperature`, `COMP`; el ampliado agrega `Reservoirs_mean_3` y `Reservoirs_rate_bar_s`, en ese orden. El promedio incluye la lectura actual y dos anteriores; la tasa usa el tiempo realmente transcurrido. `Cases.window(i, augmented=False/True)` y `Cases.label(i)` recuperan cada caso de los mismos índices. Las formas reportadas son lógicas, no arreglos densos guardados. No se consultan métricas de prueba: el lector termina al llegar al corte de junio.

```powershell
.\.venv\Scripts\python scripts\preprocess_metropt3.py
.\.venv\Scripts\python -m unittest discover -s tests -v
.\.venv\Scripts\pre-commit run --all-files
```

La salida completa y compacta por partición, incluidos los conteos por horizonte real en segundos, está en [docs/RESULTADOS_FASE02.json](docs/RESULTADOS_FASE02.json). En entrenamiento se retienen 440 342 ventanas y el MAE de persistencia es 0,552306 bar; en validación son 405 173 y 0,545120 bar. Las tres sensibilidades 7–16, 8–15 y 9–14 s producen 85 segmentos, 440 342 ventanas y MAE 0,552306 bar en entrenamiento. Esta comparación no aporta evidencia para revisar el umbral principal, aunque no establece que sea óptimo en otros datos. La comparación de redes y selección en validación se describen en la fase 03; la evaluación final posterior se describe en la fase 04. El diseño exploratorio de señales de la auditoría es histórico: esta fase usa las dos versiones especificadas aquí.

## Fase 03: entrenamiento recurrente

El código usa los índices de fase 02 y deja reservado el periodo desde 2020-06-01. El archivo [config_fase03.json](config_fase03.json) fija cinco épocas, Adam, lote 512, semilla 2026 y una capa unidireccional de 32 unidades. Se normalizan las columnas continuas usando exclusivamente filas únicas de entrenamiento incluidas en ventanas; COMP permanece sin escalar. El objetivo está en bar. Las ventanas se construyen por lote. Cada corrida guarda el mejor checkpoint por MAE de validación, el escalador y el esquema en `artifacts/<run_id>/`, además de registrarlos en MLflow. La corrida `--smoke` usa solo 4096 casos por partición durante una época y no debe utilizarse para seleccionar el modelo.

Con Docker Desktop y GPU NVIDIA habilitada, desde PowerShell:

```powershell
docker compose build train
docker compose run --rm train python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0)); print((torch.ones(1, device='cuda') + 1).cpu())"
docker compose run --rm train python scripts/train_metropt3.py --smoke
docker compose run --rm train python scripts/train_metropt3.py --architecture RNN
docker compose run --rm train python scripts/train_metropt3.py --architecture RNN --augmented
docker compose run --rm train python scripts/train_metropt3.py --architecture LSTM
docker compose run --rm train python scripts/train_metropt3.py --architecture LSTM --augmented
docker compose run --rm train python scripts/select_metropt3.py <run_id_rnn_original> <run_id_rnn_ampliada> <run_id_lstm_original> <run_id_lstm_ampliada>
docker compose run --rm --service-ports -p 5000:5000 train mlflow ui --backend-store-uri file:///workspace/mlruns --host 0.0.0.0 --port 5000
```

Abrir `http://localhost:5000` para consultar MLflow. Los directorios `mlruns/`, `artifacts/` y `outputs/` son locales y están excluidos de Git. La selección usa MAE sin redondear; un empate exacto favorece entradas originales y después RNN. Las métricas de las corridas completas usan toda la validación y nunca el tramo de prueba. Consultar [docs/FASE03.md](docs/FASE03.md) para el estado y resultados efectivamente verificados.

## Efecto de las señales en validación

La media causal de tres lecturas y la tasa de cambio de presión se añadieron a las cuatro señales originales. En validación, el MAE de la RNN pasó de 0.284004 a 0.476307 bar (subió 67.71 %), mientras que el de la LSTM pasó de 0.353838 a 0.280736 bar (bajó 20.66 %). Son resultados de estas corridas, no una garantía general. La selección se hizo con validación, antes de consultar prueba. Véase [FASE03.md](docs/FASE03.md).

## Fase 04: evaluación final e inferencia CPU

El ganador de fase 03 se congeló antes de leer prueba: [manifiesto con SHA-256, escalador y esquema](docs/FASE04_CONGELACION.json). El checkpoint `best.pt` (época 5) tiene SHA-256 `5ef6067a30e856b3bf3c9805d8b1b0c0f945a415f07f3489e8e8605c73d575ee`; sus cinco archivos coinciden con los del run `13e41b225750413a976d09f36a91ab69`. Se recalculó el escalador con entrenamiento y las dos copias del checkpoint generaron exactamente el mismo lote de 512 casos de validación. La comparación original del modelo en memoria con su checkpoint consta en la fase 03; el modelo en memoria ya no existe para repetirla.

La prueba, utilizada solo para medir el ganador ya seleccionado, contiene **650 545** ventanas idénticas para ambos métodos. Las anclas van de 2020-06-01 00:37:09 a 2020-09-01 03:54:53 y las etiquetas de 2020-06-01 00:42:06 a 2020-09-01 03:59:50; no se asigna zona horaria. Se usa el objetivo más cercano a t + 300 s dentro de ±5 s, con empate hacia el anterior, sin cruzar segmentos ni cortes.

| Método | MAE (bar) | RMSE (bar) |
|---|---:|---:|
| LSTM ampliada | 0.364242 | 0.478356 |
| Persistencia | 0.708209 | 0.862600 |

La reducción de MAE es **0.343967 bar (48.57 %)** respecto de persistencia. Valores completos y fechas están en [RESULTADOS_FASE04.json](docs/RESULTADOS_FASE04.json); el run de evaluación MLflow `af679ccd27304e2281247516d0f7d913` identifica al ganador y su hash. Los resultados son locales al dataset y reglas descritos; no constituyen validación operativa.

### Contrato y ejecución

El endpoint `POST /predict` recibe `{"readings": [...]}` con exactamente 32 objetos, ordenados cronológicamente. Cada objeto contiene exclusivamente `timestamp` ISO 8601 sin zona horaria (por ejemplo `2020-07-01T00:00:00`), `Reservoirs` (bar), `Motor_current`, `Oil_temperature` y `COMP`, estos cuatro como números finitos. Los intervalos consecutivos deben estar entre 8 y 15 segundos inclusive; no deben cruzar los cortes 2020-04-01 o 2020-06-01. No se interpola. Las dos primeras lecturas aportan historia causal y las 30 restantes forman la entrada: media móvil de tres y diferencia de presión dividida por el intervalo real; después se aplica el escalador congelado. La salida incluye `prediction_bar`, `target`, `horizon_seconds`, `anchor_timestamp` y `forecast_timestamp` para t + 300 s. El ejemplo sintético de 32 filas se genera y envía sin dependencias adicionales:

```powershell
# Primero instale el ZIP como se indica en la siguiente sección.
docker compose -f compose.inference.yaml up -d --build
Invoke-RestMethod http://localhost:8000/health
python scripts/example_request.py
```

El servicio carga `outputs/metropt3_model` desde un montaje de solo lectura y no requiere servidor de tracking, dataset ni GPU. La imagen se construye con Python público y PyTorch CPU. La comparación de la petición de ejemplo contra el checkpoint local usa tolerancia absoluta de 0.00001 bar; la verificación observada y el rechazo HTTP 422 están en [VERIFICACION_FASE04.json](docs/VERIFICACION_FASE04.json). Para detenerlo: `docker compose -f compose.inference.yaml down`.

### Distribución e instalación del modelo

El paquete versionado `metropt3-inference-v1.0.0.zip` tiene SHA-256 **`2002146f6b2899988d0e3178262251fb8b0ccd913e183a274b2b8fb8fd91332f`**. Se preparó para adjuntarlo a una futura GitHub Release; todavía no hay Release ni enlace de descarga publicado. Una vez publicada, descargue el ZIP desde los activos de esa Release y verifique este hash obtenido de una fuente confiable. Hasta entonces, transfiera el ZIP local por un canal confiable. El manifiesto interno identifica el run, los commits de procedencia, el checkpoint y todos los archivos del modelo. El hash del ZIP es la referencia externa: los hashes dentro del ZIP por sí solos no autentican su origen. Solo cargue paquetes confiables: MLflow utiliza serialización de Python.

En un clon limpio, con Python 3.11 y Docker Desktop, para instalar desde una ruta local y ejecutar únicamente la demo (sin dataset ni servidor MLflow):

```powershell
python scripts/release_metropt3.py install C:\ruta\metropt3-inference-v1.0.0.zip --sha256 2002146f6b2899988d0e3178262251fb8b0ccd913e183a274b2b8fb8fd91332f
$env:INFERENCE_PORT = "18080" # elegir un puerto libre
docker compose -f compose.inference.yaml up -d --build
Invoke-RestMethod http://localhost:18080/health
python scripts/example_request.py --url http://localhost:18080
try { Invoke-RestMethod -Method Post -Uri http://localhost:18080/predict -ContentType 'application/json' -Body '{"readings":[]}' } catch { $_.ErrorDetails.Message }
docker compose -f compose.inference.yaml down
```

La prueba independiente y sus comandos están en [VERIFICACION_RELEASE.md](docs/VERIFICACION_RELEASE.md). La respuesta válida esperada para el ejemplo es `prediction_bar = 8.821163177490234`, con tolerancia absoluta de **0.00001 bar** respecto al checkpoint de fase 04; la petición vacía devuelve HTTP 422. El instalador comprueba el hash antes de extraer, valida cada archivo y no sustituye un destino existente. El ZIP, el modelo, los checkpoints, el dataset y `mlruns/` permanecen fuera de Git.

Para reconstruir el ZIP a partir del modelo exportado local verificado (compruebe el SHA-256 resultante; las versiones de compresión pueden afectarlo):

```powershell
python scripts/release_metropt3.py build outputs/release/metropt3-inference-v1.0.0.zip
```

El contenido exportado incluye el checkpoint, escalador, esquema, código y metadatos MLflow. Si solo se conservan los cinco artefactos originales y sus copias de MLflow, el procedimiento de fase 04 requiere también recuperar el dataset UCI de la sección «Reproducir en Windows PowerShell» y ejecutar `scripts/freeze_metropt3.py` y `scripts/export_metropt3.py` en la imagen CPU. Sin el checkpoint original, reentrenar no garantiza recuperar sus pesos o hash. Para reproducir el entrenamiento completo recupere el dataset oficial como se explica arriba y siga los comandos de fase 03; la demo necesita solo el ZIP.

### Pruebas y CI

El workflow [CI CPU](.github/workflows/ci.yml) instala Python 3.11, dependencias y PyTorch CPU, y ejecuta `python -m unittest discover -s tests -v` y `pre-commit run --all-files` en cada push y pull request. Las pruebas usan datos sintéticos y no necesitan dataset, pesos ni tracking local; comprueban también la instalación y el rechazo de paquetes incorrectos. La comparación de la predicción real del ganador se realiza aparte mediante la integración Docker y el ZIP verificado: CI no la ejecuta. El workflow se agregó localmente y su ejecución remota está pendiente de un push autorizado.
