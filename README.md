# Pronóstico de presión industrial

Proyecto universitario para pronosticar la presión de salida de los depósitos (`Reservoirs`, bar) de la unidad de producción de aire de un tren, usando MetroPT-3. El proyecto incluye preprocesamiento causal y comparación de RNN y LSTM en validación con Docker y MLflow. La evaluación final de prueba y el servidor de inferencia quedan para una fase posterior.

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

La salida completa y compacta por partición, incluidos los conteos por horizonte real en segundos, está en [docs/RESULTADOS_FASE02.json](docs/RESULTADOS_FASE02.json). En entrenamiento se retienen 440 342 ventanas y el MAE de persistencia es 0,552306 bar; en validación son 405 173 y 0,545120 bar. Las tres sensibilidades 7–16, 8–15 y 9–14 s producen 85 segmentos, 440 342 ventanas y MAE 0,552306 bar en entrenamiento. Esta comparación no aporta evidencia para revisar el umbral principal, aunque no establece que sea óptimo en otros datos. La comparación de redes y selección en validación se describen en la fase 03; la evaluación final de prueba sigue reservada. El diseño exploratorio de señales de la auditoría es histórico: esta fase usa las dos versiones especificadas aquí.

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
