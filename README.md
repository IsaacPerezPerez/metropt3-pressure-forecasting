# Pronóstico de presión industrial — auditoría inicial

Proyecto universitario para pronosticar la presión de salida de los depósitos (`Reservoirs`, bar) de la unidad de producción de aire de un tren, usando MetroPT-3. Esta fase se limita a auditar los datos y diseñar el experimento: aún no implementa ni entrena modelos. El objetivo final conserva la comparación de RNN y LSTM, entrenamiento con Docker Compose, seguimiento con MLflow, pruebas e inferencia en Docker; son entregables futuros, no funcionalidades ya implementadas.

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

La salida completa y compacta por partición, incluidos los conteos por horizonte real en segundos, está en [docs/RESULTADOS_FASE02.json](docs/RESULTADOS_FASE02.json). En entrenamiento se retienen 440 342 ventanas y el MAE de persistencia es 0,552306 bar; en validación son 405 173 y 0,545120 bar. Las tres sensibilidades 7–16, 8–15 y 9–14 s producen 85 segmentos, 440 342 ventanas y MAE 0,552306 bar en entrenamiento. Esta comparación no aporta evidencia para revisar el umbral principal, aunque no establece que sea óptimo en otros datos. La comparación de redes, selección en validación y evaluación final reservada de prueba quedan para la fase siguiente. El diseño exploratorio de señales de la auditoría es histórico: esta fase usa las dos versiones especificadas aquí.
