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
