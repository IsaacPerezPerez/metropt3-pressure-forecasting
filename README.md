# Pronóstico horario de PM2.5 en Dongsi — fase 01

Objetivo: pronosticar PM2.5 de la siguiente hora en Dongsi, Beijing, a partir de las 24 horas anteriores. En fases posteriores se compararán RNN y LSTM en PyTorch. Esta fase solo prepara el repositorio y audita los datos.

## Fuente y versión

- [UCI: Beijing Multi-Site Air Quality](https://archive.ics.uci.edu/dataset/501/beijing+multi+site+air+quality); licencia [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
- Cita: Chen, S. (2017). *Beijing Multi-Site Air Quality* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5RK5G.
- Descarga oficial: https://archive.ics.uci.edu/static/public/501/beijing%2Bmulti%2Bsite%2Bair%2Bquality%2Bdata.zip. SHA-256 del paquete: `b04da438b2f331ac0ffd45aebdfec0d20d2367feb5f6948c4b1f7ce1191e33c4`.
- Archivo original dentro del paquete y gestionado por DVC: `PRSA2017_Data_20130301-20170228.zip`; SHA-256: `d1b9261c54132f04c374f762f1e5e512af19f95c95fd6bfa1e8ac7e927e3b0b8`. Contiene 13 entradas, incluido `PRSA_Data_20130301-20170228/PRSA_Data_Dongsi_20130301-20170228.csv`. La auditoría verifica integridad CRC del ZIP y el hash antes de leerlo.

## Recuperación desde un clon limpio (Windows PowerShell)

No hay remoto DVC configurado: `dvc pull` por sí solo no puede recuperar los datos. Con Python 3.11, Git y acceso a UCI:

```powershell
git clone <ruta-o-URL-del-repositorio> pm25-clon
cd pm25-clon
python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements-dev.txt
.\.venv\Scripts\python scripts\recover_data.py
.\.venv\Scripts\dvc status
.\.venv\Scripts\python scripts\audit_dongsi.py
.\.venv\Scripts\pre-commit install
.\.venv\Scripts\pre-commit run --all-files
```

La recuperación descarga el paquete oficial, comprueba ambos SHA-256 y extrae solo el ZIP original sin sobrescribir una versión diferente. Depende de la disponibilidad futura de UCI; si cambia el paquete, falla sin escribir datos y debe investigarse. El archivo `.dvc` fija también el identificador de contenido usado por DVC. No se ha configurado ni publicado un remoto DVC.

## Auditoría ejecutada

Se encontraron 35 064 filas, del 2013-03-01 00:00 al 2017-02-28 23:00, sin marcas temporales duplicadas ni horas sin registro y con 750 filas con PM2.5 `NA`. Hay 34 290 ventanas con 24 horas registradas y objetivo PM2.5 válido; de ellas, 30 114 también tienen PM2.5 válido en las 24 entradas. La hora objetivo sigue inmediatamente a la ventana. Se excluyen marcas duplicadas de los conteos de ventanas; aquí no hay duplicados. Se consideran fechas y horas tal como aparecen en la fuente, sin asignar zona horaria.

La disponibilidad de otras variables, sus faltantes, el tratamiento de valores faltantes y la división temporal sin fugas quedan para preprocesamiento. El conteo no garantiza ventanas completas para otras variables ni determina cuántas habrá después de dividir los datos. No se modificaron los datos originales.

## Entorno verificado

Windows 10 (compilación 19045), Python 3.11.9, Git 2.55.0.windows.5, DVC 3.66.1, pre-commit 4.3.0, Docker 29.8.2 y Docker Compose 5.5.1. DVC y pre-commit se instalaron en `.venv`; Docker se detectó, pero no se usó para entrenar en esta fase.
