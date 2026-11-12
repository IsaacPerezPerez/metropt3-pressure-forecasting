# Fase 03: entrenamiento y selección en validación

Se reutilizó `1d832353ccf874b7c518a79aef91fd9120c39c67`. El código de entrenamiento ejecutado es `8db576e7cc03f20ed629c0d79679b684922ad9d0`; su SHA-256 se guardó en cada corrida. El ZIP tiene SHA-256 `aab991a970e58210de853bb8078ce0e63abb4d9412fdc5c79792dae3d8e1721a`. Se reconstruyeron 440342 casos de entrenamiento y 405173 de validación, compartidos por las dos variantes, y la persistencia en validación fue 0.5451199413583829 bar (0.545120 a seis decimales). No se evaluó el tramo de prueba desde 2020-06-01. Se usa el corte de entrenamiento 2020-04-01 y validación hasta 2020-06-01, exclusivo.

Configuración fijada antes de las corridas: [config_fase03.json](../config_fase03.json); semilla 2026, 30 lecturas, horizonte 300 s, una capa unidireccional de 32 unidades, lote 512, Adam con tasa 0.001, pérdida MSE, cinco épocas y mejor checkpoint por MAE de validación completa. El escalador usa media y desviación estándar poblacional sobre filas únicas incluidas en ventanas de entrenamiento; COMP permanece sin escalar. El objetivo está en bar. Se crean ventanas solo por lote. Las cuatro corridas utilizaron CPU y cuatro hilos PyTorch.

| Modelo | MAE bar | RMSE bar | MAE − persistencia bar | Cambio porcentual | Run ID |
|---|---:|---:|---:|---:|---|
| RNN original | 0.284004 | 0.636784 | -0.261116 | -47.90 % | b1862a2da6094b568d1585fff0d3d70f |
| RNN ampliada | 0.476307 | 0.605151 | -0.068813 | -12.62 % | b3bbe1f8dafa42e69a04770a06dbe06b |
| LSTM original | 0.353838 | 0.505130 | -0.191281 | -35.09 % | 4e37271a37b24c9ca46aa3d42ffa2817 |
| LSTM ampliada | 0.280736 | 0.409218 | -0.264384 | -48.50 % | 13e41b225750413a976d09f36a91ab69 |

Las diferencias usan MAE sin redondear; signo negativo significa menor error que persistencia. Añadir las señales aumentó el MAE de RNN en 0.192303 bar (67.71 %) y redujo el de LSTM en 0.073102 bar (20.66 %), respecto de cada variante original. El ganador por MAE sin redondear es **LSTM ampliada**, run `13e41b225750413a976d09f36a91ab69` (0.2807360319524576 bar); todas las redes superaron persistencia en MAE. En las cuatro, el mejor checkpoint fue la época 5. Para cada uno, se comparó exactamente un lote de predicciones del modelo en memoria con las del checkpoint recargado. Tiempos de entrenamiento más validaciones: 41.84, 41.38, 82.58 y 86.47 s, respectivamente; no incluyen preparación inicial.

`artifacts/<run_id>/` y MLflow contienen `best.pt`, `scaler.json`, `schema.json`, `requirements-freeze.txt` y `result.json`; son locales y no están en Git. Las métricas por época, dispositivo, configuración y tiempos están en el experimento `metropt3-fase03`. El JSON [RESULTADOS_FASE03.json](RESULTADOS_FASE03.json) conserva precisión completa y la selección. No se capturaron métricas de utilización de sistema.

## Docker, CUDA y pruebas

Docker expuso la RTX 5060 con `docker run --rm --gpus all --entrypoint sh proyecto1-web:latest -c 'nvidia-smi'`. La imagen local de diagnóstico tenía PyTorch `2.7.1+cpu`, `torch.version.cuda=None` y `torch.cuda.is_available()=False`: la imposibilidad de usar CUDA en esa imagen se debe a que es una compilación CPU, no prueba un fallo del controlador. La imagen CUDA 12.8 de `Dockerfile` quedó descargándose; **no se verificó todavía PyTorch CUDA ni se utilizó la GPU para estos entrenamientos**. La prueba corta `0f90c6dc40e14f1b9540ff2487b80375` usó 4096 casos por partición, pasó registro y recarga y no participa en la selección. Hubo otra corta exitosa `7126cb7785214c22965dd83a8ca6ef69` y una fallida `797c8688c9854c4380e79916c1431f7b` por ausencia inicial de Git, corregida antes de las completas.

Como referencia, 100 lotes sintéticos CPU de 512 x 30 x 6 con cuatro hilos tomaron 0.95 s en RNN y 2.17 s en LSTM; extrapolados linealmente a cinco épocas equivalen a 40.7 y 93.2 s solo de entrenamiento, sin validación ni preparación. Los tiempos reales aparecen arriba.

La imagen CPU usada está disponible localmente como `metropt3-cpu-diagnostic:git`, ID `sha256:d9d0c47f96909bfb53c1b5c06b2f921e29b26a78f4f1136b66cf5ce4481a2dda`. [Dockerfile.cpu](../Dockerfile.cpu) documenta su construcción desde la imagen local previa `proyecto1-web` fijada por digest; esa base no es pública ni garantiza reconstrucción en otra máquina. Las dependencias exactas instaladas se conservan en los artefactos. La imagen CUDA propuesta usa una base pública, pero su construcción y funcionamiento aún no se han verificado. Los valores `source_dirty=True` en MLflow resultaron del tratamiento CRLF del montaje Windows: con `git -c safe.directory=* -c core.autocrlf=true status --short --untracked-files=no` el árbol estaba limpio en el commit de las corridas. No se modificaron esos registros históricos.

Comandos para reproducir la prueba corta, las completas y la selección. Durante la ejecución se usó `outputs/compose.cpu.yaml`, con el mismo contenido que el archivo versionado `compose.cpu.yaml`:

```powershell
docker compose -f compose.yaml -f compose.cpu.yaml run --rm train python scripts/train_metropt3.py --device cpu --smoke
docker compose -f compose.yaml -f compose.cpu.yaml run --rm train python scripts/train_metropt3.py --device cpu --architecture RNN
docker compose -f compose.yaml -f compose.cpu.yaml run --rm train python scripts/train_metropt3.py --device cpu --architecture RNN --augmented
docker compose -f compose.yaml -f compose.cpu.yaml run --rm train python scripts/train_metropt3.py --device cpu --architecture LSTM
docker compose -f compose.yaml -f compose.cpu.yaml run --rm train python scripts/train_metropt3.py --device cpu --architecture LSTM --augmented
.\.venv\Scripts\python scripts\select_metropt3.py b1862a2da6094b568d1585fff0d3d70f b3bbe1f8dafa42e69a04770a06dbe06b 4e37271a37b24c9ca46aa3d42ffa2817 13e41b225750413a976d09f36a91ab69
.\.venv\Scripts\python -m unittest discover -s tests -v
.\.venv\Scripts\pre-commit run --all-files
```

Para abrir la UI con la imagen CPU local:

```powershell
docker compose -f compose.yaml -f compose.cpu.yaml run --rm -p 5000:5000 train mlflow ui --backend-store-uri file:///workspace/mlruns --host 0.0.0.0 --port 5000
```

Abrir http://localhost:5000. Se verificó respuesta HTTP 200. Para probar CUDA cuando termine la construcción, ejecutar los comandos de la sección Fase 03 del README; no sustituir los resultados de CPU sin identificar nuevas corridas.
