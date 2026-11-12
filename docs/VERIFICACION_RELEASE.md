# Verificación local del paquete v1.0.0

El ZIP `metropt3-inference-v1.0.0.zip` tiene SHA-256 `2002146f6b2899988d0e3178262251fb8b0ccd913e183a274b2b8fb8fd91332f`. Su manifiesto identifica el commit de fase 04 `84bf8bf8c41b242cfc66c66e05b79a12c9b5169f` y el commit de entrenamiento `8db576e7cc03f20ed629c0d79679b684922ad9d0`. Se cotejaron checkpoint, escalador y esquema con el manifiesto congelado y los cuatro archivos Python exportados con las fuentes locales.

Se clonó localmente el commit `48573fd` de esta misma implementación (los cambios posteriores al clon son documentación de la prueba). Antes de instalar no existían `.venv`, `.dvc/cache`, `artifacts`, `outputs`, `mlruns` ni el dataset descargado en el clon. Solo se usaron los archivos versionados y el ZIP externo. Docker reutilizó capas de imagen locales: no se copió ninguna caché de proyecto al clon. El puerto `62773` se eligió consultando un socket libre. Comandos ejecutados en PowerShell:

```powershell
git clone --no-local --branch industrial-pressure-forecasting D:\pm25-forecasting D:\metropt3-release-check-48573fd
cd D:\metropt3-release-check-48573fd
python scripts/release_metropt3.py install D:\pm25-forecasting\outputs\release\metropt3-inference-v1.0.0.zip --sha256 2002146f6b2899988d0e3178262251fb8b0ccd913e183a274b2b8fb8fd91332f
$env:INFERENCE_PORT='62773'
docker compose -p metropt3-release-check -f compose.inference.yaml up -d --build
Invoke-WebRequest http://localhost:62773/health
python scripts/example_request.py --url http://localhost:62773
try { Invoke-WebRequest -Method Post -Uri http://localhost:62773/predict -ContentType 'application/json' -Body '{"readings":[]}' } catch { $_.Exception.Response.StatusCode.value__; $_.ErrorDetails.Message }
docker compose -p metropt3-release-check -f compose.inference.yaml down
```

Respuestas observadas:

- Salud: HTTP 200, `{"status":"ok"}`.
- Predicción: HTTP 200, `{"prediction_bar":8.821163177490234,"target":"Reservoirs","horizon_seconds":300,"anchor_timestamp":"2020-07-01T00:05:10","forecast_timestamp":"2020-07-01T00:10:10"}`. Diferencia absoluta frente al checkpoint de fase 04: 0 bar; tolerancia 0.00001 bar.
- Petición vacía: HTTP 422, `{"detail":"Se requieren exactamente 32 lecturas consecutivas"}`.

El contenedor se detuvo al terminar. No se detuvieron ni modificaron los servicios ajenos en los puertos 8000 y 5000. El modelo se cargó desde el montaje local; no se configuró conexión al tracking. La comprobación aislada de tracking de fase 04 se documenta en [VERIFICACION_FASE04.json](VERIFICACION_FASE04.json).

Las ocho pruebas pasaron en Linux con PyTorch CPU y las versiones de `requirements-dev.txt` del workflow; `pre-commit run --all-files` pasó en el entorno Windows del proyecto. La imagen de inferencia no contiene Git, por lo que no se ejecutó pre-commit dentro de esa imagen; el runner de GitHub Actions sí incluye Git. La ejecución remota está pendiente, pues no se realizó push y no hay remoto configurado.
