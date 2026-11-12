# Fase 03: estado de ejecución

La implementación y configuración se fijaron en el commit `2fdd4c0`, basado en `1d832353ccf874b7c518a79aef91fd9120c39c67`. La reconstrucción local confirmó 440342 casos de entrenamiento, 405173 de validación y MAE de persistencia de validación 0.5451199413583827 bar (0.545120 a seis decimales). Las dos variantes se extraen del mismo objeto de índices y etiquetas. No se evaluó prueba desde 2020-06-01.

Docker Desktop expone la RTX 5060 a un contenedor: `docker run --rm --gpus all --entrypoint sh proyecto1-web:latest -c 'nvidia-smi'` la identifica. La imagen ya disponible `proyecto1-web:latest` contiene PyTorch 2.7.1 **solo CPU**, con `torch.version.cuda=None` y `torch.cuda.is_available()=False`; no demuestra ejecución CUDA. Se está construyendo la imagen específica CUDA 12.8 de este proyecto.

Como referencia diagnóstica independiente, un contenedor CPU con cuatro hilos ejecutó 100 lotes sintéticos de 512 x 30 x 6: RNN 0.95 s y LSTM 2.17 s. Una extrapolación lineal a cinco épocas de entrenamiento arroja 40.7 y 93.2 s respectivamente por variante, sin incluir lectura, validación, seguimiento, diferencias de carga ni sobrecostos; no son tiempos medidos de las corridas reales.

Los resultados, la verificación CUDA por PyTorch, los identificadores y la selección deben añadirse cuando concluyan las corridas. No se han capturado ni verificado métricas de utilización de sistema.
