# Stand Visual v2 — Detección de Somnolencia

## Estructura
```
stand-v2/
├── stand_app.py          ← Servidor Flask (puerto 5001)
├── stand_detector.py     ← Detector con datos para visualización
├── requirements.txt
└── templates/
    ├── stand_home.html   ← Menú selector
    ├── stand_xray.html   ← Análisis facial (landmarks)
    ├── stand_conv.html   ← Pipeline de convoluciones
    ├── stand_monitor.html← Monitor de señales (gráficas)
    └── stand_sim.html    ← Simulación de auto en ciudad
```

## Requisito
El archivo `detector.py` original debe estar en la carpeta padre:
```
tu_proyecto/
├── detector.py           ← El original (NO modificar)
├── app.py                ← Servidor original (puerto 5000)
├── stand-v2/             ← Esta carpeta
│   ├── stand_app.py
│   └── ...
```

## Instalar
```
pip install -r requirements.txt
```

## Ejecutar
```
cd stand-v2
python stand_app.py
```

Abre http://127.0.0.1:5001 en el navegador.

## Cámara
Por defecto usa la cámara USB (índice 1 = BRobotix).
Para cambiar:
```
set CAMERA_INDEX=0    (Windows)
python stand_app.py
```
