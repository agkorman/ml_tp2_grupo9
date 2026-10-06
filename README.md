# TP2 - Clasificación supervisada

Dataset: Bank Marketing. Ejecutar los comandos desde la raíz del repositorio.

## Instalación

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Datos y análisis exploratorio

```bash
python -m datos.exportar_dataset
python -m datos.separar_datos
python -m analisis.eda_inicial
python -m analisis.analizar_outliers
```

## Modelos y evaluación

Ejecutar en este orden:

```bash
python -m modelos.validacion_cruzada --metricas roc_auc recall
python -m modelos.ajuste_hiperparametros --metrica roc_auc
python -m modelos.ajuste_rf
python -m modelos.ajuste_svm
python -m modelos.evaluacion_metricas
python -m modelos.modelo_final
```

El modelo final usa la configuración fijada en `modelos/modelo_final.py`.
Con los datos preparados, ese script también se puede ejecutar por separado.
