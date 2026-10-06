"""Entrenamiento y evaluación del modelo final."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    confusion_matrix,
    recall_score,
    roc_auc_score,
)

from datos.preprocesamiento import COLUMNAS_ENTRADA
from datos.separar_datos import ARCHIVO_TEST
from modelos.ajuste_hiperparametros import plt
from modelos.clasificadores import crear_clasificadores
from modelos.validacion_cruzada import cargar_entrenamiento


RAIZ = Path(__file__).resolve().parents[1]
SALIDA = RAIZ / "resultados" / "modelo_final"
PARAMETROS_RF = {
    "clasificador__n_estimators": 100,
    "clasificador__max_depth": 12,
    "clasificador__min_samples_split": 200,
    "clasificador__min_samples_leaf": 10,
}


def crear_modelo_final():
    return crear_clasificadores(random_state=42)["RF"].set_params(**PARAMETROS_RF)


def cargar_test():
    if not ARCHIVO_TEST.exists():
        raise FileNotFoundError("Ejecutar primero: python -m datos.separar_datos")
    datos = pd.read_csv(ARCHIVO_TEST)
    faltantes = (set(COLUMNAS_ENTRADA) | {"y"}).difference(datos.columns)
    if faltantes:
        raise ValueError(f"Faltan columnas en test: {sorted(faltantes)}")
    if datos["y"].isna().any() or set(datos["y"].unique()) != {"no", "yes"}:
        raise ValueError("Test debe contener ambas clases 'no' y 'yes', sin nulos")
    return datos.drop(columns="y"), datos["y"].map({"no": 0, "yes": 1})


def main():
    X_train, y_train = cargar_entrenamiento()
    X_test, y_test = cargar_test()
    if list(X_train.columns) != list(X_test.columns):
        raise ValueError("Train y test deben tener las mismas columnas y orden")
    train = X_train.assign(y=y_train)
    test = X_test.assign(y=y_test)
    if not train.merge(test, on=list(train.columns), how="inner").empty:
        raise ValueError("Hay registros idénticos compartidos entre train y test")

    modelo = crear_modelo_final()
    print(f"Entrenando RF final con {len(y_train)} filas de train...", flush=True)
    modelo.fit(X_train, y_train)

    print(f"Evaluando configuración fija en {len(y_test)} filas de test...", flush=True)
    indice_yes = list(modelo.classes_).index(1)
    probabilidades = modelo.predict_proba(X_test)[:, indice_yes]
    predicciones = modelo.predict(X_test)
    auc = float(roc_auc_score(y_test, probabilidades))
    recall = float(recall_score(y_test, predicciones))
    matriz = confusion_matrix(y_test, predicciones, labels=[0, 1])
    tn, fp, fn, tp = (int(x) for x in matriz.ravel())

    SALIDA.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{
        "modelo": "RF", "conjunto": "test", "filas": len(y_test),
        "roc_auc": auc, "recall_yes": recall,
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
    }]).to_csv(SALIDA / "metricas.csv", index=False)
    pd.DataFrame({
        "fila_test": np.arange(len(y_test)), "real": y_test.to_numpy(),
        "prediccion": predicciones, "probabilidad_yes": probabilidades,
    }).to_csv(SALIDA / "predicciones_test.csv", index=False)
    pd.DataFrame(
        matriz, index=["real_no", "real_yes"], columns=["pred_no", "pred_yes"]
    ).to_csv(SALIDA / "matriz_confusion.csv")
    registro = {
        "modelo": "RF", "criterio_seleccion": "roc_auc",
        "parametros": modelo["clasificador"].get_params(),
        "filas_train": len(y_train), "filas_test": len(y_test),
        "positivos_train": int(y_train.sum()), "positivos_test": int(y_test.sum()),
        "metricas_test": {"roc_auc": auc, "recall_yes": recall},
        "matriz_test": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
    }
    (SALIDA / "evaluacion.json").write_text(
        json.dumps(registro, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    fig, ejes = plt.subplots(1, 2, figsize=(11, 4.5))
    ConfusionMatrixDisplay(matriz, display_labels=["no", "yes"]).plot(
        ax=ejes[0], cmap="Blues", colorbar=False, values_format="d"
    )
    ejes[0].set(
        title=f"Test: recall de yes = {recall:.2%}",
        xlabel="Predicción", ylabel="Clase real",
    )
    RocCurveDisplay.from_predictions(y_test, probabilidades, ax=ejes[1], name="RF")
    ejes[1].plot([0, 1], [0, 1], "--", color="gray", label="Azar (AUC = 0,5)")
    ejes[1].set(
        title=f"Test: ROC AUC = {auc:.4f}",
        xlabel="Tasa de falsos positivos", ylabel="Tasa de verdaderos positivos",
    )
    ejes[1].legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(SALIDA / "evaluacion_test.png", dpi=180)
    plt.close(fig)

    print(f"ROC AUC test: {auc:.4f}")
    print(f"Recall yes test: {recall:.4f} ({recall:.2%})")
    print(f"TN={tn}, FP={fp}, FN={fn}, TP={tp}")
    print(f"Resultados guardados en {SALIDA}")


if __name__ == "__main__":
    main()
