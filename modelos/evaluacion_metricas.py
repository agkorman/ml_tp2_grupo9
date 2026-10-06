"""Comparación mediante ROC AUC, recall y matrices de confusión."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_validate

from modelos.ajuste_hiperparametros import plt
from modelos.clasificadores import crear_clasificadores
from modelos.validacion_cruzada import cargar_entrenamiento


RAIZ = Path(__file__).resolve().parents[1]
SALIDA = RAIZ / "resultados" / "metricas"


def cargar_candidatos(semilla=42):
    pipelines = crear_clasificadores(semilla)
    archivo = RAIZ / "resultados" / "hiperparametros" / "roc_auc" / "mejores_parametros.csv"
    mejores = pd.read_csv(archivo)
    fila = mejores.loc[mejores["modelo"] == "KNN"].iloc[0]
    pipelines["KNN"].set_params(**{fila["parametro"]: int(float(fila["valor"]))})
    ajuste_svm = json.loads((RAIZ / "resultados" / "ajuste_svm" / "mejor_configuracion.json").read_text())
    pipelines["SVM"].set_params(**ajuste_svm["parametros"])
    ajuste_rf = json.loads((RAIZ / "resultados" / "ajuste_rf" / "mejor_configuracion.json").read_text())
    pipelines["RF"].set_params(**ajuste_rf["parametros"])
    return pipelines


def evaluar(X, y, pipelines, k=5, semilla=42, n_jobs=1):
    if len(X) != len(y) or not X.index.equals(y.index):
        raise ValueError("X e y deben tener igual longitud e índices alineados")
    if y.isna().any() or set(y.unique()) != {0, 1}:
        raise ValueError("y debe contener ambas clases, sin nulos")
    if k < 2 or k > int(y.value_counts().min()):
        raise ValueError("k debe ser al menos 2 y no superar la clase menos frecuente")
    cv = list(StratifiedKFold(k, shuffle=True, random_state=semilla).split(X, y))
    detalle, predicciones, matrices = [], [], {}
    for nombre, pipeline in pipelines.items():
        print(f"Evaluando {nombre}: ROC AUC y recall", flush=True)
        resultado = cross_validate(
            pipeline, X, y, cv=cv, scoring=["roc_auc", "recall"],
            return_estimator=True, n_jobs=n_jobs, error_score="raise",
        )
        oof = np.full(len(y), -1, dtype=int)
        folds = np.zeros(len(y), dtype=int)
        for fold, ((_, validacion), ajustado) in enumerate(zip(cv, resultado["estimator"]), start=1):
            oof[validacion] = ajustado.predict(X.iloc[validacion])
            folds[validacion] = fold
            detalle.append({
                "modelo": nombre, "fold": fold,
                "roc_auc": resultado["test_roc_auc"][fold - 1],
                "recall": resultado["test_recall"][fold - 1],
            })
        if (oof < 0).any():
            raise ValueError("Faltan predicciones de validación")
        matrices[nombre] = confusion_matrix(y, oof, labels=[0, 1])
        predicciones.append(pd.DataFrame({
            "modelo": nombre, "fila_train": np.arange(len(y)), "fold": folds,
            "real": y.to_numpy(), "prediccion": oof,
        }))
    detalle = pd.DataFrame(detalle)
    resumen = detalle.groupby("modelo", sort=False)[["roc_auc", "recall"]].agg(["mean", "std"])
    resumen.columns = [f"{m}_{'media' if e == 'mean' else 'desvio'}" for m, e in resumen.columns]
    return detalle, resumen.reset_index(), pd.concat(predicciones, ignore_index=True), matrices


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--semilla", type=int, default=42)
    parser.add_argument("--n-jobs", type=int, default=1)
    args = parser.parse_args()
    X, y = cargar_entrenamiento()
    pipelines = cargar_candidatos(args.semilla)
    detalle, resumen, predicciones, matrices = evaluar(
        X, y, pipelines, k=args.folds, semilla=args.semilla, n_jobs=args.n_jobs,
    )
    SALIDA.mkdir(parents=True, exist_ok=True)
    detalle.to_csv(SALIDA / "puntajes_por_fold.csv", index=False)
    resumen.to_csv(SALIDA / "resumen.csv", index=False)
    predicciones.to_csv(SALIDA / "predicciones_validacion.csv", index=False)
    fig, ejes = plt.subplots(2, 2, figsize=(10, 8))
    for ax, (nombre, matriz) in zip(ejes.flat, matrices.items()):
        ConfusionMatrixDisplay(matriz, display_labels=["no", "yes"]).plot(
            ax=ax, cmap="Blues", colorbar=False, values_format="d",
        )
        ax.set_title(nombre)
        ax.set_xlabel("Predicción")
        ax.set_ylabel("Clase real")
        pd.DataFrame(matriz, index=["real_no", "real_yes"], columns=["pred_no", "pred_yes"]).to_csv(
            SALIDA / f"matriz_{nombre.lower().replace(' ', '_')}.csv"
        )
    fig.tight_layout()
    fig.savefig(SALIDA / "matrices_confusion.png", dpi=180)
    plt.close(fig)
    candidato = resumen.sort_values("roc_auc_media", ascending=False).iloc[0]["modelo"]
    decision = {
        "candidato": candidato, "criterio": "roc_auc", "folds": args.folds,
        "semilla": args.semilla,
        "parametros": {nombre: pipeline["clasificador"].get_params() for nombre, pipeline in pipelines.items()},
    }
    (SALIDA / "decision.json").write_text(json.dumps(decision, indent=2, ensure_ascii=False) + "\n")
    print(resumen.to_string(index=False, float_format=lambda valor: f"{valor:.4f}"))
    print(f"Candidato por ROC AUC: {candidato}. Revisar recall y matrices antes del modelo final.")


if __name__ == "__main__":
    main()
