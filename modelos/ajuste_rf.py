"""Ajuste individual y conjunto de parámetros de Random Forest."""

import argparse
import json
from pathlib import Path

from modelos.ajuste_hiperparametros import plt
from modelos.clasificadores import crear_clasificadores
from modelos.validacion_cruzada import cargar_entrenamiento

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedKFold


CARPETA_SALIDA = Path(__file__).resolve().parents[1] / "resultados" / "ajuste_rf"
RANGOS = {
    "clasificador__max_depth": [3, 6, 12, 20, None],
    "clasificador__min_samples_split": [2, 10, 50, 100, 200],
    "clasificador__min_samples_leaf": [1, 5, 10, 25, 50],
}
METRICAS = ("roc_auc", "recall", "accuracy")


def buscar(pipeline, X, y, particiones, rangos, n_jobs):
    """Evalúa combinaciones sin entrenar el modelo final."""
    busqueda = GridSearchCV(
        pipeline, param_grid=rangos, scoring=list(METRICAS),
        cv=particiones, refit=False, return_train_score=True,
        n_jobs=n_jobs, error_score="raise", verbose=1,
    )
    busqueda.fit(X, y)
    return pd.DataFrame(busqueda.cv_results_)


def explorar_rf(X, y, k=5, random_state=42, n_jobs=1):
    """Explora parámetros y define una grilla con valores cercanos."""
    if len(X) != len(y) or not X.index.equals(y.index):
        raise ValueError("X e y deben tener igual longitud e índices alineados")
    if y.isna().any() or set(y.unique()) != {0, 1}:
        raise ValueError("y debe contener las clases 0 y 1, sin nulos")
    if k < 2 or k > int(y.value_counts().min()):
        raise ValueError("k debe ser al menos 2 y no superar la clase menos frecuente")
    cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=random_state)
    particiones = list(cv.split(X, y))
    pipeline = crear_clasificadores(random_state)["RF"]
    resumen, detalle, rangos_grid = [], [], {}

    for parametro, valores in RANGOS.items():
        print(f"Curva: {parametro.split('__')[-1]} = {valores}", flush=True)
        resultados = buscar(pipeline, X, y, particiones, {parametro: valores}, n_jobs)
        for indice, valor in enumerate(valores):
            fila = resultados.iloc[indice]
            for metrica in METRICAS:
                train = np.array([fila[f"split{fold}_train_{metrica}"] for fold in range(k)])
                validacion = np.array([fila[f"split{fold}_test_{metrica}"] for fold in range(k)])
                resumen.append({
                    "parametro": parametro, "valor": "sin_limite" if valor is None else str(valor),
                    "metrica": metrica, "train_media": train.mean(),
                    "train_desvio": train.std(ddof=1),
                    "validacion_media": validacion.mean(),
                    "validacion_desvio": validacion.std(ddof=1),
                    "brecha": train.mean() - validacion.mean(),
                })
                for fold in range(k):
                    detalle.append({
                        "parametro": parametro, "valor": resumen[-1]["valor"],
                        "metrica": metrica, "fold": fold + 1,
                        "train": train[fold], "validacion": validacion[fold],
                    })
        mejor = int(resultados["mean_test_roc_auc"].to_numpy().argmax())
        # Incluye valores cercanos al mejor para evaluar combinaciones.
        inicio = max(0, min(mejor - 1, len(valores) - 3))
        rangos_grid[parametro] = valores[inicio:inicio + 3]

    print(f"Grid conjunta: {rangos_grid}", flush=True)
    grid = buscar(pipeline, X, y, particiones, rangos_grid, n_jobs)
    if not np.isfinite(grid[[f"mean_test_{m}" for m in METRICAS]]).all().all():
        raise ValueError("La búsqueda produjo puntajes no finitos")
    mejor = grid.loc[grid["mean_test_roc_auc"].idxmax()]
    seleccion = {
        "parametros": mejor["params"],
        "criterio": "roc_auc", "folds": k, "semilla": random_state,
        "rangos_grid": rangos_grid,
        "puntajes": {
            metrica: {
                "train_media": float(mejor[f"mean_train_{metrica}"]),
                "validacion_media": float(mejor[f"mean_test_{metrica}"]),
                "validacion_desvio": float(np.std(
                    [mejor[f"split{fold}_test_{metrica}"] for fold in range(k)], ddof=1
                )),
            }
            for metrica in METRICAS
        },
    }
    return pd.DataFrame(detalle), pd.DataFrame(resumen), grid, seleccion


def graficar_rf(resumen, salida):
    salida.mkdir(parents=True, exist_ok=True)
    for parametro, tabla in resumen.groupby("parametro", sort=False):
        nombre = parametro.split("__")[-1]
        fig, ejes = plt.subplots(1, 2, figsize=(12, 4.5))
        for ax, metrica in zip(ejes, ("accuracy", "roc_auc")):
            curva = tabla.loc[tabla["metrica"] == metrica]
            etiquetas = curva["valor"].str.replace("sin_limite", "sin límite").tolist()
            x = (np.arange(len(curva)) if nombre == "max_depth"
                 else curva["valor"].astype(float).to_numpy())
            for conjunto, etiqueta, color in (
                ("train", "Entrenamiento", "#4c78a8"),
                ("validacion", "Validación", "#f58518"),
            ):
                media = curva[f"{conjunto}_media"].to_numpy()
                desvio = curva[f"{conjunto}_desvio"].to_numpy()
                ax.plot(x, media, "o-", label=etiqueta, color=color)
                ax.fill_between(x, media - desvio, media + desvio, color=color, alpha=0.15)
            if nombre != "max_depth":
                ax.set_xscale("log")
                ax.minorticks_off()
            ax.set_xticks(x, etiquetas)
            escala = "valores evaluados" if nombre == "max_depth" else "escala log"
            ax.set_xlabel(f"{nombre} ({escala})")
            ax.set_ylabel(metrica)
            ax.grid(alpha=0.25)
            ax.legend()
        fig.suptitle(f"Random Forest: {nombre}")
        fig.tight_layout()
        fig.savefig(salida / f"curva_{nombre}.png", dpi=180)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--semilla", type=int, default=42)
    parser.add_argument("--n-jobs", type=int, default=1)
    args = parser.parse_args()
    X, y = cargar_entrenamiento()
    detalle, resumen, grid, seleccion = explorar_rf(
        X, y, k=args.folds, random_state=args.semilla, n_jobs=args.n_jobs,
    )
    CARPETA_SALIDA.mkdir(parents=True, exist_ok=True)
    detalle.to_csv(CARPETA_SALIDA / "curvas_por_fold.csv", index=False)
    resumen.to_csv(CARPETA_SALIDA / "resumen_curvas.csv", index=False)
    grid.to_csv(CARPETA_SALIDA / "grid_resultados.csv", index=False)
    (CARPETA_SALIDA / "mejor_configuracion.json").write_text(
        json.dumps(seleccion, indent=2, ensure_ascii=False) + "\n"
    )
    graficar_rf(resumen, CARPETA_SALIDA)
    print(json.dumps(seleccion, indent=2, ensure_ascii=False))
    print(f"Resultados guardados en {CARPETA_SALIDA}")


if __name__ == "__main__":
    main()
