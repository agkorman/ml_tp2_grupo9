"""Curvas de validación para SVM, KNN y RF."""

import argparse
import os
from pathlib import Path
from tempfile import gettempdir

_cache_graficos = Path(gettempdir()) / "ml_tp2_graficos_cache"
_cache_graficos.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_cache_graficos / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(_cache_graficos))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import get_scorer
from sklearn.model_selection import StratifiedKFold, validation_curve

from modelos.clasificadores import crear_clasificadores
from modelos.validacion_cruzada import cargar_entrenamiento


CARPETA_SALIDA = Path(__file__).resolve().parents[1] / "resultados" / "hiperparametros"
PARAMETROS = {
    "SVM": ("clasificador__C", [0.01, 0.1, 1.0, 10.0]),
    "KNN": ("clasificador__n_neighbors", [1, 5, 15, 31]),
    "RF": ("clasificador__max_depth", [3, 6, 12, None]),
}


def evaluar_hiperparametros(
    X: pd.DataFrame,
    y: pd.Series,
    k: int = 5,
    random_state: int = 42,
    metrica: str = "accuracy",
    n_jobs: int = 1,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Selecciona el valor con mayor promedio de validación."""
    if len(X) != len(y) or not X.index.equals(y.index):
        raise ValueError("X e y deben tener igual longitud e índices alineados")
    if y.isna().any() or set(y.unique()) != {0, 1}:
        raise ValueError("y debe contener ambas clases, codificadas como 0 y 1")
    if k < 2 or k > int(y.value_counts().min()):
        raise ValueError("k debe ser al menos 2 y no superar la clase menos frecuente")
    scoring = get_scorer(metrica)
    cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=random_state)
    particiones = list(cv.split(X, y))
    if max(PARAMETROS["KNN"][1]) > min(len(train) for train, _ in particiones):
        raise ValueError("Hay menos filas por fold que vecinos a evaluar")

    pipelines = crear_clasificadores(random_state)
    detalle, resumen, mejores = [], [], []
    for nombre, (parametro, valores) in PARAMETROS.items():
        print(f"Evaluando {nombre}: {parametro.split('__')[-1]} = {valores}", flush=True)
        train_scores, val_scores = validation_curve(
            pipelines[nombre], X, y,
            param_name=parametro, param_range=valores,
            cv=particiones, scoring=scoring, n_jobs=n_jobs,
            error_score="raise",
        )
        for indice, valor in enumerate(valores):
            etiqueta = "sin_limite" if valor is None else str(valor)
            for fold in range(k):
                detalle.append({
                    "modelo": nombre, "parametro": parametro, "valor": etiqueta,
                    "fold": fold + 1, "metrica": metrica,
                    "train": train_scores[indice, fold],
                    "validacion": val_scores[indice, fold],
                })
            resumen.append({
                "modelo": nombre, "parametro": parametro, "valor": etiqueta,
                "metrica": metrica,
                "train_media": train_scores[indice].mean(),
                "train_desvio": train_scores[indice].std(ddof=1),
                "validacion_media": val_scores[indice].mean(),
                "validacion_desvio": val_scores[indice].std(ddof=1),
                "brecha": train_scores[indice].mean() - val_scores[indice].mean(),
            })
        # En caso de empate, queda el primer valor.
        mejor = int(np.argmax(val_scores.mean(axis=1)))
        mejores.append(resumen[-len(valores) + mejor].copy())

    return pd.DataFrame(detalle), pd.DataFrame(resumen), pd.DataFrame(mejores)


def graficar_curvas(resumen: pd.DataFrame, salida: Path) -> None:
    """Grafica train y validación con sus desvíos."""
    salida.mkdir(parents=True, exist_ok=True)
    for nombre, tabla in resumen.groupby("modelo", sort=False):
        parametro = tabla["parametro"].iloc[0].split("__")[-1]
        metrica = tabla["metrica"].iloc[0]
        valores = tabla["valor"].tolist()
        x = np.arange(len(valores)) if nombre == "RF" else np.array(valores, dtype=float)
        fig, ax = plt.subplots(figsize=(8, 5))
        for conjunto, etiqueta, color in (
            ("train", "Entrenamiento", "#4c78a8"),
            ("validacion", "Validación", "#f58518"),
        ):
            media = tabla[f"{conjunto}_media"].to_numpy()
            desvio = tabla[f"{conjunto}_desvio"].to_numpy()
            ax.plot(x, media, "o-", label=etiqueta, color=color)
            ax.fill_between(x, media - desvio, media + desvio, alpha=0.15, color=color)

        mejor = int(np.argmax(tabla["validacion_media"].to_numpy()))
        ax.scatter(x[mejor], tabla["validacion_media"].iloc[mejor],
                   s=160, facecolors="none", edgecolors="#333333", zorder=5,
                   label=f"Mejor valor: {valores[mejor].replace('sin_limite', 'sin límite')}")
        if nombre == "SVM":
            ax.set_xscale("log")
        ax.set_xticks(x, [valor.replace("sin_limite", "sin límite") for valor in valores])
        ax.set_xlabel(parametro)
        ax.set_ylabel(metrica)
        ax.set_title(f"{nombre}: curva de validación")
        ax.grid(alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(salida / f"curva_{nombre.lower()}.png", dpi=180)
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--semilla", type=int, default=42)
    parser.add_argument("--metrica", default="accuracy")
    parser.add_argument("--n-jobs", type=int, default=1)
    args = parser.parse_args()

    X, y = cargar_entrenamiento()
    print(f"Entrenamiento: {len(y)} filas; métrica: {args.metrica}", flush=True)
    detalle, resumen, mejores = evaluar_hiperparametros(
        X, y, k=args.folds, random_state=args.semilla,
        metrica=args.metrica, n_jobs=args.n_jobs,
    )
    salida = CARPETA_SALIDA / args.metrica
    salida.mkdir(parents=True, exist_ok=True)
    detalle.to_csv(salida / "puntajes_por_fold.csv", index=False)
    resumen.to_csv(salida / "resumen_curvas.csv", index=False)
    mejores.to_csv(salida / "mejores_parametros.csv", index=False)
    graficar_curvas(resumen, salida)
    print(mejores.to_string(index=False, float_format=lambda valor: f"{valor:.4f}"))
    print(f"Resultados guardados en {salida}")


if __name__ == "__main__":
    main()
