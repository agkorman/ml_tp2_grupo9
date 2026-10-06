"""Ajuste de C, gamma y kernel de SVM."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from modelos.ajuste_hiperparametros import plt
from modelos.clasificadores import crear_clasificadores
from modelos.validacion_cruzada import cargar_entrenamiento


SALIDA = Path(__file__).resolve().parents[1] / "resultados" / "ajuste_svm"
VALORES_C = [0.01, 0.1, 1.0, 10.0]
VALORES_GAMMA = [0.001, 0.01, 0.1, "scale"]
GRILLA = [
    {"clasificador__kernel": ["rbf"], "clasificador__C": VALORES_C,
     "clasificador__gamma": VALORES_GAMMA},
    {"clasificador__kernel": ["linear"], "clasificador__C": VALORES_C},
]
METRICAS = ("roc_auc", "recall", "accuracy")


def explorar_svm(X, y, k=5, semilla=42, n_jobs=1):
    """Compara kernels y parámetros con los mismos folds."""
    if len(X) != len(y) or not X.index.equals(y.index):
        raise ValueError("X e y deben tener igual longitud e índices alineados")
    if y.isna().any() or set(y.unique()) != {0, 1}:
        raise ValueError("y debe contener ambas clases, sin nulos")
    if k < 2 or k > int(y.value_counts().min()):
        raise ValueError("k debe ser al menos 2 y no superar la clase menos frecuente")
    cv = list(StratifiedKFold(k, shuffle=True, random_state=semilla).split(X, y))
    busqueda = GridSearchCV(
        crear_clasificadores(semilla)["SVM"], param_grid=GRILLA,
        scoring=list(METRICAS), cv=cv, refit=False, return_train_score=True,
        n_jobs=n_jobs, pre_dispatch="n_jobs", error_score="raise", verbose=2,
    )
    busqueda.fit(X, y)
    grid = pd.DataFrame(busqueda.cv_results_)
    if not np.isfinite(grid[[f"mean_test_{m}" for m in METRICAS]]).all().all():
        raise ValueError("La búsqueda produjo puntajes no finitos")
    detalle, resumen = [], []
    for candidato, fila in grid.iterrows():
        params = fila["params"]
        base = {
            "candidato": candidato + 1,
            "kernel": params["clasificador__kernel"],
            "C": params["clasificador__C"],
            "gamma": str(params.get("clasificador__gamma", "no_aplica")),
        }
        for metrica in METRICAS:
            train = np.array([fila[f"split{i}_train_{metrica}"] for i in range(k)])
            val = np.array([fila[f"split{i}_test_{metrica}"] for i in range(k)])
            resumen.append({
                **base, "metrica": metrica,
                "train_media": train.mean(), "train_desvio": train.std(ddof=1),
                "validacion_media": val.mean(), "validacion_desvio": val.std(ddof=1),
                "brecha": train.mean() - val.mean(),
            })
            for i in range(k):
                detalle.append({**base, "metrica": metrica, "fold": i + 1,
                                "train": train[i], "validacion": val[i]})
    # En caso de empate, queda la primera combinación.
    mejor = int(grid["mean_test_roc_auc"].to_numpy().argmax())
    resumen = pd.DataFrame(resumen)
    seleccion = {
        "parametros": grid.iloc[mejor]["params"], "criterio": "roc_auc",
        "folds": k, "semilla": semilla, "filas_train": len(y), "grilla": GRILLA,
        "puntajes": {
            row.metrica: {key: float(getattr(row, key)) for key in
                          ("train_media", "validacion_media", "validacion_desvio", "brecha")}
            for row in resumen.loc[resumen.candidato == mejor + 1].itertuples()
        },
    }
    return pd.DataFrame(detalle), resumen, grid, seleccion


def graficar(resumen, salida):
    """Grafica curvas de C, la grilla RBF y los mejores kernels."""
    salida.mkdir(parents=True, exist_ok=True)
    auc = resumen.loc[resumen.metrica == "roc_auc"]
    fig, ejes = plt.subplots(2, 3, figsize=(15, 8), sharey=True)
    grupos = [("rbf", str(g)) for g in VALORES_GAMMA] + [("linear", "no_aplica")]
    for ax, (kernel, gamma) in zip(ejes.flat, grupos):
        curva = auc.loc[(auc.kernel == kernel) & (auc.gamma == gamma)].sort_values("C")
        x = curva.C.to_numpy()
        for conjunto, etiqueta, color in (
            ("train", "Entrenamiento", "#4c78a8"),
            ("validacion", "Validación", "#f58518"),
        ):
            media = curva[f"{conjunto}_media"].to_numpy()
            desvio = curva[f"{conjunto}_desvio"].to_numpy()
            ax.plot(x, media, "o-", label=etiqueta, color=color)
            ax.fill_between(x, media - desvio, media + desvio, alpha=0.15, color=color)
        ax.set_xscale("log")
        ax.set_xticks(VALORES_C, [str(c) for c in VALORES_C])
        ax.set_xlabel("C")
        ax.set_ylabel("ROC AUC")
        ax.set_title(f"RBF: gamma={gamma}" if kernel == "rbf" else "Kernel lineal")
        ax.grid(alpha=0.25)
        ax.legend()
    ejes.flat[-1].set_visible(False)
    fig.suptitle("SVM: curvas de validación de C (± un desvío)")
    fig.tight_layout()
    fig.savefig(salida / "curvas_c.png", dpi=180)
    plt.close(fig)

    tabla = auc.loc[auc.kernel == "rbf"].pivot(index="gamma", columns="C", values="validacion_media")
    tabla = tabla.reindex(index=[str(g) for g in VALORES_GAMMA], columns=VALORES_C)
    fig, ax = plt.subplots(figsize=(8, 5))
    mapa = ax.imshow(tabla.to_numpy(), cmap="viridis", aspect="auto")
    for (i, j), valor in np.ndenumerate(tabla.to_numpy()):
        ax.text(j, i, f"{valor:.4f}", ha="center", va="center", color="white",
                bbox={"facecolor": "black", "alpha": 0.3, "edgecolor": "none"})
    i, j = np.unravel_index(np.argmax(tabla.to_numpy()), tabla.shape)
    ax.scatter(j, i, marker="s", s=2200, facecolors="none", edgecolors="red", linewidths=2)
    ax.set_xticks(range(len(VALORES_C)), [str(c) for c in VALORES_C])
    ax.set_yticks(range(len(VALORES_GAMMA)), [str(g) for g in VALORES_GAMMA])
    ax.set_xlabel("C")
    ax.set_ylabel("gamma (valores evaluados)")
    ax.set_title("SVM RBF: ROC AUC medio de validación\nMejor combinación RBF en rojo")
    fig.colorbar(mapa, ax=ax, label="ROC AUC")
    fig.tight_layout()
    fig.savefig(salida / "heatmap_rbf.png", dpi=180)
    plt.close(fig)

    mejores = auc.loc[auc.groupby("kernel").validacion_media.idxmax()]
    mejores.to_csv(salida / "mejores_por_kernel.csv", index=False)
    fig, ax = plt.subplots(figsize=(7, 5))
    x = np.arange(len(mejores))
    ax.bar(x, mejores.validacion_media, yerr=mejores.validacion_desvio, capsize=5)
    ax.set_xticks(x, [f"{row.kernel}\nC={row.C}" +
                     (f"; gamma={row.gamma}" if row.kernel == "rbf" else "")
                     for row in mejores.itertuples()])
    for i, row in enumerate(mejores.itertuples()):
        ax.text(i, row.validacion_media + row.validacion_desvio + 0.015,
                f"{row.validacion_media:.4f}", ha="center")
    ax.set_ylim(0, 1)
    ax.set_ylabel("ROC AUC")
    ax.set_title("SVM: mejor configuración de cada kernel\nMedia de validación ± un desvío")
    fig.tight_layout()
    fig.savefig(salida / "comparacion_kernels.png", dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--semilla", type=int, default=42)
    parser.add_argument("--n-jobs", type=int, default=1)
    args = parser.parse_args()
    X, y = cargar_entrenamiento()
    detalle, resumen, grid, seleccion = explorar_svm(
        X, y, k=args.folds, semilla=args.semilla, n_jobs=args.n_jobs,
    )
    SALIDA.mkdir(parents=True, exist_ok=True)
    detalle.to_csv(SALIDA / "puntajes_por_fold.csv", index=False)
    resumen.to_csv(SALIDA / "resumen.csv", index=False)
    grid.to_csv(SALIDA / "grid_resultados.csv", index=False)
    (SALIDA / "mejor_configuracion.json").write_text(
        json.dumps(seleccion, indent=2, ensure_ascii=False) + "\n"
    )
    graficar(resumen, SALIDA)
    print(json.dumps(seleccion, indent=2, ensure_ascii=False))
    print(f"Resultados guardados en {SALIDA}")


if __name__ == "__main__":
    main()
