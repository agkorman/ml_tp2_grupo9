"""Validación cruzada de los clasificadores sobre train."""

import argparse
from pathlib import Path

import pandas as pd
from sklearn.metrics import get_scorer
from sklearn.model_selection import StratifiedKFold, cross_validate

from datos.preprocesamiento import COLUMNAS_ENTRADA
from modelos.clasificadores import crear_clasificadores


RAIZ_PROYECTO = Path(__file__).resolve().parents[1]
ARCHIVO_TRAIN = RAIZ_PROYECTO / "data" / "processed" / "bank_train.csv"
CARPETA_SALIDA = RAIZ_PROYECTO / "resultados" / "validacion_cruzada"


def cargar_entrenamiento() -> tuple[pd.DataFrame, pd.Series]:
    """Carga train y separa las variables del objetivo."""
    if not ARCHIVO_TRAIN.exists():
        raise FileNotFoundError(
            f"No existe {ARCHIVO_TRAIN}. Ejecutar primero "
            "datos/exportar_dataset.py y datos/separar_datos.py."
        )
    datos = pd.read_csv(ARCHIVO_TRAIN)
    faltantes = (set(COLUMNAS_ENTRADA) | {"y"}).difference(datos.columns)
    if faltantes:
        raise ValueError(f"Faltan columnas requeridas: {sorted(faltantes)}")
    if datos["y"].isna().any() or set(datos["y"].unique()) != {"no", "yes"}:
        raise ValueError("La variable objetivo debe contener 'no' y 'yes', sin nulos")
    return datos.drop(columns="y"), datos["y"].map({"no": 0, "yes": 1})


def evaluar_clasificadores(
    X: pd.DataFrame,
    y: pd.Series,
    k: int = 5,
    random_state: int = 42,
    metricas: tuple[str, ...] = ("accuracy",),
    n_jobs: int = 1,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Evalúa los modelos con los mismos folds."""
    if len(X) != len(y) or not X.index.equals(y.index):
        raise ValueError("X e y deben tener igual longitud e índices alineados")
    if y.isna().any() or set(y.unique()) != {0, 1}:
        raise ValueError("y debe contener ambas clases, codificadas como 0 y 1")
    if k < 2 or k > int(y.value_counts().min()):
        raise ValueError("k debe ser al menos 2 y no superar la clase menos frecuente")
    if not metricas or len(set(metricas)) != len(metricas):
        raise ValueError("Indicar al menos una métrica y no repetir nombres")
    scoring = {metrica: get_scorer(metrica) for metrica in metricas}

    cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=random_state)
    particiones = list(cv.split(X, y))
    composicion = pd.DataFrame(
        [
            {
                "fold": fold,
                "filas_train": len(train),
                "filas_validacion": len(validacion),
                "positivos_train": int(y.iloc[train].sum()),
                "positivos_validacion": int(y.iloc[validacion].sum()),
            }
            for fold, (train, validacion) in enumerate(particiones, start=1)
        ]
    )
    filas = []
    for nombre, pipeline in crear_clasificadores(random_state).items():
        print(f"Evaluando {nombre} ({k} folds)...", flush=True)
        resultado = cross_validate(
            pipeline,
            X,
            y,
            cv=particiones,
            scoring=scoring,
            n_jobs=n_jobs,
            return_train_score=False,
            error_score="raise",
        )
        for fold in range(k):
            filas.append(
                {
                    "modelo": nombre,
                    "fold": fold + 1,
                    "tiempo_ajuste_s": resultado["fit_time"][fold],
                    "tiempo_evaluacion_s": resultado["score_time"][fold],
                    **{
                        metrica: resultado[f"test_{metrica}"][fold]
                        for metrica in metricas
                    },
                }
            )
    detalle = pd.DataFrame(filas)
    resumen = detalle.groupby("modelo", sort=False)[list(metricas)].agg(["mean", "std"])
    resumen.columns = [
        f"{metrica}_{'media' if estadistico == 'mean' else 'desvio'}"
        for metrica, estadistico in resumen.columns
    ]
    return detalle, resumen.reset_index(), composicion


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--semilla", type=int, default=42)
    parser.add_argument("--metricas", nargs="+", default=["accuracy"])
    parser.add_argument(
        "--n-jobs", type=int, default=1,
        help="Folds en paralelo; -1 usa todos los núcleos (mayor uso de memoria)",
    )
    args = parser.parse_args()
    X, y = cargar_entrenamiento()
    print(f"Entrenamiento: {len(y)} filas; positivos: {y.mean():.2%}", flush=True)
    detalle, resumen, composicion = evaluar_clasificadores(
        X, y, k=args.folds, random_state=args.semilla,
        metricas=tuple(args.metricas), n_jobs=args.n_jobs,
    )
    CARPETA_SALIDA.mkdir(parents=True, exist_ok=True)
    detalle.to_csv(CARPETA_SALIDA / "puntajes_por_fold.csv", index=False)
    resumen.to_csv(CARPETA_SALIDA / "resumen.csv", index=False)
    composicion.to_csv(CARPETA_SALIDA / "particiones.csv", index=False)
    print(resumen.to_string(index=False, float_format=lambda valor: f"{valor:.4f}"))
    print(f"Resultados guardados en {CARPETA_SALIDA}")
    print("El test reservado no se utilizó. Los modelos no se ajustaron sobre todo train.")


if __name__ == "__main__":
    main()
