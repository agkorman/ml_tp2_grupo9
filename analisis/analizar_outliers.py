"""Revisa valores extremos en entrenamiento sin eliminar observaciones."""

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
import pandas as pd


RAIZ_PROYECTO = Path(__file__).resolve().parents[1]
ARCHIVO_TRAIN = RAIZ_PROYECTO / "data" / "processed" / "bank_train.csv"
CARPETA_DOCUMENTACION = RAIZ_PROYECTO / "documentacion"
CARPETA_TABLAS = CARPETA_DOCUMENTACION / "tablas"
CARPETA_GRAFICOS = CARPETA_DOCUMENTACION / "graficos"
VARIABLES = ("age", "campaign")
FACTOR_IQR = 1.5


def main() -> None:
    """Calcula límites IQR como diagnóstico y guarda un boxplot por variable."""
    datos = pd.read_csv(ARCHIVO_TRAIN)
    CARPETA_TABLAS.mkdir(parents=True, exist_ok=True)
    CARPETA_GRAFICOS.mkdir(parents=True, exist_ok=True)

    resumen = []
    fig, ejes = plt.subplots(1, len(VARIABLES), figsize=(12, 4.5))

    for eje, variable in zip(ejes, VARIABLES):
        valores = datos[variable]
        q1 = valores.quantile(0.25)
        q3 = valores.quantile(0.75)
        iqr = q3 - q1
        inferior = q1 - FACTOR_IQR * iqr
        superior = q3 + FACTOR_IQR * iqr
        fuera = valores.lt(inferior) | valores.gt(superior)
        respuesta_positiva = datos["y"].eq("yes")

        resumen.append(
            {
                "variable": variable,
                "q1": q1,
                "q3": q3,
                "iqr": iqr,
                "limite_inferior": inferior,
                "limite_superior": superior,
                "cantidad_fuera_iqr": int(fuera.sum()),
                "porcentaje_fuera_iqr": fuera.mean() * 100,
                "tasa_positiva_fuera_iqr": respuesta_positiva[fuera].mean(),
                "tasa_positiva_resto": respuesta_positiva[~fuera].mean(),
                "minimo": valores.min(),
                "maximo": valores.max(),
            }
        )

        eje.boxplot(valores.dropna(), vert=False, widths=0.45)
        eje.axvline(superior, color="#d95f59", linestyle="--", label="Límite IQR")
        eje.set_title(variable)
        eje.set_xlabel("Valor original")
        eje.set_yticks([])
        eje.legend(loc="lower right")

    tabla = pd.DataFrame(resumen)
    tabla.to_csv(CARPETA_TABLAS / "resumen_outliers.csv", index=False)
    fig.suptitle("Valores extremos en entrenamiento (criterio 1,5 × IQR)")
    fig.tight_layout()
    fig.savefig(CARPETA_GRAFICOS / "outliers_edad_campania.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    print(tabla[["variable", "cantidad_fuera_iqr", "porcentaje_fuera_iqr"]].to_string(index=False))
    print("El criterio IQR es descriptivo; no se eliminó ninguna fila.")


if __name__ == "__main__":
    main()
