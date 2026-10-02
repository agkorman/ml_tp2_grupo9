"""EDA inicial sobre entrenamiento: calidad, clases y relaciones con el objetivo."""

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
import seaborn as sns


RAIZ_PROYECTO = Path(__file__).resolve().parents[1]
ARCHIVO_TRAIN = RAIZ_PROYECTO / "data" / "processed" / "bank_train.csv"
CARPETA_DOCUMENTACION = RAIZ_PROYECTO / "documentacion"
CARPETA_SALIDA = CARPETA_DOCUMENTACION / "tablas"
CARPETA_GRAFICOS = CARPETA_DOCUMENTACION / "graficos"
VARIABLE_OBJETIVO = "y"
UMBRAL_CATEGORIA_RARA = 0.01


def guardar_grafico(fig: plt.Figure, nombre: str) -> None:
    fig.tight_layout()
    fig.savefig(CARPETA_GRAFICOS / nombre, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    """Genera resúmenes y gráficos reproducibles sin acceder al conjunto test."""
    datos = pd.read_csv(ARCHIVO_TRAIN)
    if VARIABLE_OBJETIVO not in datos:
        raise ValueError(f"Falta la columna {VARIABLE_OBJETIVO}")
    CARPETA_SALIDA.mkdir(parents=True, exist_ok=True)
    CARPETA_GRAFICOS.mkdir(parents=True, exist_ok=True)

    balance = datos[VARIABLE_OBJETIVO].value_counts().reindex(["no", "yes"])
    balance.rename_axis("clase").reset_index(name="cantidad").assign(
        porcentaje=lambda tabla: tabla["cantidad"] / len(datos) * 100
    ).to_csv(CARPETA_SALIDA / "balance_clases.csv", index=False)

    fig, ax = plt.subplots(figsize=(6, 4))
    barras = ax.bar(balance.index, balance.values, color=["#4c78a8", "#f58518"])
    ax.bar_label(barras, labels=[f"{valor / len(datos):.1%}" for valor in balance])
    ax.set_ylabel("Cantidad de clientes")
    ax.set_title("Balance de clases en entrenamiento")
    guardar_grafico(fig, "balance_clases.png")

    columnas_categoricas = datos.select_dtypes(include=["object", "category"]).columns.drop(
        VARIABLE_OBJETIVO
    )
    calidad = []
    categorias_raras = []
    for columna in datos.columns:
        serie = datos[columna]
        calidad.append(
            {
                "variable": columna,
                "tipo": str(serie.dtype),
                "nulos_reales": int(serie.isna().sum()),
                "unknown": int(serie.eq("unknown").sum()) if columna in columnas_categoricas else 0,
                "valores_distintos": int(serie.nunique(dropna=True)),
            }
        )
        if columna in columnas_categoricas:
            frecuencias = serie.value_counts(normalize=True, dropna=False)
            for categoria, proporcion in frecuencias.items():
                if proporcion < UMBRAL_CATEGORIA_RARA:
                    categorias_raras.append(
                        {
                            "variable": columna,
                            "categoria": categoria,
                            "porcentaje": proporcion * 100,
                        }
                    )
    pd.DataFrame(calidad).to_csv(CARPETA_SALIDA / "calidad_variables.csv", index=False)
    pd.DataFrame(
        categorias_raras, columns=["variable", "categoria", "porcentaje"]
    ).to_csv(CARPETA_SALIDA / "categorias_raras.csv", index=False)
    datos.select_dtypes(include="number").describe().T.to_csv(
        CARPETA_SALIDA / "resumen_numericas.csv"
    )

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for clase, color in (("no", "#4c78a8"), ("yes", "#f58518")):
        datos.loc[datos[VARIABLE_OBJETIVO] == clase, "age"].hist(
            bins=25, density=True, alpha=0.45, label=clase, color=color, ax=ax
        )
    ax.set_xlabel("Edad")
    ax.set_ylabel("Densidad")
    ax.set_title("Distribución de edad según respuesta")
    ax.legend(title="y")
    guardar_grafico(fig, "edad_por_clase.png")

    por_trabajo = (
        datos.assign(respuesta_positiva=datos[VARIABLE_OBJETIVO].eq("yes"))
        .groupby("job", dropna=False)["respuesta_positiva"]
        .agg(cantidad="size", tasa_positiva="mean")
        .sort_values("tasa_positiva")
    )
    por_trabajo.to_csv(CARPETA_SALIDA / "respuesta_por_trabajo.csv")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(por_trabajo.index, por_trabajo["tasa_positiva"] * 100, color="#4c78a8")
    ax.set_xlabel("Respuestas positivas (%)")
    ax.set_title("Tasa de contratación según ocupación")
    guardar_grafico(fig, "respuesta_por_trabajo.png")

    por_default = (
        datos.assign(respuesta_positiva=datos[VARIABLE_OBJETIVO].eq("yes"))
        .groupby("default")["respuesta_positiva"]
        .agg(cantidad="size", tasa_positiva="mean")
    )
    por_default.to_csv(CARPETA_SALIDA / "respuesta_por_default.csv")

    por_contacto_previo = (
        datos.assign(
            contacto_previo=datos["pdays"].ne(999),
            respuesta_positiva=datos[VARIABLE_OBJETIVO].eq("yes"),
        )
        .groupby("contacto_previo")["respuesta_positiva"]
        .agg(cantidad="size", tasa_positiva="mean")
        .reindex([False, True])
    )
    por_contacto_previo.to_csv(CARPETA_SALIDA / "respuesta_por_contacto_previo.csv")
    fig, ax = plt.subplots(figsize=(7, 4.5))
    barras = ax.bar(
        ["Sin contacto previo", "Con contacto previo"],
        por_contacto_previo["tasa_positiva"] * 100,
        color=["#4c78a8", "#f58518"],
    )
    ax.bar_label(
        barras,
        labels=[
            f"{fila.tasa_positiva:.1%} (n={fila.cantidad:,})"
            for fila in por_contacto_previo.itertuples()
        ],
        padding=3,
    )
    ax.set_ylim(0, 75)
    ax.set_ylabel("Respuestas positivas (%)")
    ax.set_title("Respuesta según contacto en campañas anteriores")
    guardar_grafico(fig, "respuesta_por_contacto_previo.png")

    numericas = datos.select_dtypes(include="number").drop(columns=["duration", "pdays"])
    correlaciones = numericas.corr()
    correlaciones.to_csv(CARPETA_SALIDA / "correlaciones_numericas.csv")
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(
        correlaciones,
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        center=0,
        vmin=-1,
        vmax=1,
        ax=ax,
    )
    ax.set_title("Correlaciones numéricas en entrenamiento")
    guardar_grafico(fig, "correlaciones_numericas.png")

    print(f"Tablas guardadas en {CARPETA_SALIDA}")
    print(f"Gráficos guardados en {CARPETA_GRAFICOS}")
    print(f"Entrenamiento: {len(datos)} filas; positivos: {balance['yes'] / len(datos):.2%}")
    print("Decisiones de preprocesamiento registradas en documentacion/decisiones.md")


if __name__ == "__main__":
    main()
