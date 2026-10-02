"""Preprocesamiento reutilizable para modelos de clasificación.

El objeto devuelto está sin ajustar. Debe incorporarse a un ``Pipeline`` junto
con el clasificador para que imputación, codificación y escalado se aprendan
dentro de cada partición de entrenamiento de la validación cruzada.
"""

from typing import Tuple

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler


COLUMNAS_NUMERICAS = (
    "age",
    "campaign",
    "previous",
    "emp.var.rate",
    "cons.price.idx",
    "cons.conf.idx",
    "euribor3m",
    "nr.employed",
)
COLUMNAS_CATEGORICAS = (
    "job",
    "marital",
    "education",
    "default",
    "housing",
    "loan",
    "contact",
    "month",
    "day_of_week",
    "poutcome",
)
COLUMNAS_ENTRADA = COLUMNAS_NUMERICAS + COLUMNAS_CATEGORICAS + ("pdays",)
COLUMNAS_NUMERICAS_MODELO = COLUMNAS_NUMERICAS + ("dias_desde_contacto",)
COLUMNAS_BINARIAS_MODELO = ("contacto_previo",)
VALOR_SIN_CONTACTO_PREVIO = 999


def preparar_variables(datos: pd.DataFrame) -> pd.DataFrame:
    """Selecciona atributos disponibles antes de la llamada y corrige ``pdays``."""
    if not isinstance(datos, pd.DataFrame):
        raise TypeError("El preprocesador espera un DataFrame de pandas")
    if "y" in datos.columns:
        raise ValueError("Separar la variable objetivo 'y' antes de preprocesar")

    faltantes = set(COLUMNAS_ENTRADA).difference(datos.columns)
    if faltantes:
        raise ValueError(f"Faltan columnas requeridas: {sorted(faltantes)}")

    variables = datos.loc[:, COLUMNAS_ENTRADA].copy()
    hubo_contacto = variables["pdays"].notna() & variables["pdays"].ne(
        VALOR_SIN_CONTACTO_PREVIO
    )
    variables["contacto_previo"] = hubo_contacto.astype(int)
    variables["dias_desde_contacto"] = variables["pdays"].where(hubo_contacto)
    return variables.drop(columns="pdays")


def crear_preprocesador(escalar_numericas: bool = True) -> Pipeline:
    """Crea un pipeline sin ajustar para acoplar a un clasificador.

    ``escalar_numericas=True`` sirve para modelos sensibles a la escala, como
    SVM y KNN. El indicador binario se mantiene en 0/1. Puede omitirse el
    escalado de las variables continuas para modelos de árboles.
    """
    pasos_numericos: list[Tuple[str, object]] = [
        ("imputacion", SimpleImputer(strategy="median", keep_empty_features=True)),
    ]
    if escalar_numericas:
        pasos_numericos.append(("escalado", StandardScaler()))

    numericas = Pipeline(pasos_numericos)
    categoricas = Pipeline(
        [
            ("imputacion", SimpleImputer(strategy="constant", fill_value="unknown")),
            ("codificacion", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    columnas = ColumnTransformer(
        [
            ("numericas", numericas, COLUMNAS_NUMERICAS_MODELO),
            ("indicadores", "passthrough", COLUMNAS_BINARIAS_MODELO),
            ("categoricas", categoricas, COLUMNAS_CATEGORICAS),
        ],
        remainder="drop",
        sparse_threshold=0,
    )
    return Pipeline(
        [
            ("variables", FunctionTransformer(preparar_variables, validate=False)),
            ("columnas", columnas),
        ]
    )
