"""Separa los datos originales antes del análisis y del preprocesamiento."""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


RAIZ_PROYECTO = Path(__file__).resolve().parents[1]
ARCHIVO_ORIGINAL = RAIZ_PROYECTO / "data" / "raw" / "bank-additional-full.csv"
ARCHIVO_TRAIN = RAIZ_PROYECTO / "data" / "processed" / "bank_train.csv"
ARCHIVO_TEST = RAIZ_PROYECTO / "data" / "test" / "bank_test.csv"
TEST_SIZE = 0.20
RANDOM_STATE = 42
VARIABLE_OBJETIVO = "y"


def main() -> None:
    """Crea una partición estratificada y conserva todas las columnas originales."""
    datos = pd.read_csv(ARCHIVO_ORIGINAL, sep=";")
    if datos.shape[1] != 21 or VARIABLE_OBJETIVO not in datos:
        raise ValueError("El dataset no tiene el formato esperado")
    if set(datos[VARIABLE_OBJETIVO].dropna().unique()) != {"no", "yes"}:
        raise ValueError("La variable objetivo debe contener exactamente 'no' y 'yes'")

    cantidad_duplicados = int(datos.duplicated().sum())
    datos = datos.drop_duplicates()

    train, test = train_test_split(
        datos,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=datos[VARIABLE_OBJETIVO],
    )

    ARCHIVO_TRAIN.parent.mkdir(parents=True, exist_ok=True)
    ARCHIVO_TEST.parent.mkdir(parents=True, exist_ok=True)
    train.reset_index(drop=True).to_csv(ARCHIVO_TRAIN, index=False)
    test.reset_index(drop=True).to_csv(ARCHIVO_TEST, index=False)

    positivos_train = (train[VARIABLE_OBJETIVO] == "yes").mean()
    print(f"Filas exactamente duplicadas retiradas: {cantidad_duplicados}")
    print(f"Entrenamiento: {len(train)} filas; positivos: {positivos_train:.2%}")
    print(f"Test reservado: {len(test)} filas en {ARCHIVO_TEST}")
    print("Usar solo entrenamiento durante EDA y validación cruzada.")


if __name__ == "__main__":
    main()
