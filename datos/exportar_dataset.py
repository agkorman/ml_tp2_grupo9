"""Descarga la versión de Bank Marketing indicada en la consigna."""

from hashlib import sha256
from io import BytesIO
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile


RAIZ_PROYECTO = Path(__file__).resolve().parents[1]
CARPETA_SALIDA = RAIZ_PROYECTO / "data" / "raw"
URL_DATASET = (
    "https://www.kaggle.com/api/v1/datasets/download/"
    "henriqueyamahata/bank-marketing"
)
ARCHIVOS = ("bank-additional-full.csv", "bank-additional-names.txt")
SHA256_CSV = "74adfc578bf77a7ff4bb1ba4a9f8709d9e3c6907342959c2c8416847e0afb4d8"


def main() -> None:
    """Guarda el dataset y su descripción, verificando el archivo de datos."""
    archivo_csv = CARPETA_SALIDA / ARCHIVOS[0]
    archivo_descripcion = CARPETA_SALIDA / ARCHIVOS[1]
    if archivo_csv.exists() and archivo_descripcion.exists():
        if sha256(archivo_csv.read_bytes()).hexdigest() == SHA256_CSV:
            print(f"Dataset ya disponible en {CARPETA_SALIDA}")
            return
        raise ValueError(f"El archivo existente no coincide con la versión esperada: {archivo_csv}")

    with urlopen(URL_DATASET, timeout=60) as respuesta:
        contenido_zip = respuesta.read()

    with ZipFile(BytesIO(contenido_zip)) as paquete:
        datos = {nombre: paquete.read(nombre) for nombre in ARCHIVOS}

    if sha256(datos[ARCHIVOS[0]]).hexdigest() != SHA256_CSV:
        raise ValueError("El CSV descargado no coincide con la versión esperada")

    CARPETA_SALIDA.mkdir(parents=True, exist_ok=True)
    for nombre, contenido in datos.items():
        (CARPETA_SALIDA / nombre).write_bytes(contenido)
        print(f"Guardado: {CARPETA_SALIDA / nombre}")


if __name__ == "__main__":
    main()
