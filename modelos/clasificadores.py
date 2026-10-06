"""Clasificadores con preprocesamiento."""

from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from datos.preprocesamiento import crear_preprocesador


def crear_clasificadores(random_state: int = 42) -> dict[str, Pipeline]:
    """Crea pipelines sin entrenar."""
    configuraciones = {
        "Naive Bayes": (GaussianNB(), False),
        "SVM": (SVC(C=1.0, kernel="rbf", gamma="scale"), True),
        "KNN": (KNeighborsClassifier(n_neighbors=5, weights="uniform"), True),
        "RF": (
            RandomForestClassifier(
                n_estimators=100, random_state=random_state, n_jobs=1
            ),
            False,
        ),
    }
    return {
        nombre: Pipeline(
            [
                ("preprocesamiento", crear_preprocesador(escalar_numericas=escalar)),
                ("clasificador", clasificador),
            ]
        )
        for nombre, (clasificador, escalar) in configuraciones.items()
    }
