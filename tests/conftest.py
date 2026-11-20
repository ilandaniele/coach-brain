"""Aislamiento del almacenamiento de Qdrant para toda la suite.

`kb/index.py` resetea borrando `qdrant_storage/collection/<nombre>` del disco
(gotcha #36). `settings.qdrant_path` por defecto es `./qdrant_storage`, que es
el índice real de 400 MB, así que cualquier test que llame a `I.run(reset=True)`
sin pisar ese ajuste lo borra de verdad. Pasó: dos tests que solo querían
comprobar que el reset borra y recrea las colecciones salieron a borrar el
índice de producción, y lo único que lo salvó fue que en ese momento había una
reindexación corriendo con el sqlite lockeado.

Esta fixture es autouse: ningún test apunta al índice real salvo que lo pida
explícitamente volviendo a pisar `qdrant_path`.
"""

from __future__ import annotations

import pytest

from coach_brain.settings import settings


@pytest.fixture(autouse=True)
def _qdrant_aislado(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "qdrant_path", str(tmp_path / "qdrant_storage"))
    monkeypatch.setattr(settings, "qdrant_url", "")
