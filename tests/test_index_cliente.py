"""Tests del cliente Qdrant: el cache, el mutex y la bifurcación local/remoto.

`index.py` tenía probado el indexado (ver `tests/test_index_extracted.py`) pero
no la función que abre el cliente, y ahí viven tres decisiones que cuestan caro
si se rompen en silencio:

1. **Una sola construcción por proceso.** En modo embebido el constructor de
   `QdrantClient` carga los 399 MB de vectores a numpy de forma eager. Sin el
   `lru_cache`, cada consulta de la app recargaría el índice entero.
2. **El mutex de verdad serializa.** `QdrantClient(path=...)` toma un file lock
   sobre el directorio, y `lru_cache` no sostiene nada mientras corre el cuerpo
   de la función: dos threads de Streamlit entraban a la vez y el segundo
   quedaba colgado esperando el lock del primero. Es el gotcha #11.
3. **Con `QDRANT_URL` no se toca el disco.** `settings.qdrant_storage` devuelve
   `Path("")` justo cuando hay URL, así que si la rama remota no cortara antes,
   el `mkdir` de la rama local recibiría una ruta vacía. En Fly.io eso sería un
   crash en el arranque.

4. **El cierre es explícito.** `close_qdrant()` existe porque el cache hace
   que el cliente viva hasta que se apaga el intérprete, y ahí el `__del__` de
   `QdrantClient` importa `portalocker` con `sys.meta_path` ya en None: un
   script que terminó bien imprime cinco líneas de traceback igual.

Además se fija que un `--reset` sobre un Qdrant vacío no aborta: en el primer
deploy las tres colecciones todavía no existen y `delete_collection` falla.

No se abre Qdrant real: `QdrantClient` se reemplaza por un doble que anota con
qué argumentos lo llamaron y si lo cerraron.
"""

from __future__ import annotations

import threading
import time

import pytest

from coach_brain.kb import index as I
from coach_brain.settings import settings


class ClienteFalso:
    """Anota los kwargs de construcción y si lo cerraron. Es lo único que se
    quiere observar."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.cerrado = False

    def close(self):
        self.cerrado = True


@pytest.fixture(autouse=True)
def cache_limpio():
    """El cache es del proceso, no del test. Sin esto el primer test que abre
    un cliente se lo presta a todos los demás y a la sesión real de pytest."""
    I._get_qdrant_cached.cache_clear()
    yield
    I._get_qdrant_cached.cache_clear()


@pytest.fixture
def construcciones(monkeypatch):
    """Reemplaza `QdrantClient` y devuelve la lista de kwargs con que se llamó."""
    llamadas: list[dict] = []

    def fabrica(**kwargs):
        llamadas.append(kwargs)
        return ClienteFalso(**kwargs)

    monkeypatch.setattr(I, "QdrantClient", fabrica)
    return llamadas


@pytest.fixture
def local(tmp_path, monkeypatch):
    """Modo embebido, con el storage en un directorio que todavía no existe."""
    monkeypatch.setattr(settings, "qdrant_url", "")
    monkeypatch.setattr(settings, "qdrant_api_key", "")
    destino = tmp_path / "qdrant_storage"
    monkeypatch.setattr(settings, "qdrant_path", str(destino))
    return destino


# --- el cache ---------------------------------------------------------------


def test_dos_llamadas_devuelven_el_mismo_cliente(local, construcciones):
    """En modo embebido el constructor carga 399 MB a numpy. Construir dos veces
    no es lento: es el índice entero otra vez en memoria."""
    primero = I.get_qdrant()
    segundo = I.get_qdrant()

    assert primero is segundo
    assert len(construcciones) == 1


def test_el_cache_guarda_uno_solo():
    """`maxsize=1`: si creciera, un cambio de `QDRANT_URL` en caliente dejaría
    dos clientes vivos sobre el mismo storage, y el segundo no consigue el lock."""
    assert I._get_qdrant_cached.cache_info().maxsize == 1


# --- el cierre --------------------------------------------------------------


def test_close_qdrant_cierra_el_cliente_y_vacia_el_cache(local, construcciones):
    """Lo que arregla: un script que llama a `get_qdrant()` deja el cliente en
    el cache, y al apagarse el intérprete `QdrantClient.__del__` intenta
    `import portalocker` con `sys.meta_path` ya en None. El script termina bien
    e igual imprime cinco líneas de traceback."""
    cliente = I.get_qdrant()

    I.close_qdrant()

    assert cliente.cerrado, "no se cerró el cliente que estaba cacheado"
    assert I._get_qdrant_cached.cache_info().currsize == 0


def test_close_qdrant_sin_cliente_no_construye_uno_para_cerrarlo(local, construcciones):
    """`_get_qdrant_cached()` a secas construiría el cliente que se quiere
    cerrar: en modo embebido son 399 MB a numpy y el file lock del directorio.
    De ahí la guarda por `currsize`."""
    I.close_qdrant()

    assert construcciones == []


def test_despues_de_cerrar_la_proxima_llamada_construye_de_nuevo(local, construcciones):
    """El cache queda vacío, no con un cliente cerrado adentro: devolver uno
    cerrado haría fallar la consulta siguiente."""
    primero = I.get_qdrant()
    I.close_qdrant()
    segundo = I.get_qdrant()

    assert segundo is not primero
    assert len(construcciones) == 2
    assert not segundo.cerrado


# --- el mutex (gotcha #11) --------------------------------------------------


def test_dos_threads_construyen_un_solo_cliente(local, monkeypatch):
    """El caso real: dos requests de Streamlit a la vez. `lru_cache` no sostiene
    nada mientras corre el cuerpo, así que sin `_QDRANT_LOCK` los dos entran, y
    el segundo se cuelga esperando el file lock del primero."""
    construidos: list[float] = []

    def fabrica_lenta(**kwargs):
        time.sleep(0.05)  # la ventana en la que entraba el segundo thread
        construidos.append(time.monotonic())
        return ClienteFalso(**kwargs)

    monkeypatch.setattr(I, "QdrantClient", fabrica_lenta)

    resultados: list[object] = []
    hilos = [
        threading.Thread(target=lambda: resultados.append(I.get_qdrant()))
        for _ in range(2)
    ]
    for h in hilos:
        h.start()
    for h in hilos:
        h.join(timeout=10)

    assert len(construidos) == 1, "se construyo dos veces"
    assert len(resultados) == 2
    assert resultados[0] is resultados[1], "los dos threads comparten cliente"


def test_el_lock_es_del_modulo():
    """Tiene que ser uno solo para todo el proceso: un lock por llamada no
    serializa nada."""
    assert isinstance(I._QDRANT_LOCK, type(threading.Lock()))


# --- la rama remota ---------------------------------------------------------


def test_con_url_se_construye_remoto(monkeypatch, construcciones):
    monkeypatch.setattr(settings, "qdrant_url", "https://qdrant.ejemplo:6333")
    monkeypatch.setattr(settings, "qdrant_api_key", "")

    I.get_qdrant()

    assert construcciones == [{"url": "https://qdrant.ejemplo:6333"}]


def test_con_url_y_api_key_se_pasa_la_key(monkeypatch, construcciones):
    monkeypatch.setattr(settings, "qdrant_url", "https://qdrant.ejemplo:6333")
    monkeypatch.setattr(settings, "qdrant_api_key", "clave-de-prueba")

    I.get_qdrant()

    assert construcciones == [
        {"url": "https://qdrant.ejemplo:6333", "api_key": "clave-de-prueba"}
    ]


def test_una_api_key_vacia_no_se_pasa(monkeypatch, construcciones):
    """Qdrant Cloud rechaza `api_key=""`, así que la clave vacía no puede
    viajar como kwarg."""
    monkeypatch.setattr(settings, "qdrant_url", "https://qdrant.ejemplo:6333")
    monkeypatch.setattr(settings, "qdrant_api_key", "")

    I.get_qdrant()

    assert "api_key" not in construcciones[0]


def test_con_url_nunca_se_pasa_path(monkeypatch, construcciones):
    """`path` y `url` juntos no son una configuración válida del cliente."""
    monkeypatch.setattr(settings, "qdrant_url", "https://qdrant.ejemplo:6333")
    monkeypatch.setattr(settings, "qdrant_api_key", "k")

    I.get_qdrant()

    assert "path" not in construcciones[0]


def test_con_url_el_storage_queda_vacio(monkeypatch):
    """La razón por la que la rama remota tiene que cortar antes: con URL,
    `qdrant_storage` es `Path("")`, y `Path("").mkdir()` no es una ruta."""
    monkeypatch.setattr(settings, "qdrant_url", "https://qdrant.ejemplo:6333")
    assert str(settings.qdrant_storage) == "."


def test_con_url_no_se_crea_ningun_directorio(tmp_path, monkeypatch, construcciones):
    """En Fly.io el disco es efímero y chico: la rama remota no tiene que
    escribir nada."""
    destino = tmp_path / "no_deberia_existir"
    monkeypatch.setattr(settings, "qdrant_url", "https://qdrant.ejemplo:6333")
    monkeypatch.setattr(settings, "qdrant_api_key", "")
    monkeypatch.setattr(settings, "qdrant_path", str(destino))

    I.get_qdrant()

    assert not destino.exists()


def test_con_url_se_avisa_a_donde_apunta(monkeypatch, construcciones, capsys):
    """El aviso es lo que distingue "consulté el índice de producción" de
    "consulté el de mi disco" cuando una respuesta sale distinta."""
    monkeypatch.setattr(settings, "qdrant_url", "https://qdrant.ejemplo:6333")
    monkeypatch.setattr(settings, "qdrant_api_key", "")

    I.get_qdrant()

    salida = capsys.readouterr().out
    assert "remoto" in salida
    assert "qdrant.ejemplo" in salida


# --- la rama local ----------------------------------------------------------


def test_sin_url_se_construye_embebido_con_path(local, construcciones):
    I.get_qdrant()
    assert construcciones == [{"path": str(local)}]


def test_el_path_viaja_como_string(local, construcciones):
    """`QdrantClient` no acepta un `Path` en todas sus versiones."""
    I.get_qdrant()
    assert isinstance(construcciones[0]["path"], str)


def test_el_storage_se_crea_si_no_existe(local, construcciones):
    """Primer arranque en una máquina limpia: el directorio no está."""
    assert not local.exists()
    I.get_qdrant()
    assert local.is_dir()


def test_el_storage_se_crea_con_los_padres(tmp_path, monkeypatch, construcciones):
    destino = tmp_path / "uno" / "dos" / "qdrant_storage"
    monkeypatch.setattr(settings, "qdrant_url", "")
    monkeypatch.setattr(settings, "qdrant_path", str(destino))

    I.get_qdrant()

    assert destino.is_dir()


def test_un_storage_que_ya_existe_no_molesta(local, construcciones):
    """`exist_ok=True`: el caso normal es que los 399 MB ya estén ahí."""
    local.mkdir(parents=True)
    (local / "meta.json").write_text("{}", encoding="utf-8")

    I.get_qdrant()

    assert (local / "meta.json").exists(), "no se toco lo que habia"
    assert len(construcciones) == 1


def test_sin_url_se_avisa_que_es_local(local, construcciones, capsys):
    I.get_qdrant()
    salida = capsys.readouterr().out
    assert "local" in salida or "embebido" in salida


# --- el reset sobre un Qdrant vacío ----------------------------------------


def test_el_reset_no_aborta_si_la_coleccion_no_existe(tmp_path, monkeypatch):
    """Primer deploy: las tres colecciones no existen todavía y
    `delete_collection` falla. Abortar ahí dejaría el índice sin construir."""

    class QdrantSinColecciones:
        def __init__(self):
            self.creadas: list[str] = []
            self.upserts: list[tuple[str, list]] = []

        def delete_collection(self, name):
            raise ValueError(f"no existe {name}")

        def get_collection(self, name):
            raise KeyError(name)

        def create_collection(self, collection_name, vectors_config):
            self.creadas.append(collection_name)

        def upsert(self, collection_name, points):
            self.upserts.append((collection_name, points))

    edir = tmp_path / "processed" / "extracted"
    edir.mkdir(parents=True)
    (edir / "uno.extracted.json").write_text(
        '{"frases": [{"id": "f1", "texto": "hola"}]}', encoding="utf-8"
    )
    monkeypatch.setattr(settings, "data_dir", str(tmp_path))

    c = QdrantSinColecciones()
    monkeypatch.setattr(I, "get_qdrant", lambda: c)
    monkeypatch.setattr(I, "embed_dim", lambda: 4)
    monkeypatch.setattr(I, "embed", lambda textos: [[0.1] * 4 for _ in textos])

    I.run(reset=True)

    assert set(c.creadas) == {I.COL_PRINCIPLES, I.COL_SITUATIONS, I.COL_STYLE}
    assert c.upserts, "el extracto se indexo igual"
