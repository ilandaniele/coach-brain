"""Tests de los dos scripts que quedaban sin pruebas: el smoke test y la migración a Qdrant Cloud.

Los dos tenían el mismo defecto de fondo, y es el que este proyecto ya vio tres
veces: **avisan OK sin haber hecho nada**.

- `scripts/smoke_test.py` listaba los seis paquetes del pipeline en el informe
  pero solo importaba `streamlit`. Con `faster-whisper` roto o ausente, el
  smoke test daba verde y el error aparecía recién al transcribir, horas
  después.
- `scripts/upload_to_qdrant_cloud.py` leía `settings.qdrant_storage`, que
  devuelve `Path("")` —el directorio actual— en cuanto `QDRANT_URL` está
  configurada, que es justo lo que el script exige para correr. `Path("")`
  existe siempre, así que el chequeo pasaba, se abría la raíz del repo como
  base Qdrant embebida, no había ninguna colección, se salteaban las tres y la
  consola imprimía "Migración completa" con cero vectores subidos.

Los tests fijan las dos cosas: que el chequeo de imports nombre el paquete que
falta, y que la carpeta local se lea de `QDRANT_PATH` y no de la propiedad que
se vacía sola.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
from rich.console import Console

from coach_brain.settings import settings

REPO = Path(__file__).resolve().parent.parent


def _cargar(nombre: str):
    spec = importlib.util.spec_from_file_location(
        nombre, REPO / "scripts" / f"{nombre}.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def smoke(monkeypatch):
    mod = _cargar("smoke_test")
    # Consola ancha: la tabla de rich envuelve las celdas al ancho de la
    # terminal, y envuelta no se puede buscar una frase en la salida.
    monkeypatch.setattr(mod, "console", Console(width=240))
    return mod


@pytest.fixture
def upload(monkeypatch):
    mod = _cargar("upload_to_qdrant_cloud")
    monkeypatch.setattr(mod, "console", Console(width=240))
    return mod


# --- smoke_test: el stack entero falseado ----------------------------------


def torch_falso(*, disponible=True):
    m = types.ModuleType("torch")
    m.cuda = SimpleNamespace(
        is_available=lambda: disponible,
        get_device_name=lambda i: "RTX 5070 Ti",
    )
    m.version = SimpleNamespace(cuda="13.0")
    return m


def anthropic_falso(texto="ok"):
    m = types.ModuleType("anthropic")
    pedidos: list[dict] = []

    class Anthropic:
        def __init__(self, api_key=None):
            self.api_key = api_key
            self.messages = SimpleNamespace(create=self._create)

        def _create(self, **kw):
            pedidos.append(kw)
            return SimpleNamespace(content=[SimpleNamespace(text=texto)])

    m.Anthropic = Anthropic
    m.pedidos = pedidos
    return m


def todo_bien(mod, monkeypatch, *, paquetes=("pypdf",), cuda=True, api_key="sk-ant-0123456789"):
    """Deja los siete chequeos en condiciones de pasar."""
    import coach_brain.kb.embed as E
    import coach_brain.kb.index as I

    # `pypdf` es real y barato de importar; los seis de verdad no se pueden
    # usar acá porque faster-whisper no está instalado en CI.
    monkeypatch.setattr(mod, "PAQUETES", paquetes)
    monkeypatch.setattr(settings, "anthropic_api_key", api_key)
    monkeypatch.setattr(mod.shutil, "which", lambda n: f"C:/bin/{n}.exe")
    monkeypatch.setitem(sys.modules, "torch", torch_falso(disponible=cuda))
    anth = anthropic_falso()
    monkeypatch.setitem(sys.modules, "anthropic", anth)
    monkeypatch.setattr(E, "embed_one", lambda t: [0.0] * 1024)
    monkeypatch.setattr(E, "embed_dim", lambda: 1024)
    monkeypatch.setattr(
        I,
        "get_qdrant",
        lambda: SimpleNamespace(
            get_collections=lambda: SimpleNamespace(collections=[1, 2, 3])
        ),
    )
    # El `close_qdrant` real mira el cache del proceso, que es compartido con el
    # resto de la sesión de pytest: si otro test dejó un cliente ahí, este lo
    # cerraría. Los dos tests que miran el cierre lo re-parchean con un
    # anotador.
    monkeypatch.setattr(I, "close_qdrant", lambda: None)
    return anth


def test_los_seis_paquetes_del_pipeline_siguen_en_la_lista(smoke):
    """Si alguien recorta `PAQUETES`, el smoke test deja de chequear eso y nadie se entera."""
    assert set(smoke.PAQUETES) == {
        "anthropic",
        "faster_whisper",
        "pypdf",
        "qdrant_client",
        "sentence_transformers",
        "streamlit",
    }


def test_todo_en_orden_devuelve_cero(smoke, monkeypatch, capsys):
    todo_bien(smoke, monkeypatch)
    assert smoke.main() == 0
    out = capsys.readouterr().out
    assert "Todo en orden" in out
    assert "FAIL" not in out


def test_el_informe_nombra_el_paquete_que_falta(smoke, monkeypatch, capsys):
    """La regresión: antes importaba solo streamlit y listaba los seis igual."""
    todo_bien(smoke, monkeypatch, paquetes=("pypdf", "no_existe_este_paquete"))
    assert smoke.main() == 1
    out = capsys.readouterr().out
    assert "no_existe_este_paquete" in out
    assert "ModuleNotFoundError" in out
    assert "1 check(s) fallaron" in out


def test_sin_api_key_el_check_de_settings_falla(smoke, monkeypatch, capsys):
    todo_bien(smoke, monkeypatch, api_key="")
    assert smoke.main() == 1
    assert "ANTHROPIC_API_KEY no está en .env" in capsys.readouterr().out


def test_sin_ffmpeg_en_path_falla(smoke, monkeypatch, capsys):
    todo_bien(smoke, monkeypatch)
    monkeypatch.setattr(smoke.shutil, "which", lambda n: None)
    assert smoke.main() == 1
    assert "ffmpeg no está en PATH" in capsys.readouterr().out


def test_sin_cuda_falla_y_lo_dice(smoke, monkeypatch, capsys):
    todo_bien(smoke, monkeypatch, cuda=False)
    assert smoke.main() == 1
    assert "torch.cuda.is_available() == False" in capsys.readouterr().out


def test_con_cuda_reporta_la_placa(smoke, monkeypatch, capsys):
    todo_bien(smoke, monkeypatch)
    smoke.main()
    assert "RTX 5070 Ti" in capsys.readouterr().out


def test_el_check_de_qdrant_cierra_el_cliente(smoke, monkeypatch, capsys):
    """`get_qdrant()` cachea el cliente, así que sin un cierre explícito el
    cliente llega vivo al apagado del intérprete y `QdrantClient.__del__`
    importa `portalocker` con `sys.meta_path` ya en None: el script imprimía
    cinco líneas de "Exception ignored in" después de decir que todo estaba en
    orden."""
    import coach_brain.kb.index as I

    todo_bien(smoke, monkeypatch)
    cierres: list[int] = []
    monkeypatch.setattr(I, "close_qdrant", lambda: cierres.append(1))

    assert smoke.main() == 0

    assert cierres == [1], "el chequeo de Qdrant no cerró el cliente"


def test_si_qdrant_falla_el_cliente_se_cierra_igual(smoke, monkeypatch, capsys):
    """De ahí el `finally`: justo cuando Qdrant falla —otro proceso con el file
    lock, gotcha #11— es cuando conviene soltar lo que se abrió."""
    import coach_brain.kb.index as I

    todo_bien(smoke, monkeypatch)

    def explota():
        raise RuntimeError("otro proceso tiene el lock")

    monkeypatch.setattr(
        I, "get_qdrant", lambda: SimpleNamespace(get_collections=explota)
    )
    cierres: list[int] = []
    monkeypatch.setattr(I, "close_qdrant", lambda: cierres.append(1))

    assert smoke.main() == 1

    assert cierres == [1], "el cliente quedó abierto cuando el chequeo falló"
    assert "otro proceso tiene el lock" in capsys.readouterr().out


def test_el_ping_usa_el_modelo_barato(smoke, monkeypatch, capsys):
    """El ping es para ver si la key responde, no para gastar: modelo fast y 20 tokens."""
    anth = todo_bien(smoke, monkeypatch)
    smoke.main()
    assert len(anth.pedidos) == 1
    assert anth.pedidos[0]["model"] == settings.model_fast
    assert anth.pedidos[0]["max_tokens"] == 20


def test_quick_saltea_los_embeddings(smoke, monkeypatch, capsys):
    """bge-m3 son 2 GB: con `--quick` no se toca, y el informe dice que se salteó."""
    import coach_brain.kb.embed as E

    todo_bien(smoke, monkeypatch)
    monkeypatch.setattr(smoke, "QUICK", True)
    monkeypatch.setattr(E, "embed_one", lambda t: pytest.fail("no se tenía que llamar"))
    assert smoke.main() == 0
    assert "skipped (--quick)" in capsys.readouterr().out


def test_sin_quick_carga_los_embeddings_y_reporta_la_dimension(smoke, monkeypatch, capsys):
    llamadas = []
    import coach_brain.kb.embed as E

    todo_bien(smoke, monkeypatch)
    monkeypatch.setattr(smoke, "QUICK", False)
    monkeypatch.setattr(E, "embed_one", lambda t: llamadas.append(t) or [0.0])
    assert smoke.main() == 0
    assert llamadas == ["hola mundo"]
    assert "dim=1024" in capsys.readouterr().out


def test_qdrant_caido_falla_pero_los_otros_checks_se_reportan(smoke, monkeypatch, capsys):
    """Un chequeo que explota no corta el informe: los siete salen en la tabla."""
    import coach_brain.kb.index as I

    todo_bien(smoke, monkeypatch)
    monkeypatch.setattr(
        I, "get_qdrant", lambda: (_ for _ in ()).throw(RuntimeError("lock tomado"))
    )
    assert smoke.main() == 1
    out = capsys.readouterr().out
    assert "lock tomado" in out
    assert "Qdrant" in out and "CUDA" in out and "ffmpeg" in out


def test_check_devuelve_ok_cuando_la_funcion_no_devuelve_nada(smoke):
    assert smoke.check("x", lambda: None) == (True, "ok")


def test_check_envuelve_la_excepcion_con_su_tipo(smoke):
    ok, msg = smoke.check("x", lambda: (_ for _ in ()).throw(ValueError("mal")))
    assert ok is False
    assert msg == "ValueError: mal"


# --- upload_to_qdrant_cloud ------------------------------------------------


class ClienteFalso:
    """Reemplaza a QdrantClient. Anota todo lo que se le pide."""

    def __init__(self, **kw):
        self.kw = kw
        self.colecciones: dict[str, SimpleNamespace] = {}
        self.creadas: list[tuple[str, int, object]] = []
        self.subidos: list[tuple[str, list]] = []
        self.paginas: dict[str, list] = {}
        self.scrolls: list[tuple[str, int, object]] = []

    def tener(self, nombre, dim=1024, total=0):
        self.colecciones[nombre] = SimpleNamespace(
            config=SimpleNamespace(params=SimpleNamespace(vectors=SimpleNamespace(size=dim))),
            points_count=total,
        )

    def get_collection(self, nombre):
        if nombre not in self.colecciones:
            raise RuntimeError(f"Collection {nombre} not found")
        return self.colecciones[nombre]

    def create_collection(self, collection_name, vectors_config):
        self.creadas.append((collection_name, vectors_config.size, vectors_config.distance))
        self.tener(collection_name, vectors_config.size)

    def scroll(self, collection_name, limit, offset, with_payload, with_vectors):
        assert with_payload and with_vectors, "sin payload ni vectores no se migra nada"
        self.scrolls.append((collection_name, limit, offset))
        return self.paginas.get(collection_name, [([], None)]).pop(0)

    def upsert(self, collection_name, points):
        self.subidos.append(
            (collection_name, [(p.id, p.vector, p.payload) for p in points])
        )


def punto(i):
    return SimpleNamespace(id=i, vector=[float(i)], payload={"n": i})


def armar(upload, monkeypatch, tmp_path, *, url="https://x.qdrant.io:6333", api_key=""):
    """Deja un Qdrant embebido creíble en tmp_path y falsea los dos clientes."""
    (tmp_path / "collection").mkdir()
    monkeypatch.setattr(settings, "qdrant_url", url)
    monkeypatch.setattr(settings, "qdrant_api_key", api_key)
    monkeypatch.setattr(settings, "qdrant_path", str(tmp_path))
    local, cloud = ClienteFalso(), ClienteFalso()

    def fabrica(**kw):
        c = local if "path" in kw else cloud
        c.kw = kw
        return c

    monkeypatch.setattr(upload, "QdrantClient", fabrica)
    return local, cloud


def test_el_storage_local_sale_de_qdrant_path_no_de_la_propiedad(upload, monkeypatch, tmp_path):
    """La regresión: con QDRANT_URL puesta, `settings.qdrant_storage` es `.`.

    El script exige QDRANT_URL, así que leía siempre el directorio actual, que
    existe, y la migración terminaba "completa" sin subir nada.
    """
    monkeypatch.setattr(settings, "qdrant_url", "https://x.qdrant.io:6333")
    monkeypatch.setattr(settings, "qdrant_path", str(tmp_path / "qs"))
    assert settings.qdrant_storage == Path("")
    assert upload.storage_local() == tmp_path / "qs"


def test_qdrant_path_relativo_se_resuelve_contra_el_repo(upload, monkeypatch):
    monkeypatch.setattr(settings, "qdrant_url", "https://x.qdrant.io:6333")
    monkeypatch.setattr(settings, "qdrant_path", "./qdrant_storage")
    assert upload.storage_local() == REPO / "qdrant_storage"


def test_sin_qdrant_url_corta(upload, monkeypatch, capsys):
    monkeypatch.setattr(settings, "qdrant_url", "")
    with pytest.raises(SystemExit) as e:
        upload.main()
    assert e.value.code == 1
    assert "QDRANT_URL no configurada" in capsys.readouterr().out


def test_sin_base_embebida_corta_en_vez_de_migrar_la_nada(upload, monkeypatch, tmp_path, capsys):
    """Una carpeta que existe pero no tiene `collection/` no es un Qdrant embebido."""
    monkeypatch.setattr(settings, "qdrant_url", "https://x.qdrant.io:6333")
    monkeypatch.setattr(settings, "qdrant_path", str(tmp_path))
    with pytest.raises(SystemExit) as e:
        upload.main()
    assert e.value.code == 1
    out = capsys.readouterr().out
    assert "No hay un Qdrant embebido" in out
    assert "QDRANT_PATH" in out


def test_sube_todos_los_vectores_en_lotes(upload, monkeypatch, tmp_path, capsys):
    local, cloud = armar(upload, monkeypatch, tmp_path)
    local.tener("principles", dim=1024, total=3)
    local.paginas["principles"] = [([punto(1), punto(2)], "sig"), ([punto(3)], None)]
    monkeypatch.setattr(upload, "COLLECTIONS", ["principles"])
    upload.main()
    assert [p[0] for p in cloud.subidos] == ["principles", "principles"]
    assert [i for _, pts in cloud.subidos for i, _, _ in pts] == [1, 2, 3]
    # el offset de la segunda página es el que devolvió la primera
    assert [s[2] for s in local.scrolls] == [None, "sig"]
    assert "3 vectores subidos" in capsys.readouterr().out


def test_el_lote_respeta_batch(upload, monkeypatch, tmp_path):
    local, _ = armar(upload, monkeypatch, tmp_path)
    local.tener("principles", total=1)
    local.paginas["principles"] = [([punto(1)], None)]
    monkeypatch.setattr(upload, "COLLECTIONS", ["principles"])
    upload.main()
    assert {s[1] for s in local.scrolls} == {upload.BATCH}


def test_crea_la_coleccion_en_cloud_con_la_dimension_del_local(
    upload, monkeypatch, tmp_path, capsys
):
    local, cloud = armar(upload, monkeypatch, tmp_path)
    local.tener("principles", dim=768, total=0)
    local.paginas["principles"] = [([], None)]
    monkeypatch.setattr(upload, "COLLECTIONS", ["principles"])
    upload.main()
    assert cloud.creadas == [("principles", 768, upload.Distance.COSINE)]
    assert "Creada en cloud" in capsys.readouterr().out


def test_si_ya_existe_en_cloud_no_la_vuelve_a_crear(upload, monkeypatch, tmp_path, capsys):
    local, cloud = armar(upload, monkeypatch, tmp_path)
    local.tener("principles", dim=1024, total=1)
    cloud.tener("principles", dim=1024)
    local.paginas["principles"] = [([punto(1)], None)]
    monkeypatch.setattr(upload, "COLLECTIONS", ["principles"])
    upload.main()
    assert cloud.creadas == []
    assert "ya existe en cloud" in capsys.readouterr().out
    assert len(cloud.subidos) == 1


def test_una_coleccion_que_no_esta_local_se_saltea_y_sigue_con_las_otras(
    upload, monkeypatch, tmp_path, capsys
):
    local, cloud = armar(upload, monkeypatch, tmp_path)
    local.tener("situations", total=1)
    local.paginas["situations"] = [([punto(7)], None)]
    upload.main()  # las tres colecciones de verdad; solo una existe
    out = capsys.readouterr().out
    assert out.count("No existe localmente, skip") == 2
    assert [p[0] for p in cloud.subidos] == ["situations"]
    assert "Migración completa" in out


def test_la_api_key_solo_se_le_pasa_al_cliente_si_esta(upload, monkeypatch, tmp_path, capsys):
    _, cloud = armar(upload, monkeypatch, tmp_path, api_key="")
    monkeypatch.setattr(upload, "COLLECTIONS", [])
    upload.main()
    assert cloud.kw == {"url": settings.qdrant_url}
    assert "QDRANT_API_KEY vacía" in capsys.readouterr().out


def test_con_api_key_va_en_el_cliente_y_sin_aviso(upload, monkeypatch, tmp_path, capsys):
    _, cloud = armar(upload, monkeypatch, tmp_path, api_key="qdr-secreta")
    monkeypatch.setattr(upload, "COLLECTIONS", [])
    upload.main()
    assert cloud.kw["api_key"] == "qdr-secreta"
    assert "QDRANT_API_KEY vacía" not in capsys.readouterr().out


def test_el_cliente_local_se_abre_en_la_carpeta_resuelta(upload, monkeypatch, tmp_path):
    local, _ = armar(upload, monkeypatch, tmp_path)
    monkeypatch.setattr(upload, "COLLECTIONS", [])
    upload.main()
    assert local.kw == {"path": str(tmp_path)}
