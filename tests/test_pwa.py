"""El PWA: el icono generado y la inyeccion en el <head>.

Streamlit no expone el `<head>`, asi que `inject()` arma un `<script>` con
los iconos embebidos como data: URI y lo manda por un componente. El riesgo
no es que falle: es que quede un placeholder sin reemplazar (`__ICON192__`)
y el navegador se trague un manifest invalido sin decir nada, con lo cual la
app deja de ser instalable y nadie se entera.

No hace falta runtime de Streamlit: `components.html` y `st.markdown` se
reemplazan por espias.
"""

from __future__ import annotations

import base64
import io
import json
import re

import pytest
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image

from coach_brain.app import pwa as P


def abrir(size: int) -> Image.Image:
    """El icono de ese tamano, decodificado como imagen."""
    return Image.open(io.BytesIO(base64.b64decode(P._icon_png_b64(size))))


# --- el icono --------------------------------------------------------------

@pytest.mark.parametrize("size", [180, 192, 512])
def test_el_icono_es_un_png_del_tamano_pedido(size):
    img = abrir(size)
    assert img.format == "PNG"
    assert img.size == (size, size)


def test_el_base64_es_ascii_puro():
    """Va dentro de un string de JavaScript: un byte raro rompe el script."""
    b64 = P._icon_png_b64(192)
    assert b64.isascii()
    assert "\n" not in b64


def test_el_centro_del_icono_es_el_acento():
    """La diana se dibuja de afuera hacia adentro: el ultimo anillo pintado es
    el acento, asi que el centro tiene que quedar rojo."""
    img = abrir(192).convert("RGB")
    assert img.getpixel((96, 96)) == (0xFF, 0x4B, 0x4B)


def test_las_esquinas_son_el_fondo_del_tema():
    """Sin transparencia: el maskable de Android necesita el fondo pintado."""
    img = abrir(192).convert("RGB")
    assert img.getpixel((0, 0)) == (0x0E, 0x11, 0x17)
    assert img.getpixel((191, 191)) == (0x0E, 0x11, 0x17)


def test_dos_tamanos_dan_dos_imagenes_distintas():
    """El cache es por tamano: si la clave se perdiera, los tres iconos
    saldrian iguales y Chrome rechazaria el manifest."""
    assert P._icon_png_b64(192) != P._icon_png_b64(512)


def test_el_icono_no_es_una_imagen_vacia():
    assert len(base64.b64decode(P._icon_png_b64(192))) > 200


# --- la inyeccion ----------------------------------------------------------

@pytest.fixture
def inyectado(monkeypatch):
    """Corre `inject()` con espias y devuelve (html, css)."""
    htmls: list[dict] = []
    css: list[dict] = []
    monkeypatch.setattr(components, "html", lambda h, **kw: htmls.append({"html": h, **kw}))
    monkeypatch.setattr(st, "markdown", lambda t, **kw: css.append({"texto": t, **kw}))
    P.inject()
    assert len(htmls) == 1 and len(css) == 1
    return htmls[0], css[0]


def test_no_queda_ningun_placeholder_sin_reemplazar(inyectado):
    """Lo que de verdad importa: un `__ICON192__` crudo deja el manifest
    invalido y la app no se puede instalar, sin error visible."""
    html = inyectado[0]["html"]
    assert "__ICON192__" not in html
    assert "__ICON512__" not in html
    assert "__ICON180__" not in html
    assert "__THEME__" not in html
    assert not re.search(r"__[A-Z0-9]+__", html)


def test_los_tres_iconos_van_como_data_uri(inyectado):
    html = inyectado[0]["html"]
    uris = re.findall(r'"(data:image/png;base64,[^"]+)"', html)
    assert len(uris) == 3
    for u in uris:
        payload = u.split(",", 1)[1]
        assert Image.open(io.BytesIO(base64.b64decode(payload))).format == "PNG"


def test_los_tres_iconos_son_distintos_entre_si(inyectado):
    uris = re.findall(r'"(data:image/png;base64,[^"]+)"', inyectado[0]["html"])
    assert len(set(uris)) == 3


def test_el_color_del_tema_llega_a_las_dos_claves(inyectado):
    """`background_color` y `theme_color` salen de la misma constante que el
    fondo dark de Streamlit: si se desincronizan, la splash parpadea."""
    html = inyectado[0]["html"]
    assert 'var THEME = "#0E1117"' in html


def test_el_componente_no_ocupa_lugar(inyectado):
    """El iframe solo inyecta: si midiera, abriria un hueco arriba de todo."""
    assert inyectado[0]["height"] == 0
    assert inyectado[0]["width"] == 0


def test_el_script_no_se_ejecuta_dos_veces(inyectado):
    """Streamlit vuelve a correr el script en cada interaccion: sin el guard,
    cada rerun agregaria otro <link rel=manifest>."""
    assert "__coachPwaReady" in inyectado[0]["html"]


def test_el_manifest_declara_standalone_y_los_tres_iconos(inyectado):
    html = inyectado[0]["html"]
    assert 'display: "standalone"' in html
    assert html.count('type: "image/png"') == 3
    assert 'purpose: "maskable"' in html  # Android recorta el icono


def test_el_manifest_va_como_blob_no_como_data_uri(inyectado):
    """Chrome rechaza un manifest `data:` porque no es same-origin, y entonces
    start_url y scope absolutos no validan."""
    html = inyectado[0]["html"]
    assert "URL.createObjectURL(blob)" in html
    assert "application/manifest+json" in html


def test_ios_recibe_sus_propios_meta_tags(inyectado):
    """iOS ignora el manifest entero: sin estos tags no hay app instalable."""
    html = inyectado[0]["html"]
    for tag in ("apple-touch-icon", "apple-mobile-web-app-capable",
                "apple-mobile-web-app-status-bar-style", "mobile-web-app-capable"):
        assert tag in html


def test_el_viewport_respeta_el_notch_y_frena_el_zoom(inyectado):
    html = inyectado[0]["html"]
    assert "viewport-fit=cover" in html
    assert "maximum-scale=1" in html


# --- el CSS mobile ---------------------------------------------------------

def test_el_css_se_pasa_como_html_crudo(inyectado):
    """Sin `unsafe_allow_html` el `<style>` se imprimiria como texto."""
    css = inyectado[1]
    assert css["unsafe_allow_html"] is True
    assert css["texto"].strip().startswith("<style>")


def test_los_inputs_miden_16px(inyectado):
    """Menos de 16px y iOS hace zoom al enfocar: el layout se descoloca."""
    assert "font-size: 16px !important" in inyectado[1]["texto"]


def test_las_tabs_se_apuntan_por_los_tres_selectores(inyectado):
    """Streamlit 1.58 cambio `data-baseweb="tab"` por `data-testid="stTab"` y
    la regla vieja dejo de matchear: las 3 tabs no entraban en 375px."""
    texto = inyectado[1]["texto"]
    assert '[data-baseweb="tab"]' in texto
    assert '[data-testid="stTab"]' in texto
    assert '[role="tab"]' in texto


def test_el_safe_area_de_abajo_se_respeta(inyectado):
    assert "env(safe-area-inset-bottom)" in inyectado[1]["texto"]


def test_las_tabs_scrollean_en_vez_de_apilarse(inyectado):
    texto = inyectado[1]["texto"]
    assert "overflow-x: auto" in texto
    assert "flex-wrap: nowrap" in texto


def test_el_chrome_de_streamlit_queda_oculto(inyectado):
    """En el celular el menu y el footer ocupan lugar y no se usan."""
    texto = inyectado[1]["texto"]
    assert "#MainMenu" in texto
    assert "visibility: hidden" in texto


def test_el_manifest_es_json_valido_una_vez_armado(inyectado):
    """El objeto se escribe como literal de JS; esta prueba lo traduce a JSON
    para confirmar que no quedo una coma colgando ni una clave sin valor."""
    html = inyectado[0]["html"]
    cuerpo = html.split("var manifest = ", 1)[1].split("\n  };", 1)[0] + "\n  }"
    # Las claves van sin comillas en JS y los iconos usan `src: ICON192`.
    cuerpo = re.sub(r"(?m)(^|[{,])(\s*)([a-z_]+):", r'\1\2"\3":', cuerpo)
    cuerpo = re.sub(r':\s*(ICON\d+|THEME|origin \+ "/")', ': "x"', cuerpo)
    datos = json.loads(cuerpo)
    assert datos["display"] == "standalone"
    assert datos["orientation"] == "portrait"
    assert len(datos["icons"]) == 3
    assert datos["short_name"] == "Coach"
