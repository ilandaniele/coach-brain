"""PWA: hace que la app se pueda instalar en el celular como app nativa.

Streamlit no expone el <head>, así que el manifest y los meta tags de iOS se
inyectan desde un componente (que corre en un iframe same-origin) al
document.head del padre. El manifest va como data: URI para no depender de
servir archivos estáticos.
"""

from __future__ import annotations

import base64
import io

import streamlit as st
from PIL import Image, ImageDraw

THEME_BG = "#0E1117"      # igual al fondo dark de Streamlit
THEME_ACCENT = "#FF4B4B"  # rojo de Streamlit


@st.cache_data(show_spinner=False)
def _icon_png_b64(size: int) -> str:
    """Ícono generado (diana concéntrica). Sin fuentes: solo círculos."""
    img = Image.new("RGB", (size, size), THEME_BG)
    d = ImageDraw.Draw(img)
    rings = [THEME_ACCENT, THEME_BG, THEME_ACCENT, THEME_BG, THEME_ACCENT]
    outer = size * 0.40
    for i, color in enumerate(rings):
        r = outer * (1 - i * 0.19)
        c = size / 2
        d.ellipse([c - r, c - r, c + r, c + r], fill=color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


_INJECT_JS = """
<script>
(function () {
  var doc = window.parent.document;
  var head = doc.head;
  if (!head || window.parent.__coachPwaReady) return;
  window.parent.__coachPwaReady = true;

  var ICON192 = "__ICON192__";
  var ICON512 = "__ICON512__";
  var ICON180 = "__ICON180__";
  var THEME = "__THEME__";

  function setMeta(key, content) {
    var el = head.querySelector('meta[name="' + key + '"]');
    if (!el) {
      el = doc.createElement('meta');
      el.setAttribute('name', key);
      head.appendChild(el);
    }
    el.setAttribute('content', content);
  }
  function setLink(rel, href) {
    var el = head.querySelector('link[rel="' + rel + '"]');
    if (!el) {
      el = doc.createElement('link');
      el.setAttribute('rel', rel);
      head.appendChild(el);
    }
    el.setAttribute('href', href);
  }

  // El manifest va como Blob (no data:) porque un blob: es same-origin y deja
  // que start_url/scope absolutos validen. Con data: Chrome lo rechaza.
  var origin = window.parent.location.origin;
  var manifest = {
    name: "Coach Asistente",
    short_name: "Coach",
    description: "Diagnostico de chats y lineas listas para usar.",
    start_url: origin + "/",
    scope: origin + "/",
    display: "standalone",
    orientation: "portrait",
    background_color: THEME,
    theme_color: THEME,
    icons: [
      { src: ICON192, sizes: "192x192", type: "image/png", purpose: "any" },
      { src: ICON512, sizes: "512x512", type: "image/png", purpose: "any" },
      { src: ICON512, sizes: "512x512", type: "image/png", purpose: "maskable" }
    ]
  };
  var blob = new Blob([JSON.stringify(manifest)], { type: "application/manifest+json" });
  setLink('manifest', URL.createObjectURL(blob));

  // iOS ignora el manifest: usa apple-touch-icon + estos meta tags.
  setLink('apple-touch-icon', ICON180);
  setMeta('apple-mobile-web-app-capable', 'yes');
  setMeta('mobile-web-app-capable', 'yes');
  setMeta('apple-mobile-web-app-status-bar-style', 'black-translucent');
  setMeta('apple-mobile-web-app-title', 'Coach');
  setMeta('theme-color', THEME);

  // viewport-fit=cover respeta el notch; maximum-scale frena el zoom de iOS
  // al enfocar un input.
  setMeta('viewport',
    'width=device-width, initial-scale=1, maximum-scale=1, viewport-fit=cover');
})();
</script>
"""

_MOBILE_CSS = """
<style>
/* Oculta el chrome de Streamlit: en el celular ocupa lugar y no se usa. */
#MainMenu, footer, header [data-testid="stStatusWidget"] { visibility: hidden; }

/* iOS hace zoom si el input mide menos de 16px. */
input, textarea, select { font-size: 16px !important; }

/* Respeta la barra de gestos / notch en modo standalone. */
.block-container {
  padding-top: 2.2rem !important;
  padding-bottom: calc(2rem + env(safe-area-inset-bottom)) !important;
}

@media (max-width: 640px) {
  .block-container { padding-left: 0.8rem !important; padding-right: 0.8rem !important; }

  /* Tap targets comodos con el pulgar. */
  .stButton button { min-height: 46px; font-size: 1rem; }

  /* Streamlit 1.58 dejo de usar data-baseweb="tab" (ahora es data-testid="stTab"),
     asi que la regla vieja no matcheaba nada y las 3 tabs no entraban en 375px.
     Se apunta a los tres selectores; role="tab" es el mas estable de todos. */
  [data-baseweb="tab"], [data-testid="stTab"], [role="tab"] {
    padding: 0.4rem 0.3rem !important;
    font-size: 0.85rem !important;
  }
  [data-baseweb="tab-list"], [role="tablist"] { overflow-x: auto; flex-wrap: nowrap; }

  h1 { font-size: 1.5rem !important; }
}
</style>
"""


def inject() -> None:
    """Registra manifest + meta tags de iOS y aplica el CSS mobile."""
    import streamlit.components.v1 as components

    js = (_INJECT_JS
          .replace("__ICON192__", f"data:image/png;base64,{_icon_png_b64(192)}")
          .replace("__ICON512__", f"data:image/png;base64,{_icon_png_b64(512)}")
          .replace("__ICON180__", f"data:image/png;base64,{_icon_png_b64(180)}")
          .replace("__THEME__", THEME_BG))
    components.html(js, height=0, width=0)
    st.markdown(_MOBILE_CSS, unsafe_allow_html=True)
