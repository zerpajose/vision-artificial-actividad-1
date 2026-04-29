# ===========================================================================
# Vision_Artificial_Actividad1_Completo
#
# Universidad Internacional de La Rioja (UNIR)
# Visión Artificial — Actividad 1
# Profesor: Javier Rodrigo Villazon Terrazas
# ===========================================================================

# INSTRUCCIONES DE USO:
#   Google Colab : sube este .py y ejecútalo con  !python <nombre>.py
#   Local        : python <nombre>.py
#   VS Code      : abre el .py y usa el botón "Run"


# -------------------------------------------------------------------------
# <div style="background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%); padding: 40px; border-radius: 15px; margin-bottom: 20px;">
# <h1 style="color: #e94560; font-size: 2.5em; text-align: center; font-family: 'Arial', sans-serif; margin-bottom: 10px;">🔬 VISIÓN ARTIFICIAL</h1>
# <h2 style="color: #ffffff; font-size: 1.6em; text-align: center; font-family: 'Arial', sans-serif; margin-bottom: 5px;">Temas 1–6 · Laboratorio Completo</h2>
# <h3 style="color: #a8dadc; font-size: 1.2em; text-align: center; font-family: 'Arial', sans-serif;">Actividad 1: Mejora de imagen — Operaciones Elementales</h3>
# <hr style="border-color: #e94560; margin: 20px 0;">
# <p style="color: #ccc; text-align: center; font-size: 0.95em;">Universidad Internacional de La Rioja (UNIR) · Procesamiento Digital de Imágenes</p>
# </div>
# 
# ---
# 
# ## 📋 Guía de uso de este notebook
# 
# Este notebook está diseñado para una **sesión de 2 horas** y cubre todo el material de los Temas 1 al 6 de la asignatura. Está estructurado en tres grandes bloques:
# 
# | Bloque | Contenido | Duración estimada |
# |--------|-----------|-------------------|
# | 🧠 **I — Fundamentos (T1-T5)** | Percepción, digitalización, modelo de imagen | ~25 min |
# | ⚙️ **II — Operaciones Elementales (T6)** | Ajuste de intensidad, histograma, operadores | ~55 min |
# | 🧪 **III — Actividad 1 Guiada** | Aplicación sobre Dark Face Dataset | ~40 min |
# 
# > ⚡ **Consejo:** Ejecuta las celdas **en orden de arriba a abajo**. La primera celda instala todas las dependencias necesarias.


# -------------------------------------------------------------------------
# ---
# # 🔧 CONFIGURACIÓN DEL ENTORNO
# > Ejecuta esta celda primero. Instala todas las librerías necesarias para el notebook.


# =========================================================================
# CELDA 1
# =========================================================================
# ============================================================
# CELDA 0 — Instalación de dependencias y configuración global
# ============================================================
import subprocess, sys; subprocess.run([sys.executable, "-m", "pip", "install", "scikit-image", "matplotlib", "numpy", "opencv-python-headless", "scipy", "Pillow"], check=False)

# ── Importaciones ────────────────────────────────────────────
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import cv2
from PIL import Image
import urllib.request
import os, io, warnings

from skimage import data, exposure, filters, morphology, util
from skimage.color import rgb2gray
from skimage.util import img_as_float, img_as_ubyte
from scipy import ndimage

warnings.filterwarnings('ignore')

# ── Estilo global de gráficas ─────────────────────────────────
plt.rcParams.update({
    'figure.facecolor': '#0d1117',
    'axes.facecolor':   '#161b22',
    'axes.edgecolor':   '#30363d',
    'axes.labelcolor':  '#c9d1d9',
    'xtick.color':      '#8b949e',
    'ytick.color':      '#8b949e',
    'text.color':       '#c9d1d9',
    'figure.titlesize': 14,
    'axes.titlesize':   12,
    'axes.titlecolor':  '#58a6ff',
    'grid.color':       '#21262d',
    'grid.linestyle':   '--',
    'grid.alpha':       0.5,
})

# ── Función utilitaria: mostrar imagen + histograma ───────────
def mostrar_imagen_histograma(imgs, titulos, figsize=(18, 5), cmap_list=None):
    """
    Muestra N imágenes con sus histogramas en una figura.
    imgs     : lista de arrays numpy (uint8 o float)
    titulos  : lista de strings con nombres
    """
    n = len(imgs)
    fig, axes = plt.subplots(2, n, figsize=figsize)
    if n == 1:
        axes = axes.reshape(2, 1)
    
    colores_hist = ['#58a6ff', '#3fb950', '#f78166', '#d2a8ff', '#ffa657']
    
    for i, (img, titulo) in enumerate(zip(imgs, titulos)):
        cmap = 'gray' if (cmap_list is None or len(cmap_list) <= i) else cmap_list[i]
        # Imagen
        axes[0, i].imshow(img if img.ndim == 3 else img, cmap=cmap)
        axes[0, i].set_title(titulo, fontsize=10, pad=8)
        axes[0, i].axis('off')
        # Histograma
        img_norm = img_as_float(img) if img.dtype != np.float64 else img
        if img.ndim == 3:
            for c, col in enumerate(['#f78166', '#3fb950', '#58a6ff']):
                axes[1, i].hist(img[:,:,c].ravel(), bins=128, color=col,
                                alpha=0.5, density=True)
        else:
            axes[1, i].hist(img_norm.ravel(), bins=128,
                            color=colores_hist[i % len(colores_hist)],
                            density=True, alpha=0.85)
        axes[1, i].set_xlabel('Nivel de intensidad')
        axes[1, i].set_ylabel('Densidad')
        axes[1, i].grid(True)
    
    fig.tight_layout(pad=2)
    plt.show()

# ── Función utilitaria: métricas de calidad ───────────────────
def calcular_metricas(img_original, img_procesada):
    """
    Calcula métricas básicas de evaluación de calidad de imagen.
    Retorna un diccionario con MSE, PSNR, contraste (STD) y entropía.
    """
    o = img_as_float(img_original)
    p = img_as_float(img_procesada)
    mse  = np.mean((o - p) ** 2)
    psnr = 10 * np.log10(1.0 / mse) if mse > 0 else float('inf')
    std_o = np.std(o)
    std_p = np.std(p)
    # Entropía
    def entropia(img):
        hist, _ = np.histogram(img, bins=256, range=(0, 1), density=True)
        hist = hist[hist > 0]
        return -np.sum(hist * np.log2(hist + 1e-10)) * (1/256)
    return {
        'MSE':           round(mse, 6),
        'PSNR (dB)':     round(psnr, 2),
        'Contraste STD orig':  round(float(std_o), 4),
        'Contraste STD proc':  round(float(std_p), 4),
        'Entropía orig':       round(float(entropia(o)), 4),
        'Entropía proc':       round(float(entropia(p)), 4),
    }

print('✅ Entorno configurado correctamente. ¡Puedes continuar!')


# -------------------------------------------------------------------------
# ---
# # 🧠 BLOQUE I — FUNDAMENTOS TEÓRICOS (Temas 1 – 5)
# 
# En este bloque repasamos los conceptos fundamentales que sustentan toda la asignatura y que son necesarios para entender las técnicas aplicadas en la Actividad 1.


# -------------------------------------------------------------------------
# ## 📖 Tema 1 · Introducción a los Sistemas de Percepción
# 
# Un **sistema de percepción** computacional imita el proceso de percepción del mundo natural y se estructura en tres módulos fundamentales:
# 
# ```
#   [ CAPTURA ] ──►  [ PROCESAMIENTO ] ──►  [ TOMA DE DECISIÓN ]
#       ↑                                          │
#       └──────────── retroalimentación ──────────┘
# ```
# 
# | Módulo | Función | Analogía biológica |
# |--------|---------|--------------------|
# | **Captura** | Adquirir información del entorno | Ojos, oídos, piel |
# | **Procesamiento** | Transformar y analizar los datos | Corteza cerebral |
# | **Decisión** | Interpretar y actuar | Respuesta motora |
# 
# ### ¿Por qué es difícil replicar la visión humana?
# - El ojo humano tiene ~120M fotorreceptores con procesamiento paralelo masivo.
# - La corteza visual ocupa ~30% del córtex cerebral.
# - Incorpora contexto, experiencia y aprendizaje en tiempo real.


# -------------------------------------------------------------------------
# ## 📖 Tema 2 · Elementos de un Sistema de Percepción — Parámetros de Captura
# 
# Los parámetros clave que definen la calidad de un sistema de captura son:
# 
# - **Especificidad**: capacidad de captar únicamente el fenómeno de interés.
# - **Precisión**: nivel de error en la medida (p.ej. resolución espacial).
# - **Rango dinámico**: relación entre el máximo y mínimo valor capturado.
# - **Frecuencia de muestreo**: cuántos valores por segundo/pixel se capturan.
# 
# > **Relevancia para la Actividad 1:** Las imágenes del *Dark Face Dataset* presentan **bajo rango dinámico** (píxeles concentrados en valores oscuros), lo que justifica aplicar técnicas de mejora de contraste.


# -------------------------------------------------------------------------
# ## 📖 Tema 3 · Captura y Digitalización de Señales


# =========================================================================
# CELDA 2
# =========================================================================
# ============================================================
# CELDA T3-1 — Muestreo y cuantificación: efecto en la imagen
# ============================================================
# Cargamos una imagen de ejemplo de skimage
img_original = data.camera()   # imagen en escala de grises, 512×512

niveles = [256, 64, 16, 4]
fig, axes = plt.subplots(1, 4, figsize=(18, 5))
fig.suptitle('Efecto de la CUANTIFICACIÓN: reducción de niveles de intensidad', fontsize=13)

for ax, L in zip(axes, niveles):
    # Reducir a L niveles uniformes
    factor = 256 // L
    img_cuant = (img_original // factor) * factor
    ax.imshow(img_cuant, cmap='gray', vmin=0, vmax=255)
    ax.set_title(f'{L} niveles de intensidad')
    ax.axis('off')

plt.tight_layout()
plt.show()

print("""
📌 INTERPRETACIÓN:
  • Con 256 niveles (8 bits): la imagen presenta transiciones suaves y
    alto detalle visual.
  • Con 64 niveles (6 bits): comienzan a apreciarse ligeras bandas
    (efecto 'posterization').
  • Con 16 niveles (4 bits): los contornos son notorios y se pierde
    información de texturas finas.
  • Con 4 niveles (2 bits): la imagen queda prácticamente binaria;
    solo se distinguen objetos grandes.

  ➤ Conclusión: 8 bits (256 niveles) es el estándar de facto para
    imágenes en escala de grises. Para imágenes médicas o satelitales
    se usan 12 o 16 bits.
""")


# =========================================================================
# CELDA 3
# =========================================================================
# ============================================================
# CELDA T3-2 — Efecto del MUESTREO ESPACIAL (resolución)
# ============================================================
img_hr = data.camera()   # 512×512 píxeles
resoluciones = [512, 256, 128, 64]

fig, axes = plt.subplots(1, 4, figsize=(18, 5))
fig.suptitle('Efecto del MUESTREO ESPACIAL: reducción de resolución', fontsize=13)

for ax, res in zip(axes, resoluciones):
    # Submuestrear y luego ampliar para visualización uniforme
    factor = 512 // res
    img_sub = img_hr[::factor, ::factor]  # submuestreo
    img_vis = np.kron(img_sub, np.ones((factor, factor), dtype=np.uint8))  # ampliar
    ax.imshow(img_vis, cmap='gray', vmin=0, vmax=255)
    ax.set_title(f'{res}×{res} px')
    ax.axis('off')

plt.tight_layout()
plt.show()

print("""
📌 INTERPRETACIÓN:
  • 512×512: imagen de referencia, sin pérdida.
  • 256×256: calidad aceptable, ligera pérdida de detalle fino.
  • 128×128: comienza a apreciarse el efecto pixelado (aliasing).
  • 64×64: la imagen es irreconocible en sus detalles.

  ➤ Teorema de Nyquist: la frecuencia de muestreo debe ser al menos
    el DOBLE de la frecuencia máxima presente en la señal.
""")


# -------------------------------------------------------------------------
# ## 📖 Temas 4 y 5 · Modelo Matemático de Imagen y Espacios de Color


# =========================================================================
# CELDA 4
# =========================================================================
# ============================================================
# CELDA T4-1 — Representación matricial de una imagen
# ============================================================
img = data.camera()

print('='*60)
print('REPRESENTACIÓN MATEMÁTICA DE UNA IMAGEN DIGITAL')
print('='*60)
print(f'  Tipo de dato : {img.dtype}')
print(f'  Dimensiones  : {img.shape}  (filas × columnas)')
print(f'  Rango de vals: [{img.min()}, {img.max()}]')
print(f'  Total píxeles: {img.size:,}')
print(f'  Memoria       : {img.nbytes / 1024:.1f} KB')
print()
print('Submatriz 8×8 (esquina superior izquierda):')
print(img[:8, :8])

# Visualización
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle('Análisis matricial de una imagen digital', fontsize=13)

axes[0].imshow(img, cmap='gray')
axes[0].set_title('Imagen completa (512×512)')
axes[0].axis('off')

# Zoom a 16×16 píxeles con valores superpuestos
patch = img[250:258, 250:258]
axes[1].imshow(patch, cmap='gray', vmin=0, vmax=255, interpolation='nearest')
axes[1].set_title('Zoom: bloque 8×8 px\n(valores visibles)')
for i in range(8):
    for j in range(8):
        color = 'white' if patch[i, j] < 128 else 'black'
        axes[1].text(j, i, str(patch[i, j]), ha='center', va='center',
                     fontsize=7, color=color, fontweight='bold')
axes[1].axis('off')

# Perfil de intensidad horizontal
fila = img[256, :]   # fila central
axes[2].plot(fila, color='#58a6ff', linewidth=1)
axes[2].fill_between(range(len(fila)), fila, alpha=0.3, color='#58a6ff')
axes[2].set_title('Perfil de intensidad — fila 256')
axes[2].set_xlabel('Posición horizontal (x)')
axes[2].set_ylabel('Nivel de intensidad f(256, x)')
axes[2].grid(True)

plt.tight_layout()
plt.show()


# =========================================================================
# CELDA 5
# =========================================================================
# ============================================================
# CELDA T4-2 — Espacios de color: RGB, HSV y LAB
# ============================================================
from skimage.color import rgb2hsv, rgb2lab

# Imagen de color de ejemplo
img_color = data.astronaut()   # 512×512×3, uint8 RGB

# Conversiones
img_hsv  = rgb2hsv(img_color)
img_lab  = rgb2lab(img_color)

fig, axes = plt.subplots(3, 4, figsize=(18, 13))
fig.suptitle('Espacios de color: RGB · HSV · LAB', fontsize=14)

# Fila 1: RGB
nombres_rgb = ['RGB — Imagen completa', 'Canal R (rojo)', 'Canal G (verde)', 'Canal B (azul)']
datos_rgb   = [img_color,
               img_color[:,:,0], img_color[:,:,1], img_color[:,:,2]]
cmaps_rgb   = [None, 'Reds_r', 'Greens_r', 'Blues_r']
for ax, dato, nombre, cmap in zip(axes[0], datos_rgb, nombres_rgb, cmaps_rgb):
    ax.imshow(dato, cmap=cmap)
    ax.set_title(nombre)
    ax.axis('off')

# Fila 2: HSV
nombres_hsv = ['HSV — Hue (matiz)', 'HSV — Saturation (saturación)', 'HSV — Value (brillo)', '']
datos_hsv   = [img_hsv[:,:,0], img_hsv[:,:,1], img_hsv[:,:,2], None]
cmaps_hsv   = ['hsv', 'gray', 'gray', None]
for ax, dato, nombre, cmap in zip(axes[1], datos_hsv, nombres_hsv, cmaps_hsv):
    if dato is not None:
        ax.imshow(dato, cmap=cmap)
        ax.set_title(nombre)
    ax.axis('off')
# Texto explicativo
axes[1, 3].text(0.1, 0.5,
    'H: ángulo de color (0–360°)\nS: pureza del color (0–1)\nV: luminosidad (0–1)\n\n'
    '→ Canal V equivale\n  aproximadamente\n  a la escala de grises',
    transform=axes[1, 3].transAxes, fontsize=10, va='center',
    bbox=dict(boxstyle='round', facecolor='#21262d', alpha=0.8))

# Fila 3: LAB
# Normalizar para visualización
L_norm = (img_lab[:,:,0] - img_lab[:,:,0].min()) / (img_lab[:,:,0].ptp() + 1e-8)
a_norm = (img_lab[:,:,1] - img_lab[:,:,1].min()) / (img_lab[:,:,1].ptp() + 1e-8)
b_norm = (img_lab[:,:,2] - img_lab[:,:,2].min()) / (img_lab[:,:,2].ptp() + 1e-8)
datos_lab  = [L_norm, a_norm, b_norm, None]
nombres_lab = ['LAB — L* (luminosidad)', 'LAB — a* (verde↔rojo)', 'LAB — b* (azul↔amarillo)', '']
cmaps_lab  = ['gray', 'RdYlGn_r', 'PiYG_r', None]
for ax, dato, nombre, cmap in zip(axes[2], datos_lab, nombres_lab, cmaps_lab):
    if dato is not None:
        ax.imshow(dato, cmap=cmap)
        ax.set_title(nombre)
    ax.axis('off')
axes[2, 3].text(0.1, 0.5,
    'L*: 0 (negro) – 100 (blanco)\na*: −128 (verde) – +127 (rojo)\nb*: −128 (azul) – +127 (amarillo)\n\n'
    '→ Perceptualmente uniforme:\n  igual Δ numérico = igual\n  diferencia visual percibida',
    transform=axes[2, 3].transAxes, fontsize=10, va='center',
    bbox=dict(boxstyle='round', facecolor='#21262d', alpha=0.8))

plt.tight_layout()
plt.show()


# -------------------------------------------------------------------------
# ---
# # ⚙️ BLOQUE II — TEMA 6: OPERACIONES ELEMENTALES DE MEJORA DE IMAGEN
# 
# > **González & Woods (2008):** *"El objetivo del realce de imagen es procesar una imagen de modo que el resultado sea más adecuado que la imagen original para una aplicación específica."*
# 
# ## Modelo matemático general
# 
# Todas las operaciones de este bloque son **operaciones punto a punto** (point operations). El píxel en la imagen de salida depende **únicamente** del píxel correspondiente en la imagen de entrada:
# 
# $$g(x, y) = T[f(x, y)]$$
# 
# Donde:
# - $f(x, y)$: imagen original
# - $g(x, y)$: imagen procesada
# - $T[\cdot]$: función/operador de transformación
# - $(x, y)$: coordenadas del píxel
# 
# Esta operación es computacionalmente muy eficiente porque no requiere considerar los píxeles vecinos.


# -------------------------------------------------------------------------
# ## 6.2 · Ajuste de Intensidad — Funciones de Transformación Punto a Punto


# =========================================================================
# CELDA 6
# =========================================================================
# ============================================================
# CELDA T6-2-A — Visualización de las funciones de transformación
# (Figura 3 de González y Woods)
# ============================================================
L = 256  # número máximo de niveles de intensidad
r = np.linspace(0, L-1, 500)  # rango de entrada

fig, ax = plt.subplots(figsize=(10, 6))
fig.suptitle('Funciones de Transformación Punto a Punto más comunes\n(Tema 6 — Figura 3, González y Woods)', fontsize=13)

# 1. Identidad
ax.plot(r, r, color='#8b949e', linewidth=1.5, linestyle='--', label='Identidad: g = f')

# 2. Negativo
ax.plot(r, L-1-r, color='#f78166', linewidth=2.5, label='Negativo: g = (L-1) - f')

# 3. Logarítmica (c ajustado para escalar a [0, 255])
c_log = (L-1) / np.log(1 + (L-1))
ax.plot(r, c_log * np.log(1 + r), color='#3fb950', linewidth=2.5, label='Logarítmica: g = c·log(1+f)')

# 4. Ley de potencia — gamma < 1 (realce oscuros)
ax.plot(r, (r/(L-1))**0.3 * (L-1), color='#58a6ff', linewidth=2.5, label='Gamma γ=0.3 (realza oscuros)')

# 5. Ley de potencia — gamma > 1 (realce claros)
ax.plot(r, (r/(L-1))**2.5 * (L-1), color='#d2a8ff', linewidth=2.5, label='Gamma γ=2.5 (realza claros)')

# 6. Realce de contraste (función a trozos)
f_piecewise = np.where(r < 50,  r * 0.3,
              np.where(r < 200, (r - 50) * (205 / 150) + 15,
                                (r - 200) * 0.3 + 220))
ax.plot(r, f_piecewise, color='#ffa657', linewidth=2.5, label='Función a trozos (realce contraste)')

ax.set_xlabel('Intensidad de entrada  f(x,y)', fontsize=11)
ax.set_ylabel('Intensidad de salida  g(x,y)', fontsize=11)
ax.set_xlim(0, 255)
ax.set_ylim(0, 255)
ax.legend(loc='lower right', fontsize=9)
ax.grid(True)
ax.set_aspect('equal')

plt.tight_layout()
plt.show()

print("""
📌 INTERPRETACIÓN DE LAS CURVAS:
  • La diagonal (identidad) significa: sin transformación.
  • Curvas POR ENCIMA de la diagonal: aumentan la luminosidad.
  • Curvas POR DEBAJO de la diagonal: oscurecen la imagen.
  • Una pendiente EMPINADA en un rango concreto: aumenta el contraste en ese rango.
  • Una pendiente PLANA: comprime ese rango de intensidades.
""")


# -------------------------------------------------------------------------
# ### 6.2.1 · Negativo de una Imagen
# 
# $$g(x,y) = (L-1) - f(x,y)$$
# 
# Para imágenes de 8 bits: $g = 255 - f$
# 
# **Utilidad:** Resaltar estructuras blancas en fondos oscuros (imágenes médicas, rayos X). La transformación es **lineal e invertible** — no modifica el contenido visual, solo invierte la representación.


# =========================================================================
# CELDA 7
# =========================================================================
# ============================================================
# CELDA T6-2-B — NEGATIVO de imagen
# ============================================================
img_rayos_x = data.chest_xray() if hasattr(data, 'chest_xray') else data.camera()
if img_rayos_x.ndim == 3:
    img_rayos_x = rgb2gray(img_rayos_x)
    img_rayos_x = img_as_ubyte(img_rayos_x)

# ── Aplicar negativo ──────────────────────────────────────────
img_negativo = 255 - img_rayos_x

# ── Visualización ─────────────────────────────────────────────
mostrar_imagen_histograma(
    [img_rayos_x, img_negativo],
    ['Original', 'Negativo: g = 255 − f'],
    figsize=(12, 5)
)

# Verificar la propiedad de invertibilidad
img_reconstruct = 255 - img_negativo
print(f'✅ Invertibilidad del negativo: MSE = {np.mean((img_rayos_x.astype(int) - img_reconstruct.astype(int))**2):.0f}')
print()
metricas = calcular_metricas(img_rayos_x, img_negativo)
print('📊 Métricas:')
for k, v in metricas.items():
    print(f'   {k:30s}: {v}')


# -------------------------------------------------------------------------
# ### 6.2.2 · Transformación Logarítmica
# 
# $$g(x,y) = c \cdot \log(1 + f(x,y))$$
# 
# Donde $c$ es una constante de escalado que mapea el resultado al rango $[0, L-1]$.
# 
# **Utilidad:** Expandir el rango de los píxeles oscuros y comprimir los claros. Ideal para visualizar el **espectro de Fourier** o imágenes con alto rango dinámico.


# =========================================================================
# CELDA 8
# =========================================================================
# ============================================================
# CELDA T6-2-C — TRANSFORMACIÓN LOGARÍTMICA
# ============================================================
img_gris = data.camera().astype(np.float64)

# Constante c para escalar la salida a [0, 255]
c = 255 / np.log(1 + img_gris.max())
img_log = (c * np.log(1 + img_gris)).astype(np.uint8)

# ── Ejemplo 2: Espectro de Fourier ──────────────────────────
# El espectro tiene rango dinámico MUY alto; sin log solo vemos un punto
F = np.fft.fft2(img_gris)
F_shift = np.fft.fftshift(F)
magnitud = np.abs(F_shift)
espectro_sin_log = (magnitud / magnitud.max() * 255).astype(np.uint8)
c2 = 255 / np.log(1 + magnitud.max())
espectro_con_log = (c2 * np.log(1 + magnitud)).astype(np.uint8)

# ── Figura 1: imagen general ─────────────────────────────────
mostrar_imagen_histograma(
    [data.camera(), img_log],
    ['Original', 'Transformación logarítmica\ng = c·log(1+f)'],
    figsize=(12, 5)
)

# ── Figura 2: espectro de Fourier ────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('Utilidad práctica: representación del espectro de Fourier\n'
             '(Figura 5 del Tema 6 — González y Woods)', fontsize=12)
axes[0].imshow(espectro_sin_log, cmap='gray')
axes[0].set_title('Módulo de la TF — sin transformación\n(solo vemos la componente DC central)')
axes[0].axis('off')
axes[1].imshow(espectro_con_log, cmap='gray')
axes[1].set_title('Módulo de la TF — con transformación log\n(ahora son visibles las componentes de alta frecuencia)')
axes[1].axis('off')
plt.tight_layout()
plt.show()

print(f'📌 Rango dinámico espectro: {magnitud.min():.0f} – {magnitud.max():.0f}')
print(f'   La diferencia es de orden {magnitud.max()/magnitud.min():.0f}x')
print('   Sin log: solo el punto central (componente DC) es visible.')
print('   Con log: toda la distribución espectral es apreciable.')


# -------------------------------------------------------------------------
# ### 6.2.3 · Ley de Potencia (Corrección Gamma)
# 
# $$g(x,y) = c \cdot f(x,y)^\gamma$$
# 
# Con $c$ y $\gamma$ constantes positivas. Esta familia de curvas permite:
# - $\gamma < 1$: expansión de niveles oscuros (imagen más brillante).
# - $\gamma > 1$: compresión de niveles oscuros (imagen más oscura).
# - $\gamma = 1$: identidad (sin cambio).
# 
# **Utilidad práctica:** corrección gamma de monitores, cámaras e impresoras.


# =========================================================================
# CELDA 9
# =========================================================================
# ============================================================
# CELDA T6-2-D — LEY DE POTENCIA (Gamma)
# ============================================================
img_base = data.camera().astype(np.float64) / 255.0   # normalizar a [0,1]

gammas = [0.1, 0.3, 0.5, 1.0, 1.5, 2.5, 5.0]
n = len(gammas)

fig, axes = plt.subplots(2, n, figsize=(22, 7))
fig.suptitle('Ley de Potencia: familia de transformaciones gamma\n'
             'g = c · f^γ  (c normalizado para que la salida esté en [0,1])', fontsize=12)

for i, gamma in enumerate(gammas):
    img_gamma = np.power(img_base, gamma)
    axes[0, i].imshow(img_gamma, cmap='gray', vmin=0, vmax=1)
    color = '#f78166' if gamma > 1 else ('#3fb950' if gamma < 1 else '#8b949e')
    axes[0, i].set_title(f'γ = {gamma}', color=color, fontsize=10)
    axes[0, i].axis('off')
    axes[1, i].hist(img_gamma.ravel(), bins=64, color=color, density=True, alpha=0.85)
    axes[1, i].set_xlim(0, 1)
    axes[1, i].set_xlabel('Intensidad', fontsize=8)

# Etiquetas especiales
axes[0, 3].set_title('γ = 1.0\n(Identidad)', color='#8b949e', fontsize=10)

plt.tight_layout()
plt.show()

# ── Curvas de transformación ─────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 6))
fig.suptitle('Curvas de transformación gamma (Figura 6 — Tema 6)', fontsize=12)
r_norm = np.linspace(0, 1, 500)
colores_curvas = plt.cm.RdYlGn(np.linspace(0, 1, len(gammas)))
for gamma, col in zip(gammas, colores_curvas):
    label = f'γ={gamma}' + (' ← expansión oscuros' if gamma == 0.3 else
                             ' ← compresión oscuros' if gamma == 2.5 else '')
    ax.plot(r_norm, np.power(r_norm, gamma), color=col, linewidth=2, label=label)
ax.plot([0,1],[0,1], 'w--', linewidth=1, alpha=0.5, label='Identidad (γ=1)')
ax.set_xlabel('Intensidad de entrada f')
ax.set_ylabel('Intensidad de salida g')
ax.legend(fontsize=8, loc='upper left')
ax.grid(True)
ax.set_xlim(0,1); ax.set_ylim(0,1)
plt.tight_layout()
plt.show()

print("""
📌 RELEVANCIA PARA LA ACTIVIDAD 1:
  Las imágenes de baja iluminación del Dark Face Dataset tienen sus
  píxeles concentrados en valores bajos. Aplicar γ < 1 (por ejemplo
  γ = 0.3 o 0.5) expande ese rango y MEJORA drásticamente la visibilidad.
  
  Experimentación sugerida: γ ∈ {0.3, 0.5, 0.7} para imágenes oscuras.
""")


# -------------------------------------------------------------------------
# ### 6.2.4 · Función Definida a Trozos — Realce de Contraste
# 
# Permite diseñar **manualmente** una transformación que actúe de forma diferente en distintos rangos de intensidad. Es muy flexible pero requiere conocimiento previo de la distribución de la imagen.


# =========================================================================
# CELDA 10
# =========================================================================
# ============================================================
# CELDA T6-2-E — FUNCIÓN A TROZOS (Realce de contraste)
# ============================================================
img_low_contrast = data.camera()   # imagen base

# Simular imagen de bajo contraste comprimiendo el rango a [60, 180]
img_lc = (img_low_contrast.astype(np.float64) / 255.0 * 120 + 60).astype(np.uint8)

def funcion_a_trozos(img, puntos):
    """
    Aplica una función de transformación a trozos lineal por partes.
    puntos: lista de (x_in, x_out) que definen los puntos de control.
    """
    img_f  = img.astype(np.float64)
    lut    = np.zeros(256, dtype=np.float64)
    pts_x  = [p[0] for p in puntos]
    pts_y  = [p[1] for p in puntos]
    for i in range(256):
        lut[i] = np.interp(i, pts_x, pts_y)
    return np.clip(lut[img], 0, 255).astype(np.uint8)

# Transformación: expandir el rango de interés [60,180] a [0,255]
# Puntos de control: (entrada, salida)
puntos_estiramiento = [(0, 0), (60, 0), (180, 255), (255, 255)]
img_stretched = funcion_a_trozos(img_lc, puntos_estiramiento)

# Transformación de 3 tramos: oscuro comprimido, medio expandido, claro comprimido
puntos_3tramos = [(0, 0), (80, 40), (150, 210), (255, 255)]
img_3tramos = funcion_a_trozos(img_lc, puntos_3tramos)

# ── Figura: imágenes + curvas de transformación ──────────────
fig = plt.figure(figsize=(18, 12))
fig.suptitle('Función definida a trozos — Realce de contraste selectivo', fontsize=13)

gs = gridspec.GridSpec(2, 4, figure=fig, hspace=0.45, wspace=0.3)

imagenes = [img_lc, img_stretched, img_3tramos]
titulos  = ['Imagen bajo contraste\n(rango comprimido 60–180)',
            'Estiramiento simple\n(expansión lineal)',
            'Función 3 tramos\n(realce selectivo zona media)']
for col, (img_show, titulo) in enumerate(zip(imagenes, titulos)):
    ax_img = fig.add_subplot(gs[0, col])
    ax_img.imshow(img_show, cmap='gray', vmin=0, vmax=255)
    ax_img.set_title(titulo, fontsize=9)
    ax_img.axis('off')
    ax_hist = fig.add_subplot(gs[1, col])
    ax_hist.hist(img_show.ravel(), bins=64, color=['#58a6ff','#3fb950','#ffa657'][col],
                 density=True, alpha=0.85)
    ax_hist.set_xlim(0, 255)
    ax_hist.grid(True)
    ax_hist.set_xlabel('Intensidad')
    ax_hist.set_ylabel('Densidad')

# Curva de transformación
ax_curva = fig.add_subplot(gs[:, 3])
r_input = np.arange(256)
# Calcular LUT manualmente para visualización
def calcular_lut(puntos):
    px = [p[0] for p in puntos]
    py = [p[1] for p in puntos]
    return np.interp(r_input, px, py)

ax_curva.plot(r_input, r_input, '--', color='#8b949e', linewidth=1.5, label='Identidad')
ax_curva.plot(r_input, calcular_lut(puntos_estiramiento),
              color='#3fb950', linewidth=2.5, label='Estiramiento')
ax_curva.plot(r_input, calcular_lut(puntos_3tramos),
              color='#ffa657', linewidth=2.5, label='3 tramos')
ax_curva.set_title('Curvas de transformación\nsuperpuestas')
ax_curva.set_xlabel('Entrada f')
ax_curva.set_ylabel('Salida g')
ax_curva.legend()
ax_curva.grid(True)

plt.show()


# -------------------------------------------------------------------------
# ## 6.3 · Procesado Sistemático del Histograma
# 
# En lugar de diseñar manualmente la función de transformación, podemos **derivarla automáticamente** a partir de las propiedades estadísticas del histograma.
# 
# ### Recordatorio: Histograma de una imagen
# 
# El histograma de una imagen digital $f$ con $L$ niveles de intensidad es:
# 
# $$h(r_k) = n_k \quad k = 0, 1, \ldots, L-1$$
# 
# Su versión normalizada (estimación de la PDF):
# 
# $$p(r_k) = \frac{n_k}{MN}$$
# 
# Donde $n_k$ = número de píxeles con valor $r_k$, y $MN$ = total de píxeles.


# =========================================================================
# CELDA 11
# =========================================================================
# ============================================================
# CELDA T6-3-A — Análisis del histograma
# ============================================================
# Tres tipos de imágenes con diferentes distribuciones
img_oscura  = (data.camera().astype(np.float64) * 0.35).astype(np.uint8)
img_normal  = data.camera()
img_clara   = np.clip(data.camera().astype(np.float64) * 1.8, 0, 255).astype(np.uint8)
img_bc      = (data.camera().astype(np.float64) / 255 * 100 + 80).astype(np.uint8)  # bajo contraste

imgs_demo = [img_oscura, img_normal, img_clara, img_bc]
labels    = ['Imagen oscura\n(subexpuesta)',
             'Imagen normal\n(bien expuesta)',
             'Imagen sobreexpuesta\n(saturada)',
             'Imagen bajo contraste\n(rango comprimido)']
colores   = ['#f78166', '#3fb950', '#ffa657', '#58a6ff']

fig, axes = plt.subplots(2, 4, figsize=(20, 8))
fig.suptitle('Relación entre el histograma y las características visuales de la imagen', fontsize=13)

for i, (img_i, label_i, color_i) in enumerate(zip(imgs_demo, labels, colores)):
    axes[0, i].imshow(img_i, cmap='gray', vmin=0, vmax=255)
    axes[0, i].set_title(label_i, fontsize=9)
    axes[0, i].axis('off')
    
    hist_vals, bins = np.histogram(img_i.ravel(), bins=256, range=(0, 255))
    axes[1, i].bar(bins[:-1], hist_vals, width=1, color=color_i, alpha=0.85)
    
    media = img_i.mean()
    std   = img_i.std()
    axes[1, i].axvline(media, color='white', linewidth=1.5, linestyle='--',
                        label=f'μ={media:.0f}')
    axes[1, i].set_xlabel('Nivel de intensidad')
    axes[1, i].set_xlim(0, 255)
    axes[1, i].set_title(f'μ={media:.1f}  σ={std:.1f}', fontsize=9)
    axes[1, i].grid(True)

plt.tight_layout()
plt.show()


# -------------------------------------------------------------------------
# ### 6.3.1 · Ecualización del Histograma
# 
# La ecualización busca convertir el histograma de la imagen en una distribución **uniforme**, maximizando el contraste de forma automática.
# 
# **Función de transformación:** la función de distribución acumulada (CDF) normalizada:
# 
# $$T(r_k) = (L-1) \sum_{j=0}^{k} p(r_j) = \frac{(L-1)}{MN} \sum_{j=0}^{k} n_j$$


# =========================================================================
# CELDA 12
# =========================================================================
# ============================================================
# CELDA T6-3-B — ECUALIZACIÓN del histograma (manual + skimage)
# ============================================================

def equalizar_histograma(img):
    """
    Implementación manual de ecualización de histograma.
    Sigue el algoritmo del Tema 6 paso a paso.
    """
    # Paso 1: Calcular histograma normalizado (PDF)
    hist, bins = np.histogram(img.ravel(), bins=256, range=(0, 256))
    pdf = hist / float(img.size)
    
    # Paso 2: Calcular CDF (función de distribución acumulada)
    cdf = np.cumsum(pdf)
    
    # Paso 3: Escalar la CDF al rango [0, 255]
    lut = np.round(cdf * 255).astype(np.uint8)
    
    # Paso 4: Aplicar la LUT (Look-Up Table)
    return lut[img], pdf, cdf, lut

# Aplicar a imagen oscura
img_entrada = img_oscura
img_eq_manual, pdf_orig, cdf_orig, lut = equalizar_histograma(img_entrada)

# Con skimage (versión AHE y CLAHE)
img_eq_global  = exposure.equalize_hist(img_entrada)
img_eq_clahe   = exposure.equalize_adapthist(img_entrada, clip_limit=0.03)

# ── Figura principal ─────────────────────────────────────────
fig = plt.figure(figsize=(20, 14))
fig.suptitle('Ecualización del Histograma — Análisis completo\n'
             '(Tema 6, sección 6.3)', fontsize=14)

gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.5, wspace=0.35)

imgs_eq   = [img_entrada, img_eq_manual,
             img_as_ubyte(img_eq_global), img_as_ubyte(img_eq_clahe)]
titulos_eq = ['Original (subexpuesta)',
              'Ecualización manual\n(implementación propia)',
              'Ecualización global\n(skimage)',
              'CLAHE — Ecualización\nadaptativa por bloques']
colores_eq = ['#f78166', '#3fb950', '#58a6ff', '#d2a8ff']

for i, (img_i, titulo_i, color_i) in enumerate(zip(imgs_eq, titulos_eq, colores_eq)):
    # Imagen
    ax_img = fig.add_subplot(gs[0, i])
    ax_img.imshow(img_i, cmap='gray', vmin=0, vmax=255)
    ax_img.set_title(titulo_i, fontsize=9)
    ax_img.axis('off')
    # Histograma
    ax_h = fig.add_subplot(gs[1, i])
    ax_h.hist(img_i.ravel(), bins=128, color=color_i, density=True, alpha=0.85)
    ax_h.set_xlim(0, 255)
    ax_h.set_xlabel('Intensidad', fontsize=8)
    ax_h.set_ylabel('Densidad', fontsize=8)
    ax_h.grid(True)
    # CDF
    ax_cdf = fig.add_subplot(gs[2, i])
    h_i, b_i = np.histogram(img_i.ravel(), bins=256, range=(0,255))
    cdf_i = np.cumsum(h_i / float(img_i.size))
    ax_cdf.plot(cdf_i, color=color_i, linewidth=2)
    ax_cdf.set_xlim(0, 255)
    ax_cdf.set_ylim(0, 1)
    ax_cdf.set_xlabel('Intensidad', fontsize=8)
    ax_cdf.set_ylabel('CDF', fontsize=8)
    ax_cdf.grid(True)

plt.show()

# ── Métricas comparativas ────────────────────────────────────
print('\n📊 Comparación de métricas:')
print(f'{"Técnica":<35} {"μ (media)":>12} {"σ (std)":>12} {"Min":>8} {"Max":>8}')
print('-' * 80)
tecnicas = ['Original', 'Ecualización manual', 'Ecualización global', 'CLAHE']
for nombre, img_m in zip(tecnicas, imgs_eq):
    print(f'{nombre:<35} {img_m.mean():>12.1f} {img_m.std():>12.2f} '
          f'{img_m.min():>8} {img_m.max():>8}')


# -------------------------------------------------------------------------
# ### 6.3.2 · Especificación del Histograma (Histogram Matching)
# 
# Mientras la ecualización fuerza una distribución uniforme, la **especificación** permite obtener cualquier distribución deseada $p_z(z)$. Útil cuando se quiere que una imagen tenga el mismo perfil tonal que una imagen de referencia.


# =========================================================================
# CELDA 13
# =========================================================================
# ============================================================
# CELDA T6-3-C — ESPECIFICACIÓN del histograma
# ============================================================
from skimage.exposure import match_histograms

# Imagen de entrada: oscura
img_source = img_oscura

# Imagen de referencia: bien expuesta (usamos la versión normal)
img_ref    = img_normal

# Especificación del histograma
img_matched = match_histograms(img_source, img_ref)
if img_matched.dtype == np.float64:
    img_matched = img_as_ubyte(np.clip(img_matched, 0, 1))

fig, axes = plt.subplots(2, 3, figsize=(16, 8))
fig.suptitle('Especificación del Histograma (Histogram Matching)\n'
             'Transformar la distribución de la imagen fuente para que coincida con la referencia',
             fontsize=12)

pares = [(img_source, 'FUENTE: Imagen oscura'),
         (img_ref,    'REFERENCIA: Imagen normal'),
         (img_matched,'RESULTADO: Matching aplicado')]
colores_spec = ['#f78166', '#3fb950', '#58a6ff']

for i, ((img_i, titulo_i), color_i) in enumerate(zip(pares, colores_spec)):
    axes[0, i].imshow(img_i, cmap='gray', vmin=0, vmax=255)
    axes[0, i].set_title(titulo_i, fontsize=10)
    axes[0, i].axis('off')
    axes[1, i].hist(img_i.ravel(), bins=128, color=color_i, density=True, alpha=0.85)
    axes[1, i].set_xlim(0, 255)
    axes[1, i].set_xlabel('Intensidad')
    axes[1, i].set_ylabel('Densidad')
    axes[1, i].grid(True)

plt.tight_layout()
plt.show()

print("""
📌 DIFERENCIA CLAVE:
  • Ecualización:   distribución objetivo = UNIFORME (predeterminada)
  • Especificación: distribución objetivo = LA QUE TÚ ELIJAS (la de la imagen de referencia)
  
  → En la Actividad 1, puedes usar especificación para que tus imágenes
    oscuras tengan un perfil tonal similar al de una imagen de referencia
    tomada de día.
""")


# -------------------------------------------------------------------------
# ## 6.4 · Suavizado y Realce mediante Operadores Aritméticos
# 
# Los operadores aritméticos sobre imágenes permiten combinar dos o más imágenes para obtener efectos concretos:
# 
# | Operación | Expresión | Aplicación |
# |-----------|-----------|------------|
# | **Suma** | $g = f_1 + f_2$ | Fusión de imágenes |
# | **Resta** | $g = f_1 - f_2$ | Detección de cambios, eliminación de fondo |
# | **Promedio** | $g = \frac{1}{M}\sum_{i=1}^M f_i$ | Reducción de ruido |
# | **Producto** | $g = f_1 \cdot f_2$ | Enmascarado de regiones |


# =========================================================================
# CELDA 14
# =========================================================================
# ============================================================
# CELDA T6-4-A — PROMEDIADO de imágenes para reducción de ruido
# ============================================================
# Simulamos M capturas de la misma escena con ruido gaussiano
np.random.seed(42)
img_limpia = data.camera().astype(np.float64)
sigma_ruido = 35  # desviación estándar del ruido

M_vals = [1, 4, 16, 64]

fig, axes = plt.subplots(2, 4, figsize=(20, 8))
fig.suptitle('Reducción de ruido por PROMEDIADO de imágenes\n'
             f'Modelo: g_i = f + η_i,  η_i ~ N(0, σ²={sigma_ruido}²)\n'
             'La potencia del ruido se atenúa por un factor M', fontsize=12)

for i, M in enumerate(M_vals):
    # Generar M imágenes con ruido y promediarlas
    acumulador = np.zeros_like(img_limpia)
    for _ in range(M):
        ruido = np.random.normal(0, sigma_ruido, img_limpia.shape)
        acumulador += img_limpia + ruido
    img_promedio = np.clip(acumulador / M, 0, 255).astype(np.uint8)
    
    # SNR teórica y empírica
    snr_teorica = 20 * np.log10(img_limpia.std() / (sigma_ruido / np.sqrt(M)))
    residual    = img_promedio.astype(float) - img_limpia
    snr_empirica= 20 * np.log10(img_limpia.std() / (residual.std() + 1e-10))
    
    axes[0, i].imshow(img_promedio, cmap='gray', vmin=0, vmax=255)
    axes[0, i].set_title(f'M = {M} imagen(es)\nSNR ≈ {snr_empirica:.1f} dB', fontsize=10)
    axes[0, i].axis('off')
    
    axes[1, i].hist(residual.ravel(), bins=64, color=['#f78166','#ffa657','#3fb950','#58a6ff'][i],
                    density=True, alpha=0.85)
    axes[1, i].set_title(f'σ residual = {residual.std():.1f}', fontsize=9)
    axes[1, i].set_xlabel('Error residual')
    axes[1, i].set_ylabel('Densidad')
    axes[1, i].grid(True)

plt.tight_layout()
plt.show()

print('\n📊 Reducción teórica del ruido:')
print(f'{"M":>6} | {"σ_ruido/√M":>12} | {"Mejora vs M=1":>15}')
print('-' * 40)
for M in M_vals:
    print(f'{M:>6} | {sigma_ruido/np.sqrt(M):>12.2f} | {10*np.log10(M):>15.1f} dB')


# =========================================================================
# CELDA 15
# =========================================================================
# ============================================================
# CELDA T6-4-B — OPERADOR RESTA: detección de cambios y
#                eliminación de fondo
# ============================================================
img_fondo    = data.camera()  # imagen base

# Simular imagen con cambio: añadir un objeto (rectángulo)
img_cambio   = img_fondo.copy()
img_cambio[150:250, 150:250] = np.clip(
    img_cambio[150:250, 150:250].astype(int) + 80, 0, 255).astype(np.uint8)
# Añadir ruido leve para simular escena real
ruido_leve = (np.random.normal(0, 5, img_fondo.shape)).astype(np.int16)
img_cambio = np.clip(img_cambio.astype(np.int16) + ruido_leve, 0, 255).astype(np.uint8)

# Operador resta
diferencia     = cv2.absdiff(img_cambio, img_fondo)   # |g - f|
# Umbralizar para detectar la región cambiada
_, mascara     = cv2.threshold(diferencia, 20, 255, cv2.THRESH_BINARY)

# Visualización
fig, axes = plt.subplots(1, 4, figsize=(20, 5))
fig.suptitle('Operador Resta: detección de cambios entre imágenes', fontsize=12)

for ax, img_i, titulo_i, cmap_i in zip(
        axes,
        [img_fondo, img_cambio, diferencia, mascara],
        ['Imagen fondo (referencia)', 'Imagen con objeto añadido',
         'Diferencia |g - f|\n(amplificada ×4 para visualizar)',
         'Máscara de cambio\n(umbral = 20)'],
        ['gray','gray','hot','gray']):
    display = np.clip(img_i.astype(int)*4, 0, 255).astype(np.uint8) if ax == axes[2] else img_i
    ax.imshow(display, cmap=cmap_i, vmin=0, vmax=255)
    ax.set_title(titulo_i, fontsize=9)
    ax.axis('off')

plt.tight_layout()
plt.show()

print(f'\n📌 Píxeles detectados como cambiados: {(mascara > 0).sum():,}  '
      f'({100*(mascara>0).mean():.1f}% del total)')
