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
print('\n' + '='*60)
print('INICIANDO BLOQUE III: PROCESAMIENTO DE IMAGEN SELECCIONADA')
print('='*60)

# 1. Configuración de carga
nombre_imagen = './images/1010.png' # Puedes cambiar esto por '1239.png', '1321.png', etc.
procesar_en_color = False   # <--- ¡CAMBIA A False PARA BLANCO Y NEGRO!

ruta_imagen = os.path.join(os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else '.', nombre_imagen)

try:
    if procesar_en_color:
        img_raw = cv2.imread(ruta_imagen, cv2.IMREAD_COLOR)
        if img_raw is None:
            img_raw = cv2.imread(nombre_imagen, cv2.IMREAD_COLOR)
    else:
        img_raw = cv2.imread(ruta_imagen, cv2.IMREAD_GRAYSCALE)
        if img_raw is None:
            img_raw = cv2.imread(nombre_imagen, cv2.IMREAD_GRAYSCALE)
            
    if img_raw is None:
        print(f'❌ No se pudo cargar la imagen {nombre_imagen}. Comprueba la ruta.')
    else:
        print(f'✅ Imagen {nombre_imagen} cargada correctamente. Resolucion: {img_raw.shape[:2]} | Modo: {"Color" if procesar_en_color else "Blanco y Negro"}')
        
        # 2. Aplicar técnicas de mejora
        gamma_val = 0.5
        
        if procesar_en_color:
            # Convertir de BGR a RGB y luego a HSV
            img_rgb = cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB)
            from skimage.color import rgb2hsv, hsv2rgb
            img_hsv = rgb2hsv(img_rgb)
            
            # El canal V (brillo) es el índice 2 (rango 0.0 a 1.0)
            v_channel = img_hsv[:,:,2]
            
            # Original para visualización
            img_real_vis = img_rgb
            
            # Técnica A: Gamma
            v_gamma = np.power(v_channel, gamma_val)
            img_hsv_gamma = img_hsv.copy()
            img_hsv_gamma[:,:,2] = v_gamma
            img_gamma_vis = img_as_ubyte(hsv2rgb(img_hsv_gamma))
            
            # Técnica B: Ecualización Global
            v_eq = exposure.equalize_hist(v_channel)
            img_hsv_eq = img_hsv.copy()
            img_hsv_eq[:,:,2] = v_eq
            img_eq_global_vis = img_as_ubyte(hsv2rgb(img_hsv_eq))
            
            # Técnica C: CLAHE
            v_clahe = exposure.equalize_adapthist(v_channel, clip_limit=0.03)
            img_hsv_clahe = img_hsv.copy()
            img_hsv_clahe[:,:,2] = v_clahe
            img_clahe_vis = img_as_ubyte(hsv2rgb(img_hsv_clahe))
            
            # Para métricas usaremos el canal V como referencia equivalente a blanco y negro
            img_real_metricas = img_as_ubyte(v_channel)
            img_gamma_metricas = img_as_ubyte(v_gamma)
            img_eq_global_metricas = img_as_ubyte(v_eq)
            img_clahe_metricas = img_as_ubyte(v_clahe)
            
        else:
            img_real = img_raw
            img_real_vis = img_real
            
            # Técnica A: Gamma
            img_norm = img_real.astype(np.float64) / 255.0
            img_gamma = np.power(img_norm, gamma_val)
            img_gamma_vis = np.clip(img_gamma * 255, 0, 255).astype(np.uint8)
            
            # Técnica B: Ecualización Global
            img_eq_global = exposure.equalize_hist(img_real)
            img_eq_global_vis = img_as_ubyte(img_eq_global)

            # Técnica C: CLAHE
            img_clahe = exposure.equalize_adapthist(img_real, clip_limit=0.03)
            img_clahe_vis = img_as_ubyte(img_clahe)
            
            # Para métricas
            img_real_metricas = img_real
            img_gamma_metricas = img_gamma_vis
            img_eq_global_metricas = img_eq_global_vis
            img_clahe_metricas = img_clahe_vis

        # 3. Mostrar resultados visuales
        imagenes_act = [img_real_vis, img_gamma_vis, img_eq_global_vis, img_clahe_vis]
        titulos_act = [
            f'Original ({nombre_imagen})',
            f'Gamma (γ={gamma_val})',
            'Ecualización Global',
            'CLAHE'
        ]
        
        # Si es a color, no forzamos cmap='gray'
        cmap_list = [None, None, None, None] if procesar_en_color else ['gray', 'gray', 'gray', 'gray']
        mostrar_imagen_histograma(imagenes_act, titulos_act, figsize=(20, 8), cmap_list=cmap_list)

        # 4. Calcular métricas
        print('\n📊 Comparación de métricas de calidad:')
        if procesar_en_color:
            print('   (Calculadas sobre el canal de luminosidad V del espacio HSV)')
        print(f'{"Técnica":<25} {"MSE":>10} {"PSNR (dB)":>12} {"Contraste (σ)":>15} {"Entropía":>12}')
        print('-' * 80)
        
        tecnicas = ['Corrección Gamma', 'Ecualización Global', 'CLAHE']
        imagenes_proc = [img_gamma_metricas, img_eq_global_metricas, img_clahe_metricas]
        
        ent_orig = calcular_metricas(img_real_metricas, img_real_metricas)['Entropía orig']
        std_orig = img_real_metricas.std()
        print(f'{"Original":<25} {"-":>10} {"-":>12} {std_orig:>15.2f} {ent_orig:>12.4f}')
        
        for nombre, img_p in zip(tecnicas, imagenes_proc):
            mets = calcular_metricas(img_real_metricas, img_p)
            print(f'{nombre:<25} {mets["MSE"]:>10.4f} {mets["PSNR (dB)"]:>12.2f} {mets["Contraste STD proc"]:>15.4f} {mets["Entropía proc"]:>12.4f}')
        
        print('\n🚀 Proceso completado. Cambia procesar_en_color para probar más opciones.')

except Exception as e:
    print(f"Error procesando la imagen: {e}")
