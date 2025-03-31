import cv2
import numpy as np
import os

def generate_visualizations(input_path, output_folder, base_filename):
    """Genera la imagen original"""
    # Cargar imagen
    img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("No se pudo cargar la imagen")
    
    # Normalizar
    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    
    # 1. Imagen original
    original_path = os.path.join(output_folder, f"{base_filename}_original.png")
    cv2.imwrite(original_path, img)
    
    return {
        'original': f"results/{base_filename}_original.png"
    }