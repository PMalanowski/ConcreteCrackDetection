import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import glob
from collections import defaultdict

# ==========================================
#               KONFIGURACJA
# ==========================================

# 1. ŚCIEŻKI
DATASET = 'dataset'  # Nazwa folderu ze zbiorami danych

# 2. METODA
# 'OTSU'     - Globalne progowanie (dobre dla równomiernego oświetlenia)
# 'ADAPTIVE' - Lokalne progowanie (lepsze przy cieniach/nierównym świetle)
ALGORITHM_TYPE = 'ADAPTIVE' 

# 3. ODSZUMIANIE WSTĘPNE (Filtr Medianowy)
USE_DENOISING = True       
MEDIAN_KERNEL_SIZE = 3     # Musi być liczbą nieparzystą w OpenCV (np. 3, 5, 7)

# 4. PARAMETRY ADAPTACYJNE (Tylko dla ALGORITHM_TYPE = 'ADAPTIVE')
ADAPTIVE_BLOCK_SIZE = 75   # Rozmiar sąsiedztwa (musi być nieparzysty!)
ADAPTIVE_C = 10            # Stała odejmowana od średniej. 
                           # Odpowiada "Sensitivity". Im wyższa, tym mniej szumu (tylko czarne pęknięcia).
                           # Typowe wartości: 5 - 20.

# 5. POST-PROCESSING (Morfologia)
USE_MORPHOLOGY = True      
MORPH_KERNEL_SIZE = 3      # Rozmiar elementu (3, 5...)
MORPH_OPERATION = 'OPEN'   # 'OPEN' (usuwa kropki) lub 'CLOSE' (łączy pęknięcia)

# 6. WIZUALIZACJA
SHOW_PLOTS = True          
# ==========================================


def load_image_gray_cv(path):
    """Wczytuje obraz w skali szarości używając OpenCV."""
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    return img # Zwraca uint8 0-255

def load_mask_cv(path):
    """Wczytuje maskę i binaryzuje ją."""
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None: return None
    # Binaryzacja: wszystko powyżej 127 staje się 1, reszta 0
    _, mask = cv2.threshold(img, 20, 1, cv2.THRESH_BINARY)
    return mask.astype(np.int8)

# --- ALGORYTMY (OPENCV WRAPPERS) ---

def apply_median_filter(image, size=5):
    """Filtr medianowy z OpenCV."""
    # Size musi być nieparzysty
    if size % 2 == 0: size += 1
    return cv2.medianBlur(image, size)

def apply_threshold_otsu(image):
    """
    Globalne progowanie Otsu.
    Używamy THRESH_BINARY_INV, ponieważ pęknięcia są ciemne (czarne), 
    a chcemy, żeby na masce były białe (1/255).
    """
    # Funkcja zwraca (wartość_progu, obraz_binarny)
    _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return binary

def apply_threshold_adaptive(image, block_size=75, C=10):
    """
    Adaptacyjne progowanie Gaussian C.
    Szuka lokalnie ciemniejszych obszarów.
    """
    if block_size % 2 == 0: block_size += 1 # Musi być nieparzysty
    
    binary = cv2.adaptiveThreshold(
        image, 
        255,                        # Wartość maksymalna
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV,      # Inwersja (szukamy ciemnego na jasnym)
        block_size, 
        C
    )
    return binary

def apply_morphological_cleanup(binary_image, operation='OPEN', size=3):
    """Operacje morfologiczne OpenCV."""
    kernel = np.ones((size,size),np.uint8)
    
    if operation == 'OPEN':
        # Erozja -> Dylatacja (Usuwanie szumu)
        return cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel)
    elif operation == 'CLOSE':
        # Dylatacja -> Erozja (Zamykanie dziur)
        return cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel)
    else:
        return binary_image

# --- POMOCNICZE: GRUPOWANIE ---

def get_file_group(filename):
    """
    Określa grupę na podstawie nazwy pliku.
    - Jeśli ma podkreślnik -> 'PREFIKS'
    - Inaczej -> 'Other'
    """
    parts = filename.split('_')
    if len(parts) > 1:
        return parts[0]
    
    return "Other"

# --- METRYKI ---

def calculate_metrics(pred, true):
    """
    Oblicza IoU, Precision, Recall, F1-Score oraz Accuracy.
    """
    # Normalizacja predykcji OpenCV (0-255) do (0-1)
    p = (pred > 127).astype(np.int8).flatten()
    t = true.flatten()
    
    # Podstawowe składniki macierzy pomyłek
    TP = np.sum((p == 1) & (t == 1)) # True Positive
    FP = np.sum((p == 1) & (t == 0)) # False Positive
    FN = np.sum((p == 0) & (t == 1)) # False Negative
    
    smooth = 1e-6 # Zabezpieczenie przed dzieleniem przez zero
    
    # 1. IoU (Jaccard Index)
    iou = TP / (TP + FP + FN + smooth)
    
    # 2. Precision
    precision = TP / (TP + FP + smooth)
    
    # 3. Recall
    recall = TP / (TP + FN + smooth)
    
    # 4. F1-Score (Średnia harmoniczna Precision i Recall)
    f1 = 2 * (precision * recall) / (precision + recall + smooth)
    
    return iou, precision, recall, f1

# --- MAIN ---

def main():
    base_dir = os.getcwd()
    images_dir = os.path.join(base_dir, DATASET, 'images')
    masks_dir = os.path.join(base_dir, DATASET, 'masks')
    files = glob.glob(os.path.join(images_dir, '*.jpg'))
    
    if not files: return print(f"Brak plików jpg w {images_dir}")

    # Słownik na statystyki
    dataset_stats = defaultdict(lambda: {'iou': [], 'prec': [], 'rec': [], 'f1': []})
    
    print(f"Metoda: {ALGORITHM_TYPE} | Denoise: {USE_DENOISING} | Morph: {USE_MORPHOLOGY}")
    print("-" * 100)
    # Szersza nagłówka tabeli
    #print(f"{'Grupa':<12} | {'Plik':<20} | {'IoU':<7} | {'Prec':<7} | {'Rec':<7} | {'F1':<7} | {'Acc':<7}")
    #print("-" * 100)

    for fpath in files:
        fname = os.path.basename(fpath)
        group_name = get_file_group(fname)
        
        # Szukanie maski
        mpath = os.path.join(masks_dir, fname.replace('.jpg', '.png'))
        if not os.path.exists(mpath): mpath = os.path.join(masks_dir, fname)
        if not os.path.exists(mpath): continue
            
        # 1. ŁADOWANIE
        raw_img = load_image_gray_cv(fpath)
        gt_mask = load_mask_cv(mpath)
        if raw_img is None or gt_mask is None: continue
        
        # 2. ODSZUMIANIE
        if USE_DENOISING:
            input_img = apply_median_filter(raw_img, size=MEDIAN_KERNEL_SIZE)
        else:
            input_img = raw_img

        # 3. SEGMENTACJA
        if ALGORITHM_TYPE == 'OTSU':
            mask = apply_threshold_otsu(input_img)
        elif ALGORITHM_TYPE == 'ADAPTIVE':
            mask = apply_threshold_adaptive(input_img, block_size=ADAPTIVE_BLOCK_SIZE, C=ADAPTIVE_C)
        
        # 4. POST-PROCESSING
        if USE_MORPHOLOGY:
            final_mask = apply_morphological_cleanup(mask, operation=MORPH_OPERATION, size=MORPH_KERNEL_SIZE)
        else:
            final_mask = mask

        # 5. METRYKI
        iou, prec, rec, f1 = calculate_metrics(final_mask, gt_mask)
        
        # Zapis do statystyk
        dataset_stats[group_name]['iou'].append(iou)
        dataset_stats[group_name]['prec'].append(prec)
        dataset_stats[group_name]['rec'].append(rec)
        dataset_stats[group_name]['f1'].append(f1)
        
        # Wyświetlanie (skracam nazwę pliku, żeby tabelka się nie rozjeżdżała)
        short_fname = (fname[:17] + '..') if len(fname) > 20 else fname
        #print(f"{group_name:<12} | {short_fname:<20} | {iou:.4f}  | {prec:.4f}  | {rec:.4f}  | {f1:.4f}")

        # 6. WIZUALIZACJA
        if fpath == files[0] and SHOW_PLOTS:
            fig, ax = plt.subplots(1, 5, figsize=(16, 5))
            ax[0].imshow(raw_img, cmap='gray'); ax[0].set_title("Oryginał")
            ax[1].imshow(input_img, cmap='gray'); ax[1].set_title("Smoothened")
            ax[2].imshow(final_mask, cmap='gray', vmin=0, vmax=255); ax[2].set_title("Output Mask")
            ax[3].imshow(gt_mask, cmap='gray', vmin=0, vmax=1); ax[3].set_title("Ground Truth")
            ax[4].imshow(mask, cmap='gray', vmin=0, vmax=255); ax[4].set_title("Initial Mask")
            for a in ax: a.axis('off')
            plt.tight_layout(); plt.show()

    # --- PODSUMOWANIE ---
    print("\n" + "=" * 85)
    print(f"{'PODSUMOWANIE WG GRUP':^85}")
    print("=" * 85)
    # Nagłówek podsumowania
    print(f"{'Grupa':<15} | {'IoU':<8} | {'Prec':<8} | {'Rec':<8} | {'F1':<8} | {'Ilość'}")
    print("-" * 85)
    
    total_iou = []
    total_prec = []
    total_rec = []
    total_f1 = []
    
    for group, stats in sorted(dataset_stats.items()):
        m_iou = np.mean(stats['iou'])
        m_prec = np.mean(stats['prec'])
        m_rec = np.mean(stats['rec'])
        m_f1 = np.mean(stats['f1'])
        count = len(stats['iou'])
        
        total_iou.extend(stats['iou'])
        total_prec.extend(stats['prec'])
        total_rec.extend(stats['rec'])
        total_f1.extend(stats['f1'])
        
        print(f"{group:<15} | {m_iou:.4f}   | {m_prec:.4f}   | {m_rec:.4f}   | {m_f1:.4f}  | {count}")

    print("-" * 85)
    if total_iou:
        print(f"\n{'STATYSTYKI GLOBALNE (WSZYSTKIE ZDJĘCIA)':^50}")
        print("-" * 50)
        print(f"Mean IoU:       {np.mean(total_iou):.4f}")
        print(f"Mean Precision: {np.mean(total_prec):.4f}")
        print(f"Mean Recall:    {np.mean(total_rec):.4f}")
        print(f"Mean F1-Score:  {np.mean(total_f1):.4f}")
        print("-" * 50)

if __name__ == "__main__":
    main()