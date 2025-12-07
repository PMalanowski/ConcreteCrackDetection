import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import glob
from numpy.lib.stride_tricks import sliding_window_view

# ==========================================
#               KONFIGURACJA
# ==========================================

# 1. METODA
# 'OTSU' lub 'ADAPTIVE'
ALGORITHM_TYPE = 'OTSU' 

# 2. ODSZUMIANIE WSTĘPNE (Filtr Medianowy)
USE_DENOISING = True       
MEDIAN_KERNEL_SIZE = 7    

# 3. PARAMETRY ADAPTACYJNE
ADAPTIVE_WINDOW_SIZE = 75  
ADAPTIVE_SENSITIVITY = 0.10 

# 4. POST-PROCESSING (Morfologia) - NOWOŚĆ!
USE_MORPHOLOGY = True      # Czy czyścić maskę wynikową?
MORPH_KERNEL_SIZE = 3     # Rozmiar elementu strukturalnego (3 jest zazwyczaj idealne do czyszczenia szumu)
MORPH_OPERATION = 'OPEN'   # 'OPEN' (Usuwa małe kropki) lub 'CLOSE' (Łączy przerwane pęknięcia)
                           # Zazwyczaj 'OPEN' daje lepszą precyzję (Precision), a 'CLOSE' lepszy Recall.

# 5. WIZUALIZACJA
SHOW_PLOTS = False          
# ==========================================


def load_image_gray(path):
    img = Image.open(path).convert('RGB')
    arr = np.array(img, dtype=np.float32)
    gray = 0.299 * arr[:,:,0] + 0.587 * arr[:,:,1] + 0.114 * arr[:,:,2]
    return gray / 255.0

def load_mask(path):
    img = Image.open(path).convert('L')
    arr = np.array(img, dtype=np.float32)
    return (arr > 20).astype(np.int8)

# --- ODSZUMIANIE ---
def apply_median_filter(image, size=3):
    pad = size // 2
    padded = np.pad(image, pad, mode='reflect')
    windows = sliding_window_view(padded, window_shape=(size, size))
    return np.median(windows, axis=(2, 3))

# --- OPTYMALIZACJA (OBRAZ INTEGRALNY) ---
def compute_integral_image(image):
    integral = np.cumsum(np.cumsum(image, axis=0), axis=1)
    pad_integral = np.pad(integral, ((1,0), (1,0)), mode='constant')
    return pad_integral

def get_local_mean_vectorized(integral_img, h, w, window_size):
    half = window_size // 2
    r, c = np.indices((h, w))
    r0 = np.maximum(r - half, 0)
    r1 = np.minimum(r + half + 1, h)
    c0 = np.maximum(c - half, 0)
    c1 = np.minimum(c + half + 1, w)
    region_sum = (integral_img[r1, c1] - integral_img[r0, c1] - integral_img[r1, c0] + integral_img[r0, c0])
    region_area = (r1 - r0) * (c1 - c0)
    region_area[region_area == 0] = 1
    return region_sum / region_area

# --- OPERACJE MORFOLOGICZNE (NOWE FUNKCJE) ---

def morphology_erode(binary_image, kernel_size=3):
    """
    Erozja: Operacja Minimum. Zmniejsza obiekty, usuwa pojedyncze piksele.
    """
    pad = kernel_size // 2
    # Padding wartością 1 (True), aby brzegi nie stały się czarne (0) przy operacji min()
    padded = np.pad(binary_image, pad, mode='constant', constant_values=1)
    
    windows = sliding_window_view(padded, window_shape=(kernel_size, kernel_size))
    # Jeśli wszystkie w oknie to 1 -> wynik 1. Jeśli choć jedno 0 -> wynik 0.
    return np.min(windows, axis=(2, 3))

def morphology_dilate(binary_image, kernel_size=3):
    """
    Dylatacja: Operacja Maximum. Powiększa obiekty, wypełnia dziury.
    """
    pad = kernel_size // 2
    # Padding wartością 0 (False), aby brzegi nie stały się białe (1) przy operacji max()
    padded = np.pad(binary_image, pad, mode='constant', constant_values=0)
    
    windows = sliding_window_view(padded, window_shape=(kernel_size, kernel_size))
    # Jeśli choć jedno w oknie to 1 -> wynik 1.
    return np.max(windows, axis=(2, 3))

def apply_morphological_cleanup(binary_mask, operation='OPEN', size=3):
    """Wykonuje złożoną operację morfologiczną."""
    if operation == 'OPEN':
        # Otwarcie = Erozja -> Dylatacja (Usuwa szum na zewnątrz obiektów)
        eroded = morphology_erode(binary_mask, size)
        opened = morphology_dilate(eroded, size)
        return opened
    elif operation == 'CLOSE':
        # Zamknięcie = Dylatacja -> Erozja (Wypełnia dziury wewnątrz obiektów)
        dilated = morphology_dilate(binary_mask, size)
        closed = morphology_erode(dilated, size)
        return closed
    else:
        return binary_mask

# --- PROGOWANIE ---
def threshold_otsu(image):
    pixels = (image * 255).flatten().astype(np.uint8)
    hist, _ = np.histogram(pixels, bins=256, range=(0, 256))
    total = pixels.size
    current_max, threshold = 0, 0
    sum_total = np.dot(np.arange(256), hist)
    w_b, sum_b = 0, 0
    for t in range(256):
        w_b += hist[t]
        if w_b == 0: continue
        w_f = total - w_b
        if w_f == 0: break
        sum_b += t * hist[t]
        m_b = sum_b / w_b
        m_f = (sum_total - sum_b) / w_f
        var = w_b * w_f * (m_b - m_f)**2
        if var > current_max:
            current_max = var
            threshold = t
    return image < (threshold / 255.0)

def threshold_adaptive(image, window_size=45, sensitivity=0.05):
    h, w = image.shape
    int_img = compute_integral_image(image)
    local_mean = get_local_mean_vectorized(int_img, h, w, window_size)
    return image < (local_mean - sensitivity)

# --- METRYKI ---
def calculate_metrics(pred, true):
    p = pred.flatten()
    t = true.flatten()
    TP = np.sum((p==1) & (t==1))
    FP = np.sum((p==1) & (t==0))
    FN = np.sum((p==0) & (t==1))
    smooth = 1e-6
    iou = TP / (TP + FP + FN + smooth)
    precision = TP / (TP + FP + smooth)
    recall = TP / (TP + FN + smooth)
    return iou, precision, recall

# --- MAIN ---
def main():
    base_dir = os.getcwd()
    images_dir = os.path.join(base_dir, 'dataset2', 'cracks')
    masks_dir = os.path.join(base_dir, 'dataset2', 'labels')
    files = glob.glob(os.path.join(images_dir, '*.jpg'))
    
    if not files: return print("Brak plików jpg!")

    scores = []
    
    print(f"Metoda: {ALGORITHM_TYPE}, Odszumianie: {USE_DENOISING}, Morfologia: {USE_MORPHOLOGY} ({MORPH_OPERATION})")
    print("-" * 65)
    print(f"{'Plik':<25} | {'IoU':<8} | {'Prec':<8} | {'Rec':<8}")
    print("-" * 65)

    for fpath in files:
        fname = os.path.basename(fpath)
        mpath = os.path.join(masks_dir, fname.replace('.jpg', '.png'))
        if not os.path.exists(mpath): mpath = os.path.join(masks_dir, fname)
        if not os.path.exists(mpath): continue
            
        # 1. ŁADOWANIE
        raw_img = load_image_gray(fpath)
        gt_mask = load_mask(mpath)
        
        # 2. ODSZUMIANIE
        if USE_DENOISING:
            input_img = apply_median_filter(raw_img, size=MEDIAN_KERNEL_SIZE)
        else:
            input_img = raw_img

        # 3. SEGMENTACJA
        if ALGORITHM_TYPE == 'OTSU':
            mask = threshold_otsu(input_img)
        elif ALGORITHM_TYPE == 'ADAPTIVE':
            mask = threshold_adaptive(input_img, ADAPTIVE_WINDOW_SIZE, ADAPTIVE_SENSITIVITY)
        
        mask = mask.astype(np.int8) # 0 i 1

        # 4. POST-PROCESSING (MORFOLOGIA)
        if USE_MORPHOLOGY:
            # Tworzymy kopię przed morfologią do wizualizacji (opcjonalnie)
            mask_before_morph = mask.copy()
            final_mask = apply_morphological_cleanup(mask, operation=MORPH_OPERATION, size=MORPH_KERNEL_SIZE)
        else:
            final_mask = mask

        # 5. METRYKI
        iou, prec, rec = calculate_metrics(final_mask, gt_mask)
        scores.append(iou)
        print(f"{fname:<25} | {iou:.4f}   | {prec:.4f}   | {rec:.4f}")

        # 6. WIZUALIZACJA
        if SHOW_PLOTS:
            fig, ax = plt.subplots(1, 4, figsize=(16, 5)) # Dodano 4 panel
            
            ax[0].imshow(raw_img, cmap='gray')
            ax[0].set_title("Oryginał")
            
            # Pokazujemy maskę surową (przed morfologią) jeśli morfologia jest włączona
            if USE_MORPHOLOGY:
                ax[1].imshow(mask_before_morph, cmap='gray', vmin=0, vmax=1)
                ax[1].set_title("Przed Morfologią")
            else:
                ax[1].imshow(input_img, cmap='gray')
                ax[1].set_title("Wejście (Odszumione)")

            ax[2].imshow(final_mask, cmap='gray', vmin=0, vmax=1)
            title_suffix = f"+ {MORPH_OPERATION}" if USE_MORPHOLOGY else ""
            ax[2].set_title(f"Wynik ({ALGORITHM_TYPE} {title_suffix})\nIoU: {iou:.3f}")
            
            ax[3].imshow(gt_mask, cmap='gray', vmin=0, vmax=1)
            ax[3].set_title("Ground Truth")
            
            for a in ax: a.axis('off')
            plt.tight_layout()
            plt.show()

    print("-" * 65)
    print(f"Średnie IoU: {np.mean(scores):.4f}")

if __name__ == "__main__":
    main()