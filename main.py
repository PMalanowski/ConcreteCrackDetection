import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import glob
from collections import defaultdict
import itertools
import random
import time
from scipy.spatial.distance import directed_hausdorff

# ==========================================
#               KONFIGURACJA
# ==========================================

DATASET = 'dataset3'

# --- 1. TRYB PRACY ---
# True: Najpierw szuka najlepszych parametrów na próbce zdjęć.
# False: Używa DEFAULT_PARAMS od razu.
ENABLE_GRID_SEARCH = True 

# --- 2. KONFIGURACJA GRID SEARCH ---
# Sprawdzamy te kombinacje. 
# UWAGA: Definiujemy wartości BAZOWE (dla szerokości 450px). 
# Skrypt sam je przeskaluje dla większych zdjęć.
GRID_SEARCH_PARAMS = {
    'BASE_BLOCK_SIZE': [149, 199, 249, 299], 
    'ADAPTIVE_C': [10, 12, 15, 20, 23],
    'BASE_MIN_AREA': [200, 300],
    'BASE_MORPH_SIZE': [3, 5],
    'MORPH_OPERATION': ['CLOSE', 'OPEN'],
    'BASE_MEDIAN_SIZE': [3, 5]
}

# Na ilu losowych zdjęciach testować każdą kombinację?
GRID_SUBSET_SIZE = 25

# --- 3. PARAMETRY DOMYŚLNE (Backup / Startowe) ---
DEFAULT_PARAMS = {
    'BASE_BLOCK_SIZE': 199,
    'ADAPTIVE_C': 20,
    'BASE_MIN_AREA': 200,
    'BASE_MEDIAN_SIZE': 3,
    'BASE_MORPH_SIZE': 3,
    'MORPH_OPERATION': 'CLOSE',
    'BASE_TOLERANCE': 3
}

# --- 4. OPCJE ---
USE_DENOISING = True
USE_MORPHOLOGY = True
SHOW_PLOTS = False # Ustaw na True tylko jeśli chcesz oglądać zdjęcia jedno po drugim
# ==========================================


def load_image_gray_cv(path):
    return cv2.imread(path, cv2.IMREAD_GRAYSCALE)

def load_mask_cv(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None: return None
    _, mask = cv2.threshold(img, 20, 1, cv2.THRESH_BINARY)
    return mask.astype(np.uint8)

def get_file_group(filename):
    parts = filename.split('_')
    if len(parts) > 1: return parts[0]
    return "Other"

# --- LOGIKA DYNAMICZNEGO SKALOWANIA ---

def get_dynamic_params(image, base_params):
    """
    Przelicza parametry bazowe na parametry rzeczywiste 
    w zależności od rozdzielczości zdjęcia.
    """
    h, w = image.shape[:2]
    # Skala względem szerokości 512px
    scale = w / 512.0
    if scale < 1.0: scale = 1.0 # Nie zmniejszamy poniżej bazy
    
    runtime_params = base_params.copy()
    
    # 1. Block Size (Musi być nieparzysty)
    bs = int(base_params['BASE_BLOCK_SIZE'] * scale)
    if bs % 2 == 0: bs += 1
    runtime_params['ACTUAL_BLOCK_SIZE'] = bs
    
    # 2. Min Area (Rośnie kwadratowo)
    runtime_params['ACTUAL_MIN_AREA'] = int(base_params['BASE_MIN_AREA'] * (scale * scale))
    
    # 3. Kernel Sizes (Mediana, Morfologia)
    ms = int(base_params.get('BASE_MEDIAN_SIZE', 5) * scale)
    if ms % 2 == 0: ms += 1
    if ms < 3: ms = 3
    runtime_params['ACTUAL_MEDIAN_SIZE'] = ms
    
    m_morph = int(base_params.get('BASE_MORPH_SIZE', 3) * scale)
    if m_morph % 2 == 0: m_morph += 1
    if m_morph < 3: m_morph = 3
    runtime_params['ACTUAL_MORPH_SIZE'] = m_morph
    
    # 4. Tolerancja Metryk
    tol = int(base_params.get('BASE_TOLERANCE', 3) * scale)
    runtime_params['ACTUAL_TOLERANCE'] = tol
    
    return runtime_params

# --- PIPELINE PRZETWARZANIA ---

def process_single_image(raw_img, base_params):
    # 1. Oblicz parametry pod to konkretne zdjęcie
    p = get_dynamic_params(raw_img, base_params)
    
    # 2. Odszumianie
    if USE_DENOISING:
        input_img = cv2.medianBlur(raw_img, p['ACTUAL_MEDIAN_SIZE'])
    else:
        input_img = raw_img

    # 3. Segmentacja (Adaptive)
    mask = cv2.adaptiveThreshold(
        input_img, 
        255, 
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 
        p['ACTUAL_BLOCK_SIZE'], 
        base_params['ADAPTIVE_C'] # C się nie skaluje!
    )
    
    # 4. Morfologia
    if USE_MORPHOLOGY:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (p['ACTUAL_MORPH_SIZE'], p['ACTUAL_MORPH_SIZE']))
        op = cv2.MORPH_CLOSE if base_params['MORPH_OPERATION'] == 'CLOSE' else cv2.MORPH_OPEN
        mask_connected = cv2.morphologyEx(mask, op, kernel)
    else:
        mask_connected = mask

    # 5. Filtracja powierzchni
    contours, _ = cv2.findContours(mask_connected, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    final_mask = np.zeros_like(mask_connected)
    
    for cnt in contours:
        if cv2.contourArea(cnt) >= p['ACTUAL_MIN_AREA']:
            cv2.drawContours(final_mask, [cnt], -1, 255, thickness=cv2.FILLED)
            
    return final_mask, p['ACTUAL_TOLERANCE']

# --- METRYKI ---

def calculate_metrics(pred_mask_255, true_mask_01, tolerance):
    p_bool = pred_mask_255 > 127
    t_bool = true_mask_01 > 0
    
    # Puste GT
    if np.sum(t_bool) == 0:
        if np.sum(p_bool) == 0:
            return 1.0, 1.0, 1.0, 1.0, 1.0, 0.0 # Idealnie
        else:
            h, w = pred_mask_255.shape[:2]
            penalty_hd = np.sqrt(h**2 + w**2)
            return 0.0, 0.0, 0.0, 0.0, 0.0, penalty_hd

    p_flat = p_bool.astype(np.int8).flatten()
    t_flat = t_bool.astype(np.int8).flatten()
    
    TP = np.sum((p_flat == 1) & (t_flat == 1))
    FP = np.sum((p_flat == 1) & (t_flat == 0))
    FN = np.sum((p_flat == 0) & (t_flat == 1))
    
    smooth = 1e-6
    iou = TP / (TP + FP + FN + smooth)
    precision = TP / (TP + FP + smooth)
    recall = TP / (TP + FN + smooth)
    f1 = 2 * (precision * recall) / (precision + recall + smooth)
    
    # Relaxed Precision
    kernel_tol = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*tolerance+1, 2*tolerance+1))
    t_dilated = cv2.dilate(t_bool.astype(np.uint8), kernel_tol)
    relaxed_tp = np.logical_and(p_bool, t_dilated).sum()
    pred_count = p_bool.sum()
    relaxed_precision = relaxed_tp / (pred_count + smooth)
    
    # Hausdorff
    if pred_count == 0:
        h, w = pred_mask_255.shape[:2]
        hd = np.sqrt(h**2 + w**2)
    else:
        p_coords = np.argwhere(p_bool)
        t_coords = np.argwhere(t_bool)
        d1 = directed_hausdorff(p_coords, t_coords)[0]
        d2 = directed_hausdorff(t_coords, p_coords)[0]
        hd = max(d1, d2)

    return iou, precision, recall, f1, relaxed_precision, hd

# --- GRID SEARCH ---

def run_grid_search(files, masks_dir):
    print("\n" + "="*80)
    print(f"URUCHAMIANIE GRID SEARCH (Szukanie najlepszych parametrów)")
    print("="*80)
    
    if len(files) > GRID_SUBSET_SIZE:
        subset_files = random.sample(files, GRID_SUBSET_SIZE)
        print(f"Wybrano losową próbkę {GRID_SUBSET_SIZE} zdjęć.")
    else:
        subset_files = files
    
    # Cache images
    loaded_data = []
    for fpath in subset_files:
        fname = os.path.basename(fpath)
        mpath = os.path.join(masks_dir, fname.replace('.jpg', '.png'))
        if not os.path.exists(mpath): mpath = os.path.join(masks_dir, fname)
        if os.path.exists(mpath):
            img = load_image_gray_cv(fpath)
            mask = load_mask_cv(mpath)
            if img is not None and mask is not None:
                loaded_data.append((img, mask))

    # Generowanie kombinacji
    keys, values = zip(*GRID_SEARCH_PARAMS.items())
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    print(f"Liczba kombinacji: {len(combinations)}")
    print(f"{'BLOCK':<6} | {'C':<4} | {'AREA':<6} | {'IoU':<8} | {'R.Prec':<8} | {'MORPH_S':<6} | {'MEDIAN':<6} | {'MORPH_OP':<8}")
    print("-" * 60)
    
    best_score = -1
    best_combo = DEFAULT_PARAMS.copy()
    
    for combo in combinations:
        # Scalanie z defaultami (żeby mieć morph_size itp.)
        current_params = DEFAULT_PARAMS.copy()
        current_params.update(combo)
        
        scores_iou = []
        scores_rp = []
        
        for img, gt_mask in loaded_data:
            final_mask, tol = process_single_image(img, current_params)
            iou, _, _, _, rel_prec, _ = calculate_metrics(final_mask, gt_mask, tol)
            scores_iou.append(iou)
            scores_rp.append(rel_prec)
            
        m_iou = np.mean(scores_iou)
        m_rp = np.mean(scores_rp)
        
        print(f"{combo['BASE_BLOCK_SIZE']:<6} | {combo['ADAPTIVE_C']:<4} | {combo['BASE_MIN_AREA']:<6} | {combo['BASE_MORPH_SIZE']:<6} | {combo['BASE_MEDIAN_SIZE']:<6} | {combo['MORPH_OPERATION']:<8} | {m_iou:.4f}   | {m_rp:.4f}")
        
        if m_iou > best_score:
            best_score = m_iou
            best_combo = current_params

    print("-" * 60)
    print(f"ZWYCIĘSKIE PARAMETRY (IoU: {best_score:.4f}):")
    print(f"Block: {best_combo['BASE_BLOCK_SIZE']}, C: {best_combo['ADAPTIVE_C']}, MinArea: {best_combo['BASE_MIN_AREA']}")
    return best_combo

# --- MAIN ---

def main():
    base_dir = os.getcwd()
    images_dir = os.path.join(base_dir, DATASET, 'images')
    masks_dir = os.path.join(base_dir, DATASET, 'masks')
    files = glob.glob(os.path.join(images_dir, '*.jpg'))
    
    if not files: return print("Brak plików!")

    # 1. Dobór parametrów
    if ENABLE_GRID_SEARCH:
        final_params = run_grid_search(files, masks_dir)
    else:
        final_params = DEFAULT_PARAMS
        print("Używam parametrów domyślnych.")

    # 2. Finalne przetwarzanie
    print("\n" + "="*100)
    print("FINALNA OCENA DLA CAŁEGO ZBIORU")
    print("="*100)
    print(f"{'Grupa':<12} | {'Plik':<20} | {'IoU':<7} | {'Prec':<7} | {'Rec':<7} | {'R.Prec':<7} | {'HD':<7}")
    print("-" * 100)
    
    stats = defaultdict(lambda: defaultdict(list))
    
    for fpath in files:
        fname = os.path.basename(fpath)
        group = get_file_group(fname)
        mpath = os.path.join(masks_dir, fname.replace('.jpg', '.png'))
        if not os.path.exists(mpath): mpath = os.path.join(masks_dir, fname)
        if not os.path.exists(mpath): continue
            
        raw_img = load_image_gray_cv(fpath)
        gt_mask = load_mask_cv(mpath)
        if raw_img is None or gt_mask is None: continue
        
        # Proces
        final_mask, tol = process_single_image(raw_img, final_params)
        
        # Metryki
        iou, prec, rec, f1, rel_prec, hd = calculate_metrics(final_mask, gt_mask, tol)
        
        stats[group]['iou'].append(iou)
        stats[group]['prec'].append(prec)
        stats[group]['rec'].append(rec)
        stats[group]['f1'].append(f1)
        stats[group]['rel_prec'].append(rel_prec)
        stats[group]['hd'].append(hd)
        
        short_name = (fname[:17] + '..') if len(fname)>20 else fname
        print(f"{group:<12} | {short_name:<20} | {iou:.4f}  | {prec:.4f}  | {rec:.4f}  | {rel_prec:.4f}  | {hd:.1f}")

        if SHOW_PLOTS and len(stats[group]['iou']) == 1:
             fig, ax = plt.subplots(1, 3, figsize=(12, 4))
             ax[0].imshow(raw_img, cmap='gray'); ax[0].set_title(f"Orig: {fname}")
             ax[1].imshow(final_mask, cmap='gray', vmin=0, vmax=255); ax[1].set_title(f"Wynik (IoU:{iou:.2f})")
             ax[2].imshow(gt_mask, cmap='gray', vmin=0, vmax=1); ax[2].set_title("Ground Truth")
             for a in ax: a.axis('off')
             plt.tight_layout(); plt.show()

    # Podsumowanie
    print(f"ZWYCIĘSKIE PARAMETRY:")
    print(f"Block: {final_params['BASE_BLOCK_SIZE']}, C: {final_params['ADAPTIVE_C']}, MinArea: {final_params['BASE_MIN_AREA']}, MorphSize: {final_params['BASE_MORPH_SIZE']}, MedianSize: {final_params['BASE_MEDIAN_SIZE']}, MorphOp: {final_params['MORPH_OPERATION']}")
    print("\n" + "=" * 100)
    print(f"{'PODSUMOWANIE WG GRUP':^100}")
    print("=" * 100)
    print(f"{'Grupa':<15} | {'IoU':<8} | {'Prec':<8} | {'Rec':<8} | {'F1':<8} | {'R.Prec':<8} | {'Mean HD':<8} | {'N'}")
    print("-" * 100)
    
    total = defaultdict(list)
    for g, metrics in sorted(stats.items()):
        means = {k: np.mean(v) for k, v in metrics.items()}
        for k, v in metrics.items(): total[k].extend(v)
        
        print(f"{g:<15} | {means['iou']:.4f}   | {means['prec']:.4f}   | {means['rec']:.4f}   | {means['f1']:.4f}   | {means['rel_prec']:.4f}   | {means['hd']:.2f}     | {len(metrics['iou'])}")
        
    print("-" * 100)
    if total['iou']:
        print(f"GLOBALNE IoU:      {np.mean(total['iou']):.4f}")
        print(f"GLOBALNE R.Prec:   {np.mean(total['rel_prec']):.4f}")
        print(f"GLOBALNE HD:       {np.mean(total['hd']):.2f}")

if __name__ == "__main__":
    main()