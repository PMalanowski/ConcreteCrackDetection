import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import glob
from collections import defaultdict
from scipy.spatial.distance import directed_hausdorff

# ==========================================
#        KONFIGURACJA
# ==========================================

DATASET = 'dataset_small'

# --- 1. ODSZUMIANIE WSTĘPNE ---
USE_DENOISING = True       
MEDIAN_KERNEL_SIZE = 7

# --- 2. SEGMENTACJA (ADAPTIVE THRESHOLD) ---
ADAPTIVE_BLOCK_SIZE = 125
ADAPTIVE_C = 7

# --- 3. POST-PROCESSING (MORFOLOGIA) ---
USE_MORPHOLOGY = True      
MORPH_KERNEL_SIZE = 3      
MORPH_OPERATION = 'CLOSE'

# --- 4. FILTRACJA PO POWIERZCHNI ---
REMOVE_SMALL_OBJECTS = True
MIN_OBJECT_AREA = 200

# --- 5. METRYKI ---
METRICS_TOLERANCE = 3

# WIZUALIZACJA
SHOW_PLOTS = False
# ==========================================


def load_image_gray_cv(path):
    return cv2.imread(path, cv2.IMREAD_GRAYSCALE)

def load_mask_cv(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None: return None
    _, mask = cv2.threshold(img, 20, 1, cv2.THRESH_BINARY)
    return mask.astype(np.uint8)

# --- ALGORYTMY ---

def apply_median_filter(image, size=5):
    if size % 2 == 0: size += 1
    return cv2.medianBlur(image, size)

def apply_threshold_adaptive(image, block_size=55, C=10):
    if block_size % 2 == 0: block_size += 1
    binary = cv2.adaptiveThreshold(
        image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, block_size, C
    )
    return binary

def apply_morphological_cleanup(binary_image, operation='CLOSE', size=3):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))
    if operation == 'OPEN':
        return cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel)
    elif operation == 'CLOSE':
        return cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel)
    else:
        return binary_image

def remove_small_objects(binary_mask, min_area):
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cleaned_mask = np.zeros_like(binary_mask)
    found_something_big = False
    for cnt in contours:
        if cv2.contourArea(cnt) >= min_area:
            cv2.drawContours(cleaned_mask, [cnt], -1, 255, thickness=cv2.FILLED)
            found_something_big = True
    return cleaned_mask, found_something_big

def get_file_group(filename):
    parts = filename.split('_')
    if len(parts) > 1: return parts[0]
    return "Other"

# --- METRYKI (ZAAWANSOWANE) ---

def calculate_advanced_metrics(pred_mask_255, true_mask_01, tolerance=3):
    p_bool = pred_mask_255 > 127
    t_bool = true_mask_01 > 0
    
    # Obsługa pustego Ground Truth
    if np.sum(t_bool) == 0:
        if np.sum(p_bool) == 0:
            return 1.0, 1.0, 1.0, 1.0, 1.0, 0.0 # Idealnie
        else:
            h, w = pred_mask_255.shape[:2]
            penalty_hd = np.sqrt(h**2 + w**2)
            return 0.0, 0.0, 0.0, 0.0, 0.0, penalty_hd # False Positive

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
    
    # Hausdorff Distance
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

# --- MAIN ---

def main():
    base_dir = os.getcwd()
    images_dir = os.path.join(base_dir, DATASET, 'images')
    masks_dir = os.path.join(base_dir, DATASET, 'masks')
    files = glob.glob(os.path.join(images_dir, '*.jpg'))
    
    if not files: return print(f"Brak plików jpg w {images_dir}")

    dataset_stats = defaultdict(lambda: {'iou': [], 'prec': [], 'rec': [], 'f1': [], 'rel_prec': [], 'hd': []})
    
    print(f"METODA: ADAPTIVE CLASSIC | C: {ADAPTIVE_C} | Block: {ADAPTIVE_BLOCK_SIZE}")
    print("-" * 105)
    print(f"{'Grupa':<12} | {'Plik':<20} | {'IoU':<7} | {'Prec':<7} | {'Rec':<7} | {'F1':<7} | {'R.Prec':<7} | {'HD':<7}")
    print("-" * 105)

    for fpath in files:
        fname = os.path.basename(fpath)
        group_name = get_file_group(fname)
        
        mpath = os.path.join(masks_dir, fname.replace('.jpg', '.png'))
        if not os.path.exists(mpath): mpath = os.path.join(masks_dir, fname)
        if not os.path.exists(mpath): continue
            
        raw_img = load_image_gray_cv(fpath)
        gt_mask = load_mask_cv(mpath)
        if raw_img is None or gt_mask is None: continue
        
        # 1. Odszumianie (Mediana)
        if USE_DENOISING:
            input_img = apply_median_filter(raw_img, size=MEDIAN_KERNEL_SIZE)
        else:
            input_img = raw_img

        # 2. Segmentacja (Adaptive Threshold)
        mask = apply_threshold_adaptive(input_img, block_size=ADAPTIVE_BLOCK_SIZE, C=ADAPTIVE_C)
        
        # 3. Morfologia (Łączenie)
        if USE_MORPHOLOGY:
            mask_connected = apply_morphological_cleanup(mask, operation=MORPH_OPERATION, size=MORPH_KERNEL_SIZE)
        else:
            mask_connected = mask

        # 4. Filtracja powierzchni (Usuwanie szumu)
        if REMOVE_SMALL_OBJECTS:
            final_mask, _ = remove_small_objects(mask_connected, min_area=MIN_OBJECT_AREA)
        else:
            final_mask = mask_connected

        # 5. Metryki
        iou, prec, rec, f1, rel_prec, hd = calculate_advanced_metrics(final_mask, gt_mask, tolerance=METRICS_TOLERANCE)
        
        dataset_stats[group_name]['iou'].append(iou)
        dataset_stats[group_name]['prec'].append(prec)
        dataset_stats[group_name]['rec'].append(rec)
        dataset_stats[group_name]['f1'].append(f1)
        dataset_stats[group_name]['rel_prec'].append(rel_prec)
        dataset_stats[group_name]['hd'].append(hd)
        
        short_fname = (fname[:17] + '..') if len(fname) > 20 else fname
        print(f"{group_name:<12} | {short_fname:<20} | {iou:.4f}  | {prec:.4f}  | {rec:.4f}  | {f1:.4f}  | {rel_prec:.4f}  | {hd:.1f}")

        if SHOW_PLOTS and iou < 0.2:
             fig, ax = plt.subplots(1, 3, figsize=(12, 4))
             ax[0].imshow(raw_img, cmap='gray'); ax[0].set_title(f"Oryginał: {fname}")
             ax[1].imshow(final_mask, cmap='gray', vmin=0, vmax=255); ax[1].set_title(f"Wynik (IoU:{iou:.2f})")
             ax[2].imshow(gt_mask, cmap='gray', vmin=0, vmax=1); ax[2].set_title("Ground Truth")
             for a in ax: a.axis('off')
             plt.tight_layout(); plt.show()

    # --- PODSUMOWANIE ---
    print("\n" + "=" * 105)
    print(f"{'PODSUMOWANIE WG GRUP':^105}")
    print("=" * 105)
    print(f"{'Grupa':<15} | {'IoU':<8} | {'Prec':<8} | {'Rec':<8} | {'F1':<8} | {'R.Prec':<8} | {'Mean HD':<8} | {'Ilość'}")
    print("-" * 105)
    
    total_stats = defaultdict(list)
    for group, stats in sorted(dataset_stats.items()):
        means = {k: np.mean(v) for k, v in stats.items()}
        for k, v in stats.items():
            total_stats[k].extend(v)
        print(f"{group:<15} | {means['iou']:.4f}   | {means['prec']:.4f}   | {means['rec']:.4f}   | {means['f1']:.4f}   | {means['rel_prec']:.4f}   | {means['hd']:.2f}     | {len(stats['iou'])}")

    print("-" * 105)
    if total_stats['iou']:
        print(f"\n{'STATYSTYKI GLOBALNE':^60}")
        print("-" * 60)
        print(f"Mean IoU:            {np.mean(total_stats['iou']):.4f}")
        print(f"Mean Relaxed Prec:   {np.mean(total_stats['rel_prec']):.4f}")
        print(f"Mean HD:             {np.mean(total_stats['hd']):.2f} px")
        print("-" * 60)

if __name__ == "__main__":
    main()