import os
import random
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from sklearn.cluster import KMeans
from imblearn.over_sampling import RandomOverSampler

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)

def segment_image_kmeans(image_pil, n_clusters=3):
   
    img_np = np.array(image_pil)
    h, w, c = img_np.shape
    
    
    pixels = img_np.reshape(-1, c)
    
    # K-Means 
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(pixels)
    centers = kmeans.cluster_centers_
    
    background_cluster_idx = np.argmin(np.mean(centers, axis=1)) 
    
    fused_pixels = pixels.copy()
    fused_pixels[labels == background_cluster_idx] = [0, 0, 0]
    
    segmented_img = fused_pixels.reshape(h, w, c)
    return Image.fromarray(segmented_img.astype(np.uint8))

if __name__ == '__main__':
    set_seed(42)
    
  
    ORIGINAL_CSV = 'split.csv'     
    ORIGINAL_IMG_DIR = './'         
    OUTPUT_DIR = './cleaned_dataset'
    OUTPUT_CSV = 'cleaned_split.csv'
    TARGET_SIZE = (224, 224)        
  
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df = pd.read_csv(ORIGINAL_CSV)
    
    print("K-Means...")
    processed_records = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        img_path = os.path.join(ORIGINAL_IMG_DIR, row['filepath'])
        if not os.path.exists(img_path):
            continue
            

        img = Image.open(img_path).convert('RGB')
        img_resized = img.resize(TARGET_SIZE)
        

        try:
            img_cleaned = segment_image_kmeans(img_resized, n_clusters=3)
        except Exception as e:

            img_cleaned = img_resized
            

        filename = os.path.basename(row['filepath'])
        save_path = os.path.join(OUTPUT_DIR, filename)
        img_cleaned.save(save_path)
        

        processed_records.append({
            'filename': filename,
            'filepath': os.path.join('cleaned_dataset', filename),
            'label_index': int(row['label_index']),
            'split': row['split']
        })
        
    df_processed = pd.DataFrame(processed_records)
    
    df_train = df_processed[df_processed['split'] == 'train'].reset_index(drop=True)
    df_val = df_processed[df_processed['split'] != 'train'].reset_index(drop=True)
    
    # RandomOverSampler 
    ros = RandomOverSampler(random_state=42)
    dummy_x = np.arange(len(df_train)).reshape(-1, 1)
    idx_resampled, _ = ros.fit_resample(dummy_x, df_train['label_index'])
    
    df_train_balanced = df_train.iloc[idx_resampled.flatten()].reset_index(drop=True)
    
    df_final = pd.concat([df_train_balanced, df_val], axis=0).reset_index(drop=True)
    
    # 4. New CSV
    df_final.to_csv(OUTPUT_CSV, index=False)
    print(f"all completed")
    print(f" -> image saved: {OUTPUT_DIR}")
    print(f" -> new csv saved: {OUTPUT_CSV}")