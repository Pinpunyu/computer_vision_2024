import cv2
import time
import matplotlib.pyplot as plt
import pandas as pd

# Load images for analysis
image_paths = ['./data/hill1.JPG', './data/hill2.JPG', './data/S1.jpg', './data/S2.jpg', './data/TV1.jpg', './data/TV2.jpg']
images = [cv2.imread(img_path, cv2.IMREAD_GRAYSCALE) for img_path in image_paths]
image_names = [path.split('/')[-1].split('.')[0] for path in image_paths]  # Extract filename without extension for x-axis labels

# Feature detectors to compare
detectors = {
    'SIFT': cv2.SIFT_create(),
    'ORB': cv2.ORB_create(),
    'BRISK': cv2.BRISK_create(),
    'AKAZE': cv2.AKAZE_create(),
    'KAZE': cv2.KAZE_create()
}

# Store results for analysis
results = []

# Apply each detector and measure performance
for name, detector in detectors.items():
    for idx, img in enumerate(images):
        start_time = time.time()
        
        # Detect keypoints and compute descriptors
        keypoints, descriptors = detector.detectAndCompute(img, None)
        end_time = time.time()
        
        # Collecting data
        results.append({
            'Detector': name,
            'Image Name': image_names[idx],
            'Keypoints': len(keypoints),
            'Descriptor Size': descriptors.shape if descriptors is not None else (0, 0),
            'Time': end_time - start_time
        })

# Convert results to a DataFrame for easy analysis
df_results = pd.DataFrame(results)

# Display results as a table
print(df_results)

# Visualize the results
plt.figure(figsize=(14, 7))
for name in detectors.keys():
    subset = df_results[df_results['Detector'] == name]
    plt.plot(subset['Image Name'], subset['Keypoints'], label=name, marker='o')
plt.title('Number of Keypoints Detected by Each Detector', fontsize=16, weight='bold')
plt.xlabel('Image Name', fontsize=14)
plt.ylabel('Number of Keypoints', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend(title='Detectors', loc='upper left', bbox_to_anchor=(1, 1), fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('./output/keypoints_detected.png') 
plt.close()
# plt.show()

# Optional: Visualization of computation time for each detector
plt.figure(figsize=(14, 7))
for name in detectors.keys():
    subset = df_results[df_results['Detector'] == name]
    plt.plot(subset['Image Name'], subset['Time'], label=name, marker='o')
plt.title('Computation Time for Each Detector', fontsize=16, weight='bold')
plt.xlabel('Image Name', fontsize=14)
plt.ylabel('Computation Time (seconds)', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend(title='Detectors', loc='upper left', bbox_to_anchor=(1, 1), fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('./output/computation_time.png')
plt.close()
# plt.show()
