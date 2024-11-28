### **README for Notebook 2: Deforestation Analysis with Sentinel-2 Data**

---

# **Deforestation Analysis with Sentinel-2 Data**

This notebook focuses on the analysis of deforestation in Ukraine using Sentinel-2 satellite imagery. The pipeline includes downloading the data, preprocessing grayscale images, detecting key features, and comparing them over time to identify changes.

---

## **Notebook Overview**

### **1. Data Download**
- **Purpose**: To acquire Sentinel-2 satellite imagery for deforestation analysis.
- **Process**:
  - Use the `kagglehub` library to download the dataset: `isaienkov/deforestation-in-ukraine`.
  - Verify the path to the downloaded data to ensure successful access.

---

### **2. Preprocessing Sentinel-2 Images**
- **Objective**: Load and prepare grayscale satellite images for feature extraction.
- **Steps**:
  1. Load images using the `rasterio` library.
  2. Convert the images to OpenCV-compatible formats (`np.uint8`).

---

### **3. Feature Detection Using SIFT**
- **Purpose**: Extract key features from the satellite images for comparison.
- **Process**:
  - Use the SIFT (Scale-Invariant Feature Transform) algorithm to detect keypoints and compute descriptors.
  - Visualize the detected keypoints on both "before" and "after" images.

---

### **4. Feature Matching**
- **Objective**: Compare features between the "before" and "after" images to identify deforestation areas.
- **Steps**:
  1. Use the `BFMatcher` (Brute Force Matcher) with `cv2.NORM_L2` norm to find matching features.
  2. Sort matches by their distance and visualize the top 50 matches.
  3. Display the number of matches to assess the similarity or changes between the images.

---

## **Outputs**
- Keypoint visualizations for both images.
- Matched keypoints displayed side-by-side.
- Quantitative summary of matching features.

---

### **Requirements**
- Libraries: `kagglehub`, `rasterio`, `cv2`, `numpy`, `matplotlib`
- Dataset: Sentinel-2 imagery from Kaggle (`isaienkov/deforestation-in-ukraine`).

---

### **Conclusion**
This notebook provides a foundational approach to analyze deforestation using feature detection and matching. The insights can be used to track environmental changes over time.

---

---
