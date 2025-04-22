# PyQt5-Image-Processing-GUI-HW5

## 📌 HW5 - PyQt5 Image Processing GUI

This project is a graphical user interface (GUI) application for image processing, developed using **PyQt5**. It supports the following functions:

- Image loading and display  
- Color space conversion (RGB, CMY, HSI, XYZ, Lab, YUV)  
- Pseudo color mapping (Jet, Hot, Cool, Spring, etc.)  
- k-means clustering in different color spaces  

---

### 🖼 Features

#### 📁 Image Loading  
- Allows users to open and display an image from the local filesystem.

#### 🎨 Color Space Conversion  
- Supports conversion to and visualization of the following color spaces:
  - RGB
  - ![GUI Screenshot](RGB.png)
  - CMY
  - ![GUI Screenshot](CMY.png)
  - HSI
  - ![GUI Screenshot](HSI.png)
  - XYZ
  - ![GUI Screenshot](XYZ.png)
  - Lab
  - ![GUI Screenshot](Lab.png)
  - YUV
  - ![GUI Screenshot](YUV.png)

#### 🌈 Pseudo Color Mapping  
- Applies OpenCV colormaps to grayscale images for pseudo color visualization:
  - Jet
  - Hot
  - Cool
  - Spring
  - ![GUI Screenshot](Pseudo_Image.png)


#### 📊 k-means Clustering  
- Performs k-means clustering on image pixels in selected color spaces:
  - RGB
  - ![GUI Screenshot](k_means_RGB.png)
  - HSI
  - ![GUI Screenshot](k_means_HSI.png)
  - Lab
  - ![GUI Screenshot](k_means_Lab.png)
- Result is displayed as a segmented image based on clustering.

---

### 🛠 How to Run

1. **Install dependencies**  
   ```bash
   pip install opencv-python-headless PyQt5 numpy scikit-learn
   ```

2. **Run the GUI application**  
   ```bash
   python hw5_gui.py
   ```

---

### 📂 File Structure

```
│
├── hw5_gui.py              # Main GUI application
├── color_convert.py        # Color space conversion functions
├── colormap.py             # Pseudo color mapping logic
├── kmeans_clustering.py    # k-means clustering logic
└── README.md               # This documentation
```

---

### 🧑‍💻 Development Environment

- Python 3.8+
- PyQt5
- OpenCV 4.x
- NumPy
- scikit-learn

---
