import cv2 as cv
import sys
import numpy as np
import matplotlib.pyplot as plt
from PyQt5 import QtWidgets, QtGui, QtCore
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas 

class HW5(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()

        self.image_data = None

        self.setWindowTitle('HW5')
        self.resize(1400, 900)
        self.scene = QtWidgets.QGraphicsScene(self)
        self.scene.setSceneRect(0, 0, 1000, 900)
        self.view = QtWidgets.QGraphicsView(self)
        self.view.setGeometry(0, 0, 1000, 900)
        self.view.setScene(self.scene)
        self.view.setBackgroundBrush(QtGui.QBrush(QtCore.Qt.white))

        self.btn_open_file = QtWidgets.QPushButton(self)
        self.btn_open_file.move(1140, 150)
        self.btn_open_file.resize(110, 40)
        self.btn_open_file.setText('開啟圖片')
        self.btn_open_file.setStyleSheet("background-color: green;")
        self.btn_open_file.clicked.connect(self.open_file)
        
        self.btn_open_file = QtWidgets.QPushButton(self)
        self.btn_open_file.move(1020, 215)
        self.btn_open_file.resize(110, 40)
        self.btn_open_file.setText('RGB')
        # self.btn_open_file.setStyleSheet("background-color: green;")
        self.btn_open_file.clicked.connect(self.rgb_image)

        self.btn_open_file = QtWidgets.QPushButton(self)
        self.btn_open_file.move(1140, 215)
        self.btn_open_file.resize(110, 40)
        self.btn_open_file.setText('CMY')
        # self.btn_open_file.setStyleSheet("background-color: green;")
        self.btn_open_file.clicked.connect(self.cmy_image)

        self.btn_open_file = QtWidgets.QPushButton(self)
        self.btn_open_file.move(1260, 215)
        self.btn_open_file.resize(110, 40)
        self.btn_open_file.setText('HSI')
        # self.btn_open_file.setStyleSheet("background-color: green;")
        self.btn_open_file.clicked.connect(self.hsi_image)

        self.btn_open_file = QtWidgets.QPushButton(self)
        self.btn_open_file.move(1020, 275)
        self.btn_open_file.resize(110, 40)
        self.btn_open_file.setText('XYZ')
        # self.btn_open_file.setStyleSheet("background-color: green;")
        self.btn_open_file.clicked.connect(self.xyz_image)

        self.btn_open_file = QtWidgets.QPushButton(self)
        self.btn_open_file.move(1140, 275)
        self.btn_open_file.resize(110, 40)
        self.btn_open_file.setText('Lab')
        # self.btn_open_file.setStyleSheet("background-color: green;")
        self.btn_open_file.clicked.connect(self.lab_image)

        self.btn_open_file = QtWidgets.QPushButton(self)
        self.btn_open_file.move(1260, 275)
        self.btn_open_file.resize(110, 40)
        self.btn_open_file.setText('YUV')
        # self.btn_open_file.setStyleSheet("background-color: green;")
        self.btn_open_file.clicked.connect(self.yuv_image)

        
        self.box = QtWidgets.QComboBox(self)   
        self.box.addItems(['autumn','bone', 'cool', 'hot', 'hsv' ,'jet', 'ocean', 'pink', 'rainbow', 'spring', 'summer' ,'winter'])   
        self.box.setGeometry(1050,335,300,30)
        self.box.currentIndexChanged.connect(self.Pseudo_color_Image)

        self.btn_open_file = QtWidgets.QPushButton(self)
        self.btn_open_file.move(1140, 375)
        self.btn_open_file.resize(120, 40)
        self.btn_open_file.setText('Pseudo_Image')
        # self.btn_open_file.setStyleSheet("background-color: green;")
        self.btn_open_file.clicked.connect(self.Pseudo_color_Image)

        self.btn_open_file = QtWidgets.QPushButton(self)
        self.btn_open_file.move(1140, 425)
        self.btn_open_file.resize(120, 40)
        self.btn_open_file.setText('k_means_RGB')
        # self.btn_open_file.setStyleSheet("background-color: green;")
        self.btn_open_file.clicked.connect(self.k_means_RGB)

        self.btn_open_file = QtWidgets.QPushButton(self)
        self.btn_open_file.move(1140, 475)
        self.btn_open_file.resize(120, 40)
        self.btn_open_file.setText('k_means_HSI')
        # self.btn_open_file.setStyleSheet("background-color: green;")
        self.btn_open_file.clicked.connect(self.k_means_HSI)

        self.btn_open_file = QtWidgets.QPushButton(self)
        self.btn_open_file.move(1140, 525)
        self.btn_open_file.resize(120, 40)
        self.btn_open_file.setText('k_means_Lab')
        # self.btn_open_file.setStyleSheet("background-color: green;")
        self.btn_open_file.clicked.connect(self.k_means_Lab)
        
        self.btn_close = QtWidgets.QPushButton(self)
        self.btn_close.setText('關閉')
        self.btn_close.setGeometry(1135, 750, 100, 30)
        self.btn_close.resize(110, 40)
        self.btn_close.clicked.connect(self.closeFile)

        self.output_height = 0
        self.output_width = 0
        self.filter_size = None

    def closeFile(self):
        ret = QtWidgets.QMessageBox.question(self, 'question', '確定關閉視窗？')
        if ret == QtWidgets.QMessageBox.Yes:
            app.quit()
        else:
            return

    def open_file(self):
        file_name, _ = QtWidgets.QFileDialog.getOpenFileName()
        if file_name:
            self.image_data = cv.imread(file_name)
            self.image_data = cv.cvtColor(self.image_data, cv.COLOR_BGR2RGB)
            fig, ax = plt.subplots(figsize=(8, 6))
            self.pic = ax.imshow(self.image_data)
            ax.set_xticks([])
            ax.set_yticks([])
            canvas = FigureCanvas(fig)
            self.view.setAlignment(QtCore.Qt.AlignCenter)
            self.scene.clear()
            self.scene.addWidget(canvas)

    # Convert to RGB color space
    def rgb_image(self):
        global rgb_image, R, G, B
        rgb_image = self.image_data
        R = rgb_image[:,:,0]
        G = rgb_image[:,:,1]
        B = rgb_image[:,:,2]

        fig, ax = plt.subplots(2, 2, figsize=(10, 8))

        ax[0, 0].imshow(rgb_image)
        ax[0, 0].set_title("RGB")
        ax[0, 0].axis('off')

        ax[0, 1].imshow(cv.cvtColor(R, cv.COLOR_RGB2BGR))
        ax[0, 1].set_title("R")
        ax[0, 1].axis('off')

        ax[1, 0].imshow(cv.cvtColor(G, cv.COLOR_RGB2BGR))
        ax[1, 0].set_title('G')
        ax[1, 0].axis('off')

        ax[1, 1].imshow(cv.cvtColor(B, cv.COLOR_RGB2BGR))
        ax[1, 1].set_title('B')
        ax[1, 1].axis('off')

        canvas = FigureCanvas(fig)
        self.view.setAlignment(QtCore.Qt.AlignCenter)
        self.scene.addWidget(canvas)

    # Convert to CMY color space
    def cmy_image(self):
        cmy_image = 255 - self.image_data
        C = cmy_image[:, :, 0]  # 色相通道
        M = cmy_image[:, :, 1]  # 饱和度通道
        Y = cmy_image[:, :, 2]  # 强度通道

        fig, ax = plt.subplots(2, 2, figsize=(10, 8))

        ax[0, 0].imshow(cmy_image)
        ax[0, 0].set_title("CMY")
        ax[0, 0].axis('off')

        ax[0, 1].imshow(cv.cvtColor(C, cv.COLOR_RGB2BGR))
        ax[0, 1].set_title("C")
        ax[0, 1].axis('off')

        ax[1, 0].imshow(cv.cvtColor(M, cv.COLOR_RGB2BGR))
        ax[1, 0].set_title('M')
        ax[1, 0].axis('off')

        ax[1, 1].imshow(cv.cvtColor(Y, cv.COLOR_RGB2BGR))
        ax[1, 1].set_title('Y')
        ax[1, 1].axis('off')

        canvas = FigureCanvas(fig)
        self.view.setAlignment(QtCore.Qt.AlignCenter)
        self.scene.addWidget(canvas)

    # Convert to HSI color space
    def hsi_image(self):
        global hsi_img
        # 保存原始圖像的維度
        row, col, _ = np.shape(self.image_data)

        # 複製原始圖像
        hsi_img = self.image_data.copy()

        # 分離通道
        B, G, R = cv.split(hsi_img)

        # 正規化通道到[0, 1]
        B, G, R = B / 255.0, G / 255.0, R / 255.0

        # 計算H、S、I通道
        # 計算H通道
        # 計算H通道
        H = np.zeros((row, col))
        for i in range(row):
            den = np.sqrt((R[i] - G[i])**2 + (R[i] - B[i]) * (G[i] - B[i]))
            # 檢查分母是否為零，避免除以零
            if np.any(den != 0):
                thetha = np.arccos(0.5 * (R[i] - B[i] + R[i] - G[i]) / den)
                h = np.zeros(col)
                h[B[i] <= G[i]] = thetha[B[i] <= G[i]]
                h[G[i] < B[i]] = 2 * np.pi - thetha[G[i] < B[i]]
                H[i] = h / (2 * np.pi)
            else:
                # 如果分母為零，可以根據實際情況進行處理，這裡先將H通道設為0
                H[i] = 0

        # 將H縮放到 [0, 255] 範圍
        H = (H * 255).astype(np.uint8)

        # 計算I和S通道
        I = (R + G + B) / 3.0
        S = 1 - (3 * np.minimum.reduce([R, G, B])) / (R + G + B)

        # 將I和S縮放到 [0, 255] 範圍
        I = (I * 255).astype(np.uint8)
        S = (S * 255).astype(np.uint8)

        # 將H、S、I通道組合成HSI圖像
        hsi_img = np.stack([H, S, I], axis=-1)

        # 顯示圖像
        fig, ax = plt.subplots(2, 2, figsize=(10, 8))

        ax[0, 0].imshow(cv.cvtColor(hsi_img, cv.COLOR_BGR2RGB))
        ax[0, 0].set_title("HSI")
        ax[0, 0].axis('off')

        ax[0, 1].imshow(H, cmap='hsv')
        ax[0, 1].set_title("H")
        ax[0, 1].axis('off')

        ax[1, 0].imshow(S, cmap='gray')
        ax[1, 0].set_title('S')
        ax[1, 0].axis('off')

        ax[1, 1].imshow(I, cmap='gray')
        ax[1, 1].set_title('I')
        ax[1, 1].axis('off')

        # 假設self.scene是有效的QGraphicsScene
        canvas = FigureCanvas(fig)
        self.view.setAlignment(QtCore.Qt.AlignCenter)
        self.scene.addWidget(canvas)

        # Convert to XYZ color space
    def xyz_image(self):
            xyz_image = cv.cvtColor(rgb_image, cv.COLOR_RGB2XYZ)
            X = xyz_image[:, :, 0]
            Y = xyz_image[:, :, 1]
            Z = xyz_image[:, :, 2]

            fig, ax = plt.subplots(2, 2, figsize=(10, 8))

            ax[0, 0].imshow(xyz_image)
            ax[0, 0].set_title("XYZ")
            ax[0, 0].axis('off')

            ax[0, 1].imshow(cv.cvtColor(X, cv.COLOR_RGB2BGR))
            ax[0, 1].set_title("X")
            ax[0, 1].axis('off')

            ax[1, 0].imshow(cv.cvtColor(Y, cv.COLOR_RGB2BGR))
            ax[1, 0].set_title('Y')
            ax[1, 0].axis('off')

            ax[1, 1].imshow(cv.cvtColor(Z, cv.COLOR_RGB2BGR))
            ax[1, 1].set_title('Z')
            ax[1, 1].axis('off')

            canvas = FigureCanvas(fig)
            self.view.setAlignment(QtCore.Qt.AlignCenter)
            self.scene.addWidget(canvas)


    # Convert to Lab color space
    def lab_image(self):
        global lab_image
        lab_image = cv.cvtColor(rgb_image, cv.COLOR_RGB2Lab)
        L = lab_image[:, :, 0]
        a = lab_image[:, :, 1]
        b = lab_image[:, :, 2]

        fig, ax = plt.subplots(2, 2, figsize=(10, 8))

        ax[0, 0].imshow(lab_image)
        ax[0, 0].set_title("Lab")
        ax[0, 0].axis('off')

        ax[0, 1].imshow(cv.cvtColor(L, cv.COLOR_RGB2BGR))
        ax[0, 1].set_title("L")
        ax[0, 1].axis('off')

        ax[1, 0].imshow(cv.cvtColor(a, cv.COLOR_RGB2BGR))
        ax[1, 0].set_title('a')
        ax[1, 0].axis('off')

        ax[1, 1].imshow(cv.cvtColor(b, cv.COLOR_RGB2BGR))
        ax[1, 1].set_title('b')
        ax[1, 1].axis('off')

        canvas = FigureCanvas(fig)
        self.view.setAlignment(QtCore.Qt.AlignCenter)
        self.scene.addWidget(canvas)

    # Convert to YUV color space
    def yuv_image(self):
        yuv_image = cv.cvtColor(rgb_image, cv.COLOR_RGB2YUV)
        y = yuv_image[:, :, 0]
        u = yuv_image[:, :, 1]
        v = yuv_image[:, :, 2]

        fig, ax = plt.subplots(2, 2, figsize=(10, 8))

        ax[0, 0].imshow(yuv_image)
        ax[0, 0].set_title("YUV")
        ax[0, 0].axis('off')

        ax[0, 1].imshow(cv.cvtColor(y, cv.COLOR_RGB2BGR))
        ax[0, 1].set_title("Y")
        ax[0, 1].axis('off')

        ax[1, 0].imshow(cv.cvtColor(u, cv.COLOR_RGB2BGR))
        ax[1, 0].set_title('U')
        ax[1, 0].axis('off')

        ax[1, 1].imshow(cv.cvtColor(v, cv.COLOR_RGB2BGR))
        ax[1, 1].set_title('V')
        ax[1, 1].axis('off')

        canvas = FigureCanvas(fig)
        self.view.setAlignment(QtCore.Qt.AlignCenter)
        self.scene.addWidget(canvas)

    def Pseudo_color_Image(self,):
        self.scene.clear()
        selected_option = self.box.currentText()
        if self.view.items():
    # 視圖不為空
            if cv.cvtColor(self.image_data, cv.COLOR_RGB2GRAY).ndim == 2:
                # 當前圖像為灰度，不做任何操作
                pass
            else:
                # 當前圖像不是灰度，提示用戶選擇新圖像
                new_file_name, _ = QtWidgets.QFileDialog.getOpenFileName()
                if new_file_name:
                    self.image_data = cv.imread(new_file_name)
                    self.image_data = cv.cvtColor(self.image_data, cv.COLOR_BGR2RGB)
                    # 視圖為空，繼續添加新圖像
                    fig, ax = plt.subplots(figsize=(8, 6))
                    self.pic = ax.imshow(self.image_data)
                    ax.set_xticks([])
                    ax.set_yticks([])
                    canvas = FigureCanvas(fig)
                    self.view.setAlignment(QtCore.Qt.AlignCenter)
                    self.scene.clear()
                    self.scene.addWidget(canvas)
        else:
            # 視圖為空，繼續添加圖像
            file_name, _ = QtWidgets.QFileDialog.getOpenFileName()
            if file_name:
                self.image_data = cv.imread(file_name)
                self.image_data = cv.cvtColor(self.image_data, cv.COLOR_BGR2RGB)
                fig, ax = plt.subplots(figsize=(8, 6))
                self.pic = ax.imshow(self.image_data)
                ax.set_xticks([])
                ax.set_yticks([])
                canvas = FigureCanvas(fig)
                self.view.setAlignment(QtCore.Qt.AlignCenter)
                self.scene.clear()
                self.scene.addWidget(canvas)

        if  selected_option == 'autumn':
            color_map = cv.COLORMAP_AUTUMN

        if  selected_option =='bone':
            color_map = cv.COLORMAP_BONE

        if  selected_option =='cool':
            color_map = cv.COLORMAP_COOL

        if  selected_option =='hot':
            color_map = cv.COLORMAP_HOT

        if  selected_option =='hsv':
            color_map = cv.COLORMAP_HSV

        if  selected_option =='jet':
            color_map = cv.COLORMAP_JET

        if  selected_option =='ocean':
            color_map = cv.COLORMAP_OCEAN

        if  selected_option =='pink':
            color_map = cv.COLORMAP_PINK

        if  selected_option =='rainbow':
            color_map = cv.COLORMAP_RAINBOW

        if  selected_option =='spring':
            color_map = cv.COLORMAP_SPRING

        if  selected_option =='summer': 
            color_map = cv.COLORMAP_SUMMER

        if  selected_option =='winter':
            color_map = cv.COLORMAP_WINTER

        # Normalize the gray image to the range [0, 1]
        normalized_gray_image = self.image_data / 255.0

        # Apply the color map to the grayscale image
        pseudo_color_image = cv.applyColorMap((normalized_gray_image * 255).astype(np.uint8), color_map)

        fig, ax = plt.subplots(1, 2, figsize=(10, 8))
        ax[0].imshow(self.image_data, cmap='gray')
        ax[0].set_title('Grayscale Image')
        ax[0].axis('off')

        im = ax[1].imshow(pseudo_color_image)
        ax[1].set_title('Pseudo-Color Image')
        ax[1].axis('off')

        cbar = fig.colorbar(im, ax=ax[1])
        cbar.set_label('Intensity')
        
        canvas = FigureCanvas(fig)
        self.view.setAlignment(QtCore.Qt.AlignCenter)
        self.scene.addWidget(canvas)

    def Color_Segmentation_RGB(self):
        # self.image_data = cv2.cvtColor(self.image_data, cv2.COLOR_RGB2BGR)
        pixels = self.image_data.reshape(-1, 3).astype(np.float32)
        segmented_images = {}

        for i in range(2, 6):
            num_clusters = i

            # Apply K-means clustering
            criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 0.2)
            _, labels, centers = cv.kmeans(pixels, num_clusters, None, criteria, 10, cv.KMEANS_RANDOM_CENTERS)

            # Convert the cluster centers to integers
            centers = np.uint8(centers)

            # Map each pixel to its cluster center color
            segmented_images[i] = centers[labels.flatten()].reshape(self.image_data.shape)

        # Plot the original and segmented images
        fig, ax = plt.subplots(2, 2, figsize=(10, 8))

        ax[0, 0].imshow(segmented_images[2])
        ax[0, 0].set_title('k = 2 ')
        ax[0, 0].axis('off')

        ax[0, 1].imshow(segmented_images[3])
        ax[0, 1].set_title('k = 3 ')
        ax[0, 1].axis('off')

        ax[1, 0].imshow(segmented_images[4])
        ax[1, 0].set_title('k = 4 ')
        ax[1, 0].axis('off')

        ax[1, 1].imshow(segmented_images[5])
        ax[1, 1].set_title('k = 5 ')
        ax[1, 1].axis('off')

        canvas = FigureCanvas(fig)
        self.view.setAlignment(QtCore.Qt.AlignCenter)
        self.scene.addWidget(canvas)

    def Color_Segmentation(self, image, num_clusters) :
        pixels = image.reshape(-1, 3).astype(np.float32)

        # Define the number of clusters (segments) you wan
        num_clusters = num_clusters

        # Apply K-means clustering
        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 0.2)
        _, labels, centers = cv.kmeans(pixels, num_clusters, None, criteria, 10, cv.KMEANS_RANDOM_CENTERS)

        # Convert the cluster centers to integers
        centers = np.uint8(centers)

        # Map each pixel to its cluster center color
        segmented_image = centers[labels.flatten()].reshape(image.shape)

        fig, ax = plt.subplots(1, 2, figsize=(10, 8))
        ax[0].imshow(image, cmap='gray')
        ax[0].set_title('Original_image')
        ax[0].axis('off')

        ax[1].imshow(segmented_image)
        ax[1].set_title('Segmented Image')
        ax[1].axis('off')

        canvas = FigureCanvas(fig)
        self.view.setAlignment(QtCore.Qt.AlignCenter)
        self.scene.addWidget(canvas)

    def k_means_RGB(self):
        file_name, _ = QtWidgets.QFileDialog.getOpenFileName()
        if file_name:
            self.image_data = cv.imread(file_name)
            self.image_data = cv.cvtColor(self.image_data, cv.COLOR_BGR2RGB)
            fig, ax = plt.subplots(figsize=(8, 6))
            self.pic = ax.imshow(self.image_data)
            ax.set_xticks([])
            ax.set_yticks([])
            canvas = FigureCanvas(fig)
            self.view.setAlignment(QtCore.Qt.AlignCenter)
            self.scene.clear()
            self.scene.addWidget(canvas)

        self.Color_Segmentation_RGB()   

    def k_means_HSI(self):
        file_name, _ = QtWidgets.QFileDialog.getOpenFileName()
        self.image_data = cv.imread(file_name)
        self.image_data = cv.cvtColor(self.image_data, cv.COLOR_BGR2RGB)
        # View is empty, proceed to add the image
        fig, ax = plt.subplots(figsize=(8, 6))
        self.pic = ax.imshow(self.image_data)
        ax.set_xticks([])
        ax.set_yticks([])
        canvas = FigureCanvas(fig)
        self.view.setAlignment(QtCore.Qt.AlignCenter)
        self.scene.clear()
        self.scene.addWidget(canvas)

        row, col, _ = np.shape(self.image_data)

        # 複製原始圖像
        hsi_img = self.image_data.copy()

        # 分離通道
        B, G, R = cv.split(hsi_img)

        # 正規化通道到[0, 1]
        B, G, R = B / 255.0, G / 255.0, R / 255.0

        # 計算H、S、I通道
        # 計算H通道
        # 計算H通道
        H = np.zeros((row, col))
        for i in range(row):
            den = np.sqrt((R[i] - G[i])**2 + (R[i] - B[i]) * (G[i] - B[i]))
            # 檢查分母是否為零，避免除以零
            if np.any(den != 0):
                thetha = np.arccos(0.5 * (R[i] - B[i] + R[i] - G[i]) / den)
                h = np.zeros(col)
                h[B[i] <= G[i]] = thetha[B[i] <= G[i]]
                h[G[i] < B[i]] = 2 * np.pi - thetha[G[i] < B[i]]
                H[i] = h / (2 * np.pi)
            else:
                # 如果分母為零，可以根據實際情況進行處理，這裡先將H通道設為0
                H[i] = 0

        # 將H縮放到 [0, 255] 範圍
        H = (H * 255).astype(np.uint8)

        # 計算I和S通道
        I = (R + G + B) / 3.0
        S = 1 - (3 * np.minimum.reduce([R, G, B])) / (R + G + B)

        # 將I和S縮放到 [0, 255] 範圍
        I = (I * 255).astype(np.uint8)
        S = (S * 255).astype(np.uint8)

        # 將H、S、I通道組合成HSI圖像
        hsi_img = np.stack([H, S, I], axis=-1)
        self.Color_Segmentation(image = hsi_img, num_clusters = 2)   
        
    def k_means_Lab(self):
        file_name, _ = QtWidgets.QFileDialog.getOpenFileName()
        self.image_data = cv.imread(file_name)
        self.image_data = cv.cvtColor(self.image_data, cv.COLOR_BGR2RGB)
        # View is empty, proceed to add the image
        fig, ax = plt.subplots(figsize=(8, 6))
        self.pic = ax.imshow(self.image_data)
        ax.set_xticks([])
        ax.set_yticks([])
        canvas = FigureCanvas(fig)
        self.view.setAlignment(QtCore.Qt.AlignCenter)
        self.scene.clear()
        self.scene.addWidget(canvas)

        lab_image = cv.cvtColor(self.image_data, cv.COLOR_RGB2Lab)

        self.Color_Segmentation(image = lab_image, num_clusters = 2)    

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    ex = HW5()
    ex.show()
    sys.exit(app.exec_())