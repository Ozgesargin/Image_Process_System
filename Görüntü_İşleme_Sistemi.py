import sys
import cv2
from PyQt5.QtWidgets import QMainWindow, QApplication, QFileDialog, QMessageBox, QGraphicsScene, QGraphicsPixmapItem
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt

from gui_main import Ui_MainWindow

from Modulus.Tone_Operations import *
from Modulus.Contrast_Operations import *
from Modulus.Smoothing_Operations import *
from Modulus.Edge_Operations import *
from Modulus.Morphological_Operations import *
from Modulus.Perspective_Operations import *
from Modulus.Shifting_Operations import *


class ImageEditor(QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        self.original_image = None
        self.processed_image = None
        self.graphic = None

        self.ui.open_file.clicked.connect(self.open_image)
        self.ui.save_file.clicked.connect(self.save_image)
        self.ui.exit.clicked.connect(self.exit_app)
        self.ui.apply.clicked.connect(self.apply_selected_operations)

    def open_image(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Image Files (*.png *.jpg *.bmp *.jpeg)")
        if file_path:
            self.original_image = cv2.imread(file_path)
            self.processed_image = self.original_image.copy()
            self.display_image(self.original_image, target="original")
            self.display_image(self.processed_image, target="processed")

    def save_image(self):
        if self.processed_image is not None:
            file_path, _ = QFileDialog.getSaveFileName(self, "Save Image", "", "PNG Files (*.png);;JPEG Files (*.jpg *.jpeg)")
            if file_path:
                cv2.imwrite(file_path, self.processed_image)
                QMessageBox.information(self, "Kaydedildi", "Resim başarıyla kaydedildi.")
        else:
            QMessageBox.warning(self, "Görüntü Yok", "Kaydedilecek işlenmiş görüntü yok.")

    def display_image(self, img, target="original"):
        if img is None:
            return

        if len(img.shape) == 2:  
            height, width = img.shape
            bytes_per_line = width
            q_img = QImage(img.data, width, height, bytes_per_line, QImage.Format_Grayscale8)
        else:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            height, width, channel = img_rgb.shape
            bytes_per_line = 3 * width
            q_img = QImage(img_rgb.data, width, height, bytes_per_line, QImage.Format_RGB888)

        pixmap = QPixmap.fromImage(q_img)

        scene = QGraphicsScene()
        item = QGraphicsPixmapItem(pixmap)
        scene.addItem(item)

        if target == "original":
            self.ui.original.setScene(scene)
            self.ui.original.fitInView(item, Qt.KeepAspectRatio)
        elif target == "processed":
            self.ui.processed.setScene(scene)
            self.ui.processed.fitInView(item, Qt.KeepAspectRatio)
        elif target == "graphic":
            self.ui.graphic.setScene(scene)
            self.ui.graphic.fitInView(item, Qt.KeepAspectRatio)

    def exit_app(self):
        self.close()

    def apply_selected_operations(self):
        if self.original_image is None:
            QMessageBox.warning(self, "Uyarı", "Lütfen önce bir resim yükleyin.")
            return

        # Tone operations
        if self.ui.grayscale.isChecked():
            self.processed_image = grayscale(self.original_image)
        elif self.ui.negative.isChecked():
            self.processed_image = negative(self.original_image)
        elif self.ui.gray_negative.isChecked():
            self.processed_image = gray_negative(self.original_image)
        elif self.ui.half_negative.isChecked():
            self.processed_image = half_negative(self.original_image)
        elif self.ui.rgb_channels.isChecked():
            r, g, b = separate_RGB_channels(self.original_image)
            combined = np.hstack((r, g, b))  
            self.processed_image = combined

        # Contrast operations
        elif self.ui.contrast_stretching.isChecked():
            try:
                in_min = int(self.ui.in_min_contrast.text())
            except ValueError:
                QMessageBox.warning(self, "Uyarı", "Lütfen önce bir sayı giriniz.")

            try:
                in_max = int(self.ui.in_max_contrast.text())
            except ValueError:
                QMessageBox.warning(self, "Uyarı", "Lütfen önce bir sayı giriniz.")

            self.processed_image = manuel_contrast_streching(self.original_image, in_min, in_max)

        elif self.ui.contrast_slider.value() != 0: 
                self.processed_image = contrast(self.original_image, self.ui.contrast_slider.value())

        elif self.ui.brightness_slider.value() != 0:
            self.processed_image = brightness(self.original_image, self.ui.brightness_slider.value())

        elif self.ui.histogram.isChecked():
            hist_img = histogram(self.original_image)
            if hist_img is not None:
                self.display_image(hist_img, target="graphic")

        elif self.ui.histogram_equalization.isChecked():
            processed_img, hist_img = histogram_equalization(self.original_image)
            self.processed_image = processed_img
            self.display_image(hist_img, target="graphic")

        elif self.ui.thresholding.isChecked():
            try:
                threshold_value = int(self.ui.threshold_value.text())
            except ValueError:
                QMessageBox.warning(self, "Uyarı", "Lütfen önce bir sayı giriniz.")
                return
            self.processed_image = thresholding(self.original_image, threshold_value)

        # Shifting operations
        elif self.ui.shifting.isChecked():
            try:
                right_shift = int(self.ui.x_shift.text())
            except ValueError:
                QMessageBox.warning(self, "Uyarı", "Lütfen önce bir sayı giriniz.")
                return
            try:
                down_shift = int(self.ui.y_shift.text())
            except ValueError:
                QMessageBox.warning(self, "Uyarı", "Lütfen önce bir sayı giriniz.")
                return
            self.processed_image = shifting(self.original_image, right_shift, down_shift)

        elif self.ui.shearing.isChecked():
            try:
                shear_val = float(self.ui.shearing_value.value())  
            except ValueError:
                QMessageBox.warning(self, "Uyarı", "Lütfen geçerli bir sayı giriniz.")
                return

            if self.ui.sh_x.isChecked():
                sh_x = shear_val
                sh_y = 0
            elif self.ui.sh_y.isChecked():
                sh_x = 0
                sh_y = shear_val
            else:
                QMessageBox.warning(self, "Uyarı", "Lütfen X veya Y eksenini seçiniz.")
                return

            self.processed_image = shearing(self.original_image, sh_x=sh_x, sh_y=sh_y)

        elif self.ui.scaling.isChecked():
            try:
                scale_factor = float(self.ui.scaling_value.value())
            except ValueError:
                QMessageBox.warning(self, "Uyarı", "Lütfen geçerli bir sayı giriniz.")
                return

            scaling_type = self.ui.scaling_type.currentText()
            scale_up = True if scaling_type == "Büyült" else False

            self.processed_image = scale_image(self.original_image, scale_factor, scale_up)

        elif self.ui.cropping.isChecked():
            try:
                x_min = int(self.ui.crop_x_min.text())
            except ValueError:
                QMessageBox.warning(self, "Uyarı", "Lütfen önce bir sayı giriniz.")
                return
            try:
                x_max = int(self.ui.crop_x_max.text())
            except ValueError:
                QMessageBox.warning(self, "Uyarı", "Lütfen önce bir sayı giriniz.")
                return
            try:
                y_min = int(self.ui.crop_y_min.text())
            except ValueError:
                QMessageBox.warning(self, "Uyarı", "Lütfen önce bir sayı giriniz.")
                return
            try:
                y_max = int(self.ui.crop_y_max.text())
            except ValueError:
                QMessageBox.warning(self, "Uyarı", "Lütfen önce bir sayı giriniz.")
                return
            self.processed_image = crop_image(self.original_image, x_min, x_max, y_min, y_max)

        elif self.ui.rotating.isChecked():
            try:
                rotate_angle = int(self.ui.rotate_angle.value())
            except ValueError:
                QMessageBox.warning(self, "Uyarı", "Lütfen önce bir sayı giriniz.")
                return
            self.processed_image = rotate_image_with_interpolation(self.original_image, rotate_angle)

        elif self.ui.mirroring.isChecked():
            try:
                mirror_angle = int(self.ui.mirror_angle.value())
            except ValueError:
                QMessageBox.warning(self, "Uyarı", "Lütfen önce bir sayı giriniz.")
                return
            self.processed_image = mirror_image(self.original_image, mirror_angle)

        # Perspective operations
        elif self.ui.perspective_correction.isChecked():
            self.processed_image = perspective_correction(self.original_image)
        elif self.ui.interactive_perspective_correction.isChecked():
            self.processed_image = interactive_perspective_correction(self.original_image)

        # Smoothing operations
        elif self.ui.mean_filter.isChecked():
            if self.original_image is None:
                QMessageBox.warning(self, "Uyarı", "Lütfen önce bir görüntü yükleyiniz.")
                return

            self.processed_image = mean_filter(self.original_image)

        elif self.ui.median_filter.isChecked():
            if self.original_image is None:
                QMessageBox.warning(self, "Uyarı", "Lütfen önce bir görüntü yükleyiniz.")
                return
            self.processed_image = median_filter(self.original_image)

        elif self.ui.gauss_filter.isChecked():
            selected_option = self.ui.gauss_filter_type.currentText()

            kernel_sigma_map = {
                "3x3, σ=1": ((3, 3), 1),
                "5x5, σ=2": ((5, 5), 2),
                "7x7, σ=3": ((7, 7), 3),
                "9x9, σ=4": ((9, 9), 4)
            }

            if selected_option in kernel_sigma_map:
                kernel_size, sigma = kernel_sigma_map[selected_option]
                self.processed_image = gauss_filter(self.original_image, kernel_size, sigma)
            else:
                QMessageBox.warning(self, "Uyarı", "Lütfen geçerli bir filtre seçiniz.")

        elif self.ui.conservative_smoothing.isChecked():
            self.processed_image = conservative_filter(self.original_image)

        elif self.ui.crimmins_speckle.isChecked():
            self.processed_image = crimmins_speckle(self.original_image)

        elif self.ui.fourier_transform.isChecked():
            selected_filter = self.ui.fourier_transform_type.currentText()
            gray_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)

            if selected_filter == "Low-Pass Filtreleme":
                self.processed_image = apply_low_pass_filter(gray_image)
            elif selected_filter == "High-Pass Filtreleme":
                self.processed_image = apply_high_pass_filter(gray_image)
            else:
                QMessageBox.warning(self, "Uyarı", "Lütfen geçerli bir filtre seçiniz.")

        elif self.ui.band_filter.isChecked():
            selected_filter = self.ui.band_filter_type.currentText()

            if selected_filter == "Band Geçiren Filtre":
                self.processed_image = apply_band_filter(self.original_image, D1=20, D2=50, band_pass=True)
            elif selected_filter == "Band Durduran Filtre":
                self.processed_image = apply_band_filter(self.original_image, D1=20, D2=50, band_pass=False)
            else:
                QMessageBox.warning(self, "Uyarı", "Lütfen geçerli bir filtre seçiniz.")

        elif self.ui.butterworth_filter.isChecked():
            selected = self.ui.butterworth_filter_type.currentText() 
    
            if selected == "Alçak Geçiren Filtre":
                self.processed_image = apply_butterworth_filter(self.original_image, D0=30, n=2, filter_type="lowpass")
            elif selected == "Yüksek Geçiren Filtre":
                self.processed_image = apply_butterworth_filter(self.original_image, D0=30, n=2, filter_type="highpass")
            else:
                QMessageBox.warning(self, "Uyarı", "Lütfen geçerli bir filtre seçiniz.")

        elif self.ui.gaussian_filter.isChecked():
            selected = self.ui.gaussian_filter_type.currentText()  
    
            if selected == "Low-Pass Filtreleme":
                self.processed_image = apply_gaussian_filter(self.original_image, D0=30, filter_type="lowpass")
            elif selected == "High-Pass Filtreleme":
                self.processed_image = apply_gaussian_filter(self.original_image, D0=30, filter_type="highpass")
            else:
                QMessageBox.warning(self, "Uyarı", "Lütfen geçerli bir filtre seçiniz.")

        elif self.ui.homomorphic_filter.isChecked():
            d0 = 50
            h_l = 0.5
            h_h = 2.0
            c = 1

            self.processed_image = homomorphic_filter(self.original_image, d0=d0, h_l=h_l, h_h=h_h, c=c)

        # Edge operations
        elif self.ui.sobel.isChecked():
            self.processed_image = sobel(self.original_image)
        elif self.ui.prewitt.isChecked():
            self.processed_image = prewitt(self.original_image)
        elif self.ui.robertss_cross.isChecked():
            self.processed_image = robertss_cross(self.original_image)
        elif self.ui.compass.isChecked():
            self.processed_image = compass(self.original_image)
        elif self.ui.canny.isChecked():
            self.processed_image = canny(self.original_image)
        elif self.ui.laplace.isChecked():
            self.processed_image = laplacian(self.original_image)
        elif self.ui.gabor.isChecked():
            self.processed_image = gabor(self.original_image)
        elif self.ui.hough_transformation.isChecked():
            selected = self.ui.hough_type.currentText()

            if selected == "Doğru Dönüşümü":
                self.processed_image = apply_hough_lines(self.original_image)
            elif selected == "Çember Dönüşümü":
                self.processed_image = apply_hough_circles(self.original_image)
            else:
                QMessageBox.warning(self, "Uyarı", "Lütfen geçerli bir dönüşüm seçiniz.")
        elif self.ui.k_means_segmentation.isChecked():
            self.processed_image = apply_kmeans_segmentation(self.original_image, k=3)

        # Morphological operations
        elif self.ui.erode.isChecked():
            self.processed_image = erode(self.original_image, kernel_size=3, iterations=1)
        elif self.ui.dilate.isChecked():
            self.processed_image = dilate(self.original_image, kernel_size=3, iterations=1)

        else:
            self.processed_image = self.original_image.copy()  

        self.display_image(self.processed_image, target="processed")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ImageEditor()
    window.show()
    sys.exit(app.exec_())
