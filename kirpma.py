import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Resimlerin bulunduğu dizin
input_dir = 'processed/3'
output_dir = 'processed/3'  # Aynı dizine yazmak için aynı dizini kullanıyoruz

# Dizindeki tüm resim dosyalarını listeleme
image_files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]

for image_file in image_files:
    image_path = os.path.join(input_dir, image_file)
    mr_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Görüntüyü bulanıklaştırma
    blurred = cv2.GaussianBlur(mr_image, (5, 5), 0)

    # İkili (binary) görüntü oluşturma
    _, binary_image = cv2.threshold(blurred, 30, 255, cv2.THRESH_BINARY)

    # Konturları bulma
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # En büyük konturu (beyin) bulma
        largest_contour = max(contours, key=cv2.contourArea)

        # En büyük konturun etrafındaki dikdörtgeni bulma
        x, y, w, h = cv2.boundingRect(largest_contour)

        # Beyin bölgesini kırpma
        cropped_image = mr_image[y:y+h, x:x+w]

        # Kırpılmış görüntüyü kaydetme (eski resmin üzerine yazma)
        cv2.imwrite(image_path, cropped_image)

    else:
        print(f"Kontur bulunamadı: {image_file}")

print("Tüm resimler başarıyla işlendi ve kaydedildi.")