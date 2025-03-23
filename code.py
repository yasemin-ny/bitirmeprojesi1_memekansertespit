# Gerekli kütüphaneler
import os                # Dosya işlemleri için
import zipfile           # Zip dosyasını açmak için
import shutil            # Dosyaları taşımak veya kopyalamak için
import numpy as np       # Sayısal hesaplamalar için
import matplotlib.pyplot as plt  # Görselleri göstermek için
import cv2               # Görsel işleme için
from sklearn.model_selection import train_test_split  # Veri setini ayırmak için

# Yüklenen zip dosyasının adı
zip_dosya_adi = "mias_mamography.zip"

# Zip dosyasını çıkart
with zipfile.ZipFile(zip_dosya_adi, 'r') as zip_ref:
    zip_ref.extractall("mias_data")

# Çıkartılan dosyaları kontrol edelim
print("Dosyalar:", os.listdir("mias_data"))

# Tar.gz dosyasını aç
import tarfile

tar_dosya_yolu = "mias_data/all-mias.tar.gz"
hedef_klasor = "mias_data/all_mias_images"

# Dosyayı çıkar
with tarfile.open(tar_dosya_yolu, "r:gz") as tar:
    tar.extractall(hedef_klasor)

# Çıkartılan dosyaları kontrol et
print("Çıkartılan klasör:", os.listdir(hedef_klasor))

# PGM formatındaki bir görüntüyü yükleyip görselleştirelim
pgm_dosya_yolu = os.path.join(hedef_klasor, 'mdb282.pgm')  # Örnek dosya ismi

# Görüntüyü oku
gorsel = cv2.imread(pgm_dosya_yolu, cv2.IMREAD_GRAYSCALE)

# Görselleştir
plt.imshow(gorsel, cmap='gray')
plt.title('Örnek PGM Görseli')
plt.axis('off')  # Eksenleri kaldır
plt.show()

# Görselleri yükleyip normalleştirme ve boyutlandırma işlemi
def gorsel_yukle_ve_isle(dosya_yolu, hedef_boyut=(128, 128)):
    # Görseli oku
    gorsel = cv2.imread(dosya_yolu, cv2.IMREAD_GRAYSCALE)

    # Boyutlandır
    gorsel = cv2.resize(gorsel, hedef_boyut)

    # Normalizasyon (0 ile 1 arasına çekme)
    gorsel = gorsel / 255.0

    return gorsel

# Örnek görselleri işleyelim
gorsel1 = gorsel_yukle_ve_isle(os.path.join(hedef_klasor, 'mdb282.pgm'))
gorsel2 = gorsel_yukle_ve_isle(os.path.join(hedef_klasor, 'mdb027.pgm'))

# Görselleri görselleştir
fig, axs = plt.subplots(1, 2, figsize=(10, 5))
axs[0].imshow(gorsel1, cmap='gray')
axs[0].set_title('Görsel 1')
axs[0].axis('off')

axs[1].imshow(gorsel2, cmap='gray')
axs[1].set_title('Görsel 2')
axs[1].axis('off')

plt.show()

# Info.txt dosyasını oku
info_dosya_yolu = "mias_data/Info.txt"

# Dosyayı aç ve içeriği yazdır
with open(info_dosya_yolu, 'r') as file:
    dosya_icerik = file.read()
    print(dosya_icerik)


###dipnot
#REFNUM: Görüntünün benzersiz kimlik numarası (örneğin, mdb001, mdb002, vb.).
#BG: Görüntü arka plan bilgisi.
#CLASS: Görüntü türü (örneğin, CIRC: yuvarlak, SPIC: dikenli, vb.).
#SEVERITY: Görüntüdeki kanserin seviyesi (B: Benign – iyi huylu, M: Malignant – kötü huylu).
#X, Y: Bölgedeki koordinatlar.
#RADIUS: Bölgenin çapı veya yarıçapı.
#INFO.txt dosyasındaki SEVERITY sütununu kullanarak her görsel için bir etiket oluşturur.
#Bu etiketler, kanserin iyi huylu (B) veya kötü huylu (M) olduğuna göre belirlenir.

import pandas as pd

# Info.txt dosyasını DataFrame olarak yükle
info_df = pd.read_csv(info_dosya_yolu, delim_whitespace=True, header=None,
                       names=['REFNUM', 'BG', 'CLASS', 'SEVERITY', 'X', 'Y', 'RADIUS'])

# SEVERITY sütununa göre etiket oluştur
info_df['LABEL'] = info_df['SEVERITY'].map({'B': 0, 'M': 1})

# İlk birkaç satırın çıktısı
print(info_df.head())

###dipnot
#sep='\s+': Bu, birden fazla boşluğu ayırıcı olarak kabul eder ve veriyi düzgün şekilde okur.
#dropna(): NaN (eksik) değerlere sahip satırları veri kümesinden çıkarır.
#header=0: İlk satırdaki sütun başlıklarını alır. Bu şekilde, eksik verileri temizleyip doğru şekilde etiketleme işlemi yapabiliriz.

import pandas as pd

# Info.txt dosyasını düzgün şekilde yükle
info_df = pd.read_csv(info_dosya_yolu, sep='\s+', header=0, names=['REFNUM', 'BG', 'CLASS', 'SEVERITY', 'X', 'Y', 'RADIUS'])

# NaN değerlerini içeren satırları kaldır
info_df = info_df.dropna()

# SEVERITY sütununa göre etiket oluştur
info_df['LABEL'] = info_df['SEVERITY'].map({'B': 0, 'M': 1})

# İlk birkaç satırın çıktısı
print(info_df.head())

#Artık eksik veriler (NaN) kaldırıldı ve her satırda LABEL sütunu doğru şekilde yer alıyor.
#Etiketler, SEVERITY sütunundaki B ve M değerlerine göre 0 (iyi) ve 1 (kötü) olarak atanmış.

from google.colab import files
uploaded = files.upload()

import zipfile
import os

# ZIP dosyasının adı (doğru dosya adını buraya yazın)
zip_dosya_yolu = 'mias_mamography.zip'

# ZIP dosyasını açma
with zipfile.ZipFile(zip_dosya_yolu, 'r') as zip_ref:
    zip_ref.extractall('/content')  # Dosyaları /content klasörüne çıkartıyoruz

### ??????

import pandas as pd

# Dosya yolunu doğru yazdığınızdan emin olun
df = pd.read_csv('/content/mias_mamography.csv', delim_whitespace=True)

### ??????

#--- eksik kısımlar tamamlanmalı

### ??????

import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# CSV dosyasını tekrar yükle
# Dosya yolunu doğru belirttiğinizden emin olun
dosya_yolu = 'mias_mamography.csv'  # Dosyanın doğru yolu

# Dosyayı yükleyelim
df = pd.read_csv(dosya_yolu, delim_whitespace=True)

# Sayısal sütunları seçelim
sayi_sutunlar = ['X', 'Y', 'RADIUS']

# Min-Max ölçeklendirme
scaler = MinMaxScaler()
df[sayi_sutunlar] = scaler.fit_transform(df[sayi_sutunlar])

# Normalizasyon sonrasında veriyi kontrol edelim
print(df.head())
from sklearn.preprocessing import MinMaxScaler

# Sadece sayısal sütunları seçelim
sayi_sutunlar = ['X', 'Y', 'RADIUS']

# Min-Max ölçeklendirme
scaler = MinMaxScaler()
df[sayi_sutunlar] = scaler.fit_transform(df[sayi_sutunlar])

# Normalizasyon sonrasında veriyi kontrol edelim
print(df.head())

### ???????

#      MODEL
import tensorflow as tf
from tensorflow.keras import layers, models

# Modelin oluşturulması
model = models.Sequential([
    # 1. Konvolüsyonel Katman
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    layers.MaxPooling2D((2, 2)),

    # 2. Konvolüsyonel Katman
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    # 3. Konvolüsyonel Katman
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2
    # Tam Bağlantılı (Fully Connected) Katmanlar
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.5),  # Aşırı öğrenmeyi (overfitting) önlemek için

    # Çıkış Katmanı (Softmax ile 3 sınıflı çıktı)
    layers.Dense(3, activation='softmax')
])

# Modelin derlenmesi
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Modelin özetini görüntüleme
model.summary()

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Veri ön işleme ve artırma (Data Augmentation)
train_datagen = ImageDataGenerator(
    rescale=1./255,      # Piksel değerlerini 0-1 arasına ölçekleme
    rotation_range=20,   # Resimleri döndürme
    width_shift_range=0.2,  # Genişlik kaydırma
    height_shift_range=0.2, # Yükseklik kaydırma
    shear_range=0.2,    # Şeffaflık dönüşümü
    zoom_range=0.2,     # Yakınlaştırma
    horizontal_flip=True, # Yatay çevirme
    fill_mode='nearest') # Boşlukları doldurma

test_datagen = ImageDataGenerator(rescale=1./255)  # Test verisi için sadece ölçekleme

# Eğitim ve test verilerinin yüklenmesi
train_generator = train_datagen.flow_from_directory(
    'breast_ultrasound/train',  # Eğitim veri yolu
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical')  # Çok sınıflı sınıflandırma için

validation_generator = test_datagen.flow_from_directory(
    'breast_ultrasound/test',  # Test veri yolu
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical')
history = model.fit(
    train_generator,
    epochs=10,
    validation_data=validation_generator)
# Modelin test verisi üzerinde değerlendirilmesi
test_loss, test_accuracy = model.evaluate(validation_generator)
print(f"Test Doğruluğu: {test_accuracy:.4f}")
print(f"Test Kaybı (Loss): {test_loss:.4f}")

from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

# Test veri kümesi için tahminler al
Y_pred = model.predict(validation_generator)
y_pred = np.argmax(Y_pred, axis=1)

# Gerçek etiketleri al
true_classes = validation_generator.classes
class_labels = list(validation_generator.class_indices.keys())

# Sınıflandırma raporunu yazdır
print(classification_report(true_classes, y_pred, target_names=class_labels))

# Karışıklık matrisini göster
print(confusion_matrix(true_classes, y_pred))




#eklemeler yap!






