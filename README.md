# 🌾 Klasifikasi Penyakit Daun Padi Menggunakan Arsitektur MobileNetV3-ECA

Proyek ini mengembangkan model deep learning untuk mengklasifikasikan penyakit daun padi menggunakan arsitektur **MobileNetV3** yang diintegrasikan dengan modul **Efficient Channel Attention (ECA)**. Model dilatih untuk mengenali 10 kelas kondisi daun padi, mencakup kondisi sehat maupun berbagai jenis penyakit.

---

## Daftar Isi

- [Latar Belakang](#latar-belakang)
- [Dataset](#dataset)
- [Arsitektur Model](#arsitektur-model)
- [Hasil Evaluasi](#hasil-evaluasi)
- [Cara Penggunaan](#cara-penggunaan)
- [Struktur Proyek](#struktur-proyek)
- [Teknologi](#teknologi)

---

## Latar Belakang

Penyakit tanaman padi menjadi salah satu ancaman utama produksi pertanian. Deteksi dini secara manual membutuhkan keahlian khusus dan memakan waktu. Proyek ini hadir sebagai solusi berbasis computer vision yang membantu petani mengidentifikasi penyakit daun padi secara otomatis dari foto, sehingga penanganan dapat dilakukan lebih cepat dan tepat.

---

## Dataset

- **Sumber:** Kaggle (Rice Leaf Disease Dataset)
- **Total gambar:** 3.345 citra daun padi (setelah pembersihan duplikat)
- **Ukuran input:** 224 × 224 piksel
- **Pembagian data:**

| Set | Proporsi | Jumlah |
|-----|----------|--------|
| Training | 80% | ~2.676 |
| Validation | 10% | ~335 |
| Testing | 10% | ~334 |

**10 Kelas Penyakit:**

| No | Kelas |
|----|-------|
| 1 | Healthy |
| 2 | Rice Hispa |
| 3 | Tungro |
| 4 | Neck Blast |
| 5 | Leaf Blast |
| 6 | Leaf Scald |
| 7 | Brown Spot |
| 8 | Sheath Blight |
| 9 | Narrow Brown Spot |
| 10 | Bacterial Leaf Blight |

### Augmentasi Data

Augmentasi diterapkan pada data latih untuk meningkatkan keragaman dataset, meliputi:
- Rotasi acak (±30°)
- Horizontal & vertical flip
- Random shift (warp)
- Random blur
- Penyesuaian brightness
- Shear transformation

---

## Arsitektur Model

Proyek ini membandingkan 4 arsitektur:

| Model | Deskripsi |
|-------|-----------|
| MobileNetV3Small Original | Baseline arsitektur kecil |
| MobileNetV3Large Original | Baseline arsitektur besar |
| **MobileNetV3Small-ECA** | MobileNetV3Small + modul ECA |
| **MobileNetV3Large-ECA** | MobileNetV3Large + modul ECA ⭐ Terbaik |

**Modul ECA (Efficient Channel Attention)** ditambahkan setelah backbone untuk meningkatkan kemampuan model dalam memperhatikan fitur-fitur channel yang paling relevan, tanpa penambahan parameter yang signifikan.

**Hyperparameter terbaik:**
- Batch size: `32`
- Learning rate: `1e-4`
- Optimizer: `Adam`
- Early stopping: 5 epoch patience

---

## Hasil Evaluasi

| Model | Accuracy | F1-Score (Weighted) | Epoch Berhenti |
|-------|----------|---------------------|----------------|
| MobileNetV3Small Original | 0.8955 | 0,8983 | 78 |
| MobileNetV3Large Original | 0,9045 | 0,9023 | 64 |
| MobileNetV3Small-ECA | 0.9313 | 0.9333 | 38 |
| **MobileNetV3Large-ECA** | **0.9373** | **0.9383** | **34** |

**Temuan utama:**
- Penambahan modul ECA meningkatkan performa pada kedua arsitektur
- MobileNetV3Large-ECA memberikan hasil terbaik dan paling stabil
- Kelas `neck_blast`, `bacterial_leaf_blight`, dan `tungro` mencapai F1-score sempurna (1.00) pada model Large-ECA

---

## Cara Penggunaan

### Training Model
```python
# Pemodelan
jupyter notebook Image_Classification_with_MobileNetV3_ECA_Architecture.ipynb
```

### Inferensi (Prediksi Gambar Baru)
```python
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Load model
model = tf.keras.models.load_model('model_mobilenetv3large_eca.h5')

# Prediksi
img = load_img('daun_padi.jpg', target_size=(224, 224))
img_array = img_to_array(img) / 127.5 - 1.0  # normalisasi ke [-1, 1]
img_array = np.expand_dims(img_array, axis=0)

pred = model.predict(img_array)
class_names = ['bacterial_leaf_blight', 'brown_spot', 'healthy', 'leaf_blast',
               'leaf_scald', 'narrow_brown_spot', 'neck_blast', 'rice_hispa',
               'sheath_blight', 'tungro']

print(f"Prediksi: {class_names[np.argmax(pred)]}")
print(f"Confidence: {np.max(pred)*100:.2f}%")
```

---

## Struktur Proyek

```
rice-leaf-disease-mobilenetv3-eca/
│
├── Image_Classification_with_MobileNetV3_ECA_Architecture.ipynb
├── requirements.txt
├── README.md
│
├── Dataset-Final/
│   ├── train/
│   │   ├── healthy/
│   │   ├── leaf_blast/
│   │   └── ... (10 kelas)
│   ├── val/
│   └── test/
│
└── models/
    ├── model_mobilenetv3small_eca.h5
    └── model_mobilenetv3large_eca.h5
```

---

## Teknologi

![Python](https://img.shields.io/badge/Python-3.9+-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.10+-orange)
![Keras](https://img.shields.io/badge/Keras-deep%20learning-red)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-yellow)
![Kaggle](https://img.shields.io/badge/Dataset-Kaggle-20BEFF)

---

## 👤 Author

**Salma Zhafira Muchtar**
📧 zhafirasalmam@gmail.com

---

## Lisensi

Proyek ini dibuat untuk keperluan penelitian dan edukasi.
