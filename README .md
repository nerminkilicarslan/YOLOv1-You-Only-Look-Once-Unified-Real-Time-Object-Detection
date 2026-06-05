# YOLOv1: You Only Look Once — Unified, Real-Time Object Detection

> **Paper:** Redmon, J., Divvala, S., Girshick, R., & Farhadi, A. (2016).
> *You Only Look Once: Unified, Real-Time Object Detection.* CVPR 2016.
> [arXiv:1506.02640](https://arxiv.org/abs/1506.02640)

---

## About This Notebook

This notebook provides an in-depth implementation of the YOLOv1 paper, translating every core concept into Python code from scratch. It serves as a comprehensive reference from theory to practice — from mathematical formulations to working visualizations.

---

## Contents

| Section | Topic |
|---------|-------|
| 1 | Motivation & comparison with R-CNN family |
| 2 | Grid-based detection — S×S cell logic |
| 3 | Darknet architecture (24 Conv + 2 FC) |
| 4 | Loss function (5 components, full derivation) |
| 5 | Bounding box encoding & IoU calculation |
| 6 | Non-Maximum Suppression (NMS) |
| 7 | Prediction decoding |
| 8 | End-to-end pipeline simulation |
| 9 | Performance analysis & limitations |
| 10 | Summary & next steps |

---

## Key Concepts

### YOLO's Revolutionary Idea
YOLO frames object detection as a **single regression problem**:

```
Image → CNN → S×S×(B×5+C) Tensor → Bounding Boxes + Class Probabilities
```

### Output Tensor
```
7 × 7 × 30  =  S × S × (B×5 + C)
                        └─ 2×5 + 20 = 30
                           ↑      ↑
                        B boxes  C classes (VOC)
```

### Loss Function (5 Components)
```
L = lambda_coord · (center coordinate loss)
  + lambda_coord · (sqrt dimension loss)
  + (confidence loss — object present)
  + lambda_noobj · (confidence loss — no object)
  + (class probability loss)

lambda_coord = 5,  lambda_noobj = 0.5
```

### Performance on Pascal VOC 2007
| Model | mAP | FPS |
|-------|-----|-----|
| Faster R-CNN | 73.2% | 7 |
| **YOLOv1** | **63.4%** | **45** |
| Fast-YOLO | 52.7% | 155 |

---

## Setup & Usage

### Requirements
```bash
pip install numpy matplotlib
```

### Run
```bash
jupyter notebook 16-YOLOv1.ipynb
```

All cells should be run sequentially. No external dataset or GPU required — pure NumPy simulation.

---

## File Structure

```
Deep-Learning-Paper-Implementations/
├── 15-Transformers.ipynb        # ViT: An Image is Worth 16x16 Words
├── 16-YOLOv1.ipynb              # <- This notebook
└── README.md
```

---

## YOLO Series Roadmap

```
YOLOv1 (CVPR 2016)        <- You are here
    |
YOLOv2 / YOLO9000 (2017)  — Anchor boxes, BatchNorm, 9000 classes
    |
YOLOv3 (2018)             — Multi-scale detection, Darknet-53, FPN-like
    |
YOLOv4 (2020)             — CSP, PANet, Mosaic augmentation
    |
YOLOv8+ (2023+)           — Anchor-free, transformer heads
```

---

## Reference

```bibtex
@inproceedings{redmon2016yolo,
  title     = {You Only Look Once: Unified, Real-Time Object Detection},
  author    = {Redmon, Joseph and Divvala, Santosh and Girshick, Ross and Farhadi, Ali},
  booktitle = {Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2016},
  pages     = {779--788}
}
```

---
---

# YOLOv1: You Only Look Once — Birlesik, Gercek Zamanli Nesne Tespiti

> **Makale:** Redmon, J., Divvala, S., Girshick, R., & Farhadi, A. (2016).
> *You Only Look Once: Unified, Real-Time Object Detection.* CVPR 2016.
> [arXiv:1506.02640](https://arxiv.org/abs/1506.02640)

---

## Bu Notebook Hakkinda

Bu notebook, YOLOv1 makalesini derinlemesine inceleyerek tum temel konseptleri sifirdan Python koduyla implemente eder. Teoriden pratige, formulden gorsellestirmeye kapsamli bir referans kaynagindan olusmaktadir.

---

## Icerik

| Bolum | Konu |
|-------|------|
| 1 | Motivasyon & R-CNN ailesiyle karsilastirma |
| 2 | Grid tabanli tespit — S×S hucre mantigi |
| 3 | Darknet mimarisi (24 Conv + 2 FC) |
| 4 | Loss fonksiyonu (5 bilesen, tam turetme) |
| 5 | Bounding box kodlamasi & IoU hesabi |
| 6 | Non-Maximum Suppression (NMS) |
| 7 | Tahmin decode etme |
| 8 | Uctan uca pipeline simulasyonu |
| 9 | Performans analizi & kisitlamalar |
| 10 | Ozet & sonraki adimlar |

---

## Temel Kavramlar

### YOLO'nun Devrimsel Fikri
YOLO, nesne tespitini **tek bir regresyon problemi** olarak tanimlar:

```
Goruntu → CNN → S×S×(B×5+C) Tensoru → Bounding Box + Sinif Olasiliklari
```

### Cikti Tensoru
```
7 × 7 × 30  =  S × S × (B×5 + C)
                        └─ 2×5 + 20 = 30
                           ↑      ↑
                       B kutular  C sinif (VOC)
```

### Loss Fonksiyonu (5 Bilesen)
```
L = lambda_coord · (merkez koordinat hatasi)
  + lambda_coord · (sqrt boyut hatasi)
  + (confidence hatasi — nesne VAR)
  + lambda_noobj · (confidence hatasi — nesne YOK)
  + (sinif olasilik hatasi)

lambda_coord = 5,  lambda_noobj = 0.5
```

### Performans (Pascal VOC 2007)
| Model | mAP | FPS |
|-------|-----|-----|
| Faster R-CNN | 73.2% | 7 |
| **YOLOv1** | **63.4%** | **45** |
| Fast-YOLO | 52.7% | 155 |

---

## Kurulum & Calistirma

### Gereksinimler
```bash
pip install numpy matplotlib
```

### Calistirma
```bash
jupyter notebook 16-YOLOv1.ipynb
```

Tum hucreler sirayla calistirilmalidir. Harici dataset veya GPU gerektirmez — saf NumPy simulasyonu.

---

## Dosya Yapisi

```
Deep-Learning-Paper-Implementations/
├── 15-Transformers.ipynb        # ViT: An Image is Worth 16x16 Words
├── 16-YOLOv1.ipynb              # <- Bu notebook
└── README.md
```

---

## YOLO Serisi Yol Haritasi

```
YOLOv1 (CVPR 2016)        <- Buradasiniz
    |
YOLOv2 / YOLO9000 (2017)  — Anchor box, BatchNorm, 9000 sinif
    |
YOLOv3 (2018)             — Multi-scale, Darknet-53, FPN benzeri
    |
YOLOv4 (2020)             — CSP, PANet, Mosaic augmentation
    |
YOLOv8+ (2023+)           — Anchor-free, transformer basliklar
```

---

## Referans

```bibtex
@inproceedings{redmon2016yolo,
  title     = {You Only Look Once: Unified, Real-Time Object Detection},
  author    = {Redmon, Joseph and Divvala, Santosh and Girshick, Ross and Farhadi, Ali},
  booktitle = {Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2016},
  pages     = {779--788}
}
```

---

*Deep Learning Paper Implementations serisi — Makale okuyup koda donusturme pratigi*
