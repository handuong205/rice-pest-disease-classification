# Nhận diện bệnh và sâu hại lúa bằng Deep Learning

Đồ án môn **MDS301**: xây dựng hệ thống phân loại ảnh **10 lớp bệnh và sâu hại trên cây lúa** bằng Transfer Learning với các mạng CNN pretrained trên ImageNet (ResNet50, EfficientNet-B3, MobileNetV3-Large). Sau đó so sánh các mô hình theo Accuracy, Precision, Recall, F1, ROC-AUC và tốc độ suy luận.

Phương pháp nghiên cứu chi tiết nằm trong [Methodology.md](docs/Methodology.md).

---

## Mục lục

1. [Bài toán](#1-bài-toán)
2. [Dữ liệu](#2-dữ-liệu)
3. [Cấu trúc thư mục](#3-cấu-trúc-thư-mục)
4. [Pipeline](#4-pipeline)
5. [Mô hình và cấu hình huấn luyện](#5-mô-hình-và-cấu-hình-huấn-luyện)
6. [Kết quả](#6-kết-quả)
7. [Cài đặt và chạy lại](#7-cài-đặt-và-chạy-lại)
8. [Sử dụng mô hình đã huấn luyện](#8-sử-dụng-mô-hình-đã-huấn-luyện)
9. [Hạn chế và hướng phát triển](#9-hạn-chế-và-hướng-phát-triển)

---

## 1. Bài toán

Phần lớn các nghiên cứu trước đây chỉ phân loại **bệnh lá lúa** trên một bộ dữ liệu công khai duy nhất. Đồ án này gộp cả **bệnh** và **sâu hại** từ nhiều nguồn vào một bài toán phân loại đa lớp:

| Nhóm | Lớp |
| --- | --- |
| Bệnh | `BacterialBlight` (bạc lá), `Blast` (đạo ôn), `BrownSpot` (đốm nâu), `SheathBlight` (khô vằn) |
| Sâu hại / triệu chứng do sâu | `DeadHeart` (nõn héo), `LeafDamage` (lá bị hại), `Planthopper` (rầy), `StemBorer` (sâu đục thân), `OtherRicePests` (sâu hại khác) |
| Bình thường | `Healthy` |

---

## 2. Dữ liệu

### 2.1 Nguồn

| Nguồn | Số ảnh | Các lớp lấy ra |
| --- | ---: | --- |
| [IP102](https://github.com/xpwu95/IP102) (thư mục `IP02`) | 6,841 | LeafDamage, OtherRicePests, Planthopper, StemBorer |
| Kaggle Rice Disease | 3,201 | BacterialBlight, Blast, BrownSpot, Healthy, SheathBlight |
| Mendeley Rice Disease | 3,224 | BacterialBlight, BrownSpot |
| Paddy Doctor | 6,388 | BacterialBlight, Blast, BrownSpot, DeadHeart, Healthy |
| **Tổng sau khi gộp** | **19,654** | 10 lớp |


### 2.2 Làm sạch

- **Ảnh lỗi:** mở và `verify()` toàn bộ ảnh bằng PIL. Không có ảnh lỗi.
- **Ảnh trùng:** dùng `imagehash.average_hash` và tìm được **2,995 ảnh trùng**. Các ảnh này được chuyển sang thư mục cách ly `Duplicated_Trash`, không xoá hẳn.
- Còn lại **16,659 ảnh** sạch. Danh sách đường dẫn và nhãn được lưu trong [metadata.csv](data/metadata.csv).

### 2.3 Chia tập

Chia phân tầng (stratified) theo nhãn với `random_state=42`, tỉ lệ 70 / 15 / 15:

| Lớp | Train | Val | Test |
| --- | ---: | ---: | ---: |
| OtherRicePests | 1,902 | 407 | 408 |
| Blast | 1,583 | 339 | 339 |
| Healthy | 1,576 | 338 | 337 |
| BrownSpot | 1,534 | 329 | 329 |
| Planthopper | 1,303 | 279 | 280 |
| BacterialBlight | 1,140 | 245 | 244 |
| DeadHeart | 942 | 202 | 202 |
| StemBorer | 631 | 136 | 135 |
| LeafDamage | 612 | 131 | 131 |
| SheathBlight | 438 | 93 | 94 |
| **Tổng** | **11,661** | **2,499** | **2,499** |

### 2.4 Mất cân bằng lớp

Dữ liệu lệch lớp khá rõ: `OtherRicePests` có số ảnh gấp khoảng 4.3 lần `SheathBlight`. Vì vậy loss được đánh trọng số theo lớp (`compute_class_weight("balanced")`), và trọng số được lưu trong [class_weights.npy](data/class_weights.npy):

| Lớp | Trọng số |
| --- | ---: |
| BacterialBlight | 1.0227 |
| Blast | 0.7368 |
| BrownSpot | 0.7600 |
| DeadHeart | 1.2377 |
| Healthy | 0.7401 |
| LeafDamage | 1.9061 |
| OtherRicePests | 0.6131 |
| Planthopper | 0.8947 |
| SheathBlight | 2.6654 |
| StemBorer | 1.8469 |

---

## 3. Cấu trúc thư mục

```text
project/
├── README.md
├── requirements.txt
├── .gitignore
│
├── docs/
│   └── Methodology.md                  # Phương pháp nghiên cứu (chương 3 báo cáo)
│
├── notebooks/                          # Chạy theo thứ tự số
│   ├── 01_data_process.ipynb           # Thống kê nguồn, gộp 4 bộ dữ liệu thành 10 lớp
│   ├── 02_data_cleaning.ipynb          # Kiểm tra ảnh lỗi/trùng, tạo metadata.csv, class weight
│   ├── 03_data_preparation.ipynb       # Chia train/val/test, transform & DataLoader
│   ├── 04_train_resnet50.ipynb         # Huấn luyện + đánh giá ResNet50
│   ├── 05_train_efficientnet_b3.ipynb  # Huấn luyện + đánh giá EfficientNet-B3
│   ├── 06_train_mobilenetv3.ipynb      # Huấn luyện + đánh giá MobileNetV3-Large
│   └── 07_model_comparison.ipynb       # So sánh 3 mô hình (metric, ROC, tốc độ)
│
├── data/
│   ├── metadata.csv                    # image_path,label (16,659 dòng)
│   └── class_weights.npy               # Trọng số lớp cho CrossEntropyLoss
│
├── models/
│   ├── best_resnet50.pth               # Checkpoint tốt nhất (~94 MB)
│   ├── best_efficientnet_b3.pth        # Checkpoint tốt nhất (~43 MB)
│   └── best_mobilenetv3.pth            # Checkpoint tốt nhất (~17 MB)
│
└── results/
    ├── class_distribution.png          # Biểu đồ phân bố lớp
    └── resnet50_test_predictions.csv   # Dự đoán + xác suất từng lớp của ResNet50 trên test set
```

> Các notebook dùng đường dẫn tương đối kiểu `../data/...`, `../models/...`, nên cần chạy với thư mục làm việc là `notebooks/`. Đây là mặc định của Jupyter và VS Code.

---

## 4. Pipeline

```text
4 bộ dữ liệu gốc (RiceDataset/)
        │  01_data_process.ipynb      gộp & chuẩn hoá nhãn
        ▼
RiceDataset_Final/  (19,654 ảnh, 10 lớp)
        │  02_data_cleaning.ipynb     loại ảnh lỗi/trùng → data/metadata.csv, data/class_weights.npy
        ▼
16,659 ảnh sạch
        │  03_data_preparation.ipynb  stratified split 70/15/15
        ▼
RiceDataset_Split/{train,val,test}/<class>/
        │  04 / 05 / 06_train_*.ipynb
        ▼
models/best_*.pth
        │  07_model_comparison.ipynb
        ▼
Bảng so sánh, đường ROC, benchmark tốc độ
```

---

## 5. Mô hình và cấu hình huấn luyện

### 5.1 Chiến lược chung (cả 3 mô hình)

- **Transfer learning** từ trọng số ImageNet của `torchvision`. Lớp cuối được thay bằng `Linear(…, 10)`.
- **Huấn luyện 2 giai đoạn:**
  1. *Warm-up* (5 epoch đầu): đóng băng backbone và giữ BatchNorm ở chế độ `eval`. Chỉ huấn luyện classifier với `AdamW(lr=3e-4)`.
  2. *Fine-tune*: mở băng toàn bộ mạng và dùng learning rate phân tầng: backbone `1e-5`, classifier `1e-4`.
- **Loss:** `CrossEntropyLoss` có trọng số lớp.
- **Regularization:** `weight_decay=1e-3`, gradient clipping `max_norm=1.0`.
- **Scheduler:** `ReduceLROnPlateau` (factor 0.5, patience 2, min_lr 1e-7) theo val loss.
- **Early stopping:** dừng sau 5 epoch không cải thiện val loss (`min_delta=1e-3`), tối đa 40 epoch. Checkpoint được lưu theo **val loss tốt nhất**.
- **Chuẩn hoá:** mean/std của ImageNet.

### 5.2 Khác biệt giữa các mô hình

| | ResNet50 | EfficientNet-B3 | MobileNetV3-Large |
| --- | --- | --- | --- |
| Kích thước ảnh | 224 × 224 | 300 × 300 | 300 × 300 |
| Batch size | 64 | 16 | 32 |
| Augmentation (train) | HFlip, Rotation 25°, Affine (translate 0.1), ColorJitter 0.3 | HFlip, Rotation 10° | HFlip, Rotation 15°, Affine (translate 0.05), ColorJitter 0.2 |
| Mixed precision | Không | Có (bfloat16, channels_last) | Không |
| Số epoch thực tế | 37 | 28 | 21 |
| Best val loss / val acc | 0.4158 / 0.8631 | 0.3341 / 0.8820 | 0.4548 / 0.8327 |

Phần cứng huấn luyện: **NVIDIA GeForce RTX 4050 Laptop GPU (6 GB VRAM)**.

---

## 6. Kết quả

Tất cả kết quả dưới đây được đo trên **cùng một test set gồm 2,499 ảnh** (xem [07_model_comparison.ipynb](notebooks/07_model_comparison.ipynb)).

### 6.1 Tổng quan

| Mô hình | Input | Accuracy | Precision (macro) | Recall (macro) | F1 (macro) | ROC-AUC (macro OVR) | ROC-AUC (weighted OVR) |
| --- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **EfficientNet-B3** | 300 | **0.8760** | 0.8690 | 0.8748 | 0.8712 | **0.9892** | **0.9887** |
| ResNet50 | 224 | 0.8756 | **0.8700** | **0.8755** | **0.8723** | 0.9874 | 0.9875 |
| MobileNetV3-Large | 300 | 0.8295 | 0.8198 | 0.8379 | 0.8254 | 0.9838 | 0.9833 |

### 6.2 Tốc độ suy luận

Đo với batch = 1 trên GPU, 20 lần warm-up và 100 lần đo:

| Mô hình | Thời gian (ms/ảnh) | FPS | Checkpoint |
| --- | ---: | ---: | ---: |
| ResNet50 | **8.36** | **119.7** | 94 MB |
| MobileNetV3-Large | 10.45 | 95.7 | **17 MB** |
| EfficientNet-B3 | 19.79 | 50.5 | 43 MB |

### 6.3 ROC-AUC theo từng lớp

| Lớp | EfficientNet-B3 | ResNet50 | MobileNetV3 |
| --- | :---: | :---: | :---: |
| BacterialBlight | 0.9991 | 0.9970 | 0.9939 |
| Blast | 0.9986 | 0.9978 | 0.9916 |
| BrownSpot | 0.9994 | 0.9987 | 0.9949 |
| DeadHeart | 0.9995 | 0.9999 | 0.9997 |
| Healthy | 0.9997 | 0.9992 | 0.9984 |
| LeafDamage | 0.9742 | 0.9682 | 0.9646 |
| OtherRicePests | 0.9656 | 0.9668 | 0.9602 |
| Planthopper | 0.9737 | 0.9716 | 0.9625 |
| SheathBlight | 0.9999 | 1.0000 | 0.9996 |
| StemBorer | 0.9827 | 0.9754 | 0.9723 |

### 6.4 F1-score theo từng lớp

| Lớp | EfficientNet-B3 | ResNet50 | MobileNetV3 |
| --- | :---: | :---: | :---: |
| BacterialBlight | 0.96 | 0.94 | 0.90 |
| Blast | 0.95 | 0.95 | 0.89 |
| BrownSpot | 0.97 | 0.96 | 0.92 |
| DeadHeart | 0.99 | 0.98 | 0.97 |
| Healthy | 0.98 | 0.96 | 0.95 |
| LeafDamage | 0.67 | 0.68 | 0.60 |
| OtherRicePests | 0.73 | 0.74 | 0.68 |
| Planthopper | 0.73 | 0.75 | 0.68 |
| SheathBlight | 0.98 | 0.98 | 0.95 |
| StemBorer | 0.75 | 0.77 | 0.71 |

### 6.5 Nhận xét

- **EfficientNet-B3 và ResNet50 gần như ngang nhau.** Accuracy chênh 0.04 điểm phần trăm và ROC-AUC macro chênh 0.0018. EfficientNet-B3 nhỉnh hơn ở ROC-AUC, ResNet50 nhỉnh hơn ở F1 macro.
- **ResNet50 cho tỉ lệ hiệu năng/tốc độ tốt nhất.** Nó suy luận nhanh gấp khoảng 2.4 lần EfficientNet-B3 mà độ chính xác gần như không đổi.
- **MobileNetV3 nhẹ nhất (17 MB)** nhưng thấp hơn khoảng 4.6 điểm accuracy. Mô hình này phù hợp khi cần triển khai trên thiết bị biên hoặc di động.
- **Nhóm bệnh lá** (BacterialBlight, Blast, BrownSpot, SheathBlight, Healthy) và DeadHeart đạt F1 ≥ 0.94 trên cả hai mô hình mạnh nhất.
- **Nhóm sâu hại từ IP102** (LeafDamage, OtherRicePests, Planthopper, StemBorer) là điểm yếu chung: F1 chỉ đạt 0.60–0.77. Nguyên nhân có thể là ảnh côn trùng đa dạng về loài, nền và tỉ lệ khung hình, và lớp `OtherRicePests` vốn là lớp "gom" nhiều loài khác nhau.

---

## 7. Cài đặt và chạy lại

### 7.1 Môi trường

- Python 3.10+ (đã dùng 3.12)
- GPU NVIDIA có CUDA (khuyến nghị; EfficientNet-B3 dùng khoảng 2.5 GB VRAM với batch 16)

```bash
# Nên cài PyTorch trước, theo đúng phiên bản CUDA của máy: https://pytorch.org/get-started/locally/
pip install -r requirements.txt
```

### 7.2 Chuẩn bị dữ liệu

1. Tải 4 bộ dữ liệu ở [mục 2.1](#21-nguồn) và sắp xếp theo cấu trúc `RiceDataset/<Nguồn>/<Lớp>/...`.
2. **Sửa đường dẫn:** các notebook đang hard-code `D:\KÌ 7\RiceDataset`, `D:\KÌ 7\RiceDataset_Final` và `D:\KÌ 7\RiceDataset_Split`. Đổi các đường dẫn này cho phù hợp với máy của bạn. File `metadata.csv` cũng lưu đường dẫn tuyệt đối nên cần tạo lại.
3. `07_model_comparison.ipynb` tự tìm `RiceDataset_Split`. Nếu không tìm thấy, đặt biến môi trường:

   ```bash
   # Linux/macOS
   export RICE_DATASET_SPLIT=/path/to/RiceDataset_Split
   # Windows PowerShell
   $env:RICE_DATASET_SPLIT = "D:\path\to\RiceDataset_Split"
   ```

### 7.3 Thứ tự chạy notebook

| Bước | Notebook | Đầu ra |
| :---: | --- | --- |
| 1 | `01_data_process.ipynb` (bỏ comment ô merge) | `RiceDataset_Final/` |
| 2 | `02_data_cleaning.ipynb` | `data/metadata.csv`, `data/class_weights.npy`, ảnh trùng được chuyển sang `Duplicated_Trash/` |
| 3 | `03_data_preparation.ipynb` | `RiceDataset_Split/{train,val,test}/` |
| 4 | `04_train_resnet50.ipynb`, `05_train_efficientnet_b3.ipynb`, `06_train_mobilenetv3.ipynb` | `models/best_*.pth` |
| 5 | `07_model_comparison.ipynb` | Bảng so sánh, đường ROC, benchmark |

> Nếu chỉ muốn **đánh giá lại** mà không huấn luyện, bỏ qua bước 4 và dùng các file `models/best_*.pth` có sẵn.

---

## 8. Sử dụng mô hình đã huấn luyện

Các file `.pth` chứa `state_dict`. Ví dụ dự đoán một ảnh bằng EfficientNet-B3:

```python
import torch
from PIL import Image
from torchvision import transforms
from torchvision.models import efficientnet_b3

CLASSES = [
    "BacterialBlight", "Blast", "BrownSpot", "DeadHeart", "Healthy",
    "LeafDamage", "OtherRicePests", "Planthopper", "SheathBlight", "StemBorer",
]

model = efficientnet_b3(weights=None)
model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, len(CLASSES))
model.load_state_dict(torch.load("models/best_efficientnet_b3.pth", map_location="cpu", weights_only=True))
model.eval()

transform = transforms.Compose([
    transforms.Resize((300, 300)),   # ResNet50 dùng 224
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

image = transform(Image.open("leaf.jpg").convert("RGB")).unsqueeze(0)
with torch.inference_mode():
    probs = torch.softmax(model(image), dim=1)[0]

top = probs.argmax().item()
print(f"{CLASSES[top]} ({probs[top]:.2%})")
```

Với các mô hình khác:

| Mô hình | Hàm khởi tạo | Thay lớp cuối | Input |
| --- | --- | --- | :---: |
| ResNet50 | `resnet50(weights=None)` | `model.fc = Linear(2048, 10)` | 224 |
| MobileNetV3 | `mobilenet_v3_large(weights=None)` | `model.classifier[3] = Linear(1280, 10)` | 300 |

---

## 9. Hạn chế và hướng phát triển

### Hạn chế hiện tại

- **Kết quả chưa thể tái lập trực tiếp:** đường dẫn dữ liệu được hard-code và ảnh không đi kèm repo.
- **Class weight được tính trên toàn bộ dữ liệu**, gồm cả val và test, thay vì chỉ trên tập train. Vì tỉ lệ lớp được giữ nguyên khi chia phân tầng nên ảnh hưởng không đáng kể, nhưng về nguyên tắc nên tính chỉ trên train.
- **Nguy cơ rò rỉ dữ liệu:** một số nguồn (ví dụ Kaggle, với các file `*_aug_*`) đã chứa sẵn ảnh augmentation. Phép khử trùng bằng `average_hash` chỉ loại được ảnh gần như giống hệt, nên các biến thể xoay/lật của cùng một ảnh gốc vẫn có thể rơi vào cả train và test, làm kết quả test cao hơn thực tế.
- Nhóm lớp sâu hại (đặc biệt `LeafDamage`, `OtherRicePests`) còn thấp.

### Kế hoạch tiếp theo (theo [Methodology.md](docs/Methodology.md))

- [ ] **Ensemble Learning:** Weighted Averaging, Stacking (meta-classifier) và Hybrid Ensemble kết hợp ResNet50 + EfficientNet + MobileNetV3.
- [ ] **Đánh giá độ bền vững:** thử với Gaussian noise, ảnh thiếu sáng, độ tương phản thấp và ảnh mờ.
- [ ] **Đánh giá khả năng tổng quát hoá** trên một bộ dữ liệu ngoài (chưa từng thấy).
- [ ] **Grad-CAM:** trực quan hoá vùng ảnh mà mô hình chú ý khi dự đoán.
- [ ] Tối ưu siêu tham số có hệ thống (learning rate, batch size, optimizer, weight decay).
