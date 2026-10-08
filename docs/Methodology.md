# 3. Phương pháp nghiên cứu (Methodology)

## 3.1 Tổng quan hệ thống

Nghiên cứu này đề xuất một hệ thống nhận diện tổng hợp bệnh và sâu hại lúa dựa trên Deep Learning và Ensemble Learning. Hệ thống được xây dựng nhằm khắc phục các hạn chế của các nghiên cứu trước đây, vốn chủ yếu tập trung vào phân loại bệnh lá lúa trên các bộ dữ liệu công khai và sử dụng các mô hình CNN đơn lẻ.

Quy trình nghiên cứu được chia thành bốn giai đoạn chính:

```text
Phase 1: Data Engineering
        ↓
Phase 2: Model Development & Optimization
        ↓
Phase 3: Comprehensive Evaluation
        ↓
Phase 4: Explainability & Analysis
```

---

# 3.2 Giai đoạn 1 – Xây dựng và xử lý dữ liệu (Data Engineering)

## 3.2.1 Thu thập dữ liệu (Dataset Collection)

Để xây dựng bộ dữ liệu đa dạng cho bài toán nhận diện sâu bệnh hại lúa, dữ liệu được thu thập từ nhiều nguồn công khai khác nhau, bao gồm:

- Kaggle Rice Disease Dataset
- Paddy Doctor Dataset
- Mendeley Rice Disease Dataset
- IP102 Pest Dataset

Các bộ dữ liệu này bao gồm cả ảnh bệnh lá lúa và ảnh sâu hại lúa được thu thập trong nhiều điều kiện môi trường khác nhau.

Sau quá trình hợp nhất dữ liệu và chuẩn hóa nhãn, bộ dữ liệu cuối cùng được chia thành 10 lớp:

| Nhãn |
| --- |
| BacterialBlight |
| Blast |
| BrownSpot |
| DeadHeart |
| Healthy |
| LeafDamage |
| OtherRicePests |
| Planthopper |
| SheathBlight |
| StemBorer |

Tổng số ảnh sau khi hợp nhất:

```text
19,654 ảnh
```

---

## 3.2.2 Làm sạch dữ liệu (Data Cleaning)

Quá trình làm sạch dữ liệu được thực hiện nhằm đảm bảo chất lượng ảnh đầu vào trước khi huấn luyện mô hình.

Các bước bao gồm:

- Loại bỏ ảnh bị lỗi (corrupted images)
- Loại bỏ ảnh không đọc được
- Chuẩn hóa định dạng ảnh
- Kiểm tra tính nhất quán của nhãn

Việc làm sạch dữ liệu giúp giảm nhiễu và nâng cao độ tin cậy của quá trình huấn luyện.

---

## 3.2.3 Tiền xử lý ảnh (Image Preprocessing)

### Thay đổi kích thước ảnh

Tất cả ảnh được chuẩn hóa về kích thước:

```text
224 × 224 pixels
```

để phù hợp với đầu vào của các mô hình CNN pretrained.

### Chuẩn hóa dữ liệu (Normalization)

Sử dụng bộ tham số chuẩn của ImageNet:

```python
mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]
```

---

## 3.2.4 Tăng cường dữ liệu (Data Augmentation)

Để giảm hiện tượng overfitting và tăng khả năng tổng quát hóa của mô hình, các kỹ thuật augmentation được áp dụng trên tập huấn luyện:

- Random Horizontal Flip
- Random Rotation
- Random Brightness Adjustment
- Random Contrast Adjustment
- Color Jitter

Các kỹ thuật này giúp mô phỏng các điều kiện thực tế như:

- Thay đổi góc chụp
- Điều kiện ánh sáng khác nhau
- Nền ảnh phức tạp
- Chất lượng ảnh không đồng nhất

---

## 3.2.5 Chia tập dữ liệu (Dataset Splitting)

Bộ dữ liệu được chia theo phương pháp Stratified Split nhằm đảm bảo tỷ lệ phân bố lớp được giữ nguyên giữa các tập dữ liệu.

| Tập dữ liệu | Tỷ lệ |
| --- | --- |
| Training Set | 70% |
| Validation Set | 15% |
| Test Set | 15% |

---

# 3.3 Giai đoạn 2 – Phát triển và tối ưu mô hình

## 3.3.1 Transfer Learning

Thay vì huấn luyện từ đầu, nghiên cứu sử dụng kỹ thuật Transfer Learning với các trọng số pretrained từ ImageNet.

Ba kiến trúc CNN phổ biến được lựa chọn làm mô hình cơ sở (Base Models):

### ResNet50

ResNet50 sử dụng cơ chế Residual Learning nhằm giải quyết hiện tượng mất mát gradient (Vanishing Gradient) khi huấn luyện các mạng sâu.

### EfficientNet-B0

EfficientNet-B0 sử dụng phương pháp Compound Scaling để đồng thời mở rộng:

- Chiều sâu mạng (Depth)
- Chiều rộng mạng (Width)
- Độ phân giải đầu vào (Resolution)

giúp đạt hiệu năng cao với số lượng tham số thấp.

### MobileNetV3

MobileNetV3 được thiết kế cho các thiết bị tài nguyên hạn chế thông qua:

- Depthwise Separable Convolution
- Squeeze-and-Excitation Block

giúp giảm chi phí tính toán nhưng vẫn duy trì độ chính xác cao.

---

## 3.3.2 Tối ưu siêu tham số (Hyperparameter Optimization)

Các siêu tham số được tối ưu dựa trên tập Validation.

Các tham số được khảo sát bao gồm:

- Learning Rate
- Batch Size
- Optimizer
- Number of Epochs
- Weight Decay

Mô hình có hiệu năng tốt nhất trên Validation Set sẽ được lựa chọn cho bước đánh giá tiếp theo.

---

## 3.3.3 Ensemble Learning

Sau khi huấn luyện các mô hình cơ sở, nghiên cứu tiến hành xây dựng các mô hình Ensemble nhằm tận dụng ưu điểm của từng kiến trúc CNN.

### Weighted Averaging Ensemble

Xác suất dự đoán cuối cùng được tính bằng công thức:

\[ P(y)=\sum\_{i=1}^{n}w_iP_i(y) \]

Trong đó:

- $P_i(y)$ là xác suất dự đoán của mô hình thứ $i$
- $w_i$ là trọng số tương ứng của mô hình thứ $i$
- $\\sum\_{i=1}^{n} w_i = 1$

---

### Stacking Ensemble

Trong phương pháp Stacking, đầu ra xác suất của các mô hình cơ sở được sử dụng làm đầu vào cho một bộ phân loại cấp cao (Meta-Classifier) để đưa ra dự đoán cuối cùng.

---

### Hybrid Ensemble (Đề xuất)

Nghiên cứu đề xuất Hybrid Ensemble kết hợp:

```text
ResNet50
+
EfficientNet-B0
+
MobileNetV3
```

nhằm khai thác tính bổ sung thông tin giữa các kiến trúc CNN khác nhau.

---

# 3.4 Giai đoạn 3 – Đánh giá mô hình

## 3.4.1 Các độ đo đánh giá

Hiệu năng mô hình được đánh giá thông qua các chỉ số phân loại phổ biến.

### Accuracy

Accuracy đo tỷ lệ dự đoán đúng trên toàn bộ tập kiểm tra:

\[ Accuracy = \frac{TP + TN}{TP + TN + FP + FN} \]

Trong đó:

- TP (True Positive): Dự đoán đúng lớp dương
- TN (True Negative): Dự đoán đúng lớp âm
- FP (False Positive): Dự đoán sai lớp dương
- FN (False Negative): Dự đoán sai lớp âm

---

### Precision

Precision đo độ chính xác của các dự đoán dương:

\[ Precision = \frac{TP}{TP + FP} \]

Precision càng cao cho thấy số lượng dự đoán sai lớp dương càng thấp.

---

### Recall

Recall đo khả năng phát hiện đúng các mẫu thuộc lớp dương:

\[ Recall = \frac{TP}{TP + FN} \]

Recall cao cho thấy mô hình bỏ sót ít mẫu thực tế.

---

### F1-Score

F1-Score là trung bình điều hòa giữa Precision và Recall:

\[ F1 = 2 \times \frac{Precision \times Recall}{Precision + Recall} \]

F1-Score đặc biệt hữu ích khi dữ liệu bị mất cân bằng giữa các lớp.

---

## 3.4.2 So sánh Benchmark

Các mô hình được so sánh bao gồm:

- ResNet50
- EfficientNet-B0
- MobileNetV3
- Weighted Averaging Ensemble
- Stacking Ensemble
- Hybrid Ensemble

Mục tiêu là xác định liệu Ensemble Learning có cải thiện hiệu năng so với các mô hình CNN đơn lẻ hay không.

---

## 3.4.3 Đánh giá độ bền vững (Robustness Evaluation)

Mô hình được đánh giá trên các ảnh có chứa các yếu tố gây nhiễu như:

- Gaussian Noise
- Độ sáng thấp
- Độ tương phản thấp
- Ảnh bị mờ

Qua đó đánh giá khả năng hoạt động trong điều kiện thực tế ngoài đồng ruộng.

---

## 3.4.4 Đánh giá khả năng tổng quát hóa (Generalization Evaluation)

Mô hình được kiểm tra trên tập dữ liệu chưa từng xuất hiện trong quá trình huấn luyện nhằm đánh giá khả năng tổng quát hóa đối với dữ liệu mới.

---

# 3.5 Giai đoạn 4 – Giải thích mô hình và phân tích kết quả

## 3.5.1 Grad-CAM

Grad-CAM được sử dụng để trực quan hóa các vùng ảnh mà mô hình tập trung khi đưa ra dự đoán.

Mục tiêu:

- Kiểm tra tính hợp lý của mô hình
- Xác định mô hình có tập trung đúng vùng sâu bệnh hay không
- Tăng tính minh bạch và độ tin cậy của hệ thống

---

## 3.5.2 Phân tích kết quả

Kết quả thực nghiệm được phân tích theo các khía cạnh:

- Hiệu quả nhận diện bệnh hại lúa
- Hiệu quả nhận diện sâu hại lúa
- So sánh giữa các kiến trúc CNN
- So sánh giữa mô hình đơn và Ensemble
- Khả năng ứng dụng trong hệ thống nông nghiệp thông minh

---

# Đóng góp kỳ vọng của nghiên cứu

Nghiên cứu kỳ vọng mang lại các đóng góp sau:

1. Xây dựng bộ dữ liệu nhận diện tổng hợp bệnh và sâu hại lúa từ nhiều nguồn dữ liệu khác nhau.
2. Đánh giá toàn diện các kiến trúc ResNet50, EfficientNet-B0 và MobileNetV3 trên cùng một bộ dữ liệu.
3. Đề xuất mô hình Hybrid Ensemble Learning cho bài toán nhận diện sâu bệnh hại lúa.
4. Chứng minh hiệu quả của Ensemble Learning thông qua các thực nghiệm benchmark.
5. Cung cấp khả năng giải thích quyết định của mô hình bằng Grad-CAM, hỗ trợ triển khai trong thực tế.


