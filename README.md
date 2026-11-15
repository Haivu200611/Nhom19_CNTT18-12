# Nhóm 19 – Phân loại 102 loài hoa (Oxford Flowers 102)

Đồ án môn Học máy – lớp CNTT18-12.

Dự án nhận dạng **102 loài hoa** từ ảnh theo hướng **Transfer Learning**:

1. Dùng **ResNet50** đã huấn luyện sẵn trên ImageNet (bỏ lớp `fc`) để trích xuất vector đặc trưng 2048 chiều cho mỗi ảnh.
2. Huấn luyện và so sánh 4 mô hình học máy cổ điển trên các vector đó: **KNN**, **Gaussian Naive Bayes**, **Logistic Regression**, **SVM (RBF)**.
3. Lưu mô hình **Logistic Regression** và đưa vào một **giao diện web (Gradio)** để tải ảnh lên và xem Top-5 loài hoa dự đoán.

---

## Mục lục

- [Cấu trúc thư mục](#cấu-trúc-thư-mục)
- [Yêu cầu hệ thống](#yêu-cầu-hệ-thống)
- [Cài đặt](#cài-đặt)
- [Cách 1 – Chỉ chạy giao diện web (không cần dataset)](#cách-1--chỉ-chạy-giao-diện-web-không-cần-dataset)
- [Cách 2 – Huấn luyện lại từ đầu](#cách-2--huấn-luyện-lại-từ-đầu)
- [Chuẩn bị dataset](#chuẩn-bị-dataset)
- [Quy trình xử lý trong `main.py`](#quy-trình-xử-lý-trong-mainpy)
- [Các file kết quả](#các-file-kết-quả)
- [Lỗi thường gặp](#lỗi-thường-gặp)

---

## Cấu trúc thư mục

Sau khi clone, repo có các file sau:

```
Nhom19_CNTT18-12/
├── main.py                    # Pipeline đầy đủ: EDA → tiền xử lý → trích đặc trưng → huấn luyện → đánh giá
├── app.py                     # Giao diện web Gradio để nhận dạng hoa
├── flower_classifier.joblib   # Mô hình đã huấn luyện sẵn (scaler + Logistic Regression + tên lớp)
├── requirements.txt           # Danh sách thư viện
└── README.md
```

Hai thư mục **không có trên GitHub** (do dung lượng lớn), bạn phải tự tạo nếu muốn huấn luyện lại:

```
├── Dataset/                   # Dữ liệu ảnh – tải về theo hướng dẫn bên dưới
└── outputs/                   # Tự sinh ra khi chạy main.py (hình, bảng kết quả, đặc trưng, mô hình)
```

---

## Yêu cầu hệ thống

| Thành phần | Yêu cầu |
|---|---|
| Python | 3.9 – 3.12 (khuyến nghị 3.10 hoặc 3.11) |
| Hệ điều hành | Windows / macOS / Linux |
| RAM | Tối thiểu 8 GB |
| GPU | Không bắt buộc. Có GPU NVIDIA thì bước trích đặc trưng nhanh hơn nhiều |
| Internet | Cần ở lần chạy đầu để tải trọng số ResNet50 (~100 MB, tự động) |

> Trên Windows, lệnh Python có thể là `py` thay vì `python`. Trong README này dùng `python`; nếu máy báo không tìm thấy lệnh thì thay bằng `py`.

---

## Cài đặt

### 1. Clone repo

```bash
git clone https://github.com/Haivu200611/Nhom19_CNTT18-12.git
cd Nhom19_CNTT18-12
```

### 2. Tạo môi trường ảo (khuyến nghị)

**Windows (PowerShell):**

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

> Nếu PowerShell báo lỗi *"running scripts is disabled"*, chạy một lần lệnh sau rồi kích hoạt lại:
> `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`

**macOS / Linux:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Khi kích hoạt thành công, đầu dòng lệnh sẽ có chữ `(.venv)`.

### 3. Cài PyTorch

Chọn **một** trong hai lệnh:

- Máy **không có** GPU NVIDIA (bản CPU, nhẹ hơn):

  ```bash
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
  ```

- Máy **có** GPU NVIDIA: lấy lệnh phù hợp với phiên bản CUDA tại <https://pytorch.org/get-started/locally/>.

### 4. Cài các thư viện còn lại

```bash
pip install -r requirements.txt
```

Kiểm tra nhanh:

```bash
python -c "import torch, sklearn, gradio; print('OK, CUDA:', torch.cuda.is_available())"
```

---

## Cách 1 – Chỉ chạy giao diện web (không cần dataset)

Repo đã có sẵn mô hình `flower_classifier.joblib`, nên **không cần tải dataset hay huấn luyện** vẫn dùng được:

```bash
python app.py
```

Trình duyệt sẽ tự mở tại `http://127.0.0.1:7860`. Lần đầu chạy sẽ mất một lúc để tải trọng số ResNet50.

Trong giao diện:

- **🔍 Nhận dạng** – tải ảnh lên, chụp bằng webcam hoặc dán từ clipboard → xem loài hoa dự đoán, độ tin cậy và Top-5.
- **📊 Đánh giá mô hình** – bảng chỉ số và biểu đồ (chỉ hiện khi đã có thư mục `outputs/` từ việc chạy `main.py`).
- **ℹ️ Giới thiệu** – mô tả quy trình và lưu ý khi dùng.

Tùy chọn:

```bash
python app.py --share        # tạo link công khai tạm thời để gửi cho người khác
python app.py --port 8000    # chạy ở cổng khác nếu 7860 đang bận
```

Dừng ứng dụng: nhấn `Ctrl + C` trong terminal.

> Nếu chưa có thư mục `Dataset/test`, phần "Ảnh mẫu" sẽ không hiển thị. Bạn vẫn tải ảnh của mình lên bình thường.

---

## Cách 2 – Huấn luyện lại từ đầu

### Bước 1. Chuẩn bị dataset

Làm theo mục [Chuẩn bị dataset](#chuẩn-bị-dataset).

### Bước 2. Chạy pipeline

```bash
python main.py
```

Các tham số:

| Tham số | Mặc định | Ý nghĩa |
|---|---|---|
| `--data` | `Dataset` | Thư mục dataset (chứa `train/`, `valid/`, `cat_to_name.json`) |
| `--out` | `outputs` | Thư mục lưu kết quả |
| `--gui` | tắt | Mở giao diện web ngay sau khi huấn luyện xong |
| `--show` | tắt | Hiện cửa sổ biểu đồ khi chạy (phải đóng cửa sổ thì chương trình mới chạy tiếp) |

Ví dụ:

```bash
python main.py --data D:/du_lieu/flowers --out ket_qua --gui
```

### Thời gian chạy

- Bước **trích xuất đặc trưng** là lâu nhất: trên CPU có thể mất **30–60 phút**, trên GPU vài phút.
- Đặc trưng được lưu vào `outputs/features_resnet50_aug2.npz`. Từ lần chạy thứ hai, chương trình **tải lại file này** và bỏ qua bước trích xuất. Muốn trích xuất lại thì xóa file đó.

### Bước 3. Dùng mô hình vừa huấn luyện cho giao diện web

`main.py` lưu mô hình mới vào `outputs/flower_classifier.joblib`. Tuy nhiên `app.py` **ưu tiên file ở thư mục gốc** trước, sau đó mới tìm trong `outputs/`. Vì vậy, để giao diện dùng mô hình mới, chọn một trong hai cách:

```bash
# Cách a: chép đè mô hình mới ra thư mục gốc
# Windows (PowerShell)
Copy-Item outputs\flower_classifier.joblib . -Force
# macOS / Linux
cp outputs/flower_classifier.joblib .

# Cách b: xóa file ở thư mục gốc để app.py tự dùng file trong outputs/
```

Sau đó chạy `python app.py`.

---

## Chuẩn bị dataset

Dự án dùng bộ **Oxford 102 Category Flower Dataset** (trang gốc: <https://www.kaggle.com/datasets/waseemalastal/the-oxford-flowers-102-dataset>), ở dạng đã chia sẵn thư mục theo lớp – thường được chia sẻ trên Kaggle với tên *"PyTorch Challenge Flower Dataset"* / *"flower_data"* (gồm `train/`, `valid/`, `test/` và `cat_to_name.json`).

Sau khi tải và giải nén, đặt vào thư mục `Dataset/` ở gốc dự án với **đúng cấu trúc** sau:

```
Dataset/
├── cat_to_name.json      # Ánh xạ số thứ tự lớp → tên loài, ví dụ {"1": "pink primrose", ...}
├── train/                # Ảnh huấn luyện, mỗi lớp một thư mục con đặt tên bằng số
│   ├── 1/
│   │   ├── image_06734.jpg
│   │   └── ...
│   ├── 2/
│   └── ... (đến 102/)
├── valid/                # Ảnh có nhãn, cùng cấu trúc như train/ → được dùng làm TẬP TEST
│   ├── 1/
│   └── ... (đến 102/)
└── test/                 # (Không bắt buộc) ảnh không nhãn, để phẳng – app.py lấy làm ảnh mẫu
    ├── image_00001.jpg
    └── ...
```

Lưu ý:

- Tên thư mục lớp **phải là số** (`1`, `2`, …, `102`) vì code sắp xếp lớp theo giá trị số.
- `cat_to_name.json` phải nằm **trực tiếp** trong `Dataset/`.
- Chấp nhận ảnh `.jpg`, `.jpeg`, `.png`.
- Nếu giải nén ra một thư mục lồng nhau (ví dụ `Dataset/flower_data/train/...`), hãy chuyển các thư mục con lên đúng vị trí như trên, hoặc chạy `python main.py --data Dataset/flower_data`.

---

## Quy trình xử lý trong `main.py`

| Bước | Nội dung |
|---|---|
| **1. Khám phá dữ liệu (EDA)** | Đếm số ảnh theo lớp, mức mất cân bằng, phân phối kích thước ảnh, hiển thị ảnh mẫu |
| **2. Tiền xử lý** | Tách 15% tập train (phân tầng) làm validation. Resize 256 → cắt giữa 224×224 → chuẩn hóa theo mean/std ImageNet. Tăng cường dữ liệu cho train: crop ngẫu nhiên, lật ngang, xoay ±20°, thay đổi độ sáng/tương phản |
| **3. Trích xuất đặc trưng** | ResNet50 (`IMAGENET1K_V2`) đóng băng trọng số, bỏ lớp `fc` → vector 2048 chiều. Mỗi ảnh train có thêm 2 bản augment |
| **4. Huấn luyện** | `StandardScaler` fit trên train. **KNN**: chọn K (1–29) và khoảng cách (euclidean / manhattan / cosine) bằng Stratified Group 5-Fold CV (ảnh gốc và bản augment luôn cùng fold). **Naive Bayes, Logistic Regression, SVM**: chọn siêu tham số trên tập validation |
| **5. Đánh giá** | Accuracy, Precision / Recall / F1 (macro) trên tập test; báo cáo chi tiết mô hình tốt nhất; ma trận nhầm lẫn; 10 cặp loài bị nhầm nhiều nhất |
| **Lưu mô hình** | Lưu `scaler` + Logistic Regression + tên lớp vào `flower_classifier.joblib` |

Các hằng số cấu hình (kích thước ảnh, batch size, số bản augment, danh sách K, seed…) nằm ở đầu file `main.py`, mục `CẤU HÌNH`.

---

## Các file kết quả

Sau khi chạy `main.py`, thư mục `outputs/` chứa:

| File | Nội dung |
|---|---|
| `eda_phan_phoi_nhan.png` | Số ảnh theo từng loài hoa |
| `eda_kich_thuoc_anh.png` | Phân phối chiều rộng / chiều cao ảnh |
| `eda_anh_mau.png` | Ảnh mẫu của 8 loài ngẫu nhiên |
| `tien_xu_ly_augment.png` | Minh họa resize và tăng cường dữ liệu |
| `knn_chon_k_cv.png` | Accuracy theo K cho từng loại khoảng cách (5-fold CV) |
| `so_sanh_mo_hinh.png` | Biểu đồ so sánh 4 mô hình trên tập test |
| `confusion_matrix.png` | Ma trận nhầm lẫn chuẩn hóa của mô hình tốt nhất |
| `ket_qua_test.csv` | Bảng chỉ số của 4 mô hình trên tập test |
| `classification_report_best.csv` | Precision / Recall / F1 từng loài của mô hình tốt nhất |
| `features_resnet50_aug2.npz` | Đặc trưng đã trích xuất (cache để chạy lại nhanh) |
| `flower_classifier.joblib` | Mô hình dùng cho giao diện web |

---

## Lỗi thường gặp

| Lỗi | Cách xử lý |
|---|---|
| `'python' is not recognized...` | Dùng `py` thay cho `python`, hoặc cài Python và tick "Add Python to PATH" |
| `ModuleNotFoundError: No module named 'torch'` (hoặc thư viện khác) | Chưa kích hoạt môi trường ảo, hoặc chưa chạy bước cài đặt. Kích hoạt `.venv` rồi cài lại |
| `FileNotFoundError: ... Dataset\train` hoặc `cat_to_name.json` | Dataset chưa đúng cấu trúc – xem [Chuẩn bị dataset](#chuẩn-bị-dataset) hoặc chỉ đường dẫn bằng `--data` |
| `ValueError: invalid literal for int()` khi chạy `main.py` | Trong `train/` hoặc `valid/` có thư mục con không phải số – xóa hoặc đổi tên |
| `Không tìm thấy flower_classifier.joblib` khi chạy `app.py` | Thiếu file mô hình ở thư mục gốc. Kéo lại repo, hoặc chạy `python main.py` để tạo trong `outputs/` |
| Lỗi tải trọng số ResNet50 (`URLError`, timeout) | Kiểm tra kết nối Internet; trọng số được lưu vào cache của PyTorch và chỉ cần tải một lần |
| Trích xuất đặc trưng quá chậm | Bình thường trên CPU (30–60 phút), chỉ chạy một lần. Có GPU NVIDIA thì cài PyTorch bản CUDA |
| Đổi dataset/cấu hình nhưng kết quả không đổi | Xóa `outputs/features_resnet50_aug2.npz` để trích xuất lại đặc trưng |
| `OSError: Cannot find empty port` khi chạy `app.py` | Cổng 7860 đang bận – dùng `python app.py --port 8000` |
| Giao diện vẫn dùng mô hình cũ sau khi huấn luyện lại | Xem [Bước 3](#bước-3-dùng-mô-hình-vừa-huấn-luyện-cho-giao-diện-web) – chép đè `flower_classifier.joblib` ra thư mục gốc |

---

## Lưu ý khi dùng mô hình

- Mô hình chỉ biết **102 loài** trong Oxford Flowers 102. Ảnh hoa loài khác (hoặc không phải hoa) vẫn bị gán vào loài gần nhất.
- Kết quả tốt nhất với ảnh chụp rõ một bông hoa, cận cảnh, đủ sáng.
- Dự đoán có độ tin cậy dưới 40% được đánh dấu *"Không chắc chắn"*.