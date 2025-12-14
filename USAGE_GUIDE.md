# Hướng Dẫn Sử Dụng Hope-AD

Tài liệu này cung cấp hướng dẫn chi tiết để sử dụng hệ thống bảo vệ hình ảnh Hope-AD.

## Mục Lục

1. [Bắt Đầu Nhanh](#quick-start)
2. [Module Python](#python-modules)
   - [Nhiễu Đối Kháng](#adversarial-perturbations)
   - [Băm Hình Ảnh (Image Hashing)](#image-hashing)
3. [Thành Phần C#](#c-components)
   - [Đóng Dấu Bản Quyền (Watermarking)](#watermarking)
   - [Băm Hình Ảnh (C#)](#image-hashing-c)
4. [Ứng Dụng WPF](#wpf-application)
5. [Các Phương Pháp Tốt Nhất](#best-practices)
6. [Khắc Phục Sự Cố](#troubleshooting)

---

## Bắt Đầu Nhanh

### Thiết Lập Python

```bash
# Clone kho mã nguồn
git clone https://github.com/quachdang122-jpg/Hope-AD.git
cd Hope-AD

# Cài đặt các thư viện phụ thuộc
pip install -r requirements.txt

# Kiểm tra các module
cd Hope/Hope
python tests/test_image_hashing.py
```

### Thiết Lập C\#

```bash
cd Hope/Hope
dotnet build
dotnet run
```

---

## Module Python

### Nhiễu Đối Kháng

Module nhiễu đối kháng sử dụng các cuộc tấn công ngữ nghĩa dựa trên CLIP để bảo vệ hình ảnh khỏi sự phát hiện của AI.

#### Sử Dụng Cơ Bản

```python
from adversarial_perturbations import protect_image

# Bảo vệ một hình ảnh với cài đặt mặc định
protect_image(
    input_path="input.jpg",
    output_path="protected.jpg",
    intensity=0.30,      # 0.0 đến 1.0
    iterations=150       # Nhiều vòng lặp hơn = bảo vệ mạnh hơn
)
```

#### Sử Dụng Nâng Cao

```python
from adversarial_perturbations import AdversarialProtector

# Tạo một thể hiện bộ bảo vệ
protector = AdversarialProtector(
    intensity=0.25,      # Thấp hơn = tinh tế hơn
    iterations=100,      # Ít hơn = nhanh hơn
    verbose=True         # Hiển thị tiến trình
)

# Bảo vệ hình ảnh
success = protector.protect_image("input.jpg", "output.jpg")
if success:
    print("Image protected successfully!")
```

#### Tham Số

- **intensity** (float, 0.0-1.0): Độ lớn nhiễu tối đa

  - 0.1-0.2: Rất tinh tế, tác động thị giác tối thiểu
  - 0.25-0.35: Phạm vi khuyến nghị (mặc định: 0.30)
  - 0.4+: Bảo vệ mạnh, có thể nhìn thấy

- **iterations** (int): Số lần lặp tối ưu hóa

  - 50-100: Nhanh, bảo vệ vừa phải
  - 150: Khuyến nghị (mặc định)
  - 200+: Bảo vệ mạnh nhất, chậm hơn

- **verbose** (bool): Hiển thị thông báo tiến trình (mặc định: True)

#### Giao Diện Dòng Lệnh (CLI)

```bash
# Sử dụng engine.py
python engine.py \
    --input image.jpg \
    --output protected.jpg \
    --intensity 0.30 \
    --iterations 150

# Với mô tả mục tiêu
python engine.py \
    --input image.jpg \
    --output protected.jpg \
    --target "artistic portrait" \
    --intensity 0.30 \
    --iterations 150

# Sử dụng file mục tiêu (cho văn bản UTF-8)
python engine.py \
    --input image.jpg \
    --output protected.jpg \
    --target-file target.txt \
    --intensity 0.30 \
    --iterations 150
```

#### Mẹo Hiệu Năng

- **Sử Dụng GPU**: GPU hỗ trợ CUDA tăng tốc độ 10-20 lần
- **Xử Lý Hàng Loạt (Batch Processing)**: Tái sử dụng thể hiện `AdversarialProtector`
- **Cân Bằng**: Cường độ/số vòng lặp cao hơn = bảo vệ mạnh hơn nhưng xử lý lâu hơn

---

### Băm Hình Ảnh

Module băm hình ảnh cung cấp băm nhận thức (perceptual hashing) để phát hiện sửa đổi và xác minh tính xác thực.

#### Sử Dụng Cơ Bản

```python
from image_hashing import ImageHasher, compare_hashes

# Tạo bộ băm
hasher = ImageHasher()

# Tính toán mã băm
hash1 = hasher.perceptual_hash("image1.jpg")
hash2 = hasher.perceptual_hash("image2.jpg")

# So sánh
similarity = compare_hashes(hash1, hash2)
print(f"Similarity: {similarity:.2%}")
```

#### Các Phương Pháp Băm

```python
from image_hashing import ImageHasher

hasher = ImageHasher()

# Average Hash (nhanh nhất)
ahash = hasher.average_hash("image.jpg")

# Difference Hash (dựa trên gradient)
dhash = hasher.difference_hash("image.jpg")

# Perceptual Hash (mạnh mẽ nhất, khuyến nghị)
phash = hasher.perceptual_hash("image.jpg")

# Hoặc sử dụng phương pháp chung
hash_value = hasher.compute_hash("image.jpg", method="phash")
```

#### Lưu và Xác Minh

```python
from image_hashing import save_hash_to_file, verify_image

# Lưu mã băm vào file
save_hash_to_file("original.jpg", "original_hash.json", method="phash")

# Sau đó, xác minh một hình ảnh khác
is_match, similarity = verify_image("test.jpg", "original_hash.json", threshold=0.9)

if is_match:
    print(f"Images match! Similarity: {similarity:.2%}")
else:
    print(f"Images different. Similarity: {similarity:.2%}")
```

#### Phát Hiện Trùng Lặp

```python
from image_hashing import ImageHasher, is_similar

hasher = ImageHasher()
images = ["img1.jpg", "img2.jpg", "img3.jpg", "img4.jpg"]

# Tính toán mã băm
hashes = {img: hasher.perceptual_hash(img) for img in images}

# Tìm các bản sao
for img1 in images:
    for img2 in images:
        if img1 < img2:  # Tránh so sánh trùng lặp
            if is_similar(hashes[img1], hashes[img2], threshold=0.95):
                print(f"Duplicates found: {img1} and {img2}")
```

#### Giao Diện Dòng Lệnh (CLI)

```bash
# Tính toán mã băm
python image_hashing.py --input image.jpg --output hash.json --method phash

# So sánh hai hình ảnh
python image_hashing.py --input image1.jpg --compare image2.jpg --method phash

# Xác minh với mã băm đã lưu trữ
python image_hashing.py --input test.jpg --verify original_hash.json --threshold 0.9
```

#### So Sánh Phương Pháp Băm

| Phương Pháp | Tốc Độ     | Độ Bền Vững | Trường Hợp Sử Dụng                   |
| ----------- | ---------- | ----------- | ------------------------------------ |
| aHash       | Nhanh nhất | Thấp        | So sánh nhanh                        |
| dHash       | Nhanh      | Trung bình  | Phát hiện cạnh                       |
| pHash       | Chậm hơn   | Cao         | Sử dụng trong sản xuất (khuyến nghị) |

---

## Thành Phần C\#

### Đóng Dấu Bản Quyền

Thêm hình mờ động, tùy chỉnh vào hình ảnh.

#### Sử Dụng Cơ Bản

```csharp
using ArtShield;

// Hình mờ mặc định
var config = ImageWatermarking.CreateDefaultConfig();
ImageWatermarking.AddWatermark("input.jpg", "watermarked.jpg", config);
```

#### Hình Mờ Tùy Chỉnh

```csharp
using ArtShield;
using System.Windows.Media;

var config = new ImageWatermarking.WatermarkConfig
{
    Text = "© 2024 My Company",
    FontSize = 48,
    FontFamily = "Arial",
    Opacity = 0.4,
    Position = ImageWatermarking.WatermarkPosition.BottomRight,
    Color = Colors.Yellow,
    RotationAngle = 0,
    Tiled = false
};

ImageWatermarking.AddWatermark("input.jpg", "watermarked.jpg", config);
```

#### Hình Mờ Lặp Lại (Tiled Watermark)

```csharp
// Tạo hình mờ lặp lại để bảo vệ tối đa
var config = ImageWatermarking.CreateTiledConfig("CONFIDENTIAL");
ImageWatermarking.AddWatermark("input.jpg", "protected.jpg", config);
```

#### Cấu Hình Cài Đặt Sẵn

```csharp
// Mặc định: Giữa, chéo, bán trong suốt
var defaultConfig = ImageWatermarking.CreateDefaultConfig();

// Lặp lại (Tiled): Mẫu lặp lại trên toàn bộ hình ảnh
var tiledConfig = ImageWatermarking.CreateTiledConfig("PROTECTED");

// Góc: Hình mờ nhỏ, tinh tế ở góc
var cornerConfig = ImageWatermarking.CreateCornerConfig("© 2024");
```

#### Vị Trí Hình Mờ

- TopLeft (Trên Trái), TopCenter (Trên Giữa), TopRight (Trên Phải)
- CenterLeft (Giữa Trái), Center (Giữa), CenterRight (Giữa Phải)
- BottomLeft (Dưới Trái), BottomCenter (Dưới Giữa), BottomRight (Dưới Phải)

#### Tùy Chọn Tùy Chỉnh

```csharp
var config = new ImageWatermarking.WatermarkConfig
{
    Text = "Your Text",           // Văn bản hình mờ
    FontSize = 36,                 // Kích thước font chữ theo điểm (points)
    FontFamily = "Arial",          // Tên họ font chữ
    Opacity = 0.3,                 // 0.0 (trong suốt) đến 1.0 (mờ đục)
    Color = Colors.White,          // Màu văn bản
    RotationAngle = -45,           // Góc xoay theo độ
    Tiled = false,                 // Đơn lẻ hoặc lặp lại
    TileSpacing = 0.25,           // Khoảng cách cho dạng lặp (0.0-1.0)
    Position = WatermarkPosition.Center
};
```

---

### Băm Hình Ảnh (C\#)

Triển khai băm hình ảnh nhận thức bằng C\#.

#### Sử Dụng Cơ Bản

```csharp
using ArtShield;

var hasher = new ImageHashing();

// Tính toán mã băm
string hash1 = hasher.ComputeHash("image1.jpg", ImageHashing.HashMethod.PerceptualHash);
string hash2 = hasher.ComputeHash("image2.jpg", ImageHashing.HashMethod.PerceptualHash);

// So sánh
double similarity = ImageHashing.CompareHashes(hash1, hash2);
Console.WriteLine($"Similarity: {similarity:P}");
```

#### Lưu và Xác Minh

```csharp
var hasher = new ImageHashing();

// Lưu mã băm
hasher.SaveHashToFile("original.jpg", "hash.json", ImageHashing.HashMethod.PerceptualHash);

// Xác minh sau này
var (isMatch, similarity) = hasher.VerifyImage("test.jpg", "hash.json", threshold: 0.9);
Console.WriteLine($"Match: {isMatch}, Similarity: {similarity:P}");
```

#### Phương Pháp Băm

```csharp
var hasher = new ImageHashing();

// Average Hash
string ahash = hasher.ComputeHash("image.jpg", ImageHashing.HashMethod.AverageHash);

// Difference Hash
string dhash = hasher.ComputeHash("image.jpg", ImageHashing.HashMethod.DifferenceHash);

// Perceptual Hash (khuyến nghị)
string phash = hasher.ComputeHash("image.jpg", ImageHashing.HashMethod.PerceptualHash);
```

---

## Ứng Dụng WPF

Ứng dụng Hope-AD WPF cung cấp giao diện người dùng đồ họa (GUI) thân thiện để bảo vệ hình ảnh.

### Tính Năng

1.  **Chọn Hình Ảnh**: Duyệt và chọn hình ảnh
2.  **Cài Đặt Bảo Vệ**:
    - Mô tả mục tiêu (tùy chọn)
    - Thanh trượt Cường độ (0.0 - 1.0)
    - Thanh trượt Vòng lặp (50 - 300)
3.  **Xem Trước (Preview)**: Xem hình ảnh gốc và hình ảnh đã bảo vệ
4.  **Theo Dõi Tiến Trình**: Cập nhật trạng thái theo thời gian thực

### Sử Dụng

1.  Khởi chạy ứng dụng: `dotnet run`
2.  Nhấp vào "Select Image" để chọn một hình ảnh
3.  (Tùy chọn) Nhập mô tả mục tiêu
4.  Điều chỉnh thanh trượt cường độ và số vòng lặp
5.  Nhấp vào "Run Protection"
6.  Chờ quá trình xử lý (2-4 phút trên CPU, 10-30 giây trên GPU)
7.  Xem hình ảnh đã bảo vệ trong phần xem trước
8.  Hình ảnh đã bảo vệ được lưu với hậu tố `_protected`

### Cấu Hình

Biến môi trường:

- `ENGINE_PYTHON`: Đường dẫn đến trình thông dịch Python (mặc định: `python`)

---

## Các Phương Pháp Tốt Nhất

### Chiến Lược Bảo Vệ Hình Ảnh

1.  **Bảo Vệ Đa Lớp**: Kết hợp nhiều kỹ thuật

    ```
    Original → Adversarial Perturbations → Watermark → Hash
    ```

2.  **Lựa Chọn Cường Độ**:

    - Hình ảnh công khai: 0.25-0.30 (tinh tế)
    - Hình ảnh nhạy cảm: 0.30-0.35 (cân bằng)
    - Hình ảnh giá trị cao: 0.35-0.40 (mạnh)

3.  **Chiến Lược Hình Mờ (Watermark)**:

    - Hình mờ lặp lại (Tiled watermarks): Bảo vệ tối đa, dễ nhìn thấy hơn
    - Hình mờ đơn lẻ: Ít xâm lấn hơn, dễ bị cắt hơn
    - Kết hợp: Hình mờ lặp lại tinh tế + hình mờ đơn lẻ nổi bật

4.  **Lưu Trữ Mã Băm (Hash Storage)**:

    - Lưu trữ mã băm riêng biệt với hình ảnh
    - Sử dụng băm nhận thức (pHash) để có kết quả tốt nhất
    - Đặt ngưỡng dựa trên trường hợp sử dụng (0.85-0.95)

### Tối Ưu Hóa Hiệu Năng

1.  **Xử Lý Hàng Loạt (Batch Processing)**:

    ```python
    protector = AdversarialProtector(intensity=0.30, iterations=150)
    for image in image_list:
        protector.protect_image(image, f"protected_{image}")
    ```

2.  **Tăng Tốc GPU**:

    - Cài đặt bộ công cụ CUDA
    - Cài đặt PyTorch với hỗ trợ CUDA
    - Xử lý nhanh hơn 10-20 lần

3.  **Xử Lý Song Song**:

    ```python
    from concurrent.futures import ThreadPoolExecutor

    def protect(img):
        protector = AdversarialProtector()
        return protector.protect_image(img, f"protected_{img}")

    with ThreadPoolExecutor(max_workers=4) as executor:
        results = executor.map(protect, images)
    ```

### Cân Nhắc về Bảo Mật

1.  **Không Tuyệt Đối**: Không có biện pháp bảo vệ nào hiệu quả 100%
2.  **Cập Nhật Thường Xuyên**: Các mô hình AI phát triển; cập nhật các phương pháp bảo vệ
3.  **Theo Dõi Sử Dụng**: Theo dõi nơi các hình ảnh được bảo vệ được sử dụng
4.  **Bảo Vệ Pháp Lý**: Sử dụng hình mờ như bằng chứng pháp lý
5.  **Nhiều Lớp**: Không dựa vào một phương pháp bảo vệ duy nhất

---

## Khắc Phục Sự Cố

### Các Vấn Đề Thường Gặp

#### CLIP Chưa Được Cài Đặt

**Lỗi**: `ERROR: CLIP required!`

**Giải Pháp**:

```bash
pip install git+[https://github.com/openai/CLIP.git](https://github.com/openai/CLIP.git)
```

#### Hết Bộ Nhớ (GPU)

**Lỗi**: `CUDA out of memory`

**Giải Pháp**:

- Giảm kích thước hình ảnh trước khi xử lý
- Sử dụng CPU thay thế: Đặt `CUDA_VISIBLE_DEVICES=-1`
- Đóng các ứng dụng GPU khác

#### Xử Lý Chậm (CPU)

**Vấn Đề**: Xử lý mất hơn 5 phút mỗi hình ảnh

**Giải Pháp**:

- Giảm số vòng lặp (100 thay vì 150)
- Giảm cường độ một chút
- Cân nhắc tăng tốc GPU
- Sử dụng hình ảnh nhỏ hơn (thay đổi kích thước trước khi bảo vệ)

#### C\# Xây Dựng Thất Bại trên Linux

**Lỗi**: `To build a project targeting Windows...`

**Giải Pháp**: Điều này là bình thường. Ứng dụng WPF yêu cầu Windows. Sử dụng module Python trên Linux/Mac.

#### Hình Mờ Không Hiển Thị

**Vấn Đề**: Hình mờ quá tinh tế hoặc không hiển thị

**Giải Pháp**:

- Tăng độ mờ (0.3 → 0.5)
- Sử dụng màu tối hơn/sáng hơn để tạo độ tương phản
- Tăng kích thước font chữ
- Thử hình mờ lặp lại (tiled watermark)

#### So Sánh Mã Băm Cho Thấy Độ Tương Đồng Thấp

**Vấn Đề**: Hình ảnh tương tự có điểm tương đồng thấp

**Giải Pháp**:

- Sử dụng băm nhận thức (pHash) thay vì aHash/dHash
- Kiểm tra xem hình ảnh có khác biệt đáng kể không
- Điều chỉnh ngưỡng (giảm xuống 0.85 hoặc 0.80)
- Xác minh hình ảnh ở định dạng chính xác (RGB, không phải RGBA)

### Nhận Trợ Giúp

1.  Kiểm tra `README.md` để biết thông tin cơ bản
2.  Xem lại mã ví dụ trong thư mục `examples/`
3.  Chạy bộ kiểm thử: `python tests/test_image_hashing.py`
4.  Kiểm tra các vấn đề tương tự trên GitHub issues
5.  Tạo một issue mới với thông báo lỗi chi tiết

---

## Tài Nguyên Bổ Sung

- **README.md**: Tổng quan và bắt đầu nhanh
- **examples/**: Các script và mã ví dụ
- **tests/**: Các script kiểm thử để xác thực
- **requirements.txt**: Các thư viện phụ thuộc Python

## Thông Tin Phiên Bản

- Python: 3.8+
- .NET: 10.0+
- PyTorch: 2.0+
- CLIP: Phiên bản mới nhất từ GitHub

---

**Cập Nhật Lần Cuối**: 2024-12-11
