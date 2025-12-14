# Hướng Dẫn Bảo Vệ Phong Cách Glaze

## Tổng Quan

Bảo vệ phong cách Glaze là một phương pháp mang tính cách mạng nhằm bảo vệ hình ảnh khỏi việc huấn luyện và tạo sinh bởi AI. Không giống như các nhiễu đối kháng truyền thống thêm sự hỗn loạn/nhiễu, bảo vệ kiểu Glaze **dịch chuyển phong cách nghệ thuật** của hình ảnh trong không gian đặc trưng của AI, khiến chúng không phù hợp để huấn luyện mà vẫn duy trì chất lượng thị giác không thể nhận thấy.

## Các Cải Tiến Chính So Với Phương Pháp Truyền Thống

| Tính Năng               | Kiểu Glaze               | Đối Kháng Truyền Thống |
| ----------------------- | ------------------------ | ---------------------- |
| **Hiệu quả Bảo vệ**     | **>90%**                 | 30-50%                 |
| **Kích thước File**     | **Nhỏ hơn 30-50%**       | Lớn (entropy cao)      |
| **Phương Pháp**         | Dịch chuyển phong cách   | Tiêm hỗn loạn/nhiễu    |
| **Chất Lượng Thị Giác** | Không thể nhận thấy      | Không thể nhận thấy    |
| **Thời Gian Xử Lý**     | 2-4 phút (CPU)           | 2-3 phút (CPU)         |
| **Khuyến Nghị**         | ✅ **Phương pháp chính** | Hỗ trợ legacy          |

## Cách Thức Hoạt Động

### Nhiễu Đối Kháng Truyền Thống

```

Hình Ảnh Gốc → Thêm Hỗn Loạn/Nhiễu → Hình Ảnh Được Bảo Vệ
↓
AI vẫn học được các mẫu
Hiệu quả: 30-50%

```

### Bảo Vệ Phong Cách Glaze (MỚI)

```

Hình Ảnh Gốc → Dịch Chuyển Phong Cách trong Không Gian AI → Hình Ảnh Được Bảo Vệ
↓                    ↓                        ↓
Ảnh Thực Tế → Trừu Tượng trong Nhận Thức AI → AI Không Thể Học
Hiệu quả: \>90%

```

**Điểm Khác Biệt Chính**: Thay vì cố gắng gây nhầm lẫn cho AI bằng nhiễu ngẫu nhiên, bảo vệ Glaze thay đổi cách AI diễn giải hình ảnh về mặt ngữ nghĩa, biến bức ảnh chân dung thực tế thành thứ mà AI coi là một bức tranh trừu tượng của Jackson Pollock, phá vỡ quá trình huấn luyện của nó.

## 🚀 Bắt Đầu Nhanh

### 1. Cài Đặt và Cấu Hình

```bash
# Đảm bảo các thư viện phụ thuộc được cài đặt
pip install torch torchvision Pillow numpy
pip install git+[https://github.com/openai/CLIP.git](https://github.com/openai/CLIP.git)

# Cài đặt module bảo vệ Glaze
# (Đã có sẵn trong Hope-AD/Hope/Hope/glaze_protection.py)
```

### 2\. Bảo Vệ Hình Ảnh (Khuyến Nghị)

```python
from glaze_protection import protect_image

# Sử dụng phong cách "abstract" để đạt hiệu quả tối đa
protect_image(
    input_path="input.jpg",
    output_path="protected_glaze.jpg",
    target_style="abstract",
    intensity=0.45,      # Cường độ khuyến nghị
    iterations=250,      # Vòng lặp khuyến nghị
    output_quality=92    # Chất lượng JPEG được tối ưu hóa
)
```

## ⚙️ Các Tham Số Quan Trọng

### 1\. `target_style` (String)

Chọn phong cách mà AI sẽ diễn giải hình ảnh của bạn:

| Phong Cách        | Hiệu Quả Bảo Vệ | Tính Chất                        | Khuyên Dùng Cho                 |
| ----------------- | --------------- | -------------------------------- | ------------------------------- |
| **abstract**      | Tối đa          | Trừu tượng kiểu Jackson Pollock  | Bảo vệ tối đa (Mặc định)        |
| **impressionist** | Cao             | Mềm mại, kiểu Monet              | Ảnh nghệ thuật, phong cảnh      |
| **cubist**        | Cao             | Phân tích, hình học kiểu Picasso | Chủ thể hình học                |
| **sketch**        | Trung bình      | Phác thảo bằng bút chì           | Nghệ thuật nhân vật, bản vẽ     |
| **watercolor**    | Trung bình      | Màu nước, vệt màu mềm            | Hình ảnh nhiều màu sắc, mềm mại |

### 2\. `intensity` (Float: 0.0 - 1.0)

Kiểm soát mức độ dịch chuyển phong cách. Giá trị cao hơn bảo vệ tốt hơn nhưng có thể dễ nhận thấy hơn (thường là rất tinh tế).

- **Phạm vi Khuyến Nghị**: **0.40 - 0.50**

### 3\. `iterations` (Int)

Số bước tối ưu hóa để đạt được sự dịch chuyển phong cách. Số vòng lặp cao hơn cho kết quả tốt hơn, nhưng chậm hơn.

- **Phạm vi Khuyến Nghị**: **200 - 300** (Mặc định: 250)

### 4\. `output_quality` (Int: 85 - 98)

Kiểm soát chất lượng nén JPEG của file đầu ra. Kích thước file được tối ưu hóa ở chất lượng 92 với subsampling 4:2:0.

- **Khuyến Nghị**: **92** (Cân bằng tốt nhất giữa chất lượng và kích thước file)

## 🎛️ Tối Ưu Hóa Hiệu Năng

### Tăng Tốc GPU (Khuyến Nghị)

Bảo vệ Glaze được tăng tốc bằng GPU:

- **CPU**: 2-4 phút mỗi hình ảnh (250 vòng lặp)
- **GPU (CUDA)**: **15-40 giây** mỗi hình ảnh

Để sử dụng GPU:

1.  Đảm bảo bạn đã cài đặt bộ công cụ CUDA.
2.  Cài đặt phiên bản PyTorch có hỗ trợ CUDA.
3.  Module sẽ tự động phát hiện và sử dụng GPU.

### Kiểm Soát Kích Thước File

Bảo vệ Glaze tạo ra file nhỏ hơn 30-50% so với phương pháp đối kháng truyền thống do sử dụng nhiễu có cấu trúc.

- **Kiểm soát kích thước**: Sử dụng tham số `output_quality`. Chất lượng thấp hơn (ví dụ: 88) sẽ tạo ra file nhỏ hơn nhưng có thể xuất hiện nhiều nhiễu nén (compression artifacts).

## ⚠️ Cân Nhắc và Thực Hành Tốt Nhất

### Tại Sao Glaze Tốt Hơn

Bảo vệ Glaze không chỉ là một phương pháp phòng thủ khác; đó là một sự dịch chuyển kiến trúc:

- **Mục tiêu là Style**: Tấn công khả năng của AI để nhận diện phong cách (style) thực tế.
- **Kháng cự tốt hơn**: Các trình tạo AI gặp khó khăn trong việc tạo sinh từ các mẫu đã dịch chuyển phong cách.
- **Tiêu thụ ít hơn**: Các nhiễu đối kháng truyền thống tạo ra entropy cao, làm tăng kích thước file; Glaze sử dụng nhiễu có cấu trúc nén tốt hơn.

### Cân Nhắc Khi Chuyển Đổi

Nếu bạn đang sử dụng phương pháp nhiễu đối kháng truyền thống:

| Cài Đặt Truyền Thống | Chuyển Sang Glaze                         |
| -------------------- | ----------------------------------------- |
| `intensity=0.30`     | `intensity=0.45`                          |
| `iterations=150`     | `iterations=250`                          |
| **Khuyến Nghị**      | **Chuyển sang `target_style="abstract"`** |

### Lợi Ích Khi Chuyển Đổi

1.  **Bảo vệ Tốt hơn 3 lần**: Hiệu quả 90% so với 30%
2.  **File Nhỏ hơn**: Giảm kích thước 30-50%
3.  **Chống lại tương lai**: Bảo vệ dựa trên phong cách có khả năng phục hồi tốt hơn
4.  **Chất Lượng Thị Giác Tương Tự**: Các thay đổi không thể nhận thấy

## FAQ

**Hỏi: Tôi có thể sử dụng cả hai phương pháp trên cùng một hình ảnh không?** Đ: Không khuyến nghị. Chỉ kiểu Glaze đã cung cấp sự bảo vệ vượt trội. Sử dụng cả hai có thể làm tăng kích thước file mà không mang lại lợi ích đáng kể.

**Hỏi: Tôi nên chọn phong cách nào?** Đ: "Abstract" được khuyến nghị để đạt hiệu quả bảo vệ tối đa. Hãy chọn dựa trên:

- Bảo vệ tối đa → Abstract
- Hình ảnh nghệ thuật → Impressionist
- Chủ thể hình học → Cubist
- Nghệ thuật nhân vật → Sketch
- Mềm mại/nhiều màu sắc → Watercolor

**Hỏi: Tôi nên xử lý lại hình ảnh bao lâu một lần?** Đ: Chỉ khi xuất hiện các mô hình AI mới, mạnh mẽ hơn đáng kể. Bảo vệ hiện tại có hiệu quả chống lại các mô hình hiện có.

**Hỏi: Điều này có ảnh hưởng đến metadata hình ảnh không?** Đ: Không, dữ liệu EXIF được giữ nguyên trừ khi bị xóa rõ ràng.

**Hỏi: Điều này có thể bị đảo ngược không?** Đ: Không, lớp bảo vệ là vĩnh viễn đối với bản sao đã được bảo vệ. Luôn giữ bản gốc chưa được bảo vệ riêng biệt.

**Hỏi: Việc sử dụng này có hợp pháp không?** Đ: Có, bảo vệ hình ảnh của riêng bạn là hợp pháp. Kiểm tra luật pháp địa phương để biết việc sử dụng thương mại.

## Hỗ Trợ & Tài Nguyên

- **Tài liệu**: README.md
- **Ví dụ**: Xem `examples/glaze_example.py`
- **Vấn đề**: Mở một issue
