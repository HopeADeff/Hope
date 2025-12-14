# Hope-AD

![Hope-AD Icon](Hope/Hope/Hope.ico)
## Download

**[Download Hope-AD Installer (v1.0.0)](https://github.com/HopeADeff/Hope/releases)**

**Cách cài đặt:**
1. Tải file `Hope-AD-Setup-1.0.0.exe`
2. Chạy file cài đặt
3. Mở **Hope-AD** từ Desktop

---

## Mục Đích

Dự án được thiết kế để bảo vệ bản quyền hình ảnh bằng kỹ thuật **Adversarial Machine Learning**
- **Chống AI Training**: Ngăn chặn Stable Diffusion, Midjourney mimic style của bạn.
- **Nightshade (Poison)**: "Đầu độc" dữ liệu training. Mặc định biến概念 "Artwork" thành "Noise" (Nhiễu), khiến AI không thể học được khái niệm về tranh vẽ.
- **Style Cloaking (Glaze)**:

## Tính Năng

| Tính năng | Mô tả | Hiệu quả |
|-----------|-------|----------|
| **Glaze Protection** | Giả lập phong cách khác (Abstract, Impressionist...) để đánh lừa AI | Cao |
| **Nightshade** | Gây nhiễu khái niệm (Default: Artwork → Noise) | Rất cao |
| **AI Detect** | Phát hiện ảnh do AI tạo ra | Trung bình |
| **Watermark** | Đóng dấu bản quyền vô hình (Invisible Watermark) | Cao |

## FAQ

### Q: Tại sao tôi upload ảnh đã protect lên ChatGPT/Gemini, nó vẫn mô tả được bình thường?

**A: Tool bảo vệ "Phong cách" (Style), không phải che giấu "Nội dung" (Content).**

1. **Khác biệt về Mục tiêu**:
   - Hope-AD tấn công vào **CLIP Encoder** - "con mắt" của các AI vẽ tranh như Stable Diffusion/Midjourney.
   - ChatGPT/Gemini dùng các mô hình Vision hoàn toàn khác (lớn hơn gấp 100 lần) để hiểu nội dung.

2. **Cơ chế Bảo vệ**:
   - Mục tiêu của Glaze/Nightshade là ngăn AI **bắt chước nét vẽ** của bạn.
   - Việc Gemini nhận ra "trong tranh có con mèo" là bình thường. Nhưng nếu ai đó dùng ảnh đó để train AI vẽ "con mèo theo phong cách của bạn", AI sẽ thất bại.

### Q: Ảnh sau khi protect có bị vỡ nét không?
A: Tool sử dụng thuật toán tối ưu hóa để giữ sự thay đổi ở mức thấp nhất (gần như vô hình với mắt thường). Tuy nhiên, với setting `Intensity` cao, có thể xuất hiện nhiễu hạt nhẹ.

### Q: Nên chọn mức Intensity nào phù hợp (giống 80-90% gốc)?
**Khuyến nghị:**
- **Rất giống bản gốc (95%+)**: `0.05` (5%) -> Phù hợp nếu bạn muốn ảnh giữ nguyên vẻ đẹp tối đa.
- **Khuyên dùng (Balanced)**: `0.08 - 0.10` (8-10%) -> Cân bằng giữa bảo vệ và thẩm mỹ (giống ~90%).
- **Bảo vệ mạnh**: `0.15+` -> Có thể xuất hiện nhiễu (noise) nhẹ nhưng bảo vệ tốt hơn.

### Q: Tại sao các model AI vẫn tạo ra được nhân vật hoàn chỉnh từ ảnh được sử dụng phương pháp Glaze của tôi?
**A: Đây là sự khác biệt giữa hình thức Training và Inference:**

1. **Inference (Tạo ảnh/Img2Img)**: Khi bạn đưa ảnh vào để AI vẽ lại, AI có khả năng **khử nhiễu** (denoise) rất mạnh. Nó có thể nhìn xuyên qua lớp Glaze mỏng để tái tạo lại đường nét. **Glaze KHÔNG được thiết kế để chặn việc này :(.**
2. **Training (Học Style)**: Đây là mục đích chính của Glaze. Nếu ai đó dùng ảnh Glaze của bạn để **Train LoRA**, model đó sẽ bị hỏng (học ra nhiễu hoặc phong cách lập thể thay vì tranh gốc).

=> **Kết luận**: Việc AI vẫn nhìn thấy nhân vật để vẽ lại (i2i) là bình thường. Glaze bảo vệ bạn khỏi việc bị **đánh cắp style** để tạo ra Model riêng.

---

## Cài đặt cho dev/contributors

Nếu bạn muốn phát triển hoặc chạy từ mã nguồn:

```bash
git clone https://github.com/HopeADeff/Hope.git
cd Hope
.\setup.bat  
.\run.bat    
```

### Yêu cầu hệ thống (Source Code)
- Python 3.10+
- .NET 8.0+
- NVIDIA GPU (Khuyên dùng - nhanh hơn 20x so với CPU)

## Disk Space

| Phiên bản | Kích thước | Ghi chú |
|-----------|------------|---------|
| **Installer (.exe)** | **~130 MB** | Đã bao gồm tất cả (Python, Torch...) |
| **Source Code** | ~1 MB | Chưa bao gồm venv |
| **Installed (Full)** | ~3 GB | PyTorch + Dependencies |

## References & Credits

Dự án được xây dựng dựa trên các nghiên cứu khoa học:

- **Nightshade**: [Shawn Shan et al., "Nightshade: Prompt-Specific Poisoning Attacks on Text-to-Image Generative Models"](https://arxiv.org/abs/2310.13828)
- **Glaze**: [Shawn Shan et al., "Glaze: Protecting Artists from Style Mimicry by Text-to-Image Models"](https://arxiv.org/abs/2302.04222)
- **CLIP**: [OpenAI, "Learning Transferable Visual Models From Natural Language Supervision"](https://github.com/openai/CLIP)

## Special Thanks

- [Noah Trần](https://github.com/Coder-Blue)
- [Nguyễn Trí Nhân](https://www.facebook.com/nguyen.ala.142)
