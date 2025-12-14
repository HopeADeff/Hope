# TLDR;

Hệ thống bảo vệ hình ảnh khỏi AI, sử dụng các công nghệ sau:

- [PyTorch](https://pytorch.org/) - Deep learning framework

- [CLIP](https://github.com/openai/CLIP) - Vision-language model

- [.NET 8+](https://dotnet.microsoft.com/) - WPF desktop app

# Mục đích repo

- Bảo vệ artwork khỏi bị AI training/generation (>90% hiệu quả)

- Đầu độc concept để AI học sai (Nightshade)

- Nhúng copyright vô hình vào ảnh (steganography)

- Detect ảnh AI-generated

# Tính năng

| Feature | Mô tả |
|---------|-------|
| Glaze | Style cloaking - AI thấy style khác |
| Nightshade | Concept poisoning - AI học sai |
| AI Detect | Phát hiện ảnh AI-generated |
| Stego | Nhúng copyright ẩn (LSB/DCT) |
| Watermark | DWT/FFT watermark chống nén |

# Cách chạy?

```bash
git clone https://github.com/quachdang122-jpg/Hope.git
cd Hope-AD

.\setup.bat

.\run.bat
```

# CLI

```bash
python engine.py protect -i art.jpg -o safe.jpg --target-style abstract

python engine.py protect -i dog.jpg -o safe.jpg --nightshade

python engine.py detect -i image.jpg

python engine.py embed -i art.jpg -o safe.jpg --owner "Your Name"
```

# Requirements

- Python 3.10+
- .NET 8.0+
- NVIDIA GPU (recommended) hoặc CPU

# Disk Space

| Version | Size |
|---------|------|
| Download (zip) | 0.66 MB |
| CPU install | ~3 GB |
| GPU (CUDA) install | ~9 GB |

# Special Thanks

- [Noah Trần](https://github.com/Coder-Blue)
- [Nguyễn Trí Nhân](https://www.facebook.com/nguyen.ala.142)
