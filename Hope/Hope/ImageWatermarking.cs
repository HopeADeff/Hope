using System;
using System.IO;
using System.Windows;
using System.Windows.Media;
using System.Windows.Media.Imaging;

namespace ArtShield
{
    /// <summary>
    /// Image watermarking functionality for adding text overlays to images.
    /// NOTE: This is an experimental feature.
    /// </summary>
    public static class ImageWatermarking
    {
        /// <summary>
        /// Watermark position options.
        /// </summary>
        public enum WatermarkPosition
        {
            TopLeft,
            TopCenter,
            TopRight,
            MiddleLeft,
            Center,
            MiddleRight,
            BottomLeft,
            BottomCenter,
            BottomRight
        }

        /// <summary>
        /// Configuration for watermark appearance and placement.
        /// </summary>
        public class WatermarkConfig
        {
            public string Text { get; set; } = "PROTECTED";
            public string FontFamily { get; set; } = "Arial";
            public double FontSize { get; set; } = 36;
            public double Opacity { get; set; } = 0.3;
            public WatermarkPosition Position { get; set; } = WatermarkPosition.BottomRight;
            public Color Color { get; set; } = Colors.White;
            public double RotationAngle { get; set; } = 0;
            public bool Tiled { get; set; } = false;
            public int TileSpacing { get; set; } = 100;
            public int Margin { get; set; } = 20;
        }

        /// <summary>
        /// Creates a default watermark configuration.
        /// </summary>
        public static WatermarkConfig CreateDefaultConfig()
        {
            return new WatermarkConfig();
        }

        /// <summary>
        /// Creates a tiled watermark configuration.
        /// </summary>
        public static WatermarkConfig CreateTiledConfig(string text)
        {
            return new WatermarkConfig
            {
                Text = text,
                Tiled = true,
                Opacity = 0.15,
                RotationAngle = -45,
                FontSize = 24
            };
        }

        /// <summary>
        /// Creates a corner watermark configuration.
        /// </summary>
        public static WatermarkConfig CreateCornerConfig(string text)
        {
            return new WatermarkConfig
            {
                Text = text,
                Position = WatermarkPosition.BottomRight,
                Opacity = 0.5,
                FontSize = 18
            };
        }

        /// <summary>
        /// Adds a watermark to an image.
        /// </summary>
        /// <param name="inputPath">Path to the input image.</param>
        /// <param name="outputPath">Path to save the watermarked image.</param>
        /// <param name="config">Watermark configuration.</param>
        /// <returns>True if successful, false otherwise.</returns>
        public static bool AddWatermark(string inputPath, string outputPath, WatermarkConfig config)
        {
            try
            {
                if (!File.Exists(inputPath))
                {
                    Console.WriteLine($"Error: Input file not found: {inputPath}");
                    return false;
                }

                // Load the source image
                BitmapImage sourceImage = new BitmapImage();
                sourceImage.BeginInit();
                sourceImage.CacheOption = BitmapCacheOption.OnLoad;
                sourceImage.UriSource = new Uri(Path.GetFullPath(inputPath), UriKind.Absolute);
                sourceImage.EndInit();
                sourceImage.Freeze();

                int width = sourceImage.PixelWidth;
                int height = sourceImage.PixelHeight;

                // Create drawing visual for watermark
                DrawingVisual visual = new DrawingVisual();
                using (DrawingContext context = visual.RenderOpen())
                {
                    // Draw the original image
                    context.DrawImage(sourceImage, new Rect(0, 0, width, height));

                    // Create text formatting
                    var typeface = new Typeface(new FontFamily(config.FontFamily), 
                                                FontStyles.Normal, 
                                                FontWeights.Bold, 
                                                FontStretches.Normal);
                    
                    var brush = new SolidColorBrush(config.Color);
                    brush.Opacity = config.Opacity;
                    brush.Freeze();

                    var formattedText = new FormattedText(
                        config.Text,
                        System.Globalization.CultureInfo.CurrentCulture,
                        FlowDirection.LeftToRight,
                        typeface,
                        config.FontSize,
                        brush,
                        VisualTreeHelper.GetDpi(visual).PixelsPerDip);

                    if (config.Tiled)
                    {
                        // Draw tiled watermarks
                        DrawTiledWatermarks(context, formattedText, width, height, config);
                    }
                    else
                    {
                        // Draw single watermark
                        Point position = CalculatePosition(formattedText, width, height, config);
                        
                        if (config.RotationAngle != 0)
                        {
                            context.PushTransform(new RotateTransform(config.RotationAngle, 
                                position.X + formattedText.Width / 2, 
                                position.Y + formattedText.Height / 2));
                        }
                        
                        context.DrawText(formattedText, position);
                        
                        if (config.RotationAngle != 0)
                        {
                            context.Pop();
                        }
                    }
                }

                // Render to bitmap
                RenderTargetBitmap renderBitmap = new RenderTargetBitmap(
                    width, height, 96, 96, PixelFormats.Pbgra32);
                renderBitmap.Render(visual);
                renderBitmap.Freeze();

                // Save to file
                string extension = Path.GetExtension(outputPath).ToLowerInvariant();
                BitmapEncoder encoder = extension switch
                {
                    ".png" => new PngBitmapEncoder(),
                    ".bmp" => new BmpBitmapEncoder(),
                    _ => new JpegBitmapEncoder { QualityLevel = 95 }
                };

                encoder.Frames.Add(BitmapFrame.Create(renderBitmap));

                using (FileStream stream = new FileStream(outputPath, FileMode.Create))
                {
                    encoder.Save(stream);
                }

                return true;
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error adding watermark: {ex.Message}");
                return false;
            }
        }

        private static void DrawTiledWatermarks(DrawingContext context, FormattedText text, 
            int width, int height, WatermarkConfig config)
        {
            double textWidth = text.Width;
            double textHeight = text.Height;
            double spacing = config.TileSpacing;

            for (double y = -textHeight; y < height + textHeight; y += textHeight + spacing)
            {
                for (double x = -textWidth; x < width + textWidth; x += textWidth + spacing)
                {
                    context.PushTransform(new RotateTransform(config.RotationAngle, 
                        x + textWidth / 2, y + textHeight / 2));
                    context.DrawText(text, new Point(x, y));
                    context.Pop();
                }
            }
        }

        private static Point CalculatePosition(FormattedText text, int width, int height, WatermarkConfig config)
        {
            double textWidth = text.Width;
            double textHeight = text.Height;
            int margin = config.Margin;

            return config.Position switch
            {
                WatermarkPosition.TopLeft => new Point(margin, margin),
                WatermarkPosition.TopCenter => new Point((width - textWidth) / 2, margin),
                WatermarkPosition.TopRight => new Point(width - textWidth - margin, margin),
                WatermarkPosition.MiddleLeft => new Point(margin, (height - textHeight) / 2),
                WatermarkPosition.Center => new Point((width - textWidth) / 2, (height - textHeight) / 2),
                WatermarkPosition.MiddleRight => new Point(width - textWidth - margin, (height - textHeight) / 2),
                WatermarkPosition.BottomLeft => new Point(margin, height - textHeight - margin),
                WatermarkPosition.BottomCenter => new Point((width - textWidth) / 2, height - textHeight - margin),
                WatermarkPosition.BottomRight => new Point(width - textWidth - margin, height - textHeight - margin),
                _ => new Point(margin, margin)
            };
        }
    }
}
