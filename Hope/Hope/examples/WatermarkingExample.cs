using System;
using System.Windows.Media;
using ArtShield;

namespace ArtShield.Examples
{
    /// <summary>
    /// Example usage of the ImageWatermarking class.
    /// This class demonstrates various watermarking scenarios.
    /// </summary>
    public class WatermarkingExample
    {
        /// <summary>
        /// Example 1: Add a default watermark to an image.
        /// </summary>
        public static void Example1_DefaultWatermark()
        {
            Console.WriteLine("Example 1: Default Watermark");
            Console.WriteLine("=" + new string('=', 59));

            string inputPath = "input.jpg";
            string outputPath = "watermarked_default.jpg";

            var config = ImageWatermarking.CreateDefaultConfig();
            bool success = ImageWatermarking.AddWatermark(inputPath, outputPath, config);

            if (success)
            {
                Console.WriteLine($"✓ Watermark added successfully: {outputPath}");
            }
            else
            {
                Console.WriteLine("✗ Failed to add watermark");
            }
            Console.WriteLine();
        }

        /// <summary>
        /// Example 2: Add a custom watermark with specific settings.
        /// </summary>
        public static void Example2_CustomWatermark()
        {
            Console.WriteLine("Example 2: Custom Watermark");
            Console.WriteLine("=" + new string('=', 59));

            string inputPath = "input.jpg";
            string outputPath = "watermarked_custom.jpg";

            var config = new ImageWatermarking.WatermarkConfig
            {
                Text = "© 2024 My Company",
                FontSize = 48,
                Opacity = 0.5,
                Position = ImageWatermarking.WatermarkPosition.BottomRight,
                Color = Colors.Yellow,
                RotationAngle = 0,
                Tiled = false
            };

            bool success = ImageWatermarking.AddWatermark(inputPath, outputPath, config);

            if (success)
            {
                Console.WriteLine($"✓ Custom watermark added: {outputPath}");
            }
            else
            {
                Console.WriteLine("✗ Failed to add watermark");
            }
            Console.WriteLine();
        }

        /// <summary>
        /// Example 3: Add a tiled watermark for maximum protection.
        /// </summary>
        public static void Example3_TiledWatermark()
        {
            Console.WriteLine("Example 3: Tiled Watermark");
            Console.WriteLine("=" + new string('=', 59));

            string inputPath = "input.jpg";
            string outputPath = "watermarked_tiled.jpg";

            var config = ImageWatermarking.CreateTiledConfig("CONFIDENTIAL");
            bool success = ImageWatermarking.AddWatermark(inputPath, outputPath, config);

            if (success)
            {
                Console.WriteLine($"✓ Tiled watermark added: {outputPath}");
            }
            else
            {
                Console.WriteLine("✗ Failed to add watermark");
            }
            Console.WriteLine();
        }

        /// <summary>
        /// Example 4: Add a subtle corner watermark.
        /// </summary>
        public static void Example4_CornerWatermark()
        {
            Console.WriteLine("Example 4: Corner Watermark");
            Console.WriteLine("=" + new string('=', 59));

            string inputPath = "input.jpg";
            string outputPath = "watermarked_corner.jpg";

            var config = ImageWatermarking.CreateCornerConfig("© Protected 2024");
            bool success = ImageWatermarking.AddWatermark(inputPath, outputPath, config);

            if (success)
            {
                Console.WriteLine($"✓ Corner watermark added: {outputPath}");
            }
            else
            {
                Console.WriteLine("✗ Failed to add watermark");
            }
            Console.WriteLine();
        }

        /// <summary>
        /// Example 5: Batch watermarking multiple images.
        /// </summary>
        public static void Example5_BatchWatermark()
        {
            Console.WriteLine("Example 5: Batch Watermarking");
            Console.WriteLine("=" + new string('=', 59));

            string[] images = { "image1.jpg", "image2.jpg", "image3.jpg" };
            var config = ImageWatermarking.CreateDefaultConfig();

            foreach (string imagePath in images)
            {
                string outputPath = $"watermarked_{imagePath}";
                Console.Write($"Processing {imagePath}... ");

                bool success = ImageWatermarking.AddWatermark(imagePath, outputPath, config);

                if (success)
                {
                    Console.WriteLine($"✓ Done");
                }
                else
                {
                    Console.WriteLine($"✗ Failed");
                }
            }
            Console.WriteLine();
        }

        /// <summary>
        /// Display usage information.
        /// </summary>
        public static void DisplayUsage()
        {
            Console.WriteLine("\nWatermarking API Usage:");
            Console.WriteLine(new string('-', 60));
            Console.WriteLine("using ArtShield;");
            Console.WriteLine();
            Console.WriteLine("// Create configuration");
            Console.WriteLine("var config = new ImageWatermarking.WatermarkConfig");
            Console.WriteLine("{");
            Console.WriteLine("    Text = \"PROTECTED\",");
            Console.WriteLine("    FontSize = 36,");
            Console.WriteLine("    Opacity = 0.3,");
            Console.WriteLine("    Position = ImageWatermarking.WatermarkPosition.Center,");
            Console.WriteLine("    Color = Colors.White,");
            Console.WriteLine("    RotationAngle = -45");
            Console.WriteLine("};");
            Console.WriteLine();
            Console.WriteLine("// Apply watermark");
            Console.WriteLine("ImageWatermarking.AddWatermark(\"input.jpg\", \"output.jpg\", config);");
            Console.WriteLine(new string('-', 60));
        }

        /// <summary>
        /// Run all examples.
        /// </summary>
        public static void RunAllExamples()
        {
            Console.WriteLine("\n" + new string('=', 60));
            Console.WriteLine("IMAGE WATERMARKING - EXAMPLES");
            Console.WriteLine(new string('=', 60) + "\n");

            Console.WriteLine("NOTE: These examples demonstrate the API.");
            Console.WriteLine("      Actual image files are required to run successfully.\n");

            // Uncomment to run examples:
            // Example1_DefaultWatermark();
            // Example2_CustomWatermark();
            // Example3_TiledWatermark();
            // Example4_CornerWatermark();
            // Example5_BatchWatermark();

            DisplayUsage();
        }
    }
}
