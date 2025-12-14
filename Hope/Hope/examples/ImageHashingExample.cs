using System;
using ArtShield;

namespace ArtShield.Examples
{
    /// <summary>
    /// Example usage of the ImageHashing class.
    /// This class demonstrates various hashing scenarios.
    /// </summary>
    public class ImageHashingExample
    {
        /// <summary>
        /// Example 1: Compute hash for a single image.
        /// </summary>
        public static void Example1_ComputeHash()
        {
            Console.WriteLine("Example 1: Compute Image Hash");
            Console.WriteLine("=" + new string('=', 59));

            string imagePath = "sample_image.jpg";

            try
            {
                var hasher = new ImageHashing();

                string ahash = hasher.ComputeHash(imagePath, ImageHashing.HashMethod.AverageHash);
                string dhash = hasher.ComputeHash(imagePath, ImageHashing.HashMethod.DifferenceHash);
                string phash = hasher.ComputeHash(imagePath, ImageHashing.HashMethod.PerceptualHash);

                Console.WriteLine($"Image: {imagePath}\n");
                Console.WriteLine($"Average Hash:    {ahash}");
                Console.WriteLine($"Difference Hash: {dhash}");
                Console.WriteLine($"Perceptual Hash: {phash}");
                Console.WriteLine("\n✓ Hashes computed successfully");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"✗ Error: {ex.Message}");
            }
            Console.WriteLine();
        }

        /// <summary>
        /// Example 2: Compare two images.
        /// </summary>
        public static void Example2_CompareImages()
        {
            Console.WriteLine("Example 2: Compare Two Images");
            Console.WriteLine("=" + new string('=', 59));

            string image1 = "original.jpg";
            string image2 = "modified.jpg";

            try
            {
                var hasher = new ImageHashing();

                string hash1 = hasher.ComputeHash(image1, ImageHashing.HashMethod.PerceptualHash);
                string hash2 = hasher.ComputeHash(image2, ImageHashing.HashMethod.PerceptualHash);

                double similarity = ImageHashing.CompareHashes(hash1, hash2);
                int distance = ImageHashing.HammingDistance(hash1, hash2);

                Console.WriteLine($"Image 1: {image1}");
                Console.WriteLine($"Hash 1:  {hash1}\n");
                Console.WriteLine($"Image 2: {image2}");
                Console.WriteLine($"Hash 2:  {hash2}\n");
                Console.WriteLine($"Similarity: {similarity:P2}");
                Console.WriteLine($"Hamming Distance: {distance} bits");
                Console.WriteLine($"Similar: {ImageHashing.AreSimilar(hash1, hash2, 0.9)}");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"✗ Error: {ex.Message}");
            }
            Console.WriteLine();
        }

        /// <summary>
        /// Example 3: Save and verify hash.
        /// </summary>
        public static void Example3_SaveAndVerify()
        {
            Console.WriteLine("Example 3: Save and Verify Hash");
            Console.WriteLine("=" + new string('=', 59));

            string originalImage = "protected_image.jpg";
            string hashFile = "image_hash.json";
            string testImage = "test_image.jpg";

            try
            {
                var hasher = new ImageHashing();

                // Save hash
                hasher.SaveHashToFile(originalImage, hashFile);
                Console.WriteLine($"✓ Hash saved to {hashFile}");

                // Verify later
                var (isMatch, similarity) = hasher.VerifyImage(testImage, hashFile, threshold: 0.9);

                Console.WriteLine($"\nVerifying: {testImage}");
                Console.WriteLine($"Similarity: {similarity:P2}");
                Console.WriteLine($"Match: {(isMatch ? "YES ✓" : "NO ✗")}");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"✗ Error: {ex.Message}");
            }
            Console.WriteLine();
        }

        /// <summary>
        /// Example 4: Detect unauthorized modifications.
        /// </summary>
        public static void Example4_DetectModifications()
        {
            Console.WriteLine("Example 4: Detect Unauthorized Modifications");
            Console.WriteLine("=" + new string('=', 59));

            string referencePath = "reference_image.jpg";

            var testCases = new[]
            {
                ("original.jpg", "Original image"),
                ("slightly_modified.jpg", "Slightly modified"),
                ("heavily_modified.jpg", "Heavily modified"),
                ("unauthorized_copy.jpg", "Unauthorized copy")
            };

            try
            {
                var hasher = new ImageHashing();
                string referenceHash = hasher.ComputeHash(referencePath, ImageHashing.HashMethod.PerceptualHash);

                Console.WriteLine($"Reference: {referencePath}");
                Console.WriteLine($"Hash: {referenceHash}\n");

                foreach (var (testPath, description) in testCases)
                {
                    try
                    {
                        string testHash = hasher.ComputeHash(testPath, ImageHashing.HashMethod.PerceptualHash);
                        double similarity = ImageHashing.CompareHashes(referenceHash, testHash);
                        string status = similarity >= 0.9 ? "✓ MATCH" : "✗ DIFFERENT";

                        Console.WriteLine($"{description,-30} | Similarity: {similarity:P2} | {status}");
                    }
                    catch
                    {
                        Console.WriteLine($"{description,-30} | ⚠ File not found");
                    }
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"✗ Error: {ex.Message}");
            }
            Console.WriteLine();
        }

        /// <summary>
        /// Example 5: Batch hashing for duplicate detection.
        /// </summary>
        public static void Example5_BatchHashing()
        {
            Console.WriteLine("Example 5: Batch Hashing & Duplicate Detection");
            Console.WriteLine("=" + new string('=', 59));

            string[] images = { "image1.jpg", "image2.jpg", "image3.jpg", "image4.jpg" };

            try
            {
                var hasher = new ImageHashing();
                var hashes = new System.Collections.Generic.Dictionary<string, string>();

                Console.WriteLine("Computing hashes...\n");

                foreach (string imagePath in images)
                {
                    try
                    {
                        string hash = hasher.ComputeHash(imagePath, ImageHashing.HashMethod.PerceptualHash);
                        hashes[imagePath] = hash;
                        Console.WriteLine($"✓ {imagePath,-30} | {hash}");
                    }
                    catch
                    {
                        Console.WriteLine($"✗ {imagePath,-30} | Error");
                    }
                }

                // Find duplicates
                Console.WriteLine("\n" + new string('-', 60));
                Console.WriteLine("Duplicate Detection:");

                bool foundDuplicates = false;
                var checked_pairs = new System.Collections.Generic.HashSet<(string, string)>();

                foreach (var kvp1 in hashes)
                {
                    foreach (var kvp2 in hashes)
                    {
                        if (kvp1.Key != kvp2.Key &&
                            !checked_pairs.Contains((kvp1.Key, kvp2.Key)) &&
                            !checked_pairs.Contains((kvp2.Key, kvp1.Key)))
                        {
                            checked_pairs.Add((kvp1.Key, kvp2.Key));

                            if (ImageHashing.AreSimilar(kvp1.Value, kvp2.Value, 0.95))
                            {
                                double similarity = ImageHashing.CompareHashes(kvp1.Value, kvp2.Value);
                                Console.WriteLine($"  {kvp1.Key} ≈ {kvp2.Key} ({similarity:P2})");
                                foundDuplicates = true;
                            }
                        }
                    }
                }

                if (!foundDuplicates)
                {
                    Console.WriteLine("  No duplicates found");
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"✗ Error: {ex.Message}");
            }
            Console.WriteLine();
        }

        /// <summary>
        /// Display usage information.
        /// </summary>
        public static void DisplayUsage()
        {
            Console.WriteLine("\nImage Hashing API Usage:");
            Console.WriteLine(new string('-', 60));
            Console.WriteLine("using ArtShield;");
            Console.WriteLine();
            Console.WriteLine("// Create hasher");
            Console.WriteLine("var hasher = new ImageHashing();");
            Console.WriteLine();
            Console.WriteLine("// Compute hash");
            Console.WriteLine("string hash1 = hasher.ComputeHash(\"image1.jpg\",");
            Console.WriteLine("    ImageHashing.HashMethod.PerceptualHash);");
            Console.WriteLine("string hash2 = hasher.ComputeHash(\"image2.jpg\",");
            Console.WriteLine("    ImageHashing.HashMethod.PerceptualHash);");
            Console.WriteLine();
            Console.WriteLine("// Compare");
            Console.WriteLine("double similarity = ImageHashing.CompareHashes(hash1, hash2);");
            Console.WriteLine("Console.WriteLine($\"Similarity: {similarity:P}\");");
            Console.WriteLine(new string('-', 60));
        }

        /// <summary>
        /// Run all examples.
        /// </summary>
        public static void RunAllExamples()
        {
            Console.WriteLine("\n" + new string('=', 60));
            Console.WriteLine("IMAGE HASHING - EXAMPLES");
            Console.WriteLine(new string('=', 60) + "\n");

            Console.WriteLine("NOTE: These examples demonstrate the API.");
            Console.WriteLine("      Actual image files are required to run successfully.\n");

            // Uncomment to run examples:
            // Example1_ComputeHash();
            // Example2_CompareImages();
            // Example3_SaveAndVerify();
            // Example4_DetectModifications();
            // Example5_BatchHashing();

            DisplayUsage();
        }
    }
}
