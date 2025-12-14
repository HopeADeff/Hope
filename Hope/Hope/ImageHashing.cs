using System;
using System.IO;
using System.Text;
using System.Windows.Media.Imaging;
using System.Text.Json;

namespace ArtShield
{
    public class ImageHashing
    {
        private readonly int hashSize;
        /// <param name="hashSize">Size of the hash grid </param>
        public ImageHashing(int hashSize = 8)
        {
            this.hashSize = hashSize;
        }
        public enum HashMethod
        {
            AverageHash,
            DifferenceHash,
            PerceptualHash
        }


        /// <param name="imagePath">Path to the img</param>
        /// <param name="method">Hash method</param>
        public string ComputeHash(string imagePath, HashMethod method = HashMethod.PerceptualHash)
        {
            return method switch
            {
                HashMethod.AverageHash => ComputeAverageHash(imagePath),
                HashMethod.DifferenceHash => ComputeDifferenceHash(imagePath),
                HashMethod.PerceptualHash => ComputePerceptualHash(imagePath),
                _ => throw new ArgumentException($"Unknown hash method: {method}")
            };
        }
        private string ComputeAverageHash(string imagePath)
        {
            byte[,] pixels = LoadAndResizeGrayscale(imagePath, hashSize, hashSize);
            double sum = 0;
            for (int i = 0; i < hashSize; i++)
            {
                for (int j = 0; j < hashSize; j++)
                {
                    sum += pixels[i, j];
                }
            }
            double average = sum / (hashSize * hashSize);
            bool[] bits = new bool[hashSize * hashSize];
            int index = 0;
            for (int i = 0; i < hashSize; i++)
            {
                for (int j = 0; j < hashSize; j++)
                {
                    bits[index++] = pixels[i, j] > average;
                }
            }

            return BitsToHex(bits);
        }
        private string ComputeDifferenceHash(string imagePath)
        {
            byte[,] pixels = LoadAndResizeGrayscale(imagePath, hashSize + 1, hashSize);
            bool[] bits = new bool[hashSize * hashSize];
            int index = 0;
            for (int i = 0; i < hashSize; i++)
            {
                for (int j = 0; j < hashSize; j++)
                {
                    bits[index++] = pixels[i, j + 1] > pixels[i, j];
                }
            }
            return BitsToHex(bits);
        }
        private string ComputePerceptualHash(string imagePath)
        {
            byte[,] pixels = LoadAndResizeGrayscale(imagePath, 32, 32);
            double[,] imageData = new double[32, 32];
            for (int i = 0; i < 32; i++)
            {
                for (int j = 0; j < 32; j++)
                {
                    imageData[i, j] = pixels[i, j];
                }
            }
            double[,] dct = ComputeDCT(imageData);
            double[] lowFreq = new double[hashSize * hashSize];
            int index = 0;
            for (int i = 0; i < hashSize; i++)
            {
                for (int j = 0; j < hashSize; j++)
                {
                    lowFreq[index++] = dct[i, j];
                }
            }
            Array.Sort(lowFreq);
            double median = lowFreq[lowFreq.Length / 2];
            bool[] bits = new bool[hashSize * hashSize];
            index = 0;
            for (int i = 0; i < hashSize; i++)
            {
                for (int j = 0; j < hashSize; j++)
                {
                    bits[index++] = dct[i, j] > median;
                }
            }

            return BitsToHex(bits);
        }
        private byte[,] LoadAndResizeGrayscale(string imagePath, int width, int height)
        {
            BitmapImage bitmap = new BitmapImage();
            bitmap.BeginInit();
            bitmap.CacheOption = BitmapCacheOption.OnLoad;
            bitmap.UriSource = new Uri(imagePath, UriKind.Absolute);
            bitmap.DecodePixelWidth = width;
            bitmap.DecodePixelHeight = height;
            bitmap.EndInit();
            bitmap.Freeze();
            FormatConvertedBitmap grayBitmap = new FormatConvertedBitmap(
                bitmap, 
                System.Windows.Media.PixelFormats.Gray8, 
                null, 
                0
            );

            byte[] pixels = new byte[width * height];
            grayBitmap.CopyPixels(pixels, width, 0);
            byte[,] result = new byte[height, width];
            for (int i = 0; i < height; i++)
            {
                for (int j = 0; j < width; j++)
                {
                    result[i, j] = pixels[i * width + j];
                }
            }

            return result;
        }
        private double[,] ComputeDCT(double[,] input)
        {
            int N = input.GetLength(0);
            int M = input.GetLength(1);
            double[,] output = new double[N, M];

            for (int u = 0; u < N; u++)
            {
                for (int v = 0; v < M; v++)
                {
                    double sum = 0.0;
                    for (int i = 0; i < N; i++)
                    {
                        for (int j = 0; j < M; j++)
                        {
                            sum += input[i, j] *
                                   Math.Cos((2 * i + 1) * u * Math.PI / (2 * N)) *
                                   Math.Cos((2 * j + 1) * v * Math.PI / (2 * M));
                        }
                    }

                    double cu = (u == 0) ? 1.0 / Math.Sqrt(N) : Math.Sqrt(2.0 / N);
                    double cv = (v == 0) ? 1.0 / Math.Sqrt(M) : Math.Sqrt(2.0 / M);
                    output[u, v] = cu * cv * sum;
                }
            }

            return output;
        }
        private string BitsToHex(bool[] bits)
        {
            StringBuilder binary = new StringBuilder();
            foreach (bool bit in bits)
            {
                binary.Append(bit ? '1' : '0');
            }
            while (binary.Length % 4 != 0)
            {
                binary.Append('0');
            }
            StringBuilder hex = new StringBuilder();
            for (int i = 0; i < binary.Length; i += 4)
            {
                string chunk = binary.ToString(i, 4);
                int value = Convert.ToInt32(chunk, 2);
                hex.Append(value.ToString("x"));
            }

            return hex.ToString();
        }
        /// <param name="hash1">First hash.</param>
        /// <param name="hash2">Second hash.</param>
        public static int HammingDistance(string hash1, string hash2)
        {
            if (hash1.Length != hash2.Length)
            {
                throw new ArgumentException("Hashes must be the same length");
            }
            long value1 = Convert.ToInt64(hash1, 16);
            long value2 = Convert.ToInt64(hash2, 16);
            long xor = value1 ^ value2;
            int distance = 0;
            while (xor != 0)
            {
                distance += (int)(xor & 1);
                xor >>= 1;
            }

            return distance;
        }

        /// <param name="hash1">First hash</param>
        /// <param name="hash2">Second hash</param>
        /// <param name="maxBits">Maximum number of bits</param>
        public static double CompareHashes(string hash1, string hash2, int maxBits = 64)
        {
            int distance = HammingDistance(hash1, hash2);
            return 1.0 - ((double)distance / maxBits);
        }
        /// <param name="hash1">First hash.</param>
        /// <param name="hash2">Second hash.</param>
        /// <param name="threshold">Similarity threshold</param>
        /// <param name="maxBits">Maximum number of bits</param>
        public static bool AreSimilar(string hash1, string hash2, double threshold = 0.9, int maxBits = 64)
        {
            double similarity = CompareHashes(hash1, hash2, maxBits);
            return similarity >= threshold;
        }
        public class HashData
        {
            public string ImagePath { get; set; } = string.Empty;
            public string Method { get; set; } = string.Empty;
            public string Hash { get; set; } = string.Empty;
            public int HashSize { get; set; }
        }
        /// <param name="imagePath">Path to the img</param>
        /// <param name="outputPath">Path to save the hash</param>
        /// <param name="method">Hash method</param>
        public void SaveHashToFile(string imagePath, string outputPath, HashMethod method = HashMethod.PerceptualHash)
        {
            string hash = ComputeHash(imagePath, method);

            var hashData = new HashData
            {
                ImagePath = Path.GetFullPath(imagePath),
                Method = method.ToString(),
                Hash = hash,
                HashSize = hashSize
            };

            string json = JsonSerializer.Serialize(hashData, new JsonSerializerOptions { WriteIndented = true });
            File.WriteAllText(outputPath, json);
        }
        /// <param name="hashFile">Path to the hash JSON file.</param>
        /// <returns>Hash data.</returns>
        public static HashData LoadHashFromFile(string hashFile)
        {
            string json = File.ReadAllText(hashFile);
            return JsonSerializer.Deserialize<HashData>(json) ?? throw new InvalidOperationException("Failed to deserialize hash data");
        }
        /// <param name="imagePath">Path to the img</param>
        /// <param name="hashFile">Path to the hash</param>
        /// <param name="threshold">Similarity threshold</param>
        /// <returns>Tuple of (isMatch, similarity).</returns>
        public (bool isMatch, double similarity) VerifyImage(string imagePath, string hashFile, double threshold = 0.9)
        {
            HashData hashData = LoadHashFromFile(hashFile);
            if (!Enum.TryParse<HashMethod>(hashData.Method, out HashMethod method))
            {
                method = HashMethod.PerceptualHash;
            }
            string currentHash = ComputeHash(imagePath, method);
            int maxBits = hashData.HashSize * hashData.HashSize;
            double similarity = CompareHashes(hashData.Hash, currentHash, maxBits);
            bool isMatch = similarity >= threshold;

            return (isMatch, similarity);
        }
    }
}
