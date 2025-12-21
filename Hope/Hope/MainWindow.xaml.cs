using System;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Media.Imaging;
using Microsoft.Win32;

namespace ArtShield
{
    public partial class MainWindow : Window
    {
        string selectedFilePath = "";

        public MainWindow()
        {
            InitializeComponent();
        }

        private void ProtectionMethod_Changed(object sender, SelectionChangedEventArgs e)
        {
            if (ProtectionMethodCombo == null) return;
            
            if (TargetInput == null || TargetStyleCombo == null) return;


            int index = ProtectionMethodCombo.SelectedIndex;
            if (index == 0) // Nightshade
            {
                TargetLabel.Text = "Input";
                TargetInput.Visibility = Visibility.Visible;
                TargetStyleCombo.Visibility = Visibility.Collapsed;
                if (string.IsNullOrWhiteSpace(TargetInput.Text)) TargetInput.Text = "noise";
            }
            else if (index == 1) // Glaze
            {
                TargetLabel.Text = "Target Style";
                TargetInput.Visibility = Visibility.Collapsed;
                TargetStyleCombo.Visibility = Visibility.Visible;
            }
            else // Noise
            {
                TargetLabel.Text = "Target Description";
                TargetInput.Visibility = Visibility.Visible;
                TargetStyleCombo.Visibility = Visibility.Collapsed;
            }
        }

        private void SelectImage_Click(object sender, RoutedEventArgs e)
        {
            OpenFileDialog openFileDialog = new OpenFileDialog();
            openFileDialog.Filter = "Ảnh (*.png;*.jpg;*.bmp)|*.png;*.jpg;*.bmp";
            if (openFileDialog.ShowDialog() == true)
            {
                selectedFilePath = openFileDialog.FileName;
                PreviewImage.Source = new BitmapImage(new Uri(selectedFilePath));
                
                var fileInfo = new FileInfo(selectedFilePath);
                double fileSizeKB = fileInfo.Length / 1024.0;
                FileSizeText.Text = $"Size: {fileSizeKB:F1} KB";
            }
        }

        private async void RunProtection_Click(object sender, RoutedEventArgs e)
        {
            if (string.IsNullOrEmpty(selectedFilePath))
            {
                MessageBox.Show("Please select an image first!");
                return;
            }

            string directory = Path.GetDirectoryName(selectedFilePath) ?? Environment.CurrentDirectory;
            string fileNameNoExt = Path.GetFileNameWithoutExtension(selectedFilePath);
            string ext = Path.GetExtension(selectedFilePath);
            string outputFileName = $"{fileNameNoExt}_protected{ext}";
            string outputPath = Path.Combine(directory, outputFileName);
            string appFolder = AppDomain.CurrentDomain.BaseDirectory;
            string engineExePath = Path.Combine(appFolder, "engine", "engine.exe");
            
            // STRICT CHECK: Only use engine.exe
            if (!File.Exists(engineExePath))
            {
                MessageBox.Show($"Engine NOT found!\nLooking for: {engineExePath}\n\nCurrent Directory: {Environment.CurrentDirectory}\nBase Directory: {appFolder}");
                return;
            }

            // Get UI Variables
            string target = TargetInput.Text;
            double intensity = IntensitySlider.Value;
            int iterations = 100; // Default, will be overridden by Render Quality
            
            // Override iterations based on Render Quality selection
            int renderQuality = (int)RenderQualitySlider.Value;
            switch (renderQuality)
            {
                case 1: iterations = 50; break;   // Faster
                case 2: iterations = 100; break;  // DEFAULT
                case 3: iterations = 200; break;  // Slower
                case 4: iterations = 250; break;  // Slowest
            }
            
            int quality = (int)QualitySlider.Value;
            bool useNightshadeMethod = ProtectionMethodCombo.SelectedIndex == 0;
            bool useGlazeMethod = ProtectionMethodCombo.SelectedIndex == 1;
            string targetStyle = "abstract"; 
            
            if (useGlazeMethod && TargetStyleCombo.SelectedItem != null)
            {
                var selectedItem = TargetStyleCombo.SelectedItem as System.Windows.Controls.ComboBoxItem;
                if (selectedItem?.Tag != null)
                {
                    targetStyle = selectedItem.Tag?.ToString() ?? "abstract";
                }
            }

            string methodName = useGlazeMethod ? "Glaze-Style" : (useNightshadeMethod ? "Nightshade" : "Adversarial");
            StatusText.Text = $"Starting {methodName} (Intensity: {intensity:F2}, Iterations: {iterations}, Quality: {quality})...";
            RunBtn.IsEnabled = false;
            ProgBar.IsIndeterminate = true;
            ProgBar.Value = 0;

            string stdErr = string.Empty;
            int exitCode = -1;

            await Task.Run(() =>
            {
                string args = $"--input \"{selectedFilePath}\" --output \"{outputPath}\"";
                
                if (useGlazeMethod)
                {
                    args += $" --target-style {targetStyle}";
                }
                else if (useNightshadeMethod)
                {
                     args += " --nightshade";
                     args += " --source-concept artwork --target-concept noise";
                }
                else
                {
                    args += $" --target \"{target}\"";
                }
                
                args += $" --intensity {intensity.ToString(CultureInfo.InvariantCulture)}";
                args += $" --iterations {iterations}";
                args += $" --output-quality {quality}";
                
                var start = new ProcessStartInfo
                {
                    FileName = engineExePath,
                    Arguments = args,
                    UseShellExecute = false,
                    RedirectStandardOutput = true,
                    RedirectStandardError = true,
                    CreateNoWindow = true,
                    StandardOutputEncoding = Encoding.UTF8,
                    StandardErrorEncoding = Encoding.UTF8
                };

                using (var process = new Process { StartInfo = start, EnableRaisingEvents = true })
                {
                    var outputBuilder = new StringBuilder();
                    var errorBuilder = new StringBuilder();

                    process.OutputDataReceived += (s, ea) =>
                    {
                        if (ea.Data == null) return;
                        outputBuilder.AppendLine(ea.Data);
                        bool isStatus = ea.Data.StartsWith("STATUS:");
                        bool isLoading = ea.Data.Contains("Loading models");
                        bool isDevice = ea.Data.Contains("Device:");
                        
                        if (isStatus || isLoading || isDevice)
                        {
                            string status = isStatus ? (ea.Data.Length > 7 ? ea.Data.Substring(7).Trim() : string.Empty) : ea.Data;
                            Dispatcher.Invoke(() => StatusText.Text = status);
                            if (status.Contains("Iter "))
                            {
                                try
                                {
                                    var match = System.Text.RegularExpressions.Regex.Match(status, @"Iter (\d+)/(\d+)");
                                    if (match.Success)
                                    {
                                        int current = int.Parse(match.Groups[1].Value);
                                        int total = int.Parse(match.Groups[2].Value);
                                        double percent = (double)current / total * 100.0;
                                        Dispatcher.Invoke(() =>
                                        {
                                            StatusText.Text = $"{status} ({percent:F0}%)";
                                            ProgBar.IsIndeterminate = false;
                                            ProgBar.Maximum = total;
                                            ProgBar.Value = current;
                                        });
                                    }
                                }
                                catch { }
                            }
                        }
                    };

                    process.ErrorDataReceived += (s, ea) =>
                    {
                        if (ea.Data == null) return;
                        errorBuilder.AppendLine(ea.Data);
                        System.Diagnostics.Debug.WriteLine($"[Engine Stderr]: {ea.Data}");
                    };

                    process.Start();
                    process.BeginOutputReadLine();
                    process.BeginErrorReadLine();
                    process.WaitForExit();

                    exitCode = process.ExitCode;
                    stdErr = errorBuilder.ToString();
                }
            });
            
            ProgBar.IsIndeterminate = false;
            ProgBar.Value = 0;

            if (exitCode != 0)
            {
                StatusText.Text = $"Engine error (code {exitCode})";
                RunBtn.IsEnabled = true;
                MessageBox.Show($"Engine exited with code {exitCode}.\n\nStderr:\n{stdErr}");
                return;
            }

            if (!File.Exists(outputPath))
            {
                StatusText.Text = "Output file not found.";
                RunBtn.IsEnabled = true;
                MessageBox.Show($"Expected output file not found: {outputPath}\n\nEngine stderr:\n{stdErr}");
                return;
            }

            try
            {
                var bitmap = new BitmapImage();
                bitmap.BeginInit();
                bitmap.CacheOption = BitmapCacheOption.OnLoad;
                bitmap.UriSource = new Uri(outputPath);
                bitmap.EndInit();
                bitmap.Freeze();

                PreviewImage.Source = bitmap;
                StatusText.Text = "Protection complete!";
                
                var outputFileInfo = new FileInfo(outputPath);
                double outputSizeKB = outputFileInfo.Length / 1024.0;
                var inputFileInfo = new FileInfo(selectedFilePath);
                double inputSizeKB = inputFileInfo.Length / 1024.0;
                FileSizeText.Text = $"Size: {inputSizeKB:F1} KB → {outputSizeKB:F1} KB";
                
                string methodInfo = useGlazeMethod ? $"Glaze ({targetStyle})" : (useNightshadeMethod ? "Nightshade" : "Adversarial");
                MessageBox.Show($"File saved at: {outputPath}\n\nSettings:\n- Method: {methodInfo}\n- Intensity: {intensity:F2}\n- Iterations: {iterations}\n- Quality: {quality}\n- Size: {inputSizeKB:F1}KB → {outputSizeKB:F1}KB");
            }
            catch (Exception ex)
            {
                StatusText.Text = "Could not load output image.";
                MessageBox.Show($"File exists but could not be loaded: {ex.Message}");
            }
            finally
            {
                RunBtn.IsEnabled = true;
            }
        }
    }
}