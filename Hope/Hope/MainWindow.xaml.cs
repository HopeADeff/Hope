using System;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
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

        private void ProtectionMethod_Changed(object sender, System.Windows.Controls.SelectionChangedEventArgs e)
        {
            if (ProtectionMethodCombo == null) return;
            
            bool isGlazeMethod = ProtectionMethodCombo.SelectedIndex == 1;
            
            if (TargetLabel != null) TargetLabel.Visibility = isGlazeMethod ? Visibility.Collapsed : Visibility.Visible;
            if (TargetInput != null) TargetInput.Visibility = isGlazeMethod ? Visibility.Collapsed : Visibility.Visible;
            if (StyleLabel != null) StyleLabel.Visibility = isGlazeMethod ? Visibility.Visible : Visibility.Collapsed;
            if (TargetStyleCombo != null) TargetStyleCombo.Visibility = isGlazeMethod ? Visibility.Visible : Visibility.Collapsed;
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
            string engineExePath = Path.Combine(appFolder, "engine.exe");
            string engineScriptPath = Path.Combine(appFolder, "engine.py");
            string engineInterpreter = Environment.GetEnvironmentVariable("ENGINE_PYTHON") ?? "python";

            bool scriptAvailable = File.Exists(engineScriptPath);
            bool exeAvailable = File.Exists(engineExePath);

            if (!scriptAvailable && !exeAvailable)
            {
                MessageBox.Show($"Engine file not found in: {appFolder}\nPlace engine file there or set ENGINE_PYTHON environment variable.");
                return;
            }


            string target = TargetInput.Text;
            double intensity = IntensitySlider.Value;
            int iterations = (int)IterationSlider.Value;
            int quality = (int)QualitySlider.Value;
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

            string targetFilePath = null;
            if (scriptAvailable)
            {
                targetFilePath = Path.Combine(Path.GetTempPath(), $"{fileNameNoExt}_target_{Guid.NewGuid()}.txt");
                File.WriteAllText(targetFilePath, target, Encoding.UTF8);
            }

            string methodName = useGlazeMethod ? "Glaze-Style" : "Adversarial";
            StatusText.Text = $"Starting {methodName} (Intensity: {intensity:F2}, Iterations: {iterations}, Quality: {quality})...";
            RunBtn.IsEnabled = false;
            ProgBar.IsIndeterminate = true;
            ProgBar.Value = 0;

            string stdErr = string.Empty;
            int exitCode = -1;

            await Task.Run(() =>
            {
                ProcessStartInfo start;

                if (scriptAvailable)
                {
                    string args = $"\"{engineScriptPath}\" --input \"{selectedFilePath}\" --output \"{outputPath}\"";
                    
                    if (useGlazeMethod)
                    {
                        args += $" --target-style {targetStyle}";
                    }
                    else
                    {
                        args += $" --target-file \"{targetFilePath}\"";
                    }
                    
                    args += $" --intensity {intensity.ToString(CultureInfo.InvariantCulture)}";
                    args += $" --iterations {iterations}";
                    args += $" --output-quality {quality}";
                    
                    start = new ProcessStartInfo
                    {
                        FileName = engineInterpreter,
                        Arguments = args,
                        UseShellExecute = false,
                        RedirectStandardOutput = true,
                        RedirectStandardError = true,
                        CreateNoWindow = true,
                        StandardOutputEncoding = Encoding.UTF8,
                        StandardErrorEncoding = Encoding.UTF8
                    };
                }
                else
                {
                    string args = $"--input \"{selectedFilePath}\" --output \"{outputPath}\"";
                    
                    if (useGlazeMethod)
                    {
                        args += $" --target-style {targetStyle}";
                    }
                    else
                    {
                        args += $" --target \"{target}\"";
                    }
                    
                    args += $" --intensity {intensity.ToString(CultureInfo.InvariantCulture)}";
                    args += $" --iterations {iterations}";
                    args += $" --output-quality {quality}";
                    
                    start = new ProcessStartInfo
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
                }

                using (var process = new Process { StartInfo = start, EnableRaisingEvents = true })
                {
                    var outputBuilder = new StringBuilder();
                    var errorBuilder = new StringBuilder();

                    process.OutputDataReceived += (s, ea) =>
                    {
                        if (ea.Data == null) return;
                        outputBuilder.AppendLine(ea.Data);
                        if (ea.Data.StartsWith("STATUS:"))
                        {
                            string status = ea.Data.Length > 7 ? ea.Data.Substring(7).Trim() : string.Empty;
                            Dispatcher.Invoke(() => StatusText.Text = status);
                            
                            // Update progress bar based on iteration status
                            if (status.Contains("Iter "))
                            {
                                try
                                {
                                    var match = System.Text.RegularExpressions.Regex.Match(status, @"Iter (\d+)/(\d+)");
                                    if (match.Success)
                                    {
                                        int current = int.Parse(match.Groups[1].Value);
                                        int total = int.Parse(match.Groups[2].Value);
                                        Dispatcher.Invoke(() =>
                                        {
                                            ProgBar.IsIndeterminate = false;
                                            ProgBar.Maximum = total;
                                            ProgBar.Value = current;
                                        });
                                    }
                                }
                                catch { /* ignore parsing errors */ }
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

            // Reset progress bar
            ProgBar.IsIndeterminate = false;
            ProgBar.Value = 0;

            if (!string.IsNullOrEmpty(targetFilePath) && File.Exists(targetFilePath))
            {
                try { File.Delete(targetFilePath); } catch (Exception ex) { System.Diagnostics.Debug.WriteLine($"Failed to delete temp file: {ex.Message}"); }
            }

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
                string dir = Path.GetDirectoryName(outputPath) ?? string.Empty;
                string[] files = Array.Empty<string>();
                try
                {
                    files = Directory.GetFiles(dir);
                }
                catch { /* ignore directory enumeration errors for diagnostics */ }

                var listing = new StringBuilder();
                foreach (var f in files) listing.AppendLine(f);

                MessageBox.Show($"Expected output file not found: {outputPath}\n\nEngine stderr:\n{stdErr}\n\nDirectory listing:\n{listing}");
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
                
                string methodInfo = useGlazeMethod ? $"Glaze ({targetStyle})" : "Adversarial";
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