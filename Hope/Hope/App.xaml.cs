using System.Configuration;
using System.Data;
using System.Windows;
using System.IO;
using System.Threading.Tasks;

namespace Hope
{
    /// <summary>
    /// Interaction logic for App.xaml
    /// </summary>
    public partial class App : Application
    {
        public App()
        {
            AppDomain.CurrentDomain.UnhandledException += (s, e) =>
                LogException((Exception)e.ExceptionObject, "AppDomain.UnhandledException");

            DispatcherUnhandledException += (s, e) =>
            {
                LogException(e.Exception, "DispatcherUnhandledException");
                e.Handled = true; 
            };

            TaskScheduler.UnobservedTaskException += (s, e) =>
            {
                LogException(e.Exception, "TaskScheduler.UnobservedTaskException");
                e.SetObserved();
            };
        }

        private void LogException(Exception ex, string source)
        {
            string logFile = "crash_log.txt";
            string message = $"[{DateTime.Now}] {source}:\n{ex.ToString()}\n\n";
            File.AppendAllText(logFile, message);
            MessageBox.Show($"Application Crashed! Log saved to {logFile}\nError: {ex.Message}", "Critical Error");
        }
    }

}
