using System;
using System.Diagnostics;
using System.IO;
using System.Reflection;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text;
using System.Threading;
using System.Windows.Forms;

[assembly: System.Reflection.AssemblyVersion("1.0.0.0")]
[assembly: System.Reflection.AssemblyFileVersion("1.0.0.0")]
[assembly: System.Reflection.AssemblyInformationalVersion("1.0.0")]
internal static class Program
{
    // 项目、桌面组件与当前用户数据目录。
    internal static readonly string Root = AppDomain.CurrentDomain.BaseDirectory;
    internal static readonly string Runtime = Path.Combine(Root, "launcher", "runtime");
    internal static readonly string Data = Path.Combine(
        Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData), "MonoPlanner", "Data");

    [DllImport("kernel32.dll", CharSet = CharSet.Unicode, SetLastError = true)]
    private static extern bool SetDllDirectory(string path);

    // 调用服务启动脚本并读取最近一次失败日志。
    internal static int RunService(string action, bool noBrowser)
    {
        string script = Path.Combine(Root, "scripts", "start-app.ps1");
        if (!File.Exists(script))
            throw new FileNotFoundException("请将 MonoPlanner.exe 保留在项目根目录。");
        string shell = Path.Combine(
            Environment.GetFolderPath(Environment.SpecialFolder.System),
            "WindowsPowerShell", "v1.0", "powershell.exe");
        var start = new ProcessStartInfo(shell,
            "-NoLogo -NoProfile -ExecutionPolicy Bypass -File \"" + script + "\" -Action " + action +
            (noBrowser ? " -NoBrowser" : ""));
        start.WorkingDirectory = Root;
        start.UseShellExecute = false;
        start.CreateNoWindow = true;
        start.WindowStyle = ProcessWindowStyle.Hidden;
        using (Process child = Process.Start(start))
        {
            child.WaitForExit();
            return child.ExitCode;
        }
    }

    internal static string Failure()
    {
        string logs = File.Exists(Path.Combine(Root, "portable.json"))
            ? Path.Combine(Data, "launcher-logs")
            : Path.Combine(Root, "validation", "launcher-logs");
        string file = Path.Combine(logs, "last-error.txt");
        return File.Exists(file) ? File.ReadAllText(file) : "操作失败，请查看启动日志：" + logs;
    }

    // 校验入口参数，并协调每个项目唯一的桌面窗口。
    [STAThread]
    private static int Main(string[] args)
    {
        bool quiet = Array.IndexOf(args, "--quiet") >= 0;
        try
        {
            foreach (string arg in args)
                if (arg != "--no-browser" && arg != "--quiet")
                    throw new ArgumentException("支持的参数：--no-browser、--quiet。");
            if (quiet || Array.IndexOf(args, "--no-browser") >= 0)
            {
                int code = RunService("Start", true);
                if (code != 0 && !quiet)
                    MessageBox.Show(Failure(), "MonoPlanner", MessageBoxButtons.OK, MessageBoxIcon.Error);
                return code;
            }

            // 使用项目路径派生窗口互斥量与激活事件的名称。
            string key;
            using (SHA256 hash = SHA256.Create())
                key = BitConverter.ToString(hash.ComputeHash(Encoding.UTF8.GetBytes(Root.TrimEnd('\\').ToLowerInvariant()))).Replace("-", "");
            using (var mutex = new Mutex(false, "Local\\MonoPlanner-Window-" + key))
            using (var activate = new EventWaitHandle(false, EventResetMode.AutoReset, "Local\\MonoPlanner-Activate-" + key))
            {
                bool owned;
                try
                {
                    owned = mutex.WaitOne(0);
                }
                catch (AbandonedMutexException)
                {
                    owned = true;
                }

                if (!owned)
                {
                    activate.Set();
                    return 0;
                }

                try
                {
                    Application.EnableVisualStyles();
                    Application.SetCompatibleTextRenderingDefault(false);

                    // 先安装本地组件解析器，再创建桌面窗口。
                    AppDomain.CurrentDomain.AssemblyResolve += delegate (object sender, ResolveEventArgs e)
                    {
                        string name = new AssemblyName(e.Name).Name;
                        if (name != "Microsoft.Web.WebView2.Core" && name != "Microsoft.Web.WebView2.WinForms")
                            return null;
                        return Assembly.LoadFrom(Path.Combine(Runtime, name + ".dll"));
                    };
                    if (!SetDllDirectory(Runtime))
                        throw new IOException("无法加载桌面组件目录。");
                    OpenDesktop(activate, key);
                }
                finally
                {
                    mutex.ReleaseMutex();
                }
            }

            return 0;
        }
        catch (Exception error)
        {
            if (!quiet)
                MessageBox.Show(error.Message, "MonoPlanner 启动失败", MessageBoxButtons.OK, MessageBoxIcon.Error);
            return 1;
        }
    }

    // 安装本地依赖解析器后，再加载 WebView2 类型。
    [System.Runtime.CompilerServices.MethodImpl(System.Runtime.CompilerServices.MethodImplOptions.NoInlining)]
    private static void OpenDesktop(EventWaitHandle activate, string key)
    {
        Application.Run(new DesktopWindow(activate, key));
    }
}
