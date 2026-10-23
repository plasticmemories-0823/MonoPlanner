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
    private static DesktopLifetime desktop;
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
        if (desktop != null)
        {
            start.EnvironmentVariables["MONOPLANER_DESKTOP_JOB"] = desktop.Name;
            start.EnvironmentVariables["MONOPLANER_CLOSE_SIGNAL"] = desktop.SignalName;
            start.EnvironmentVariables["MONOPLANER_DESKTOP_HEARTBEAT"] = desktop.HeartbeatName;
        }
        string gateName = desktop == null ? null : desktop.Name + "-Start-" + Guid.NewGuid().ToString("N");
        using (var gate = gateName == null ? null : new EventWaitHandle(false, EventResetMode.ManualReset, gateName))
        {
            if (gateName != null)
                start.EnvironmentVariables["MONOPLANER_START_GATE"] = gateName;
            using (Process child = Process.Start(start))
            {
                if (desktop != null)
                    desktop.Assign(child);
                if (gate != null)
                    gate.Set();
                var elapsed = Stopwatch.StartNew();
                int timeout = action == "Stop" ? 12000 : 210000;
                while (!child.WaitForExit(100))
                {
                    if (action != "Stop" && desktop != null && desktop.Closing)
                    {
                        child.Kill();
                        child.WaitForExit(1000);
                        return 0;
                    }
                    if (elapsed.ElapsedMilliseconds >= timeout)
                    {
                        child.Kill();
                        child.WaitForExit(1000);
                        throw new TimeoutException("服务控制已超时；本实例的进程将随窗口退出，未确认的操作保留待核对记录。");
                    }
                }
                return child.ExitCode;
            }
        }
    }

    internal static void RequestClose()
    {
        if (desktop != null)
            desktop.RequestClose();
    }

    internal static void PulseDesktop()
    {
        if (desktop != null)
            desktop.Pulse();
    }

    internal static string Failure()
    {
        string logs = File.Exists(Path.Combine(Root, "portable.json"))
            ? Path.Combine(Data, "launcher-logs")
            : Path.Combine(Root, "validation", "launcher-logs");
        string file = Path.Combine(logs, "last-error.txt");
        return File.Exists(file) ? File.ReadAllText(file) : "操作失败，请查看启动日志：" + logs;
    }

    internal static void RecordExitFailure(string failure)
    {
        try
        {
            string logs = File.Exists(Path.Combine(Root, "portable.json"))
                ? Path.Combine(Data, "launcher-logs")
                : Path.Combine(Root, "validation", "launcher-logs");
            Directory.CreateDirectory(logs);
            File.WriteAllText(Path.Combine(logs, "shutdown-error.txt"),
                DateTime.UtcNow.ToString("o") + "\n" + failure, Encoding.UTF8);
        }
        catch (IOException) { }
        catch (UnauthorizedAccessException) { }
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
                    using (desktop = new DesktopLifetime(key))
                        OpenDesktop(activate, key);
                    desktop = null;
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

// 窗口只拥有服务控制器及其子进程；外部打开的应用不加入此作业对象。
internal sealed class DesktopLifetime : IDisposable
{
    internal readonly string Name;
    internal readonly string SignalName;
    internal readonly string HeartbeatName;
    private readonly EventWaitHandle signal;
    private readonly EventWaitHandle heartbeat;
    private IntPtr job;
    internal bool Closing { get { return signal.WaitOne(0); } }

    [StructLayout(LayoutKind.Sequential)]
    private struct BasicLimits
    {
        public long ProcessTime, JobTime;
        public uint Flags;
        public UIntPtr MinimumWorkingSet, MaximumWorkingSet;
        public uint ActiveProcessLimit;
        public UIntPtr Affinity;
        public uint PriorityClass, SchedulingClass;
    }

    [StructLayout(LayoutKind.Sequential)]
    private struct IoCounters
    {
        public ulong ReadOperations, WriteOperations, OtherOperations;
        public ulong ReadBytes, WriteBytes, OtherBytes;
    }

    [StructLayout(LayoutKind.Sequential)]
    private struct ExtendedLimits
    {
        public BasicLimits Basic;
        public IoCounters Io;
        public UIntPtr ProcessMemory, JobMemory, PeakProcessMemory, PeakJobMemory;
    }

    [DllImport("kernel32.dll", CharSet = CharSet.Unicode, SetLastError = true)]
    private static extern IntPtr CreateJobObject(IntPtr attributes, string name);
    [DllImport("kernel32.dll", SetLastError = true)]
    private static extern bool SetInformationJobObject(IntPtr job, int kind, IntPtr info, uint size);
    [DllImport("kernel32.dll", SetLastError = true)]
    private static extern bool AssignProcessToJobObject(IntPtr job, IntPtr process);
    [DllImport("kernel32.dll")]
    private static extern bool CloseHandle(IntPtr handle);

    internal DesktopLifetime(string key)
    {
        Name = "Local\\MonoPlanner-Services-" + key + "-" + Process.GetCurrentProcess().Id;
        SignalName = Name + "-Close";
        HeartbeatName = Name + "-Heartbeat";
        signal = new EventWaitHandle(false, EventResetMode.ManualReset, SignalName);
        heartbeat = new EventWaitHandle(true, EventResetMode.AutoReset, HeartbeatName);
        job = CreateJobObject(IntPtr.Zero, Name);
        if (job == IntPtr.Zero)
            throw new System.ComponentModel.Win32Exception(Marshal.GetLastWin32Error());
        var limits = new ExtendedLimits();
        // 结束内部服务；只有受信任的外部应用打开入口显式请求脱离。
        limits.Basic.Flags = 0x2000 | 0x800;
        int size = Marshal.SizeOf(limits);
        IntPtr memory = Marshal.AllocHGlobal(size);
        try
        {
            Marshal.StructureToPtr(limits, memory, false);
            if (!SetInformationJobObject(job, 9, memory, (uint)size))
                throw new System.ComponentModel.Win32Exception(Marshal.GetLastWin32Error());
        }
        catch
        {
            Dispose();
            throw;
        }
        finally
        {
            Marshal.FreeHGlobal(memory);
        }
    }

    internal void Assign(Process process)
    {
        if (!AssignProcessToJobObject(job, process.Handle))
        {
            int error = Marshal.GetLastWin32Error();
            if (!process.HasExited)
                process.Kill();
            throw new System.ComponentModel.Win32Exception(error, "无法接管服务进程，已停止启动。");
        }
    }

    internal void RequestClose() { signal.Set(); }
    internal void Pulse() { if (!Closing) heartbeat.Set(); }

    public void Dispose()
    {
        if (job != IntPtr.Zero)
        {
            CloseHandle(job);
            job = IntPtr.Zero;
        }
        signal.Dispose();
        heartbeat.Dispose();
    }
}
