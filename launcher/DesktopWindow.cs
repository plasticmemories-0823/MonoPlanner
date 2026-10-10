using System;
using System.Diagnostics;
using System.Drawing;
using System.IO;
using System.Threading;
using System.Threading.Tasks;
using System.Windows.Forms;
using Microsoft.Web.WebView2.Core;
using Microsoft.Web.WebView2.WinForms;

internal sealed class DesktopWindow : Form
{
    private static readonly Uri Home = new Uri("http://127.0.0.1:3000/");
    private WebView2 view = new WebView2();
    private readonly Panel loading = new Panel();
    private readonly Label message = new Label();
    private readonly Button retry = new Button();
    private readonly ToolStripMenuItem restart = new ToolStripMenuItem("重启服务");
    private readonly ToolStripMenuItem reload = new ToolStripMenuItem("刷新界面");
    private readonly ToolStripStatusLabel status = new ToolStripStatusLabel("正在启动…");
    private readonly System.Windows.Forms.Timer focusTimer = new System.Windows.Forms.Timer();
    private readonly string profile;
    private bool busy;
    private bool stopped;
    private bool browserLost;

    internal DesktopWindow(EventWaitHandle activate, string key)
    {
        Text = "MonoPlanner";
        Font = new Font("Microsoft YaHei UI", 9);
        ClientSize = new Size(1240, 820);
        MinimumSize = new Size(850, 600);
        StartPosition = FormStartPosition.CenterScreen;
        BackColor = Color.White;
        AutoScaleMode = AutoScaleMode.Dpi;
        profile = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData),
                               "MonoPlanner", "WebView2", key);
        string logo = Path.Combine(Program.Root, "web", "public", "brand", "monoplanner.png");
        if (File.Exists(logo))
        {
            using (var bitmap = new Bitmap(logo))
            {
                IntPtr handle = bitmap.GetHicon();
                Icon = (Icon)Icon.FromHandle(handle).Clone();
                DestroyIcon(handle);
            }
        }
        var menu = new MenuStrip { BackColor = Color.White, Padding = new Padding(8, 3, 8, 3) };
        var app = new ToolStripMenuItem("MonoPlanner");
        var exit = new ToolStripMenuItem("退出并停止服务");
        reload.ShortcutKeys = Keys.F5;
        app.DropDownItems.AddRange(new ToolStripItem[] { reload, restart, new ToolStripSeparator(), exit });
        menu.Items.Add(app);
        MainMenuStrip = menu;
        var bar = new StatusStrip { BackColor = Color.White, SizingGrip = false };
        bar.Items.Add(status);
        view.Dock = DockStyle.Fill;
        view.DefaultBackgroundColor = Color.White;
        loading.Dock = DockStyle.Fill;
        var title = new Label { Text = "MonoPlanner", AutoSize = false, Font = new Font(Font.FontFamily, 26),
                                TextAlign = ContentAlignment.MiddleCenter, Dock = DockStyle.Top, Height = 92 };
        message.Text = "正在准备工作区…";
        message.Dock = DockStyle.Top;
        message.Height = 125;
        message.TextAlign = ContentAlignment.MiddleCenter;
        message.Padding = new Padding(35, 0, 35, 0);
        retry.Text = "重新连接";
        retry.Size = new Size(120, 36);
        retry.Visible = false;
        var controls = new FlowLayoutPanel { Dock = DockStyle.Top, Height = 50,
                                            FlowDirection = FlowDirection.LeftToRight };
        controls.Controls.Add(retry);
        loading.Controls.Add(controls);
        loading.Controls.Add(message);
        loading.Controls.Add(title);
        loading.Padding = new Padding(30, 100, 30, 30);
        Controls.Add(view);
        Controls.Add(loading);
        Controls.Add(bar);
        Controls.Add(menu);
        loading.BringToFront();
        retry.Click += delegate { RunService("Start"); };
        restart.Click += delegate { RunService("Restart"); };
        reload.Click += delegate { if (!busy && view.CoreWebView2 != null) view.Reload(); };
        exit.Click += delegate { Close(); };
        Shown += delegate { RunService("Start"); };
        FormClosing += delegate(object sender, FormClosingEventArgs e) {
            if (stopped) return;
            e.Cancel = true;
            if (!busy) RunService("Stop");
        };
        focusTimer.Interval = 300;
        focusTimer.Tick += delegate {
            if (!activate.WaitOne(0)) return;
            if (WindowState == FormWindowState.Minimized) WindowState = FormWindowState.Normal;
            Show();
            Activate();
        };
        focusTimer.Start();
    }

    [System.Runtime.InteropServices.DllImport("user32.dll")]
    private static extern bool DestroyIcon(IntPtr handle);

    private void SetBusy(bool value)
    {
        busy = value;
        restart.Enabled = reload.Enabled = retry.Enabled = view.Enabled = !value;
    }

    private void RunService(string action)
    {
        if (busy) return;
        SetBusy(true);
        status.Text = action == "Stop" ? "正在停止服务…" : "正在准备工作区…";
        ThreadPool.QueueUserWorkItem(delegate {
            string failure = null;
            try { if (Program.RunService(action, true) != 0) failure = Program.Failure(); }
            catch (Exception error) { failure = error.Message; }
            if (IsDisposed) return;
            BeginInvoke(new Action(delegate {
                if (failure != null)
                {
                    SetBusy(false);
                    status.Text = "操作未完成";
                    if (view.CoreWebView2 == null) ShowError(failure);
                    else MessageBox.Show(this, failure, "MonoPlanner", MessageBoxButtons.OK, MessageBoxIcon.Information);
                }
                else if (action == "Stop") { stopped = true; Close(); }
                else OpenWorkspace();
            }));
        });
    }

    private void ShowError(string error)
    {
        loading.Visible = true;
        loading.BringToFront();
        message.Text = error;
        retry.Visible = true;
        status.Text = "工作区暂未连接";
    }

    private void OpenWorkspace()
    {
        if (browserLost)
        {
            int position = Controls.GetChildIndex(view);
            Controls.Remove(view);
            view.Dispose();
            view = new WebView2 { Dock = DockStyle.Fill, DefaultBackgroundColor = Color.White };
            Controls.Add(view);
            Controls.SetChildIndex(view, position);
            browserLost = false;
        }
        if (view.CoreWebView2 != null)
        {
            SetBusy(false);
            try { view.Reload(); }
            catch (Exception error) { ShowError(error.Message); }
            return;
        }
        var scheduler = TaskScheduler.FromCurrentSynchronizationContext();
        try
        {
            string browser = null;
            try { CoreWebView2Environment.GetAvailableBrowserVersionString(); }
            catch(WebView2RuntimeNotFoundException) {
                var candidates=Directory.GetDirectories(Path.Combine(Program.Runtime,"webview"));
                if(candidates.Length!=1) throw new IOException("备用 WebView2 运行时缺失或不明确");
                browser=candidates[0];
            }
            if(Environment.GetEnvironmentVariable("MONOPLANER_FIXED_WEBVIEW")=="1") {
                browser=Directory.GetDirectories(Path.Combine(Program.Runtime,"webview"))[0];
            }
            CoreWebView2Environment.CreateAsync(browser, profile).ContinueWith(environment => {
                if (environment.IsFaulted) { SetBusy(false); ShowError(environment.Exception.GetBaseException().Message); return; }
                try
                {
                    view.EnsureCoreWebView2Async(environment.Result).ContinueWith(initialized => {
                        SetBusy(false);
                        if (initialized.IsFaulted) { ShowError(initialized.Exception.GetBaseException().Message); return; }
                        try { ConfigureBrowser(); view.Source = Home; }
                        catch (Exception error) { ShowError(error.Message); }
                    }, scheduler);
                }
                catch (Exception error) { SetBusy(false); ShowError(error.Message); }
            }, scheduler);
        }
        catch (Exception error) { SetBusy(false); ShowError(error.Message); }
    }

    private static bool IsWorkspace(string address)
    {
        Uri target;
        return Uri.TryCreate(address, UriKind.Absolute, out target) &&
               target.Scheme == Home.Scheme && target.Host == Home.Host && target.Port == Home.Port;
    }

    private void OpenExternal(string address)
    {
        Uri target;
        if (!Uri.TryCreate(address, UriKind.Absolute, out target) ||
            (target.Scheme != "http" && target.Scheme != "https")) return;
        try { Process.Start(new ProcessStartInfo(target.AbsoluteUri) { UseShellExecute = true }); }
        catch (Exception error) { status.Text = "链接未打开：" + error.Message; }
    }

    private void ConfigureBrowser()
    {
        view.CoreWebView2.Settings.AreHostObjectsAllowed = false;
        view.CoreWebView2.Settings.IsWebMessageEnabled = false;
        view.CoreWebView2.Settings.AreDevToolsEnabled = false;
        view.CoreWebView2.Settings.IsStatusBarEnabled = false;
        view.CoreWebView2.Settings.IsZoomControlEnabled = true;
        view.CoreWebView2.NavigationStarting += delegate(object sender, CoreWebView2NavigationStartingEventArgs e) {
            if (IsWorkspace(e.Uri)) return;
            e.Cancel = true;
            // Only a user-initiated external link can open another application.
            if (e.IsUserInitiated && !e.IsRedirected) OpenExternal(e.Uri);
        };
        view.CoreWebView2.FrameNavigationStarting += delegate(object sender, CoreWebView2NavigationStartingEventArgs e) {
            if (!IsWorkspace(e.Uri)) e.Cancel = true;
        };
        view.CoreWebView2.NewWindowRequested += delegate(object sender, CoreWebView2NewWindowRequestedEventArgs e) {
            e.Handled = true;
            if (e.IsUserInitiated) OpenExternal(e.Uri);
        };
        view.CoreWebView2.PermissionRequested += delegate(object sender, CoreWebView2PermissionRequestedEventArgs e) {
            e.State = CoreWebView2PermissionState.Deny;
        };
        view.CoreWebView2.NavigationCompleted += delegate(object sender, CoreWebView2NavigationCompletedEventArgs e) {
            if (!e.IsSuccess) { ShowError("界面暂未加载完成：" + e.WebErrorStatus + "。可重新连接，任务记录会保留。"); return; }
            loading.Visible = false;
            status.Text = "本机工作区 · 就绪";
        };
        view.CoreWebView2.ProcessFailed += delegate(object sender, CoreWebView2ProcessFailedEventArgs e) {
            browserLost = e.ProcessFailedKind == CoreWebView2ProcessFailedKind.BrowserProcessExited;
            ShowError("界面进程已中断。可重新连接，后台任务与记录保留。");
        };
    }

    protected override void Dispose(bool disposing)
    {
        if (disposing) focusTimer.Dispose();
        base.Dispose(disposing);
    }
}
