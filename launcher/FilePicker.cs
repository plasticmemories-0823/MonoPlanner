using System;
using System.Text;
using System.Windows.Forms;

internal static class FilePicker
{
    [STAThread]
    public static int Main(string[] arguments)
    {
        Console.OutputEncoding = new UTF8Encoding(false);
        Application.EnableVisualStyles();

        // 文件夹选择模式。
        if (arguments.Length == 1 && arguments[0] == "directory")
        {
            using (var folder = new FolderBrowserDialog
            {
                Description = "选择本对话可读取的资料文件夹"
            })
            {
                if (folder.ShowDialog() == DialogResult.OK)
                    Console.WriteLine(folder.SelectedPath);
            }

            return 0;
        }

        // 默认使用单文件选择模式。
        using (var dialog = new OpenFileDialog
        {
            Title = "选择本对话可读取的文件",
            CheckFileExists = true,
            Multiselect = false
        })
        {
            if (dialog.ShowDialog() == DialogResult.OK)
                Console.WriteLine(dialog.FileName);
        }

        return 0;
    }
}
