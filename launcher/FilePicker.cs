using System;
using System.Text;
using System.Windows.Forms;

internal static class FilePicker
{
    [STAThread] public static int Main()
    {
        Console.OutputEncoding=new UTF8Encoding(false);
        Application.EnableVisualStyles();
        using(var dialog=new OpenFileDialog { Title="选择本对话可读取的文件", CheckFileExists=true, Multiselect=false }) {
            if(dialog.ShowDialog()==DialogResult.OK) Console.WriteLine(dialog.FileName);
        }
        return 0;
    }
}
