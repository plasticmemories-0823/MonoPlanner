using System;
using System.Collections.Generic;
using System.IO;
using System.Reflection;
using System.Runtime.InteropServices;
using System.Text;
using System.Web.Script.Serialization;

internal static class Applications
{
    [ComImport, Guid("2E941141-7F97-4756-BA1D-9DECDE894A3D"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
    interface ActivationManager {
        [PreserveSig] int ActivateApplication([MarshalAs(UnmanagedType.LPWStr)] string id,
            [MarshalAs(UnmanagedType.LPWStr)] string args, uint options, out uint pid);
        [PreserveSig] int ActivateForFile([MarshalAs(UnmanagedType.LPWStr)] string id, IntPtr items,
            [MarshalAs(UnmanagedType.LPWStr)] string verb, out uint pid);
        [PreserveSig] int ActivateForProtocol([MarshalAs(UnmanagedType.LPWStr)] string id, IntPtr items, out uint pid);
    }
    [DllImport("shell32.dll", CharSet=CharSet.Unicode, PreserveSig=false)]
    static extern void SHCreateItemFromParsingName(string path, IntPtr context, ref Guid iid, out IntPtr item);
    [DllImport("shell32.dll", PreserveSig=false)]
    static extern void SHCreateShellItemArrayFromShellItem(IntPtr item, ref Guid iid, out IntPtr array);

    static object Get(object target, string name, params object[] args) {
        return target.GetType().InvokeMember(name, BindingFlags.GetProperty, null, target, args);
    }
    static object Call(object target, string name, params object[] args) {
        return target.GetType().InvokeMember(name, BindingFlags.InvokeMethod, null, target, args);
    }
    static IEnumerable<string> Shortcuts(string root) {
        var pending=new Stack<string>();pending.Push(root);
        while(pending.Count>0) {
            string directory=pending.Pop(); string[] files,folders;
            try {files=Directory.GetFiles(directory,"*.lnk");folders=Directory.GetDirectories(directory);}
            catch(UnauthorizedAccessException) {continue;} catch(IOException) {continue;}
            foreach(string file in files) yield return file;
            foreach(string folder in folders) {
                try {if((File.GetAttributes(folder)&FileAttributes.ReparsePoint)==0)pending.Push(folder);}
                catch(UnauthorizedAccessException) { } catch(IOException) { }
            }
        }
    }
    static List<Dictionary<string,string>> Catalog() {
        var rows=new List<Dictionary<string,string>>();
        object shell=Activator.CreateInstance(Type.GetTypeFromProgID("Shell.Application"));
        object scripting=Activator.CreateInstance(Type.GetTypeFromProgID("WScript.Shell"));
        try {
            foreach(string root in new [] {Environment.GetFolderPath(Environment.SpecialFolder.StartMenu),
                                           Environment.GetFolderPath(Environment.SpecialFolder.CommonStartMenu)}) {
                if(!Directory.Exists(root)) continue;
                foreach(string path in Shortcuts(root)) {
                    object link=null;
                    try {
                        link=Call(scripting,"CreateShortcut",path);
                        string target=Convert.ToString(Get(link,"TargetPath"));
                        if(Path.IsPathRooted(target) && File.Exists(target) && Path.GetExtension(target).Equals(".exe",StringComparison.OrdinalIgnoreCase))
                            rows.Add(new Dictionary<string,string>{{"name",Path.GetFileNameWithoutExtension(path)},
                                {"path",Path.GetFullPath(target)},{"kind","exe"}});
                    } catch { } finally {if(link!=null) Marshal.FinalReleaseComObject(link);}
                }
            }
            object folder=Call(shell,"NameSpace","shell:AppsFolder");
            object items=Call(folder,"Items");
            try {
                int count=Convert.ToInt32(Get(items,"Count"));
                for(int i=0;i<count;i++) {
                    object item=Call(items,"Item",i);
                    try {
                        string id=Convert.ToString(Get(item,"Path"));
                        if(!id.Contains("!")) continue;
                        string package=Convert.ToString(Call(item,"ExtendedProperty","System.AppUserModel.PackageFullName"));
                        rows.Add(new Dictionary<string,string>{{"name",Convert.ToString(Get(item,"Name"))},
                            {"aumid",id},{"package",package},{"kind","packaged"}});
                    } finally {Marshal.FinalReleaseComObject(item);}
                }
            } finally {Marshal.FinalReleaseComObject(items);Marshal.FinalReleaseComObject(folder);}
        } finally {Marshal.FinalReleaseComObject(shell);Marshal.FinalReleaseComObject(scripting);}
        return rows;
    }
    [STAThread] public static int Main(string[] args) {
        Console.OutputEncoding=new UTF8Encoding(false);
        try {
            if(args.Length==1 && args[0]=="list") {
                Console.WriteLine(new JavaScriptSerializer().Serialize(Catalog()));return 0;
            }
            if(args.Length!=3 || args[0]!="activate") throw new ArgumentException("Invalid activation request");
            var manager=(ActivationManager)Activator.CreateInstance(Type.GetTypeFromCLSID(new Guid("45BA127D-10A8-46EA-8AB7-56EA9078943C")));
            uint pid;
            try {
                if(args[2]=="") Marshal.ThrowExceptionForHR(manager.ActivateApplication(args[1],"",0,out pid));
                else {
                    IntPtr item=IntPtr.Zero,array=IntPtr.Zero;
                    Guid itemId=new Guid("43826D1E-E718-42EE-BC55-A1E261C37BFE"),arrayId=new Guid("B63EA76D-1F85-456F-A19C-48159EFA858B");
                    try {
                        SHCreateItemFromParsingName(Path.GetFullPath(args[2]),IntPtr.Zero,ref itemId,out item);
                        SHCreateShellItemArrayFromShellItem(item,ref arrayId,out array);
                        Marshal.ThrowExceptionForHR(manager.ActivateForFile(args[1],array,"open",out pid));
                    } finally {if(array!=IntPtr.Zero)Marshal.Release(array);if(item!=IntPtr.Zero)Marshal.Release(item);}
                }
                Console.WriteLine(pid);
            } finally {Marshal.FinalReleaseComObject(manager);}
            return 0;
        } catch(Exception error) {Console.Error.WriteLine(error.Message);return 1;}
    }
}
