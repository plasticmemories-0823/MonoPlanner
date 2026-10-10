// A trusted launcher only. Supplied programs run in a networkless LPAC and Job.
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.IO;
using System.Runtime.InteropServices;
using System.Security.AccessControl;
using System.Security.Principal;
using System.Text;
using System.Web.Script.Serialization;

internal static class Sandbox
{
    [StructLayout(LayoutKind.Sequential)] struct Startup {
        public int cb; public IntPtr reserved, desktop, title;
        public int x,y,cx,cy,xchars,ychars,fill,flags; public short show, count;
        public IntPtr bytes, input, output, error;
    }
    [StructLayout(LayoutKind.Sequential)] struct StartupEx { public Startup start; public IntPtr attrs; }
    [StructLayout(LayoutKind.Sequential)] struct ProcessInfo { public IntPtr process, thread; public int pid, tid; }
    [StructLayout(LayoutKind.Sequential)] struct Capabilities { public IntPtr sid, caps; public int count, reserved; }
    [StructLayout(LayoutKind.Sequential)] struct SidAttributes { public IntPtr sid; public int attributes; }
    [StructLayout(LayoutKind.Sequential)] struct Limits {
        public long processTime, jobTime; public int flags; public UIntPtr min, max;
        public int count; public UIntPtr affinity; public int priority, scheduling;
    }
    [StructLayout(LayoutKind.Sequential)] struct ExtendedLimits {
        public Limits basic; public ulong readOps,writeOps,otherOps,readBytes,writeBytes,otherBytes;
        public UIntPtr processMemory,jobMemory,peakProcess,peakJob;
    }
    [DllImport("userenv.dll", CharSet=CharSet.Unicode)] static extern int CreateAppContainerProfile(string name,string display,string description,IntPtr caps,int count,out IntPtr sid);
    [DllImport("userenv.dll", CharSet=CharSet.Unicode)] static extern int DeleteAppContainerProfile(string name);
    [DllImport("userenv.dll", CharSet=CharSet.Unicode)] static extern int DeriveAppContainerSidFromAppContainerName(string name,out IntPtr sid);
    [DllImport("advapi32.dll")] static extern IntPtr FreeSid(IntPtr sid);
    [DllImport("advapi32.dll", CharSet=CharSet.Unicode, SetLastError=true)] static extern bool SetFileSecurity(string path,int info,byte[] descriptor);
    [DllImport("kernelbase.dll", CharSet=CharSet.Unicode, SetLastError=true)] static extern bool DeriveCapabilitySidsFromName(string name,out IntPtr groups,out int groupCount,out IntPtr capabilities,out int count);
    [DllImport("kernel32.dll")] static extern IntPtr LocalFree(IntPtr value);
    [DllImport("kernel32.dll", SetLastError=true)] static extern bool InitializeProcThreadAttributeList(IntPtr list,int count,int flags,ref IntPtr size);
    [DllImport("kernel32.dll", SetLastError=true)] static extern bool UpdateProcThreadAttribute(IntPtr list,int flags,IntPtr key,IntPtr value,IntPtr size,IntPtr previous,IntPtr returned);
    [DllImport("kernel32.dll")] static extern void DeleteProcThreadAttributeList(IntPtr list);
    [DllImport("kernel32.dll", CharSet=CharSet.Unicode, SetLastError=true)] static extern bool CreateProcess(string app,StringBuilder command,IntPtr pa,IntPtr ta,bool inherit,int flags,IntPtr env,string cwd,ref StartupEx startup,out ProcessInfo process);
    [DllImport("kernel32.dll", SetLastError=true)] static extern IntPtr CreateJobObject(IntPtr security,string name);
    [DllImport("kernel32.dll", SetLastError=true)] static extern bool SetInformationJobObject(IntPtr job,int type,ref ExtendedLimits limits,int size);
    [DllImport("kernel32.dll", SetLastError=true)] static extern bool AssignProcessToJobObject(IntPtr job,IntPtr process);
    [DllImport("kernel32.dll")] static extern bool TerminateJobObject(IntPtr job,int exit);
    [DllImport("kernel32.dll")] static extern uint WaitForSingleObject(IntPtr handle,uint milliseconds);
    [DllImport("kernel32.dll")] static extern bool GetExitCodeProcess(IntPtr handle,out int code);
    [DllImport("kernel32.dll")] static extern uint ResumeThread(IntPtr thread);
    [DllImport("kernel32.dll")] static extern bool CloseHandle(IntPtr handle);
    [DllImport("kernel32.dll")] static extern IntPtr GetStdHandle(int kind);
    [DllImport("advapi32.dll", SetLastError=true)] static extern bool OpenProcessToken(IntPtr process,int access,out IntPtr token);
    [DllImport("advapi32.dll", SetLastError=true)] static extern bool GetTokenInformation(IntPtr token,int type,out int value,int size,out int needed);

    static void Check(bool success) { if(!success) throw new Win32Exception(Marshal.GetLastWin32Error()); }
    static string Quote(string text) {
        if(text.Length>0 && text.IndexOfAny(new [] {' ', '\t', '\n', '"'})<0) return text;
        var result=new StringBuilder("\""); int slash=0;
        foreach(char c in text) { if(c=='\\') { slash++; continue; }
            if(c=='\"') result.Append('\\',slash*2+1); else result.Append('\\',slash);
            slash=0; result.Append(c);
        }
        return result.Append('\\',slash*2).Append('"').ToString();
    }
    static void Access(string path,SecurityIdentifier sid,FileSystemRights rights,bool remove) {
        if(!Directory.Exists(path) && !File.Exists(path)) throw new FileNotFoundException(path);
        if((File.GetAttributes(path)&FileAttributes.ReparsePoint)!=0) throw new IOException("Reparse point is not an ACL target: "+path);
        bool directory=Directory.Exists(path);
        var security=directory ? (FileSystemSecurity)Directory.GetAccessControl(path) : File.GetAccessControl(path);
        var traversal=FileSystemRights.Traverse|FileSystemRights.ReadAttributes;
        var inheritance=directory && rights!=traversal ? InheritanceFlags.ContainerInherit|InheritanceFlags.ObjectInherit : InheritanceFlags.None;
        var descriptor=new RawSecurityDescriptor(security.GetSecurityDescriptorBinaryForm(),0);
        var acl=descriptor.DiscretionaryAcl;
        for(int i=acl.Count-1;i>=0;i--) {
            var ace=acl[i] as CommonAce;
            if(ace!=null && ace.SecurityIdentifier.Equals(sid)) acl.RemoveAce(i);
        }
        if(!remove) {
            var flags=inheritance==InheritanceFlags.None ? AceFlags.None : AceFlags.ContainerInherit|AceFlags.ObjectInherit;
            acl.InsertAce(0,new CommonAce(flags,AceQualifier.AccessDenied,(int)(FileSystemRights.ChangePermissions|FileSystemRights.TakeOwnership),sid,false,null));
            int index=1;
            while(index<acl.Count) {var ace=acl[index] as CommonAce;
                if(ace==null || ace.AceQualifier!=AceQualifier.AccessDenied || (ace.AceFlags&AceFlags.Inherited)!=0) break; index++;}
            acl.InsertAce(index,new CommonAce(flags,AceQualifier.AccessAllowed,(int)(rights|FileSystemRights.Synchronize),sid,false,null));
        }
        var bytes=new byte[descriptor.BinaryLength];descriptor.GetBinaryForm(bytes,0);
        security.SetSecurityDescriptorBinaryForm(bytes,AccessControlSections.Access);
        if(rights==traversal) Check(SetFileSecurity(path,4,bytes));
        else if(directory) Directory.SetAccessControl(path,(DirectorySecurity)security); else File.SetAccessControl(path,(FileSecurity)security);
    }

    public static int Main(string[] arguments) {
        Console.InputEncoding=new UTF8Encoding(false); Console.OutputEncoding=new UTF8Encoding(false);
        var json=new JavaScriptSerializer { MaxJsonLength=16*1024*1024 };
        string profile="MonoPlanner.Run."+Guid.NewGuid().ToString("N");
        IntPtr sid=IntPtr.Zero,attrs=IntPtr.Zero,capPtr=IntPtr.Zero,policy=IntPtr.Zero,envPtr=IntPtr.Zero,handles=IntPtr.Zero,job=IntPtr.Zero;
        ProcessInfo process=new ProcessInfo();
        var grants=new List<Tuple<string,FileSystemRights>>();
        SecurityIdentifier identity=null;
        try {
            var request=json.Deserialize<Dictionary<string,object>>(Console.In.ReadToEnd());
            if(request.ContainsKey("cleanup_profile")) {
                profile=(string)request["cleanup_profile"];
                if(!profile.StartsWith("monoplanner.run.",StringComparison.OrdinalIgnoreCase)) throw new ArgumentException("Not a MonoPlanner execution profile");
                Marshal.ThrowExceptionForHR(DeriveAppContainerSidFromAppContainerName(profile,out sid)); identity=new SecurityIdentifier(sid);
                foreach(string path in (System.Collections.IEnumerable)request["paths"]) {
                    var access=Directory.GetAccessControl(path);
                    bool found=false; foreach(FileSystemAccessRule rule in access.GetAccessRules(true,false,typeof(SecurityIdentifier))) if(rule.IdentityReference.Equals(identity)) found=true;
                    if(found) {access.PurgeAccessRules(identity); Check(SetFileSecurity(path,4,access.GetSecurityDescriptorBinaryForm()));}
                }
                return 0;
            }
            string cwd=Path.GetFullPath((string)request["cwd"]);
            int timeout=Convert.ToInt32(request["timeout_ms"]);
            if(timeout<=0) throw new ArgumentException("timeout must be positive");
            int hr=CreateAppContainerProfile(profile,profile,"MonoPlanner isolated execution",IntPtr.Zero,0,out sid);
            Marshal.ThrowExceptionForHR(hr); identity=new SecurityIdentifier(sid);
            foreach(string path in (System.Collections.IEnumerable)request["writable"]) grants.Add(Tuple.Create(Path.GetFullPath(path),FileSystemRights.Modify));
            foreach(string path in (System.Collections.IEnumerable)request["readable"]) grants.Add(Tuple.Create(Path.GetFullPath(path),FileSystemRights.ReadAndExecute));
            File.WriteAllText((string)request["result"],json.Serialize(new {status="initializing",profile=profile,sid=identity.Value}),Encoding.UTF8);
            // Unique per-run SID; cleanup removes only our rules, never restores a stale ACL.
            foreach(var grant in grants) Access(grant.Item1,identity,grant.Item2,false);
            IntPtr size=IntPtr.Zero; InitializeProcThreadAttributeList(IntPtr.Zero,3,0,ref size);
            attrs=Marshal.AllocHGlobal(size); Check(InitializeProcThreadAttributeList(attrs,3,0,ref size));
            string[] names={"registryRead"};
            int capSize=Marshal.SizeOf(typeof(SidAttributes));
            var capArray=Marshal.AllocHGlobal(capSize*names.Length);
            for(int i=0;i<names.Length;i++) {
                IntPtr groups,capabilities; int groupCount,capCount;
                Check(DeriveCapabilitySidsFromName(names[i],out groups,out groupCount,out capabilities,out capCount));
                Marshal.StructureToPtr(new SidAttributes {sid=Marshal.ReadIntPtr(capabilities),attributes=4},IntPtr.Add(capArray,capSize*i),false);
            }
            var caps=new Capabilities { sid=sid,caps=capArray,count=names.Length }; capPtr=Marshal.AllocHGlobal(Marshal.SizeOf(caps)); Marshal.StructureToPtr(caps,capPtr,false);
            Check(UpdateProcThreadAttribute(attrs,0,(IntPtr)0x20009,capPtr,(IntPtr)Marshal.SizeOf(caps),IntPtr.Zero,IntPtr.Zero));
            policy=Marshal.AllocHGlobal(4); Marshal.WriteInt32(policy,1); // LPAC.
            Check(UpdateProcThreadAttribute(attrs,0,(IntPtr)0x2000f,policy,(IntPtr)4,IntPtr.Zero,IntPtr.Zero));
            var std=new [] {GetStdHandle(-10),GetStdHandle(-11),GetStdHandle(-12)};
            handles=Marshal.AllocHGlobal(3*IntPtr.Size); for(int i=0;i<3;i++) Marshal.WriteIntPtr(handles,i*IntPtr.Size,std[i]);
            Check(UpdateProcThreadAttribute(attrs,0,(IntPtr)0x20002,handles,(IntPtr)(3*IntPtr.Size),IntPtr.Zero,IntPtr.Zero));
            var environment=(Dictionary<string,object>)request["env"];
            var lines=new List<string>(); foreach(var pair in environment) lines.Add(pair.Key+"="+(string)pair.Value);
            lines.Sort(StringComparer.OrdinalIgnoreCase); envPtr=Marshal.StringToHGlobalUni(string.Join("\0",lines.ToArray())+"\0\0");
            var argv=((System.Collections.ArrayList)request["argv"]).ToArray(); var command=new StringBuilder();
            foreach(string arg in argv) { if(command.Length>0) command.Append(' '); command.Append(Quote(arg)); }
            if(request.ContainsKey("shell_body")) {
                command=new StringBuilder(Quote((string)argv[0])+" /d /s /c \""+(string)request["shell_body"]+"\"");
            }
            var startup=new StartupEx { start=new Startup { cb=Marshal.SizeOf(typeof(StartupEx)),flags=0x100,input=std[0],output=std[1],error=std[2] },attrs=attrs };
            job=CreateJobObject(IntPtr.Zero,null); if(job==IntPtr.Zero) Check(false);
            var limits=new ExtendedLimits { basic=new Limits { flags=0x2000|0x8,count=64 } };
            Check(SetInformationJobObject(job,9,ref limits,Marshal.SizeOf(limits)));
            if(!CreateProcess((string)argv[0],command,IntPtr.Zero,IntPtr.Zero,true,0x80000|0x400|0x4|0x08000000,envPtr,cwd,ref startup,out process)) throw new Win32Exception(Marshal.GetLastWin32Error(),"CreateProcess LPAC");
            Check(AssignProcessToJobObject(job,process.process));
            IntPtr token; Check(OpenProcessToken(process.process,8,out token));
            try { int value,needed; Check(GetTokenInformation(token,29,out value,4,out needed)); if(value!=1) throw new IOException("Child is not AppContainer; execution refused"); } finally {CloseHandle(token);}
            if(ResumeThread(process.thread)==uint.MaxValue) Check(false);
            bool timedOut=WaitForSingleObject(process.process,(uint)timeout)==258;
            int exit; if(timedOut) TerminateJobObject(job,124);
            WaitForSingleObject(process.process,5000); Check(GetExitCodeProcess(process.process,out exit));
            TerminateJobObject(job,0); // No background children survive a completed operation.
            File.WriteAllText((string)request["result"],json.Serialize(new {exit_code=timedOut?124:exit,timed_out=timedOut,pid=process.pid,isolation="lpac",profile=profile}),Encoding.UTF8);
            return 0;
        } catch(Exception error) { Console.Error.WriteLine("SANDBOX_INIT_OR_RUNTIME_ERROR: "+error); return 125; }
        finally {
            if(job!=IntPtr.Zero) {TerminateJobObject(job,125); CloseHandle(job);}
            if(process.thread!=IntPtr.Zero) CloseHandle(process.thread);
            if(process.process!=IntPtr.Zero) CloseHandle(process.process);
            if(identity!=null) for(int i=grants.Count-1;i>=0;i--) try {Access(grants[i].Item1,identity,grants[i].Item2,true);} catch(Exception error) {Console.Error.WriteLine("SANDBOX_ACL_CLEANUP: "+error.Message);}
            if(attrs!=IntPtr.Zero) {DeleteProcThreadAttributeList(attrs);Marshal.FreeHGlobal(attrs);}
            foreach(IntPtr p in new [] {capPtr,policy,envPtr,handles}) if(p!=IntPtr.Zero) Marshal.FreeHGlobal(p);
            if(sid!=IntPtr.Zero) {FreeSid(sid); DeleteAppContainerProfile(profile);}
        }
    }
}
