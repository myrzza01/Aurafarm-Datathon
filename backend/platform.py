"""Local interprocess locks and process-tree containment on Windows and POSIX."""
from contextlib import AbstractContextManager
import errno
import os
from pathlib import Path
import signal
import time
from uuid import uuid4


class FileLock(AbstractContextManager):
    """An OS-owned lock, automatically released when its process exits."""

    def __init__(self, path: Path, *, blocking: bool = True):
        self.path, self.blocking = path, blocking
        self.stream = None

    def __enter__(self):
        self.stream = self.path.open("a+b")
        try:
            if os.name == "nt":
                import msvcrt
                if self.stream.seek(0, os.SEEK_END) == 0:
                    self.stream.write(b"\0")
                    self.stream.flush()
                while True:
                    self.stream.seek(0)
                    try:
                        msvcrt.locking(self.stream.fileno(), msvcrt.LK_NBLCK, 1)
                        break
                    except OSError as exc:
                        if exc.errno not in (errno.EACCES, errno.EAGAIN, errno.EDEADLK):
                            raise
                        if not self.blocking:
                            raise BlockingIOError("Lock is held by another process") from exc
                        time.sleep(.02)
            else:
                import fcntl
                flags = fcntl.LOCK_EX | (0 if self.blocking else fcntl.LOCK_NB)
                fcntl.flock(self.stream, flags)
        except BaseException:
            self.stream.close()
            self.stream = None
            raise
        return self

    def __exit__(self, *args):
        if self.stream is not None:
            try:
                if os.name == "nt":
                    import msvcrt
                    self.stream.seek(0)
                    msvcrt.locking(self.stream.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(self.stream, fcntl.LOCK_UN)
            finally:
                self.stream.close()
                self.stream = None


if os.name == "nt":
    import ctypes
    from ctypes import wintypes

    _kernel = ctypes.WinDLL("kernel32", use_last_error=True)

    class _BasicLimits(ctypes.Structure):
        _fields_ = [
            ("PerProcessUserTimeLimit", ctypes.c_longlong), ("PerJobUserTimeLimit", ctypes.c_longlong),
            ("LimitFlags", wintypes.DWORD), ("MinimumWorkingSetSize", ctypes.c_size_t),
            ("MaximumWorkingSetSize", ctypes.c_size_t), ("ActiveProcessLimit", wintypes.DWORD),
            ("Affinity", ctypes.c_size_t), ("PriorityClass", wintypes.DWORD),
            ("SchedulingClass", wintypes.DWORD),
        ]

    class _IOCounters(ctypes.Structure):
        _fields_ = [(name, ctypes.c_ulonglong) for name in (
            "ReadOperationCount", "WriteOperationCount", "OtherOperationCount",
            "ReadTransferCount", "WriteTransferCount", "OtherTransferCount",
        )]

    class _ExtendedLimits(ctypes.Structure):
        _fields_ = [
            ("BasicLimitInformation", _BasicLimits), ("IoInfo", _IOCounters),
            ("ProcessMemoryLimit", ctypes.c_size_t), ("JobMemoryLimit", ctypes.c_size_t),
            ("PeakProcessMemoryUsed", ctypes.c_size_t), ("PeakJobMemoryUsed", ctypes.c_size_t),
        ]

    _kernel.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
    _kernel.CreateJobObjectW.restype = wintypes.HANDLE
    _kernel.OpenJobObjectW.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.LPCWSTR]
    _kernel.OpenJobObjectW.restype = wintypes.HANDLE
    _kernel.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
    _kernel.SetInformationJobObject.restype = wintypes.BOOL
    _kernel.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
    _kernel.AssignProcessToJobObject.restype = wintypes.BOOL
    _kernel.TerminateJobObject.argtypes = [wintypes.HANDLE, wintypes.UINT]
    _kernel.TerminateJobObject.restype = wintypes.BOOL
    _kernel.QueryInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD, ctypes.c_void_p]
    _kernel.QueryInformationJobObject.restype = wintypes.BOOL
    _kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    _kernel.OpenProcess.restype = wintypes.HANDLE
    _kernel.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    _kernel.WaitForSingleObject.restype = wintypes.DWORD
    _kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    _kernel.CloseHandle.restype = wintypes.BOOL

    def _checked(value):
        if not value:
            raise ctypes.WinError(ctypes.get_last_error())
        return value

    def _job_process_handles(job):
        capacity = 32
        while True:
            class ProcessIds(ctypes.Structure):
                _fields_ = [
                    ("NumberOfAssignedProcesses", wintypes.DWORD),
                    ("NumberOfProcessIdsInList", wintypes.DWORD),
                    ("ProcessIdList", ctypes.c_size_t * capacity),
                ]
            members = ProcessIds()
            if _kernel.QueryInformationJobObject(job, 3, ctypes.byref(members), ctypes.sizeof(members), None):
                break
            if ctypes.get_last_error() != 234:  # ERROR_MORE_DATA
                raise ctypes.WinError(ctypes.get_last_error())
            capacity = max(capacity * 2, members.NumberOfAssignedProcesses)
        handles = []
        for pid in members.ProcessIdList[:members.NumberOfProcessIdsInList]:
            handle = _kernel.OpenProcess(0x00100000, False, pid)  # SYNCHRONIZE
            if handle:
                handles.append(handle)
        return handles


class ProcessTree:
    """Keep a Windows Job handle in the API parent; POSIX uses a new session."""

    def __init__(self):
        self.name = f"Local\\ForteAnalysis-{uuid4().hex}" if os.name == "nt" else None
        self.handle = None
        if os.name == "nt":
            self.handle = _checked(_kernel.CreateJobObjectW(None, self.name))
            limits = _ExtendedLimits()
            limits.BasicLimitInformation.LimitFlags = 0x00002000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
            try:
                _checked(_kernel.SetInformationJobObject(self.handle, 9, ctypes.byref(limits), ctypes.sizeof(limits)))
            except BaseException:
                self.close()
                raise

    def attach(self, process):
        if os.name == "nt":
            # The child waits on a gate until containment succeeds, before loading model code.
            handle = _checked(_kernel.OpenProcess(0x0100 | 0x0001, False, process.pid))
            try:
                _checked(_kernel.AssignProcessToJobObject(self.handle, handle))
            finally:
                _kernel.CloseHandle(handle)

    def terminate(self, process):
        handles = []
        try:
            if os.name == "nt":
                if self.handle is not None:
                    handles = _job_process_handles(self.handle)
                    _checked(_kernel.TerminateJobObject(self.handle, 1))
            else:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    pass
            if process.is_alive():
                process.kill()
            process.join()
            if os.name == "nt":
                # Job accounting can reach zero before process handles are
                # signaled. Wait for OS teardown before deleting open videos.
                deadline = time.monotonic() + 5
                for handle in handles:
                    _kernel.WaitForSingleObject(handle, max(0, int((deadline - time.monotonic()) * 1000)))
        finally:
            for handle in handles:
                _kernel.CloseHandle(handle)

    def close(self):
        if self.handle is not None:
            _kernel.CloseHandle(self.handle)
            self.handle = None


class ParentMonitor(AbstractContextManager):
    def __init__(self, parent_pid):
        self.pid, self.handle = parent_pid, None
        if os.name == "nt":
            # Holding the process handle also avoids PID-reuse ambiguity.
            self.handle = _kernel.OpenProcess(0x00100000, False, parent_pid)  # SYNCHRONIZE

    def alive(self):
        if os.name == "nt":
            return bool(self.handle) and _kernel.WaitForSingleObject(self.handle, 0) == 258  # WAIT_TIMEOUT
        return os.getppid() == self.pid

    def __exit__(self, *args):
        if self.handle:
            _kernel.CloseHandle(self.handle)
            self.handle = None


def prepare_child():
    if os.name != "nt":
        os.setsid()


def kill_current_tree(job_name):
    if os.name == "nt":
        # Do not retain a Job handle in the child: parent death must close the last handle.
        handle = _kernel.OpenJobObjectW(0x0008, False, job_name) if job_name else None
        if handle:
            try:
                _kernel.TerminateJobObject(handle, 1)
            finally:
                _kernel.CloseHandle(handle)
    else:
        try:
            os.killpg(os.getpid(), signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
    os._exit(1)
