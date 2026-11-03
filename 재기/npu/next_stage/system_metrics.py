import ctypes
import os
import threading
import time
from ctypes import wintypes

class Memory(ctypes.Structure):
    _fields_=[('cb',wintypes.DWORD),('faults',wintypes.DWORD)]+[(n,ctypes.c_size_t) for n in ('peak_rss','rss','peak_paged','paged','peak_nonpaged','nonpaged','pagefile','peak_pagefile','private')]
kernel=ctypes.WinDLL('kernel32',use_last_error=True)
psapi=ctypes.WinDLL('psapi',use_last_error=True)
kernel.GetCurrentProcess.restype=wintypes.HANDLE
kernel.OpenProcess.argtypes=[wintypes.DWORD,wintypes.BOOL,wintypes.DWORD]
kernel.OpenProcess.restype=wintypes.HANDLE
kernel.CloseHandle.argtypes=[wintypes.HANDLE]
kernel.GetProcessTimes.argtypes=[wintypes.HANDLE]+[ctypes.POINTER(wintypes.FILETIME)]*4
psapi.GetProcessMemoryInfo.argtypes=[wintypes.HANDLE,ctypes.POINTER(Memory),wintypes.DWORD]

def memory(handle=None):
    data=Memory()
    data.cb=ctypes.sizeof(data)
    if not psapi.GetProcessMemoryInfo(handle or kernel.GetCurrentProcess(),ctypes.byref(data),data.cb):raise ctypes.WinError(ctypes.get_last_error())
    return dict(rss=data.rss,private=data.private)

class Resources:
    def __init__(self,handle=None):self.handle=handle
    def cpu_time(self):
        if self.handle is None:return time.process_time()
        values=[wintypes.FILETIME() for _ in range(4)]
        if not kernel.GetProcessTimes(self.handle,*(ctypes.byref(v) for v in values)):raise ctypes.WinError(ctypes.get_last_error())
        return sum((v.dwHighDateTime<<32)|v.dwLowDateTime for v in values[2:])/1e7
    def __enter__(self):
        self.stop=threading.Event()
        self.peak=memory(self.handle)
        self.start_memory=self.peak.copy()
        def sample():
            while not self.stop.wait(.1):
                for key,val in memory(self.handle).items():self.peak[key]=max(self.peak[key],val)
        self.thread=threading.Thread(target=sample,daemon=True)
        self.thread.start()
        self.cpu=self.cpu_time()
        self.wall=time.perf_counter()
        return self
    def __exit__(self,*args):
        wall=time.perf_counter()-self.wall
        cpu=self.cpu_time()-self.cpu
        self.stop.set()
        self.thread.join()
        for key,val in memory(self.handle).items():self.peak[key]=max(self.peak[key],val)
        self.result=dict(wall_seconds=wall,cpu_seconds=cpu,average_cpu_cores=cpu/wall,
            logical_cpu_percent=100*cpu/wall/os.cpu_count(),peak_sampled_rss_bytes=self.peak['rss'],
            peak_sampled_private_bytes=self.peak['private'],start_memory=self.start_memory)
