"""Fixed-shape research device: retain the loaded IRON execution handle."""
from reuse_device import ReuseDevice
from diagnose_transfer import Measured

class RetainedDevice(Measured,ReuseDevice):
    def __init__(self,frames=4096):
        super().__init__(frames)
        self.handle=None
        self.handle_loads=0
        self.reset_stages()

    def kernel(self,*args,**kwargs):
        from aie.utils import DefaultNPURuntime
        if self.handle is None:
            self.handle,result=ReuseDevice.kernel(*args,**kwargs)
            self.handle_loads+=1
            return self.handle,result
        try:
            result=DefaultNPURuntime.run(self.handle,list(args))
            return self.handle,result
        except RuntimeError as exc:
            message=str(exc)
            if not ('err=-22' in message or ('Invalid argument' in message and 'XRT' in message)):
                raise
            # Same stale-context condition handled by CallableDesign.
            self.handle=None
            return self.kernel(*args,**kwargs)

    def clear_handle(self):
        self.handle=None
