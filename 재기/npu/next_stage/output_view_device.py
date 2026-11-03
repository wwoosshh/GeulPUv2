"""Keep contiguous input preparation; omit only the intermediate output copy."""
from retained_device import RetainedDevice

class OutputViewDevice(RetainedDevice):
    def read_output(self,view,count):
        # Measured.grouped scatters this into an owned chunk before the next run.
        return view.reshape(-1,self.output_words)[:count]
