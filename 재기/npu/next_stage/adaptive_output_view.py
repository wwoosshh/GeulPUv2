"""Adaptive capacities with independent retained handles and owned output chunks."""
from adaptive_reuse import AdaptiveReuseDevice
from output_view_device import OutputViewDevice

class AdaptiveOutputViewDevice(AdaptiveReuseDevice):
    def _device(self,frames):
        if frames not in self.devices:
            self.devices[frames]=OutputViewDevice(frames)
        return self.devices[frames]
