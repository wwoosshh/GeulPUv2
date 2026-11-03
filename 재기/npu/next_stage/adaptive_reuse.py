"""Conservative per-chunk capacity policy; not a learned performance model."""


def select_capacity(groups, span):
    if groups <= 0 or span <= 0:
        raise ValueError('positive group and weight tile counts required')
    padded_groups=(groups+3)//4*4
    padded_span=(span+3)//4*4
    total=padded_groups*padded_span
    # Existing 64-row results do not favor 8192 over 4096.
    capacities=(2048,4096) if groups<=16 else (2048,4096,8192)
    # First minimize transferred padding, then submission count.
    return min(capacities,key=lambda f: (((total+f-1)//f)*f,(total+f-1)//f))


class AdaptiveReuseDevice:
    rows=4
    fused=True
    frames=2048

    def __init__(self):
        self.devices={}
        self.last_calls=0
        self.selections={}

    def _device(self,frames):
        if frames not in self.devices:
            from reuse_device import ReuseDevice
            self.devices[frames]=ReuseDevice(frames)
        return self.devices[frames]

    def grouped(self,inputs,weights,groups):
        frames=select_capacity(groups,len(weights))
        device=self._device(frames)
        result=device.grouped(inputs,weights,groups)
        self.frames=frames
        self.last_calls=device.last_calls
        self.selections[frames]=self.selections.get(frames,0)+1
        return result

    def __call__(self,a,b):
        self.frames=2048
        return self._device(2048)(a,b)
