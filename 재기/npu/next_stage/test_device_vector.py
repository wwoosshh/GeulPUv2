import os,runpy
from pathlib import Path
os.environ['GEUL_FP_VECTOR']='1'
runpy.run_path(str(Path(__file__).with_name('test_device.py')),run_name='__main__')
