import sys,runpy
from pathlib import Path
sys.argv.append('--profile')
runpy.run_path(str(Path(__file__).with_name('test_generation.py')),run_name='__main__')
