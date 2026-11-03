import argparse
import sys
from pathlib import Path
HERE=Path(__file__).parent
ROOT=HERE.parents[2]
sys.path.insert(0,str(HERE.parent))
import trace_model

p=argparse.ArgumentParser()
p.add_argument('--backend',choices=['gl','c','c_unrolled','aie_cpp'],required=True)
args=p.parse_args()
kind={'gl':'gl-aie','c':'native-c','c_unrolled':'native-c-unrolled','aie_cpp':'native-aie'}[args.backend]
sys.argv=['trace_model.py',str(ROOT/'빌드/npu/models/Qwen3-0.6B-Q8_0.gguf'),
          '--backend',kind,'--tokens','1,2,3','--repeats','2',
          '--output',str(HERE/f'results/model-{args.backend}.jsonl')]
trace_model.main()
