import tempfile
import unittest
from pathlib import Path
from lower_gl import compile_gl,BackendError
from geul.diagnostics import CompileError

HERE=Path(__file__).parent
BASE=(HERE/'tile.gl').read_text(encoding='utf-8')

class LoweringTests(unittest.TestCase):
    def compile(self,text):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'kernel.gl'
            path.write_text(text,encoding='utf-8')
            return compile_gl(path)

    def test_real_source_drives_generated_body(self):
        original,_=self.compile(BASE)
        changed,_=self.compile(BASE.replace('256 * s1','128 * s1'))
        self.assertNotEqual(original,changed)
        self.assertIn('(128 * s1)',changed)

    def test_role_reordering(self):
        original,_=self.compile(BASE)
        text=BASE.replace('a 에서, ai 를, b 로, bi','b 로, ai 를, a 에서, bi')
        self.assertEqual(original,self.compile(text)[0])

    def test_load_hoisting_and_reuse(self):
        code,_=self.compile(BASE)
        self.assertEqual(code.count('aie::load_v<32>'),3)
        self.assertLess(code.index('a + ah'),code.index('for (int32_t n'))
        self.assertGreater(code.index('b + bi'),code.index('for (int32_t n'))

    def test_bounds(self):
        for before,after in [('r < 4','r < 5'),('n < 32','n < 33'),('r * 32 + n]','r * 32 + n + 1]')]:
            with self.subTest(after=after),self.assertRaises(BackendError):
                self.compile(BASE.replace(before,after))

    def test_alignment(self):
        with self.assertRaises(BackendError): self.compile(BASE.replace('ai = r * 32','ai = r * 32 + 1'))

    def test_overflow(self):
        with self.assertRaises(BackendError): self.compile(BASE.replace('256 * s1','65536 * s1'))

    def test_intrinsic_rejection(self):
        with self.assertRaises(BackendError): self.compile(BASE.replace('"dot32"','"unknown"'))

    def test_signature(self):
        with self.assertRaises((BackendError,CompileError)): self.compile(BASE.replace('중간정수 참조 dst','짧은실수 참조 dst'))

if __name__=='__main__': unittest.main()
