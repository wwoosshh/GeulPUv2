# Third-party notices

## llama.cpp

The following files are generated from llama.cpp (https://github.com/ggml-org/llama.cpp) sources:

- `엔진/유니코드범주.gl` — Unicode category ranges and the whitespace set of `src/unicode-data.cpp` (generator: `도구/유니코드범주.py`)
- `엔진/사전처리표.gl` — the pre-tokenizer names, regex lists and settings of `src/llama-vocab.cpp`, and the fast-path regexes of
  `src/unicode.cpp` (generator: `도구/사전처리표.py`)
- `시험/어휘시험.txt` lists llama.cpp's vocabulary test files (`models/ggml-vocab-*`), which the tests download and do not ship.

```
MIT License

Copyright (c) 2023-2026 The ggml authors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Unicode Character Database

The character data in `엔진/유니코드표.gl` (normalization) and `엔진/유니코드범주.gl` (categories, through llama.cpp) is derived from the
Unicode Character Database, Copyright © 1991-2026 Unicode, Inc., under the Unicode License v3 (https://www.unicode.org/license.txt).
