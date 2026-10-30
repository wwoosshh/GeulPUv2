갈래: 더함

- **명령줄 옵션의 영어 별칭과 `--help`** ([#137](https://github.com/wwoosshh/GeulPUv2/issues/137)): 서버(`--host` · `--port` · `--slots` ·
  `--ctx` · `--chunk` · `--name` · `--log-dir` · `--no-spec` · `--store` · `--store-size` · `--model-hash` · `--contract exact|fast` ·
  `--store-write all|evict|off` · `--tool-errors` · `--model-check` · `--header-only` · `--archs` 등), 커널 생성기(구조 이름 `qwen2` ·
  `qwen3` · `llama` · `qwen35` · `gemma4` …, `--fast` · `--exact` · `--shared32` · `--coop` · `--head` 등), 터미널 실행기, GeulPU(`--port` ·
  `--data` · `--no-open`)가 영어 이름을 한글 이름의 별칭으로 받는다. 한글 이름은 그대로이고, 어느 이름으로 불러도 결과는 바이트까지 같다.
  `--help` 가 두 이름을 함께 적는다. 한글을 못 쓰는 터미널(SSH · JupyterHub)을 위해 `build-linux.sh` 하나가 글(geul) 저장소 받기 → 컴파일러
  `geulc` → `ptx-backend` · `kernel-gen` · `server-cuda` · `runner-cuda` 를 ASCII 이름으로 빌드한다.

---

- **English aliases for command-line options, and `--help`** ([#137](https://github.com/wwoosshh/GeulPUv2/issues/137)). The server
  (`--host` · `--port` · `--slots` · `--ctx` · `--chunk` · `--name` · `--log-dir` · `--no-spec` · `--store` · `--store-size` ·
  `--model-hash` · `--contract exact|fast` · `--store-write all|evict|off` · `--tool-errors` · `--model-check` · `--header-only` ·
  `--archs` and others), the kernel generator (architecture names `qwen2` · `qwen3` · `llama` · `qwen35` · `gemma4` …, `--fast` ·
  `--exact` · `--shared32` · `--coop` · `--head` and others), the terminal runner and GeulPU (`--port` · `--data` · `--no-open`) take
  English names as aliases of the Korean ones. The Korean names stay, and either name gives byte-identical results. `--help` lists
  both. For terminals that cannot show Korean (SSH, JupyterHub), `build-linux.sh` clones the geul repository, builds the compiler as
  `geulc` and builds `ptx-backend` · `kernel-gen` · `server-cuda` · `runner-cuda` under ASCII names.
