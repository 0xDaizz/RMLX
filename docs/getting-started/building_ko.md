# 빌드 및 실행

rmlx 프로젝트를 클론하고, 빌드하고, 테스트하는 방법을 안내합니다.

---

## 빠른 시작

```bash
# 1. 리포지토리 클론
git clone https://github.com/0xDaizz/RMLX.git
cd rmlx

# 2. 전체 워크스페이스 빌드
cargo build --workspace

# 3. 테스트 실행
cargo test --workspace

# 4. 포맷 및 린트 확인
cargo fmt --all --check
cargo clippy --workspace -- -D warnings
```

> **참고**: 처음 빌드 시 의존성 다운로드와 컴파일에 수 분이 소요될 수 있습니다.

---

## 워크스페이스 구조

rmlx는 Cargo workspace로 구성되어 있으며, 다음 크레이트들을 포함합니다.

```
rmlx/
├── Cargo.toml              # 워크스페이스 루트
├── crates/
│   ├── rmlx-metal/         # Metal GPU 추상화
│   ├── rmlx-alloc/         # Zero-copy 메모리 할당
│   ├── rmlx-rdma/          # RDMA 통신 (Thunderbolt 5)
│   ├── rmlx-core/          # Array 타입, 커널 레지스트리
│   ├── rmlx-distributed/   # 분산 추론 (EP, 파이프라인)
│   └── rmlx-nn/            # 신경망 레이어 (Transformer 등)
```

---

## 빌드 시스템 상세

### Metal 셰이더 AOT 컴파일

`rmlx-core` 크레이트의 `build.rs`는 빌드 과정에서 Metal 셰이더를 사전 컴파일합니다.

**컴파일 파이프라인**:

```
kernels/*.metal  →  xcrun metal -c  →  *.air  →  xcrun metallib  →  rmlx_kernels.metallib
```

1. `kernels/` 디렉토리의 모든 `.metal` 파일을 탐색합니다
2. 각 파일을 `xcrun -sdk macosx metal -c`로 중간 표현(`.air`)으로 컴파일합니다
3. 모든 `.air` 파일을 `xcrun -sdk macosx metallib`로 링크하여 단일 `.metallib` 파일을 생성합니다
4. `RMLX_METALLIB_PATH` 환경 변수에 생성된 `.metallib` 경로를 설정합니다

### Xcode가 없는 경우

Xcode가 설치되어 있지 않으면 `build.rs`가 다음과 같이 동작합니다.

```
cargo:warning=Metal compiler not found (requires Xcode, not just Command Line Tools).
               Skipping shader AOT compilation. GPU kernels will not be available at runtime.
```

- AOT 컴파일을 건너뛰고 **경고만 출력**합니다 (빌드 실패가 아닙니다)
- `RMLX_METALLIB_PATH`가 빈 문자열로 설정됩니다
- **테스트 코드는 JIT 컴파일(`compile_source`)을 사용**하므로 Xcode 없이도 정상 실행됩니다

> **핵심**: Xcode 미설치 시에도 빌드와 테스트는 정상적으로 진행됩니다.
> 프로덕션 환경에서 최적 성능을 위해서는 AOT 컴파일된 `.metallib`를 사용하는 것을 권장합니다.

### AOT metallib 경로 수동 지정

사전 컴파일된 `.metallib` 파일이 별도로 있는 경우, 환경 변수로 직접 경로를 지정할 수 있습니다.

```bash
# 환경 변수로 AOT metallib 경로 전달
RMLX_METALLIB_PATH=/path/to/rmlx_kernels.metallib cargo build --workspace
```

---

## 개별 크레이트 빌드

특정 크레이트만 빌드하거나 테스트하려면 `-p` 플래그를 사용합니다.

```bash
# rmlx-metal 크레이트만 빌드
cargo build -p rmlx-metal

# rmlx-metal 테스트만 실행
cargo test -p rmlx-metal

# rmlx-core 크레이트만 빌드
cargo build -p rmlx-core

# 특정 테스트만 실행
cargo test -p rmlx-metal -- test_basic_metal_compute
```

---

## 개발 워크플로

### 자동 빌드 (cargo-watch)

파일 변경 시 자동으로 빌드와 테스트를 수행합니다.

```bash
# 파일 변경 시 자동 체크 (컴파일 에러 확인)
cargo watch -x check

# 파일 변경 시 자동 테스트
cargo watch -x 'test --workspace'

# 특정 크레이트만 감시
cargo watch -x 'test -p rmlx-metal'
```

### CI와 동일한 검증

Pull Request를 생성하기 전에, CI에서 실행되는 것과 동일한 검증을 로컬에서 수행할 수 있습니다.

```bash
# 전체 CI 검증 순서
cargo build --workspace \
  && cargo fmt --all --check \
  && cargo clippy --workspace -- -D warnings \
  && cargo test --workspace
```

---

## 문제 해결

### `xcrun: error: unable to find utility "metal"`

**원인**: Xcode가 설치되어 있지 않거나, Command Line Tools만 설치된 경우입니다.

**해결 방법**:
- App Store에서 Xcode를 설치합니다
- 또는 AOT 컴파일 없이 JIT 모드로 사용합니다 (빌드 경고를 무시해도 됩니다)

### `error: no Metal device available`

**원인**: Metal을 지원하지 않는 환경(VM, Intel Mac, CI 러너 등)에서 실행한 경우입니다.

**해결 방법**:
- Apple Silicon Mac에서 실행하세요
- Metal 관련 테스트는 디바이스가 없으면 자동으로 skip됩니다

### 빌드 시 `kernels/` 디렉토리 관련 에러

**원인**: `rmlx-core/kernels/` 디렉토리가 비어있거나 존재하지 않는 경우입니다.

**해결 방법**:
```bash
# kernels 디렉토리 확인
ls crates/rmlx-core/kernels/

# 디렉토리가 없는 경우 (Phase 2에서 추가 예정)
mkdir -p crates/rmlx-core/kernels
```

---

다음 단계: [첫 번째 Metal 커널 실행하기](first-kernel.md)
