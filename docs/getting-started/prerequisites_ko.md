# 시스템 요구사항

rmlx를 빌드하고 실행하기 위한 하드웨어 및 소프트웨어 요구사항을 안내합니다.

---

## 하드웨어

### 필수

| 항목 | 요구사항 | 비고 |
|------|----------|------|
| **프로세서** | Apple Silicon (M1 이상) | Metal GPU + Unified Memory Architecture(UMA) 필수 |
| **메모리** | 16GB 이상 권장 | 대형 모델 추론 시 32GB+ 권장 |
| **macOS** | macOS 14.0+ (Sonoma) | Metal 3 API 기능 활용 |

> **참고**: Intel Mac에서는 Metal compute 기능이 제한되므로, Apple Silicon Mac이 필수입니다.
> UMA 구조 덕분에 GPU와 CPU가 동일한 물리 메모리를 공유하여 zero-copy 최적화가 가능합니다.

### 분산 추론 (선택)

분산 추론(Expert Parallelism)을 사용하려면 아래 추가 하드웨어가 필요합니다.

| 항목 | 요구사항 |
|------|----------|
| **Thunderbolt 5 포트** | 각 Mac에 1개 이상 |
| **TB5 호환 케이블** | Thunderbolt 5 인증 케이블 |
| **노드 수** | 2대 이상의 Apple Silicon Mac |

> **참고**: Thunderbolt 5를 통한 RDMA 통신으로 노드 간 6.89 GB/s 이상의 대역폭을 달성합니다.
> 듀얼 TB5 포트를 사용하면 12 GB/s 이상으로 확장할 수 있습니다 (Phase 6).

### RDMA 셋업 (클러스터 검증)

RMLX는 `mlx.distributed_config`, `mlx.launch`를 벤치마킹한
분산 설정/실행 헬퍼 스크립트를 이 저장소에 포함합니다.

```bash
# 1) 호스트파일 생성 + 기본 호스트 셋업 (컨트롤 노드에서 실행)
python3 scripts/rmlx_distributed_config.py \
  --hosts node1,node2 \
  --backend rdma \
  --over thunderbolt \
  --control-iface en0 \
  --auto-setup \
  --output rmlx-hosts.json \
  --verbose

# 2) 각 노드 RDMA 디바이스 가시성 검증
python3 scripts/rmlx_launch.py \
  --backend rdma \
  --hostfile rmlx-hosts.json \
  -- ibv_devices

# 3) 양 노드에서 RDMA 크레이트 테스트 실행
python3 scripts/rmlx_launch.py \
  --backend rdma \
  --hostfile rmlx-hosts.json \
  -- cargo test -p rmlx-rdma -- --nocapture
```

> **참고**: `--auto-setup`은 각 노드의 SSH 환경에서 passwordless `sudo`가 필요합니다.
> 호스트파일만 만들고 싶다면 `--auto-setup`을 빼고 실행하세요.

---

## 소프트웨어

### 필수

| 소프트웨어 | 버전 | 설치 방법 |
|-----------|------|----------|
| **macOS** | 14.0+ (Sonoma) | 시스템 업데이트 |
| **Rust toolchain** | stable (1.80+) | [rustup](https://rustup.rs/) |

#### Rust 설치

```bash
# rustup 설치 (처음 설치하는 경우)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# 이미 설치된 경우 최신 stable로 업데이트
rustup update stable
```

> **참고**: 프로젝트 루트의 `rust-toolchain.toml` 파일이 정확한 Rust 버전을 자동으로 관리합니다.
> `rustup`이 설치되어 있으면 `cargo build` 실행 시 필요한 toolchain이 자동으로 설치됩니다.

### Xcode (선택)

| 구성 | AOT 컴파일 | JIT 컴파일 | 비고 |
|------|-----------|-----------|------|
| **Xcode 전체 설치** | O | O | `.metal` -> `.metallib` AOT 컴파일 지원 |
| **Command Line Tools만** | X | O | AOT 불가, JIT(`new_library_with_source`)로 동작 |
| **미설치** | X | O | JIT로 동작, 테스트 실행 가능 |

- **AOT 컴파일**: `build.rs`가 빌드 시 `xcrun metal` 컴파일러를 호출하여 `.metal` 셰이더를 `.metallib`로 사전 컴파일합니다. **전체 Xcode 설치가 필요합니다** (Command Line Tools만으로는 `xcrun -sdk macosx metal` 명령을 사용할 수 없습니다).
- **JIT 컴파일**: Xcode가 없어도 `MTLDevice::newLibraryWithSource`를 통해 런타임에 셰이더를 컴파일할 수 있습니다. 테스트 코드는 JIT 컴파일을 사용하므로 Xcode 없이도 실행 가능합니다.

```bash
# Xcode 설치 (App Store 또는 xcode-select)
xcode-select --install  # Command Line Tools만 설치
# 전체 Xcode는 App Store에서 설치하세요
```

---

## 개발 도구 (권장)

필수는 아니지만, 개발 생산성을 높여주는 도구들입니다.

| 도구 | 용도 | 설치 방법 |
|------|------|----------|
| **cargo-watch** | 파일 변경 시 자동 빌드/테스트 | `cargo install cargo-watch` |
| **rust-analyzer** | IDE 언어 서버 (자동완성, 타입 추론) | VS Code 확장 또는 에디터별 설치 |

```bash
# cargo-watch 설치
cargo install cargo-watch

# 사용 예시: 파일 변경 시 자동 테스트
cargo watch -x test

# 사용 예시: 특정 크레이트만 감시
cargo watch -x 'test -p rmlx-metal'
```

---

## 환경 확인

설치가 올바르게 완료되었는지 아래 명령으로 확인할 수 있습니다.

```bash
# Rust toolchain 확인
rustc --version
cargo --version

# macOS 버전 확인
sw_vers

# Metal 지원 확인 (Apple Silicon인지 확인)
system_profiler SPDisplaysDataType | grep "Metal"

# Xcode Metal 컴파일러 확인 (선택)
xcrun -sdk macosx --find metal 2>/dev/null && echo "AOT 컴파일 가능" || echo "JIT 전용 모드"
```

정상적인 출력 예시:

```
rustc 1.80.0 (051478957 2024-07-21)
cargo 1.80.0 (376290515 2024-07-16)
ProductName:    macOS
ProductVersion: 14.x
Metal:          Apple M1 Pro
AOT 컴파일 가능
```

---

다음 단계: [빌드 및 실행](building.md)
