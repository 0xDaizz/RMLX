# RMLX vs MLX vs CUDA -- 아키텍처 비교

> 세 GPU 추론 스택에 대한 솔직한 엔지니어링 관점의 비교 문서입니다.
> RMLX는 특정 영역에서 명확한 장점을, 다른 영역에서 명확한 단점을 가지고 있습니다.
> 이 문서는 마케팅 없이 양쪽 모두를 있는 그대로 제시합니다.
>
> 관련 문서: [아키텍처 개요](architecture/overview_ko.md) | [설계 결정](architecture/design-decisions_ko.md) | [로드맵](roadmap/phases_ko.md)

---

## 1. 아키텍처 비교

| 특성 | RMLX | MLX | CUDA (PyTorch / vLLM) |
|------|------|-----|-----------------------|
| **언어** | Rust 1.80+ | C++ / Python (nanobind) | C++ / Python (pybind11) |
| **GPU API** | Apple Metal (metal-rs 0.31) | Apple Metal (metal-cpp) | NVIDIA CUDA |
| **메모리 모델** | Unified Memory (UMA) | Unified Memory (UMA) | 개별 VRAM + 호스트 RAM |
| **실행 모델** | Eager-first (prefill에 선택적 tracing) | Lazy evaluation (그래프 레벨 퓨전) | Eager (PyTorch) / Graph (CUDA Graphs) |
| **CB 관리** | ExecGraph: 65 CBs/layer -> 5 CBs/layer (92.3% 감소) | eval() 배치당 1 CB | Stream 기반, CUDA Graphs capture-replay |
| **동기화 메커니즘** | MTLSharedEvent signal/wait (263.9us) | waitUntilCompleted (424.9us) | CUDA events / streams |
| **RDMA** | Zero-copy: posix_memalign + NoCopy + ibv_reg_mr | std::copy로 전송 버퍼에 복사 | GPUDirect RDMA / NVLink |
| **Expert Parallelism** | 네이티브 EP (3-zone auto backend, 7 MoE 커널) | EP 미지원 | DeepSpeed-MoE, Tutel |
| **양자화** | Q4_0, Q4_1, Q8_0, GGUF, AWQ, GPTQ, FP8 | Q4_0, Q4_1, Q8_0, GGUF, AWQ, GPTQ | FP8, AWQ, GPTQ, INT4/INT8 |
| **Flash Attention** | Flash Attention 2 (fused SDPA, tiled K/V) | 미지원 (fused SDPA) | FlashAttention-2/3 |
| **KV 캐시** | 정적/순환/배치/양자화 KV 캐시 | 정적 레이어별 캐시 | PagedAttention (vLLM) |
| **학습** | LoRA 파인튜닝만 지원 | 전체 학습 지원 | 전체 학습 + LoRA + QLoRA |
| **Op 모듈 수** | 25 | ~50+ | 수백 개 |
| **테스트 수** | 543 | 광범위 | 광범위 |
| **완료된 Phase** | 0-9B-opt + S1-S5 | N/A (안정 릴리스) | N/A (안정 릴리스) |

---

## 2. RMLX가 MLX보다 우수한 부분

### 2.1 MTLSharedEvent 동기화

MLX는 `waitUntilCompleted`를 사용하여 Command Buffer 완료까지 CPU 스레드를 블로킹합니다. RMLX는 이를 논블로킹 `MTLSharedEvent` signal/wait로 대체합니다.

| 방식 | 지연 시간 | 메커니즘 |
|------|-----------|----------|
| `waitUntilCompleted` (MLX) | 424.9us | CB 완료까지 CPU 블로킹 |
| `MTLSharedEvent` spin-wait (RMLX) | 263.9us | signaledValue 논블로킹 폴링 |
| **개선** | **1.61x** | |

이 차이는 레이어를 거치며 누적됩니다. 60레이어 모델에서 동기화 오버헤드만으로 수십 밀리초가 절감됩니다.

### 2.2 ExecGraph Command Buffer 배칭

MLX는 연산별로 개별 Command Buffer를 제출합니다. RMLX의 ExecGraph는 결정론적 연산 시퀀스를 소수의 Command Buffer로 사전 인코딩합니다.

| 지표 | 베이스라인 (연산별) | ExecGraph |
|------|---------------------|-----------|
| 레이어당 CB 수 | 65 | 5 |
| CB 감소율 | -- | 92.3% |
| 레이어당 CPU-GPU 동기화 지점 | ~65 | ~1 |
| CPU-GPU 동기화 감소율 | -- | 98.5% |
| 레이어당 지연 시간 | ~112ms | ~6.4ms |
| **속도 향상** | -- | **17.4x** |
| 지연 시간 감소율 | -- | 94.3% |

수치적 동등성이 유지됩니다: 베이스라인과 ExecGraph 출력 간 max_diff = 6.4e-6.

### 2.3 Zero-Copy RDMA 데이터 경로

MLX는 RDMA 전송 전에 `std::copy`로 별도 버퍼에 데이터를 복사합니다. RMLX는 핫 패스에서 모든 복사를 제거합니다.

```
posix_memalign  ->  newBufferWithBytesNoCopy  ->  ibv_reg_mr
    (CPU)              (Metal GPU 뷰)             (RDMA MR)
         \                   |                    /
          +----- 동일 물리 주소 ------------------+
```

세 개의 뷰가 하나의 물리 주소를 공유합니다. GPU 연산 결과가 중간 복사 없이 RDMA를 통해 전송됩니다.

### 2.4 CPU-Minimal 실행

ExecGraph는 CPU의 역할을 얇은 제출 레이어로 축소합니다. 레이어당 CPU-GPU 동기화가 ~65 지점에서 ~1 지점으로 감소합니다(98.5% 감소). Apple Silicon에서 CPU와 GPU가 동일한 메모리 버스를 공유하므로, 추론 중 CPU 활동을 줄이면 GPU에 더 많은 대역폭이 확보됩니다.

### 2.5 단일 Rust 바이너리

| 측면 | MLX | RMLX |
|------|-----|------|
| 빌드 시스템 | CMake + Python setuptools + nanobind | `cargo build` (단일 명령) |
| 런타임 | Python 인터프리터 + C++ 공유 라이브러리 | 단일 정적 바이너리 |
| 배포 | pip install + 의존성 설치 | 바이너리 복사 |
| 패키지 관리 | pip + CMake find_package | Cargo.toml |

Python 인터프리터 오버헤드가 없습니다. 시작 시 동적 라이브러리 해석이 불필요합니다.

### 2.6 Expert Parallelism (EP)

MLX에는 Expert Parallelism 지원이 없습니다. 사용자가 MoE dispatch/combine을 직접 구현해야 합니다. RMLX는 완전한 EP 스택을 일급(first-class) 기능으로 제공합니다:

| 컴포넌트 | RMLX | MLX |
|----------|------|-----|
| MoE dispatch/combine | 3-zone 자동 백엔드 (CPU/Metal/RDMA) | 수동 구현 필요 |
| MoE Metal 커널 | 7개 전용 GPU 커널 | 없음 |
| 백엔드 선택 | 데이터 크기 기반 자동 선택 (쿨다운/히스테리시스) | 해당 없음 |
| SparseGuard | 오버플로우 모니터링 + 용량 자동 조절 | 해당 없음 |
| Compute-RDMA 오버랩 | 듀얼 커맨드 큐 파이프라인 | 해당 없음 |
| Zero-copy EP 경로 | 동일 물리 주소에 ibv_mr + Metal 버퍼 | 해당 없음 |

3-zone 정책이 최적 백엔드를 자동으로 선택합니다: 소규모 페이로드(N ≤ 64)에는 CPU, 중규모(N ≥ 320)에는 Metal GPU, 노드 간 통신에는 RDMA. Mixtral, DeepSeek-V3 같은 MoE 모델에서 수동 백엔드 튜닝이 필요 없습니다.

### 2.7 소유권 안전성

Rust의 소유권 시스템이 `unsafe` Metal/RDMA FFI 경계를 컴파일 타임에 격리합니다. 공개 API 표면은 전부 안전한 Rust입니다. 내부적으로 `posix_memalign`, `newBufferWithBytesNoCopy`, `ibv_post_send`에 대한 `unsafe` 블록은 `SAFETY` 주석과 함께 명시적으로 경계가 지정됩니다. 동시성 GPU/RDMA 코드 경로의 데이터 레이스는 `Send`/`Sync` 트레이트를 통해 컴파일 타임에 감지됩니다.

---

## 3. RMLX가 CUDA보다 우수한 부분

### 3.1 Unified Memory Architecture (UMA)

Apple Silicon의 UMA는 GPU와 CPU가 동일한 물리 메모리를 공유합니다. 둘 사이에 PCIe 버스가 없습니다.

| 연산 | CUDA | RMLX (UMA) |
|------|------|------------|
| Host-to-device 전송 | PCIe를 통한 `cudaMemcpy` (32-64 GB/s) | 불필요 -- 동일 메모리 |
| Device-to-host 읽기 | `cudaMemcpyDeviceToHost` | 직접 포인터 접근 |
| 메모리 용량 | VRAM 제한 (일반적으로 24-80GB) | 전체 시스템 RAM (최대 512GB) |

이산 GPU에서 VRAM 용량을 초과하는 모델의 경우, UMA가 큰 이점을 제공합니다. 512GB Mac Studio M3 Ultra는 오프로딩 없이 약 250B 파라미터 모델을 메모리에 수용할 수 있습니다.

### 3.2 단일 바이너리 배포

| 측면 | CUDA 스택 | RMLX |
|------|-----------|------|
| 드라이버 | NVIDIA 드라이버 (버전 종속) | macOS 내장 Metal 드라이버 |
| 런타임 | CUDA 툴킷 + cuDNN + cuBLAS | 불필요 (Metal은 macOS에 포함) |
| Python | Python + PyTorch + transformers | 불필요 |
| 바이너리 | 다수의 공유 라이브러리 | 단일 정적 바이너리 |

드라이버 호환성 매트릭스가 없습니다. `CUDA_HOME` 환경 변수 설정이 불필요합니다. `nvidia-smi` 디버깅이 필요 없습니다.

### 3.3 전력 효율

Apple Silicon (5nm / 3nm 공정)은 데이터센터 GPU 대비 현저히 낮은 전력으로 동작합니다.

| 플랫폼 | 일반적 전력 | 용도 |
|--------|-------------|------|
| M3 Ultra (Mac Studio) | ~100W 전체 시스템 | 데스크톱 추론 |
| NVIDIA H100 (SXM5) | GPU당 700W | 데이터센터 |
| NVIDIA A100 (SXM4) | GPU당 400W | 데이터센터 |

UMA에 맞는 추론 워크로드에서 성능 대비 전력 효율(performance-per-watt)의 이점이 상당합니다.

### 3.4 Zero-Copy 버퍼 공유

RMLX의 이중 등록 기법(동일 물리 주소에 Metal 버퍼 + RDMA MR)은 커널 레벨 피닝과 드라이버 조율이 필요한 CUDA의 GPUDirect RDMA보다 단순한 zero-copy 경로를 제공합니다.

---

## 4. RMLX가 MLX보다 뒤처지는 부분

이 섹션은 의도적으로 상세하게 작성되었습니다. 한계에 대한 솔직한 평가가 마케팅보다 유용합니다.

### 4.1 Flash Attention -- 구현 완료

RMLX는 Phase S3에서 Flash Attention 2를 구현했습니다. tiled K/V 처리와 online softmax를 사용하여 어텐션 행렬의 메모리 사용량을 O(N^2)에서 O(N)으로 줄입니다. MLX는 아직 완전한 Flash Attention을 지원하지 않으므로, 이 영역에서 RMLX가 더 진보한 상태입니다.

### 4.2 작은 생태계

MLX는 pip 설치가 가능한 패키지, Hugging Face 통합, 활발한 서드파티 기여를 가진 대규모 Python 커뮤니티를 보유합니다. RMLX는 단일 팀 프로젝트로 공개 패키지 레지스트리에 등록되어 있지 않습니다.

### 4.3 Python API 미지원

머신러닝 연구 커뮤니티는 압도적으로 Python을 사용합니다. RMLX는 Rust 전용이므로 연구자들의 채택에 한계가 있습니다. PyO3를 통한 Python 바인딩 레이어 제공은 아직 로드맵에 포함되지 않았습니다.

### 4.4 양자화 지원 -- MLX와 동등

| 형식 | RMLX | MLX |
|------|------|-----|
| Q4_0 / Q4_1 | 지원 | 지원 |
| Q8_0 | 지원 | 지원 |
| GGUF | 지원 | 지원 |
| AWQ | 지원 | 지원 |
| GPTQ | 지원 | 지원 |
| FP8 (E4M3/E5M2) | 지원 | 부분 지원 |

Phase S2에서 GGUF, AWQ, GPTQ, FP8 지원이 추가되어 MLX와 동등한 양자화 범위를 달성했습니다. FP8의 경우 RMLX가 E4M3과 E5M2 두 변형을 모두 완전 지원하는 반면, MLX는 부분적으로만 지원합니다.

### 4.5 그래프 최적화 미지원

MLX는 lazy evaluation과 그래프 레벨 퓨전을 사용합니다. 그래프 컴파일러가 element-wise 연산 퓨전, 데드 코드 제거, 전체 연산 그래프에 걸친 메모리 할당 최적화를 수행할 수 있습니다. RMLX는 선택적 tracing과 함께 eager-first 실행을 사용하여 연산별 지연 시간은 낮지만 그래프 레벨 최적화 기회를 놓칩니다.

### 4.6 학습 미지원

MLX는 자동 미분, 옵티마이저 상태, 그래디언트 누적을 갖춘 전체 학습을 지원합니다. RMLX는 VJP 자동미분과 LoRA 파인튜닝만 지원합니다. 전체 사전학습이나 전체 파라미터 파인튜닝은 지원하지 않습니다.

---

## 5. RMLX가 CUDA보다 뒤처지는 부분

RMLX와 CUDA 생태계 간의 격차는 RMLX와 MLX 간의 격차보다 큽니다. 이는 예상된 것으로, CUDA는 수십 년간의 투자가 축적되어 있습니다.

### 5.1 Flash Attention -- 부분적 동등

RMLX는 Phase S3에서 Flash Attention 2를 Metal 커널로 구현했습니다. tiled K/V와 online softmax를 사용하여 O(N) 메모리 효율을 달성합니다. 다만 CUDA 생태계의 FlashAttention-3은 NVIDIA 하드웨어의 전용 SRAM을 활용한 추가 최적화를 제공하므로, 절대적인 연산 효율에서는 여전히 격차가 있습니다.

### 5.2 인터커넥트 대역폭

| 인터커넥트 | 링크당 대역폭 | 일반적 구성 | 합계 |
|------------|---------------|-------------|------|
| Thunderbolt 5 | 포트당 ~16 GB/s | 2포트 | ~32 GB/s |
| NVLink (H100) | 링크당 50 GB/s | 12링크 | 600 GB/s |
| **비율** | | | **~18.7배 불리** |

분산 추론에서 이 대역폭 격차는 통신 집약적 워크로드(예: tensor parallelism)에서 RMLX의 스케일링 효율을 제한합니다.

### 5.6 CUDA Graphs Capture-Replay 미지원

CUDA Graphs는 GPU 연산 시퀀스를 캡처한 후 거의 제로에 가까운 CPU 오버헤드로 재생합니다. RMLX의 ExecGraph는 매번 결정론적 시퀀스를 재인코딩하므로, 연산별 제출보다는 빠르지만 진정한 재생(replay)보다는 느립니다. 상세 비교는 6절을 참조하세요.

### 5.7 Tensor Cores에 해당하는 하드웨어 미보유

NVIDIA GPU는 행렬 곱셈(FP16, BF16, FP8, INT8)을 위한 전용 Tensor Cores를 보유합니다. Apple GPU는 행렬 연산에 SIMD 그룹을 사용하며, 좋은 처리량을 제공하지만 전용 고정 기능 행렬 하드웨어는 없습니다.

### 5.8 생태계 성숙도

CUDA는 컴파일러(NVCC, Triton), 라이브러리(cuBLAS, cuDNN, NCCL, cuSPARSE), 프로파일링 도구(Nsight, nvprof), 커뮤니티 지식에 걸쳐 수십 년간의 최적화가 축적되어 있습니다. RMLX는 이 범위에 필적할 수 없습니다.

---

## 6. ExecGraph vs CUDA Graphs -- 상세 비교

| 측면 | ExecGraph (RMLX) | CUDA Graphs |
|------|-------------------|-------------|
| **메커니즘** | 결정론적 연산 시퀀스를 배칭된 CB로 재인코딩 | GPU 연산을 캡처하고, 기록된 스트림을 재생 |
| **CPU 오버헤드** | 추론 스텝당 재인코딩 비용 | 거의 제로 (재생만) |
| **유연성** | 항상 결정론적; 모든 shape에서 동작 | 입력 shape 변경 시 재캡처 필요 |
| **CB 감소** | 레이어당 65 -> 5 (92.3%) | 단일 그래프로 전체 병합 |
| **동기화 모델** | MTLSharedEvent (논블로킹) | CUDA events (스트림 순서) |
| **Shape 동적성** | 재인코딩이 shape 변경을 자연스럽게 처리 | 재캡처 또는 CUDA Graph 업데이트 필요 |
| **지연 시간** | 레이어당 ~6.4ms | 서브밀리초 재생 |
| **베이스라인 대비 속도 향상** | 17.4x | 일반적으로 2-5x (이미 효율적인 베이스라인 기준) |
| **구현 복잡도** | 중간 (결정론적 시퀀싱) | 낮음 (캡처 API가 직관적) |
| **메모리 오버헤드** | 최소 (재인코딩 시 버퍼 재사용) | 그래프 저장 (캡처된 연산) |

**핵심 인사이트**: ExecGraph의 17.4x 속도 향상은 CUDA의 기본 실행 모델보다 오버헤드가 큰 연산별 베이스라인 대비입니다. CUDA의 베이스라인은 스트림 순서 실행으로 이미 더 효율적이므로, CUDA Graphs는 더 강한 시작점에서 상대적으로 작은 개선을 제공합니다.

---

## 7. 메모리 모델 비교

### UMA (Apple Silicon) vs 이산 GPU 메모리 (NVIDIA)

```
Apple Silicon UMA:
┌────────────────────────────────────┐
│        512GB Unified Memory        │
│   ┌──────────┐  ┌──────────────┐   │
│   │ CPU 코어  │  │ GPU 코어     │   │
│   │ (읽기/    │  │ (읽기/       │   │
│   │  쓰기)    │  │  쓰기)       │   │
│   └──────────┘  └──────────────┘   │
└────────────────────────────────────┘

NVIDIA 이산 GPU:
┌───────────────┐    PCIe/NVLink    ┌─────────────────┐
│  호스트 RAM    │ <===============> │  GPU VRAM       │
│  (CPU 접근)    │    32-64 GB/s     │  (24-80GB HBM)  │
│               │                   │  ~3.35 TB/s BW  │
└───────────────┘                   └─────────────────┘
```

| 측면 | UMA (RMLX / MLX) | 이산 GPU (CUDA) |
|------|-------------------|-----------------|
| **메모리 용량** | 최대 512GB (M3 Ultra) | 일반적으로 24-80GB VRAM |
| **CPU-GPU 전송** | Zero-copy (동일 물리 메모리) | 명시적 cudaMemcpy 필요 |
| **메모리 대역폭** | ~400 GB/s | ~3.35 TB/s (H100 HBM3e) |
| **대역폭 비율** | 1x | 이산 GPU가 ~8.4x 유리 |
| **대형 모델 지원** | 512GB에 ~250B 파라미터 수용 가능 | VRAM 초과 시 멀티 GPU 또는 오프로딩 필요 |
| **RDMA 통합** | 동일 물리 주소에 이중 등록 | GPUDirect RDMA (커널 드라이버 필요) |

**UMA 장점**: 복사 오버헤드 없음, 더 큰 유효 용량, 단순한 프로그래밍 모델, zero-copy RDMA 경로.

**이산 GPU 장점**: 훨씬 높은 메모리 대역폭 (HBM3e: 3.35 TB/s vs UMA: ~400 GB/s), 전용 VRAM이 CPU 메모리 압력으로부터 GPU 워크로드를 격리.

---

## 8. 양자화 지원 비교

| 형식 | RMLX | MLX | CUDA 생태계 |
|------|------|-----|-------------|
| Q4_0 (4-bit block, absmax) | 지원 | 지원 | llama.cpp 경유 |
| Q4_1 (4-bit block, min+scale) | 지원 | 지원 | llama.cpp 경유 |
| Q8_0 (8-bit block) | 지원 | 지원 | llama.cpp 경유 |
| GGUF (llama.cpp 형식) | 지원 | 지원 | 지원 (네이티브 llama.cpp) |
| AWQ (activation-aware) | 지원 | 지원 | 지원 (AutoAWQ) |
| GPTQ (post-training) | 지원 | 지원 | 지원 (AutoGPTQ) |
| FP8 (E4M3/E5M2) | 지원 | 부분 지원 | 지원 (H100+, 네이티브) |
| INT4 (NVIDIA) | 미지원 | 미지원 | 지원 (TensorRT-LLM) |
| INT8 (smooth quant) | 미지원 | 부분 지원 | 지원 (TensorRT-LLM) |
| W4A16 (가중치 4-bit, 활성화 16-bit) | 지원 (Q4_0/Q4_1) | 지원 | 지원 |

Phase S2에서 GGUF, AWQ, GPTQ, FP8 지원이 추가되어 RMLX의 양자화 범위가 MLX와 동등해졌습니다. CUDA 생태계의 INT4/INT8 (TensorRT-LLM)만 아직 미지원이며, 이는 NVIDIA 전용 하드웨어 기능입니다.

---

## 9. 수치적 동등성

RMLX는 베이스라인 실행 경로와 ExecGraph 최적화 경로 사이의 엄격한 수치적 동등성을 유지합니다.

| 지표 | 값 |
|------|----|
| max_diff (베이스라인 vs ExecGraph) | 6.4e-6 |
| 허용 임계값 | 1e-4 |
| 테스트 커버리지 | 543 테스트 |
| 검증 방법 | 모든 Op에 대한 element-wise 비교 |

이는 ExecGraph의 17.4x 성능 향상이 의미 있는 수치적 드리프트를 유발하지 않음을 보장합니다.

---

## 10. 동등성 달성 현황

Phase S1-S5를 통해 4절과 5절에서 식별된 주요 격차가 해소되었습니다:

| Phase | 초점 | 주요 결과물 | 해소된 격차 | 상태 |
|-------|------|------------|-------------|------|
| **Phase S1** | Quick Wins | GELU, RotatingKV, BatchKV | 서빙 기본 기능 | ✅ 완료 |
| **Phase S2** | DType + 양자화 | FP8, GGUF, AWQ/GPTQ | 4.4절, 8절 | ✅ 완료 |
| **Phase S3** | 어텐션 업그레이드 | Flash Attention 2, QuantizedKV | 4.1절, 5.1절 | ✅ 완료 |
| **Phase S4** | 런타임 유연성 | Array 수준 집합 연산, 동적 shape | 런타임 편의성 | ✅ 완료 |
| **Phase S5** | 멀티모달 확장 | Conv1d/Conv2d | 멀티모달 지원 | ✅ 완료 |

Flash Attention 2와 고급 양자화(GGUF, AWQ, GPTQ, FP8)가 구현되어, MLX 대비 주요 기술 격차가 해소되었습니다. CUDA 생태계와의 격차는 여전히 존재하나, 주로 하드웨어 차이(Tensor Cores, NVLink 대역폭)와 생태계 성숙도에 기인합니다.

---

## 요약

RMLX는 MLX나 CUDA의 대체품이 아닙니다. 특정 니치를 차지합니다: **Apple Silicon을 위한 고성능 Metal GPU ML 프레임워크, Rust로 작성, Thunderbolt 5 RDMA 기반 분산 실행에 최적화**.

**RMLX를 선택해야 할 때**: Apple Silicon 하드웨어에서 최대 GPU 성능이 필요할 때, Mac Studio 클러스터에서 MoE 모델의 Expert Parallelism과 zero-copy RDMA가 필요할 때, Python 의존성 없는 단일 Rust 바이너리를 원할 때, 또는 분산 추론 클러스터를 구축할 때.

**MLX를 선택해야 할 때**: Python 호환성, 성숙한 생태계, 학습 지원, 또는 광범위한 양자화 형식 지원이 필요할 때.

**CUDA를 선택해야 할 때**: 최대 절대 성능, 대규모 프로덕션 서빙, Flash Attention, speculative decoding, 또는 가장 큰 도구 및 라이브러리 생태계에 대한 접근이 필요할 때.
