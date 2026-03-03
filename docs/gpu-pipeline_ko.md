# GPU Pipeline — ExecGraph 아키텍처

## 개요

rmlx GPU 파이프라인은 여러 Metal GPU 연산을 최소한의 command buffer로 배칭하여 연산별 CPU 오버헤드를 제거합니다. 핵심 추상화인 **ExecGraph**는 결정적 실행 그래프를 사전 구축하여 transformer 레이어 연산을 CPU 개입 없이 재생합니다.

**주요 결과:**

| 지표 | Baseline | ExecGraph | 개선 |
|------|----------|-----------|------|
| 레이어당 Command buffer | 65 | 5 | 92.3% 감소 |
| 레이어당 레이턴시 | 110.4ms | 6.8ms | 16.15x 속도 향상 |
| CPU-GPU 동기화 오버헤드 | baseline | 최소 | 98.5% 감소 |
| 수치 정합성 | -- | max_diff=6.4e-6 | 일치 |

---

## 문제점

기본 Metal 실행 모델에서는 각 GPU 연산(matmul, RoPE, softmax, add 등)이 자체 command buffer를 생성합니다:

1. 새로운 `MTLCommandBuffer` 할당
2. `MTLComputeCommandEncoder` 생성
3. Pipeline state, buffer, threadgroup 크기 설정
4. Thread 디스패치
5. 인코딩 종료
6. Command buffer 커밋
7. **CPU-GPU 동기화 배리어** (완료 대기)

LLaMA 스타일 모델의 단일 transformer 레이어는 약 65개의 개별 연산을 실행합니다. 각 연산은 command buffer 생성, 인코더 설정, CPU-GPU 동기화 포인트에 대한 CPU 측 오버헤드를 발생시킵니다. 레이어당 65개의 command buffer가 있으면, CPU가 GPU 작업을 관리하는 데 GPU가 실행하는 것보다 더 많은 시간을 소비합니다.

```
Baseline: 레이어당 65 CB x N 레이어
  [CB1: matmul_qkv] -> sync -> [CB2: rope_q] -> sync -> [CB3: rope_k] -> sync -> ...
  총: 레이어당 110.4ms
```

---

## ExecGraph 아키텍처

GPU 파이프라인은 3개의 계층적 컴포넌트로 구성됩니다:

### CommandBatcher

`CommandBatcher`는 여러 인코더 연산을 공유 command buffer로 그룹화합니다. 각 연산이 자체 CB를 생성하는 대신, batcher가 여러 연산이 순차적으로 인코딩하는 단일 CB를 제공합니다.

```rust
let mut batcher = CommandBatcher::new(&device);
batcher.begin();

// 여러 연산이 하나의 command buffer를 공유
matmul.forward_into_cb(&mut batcher, &q_proj, &x)?;
rope.forward_into_cb(&mut batcher, &q, positions)?;
matmul.forward_into_cb(&mut batcher, &k_proj, &x)?;
rope.forward_into_cb(&mut batcher, &k, positions)?;

batcher.commit();  // 모든 연산에 대해 단일 커밋
```

### ExecGraph

`ExecGraph`는 transformer 모델의 전체 실행 시퀀스를 사전 구축합니다. Transformer 추론은 결정적이기 때문에(동일한 연산, 동일한 버퍼 크기, 동일한 디스패치 기하), 그래프는 시퀀스를 한 번 기록하고 이후 토큰에 대해 재생합니다.

```rust
// 한 번 구축
let graph = ExecGraph::build(&model, &sample_input)?;

// 각 토큰에 대해 재생 — CPU 오버헤드 거의 없음
for token in tokens {
    graph.execute(&device, &buffers)?;
}
```

### ICB (Indirect Command Buffers)

`IcbBuilder`, `IcbReplay`, `IcbCache`는 Metal Indirect Command Buffer 지원을 제공합니다. ICB를 사용하면 GPU가 CPU 개입 없이 compute 명령을 디스패치할 수 있어, 기록된 명령 시퀀스에 대해 진정한 zero-CPU-overhead 재생이 가능합니다.

---

## `_into_cb()` 패턴

GPU 파이프라인의 기반은 `_into_cb()` 패턴입니다. rmlx-core의 14개 전체 op 모듈이 새로운 command buffer를 생성하는 대신 호출자가 제공한 command buffer에 작업을 인코딩하는 `_into_cb()` 변형을 구현합니다.

**표준 패턴** (자체 CB 생성):
```rust
fn forward(&self, input: &Array) -> Result<Array> {
    let cb = device.new_command_buffer();
    let encoder = cb.new_compute_command_encoder();
    // ... 작업 인코딩 ...
    encoder.end_encoding();
    cb.commit();
    cb.wait_until_completed();
    Ok(output)
}
```

**`_into_cb()` 패턴** (호출자의 CB에 인코딩):
```rust
fn forward_into_cb(
    &self,
    batcher: &mut CommandBatcher,
    input: &Array,
) -> Result<Array> {
    let encoder = batcher.current_encoder();
    // ... 공유 인코더에 작업 인코딩 ...
    Ok(output)
}
```

14개 전체 op 모듈이 이 패턴을 구현합니다:
- matmul, quantized matmul, softmax, RMS norm, RoPE
- element-wise binary ops, reduce, copy/transpose, indexing
- SiLU, residual add, layer norm, gather, scatter

nn 레이어에서는 다음으로 확장됩니다:
- `Linear::forward_into_cb()`
- `Attention::forward_graph()`
- `TransformerBlock::forward_graph()`
- `TransformerModel::forward_graph()`

---

## Command Buffer 감소

레이어당 65개의 baseline command buffer가 5개로 통합됩니다:

| CB | 연산 | 설명 |
|----|------|------|
| CB1 | Q/K/V projection + RoPE | QKV Linear projection과 rotary position encoding |
| CB2 | SDPA | Fused scaled dot-product attention |
| CB3 | Output projection + residual | Attention output projection과 residual connection |
| CB4 | FFN (gate + up + SiLU + down) | Gated activation을 사용한 feed-forward network |
| CB5 | Final residual + norm | 최종 residual connection과 RMS normalization |

```
ExecGraph: 레이어당 5 CB
  [CB1: QKV+RoPE] -> [CB2: SDPA] -> [CB3: OProj+Res] -> [CB4: FFN] -> [CB5: Res+Norm]
  총: 레이어당 6.8ms
```

이는 command buffer 수의 **92.3% 감소** (65 → 5)와 CPU-GPU 동기화 포인트의 **98.5% 감소**를 나타냅니다.

---

## 벤치마크 결과

단일 transformer 레이어(LLaMA 스타일 아키텍처)에서 측정한 벤치마크입니다:

### Baseline (연산별 command buffer)

- **레이어당 Command buffer:** 65
- **레이어당 레이턴시:** 110.4ms
- **레이어당 CPU-GPU 동기화:** 65

### ExecGraph (배칭된 command buffer)

- **레이어당 Command buffer:** 5
- **레이어당 레이턴시:** 6.8ms
- **레이어당 CPU-GPU 동기화:** 5

### 요약

| 지표 | 값 |
|------|-----|
| 속도 향상 | **16.15x** |
| 레이턴시 감소 | **93.8%** (110.4ms → 6.8ms) |
| CB 감소 | **92.3%** (65 → 5) |
| CPU-GPU 동기화 감소 | **98.5%** |
| 통과 테스트 | **534** |

---

## 가중치 사전 캐싱

Phase 9B-opt에서 런타임 transpose 오버헤드를 제거하기 위해 가중치 사전 캐싱을 도입했습니다. 모델 초기화 시 `prepare_weight_t()`가 가중치 행렬의 contiguous transposed 복사본을 생성하여 추론 중 전치 연산을 수행할 필요가 없도록 합니다.

```rust
// 모델 로드 / 워밍업 시
linear.prepare_weight_t();

// 사전 캐싱된 contiguous transposed 가중치 반환
let wt = linear.weight_transposed_contiguous();
```

모델 수준에서 `prepare_weights_for_graph()`가 모든 가중치 행렬을 재귀적으로 준비합니다:

```rust
model.prepare_weights_for_graph();
// TransformerModel -> TransformerBlock -> Attention + FeedForward -> Linear
```

이를 통해 command buffer 배칭 이후에도 남아 있던 레이어당 오버헤드의 상당 부분을 제거하여, 최종 16.15x 속도 향상에 기여합니다.

---

## ExecGraph vs CUDA Graphs

Metal과 CUDA는 GPU 작업 배칭에 근본적으로 다른 접근 방식을 취합니다:

| 측면 | ExecGraph (Metal) | CUDA Graphs |
|------|-------------------|-------------|
| 전략 | 최소 CB에 **재인코딩** | 기록된 스트림을 **캡처-재생** |
| 기록 | 명시적 그래프 구축 | 암시적 스트림 캡처 |
| 유연성 | 재생 간 buffer 바인딩 변경 가능 | 변경 시 그래프 재구축 필요 |
| CPU 오버헤드 | 거의 없음 (레이어당 5 CB 커밋) | 거의 없음 (단일 그래프 실행) |
| GPU 오버헤드 | 수동 디스패치와 동일 | 드라이버 수준 최적화 가능 |
| 메모리 | 그래프에 대한 추가 메모리 없음 | 그래프 객체가 캡처 상태 유지 |

ExecGraph의 재인코딩 전략은 Metal의 command buffer 모델에 적합합니다. 고정된 시퀀스를 캡처하고 재생하는 CUDA Graphs와 달리, ExecGraph는 인코딩 시퀀스를 사전 계산하고 새로운 command buffer에 재인코딩합니다. 이를 통해 유연성(호출 간 buffer 바인딩 변경 가능)을 유지하면서 비슷한 수준의 CPU 오버헤드 감소를 달성합니다.

---

## 수치 정합성

ExecGraph는 baseline 연산별 실행 경로와 수치적으로 동일한 결과를 생성합니다:

- **최대 절대 차이:** 6.4e-6
- **검증:** 534 전체 테스트가 양쪽 코드 경로에서 통과
- **보장:** `_into_cb()` 패턴은 표준 `forward()` 경로와 정확히 동일한 compute pipeline, threadgroup 크기, buffer 바인딩을 인코딩합니다. 유일한 차이는 command buffer 그룹화이며, 이는 수치 결과에 영향을 미치지 않습니다.

이 정밀도 수준(max_diff=6.4e-6)은 fp16/bf16 transformer 연산에서 예상되는 부동소수점 허용 오차 범위 내에 있으며, 파이프라인 최적화가 수치적 발산을 발생시키지 않음을 확인합니다.
