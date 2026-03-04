# Building and Running

This guide covers cloning, building, and testing the rmlx project.

---

## Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/0xDaizz/RMLX.git
cd rmlx

# 2. Build the entire workspace
cargo build --workspace

# 3. Run tests
cargo test --workspace

# 4. Check formatting and linting
cargo fmt --all --check
cargo clippy --workspace -- -D warnings
```

> **Note**: The first build may take several minutes to download and compile dependencies.

---

## Workspace Structure

rmlx is organized as a Cargo workspace containing the following crates:

```
rmlx/
├── Cargo.toml              # workspace root
├── crates/
│   ├── rmlx-metal/         # Metal GPU abstractions
│   ├── rmlx-alloc/         # Zero-copy memory allocation
│   ├── rmlx-rdma/          # RDMA communication (Thunderbolt 5)
│   ├── rmlx-core/          # Array type, kernel registry
│   ├── rmlx-distributed/   # Distributed inference (EP, pipeline)
│   └── rmlx-nn/            # Neural network layers (Transformer, etc.)
```

---

## Build System Details

### Metal Shader AOT Compilation

The `rmlx-core` crate's `build.rs` pre-compiles Metal shaders during the build process.

**Compilation pipeline**:

```
kernels/*.metal  ->  xcrun metal -c  ->  *.air  ->  xcrun metallib  ->  rmlx_kernels.metallib
```

1. Discovers all `.metal` files in the `kernels/` directory
2. Compiles each file to an intermediate representation (`.air`) using `xcrun -sdk macosx metal -c`
3. Links all `.air` files into a single `.metallib` file using `xcrun -sdk macosx metallib`
4. Sets the `RMLX_METALLIB_PATH` environment variable to the generated `.metallib` path

### Without Xcode

If Xcode is not installed, `build.rs` behaves as follows:

```
cargo:warning=Metal compiler not found (requires Xcode, not just Command Line Tools).
               Skipping shader AOT compilation. GPU kernels will not be available at runtime.
```

- Skips AOT compilation and **only prints a warning** (this is not a build failure)
- `RMLX_METALLIB_PATH` is set to an empty string
- **Test code uses JIT compilation (`compile_source`)**, so it runs normally without Xcode

> **Key point**: Builds and tests proceed normally even without Xcode.
> For optimal performance in production, using AOT-compiled `.metallib` files is recommended.

### Manually Specifying the AOT metallib Path

If you have a pre-compiled `.metallib` file available separately, you can specify its path directly via an environment variable:

```bash
# Pass the AOT metallib path via environment variable
RMLX_METALLIB_PATH=/path/to/rmlx_kernels.metallib cargo build --workspace
```

---

## Building Individual Crates

To build or test a specific crate, use the `-p` flag:

```bash
# Build only the rmlx-metal crate
cargo build -p rmlx-metal

# Run only rmlx-metal tests
cargo test -p rmlx-metal

# Build only the rmlx-core crate
cargo build -p rmlx-core

# Run a specific test
cargo test -p rmlx-metal -- test_basic_metal_compute
```

---

## Development Workflow

### Auto-Build (cargo-watch)

Automatically build and test on file changes:

```bash
# Auto-check on file changes (check for compile errors)
cargo watch -x check

# Auto-test on file changes
cargo watch -x 'test --workspace'

# Watch a specific crate only
cargo watch -x 'test -p rmlx-metal'
```

### CI-Equivalent Verification

Before creating a pull request, you can run the same verification locally that CI performs:

```bash
# Full CI verification sequence
cargo build --workspace \
  && cargo fmt --all --check \
  && cargo clippy --workspace -- -D warnings \
  && cargo test --workspace
```

---

## Distributed 2-Node Validation (RDMA)

For multi-node environments, run this minimal sequence from the control node:

```bash
# Generate hostfile and apply baseline host setup
rmlx config \
  --hosts node1,node2 \
  --backend rdma \
  --over thunderbolt \
  --control-iface en0 \
  --auto-setup \
  --output rmlx-hosts.json \
  --verbose

# Verify RDMA device visibility
rmlx launch \
  --backend rdma \
  --hostfile rmlx-hosts.json \
  -- ibv_devices

# Run RDMA crate tests on both nodes
rmlx launch \
  --backend rdma \
  --hostfile rmlx-hosts.json \
  -- cargo test -p rmlx-rdma -- --nocapture
```

If you only need hostfile generation, omit `--auto-setup`.
For full setup requirements, see [Prerequisites](prerequisites.md).

---

## Troubleshooting

### `xcrun: error: unable to find utility "metal"`

**Cause**: Xcode is not installed, or only Command Line Tools are installed.

**Solution**:
- Install Xcode from the App Store
- Or use JIT mode without AOT compilation (build warnings can be safely ignored)

### `error: no Metal device available`

**Cause**: Running in an environment that does not support Metal (VM, Intel Mac, CI runner, etc.).

**Solution**:
- Run on an Apple Silicon Mac
- Metal-related tests are automatically skipped when no device is available

### Build Error Related to `kernels/` Directory

**Cause**: The `rmlx-core/kernels/` directory is empty or does not exist.

**Solution**:
```bash
# Check the kernels directory
ls crates/rmlx-core/kernels/

# If the directory does not exist (will be added in Phase 2)
mkdir -p crates/rmlx-core/kernels
```

---

Next step: [Running Your First Metal Kernel](first-kernel.md)
