use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;

fn metal_compiler_available() -> bool {
    Command::new("xcrun")
        .args(["-sdk", "macosx", "--find", "metal"])
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

/// Recursively collect all .metal files from a directory.
fn collect_metal_files(dir: &Path) -> Vec<PathBuf> {
    let mut files = Vec::new();
    let entries = match std::fs::read_dir(dir) {
        Ok(entries) => entries,
        Err(_) => return files,
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            files.extend(collect_metal_files(&path));
        } else if path.extension().and_then(|e| e.to_str()) == Some("metal") {
            files.push(path);
        }
    }
    files
}

fn main() {
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());

    // --- Source directories ---
    let kernel_dir = Path::new("kernels");
    // Workspace root relative to this crate (crates/rmlx-core/)
    let workspace_root = Path::new("../..").canonicalize().unwrap_or_else(|_| {
        PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap())
            .join("../..")
            .to_path_buf()
    });
    let mlx_compat_dir = workspace_root.join("shaders/mlx_compat");

    // --- Rerun triggers ---
    println!("cargo:rerun-if-changed=kernels/");
    if mlx_compat_dir.exists() {
        println!("cargo:rerun-if-changed={}", mlx_compat_dir.display());
    }
    println!("cargo:rerun-if-env-changed=RMLX_METALLIB_PATH");

    // --- Allow prebuilt metallib override ---
    if let Ok(prebuilt) = env::var("RMLX_METALLIB_PATH") {
        if !prebuilt.is_empty() {
            println!("cargo:rustc-env=RMLX_METALLIB_PATH={prebuilt}");
            return;
        }
    }

    // --- Collect all .metal source files ---
    let mut metal_files: Vec<PathBuf> = collect_metal_files(kernel_dir);
    if mlx_compat_dir.exists() {
        metal_files.extend(collect_metal_files(&mlx_compat_dir));
    }

    if metal_files.is_empty() {
        println!("cargo:rustc-env=RMLX_METALLIB_PATH=");
        return;
    }

    // --- Check Metal compiler availability ---
    if !metal_compiler_available() {
        println!(
            "cargo:warning=Metal compiler not found (requires Xcode, not just Command Line Tools). \
             Skipping shader AOT compilation. GPU kernels will not be available at runtime."
        );
        println!("cargo:rustc-env=RMLX_METALLIB_PATH=");
        return;
    }

    // --- Build include paths ---
    let mut include_paths: Vec<PathBuf> = Vec::new();
    if mlx_compat_dir.exists() {
        include_paths.push(mlx_compat_dir.clone());
    }

    // --- Compile each .metal → .air ---
    let mut air_files = Vec::new();
    for metal_file in &metal_files {
        let stem = metal_file.file_stem().unwrap().to_str().unwrap();
        // Use a unique name to avoid collisions between kernel dirs
        let unique_name = if metal_file.starts_with(&mlx_compat_dir) {
            format!("mlx_{stem}")
        } else {
            stem.to_string()
        };
        let air_path = out_dir.join(format!("{unique_name}.air"));

        let mut cmd = Command::new("xcrun");
        cmd.args(["-sdk", "macosx", "metal"]);
        cmd.arg("-c");
        // Add include paths so headers resolve
        for inc in &include_paths {
            cmd.arg("-I").arg(inc);
        }
        cmd.arg(metal_file);
        cmd.arg("-o").arg(&air_path);

        let status = cmd
            .status()
            .expect("failed to execute xcrun metal compiler");

        assert!(
            status.success(),
            "Metal shader compilation failed for {}",
            metal_file.display()
        );
        air_files.push(air_path);
    }

    // --- Link .air → .metallib ---
    let metallib_path = out_dir.join("rmlx_kernels.metallib");
    let mut cmd = Command::new("xcrun");
    cmd.args(["-sdk", "macosx", "metallib"]);
    for air in &air_files {
        cmd.arg(air);
    }
    cmd.arg("-o").arg(&metallib_path);

    let status = cmd
        .status()
        .expect("failed to execute xcrun metallib linker");
    assert!(status.success(), "Metal library linking failed");

    println!(
        "cargo:rustc-env=RMLX_METALLIB_PATH={}",
        metallib_path.display()
    );
}
