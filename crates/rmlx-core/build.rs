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

fn main() {
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let kernel_dir = Path::new("kernels");

    println!("cargo:rerun-if-changed=kernels/");

    // Find all .metal files
    let metal_files: Vec<PathBuf> = std::fs::read_dir(kernel_dir)
        .expect("failed to read kernels/ directory")
        .filter_map(|entry| {
            let entry = entry.ok()?;
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) == Some("metal") {
                Some(path)
            } else {
                None
            }
        })
        .collect();

    if metal_files.is_empty() {
        println!("cargo:rustc-env=RMLX_METALLIB_PATH=");
        return; // No shaders to compile
    }

    // Check if the Metal compiler toolchain is available (requires full Xcode)
    if !metal_compiler_available() {
        println!(
            "cargo:warning=Metal compiler not found (requires Xcode, not just Command Line Tools). \
             Skipping shader AOT compilation. GPU kernels will not be available at runtime."
        );
        println!("cargo:rustc-env=RMLX_METALLIB_PATH=");
        return;
    }

    // Compile each .metal → .air
    let mut air_files = Vec::new();
    for metal_file in &metal_files {
        let stem = metal_file.file_stem().unwrap().to_str().unwrap();
        let air_path = out_dir.join(format!("{stem}.air"));

        let status = Command::new("xcrun")
            .args(["-sdk", "macosx", "metal"])
            .arg("-c")
            .arg(metal_file)
            .arg("-o")
            .arg(&air_path)
            .status()
            .expect("failed to execute xcrun metal compiler");

        assert!(
            status.success(),
            "Metal shader compilation failed for {}",
            metal_file.display()
        );
        air_files.push(air_path);
    }

    // Link .air → .metallib
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
