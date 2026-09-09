use std::{env, path::PathBuf, process::Command};

const NATIVE_BUILD_PROFILE: &str = "Release";
const NATIVE_BUILD_JOB_COUNT: &str = "1";

fn run_command(command: &mut Command, description: &str) {
    assert!(
        !description.is_empty(),
        "command description must not be empty"
    );
    assert!(
        !NATIVE_BUILD_JOB_COUNT.is_empty(),
        "native build job count must be configured"
    );

    let status = command
        .status()
        .unwrap_or_else(|error| panic!("failed to start {description}: {error}"));
    assert!(status.code().is_some(), "{description} must exit normally");
    assert!(status.success(), "{description} failed with {status}");
}

fn main() {
    assert!(
        NATIVE_BUILD_PROFILE == "Release",
        "native dependency must use its low-overhead release profile"
    );
    assert!(
        NATIVE_BUILD_JOB_COUNT == "1",
        "native dependency must compile serially"
    );

    let manifest_directory = PathBuf::from(
        env::var_os("CARGO_MANIFEST_DIR").expect("Cargo must provide CARGO_MANIFEST_DIR"),
    );
    let output_directory =
        PathBuf::from(env::var_os("OUT_DIR").expect("Cargo must provide OUT_DIR"));
    let source_directory = manifest_directory.join("native");
    let build_directory = output_directory.join("native-build");

    assert!(
        source_directory.is_dir(),
        "native source directory must exist"
    );
    assert!(
        output_directory.is_dir(),
        "Cargo output directory must exist"
    );

    run_command(
        Command::new("cmake")
            .arg("-S")
            .arg(&source_directory)
            .arg("-B")
            .arg(&build_directory)
            .arg(format!("-DCMAKE_BUILD_TYPE={NATIVE_BUILD_PROFILE}"))
            .arg("-DCMAKE_INTERPROCEDURAL_OPTIMIZATION=OFF")
            .env("CMAKE_BUILD_PARALLEL_LEVEL", NATIVE_BUILD_JOB_COUNT),
        "native CMake configuration",
    );
    run_command(
        Command::new("cmake")
            .arg("--build")
            .arg(&build_directory)
            .arg("--config")
            .arg(NATIVE_BUILD_PROFILE)
            .arg("--target")
            .arg("interpolate_backend")
            .arg("--parallel")
            .arg(NATIVE_BUILD_JOB_COUNT)
            .env("CMAKE_BUILD_PARALLEL_LEVEL", NATIVE_BUILD_JOB_COUNT),
        "native CMake build",
    );

    let backend_library = build_directory.join("libinterpolate_backend.a");
    assert!(
        backend_library.is_file(),
        "native backend library must exist"
    );
    assert!(
        build_directory.is_dir(),
        "native build directory must remain valid"
    );

    println!(
        "cargo:rustc-link-search=native={}",
        build_directory.display()
    );
    println!(
        "cargo:rustc-link-search=native={}",
        build_directory.join("ncnn/src").display()
    );
    println!(
        "cargo:rustc-link-search=native={}",
        build_directory.join("ncnn/glslang/SPIRV").display()
    );
    println!(
        "cargo:rustc-link-search=native={}",
        build_directory.join("ncnn/glslang/glslang").display()
    );
    println!("cargo:rustc-link-lib=static=interpolate_backend");
    println!("cargo:rustc-link-lib=static=ncnn");
    println!("cargo:rustc-link-lib=dylib=vulkan");
    println!("cargo:rustc-link-lib=dylib=dl");
    println!("cargo:rustc-link-lib=static=SPIRV");
    println!("cargo:rustc-link-lib=static=glslang");
    println!("cargo:rustc-link-lib=dylib=stdc++");
    println!("cargo:rerun-if-changed=native/CMakeLists.txt");
    println!("cargo:rerun-if-changed=native/interpolate_backend.cpp");
    println!("cargo:rerun-if-changed=native/interpolate_backend.h");
    println!("cargo:rerun-if-changed=native/rife");
}
