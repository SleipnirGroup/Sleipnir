use std::env;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

fn main() {
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let sleipnir_root = manifest_dir
        .parent()
        .expect("rust/ must sit inside the Sleipnir repo")
        .to_path_buf();

    // Re-run build.rs when the diagnostics feature flips so the cmake
    // invalidation block below sees the new value.
    println!("cargo:rerun-if-env-changed=CARGO_FEATURE_DIAGNOSTICS");

    let feature_requested = env::var("CARGO_FEATURE_DIAGNOSTICS").is_ok();

    // Sleipnir's diagnostics path pulls in C++23's `<print>`, which
    // ships in GCC 14+ libstdc++ and libc++ 19+. On older toolchains
    // the include just errors. Probe for it and auto-fall-back if the
    // feature was requested but the header isn't there.
    let toolchain_has_print = feature_requested && probe_cpp23_print();
    if feature_requested && !toolchain_has_print {
        println!(
            "cargo:warning=hafgufa: `diagnostics` feature requested but \
             your C++ toolchain has no `<print>` header (needs GCC 14+ \
             libstdc++ or libc++ 19+). Building without diagnostic \
             output; `Options::diagnostics(true)` calls will be silent."
        );
    }
    let disable_diagnostics = !toolchain_has_print;

    // Cmake's own up-to-date check doesn't re-compile Sleipnir's
    // objects when only a `-D` define changed between runs — the
    // Ninja/Make layer compares source mtimes but not preprocessor
    // flags. If the cached `SLEIPNIR_DISABLE_DIAGNOSTICS` value
    // disagrees with what we'd pass on this invocation, wipe the
    // build tree so cmake reconfigures from scratch. Without this,
    // flipping the `diagnostics` feature ships a library whose solver
    // symbols were compiled against the old define value.
    let cmake_build_dir = PathBuf::from(env::var("OUT_DIR").unwrap()).join("build");
    let cache_path = cmake_build_dir.join("CMakeCache.txt");
    if let Ok(cache) = std::fs::read_to_string(&cache_path) {
        let expected = if disable_diagnostics { "ON" } else { "OFF" };
        let matches = cache.lines().any(|line| {
            // Typical cache entry: `SLEIPNIR_DISABLE_DIAGNOSTICS:BOOL=ON`.
            let line = line.trim();
            line.starts_with("SLEIPNIR_DISABLE_DIAGNOSTICS:")
                && line.split('=').nth(1).map(str::trim) == Some(expected)
        });
        if !matches {
            let _ = std::fs::remove_dir_all(&cmake_build_dir);
        }
    }

    let mut config = cmake::Config::new(&sleipnir_root);
    config
        .define("BUILD_SHARED_LIBS", "OFF")
        .define("BUILD_TESTING", "OFF")
        .define("SLEIPNIR_BUILD_BENCHMARKS", "OFF")
        .define("SLEIPNIR_BUILD_EXAMPLES", "OFF")
        .define("SLEIPNIR_BUILD_PYTHON", "OFF")
        .define("CMAKE_POSITION_INDEPENDENT_CODE", "ON")
        .define("CMAKE_CXX_STANDARD", "23");
    // Always pass the define explicitly so the cmake cache always
    // contains an up-to-date value (whether ON or OFF) that the
    // invalidation logic above can match against on the next run.
    config.define(
        "SLEIPNIR_DISABLE_DIAGNOSTICS",
        if disable_diagnostics { "ON" } else { "OFF" },
    );
    let dst = config.build();

    let build_dir = dst.join("build");
    let install_include = dst.join("include");
    let install_lib = dst.join("lib");

    let eigen_include = find_dep_include(&build_dir, "eigen")
        .unwrap_or_else(|| panic!("could not locate fetched Eigen headers under {}", build_dir.display()));
    let small_vector_include = find_dep_include(&build_dir, "small_vector")
        .unwrap_or_else(|| panic!("could not locate fetched small_vector headers under {}", build_dir.display()));

    let shim_dir = manifest_dir.join("cxx");

    let mut shim_build = cxx_build::bridge("src/ffi.rs");
    shim_build
        .file(shim_dir.join("shim.cpp"))
        .include(&install_include)
        .include(&eigen_include)
        .include(&small_vector_include)
        .include(&shim_dir)
        .flag_if_supported("-std=c++23")
        .flag_if_supported("/std:c++latest")
        .flag_if_supported("-Wno-unused-parameter");
    if disable_diagnostics {
        shim_build.define("SLEIPNIR_DISABLE_DIAGNOSTICS", None);
    }
    shim_build.compile("hafgufa_shim");

    println!("cargo:rerun-if-changed=src/ffi.rs");
    println!("cargo:rerun-if-changed=cxx/shim.h");
    println!("cargo:rerun-if-changed=cxx/shim.cpp");
    println!("cargo:rerun-if-changed={}", sleipnir_root.join("include").display());
    println!("cargo:rerun-if-changed={}", sleipnir_root.join("src").display());
    println!("cargo:rerun-if-changed={}", sleipnir_root.join("CMakeLists.txt").display());

    println!("cargo:rustc-link-search=native={}", install_lib.display());
    println!("cargo:rustc-link-search=native={}", dst.join("lib64").display());
    println!("cargo:rustc-link-lib=static=Sleipnir");

    link_cxx_stdlib();
}

fn find_dep_include(build_dir: &Path, name_fragment: &str) -> Option<PathBuf> {
    let deps = build_dir.join("_deps");
    if !deps.exists() {
        return None;
    }
    let name_fragment = name_fragment.to_lowercase();
    let entries = std::fs::read_dir(&deps).ok()?;
    for entry in entries.flatten() {
        let path = entry.path();
        let file_name = path.file_name()?.to_string_lossy().to_lowercase();
        if file_name.contains(&name_fragment) && file_name.ends_with("-src") {
            let include = path.join("include");
            if include.exists() {
                return Some(include);
            }
            return Some(path);
        }
    }
    None
}

/// Try to compile a trivial `#include <print>` C++ translation unit.
/// Returns true if the compiler accepts the header, false otherwise.
/// Squashes stderr to keep cargo output tidy.
fn probe_cpp23_print() -> bool {
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let probe_src = out_dir.join("probe_print.cc");
    let probe_obj = out_dir.join("probe_print.o");
    if std::fs::write(&probe_src, "#include <print>\nint main(){}\n").is_err() {
        return false;
    }

    // Pick the compiler the same way cmake/cc-rs would: honour $CXX,
    // fall back to the target-default.
    let cxx = env::var("CXX").unwrap_or_else(|_| {
        let target = env::var("TARGET").unwrap_or_default();
        if target.contains("msvc") {
            "cl".to_string()
        } else if target.contains("apple") {
            "clang++".to_string()
        } else {
            "c++".to_string()
        }
    });

    let ok = Command::new(&cxx)
        .arg("-std=c++23")
        .arg("-c")
        .arg(&probe_src)
        .arg("-o")
        .arg(&probe_obj)
        .stderr(Stdio::null())
        .stdout(Stdio::null())
        .status()
        .map(|s| s.success())
        .unwrap_or(false);

    let _ = std::fs::remove_file(&probe_src);
    let _ = std::fs::remove_file(&probe_obj);
    ok
}

fn link_cxx_stdlib() {
    let target = env::var("TARGET").unwrap_or_default();
    if target.contains("apple") {
        println!("cargo:rustc-link-lib=c++");
    } else if target.contains("msvc") {
        // MSVC links the C++ runtime automatically.
    } else {
        println!("cargo:rustc-link-lib=stdc++");
        println!("cargo:rustc-link-lib=pthread");
    }
}
