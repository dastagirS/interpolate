use std::{
    ffi::{CStr, c_char, c_float, c_int, c_uchar, c_uint},
    io::{BufReader, Read, Write},
    path::{Path, PathBuf},
    process::{Child, ChildStdin, ChildStdout, Command, Stdio},
    ptr::NonNull,
    time::Duration,
};

const BACKEND_ABI_VERSION: u32 = 2;
const ERROR_MESSAGE_SIZE: usize = 512;
const GPU_NAME_SIZE: usize = 256;
const RGB_CHANNEL_COUNT: usize = 3;
const CPU_THREAD_COUNT_MIN: i32 = 1;
const CPU_THREAD_COUNT_MAX: i32 = 16;
const STATUS_OK: i32 = 0;
const CUDA_WORKER_STARTUP_TIMEOUT: Duration = Duration::from_secs(30);
const CUDA_FRAME_SIZE_MAX: usize = 128 * 1024 * 1024;
const CUDA_WORKER_READY: &[u8; 4] = b"RIF1";
const PYTORCH_MODEL_RELATIVE: &str = "models/rife-v4.25/flownet_v4.25.pkl";
const PYTORCH_MODEL_ENVIRONMENT: &str = "INTERPOLATE_PYTORCH_MODEL";
const PYTHON_EXECUTABLE_ENVIRONMENT: &str = "INTERPOLATE_PYTHON_EXECUTABLE";
const PYTHON_EXECUTABLE_RELATIVE: &str = "runtime/python/bin/python3";
const PYTORCH_WORKER_SOURCE: &str = include_str!("../scripts/pytorch_rife_worker.py");

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InferenceBackend {
    VulkanNcnn,
    CudaPytorchVapourSynth,
}

pub struct CudaInferenceStatus {
    pub available: bool,
    pub reason: String,
}

pub fn cuda_inference_status() -> CudaInferenceStatus {
    let model_available = resolve_pytorch_model_path().is_some();
    let dependency_status = python_executable().map_or_else(
        || {
            Err(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                "Python runtime not found",
            ))
        },
        |python| {
            Command::new(python)
                .args([
                    "-c",
                    "import torch, vapoursynth, vsrife; raise SystemExit(0 if torch.cuda.is_available() else 1)",
                ])
                .stdin(Stdio::null())
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .status()
        },
    );
    let (available, reason) = match (model_available, dependency_status) {
        (false, _) => (
            false,
            format!("PyTorch RIFE model is missing at {PYTORCH_MODEL_RELATIVE}"),
        ),
        (true, Ok(status)) if status.success() => (
            true,
            "PyTorch CUDA with VapourSynth/vs-rife is available".to_owned(),
        ),
        (true, Ok(status)) => (
            false,
            format!("Python CUDA dependencies are unavailable (exit status {status})"),
        ),
        (true, Err(error)) => (
            false,
            format!("failed to probe Python CUDA dependencies: {error}"),
        ),
    };
    assert!(!reason.is_empty(), "CUDA status reason must not be empty");
    assert!(reason.len() < 256, "CUDA status reason must remain bounded");
    CudaInferenceStatus { available, reason }
}

fn resolve_pytorch_model_path() -> Option<PathBuf> {
    if let Some(path) = std::env::var_os(PYTORCH_MODEL_ENVIRONMENT) {
        let path = PathBuf::from(path);
        return path.is_file().then_some(path);
    }
    let executable = std::env::current_exe().ok()?;
    let executable_directory = executable.parent()?;
    let candidates = [
        executable_directory.join(PYTORCH_MODEL_RELATIVE),
        Path::new(env!("CARGO_MANIFEST_DIR")).join(PYTORCH_MODEL_RELATIVE),
    ];
    candidates.into_iter().find(|path| path.is_file())
}

fn python_executable() -> Option<PathBuf> {
    if let Some(path) = std::env::var_os(PYTHON_EXECUTABLE_ENVIRONMENT) {
        let path = PathBuf::from(path);
        return path.is_file().then_some(path);
    }
    let executable = std::env::current_exe().ok()?;
    let executable_directory = executable.parent()?;
    let bundled_path = executable_directory.join(PYTHON_EXECUTABLE_RELATIVE);
    if bundled_path.is_file() {
        return Some(bundled_path);
    }
    if cfg!(debug_assertions) {
        return Some(PathBuf::from("python3"));
    }
    None
}

#[cfg(test)]
const STATUS_INVALID_ARGUMENT: i32 = 1;

#[repr(C)]
struct NativeBackendConfiguration {
    gpu_index: c_int,
    cpu_thread_count: c_int,
    use_uhd_mode: c_int,
}

#[repr(C)]
struct NativeBackend {
    _private: [u8; 0],
}

unsafe extern "C" {
    fn interpolate_backend_abi_version() -> c_uint;
    fn interpolate_backend_gpu_count(
        error_message: *mut c_char,
        error_message_size: usize,
    ) -> c_int;
    fn interpolate_backend_gpu_name(
        gpu_index: c_int,
        gpu_name: *mut c_char,
        gpu_name_size: usize,
        error_message: *mut c_char,
        error_message_size: usize,
    ) -> c_int;
    fn interpolate_backend_create(
        configuration: *const NativeBackendConfiguration,
        model_directory: *const c_char,
        model_directory_size: usize,
        backend_out: *mut *mut NativeBackend,
        error_message: *mut c_char,
        error_message_size: usize,
    ) -> c_int;
    fn interpolate_backend_process_rgb24(
        backend: *mut NativeBackend,
        frame_before: *const c_uchar,
        frame_before_size: usize,
        frame_after: *const c_uchar,
        frame_after_size: usize,
        width: c_uint,
        height: c_uint,
        row_stride: usize,
        timestep: c_float,
        frame_output: *mut c_uchar,
        output_size: usize,
        error_message: *mut c_char,
        error_message_size: usize,
    ) -> c_int;
    fn interpolate_backend_destroy(backend: *mut NativeBackend);
    fn interpolate_backend_shutdown();
}

pub struct Backend {
    native: NonNull<NativeBackend>,
}

// A backend is moved to exactly one processing worker. Native calls remain
// serial because Backend does not implement Sync.
unsafe impl Send for Backend {}

impl Backend {
    pub fn create(
        model_directory: &Path,
        gpu_index: i32,
        use_uhd_mode: bool,
    ) -> Result<Self, String> {
        assert!(
            CPU_THREAD_COUNT_MIN > 0,
            "thread count minimum must be positive"
        );
        assert!(
            CPU_THREAD_COUNT_MAX >= CPU_THREAD_COUNT_MIN,
            "thread limits must be ordered"
        );

        let native_abi_version = unsafe { interpolate_backend_abi_version() };
        if native_abi_version != BACKEND_ABI_VERSION {
            return Err(format!(
                "native ABI version {native_abi_version} does not match required version {BACKEND_ABI_VERSION}"
            ));
        }
        let model_directory_bytes = path_bytes(model_directory)?;
        let configuration = NativeBackendConfiguration {
            gpu_index,
            cpu_thread_count: CPU_THREAD_COUNT_MIN,
            use_uhd_mode: i32::from(use_uhd_mode),
        };
        let mut native = std::ptr::null_mut();
        let mut error_message = [0_i8; ERROR_MESSAGE_SIZE];
        let status = unsafe {
            interpolate_backend_create(
                &configuration,
                model_directory_bytes.as_ptr().cast(),
                model_directory_bytes.len(),
                &mut native,
                error_message.as_mut_ptr(),
                error_message.len(),
            )
        };
        if status != STATUS_OK {
            return Err(native_error(status, &error_message));
        }
        let native = NonNull::new(native)
            .ok_or_else(|| "native backend returned a null handle".to_owned())?;
        assert!(
            !model_directory_bytes.is_empty(),
            "validated model path must not be empty"
        );
        assert_eq!(
            native_abi_version, BACKEND_ABI_VERSION,
            "validated backend ABI must remain compatible"
        );
        Ok(Self { native })
    }

    pub fn interpolate_rgb24(
        &mut self,
        frame_before: &[u8],
        frame_after: &[u8],
        width: u32,
        height: u32,
        timestep: f32,
        frame_output: &mut [u8],
    ) -> Result<(), String> {
        assert!(width > 0, "frame width must be positive");
        assert!(height > 0, "frame height must be positive");

        let row_stride = usize::try_from(width)
            .ok()
            .and_then(|value| value.checked_mul(RGB_CHANNEL_COUNT))
            .ok_or_else(|| "frame row size exceeds addressable memory".to_owned())?;
        let frame_size = usize::try_from(height)
            .ok()
            .and_then(|value| value.checked_mul(row_stride))
            .ok_or_else(|| "frame size exceeds addressable memory".to_owned())?;
        if frame_before.len() != frame_size
            || frame_after.len() != frame_size
            || frame_output.len() != frame_size
        {
            return Err(format!(
                "RGB24 buffers must each contain exactly {frame_size} bytes"
            ));
        }
        if !timestep.is_finite() || timestep <= 0.0 || timestep >= 1.0 {
            return Err("interpolation timestep must be strictly between zero and one".to_owned());
        }

        let mut error_message = [0_i8; ERROR_MESSAGE_SIZE];
        let status = unsafe {
            interpolate_backend_process_rgb24(
                self.native.as_ptr(),
                frame_before.as_ptr(),
                frame_before.len(),
                frame_after.as_ptr(),
                frame_after.len(),
                width,
                height,
                row_stride,
                timestep,
                frame_output.as_mut_ptr(),
                frame_output.len(),
                error_message.as_mut_ptr(),
                error_message.len(),
            )
        };
        if status != STATUS_OK {
            return Err(native_error(status, &error_message));
        }
        assert_eq!(
            frame_output.len(),
            frame_size,
            "output size must remain unchanged"
        );
        assert_eq!(
            usize::try_from(width)
                .ok()
                .and_then(|value| value.checked_mul(RGB_CHANNEL_COUNT)),
            Some(row_stride),
            "row stride must remain RGB24"
        );
        Ok(())
    }
}

pub struct CudaBackend {
    child: Child,
    input: ChildStdin,
    output: BufReader<ChildStdout>,
    frame_size: usize,
}

impl CudaBackend {
    pub fn create(
        width: u32,
        height: u32,
        gpu_index: i32,
        use_half_scale: bool,
    ) -> Result<Self, String> {
        assert!(width > 0, "CUDA frame width must be positive");
        assert!(height > 0, "CUDA frame height must be positive");
        assert!(gpu_index >= 0, "CUDA GPU index must be non-negative");
        assert!(width <= 16_384, "CUDA frame width must remain bounded");
        assert!(height <= 16_384, "CUDA frame height must remain bounded");
        let frame_size = usize::try_from(width)
            .ok()
            .and_then(|value| value.checked_mul(usize::try_from(height).ok()?))
            .and_then(|value| value.checked_mul(RGB_CHANNEL_COUNT))
            .ok_or_else(|| "CUDA frame size exceeds addressable memory".to_owned())?;
        if frame_size > CUDA_FRAME_SIZE_MAX {
            return Err("CUDA frame exceeds the safety limit".to_owned());
        }
        let model_path = resolve_pytorch_model_path()
            .ok_or_else(|| "bundled PyTorch RIFE 4.25 model was not found".to_owned())?;
        let width_text = width.to_string();
        let height_text = height.to_string();
        let gpu_text = gpu_index.to_string();
        let python = python_executable()
            .ok_or_else(|| "bundled Python runtime could not be located".to_owned())?;
        let mut command = Command::new(python);
        command
            .args(["-u", "-c", PYTORCH_WORKER_SOURCE, "--width"])
            .arg(&width_text)
            .args(["--height"])
            .arg(&height_text)
            .args(["--gpu"])
            .arg(&gpu_text)
            .args(["--model"])
            .arg(&model_path);
        if use_half_scale {
            command.arg("--half-scale");
        }
        let mut child = command
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|error| format!("failed to start PyTorch RIFE worker: {error}"))?;
        let Some(input) = child.stdin.take() else {
            let _ = terminate_process(&mut child);
            return Err("PyTorch RIFE worker stdin is unavailable".to_owned());
        };
        let Some(output) = child.stdout.take() else {
            let _ = terminate_process(&mut child);
            return Err("PyTorch RIFE worker stdout is unavailable".to_owned());
        };
        let (ready_sender, ready_receiver) = std::sync::mpsc::sync_channel(1);
        std::thread::spawn(move || {
            let mut output = BufReader::new(output);
            let mut ready = [0_u8; CUDA_WORKER_READY.len()];
            let result = output
                .read_exact(&mut ready)
                .map(|()| (ready == *CUDA_WORKER_READY, output));
            let _ = ready_sender.send(result);
        });
        let ready_result = match ready_receiver.recv_timeout(CUDA_WORKER_STARTUP_TIMEOUT) {
            Ok(result) => result,
            Err(_) => {
                let _ = terminate_process(&mut child);
                return Err("PyTorch RIFE worker startup timed out".to_owned());
            }
        };
        let output = match ready_result {
            Ok((true, output)) => output,
            Ok((false, _)) => {
                let _ = terminate_process(&mut child);
                return Err("PyTorch RIFE worker returned an invalid handshake".to_owned());
            }
            Err(error) => {
                let _ = terminate_process(&mut child);
                return Err(format!(
                    "PyTorch RIFE worker failed during startup: {error}"
                ));
            }
        };
        assert!(frame_size > 0, "CUDA frame size must remain positive");
        Ok(Self {
            child,
            input,
            output,
            frame_size,
        })
    }

    pub fn interpolate_rgb24(
        &mut self,
        frame_before: &[u8],
        frame_after: &[u8],
        width: u32,
        height: u32,
        timestep: f32,
        frame_output: &mut [u8],
    ) -> Result<(), String> {
        assert!(width > 0, "CUDA frame width must be positive");
        assert!(height > 0, "CUDA frame height must be positive");
        let expected_size = usize::try_from(width)
            .ok()
            .and_then(|value| value.checked_mul(usize::try_from(height).ok()?))
            .and_then(|value| value.checked_mul(RGB_CHANNEL_COUNT))
            .ok_or_else(|| "CUDA frame size exceeds addressable memory".to_owned())?;
        if self.frame_size != expected_size
            || frame_before.len() != expected_size
            || frame_after.len() != expected_size
            || frame_output.len() != expected_size
        {
            return Err("CUDA worker frame dimensions changed during the job".to_owned());
        }
        if !timestep.is_finite() || !(0.0..1.0).contains(&timestep) {
            return Err("CUDA worker timestep must be strictly between zero and one".to_owned());
        }
        self.input
            .write_all(&timestep.to_le_bytes())
            .and_then(|()| self.input.write_all(frame_before))
            .and_then(|()| self.input.write_all(frame_after))
            .and_then(|()| self.input.flush())
            .map_err(|error| format!("failed to send frames to PyTorch RIFE worker: {error}"))?;
        self.output.read_exact(frame_output).map_err(|error| {
            format!("failed to receive frame from PyTorch RIFE worker: {error}")
        })?;
        assert_eq!(frame_output.len(), expected_size);
        Ok(())
    }
}

impl Drop for CudaBackend {
    fn drop(&mut self) {
        assert!(self.frame_size > 0, "CUDA frame size must remain positive");
        let _ = terminate_process(&mut self.child);
        assert!(
            self.frame_size <= CUDA_FRAME_SIZE_MAX,
            "CUDA frame must remain bounded"
        );
    }
}

fn terminate_process(child: &mut Child) -> Result<(), String> {
    if child
        .try_wait()
        .map_err(|error| format!("failed to query worker: {error}"))?
        .is_none()
    {
        child
            .kill()
            .map_err(|error| format!("failed to stop worker: {error}"))?;
    }
    child
        .wait()
        .map(|_| ())
        .map_err(|error| format!("failed to reap worker: {error}"))
}

pub enum InferenceEngine {
    Vulkan(Backend),
    Cuda(CudaBackend),
}

impl InferenceEngine {
    pub fn interpolate_rgb24(
        &mut self,
        frame_before: &[u8],
        frame_after: &[u8],
        width: u32,
        height: u32,
        timestep: f32,
        frame_output: &mut [u8],
    ) -> Result<(), String> {
        match self {
            Self::Vulkan(backend) => backend.interpolate_rgb24(
                frame_before,
                frame_after,
                width,
                height,
                timestep,
                frame_output,
            ),
            Self::Cuda(backend) => backend.interpolate_rgb24(
                frame_before,
                frame_after,
                width,
                height,
                timestep,
                frame_output,
            ),
        }
    }
}

impl Drop for Backend {
    fn drop(&mut self) {
        assert_eq!(
            unsafe { interpolate_backend_abi_version() },
            BACKEND_ABI_VERSION,
            "backend ABI must be valid before destruction"
        );
        assert!(BACKEND_ABI_VERSION > 0, "backend ABI version must be valid");
        unsafe { interpolate_backend_destroy(self.native.as_ptr()) };
        assert!(
            ERROR_MESSAGE_SIZE > 0,
            "error storage invariant must remain valid"
        );
        assert!(
            GPU_NAME_SIZE > 0,
            "GPU name storage invariant must remain valid"
        );
    }
}

pub fn shutdown() {
    assert!(
        BACKEND_ABI_VERSION > 0,
        "backend ABI version must be positive"
    );
    assert!(
        ERROR_MESSAGE_SIZE > 0,
        "backend error storage must remain configured"
    );
    unsafe { interpolate_backend_shutdown() };
    assert_eq!(
        unsafe { interpolate_backend_abi_version() },
        BACKEND_ABI_VERSION,
        "backend ABI must remain available"
    );
    assert!(GPU_NAME_SIZE > 0, "GPU name storage must remain configured");
}

pub fn gpu_names() -> Result<Vec<String>, String> {
    assert!(
        GPU_NAME_SIZE > 1,
        "GPU name buffer must hold text and a terminator"
    );
    assert!(
        ERROR_MESSAGE_SIZE > 1,
        "error buffer must hold text and a terminator"
    );

    let native_abi_version = unsafe { interpolate_backend_abi_version() };
    if native_abi_version != BACKEND_ABI_VERSION {
        return Err(format!(
            "native ABI version {native_abi_version} does not match required version {BACKEND_ABI_VERSION}"
        ));
    }
    let mut error_message = [0_i8; ERROR_MESSAGE_SIZE];
    let gpu_count =
        unsafe { interpolate_backend_gpu_count(error_message.as_mut_ptr(), error_message.len()) };
    if gpu_count < 0 {
        return Err(native_error(-gpu_count, &error_message));
    }
    let gpu_count = usize::try_from(gpu_count)
        .map_err(|_| "native backend returned an invalid GPU count".to_owned())?;
    const GPU_COUNT_MAX: usize = 16;
    if gpu_count > GPU_COUNT_MAX {
        return Err(format!(
            "native backend reported more than {GPU_COUNT_MAX} GPUs"
        ));
    }

    let mut names = Vec::with_capacity(gpu_count);
    for gpu_index in 0..gpu_count {
        let mut gpu_name = [0_i8; GPU_NAME_SIZE];
        error_message.fill(0);
        let gpu_index_native = i32::try_from(gpu_index)
            .map_err(|_| "bounded GPU index does not fit the native ABI".to_owned())?;
        let status = unsafe {
            interpolate_backend_gpu_name(
                gpu_index_native,
                gpu_name.as_mut_ptr(),
                gpu_name.len(),
                error_message.as_mut_ptr(),
                error_message.len(),
            )
        };
        if status != STATUS_OK {
            return Err(native_error(status, &error_message));
        }
        names.push(c_buffer_to_string(&gpu_name));
    }
    assert_eq!(names.len(), gpu_count, "every GPU must have one name");
    assert!(names.len() <= GPU_COUNT_MAX, "GPU list must remain bounded");
    Ok(names)
}

fn path_bytes(path: &Path) -> Result<&[u8], String> {
    assert!(
        ERROR_MESSAGE_SIZE > 0,
        "error buffer size must be configured"
    );
    assert!(
        RGB_CHANNEL_COUNT == 3,
        "backend pixel format must remain RGB24"
    );
    #[cfg(unix)]
    {
        use std::os::unix::ffi::OsStrExt;
        let bytes = path.as_os_str().as_bytes();
        if bytes.is_empty() {
            return Err("model directory must not be empty".to_owned());
        }
        assert!(!bytes.is_empty(), "validated path must not be empty");
        assert!(
            bytes.len() <= isize::MAX as usize,
            "path must fit a Rust slice"
        );
        Ok(bytes)
    }
    #[cfg(not(unix))]
    {
        let _ = path;
        Err("the native backend currently supports Linux paths only".to_owned())
    }
}

fn c_buffer_to_string(buffer: &[c_char]) -> String {
    assert!(!buffer.is_empty(), "C text buffer must not be empty");
    assert_eq!(
        buffer[buffer.len() - 1],
        0,
        "C text buffer must be terminated"
    );
    let text = unsafe { CStr::from_ptr(buffer.as_ptr()) }
        .to_string_lossy()
        .into_owned();
    assert!(
        text.len() < buffer.len(),
        "decoded C string must fit its buffer"
    );
    assert!(
        text.capacity() >= text.len(),
        "decoded string capacity must be valid"
    );
    text
}

fn native_error(status: i32, buffer: &[c_char]) -> String {
    assert!(status != STATUS_OK, "native error status must be nonzero");
    assert!(!buffer.is_empty(), "native error buffer must not be empty");
    let message = c_buffer_to_string(buffer);
    let result = if message.is_empty() {
        format!("native backend failed with status {status}")
    } else {
        format!("{message} (native status {status})")
    };
    assert!(
        !result.is_empty(),
        "formatted native error must not be empty"
    );
    assert!(
        result.len() >= message.len(),
        "formatted error must retain the native message"
    );
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn native_abi_rejects_small_outputs_without_aborting() {
        let mut gpu_name = [0_i8; 1];
        let mut error_message = [0_i8; ERROR_MESSAGE_SIZE];
        let gpu_name_status = unsafe {
            interpolate_backend_gpu_name(
                0,
                gpu_name.as_mut_ptr(),
                gpu_name.len(),
                error_message.as_mut_ptr(),
                error_message.len(),
            )
        };
        assert_eq!(gpu_name_status, STATUS_INVALID_ARGUMENT);
        assert_eq!(gpu_name[0], 0, "rejected output must remain terminated");

        let configuration = NativeBackendConfiguration {
            gpu_index: 0,
            cpu_thread_count: CPU_THREAD_COUNT_MIN,
            use_uhd_mode: 0,
        };
        let model_directory = Path::new(env!("CARGO_MANIFEST_DIR")).join("models/rife-v4.25");
        let model_directory_bytes = path_bytes(&model_directory).expect("model path must be valid");
        let create_status = unsafe {
            interpolate_backend_create(
                &configuration,
                model_directory_bytes.as_ptr().cast(),
                model_directory_bytes.len(),
                std::ptr::null_mut(),
                error_message.as_mut_ptr(),
                error_message.len(),
            )
        };
        assert_eq!(create_status, STATUS_INVALID_ARGUMENT);
        assert!(!c_buffer_to_string(&error_message).is_empty());

        let mut native = std::ptr::null_mut();
        let model_directory_with_null = b"/tmp/interpolate\0invalid";
        error_message.fill(0);
        let null_path_status = unsafe {
            interpolate_backend_create(
                &configuration,
                model_directory_with_null.as_ptr().cast(),
                model_directory_with_null.len(),
                &mut native,
                error_message.as_mut_ptr(),
                error_message.len(),
            )
        };
        assert_eq!(null_path_status, STATUS_INVALID_ARGUMENT);
        assert!(
            native.is_null(),
            "rejected model path must not create a backend"
        );
    }

    #[test]
    fn native_backend_interpolates_fixed_rgb_frames() {
        const WIDTH: u32 = 64;
        const HEIGHT: u32 = 64;
        const FRAME_SIZE: usize = WIDTH as usize * HEIGHT as usize * RGB_CHANNEL_COUNT;
        assert!(FRAME_SIZE > 0, "test frame must not be empty");
        assert!(
            WIDTH.is_multiple_of(64),
            "test width should avoid external padding variables"
        );

        let names = gpu_names().expect("Vulkan GPU enumeration should succeed");
        assert!(
            !names.is_empty(),
            "at least one Vulkan GPU is required for this test"
        );
        assert!(names.len() <= 16, "GPU count must stay bounded");

        let model_directory = Path::new(env!("CARGO_MANIFEST_DIR")).join("models/rife-v4.25");
        let mut backend = Backend::create(&model_directory, 0, false)
            .expect("RIFE 4.25 backend should initialize");
        let frame_before = vec![0_u8; FRAME_SIZE];
        let frame_after = vec![255_u8; FRAME_SIZE];
        let mut frame_output = vec![0_u8; FRAME_SIZE];
        let short_frame = vec![0_u8; FRAME_SIZE - 1];
        let mut native_error_message = [0_i8; ERROR_MESSAGE_SIZE];
        let native_size_status = unsafe {
            interpolate_backend_process_rgb24(
                backend.native.as_ptr(),
                frame_before.as_ptr(),
                short_frame.len(),
                frame_after.as_ptr(),
                frame_after.len(),
                WIDTH,
                HEIGHT,
                WIDTH as usize * RGB_CHANNEL_COUNT,
                0.5,
                frame_output.as_mut_ptr(),
                frame_output.len(),
                native_error_message.as_mut_ptr(),
                native_error_message.len(),
            )
        };
        assert_eq!(native_size_status, STATUS_INVALID_ARGUMENT);
        assert!(frame_output.iter().all(|value| *value == 0));

        let invalid_size_error = backend
            .interpolate_rgb24(
                &short_frame,
                &frame_after,
                WIDTH,
                HEIGHT,
                0.5,
                &mut frame_output,
            )
            .expect_err("short input must be rejected");
        assert!(invalid_size_error.contains("exactly"));
        assert!(frame_output.iter().all(|value| *value == 0));

        for invalid_timestep in [f32::NAN, 0.0, 1.0] {
            let invalid_timestep_error = backend
                .interpolate_rgb24(
                    &frame_before,
                    &frame_after,
                    WIDTH,
                    HEIGHT,
                    invalid_timestep,
                    &mut frame_output,
                )
                .expect_err("invalid timestep must be rejected");
            assert!(invalid_timestep_error.contains("strictly between"));
            assert!(frame_output.iter().all(|value| *value == 0));
        }

        backend
            .interpolate_rgb24(
                &frame_before,
                &frame_after,
                WIDTH,
                HEIGHT,
                0.5,
                &mut frame_output,
            )
            .expect("RIFE inference should succeed");
        assert!(
            frame_output.iter().any(|value| *value > 0),
            "output must differ from black"
        );
        assert!(
            frame_output.iter().any(|value| *value < 255),
            "output must differ from white"
        );
    }
}
