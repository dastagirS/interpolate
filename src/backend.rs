use std::{
    ffi::{CStr, c_char, c_float, c_int, c_uchar, c_uint},
    path::Path,
    ptr::NonNull,
};

const BACKEND_ABI_VERSION: u32 = 2;
const ERROR_MESSAGE_SIZE: usize = 512;
const GPU_NAME_SIZE: usize = 256;
const RGB_CHANNEL_COUNT: usize = 3;
const CPU_THREAD_COUNT_MIN: i32 = 1;
const CPU_THREAD_COUNT_MAX: i32 = 16;
const STATUS_OK: i32 = 0;
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
