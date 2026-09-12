mod scheduler;

pub use crate::cadence::{CadenceDiagnostics, ContentPreset};
use crate::{
    backend::{Backend, CudaBackend, InferenceBackend, InferenceEngine},
    cadence::{
        FrameDifference, is_confident_duplicate, is_smoothable_cadence_run,
        measure_frame_difference_rgb24,
    },
    logging::JobLog,
};
use scheduler::OutputScheduler;
use serde::Deserialize;
use std::{
    fs,
    io::Read,
    path::{Path, PathBuf},
    process::{Child, ChildStdout, Command, ExitStatus, Stdio},
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
        mpsc::SyncSender,
    },
    thread::{self, JoinHandle},
    time::{Duration, Instant},
};

const RGB_CHANNEL_COUNT: usize = 3;
const FRAME_SIZE_BYTES_MAX: usize = 128 * 1024 * 1024;
const FRAME_BUFFER_COUNT: u64 = 3;
const FRAME_DIMENSION_MAX: u32 = 16_384;
const MEMORY_CONSTRAINED_AVAILABLE_BYTES: u64 = 6 * 1024 * 1024 * 1024;
const MEMORY_SAFETY_RESERVE_BYTES: u64 = 2 * 1024 * 1024 * 1024;
const CGROUP_MEMORY_MAX_PATH: &str = "/sys/fs/cgroup/memory.max";
const CGROUP_MEMORY_CURRENT_PATH: &str = "/sys/fs/cgroup/memory.current";
const PROC_MEMORY_INFO_PATH: &str = "/proc/meminfo";
const OUTPUT_FRAME_COUNT_MAX: u64 = 100_000_000;
const TARGET_FPS_MAX: u32 = 480;
const ENCODER_QUALITY_MAX: u8 = 51;
const ENCODER_THREAD_COUNT_MAX: u8 = 16;
const FFMPEG_THREAD_COUNT: &str = "2";
const ENCODER_PROBE_WIDTH: &str = "256";
const ENCODER_PROBE_HEIGHT: &str = "256";
const ENCODER_PROBE_FRAME_COUNT: &str = "1";
const ENCODER_PROBE_TIMEOUT: Duration = Duration::from_secs(5);
const ENCODER_PROBE_POLL_INTERVAL: Duration = Duration::from_millis(20);
const PROGRESS_INTERVAL: Duration = Duration::from_millis(250);
const PROBE_DIAGNOSTIC_SIZE_BYTES_MAX: usize = 64 * 1024;
const PROBE_DIAGNOSTIC_LINE_COUNT_MAX: usize = 200;
const PROBE_OUTPUT_SIZE_BYTES_MAX: usize = 1024 * 1024;
const PROCESS_OUTPUT_READ_SIZE_BYTES: usize = 4096;
const PROCESS_OUTPUT_READ_COUNT_MAX: usize = 1_048_576;
const SCENE_THRESHOLD_DEFAULT: f64 = 0.15;
const MODEL_DIRECTORY_ENVIRONMENT: &str = "INTERPOLATE_MODEL_DIRECTORY";
const MODEL_DIRECTORY_RELATIVE: &str = "share/interpolate/models/rife-v4.25";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VideoEncoder {
    Automatic,
    SoftwareH264,
    NvidiaH264,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EncoderPreset {
    X264VeryFast,
    X264Faster,
    X264Fast,
    X264Medium,
    X264Slow,
    X264Slower,
    X264VerySlow,
    X264Placebo,
    NvidiaP1,
    NvidiaP2,
    NvidiaP3,
    NvidiaP4,
    NvidiaP5,
    NvidiaP6,
    NvidiaP7,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum H264Profile {
    Auto,
    Main,
    High,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct PerformanceDiagnostics {
    pub inference_fps: f64,
    pub decode_fps: f64,
    pub encode_fps: f64,
}

#[derive(Clone)]
pub struct JobConfiguration {
    pub input_path: PathBuf,
    pub output_path: PathBuf,
    pub target_fps_num: u32,
    pub target_fps_den: u32,
    pub gpu_index: i32,
    pub content_preset: ContentPreset,
    pub scene_detection: bool,
    pub use_uhd_mode: bool,
    pub use_nvdec: bool,
    pub inference_backend: InferenceBackend,
    pub video_encoder: VideoEncoder,
    pub encoder_preset: EncoderPreset,
    pub quality_level: u8,
    pub h264_profile: H264Profile,
    pub encoder_thread_count: u8,
    pub preserve_audio: bool,
    pub preserve_subtitles: bool,
    pub preserve_metadata: bool,
}

#[derive(Clone, Debug)]
pub enum JobUpdate {
    Phase(&'static str),
    Progress {
        frame_count: u64,
        frame_count_estimate: u64,
        processing_fps: f64,
        progress: f32,
        cadence_diagnostics: CadenceDiagnostics,
        performance: PerformanceDiagnostics,
    },
    Completed {
        path: PathBuf,
        cadence_diagnostics: CadenceDiagnostics,
    },
    Cancelled {
        partial_path: Option<PathBuf>,
    },
    Failed(String),
}

#[derive(Clone, Debug)]
pub struct VideoMetadata {
    pub width: u32,
    pub height: u32,
    pub source_fps_num: u64,
    pub source_fps_den: u64,
    pub duration_seconds: f64,
    pub pixel_format: String,
}

#[derive(Deserialize)]
struct ProbeOutput {
    streams: Vec<ProbeStream>,
    format: Option<ProbeFormat>,
}

#[derive(Deserialize)]
struct ProbeStream {
    width: Option<u32>,
    height: Option<u32>,
    avg_frame_rate: Option<String>,
    pix_fmt: Option<String>,
    duration: Option<String>,
}

#[derive(Deserialize)]
struct ProbeFormat {
    duration: Option<String>,
}

pub fn default_output_path(
    input_path: &Path,
    target_fps_num: u32,
    scene_detection: bool,
) -> PathBuf {
    assert!(target_fps_num > 0, "target FPS must be positive");
    assert!(
        target_fps_num <= TARGET_FPS_MAX,
        "target FPS must remain bounded"
    );
    let parent = input_path.parent().unwrap_or_else(|| Path::new("."));
    let stem = input_path
        .file_stem()
        .and_then(|value| value.to_str())
        .filter(|value| !value.is_empty())
        .unwrap_or("video");
    let scene_suffix = if scene_detection { "__sc" } else { "" };
    let filename = format!("{stem}__rife-4.25__{target_fps_num}fps{scene_suffix}.mkv");
    let output = parent.join(filename);
    assert!(
        output.file_name().is_some(),
        "default output must have a filename"
    );
    assert_eq!(
        output.extension().and_then(|value| value.to_str()),
        Some("mkv"),
        "default container must be MKV"
    );
    output
}

pub fn probe_video(input_path: &Path) -> Result<VideoMetadata, String> {
    assert!(
        !input_path.as_os_str().is_empty(),
        "input path must not be empty"
    );
    assert!(FRAME_DIMENSION_MAX > 0, "dimension limit must be positive");
    probe_video_logged(input_path, None)
}

pub fn available_video_encoders() -> Vec<VideoEncoder> {
    assert!(
        !ENCODER_PROBE_WIDTH.is_empty(),
        "encoder probe width must be set"
    );
    assert!(
        !ENCODER_PROBE_HEIGHT.is_empty(),
        "encoder probe height must be set"
    );
    assert!(
        ENCODER_PROBE_TIMEOUT > Duration::ZERO,
        "encoder probe timeout must be positive"
    );
    assert!(
        ENCODER_PROBE_POLL_INTERVAL > Duration::ZERO,
        "encoder probe poll interval must be positive"
    );
    let mut encoders = Vec::with_capacity(3);
    let software_available = encoder_is_available("libx264");
    let nvidia_available = encoder_is_available("h264_nvenc");
    if software_available || nvidia_available {
        encoders.push(VideoEncoder::Automatic);
    }
    if software_available {
        encoders.push(VideoEncoder::SoftwareH264);
    }
    if nvidia_available {
        encoders.push(VideoEncoder::NvidiaH264);
    }
    assert!(encoders.len() <= 3, "encoder list must remain bounded");
    encoders
}

fn encoder_is_available(encoder_name: &str) -> bool {
    assert!(!encoder_name.is_empty(), "encoder name must not be empty");
    assert!(encoder_name.len() < 64, "encoder name must remain bounded");
    let available = encoder_is_available_for_gpu(encoder_name, None);
    assert!(
        !encoder_name.is_empty(),
        "encoder name must remain non-empty"
    );
    assert!(encoder_name.len() < 64, "encoder name must remain bounded");
    available
}

fn encoder_is_available_for_gpu(encoder_name: &str, gpu_index: Option<i32>) -> bool {
    assert!(!encoder_name.is_empty(), "encoder name must not be empty");
    assert!(encoder_name.len() < 64, "encoder name must remain bounded");
    assert!(
        gpu_index.is_none_or(|index| index >= 0),
        "GPU index must be non-negative"
    );
    let mut command = Command::new("ffmpeg");
    command
        .args([
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
        ])
        .arg(format!(
            "color=c=black:s={ENCODER_PROBE_WIDTH}x{ENCODER_PROBE_HEIGHT}:r=1"
        ))
        .args(["-frames:v", ENCODER_PROBE_FRAME_COUNT, "-c:v"])
        .arg(encoder_name);
    if let Some(gpu_index) = gpu_index {
        command.args(["-gpu"]).arg(gpu_index.to_string());
    }
    let mut child = match command
        .args(["-f", "null", "-"])
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
    {
        Ok(child) => child,
        Err(_) => return false,
    };
    let deadline = Instant::now() + ENCODER_PROBE_TIMEOUT;
    let mut available = false;
    for _ in 0..=ENCODER_PROBE_TIMEOUT.as_millis() / ENCODER_PROBE_POLL_INTERVAL.as_millis() {
        match child.try_wait() {
            Ok(Some(status)) => {
                available = status.success();
                break;
            }
            Ok(None) => thread::sleep(ENCODER_PROBE_POLL_INTERVAL),
            Err(_) => {
                let _ = terminate_child(&mut child);
                break;
            }
        }
        if Instant::now() >= deadline {
            let _ = terminate_child(&mut child);
            break;
        }
    }
    assert!(
        !encoder_name.is_empty(),
        "encoder name must remain non-empty"
    );
    assert!(encoder_name.len() < 64, "encoder name must remain bounded");
    available
}

fn probe_video_logged(input_path: &Path, log: Option<&JobLog>) -> Result<VideoMetadata, String> {
    assert!(
        !input_path.as_os_str().is_empty(),
        "input path must not be empty"
    );
    assert!(FRAME_DIMENSION_MAX > 0, "dimension limit must be positive");
    if !input_path.is_file() {
        return Err("input video does not exist or is not a regular file".to_owned());
    }

    let output = run_ffprobe(input_path)?;
    if let Some(log) = log
        && !output.stderr.is_empty()
    {
        let diagnostic_size = output.stderr.len().min(PROBE_DIAGNOSTIC_SIZE_BYTES_MAX);
        let diagnostic = String::from_utf8_lossy(&output.stderr[..diagnostic_size]);
        for line in diagnostic.lines().take(PROBE_DIAGNOSTIC_LINE_COUNT_MAX) {
            let _ = log.write("ffprobe", line);
        }
    }
    if !output.status.success() {
        return Err(if output.stderr_exceeded {
            "ffprobe could not inspect the selected video; diagnostics exceeded the capture limit"
                .to_owned()
        } else {
            "ffprobe could not inspect the selected video".to_owned()
        });
    }
    if output.stdout_exceeded {
        return Err("ffprobe returned unexpectedly large metadata".to_owned());
    }
    let probe: ProbeOutput = serde_json::from_slice(&output.stdout)
        .map_err(|error| format!("failed to parse ffprobe metadata: {error}"))?;
    let stream = probe
        .streams
        .first()
        .ok_or_else(|| "input does not contain a video stream".to_owned())?;
    let width = stream
        .width
        .ok_or_else(|| "video width is unavailable".to_owned())?;
    let height = stream
        .height
        .ok_or_else(|| "video height is unavailable".to_owned())?;
    if width == 0 || height == 0 || width > FRAME_DIMENSION_MAX || height > FRAME_DIMENSION_MAX {
        return Err(format!(
            "video dimensions must be between 1 and {FRAME_DIMENSION_MAX}"
        ));
    }
    checked_frame_size(width, height)?;

    let (source_fps_num, source_fps_den) = parse_rational(
        stream
            .avg_frame_rate
            .as_deref()
            .ok_or_else(|| "video frame rate is unavailable".to_owned())?,
    )?;
    let duration_text = stream
        .duration
        .as_deref()
        .or_else(|| {
            probe
                .format
                .as_ref()
                .and_then(|format| format.duration.as_deref())
        })
        .ok_or_else(|| "video duration is unavailable".to_owned())?;
    let duration_seconds = duration_text
        .parse::<f64>()
        .map_err(|_| "video duration is invalid".to_owned())?;
    if !duration_seconds.is_finite() || duration_seconds <= 0.0 {
        return Err("video duration must be finite and positive".to_owned());
    }

    let metadata = VideoMetadata {
        width,
        height,
        source_fps_num,
        source_fps_den,
        duration_seconds,
        pixel_format: stream
            .pix_fmt
            .clone()
            .unwrap_or_else(|| "unknown".to_owned()),
    };
    assert!(
        metadata.width > 0 && metadata.height > 0,
        "validated dimensions must be positive"
    );
    assert!(
        metadata.source_fps_num > 0 && metadata.source_fps_den > 0,
        "validated FPS must be positive"
    );
    Ok(metadata)
}

struct ProbeProcessOutput {
    status: ExitStatus,
    stdout: Vec<u8>,
    stdout_exceeded: bool,
    stderr: Vec<u8>,
    stderr_exceeded: bool,
}

fn run_ffprobe(input_path: &Path) -> Result<ProbeProcessOutput, String> {
    assert!(
        !input_path.as_os_str().is_empty(),
        "ffprobe input must not be empty"
    );
    assert!(
        PROBE_OUTPUT_SIZE_BYTES_MAX > 0,
        "probe output limit must be positive"
    );
    let mut child = Command::new("ffprobe")
        .args([
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,avg_frame_rate,pix_fmt,duration:format=duration",
            "-of",
            "json",
        ])
        .arg(input_path)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|error| format!("failed to start ffprobe: {error}"))?;
    let stdout = match child.stdout.take() {
        Some(stdout) => stdout,
        None => {
            let cleanup_error = terminate_child(&mut child).err();
            return Err(match cleanup_error {
                Some(error) => {
                    format!("ffprobe metadata pipe is unavailable; cleanup failed: {error}")
                }
                None => "ffprobe metadata pipe is unavailable".to_owned(),
            });
        }
    };
    let stderr = match child.stderr.take() {
        Some(stderr) => stderr,
        None => {
            drop(stdout);
            let cleanup_error = terminate_child(&mut child).err();
            return Err(match cleanup_error {
                Some(error) => {
                    format!("ffprobe diagnostics pipe is unavailable; cleanup failed: {error}")
                }
                None => "ffprobe diagnostics pipe is unavailable".to_owned(),
            });
        }
    };
    let stderr_worker = match thread::Builder::new()
        .name("ffprobe-log-reader".to_owned())
        .spawn(move || drain_output_bounded(stderr, PROBE_DIAGNOSTIC_SIZE_BYTES_MAX))
    {
        Ok(stderr_worker) => stderr_worker,
        Err(error) => {
            drop(stdout);
            let cleanup_error = terminate_child(&mut child).err();
            return Err(match cleanup_error {
                Some(cleanup_error) => format!(
                    "failed to start ffprobe diagnostics reader: {error}; cleanup failed: {cleanup_error}"
                ),
                None => format!("failed to start ffprobe diagnostics reader: {error}"),
            });
        }
    };
    let stdout_result = drain_output_bounded(stdout, PROBE_OUTPUT_SIZE_BYTES_MAX);
    let status_result = child
        .wait()
        .map_err(|error| format!("failed to wait for ffprobe: {error}"));
    let stderr_result = stderr_worker
        .join()
        .map_err(|_| "ffprobe diagnostics reader panicked".to_owned())?;
    let (stdout, stdout_exceeded) = stdout_result?;
    let (stderr, stderr_exceeded) = stderr_result?;
    let status = status_result?;
    assert!(
        stdout.len() <= PROBE_OUTPUT_SIZE_BYTES_MAX,
        "probe output must remain bounded"
    );
    assert!(
        stderr.len() <= PROBE_DIAGNOSTIC_SIZE_BYTES_MAX,
        "probe diagnostics must remain bounded"
    );
    Ok(ProbeProcessOutput {
        status,
        stdout,
        stdout_exceeded,
        stderr,
        stderr_exceeded,
    })
}

fn drain_output_bounded<R: Read>(
    mut reader: R,
    size_bytes_max: usize,
) -> Result<(Vec<u8>, bool), String> {
    assert!(size_bytes_max > 0, "process output limit must be positive");
    assert!(
        PROCESS_OUTPUT_READ_SIZE_BYTES > 0,
        "process read size must be positive"
    );
    let mut output = Vec::with_capacity(size_bytes_max.min(PROCESS_OUTPUT_READ_SIZE_BYTES));
    let mut exceeded = false;
    let mut buffer = [0_u8; PROCESS_OUTPUT_READ_SIZE_BYTES];
    for _ in 0..PROCESS_OUTPUT_READ_COUNT_MAX {
        let count = reader
            .read(&mut buffer)
            .map_err(|error| format!("failed to read process output: {error}"))?;
        if count == 0 {
            assert!(
                output.len() <= size_bytes_max,
                "captured output must remain bounded"
            );
            assert!(
                !exceeded || output.len() == size_bytes_max,
                "truncated output must fill its limit"
            );
            return Ok((output, exceeded));
        }
        let remaining_size = size_bytes_max.saturating_sub(output.len());
        let copy_size = count.min(remaining_size);
        output.extend_from_slice(&buffer[..copy_size]);
        exceeded |= copy_size < count;
    }
    Err("process output exceeded the read-count safety limit".to_owned())
}

pub fn run_job(
    configuration: JobConfiguration,
    cancelled: Arc<AtomicBool>,
    updates: SyncSender<JobUpdate>,
) {
    assert!(TARGET_FPS_MAX > 0, "target FPS limit must be positive");
    assert!(
        OUTPUT_FRAME_COUNT_MAX > 0,
        "output frame limit must be positive"
    );
    let result = match JobLog::create() {
        Ok(log) => {
            let _ = log.write(
                "application",
                &format!(
                    "starting {:?} job: {} -> {}",
                    configuration.content_preset,
                    configuration.input_path.display(),
                    configuration.output_path.display()
                ),
            );
            match run_job_inner(&configuration, &cancelled, &updates, &log) {
                Ok(cadence_diagnostics) => {
                    let _ = log.write("application", "job completed successfully");
                    Ok(cadence_diagnostics)
                }
                Err(error) => {
                    let _ = log.write("application", &format!("job failed: {error}"));
                    let summary = log.recent_summary();
                    Err(format!(
                        "{error}\nDiagnostics: {}\n{summary}",
                        log.path().display()
                    ))
                }
            }
        }
        Err(error) => Err(error),
    };
    let terminal_update = match result {
        Ok(cadence_diagnostics) => JobUpdate::Completed {
            path: configuration.output_path.clone(),
            cadence_diagnostics,
        },
        Err(_) if cancelled.load(Ordering::Acquire) => JobUpdate::Cancelled {
            partial_path: partial_output_path(&configuration.output_path)
                .ok()
                .filter(|path| path.is_file()),
        },
        Err(error) => JobUpdate::Failed(error),
    };
    let _ = updates.send(terminal_update);
    assert!(TARGET_FPS_MAX > 0, "target FPS limit must remain valid");
    assert!(
        FRAME_SIZE_BYTES_MAX > 0,
        "frame size limit must remain valid"
    );
}

fn resolve_model_directory() -> Result<PathBuf, String> {
    assert!(
        !MODEL_DIRECTORY_ENVIRONMENT.is_empty(),
        "model environment variable must be named"
    );
    assert!(
        !MODEL_DIRECTORY_RELATIVE.is_empty(),
        "installed model path must be configured"
    );

    if let Some(value) = std::env::var_os(MODEL_DIRECTORY_ENVIRONMENT) {
        if value.is_empty() {
            return Err(format!("{MODEL_DIRECTORY_ENVIRONMENT} must not be empty"));
        }
        let model_directory = PathBuf::from(value);
        if !model_files_exist(&model_directory) {
            return Err(format!(
                "{MODEL_DIRECTORY_ENVIRONMENT} does not contain the RIFE 4.25 model"
            ));
        }
        assert!(
            model_directory.is_dir(),
            "validated model path must be a directory"
        );
        assert!(
            model_files_exist(&model_directory),
            "validated model files must exist"
        );
        return Ok(model_directory);
    }

    let executable = std::env::current_exe()
        .map_err(|error| format!("failed to locate the application executable: {error}"))?;
    for model_directory in packaged_model_directories(&executable) {
        if model_files_exist(&model_directory) {
            assert!(
                model_directory.is_dir(),
                "packaged model path must be a directory"
            );
            assert!(
                model_files_exist(&model_directory),
                "packaged model files must exist"
            );
            return Ok(model_directory);
        }
    }

    let development_directory = Path::new(env!("CARGO_MANIFEST_DIR")).join("models/rife-v4.25");
    if !model_files_exist(&development_directory) {
        return Err("RIFE 4.25 model files could not be located".to_owned());
    }
    assert!(
        development_directory.is_dir(),
        "development model path must be a directory"
    );
    assert!(
        model_files_exist(&development_directory),
        "development model files must exist"
    );
    Ok(development_directory)
}

fn packaged_model_directories(executable: &Path) -> Vec<PathBuf> {
    assert!(
        !executable.as_os_str().is_empty(),
        "executable path must not be empty"
    );
    assert!(
        !MODEL_DIRECTORY_RELATIVE.is_empty(),
        "installed model path must be configured"
    );
    let mut directories = Vec::new();
    if let Some(executable_directory) = executable.parent() {
        directories.push(executable_directory.join("models/rife-v4.25"));
        directories.push(executable_directory.join(MODEL_DIRECTORY_RELATIVE));
        if let Some(installation_root) = executable_directory.parent() {
            directories.push(installation_root.join(MODEL_DIRECTORY_RELATIVE));
        }
    }
    assert!(
        directories.len() <= 3,
        "packaged model search must remain bounded"
    );
    directories
}

fn model_files_exist(model_directory: &Path) -> bool {
    assert!(
        !model_directory.as_os_str().is_empty(),
        "model directory must not be empty"
    );
    assert!(
        !MODEL_DIRECTORY_RELATIVE.is_empty(),
        "model relative path must remain configured"
    );
    let files_exist = model_directory.is_dir()
        && model_directory.join("flownet.param").is_file()
        && model_directory.join("flownet.bin").is_file();
    assert!(
        !files_exist || model_directory.is_dir(),
        "model files require a containing directory"
    );
    assert!(
        !files_exist || model_directory.join("flownet.param").is_file(),
        "a valid model requires its parameter file"
    );
    files_exist
}

fn run_job_inner(
    configuration: &JobConfiguration,
    cancelled: &AtomicBool,
    updates: &SyncSender<JobUpdate>,
    log: &JobLog,
) -> Result<CadenceDiagnostics, String> {
    assert!(TARGET_FPS_MAX > 0, "target FPS limit must be positive");
    assert!(
        OUTPUT_FRAME_COUNT_MAX > 0,
        "output frame limit must be positive"
    );
    validate_configuration(configuration)?;
    send_update(updates, JobUpdate::Phase("Probing video"));
    let metadata = probe_video_logged(&configuration.input_path, Some(log))?;
    validate_target_fps(configuration, &metadata)?;
    if cancelled.load(Ordering::Acquire) {
        return Err("job cancelled".to_owned());
    }
    let (effective_configuration, memory_constrained) =
        adapt_configuration_for_memory(configuration, &metadata)?;
    if memory_constrained {
        send_update(
            updates,
            JobUpdate::Phase("Using low-memory processing mode"),
        );
    }
    let configuration = effective_configuration;

    send_update(updates, JobUpdate::Phase("Initializing inference backend"));
    let mut backend = match configuration.inference_backend {
        InferenceBackend::VulkanNcnn => {
            let model_directory = resolve_model_directory()?;
            InferenceEngine::Vulkan(Backend::create(
                &model_directory,
                configuration.gpu_index,
                configuration.use_uhd_mode,
            )?)
        }
        InferenceBackend::CudaPytorchVapourSynth => InferenceEngine::Cuda(CudaBackend::create(
            metadata.width,
            metadata.height,
            configuration.gpu_index,
            configuration.use_uhd_mode,
        )?),
    };

    let partial_path = partial_output_path(&configuration.output_path)?;
    if partial_path.exists() {
        fs::remove_file(&partial_path)
            .map_err(|error| format!("failed to remove stale partial output: {error}"))?;
    }

    send_update(updates, JobUpdate::Phase("Starting media pipeline"));
    let mut decoder = spawn_decoder(&configuration, &metadata, log)?;
    let mut encoder_configuration = configuration.clone();
    let encoder_result = match spawn_encoder(&configuration, &metadata, &partial_path, log) {
        Ok(encoder) => Ok(encoder),
        Err(error)
            if matches!(
                configuration.video_encoder,
                VideoEncoder::Automatic | VideoEncoder::NvidiaH264
            ) =>
        {
            send_update(
                updates,
                JobUpdate::Phase("NVENC unavailable; falling back to CPU H.264"),
            );
            encoder_configuration.video_encoder = VideoEncoder::SoftwareH264;
            spawn_encoder(&encoder_configuration, &metadata, &partial_path, log).map_err(
                |fallback_error| {
                    format!("NVENC failed: {error}; CPU H.264 fallback failed: {fallback_error}")
                },
            )
        }
        Err(error) => Err(error),
    };
    let mut encoder = match encoder_result {
        Ok(encoder) => encoder,
        Err(error) => {
            let terminate_result = terminate_child(&mut decoder.child);
            let log_result = finish_log_worker(&mut decoder);
            let mut combined_error = error;
            if let Err(terminate_error) = terminate_result {
                combined_error.push_str(&format!("; decoder cleanup failed: {terminate_error}"));
            }
            if let Err(log_error) = log_result {
                combined_error.push_str(&format!(
                    "; decoder diagnostics failed during cleanup: {log_error}"
                ));
            }
            return Err(combined_error);
        }
    };

    let cadence_diagnostics = match process_frames(
        &configuration,
        &metadata,
        cancelled,
        updates,
        &mut backend,
        &mut decoder.child,
        &mut encoder.child,
    ) {
        Ok(cadence_diagnostics) => cadence_diagnostics,
        Err(error) => {
            let decoder_terminate_result = terminate_child(&mut decoder.child);
            let encoder_terminate_result = terminate_child(&mut encoder.child);
            let decoder_log_result = finish_log_worker(&mut decoder);
            let encoder_log_result = finish_log_worker(&mut encoder);
            let was_cancelled = cancelled.load(Ordering::Acquire);
            let mut combined_error = error;
            if partial_path.exists() {
                combined_error.push_str(&format!(
                    "; recoverable partial output remains at {}",
                    partial_path.display()
                ));
            }
            if was_cancelled {
                combined_error.push_str("; encoded frames were preserved");
            }
            if let Err(terminate_error) = decoder_terminate_result {
                combined_error.push_str(&format!("; decoder cleanup failed: {terminate_error}"));
            }
            if let Err(terminate_error) = encoder_terminate_result {
                combined_error.push_str(&format!("; encoder cleanup failed: {terminate_error}"));
            }
            if let Err(log_error) = decoder_log_result {
                combined_error.push_str(&format!("; decoder diagnostics failed: {log_error}"));
            }
            if let Err(log_error) = encoder_log_result {
                combined_error.push_str(&format!("; encoder diagnostics failed: {log_error}"));
            }
            return Err(combined_error);
        }
    };

    send_update(updates, JobUpdate::Phase("Finalizing output"));
    let decoder_result = wait_media_child(&mut decoder, "decoder");
    let encoder_result = wait_media_child(&mut encoder, "encoder");
    let decoder_status = decoder_result?;
    let encoder_status = encoder_result?;
    if !decoder_status.success() {
        return Err(format!(
            "FFmpeg decoder failed; recoverable partial output, if any, remains at {}",
            partial_path.display()
        ));
    }
    if !encoder_status.success() {
        return Err(format!(
            "FFmpeg encoder failed; an input stream may not be compatible with MKV. Recoverable partial output, if any, remains at {}",
            partial_path.display()
        ));
    }
    fs::rename(&partial_path, &configuration.output_path).map_err(|error| {
        format!(
            "failed to publish completed output: {error}. Completed partial output remains at {}",
            partial_path.display()
        )
    })?;
    assert!(
        configuration.output_path.is_file(),
        "completed output must exist"
    );
    assert!(
        !partial_path.exists(),
        "partial output must be gone after rename"
    );
    Ok(cadence_diagnostics)
}

fn validate_configuration(configuration: &JobConfiguration) -> Result<(), String> {
    assert!(TARGET_FPS_MAX > 0, "target FPS limit must be positive");
    assert!(
        OUTPUT_FRAME_COUNT_MAX > 0,
        "output frame limit must be positive"
    );
    if !configuration.input_path.is_file() {
        return Err("select an existing input video".to_owned());
    }
    if configuration.output_path.as_os_str().is_empty() {
        return Err("select an output filename".to_owned());
    }
    if configuration.input_path == configuration.output_path {
        return Err("output path must differ from input path".to_owned());
    }
    if configuration.output_path.exists() {
        return Err("output already exists; choose another filename".to_owned());
    }
    if configuration.target_fps_num == 0
        || configuration.target_fps_den == 0
        || configuration.target_fps_num > TARGET_FPS_MAX
    {
        return Err(format!("target FPS must be between 1 and {TARGET_FPS_MAX}"));
    }
    if configuration.quality_level > ENCODER_QUALITY_MAX {
        return Err(format!(
            "quality must be between 0 and {ENCODER_QUALITY_MAX}"
        ));
    }
    if configuration.encoder_thread_count > ENCODER_THREAD_COUNT_MAX {
        return Err(format!(
            "encoder threads must be between 0 and {ENCODER_THREAD_COUNT_MAX}"
        ));
    }
    assert!(matches!(
        configuration.inference_backend,
        InferenceBackend::VulkanNcnn | InferenceBackend::CudaPytorchVapourSynth
    ));
    if matches!(
        configuration.inference_backend,
        InferenceBackend::CudaPytorchVapourSynth
    ) && configuration.gpu_index < 0
    {
        return Err("CUDA inference GPU index must be non-negative".to_owned());
    }
    assert_ne!(
        configuration.input_path, configuration.output_path,
        "validated paths must differ"
    );
    assert!(
        configuration.target_fps_den > 0,
        "validated denominator must be positive"
    );
    Ok(())
}

fn validate_target_fps(
    configuration: &JobConfiguration,
    metadata: &VideoMetadata,
) -> Result<(), String> {
    assert!(
        metadata.source_fps_num > 0,
        "source FPS numerator must be positive"
    );
    assert!(
        metadata.source_fps_den > 0,
        "source FPS denominator must be positive"
    );
    let target_scaled =
        u128::from(configuration.target_fps_num) * u128::from(metadata.source_fps_den);
    let source_scaled =
        u128::from(metadata.source_fps_num) * u128::from(configuration.target_fps_den);
    if target_scaled <= source_scaled {
        return Err("target FPS must be higher than source FPS".to_owned());
    }
    assert!(
        target_scaled > source_scaled,
        "validated target FPS must exceed source FPS"
    );
    assert!(
        target_scaled > 0 && source_scaled > 0,
        "scaled frame rates must be positive"
    );
    Ok(())
}

struct MediaChild {
    child: Child,
    log_worker: Option<JoinHandle<Result<(), String>>>,
}

fn spawn_decoder(
    configuration: &JobConfiguration,
    metadata: &VideoMetadata,
    log: &JobLog,
) -> Result<MediaChild, String> {
    assert!(metadata.width > 0, "decoder width must be positive");
    assert!(metadata.height > 0, "decoder height must be positive");
    let mut command = Command::new("ffmpeg");
    command.args(["-nostdin", "-v", "error", "-threads", FFMPEG_THREAD_COUNT]);
    if configuration.use_nvdec {
        command.args(["-hwaccel", "cuda", "-hwaccel_output_format", "cuda"]);
    }
    command.arg("-i").arg(&configuration.input_path).args([
        "-map",
        "0:v:0",
        "-an",
        "-sn",
        "-dn",
        "-fps_mode",
        "passthrough",
    ]);
    if configuration.use_nvdec {
        command.args(["-vf", "hwdownload,format=rgb24"]);
    } else {
        command.args(["-pix_fmt", "rgb24"]);
    }
    let mut child = command
        .args(["-f", "rawvideo", "pipe:1"])
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|error| format!("failed to start FFmpeg decoder: {error}"))?;
    let decoder_error = match child.stderr.take() {
        Some(decoder_error) => decoder_error,
        None => {
            let cleanup_error = terminate_child(&mut child).err();
            return Err(match cleanup_error {
                Some(error) => {
                    format!("decoder diagnostics pipe is unavailable; cleanup failed: {error}")
                }
                None => "decoder diagnostics pipe is unavailable".to_owned(),
            });
        }
    };
    let log_worker = match log.spawn_reader("decoder", decoder_error) {
        Ok(log_worker) => log_worker,
        Err(error) => {
            return match terminate_child(&mut child) {
                Ok(()) => Err(error),
                Err(cleanup_error) => {
                    Err(format!("{error}; decoder cleanup failed: {cleanup_error}"))
                }
            };
        }
    };
    assert!(child.stdout.is_some(), "decoder stdout must be piped");
    assert!(child.stdin.is_none(), "decoder stdin must be closed");
    Ok(MediaChild {
        child,
        log_worker: Some(log_worker),
    })
}

fn resolve_video_encoder(configuration: &JobConfiguration) -> VideoEncoder {
    assert!(
        configuration.gpu_index >= 0,
        "encoder GPU index must be non-negative"
    );
    assert!(matches!(
        configuration.video_encoder,
        VideoEncoder::Automatic | VideoEncoder::SoftwareH264 | VideoEncoder::NvidiaH264
    ));
    let resolved_encoder = match configuration.video_encoder {
        VideoEncoder::Automatic
            if encoder_is_available_for_gpu("h264_nvenc", Some(configuration.gpu_index)) =>
        {
            VideoEncoder::NvidiaH264
        }
        VideoEncoder::Automatic | VideoEncoder::SoftwareH264 => VideoEncoder::SoftwareH264,
        VideoEncoder::NvidiaH264 => VideoEncoder::NvidiaH264,
    };
    assert!(!matches!(resolved_encoder, VideoEncoder::Automatic));
    assert!(matches!(
        resolved_encoder,
        VideoEncoder::SoftwareH264 | VideoEncoder::NvidiaH264
    ));
    resolved_encoder
}

fn encoder_preset_argument(encoder: VideoEncoder, preset: EncoderPreset) -> &'static str {
    assert!(matches!(
        encoder,
        VideoEncoder::SoftwareH264 | VideoEncoder::NvidiaH264
    ));
    let argument = match encoder {
        VideoEncoder::SoftwareH264 => match preset {
            EncoderPreset::X264VeryFast => "veryfast",
            EncoderPreset::X264Faster => "faster",
            EncoderPreset::X264Fast => "fast",
            EncoderPreset::X264Medium => "medium",
            EncoderPreset::X264Slow => "slow",
            EncoderPreset::X264Slower => "slower",
            EncoderPreset::X264VerySlow => "veryslow",
            EncoderPreset::X264Placebo => "placebo",
            _ => "medium",
        },
        VideoEncoder::NvidiaH264 => match preset {
            EncoderPreset::NvidiaP1 => "p1",
            EncoderPreset::NvidiaP2 => "p2",
            EncoderPreset::NvidiaP3 => "p3",
            EncoderPreset::NvidiaP4 => "p4",
            EncoderPreset::NvidiaP5 => "p5",
            EncoderPreset::NvidiaP6 => "p6",
            EncoderPreset::NvidiaP7 => "p7",
            _ => "p5",
        },
        VideoEncoder::Automatic => unreachable!("automatic encoder must be resolved"),
    };
    assert!(
        !argument.is_empty(),
        "encoder preset argument must not be empty"
    );
    assert!(
        argument.len() <= 8,
        "encoder preset argument must remain bounded"
    );
    argument
}

fn spawn_encoder(
    configuration: &JobConfiguration,
    metadata: &VideoMetadata,
    partial_path: &Path,
    log: &JobLog,
) -> Result<MediaChild, String> {
    assert!(metadata.width > 0, "encoder width must be positive");
    assert!(
        configuration.target_fps_den > 0,
        "encoder FPS denominator must be positive"
    );
    let video_size = format!("{}x{}", metadata.width, metadata.height);
    let frame_rate = format!(
        "{}/{}",
        configuration.target_fps_num, configuration.target_fps_den
    );
    assert!(
        configuration.quality_level <= ENCODER_QUALITY_MAX,
        "quality must remain bounded"
    );
    assert!(
        configuration.encoder_thread_count <= ENCODER_THREAD_COUNT_MAX,
        "encoder threads must remain bounded"
    );
    let effective_video_encoder = resolve_video_encoder(configuration);
    if effective_video_encoder == VideoEncoder::NvidiaH264
        && !encoder_is_available_for_gpu("h264_nvenc", Some(configuration.gpu_index))
    {
        return Err(format!(
            "NVIDIA NVENC is unavailable on GPU {}",
            configuration.gpu_index
        ));
    }
    let encoder_preset =
        encoder_preset_argument(effective_video_encoder, configuration.encoder_preset);
    let quality_argument = configuration.quality_level.to_string();
    let mut encoder_arguments = vec![
        match effective_video_encoder {
            VideoEncoder::SoftwareH264 => "libx264",
            VideoEncoder::NvidiaH264 => "h264_nvenc",
            VideoEncoder::Automatic => unreachable!("automatic encoder must be resolved"),
        }
        .to_owned(),
        "-preset".to_owned(),
        encoder_preset.to_owned(),
    ];
    match effective_video_encoder {
        VideoEncoder::SoftwareH264 => {
            encoder_arguments.extend(["-crf".to_owned(), quality_argument])
        }
        VideoEncoder::NvidiaH264 => encoder_arguments.extend([
            "-rc".to_owned(),
            "vbr".to_owned(),
            "-cq".to_owned(),
            quality_argument,
            "-b:v".to_owned(),
            "0".to_owned(),
            "-gpu".to_owned(),
            configuration.gpu_index.to_string(),
        ]),
        VideoEncoder::Automatic => unreachable!("automatic encoder must be resolved"),
    }
    if configuration.h264_profile != H264Profile::Auto {
        encoder_arguments.extend([
            "-profile:v".to_owned(),
            match configuration.h264_profile {
                H264Profile::Auto => unreachable!("automatic profile is handled above"),
                H264Profile::Main => "main".to_owned(),
                H264Profile::High => "high".to_owned(),
            },
        ]);
    }
    let mut command = Command::new("ffmpeg");
    command
        .args([
            "-nostdin",
            "-v",
            "error",
            "-y",
            "-threads",
            FFMPEG_THREAD_COUNT,
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-video_size",
        ])
        .arg(video_size)
        .args(["-framerate"])
        .arg(frame_rate)
        .args(["-i", "pipe:0", "-i"])
        .arg(&configuration.input_path)
        .args(["-map", "0:v:0"]);
    if configuration.preserve_audio {
        command.args(["-map", "1:a?"]);
    }
    if configuration.preserve_subtitles {
        command.args(["-map", "1:s?"]);
    }
    if configuration.preserve_metadata {
        command.args(["-map_metadata", "1", "-map_chapters", "1"]);
    }
    command.args(["-c:v"]).args(&encoder_arguments);
    command.args(["-pix_fmt", "yuv420p"]);
    if configuration.encoder_thread_count > 0 {
        command
            .args(["-threads"])
            .arg(configuration.encoder_thread_count.to_string());
    }
    if configuration.preserve_audio {
        command.args(["-c:a", "copy"]);
    }
    if configuration.preserve_subtitles {
        command.args(["-c:s", "copy"]);
    }
    let mut child = command
        .arg(partial_path)
        .stdin(Stdio::piped())
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|error| format!("failed to start FFmpeg encoder: {error}"))?;
    let encoder_error = match child.stderr.take() {
        Some(encoder_error) => encoder_error,
        None => {
            let cleanup_error = terminate_child(&mut child).err();
            return Err(match cleanup_error {
                Some(error) => {
                    format!("encoder diagnostics pipe is unavailable; cleanup failed: {error}")
                }
                None => "encoder diagnostics pipe is unavailable".to_owned(),
            });
        }
    };
    let log_worker = match log.spawn_reader("encoder", encoder_error) {
        Ok(log_worker) => log_worker,
        Err(error) => {
            return match terminate_child(&mut child) {
                Ok(()) => Err(error),
                Err(cleanup_error) => {
                    Err(format!("{error}; encoder cleanup failed: {cleanup_error}"))
                }
            };
        }
    };
    assert!(child.stdin.is_some(), "encoder stdin must be piped");
    assert!(child.stdout.is_none(), "encoder stdout must be closed");
    Ok(MediaChild {
        child,
        log_worker: Some(log_worker),
    })
}

fn process_frames(
    configuration: &JobConfiguration,
    metadata: &VideoMetadata,
    cancelled: &AtomicBool,
    updates: &SyncSender<JobUpdate>,
    backend: &mut InferenceEngine,
    decoder: &mut Child,
    encoder: &mut Child,
) -> Result<CadenceDiagnostics, String> {
    assert!(metadata.source_fps_num > 0, "source FPS must be positive");
    assert!(
        configuration.target_fps_num > 0,
        "target FPS must be positive"
    );
    let frame_size = checked_frame_size(metadata.width, metadata.height)?;
    let mut decoder_output = decoder
        .stdout
        .take()
        .ok_or_else(|| "decoder output pipe is unavailable".to_owned())?;
    let mut encoder_input = encoder
        .stdin
        .take()
        .ok_or_else(|| "encoder input pipe is unavailable".to_owned())?;
    let mut frame_anchor = vec![0_u8; frame_size];
    let mut frame_candidate = vec![0_u8; frame_size];
    let anchor_read_started_at = Instant::now();
    let anchor_available = read_frame(&mut decoder_output, &mut frame_anchor)?;
    let anchor_read_elapsed = anchor_read_started_at.elapsed();
    if !anchor_available {
        return Err("decoder produced no video frames".to_owned());
    }

    let mut scheduler = OutputScheduler::new(
        configuration,
        metadata,
        cancelled,
        updates,
        backend,
        &mut encoder_input,
        frame_size,
    );
    scheduler.record_decode(anchor_read_elapsed);
    let anime_mode = configuration.content_preset == ContentPreset::Anime;
    let scene_detection = configuration.scene_detection || anime_mode;
    let mut anchor_index = 0_u64;
    let mut candidate_index = 1_u64;
    let mut cadence_run_frame_count = 1_u64;
    let mut preserving_long_hold = false;

    send_update(
        updates,
        JobUpdate::Phase(if anime_mode {
            "Analyzing anime cadence, interpolating, and encoding"
        } else {
            "Interpolating and encoding"
        }),
    );

    loop {
        if cancelled.load(Ordering::Acquire) {
            return Err("job cancelled".to_owned());
        }
        if candidate_index >= OUTPUT_FRAME_COUNT_MAX {
            return Err(format!(
                "source exceeds the {OUTPUT_FRAME_COUNT_MAX}-frame safety limit"
            ));
        }
        let candidate_read_started_at = Instant::now();
        let has_candidate = read_frame(&mut decoder_output, &mut frame_candidate)?;
        if !has_candidate {
            let source_index_end = candidate_index;
            scheduler.write_span(
                &frame_anchor,
                &frame_anchor,
                anchor_index,
                source_index_end,
                false,
            )?;
            break;
        }
        scheduler.record_decode(candidate_read_started_at.elapsed());

        let difference = if anime_mode || scene_detection {
            measure_frame_difference_rgb24(
                &frame_anchor,
                &frame_candidate,
                metadata.width,
                metadata.height,
            )
        } else {
            FrameDifference::default()
        };
        let confident_duplicate = anime_mode && is_confident_duplicate(difference);

        if preserving_long_hold {
            if confident_duplicate {
                scheduler.cadence_diagnostics.duplicate_frame_count = scheduler
                    .cadence_diagnostics
                    .duplicate_frame_count
                    .saturating_add(1);
                scheduler.write_span(
                    &frame_anchor,
                    &frame_candidate,
                    anchor_index,
                    candidate_index,
                    false,
                )?;
                std::mem::swap(&mut frame_anchor, &mut frame_candidate);
                anchor_index = candidate_index;
                candidate_index = candidate_index
                    .checked_add(1)
                    .ok_or_else(|| "source frame index overflowed".to_owned())?;
                continue;
            }

            let scene_change = scene_detection && difference.mean >= SCENE_THRESHOLD_DEFAULT;
            if scene_change {
                scheduler.cadence_diagnostics.scene_cut_count = scheduler
                    .cadence_diagnostics
                    .scene_cut_count
                    .saturating_add(1);
            }
            scheduler.write_span(
                &frame_anchor,
                &frame_candidate,
                anchor_index,
                candidate_index,
                !scene_change,
            )?;
            std::mem::swap(&mut frame_anchor, &mut frame_candidate);
            anchor_index = candidate_index;
            candidate_index = candidate_index
                .checked_add(1)
                .ok_or_else(|| "source frame index overflowed".to_owned())?;
            cadence_run_frame_count = 1;
            preserving_long_hold = false;
            continue;
        }

        if confident_duplicate {
            scheduler.cadence_diagnostics.duplicate_frame_count = scheduler
                .cadence_diagnostics
                .duplicate_frame_count
                .saturating_add(1);
            cadence_run_frame_count = cadence_run_frame_count
                .checked_add(1)
                .ok_or_else(|| "cadence run length overflowed".to_owned())?;
            if is_smoothable_cadence_run(cadence_run_frame_count) {
                candidate_index = candidate_index
                    .checked_add(1)
                    .ok_or_else(|| "source frame index overflowed".to_owned())?;
                continue;
            }

            scheduler.cadence_diagnostics.long_hold_count = scheduler
                .cadence_diagnostics
                .long_hold_count
                .saturating_add(1);
            scheduler.write_span(
                &frame_anchor,
                &frame_candidate,
                anchor_index,
                candidate_index,
                false,
            )?;
            std::mem::swap(&mut frame_anchor, &mut frame_candidate);
            anchor_index = candidate_index;
            candidate_index = candidate_index
                .checked_add(1)
                .ok_or_else(|| "source frame index overflowed".to_owned())?;
            cadence_run_frame_count = 1;
            preserving_long_hold = true;
            continue;
        }

        let scene_change = scene_detection && difference.mean >= SCENE_THRESHOLD_DEFAULT;
        if scene_change {
            scheduler.cadence_diagnostics.scene_cut_count = scheduler
                .cadence_diagnostics
                .scene_cut_count
                .saturating_add(1);
        }
        if anime_mode && is_smoothable_cadence_run(cadence_run_frame_count) && !scene_change {
            scheduler.cadence_diagnostics.cadence_run_count = scheduler
                .cadence_diagnostics
                .cadence_run_count
                .saturating_add(1);
        }
        scheduler.write_span(
            &frame_anchor,
            &frame_candidate,
            anchor_index,
            candidate_index,
            !scene_change,
        )?;
        std::mem::swap(&mut frame_anchor, &mut frame_candidate);
        anchor_index = candidate_index;
        candidate_index = candidate_index
            .checked_add(1)
            .ok_or_else(|| "source frame index overflowed".to_owned())?;
        cadence_run_frame_count = 1;
    }

    scheduler.finish()?;
    let cadence_diagnostics = scheduler.cadence_diagnostics;
    drop(scheduler);
    drop(encoder_input);
    assert!(
        candidate_index > 0,
        "processing must inspect at least one source position"
    );
    assert!(
        anchor_index < candidate_index,
        "source frame indices must remain ordered"
    );
    Ok(cadence_diagnostics)
}

fn read_frame(reader: &mut ChildStdout, frame: &mut [u8]) -> Result<bool, String> {
    assert!(!frame.is_empty(), "frame buffer must not be empty");
    assert!(
        frame.len() <= FRAME_SIZE_BYTES_MAX,
        "frame buffer must remain bounded"
    );
    let mut offset = 0_usize;
    while offset < frame.len() {
        let count = reader
            .read(&mut frame[offset..])
            .map_err(|error| format!("failed to read decoded frame: {error}"))?;
        if count == 0 {
            if offset == 0 {
                return Ok(false);
            }
            return Err("decoder ended in the middle of a video frame".to_owned());
        }
        offset = offset
            .checked_add(count)
            .ok_or_else(|| "decoded frame offset overflowed".to_owned())?;
    }
    assert_eq!(offset, frame.len(), "reader must fill one complete frame");
    assert!(
        offset <= FRAME_SIZE_BYTES_MAX,
        "read size must stay bounded"
    );
    Ok(true)
}

fn parse_rational(text: &str) -> Result<(u64, u64), String> {
    assert!(!text.is_empty(), "rational text must not be empty");
    assert!(text.len() <= 64, "rational text must remain bounded");
    let (numerator_text, denominator_text) = text
        .split_once('/')
        .ok_or_else(|| "frame rate is not rational".to_owned())?;
    let numerator = numerator_text
        .parse::<u64>()
        .map_err(|_| "frame-rate numerator is invalid".to_owned())?;
    let denominator = denominator_text
        .parse::<u64>()
        .map_err(|_| "frame-rate denominator is invalid".to_owned())?;
    if numerator == 0 || denominator == 0 {
        return Err("frame rate must be positive".to_owned());
    }
    let divisor = greatest_common_divisor(numerator, denominator);
    let result = (numerator / divisor, denominator / divisor);
    assert!(result.0 > 0, "reduced numerator must be positive");
    assert!(result.1 > 0, "reduced denominator must be positive");
    Ok(result)
}

fn greatest_common_divisor(mut left: u64, mut right: u64) -> u64 {
    assert!(left > 0, "left divisor input must be positive");
    assert!(right > 0, "right divisor input must be positive");
    const ITERATION_COUNT_MAX: usize = 128;
    for _ in 0..ITERATION_COUNT_MAX {
        if right == 0 {
            break;
        }
        let remainder = left % right;
        left = right;
        right = remainder;
    }
    assert!(left > 0, "greatest common divisor must be positive");
    assert_eq!(right, 0, "bounded Euclidean algorithm must complete");
    left
}

fn adapt_configuration_for_memory(
    configuration: &JobConfiguration,
    metadata: &VideoMetadata,
) -> Result<(JobConfiguration, bool), String> {
    assert!(metadata.width > 0, "memory policy width must be positive");
    assert!(metadata.height > 0, "memory policy height must be positive");
    let frame_size = checked_frame_size(metadata.width, metadata.height)?;
    let frame_buffer_bytes = u64::try_from(frame_size)
        .ok()
        .and_then(|size| size.checked_mul(FRAME_BUFFER_COUNT))
        .ok_or_else(|| "memory policy frame buffers exceed addressable memory".to_owned())?;
    let available_bytes = available_memory_bytes()?;
    let normal_backend_reserve = match configuration.inference_backend {
        InferenceBackend::VulkanNcnn => MEMORY_SAFETY_RESERVE_BYTES / 2,
        InferenceBackend::CudaPytorchVapourSynth => MEMORY_SAFETY_RESERVE_BYTES,
    };
    let normal_required_bytes = frame_buffer_bytes
        .checked_add(normal_backend_reserve)
        .ok_or_else(|| "normal memory budget overflowed".to_owned())?;
    let memory_constrained = available_bytes < MEMORY_CONSTRAINED_AVAILABLE_BYTES
        || available_bytes < normal_required_bytes;
    if !memory_constrained {
        assert!(available_bytes >= normal_required_bytes);
        return Ok((configuration.clone(), false));
    }

    let constrained_required_bytes = frame_buffer_bytes
        .checked_add(MEMORY_SAFETY_RESERVE_BYTES / 2)
        .ok_or_else(|| "constrained memory budget overflowed".to_owned())?;
    if available_bytes < constrained_required_bytes {
        return Err(format!(
            "not enough available memory for a safe job: {} MiB available, {} MiB required",
            available_bytes / 1024 / 1024,
            constrained_required_bytes / 1024 / 1024
        ));
    }
    let mut constrained_configuration = configuration.clone();
    constrained_configuration.use_uhd_mode = true;
    constrained_configuration.use_nvdec = false;
    constrained_configuration.encoder_thread_count = 1;
    assert!(constrained_configuration.use_uhd_mode);
    assert!(!constrained_configuration.use_nvdec);
    Ok((constrained_configuration, true))
}

fn available_memory_bytes() -> Result<u64, String> {
    assert!(
        !PROC_MEMORY_INFO_PATH.is_empty(),
        "memory info path must be set"
    );
    assert!(
        !CGROUP_MEMORY_MAX_PATH.is_empty(),
        "cgroup max path must be set"
    );
    let memory_info = fs::read_to_string(PROC_MEMORY_INFO_PATH)
        .map_err(|error| format!("failed to read available memory: {error}"))?;
    let available_kib = memory_info
        .lines()
        .find_map(|line| {
            let mut fields = line.split_whitespace();
            (fields.next() == Some("MemAvailable:"))
                .then(|| fields.next()?.parse::<u64>().ok())
                .flatten()
        })
        .ok_or_else(|| "available memory was not reported by the operating system".to_owned())?;
    let host_available_bytes = available_kib
        .checked_mul(1024)
        .ok_or_else(|| "available memory value overflowed".to_owned())?;
    let available_bytes = match cgroup_available_memory_bytes()? {
        Some(cgroup_bytes) => host_available_bytes.min(cgroup_bytes),
        None => host_available_bytes,
    };
    assert!(available_bytes <= host_available_bytes);
    assert!(available_bytes > 0, "available memory must be positive");
    Ok(available_bytes)
}

fn cgroup_available_memory_bytes() -> Result<Option<u64>, String> {
    assert!(
        !CGROUP_MEMORY_MAX_PATH.is_empty(),
        "cgroup max path must be set"
    );
    assert!(
        !CGROUP_MEMORY_CURRENT_PATH.is_empty(),
        "cgroup current path must be set"
    );
    let maximum = match fs::read_to_string(CGROUP_MEMORY_MAX_PATH) {
        Ok(value) => value,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(error) => return Err(format!("failed to read cgroup memory limit: {error}")),
    };
    if maximum.trim() == "max" {
        return Ok(None);
    }
    let maximum_bytes = maximum
        .trim()
        .parse::<u64>()
        .map_err(|error| format!("invalid cgroup memory limit: {error}"))?;
    let current = fs::read_to_string(CGROUP_MEMORY_CURRENT_PATH)
        .map_err(|error| format!("failed to read cgroup memory usage: {error}"))?;
    let current_bytes = current
        .trim()
        .parse::<u64>()
        .map_err(|error| format!("invalid cgroup memory usage: {error}"))?;
    let available_bytes = maximum_bytes
        .checked_sub(current_bytes)
        .ok_or_else(|| "cgroup memory usage exceeds its configured limit".to_owned())?;
    assert!(maximum_bytes >= current_bytes);
    assert!(available_bytes <= maximum_bytes);
    Ok(Some(available_bytes))
}

fn checked_frame_size(width: u32, height: u32) -> Result<usize, String> {
    assert!(width > 0, "frame width must be positive");
    assert!(height > 0, "frame height must be positive");
    let frame_size = usize::try_from(width)
        .ok()
        .and_then(|value| value.checked_mul(usize::try_from(height).ok()?))
        .and_then(|value| value.checked_mul(RGB_CHANNEL_COUNT))
        .ok_or_else(|| "frame size exceeds addressable memory".to_owned())?;
    if frame_size > FRAME_SIZE_BYTES_MAX {
        return Err(format!(
            "decoded frame exceeds the {} MiB safety limit",
            FRAME_SIZE_BYTES_MAX / 1024 / 1024
        ));
    }
    assert!(frame_size > 0, "validated frame size must be positive");
    assert!(
        frame_size <= FRAME_SIZE_BYTES_MAX,
        "validated frame size must remain bounded"
    );
    Ok(frame_size)
}

fn partial_output_path(output_path: &Path) -> Result<PathBuf, String> {
    assert!(
        !output_path.as_os_str().is_empty(),
        "output path must not be empty"
    );
    assert!(
        output_path.file_name().is_some(),
        "output path must include a filename"
    );
    let stem = output_path
        .file_stem()
        .and_then(|value| value.to_str())
        .ok_or_else(|| "output filename must be valid UTF-8".to_owned())?;
    let extension = output_path
        .extension()
        .and_then(|value| value.to_str())
        .ok_or_else(|| "output filename must have an extension".to_owned())?;
    let partial = output_path.with_file_name(format!("{stem}.partial.{extension}"));
    assert_ne!(partial, output_path, "partial and final paths must differ");
    assert_eq!(
        partial.extension(),
        output_path.extension(),
        "partial path must preserve the container extension"
    );
    Ok(partial)
}

fn send_update(updates: &SyncSender<JobUpdate>, update: JobUpdate) {
    assert!(
        PROGRESS_INTERVAL > Duration::ZERO,
        "progress interval must be positive"
    );
    assert!(
        OUTPUT_FRAME_COUNT_MAX > 0,
        "output frame limit must be positive"
    );
    let _ = updates.try_send(update);
    assert!(TARGET_FPS_MAX > 0, "target FPS limit must remain valid");
    assert!(
        FRAME_SIZE_BYTES_MAX > 0,
        "frame size limit must remain valid"
    );
}

fn wait_media_child(media_child: &mut MediaChild, description: &str) -> Result<ExitStatus, String> {
    assert!(
        !description.is_empty(),
        "media child description must not be empty"
    );
    assert!(
        description.len() <= 32,
        "media child description must remain bounded"
    );
    let wait_result = media_child
        .child
        .wait()
        .map_err(|error| format!("failed to wait for {description}: {error}"));
    let terminate_result = if wait_result.is_err() {
        terminate_child(&mut media_child.child)
    } else {
        Ok(())
    };
    let log_result = finish_log_worker(media_child);
    let status = wait_result?;
    terminate_result?;
    log_result?;
    assert!(
        media_child.log_worker.is_none(),
        "waited media log must be released"
    );
    assert!(
        !description.is_empty(),
        "media child description must remain valid"
    );
    Ok(status)
}

fn finish_log_worker(media_child: &mut MediaChild) -> Result<(), String> {
    assert!(
        FFMPEG_THREAD_COUNT == "2",
        "FFmpeg thread limit must remain explicit"
    );
    assert!(
        FRAME_SIZE_BYTES_MAX > 0,
        "memory limit must remain configured"
    );
    let log_worker = media_child
        .log_worker
        .take()
        .ok_or_else(|| "media log worker is unavailable".to_owned())?;
    let result = log_worker
        .join()
        .map_err(|_| "media log worker panicked".to_owned())?;
    result?;
    assert!(
        media_child.log_worker.is_none(),
        "finished log worker must be released"
    );
    assert!(
        OUTPUT_FRAME_COUNT_MAX > 0,
        "frame count limit must remain configured"
    );
    Ok(())
}

fn terminate_child(child: &mut Child) -> Result<(), String> {
    assert!(
        FFMPEG_THREAD_COUNT == "2",
        "FFmpeg thread limit must remain explicit"
    );
    assert!(
        FRAME_SIZE_BYTES_MAX > 0,
        "memory limit must remain configured"
    );
    match child.try_wait() {
        Ok(Some(_)) => {}
        Ok(None) => {
            child
                .kill()
                .map_err(|error| format!("failed to terminate media process: {error}"))?;
            child
                .wait()
                .map_err(|error| format!("failed to reap terminated media process: {error}"))?;
        }
        Err(error) => {
            let try_wait_error = error;
            child.kill().map_err(|kill_error| {
                format!(
                    "failed to inspect media process ({try_wait_error}) and terminate it: {kill_error}"
                )
            })?;
            child.wait().map_err(|wait_error| {
                format!("failed to reap media process after inspection failed: {wait_error}")
            })?;
        }
    }
    let final_status = child
        .try_wait()
        .map_err(|error| format!("failed to confirm media process termination: {error}"))?;
    if final_status.is_none() {
        return Err("media process remained active after termination".to_owned());
    }
    assert!(final_status.is_some(), "terminated child must be reaped");
    assert!(
        OUTPUT_FRAME_COUNT_MAX > 0,
        "frame count limit must remain configured"
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn output_name_includes_configuration() {
        let output = default_output_path(Path::new("/tmp/movie.mp4"), 120, true);
        assert_eq!(output, Path::new("/tmp/movie__rife-4.25__120fps__sc.mkv"));
        assert_eq!(
            output.extension().and_then(|value| value.to_str()),
            Some("mkv")
        );
    }

    #[test]
    fn bounded_process_output_is_drained_and_truncated() {
        let (captured, exceeded) =
            drain_output_bounded(&b"0123456789"[..], 4).expect("bounded output should be readable");
        assert_eq!(captured, b"0123");
        assert!(exceeded, "oversized process output must be reported");

        let (captured, exceeded) =
            drain_output_bounded(&b"ok"[..], 4).expect("small output should be readable");
        assert_eq!(captured, b"ok");
        assert!(!exceeded, "small process output must not be truncated");
    }

    #[test]
    fn rational_parser_reduces_ntsc_rate() {
        let rational = parse_rational("60000/2002").expect("rational should parse");
        assert_eq!(rational, (30_000, 1_001));
        assert!(rational.0 > rational.1);
    }

    #[test]
    fn rational_parser_rejects_malformed_rates() {
        assert!(parse_rational("24").is_err());
        assert!(parse_rational("24/0").is_err());
        assert!(parse_rational("nope/1").is_err());
        assert!(parse_rational("1/nope").is_err());
    }

    #[test]
    fn frame_size_and_partial_path_limits_are_enforced() {
        assert_eq!(checked_frame_size(64, 64), Ok(64 * 64 * RGB_CHANNEL_COUNT));
        assert!(checked_frame_size(FRAME_DIMENSION_MAX, FRAME_DIMENSION_MAX).is_err());
        let output_path = Path::new("/tmp/example.output.mkv");
        let partial_path = partial_output_path(output_path).expect("partial path should be valid");
        assert_eq!(partial_path, Path::new("/tmp/example.output.partial.mkv"));
        assert_eq!(partial_path.extension(), output_path.extension());
    }

    #[test]
    fn target_rate_must_exceed_source_rate() {
        let metadata = VideoMetadata {
            width: 64,
            height: 64,
            source_fps_num: 24_000,
            source_fps_den: 1_001,
            duration_seconds: 1.0,
            pixel_format: "yuv420p".to_owned(),
        };
        let mut configuration = JobConfiguration {
            input_path: PathBuf::from("/tmp/input.mkv"),
            output_path: PathBuf::from("/tmp/output.mkv"),
            target_fps_num: 24,
            target_fps_den: 1,
            gpu_index: 0,
            content_preset: ContentPreset::Movie,
            scene_detection: true,
            use_uhd_mode: false,
            use_nvdec: false,
            inference_backend: InferenceBackend::VulkanNcnn,
            video_encoder: VideoEncoder::SoftwareH264,
            encoder_preset: EncoderPreset::X264Medium,
            quality_level: 18,
            h264_profile: H264Profile::Auto,
            encoder_thread_count: 2,
            preserve_audio: true,
            preserve_subtitles: true,
            preserve_metadata: true,
        };
        assert!(validate_target_fps(&configuration, &metadata).is_ok());
        configuration.target_fps_num = 23;
        assert!(validate_target_fps(&configuration, &metadata).is_err());
        configuration.target_fps_num = 48;
        assert!(validate_target_fps(&configuration, &metadata).is_ok());
    }

    #[test]
    fn configuration_rejects_colliding_and_existing_outputs() {
        let process_id = std::process::id();
        let input_path = PathBuf::from(format!("/tmp/interpolate-config-input-{process_id}.mkv"));
        let output_path = PathBuf::from(format!("/tmp/interpolate-config-output-{process_id}.mkv"));
        fs::write(&input_path, b"input").expect("test input should be created");
        let _ = fs::remove_file(&output_path);
        let mut configuration = JobConfiguration {
            input_path: input_path.clone(),
            output_path: output_path.clone(),
            target_fps_num: 60,
            target_fps_den: 1,
            gpu_index: 0,
            content_preset: ContentPreset::Movie,
            scene_detection: true,
            use_uhd_mode: false,
            use_nvdec: false,
            inference_backend: InferenceBackend::VulkanNcnn,
            video_encoder: VideoEncoder::SoftwareH264,
            encoder_preset: EncoderPreset::X264Medium,
            quality_level: 18,
            h264_profile: H264Profile::Auto,
            encoder_thread_count: 2,
            preserve_audio: true,
            preserve_subtitles: true,
            preserve_metadata: true,
        };
        assert!(validate_configuration(&configuration).is_ok());
        configuration.output_path = input_path.clone();
        assert!(validate_configuration(&configuration).is_err());
        configuration.output_path = output_path.clone();
        fs::write(&output_path, b"output").expect("test output should be created");
        assert!(validate_configuration(&configuration).is_err());
        fs::remove_file(input_path).expect("test input should be removed");
        fs::remove_file(output_path).expect("test output should be removed");
    }

    #[test]
    fn complete_pipeline_smooths_a_tiny_anime_cadence() {
        let process_id = std::process::id();
        assert!(process_id > 0, "test process ID must be positive");
        assert!(
            OUTPUT_FRAME_COUNT_MAX > 8,
            "test must remain below output safety limit"
        );
        let input_path = PathBuf::from(format!("/tmp/interpolate-input-{process_id}.mkv"));
        let output_path = PathBuf::from(format!("/tmp/interpolate-output-{process_id}.mkv"));
        let subtitle_path = PathBuf::from(format!("/tmp/interpolate-subtitle-{process_id}.srt"));
        let _ = fs::remove_file(&input_path);
        let _ = fs::remove_file(&output_path);
        let _ = fs::remove_file(&subtitle_path);
        fs::write(
            &subtitle_path,
            "1\n00:00:00,000 --> 00:00:00,900\nInterpolation test\n",
        )
        .expect("test subtitle should be created");
        assert!(subtitle_path.is_file(), "test subtitle must exist");
        assert!(
            fs::metadata(&subtitle_path)
                .map(|metadata| metadata.len() > 0)
                .unwrap_or(false),
            "test subtitle must not be empty"
        );
        let generated = Command::new("ffmpeg")
            .args([
                "-nostdin",
                "-v",
                "error",
                "-y",
                "-f",
                "lavfi",
                "-i",
                "color=c=black:size=64x64:rate=4:duration=1,drawbox=color=0x202020:t=fill:enable=gte(n\\,3)",
                "-f",
                "lavfi",
                "-i",
                "sine=frequency=1000:sample_rate=48000:duration=1",
                "-f",
                "srt",
                "-i",
            ])
            .arg(&subtitle_path)
            .args([
                "-map",
                "0:v:0",
                "-map",
                "1:a:0",
                "-map",
                "2:s:0",
                "-c:a",
                "aac",
                "-c:s",
                "srt",
                "-pix_fmt",
                "yuv420p",
                "-c:v",
                "libx264",
                "-threads",
                "1",
            ])
            .arg(&input_path)
            .status()
            .expect("test FFmpeg generator should start");
        assert!(generated.success(), "test video generation must succeed");
        assert!(input_path.is_file(), "test input must exist");

        let configuration = JobConfiguration {
            input_path: input_path.clone(),
            output_path: output_path.clone(),
            target_fps_num: 8,
            target_fps_den: 1,
            gpu_index: 0,
            content_preset: ContentPreset::Anime,
            scene_detection: true,
            use_uhd_mode: false,
            use_nvdec: false,
            inference_backend: InferenceBackend::VulkanNcnn,
            video_encoder: VideoEncoder::SoftwareH264,
            encoder_preset: EncoderPreset::X264Medium,
            quality_level: 18,
            h264_profile: H264Profile::Auto,
            encoder_thread_count: 2,
            preserve_audio: true,
            preserve_subtitles: true,
            preserve_metadata: true,
        };
        let cancelled_output_path =
            PathBuf::from(format!("/tmp/interpolate-cancelled-{process_id}.mkv"));
        let _ = fs::remove_file(&cancelled_output_path);
        let mut cancelled_configuration = configuration.clone();
        cancelled_configuration.output_path = cancelled_output_path.clone();
        let cancelled_before_start = Arc::new(AtomicBool::new(true));
        let (cancelled_sender, cancelled_receiver) = std::sync::mpsc::sync_channel(4);
        run_job(
            cancelled_configuration,
            cancelled_before_start,
            cancelled_sender,
        );
        let mut cancellation_reported = false;
        for _ in 0..4 {
            match cancelled_receiver.try_recv() {
                Ok(JobUpdate::Cancelled { .. }) => {
                    cancellation_reported = true;
                    break;
                }
                Ok(JobUpdate::Phase(_)) => {}
                Ok(update) => panic!("unexpected cancellation update: {update:?}"),
                Err(error) => panic!("cancelled pipeline update missing: {error}"),
            }
        }
        assert!(
            cancellation_reported,
            "pre-start cancellation must be reported"
        );
        assert!(
            !cancelled_output_path.exists(),
            "cancelled output must not be published"
        );

        let active_cancelled_output_path = PathBuf::from(format!(
            "/tmp/interpolate-active-cancelled-{process_id}.mkv"
        ));
        let _ = fs::remove_file(&active_cancelled_output_path);
        let mut active_cancelled_configuration = configuration.clone();
        active_cancelled_configuration.output_path = active_cancelled_output_path.clone();
        let active_cancelled = Arc::new(AtomicBool::new(false));
        let active_worker_cancelled = Arc::clone(&active_cancelled);
        let (active_sender, active_receiver) = std::sync::mpsc::sync_channel(4);
        let active_worker = std::thread::spawn(move || {
            run_job(
                active_cancelled_configuration,
                active_worker_cancelled,
                active_sender,
            )
        });
        let mut active_phase_seen = false;
        let mut active_cancellation_reported = false;
        const ACTIVE_CANCELLATION_UPDATE_COUNT_MAX: usize = 30;
        for _ in 0..ACTIVE_CANCELLATION_UPDATE_COUNT_MAX {
            match active_receiver.recv_timeout(Duration::from_secs(1)) {
                Ok(JobUpdate::Phase("Starting media pipeline")) => {
                    active_phase_seen = true;
                    active_cancelled.store(true, Ordering::Release);
                }
                Ok(JobUpdate::Cancelled { .. }) => {
                    active_cancellation_reported = true;
                    break;
                }
                Ok(JobUpdate::Phase(_) | JobUpdate::Progress { .. }) => {}
                Ok(JobUpdate::Completed { .. }) => {
                    panic!("active cancellation must not complete the pipeline")
                }
                Ok(JobUpdate::Failed(error)) => {
                    panic!("active cancellation failed instead of cancelling: {error}")
                }
                Err(error) => panic!("active cancellation update failed: {error}"),
            }
        }
        active_worker
            .join()
            .expect("actively cancelled worker must exit cleanly");
        assert!(
            active_phase_seen,
            "cancellation must occur after media startup"
        );
        assert!(
            active_cancellation_reported,
            "active cancellation must report its terminal state"
        );
        assert!(
            !active_cancelled_output_path.exists(),
            "active cancellation must not publish output"
        );
        let active_partial_path = partial_output_path(&active_cancelled_output_path)
            .expect("active cancellation partial path must be valid");
        if active_partial_path.is_file() {
            fs::remove_file(&active_partial_path)
                .expect("active cancellation test must clean up partial output");
        }

        let cancelled = Arc::new(AtomicBool::new(false));
        let (sender, receiver) = std::sync::mpsc::sync_channel(1);
        let worker_cancelled = Arc::clone(&cancelled);
        let worker = std::thread::spawn(move || run_job(configuration, worker_cancelled, sender));
        let mut completed = false;
        for _ in 0..100 {
            match receiver.recv_timeout(Duration::from_secs(1)) {
                Ok(JobUpdate::Completed {
                    path,
                    cadence_diagnostics,
                }) => {
                    assert_eq!(
                        path, output_path,
                        "completion path must match requested output"
                    );
                    assert_eq!(
                        cadence_diagnostics.duplicate_frame_count, 2,
                        "Anime mode must detect the two repeated held drawings"
                    );
                    assert_eq!(
                        cadence_diagnostics.cadence_run_count, 1,
                        "Anime mode must smooth the three-frame cadence run"
                    );
                    completed = true;
                    break;
                }
                Ok(JobUpdate::Failed(error)) => {
                    panic!("tiny interpolation pipeline failed: {error}")
                }
                Ok(JobUpdate::Cancelled { .. }) => {
                    panic!("tiny interpolation pipeline was cancelled")
                }
                Ok(JobUpdate::Phase(_) | JobUpdate::Progress { .. }) => {}
                Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {}
                Err(error) => panic!("pipeline update channel failed: {error}"),
            }
        }
        worker.join().expect("pipeline worker must exit cleanly");
        assert!(
            completed,
            "pipeline must report completion within its bounded wait"
        );
        assert!(
            output_path.is_file(),
            "pipeline must produce an output file"
        );
        let output_metadata = probe_video(&output_path).expect("output should be probeable");
        assert_eq!(
            (
                output_metadata.source_fps_num,
                output_metadata.source_fps_den
            ),
            (8, 1),
            "output frame rate must match the request"
        );
        assert!(
            (0.9..=1.1).contains(&output_metadata.duration_seconds),
            "output duration must preserve the one-second source timeline"
        );
        let audio_probe = Command::new("ffprobe")
            .args([
                "-v",
                "error",
                "-select_streams",
                "a:0",
                "-show_entries",
                "stream=codec_type",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
            ])
            .arg(&output_path)
            .output()
            .expect("output audio probe should start");
        assert!(
            audio_probe.status.success(),
            "copied output audio must be probeable"
        );
        assert_eq!(
            String::from_utf8_lossy(&audio_probe.stdout).trim(),
            "audio",
            "pipeline must preserve the source audio stream"
        );
        let subtitle_probe = Command::new("ffprobe")
            .args([
                "-v",
                "error",
                "-select_streams",
                "s:0",
                "-show_entries",
                "stream=codec_type",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
            ])
            .arg(&output_path)
            .output()
            .expect("output subtitle probe should start");
        assert!(
            subtitle_probe.status.success(),
            "copied output subtitle must be probeable"
        );
        assert_eq!(
            String::from_utf8_lossy(&subtitle_probe.stdout).trim(),
            "subtitle",
            "pipeline must preserve the compatible source subtitle stream"
        );
        let _ = fs::remove_file(input_path);
        let _ = fs::remove_file(output_path);
        let _ = fs::remove_file(subtitle_path);
    }
}
