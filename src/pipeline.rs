use crate::backend::Backend;
use serde::Deserialize;
use std::{
    fs,
    io::{Read, Write},
    path::{Path, PathBuf},
    process::{Child, ChildStdin, ChildStdout, Command, Stdio},
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
        mpsc::SyncSender,
    },
    time::{Duration, Instant},
};

const RGB_CHANNEL_COUNT: usize = 3;
const FRAME_SIZE_BYTES_MAX: usize = 128 * 1024 * 1024;
const FRAME_DIMENSION_MAX: u32 = 16_384;
const OUTPUT_FRAME_COUNT_MAX: u64 = 100_000_000;
const TARGET_FPS_MAX: u32 = 480;
const FFMPEG_THREAD_COUNT: &str = "2";
const PROGRESS_INTERVAL: Duration = Duration::from_millis(250);
const FRAME_SAMPLE_STEP: usize = 4;
const SCENE_THRESHOLD_DEFAULT: f64 = 0.15;
const DUPLICATE_MEAN_DIFFERENCE_MAX: f64 = 0.006;
const DUPLICATE_CHANGED_PIXEL_RATIO_MAX: f64 = 0.005;
const DUPLICATE_TILE_DIFFERENCE_MAX: f64 = 0.01;
const DUPLICATE_PIXEL_DIFFERENCE_MIN: u32 = 4 * 256;
const DIFFERENCE_TILE_COLUMN_COUNT: usize = 8;
const DIFFERENCE_TILE_ROW_COUNT: usize = 8;
const DIFFERENCE_TILE_COUNT: usize = DIFFERENCE_TILE_COLUMN_COUNT * DIFFERENCE_TILE_ROW_COUNT;
const CADENCE_RUN_FRAME_COUNT_MIN: u64 = 2;
const CADENCE_RUN_FRAME_COUNT_MAX: u64 = 3;
const MODEL_DIRECTORY_ENVIRONMENT: &str = "INTERPOLATE_MODEL_DIRECTORY";
const MODEL_DIRECTORY_RELATIVE: &str = "share/interpolate/models/rife-v4.25";

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ContentPreset {
    Anime,
    Movie,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct CadenceDiagnostics {
    pub duplicate_frame_count: u64,
    pub cadence_run_count: u64,
    pub long_hold_count: u64,
    pub scene_cut_count: u64,
    pub inference_bypass_count: u64,
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
    },
    Completed {
        path: PathBuf,
        cadence_diagnostics: CadenceDiagnostics,
    },
    Cancelled,
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
    if !input_path.is_file() {
        return Err("input video does not exist or is not a regular file".to_owned());
    }

    let output = Command::new("ffprobe")
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
        .output()
        .map_err(|error| format!("failed to start ffprobe: {error}"))?;
    if !output.status.success() {
        return Err("ffprobe could not inspect the selected video".to_owned());
    }
    const PROBE_OUTPUT_SIZE_MAX: usize = 1024 * 1024;
    if output.stdout.len() > PROBE_OUTPUT_SIZE_MAX {
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

pub fn run_job(
    configuration: JobConfiguration,
    cancelled: Arc<AtomicBool>,
    updates: SyncSender<JobUpdate>,
) {
    assert!(
        configuration.target_fps_num > 0,
        "target FPS numerator must be positive"
    );
    assert!(
        configuration.target_fps_den > 0,
        "target FPS denominator must be positive"
    );
    let result = run_job_inner(&configuration, &cancelled, &updates);
    let terminal_update = match result {
        Ok(cadence_diagnostics) => JobUpdate::Completed {
            path: configuration.output_path.clone(),
            cadence_diagnostics,
        },
        Err(_) if cancelled.load(Ordering::Acquire) => JobUpdate::Cancelled,
        Err(error) => JobUpdate::Failed(error),
    };
    let _ = updates.send(terminal_update);
    assert!(
        configuration.target_fps_num <= TARGET_FPS_MAX,
        "target FPS must remain bounded"
    );
    assert!(
        !configuration.output_path.as_os_str().is_empty(),
        "output path must remain valid"
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
    if let Some(installation_root) = executable.parent().and_then(Path::parent) {
        let installed_directory = installation_root.join(MODEL_DIRECTORY_RELATIVE);
        if model_files_exist(&installed_directory) {
            assert!(
                installed_directory.is_dir(),
                "installed model path must be a directory"
            );
            assert!(
                model_files_exist(&installed_directory),
                "installed model files must exist"
            );
            return Ok(installed_directory);
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
) -> Result<CadenceDiagnostics, String> {
    assert!(
        configuration.target_fps_num > 0,
        "target FPS must be positive"
    );
    assert!(
        !configuration.input_path.as_os_str().is_empty(),
        "input path must be present"
    );
    validate_configuration(configuration)?;
    send_update(updates, JobUpdate::Phase("Probing video"));
    let metadata = probe_video(&configuration.input_path)?;
    validate_target_fps(configuration, &metadata)?;
    if cancelled.load(Ordering::Acquire) {
        return Err("job cancelled".to_owned());
    }

    send_update(updates, JobUpdate::Phase("Initializing RIFE 4.25"));
    let model_directory = resolve_model_directory()?;
    let mut backend = Backend::create(
        &model_directory,
        configuration.gpu_index,
        configuration.use_uhd_mode,
    )?;

    let partial_path = partial_output_path(&configuration.output_path)?;
    if partial_path.exists() {
        fs::remove_file(&partial_path)
            .map_err(|error| format!("failed to remove stale partial output: {error}"))?;
    }

    send_update(updates, JobUpdate::Phase("Starting media pipeline"));
    let mut decoder = spawn_decoder(configuration, &metadata)?;
    let mut encoder = match spawn_encoder(configuration, &metadata, &partial_path) {
        Ok(encoder) => encoder,
        Err(error) => {
            terminate_child(&mut decoder);
            return Err(error);
        }
    };

    let cadence_diagnostics = match process_frames(
        configuration,
        &metadata,
        cancelled,
        updates,
        &mut backend,
        &mut decoder,
        &mut encoder,
    ) {
        Ok(cadence_diagnostics) => cadence_diagnostics,
        Err(error) => {
            terminate_child(&mut decoder);
            terminate_child(&mut encoder);
            let _ = fs::remove_file(&partial_path);
            return Err(error);
        }
    };

    send_update(updates, JobUpdate::Phase("Finalizing output"));
    let decoder_status = decoder
        .wait()
        .map_err(|error| format!("failed to wait for decoder: {error}"))?;
    let encoder_status = encoder
        .wait()
        .map_err(|error| format!("failed to wait for encoder: {error}"))?;
    if !decoder_status.success() {
        let _ = fs::remove_file(&partial_path);
        return Err("FFmpeg decoder failed".to_owned());
    }
    if !encoder_status.success() {
        let _ = fs::remove_file(&partial_path);
        return Err(
            "FFmpeg encoder failed; an input stream may not be compatible with MKV".to_owned(),
        );
    }
    fs::rename(&partial_path, &configuration.output_path)
        .map_err(|error| format!("failed to publish completed output: {error}"))?;
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

fn spawn_decoder(
    configuration: &JobConfiguration,
    metadata: &VideoMetadata,
) -> Result<Child, String> {
    assert!(metadata.width > 0, "decoder width must be positive");
    assert!(metadata.height > 0, "decoder height must be positive");
    let child = Command::new("ffmpeg")
        .args([
            "-nostdin",
            "-v",
            "error",
            "-threads",
            FFMPEG_THREAD_COUNT,
            "-i",
        ])
        .arg(&configuration.input_path)
        .args([
            "-map",
            "0:v:0",
            "-an",
            "-sn",
            "-dn",
            "-fps_mode",
            "passthrough",
            "-pix_fmt",
            "rgb24",
            "-f",
            "rawvideo",
            "pipe:1",
        ])
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .map_err(|error| format!("failed to start FFmpeg decoder: {error}"))?;
    assert!(child.stdout.is_some(), "decoder stdout must be piped");
    assert!(child.stdin.is_none(), "decoder stdin must be closed");
    Ok(child)
}

fn spawn_encoder(
    configuration: &JobConfiguration,
    metadata: &VideoMetadata,
    partial_path: &Path,
) -> Result<Child, String> {
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
    let child = Command::new("ffmpeg")
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
        .args([
            "-map",
            "0:v:0",
            "-map",
            "1:a?",
            "-map",
            "1:s?",
            "-map_metadata",
            "1",
            "-map_chapters",
            "1",
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            "-threads",
            FFMPEG_THREAD_COUNT,
            "-c:a",
            "copy",
            "-c:s",
            "copy",
        ])
        .arg(partial_path)
        .stdin(Stdio::piped())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .map_err(|error| format!("failed to start FFmpeg encoder: {error}"))?;
    assert!(child.stdin.is_some(), "encoder stdin must be piped");
    assert!(child.stdout.is_none(), "encoder stdout must be closed");
    Ok(child)
}

struct OutputScheduler<'a> {
    factor_num: u128,
    factor_den: u128,
    frame_count_estimate: u64,
    output_index: u64,
    started_at: Instant,
    progress_at: Instant,
    metadata: &'a VideoMetadata,
    cancelled: &'a AtomicBool,
    updates: &'a SyncSender<JobUpdate>,
    backend: &'a mut Backend,
    encoder_input: &'a mut ChildStdin,
    frame_output: Vec<u8>,
    cadence_diagnostics: CadenceDiagnostics,
}

impl<'a> OutputScheduler<'a> {
    fn new(
        configuration: &JobConfiguration,
        metadata: &'a VideoMetadata,
        cancelled: &'a AtomicBool,
        updates: &'a SyncSender<JobUpdate>,
        backend: &'a mut Backend,
        encoder_input: &'a mut ChildStdin,
        frame_size: usize,
    ) -> Self {
        assert!(
            configuration.target_fps_num > 0,
            "target FPS must be positive"
        );
        assert!(metadata.source_fps_num > 0, "source FPS must be positive");
        let factor_num =
            u128::from(configuration.target_fps_num) * u128::from(metadata.source_fps_den);
        let factor_den =
            u128::from(configuration.target_fps_den) * u128::from(metadata.source_fps_num);
        let frame_count_estimate = ((metadata.duration_seconds
            * f64::from(configuration.target_fps_num)
            / f64::from(configuration.target_fps_den))
        .ceil() as u64)
            .clamp(1, OUTPUT_FRAME_COUNT_MAX);
        let started_at = Instant::now();
        let scheduler = Self {
            factor_num,
            factor_den,
            frame_count_estimate,
            output_index: 0,
            started_at,
            progress_at: started_at,
            metadata,
            cancelled,
            updates,
            backend,
            encoder_input,
            frame_output: vec![0_u8; frame_size],
            cadence_diagnostics: CadenceDiagnostics::default(),
        };
        assert!(
            scheduler.factor_num > scheduler.factor_den,
            "target FPS must exceed source FPS"
        );
        assert_eq!(
            scheduler.frame_output.len(),
            frame_size,
            "output frame must have the requested size"
        );
        scheduler
    }

    fn write_span(
        &mut self,
        frame_before: &[u8],
        frame_after: &[u8],
        source_index_start: u64,
        source_index_end: u64,
        interpolate: bool,
    ) -> Result<(), String> {
        assert!(
            source_index_end > source_index_start,
            "source span must be positive"
        );
        assert_eq!(
            frame_before.len(),
            self.frame_output.len(),
            "source frame size must match output"
        );
        assert_eq!(
            frame_after.len(),
            self.frame_output.len(),
            "endpoint frame size must match output"
        );

        let position_scaled_start = u128::from(source_index_start)
            .checked_mul(self.factor_num)
            .ok_or_else(|| "source span start overflowed".to_owned())?;
        let position_scaled_end = u128::from(source_index_end)
            .checked_mul(self.factor_num)
            .ok_or_else(|| "source span end overflowed".to_owned())?;
        let duration_scaled = position_scaled_end - position_scaled_start;

        while self.output_index < OUTPUT_FRAME_COUNT_MAX {
            if self.cancelled.load(Ordering::Acquire) {
                return Err("job cancelled".to_owned());
            }
            let position_scaled = u128::from(self.output_index)
                .checked_mul(self.factor_den)
                .ok_or_else(|| "output frame position overflowed".to_owned())?;
            if position_scaled >= position_scaled_end {
                break;
            }
            if position_scaled < position_scaled_start {
                return Err("output scheduler encountered an overlapping source span".to_owned());
            }

            let relative_scaled = position_scaled - position_scaled_start;
            if interpolate && relative_scaled > 0 {
                let timestep = (relative_scaled as f64 / duration_scaled as f64) as f32;
                if !(0.0..1.0).contains(&timestep) {
                    return Err("interpolation timestep escaped its source span".to_owned());
                }
                self.backend.interpolate_rgb24(
                    frame_before,
                    frame_after,
                    self.metadata.width,
                    self.metadata.height,
                    timestep,
                    &mut self.frame_output,
                )?;
                self.encoder_input
                    .write_all(&self.frame_output)
                    .map_err(|error| {
                        format!("failed to send interpolated frame to encoder: {error}")
                    })?;
            } else {
                self.encoder_input
                    .write_all(frame_before)
                    .map_err(|error| format!("failed to send source frame to encoder: {error}"))?;
                if relative_scaled > 0 {
                    self.cadence_diagnostics.inference_bypass_count = self
                        .cadence_diagnostics
                        .inference_bypass_count
                        .saturating_add(1);
                }
            }
            self.output_index = self
                .output_index
                .checked_add(1)
                .ok_or_else(|| "output frame index overflowed".to_owned())?;
            self.report_progress();
        }
        if self.output_index >= OUTPUT_FRAME_COUNT_MAX {
            return Err(format!(
                "output exceeds the {OUTPUT_FRAME_COUNT_MAX}-frame safety limit"
            ));
        }
        assert!(
            self.output_index < OUTPUT_FRAME_COUNT_MAX,
            "output count must remain bounded"
        );
        assert!(
            duration_scaled > 0,
            "scaled source duration must be positive"
        );
        Ok(())
    }

    fn report_progress(&mut self) {
        assert!(
            self.frame_count_estimate > 0,
            "estimated frame count must be positive"
        );
        assert!(
            self.output_index < OUTPUT_FRAME_COUNT_MAX,
            "reported frame count must be bounded"
        );
        let now = Instant::now();
        if now.duration_since(self.progress_at) >= PROGRESS_INTERVAL {
            let elapsed_seconds = now.duration_since(self.started_at).as_secs_f64().max(0.001);
            let processing_fps = self.output_index as f64 / elapsed_seconds;
            let progress = (self.output_index as f64 / self.frame_count_estimate as f64)
                .clamp(0.0, 1.0) as f32;
            send_update(
                self.updates,
                JobUpdate::Progress {
                    frame_count: self.output_index,
                    frame_count_estimate: self.frame_count_estimate,
                    processing_fps,
                    progress,
                    cadence_diagnostics: self.cadence_diagnostics,
                },
            );
            self.progress_at = now;
        }
        assert!(self.started_at <= now, "progress time must be monotonic");
        assert!(
            (0.0..=1.0).contains(
                &(self.output_index as f64 / self.frame_count_estimate as f64).clamp(0.0, 1.0)
            ),
            "progress must remain bounded"
        );
    }

    fn finish(&mut self) -> Result<(), String> {
        assert!(
            self.output_index > 0,
            "processing must produce at least one frame"
        );
        assert!(
            self.output_index < OUTPUT_FRAME_COUNT_MAX,
            "output must stay below its safety limit"
        );
        self.encoder_input
            .flush()
            .map_err(|error| format!("failed to flush encoder input: {error}"))?;
        send_update(
            self.updates,
            JobUpdate::Progress {
                frame_count: self.output_index,
                frame_count_estimate: self.output_index,
                processing_fps: self.output_index as f64
                    / self.started_at.elapsed().as_secs_f64().max(0.001),
                progress: 1.0,
                cadence_diagnostics: self.cadence_diagnostics,
            },
        );
        assert!(self.output_index > 0, "finished output must contain frames");
        assert_eq!(
            self.frame_output.len(),
            checked_frame_size(self.metadata.width, self.metadata.height)?,
            "working frame size must remain valid"
        );
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct FrameDifference {
    mean: f64,
    changed_pixel_ratio: f64,
    tile_mean_max: f64,
}

fn process_frames(
    configuration: &JobConfiguration,
    metadata: &VideoMetadata,
    cancelled: &AtomicBool,
    updates: &SyncSender<JobUpdate>,
    backend: &mut Backend,
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
    if !read_frame(&mut decoder_output, &mut frame_anchor)? {
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

fn measure_frame_difference_rgb24(
    frame_before: &[u8],
    frame_after: &[u8],
    width: u32,
    height: u32,
) -> FrameDifference {
    assert_eq!(
        frame_before.len(),
        frame_after.len(),
        "difference frames must have equal size"
    );
    assert!(
        width > 0 && height > 0,
        "difference dimensions must be positive"
    );
    let width = width as usize;
    let height = height as usize;
    let row_stride = width * RGB_CHANNEL_COUNT;
    let mut difference_sum = 0_u64;
    let mut changed_pixel_count = 0_u64;
    let mut sample_count = 0_u64;
    let mut tile_difference_sums = [0_u64; DIFFERENCE_TILE_COUNT];
    let mut tile_sample_counts = [0_u64; DIFFERENCE_TILE_COUNT];

    for y in (0..height).step_by(FRAME_SAMPLE_STEP) {
        for x in (0..width).step_by(FRAME_SAMPLE_STEP) {
            let index = y * row_stride + x * RGB_CHANNEL_COUNT;
            let before_luma = 54_u32 * u32::from(frame_before[index])
                + 183_u32 * u32::from(frame_before[index + 1])
                + 19_u32 * u32::from(frame_before[index + 2]);
            let after_luma = 54_u32 * u32::from(frame_after[index])
                + 183_u32 * u32::from(frame_after[index + 1])
                + 19_u32 * u32::from(frame_after[index + 2]);
            let difference = before_luma.abs_diff(after_luma);
            difference_sum += u64::from(difference);
            if difference >= DUPLICATE_PIXEL_DIFFERENCE_MIN {
                changed_pixel_count += 1;
            }
            let tile_x =
                (x * DIFFERENCE_TILE_COLUMN_COUNT / width).min(DIFFERENCE_TILE_COLUMN_COUNT - 1);
            let tile_y =
                (y * DIFFERENCE_TILE_ROW_COUNT / height).min(DIFFERENCE_TILE_ROW_COUNT - 1);
            let tile_index = tile_y * DIFFERENCE_TILE_COLUMN_COUNT + tile_x;
            tile_difference_sums[tile_index] += u64::from(difference);
            tile_sample_counts[tile_index] += 1;
            sample_count += 1;
        }
    }

    let difference_max = 255_u64 * 256_u64;
    let mean = difference_sum as f64 / (sample_count.max(1) * difference_max) as f64;
    let changed_pixel_ratio = changed_pixel_count as f64 / sample_count.max(1) as f64;
    let mut tile_mean_max = 0.0_f64;
    for tile_index in 0..DIFFERENCE_TILE_COUNT {
        let tile_sample_count = tile_sample_counts[tile_index];
        if tile_sample_count > 0 {
            let tile_mean = tile_difference_sums[tile_index] as f64
                / (tile_sample_count * difference_max) as f64;
            tile_mean_max = tile_mean_max.max(tile_mean);
        }
    }
    let difference = FrameDifference {
        mean,
        changed_pixel_ratio,
        tile_mean_max,
    };
    assert!(
        sample_count > 0,
        "difference detector must sample at least one pixel"
    );
    assert!(
        (0.0..=1.0).contains(&difference.mean),
        "mean difference must be normalized"
    );
    difference
}

fn is_smoothable_cadence_run(cadence_run_frame_count: u64) -> bool {
    assert!(
        cadence_run_frame_count > 0,
        "cadence run must contain at least one frame"
    );
    assert!(
        cadence_run_frame_count < OUTPUT_FRAME_COUNT_MAX,
        "cadence run must remain bounded"
    );
    let smoothable = (CADENCE_RUN_FRAME_COUNT_MIN..=CADENCE_RUN_FRAME_COUNT_MAX)
        .contains(&cadence_run_frame_count);
    assert!(
        !smoothable || cadence_run_frame_count >= CADENCE_RUN_FRAME_COUNT_MIN,
        "smoothable cadence must contain a held drawing"
    );
    assert!(
        !smoothable || cadence_run_frame_count <= CADENCE_RUN_FRAME_COUNT_MAX,
        "smoothable cadence must satisfy its upper limit"
    );
    smoothable
}

fn is_confident_duplicate(difference: FrameDifference) -> bool {
    assert!(
        (0.0..=1.0).contains(&difference.mean),
        "mean difference must be normalized"
    );
    assert!(
        (0.0..=1.0).contains(&difference.changed_pixel_ratio),
        "changed-pixel ratio must be normalized"
    );
    let duplicate = difference.mean <= DUPLICATE_MEAN_DIFFERENCE_MAX
        && difference.changed_pixel_ratio <= DUPLICATE_CHANGED_PIXEL_RATIO_MAX
        && difference.tile_mean_max <= DUPLICATE_TILE_DIFFERENCE_MAX;
    assert!(
        !duplicate || difference.tile_mean_max <= DUPLICATE_TILE_DIFFERENCE_MAX,
        "duplicate tiles must satisfy their limit"
    );
    assert!(
        !duplicate || difference.mean <= DUPLICATE_MEAN_DIFFERENCE_MAX,
        "duplicate mean must satisfy its limit"
    );
    duplicate
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

fn terminate_child(child: &mut Child) {
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
            let _ = child.kill();
            let _ = child.wait();
        }
        Err(_) => {
            let _ = child.kill();
            let _ = child.wait();
        }
    }
    assert!(
        child.try_wait().ok().flatten().is_some(),
        "terminated child must be reaped"
    );
    assert!(
        OUTPUT_FRAME_COUNT_MAX > 0,
        "frame count limit must remain configured"
    );
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
    fn scene_difference_detects_a_hard_cut() {
        let black = vec![0_u8; 12 * 12 * RGB_CHANNEL_COUNT];
        let white = vec![255_u8; 12 * 12 * RGB_CHANNEL_COUNT];
        let difference = measure_frame_difference_rgb24(&black, &white, 12, 12);
        assert!(difference.mean > 0.99);
        assert!(difference.mean <= 1.0);
    }

    #[test]
    fn confident_duplicate_accepts_uniform_decode_noise() {
        let frame_before = vec![64_u8; 64 * 64 * RGB_CHANNEL_COUNT];
        let frame_after = vec![65_u8; 64 * 64 * RGB_CHANNEL_COUNT];
        let difference = measure_frame_difference_rgb24(&frame_before, &frame_after, 64, 64);
        assert!(is_confident_duplicate(difference));
        assert!(difference.mean > 0.0);
    }

    #[test]
    fn confident_duplicate_rejects_localized_motion() {
        let frame_before = vec![0_u8; 64 * 64 * RGB_CHANNEL_COUNT];
        let mut frame_after = frame_before.clone();
        frame_after[..RGB_CHANNEL_COUNT].fill(255);
        let difference = measure_frame_difference_rgb24(&frame_before, &frame_after, 64, 64);
        assert!(!is_confident_duplicate(difference));
        assert!(difference.tile_mean_max > DUPLICATE_TILE_DIFFERENCE_MAX);
    }

    #[test]
    fn cadence_smoothing_accepts_only_two_or_three_frame_runs() {
        assert!(!is_smoothable_cadence_run(1));
        assert!(is_smoothable_cadence_run(2));
        assert!(is_smoothable_cadence_run(3));
        assert!(!is_smoothable_cadence_run(4));
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
    fn duplicate_confidence_requires_every_metric() {
        let accepted = FrameDifference {
            mean: DUPLICATE_MEAN_DIFFERENCE_MAX,
            changed_pixel_ratio: DUPLICATE_CHANGED_PIXEL_RATIO_MAX,
            tile_mean_max: DUPLICATE_TILE_DIFFERENCE_MAX,
        };
        assert!(is_confident_duplicate(accepted));
        assert!(!is_confident_duplicate(FrameDifference {
            mean: DUPLICATE_MEAN_DIFFERENCE_MAX + f64::EPSILON,
            ..accepted
        }));
        assert!(!is_confident_duplicate(FrameDifference {
            changed_pixel_ratio: DUPLICATE_CHANGED_PIXEL_RATIO_MAX + f64::EPSILON,
            ..accepted
        }));
        assert!(!is_confident_duplicate(FrameDifference {
            tile_mean_max: DUPLICATE_TILE_DIFFERENCE_MAX + f64::EPSILON,
            ..accepted
        }));
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
        let input_path = PathBuf::from(format!("/tmp/interpolate-input-{process_id}.mp4"));
        let output_path = PathBuf::from(format!("/tmp/interpolate-output-{process_id}.mkv"));
        let _ = fs::remove_file(&input_path);
        let _ = fs::remove_file(&output_path);
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
                Ok(JobUpdate::Cancelled) => {
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
                Ok(JobUpdate::Cancelled) => panic!("tiny interpolation pipeline was cancelled"),
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
        let _ = fs::remove_file(input_path);
        let _ = fs::remove_file(output_path);
    }
}
