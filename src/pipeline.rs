use crate::backend::Backend;
use serde::Deserialize;
use std::{
    fs,
    io::{Read, Write},
    path::{Path, PathBuf},
    process::{Child, ChildStdout, Command, Stdio},
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
const SCENE_SAMPLE_STEP: usize = 4;
const SCENE_THRESHOLD_DEFAULT: f32 = 0.15;

#[derive(Clone)]
pub struct JobConfiguration {
    pub input_path: PathBuf,
    pub output_path: PathBuf,
    pub target_fps_num: u32,
    pub target_fps_den: u32,
    pub gpu_index: i32,
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
    },
    Completed(PathBuf),
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
        Ok(()) => JobUpdate::Completed(configuration.output_path.clone()),
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

fn run_job_inner(
    configuration: &JobConfiguration,
    cancelled: &AtomicBool,
    updates: &SyncSender<JobUpdate>,
) -> Result<(), String> {
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
    let model_directory = Path::new(env!("CARGO_MANIFEST_DIR")).join("models/rife-v4.25");
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

    let processing_result = process_frames(
        configuration,
        &metadata,
        cancelled,
        updates,
        &mut backend,
        &mut decoder,
        &mut encoder,
    );
    if processing_result.is_err() {
        terminate_child(&mut decoder);
        terminate_child(&mut encoder);
        let _ = fs::remove_file(&partial_path);
        return processing_result;
    }

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
    Ok(())
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

fn process_frames(
    configuration: &JobConfiguration,
    metadata: &VideoMetadata,
    cancelled: &AtomicBool,
    updates: &SyncSender<JobUpdate>,
    backend: &mut Backend,
    decoder: &mut Child,
    encoder: &mut Child,
) -> Result<(), String> {
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
    let mut frame_before = vec![0_u8; frame_size];
    let mut frame_after = vec![0_u8; frame_size];
    let mut frame_output = vec![0_u8; frame_size];
    if !read_frame(&mut decoder_output, &mut frame_before)? {
        return Err("decoder produced no video frames".to_owned());
    }

    let factor_num = u128::from(configuration.target_fps_num) * u128::from(metadata.source_fps_den);
    let factor_den = u128::from(configuration.target_fps_den) * u128::from(metadata.source_fps_num);
    let frame_count_estimate = ((metadata.duration_seconds
        * f64::from(configuration.target_fps_num)
        / f64::from(configuration.target_fps_den))
    .ceil() as u64)
        .clamp(1, OUTPUT_FRAME_COUNT_MAX);
    let started_at = Instant::now();
    let mut progress_at = started_at;
    let mut source_index = 0_u64;
    let mut output_index = 0_u64;

    send_update(updates, JobUpdate::Phase("Interpolating and encoding"));
    loop {
        if cancelled.load(Ordering::Acquire) {
            return Err("job cancelled".to_owned());
        }
        let has_next = read_frame(&mut decoder_output, &mut frame_after)?;
        if !has_next {
            frame_after.copy_from_slice(&frame_before);
        }
        let scene_change = has_next
            && configuration.scene_detection
            && scene_difference_rgb24(&frame_before, &frame_after, metadata.width, metadata.height)
                >= SCENE_THRESHOLD_DEFAULT;

        while output_index < OUTPUT_FRAME_COUNT_MAX {
            let scaled_position = u128::from(output_index) * factor_den;
            let real_source_index = scaled_position / factor_num;
            if real_source_index != u128::from(source_index) {
                break;
            }
            let remainder = scaled_position % factor_num;
            if remainder == 0 || !has_next || scene_change {
                encoder_input
                    .write_all(&frame_before)
                    .map_err(|error| format!("failed to send source frame to encoder: {error}"))?;
            } else {
                let timestep = (remainder as f64 / factor_num as f64) as f32;
                backend.interpolate_rgb24(
                    &frame_before,
                    &frame_after,
                    metadata.width,
                    metadata.height,
                    timestep,
                    &mut frame_output,
                )?;
                encoder_input.write_all(&frame_output).map_err(|error| {
                    format!("failed to send interpolated frame to encoder: {error}")
                })?;
            }
            output_index += 1;

            let now = Instant::now();
            if now.duration_since(progress_at) >= PROGRESS_INTERVAL {
                let elapsed_seconds = now.duration_since(started_at).as_secs_f64().max(0.001);
                let processing_fps = output_index as f64 / elapsed_seconds;
                let progress =
                    (output_index as f64 / frame_count_estimate as f64).clamp(0.0, 1.0) as f32;
                send_update(
                    updates,
                    JobUpdate::Progress {
                        frame_count: output_index,
                        frame_count_estimate,
                        processing_fps,
                        progress,
                    },
                );
                progress_at = now;
            }
        }
        if output_index >= OUTPUT_FRAME_COUNT_MAX {
            return Err(format!(
                "output exceeds the {OUTPUT_FRAME_COUNT_MAX}-frame safety limit"
            ));
        }
        if !has_next {
            break;
        }
        std::mem::swap(&mut frame_before, &mut frame_after);
        source_index = source_index
            .checked_add(1)
            .ok_or_else(|| "source frame index overflowed".to_owned())?;
    }

    encoder_input
        .flush()
        .map_err(|error| format!("failed to flush encoder input: {error}"))?;
    drop(encoder_input);
    send_update(
        updates,
        JobUpdate::Progress {
            frame_count: output_index,
            frame_count_estimate: output_index,
            processing_fps: output_index as f64 / started_at.elapsed().as_secs_f64().max(0.001),
            progress: 1.0,
        },
    );
    assert!(
        output_index > 0,
        "processing must produce at least one frame"
    );
    assert!(
        output_index < OUTPUT_FRAME_COUNT_MAX,
        "output must stay below its safety limit"
    );
    Ok(())
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

fn scene_difference_rgb24(frame_before: &[u8], frame_after: &[u8], width: u32, height: u32) -> f32 {
    assert_eq!(
        frame_before.len(),
        frame_after.len(),
        "scene frames must have equal size"
    );
    assert!(width > 0 && height > 0, "scene dimensions must be positive");
    let width = width as usize;
    let height = height as usize;
    let row_stride = width * RGB_CHANNEL_COUNT;
    let mut difference_sum = 0_u64;
    let mut sample_count = 0_u64;
    for y in (0..height).step_by(SCENE_SAMPLE_STEP) {
        for x in (0..width).step_by(SCENE_SAMPLE_STEP) {
            let index = y * row_stride + x * RGB_CHANNEL_COUNT;
            let before_luma = 54_u32 * u32::from(frame_before[index])
                + 183_u32 * u32::from(frame_before[index + 1])
                + 19_u32 * u32::from(frame_before[index + 2]);
            let after_luma = 54_u32 * u32::from(frame_after[index])
                + 183_u32 * u32::from(frame_after[index + 1])
                + 19_u32 * u32::from(frame_after[index + 2]);
            difference_sum += u64::from(before_luma.abs_diff(after_luma));
            sample_count += 1;
        }
    }
    let maximum_difference = sample_count * 255 * 256;
    let difference = difference_sum as f64 / maximum_difference.max(1) as f64;
    assert!(
        sample_count > 0,
        "scene detector must sample at least one pixel"
    );
    assert!(
        (0.0..=1.0).contains(&difference),
        "normalized scene difference must be bounded"
    );
    difference as f32
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
        let difference = scene_difference_rgb24(&black, &white, 12, 12);
        assert!(difference > 0.99);
        assert!(difference <= 1.0);
    }

    #[test]
    fn rational_parser_reduces_ntsc_rate() {
        let rational = parse_rational("60000/2002").expect("rational should parse");
        assert_eq!(rational, (30_000, 1_001));
        assert!(rational.0 > rational.1);
    }

    #[test]
    fn complete_pipeline_interpolates_a_tiny_video() {
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
                "testsrc=size=64x64:rate=4:duration=1",
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
            scene_detection: true,
            use_uhd_mode: false,
        };
        let cancelled = Arc::new(AtomicBool::new(false));
        let (sender, receiver) = std::sync::mpsc::sync_channel(1);
        let worker_cancelled = Arc::clone(&cancelled);
        let worker = std::thread::spawn(move || run_job(configuration, worker_cancelled, sender));
        let mut completed = false;
        for _ in 0..100 {
            match receiver.recv_timeout(Duration::from_secs(1)) {
                Ok(JobUpdate::Completed(path)) => {
                    assert_eq!(
                        path, output_path,
                        "completion path must match requested output"
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
        let _ = fs::remove_file(input_path);
        let _ = fs::remove_file(output_path);
    }
}
