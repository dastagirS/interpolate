mod backend;
mod pipeline;

use backend::{gpu_names, shutdown as shutdown_backend};
use gpui::{
    App, Application, Bounds, Context, FocusHandle, KeyDownEvent, PathPromptOptions, Render, Timer,
    Window, WindowBounds, WindowOptions, div, prelude::*, px, relative, rgb, size,
};
use pipeline::{
    CadenceDiagnostics, ContentPreset, JobConfiguration, JobUpdate, VideoMetadata,
    default_output_path, probe_video, run_job,
};
use std::{
    path::{Path, PathBuf},
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
        mpsc::{TryRecvError, sync_channel},
    },
    thread,
    time::Duration,
};

const WINDOW_WIDTH_PIXELS: f32 = 960.0;
const WINDOW_HEIGHT_PIXELS: f32 = 680.0;
const BACKGROUND_COLOR: u32 = 0x0a0a0a;
const SIDEBAR_COLOR: u32 = 0x0c0c0c;
const PANEL_COLOR: u32 = 0x111111;
const PANEL_HOVER_COLOR: u32 = 0x171717;
const SELECTED_COLOR: u32 = 0x242424;
const BORDER_COLOR: u32 = 0x262626;
const BORDER_STRONG_COLOR: u32 = 0x3f3f46;
const TEXT_COLOR: u32 = 0xededed;
const TEXT_MUTED_COLOR: u32 = 0xa1a1aa;
const TEXT_SUBTLE_COLOR: u32 = 0x71717a;
const ACCENT_COLOR: u32 = 0x3b82f6;
const SUCCESS_COLOR: u32 = 0x3ecf8e;
const ERROR_COLOR: u32 = 0xf87171;
const PRIMARY_BUTTON_COLOR: u32 = 0xededed;
const PRIMARY_BUTTON_HOVER_COLOR: u32 = 0xffffff;
const PRIMARY_BUTTON_TEXT_COLOR: u32 = 0x0a0a0a;
const PROGRESS_POLL_INTERVAL: Duration = Duration::from_millis(125);
const PROGRESS_POLL_COUNT_MAX: usize = 4_838_400;
const UPDATE_DRAIN_COUNT_MAX: usize = 4;
const SECONDS_PER_MINUTE: u64 = 60;
const MINUTES_PER_HOUR: u64 = 60;
const SECONDS_PER_HOUR: u64 = SECONDS_PER_MINUTE * MINUTES_PER_HOUR;

fn use_uhd_mode_default(content_preset: ContentPreset, metadata: Option<&VideoMetadata>) -> bool {
    assert!(UHD_WIDTH_MIN > 0, "UHD width threshold must be positive");
    assert!(UHD_HEIGHT_MIN > 0, "UHD height threshold must be positive");
    let use_uhd_mode = content_preset == ContentPreset::Anime
        && metadata.is_some_and(|metadata| {
            let landscape_uhd =
                metadata.width >= UHD_WIDTH_MIN && metadata.height >= UHD_HEIGHT_MIN;
            let portrait_uhd = metadata.width >= UHD_HEIGHT_MIN && metadata.height >= UHD_WIDTH_MIN;
            landscape_uhd || portrait_uhd
        });
    assert!(
        !use_uhd_mode || content_preset == ContentPreset::Anime,
        "automatic UHD mode is reserved for Anime"
    );
    assert!(
        !use_uhd_mode || metadata.is_some(),
        "automatic UHD mode requires source metadata"
    );
    use_uhd_mode
}

fn format_remaining_time(remaining_seconds: u64) -> String {
    assert!(
        SECONDS_PER_MINUTE > 0,
        "seconds per minute must be positive"
    );
    assert_eq!(
        SECONDS_PER_HOUR,
        SECONDS_PER_MINUTE * MINUTES_PER_HOUR,
        "hour conversion must remain exact"
    );

    let hours = remaining_seconds / SECONDS_PER_HOUR;
    let minutes = (remaining_seconds % SECONDS_PER_HOUR) / SECONDS_PER_MINUTE;
    let seconds = remaining_seconds % SECONDS_PER_MINUTE;
    let formatted = if hours > 0 {
        format!("{hours}:{minutes:02}:{seconds:02} remaining")
    } else {
        format!("{minutes}:{seconds:02} remaining")
    };

    assert!(!formatted.is_empty(), "remaining time must be displayed");
    assert!(
        formatted.ends_with(" remaining"),
        "remaining time must retain its label"
    );
    formatted
}

const UHD_WIDTH_MIN: u32 = 3_840;
const UHD_HEIGHT_MIN: u32 = 2_160;

struct InterpolateApp {
    input_path: Option<PathBuf>,
    output_path: Option<PathBuf>,
    metadata: Option<VideoMetadata>,
    gpu_names: Vec<String>,
    selected_gpu_index: usize,
    gpu_menu_open: bool,
    preset_menu_open: bool,
    target_fps_num: u32,
    target_fps_input: String,
    target_fps_replace_on_type: bool,
    target_fps_focus: FocusHandle,
    content_preset: ContentPreset,
    scene_detection: bool,
    scene_detection_overridden: bool,
    use_uhd_mode: bool,
    use_uhd_mode_overridden: bool,
    cadence_diagnostics: CadenceDiagnostics,
    running: bool,
    status: String,
    error: Option<String>,
    progress: f32,
    frame_count: u64,
    frame_count_estimate: u64,
    processing_fps: f64,
    cancellation: Option<Arc<AtomicBool>>,
}

impl InterpolateApp {
    fn new(cx: &mut Context<Self>) -> Self {
        assert!(WINDOW_WIDTH_PIXELS > 0.0, "window width must be positive");
        assert!(WINDOW_HEIGHT_PIXELS > 0.0, "window height must be positive");
        let arguments: Vec<String> = std::env::args().take(3).collect();
        let input_path = arguments
            .get(1)
            .map(PathBuf::from)
            .filter(|path| path.is_file());
        let output_path = arguments.get(2).map(PathBuf::from).or_else(|| {
            input_path
                .as_deref()
                .map(|path| default_output_path(path, 120, true))
        });
        let metadata = input_path
            .as_deref()
            .and_then(|path| probe_video(path).ok());
        let (gpu_names, error) = match gpu_names() {
            Ok(names) if !names.is_empty() => (names, None),
            Ok(_) => (Vec::new(), Some("No Vulkan GPU was found".to_owned())),
            Err(error) => (Vec::new(), Some(error)),
        };
        let app = Self {
            input_path,
            output_path,
            metadata,
            gpu_names,
            selected_gpu_index: 0,
            gpu_menu_open: false,
            preset_menu_open: false,
            target_fps_num: 120,
            target_fps_input: "120".to_owned(),
            target_fps_replace_on_type: false,
            target_fps_focus: cx.focus_handle(),
            content_preset: ContentPreset::Movie,
            scene_detection: true,
            scene_detection_overridden: false,
            use_uhd_mode: false,
            use_uhd_mode_overridden: false,
            cadence_diagnostics: CadenceDiagnostics::default(),
            running: false,
            status: "Ready to interpolate".to_owned(),
            error,
            progress: 0.0,
            frame_count: 0,
            frame_count_estimate: 0,
            processing_fps: 0.0,
            cancellation: None,
        };
        assert!(
            app.target_fps_num > 0,
            "default target FPS must be positive"
        );
        assert!(
            app.selected_gpu_index <= app.gpu_names.len(),
            "selected GPU index must be bounded"
        );
        app
    }

    fn select_input(&mut self, _: &gpui::ClickEvent, _: &mut Window, cx: &mut Context<Self>) {
        assert!(self.target_fps_num > 0, "target FPS must be positive");
        assert!(
            self.selected_gpu_index <= self.gpu_names.len(),
            "GPU selection must be bounded"
        );
        if self.running {
            return;
        }
        let receiver = cx.prompt_for_paths(PathPromptOptions {
            files: true,
            directories: false,
            multiple: false,
            prompt: Some("Choose video".into()),
        });
        cx.spawn(async move |this, cx| {
            if let Ok(Ok(Some(paths))) = receiver.await
                && let Some(path) = paths.into_iter().next()
            {
                let metadata = probe_video(&path);
                let _ = this.update(cx, |app, cx| {
                    match metadata {
                        Ok(metadata) => {
                            app.output_path = Some(default_output_path(
                                &path,
                                app.target_fps_num,
                                app.scene_detection,
                            ));
                            app.input_path = Some(path);
                            app.metadata = Some(metadata);
                            if !app.use_uhd_mode_overridden {
                                app.use_uhd_mode =
                                    use_uhd_mode_default(app.content_preset, app.metadata.as_ref());
                            }
                            app.error = None;
                            app.status = "Video ready".to_owned();
                        }
                        Err(error) => {
                            app.input_path = Some(path);
                            app.metadata = None;
                            if !app.use_uhd_mode_overridden {
                                app.use_uhd_mode = false;
                            }
                            app.error = Some(error);
                            app.status = "Input needs attention".to_owned();
                        }
                    }
                    cx.notify();
                });
            }
        })
        .detach();
        assert!(!self.running, "file prompt must not start while processing");
        assert!(
            self.cancellation.is_none(),
            "idle file prompt must not have cancellation state"
        );
    }

    fn select_output(&mut self, _: &gpui::ClickEvent, _: &mut Window, cx: &mut Context<Self>) {
        assert!(self.target_fps_num > 0, "target FPS must be positive");
        assert!(
            self.output_path
                .as_ref()
                .is_none_or(|path| !path.as_os_str().is_empty()),
            "output path must be valid"
        );
        if self.running {
            return;
        }
        let Some(input_path) = self.input_path.as_deref() else {
            self.error = Some("Choose an input video first".to_owned());
            cx.notify();
            return;
        };
        let directory = input_path.parent().unwrap_or_else(|| Path::new("."));
        let suggested = self
            .output_path
            .as_ref()
            .and_then(|path| path.file_name())
            .and_then(|value| value.to_str());
        let receiver = cx.prompt_for_new_path(directory, suggested);
        cx.spawn(async move |this, cx| {
            if let Ok(Ok(Some(path))) = receiver.await {
                let _ = this.update(cx, |app, cx| {
                    app.output_path = Some(path);
                    app.error = None;
                    cx.notify();
                });
            }
        })
        .detach();
        assert!(!self.running, "output prompt must be idle");
        assert!(self.input_path.is_some(), "output prompt requires an input");
    }

    fn focus_target_fps(
        &mut self,
        _: &gpui::ClickEvent,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        assert!(self.target_fps_num > 0, "target FPS must be positive");
        assert!(self.target_fps_num <= 480, "target FPS must remain bounded");
        if !self.running {
            self.target_fps_replace_on_type = true;
            window.focus(&self.target_fps_focus);
            cx.notify();
        }
        assert!(
            self.target_fps_input.len() <= 3,
            "focused FPS input must remain bounded"
        );
        assert!(
            self.running || self.cancellation.is_none(),
            "idle state must not retain cancellation"
        );
    }

    fn edit_target_fps(
        &mut self,
        event: &KeyDownEvent,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        assert!(
            self.target_fps_input.len() <= 3,
            "FPS input must remain bounded"
        );
        assert!(self.target_fps_num <= 480, "target FPS must remain bounded");
        if self.running {
            return;
        }

        match event.keystroke.key.as_str() {
            "backspace" => {
                if self.target_fps_replace_on_type {
                    self.target_fps_input.clear();
                    self.target_fps_replace_on_type = false;
                } else {
                    self.target_fps_input.pop();
                }
            }
            "enter" | "escape" => {
                window.blur();
                self.target_fps_replace_on_type = false;
            }
            _ => {
                if let Some(character) = event.keystroke.key_char.as_deref()
                    && character.len() == 1
                    && character.as_bytes()[0].is_ascii_digit()
                {
                    if self.target_fps_replace_on_type {
                        self.target_fps_input.clear();
                        self.target_fps_replace_on_type = false;
                    }
                    if self.target_fps_input.len() < 3 {
                        self.target_fps_input.push_str(character);
                    }
                } else {
                    return;
                }
            }
        }

        match self.target_fps_input.parse::<u32>() {
            Ok(target_fps_num) if (1..=480).contains(&target_fps_num) => {
                self.target_fps_num = target_fps_num;
                self.refresh_default_output();
                self.error = None;
            }
            _ => self.error = Some("Target frame rate must be between 1 and 480 FPS".to_owned()),
        }
        cx.stop_propagation();
        cx.notify();
        assert!(
            self.target_fps_input.len() <= 3,
            "edited FPS input must remain bounded"
        );
        assert!(
            self.target_fps_num > 0,
            "last valid target FPS must remain positive"
        );
    }

    fn toggle_gpu_menu(&mut self, _: &gpui::ClickEvent, _: &mut Window, cx: &mut Context<Self>) {
        assert!(
            self.selected_gpu_index <= self.gpu_names.len(),
            "GPU selection must be bounded"
        );
        assert!(self.gpu_names.len() <= 16, "GPU list must remain bounded");
        if !self.running && !self.gpu_names.is_empty() {
            self.gpu_menu_open = !self.gpu_menu_open;
            self.preset_menu_open = false;
            cx.notify();
        }
        assert!(
            !self.gpu_names.is_empty() || !self.gpu_menu_open,
            "empty GPU list cannot open a menu"
        );
        assert!(
            self.running || self.cancellation.is_none(),
            "idle state must not retain cancellation"
        );
    }

    fn select_gpu(&mut self, gpu_index: usize, cx: &mut Context<Self>) {
        assert!(
            gpu_index < self.gpu_names.len(),
            "selected GPU index must exist"
        );
        assert!(self.gpu_names.len() <= 16, "GPU list must remain bounded");
        if !self.running {
            self.selected_gpu_index = gpu_index;
            self.gpu_menu_open = false;
            cx.notify();
        }
        assert!(
            self.selected_gpu_index < self.gpu_names.len(),
            "updated GPU selection must be valid"
        );
        assert!(
            !self.gpu_menu_open || !self.running,
            "running state cannot retain an open GPU menu"
        );
    }

    fn toggle_preset_menu(&mut self, _: &gpui::ClickEvent, _: &mut Window, cx: &mut Context<Self>) {
        assert!(self.target_fps_num > 0, "target FPS must be positive");
        assert!(self.target_fps_num <= 480, "target FPS must stay bounded");
        if !self.running {
            self.preset_menu_open = !self.preset_menu_open;
            self.gpu_menu_open = false;
            cx.notify();
        }
        assert!(
            !self.preset_menu_open || !self.running,
            "running state cannot retain an open preset menu"
        );
        assert!(
            !self.preset_menu_open || !self.gpu_menu_open,
            "only one settings menu may be open"
        );
    }

    fn select_content_preset(&mut self, preset: ContentPreset, cx: &mut Context<Self>) {
        assert!(self.target_fps_num > 0, "target FPS must be positive");
        assert!(
            self.selected_gpu_index <= self.gpu_names.len(),
            "GPU selection must be bounded"
        );
        if !self.running {
            self.content_preset = preset;
            self.scene_detection = true;
            self.scene_detection_overridden = false;
            self.use_uhd_mode = use_uhd_mode_default(preset, self.metadata.as_ref());
            self.use_uhd_mode_overridden = false;
            self.preset_menu_open = false;
            self.refresh_default_output();
            cx.notify();
        }
        assert!(
            self.running || self.content_preset == preset,
            "idle preset selection must be applied"
        );
        assert!(
            !self.preset_menu_open || !self.running,
            "running state cannot retain an open preset menu"
        );
        assert!(
            self.content_preset != ContentPreset::Anime || self.scene_detection,
            "Anime preset must retain scene protection"
        );
    }

    fn toggle_scene_detection(
        &mut self,
        _: &gpui::ClickEvent,
        _: &mut Window,
        cx: &mut Context<Self>,
    ) {
        assert!(self.target_fps_num > 0, "target FPS must be positive");
        assert!(
            self.output_path
                .as_ref()
                .is_none_or(|path| !path.as_os_str().is_empty()),
            "output path must be valid"
        );
        if !self.running && self.content_preset == ContentPreset::Movie {
            self.scene_detection = !self.scene_detection;
            self.scene_detection_overridden = !self.scene_detection;
            self.refresh_default_output();
            cx.notify();
        }
        assert!(
            self.running || self.cancellation.is_none(),
            "idle state must not retain cancellation"
        );
        assert!(self.target_fps_num <= 480, "target FPS must stay bounded");
    }

    fn toggle_uhd_mode(&mut self, _: &gpui::ClickEvent, _: &mut Window, cx: &mut Context<Self>) {
        assert!(self.target_fps_num > 0, "target FPS must be positive");
        assert!(
            self.selected_gpu_index <= self.gpu_names.len(),
            "GPU selection must be bounded"
        );
        if !self.running {
            self.use_uhd_mode = !self.use_uhd_mode;
            self.use_uhd_mode_overridden = self.use_uhd_mode
                != use_uhd_mode_default(self.content_preset, self.metadata.as_ref());
            cx.notify();
        }
        assert!(
            self.running || self.cancellation.is_none(),
            "idle state must not retain cancellation"
        );
        assert!(self.target_fps_num <= 480, "target FPS must stay bounded");
    }

    fn refresh_default_output(&mut self) {
        assert!(self.target_fps_num > 0, "target FPS must be positive");
        assert!(self.target_fps_num <= 480, "target FPS must remain bounded");
        if let Some(input_path) = self.input_path.as_deref() {
            self.output_path = Some(default_output_path(
                input_path,
                self.target_fps_num,
                self.scene_detection,
            ));
        }
        assert!(
            self.output_path
                .as_ref()
                .is_none_or(|path| path.file_name().is_some()),
            "output must include a filename"
        );
        assert!(
            self.input_path.is_some() || self.output_path.is_none(),
            "output requires an input unless explicitly provided"
        );
    }

    fn start_job(&mut self, _: &gpui::ClickEvent, _: &mut Window, cx: &mut Context<Self>) {
        assert!(self.target_fps_num > 0, "target FPS must be positive");
        assert!(
            self.selected_gpu_index <= self.gpu_names.len(),
            "GPU selection must be bounded"
        );
        if self.running {
            return;
        }
        let Ok(target_fps_num) = self.target_fps_input.parse::<u32>() else {
            self.error = Some("Enter a target frame rate between 1 and 480 FPS".to_owned());
            cx.notify();
            return;
        };
        if !(1..=480).contains(&target_fps_num) {
            self.error = Some("Target frame rate must be between 1 and 480 FPS".to_owned());
            cx.notify();
            return;
        }
        self.target_fps_num = target_fps_num;
        let Some(input_path) = self.input_path.clone() else {
            self.error = Some("Choose an input video".to_owned());
            cx.notify();
            return;
        };
        let Some(output_path) = self.output_path.clone() else {
            self.error = Some("Choose an output filename".to_owned());
            cx.notify();
            return;
        };
        if self.gpu_names.is_empty() {
            self.error = Some("A Vulkan GPU is required".to_owned());
            cx.notify();
            return;
        }

        let Ok(gpu_index) = i32::try_from(self.selected_gpu_index) else {
            self.error = Some("selected GPU index exceeds the native ABI".to_owned());
            cx.notify();
            return;
        };
        let configuration = JobConfiguration {
            input_path,
            output_path,
            target_fps_num: self.target_fps_num,
            target_fps_den: 1,
            gpu_index,
            content_preset: self.content_preset,
            scene_detection: self.scene_detection,
            use_uhd_mode: self.use_uhd_mode,
        };
        let cancellation = Arc::new(AtomicBool::new(false));
        let worker_cancellation = Arc::clone(&cancellation);
        let (update_sender, update_receiver) = sync_channel(1);
        let spawn_result = thread::Builder::new()
            .name("interpolation-worker".to_owned())
            .spawn(move || run_job(configuration, worker_cancellation, update_sender));
        if let Err(error) = spawn_result {
            self.error = Some(format!("failed to start interpolation worker: {error}"));
            cx.notify();
            return;
        }

        self.running = true;
        self.gpu_menu_open = false;
        self.preset_menu_open = false;
        self.status = "Starting job".to_owned();
        self.error = None;
        self.progress = 0.0;
        self.frame_count = 0;
        self.frame_count_estimate = 0;
        self.processing_fps = 0.0;
        self.cadence_diagnostics = CadenceDiagnostics::default();
        self.cancellation = Some(cancellation);
        cx.notify();

        cx.spawn(async move |this, cx| {
            for _ in 0..PROGRESS_POLL_COUNT_MAX {
                Timer::after(PROGRESS_POLL_INTERVAL).await;
                let mut terminal = false;
                for _ in 0..UPDATE_DRAIN_COUNT_MAX {
                    match update_receiver.try_recv() {
                        Ok(update) => {
                            terminal = matches!(
                                update,
                                JobUpdate::Completed { .. }
                                    | JobUpdate::Cancelled
                                    | JobUpdate::Failed(_)
                            );
                            let _ = this.update(cx, |app, cx| {
                                app.apply_update(update);
                                cx.notify();
                            });
                            if terminal {
                                break;
                            }
                        }
                        Err(TryRecvError::Empty) => break,
                        Err(TryRecvError::Disconnected) => {
                            terminal = true;
                            break;
                        }
                    }
                }
                if terminal {
                    break;
                }
            }
        })
        .detach();
        assert!(self.running, "job must be running after worker spawn");
        assert!(
            self.cancellation.is_some(),
            "running job must have cancellation state"
        );
    }

    fn cancel_job(&mut self, _: &gpui::ClickEvent, _: &mut Window, cx: &mut Context<Self>) {
        assert!(self.target_fps_num > 0, "target FPS must be positive");
        assert!(
            self.cancellation.is_some() || !self.running,
            "running job must be cancellable"
        );
        if let Some(cancellation) = &self.cancellation {
            cancellation.store(true, Ordering::Release);
            self.status = "Cancelling after current frame".to_owned();
            cx.notify();
        }
        assert!(self.target_fps_num <= 480, "target FPS must remain bounded");
        assert!(
            self.running || self.cancellation.is_none(),
            "idle job must not retain cancellation"
        );
    }

    fn apply_update(&mut self, update: JobUpdate) {
        assert!(
            (0.0..=1.0).contains(&self.progress),
            "existing progress must be bounded"
        );
        assert!(self.target_fps_num > 0, "target FPS must remain valid");
        match update {
            JobUpdate::Phase(phase) => self.status = phase.to_owned(),
            JobUpdate::Progress {
                frame_count,
                frame_count_estimate,
                processing_fps,
                progress,
                cadence_diagnostics,
            } => {
                self.frame_count = frame_count;
                self.frame_count_estimate = frame_count_estimate;
                self.processing_fps = processing_fps;
                self.progress = progress.clamp(0.0, 1.0);
                self.cadence_diagnostics = cadence_diagnostics;
            }
            JobUpdate::Completed {
                path,
                cadence_diagnostics,
            } => {
                self.running = false;
                self.cancellation = None;
                self.progress = 1.0;
                self.cadence_diagnostics = cadence_diagnostics;
                self.status = format!("Completed · {}", path.display());
            }
            JobUpdate::Cancelled => {
                self.running = false;
                self.cancellation = None;
                self.status = "Cancelled safely".to_owned();
            }
            JobUpdate::Failed(error) => {
                self.running = false;
                self.cancellation = None;
                self.status = "Job failed".to_owned();
                self.error = Some(error);
            }
        }
        assert!(
            (0.0..=1.0).contains(&self.progress),
            "updated progress must be bounded"
        );
        assert!(
            self.running || self.cancellation.is_none(),
            "finished job must release cancellation state"
        );
    }
}

impl Render for InterpolateApp {
    fn render(&mut self, window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        assert!(
            (0.0..=1.0).contains(&self.progress),
            "rendered progress must be bounded"
        );
        assert!(
            self.selected_gpu_index <= self.gpu_names.len(),
            "rendered GPU index must be bounded"
        );

        let input_name = self
            .input_path
            .as_ref()
            .and_then(|path| path.file_name())
            .and_then(|name| name.to_str())
            .unwrap_or("No source selected")
            .to_owned();
        let output_name = self
            .output_path
            .as_ref()
            .and_then(|path| path.file_name())
            .and_then(|name| name.to_str())
            .unwrap_or("Output is created after choosing a source")
            .to_owned();
        let output_directory = self
            .output_path
            .as_ref()
            .and_then(|path| path.parent())
            .map(|path| path.display().to_string())
            .unwrap_or_else(|| "—".to_owned());
        let source_details = self.metadata.as_ref().map_or_else(
            || "—".to_owned(),
            |metadata| {
                format!(
                    "{}×{}  ·  {:.3} fps  ·  {:.0}s  ·  {}",
                    metadata.width,
                    metadata.height,
                    metadata.source_fps_num as f64 / metadata.source_fps_den as f64,
                    metadata.duration_seconds,
                    metadata.pixel_format
                )
            },
        );
        let output_resolution = self.metadata.as_ref().map_or_else(
            || "Same as source".to_owned(),
            |metadata| format!("{}×{}", metadata.width, metadata.height),
        );
        let output_frame_estimate = self.metadata.as_ref().map_or(0_u64, |metadata| {
            (metadata.duration_seconds * f64::from(self.target_fps_num)).ceil() as u64
        });
        let gpu_label = self
            .gpu_names
            .get(self.selected_gpu_index)
            .cloned()
            .unwrap_or_else(|| "No Vulkan device".to_owned());
        let frame_label = if self.frame_count_estimate > 0 {
            format!(
                "{} / {} frames",
                self.frame_count, self.frame_count_estimate
            )
        } else {
            "Waiting".to_owned()
        };
        let speed_label = if self.processing_fps > 0.0 {
            format!("{:.2} fps", self.processing_fps)
        } else {
            "GPU idle".to_owned()
        };
        let eta_label = if self.processing_fps > 0.0 && self.frame_count_estimate > self.frame_count
        {
            let remaining_seconds = ((self.frame_count_estimate - self.frame_count) as f64
                / self.processing_fps)
                .ceil() as u64;
            format_remaining_time(remaining_seconds)
        } else {
            "—".to_owned()
        };
        let target_fps_valid = self
            .target_fps_input
            .parse::<u32>()
            .is_ok_and(|target_fps_num| (1..=480).contains(&target_fps_num));
        let target_fps_focused = self.target_fps_focus.is_focused(window);
        let anime_preset_selected = self.content_preset == ContentPreset::Anime;
        let preset_modified = self.scene_detection_overridden || self.use_uhd_mode_overridden;
        let preset_name = if anime_preset_selected {
            "Anime"
        } else {
            "Movie"
        };
        let preset_label = if preset_modified {
            format!("{preset_name} · Modified")
        } else {
            preset_name.to_owned()
        };
        let gpu_menu_height = px((self.gpu_names.len().clamp(1, 4) * 36) as f32);
        let can_start = !self.running
            && self.input_path.is_some()
            && !self.gpu_names.is_empty()
            && target_fps_valid;
        let status_color = if self.error.is_some() {
            ERROR_COLOR
        } else if self.running {
            ACCENT_COLOR
        } else {
            SUCCESS_COLOR
        };

        let root = div()
            .flex()
            .flex_col()
            .size_full()
            .bg(rgb(BACKGROUND_COLOR))
            .text_color(rgb(TEXT_COLOR))
            .font_family("Inter")
            .text_sm()
            .child(
                div()
                    .h(px(52.0))
                    .px_4()
                    .flex_shrink_0()
                    .flex()
                    .items_center()
                    .justify_between()
                    .border_b_1()
                    .border_color(rgb(BORDER_COLOR))
                    .child(
                        div()
                            .flex()
                            .items_center()
                            .gap_2()
                            .child(
                                div()
                                    .size(px(24.0))
                                    .rounded_sm()
                                    .border_1()
                                    .border_color(rgb(BORDER_STRONG_COLOR))
                                    .flex()
                                    .items_center()
                                    .justify_center()
                                    .text_xs()
                                    .child("I"),
                            )
                            .child("Interpolate"),
                    )
                    .child(
                        div()
                            .flex()
                            .items_center()
                            .gap_3()
                            .child(
                                div()
                                    .font_family("JetBrains Mono")
                                    .text_xs()
                                    .text_color(rgb(TEXT_SUBTLE_COLOR))
                                    .child("RIFE 4.25  ·  Vulkan"),
                            )
                            .child(if self.running {
                                div()
                                    .id("cancel-job-toolbar")
                                    .px_3()
                                    .py_2()
                                    .rounded_md()
                                    .cursor_pointer()
                                    .border_1()
                                    .border_color(rgb(BORDER_STRONG_COLOR))
                                    .hover(|style| style.bg(rgb(PANEL_HOVER_COLOR)))
                                    .on_click(cx.listener(Self::cancel_job))
                                    .child("Cancel")
                            } else {
                                div()
                                    .id("start-job-toolbar")
                                    .px_4()
                                    .py_2()
                                    .rounded_md()
                                    .cursor_pointer()
                                    .bg(rgb(if can_start {
                                        PRIMARY_BUTTON_COLOR
                                    } else {
                                        PANEL_HOVER_COLOR
                                    }))
                                    .text_color(rgb(if can_start {
                                        PRIMARY_BUTTON_TEXT_COLOR
                                    } else {
                                        TEXT_SUBTLE_COLOR
                                    }))
                                    .border_1()
                                    .border_color(rgb(if can_start {
                                        PRIMARY_BUTTON_COLOR
                                    } else {
                                        BORDER_COLOR
                                    }))
                                    .hover(|style| {
                                        if can_start {
                                            style.bg(rgb(PRIMARY_BUTTON_HOVER_COLOR))
                                        } else {
                                            style
                                        }
                                    })
                                    .on_click(cx.listener(Self::start_job))
                                    .child("Start encode")
                            }),
                    ),
            )
            .child(
                div()
                    .px_5()
                    .py_4()
                    .flex_shrink_0()
                    .flex()
                    .items_center()
                    .justify_between()
                    .border_b_1()
                    .border_color(rgb(BORDER_COLOR))
                    .bg(rgb(SIDEBAR_COLOR))
                    .child(
                        div()
                            .child(
                                div()
                                    .flex()
                                    .items_center()
                                    .gap_2()
                                    .child(
                                        div()
                                            .text_xs()
                                            .text_color(rgb(TEXT_SUBTLE_COLOR))
                                            .child("SOURCE"),
                                    )
                                    .child(
                                        div()
                                            .whitespace_nowrap()
                                            .overflow_hidden()
                                            .text_ellipsis()
                                            .child(input_name),
                                    ),
                            )
                            .child(
                                div()
                                    .mt_1()
                                    .font_family("JetBrains Mono")
                                    .text_xs()
                                    .text_color(rgb(TEXT_MUTED_COLOR))
                                    .child(source_details),
                            ),
                    )
                    .child(
                        div()
                            .id("choose-input-source")
                            .ml_5()
                            .px_3()
                            .py_2()
                            .flex_shrink_0()
                            .rounded_md()
                            .border_1()
                            .border_color(rgb(BORDER_COLOR))
                            .bg(rgb(PANEL_COLOR))
                            .cursor_pointer()
                            .hover(|style| {
                                style
                                    .border_color(rgb(BORDER_STRONG_COLOR))
                                    .bg(rgb(PANEL_HOVER_COLOR))
                            })
                            .on_click(cx.listener(Self::select_input))
                            .child("Source…"),
                    ),
            )
            .child(
                div()
                    .flex_1()
                    .overflow_hidden()
                    .p_5()
                    .flex()
                    .flex_col()
                    .gap_4()
                    .child(
                        div()
                            .flex_1()
                            .flex()
                            .gap_4()
                            .overflow_hidden()
                            .child(
                                div()
                                    .flex_1()
                                    .flex()
                                    .flex_col()
                                    .child(
                                        div()
                                            .mb_3()
                                            .text_xs()
                                            .text_color(rgb(TEXT_SUBTLE_COLOR))
                                            .child("INTERPOLATION"),
                                    )
                                    .child(
                                        div()
                                            .relative()
                                            .rounded_lg()
                                            .border_1()
                                            .border_color(rgb(BORDER_COLOR))
                                            .child(
                                                div()
                                                    .h(px(64.0))
                                                    .px_4()
                                                    .flex()
                                                    .items_center()
                                                    .justify_between()
                                                    .child(
                                                        div().child("Content preset").child(
                                                            div()
                                                                .mt_1()
                                                                .text_xs()
                                                                .text_color(rgb(TEXT_MUTED_COLOR))
                                                                .child("Profile tuning follows clip tests"),
                                                        ),
                                                    )
                                                    .child(
                                                        div()
                                                            .id("preset-selector")
                                                            .w(px(270.0))
                                                            .px_3()
                                                            .py_2()
                                                            .rounded_md()
                                                            .cursor_pointer()
                                                            .border_1()
                                                            .border_color(rgb(if self.preset_menu_open {
                                                                ACCENT_COLOR
                                                            } else {
                                                                BORDER_COLOR
                                                            }))
                                                            .bg(rgb(PANEL_COLOR))
                                                            .flex()
                                                            .items_center()
                                                            .justify_between()
                                                            .font_family("JetBrains Mono")
                                                            .text_xs()
                                                            .text_color(rgb(TEXT_MUTED_COLOR))
                                                            .hover(|style| {
                                                                style.border_color(rgb(BORDER_STRONG_COLOR))
                                                            })
                                                            .on_click(cx.listener(Self::toggle_preset_menu))
                                                            .child(preset_label.clone())
                                                            .child(
                                                                div()
                                                                    .ml_2()
                                                                    .text_color(rgb(TEXT_SUBTLE_COLOR))
                                                                    .child(if self.preset_menu_open {
                                                                        "⌃"
                                                                    } else {
                                                                        "⌄"
                                                                    }),
                                                            ),
                                                    ),
                                            )
                                            .child(
                                                div()
                                                    .h(px(64.0))
                                                    .px_4()
                                                    .flex()
                                                    .items_center()
                                                    .justify_between()
                                                    .border_t_1()
                                                    .border_color(rgb(BORDER_COLOR))
                                                    .child(
                                                        div().child("Target frame rate").child(
                                                            div()
                                                                .mt_1()
                                                                .text_xs()
                                                                .text_color(rgb(TEXT_MUTED_COLOR))
                                                                .child("Enter 1–480 frames per second"),
                                                        ),
                                                    )
                                                    .child(
                                                        div()
                                                            .id("target-fps-input")
                                                            .track_focus(&self.target_fps_focus)
                                                            .on_key_down(cx.listener(Self::edit_target_fps))
                                                            .on_click(cx.listener(Self::focus_target_fps))
                                                            .w(px(112.0))
                                                            .px_3()
                                                            .py_2()
                                                            .flex()
                                                            .items_center()
                                                            .justify_between()
                                                            .rounded_md()
                                                            .cursor_text()
                                                            .border_1()
                                                            .border_color(rgb(if !target_fps_valid {
                                                                ERROR_COLOR
                                                            } else if target_fps_focused {
                                                                ACCENT_COLOR
                                                            } else {
                                                                BORDER_COLOR
                                                            }))
                                                            .bg(rgb(PANEL_COLOR))
                                                            .font_family("JetBrains Mono")
                                                            .child(self.target_fps_input.clone())
                                                            .child(
                                                                div()
                                                                    .ml_2()
                                                                    .text_color(rgb(TEXT_SUBTLE_COLOR))
                                                                    .child("FPS"),
                                                            ),
                                                    ),
                                            )
                                            .child(
                                                div()
                                                    .h(px(64.0))
                                                    .px_4()
                                                    .flex()
                                                    .items_center()
                                                    .justify_between()
                                                    .border_t_1()
                                                    .border_color(rgb(BORDER_COLOR))
                                                    .child(
                                                        div().child("Compute device").child(
                                                            div()
                                                                .mt_1()
                                                                .text_xs()
                                                                .text_color(rgb(TEXT_MUTED_COLOR))
                                                                .child("Vulkan inference device"),
                                                        ),
                                                    )
                                                    .child(
                                                        div()
                                                            .id("gpu-selector")
                                                            .w(px(270.0))
                                                            .px_3()
                                                            .py_2()
                                                            .rounded_md()
                                                            .cursor_pointer()
                                                            .border_1()
                                                            .border_color(rgb(if self.gpu_menu_open {
                                                                ACCENT_COLOR
                                                            } else {
                                                                BORDER_COLOR
                                                            }))
                                                            .bg(rgb(PANEL_COLOR))
                                                            .flex()
                                                            .items_center()
                                                            .justify_between()
                                                            .font_family("JetBrains Mono")
                                                            .text_xs()
                                                            .text_color(rgb(TEXT_MUTED_COLOR))
                                                            .hover(|style| style.border_color(rgb(BORDER_STRONG_COLOR)))
                                                            .on_click(cx.listener(Self::toggle_gpu_menu))
                                                            .child(
                                                                div()
                                                                    .flex_1()
                                                                    .whitespace_nowrap()
                                                                    .overflow_hidden()
                                                                    .text_ellipsis()
                                                                    .child(gpu_label),
                                                            )
                                                            .child(
                                                                div()
                                                                    .ml_2()
                                                                    .text_color(rgb(TEXT_SUBTLE_COLOR))
                                                                    .child(if self.gpu_menu_open { "⌃" } else { "⌄" }),
                                                            ),
                                                    ),
                                            )
                                            .child(
                                                div()
                                                    .h(px(64.0))
                                                    .px_4()
                                                    .flex()
                                                    .items_center()
                                                    .justify_between()
                                                    .border_t_1()
                                                    .border_color(rgb(BORDER_COLOR))
                                                    .child(
                                                        div().child("Scene change protection").child(
                                                            div()
                                                                .mt_1()
                                                                .text_xs()
                                                                .text_color(rgb(TEXT_MUTED_COLOR))
                                                                .child(if anime_preset_selected {
                                                                    "Required for cadence safety"
                                                                } else {
                                                                    "Duplicate at hard cuts"
                                                                }),
                                                        ),
                                                    )
                                                    .child(
                                                        div()
                                                            .id("scene-detection")
                                                            .w(px(36.0))
                                                            .h(px(20.0))
                                                            .p(px(2.0))
                                                            .flex()
                                                            .justify_end()
                                                            .when(!self.scene_detection, |element| element.justify_start())
                                                            .items_center()
                                                            .rounded_full()
                                                            .bg(rgb(if self.scene_detection {
                                                                ACCENT_COLOR
                                                            } else {
                                                                BORDER_STRONG_COLOR
                                                            }))
                                                            .when(!anime_preset_selected, |element| {
                                                                element
                                                                    .cursor_pointer()
                                                                    .on_click(cx.listener(
                                                                        Self::toggle_scene_detection,
                                                                    ))
                                                            })
                                                            .child(div().size(px(16.0)).rounded_full().bg(rgb(TEXT_COLOR))),
                                                    ),
                                            )
                                            .child(
                                                div()
                                                    .h(px(64.0))
                                                    .px_4()
                                                    .flex()
                                                    .items_center()
                                                    .justify_between()
                                                    .border_t_1()
                                                    .border_color(rgb(BORDER_COLOR))
                                                    .child(
                                                        div().child("Half-scale UHD flow").child(
                                                            div()
                                                                .mt_1()
                                                                .text_xs()
                                                                .text_color(rgb(TEXT_MUTED_COLOR))
                                                                .child(if anime_preset_selected {
                                                                    "Automatic for 4K; click to override"
                                                                } else {
                                                                    "Reduce memory for 4K sources"
                                                                }),
                                                        ),
                                                    )
                                                    .child(
                                                        div()
                                                            .id("uhd-mode")
                                                            .w(px(36.0))
                                                            .h(px(20.0))
                                                            .p(px(2.0))
                                                            .flex()
                                                            .justify_end()
                                                            .when(!self.use_uhd_mode, |element| element.justify_start())
                                                            .items_center()
                                                            .rounded_full()
                                                            .cursor_pointer()
                                                            .bg(rgb(if self.use_uhd_mode {
                                                                ACCENT_COLOR
                                                            } else {
                                                                BORDER_STRONG_COLOR
                                                            }))
                                                            .on_click(cx.listener(Self::toggle_uhd_mode))
                                                            .child(div().size(px(16.0)).rounded_full().bg(rgb(TEXT_COLOR))),
                                                    ),
                                            )
                                            .when(self.preset_menu_open, |element| {
                                                element.child(
                                                    div()
                                                        .id("preset-options")
                                                        .absolute()
                                                        .top(px(52.0))
                                                        .right(px(16.0))
                                                        .w(px(270.0))
                                                        .rounded_md()
                                                        .border_1()
                                                        .border_color(rgb(BORDER_STRONG_COLOR))
                                                        .bg(rgb(PANEL_COLOR))
                                                        .shadow_lg()
                                                        .children(
                                                            [
                                                                (ContentPreset::Movie, "Movie"),
                                                                (ContentPreset::Anime, "Anime"),
                                                            ]
                                                            .into_iter()
                                                            .enumerate()
                                                            .map(|(preset_index, (preset, label))| {
                                                                let selected = preset == self.content_preset;
                                                                div()
                                                                    .id(("preset-option", preset_index))
                                                                    .h(px(36.0))
                                                                    .px_3()
                                                                    .flex()
                                                                    .items_center()
                                                                    .justify_between()
                                                                    .cursor_pointer()
                                                                    .border_b_1()
                                                                    .border_color(rgb(BORDER_COLOR))
                                                                    .bg(rgb(if selected {
                                                                        SELECTED_COLOR
                                                                    } else {
                                                                        PANEL_COLOR
                                                                    }))
                                                                    .hover(|style| {
                                                                        style.bg(rgb(PANEL_HOVER_COLOR))
                                                                    })
                                                                    .on_click(cx.listener(
                                                                        move |app, _, _, cx| {
                                                                            app.select_content_preset(
                                                                                preset, cx,
                                                                            )
                                                                        },
                                                                    ))
                                                                    .child(
                                                                        div()
                                                                            .font_family(
                                                                                "JetBrains Mono",
                                                                            )
                                                                            .text_xs()
                                                                            .text_color(rgb(
                                                                                if selected {
                                                                                    TEXT_COLOR
                                                                                } else {
                                                                                    TEXT_MUTED_COLOR
                                                                                },
                                                                            ))
                                                                            .child(label),
                                                                    )
                                                                    .child(if selected { "✓" } else { "" })
                                                            }),
                                                        ),
                                                )
                                            })
                                            .when(self.gpu_menu_open, |element| {
                                                element.child(
                                                    div()
                                                        .id("gpu-options")
                                                        .absolute()
                                                        .top(px(180.0))
                                                        .right(px(16.0))
                                                        .w(px(270.0))
                                                        .h(gpu_menu_height)
                                                        .overflow_y_scroll()
                                                        .rounded_md()
                                                        .border_1()
                                                        .border_color(rgb(BORDER_STRONG_COLOR))
                                                        .bg(rgb(PANEL_COLOR))
                                                        .shadow_lg()
                                                        .children(self.gpu_names.iter().enumerate().map(
                                                            |(gpu_index, gpu_name)| {
                                                                let selected = gpu_index == self.selected_gpu_index;
                                                                div()
                                                                    .id(("gpu-option", gpu_index))
                                                                    .h(px(36.0))
                                                                    .px_3()
                                                                    .flex()
                                                                    .items_center()
                                                                    .justify_between()
                                                                    .cursor_pointer()
                                                                    .border_b_1()
                                                                    .border_color(rgb(BORDER_COLOR))
                                                                    .bg(rgb(if selected {
                                                                        SELECTED_COLOR
                                                                    } else {
                                                                        PANEL_COLOR
                                                                    }))
                                                                    .hover(|style| style.bg(rgb(PANEL_HOVER_COLOR)))
                                                                    .on_click(cx.listener(move |app, _, _, cx| {
                                                                        app.select_gpu(gpu_index, cx)
                                                                    }))
                                                                    .child(
                                                                        div()
                                                                            .flex_1()
                                                                            .font_family("JetBrains Mono")
                                                                            .text_xs()
                                                                            .text_color(rgb(if selected {
                                                                                TEXT_COLOR
                                                                            } else {
                                                                                TEXT_MUTED_COLOR
                                                                            }))
                                                                            .whitespace_nowrap()
                                                                            .overflow_hidden()
                                                                            .text_ellipsis()
                                                                            .child(gpu_name.clone()),
                                                                    )
                                                                    .child(if selected { "✓" } else { "" })
                                                            },
                                                        )),
                                                )
                                            }),
                                    ),
                            )
                            .child(
                                div()
                                    .w(px(260.0))
                                    .flex_shrink_0()
                                    .flex()
                                    .flex_col()
                                    .child(
                                        div()
                                            .mb_3()
                                            .text_xs()
                                            .text_color(rgb(TEXT_SUBTLE_COLOR))
                                            .child("ESTIMATED OUTPUT"),
                                    )
                                    .child(
                                        div()
                                            .flex_1()
                                            .rounded_lg()
                                            .border_1()
                                            .border_color(rgb(BORDER_COLOR))
                                            .bg(rgb(PANEL_COLOR))
                                            .p_4()
                                            .child(
                                                div()
                                                    .font_family("JetBrains Mono")
                                                    .text_3xl()
                                                    .child(if target_fps_valid {
                                                        format!("{} fps", self.target_fps_input)
                                                    } else {
                                                        "Invalid fps".to_owned()
                                                    }),
                                            )
                                            .child(
                                                div()
                                                    .mt_1()
                                                    .text_xs()
                                                    .text_color(rgb(TEXT_MUTED_COLOR))
                                                    .child(if anime_preset_selected {
                                                        "RIFE 4.25 · cadence protected"
                                                    } else {
                                                        "RIFE 4.25 standard"
                                                    }),
                                            )
                                            .child(
                                                div()
                                                    .mt_5()
                                                    .pt_4()
                                                    .border_t_1()
                                                    .border_color(rgb(BORDER_COLOR))
                                                    .flex()
                                                    .flex_col()
                                                    .gap_3()
                                                    .child(
                                                        div()
                                                            .flex()
                                                            .justify_between()
                                                            .child(
                                                                div()
                                                                    .text_color(rgb(
                                                                        TEXT_MUTED_COLOR,
                                                                    ))
                                                                    .child("Resolution"),
                                                            )
                                                            .child(
                                                                div()
                                                                    .font_family("JetBrains Mono")
                                                                    .text_xs()
                                                                    .child(output_resolution),
                                                            ),
                                                    )
                                                    .child(
                                                        div()
                                                            .flex()
                                                            .justify_between()
                                                            .child(
                                                                div()
                                                                    .text_color(rgb(
                                                                        TEXT_MUTED_COLOR,
                                                                    ))
                                                                    .child("Container"),
                                                            )
                                                            .child(
                                                                div()
                                                                    .font_family("JetBrains Mono")
                                                                    .text_xs()
                                                                    .child("MKV"),
                                                            ),
                                                    )
                                                    .child(
                                                        div()
                                                            .flex()
                                                            .justify_between()
                                                            .child(
                                                                div()
                                                                    .text_color(rgb(
                                                                        TEXT_MUTED_COLOR,
                                                                    ))
                                                                    .child("Video"),
                                                            )
                                                            .child(
                                                                div()
                                                                    .font_family("JetBrains Mono")
                                                                    .text_xs()
                                                                    .child("H.264"),
                                                            ),
                                                    )
                                                    .child(
                                                        div()
                                                            .flex()
                                                            .justify_between()
                                                            .child(
                                                                div()
                                                                    .text_color(rgb(
                                                                        TEXT_MUTED_COLOR,
                                                                    ))
                                                                    .child("Audio"),
                                                            )
                                                            .child(
                                                                div()
                                                                    .font_family("JetBrains Mono")
                                                                    .text_xs()
                                                                    .child("Copy"),
                                                            ),
                                                    )
                                                    .child(
                                                        div()
                                                            .flex()
                                                            .justify_between()
                                                            .child(
                                                                div()
                                                                    .text_color(rgb(
                                                                        TEXT_MUTED_COLOR,
                                                                    ))
                                                                    .child("Frames"),
                                                            )
                                                            .child(
                                                                div()
                                                                    .font_family("JetBrains Mono")
                                                                    .text_xs()
                                                                    .child(
                                                                        if output_frame_estimate > 0
                                                                        {
                                                                            output_frame_estimate
                                                                                .to_string()
                                                                        } else {
                                                                            "—".to_owned()
                                                                        },
                                                                    ),
                                                            ),
                                                    ),
                                            )
                                            .child(
                                                div()
                                                    .mt_5()
                                                    .pt_4()
                                                    .border_t_1()
                                                    .border_color(rgb(BORDER_COLOR))
                                                    .text_xs()
                                                    .text_color(rgb(TEXT_SUBTLE_COLOR))
                                                    .child(format!("{preset_label} content preset"))
                                                    .when(anime_preset_selected, |element| {
                                                        element.child(div().mt_1().child(format!(
                                                            "{} held frames · {} smoothed runs",
                                                            self.cadence_diagnostics
                                                                .duplicate_frame_count,
                                                            self.cadence_diagnostics
                                                                .cadence_run_count
                                                        )))
                                                    })
                                                    .child(div().mt_1().child(if self.scene_detection {
                                                        "Scene cuts protected"
                                                    } else {
                                                        "Scene protection disabled"
                                                    }))
                                                    .child(div().mt_1().child(
                                                        if self.use_uhd_mode {
                                                            "Half-scale optical flow"
                                                        } else {
                                                            "Full-scale optical flow"
                                                        },
                                                    )),
                                            ),
                                    ),
                            ),
                    )
                    .child(
                        div()
                            .flex_shrink_0()
                            .child(
                                div()
                                    .mb_3()
                                    .text_xs()
                                    .text_color(rgb(TEXT_SUBTLE_COLOR))
                                    .child("DESTINATION"),
                            )
                            .child(
                                div()
                                    .h(px(62.0))
                                    .px_4()
                                    .rounded_lg()
                                    .border_1()
                                    .border_color(rgb(BORDER_COLOR))
                                    .flex()
                                    .items_center()
                                    .justify_between()
                                    .child(
                                        div()
                                            .flex_1()
                                            .child(
                                                div()
                                                    .whitespace_nowrap()
                                                    .overflow_hidden()
                                                    .text_ellipsis()
                                                    .child(output_name),
                                            )
                                            .child(
                                                div()
                                                    .mt_1()
                                                    .font_family("JetBrains Mono")
                                                    .text_xs()
                                                    .text_color(rgb(TEXT_MUTED_COLOR))
                                                    .whitespace_nowrap()
                                                    .overflow_hidden()
                                                    .text_ellipsis()
                                                    .child(output_directory),
                                            ),
                                    )
                                    .child(
                                        div()
                                            .id("choose-output")
                                            .ml_4()
                                            .px_3()
                                            .py_2()
                                            .rounded_md()
                                            .border_1()
                                            .border_color(rgb(BORDER_COLOR))
                                            .bg(rgb(PANEL_COLOR))
                                            .cursor_pointer()
                                            .hover(|style| {
                                                style
                                                    .border_color(rgb(BORDER_STRONG_COLOR))
                                                    .bg(rgb(PANEL_HOVER_COLOR))
                                            })
                                            .on_click(cx.listener(Self::select_output))
                                            .child("Browse…"),
                                    ),
                            ),
                    ),
            )
            .child(
                div()
                    .h(px(68.0))
                    .flex_shrink_0()
                    .border_t_1()
                    .border_color(rgb(BORDER_COLOR))
                    .child(
                        div().h(px(2.0)).w_full().bg(rgb(BORDER_COLOR)).child(
                            div()
                                .h_full()
                                .w(relative(self.progress))
                                .bg(rgb(ACCENT_COLOR)),
                        ),
                    )
                    .child(
                        div()
                            .h(px(65.0))
                            .px_5()
                            .flex()
                            .items_center()
                            .justify_between()
                            .child(
                                div()
                                    .flex()
                                    .items_center()
                                    .gap_2()
                                    .child(div().size(px(6.0)).rounded_full().bg(rgb(status_color)))
                                    .child(
                                        div()
                                            .whitespace_nowrap()
                                            .overflow_hidden()
                                            .text_ellipsis()
                                            .child(self.status.clone()),
                                    ),
                            )
                            .child(
                                div()
                                    .ml_5()
                                    .flex_shrink_0()
                                    .flex()
                                    .gap_5()
                                    .font_family("JetBrains Mono")
                                    .text_xs()
                                    .text_color(rgb(TEXT_MUTED_COLOR))
                                    .child(frame_label)
                                    .child(speed_label)
                                    .child(eta_label),
                            ),
                    )
                    .when_some(self.error.clone(), |element, error| {
                        element.child(
                            div()
                                .absolute()
                                .bottom(px(68.0))
                                .left(px(0.0))
                                .right(px(0.0))
                                .px_5()
                                .py_2()
                                .bg(rgb(PANEL_COLOR))
                                .border_t_1()
                                .border_color(rgb(ERROR_COLOR))
                                .text_xs()
                                .text_color(rgb(ERROR_COLOR))
                                .child(error),
                        )
                    }),
            );
        assert!(
            WINDOW_WIDTH_PIXELS.is_finite(),
            "window width must remain finite"
        );
        assert!(
            WINDOW_HEIGHT_PIXELS.is_finite(),
            "window height must remain finite"
        );
        root
    }
}

impl Drop for InterpolateApp {
    fn drop(&mut self) {
        assert!(
            self.target_fps_num > 0,
            "target FPS must remain valid during shutdown"
        );
        assert!(
            self.selected_gpu_index <= self.gpu_names.len(),
            "GPU index must remain valid during shutdown"
        );
        if let Some(cancellation) = &self.cancellation {
            cancellation.store(true, Ordering::Release);
        }
        shutdown_backend();
        assert!(
            self.target_fps_num <= 480,
            "target FPS must remain bounded during shutdown"
        );
        assert!(
            (0.0..=1.0).contains(&self.progress),
            "progress must remain bounded during shutdown"
        );
    }
}

fn main() {
    assert!(WINDOW_WIDTH_PIXELS > 0.0, "window width must be positive");
    assert!(WINDOW_HEIGHT_PIXELS > 0.0, "window height must be positive");
    Application::new().run(|context: &mut App| {
        let window_size = size(px(WINDOW_WIDTH_PIXELS), px(WINDOW_HEIGHT_PIXELS));
        let window_bounds = Bounds::centered(None, window_size, context);
        let window_result = context.open_window(
            WindowOptions {
                window_bounds: Some(WindowBounds::Windowed(window_bounds)),
                ..WindowOptions::default()
            },
            |_window, context| context.new(InterpolateApp::new),
        );
        if let Err(error) = window_result {
            eprintln!("failed to open the GPUI window: {error:#}");
            context.quit();
            return;
        }
        context.activate(true);
    });
    assert!(
        WINDOW_WIDTH_PIXELS.is_finite(),
        "window width must remain finite"
    );
    assert!(
        WINDOW_HEIGHT_PIXELS.is_finite(),
        "window height must remain finite"
    );
}

#[cfg(test)]
mod tests {
    use super::{ContentPreset, VideoMetadata, format_remaining_time, use_uhd_mode_default};

    #[test]
    fn anime_enables_uhd_for_4k_sources() {
        let metadata = VideoMetadata {
            width: 3_840,
            height: 2_160,
            source_fps_num: 24,
            source_fps_den: 1,
            duration_seconds: 1.0,
            pixel_format: "yuv420p".to_owned(),
        };
        assert!(use_uhd_mode_default(ContentPreset::Anime, Some(&metadata)));
        assert!(!use_uhd_mode_default(ContentPreset::Movie, Some(&metadata)));
        let portrait_metadata = VideoMetadata {
            width: metadata.height,
            height: metadata.width,
            ..metadata
        };
        assert!(use_uhd_mode_default(
            ContentPreset::Anime,
            Some(&portrait_metadata)
        ));
    }

    #[test]
    fn anime_keeps_full_scale_flow_below_4k() {
        let mut metadata = VideoMetadata {
            width: 1_920,
            height: 1_080,
            source_fps_num: 24,
            source_fps_den: 1,
            duration_seconds: 1.0,
            pixel_format: "yuv420p".to_owned(),
        };
        assert!(!use_uhd_mode_default(ContentPreset::Anime, Some(&metadata)));
        metadata.width = 3_840;
        assert!(!use_uhd_mode_default(ContentPreset::Anime, Some(&metadata)));
        assert!(!use_uhd_mode_default(ContentPreset::Anime, None));
    }

    #[test]
    fn remaining_time_uses_hours_for_long_jobs() {
        assert_eq!(format_remaining_time(11_558), "3:12:38 remaining");
        assert_eq!(format_remaining_time(18_758), "5:12:38 remaining");
    }

    #[test]
    fn remaining_time_keeps_minutes_for_short_jobs() {
        assert_eq!(format_remaining_time(758), "12:38 remaining");
        assert_eq!(format_remaining_time(59), "0:59 remaining");
    }
}
