use crate::backend::{
    InferenceBackend, cuda_inference_status, gpu_names, shutdown as shutdown_backend,
};
use crate::pipeline::{
    CadenceDiagnostics, ContentPreset, EncoderPreset, H264Profile, JobConfiguration, JobUpdate,
    PerformanceDiagnostics, VideoEncoder, VideoMetadata, available_video_encoders,
    default_output_path, probe_video, run_job,
};
use crate::tray::{TrayCommand, TrayController};
use gpui::{
    AnyWindowHandle, App, Application, AssetSource, Bounds, Context, FocusHandle, KeyDownEvent,
    PathPromptOptions, Render, SharedString, StatefulInteractiveElement, Timer, Window,
    WindowBounds, WindowOptions, div, prelude::*, px, relative, rgb, size, svg,
};
use std::{
    borrow::Cow,
    path::{Path, PathBuf},
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
        mpsc::{Receiver, TryRecvError, sync_channel},
    },
    thread::{self, JoinHandle},
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
const TRAY_POLL_INTERVAL: Duration = Duration::from_millis(125);
const TRAY_POLL_COUNT_MAX: usize = 252_288_000;
const TRAY_COMMAND_DRAIN_COUNT_MAX: usize = 4;
const SECONDS_PER_MINUTE: u64 = 60;
const MINUTES_PER_HOUR: u64 = 60;
const SECONDS_PER_HOUR: u64 = SECONDS_PER_MINUTE * MINUTES_PER_HOUR;
const ENCODING_QUALITY_MAX: u8 = 51;
const ENCODER_THREAD_COUNT_MAX: u8 = 16;
const X264_ENCODER_PRESETS: &[EncoderPreset] = &[
    EncoderPreset::X264VeryFast,
    EncoderPreset::X264Faster,
    EncoderPreset::X264Fast,
    EncoderPreset::X264Medium,
    EncoderPreset::X264Slow,
    EncoderPreset::X264Slower,
    EncoderPreset::X264VerySlow,
    EncoderPreset::X264Placebo,
];
const H264_PROFILES: &[H264Profile] = &[H264Profile::Auto, H264Profile::Main, H264Profile::High];
const NVIDIA_ENCODER_PRESETS: &[EncoderPreset] = &[
    EncoderPreset::NvidiaP1,
    EncoderPreset::NvidiaP2,
    EncoderPreset::NvidiaP3,
    EncoderPreset::NvidiaP4,
    EncoderPreset::NvidiaP5,
    EncoderPreset::NvidiaP6,
    EncoderPreset::NvidiaP7,
];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum MainPageTab {
    Interpolation,
    Hardware,
    Encoding,
    Media,
}

struct SettingTooltip {
    description: &'static str,
}

impl Render for SettingTooltip {
    fn render(&mut self, _window: &mut Window, _cx: &mut Context<Self>) -> impl IntoElement {
        assert!(
            !self.description.is_empty(),
            "tooltip description must not be empty"
        );
        assert!(
            self.description.len() <= 512,
            "tooltip description must remain bounded"
        );
        div()
            .w(px(280.0))
            .px_3()
            .py_2()
            .rounded_md()
            .border_1()
            .border_color(rgb(BORDER_STRONG_COLOR))
            .bg(rgb(PANEL_COLOR))
            .text_xs()
            .text_color(rgb(TEXT_COLOR))
            .child(self.description)
    }
}

fn setting_tooltip_icon(description: &'static str) -> impl IntoElement {
    assert!(
        !description.is_empty(),
        "tooltip description must not be empty"
    );
    assert!(
        description.len() <= 512,
        "tooltip description must remain bounded"
    );
    div()
        .id(description)
        .ml_1()
        .text_color(rgb(TEXT_SUBTLE_COLOR))
        .child("ⓘ")
        .tooltip(move |_, cx| cx.new(|_| SettingTooltip { description }).into())
}

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

fn video_encoder_label(encoder: VideoEncoder) -> &'static str {
    assert!(matches!(
        encoder,
        VideoEncoder::Automatic | VideoEncoder::SoftwareH264 | VideoEncoder::NvidiaH264
    ));
    let label = match encoder {
        VideoEncoder::Automatic => "Auto · Best available H.264",
        VideoEncoder::SoftwareH264 => "H.264 · CPU (x264)",
        VideoEncoder::NvidiaH264 => "H.264 · NVIDIA NVENC",
    };
    assert!(!label.is_empty(), "encoder label must not be empty");
    label
}

fn encoder_preset_label(preset: EncoderPreset) -> &'static str {
    let label = match preset {
        EncoderPreset::X264VeryFast => "x264 · Very Fast",
        EncoderPreset::X264Faster => "x264 · Faster",
        EncoderPreset::X264Fast => "x264 · Fast",
        EncoderPreset::X264Medium => "x264 · Medium",
        EncoderPreset::X264Slow => "x264 · Slow",
        EncoderPreset::X264Slower => "x264 · Slower",
        EncoderPreset::X264VerySlow => "x264 · Very Slow",
        EncoderPreset::X264Placebo => "x264 · Placebo",
        EncoderPreset::NvidiaP1 => "NVENC · P1",
        EncoderPreset::NvidiaP2 => "NVENC · P2",
        EncoderPreset::NvidiaP3 => "NVENC · P3",
        EncoderPreset::NvidiaP4 => "NVENC · P4",
        EncoderPreset::NvidiaP5 => "NVENC · P5",
        EncoderPreset::NvidiaP6 => "NVENC · P6",
        EncoderPreset::NvidiaP7 => "NVENC · P7",
    };
    assert!(!label.is_empty(), "encoder preset label must not be empty");
    assert!(label.len() < 32, "encoder preset label must remain bounded");
    label
}

fn encoder_preset_options(encoder: VideoEncoder) -> &'static [EncoderPreset] {
    assert!(matches!(
        encoder,
        VideoEncoder::SoftwareH264 | VideoEncoder::NvidiaH264
    ));
    let options = match encoder {
        VideoEncoder::SoftwareH264 => X264_ENCODER_PRESETS,
        VideoEncoder::NvidiaH264 => NVIDIA_ENCODER_PRESETS,
        VideoEncoder::Automatic => unreachable!("automatic encoder must be resolved"),
    };
    assert!(
        !options.is_empty(),
        "encoder preset options must not be empty"
    );
    assert!(
        options.len() <= 9,
        "encoder preset options must remain bounded"
    );
    options
}

fn h264_profile_label(profile: H264Profile) -> &'static str {
    let label = match profile {
        H264Profile::Auto => "Auto",
        H264Profile::Main => "Main",
        H264Profile::High => "High",
    };
    assert!(!label.is_empty(), "H.264 profile label must not be empty");
    assert!(label.len() < 16, "H.264 profile label must remain bounded");
    label
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
const HEADER_NAV_ICON_SIZE_PIXELS: f32 = 14.0;
const HEADER_NAV_ICON_PATH_PAUSE: &str = "icons/pause.svg";
const HEADER_NAV_ICON_PATH_SETTINGS: &str = "icons/settings.svg";

struct BundledAssets;

impl AssetSource for BundledAssets {
    fn load(&self, path: &str) -> gpui::Result<Option<Cow<'static, [u8]>>> {
        assert!(!path.is_empty(), "asset path must not be empty");
        assert!(path.len() < 128, "asset path must remain bounded");
        let bytes = match path {
            HEADER_NAV_ICON_PATH_PAUSE => Some(&include_bytes!("../assets/icons/pause.svg")[..]),
            HEADER_NAV_ICON_PATH_SETTINGS => {
                Some(&include_bytes!("../assets/icons/settings.svg")[..])
            }
            _ => None,
        };
        Ok(bytes.map(Cow::Borrowed))
    }

    fn list(&self, path: &str) -> gpui::Result<Vec<SharedString>> {
        assert!(path.len() < 128, "asset directory path must remain bounded");
        let icons = if path.is_empty() || path == "icons" || path == "icons/" {
            vec![
                SharedString::from(HEADER_NAV_ICON_PATH_PAUSE),
                SharedString::from(HEADER_NAV_ICON_PATH_SETTINGS),
            ]
        } else {
            Vec::new()
        };
        assert!(icons.len() <= 2, "bundled icon listing must remain bounded");
        Ok(icons)
    }
}

fn header_nav_icon(path: &'static str, color: u32) -> gpui::Svg {
    assert!(!path.is_empty(), "header icon path must not be empty");
    assert!(path.len() < 128, "header icon path must remain bounded");
    svg()
        .path(path)
        .size(px(HEADER_NAV_ICON_SIZE_PIXELS))
        .flex_shrink_0()
        .text_color(rgb(color))
}

struct InterpolateApp {
    input_path: Option<PathBuf>,
    output_path: Option<PathBuf>,
    metadata: Option<VideoMetadata>,
    gpu_names: Vec<String>,
    selected_gpu_index: usize,
    video_encoders: Vec<VideoEncoder>,
    selected_video_encoder: usize,
    use_nvdec: bool,
    cuda_inference_available: bool,
    cuda_inference_unavailable_reason: String,
    cuda_inference_enabled: bool,
    inference_backend_menu_open: bool,
    gpu_menu_open: bool,
    video_encoder_menu_open: bool,
    preset_menu_open: bool,
    settings_page: bool,
    main_page_tab: MainPageTab,
    target_fps_num: u32,
    target_fps_input: String,
    target_fps_replace_on_type: bool,
    target_fps_focus: FocusHandle,
    encoding_quality: u8,
    encoding_quality_input: String,
    encoding_quality_replace_on_type: bool,
    encoding_quality_focus: FocusHandle,
    encoder_preset: EncoderPreset,
    encoder_preset_menu_open: bool,
    h264_profile: H264Profile,
    h264_profile_menu_open: bool,
    encoder_thread_count: u8,
    encoder_thread_count_input: String,
    encoder_thread_count_replace_on_type: bool,
    encoder_thread_count_focus: FocusHandle,
    preserve_audio: bool,
    preserve_subtitles: bool,
    preserve_metadata: bool,
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
    performance: PerformanceDiagnostics,
    cancellation: Option<Arc<AtomicBool>>,
    worker: Option<JoinHandle<()>>,
    tray_controller: Option<TrayController>,
    tray_unavailable_reason: Option<String>,
    keep_running_background: bool,
    quit_after_job: bool,
}

impl InterpolateApp {
    fn new(
        cx: &mut Context<Self>,
        tray_controller: Option<TrayController>,
        tray_commands: Option<Receiver<TrayCommand>>,
        window_handle: AnyWindowHandle,
        tray_error: Option<String>,
    ) -> Self {
        assert!(WINDOW_WIDTH_PIXELS > 0.0, "window width must be positive");
        assert!(WINDOW_HEIGHT_PIXELS > 0.0, "window height must be positive");
        let arguments: Vec<_> = std::env::args_os().take(3).collect();
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
        let (gpu_names, gpu_error) = match gpu_names() {
            Ok(names) if !names.is_empty() => (names, None),
            Ok(_) => (Vec::new(), Some("No Vulkan GPU was found".to_owned())),
            Err(error) => (Vec::new(), Some(error)),
        };
        let video_encoders = available_video_encoders();
        let cuda_status = cuda_inference_status();
        let default_encoder_preset = if video_encoders.contains(&VideoEncoder::NvidiaH264) {
            EncoderPreset::NvidiaP5
        } else {
            EncoderPreset::X264Medium
        };
        let tray_available = tray_controller.is_some();
        let app = Self {
            input_path,
            output_path,
            metadata,
            gpu_names,
            selected_gpu_index: 0,
            video_encoders,
            selected_video_encoder: 0,
            use_nvdec: false,
            cuda_inference_available: cuda_status.available,
            cuda_inference_unavailable_reason: cuda_status.reason,
            cuda_inference_enabled: false,
            inference_backend_menu_open: false,
            gpu_menu_open: false,
            video_encoder_menu_open: false,
            preset_menu_open: false,
            settings_page: false,
            main_page_tab: MainPageTab::Interpolation,
            target_fps_num: 120,
            target_fps_input: "120".to_owned(),
            target_fps_replace_on_type: false,
            target_fps_focus: cx.focus_handle(),
            encoding_quality: 18,
            encoding_quality_input: "18".to_owned(),
            encoding_quality_replace_on_type: false,
            encoding_quality_focus: cx.focus_handle(),
            encoder_preset: default_encoder_preset,
            encoder_preset_menu_open: false,
            h264_profile: H264Profile::Auto,
            h264_profile_menu_open: false,
            encoder_thread_count: 2,
            encoder_thread_count_input: "2".to_owned(),
            encoder_thread_count_replace_on_type: false,
            encoder_thread_count_focus: cx.focus_handle(),
            preserve_audio: true,
            preserve_subtitles: true,
            preserve_metadata: true,
            content_preset: ContentPreset::Movie,
            scene_detection: true,
            scene_detection_overridden: false,
            use_uhd_mode: false,
            use_uhd_mode_overridden: false,
            cadence_diagnostics: CadenceDiagnostics::default(),
            running: false,
            status: "Ready to interpolate".to_owned(),
            error: gpu_error,
            progress: 0.0,
            frame_count: 0,
            frame_count_estimate: 0,
            processing_fps: 0.0,
            performance: PerformanceDiagnostics::default(),
            cancellation: None,
            worker: None,
            tray_controller,
            tray_unavailable_reason: tray_error,
            keep_running_background: tray_available,
            quit_after_job: false,
        };
        assert!(
            app.target_fps_num > 0,
            "default target FPS must be positive"
        );
        assert!(
            app.selected_gpu_index <= app.gpu_names.len(),
            "selected GPU index must be bounded"
        );
        assert!(
            app.video_encoders.len() <= 3,
            "encoder list must remain bounded"
        );
        assert!(
            !app.use_nvdec || app.video_encoders.contains(&VideoEncoder::NvidiaH264),
            "NVDEC requires an available NVIDIA encoder"
        );
        assert!(
            !app.cuda_inference_enabled || app.cuda_inference_available,
            "CUDA inference requires an available backend"
        );
        assert!(
            !app.cuda_inference_unavailable_reason.is_empty(),
            "CUDA availability must explain an unavailable backend"
        );
        assert!(
            app.encoding_quality <= ENCODING_QUALITY_MAX,
            "encoding quality must remain bounded"
        );
        assert!(
            app.encoder_thread_count <= ENCODER_THREAD_COUNT_MAX,
            "encoder thread count must remain bounded"
        );
        if let Some(tray_commands) = tray_commands {
            Self::poll_tray_commands(cx, tray_commands, window_handle);
        }
        app
    }

    fn poll_tray_commands(
        cx: &mut Context<Self>,
        tray_commands: Receiver<TrayCommand>,
        window_handle: AnyWindowHandle,
    ) {
        assert!(
            TRAY_POLL_INTERVAL > Duration::ZERO,
            "tray poll interval must be positive"
        );
        assert!(
            TRAY_COMMAND_DRAIN_COUNT_MAX > 0,
            "tray command drain must be bounded"
        );
        cx.spawn(async move |this, cx| {
            for _ in 0..TRAY_POLL_COUNT_MAX {
                Timer::after(TRAY_POLL_INTERVAL).await;
                let mut disconnected = false;
                for _ in 0..TRAY_COMMAND_DRAIN_COUNT_MAX {
                    match tray_commands.try_recv() {
                        Ok(TrayCommand::Show) => {
                            let _ = window_handle.update(cx, |_, window, cx| {
                                window.activate_window();
                                cx.activate(true);
                            });
                        }
                        Ok(TrayCommand::Cancel) => {
                            let _ = this.update(cx, |app, cx| {
                                if let Some(cancellation) = &app.cancellation {
                                    cancellation.store(true, Ordering::Release);
                                    app.status = "Cancelling after current frame".to_owned();
                                    cx.notify();
                                }
                            });
                        }
                        Ok(TrayCommand::Quit) => {
                            let _ = this.update(cx, |app, cx| {
                                if let Some(cancellation) = &app.cancellation {
                                    cancellation.store(true, Ordering::Release);
                                    app.quit_after_job = true;
                                    app.status = "Cancelling before exit".to_owned();
                                    cx.notify();
                                } else {
                                    cx.quit();
                                }
                            });
                        }
                        Err(TryRecvError::Empty) => break,
                        Err(TryRecvError::Disconnected) => {
                            disconnected = true;
                            break;
                        }
                    }
                }
                if disconnected {
                    break;
                }
            }
        })
        .detach();
        assert!(
            TRAY_POLL_COUNT_MAX > 0,
            "tray polling must have a finite lifetime"
        );
        assert!(
            TRAY_COMMAND_DRAIN_COUNT_MAX <= 16,
            "tray work per poll must remain bounded"
        );
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

    fn focus_encoding_quality(
        &mut self,
        _: &gpui::ClickEvent,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        assert!(
            self.encoding_quality <= ENCODING_QUALITY_MAX,
            "encoding quality must remain bounded"
        );
        assert!(
            self.encoding_quality_input.len() <= 2,
            "encoding quality input must remain bounded"
        );
        if !self.running {
            self.encoding_quality_replace_on_type = true;
            window.focus(&self.encoding_quality_focus);
            cx.notify();
        }
        assert!(
            self.encoding_quality_input.len() <= 2,
            "focused encoding quality input must remain bounded"
        );
        assert!(self.encoding_quality <= ENCODING_QUALITY_MAX);
    }

    fn edit_encoding_quality(
        &mut self,
        event: &KeyDownEvent,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        assert!(
            self.encoding_quality_input.len() <= 2,
            "encoding quality input must remain bounded"
        );
        assert!(
            self.encoding_quality <= ENCODING_QUALITY_MAX,
            "encoding quality must remain bounded"
        );
        if self.running {
            return;
        }
        match event.keystroke.key.as_str() {
            "backspace" => {
                if self.encoding_quality_replace_on_type {
                    self.encoding_quality_input.clear();
                    self.encoding_quality_replace_on_type = false;
                } else {
                    self.encoding_quality_input.pop();
                }
            }
            "enter" | "escape" => {
                window.blur();
                self.encoding_quality_replace_on_type = false;
            }
            _ => {
                if let Some(character) = event.keystroke.key_char.as_deref()
                    && character.len() == 1
                    && character.as_bytes()[0].is_ascii_digit()
                {
                    if self.encoding_quality_replace_on_type {
                        self.encoding_quality_input.clear();
                        self.encoding_quality_replace_on_type = false;
                    }
                    if self.encoding_quality_input.len() < 2 {
                        self.encoding_quality_input.push_str(character);
                    }
                } else {
                    return;
                }
            }
        }
        match self.encoding_quality_input.parse::<u8>() {
            Ok(quality) if quality <= ENCODING_QUALITY_MAX => {
                self.encoding_quality = quality;
                self.error = None;
            }
            _ => {
                self.error = Some(format!(
                    "Encoding quality must be between 0 and {ENCODING_QUALITY_MAX}"
                ))
            }
        }
        cx.stop_propagation();
        cx.notify();
        assert!(
            self.encoding_quality_input.len() <= 2,
            "edited encoding quality input must remain bounded"
        );
        assert!(
            self.encoding_quality <= ENCODING_QUALITY_MAX,
            "last valid encoding quality must remain bounded"
        );
    }

    fn focus_encoder_thread_count(
        &mut self,
        _: &gpui::ClickEvent,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        assert!(
            self.encoder_thread_count <= ENCODER_THREAD_COUNT_MAX,
            "encoder thread count must remain bounded"
        );
        assert!(
            self.encoder_thread_count_input.len() <= 2,
            "encoder thread input must remain bounded"
        );
        if !self.running {
            self.encoder_thread_count_replace_on_type = true;
            window.focus(&self.encoder_thread_count_focus);
            cx.notify();
        }
        assert!(
            self.encoder_thread_count_input.len() <= 2,
            "focused encoder thread input must remain bounded"
        );
        assert!(self.encoder_thread_count <= ENCODER_THREAD_COUNT_MAX);
    }

    fn edit_encoder_thread_count(
        &mut self,
        event: &KeyDownEvent,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        assert!(
            self.encoder_thread_count_input.len() <= 2,
            "encoder thread input must remain bounded"
        );
        assert!(
            self.encoder_thread_count <= ENCODER_THREAD_COUNT_MAX,
            "encoder thread count must remain bounded"
        );
        if self.running {
            return;
        }
        match event.keystroke.key.as_str() {
            "backspace" => {
                if self.encoder_thread_count_replace_on_type {
                    self.encoder_thread_count_input.clear();
                    self.encoder_thread_count_replace_on_type = false;
                } else {
                    self.encoder_thread_count_input.pop();
                }
            }
            "enter" | "escape" => {
                window.blur();
                self.encoder_thread_count_replace_on_type = false;
            }
            _ => {
                if let Some(character) = event.keystroke.key_char.as_deref()
                    && character.len() == 1
                    && character.as_bytes()[0].is_ascii_digit()
                {
                    if self.encoder_thread_count_replace_on_type {
                        self.encoder_thread_count_input.clear();
                        self.encoder_thread_count_replace_on_type = false;
                    }
                    if self.encoder_thread_count_input.len() < 2 {
                        self.encoder_thread_count_input.push_str(character);
                    }
                } else {
                    return;
                }
            }
        }
        match self.encoder_thread_count_input.parse::<u8>() {
            Ok(thread_count) if thread_count <= ENCODER_THREAD_COUNT_MAX => {
                self.encoder_thread_count = thread_count;
                self.error = None;
            }
            _ => {
                self.error = Some(format!(
                    "Encoder threads must be between 0 and {ENCODER_THREAD_COUNT_MAX}"
                ))
            }
        }
        cx.stop_propagation();
        cx.notify();
        assert!(
            self.encoder_thread_count_input.len() <= 2,
            "edited encoder thread input must remain bounded"
        );
        assert!(
            self.encoder_thread_count <= ENCODER_THREAD_COUNT_MAX,
            "last valid encoder thread count must remain bounded"
        );
    }

    fn default_encoder_preset(&self, encoder: VideoEncoder) -> EncoderPreset {
        assert!(
            self.video_encoders.len() <= 3,
            "encoder list must remain bounded"
        );
        assert!(self.video_encoders.contains(&encoder) || encoder == VideoEncoder::Automatic);
        let use_nvidia = encoder == VideoEncoder::NvidiaH264
            || (encoder == VideoEncoder::Automatic
                && self.video_encoders.contains(&VideoEncoder::NvidiaH264));
        let preset = if use_nvidia {
            EncoderPreset::NvidiaP5
        } else {
            EncoderPreset::X264Medium
        };
        assert!(matches!(
            preset,
            EncoderPreset::X264Medium | EncoderPreset::NvidiaP5
        ));
        preset
    }

    fn toggle_encoder_preset_menu(
        &mut self,
        _: &gpui::ClickEvent,
        _: &mut Window,
        cx: &mut Context<Self>,
    ) {
        assert!(
            self.video_encoders.len() <= 3,
            "encoder list must remain bounded"
        );
        assert!(self.encoder_preset_menu_open || !self.running);
        if !self.running {
            self.encoder_preset_menu_open = !self.encoder_preset_menu_open;
            self.h264_profile_menu_open = false;
            cx.notify();
        }
        assert!(!self.encoder_preset_menu_open || !self.running);
        assert!(!self.encoder_preset_menu_open || !self.h264_profile_menu_open);
    }

    fn select_encoder_preset(&mut self, preset: EncoderPreset, cx: &mut Context<Self>) {
        assert!(
            self.video_encoders.len() <= 3,
            "encoder list must remain bounded"
        );
        if !self.running {
            self.encoder_preset = preset;
            self.encoder_preset_menu_open = false;
            cx.notify();
        }
        assert!(!self.encoder_preset_menu_open || !self.running);
        assert!(matches!(
            self.encoder_preset,
            EncoderPreset::X264VeryFast
                | EncoderPreset::X264Faster
                | EncoderPreset::X264Fast
                | EncoderPreset::X264Medium
                | EncoderPreset::X264Slow
                | EncoderPreset::X264Slower
                | EncoderPreset::X264VerySlow
                | EncoderPreset::X264Placebo
                | EncoderPreset::NvidiaP1
                | EncoderPreset::NvidiaP2
                | EncoderPreset::NvidiaP3
                | EncoderPreset::NvidiaP4
                | EncoderPreset::NvidiaP5
                | EncoderPreset::NvidiaP6
                | EncoderPreset::NvidiaP7
        ));
    }

    fn toggle_h264_profile_menu(
        &mut self,
        _: &gpui::ClickEvent,
        _: &mut Window,
        cx: &mut Context<Self>,
    ) {
        assert!(
            self.video_encoders.len() <= 3,
            "encoder list must remain bounded"
        );
        assert!(!self.h264_profile_menu_open || !self.running);
        if !self.running {
            self.h264_profile_menu_open = !self.h264_profile_menu_open;
            self.encoder_preset_menu_open = false;
            cx.notify();
        }
        assert!(!self.h264_profile_menu_open || !self.running);
        assert!(!self.h264_profile_menu_open || !self.encoder_preset_menu_open);
    }

    fn select_h264_profile(&mut self, profile: H264Profile, cx: &mut Context<Self>) {
        assert!(
            self.video_encoders.len() <= 3,
            "encoder list must remain bounded"
        );
        if !self.running {
            self.h264_profile = profile;
            self.h264_profile_menu_open = false;
            cx.notify();
        }
        assert!(!self.h264_profile_menu_open || !self.running);
        assert!(matches!(
            self.h264_profile,
            H264Profile::Auto | H264Profile::Main | H264Profile::High
        ));
    }

    fn toggle_preserve_audio(
        &mut self,
        _: &gpui::ClickEvent,
        _: &mut Window,
        cx: &mut Context<Self>,
    ) {
        assert!(
            self.video_encoders.len() <= 3,
            "encoder list must remain bounded"
        );
        let preserve_audio_before = self.preserve_audio;
        if !self.running {
            self.preserve_audio = !self.preserve_audio;
            cx.notify();
        }
        assert!(
            self.running || self.preserve_audio != preserve_audio_before,
            "idle audio preservation toggle must change state"
        );
        assert!(
            !self.running || self.preserve_audio == preserve_audio_before,
            "running audio preservation toggle must not change state"
        );
    }

    fn toggle_preserve_subtitles(
        &mut self,
        _: &gpui::ClickEvent,
        _: &mut Window,
        cx: &mut Context<Self>,
    ) {
        assert!(
            self.video_encoders.len() <= 3,
            "encoder list must remain bounded"
        );
        let preserve_subtitles_before = self.preserve_subtitles;
        if !self.running {
            self.preserve_subtitles = !self.preserve_subtitles;
            cx.notify();
        }
        assert!(
            self.running || self.preserve_subtitles != preserve_subtitles_before,
            "idle subtitle preservation toggle must change state"
        );
        assert!(
            !self.running || self.preserve_subtitles == preserve_subtitles_before,
            "running subtitle preservation toggle must not change state"
        );
    }

    fn toggle_preserve_metadata(
        &mut self,
        _: &gpui::ClickEvent,
        _: &mut Window,
        cx: &mut Context<Self>,
    ) {
        assert!(
            self.video_encoders.len() <= 3,
            "encoder list must remain bounded"
        );
        let preserve_metadata_before = self.preserve_metadata;
        if !self.running {
            self.preserve_metadata = !self.preserve_metadata;
            cx.notify();
        }
        assert!(
            self.running || self.preserve_metadata != preserve_metadata_before,
            "idle metadata preservation toggle must change state"
        );
        assert!(
            !self.running || self.preserve_metadata == preserve_metadata_before,
            "running metadata preservation toggle must not change state"
        );
    }

    fn set_settings_page(&mut self, settings_page: bool, cx: &mut Context<Self>) {
        assert!(self.target_fps_num > 0, "target FPS must remain valid");
        assert!(self.gpu_names.len() <= 16, "GPU list must remain bounded");
        self.settings_page = settings_page;
        self.inference_backend_menu_open = false;
        self.gpu_menu_open = false;
        self.video_encoder_menu_open = false;
        self.preset_menu_open = false;
        self.encoder_preset_menu_open = false;
        self.h264_profile_menu_open = false;
        cx.notify();
        assert!(
            !self.inference_backend_menu_open,
            "settings navigation must close inference backend menus"
        );
        assert!(
            !self.gpu_menu_open,
            "settings navigation must close GPU menus"
        );
        assert!(
            !self.preset_menu_open,
            "settings navigation must close preset menus"
        );
        assert!(
            !self.video_encoder_menu_open,
            "settings navigation must close encoder menus"
        );
        assert!(
            !self.encoder_preset_menu_open && !self.h264_profile_menu_open,
            "settings navigation must close encoding menus"
        );
    }

    fn set_main_page_tab(&mut self, tab: MainPageTab, cx: &mut Context<Self>) {
        assert!(self.target_fps_num > 0, "target FPS must remain valid");
        assert!(self.gpu_names.len() <= 16, "GPU list must remain bounded");
        self.main_page_tab = tab;
        self.inference_backend_menu_open = false;
        self.preset_menu_open = false;
        self.video_encoder_menu_open = false;
        self.gpu_menu_open = false;
        self.encoder_preset_menu_open = false;
        self.h264_profile_menu_open = false;
        cx.notify();
        assert_eq!(self.main_page_tab, tab, "tab selection must be retained");
        assert!(!self.inference_backend_menu_open);
        assert!(!self.preset_menu_open);
    }

    fn select_main_page_tab(&mut self, tab: MainPageTab, cx: &mut Context<Self>) {
        assert!(self.target_fps_num > 0, "target FPS must remain valid");
        assert!(self.gpu_names.len() <= 16, "GPU list must remain bounded");
        self.set_main_page_tab(tab, cx);
        assert_eq!(self.main_page_tab, tab, "selected tab must be retained");
        assert!(
            !self.preset_menu_open,
            "selected tabs must close preset menus"
        );
    }

    fn show_encode_page(&mut self, _: &gpui::ClickEvent, _: &mut Window, cx: &mut Context<Self>) {
        assert!(self.target_fps_num > 0, "target FPS must remain valid");
        assert!(self.gpu_names.len() <= 16, "GPU list must remain bounded");
        assert!(
            self.video_encoders.len() <= 3,
            "encoder list must remain bounded"
        );
        self.set_settings_page(false, cx);
    }

    fn show_settings_page(&mut self, _: &gpui::ClickEvent, _: &mut Window, cx: &mut Context<Self>) {
        assert!(self.target_fps_num > 0, "target FPS must remain valid");
        assert!(self.gpu_names.len() <= 16, "GPU list must remain bounded");
        assert!(
            self.video_encoders.len() <= 3,
            "encoder list must remain bounded"
        );
        self.set_settings_page(true, cx);
    }

    fn toggle_gpu_menu(&mut self, _: &gpui::ClickEvent, _: &mut Window, cx: &mut Context<Self>) {
        assert!(
            self.selected_gpu_index <= self.gpu_names.len(),
            "GPU selection must be bounded"
        );
        assert!(self.gpu_names.len() <= 16, "GPU list must remain bounded");
        if !self.running && !self.gpu_names.is_empty() {
            self.gpu_menu_open = !self.gpu_menu_open;
            self.video_encoder_menu_open = false;
            self.preset_menu_open = false;
            self.encoder_preset_menu_open = false;
            self.h264_profile_menu_open = false;
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

    fn toggle_video_encoder_menu(
        &mut self,
        _: &gpui::ClickEvent,
        _: &mut Window,
        cx: &mut Context<Self>,
    ) {
        assert!(
            self.video_encoders.len() <= 3,
            "encoder list must remain bounded"
        );
        assert!(
            self.selected_video_encoder <= self.video_encoders.len(),
            "selected encoder index must be bounded"
        );
        if !self.running && !self.video_encoders.is_empty() {
            self.video_encoder_menu_open = !self.video_encoder_menu_open;
            self.gpu_menu_open = false;
            self.preset_menu_open = false;
            self.encoder_preset_menu_open = false;
            self.h264_profile_menu_open = false;
            cx.notify();
        }
        assert!(
            !self.video_encoders.is_empty() || !self.video_encoder_menu_open,
            "empty encoder list cannot open a menu"
        );
        assert!(
            !self.video_encoder_menu_open || !self.running,
            "running state cannot retain an encoder menu"
        );
    }

    fn select_video_encoder(&mut self, encoder_index: usize, cx: &mut Context<Self>) {
        assert!(
            encoder_index < self.video_encoders.len(),
            "selected encoder index must exist"
        );
        assert!(
            self.video_encoders.len() <= 3,
            "encoder list must remain bounded"
        );
        if !self.running {
            self.selected_video_encoder = encoder_index;
            let selected_encoder = self.video_encoders[encoder_index];
            self.encoder_preset = self.default_encoder_preset(selected_encoder);
            self.video_encoder_menu_open = false;
            cx.notify();
        }
        assert!(
            self.selected_video_encoder < self.video_encoders.len(),
            "updated encoder selection must be valid"
        );
        assert!(
            !self.video_encoder_menu_open || !self.running,
            "running state cannot retain an encoder menu"
        );
    }

    fn toggle_inference_backend_menu(
        &mut self,
        _: &gpui::ClickEvent,
        _: &mut Window,
        cx: &mut Context<Self>,
    ) {
        assert!(
            self.cuda_inference_unavailable_reason.len() < 256,
            "CUDA availability reason must remain bounded"
        );
        assert!(
            !self.cuda_inference_enabled || self.cuda_inference_available,
            "CUDA inference requires an available backend"
        );
        if !self.running {
            self.inference_backend_menu_open = !self.inference_backend_menu_open;
            self.gpu_menu_open = false;
            self.video_encoder_menu_open = false;
            self.preset_menu_open = false;
            self.encoder_preset_menu_open = false;
            self.h264_profile_menu_open = false;
            cx.notify();
        }
        assert!(
            !self.inference_backend_menu_open || !self.running,
            "running state cannot retain an inference backend menu"
        );
        assert!(
            !self.cuda_inference_enabled || self.cuda_inference_available,
            "selected CUDA inference requires an available backend"
        );
    }

    fn select_inference_backend(&mut self, use_cuda: bool, cx: &mut Context<Self>) {
        assert!(
            !use_cuda || self.cuda_inference_available,
            "CUDA inference requires an available backend"
        );
        assert!(
            self.cuda_inference_unavailable_reason.len() < 256,
            "CUDA availability reason must remain bounded"
        );
        if !self.running && (!use_cuda || self.cuda_inference_available) {
            self.cuda_inference_enabled = use_cuda;
            self.inference_backend_menu_open = false;
            cx.notify();
        }
        assert!(
            !self.cuda_inference_enabled || self.cuda_inference_available,
            "updated CUDA inference selection requires an available backend"
        );
        assert!(!self.inference_backend_menu_open || !self.running);
    }

    fn toggle_nvdec(&mut self, _: &gpui::ClickEvent, _: &mut Window, cx: &mut Context<Self>) {
        assert!(
            self.video_encoders.len() <= 3,
            "encoder list must remain bounded"
        );
        let nvdec_available = self.video_encoders.contains(&VideoEncoder::NvidiaH264);
        assert!(
            !self.use_nvdec || nvdec_available,
            "NVDEC cannot be enabled without NVIDIA support"
        );
        if !self.running && nvdec_available {
            self.use_nvdec = !self.use_nvdec;
            cx.notify();
        }
        assert!(
            !self.use_nvdec || self.video_encoders.contains(&VideoEncoder::NvidiaH264),
            "updated NVDEC selection requires NVIDIA support"
        );
        assert!(
            !self.running || !self.use_nvdec || nvdec_available,
            "running NVDEC requires support"
        );
    }

    fn toggle_preset_menu(&mut self, _: &gpui::ClickEvent, _: &mut Window, cx: &mut Context<Self>) {
        assert!(self.target_fps_num > 0, "target FPS must be positive");
        assert!(self.target_fps_num <= 480, "target FPS must stay bounded");
        if !self.running {
            self.preset_menu_open = !self.preset_menu_open;
            self.gpu_menu_open = false;
            self.video_encoder_menu_open = false;
            self.encoder_preset_menu_open = false;
            self.h264_profile_menu_open = false;
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

    fn toggle_background_mode(
        &mut self,
        _: &gpui::ClickEvent,
        _: &mut Window,
        cx: &mut Context<Self>,
    ) {
        assert!(
            TRAY_COMMAND_DRAIN_COUNT_MAX > 0,
            "tray command drain must be bounded"
        );
        assert!(
            TRAY_POLL_COUNT_MAX > 0,
            "tray polling must have a finite lifetime"
        );
        if self.tray_controller.is_some() {
            self.keep_running_background = !self.keep_running_background;
            self.error = None;
            cx.notify();
        } else {
            self.keep_running_background = false;
            self.error = Some(self.tray_unavailable_reason.clone().unwrap_or_else(|| {
                "System tray integration is unavailable on this desktop".to_owned()
            }));
            cx.notify();
        }
        assert!(
            !self.keep_running_background || self.tray_controller.is_some(),
            "background mode requires a system tray"
        );
        assert!(self.target_fps_num > 0, "target FPS must remain valid");
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
        assert!(
            self.video_encoders.len() <= 3,
            "encoder list must remain bounded"
        );
        assert!(
            self.selected_video_encoder <= self.video_encoders.len(),
            "encoder selection must be bounded"
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
        if self.encoding_quality_input.parse::<u8>().is_err()
            || self
                .encoding_quality_input
                .parse::<u8>()
                .is_ok_and(|value| value > ENCODING_QUALITY_MAX)
        {
            self.error = Some(format!(
                "Encoding quality must be between 0 and {ENCODING_QUALITY_MAX}"
            ));
            cx.notify();
            return;
        }
        if self.encoder_thread_count_input.parse::<u8>().is_err()
            || self
                .encoder_thread_count_input
                .parse::<u8>()
                .is_ok_and(|value| value > ENCODER_THREAD_COUNT_MAX)
        {
            self.error = Some(format!(
                "Encoder threads must be between 0 and {ENCODER_THREAD_COUNT_MAX}"
            ));
            cx.notify();
            return;
        }
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
        if self.video_encoders.is_empty() {
            self.error = Some("FFmpeg has no supported H.264 encoder".to_owned());
            cx.notify();
            return;
        }
        if self.selected_video_encoder >= self.video_encoders.len() {
            self.error = Some("Choose a supported video encoder".to_owned());
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
            use_nvdec: self.use_nvdec,
            inference_backend: if self.cuda_inference_enabled {
                InferenceBackend::CudaPytorchVapourSynth
            } else {
                InferenceBackend::VulkanNcnn
            },
            video_encoder: *self
                .video_encoders
                .get(self.selected_video_encoder)
                .unwrap_or(&VideoEncoder::Automatic),
            encoder_preset: self.encoder_preset,
            quality_level: self.encoding_quality,
            h264_profile: self.h264_profile,
            encoder_thread_count: self.encoder_thread_count,
            preserve_audio: self.preserve_audio,
            preserve_subtitles: self.preserve_subtitles,
            preserve_metadata: self.preserve_metadata,
        };
        let cancellation = Arc::new(AtomicBool::new(false));
        let worker_cancellation = Arc::clone(&cancellation);
        let (update_sender, update_receiver) = sync_channel(1);
        let worker = match thread::Builder::new()
            .name("interpolation-worker".to_owned())
            .spawn(move || run_job(configuration, worker_cancellation, update_sender))
        {
            Ok(worker) => worker,
            Err(error) => {
                self.error = Some(format!("failed to start interpolation worker: {error}"));
                cx.notify();
                return;
            }
        };

        self.running = true;
        self.worker = Some(worker);
        self.gpu_menu_open = false;
        self.video_encoder_menu_open = false;
        self.preset_menu_open = false;
        self.encoder_preset_menu_open = false;
        self.h264_profile_menu_open = false;
        self.status = "Starting job".to_owned();
        self.error = None;
        self.progress = 0.0;
        self.frame_count = 0;
        self.frame_count_estimate = 0;
        self.processing_fps = 0.0;
        self.performance = PerformanceDiagnostics::default();
        self.cadence_diagnostics = CadenceDiagnostics::default();
        self.cancellation = Some(cancellation);
        if let Some(tray_controller) = &self.tray_controller
            && let Err(error) = tray_controller.set_running(true)
        {
            self.keep_running_background = false;
            self.error = Some(error);
        }
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
                                    | JobUpdate::Cancelled { .. }
                                    | JobUpdate::Failed(_)
                            );
                            let _ = this.update(cx, |app, cx| {
                                app.apply_update(update, cx);
                                cx.notify();
                            });
                            if terminal {
                                break;
                            }
                        }
                        Err(TryRecvError::Empty) => break,
                        Err(TryRecvError::Disconnected) => {
                            terminal = true;
                            let _ = this.update(cx, |app, cx| {
                                if app.running {
                                    app.apply_update(
                                        JobUpdate::Failed(
                                            "processing worker stopped unexpectedly".to_owned(),
                                        ),
                                        cx,
                                    );
                                    cx.notify();
                                }
                            });
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

    fn apply_update(&mut self, update: JobUpdate, cx: &mut Context<Self>) {
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
                performance,
            } => {
                self.frame_count = frame_count;
                self.frame_count_estimate = frame_count_estimate;
                self.processing_fps = processing_fps;
                self.performance = performance;
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
            JobUpdate::Cancelled { partial_path } => {
                self.running = false;
                self.cancellation = None;
                self.status = match partial_path {
                    Some(path) => {
                        format!("Cancelled · partial output saved: {}", path.display())
                    }
                    None => "Cancelled safely".to_owned(),
                };
            }
            JobUpdate::Failed(error) => {
                self.running = false;
                self.cancellation = None;
                self.status = "Job failed".to_owned();
                self.error = Some(error);
            }
        }
        if !self.running {
            if let Some(worker) = self.worker.take()
                && worker.join().is_err()
            {
                self.status = "Job failed".to_owned();
                self.error = Some("processing worker panicked".to_owned());
            }
            if let Some(tray_controller) = &self.tray_controller
                && let Err(error) = tray_controller.set_running(false)
            {
                self.keep_running_background = false;
                self.error = Some(error);
            }
            if self.quit_after_job {
                self.quit_after_job = false;
                cx.quit();
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

impl InterpolateApp {
    fn render_encoding_tab(
        &mut self,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> impl IntoElement {
        assert!(
            self.encoding_quality <= ENCODING_QUALITY_MAX,
            "encoding quality must remain bounded"
        );
        assert!(
            self.encoder_thread_count <= ENCODER_THREAD_COUNT_MAX,
            "encoder threads must remain bounded"
        );
        let selected_encoder = self
            .video_encoders
            .get(self.selected_video_encoder)
            .copied()
            .unwrap_or(VideoEncoder::Automatic);
        let resolved_encoder = if selected_encoder == VideoEncoder::NvidiaH264 {
            VideoEncoder::NvidiaH264
        } else {
            VideoEncoder::SoftwareH264
        };
        let preset_options = encoder_preset_options(resolved_encoder);
        let quality_mode_label = if resolved_encoder == VideoEncoder::NvidiaH264 {
            "CQ"
        } else {
            "CRF"
        };
        let quality_focused = self.encoding_quality_focus.is_focused(window);
        let quality_valid = self
            .encoding_quality_input
            .parse::<u8>()
            .is_ok_and(|value| value <= ENCODING_QUALITY_MAX);
        let thread_focused = self.encoder_thread_count_focus.is_focused(window);
        let thread_valid = self
            .encoder_thread_count_input
            .parse::<u8>()
            .is_ok_and(|value| value <= ENCODER_THREAD_COUNT_MAX);
        let card = div()
            .w_full()
            .relative()
            .rounded_lg()
            .border_1()
            .border_color(rgb(BORDER_COLOR))
            .bg(rgb(PANEL_COLOR))
            .child(
                div()
                    .px_4()
                    .py_4()
                    .border_b_1()
                    .border_color(rgb(BORDER_COLOR))
                    .child(div().font_family("JetBrains Mono").child("Encoding"))
                    .child(
                        div()
                            .mt_1()
                            .text_xs()
                            .text_color(rgb(TEXT_MUTED_COLOR))
                            .child("Output controls for this encoding"),
                    ),
            )
            .child(
                div()
                    .h(px(64.0))
                    .px_4()
                    .flex()
                    .items_center()
                    .justify_between()
                    .border_b_1()
                    .border_color(rgb(BORDER_COLOR))
                    .child(
                        div()
                            .child(
                                div()
                                    .flex()
                                    .items_center()
                                    .child("Quality")
                                    .child(setting_tooltip_icon(
                                        "CRF (CPU) and CQ (NVENC) target visual quality rather than a fixed bitrate. Lower values preserve more detail and produce larger files.",
                                    )),
                            )
                            .child(
                                div()
                                    .mt_1()
                                    .text_xs()
                                    .text_color(rgb(TEXT_MUTED_COLOR))
                                    .child(format!(
                                        "{quality_mode_label}; lower values mean higher quality"
                                    )),
                            ),
                    )
                    .child(
                        div()
                            .id("main-encoding-quality-input")
                            .track_focus(&self.encoding_quality_focus)
                            .on_key_down(cx.listener(Self::edit_encoding_quality))
                            .on_click(cx.listener(Self::focus_encoding_quality))
                            .w(px(112.0))
                            .px_3()
                            .py_2()
                            .flex()
                            .items_center()
                            .justify_between()
                            .rounded_md()
                            .cursor_text()
                            .border_1()
                            .border_color(rgb(if !quality_valid {
                                ERROR_COLOR
                            } else if quality_focused {
                                ACCENT_COLOR
                            } else {
                                BORDER_COLOR
                            }))
                            .bg(rgb(PANEL_COLOR))
                            .font_family("JetBrains Mono")
                            .child(self.encoding_quality_input.clone())
                            .child(
                                div()
                                    .ml_2()
                                    .text_color(rgb(TEXT_SUBTLE_COLOR))
                                    .child(quality_mode_label),
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
                    .border_b_1()
                    .border_color(rgb(BORDER_COLOR))
                    .child(
                        div()
                            .child(
                                div()
                                    .flex()
                                    .items_center()
                                    .child("Speed preset")
                                    .child(setting_tooltip_icon(
                                        "Controls encoder complexity. Faster presets finish sooner and usually create larger files; slower presets spend more CPU/GPU time to improve compression efficiency.",
                                    )),
                            )
                            .child(
                                div()
                                    .mt_1()
                                    .text_xs()
                                    .text_color(rgb(TEXT_MUTED_COLOR))
                                    .child("Controls the quality/speed trade-off"),
                            ),
                    )
                    .child(
                        div()
                            .id("main-encoder-preset-selector")
                            .w(px(270.0))
                            .px_3()
                            .py_2()
                            .rounded_md()
                            .cursor_pointer()
                            .border_1()
                            .border_color(rgb(if self.encoder_preset_menu_open {
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
                            .on_click(cx.listener(Self::toggle_encoder_preset_menu))
                            .child(encoder_preset_label(self.encoder_preset))
                            .child(if self.encoder_preset_menu_open {
                                "⌃"
                            } else {
                                "⌄"
                            }),
                    ),
            )
            .child(
                div()
                    .h(px(64.0))
                    .px_4()
                    .flex()
                    .items_center()
                    .justify_between()
                    .border_b_1()
                    .border_color(rgb(BORDER_COLOR))
                    .child(
                        div()
                            .child(
                                div()
                                    .flex()
                                    .items_center()
                                    .child("H.264 profile")
                                    .child(setting_tooltip_icon(
                                        "Sets which H.264 features the output may use. Auto chooses a broadly compatible profile; High can improve compression but may not play everywhere.",
                                    )),
                            )
                            .child(
                                div()
                                    .mt_1()
                                    .text_xs()
                                    .text_color(rgb(TEXT_MUTED_COLOR))
                                    .child("Auto is recommended for compatibility"),
                            ),
                    )
                    .child(
                        div()
                            .id("main-h264-profile-selector")
                            .w(px(270.0))
                            .px_3()
                            .py_2()
                            .rounded_md()
                            .cursor_pointer()
                            .border_1()
                            .border_color(rgb(if self.h264_profile_menu_open {
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
                            .on_click(cx.listener(Self::toggle_h264_profile_menu))
                            .child(h264_profile_label(self.h264_profile))
                            .child(if self.h264_profile_menu_open {
                                "⌃"
                            } else {
                                "⌄"
                            }),
                    ),
            )
            .child(
                div()
                    .h(px(64.0))
                    .px_4()
                    .flex()
                    .items_center()
                    .justify_between()
                    .child(
                        div()
                            .child(
                                div()
                                    .flex()
                                    .items_center()
                                    .child("Encoder threads")
                                    .child(setting_tooltip_icon(
                                        "Limits the number of CPU threads used by FFmpeg while encoding. Zero delegates the choice to FFmpeg; fewer threads reduce contention but may slow encoding.",
                                    )),
                            )
                            .child(
                                div()
                                    .mt_1()
                                    .text_xs()
                                    .text_color(rgb(TEXT_MUTED_COLOR))
                                    .child("0 means FFmpeg automatic mode"),
                            ),
                    )
                    .child(
                        div()
                            .id("main-encoder-thread-count-input")
                            .track_focus(&self.encoder_thread_count_focus)
                            .on_key_down(cx.listener(Self::edit_encoder_thread_count))
                            .on_click(cx.listener(Self::focus_encoder_thread_count))
                            .w(px(112.0))
                            .px_3()
                            .py_2()
                            .flex()
                            .items_center()
                            .justify_between()
                            .rounded_md()
                            .cursor_text()
                            .border_1()
                            .border_color(rgb(if !thread_valid {
                                ERROR_COLOR
                            } else if thread_focused {
                                ACCENT_COLOR
                            } else {
                                BORDER_COLOR
                            }))
                            .bg(rgb(PANEL_COLOR))
                            .font_family("JetBrains Mono")
                            .child(self.encoder_thread_count_input.clone())
                            .child(
                                div()
                                    .ml_2()
                                    .text_color(rgb(TEXT_SUBTLE_COLOR))
                                    .child("threads"),
                            ),
                    ),
            )
            .when(self.encoder_preset_menu_open, |element| {
                element.child(
                    div()
                        .id("main-encoder-preset-options")
                        .absolute()
                        .top(px(140.0))
                        .right(px(16.0))
                        .w(px(270.0))
                        .h(px(324.0))
                        .overflow_y_scroll()
                        .rounded_md()
                        .border_1()
                        .border_color(rgb(BORDER_STRONG_COLOR))
                        .bg(rgb(PANEL_COLOR))
                        .shadow_lg()
                        .children(preset_options.iter().copied().enumerate().map(
                            |(index, preset)| {
                                let selected = preset == self.encoder_preset;
                                div()
                                    .id(("main-encoder-preset-option", index))
                                    .h(px(36.0))
                                    .px_3()
                                    .flex()
                                    .items_center()
                                    .justify_between()
                                    .cursor_pointer()
                                    .bg(rgb(if selected {
                                        SELECTED_COLOR
                                    } else {
                                        PANEL_COLOR
                                    }))
                                    .on_click(cx.listener(move |app, _, _, cx| {
                                        app.select_encoder_preset(preset, cx)
                                    }))
                                    .child(encoder_preset_label(preset))
                                    .child(if selected { "✓" } else { "" })
                            },
                        )),
                )
            })
            .when(self.h264_profile_menu_open, |element| {
                element.child(
                    div()
                        .id("main-h264-profile-options")
                        .absolute()
                        .top(px(204.0))
                        .right(px(16.0))
                        .w(px(270.0))
                        .h(px(108.0))
                        .rounded_md()
                        .border_1()
                        .border_color(rgb(BORDER_STRONG_COLOR))
                        .bg(rgb(PANEL_COLOR))
                        .shadow_lg()
                        .children(H264_PROFILES.iter().copied().enumerate().map(
                            |(index, profile)| {
                                let selected = profile == self.h264_profile;
                                div()
                                    .id(("main-h264-profile-option", index))
                                    .h(px(36.0))
                                    .px_3()
                                    .flex()
                                    .items_center()
                                    .justify_between()
                                    .cursor_pointer()
                                    .bg(rgb(if selected {
                                        SELECTED_COLOR
                                    } else {
                                        PANEL_COLOR
                                    }))
                                    .on_click(cx.listener(move |app, _, _, cx| {
                                        app.select_h264_profile(profile, cx)
                                    }))
                                    .child(h264_profile_label(profile))
                                    .child(if selected { "✓" } else { "" })
                            },
                        )),
                )
            });
        assert!(
            preset_options.len() <= 9,
            "encoding preset options must remain bounded"
        );
        assert!(
            H264_PROFILES.len() <= 3,
            "H.264 profile options must remain bounded"
        );
        card
    }

    fn render_hardware_tab(&mut self, cx: &mut Context<Self>) -> impl IntoElement {
        assert!(
            self.video_encoders.len() <= 3,
            "encoder list must remain bounded"
        );
        assert!(
            self.selected_gpu_index <= self.gpu_names.len(),
            "GPU selection must remain bounded"
        );
        let selected_encoder = self
            .video_encoders
            .get(self.selected_video_encoder)
            .copied()
            .unwrap_or(VideoEncoder::Automatic);
        let gpu_label = self
            .gpu_names
            .get(self.selected_gpu_index)
            .cloned()
            .unwrap_or_else(|| "No Vulkan device".to_owned());
        let cuda_description = if self.cuda_inference_available {
            "Vulkan or CUDA/PyTorch + VapourSynth".to_owned()
        } else {
            self.cuda_inference_unavailable_reason.clone()
        };
        let nvdec_available = self.video_encoders.contains(&VideoEncoder::NvidiaH264);
        div()
            .w_full()
            .relative()
            .rounded_lg()
            .border_1()
            .border_color(rgb(BORDER_COLOR))
            .bg(rgb(PANEL_COLOR))
            .child(
                div()
                    .px_4()
                    .py_4()
                    .border_b_1()
                    .border_color(rgb(BORDER_COLOR))
                    .child(div().font_family("JetBrains Mono").child("Hardware"))
                    .child(
                        div()
                            .mt_1()
                            .text_xs()
                            .text_color(rgb(TEXT_MUTED_COLOR))
                            .child("Select the acceleration devices for this encoding."),
                    ),
            )
            .child(
                div()
                    .h(px(76.0))
                    .px_4()
                    .flex()
                    .items_center()
                    .justify_between()
                    .border_b_1()
                    .border_color(rgb(BORDER_COLOR))
                    .child(
                        div()
                            .child(
                                div()
                                    .flex()
                                    .items_center()
                                    .child("RIFE inference backend")
                                    .child(setting_tooltip_icon(
                                        "Vulkan / ncnn is the self-contained fallback. CUDA / PyTorch uses the bundled Python runtime and RIFE model, and requires a compatible NVIDIA driver.",
                                    )),
                            )
                            .child(
                                div()
                                    .mt_1()
                                    .text_xs()
                                    .text_color(rgb(TEXT_MUTED_COLOR))
                                    .child(cuda_description),
                            ),
                    )
                    .child(
                        div()
                            .id("main-inference-backend-selector")
                            .w(px(270.0))
                            .px_3()
                            .py_2()
                            .rounded_md()
                            .cursor_pointer()
                            .border_1()
                            .border_color(rgb(if self.inference_backend_menu_open {
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
                            .text_color(rgb(TEXT_COLOR))
                            .on_click(cx.listener(Self::toggle_inference_backend_menu))
                            .child(if self.cuda_inference_enabled {
                                "CUDA / PyTorch"
                            } else {
                                "Vulkan / ncnn"
                            })
                            .child(if self.inference_backend_menu_open {
                                "⌃"
                            } else {
                                "⌄"
                            }),
                    ),
            )
            .child(
                div()
                    .h(px(64.0))
                    .px_4()
                    .flex()
                    .items_center()
                    .justify_between()
                    .border_b_1()
                    .border_color(rgb(BORDER_COLOR))
                    .child(
                        div()
                            .child(
                                div()
                                    .flex()
                                    .items_center()
                                    .child("Video encoder")
                                    .child(setting_tooltip_icon(
                                        "Controls how the final H.264 video is compressed. NVENC uses the NVIDIA GPU; CPU H.264 is slower but works without NVIDIA encoding support.",
                                    )),
                            )
                            .child(
                                div()
                                    .mt_1()
                                    .text_xs()
                                    .text_color(rgb(TEXT_MUTED_COLOR))
                                    .child("Hardware acceleration is tested at startup"),
                            ),
                    )
                    .child(
                        div()
                            .id("main-hardware-encoder-selector")
                            .w(px(270.0))
                            .px_3()
                            .py_2()
                            .rounded_md()
                            .cursor_pointer()
                            .border_1()
                            .border_color(rgb(if self.video_encoder_menu_open {
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
                            .on_click(cx.listener(Self::toggle_video_encoder_menu))
                            .child(video_encoder_label(selected_encoder))
                            .child(if self.video_encoder_menu_open {
                                "⌃"
                            } else {
                                "⌄"
                            }),
                    ),
            )
            .child(
                div()
                    .h(px(64.0))
                    .px_4()
                    .flex()
                    .items_center()
                    .justify_between()
                    .border_b_1()
                    .border_color(rgb(BORDER_COLOR))
                    .child(
                        div()
                            .child(
                                div()
                                    .flex()
                                    .items_center()
                                    .child("NVIDIA hardware decode")
                                    .child(setting_tooltip_icon(
                                        "Uses NVDEC to decode the input on the NVIDIA GPU. It can reduce CPU use, but it is independent of RIFE inference and uses additional GPU resources.",
                                    )),
                            )
                            .child(
                                div()
                                    .mt_1()
                                    .text_xs()
                                    .text_color(rgb(TEXT_MUTED_COLOR))
                                    .child(if nvdec_available {
                                        "Optional NVDEC acceleration for video decoding"
                                    } else {
                                        "Unavailable: NVIDIA FFmpeg decode support not detected"
                                    }),
                            ),
                    )
                    .child(
                        div()
                            .id("main-hardware-nvdec-toggle")
                            .w(px(40.0))
                            .h(px(22.0))
                            .p(px(2.0))
                            .flex()
                            .items_center()
                            .justify_start()
                            .when(self.use_nvdec, |element| element.justify_end())
                            .rounded_full()
                            .cursor_pointer()
                            .border_1()
                            .border_color(rgb(if self.use_nvdec {
                                ACCENT_COLOR
                            } else {
                                BORDER_STRONG_COLOR
                            }))
                            .bg(rgb(if self.use_nvdec {
                                ACCENT_COLOR
                            } else {
                                BORDER_COLOR
                            }))
                            .on_click(cx.listener(Self::toggle_nvdec))
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
                    .child(
                        div()
                            .child(
                                div()
                                    .flex()
                                    .items_center()
                                    .child("Compute device")
                                    .child(setting_tooltip_icon(
                                        "Selects the Vulkan GPU used by the ncnn RIFE backend. This setting does not select the NVENC/NVDEC device or the CUDA backend.",
                                    )),
                            )
                            .child(
                                div()
                                    .mt_1()
                                    .text_xs()
                                    .text_color(rgb(TEXT_MUTED_COLOR))
                                    .child("Vulkan inference device"),
                            ),
                    )
                    .child(
                        div()
                            .id("main-hardware-gpu-selector")
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
                            .on_click(cx.listener(Self::toggle_gpu_menu))
                            .child(
                                div()
                                    .flex_1()
                                    .whitespace_nowrap()
                                    .overflow_hidden()
                                    .text_ellipsis()
                                    .child(gpu_label),
                            )
                            .child(if self.gpu_menu_open { "⌃" } else { "⌄" }),
                    ),
            )
            .when(self.inference_backend_menu_open, |element| {
                element.child(
                    div()
                        .id("main-inference-backend-options")
                        .absolute()
                        .top(px(76.0))
                        .right(px(16.0))
                        .w(px(270.0))
                        .h(px(72.0))
                        .rounded_md()
                        .border_1()
                        .border_color(rgb(BORDER_STRONG_COLOR))
                        .bg(rgb(PANEL_COLOR))
                        .shadow_lg()
                        .child(
                            div()
                                .id("main-inference-backend-vulkan-option")
                                .h(px(36.0))
                                .px_3()
                                .flex()
                                .items_center()
                                .justify_between()
                                .cursor_pointer()
                                .bg(rgb(if !self.cuda_inference_enabled {
                                    SELECTED_COLOR
                                } else {
                                    PANEL_COLOR
                                }))
                                .on_click(cx.listener(|app, _, _, cx| {
                                    app.select_inference_backend(false, cx);
                                    cx.stop_propagation();
                                }))
                                .child("Vulkan / ncnn")
                                .child(if !self.cuda_inference_enabled {
                                    "✓"
                                } else {
                                    ""
                                }),
                        )
                        .child(
                            div()
                                .id("main-inference-backend-cuda-option")
                                .h(px(36.0))
                                .px_3()
                                .flex()
                                .items_center()
                                .justify_between()
                                .when(self.cuda_inference_available, |element| {
                                    element.cursor_pointer().on_click(cx.listener(
                                        |app, _, _, cx| {
                                            app.select_inference_backend(true, cx);
                                            cx.stop_propagation();
                                        },
                                    ))
                                })
                                .bg(rgb(if self.cuda_inference_enabled {
                                    SELECTED_COLOR
                                } else {
                                    PANEL_COLOR
                                }))
                                .text_color(rgb(if self.cuda_inference_available {
                                    TEXT_COLOR
                                } else {
                                    TEXT_MUTED_COLOR
                                }))
                                .child(if self.cuda_inference_available {
                                    "CUDA / PyTorch"
                                } else {
                                    "CUDA / PyTorch (unavailable)"
                                })
                                .child(if self.cuda_inference_enabled {
                                    "✓"
                                } else {
                                    ""
                                }),
                        ),
                )
            })
            .when(self.video_encoder_menu_open, |element| {
                element.child(
                    div()
                        .id("main-hardware-encoder-options")
                        .absolute()
                        .top(px(140.0))
                        .right(px(16.0))
                        .w(px(270.0))
                        .h(px(108.0))
                        .rounded_md()
                        .border_1()
                        .border_color(rgb(BORDER_STRONG_COLOR))
                        .bg(rgb(PANEL_COLOR))
                        .shadow_lg()
                        .children(self.video_encoders.iter().copied().enumerate().map(
                            |(index, encoder)| {
                                let selected = index == self.selected_video_encoder;
                                div()
                                    .id(("main-hardware-encoder-option", index))
                                    .h(px(36.0))
                                    .px_3()
                                    .flex()
                                    .items_center()
                                    .justify_between()
                                    .cursor_pointer()
                                    .bg(rgb(if selected {
                                        SELECTED_COLOR
                                    } else {
                                        PANEL_COLOR
                                    }))
                                    .on_click(cx.listener(move |app, _, _, cx| {
                                        app.select_video_encoder(index, cx)
                                    }))
                                    .child(video_encoder_label(encoder))
                                    .child(if selected { "✓" } else { "" })
                            },
                        )),
                )
            })
            .when(self.gpu_menu_open, |element| {
                element.child(
                    div()
                        .id("main-hardware-gpu-options")
                        .absolute()
                        .top(px(268.0))
                        .right(px(16.0))
                        .w(px(270.0))
                        .h(px(144.0))
                        .overflow_y_scroll()
                        .rounded_md()
                        .border_1()
                        .border_color(rgb(BORDER_STRONG_COLOR))
                        .bg(rgb(PANEL_COLOR))
                        .shadow_lg()
                        .children(self.gpu_names.iter().enumerate().map(|(index, name)| {
                            let selected = index == self.selected_gpu_index;
                            div()
                                .id(("main-hardware-gpu-option", index))
                                .h(px(36.0))
                                .px_3()
                                .flex()
                                .items_center()
                                .justify_between()
                                .cursor_pointer()
                                .bg(rgb(if selected {
                                    SELECTED_COLOR
                                } else {
                                    PANEL_COLOR
                                }))
                                .on_click(
                                    cx.listener(move |app, _, _, cx| app.select_gpu(index, cx)),
                                )
                                .child(name.clone())
                                .child(if selected { "✓" } else { "" })
                        })),
                )
            })
    }

    fn render_media_tab(&mut self, cx: &mut Context<Self>) -> impl IntoElement {
        assert!(
            self.video_encoders.len() <= 3,
            "encoder list must remain bounded"
        );
        assert!(
            !self.running || self.cancellation.is_some(),
            "running media settings require a cancellation handle"
        );
        div()
            .w_full()
            .rounded_lg()
            .border_1()
            .border_color(rgb(BORDER_COLOR))
            .bg(rgb(PANEL_COLOR))
            .child(
                div()
                    .px_4()
                    .py_4()
                    .border_b_1()
                    .border_color(rgb(BORDER_COLOR))
                    .child(div().font_family("JetBrains Mono").child("Media streams"))
                    .child(
                        div()
                            .mt_1()
                            .text_xs()
                            .text_color(rgb(TEXT_MUTED_COLOR))
                            .child("Choose which source streams are copied into this encoding."),
                    ),
            )
            .children(
                [
                    (
                        "Audio",
                        "preserve-audio-main",
                        self.preserve_audio,
                        "Copies the primary audio stream into the MKV without re-encoding. Disable it to create a silent video output.",
                    ),
                    (
                        "Subtitles",
                        "preserve-subtitles-main",
                        self.preserve_subtitles,
                        "Copies subtitle streams into the MKV without re-encoding. It does not burn subtitles into the video image.",
                    ),
                    (
                        "Metadata",
                        "preserve-metadata-main",
                        self.preserve_metadata,
                        "Copies compatible stream and container metadata such as language and title tags. It does not copy the video or audio data itself.",
                    ),
                ]
                .into_iter()
                .map(|(label, id, enabled, description)| {
                    let toggle = div()
                        .id(id)
                        .px_3()
                        .py_1()
                        .rounded_full()
                        .cursor_pointer()
                        .border_1()
                        .border_color(rgb(if enabled { ACCENT_COLOR } else { BORDER_COLOR }))
                        .bg(rgb(if enabled { ACCENT_COLOR } else { PANEL_COLOR }))
                        .child(if enabled { "✓ Keep" } else { "Do not keep" });
                    let toggle = match label {
                        "Audio" => toggle.on_click(cx.listener(Self::toggle_preserve_audio)),
                        "Subtitles" => {
                            toggle.on_click(cx.listener(Self::toggle_preserve_subtitles))
                        }
                        "Metadata" => toggle.on_click(cx.listener(Self::toggle_preserve_metadata)),
                        _ => unreachable!("media tab label must be known"),
                    };
                    div()
                        .h(px(56.0))
                        .px_4()
                        .flex()
                        .items_center()
                        .justify_between()
                        .border_b_1()
                        .border_color(rgb(BORDER_COLOR))
                        .child(
                            div()
                                .flex()
                                .items_center()
                                .child(label)
                                .child(setting_tooltip_icon(description)),
                        )
                        .child(toggle)
                }),
            )
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
        assert!(
            self.selected_video_encoder <= self.video_encoders.len(),
            "rendered encoder index must be bounded"
        );
        assert!(
            self.video_encoders.len() <= 3,
            "encoder list must remain bounded"
        );
        assert!(
            !self.cuda_inference_enabled || self.cuda_inference_available,
            "rendered CUDA inference must be available when enabled"
        );
        assert!(
            !self.cuda_inference_unavailable_reason.is_empty(),
            "CUDA availability reason must not be empty"
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
        let stage_speed_label = format!(
            "I {:.2} · D {:.2} · E {:.2}",
            self.performance.inference_fps,
            self.performance.decode_fps,
            self.performance.encode_fps,
        );
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
        let can_start = !self.running
            && self.input_path.is_some()
            && !self.gpu_names.is_empty()
            && !self.video_encoders.is_empty()
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
                            .flex_1()
                            .flex()
                            .justify_center()
                            .child(
                                div()
                                    .h(px(34.0))
                                    .flex()
                                    .items_center()
                                    .rounded_full()
                                    .border_1()
                                    .border_color(rgb(BORDER_COLOR))
                                    .bg(rgb(PANEL_COLOR))
                                    .child(
                                        div()
                                            .id("encode-page")
                                            .h(px(30.0))
                                            .px_4()
                                            .rounded_full()
                                            .flex()
                                            .items_center()
                                            .gap_2()
                                            .cursor_pointer()
                                            .bg(rgb(if !self.settings_page {
                                                SELECTED_COLOR
                                            } else {
                                                PANEL_COLOR
                                            }))
                                            .hover(|style| style.bg(rgb(PANEL_HOVER_COLOR)))
                                            .on_click(cx.listener(Self::show_encode_page))
                                            .child(header_nav_icon(
                                                HEADER_NAV_ICON_PATH_PAUSE,
                                                if !self.settings_page {
                                                    TEXT_COLOR
                                                } else {
                                                    TEXT_MUTED_COLOR
                                                },
                                            ))
                                            .child("Encode"),
                                    )
                                    .child(
                                        div()
                                            .id("settings-page")
                                            .h(px(30.0))
                                            .px_4()
                                            .rounded_full()
                                            .flex()
                                            .items_center()
                                            .gap_2()
                                            .cursor_pointer()
                                            .bg(rgb(if self.settings_page {
                                                SELECTED_COLOR
                                            } else {
                                                PANEL_COLOR
                                            }))
                                            .hover(|style| style.bg(rgb(PANEL_HOVER_COLOR)))
                                            .on_click(cx.listener(Self::show_settings_page))
                                            .child(header_nav_icon(
                                                HEADER_NAV_ICON_PATH_SETTINGS,
                                                if self.settings_page {
                                                    TEXT_COLOR
                                                } else {
                                                    TEXT_MUTED_COLOR
                                                },
                                            ))
                                            .child("Settings"),
                                    ),
                            ),
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
                                    .rounded_full()
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
                                    .rounded_full()
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
            .child(if self.settings_page {
                div()
                    .id("settings-scroll")
                    .flex_1()
                    .overflow_y_scroll()
                    .p_5()
                    .flex()
                    .flex_col()
                    .items_center()
                    .gap_4()
                    .child(
                        div()
                            .mb_3()
                            .text_xs()
                            .text_color(rgb(TEXT_SUBTLE_COLOR))
                            .child("SETTINGS"),
                    )
                    .child(
                        div()
                            .w(px(640.0))
                            .rounded_lg()
                            .border_1()
                            .border_color(rgb(BORDER_COLOR))
                            .bg(rgb(PANEL_COLOR))
                            .child(
                                div()
                                    .px_4()
                                    .py_4()
                                    .border_b_1()
                                    .border_color(rgb(BORDER_COLOR))
                                    .child(
                                        div()
                                            .font_family("JetBrains Mono")
                                            .child("Application behavior"),
                                    )
                                    .child(
                                        div()
                                            .mt_1()
                                            .text_xs()
                                            .text_color(rgb(TEXT_MUTED_COLOR))
                                            .child("Control how Interpolate behaves while a job is running."),
                                    ),
                            )
                            .child(
                                div()
                                    .h(px(76.0))
                                    .px_4()
                                    .flex()
                                    .items_center()
                                    .justify_between()
                                    .child(
                                        div()
                                            .child(
                                                div()
                                                    .flex()
                                                    .items_center()
                                                    .child("Keep running in background")
                                                    .child(setting_tooltip_icon(
                                                        "When enabled, closing the window hides the interface while the active job continues. Use the system-tray menu to restore the window or cancel the job.",
                                                    )),
                                            )
                                            .child(
                                                div()
                                                    .mt_1()
                                                    .text_xs()
                                                    .text_color(rgb(TEXT_MUTED_COLOR))
                                                    .child(if self.tray_controller.is_some() {
                                                        "Closing the window minimizes the job to the system tray"
                                                    } else {
                                                        "System tray unavailable on this desktop"
                                                    }),
                                            ),
                                    )
                                    .child(
                                        div()
                                            .id("settings-background-mode")
                                            .w(px(36.0))
                                            .h(px(20.0))
                                            .p(px(2.0))
                                            .flex()
                                            .justify_end()
                                            .when(!self.keep_running_background, |element| {
                                                element.justify_start()
                                            })
                                            .items_center()
                                            .rounded_full()
                                            .when(self.tray_controller.is_some(), |element| {
                                                element
                                                    .cursor_pointer()
                                                    .on_click(cx.listener(
                                                        Self::toggle_background_mode,
                                                    ))
                                            })
                                            .bg(rgb(if self.keep_running_background {
                                                ACCENT_COLOR
                                            } else {
                                                BORDER_STRONG_COLOR
                                            }))
                                            .child(
                                                div()
                                                    .size(px(16.0))
                                                    .rounded_full()
                                                    .bg(rgb(TEXT_COLOR)),
                                            ),
                                    ),
                            ),
                    )
                                        .child(
                        div()
                            .text_xs()
                            .text_color(rgb(TEXT_MUTED_COLOR))
                            .child(if self.tray_controller.is_some() {
                                "Background mode is enabled by default when a compatible system tray is available."
                            } else {
                                "Background mode requires a freedesktop StatusNotifierItem system tray."
                            }),
                    )
            } else {
                div()
                    .id("encode-scroll")
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
                                    .w_full()
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
                                            .mb_3()
                                            .h(px(36.0))
                                            .w_full()
                                            .flex()
                                            .items_end()
                                            .border_b_1()
                                            .border_color(rgb(BORDER_COLOR))
                                            .bg(rgb(BACKGROUND_COLOR))
                                            .children(
                                                [
                                                    (MainPageTab::Interpolation, "Interpolation"),
                                                    (MainPageTab::Hardware, "Hardware"),
                                                    (MainPageTab::Encoding, "Encoding"),
                                                    (MainPageTab::Media, "Media"),
                                                ]
                                                .into_iter()
                                                .enumerate()
                                                .map(|(tab_index, (tab, label))| {
                                                    let selected = self.main_page_tab == tab;
                                                    div()
                                                        .id(("main-page-tab", tab_index))
                                                        .h(px(36.0))
                                                        .px_4()
                                                        .flex()
                                                        .items_center()
                                                        .justify_center()
                                                        .cursor_pointer()
                                                        .border_1()
                                                        .border_color(rgb(if selected {
                                                            BORDER_COLOR
                                                        } else {
                                                            BACKGROUND_COLOR
                                                        }))
                                                        .bg(rgb(if selected {
                                                            PANEL_COLOR
                                                        } else {
                                                            BACKGROUND_COLOR
                                                        }))
                                                        .text_color(rgb(if selected {
                                                            TEXT_COLOR
                                                        } else {
                                                            TEXT_MUTED_COLOR
                                                        }))
                                                        .hover(|style| style.bg(rgb(PANEL_HOVER_COLOR)))
                                                        .on_click(cx.listener(move |app, _, _, cx| {
                                                            app.select_main_page_tab(tab, cx)
                                                        }))
                                                        .child(label)
                                                }),
                                            ),
                                    )
                                    .child(
                                        div()
                                            .when(self.main_page_tab != MainPageTab::Interpolation, |element| {
                                                element.hidden()
                                            })
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
                                                        div()
                                                            .child(
                                                                div()
                                                                    .flex()
                                                                    .items_center()
                                                                    .child("Content preset")
                                                                    .child(setting_tooltip_icon(
                                                                        "Movie is tuned for live-action footage. Anime enables cadence protection and automatically favors half-scale flow for 4K sources.",
                                                                    )),
                                                            )
                                                            .child(
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
                                                        div()
                                                            .child(
                                                                div()
                                                                    .flex()
                                                                    .items_center()
                                                                    .child("Target frame rate")
                                                                    .child(setting_tooltip_icon(
                                                                        "The output frame rate. RIFE creates intermediate frames only as needed to reach this rate; it does not change the source resolution.",
                                                                    )),
                                                            )
                                                            .child(
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
                                                        div()
                                                            .child(
                                                                div()
                                                                    .flex()
                                                                    .items_center()
                                                                    .child("Scene change protection")
                                                                    .child(setting_tooltip_icon(
                                                                        "Detects hard cuts and avoids inventing motion across them. The frame before a cut is duplicated instead, preventing flashes and blended scenes.",
                                                                    )),
                                                            )
                                                            .child(
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
                                                        div()
                                                            .child(
                                                                div()
                                                                    .flex()
                                                                    .items_center()
                                                                    .child("Half-scale UHD flow")
                                                                    .child(setting_tooltip_icon(
                                                                        "Runs the RIFE model at half resolution and scales the result back up. This greatly reduces GPU memory use for UHD video, with some loss of fine detail.",
                                                                    )),
                                                            )
                                                            .child(
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

                                            ,
                                    )
                                    .when(self.main_page_tab == MainPageTab::Hardware, |element| {
                                        element.child(self.render_hardware_tab(cx))
                                    })
                                    .when(self.main_page_tab == MainPageTab::Encoding, |element| {
                                        element.child(self.render_encoding_tab(window, cx))
                                    })
                                    .when(self.main_page_tab == MainPageTab::Media, |element| {
                                        element.child(self.render_media_tab(cx))
                                    })
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
                                    )
                            )
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
                    )
            })
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
                                    .child(stage_speed_label)
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
        if let Some(worker) = self.worker.take()
            && worker.join().is_err()
        {
            eprintln!("processing worker panicked during shutdown");
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

pub fn run() {
    assert!(WINDOW_WIDTH_PIXELS > 0.0, "window width must be positive");
    assert!(WINDOW_HEIGHT_PIXELS > 0.0, "window height must be positive");
    Application::new()
        .with_assets(BundledAssets)
        .run(|context: &mut App| {
            let (tray_controller, tray_commands, tray_error) = match TrayController::start() {
                Ok((tray_controller, tray_commands)) => {
                    (Some(tray_controller), Some(tray_commands), None)
                }
                Err(error) => (None, None, Some(error)),
            };
            let window_size = size(px(WINDOW_WIDTH_PIXELS), px(WINDOW_HEIGHT_PIXELS));
            let window_bounds = Bounds::centered(None, window_size, context);
            let window_result = context.open_window(
                WindowOptions {
                    window_bounds: Some(WindowBounds::Windowed(window_bounds)),
                    ..WindowOptions::default()
                },
                move |window, context| {
                    let window_handle = window.window_handle();
                    let app = context.new(|cx| {
                        InterpolateApp::new(
                            cx,
                            tray_controller,
                            tray_commands,
                            window_handle,
                            tray_error,
                        )
                    });
                    let weak_app = app.downgrade();
                    window.on_window_should_close(context, move |window, context| {
                        weak_app
                            .update(context, |app, cx| {
                                if app.running
                                    && app.keep_running_background
                                    && app.tray_controller.is_some()
                                {
                                    window.minimize_window();
                                    app.status = "Running in background · use the tray to restore"
                                        .to_owned();
                                    cx.notify();
                                    false
                                } else {
                                    true
                                }
                            })
                            .unwrap_or(true)
                    });
                    app
                },
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
