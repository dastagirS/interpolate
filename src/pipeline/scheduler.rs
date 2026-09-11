use super::{
    JobConfiguration, JobUpdate, OUTPUT_FRAME_COUNT_MAX, PROGRESS_INTERVAL, VideoMetadata,
    checked_frame_size, send_update,
};
use crate::{backend::InferenceEngine, cadence::CadenceDiagnostics};
use std::{
    io::Write,
    process::ChildStdin,
    sync::{
        atomic::{AtomicBool, Ordering},
        mpsc::SyncSender,
    },
    time::{Duration, Instant},
};

pub(super) struct OutputScheduler<'a> {
    factor_num: u128,
    factor_den: u128,
    frame_count_estimate: u64,
    output_index: u64,
    started_at: Instant,
    progress_at: Instant,
    metadata: &'a VideoMetadata,
    cancelled: &'a AtomicBool,
    updates: &'a SyncSender<JobUpdate>,
    backend: &'a mut InferenceEngine,
    encoder_input: &'a mut ChildStdin,
    frame_output: Vec<u8>,
    inference_count: u64,
    decode_count: u64,
    encode_count: u64,
    inference_elapsed: Duration,
    decode_elapsed: Duration,
    encode_elapsed: Duration,
    pub(super) cadence_diagnostics: CadenceDiagnostics,
}

impl<'a> OutputScheduler<'a> {
    pub(super) fn new(
        configuration: &JobConfiguration,
        metadata: &'a VideoMetadata,
        cancelled: &'a AtomicBool,
        updates: &'a SyncSender<JobUpdate>,
        backend: &'a mut InferenceEngine,
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
            inference_count: 0,
            decode_count: 0,
            encode_count: 0,
            inference_elapsed: Duration::ZERO,
            decode_elapsed: Duration::ZERO,
            encode_elapsed: Duration::ZERO,
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

    pub(super) fn write_span(
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
                let inference_started_at = Instant::now();
                let inference_result = self.backend.interpolate_rgb24(
                    frame_before,
                    frame_after,
                    self.metadata.width,
                    self.metadata.height,
                    timestep,
                    &mut self.frame_output,
                );
                self.inference_elapsed += inference_started_at.elapsed();
                self.inference_count = self.inference_count.saturating_add(1);
                inference_result?;
                self.write_output_frame()?;
            } else {
                self.write_frame(frame_before)?;
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

    fn write_output_frame(&mut self) -> Result<(), String> {
        assert!(
            !self.frame_output.is_empty(),
            "output frame must contain bytes"
        );
        assert!(
            self.encode_count < OUTPUT_FRAME_COUNT_MAX,
            "encoded frames must remain bounded"
        );
        let write_started_at = Instant::now();
        self.encoder_input
            .write_all(&self.frame_output)
            .map_err(|error| format!("failed to send interpolated frame to encoder: {error}"))?;
        self.encode_elapsed += write_started_at.elapsed();
        self.encode_count = self.encode_count.saturating_add(1);
        assert!(self.encode_count > 0, "successful writes must be counted");
        assert!(
            self.encode_elapsed >= Duration::ZERO,
            "encode duration must be valid"
        );
        Ok(())
    }

    fn write_frame(&mut self, frame: &[u8]) -> Result<(), String> {
        assert_eq!(
            frame.len(),
            self.frame_output.len(),
            "frame size must remain stable"
        );
        assert!(
            self.encode_count < OUTPUT_FRAME_COUNT_MAX,
            "encoded frames must remain bounded"
        );
        let write_started_at = Instant::now();
        self.encoder_input
            .write_all(frame)
            .map_err(|error| format!("failed to send source frame to encoder: {error}"))?;
        self.encode_elapsed += write_started_at.elapsed();
        self.encode_count = self.encode_count.saturating_add(1);
        assert!(self.encode_count > 0, "successful writes must be counted");
        assert!(
            self.encode_elapsed >= Duration::ZERO,
            "encode duration must be valid"
        );
        Ok(())
    }

    pub(super) fn record_decode(&mut self, elapsed: Duration) {
        assert!(elapsed >= Duration::ZERO, "decode duration must be valid");
        assert!(
            self.decode_count < OUTPUT_FRAME_COUNT_MAX,
            "decoded frames must remain bounded"
        );
        self.decode_elapsed += elapsed;
        self.decode_count = self.decode_count.saturating_add(1);
        assert!(self.decode_count > 0, "successful reads must be counted");
        assert!(
            self.decode_elapsed >= Duration::ZERO,
            "decode duration must remain valid"
        );
    }

    fn stage_fps(frame_count: u64, elapsed: Duration) -> f64 {
        assert!(elapsed >= Duration::ZERO, "stage duration must be valid");
        assert!(
            frame_count <= OUTPUT_FRAME_COUNT_MAX,
            "stage frame count must remain bounded"
        );
        if elapsed.is_zero() {
            return 0.0;
        }
        frame_count as f64 / elapsed.as_secs_f64()
    }

    fn performance(&self) -> super::PerformanceDiagnostics {
        assert!(
            self.output_index < OUTPUT_FRAME_COUNT_MAX,
            "output count must remain bounded"
        );
        assert!(
            self.inference_count <= self.output_index,
            "inference count cannot exceed output count"
        );
        super::PerformanceDiagnostics {
            inference_fps: Self::stage_fps(self.inference_count, self.inference_elapsed),
            decode_fps: Self::stage_fps(self.decode_count, self.decode_elapsed),
            encode_fps: Self::stage_fps(self.encode_count, self.encode_elapsed),
        }
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
                    performance: self.performance(),
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

    pub(super) fn finish(&mut self) -> Result<(), String> {
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
                performance: self.performance(),
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
