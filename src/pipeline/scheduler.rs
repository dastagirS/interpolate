use super::{
    JobConfiguration, JobUpdate, OUTPUT_FRAME_COUNT_MAX, PROGRESS_INTERVAL, VideoMetadata,
    checked_frame_size, send_update,
};
use crate::{backend::Backend, cadence::CadenceDiagnostics};
use std::{
    io::Write,
    process::ChildStdin,
    sync::{
        atomic::{AtomicBool, Ordering},
        mpsc::SyncSender,
    },
    time::Instant,
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
    backend: &'a mut Backend,
    encoder_input: &'a mut ChildStdin,
    frame_output: Vec<u8>,
    pub(super) cadence_diagnostics: CadenceDiagnostics,
}

impl<'a> OutputScheduler<'a> {
    pub(super) fn new(
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
