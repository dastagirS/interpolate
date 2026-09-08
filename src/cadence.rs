const RGB_CHANNEL_COUNT: usize = 3;
const FRAME_SAMPLE_STEP: usize = 1;
const DUPLICATE_MEAN_DIFFERENCE_MAX: f64 = 0.006;
const DUPLICATE_CHANGED_PIXEL_RATIO_MAX: f64 = 0.005;
const DUPLICATE_TILE_DIFFERENCE_MAX: f64 = 0.01;
const DUPLICATE_PIXEL_DIFFERENCE_MIN: u32 = 4 * 256;
const DIFFERENCE_TILE_COLUMN_COUNT: usize = 8;
const DIFFERENCE_TILE_ROW_COUNT: usize = 8;
const DIFFERENCE_TILE_COUNT: usize = DIFFERENCE_TILE_COLUMN_COUNT * DIFFERENCE_TILE_ROW_COUNT;
const CADENCE_RUN_FRAME_COUNT_MIN: u64 = 2;
const CADENCE_RUN_FRAME_COUNT_MAX: u64 = 3;
const CADENCE_RUN_FRAME_COUNT_INPUT_MAX: u64 = 4;

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

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct FrameDifference {
    pub(crate) mean: f64,
    pub(crate) changed_pixel_ratio: f64,
    pub(crate) tile_mean_max: f64,
}

pub(crate) fn measure_frame_difference_rgb24(
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
    assert_eq!(
        frame_before.len(),
        row_stride * height,
        "difference frame size must match its dimensions"
    );
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

pub(crate) fn is_smoothable_cadence_run(cadence_run_frame_count: u64) -> bool {
    assert!(
        cadence_run_frame_count > 0,
        "cadence run must contain at least one frame"
    );
    assert!(
        cadence_run_frame_count <= CADENCE_RUN_FRAME_COUNT_INPUT_MAX,
        "cadence decision input must remain bounded"
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

pub(crate) fn is_confident_duplicate(difference: FrameDifference) -> bool {
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

#[cfg(test)]
mod tests {
    use super::*;

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
        let changed_pixel_index = RGB_CHANNEL_COUNT;
        frame_after[changed_pixel_index..changed_pixel_index + RGB_CHANNEL_COUNT].fill(255);
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
}
