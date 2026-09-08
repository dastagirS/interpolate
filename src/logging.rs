use std::{
    collections::VecDeque,
    fs::{self, File, OpenOptions},
    io::{Read, Write},
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
    thread::{self, JoinHandle},
    time::{SystemTime, UNIX_EPOCH},
};

const LOG_FILE_COUNT_MAX: usize = 5;
const LOG_FILE_SIZE_BYTES_MAX: usize = 1024 * 1024;
const LOG_LINE_COUNT_MAX: usize = 200;
const LOG_READ_SIZE_BYTES: usize = 4096;
const LOG_READ_COUNT_MAX: usize = 1_048_576;
const LOG_SOURCE_SIZE_MAX: usize = 32;
const LOG_FILE_NAME: &str = "interpolate.log";

struct LogState {
    file: File,
    file_size_bytes: usize,
    recent_lines: VecDeque<String>,
}

#[derive(Clone)]
pub struct JobLog {
    path: Arc<PathBuf>,
    state: Arc<Mutex<LogState>>,
}

impl JobLog {
    pub fn create() -> Result<Self, String> {
        assert!(LOG_FILE_COUNT_MAX > 1, "log rotation must retain history");
        assert!(
            LOG_FILE_SIZE_BYTES_MAX > LOG_READ_SIZE_BYTES,
            "log file must hold multiple reads"
        );
        let directory = log_directory();
        fs::create_dir_all(&directory).map_err(|error| {
            format!(
                "failed to create log directory {}: {error}",
                directory.display()
            )
        })?;
        rotate_logs(&directory)?;
        let path = directory.join(LOG_FILE_NAME);
        let mut options = OpenOptions::new();
        options.create(true).write(true).truncate(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            options.mode(0o600);
        }
        let file = options.open(&path).map_err(|error| {
            format!("failed to open application log {}: {error}", path.display())
        })?;
        let log = Self {
            path: Arc::new(path),
            state: Arc::new(Mutex::new(LogState {
                file,
                file_size_bytes: 0,
                recent_lines: VecDeque::with_capacity(LOG_LINE_COUNT_MAX),
            })),
        };
        log.write("application", "job log started")?;
        assert!(log.path.is_file(), "new log file must exist");
        assert!(LOG_LINE_COUNT_MAX > 0, "recent log storage must be bounded");
        Ok(log)
    }

    pub fn path(&self) -> &Path {
        assert!(
            !self.path.as_os_str().is_empty(),
            "log path must not be empty"
        );
        assert!(
            LOG_FILE_NAME.len() < 128,
            "log filename must remain bounded"
        );
        self.path.as_path()
    }

    pub fn write(&self, source: &str, message: &str) -> Result<(), String> {
        assert!(!source.is_empty(), "log source must not be empty");
        assert!(
            source.len() <= LOG_SOURCE_SIZE_MAX,
            "log source must remain bounded"
        );
        let timestamp_milliseconds = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis();
        let message = message.trim_end_matches(['\r', '\n']);
        let line = format!("{timestamp_milliseconds} [{source}] {message}\n");
        let mut state = self
            .state
            .lock()
            .map_err(|_| "application log lock was poisoned".to_owned())?;
        if state.recent_lines.len() == LOG_LINE_COUNT_MAX {
            state.recent_lines.pop_front();
        }
        state.recent_lines.push_back(line.trim_end().to_owned());

        let remaining_size = LOG_FILE_SIZE_BYTES_MAX.saturating_sub(state.file_size_bytes);
        if remaining_size > 0 {
            let bytes = line.as_bytes();
            let write_size = bytes.len().min(remaining_size);
            state
                .file
                .write_all(&bytes[..write_size])
                .map_err(|error| format!("failed to write application log: {error}"))?;
            state.file_size_bytes = state
                .file_size_bytes
                .checked_add(write_size)
                .ok_or_else(|| "application log size overflowed".to_owned())?;
        }
        assert!(
            state.file_size_bytes <= LOG_FILE_SIZE_BYTES_MAX,
            "log file must remain bounded"
        );
        assert!(
            state.recent_lines.len() <= LOG_LINE_COUNT_MAX,
            "recent logs must remain bounded"
        );
        Ok(())
    }

    pub fn recent_summary(&self) -> String {
        assert!(LOG_LINE_COUNT_MAX > 0, "recent log storage must be bounded");
        assert!(LOG_FILE_SIZE_BYTES_MAX > 0, "log file size must be bounded");
        let summary = match self.state.lock() {
            Ok(state) => state
                .recent_lines
                .iter()
                .rev()
                .take(8)
                .cloned()
                .collect::<Vec<_>>()
                .into_iter()
                .rev()
                .collect::<Vec<_>>()
                .join("\n"),
            Err(_) => "application log is unavailable".to_owned(),
        };
        assert!(
            summary.len() <= LOG_LINE_COUNT_MAX * (LOG_READ_SIZE_BYTES + 128),
            "log summary must remain bounded"
        );
        assert!(
            LOG_LINE_COUNT_MAX >= 8,
            "summary limit must fit recent storage"
        );
        summary
    }

    pub fn spawn_reader<R>(
        &self,
        source: &'static str,
        mut reader: R,
    ) -> Result<JoinHandle<Result<(), String>>, String>
    where
        R: Read + Send + 'static,
    {
        assert!(!source.is_empty(), "log source must not be empty");
        assert!(
            source.len() <= LOG_SOURCE_SIZE_MAX,
            "log source must remain bounded"
        );
        let log = self.clone();
        thread::Builder::new()
            .name(format!("{source}-log-reader"))
            .spawn(move || {
                let mut buffer = [0_u8; LOG_READ_SIZE_BYTES];
                let mut logging_error = None;
                for _ in 0..LOG_READ_COUNT_MAX {
                    let count = reader
                        .read(&mut buffer)
                        .map_err(|error| format!("failed to read {source} diagnostics: {error}"))?;
                    if count == 0 {
                        return match logging_error {
                            Some(error) => Err(error),
                            None => Ok(()),
                        };
                    }
                    let text = String::from_utf8_lossy(&buffer[..count]);
                    for line in text
                        .split_terminator(['\r', '\n'])
                        .take(LOG_READ_SIZE_BYTES)
                    {
                        if !line.is_empty()
                            && let Err(error) = log.write(source, line)
                            && logging_error.is_none()
                        {
                            logging_error = Some(error);
                        }
                    }
                }
                Err(format!(
                    "{source} diagnostics exceeded the read safety limit"
                ))
            })
            .map_err(|error| format!("failed to start {source} log reader: {error}"))
    }
}

fn log_directory() -> PathBuf {
    assert!(
        LOG_FILE_NAME.len() < 128,
        "log filename must remain bounded"
    );
    assert!(LOG_FILE_COUNT_MAX > 1, "log rotation must retain history");
    if let Some(state_home) = std::env::var_os("XDG_STATE_HOME") {
        let path = PathBuf::from(state_home);
        if path.is_absolute() {
            return path.join("interpolate");
        }
    }
    if let Some(home) = std::env::var_os("HOME") {
        let path = PathBuf::from(home);
        if path.is_absolute() {
            return path.join(".local/state/interpolate");
        }
    }
    std::env::temp_dir().join(format!("interpolate-{}", std::process::id()))
}

fn rotate_logs(directory: &Path) -> Result<(), String> {
    assert!(directory.is_absolute(), "log directory must be absolute");
    assert!(LOG_FILE_COUNT_MAX > 1, "log rotation must retain history");
    for target_index in (1..LOG_FILE_COUNT_MAX).rev() {
        let source = if target_index == 1 {
            directory.join(LOG_FILE_NAME)
        } else {
            directory.join(format!("{LOG_FILE_NAME}.{}", target_index - 1))
        };
        let target = directory.join(format!("{LOG_FILE_NAME}.{target_index}"));
        if target.exists() {
            fs::remove_file(&target).map_err(|error| {
                format!("failed to remove old log {}: {error}", target.display())
            })?;
        }
        if source.exists() {
            fs::rename(&source, &target).map_err(|error| {
                format!(
                    "failed to rotate log {} to {}: {error}",
                    source.display(),
                    target.display()
                )
            })?;
        }
    }
    assert!(
        !directory.join(LOG_FILE_NAME).exists(),
        "current log path must be available after rotation"
    );
    assert!(
        LOG_FILE_COUNT_MAX <= 16,
        "log file count must remain bounded"
    );
    Ok(())
}
