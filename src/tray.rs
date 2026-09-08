use ksni::{Tray, blocking::TrayMethods, menu::StandardItem};
use std::sync::mpsc::{Receiver, SyncSender, sync_channel};

const TRAY_COMMAND_COUNT_MAX: usize = 3;
const TRAY_COMMAND_QUEUE_SIZE: usize = 8;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TrayCommand {
    Show,
    Cancel,
    Quit,
}

struct InterpolateTray {
    command_sender: SyncSender<TrayCommand>,
    running: bool,
}

impl Tray for InterpolateTray {
    fn id(&self) -> String {
        assert!(
            TRAY_COMMAND_COUNT_MAX > 0,
            "tray command limit must be positive"
        );
        assert!(
            env!("CARGO_PKG_NAME").len() < 128,
            "tray identifier must remain bounded"
        );
        env!("CARGO_PKG_NAME").to_owned()
    }

    fn title(&self) -> String {
        assert!(
            TRAY_COMMAND_COUNT_MAX > 0,
            "tray command limit must be positive"
        );
        assert!(
            env!("CARGO_PKG_NAME").len() < 128,
            "tray title must remain bounded"
        );
        if self.running {
            "Interpolate · Processing".to_owned()
        } else {
            "Interpolate".to_owned()
        }
    }

    fn icon_name(&self) -> String {
        assert!(
            TRAY_COMMAND_COUNT_MAX > 0,
            "tray command limit must be positive"
        );
        assert!(
            env!("CARGO_PKG_NAME").len() < 128,
            "tray icon name must remain bounded"
        );
        "video-x-generic".to_owned()
    }

    fn activate(&mut self, _x: i32, _y: i32) {
        assert!(
            TRAY_COMMAND_COUNT_MAX > 0,
            "tray command limit must be positive"
        );
        assert!(
            env!("CARGO_PKG_NAME").len() < 128,
            "tray identifier must remain bounded"
        );
        let _ = self.command_sender.try_send(TrayCommand::Show);
        assert!(
            TRAY_COMMAND_COUNT_MAX >= 3,
            "all tray commands must remain supported"
        );
        assert!(
            env!("CARGO_PKG_NAME").len() < 128,
            "tray identifier must remain bounded"
        );
    }

    fn menu(&self) -> Vec<ksni::MenuItem<Self>> {
        assert!(
            TRAY_COMMAND_COUNT_MAX >= 3,
            "all tray commands must remain supported"
        );
        assert!(
            env!("CARGO_PKG_NAME").len() < 128,
            "tray identifier must remain bounded"
        );
        let show_sender = self.command_sender.clone();
        let cancel_sender = self.command_sender.clone();
        let quit_sender = self.command_sender.clone();
        let items = vec![
            StandardItem {
                label: "Show Interpolate".to_owned(),
                icon_name: "window-restore".to_owned(),
                activate: Box::new(move |_| {
                    let _ = show_sender.try_send(TrayCommand::Show);
                }),
                ..Default::default()
            }
            .into(),
            StandardItem {
                label: "Cancel job".to_owned(),
                icon_name: "process-stop".to_owned(),
                enabled: self.running,
                activate: Box::new(move |_| {
                    let _ = cancel_sender.try_send(TrayCommand::Cancel);
                }),
                ..Default::default()
            }
            .into(),
            StandardItem {
                label: "Quit".to_owned(),
                icon_name: "application-exit".to_owned(),
                activate: Box::new(move |_| {
                    let _ = quit_sender.try_send(TrayCommand::Quit);
                }),
                ..Default::default()
            }
            .into(),
        ];
        assert_eq!(
            items.len(),
            TRAY_COMMAND_COUNT_MAX,
            "tray menu must remain bounded"
        );
        assert!(!items.is_empty(), "tray menu must not be empty");
        items
    }
}

pub struct TrayController {
    handle: ksni::blocking::Handle<InterpolateTray>,
}

impl TrayController {
    pub fn start() -> Result<(Self, Receiver<TrayCommand>), String> {
        assert!(
            TRAY_COMMAND_COUNT_MAX >= 3,
            "all tray commands must remain supported"
        );
        assert!(
            env!("CARGO_PKG_NAME").len() < 128,
            "tray identifier must remain bounded"
        );
        let (command_sender, command_receiver) = sync_channel(TRAY_COMMAND_QUEUE_SIZE);
        let handle = InterpolateTray {
            command_sender,
            running: false,
        }
        .spawn()
        .map_err(|error| format!("system tray is unavailable: {error}"))?;
        if handle.is_closed() {
            return Err("system tray stopped during startup".to_owned());
        }
        assert!(
            TRAY_COMMAND_COUNT_MAX > 0,
            "tray command limit must remain valid"
        );
        assert!(
            env!("CARGO_PKG_NAME").len() < 128,
            "tray identifier must remain valid"
        );
        Ok((Self { handle }, command_receiver))
    }

    pub fn set_running(&self, running: bool) -> Result<(), String> {
        assert!(
            TRAY_COMMAND_COUNT_MAX >= 3,
            "all tray commands must remain supported"
        );
        if self.handle.is_closed() {
            return Err("system tray stopped unexpectedly".to_owned());
        }
        self.handle
            .update(|tray| tray.running = running)
            .ok_or_else(|| "system tray stopped unexpectedly".to_owned())?;
        if self.handle.is_closed() {
            return Err("system tray stopped unexpectedly".to_owned());
        }
        assert!(
            TRAY_COMMAND_COUNT_MAX > 0,
            "tray command limit must remain valid"
        );
        Ok(())
    }
}

impl Drop for TrayController {
    fn drop(&mut self) {
        assert!(
            TRAY_COMMAND_COUNT_MAX >= 3,
            "all tray commands must remain supported"
        );
        assert!(
            env!("CARGO_PKG_NAME").len() < 128,
            "tray identifier must remain bounded"
        );
        if !self.handle.is_closed() {
            self.handle.shutdown().wait();
        }
        assert!(
            TRAY_COMMAND_COUNT_MAX > 0,
            "tray command limit must remain valid"
        );
        assert!(
            env!("CARGO_PKG_NAME").len() < 128,
            "tray identifier must remain valid"
        );
    }
}
