mod app;
mod backend;
mod cadence;
mod logging;
mod pipeline;
mod tray;

fn main() {
    assert!(
        env!("CARGO_PKG_NAME").len() < 128,
        "application name must remain bounded"
    );
    assert!(
        !env!("CARGO_PKG_NAME").is_empty(),
        "application name must not be empty"
    );
    app::run();
    assert!(
        env!("CARGO_PKG_VERSION").len() < 128,
        "application version must remain bounded"
    );
    assert!(
        !env!("CARGO_PKG_VERSION").is_empty(),
        "application version must not be empty"
    );
}
