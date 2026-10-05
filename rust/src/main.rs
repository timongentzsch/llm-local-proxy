//! Process entry point: bind the sockets, serve, shut down cleanly.

use llm_local_proxy::config::{self, Config};
use llm_local_proxy::http::handler::Listener;
use llm_local_proxy::http::server::serve;
use llm_local_proxy::service::Service;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::net::TcpListener;
use tokio::signal::unix::{signal, SignalKind};

const USAGE: &str = "\
Local proxy for Codex and Claude subscriptions

Usage: llm-local-proxy [--config PATH] [--show-config]

  --config PATH    read this config file instead of the default
  --show-config    print the config path and exit
  --version        print the version and exit";

struct Args {
    config: Option<PathBuf>,
    show_config: bool,
}

fn args() -> Result<Args, String> {
    let mut parsed = Args {
        config: None,
        show_config: false,
    };
    let mut rest = std::env::args().skip(1);
    while let Some(arg) = rest.next() {
        match arg.as_str() {
            "--config" => {
                parsed.config = Some(rest.next().ok_or("--config needs a path")?.into());
            }
            "--show-config" => parsed.show_config = true,
            "--version" => {
                println!("llm-local-proxy {}", env!("CARGO_PKG_VERSION"));
                std::process::exit(0);
            }
            "-h" | "--help" => {
                println!("{USAGE}");
                std::process::exit(0);
            }
            other => match other.strip_prefix("--config=") {
                Some(path) => parsed.config = Some(path.into()),
                None => return Err(format!("unknown argument: {other}\n\n{USAGE}")),
            },
        }
    }
    Ok(parsed)
}

/// `urllib.parse.quote`, for the key in the dashboard link's fragment.
fn quote(value: &str) -> String {
    value
        .bytes()
        .map(|byte| match byte {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'_' | b'.' | b'-' | b'~' | b'/' => {
                (byte as char).to_string()
            }
            other => format!("%{other:02X}"),
        })
        .collect()
}

async fn bind(host: &str, port: u16) -> Result<TcpListener, String> {
    let address = if host.contains(':') {
        format!("[{host}]:{port}")
    } else {
        format!("{host}:{port}")
    };
    TcpListener::bind(&address)
        .await
        .map_err(|error| format!("{address}: {error}"))
}

async fn run(config: Config) -> Result<(), String> {
    // Bind first: a port already in use should not start any app-server.
    let admin = bind(&config.host, config.port).await?;
    let public = match config.public_port {
        0 => None,
        port => Some(bind(&config.public_host, port).await?),
    };
    let service = Arc::new(
        Service::new(config)
            .await
            .map_err(|error| error.to_string())?,
    );
    let config = &service.config;

    let fragment = match config.api_key.as_str() {
        "" => String::new(),
        key => format!("#key={}", quote(key)),
    };
    println!("LLM Local Proxy: {}/{fragment}", config.origin());
    println!("Config: {}", config.path.display());
    if public.is_some() {
        let address = format!("{}:{}", config.public_host, config.public_port);
        let reach = if config.public_url.is_empty() {
            &address
        } else {
            &config.public_url
        };
        println!("Named keys only: {reach}");
    }

    let listener = |public| {
        Arc::new(Listener {
            service: service.clone(),
            public,
        })
    };
    let serving = async {
        match public {
            Some(socket) => {
                tokio::join!(serve(admin, listener(false)), serve(socket, listener(true)));
            }
            None => serve(admin, listener(false)).await,
        }
    };
    // `docker stop` sends SIGTERM; take the same path as Ctrl-C so ledgers
    // are flushed and the app-server children are closed.
    let mut terminate = signal(SignalKind::terminate()).map_err(|error| error.to_string())?;
    let mut interrupt = signal(SignalKind::interrupt()).map_err(|error| error.to_string())?;
    tokio::select! {
        _ = serving => {}
        _ = terminate.recv() => {}
        _ = interrupt.recv() => {}
    }
    service.close().await;
    Ok(())
}

#[tokio::main]
async fn main() {
    let outcome = async {
        let args = args()?;
        let config = config::load(args.config.as_deref())?;
        if args.show_config {
            println!("{}", config.path.display());
            return Ok(());
        }
        run(config).await
    }
    .await;
    if let Err(error) = outcome {
        eprintln!("llm-local-proxy: {error}");
        std::process::exit(1);
    }
}
