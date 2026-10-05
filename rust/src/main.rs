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
  --healthcheck    exit 0 if the running proxy answers /healthz, for containers
  --version        print the version and exit";

struct Args {
    config: Option<PathBuf>,
    show_config: bool,
    healthcheck: bool,
}

fn args() -> Result<Args, String> {
    let mut parsed = Args {
        config: None,
        show_config: false,
        healthcheck: false,
    };
    let mut rest = std::env::args().skip(1);
    while let Some(arg) = rest.next() {
        match arg.as_str() {
            "--config" => {
                parsed.config = Some(rest.next().ok_or("--config needs a path")?.into());
            }
            "--show-config" => parsed.show_config = true,
            "--healthcheck" => parsed.healthcheck = true,
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

/// Ask the proxy this config describes whether it is healthy.
fn healthcheck(config: &Config) -> Result<(), String> {
    use std::io::{Read, Write};
    let host = match config.host.as_str() {
        "0.0.0.0" => "127.0.0.1",
        "::" => "::1",
        host => host,
    };
    let fail = |error: std::io::Error| error.to_string();
    let mut stream = std::net::TcpStream::connect((host, config.port)).map_err(fail)?;
    let timeout = Some(std::time::Duration::from_secs(3));
    stream.set_read_timeout(timeout).map_err(fail)?;
    stream.set_write_timeout(timeout).map_err(fail)?;
    let request = "GET /healthz HTTP/1.1\r\nHost: 127.0.0.1\r\nConnection: close\r\n\r\n";
    stream.write_all(request.as_bytes()).map_err(fail)?;
    let mut answer = String::new();
    let _ = stream.read_to_string(&mut answer);
    match answer.lines().next() {
        Some(status) if status.contains(" 200 ") => Ok(()),
        Some(status) => Err(format!("unhealthy: {status}")),
        None => Err("no answer".into()),
    }
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
        if args.healthcheck {
            return healthcheck(&config);
        }
        run(config).await
    }
    .await;
    if let Err(error) = outcome {
        eprintln!("llm-local-proxy: {error}");
        std::process::exit(1);
    }
}
