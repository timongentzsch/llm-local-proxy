//! Listening sockets.

use super::handler::{handle, Listener};
use hyper::server::conn::http1;
use hyper::service::service_fn;
use hyper_util::rt::{TokioIo, TokioTimer};
use std::convert::Infallible;
use std::sync::Arc;
use std::time::Duration;
use tokio::net::TcpListener;
use tokio::sync::Semaphore;

/// Connections served at once per listener. A connection is cheap, but an
/// unbounded number of half-open ones is how a listener is starved.
const MAX_CONNECTIONS: usize = 1024;
/// How long a client may take to send its request headers.
const HEADER_TIMEOUT: Duration = Duration::from_secs(30);

/// Accept connections until the task is dropped.
pub async fn serve(socket: TcpListener, listener: Arc<Listener>) {
    let slots = Arc::new(Semaphore::new(MAX_CONNECTIONS));
    loop {
        let Ok(slot) = slots.clone().acquire_owned().await else {
            return;
        };
        let (stream, peer) = match socket.accept().await {
            Ok(accepted) => accepted,
            // Out of descriptors or a connection reset before accept: wait
            // rather than spin.
            Err(_) => {
                tokio::time::sleep(Duration::from_millis(50)).await;
                continue;
            }
        };
        // Frames are small and each should leave as it is written.
        let _ = stream.set_nodelay(true);
        let listener = listener.clone();
        tokio::spawn(async move {
            let service = service_fn(move |request| {
                let listener = listener.clone();
                async move { Ok::<_, Infallible>(handle(listener, peer, request).await) }
            });
            let _ = http1::Builder::new()
                .timer(TokioTimer::new())
                .header_read_timeout(HEADER_TIMEOUT)
                .serve_connection(TokioIo::new(stream), service)
                .await;
            drop(slot);
        });
    }
}
