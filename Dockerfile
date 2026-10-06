# One static binary on an empty image: no shell, no libc, no package manager.
# TLS roots are compiled in.
FROM rust:1-alpine AS build
RUN apk add --no-cache musl-dev
WORKDIR /src
COPY Cargo.toml Cargo.lock ./
COPY src src
RUN cargo build --release --locked && mkdir -p /out/config /out/codex

FROM scratch
COPY --from=build /src/target/release/llm-local-proxy /llm-local-proxy
# Owned by the runtime user, so a fresh volume mounted here is writable.
COPY --from=build --chown=10001:10001 /out/config /config
COPY --from=build --chown=10001:10001 /out/codex /codex

ENV CODEX_HOME=/codex \
    LLM_PROXY_CONTAINER=1 \
    XDG_CONFIG_HOME=/config

USER 10001:10001
EXPOSE 8787
ENTRYPOINT ["/llm-local-proxy"]
