# Build from the repository root: the dashboard is shared with the Python tree.
#
#   docker build -f rust/Dockerfile -t llm-local-proxy:rust .
FROM rust:1-slim-trixie AS build
WORKDIR /src
COPY rust/Cargo.toml rust/Cargo.lock rust/
COPY rust/src rust/src
COPY src/llm_local_proxy/static src/llm_local_proxy/static
RUN cargo build --release --locked --manifest-path rust/Cargo.toml

FROM debian:trixie-slim

ARG CODEX_VERSION=0.153.0
ARG CODEX_INSTALLER_SHA256=150e3cf675682efeaac115aa3747add3f27887896d04ce6d0b56478d8b428bf6

RUN apt-get update \
    && apt-get install --no-install-recommends -y ca-certificates curl \
    && curl -fsSL https://chatgpt.com/codex/install.sh -o /tmp/codex-install.sh \
    && printf '%s  %s\n' "$CODEX_INSTALLER_SHA256" /tmp/codex-install.sh | sha256sum -c - \
    && CODEX_RELEASE="$CODEX_VERSION" CODEX_NON_INTERACTIVE=1 \
       CODEX_INSTALL_DIR=/usr/local/bin CODEX_HOME=/opt/codex \
       sh /tmp/codex-install.sh \
    && rm /tmp/codex-install.sh \
    && apt-get purge --auto-remove -y curl \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 runner \
    && mkdir /config /codex \
    && chown runner:runner /config /codex

COPY --from=build /src/rust/target/release/llm-local-proxy /usr/local/bin/llm-local-proxy

ENV CODEX_HOME=/codex \
    LLM_PROXY_CONTAINER=1 \
    XDG_CONFIG_HOME=/config

USER runner
EXPOSE 8787
CMD ["llm-local-proxy"]
