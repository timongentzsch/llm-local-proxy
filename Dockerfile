FROM ghcr.io/astral-sh/uv:0.11.28-python3.12-trixie-slim@sha256:3137a0b606f65a74ee0245f43dae219b09e8af98fc37fef20841cbceef35a646

ARG CODEX_VERSION=0.145.0
ARG CODEX_INSTALLER_SHA256=ba92dd27e5c06f0d3bbc58bfa4b9cfb6599cd2742fbb1f92a2765e6c07dedb5a

RUN apt-get update \
    && apt-get install --no-install-recommends -y ca-certificates curl \
    && curl -fsSL https://chatgpt.com/codex/install.sh -o /tmp/codex-install.sh \
    && printf '%s  %s\n' "$CODEX_INSTALLER_SHA256" /tmp/codex-install.sh | sha256sum -c - \
    && CODEX_RELEASE="$CODEX_VERSION" CODEX_NON_INTERACTIVE=1 \
       CODEX_INSTALL_DIR=/usr/local/bin CODEX_HOME=/opt/codex \
       sh /tmp/codex-install.sh \
    && rm /tmp/codex-install.sh \
    && apt-get purge --auto-remove -y curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
COPY src src
RUN uv sync --locked --no-dev \
    && useradd --create-home --uid 10001 runner \
    && mkdir /config /codex \
    && chown runner:runner /config /codex

ENV CODEX_HOME=/codex \
    CODEX_PROXY_CONTAINER=1 \
    PATH="/app/.venv/bin:$PATH" \
    XDG_CONFIG_HOME=/config

USER runner
EXPOSE 8787
CMD ["codex-local-proxy"]
