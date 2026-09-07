#!/usr/bin/env bash
# Fetch reviewed snapshots. Update URLs and hashes together when reviewing a spec upgrade.
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "$root/specs"
tmp="$(mktemp -d "$root/specs/.fetch.XXXXXX")"
trap 'rm -rf "$tmp"' EXIT

fetch() {
  local name="$1" expected="$2" url="$3" actual
  curl -fsSL --retry 2 --max-time 60 "$url" -o "$tmp/$name"
  actual="$(shasum -a 256 "$tmp/$name" | cut -d' ' -f1)"
  if [ "$actual" != "$expected" ]; then
    echo "checksum mismatch: $name (expected $expected, got $actual)" >&2
    exit 1
  fi
  chmod 644 "$tmp/$name"
  mv "$tmp/$name" "$root/specs/$name"
  echo "verified $name $actual"
}

fetch anthropic-openapi.json \
  717ab2a5efd6263fc76a03b1b361c03d34fe3a0987c2b8f445b6537ede0c991a \
  https://storage.googleapis.com/stainless-sdk-openapi-specs/anthropic/anthropic-319861ef873b46e22d6feb51442e743643815093bdd2b3324df52ed202d7ab93.yml
fetch openai-openapi.yaml \
  77a517da92356a777eb9be7ecc978c15adcc2f17ee387c282090e5a890170cf5 \
  https://raw.githubusercontent.com/openai/openai-openapi/b61ced96515cb6e73794ff459e9e12ca57596c72/openapi.yaml
