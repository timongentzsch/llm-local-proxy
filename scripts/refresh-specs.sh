#!/usr/bin/env bash
# Re-resolve the pinned downstream wire specifications and report what moved.
#
# The OpenAI spec lives at a stable URL. Anthropic's is content-addressed and
# discovered indirectly: their TypeScript SDK pins the generator input in
# .stats.yml, so resolve that first rather than hardcoding a hash that rots.
#
# This script never edits specs/PINNED.md. A changed hash is a wire contract
# change and wants a human reading the diff.
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
specs="$root/specs"
stats="https://raw.githubusercontent.com/anthropics/anthropic-sdk-typescript/main/.stats.yml"
openai="https://raw.githubusercontent.com/openai/openai-openapi/master/openapi.yaml"

sha() {
  if command -v shasum >/dev/null 2>&1; then
    shasum -a 256 "$1" | cut -d' ' -f1
  else
    sha256sum "$1" | cut -d' ' -f1
  fi
}

fetch() {
  local url="$1" dest="$2" tmp
  tmp="$(mktemp)"
  curl -fsSL --max-time 120 "$url" -o "$tmp"
  if [ -f "$dest" ] && [ "$(sha "$tmp")" = "$(sha "$dest")" ]; then
    echo "unchanged  $(basename "$dest")  $(sha "$dest")"
    rm -f "$tmp"
    return 0
  fi
  if [ -f "$dest" ]; then
    echo "WAS        $(basename "$dest")  $(sha "$dest")"
  fi
  mv "$tmp" "$dest"
  chmod 644 "$dest"
  echo "UPDATED    $(basename "$dest")  $(sha "$dest")  $(wc -c <"$dest" | tr -d ' ') bytes"
  echo "           source: $url"
  changed=1
}

changed=0
mkdir -p "$specs"

anthropic="$(curl -fsSL --max-time 60 "$stats" | awk '/^openapi_spec_url:/ {print $2}')"
[ -n "$anthropic" ] || { echo "could not resolve openapi_spec_url from $stats" >&2; exit 1; }

fetch "$anthropic" "$specs/anthropic-openapi.json"
fetch "$openai" "$specs/openai-openapi.yaml"

if [ "$changed" = 1 ]; then
  echo
  echo "A spec moved. Update the table and date in specs/PINNED.md, then run:"
  echo "  uv run python -m unittest discover -s tests -v"
  echo "Conformance failures here are real downstream contract changes."
fi
