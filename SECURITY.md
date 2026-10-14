# Security

## Reporting a vulnerability

Please report it privately through GitHub:
[Security advisories](https://github.com/timongentzsch/llm-local-proxy/security/advisories/new).
Do not open a public issue for a vulnerability. Include the version, what you
did and what happened; you can expect a first answer within a week.

## Supported versions

Only the latest release receives fixes.

## What the proxy holds

- **Subscription credentials**: OAuth tokens for each signed-in Claude and
  Codex account. They are stored locally, sent only to their own provider, and
  written atomically to private files (`0600`, in directories `0700`). Claude
  logins live under the proxy's config directory, Codex logins under
  `codex_home`.
- **The master key** in `config.toml`, and named keys in `keys.json` next to
  it. Named keys are kept readable so the dashboard can show them again.
- **Usage counts**, no request or response content.

## Protections

- The listener binds to loopback by default; a non-loopback `host` is refused
  outside the container image.
- `config.toml` must not be readable by group or others, or the proxy refuses
  to start.
- Requests with a foreign `Origin`, or addressed to a host name the proxy does
  not serve, are refused (cross-site forgery and DNS rebinding).
- Keys are compared in constant time. The master key works only on the local
  listener; the optional public listener (`public_port`) serves only the model
  API to named keys, and has no dashboard, account or key routes.
- Setting `api_key` to empty turns authentication off; only do that on a
  machine you alone use.
- The public listener speaks plain HTTP. Put TLS in front of it, for example
  with Tailscale Serve.

## Out of scope

Anyone who can read your files or run code as your user can read the
credentials. The proxy does not limit the rate of failed key attempts; keys
are 256-bit random values for named keys, so guessing is not practical, but
choose a long master key if you set one yourself.
