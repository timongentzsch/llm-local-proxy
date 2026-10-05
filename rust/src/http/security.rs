//! Loopback hardening, independent of dialect and provider.
//!
//! The proxy holds live subscription credentials, so it refuses requests that
//! a browser on another origin could have forged, and requests addressed to a
//! host name it does not serve (DNS rebinding).

use crate::keys::{constant_time_eq, MASTER};
use hyper::header::HeaderMap;

const LOOPBACK: [&str; 3] = ["127.0.0.1", "::1", "localhost"];

/// Headers that may carry the proxy's own key, with the scheme inside each.
/// Every mount accepts all of them: refusing one only yields a confusing 401.
const CREDENTIALS: [(&str, &str); 2] = [("authorization", "bearer"), ("x-api-key", "")];

fn header<'a>(headers: &'a HeaderMap, name: &str) -> &'a str {
    headers
        .get(name)
        .and_then(|value| value.to_str().ok())
        .unwrap_or_default()
}

/// Host and port from an authority such as `localhost:8787` or `[::1]:8787`.
/// Anything that is more than an authority is no host at all.
fn authority(value: &str) -> Option<(String, Option<u16>)> {
    if value.is_empty() || value.contains(['/', '?', '#', '@', ' ']) {
        return None;
    }
    let (host, port) = if let Some(rest) = value.strip_prefix('[') {
        let (host, suffix) = rest.split_once(']')?;
        match suffix {
            "" => (host, None),
            suffix => (host, Some(suffix.strip_prefix(':')?)),
        }
    } else {
        match value.rsplit_once(':') {
            Some((host, port)) if !host.contains(':') => (host, Some(port)),
            Some(_) => return None,
            None => (value, None),
        }
    };
    let port = match port {
        None | Some("") => None,
        Some(port) => Some(port.parse().ok()?),
    };
    Some((host.to_lowercase(), port))
}

/// The host a request was addressed to and its port (80 when unnamed).
pub fn request_host(headers: &HeaderMap) -> (String, u16) {
    match authority(header(headers, "host")) {
        Some((host, port)) => (host, port.unwrap_or(80)),
        None => (String::new(), 0),
    }
}

pub fn valid_host(headers: &HeaderMap, configured: &str) -> bool {
    let (host, _) = request_host(headers);
    LOOPBACK.contains(&host.as_str()) || host == configured
}

pub fn same_origin(headers: &HeaderMap) -> bool {
    let origin = header(headers, "origin");
    if origin.is_empty() {
        return true;
    }
    let Some((scheme, rest)) = origin.split_once("://") else {
        return false;
    };
    let default = match scheme {
        "http" => 80,
        "https" => 443,
        _ => return false,
    };
    let Some((host, port)) = authority(rest.trim_end_matches('/')) else {
        return false;
    };
    let (_, served) = request_host(headers);
    LOOPBACK.contains(&host.as_str()) && port.unwrap_or(default) == served
}

/// Who the request's key belongs to, in any accepted form, or None.
///
/// `MASTER` for the configured key (or for anyone when none is configured, as
/// auth is then off), otherwise the name of a matching named key.
pub fn identify(
    headers: &HeaderMap,
    master: &str,
    named: impl Fn(&str) -> Option<String>,
) -> Option<String> {
    for (name, scheme) in CREDENTIALS {
        let mut value = header(headers, name);
        if value.is_empty() {
            continue;
        }
        if !scheme.is_empty() {
            let (sent, rest) = value.split_once(' ').unwrap_or((value, ""));
            if !sent.eq_ignore_ascii_case(scheme) {
                continue;
            }
            value = rest;
        }
        if !master.is_empty() && constant_time_eq(value, master) {
            return Some(MASTER.to_string());
        }
        if let Some(name) = named(value) {
            return Some(name);
        }
    }
    master.is_empty().then(|| MASTER.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn headers(pairs: &[(&'static str, &str)]) -> HeaderMap {
        let mut map = HeaderMap::new();
        for (name, value) in pairs {
            map.insert(*name, value.parse().unwrap());
        }
        map
    }

    #[test]
    fn only_loopback_hosts_are_served() {
        for good in ["127.0.0.1:8787", "localhost:8787", "[::1]:8787", "LOCALHOST"] {
            assert!(valid_host(&headers(&[("host", good)]), "127.0.0.1"), "{good}");
        }
        for bad in ["evil.com", "evil.com:8787", "127.0.0.1@evil.com", "", "[::1]x:1", "a/b"] {
            assert!(!valid_host(&headers(&[("host", bad)]), "127.0.0.1"), "{bad}");
        }
    }

    #[test]
    fn a_foreign_origin_is_refused() {
        let ok = |origin: &str| same_origin(&headers(&[("host", "127.0.0.1:8787"), ("origin", origin)]));
        assert!(same_origin(&headers(&[("host", "127.0.0.1:8787")])));
        assert!(ok("http://127.0.0.1:8787"));
        assert!(ok("http://localhost:8787"));
        assert!(!ok("http://localhost:9999"));
        assert!(!ok("http://evil.com:8787"));
        assert!(!ok("null"));
        assert!(!ok("file://localhost:8787"));
    }

    #[test]
    fn keys_are_accepted_in_either_header() {
        let named = |token: &str| (token == "llp_x").then(|| "alice".to_string());
        let master = "m".repeat(24);
        let bearer = format!("Bearer {master}");
        assert_eq!(identify(&headers(&[("authorization", &bearer)]), &master, named).as_deref(), Some("master"));
        assert_eq!(identify(&headers(&[("x-api-key", &master)]), &master, named).as_deref(), Some("master"));
        assert_eq!(identify(&headers(&[("x-api-key", "llp_x")]), &master, named).as_deref(), Some("alice"));
        assert_eq!(identify(&headers(&[("authorization", "Basic llp_x")]), &master, named), None);
        assert_eq!(identify(&headers(&[]), &master, named), None);
        assert_eq!(identify(&headers(&[]), "", named).as_deref(), Some("master"));
    }
}
