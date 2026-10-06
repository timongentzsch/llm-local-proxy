//! The Claude subscription.

pub mod auth;
pub mod catalog;
pub mod events;
pub mod provider;
pub mod request;
pub mod subscription;
pub mod thinking;
pub mod upstream;
pub mod usage;

/// Python's `str.strip()`: unlike `str::trim`, it also strips U+001C..U+001F.
pub(crate) fn strip(text: &str) -> &str {
    text.trim_matches(|c: char| c.is_whitespace() || ('\u{1c}'..='\u{1f}').contains(&c))
}
