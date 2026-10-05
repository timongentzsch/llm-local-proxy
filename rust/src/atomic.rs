//! Atomic private-file writes shared by config and auth persistence.

use serde_json::Value;
use std::fs::{self, OpenOptions};
use std::io::{self, Write};
use std::os::unix::fs::{DirBuilderExt, OpenOptionsExt};
use std::path::Path;

/// Create `path` and its parents, private to the owner.
pub fn private_dir(path: &Path) -> io::Result<()> {
    fs::DirBuilder::new()
        .recursive(true)
        .mode(0o700)
        .create(path)
}

/// Write `data` to `path` via a same-directory O_EXCL temp file and a rename.
pub fn write_bytes(path: &Path, data: &[u8]) -> io::Result<()> {
    let parent = path.parent().unwrap_or(Path::new("."));
    private_dir(parent)?;
    let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("file");
    let temp = parent.join(format!(".{name}.{}.tmp", std::process::id()));
    // Left behind by a crash in an earlier process with this pid.
    let _ = fs::remove_file(&temp);
    let result = (|| {
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .mode(0o600)
            .open(&temp)?;
        file.write_all(data)?;
        file.sync_all()?;
        fs::rename(&temp, path)?;
        // The rename is durable only once its directory entry is.
        if let Ok(directory) = fs::File::open(parent) {
            let _ = directory.sync_all();
        }
        Ok(())
    })();
    if result.is_err() {
        let _ = fs::remove_file(&temp);
    }
    result
}

pub fn write_json(path: &Path, value: &Value) -> io::Result<()> {
    write_bytes(path, value.to_string().as_bytes())
}

/// The JSON document at `path`; `None` when the file does not exist.
pub fn read_json(path: &Path) -> io::Result<Option<Result<Value, serde_json::Error>>> {
    match fs::read_to_string(path) {
        Ok(text) => Ok(Some(serde_json::from_str(&text))),
        Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(None),
        Err(error) => Err(error),
    }
}

/// 32 random bytes as URL-safe base64 without padding (`secrets.token_urlsafe`).
pub fn token_urlsafe(bytes: usize) -> String {
    use base64::Engine;
    let mut raw = vec![0u8; bytes];
    getrandom::getrandom(&mut raw).expect("the operating system has randomness");
    base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(raw)
}
