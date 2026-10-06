//! Atomic private-file writes shared by config and auth persistence.

use serde_json::{Map, Value};
use std::fs::{self, OpenOptions};
use std::io::{self, Write};
use std::os::unix::fs::{DirBuilderExt, OpenOptionsExt};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::SystemTime;

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

/// What identifies one version of a file: modification time and length.
type Stamp = Option<(SystemTime, u64)>;

fn stamp(path: &Path) -> Stamp {
    let metadata = fs::metadata(path).ok()?;
    Some((metadata.modified().ok()?, metadata.len()))
}

/// A private JSON object on disk, kept parsed in memory.
///
/// Credentials are read on every request; this costs one `stat` instead of
/// opening and parsing the file, and still notices a change made by another
/// program. It also holds on to a value the disk refused to take: a rotated
/// refresh token must not be lost because a write failed.
pub struct JsonFile {
    path: PathBuf,
    cached: Mutex<Cached>,
}

#[derive(Default)]
struct Cached {
    loaded: bool,
    stamp: Stamp,
    value: Option<Arc<Map<String, Value>>>,
    /// The value is newer than the file, whose write failed.
    unsaved: bool,
}

impl JsonFile {
    pub fn new(path: PathBuf) -> Self {
        JsonFile {
            path,
            cached: Mutex::new(Cached::default()),
        }
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    /// The object in the file; None when it is missing or not an object.
    pub fn read(&self) -> Option<Arc<Map<String, Value>>> {
        let mut cached = self.cached.lock().unwrap();
        if cached.unsaved {
            return cached.value.clone();
        }
        let current = stamp(&self.path);
        if cached.loaded && cached.stamp == current {
            return cached.value.clone();
        }
        let parsed = match read_json(&self.path) {
            Ok(Some(Ok(Value::Object(value)))) => Some(Arc::new(value)),
            // Caught mid-rewrite by another program: keep what was last read
            // and look again next time.
            Ok(Some(Err(_))) if cached.value.is_some() => return cached.value.clone(),
            _ => None,
        };
        *cached = Cached {
            loaded: true,
            stamp: current,
            value: parsed.clone(),
            unsaved: false,
        };
        parsed
    }

    /// Replace the object. It is kept in memory even if the disk refuses it.
    pub fn write(&self, value: Map<String, Value>) -> io::Result<()> {
        let mut cached = self.cached.lock().unwrap();
        let outcome = write_json(&self.path, &Value::Object(value.clone()));
        *cached = Cached {
            loaded: true,
            stamp: stamp(&self.path),
            value: Some(Arc::new(value)),
            unsaved: outcome.is_err(),
        };
        outcome
    }

    pub fn remove(&self) {
        let mut cached = self.cached.lock().unwrap();
        let _ = fs::remove_file(&self.path);
        *cached = Cached {
            loaded: true,
            ..Cached::default()
        };
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn object(value: Value) -> Map<String, Value> {
        value.as_object().cloned().unwrap()
    }

    #[test]
    fn a_json_file_is_cached_until_it_changes() {
        let dir = std::env::temp_dir().join(format!("llp-jsonfile-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        let file = JsonFile::new(dir.join("a.json"));
        assert!(file.read().is_none());
        file.write(object(json!({"n": 1}))).unwrap();
        let first = file.read().unwrap();
        assert_eq!(first["n"], 1);
        // Unchanged on disk: the same parsed value, not a new read.
        assert!(Arc::ptr_eq(&first, &file.read().unwrap()));
        // Another program rewrites it.
        write_bytes(file.path(), br#"{"n": 22}"#).unwrap();
        assert_eq!(file.read().unwrap()["n"], 22);
        file.remove();
        assert!(file.read().is_none());
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn a_value_the_disk_refused_is_kept_in_memory() {
        // A directory where the file should be: every write fails.
        let dir = std::env::temp_dir().join(format!("llp-jsonfile-ro-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(dir.join("a.json")).unwrap();
        let file = JsonFile::new(dir.join("a.json"));
        assert!(file.write(object(json!({"token": "rotated"}))).is_err());
        assert_eq!(file.read().unwrap()["token"], "rotated");
        let _ = fs::remove_dir_all(dir);
    }
}
