//! Bounded invocations of the host's ffmpeg and ffprobe for container formats Helios does not
//! decode itself. Inputs are written to private temporary files because container parsers need
//! seekable input; outputs are read through a hard byte limit so a hostile stream cannot grow
//! past what admission accepted.

use std::io::{Read, Write};
use std::path::PathBuf;
use std::process::{Command, Stdio};

use crate::{Error, Failure, FrontendError, Result};

pub(crate) struct TempFile {
    pub(crate) path: PathBuf,
}

impl Drop for TempFile {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.path);
    }
}

static COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

fn failure(message: impl Into<String>) -> Error {
    Error::frontend(FrontendError::Input(Failure::new(message.into())))
}

pub(crate) fn temp_file(bytes: &[u8]) -> Result<TempFile> {
    use std::os::unix::fs::OpenOptionsExt;
    let serial = COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_nanos());
    let path = std::env::temp_dir().join(format!("flare-media-{}-{serial}-{nanos}.bin", std::process::id()));
    let mut file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .mode(0o600)
        .open(&path)
        .map_err(|error| failure(format!("cannot stage media for decoding: {error}")))?;
    let temp = TempFile { path };
    file.write_all(bytes)
        .and_then(|()| file.sync_data())
        .map_err(|error| failure(format!("cannot stage media for decoding: {error}")))?;
    Ok(temp)
}

fn spawn_error(tool: &str, error: std::io::Error) -> Error {
    if error.kind() == std::io::ErrorKind::NotFound {
        failure(format!("{tool} is required to decode this media type and was not found on PATH"))
    } else {
        failure(format!("cannot start {tool}: {error}"))
    }
}

/// Runs `command`, returning its standard output when it exits successfully and the output
/// stays within `limit` bytes; a larger output kills the process.
pub(crate) fn run_bounded(mut command: Command, tool: &str, limit: usize) -> Result<Vec<u8>> {
    command.stdin(Stdio::null()).stdout(Stdio::piped()).stderr(Stdio::piped());
    let mut child = command.spawn().map_err(|error| spawn_error(tool, error))?;
    let mut stdout = child.stdout.take().expect("piped stdout");
    let mut stderr = child.stderr.take().expect("piped stderr");
    let errors = std::thread::spawn(move || {
        let mut text = Vec::new();
        let _ = (&mut stderr).take(16 << 10).read_to_end(&mut text);
        String::from_utf8_lossy(&text).trim().to_owned()
    });
    let mut output = Vec::new();
    let read = (&mut stdout).take(limit as u64 + 1).read_to_end(&mut output);
    let overflow = output.len() > limit;
    if overflow || read.is_err() {
        let _ = child.kill();
    }
    let status = child.wait().map_err(|error| failure(format!("{tool} did not exit: {error}")))?;
    let errors = errors.join().unwrap_or_default();
    if overflow {
        return Err(failure(format!("{tool} produced more output than the accepted media extent")));
    }
    read.map_err(|error| failure(format!("cannot read {tool} output: {error}")))?;
    if !status.success() {
        return Err(failure(format!(
            "{tool} could not decode the media{}{}",
            if errors.is_empty() { "" } else { ": " },
            errors
        )));
    }
    Ok(output)
}

/// Runs `ffprobe` over one stream of `kind` with the given `-show_entries` selection.
fn probe(path: &std::path::Path, kind: &str, entries: &str) -> Result<serde_json::Value> {
    let mut command = Command::new("ffprobe");
    command
        .arg("-v")
        .arg("error")
        .arg("-select_streams")
        .arg(format!("{kind}:0"))
        .arg("-show_entries")
        .arg(entries)
        .arg("-of")
        .arg("json")
        .arg(path);
    let output = run_bounded(command, "ffprobe", 1 << 20)?;
    serde_json::from_slice(&output).map_err(|error| failure(format!("invalid ffprobe output: {error}")))
}

fn first_stream(value: &serde_json::Value, kind: &str) -> Result<serde_json::Value> {
    value["streams"]
        .get(0)
        .cloned()
        .ok_or_else(|| failure(format!("the media contains no {} stream", if kind == "v" { "video" } else { "audio" })))
}

/// The first stream of `kind` (`v` or `a`) as ffprobe's JSON object.
pub(crate) fn probe_stream(path: &std::path::Path, kind: &str, entries: &str) -> Result<serde_json::Value> {
    first_stream(&probe(path, kind, &format!("stream={entries}"))?, kind)
}

/// The first stream of `kind`, with its side data under `side_data_list`, and the container's
/// own fields. Some muxers, Matroska among them, keep the duration in the container while
/// leaving it out of the stream.
pub(crate) fn probe_stream_and_format(
    path: &std::path::Path,
    kind: &str,
    stream_entries: &str,
    side_data_entries: &str,
    format_entries: &str,
) -> Result<(serde_json::Value, serde_json::Value)> {
    let value = probe(
        path,
        kind,
        &format!("stream={stream_entries}:stream_side_data={side_data_entries}:format={format_entries}"),
    )?;
    Ok((first_stream(&value, kind)?, value["format"].clone()))
}

/// Parses an ffprobe rational such as `30000/1001`.
pub(crate) fn rational(value: &serde_json::Value) -> Option<f64> {
    let text = value.as_str()?;
    let (numerator, denominator) = text.split_once('/').unwrap_or((text, "1"));
    let (numerator, denominator): (f64, f64) = (numerator.trim().parse().ok()?, denominator.trim().parse().ok()?);
    (denominator != 0.0 && numerator.is_finite() && denominator.is_finite()).then(|| numerator / denominator)
}

pub(crate) fn integer(value: &serde_json::Value) -> Option<u64> {
    value.as_u64().or_else(|| value.as_str()?.trim().parse().ok())
}

pub(crate) fn float(value: &serde_json::Value) -> Option<f64> {
    value.as_f64().or_else(|| value.as_str()?.trim().parse().ok()).filter(|v: &f64| v.is_finite())
}
