mod wpt_conformance;

use std::io::{Read, Write};
use std::process::{Command, Stdio};
use std::time::Instant;

use libtest_mimic::{Arguments, Completion, Failed, Trial};
use serde::{Deserialize, Serialize};
use wpt_conformance::wpt_backend::WptBackend;
use wpt_conformance::wpt_js_loader::{
    default_wpt_dir, load_wpt_corpus, sanitize_test_id, trial_name,
};
use wpt_conformance::wpt_report::{WptReportCollector, report_output_path};
use wpt_conformance::wpt_types::WptLoadedCase;
use wpt_conformance::{run_one_test_case, should_skip_test, wpt_types::WptTestCase};

/// Path the worker writes its outcome to (set on the spawned child). A dedicated
/// file, not stdout: CoreML writes diagnostics to stdout that would otherwise
/// contaminate the outcome.
const ISOLATE_OUTCOME_ENV: &str = "WPT_ISOLATE_OUTCOME";
/// Optional override: `1`/`true` forces isolation on for every backend, `0`/`false` off.
const ISOLATE_ENV: &str = "WPT_ISOLATE";

/// Whether trials for `backend_prefix` run in their own worker process. Enabled by
/// default only for CoreML, whose models can trigger uncatchable native crashes that
/// would otherwise abort the whole suite; [`ISOLATE_ENV`] overrides this.
fn should_isolate(backend_prefix: &str) -> bool {
    match std::env::var(ISOLATE_ENV).ok().as_deref() {
        Some("1") | Some("true") => true,
        Some("0") | Some("false") => false,
        _ => backend_prefix == "coreml",
    }
}

#[derive(Serialize, Deserialize)]
struct IsolatedJob {
    backend: String,
    operation: String,
    case: WptTestCase,
}

fn run_trial(
    backend: &WptBackend,
    operation: &str,
    test_case: &WptTestCase,
) -> Result<Completion, Failed> {
    if let Some(reason) = should_skip_test(&test_case.graph) {
        return Ok(Completion::ignored_with(reason));
    }

    run_one_test_case(backend, operation, test_case)
        .map(|()| Completion::Completed)
        .map_err(Failed::from)
}

/// Worker entry point: run one [`IsolatedJob`] from stdin and write its outcome to
/// the [`ISOLATE_OUTCOME_ENV`] file. A native crash kills this process before it
/// writes, which the parent detects. stdout/stderr are left for CoreML's own
/// diagnostics so they never contaminate the outcome.
fn run_isolated_child() -> ! {
    let outcome_path = match std::env::var(ISOLATE_OUTCOME_ENV) {
        Ok(p) => p,
        Err(_) => {
            eprintln!("[WPT worker] {ISOLATE_OUTCOME_ENV} not set");
            std::process::exit(2);
        }
    };
    let mut input = String::new();
    if let Err(e) = std::io::stdin().read_to_string(&mut input) {
        eprintln!("[WPT worker] failed to read job from stdin: {e}");
        std::process::exit(2);
    }
    let job: IsolatedJob = match serde_json::from_str(&input) {
        Ok(job) => job,
        Err(e) => {
            eprintln!("[WPT worker] failed to parse job: {e}");
            std::process::exit(2);
        }
    };
    let Some(backend) = WptBackend::parse_name(&job.backend) else {
        eprintln!("[WPT worker] unknown backend '{}'", job.backend);
        std::process::exit(2);
    };

    let payload = match run_trial(&backend, &job.operation, &job.case) {
        Ok(Completion::Completed) => "COMPLETED".to_string(),
        Ok(Completion::Ignored { reason }) => format!("IGNORED\n{}", reason.unwrap_or_default()),
        Err(failed) => format!("FAILED\n{}", failed.message().unwrap_or("test failed")),
    };
    let _ = std::fs::write(&outcome_path, payload);
    std::process::exit(0);
}

/// Describes how a worker exited, naming the signal for native crashes.
fn describe_exit(status: std::process::ExitStatus) -> String {
    #[cfg(unix)]
    {
        use std::os::unix::process::ExitStatusExt;
        if let Some(sig) = status.signal() {
            let name = match sig {
                6 => " (SIGABRT)",
                11 => " (SIGSEGV)",
                4 => " (SIGILL)",
                8 => " (SIGFPE)",
                10 => " (SIGBUS)",
                _ => "",
            };
            return format!("killed by signal {sig}{name}");
        }
    }
    match status.code() {
        Some(code) => format!("exited with code {code}"),
        None => "terminated abnormally".to_string(),
    }
}

/// Runs one trial in a worker process, returning its outcome or a synthesized
/// failure if the worker crashed natively.
fn run_trial_isolated(
    backend_prefix: &str,
    operation: &str,
    test_case: &WptTestCase,
) -> Result<Completion, Failed> {
    // Skips never crash; handle in-process to avoid spawning.
    if let Some(reason) = should_skip_test(&test_case.graph) {
        return Ok(Completion::ignored_with(reason));
    }

    let job = IsolatedJob {
        backend: backend_prefix.to_string(),
        operation: operation.to_string(),
        case: test_case.clone(),
    };
    let job_json = serde_json::to_string(&job).map_err(|e| Failed::from(e.to_string()))?;
    let exe = std::env::current_exe().map_err(|e| Failed::from(e.to_string()))?;
    let outcome_path = unique_outcome_path();

    let mut child = Command::new(exe)
        .env(ISOLATE_OUTCOME_ENV, &outcome_path)
        .stdin(Stdio::piped())
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit())
        .spawn()
        .map_err(|e| Failed::from(format!("failed to spawn WPT worker: {e}")))?;

    if let Some(mut stdin) = child.stdin.take() {
        let _ = stdin.write_all(job_json.as_bytes());
        // Drop closes stdin so the worker's read returns.
    }

    let status = child
        .wait()
        .map_err(|e| Failed::from(format!("failed to wait for WPT worker: {e}")))?;
    let outcome = std::fs::read_to_string(&outcome_path).ok();
    let _ = std::fs::remove_file(&outcome_path);

    if let Some(text) = outcome {
        let (status_line, message) = text.split_once('\n').unwrap_or((text.as_str(), ""));
        match status_line.trim() {
            "COMPLETED" => return Ok(Completion::Completed),
            "IGNORED" => return Ok(Completion::ignored_with(message.to_string())),
            "FAILED" => return Err(Failed::from(message.to_string())),
            _ => {}
        }
    }

    // No outcome written: the worker crashed before reporting.
    Err(Failed::from(format!(
        "WPT worker did not report an outcome ({}); treated as a native crash",
        describe_exit(status)
    )))
}

/// A unique temp path for one worker's outcome file.
fn unique_outcome_path() -> std::path::PathBuf {
    use std::sync::atomic::{AtomicU64, Ordering};
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    std::env::temp_dir().join(format!("wpt_outcome_{}_{}", std::process::id(), n))
}

fn push_backend_trials(
    trials: &mut Vec<Trial>,
    backend: WptBackend,
    cases: &[WptLoadedCase],
    report: &WptReportCollector,
) {
    let prefix = backend.trial_prefix();
    let isolate = should_isolate(prefix);
    for case in cases {
        let operation = case.operation.clone();
        let test_case = case.as_test_case();
        let name = trial_name(prefix, case);
        let file_name = case.file_name.clone();
        let test_name = case.name.clone();
        let backend_prefix = prefix.to_string();
        let snapshot_name = format!("{backend_prefix}_{}", sanitize_test_id(&test_name));
        let report = report.clone();
        let backend = backend.clone();
        trials.push(Trial::ignorable_test(name, move || {
            let started = Instant::now();
            // Isolated trials run in a worker process so a native crash is a failure, not an abort.
            let result = if isolate {
                run_trial_isolated(&backend_prefix, &operation, &test_case)
            } else {
                run_trial(&backend, &operation, &test_case)
            };
            let duration = started.elapsed();

            match &result {
                Ok(Completion::Completed) => {
                    insta::assert_debug_snapshot!(
                        snapshot_name,
                        (&file_name, &test_name, &backend_prefix, "PASS")
                    );
                    report.record_pass(&file_name, &test_name, &backend_prefix, duration);
                }
                Ok(Completion::Ignored { reason }) => {
                    let reason = reason.as_deref().unwrap_or("ignored").to_string();
                    report.record_skip(&file_name, &test_name, &backend_prefix, reason, duration);
                }
                Err(err) => {
                    let msg = err.message().unwrap_or("test failed").to_string();
                    insta::assert_snapshot!(
                        snapshot_name,
                        format!("{file_name} {test_name}, {backend_prefix}\n {msg}")
                    );
                    report.record_fail(&file_name, &test_name, &backend_prefix, msg, duration);
                }
            }
            result
        }));
    }
}

fn main() {
    let _ = pretty_env_logger::try_init();

    // Worker mode: run exactly one trial handed to us over stdin, then exit.
    if std::env::var(ISOLATE_OUTCOME_ENV).is_ok() {
        run_isolated_child();
    }

    let args = Arguments::from_args();
    let wpt_dir = default_wpt_dir();

    let corpus = match load_wpt_corpus(&wpt_dir) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("{e}");
            eprintln!();
            eprintln!("Ensure Node.js is on PATH and fetch WPT:");
            eprintln!("  node scripts/fetch_wpt.mjs");
            eprintln!("Or set WPT_DIR to an existing WPT checkout.");
            std::process::exit(2);
        }
    };

    eprintln!(
        "[WPT] loaded {} case(s) from {} via Node bridge",
        corpus.cases.len(),
        if corpus.wpt_dir.is_empty() {
            wpt_dir.display().to_string()
        } else {
            corpus.wpt_dir.clone()
        }
    );

    let backends = WptBackend::selected();
    if backends.is_empty() {
        eprintln!("No WPT backends available (enable onnx-runtime and/or trtx-runtime).");
        eprintln!("Set WPT_BACKEND=onnx or trtx to limit registered trials.");
        std::process::exit(2);
    }
    let backend_prefixes: Vec<&str> = backends.iter().map(|b| b.trial_prefix()).collect();
    eprintln!("[WPT] backends: {}", backend_prefixes.join(", "));

    let skip_eligible = corpus
        .cases
        .iter()
        .filter(|c| should_skip_test(&c.graph).is_some())
        .count()
        * backends.len();
    let trial_count = corpus.cases.len() * backends.len();
    eprintln!(
        "[WPT] registering {trial_count} trial(s) ({skip_eligible} dtype-skipped, {} executed)",
        trial_count.saturating_sub(skip_eligible)
    );
    if wpt_conformance::wpt_config::REUSE_ML_CONTEXT {
        eprintln!("[WPT] MLContext reuse: enabled (one context per backend per thread)");
    }

    let wpt_dir_label = if corpus.wpt_dir.is_empty() {
        wpt_dir.display().to_string()
    } else {
        corpus.wpt_dir.clone()
    };
    let report = WptReportCollector::new(
        wpt_dir_label,
        &backend_prefixes,
        &corpus,
        args.filter.clone(),
        report_output_path(),
    );

    let mut trials = Vec::new();
    for backend in backends {
        push_backend_trials(&mut trials, backend, &corpus.cases, &report);
    }

    let conclusion = libtest_mimic::run(&args, trials);
    eprintln!(
        "[WPT] result: {} passed, {} skipped, {} failed; {} filtered out",
        conclusion.num_passed,
        conclusion.num_ignored,
        conclusion.num_failed,
        conclusion.num_filtered_out
    );

    if let Err(e) = report.write_json() {
        eprintln!("[WPT] warning: failed to write JSON report: {e}");
    }

    conclusion.exit();
}
