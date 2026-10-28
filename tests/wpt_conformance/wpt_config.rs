//! WPT harness compile-time options.

/// Reuse one [`MLContext`](rustnn::mlcontext::MLContext) per [`WptBackend`](super::wpt_backend::WptBackend) per test thread.
///
/// When `true`, trials on the same thread share a context via `wpt_context_pool` (thread-local),
/// so no context is ever shared between libtest threads. CoreML is validated for parallel trials
/// and its Make targets use libtest's default thread count; the other backends are not validated
/// for concurrent use and their targets pass `--test-threads 1`.
pub const REUSE_ML_CONTEXT: bool = true;

/// Opt in to upstream budgets instead of the historical compatibility floors.
pub fn strict_wpt_tolerance() -> bool {
    std::env::var("WPT_STRICT_TOLERANCE").is_ok_and(|value| value == "1")
}

/// Requested CoreML policy, shared by execution and reporting (not placement).
pub fn coreml_requested_device() -> &'static str {
    parse_coreml_device(std::env::var("WPT_COREML_DEVICE").ok().as_deref())
}

fn parse_coreml_device(value: Option<&str>) -> &'static str {
    match value {
        None | Some("cpu") => "cpu",
        Some("gpu") => "gpu",
        Some("npu") => "npu",
        Some(other) => panic!("WPT_COREML_DEVICE must be cpu, gpu, or npu; got {other:?}"),
    }
}

/// CoreML float16 lowering for the trials: `WPT_COREML_FLOAT16=native` (default) or `exact`.
pub fn coreml_float16_precision() -> rustnn::mlcontext::CoremlFloat16Precision {
    parse_coreml_float16(std::env::var("WPT_COREML_FLOAT16").ok().as_deref())
}

fn parse_coreml_float16(value: Option<&str>) -> rustnn::mlcontext::CoremlFloat16Precision {
    use rustnn::mlcontext::CoremlFloat16Precision;
    match value {
        None | Some("native") => CoremlFloat16Precision::Native,
        Some("exact") => CoremlFloat16Precision::Exact,
        Some(other) => panic!("WPT_COREML_FLOAT16 must be native or exact; got {other:?}"),
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn coreml_float16_defaults_to_native_and_accepts_exact() {
        use rustnn::mlcontext::CoremlFloat16Precision;
        assert_eq!(
            super::parse_coreml_float16(None),
            CoremlFloat16Precision::Native
        );
        assert_eq!(
            super::parse_coreml_float16(Some("exact")),
            CoremlFloat16Precision::Exact
        );
    }

    #[test]
    fn coreml_policy_defaults_to_cpu_and_preserves_valid_requests() {
        assert_eq!(super::parse_coreml_device(None), "cpu");
        for value in ["cpu", "gpu", "npu"] {
            assert_eq!(super::parse_coreml_device(Some(value)), value);
        }
    }

    #[test]
    #[should_panic(expected = "WPT_COREML_DEVICE must be cpu, gpu, or npu")]
    fn invalid_coreml_policy_is_not_reported_as_a_valid_request() {
        super::parse_coreml_device(Some("all"));
    }
}
