use once_cell::sync::Lazy;
use regex::Regex;

const MODEL_NAME_ARCH_FALLBACKS: [(&str, &str); 10] = [
    (r"\bdeepseek-r1-distill-qwen\b", "qwen2"),
    (r"\bdeepseek-r1-distill-llama\b", "llama"),
    (r"\bdeepseek-v3\b", "deepseek3"),
    (r"\bdeepseek-r1\b", "deepseek3"),
    (r"\bdeepseek-v2\b", "deepseek2"),
    (r"\bmixtral\b", "mistral"),
    (r"\bcommand-r\+", "command-r-plus"),
    (r"\bcommand-r\b", "command-r"),
    (r"^\s*yi\b", "llama"),
    (r"\bglm-dsa\b", "glm-dsa"),
];

static MODEL_NAME_ARCH_FALLBACKS_REGEX: Lazy<Vec<(Regex, &'static str)>> = Lazy::new(|| {
    MODEL_NAME_ARCH_FALLBACKS
        .iter()
        .map(|(pattern, mapped)| {
            (
                Regex::new(pattern).expect("MODEL_NAME_ARCH_FALLBACKS contains an invalid regex"),
                *mapped,
            )
        })
        .collect()
});

const ARCH_SUBSTRINGS: [&str; 40] = [
    "llama",
    "mistral",
    "falcon",
    "mpt",
    "gptneox",
    "gpt2",
    "bert",
    "bloom",
    "starcoder",
    "refact",
    "command-r-plus",
    "command-r",
    "qwen2moe",
    "qwen3moe",
    "qwen2",
    "phi3",
    "phi2",
    "phi",
    "gemma3",
    "gemma2",
    "gemma",
    "stablelm",
    "deepseek2",
    "deepseek3",
    "chatglm",
    "glm-dsa",
    "glm4moe",
    "baichuan",
    "xverse",
    "orion",
    "bitnet",
    "plamo",
    "codeshell",
    "minicpm3",
    "minicpm",
    "t5",
    "jais",
    "olmo2",
    "olmo",
    "openelm",
];

pub fn detect_architecture(
    general_architecture: Option<&str>,
    general_name: Option<&str>,
) -> String {
    if let Some(arch) = general_architecture {
        if !arch.is_empty() {
            return arch.to_string();
        }
    }
    if let Some(name) = general_name {
        let lowered = name.to_lowercase();
        for (pattern, mapped) in MODEL_NAME_ARCH_FALLBACKS_REGEX.iter() {
            if pattern.is_match(&lowered) {
                return (*mapped).to_string();
            }
        }
        for arch in ARCH_SUBSTRINGS {
            if lowered.contains(arch) {
                return arch.to_string();
            }
        }
    }
    "unknown".to_string()
}

const MOE_ARCHES: [&str; 6] = [
    "qwen2moe",
    "qwen3moe",
    "deepseek2",
    "deepseek3",
    "glm4moe",
    "glm-dsa",
];

pub fn classify_architecture(architecture: &str, tensor_names: &[String]) -> String {
    let has_experts = tensor_names.iter().any(|name| {
        name.contains("ffn_gate_exps")
            || name.contains("ffn_up_exps")
            || name.contains("ffn_down_exps")
            || name.contains(".experts.")
            || name.contains("switch_mlp")
    });
    let has_dense_mlp = tensor_names.iter().any(|name| {
        (name.contains("ffn_gate.weight")
            || name.contains("ffn_up.weight")
            || name.contains("ffn_down.weight"))
            && !name.contains("_exps")
    });
    let declared_moe = MOE_ARCHES.contains(&architecture);

    if (declared_moe || has_experts) && has_dense_mlp {
        "hybrid".to_string()
    } else if declared_moe || has_experts {
        "moe".to_string()
    } else {
        "dense".to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::{classify_architecture, detect_architecture};

    #[test]
    fn respects_more_specific_substrings_first() {
        assert_eq!(detect_architecture(None, Some("phi3-mini")), "phi3");
        assert_eq!(detect_architecture(None, Some("phi2")), "phi2");
        assert_eq!(detect_architecture(None, Some("gemma2-9b")), "gemma2");
        assert_eq!(detect_architecture(None, Some("gemma3-27b")), "gemma3");
        assert_eq!(detect_architecture(None, Some("minicpm3-4b")), "minicpm3");
        assert_eq!(detect_architecture(None, Some("olmo2-13b")), "olmo2");
        assert_eq!(
            detect_architecture(None, Some("Command-R+ 104B")),
            "command-r-plus"
        );
    }

    #[test]
    fn mirrors_existing_python_fallback_examples() {
        assert_eq!(
            detect_architecture(None, Some("DeepSeek-R1-Distill-Qwen-32B")),
            "qwen2"
        );
        assert_eq!(
            detect_architecture(None, Some("Mixtral-8x7B-Instruct-v0.1")),
            "mistral"
        );
        assert_eq!(detect_architecture(None, Some("Yi-34B-Chat")), "llama");
        assert_eq!(detect_architecture(None, None), "unknown");
    }

    #[test]
    fn classifies_moe_and_hybrid_layouts() {
        assert_eq!(classify_architecture("llama", &[]), "dense");
        assert_eq!(classify_architecture("qwen3moe", &[]), "moe");
        assert_eq!(
            classify_architecture(
                "qwen3moe",
                &[
                    "blk.0.ffn_gate_exps.weight".to_string(),
                    "blk.1.ffn_gate.weight".to_string(),
                ],
            ),
            "hybrid"
        );
    }
}
