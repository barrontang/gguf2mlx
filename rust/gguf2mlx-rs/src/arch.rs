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

const ARCH_SUBSTRINGS: [&str; 39] = [
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
    "command-r",
    "command-r-plus",
    "qwen2",
    "qwen2moe",
    "qwen3moe",
    "phi3",
    "phi",
    "gemma",
    "gemma2",
    "gemma3",
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
    "minicpm",
    "minicpm3",
    "t5",
    "jais",
    "olmo",
    "olmo2",
    "openelm",
];

pub fn detect_architecture(general_architecture: Option<&str>, general_name: Option<&str>) -> String {
    if let Some(arch) = general_architecture {
        if !arch.is_empty() {
            return arch.to_string();
        }
    }
    if let Some(name) = general_name {
        let lowered = name.to_lowercase();
        for (pattern, mapped) in MODEL_NAME_ARCH_FALLBACKS {
            if Regex::new(pattern).map(|re| re.is_match(&lowered)).unwrap_or(false) {
                return mapped.to_string();
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
