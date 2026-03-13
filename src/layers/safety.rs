use crate::config::TrustConfig;
use crate::types::{RetrievedDocument, TrustAssessment};
use reqwest::Url;

pub struct TrustSafetyValidator;

impl TrustSafetyValidator {
    pub fn assess(&self, source_url: &str, content: &str, config: &TrustConfig) -> TrustAssessment {
        let mut trust_score = config.default_source_trust;
        let mut warnings = Vec::new();

        if source_url.starts_with("https://") {
            trust_score += config.https_bonus;
        } else {
            warnings.push("non_https_source".to_string());
            if config.require_https {
                warnings.push("https_required".to_string());
            }
        }

        if is_allowlisted_source(source_url, config) {
            trust_score += config.allowlist_bonus;
        }

        if source_url.contains(".gov") || source_url.contains(".edu") {
            trust_score += config.allowlist_bonus;
        }

        for noisy in [
            "ignore previous instructions",
            "act as",
            "<script",
            "buy now",
            "sponsored",
        ] {
            if content.to_lowercase().contains(noisy) {
                trust_score -= 0.12;
                warnings.push(format!("unsafe_pattern:{noisy}"));
            }
        }

        if has_parser_warning(content) {
            trust_score -= config.parser_warning_penalty;
            warnings.push("parser_warning_detected".to_string());
        }

        let accepted = (!config.require_https || source_url.starts_with("https://"))
            && trust_score >= config.min_source_trust;
        if !accepted {
            warnings.push("below_trust_threshold".to_string());
        }

        TrustAssessment {
            source_url: source_url.to_string(),
            trust_score: trust_score.clamp(0.0, 1.0),
            accepted,
            warnings,
        }
    }

    pub fn filter_documents(
        &self,
        docs: Vec<RetrievedDocument>,
        config: &TrustConfig,
    ) -> (Vec<RetrievedDocument>, Vec<String>) {
        let mut accepted = Vec::new();
        let mut warnings = Vec::new();
        for doc in docs {
            let assessment = self.assess(&doc.source_url, &doc.normalized_content, config);
            warnings.extend(assessment.warnings.clone());
            if assessment.accepted {
                let mut doc = doc;
                doc.trust_score = assessment.trust_score;
                accepted.push(doc);
            }
        }
        (accepted, warnings)
    }
}

fn is_allowlisted_source(source_url: &str, config: &TrustConfig) -> bool {
    let host = Url::parse(source_url)
        .ok()
        .and_then(|url| url.host_str().map(|host| host.to_string()))
        .unwrap_or_default();
    if host.is_empty() {
        return false;
    }

    config
        .allowlist_domains
        .iter()
        .any(|domain| host == *domain || host.ends_with(&format!(".{domain}")))
}

fn has_parser_warning(content: &str) -> bool {
    let content = content.to_ascii_lowercase();
    [
        "parser warning",
        "parse error",
        "syntaxerror",
        "access denied",
        "temporarily unavailable",
    ]
    .iter()
    .any(|pattern| content.contains(pattern))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn require_https_blocks_non_https_sources() {
        let mut config = TrustConfig::default();
        config.require_https = true;
        let assessment =
            TrustSafetyValidator.assess("http://example.com/article", "safe content", &config);
        assert!(!assessment.accepted);
        assert!(assessment
            .warnings
            .iter()
            .any(|warning| warning == "https_required"));
    }
}
