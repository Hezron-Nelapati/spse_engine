use crate::config::TrustConfig;
use crate::types::{RetrievedDocument, TrustAssessment};
use regex::Regex;
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

        for noisy in &config.unsafe_patterns {
            if content.to_lowercase().contains(noisy) {
                trust_score -= config.unsafe_pattern_penalty;
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

    /// Detect PII in text (emails, phone numbers, SSN, credit cards)
    pub fn detect_pii(&self, text: &str) -> Vec<String> {
        let mut detected = Vec::new();
        let text_lower = text.to_lowercase();

        // Email detection
        let email_pattern =
            Regex::new(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b").unwrap();
        if email_pattern.is_match(text) {
            detected.push("pii:email".to_string());
        }

        // Phone number detection (various formats)
        let phone_patterns = [
            r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b",         // US format
            r"\b\(\d{3}\)\s*\d{3}[-.]?\d{4}\b",       // (xxx) xxx-xxxx
            r"\b\+\d{1,3}[\s-]?\d{3,4}[\s-]?\d{4}\b", // International
        ];
        for pattern in phone_patterns {
            if Regex::new(pattern).unwrap().is_match(text) {
                detected.push("pii:phone".to_string());
                break;
            }
        }

        // SSN detection
        if Regex::new(r"\b\d{3}[-]?\d{2}[-]?\d{4}\b")
            .unwrap()
            .is_match(text)
        {
            detected.push("pii:ssn".to_string());
        }

        // Credit card detection
        if Regex::new(r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b")
            .unwrap()
            .is_match(text)
        {
            detected.push("pii:credit_card".to_string());
        }

        // Address detection (street addresses)
        if Regex::new(r"\b\d+\s+[A-Za-z]+\s+(Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Drive|Dr|Lane|Ln|Way|Court|Ct)\b")
            .unwrap().is_match(text)
        {
            detected.push("pii:address".to_string());
        }

        // IP address detection
        if Regex::new(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
            .unwrap()
            .is_match(text)
        {
            detected.push("pii:ip_address".to_string());
        }

        // Name patterns (common PII indicators)
        let name_indicators = ["my name is", "i am", "my ssn is", "my social security"];
        for indicator in name_indicators {
            if text_lower.contains(indicator) {
                detected.push(format!("pii:name_indicator:{}", indicator));
            }
        }

        // PII request detection - asking for personal information
        let pii_request_patterns = [
            "home address of",
            "personal address of",
            "home address for",
            "phone number of",
            "phone number for",
            "cell phone of",
            "email address of",
            "personal email of",
            "ssn of",
            "social security of",
            "private number of",
            "personal number of",
            "home address",
            "personal phone",
        ];
        for pattern in pii_request_patterns {
            if text_lower.contains(pattern) {
                detected.push(format!("pii:request:{}", pattern));
            }
        }

        detected
    }

    /// Detect harmful/dangerous content
    pub fn detect_harmful_content(&self, text: &str) -> Vec<String> {
        let mut detected = Vec::new();
        let text_lower = text.to_lowercase();

        // Violence and harm
        let violence_patterns = [
            "how to kill",
            "how to murder",
            "how to make a bomb",
            "how to create a bomb",
            "explosive recipe",
            "poison recipe",
            "how to poison",
            "terrorist",
            "terrorism guide",
            "mass shooting",
            "school shooting",
            "self harm",
            "how to commit suicide",
            "suicide methods",
            "kill yourself",
        ];
        for pattern in violence_patterns {
            if text_lower.contains(pattern) {
                detected.push(format!("harmful:violence:{}", pattern));
            }
        }

        // Illegal activities
        let illegal_patterns = [
            "how to steal",
            "how to rob",
            "shoplifting guide",
            "how to hack",
            "hacking tutorial",
            "crack password",
            "identity theft",
            "how to forge",
            "counterfeit",
            "drug manufacturing",
            "meth recipe",
            "cocaine production",
        ];
        for pattern in illegal_patterns {
            if text_lower.contains(pattern) {
                detected.push(format!("harmful:illegal:{}", pattern));
            }
        }

        // Adult content
        let adult_patterns = ["pornography", "explicit sexual", "adult content"];
        for pattern in adult_patterns {
            if text_lower.contains(pattern) {
                detected.push(format!("harmful:adult:{}", pattern));
            }
        }

        // Hate speech indicators
        let hate_patterns = [
            "kill all",
            "hate all",
            "destroy all",
            "racial slur",
            "ethnic cleansing",
        ];
        for pattern in hate_patterns {
            if text_lower.contains(pattern) {
                detected.push(format!("harmful:hate_speech:{}", pattern));
            }
        }

        detected
    }

    /// Detect prompt injection attempts
    pub fn detect_prompt_injection(&self, text: &str) -> Vec<String> {
        let mut detected = Vec::new();
        let text_lower = text.to_lowercase();

        // Direct instruction injection
        let injection_patterns = [
            "ignore previous instructions",
            "ignore all previous",
            "disregard all",
            "forget your instructions",
            "new instructions:",
            "system override",
            "admin mode",
            "developer mode",
            "debug mode enabled",
            "ignore safety",
            "bypass safety",
            "override safety",
            "disable safety",
            "ignore constraints",
            "break the rules",
        ];
        for pattern in injection_patterns {
            if text_lower.contains(pattern) {
                detected.push(format!("injection:instruction:{}", pattern));
            }
        }

        // Role manipulation
        let role_patterns = [
            "act as",
            "pretend to be",
            "roleplay as",
            "you are now",
            "your new role",
            "switch persona",
            "imagine you are",
            "play the role of",
        ];
        for pattern in role_patterns {
            if text_lower.contains(pattern) {
                detected.push(format!("injection:role:{}", pattern));
            }
        }

        // Output manipulation
        let output_patterns = [
            "print exactly",
            "output exactly",
            "repeat after me",
            "say exactly",
            "respond with only",
            "only respond",
            "do not include",
            "exclude all",
        ];
        for pattern in output_patterns {
            if text_lower.contains(pattern) {
                detected.push(format!("injection:output:{}", pattern));
            }
        }

        // Encoding/obfuscation attempts
        let obfuscation_patterns = [
            "base64:",
            "decode this:",
            "hex encoded",
            "rot13",
            "encrypted message",
        ];
        for pattern in obfuscation_patterns {
            if text_lower.contains(pattern) {
                detected.push(format!("injection:obfuscation:{}", pattern));
            }
        }

        detected
    }

    /// Comprehensive safety check combining all detectors
    pub fn check_safety(&self, text: &str) -> (bool, Vec<String>) {
        let mut all_warnings = Vec::new();
        let mut blocked = false;

        let pii = self.detect_pii(text);
        if !pii.is_empty() {
            all_warnings.extend(pii);
            blocked = true;
        }

        let harmful = self.detect_harmful_content(text);
        if !harmful.is_empty() {
            all_warnings.extend(harmful);
            blocked = true;
        }

        let injection = self.detect_prompt_injection(text);
        if !injection.is_empty() {
            all_warnings.extend(injection);
            blocked = true;
        }

        (blocked, all_warnings)
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
