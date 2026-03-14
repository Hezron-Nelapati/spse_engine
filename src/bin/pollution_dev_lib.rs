//! Pollution Development Library
//!
//! Core logic for pollution testing, config sweeping, and analysis.

use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::Path;

use spse_engine::config::{GovernanceConfig, UnitBuilderConfig};
use spse_engine::layers::{builder::UnitBuilder, hierarchy::HierarchicalUnitOrganizer, input};
use spse_engine::memory::store::MemoryStore;
use spse_engine::types::{SourceKind, UnitLevel};

// ============================================================================
// Test Content Generator
// ============================================================================

#[derive(Debug, Clone)]
pub struct TestContent {
    pub documents: Vec<TestDocument>,
    pub expected_clean: HashSet<String>,
    pub expected_polluted: HashSet<String>,
    pub total_bytes: usize,
}

#[derive(Debug, Clone)]
pub struct TestDocument {
    pub content: String,
    pub category: String,
    pub bytes: usize,
}

/// Load large corpus from test_data/large_corpus directory (70MB+)
pub fn load_large_corpus() -> Result<TestContent, Box<dyn std::error::Error>> {
    let corpus_dir = Path::new("test_data/large_corpus");
    
    if !corpus_dir.exists() {
        return Err("Large corpus not found. Run: python3 scripts/generate_large_corpus.py".into());
    }
    
    let mut documents = Vec::new();
    let mut total_bytes = 0;
    
    let categories = [
        "escaped_unicode", "outer_punct", "url_fragments", "broken_json",
        "encoding_issues", "sql_patterns", "html_markup", "control_chars",
        "emoji_content", "code_fragments", "adversarial", "whitespace_anomaly",
        "numeric_fragments", "mixed_realistic",
    ];
    
    for category in &categories {
        let file_path = corpus_dir.join(format!("{}.txt", category));
        if file_path.exists() {
            let content = fs::read_to_string(&file_path)?;
            let bytes = content.len();
            total_bytes += bytes;
            
            // Split into chunks for processing (each line is a document)
            for line in content.lines() {
                if !line.trim().is_empty() {
                    documents.push(TestDocument {
                        content: line.to_string(),
                        category: category.to_string(),
                        bytes: line.len(),
                    });
                }
            }
            
            println!("  {}: {:.2} MB", category, bytes as f64 / 1024.0 / 1024.0);
        }
    }
    
    println!("Total corpus: {:.2} MB ({} documents)", 
        total_bytes as f64 / 1024.0 / 1024.0, documents.len());
    
    Ok(TestContent {
        documents,
        expected_clean: HashSet::new(),
        expected_polluted: HashSet::new(),
        total_bytes,
    })
}

/// Generate test content with controlled pollution patterns
pub fn generate_test_content() -> TestContent {
    let mut documents = Vec::new();
    let mut expected_clean = HashSet::new();
    let mut expected_polluted = HashSet::new();

    // Category 1: Escaped Unicode (common pollution source)
    // These appear when JSON escape sequences leak into text
    let unicode_pollution_docs = vec![
        // Unicode escape sequences that should NOT become units
        ("The symbol \\u0259 represents a schwa.", "escaped_unicode"),
        ("Temperature: 90\\u00B0C (194\\u00B0F)", "escaped_unicode"),
        ("File: \\u00E9\\u00E8\\u00EA accented chars", "escaped_unicode"),
        ("Price: 99\\u20AC for the item", "escaped_unicode"),
        // These should become clean units
        ("The café has naïve façade designs.", "clean_unicode"),
        ("München and Zürich are cities.", "clean_unicode"),
    ];
    
    for (text, category) in unicode_pollution_docs {
        documents.push(TestDocument {
            content: text.to_string(),
            category: category.to_string(),
            bytes: text.len(),
        });
        
        if category == "clean_unicode" {
            for word in text.split_whitespace() {
                let clean = word.trim_matches(|c: char| !c.is_alphanumeric()).to_lowercase();
                if clean.len() >= 3 {
                    expected_clean.insert(clean);
                }
            }
        } else {
            expected_polluted.insert("u0259".to_string());
            expected_polluted.insert("u00b0".to_string());
            expected_polluted.insert("u00e9".to_string());
            expected_polluted.insert("u20ac".to_string());
        }
    }

    // Category 2: Outer Punctuation Fragments
    // Byte-window creates fragments with trailing/leading punctuation
    let punct_pollution_docs = vec![
        // These create polluted fragments like "claude-", "sudan_", "file."
        ("Claude- is an AI assistant. Claude-3 is newer.", "outer_punct"),
        ("Check the file.txt for details. file.png is an image.", "outer_punct"),
        ("Sudan_ is a country. Sudan_r is a region.", "outer_punct"),
        ("Visit https://example.com/page-1 for info.", "outer_punct"),
        ("The result: 42.5 is the answer.", "outer_punct"),
        // Clean versions
        ("Claude is an AI assistant.", "clean"),
        ("The file contains important data.", "clean"),
    ];
    
    for (text, category) in punct_pollution_docs {
        documents.push(TestDocument {
            content: text.to_string(),
            category: category.to_string(),
            bytes: text.len(),
        });
        
        if category == "clean" {
            for word in text.split_whitespace() {
                let clean = word.trim_matches(|c: char| !c.is_alphanumeric()).to_lowercase();
                if clean.len() >= 3 {
                    expected_clean.insert(clean);
                }
            }
        } else {
            expected_polluted.insert("claude-".to_string());
            expected_polluted.insert("sudan_".to_string());
            expected_polluted.insert("file.".to_string());
        }
    }

    // Category 3: Edge-Trimmed Partial Words
    // Byte-window cuts words mid-character, creating partials
    let edge_trim_docs = vec![
        // These create partials like "atholic", "audel", "ubang"
        ("Catholic Church is ancient. Catholic traditions continue.", "edge_trim"),
        ("Baudelaire was a poet. Baudelaire wrote Les Fleurs.", "edge_trim"),
        ("Subang is a town. Subang Jaya is a city.", "edge_trim"),
        // Clean versions
        ("The Catholic Church has traditions.", "clean"),
        ("Baudelaire was a famous poet.", "clean"),
    ];
    
    for (text, category) in edge_trim_docs {
        documents.push(TestDocument {
            content: text.to_string(),
            category: category.to_string(),
            bytes: text.len(),
        });
        
        if category == "clean" {
            for word in text.split_whitespace() {
                let clean = word.trim_matches(|c: char| !c.is_alphanumeric()).to_lowercase();
                if clean.len() >= 3 {
                    expected_clean.insert(clean);
                }
            }
        } else {
            expected_polluted.insert("atholic".to_string());
            expected_polluted.insert("audelaire".to_string());
            expected_polluted.insert("ubang".to_string());
        }
    }

    // Category 4: URL/Path Fragments
    let url_docs = vec![
        ("Download from https://example.com/file-v2.3.1.tar.gz", "url_fragments"),
        ("Path: /home/user/documents/report-2024.pdf", "url_fragments"),
        ("API endpoint: https://api.example.com/v1/users/123", "url_fragments"),
        ("The map.png file is an image.", "url_fragments"),
    ];
    
    for (text, category) in url_docs {
        documents.push(TestDocument {
            content: text.to_string(),
            category: category.to_string(),
            bytes: text.len(),
        });
    }
    
    expected_polluted.insert("-map.png".to_string());
    expected_polluted.insert("api/".to_string());
    expected_polluted.insert("v1/".to_string());

    // Category 5: Repetitive Content (should create good units)
    let repetitive_docs = vec![
        ("reasoning reasoning reasoning is important for AI.", "repetitive"),
        ("The anchor anchor anchor provides stability.", "repetitive"),
        ("Memory memory memory stores information.", "repetitive"),
    ];
    
    for (text, category) in repetitive_docs {
        documents.push(TestDocument {
            content: text.to_string(),
            category: category.to_string(),
            bytes: text.len(),
        });
        
        for word in text.split_whitespace() {
            let clean = word.trim_matches(|c: char| !c.is_alphanumeric()).to_lowercase();
            if clean.len() >= 3 && clean != "is" && clean != "the" && clean != "for" {
                expected_clean.insert(clean);
            }
        }
    }

    // Category 6: Mixed Content (realistic test)
    let mixed_docs = vec![
        (r#"The temperature today is 25°C (77°F). The café "L'Étoile" serves coffee.
            Visit https://example.com for more info.
            The Catholic Church in München has historical significance.
            Reasoning about these topics requires careful analysis."#, "mixed"),
        (r#"Data: {"temp": 90, "unit": "°C", "file": "report-2024.pdf"}
            The analysis shows significant patterns.
            Subang Jaya and Ulan-Ude are cities in Asia."#, "mixed"),
    ];
    
    for (text, category) in mixed_docs {
        documents.push(TestDocument {
            content: text.to_string(),
            category: category.to_string(),
            bytes: text.len(),
        });
        
        if category == "mixed" {
            expected_clean.insert("temperature".to_string());
            expected_clean.insert("café".to_string());
            expected_clean.insert("catholic".to_string());
            expected_clean.insert("reasoning".to_string());
            expected_clean.insert("analysis".to_string());
        }
    }

    // Category 7: Known Pollution Patterns from TSV
    let known_pollution_patterns = vec![
        ("E.u0259. is a notation. E.u0259 appears often.", "known_pollution"),
        ("sudan_ is a variant. sudan_ appears in data.", "known_pollution"),
        ("Colucci. is a name. Colucci. appears frequently.", "known_pollution"),
        ("Claude_D is a version. Claude_D is referenced.", "known_pollution"),
        ("lan-Ude is a city. lan-Ude is in Russia.", "known_pollution"),
    ];
    
    for (text, category) in known_pollution_patterns {
        documents.push(TestDocument {
            content: text.to_string(),
            category: category.to_string(),
            bytes: text.len(),
        });
    }
    
    // Add known pollution patterns
    expected_polluted.insert("e.u0259.".to_string());
    expected_polluted.insert("sudan_".to_string());
    expected_polluted.insert("colucci.".to_string());
    expected_polluted.insert("claude_d".to_string());
    expected_polluted.insert("lan-ude".to_string());

    // ========================================================================
    // ILL-MANNERED / MALFORMED INPUTS (Expanded Corpus)
    // ========================================================================

    // Category 8: Null bytes and control characters
    let control_char_docs = vec![
        ("Text with\x00null\x00bytes inside.", "control_chars"),
        ("Line1\nLine2\nLine3\nLine4\nLine5", "control_chars"),
        ("Tab\there\tand\tthere", "control_chars"),
        ("Carriage\rreturn\rtest", "control_chars"),
        ("Mixed\r\nline\r\nendings", "control_chars"),
        ("Bell\x07and\x08backspace", "control_chars"),
    ];
    
    for (text, category) in control_char_docs {
        documents.push(TestDocument {
            content: text.to_string(),
            category: category.to_string(),
            bytes: text.len(),
        });
    }

    // Category 9: Broken/malformed JSON fragments
    let broken_json_docs = vec![
        (r#"{"key": "value", "broken: missing_quote}"#, "broken_json"),
        (r#"{"nested": {"deep": {"unclosed": "yes""#, "broken_json"),
        (r#"[1, 2, 3, {"mixed": array}]"#, "broken_json"),
        (r#"{"escaped": \"bad\", "quote": "unmatched}"#, "broken_json"),
        (r#"{"unicode": "\u00", "truncated": true}"#, "broken_json"),
        (r#"{"a":1,"b":2,"c":3,"d":4,"e":5,"f":6,"g":7}"#, "broken_json"),
        (r#"{"key": "value with \n newline \t tab"}"#, "broken_json"),
    ];
    
    for (text, category) in broken_json_docs {
        documents.push(TestDocument {
            content: text.to_string(),
            category: category.to_string(),
            bytes: text.len(),
        });
    }

    // Category 10: Mixed encodings and mojibake
    let encoding_docs = vec![
        ("CafÃ© with mojibake from UTF-8 misdecoded as Latin-1", "encoding_issues"),
        ("Ã¼ber Ã¶ffentlich Ã¤ndern", "encoding_issues"),
        ("ÐÑÑÑÐ¸Ð¹ (Russian in wrong encoding)", "encoding_issues"),
        ("æ—¥æœ¬èªž (Japanese mojibake)", "encoding_issues"),
        ("Valid UTF-8: 日本語 한국어 العربية", "encoding_issues"),
        ("Mixed: café, über, naïve, 日本語", "encoding_issues"),
    ];
    
    for (text, category) in encoding_docs {
        documents.push(TestDocument {
            content: text.to_string(),
            category: category.to_string(),
            bytes: text.len(),
        });
    }

    // Category 11: Extremely long tokens and repeated characters
    let extreme_length_docs = vec![
        ("Aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", "extreme_length"),
        ("Supercalifragilisticexpialidocious is a long word.", "extreme_length"),
        ("Pneumonoultramicroscopicsilicovolcanoconiosis is even longer.", "extreme_length"),
        ("Hippopotomonstrosesquippedaliophobia is ironic.", "extreme_length"),
        ("Repetition: aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", "extreme_length"),
        ("Numbers: 1234567890123456789012345678901234567890", "extreme_length"),
    ];
    
    for (text, category) in extreme_length_docs {
        documents.push(TestDocument {
            content: text.to_string(),
            category: category.to_string(),
            bytes: text.len(),
        });
    }

    // Category 12: SQL injection-like patterns (should not pollute)
    let sql_patterns_docs = vec![
        ("SELECT * FROM users WHERE id = 1", "sql_patterns"),
        ("'; DROP TABLE users; --", "sql_patterns"),
        ("UNION SELECT password FROM admin", "sql_patterns"),
        ("1' OR '1'='1", "sql_patterns"),
        ("INSERT INTO logs VALUES ('test')", "sql_patterns"),
    ];
    
    for (text, category) in sql_patterns_docs {
        documents.push(TestDocument {
            content: text.to_string(),
            category: category.to_string(),
            bytes: text.len(),
        });
    }

    // Category 13: HTML/XML fragments and markup
    let markup_docs = vec![
        ("<div class=\"container\"><p>Text</p></div>", "markup"),
        ("&lt;escaped&gt; &amp; &quot;entities&quot;", "markup"),
        ("<![CDATA[Some <cdata> content]]>", "markup"),
        ("<!-- HTML comment --><span>visible</span>", "markup"),
        ("<script>alert('xss')</script>", "markup"),
        ("Text with <b>bold</b> and <i>italic</i>", "markup"),
    ];
    
    for (text, category) in markup_docs {
        documents.push(TestDocument {
            content: text.to_string(),
            category: category.to_string(),
            bytes: text.len(),
        });
    }

    // Category 14: Email and address fragments
    let contact_docs = vec![
        ("Contact: user@example.com for support", "contact_fragments"),
        ("Email: john.doe+tag@subdomain.example.org", "contact_fragments"),
        ("Phone: +1 (555) 123-4567", "contact_fragments"),
        ("Address: 123 Main St, City, State 12345", "contact_fragments"),
        ("IP: 192.168.1.1 is private", "contact_fragments"),
        ("MAC: 00:1A:2B:3C:4D:5E", "contact_fragments"),
    ];
    
    for (text, category) in contact_docs {
        documents.push(TestDocument {
            content: text.to_string(),
            category: category.to_string(),
            bytes: text.len(),
        });
    }

    // Category 15: Base64 and encoded content
    let encoded_docs = vec![
        ("Base64: SGVsbG8gV29ybGQh", "encoded_content"),
        ("Hex: 48656c6c6f20576f726c64", "encoded_content"),
        ("URL encoded: Hello%20World%21", "encoded_content"),
        ("HTML entity: &#72;&#101;&#108;&#108;&#111;", "encoded_content"),
        ("Quoted printable: Hello=20World=21", "encoded_content"),
    ];
    
    for (text, category) in encoded_docs {
        documents.push(TestDocument {
            content: text.to_string(),
            category: category.to_string(),
            bytes: text.len(),
        });
    }

    // Category 16: Whitespace anomalies
    let whitespace_docs = vec![
        ("Multiple   spaces   between   words", "whitespace_anomaly"),
        ("Tabs\t\t\tmultiple\t\ttabs", "whitespace_anomaly"),
        ("  Leading and trailing  ", "whitespace_anomaly"),
        ("Non-breaking\u{00A0}space\u{00A0}here", "whitespace_anomaly"),
        ("Zero\u{200B}width\u{200B}space", "whitespace_anomaly"),
        ("Em\u{2003}space\u{2003}wide", "whitespace_anomaly"),
    ];
    
    for (text, category) in whitespace_docs {
        documents.push(TestDocument {
            content: text.to_string(),
            category: category.to_string(),
            bytes: text.len(),
        });
    }

    // Category 17: Numeric and date fragments
    let numeric_docs = vec![
        ("Date: 2024-01-15T10:30:00Z", "numeric_fragments"),
        ("Time: 10:30:45.123", "numeric_fragments"),
        ("Money: $1,234.56 USD", "numeric_fragments"),
        ("Percentage: 99.9%", "numeric_fragments"),
        ("Scientific: 1.23e-10", "numeric_fragments"),
        ("Hex color: #FF5733", "numeric_fragments"),
        ("UUID: 550e8400-e29b-41d4-a716-446655440000", "numeric_fragments"),
        ("Version: v2.3.1-beta.2+build.123", "numeric_fragments"),
    ];
    
    for (text, category) in numeric_docs {
        documents.push(TestDocument {
            content: text.to_string(),
            category: category.to_string(),
            bytes: text.len(),
        });
    }

    // Category 18: Emoji and special unicode
    let emoji_docs = vec![
        ("Hello 👋 World 🌍!", "emoji_content"),
        ("Math: 2×3=6, a²+b²=c²", "emoji_content"),
        ("Arrows: → ← ↑ ↓ ↔", "emoji_content"),
        ("Symbols: © ® ™ § ¶", "emoji_content"),
        ("Currency: $ € £ ¥ ₹ ₿", "emoji_content"),
        ("Dingbats: ✆ ✇ ✈ ✉ ☎ ☏", "emoji_content"),
    ];
    
    for (text, category) in emoji_docs {
        documents.push(TestDocument {
            content: text.to_string(),
            category: category.to_string(),
            bytes: text.len(),
        });
    }

    // Category 19: Programming language fragments
    let code_fragments_docs = vec![
        ("fn main() { println!(\"Hello\"); }", "code_fragments"),
        ("def function(): return True", "code_fragments"),
        ("const x = () => { return 42; };", "code_fragments"),
        ("public static void main(String[] args)", "code_fragments"),
        ("#include <stdio.h>\nint main()", "code_fragments"),
        ("package main\nimport \"fmt\"", "code_fragments"),
    ];
    
    for (text, category) in code_fragments_docs {
        documents.push(TestDocument {
            content: text.to_string(),
            category: category.to_string(),
            bytes: text.len(),
        });
    }

    // Category 20: Adversarial patterns
    let adversarial_docs = vec![
        ("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA", "adversarial"),
        ("a a a a a a a a a a a a a a a a a a a a", "adversarial"),
        ("wordwordwordwordwordwordwordwordwordword", "adversarial"),
        ("AaAaAaAaAaAaAaAaAaAaAaAaAaAaAaAaAaAaAaAa", "adversarial"),
        ("123123123123123123123123123123123123123", "adversarial"),
        ("!@#$%^&*()_+-=[]{}|;':\",./<>?", "adversarial"),
    ];
    
    for (text, category) in adversarial_docs {
        documents.push(TestDocument {
            content: text.to_string(),
            category: category.to_string(),
            bytes: text.len(),
        });
    }

    // Category 21: Real-world messy data
    let messy_real_docs = vec![
        (r#"{"question":"What is 2+2?","context":"Math basics","answer":"4"}"#, "messy_real"),
        ("User123 commented: 'Great article!!! 5/5 stars 👍👍👍'", "messy_real"),
        ("RT @user: Check this out! https://t.co/abc123 #hashtag", "messy_real"),
        ("[ERROR] 2024-01-15 10:30:45 - Connection failed (code: 500)", "messy_real"),
        ("SELECT u.name, COUNT(*) FROM users u JOIN posts p ON u.id = p.user_id", "messy_real"),
        ("Lorem ipsum dolor sit amet, consectetur adipiscing elit.", "messy_real"),
    ];
    
    for (text, category) in messy_real_docs {
        documents.push(TestDocument {
            content: text.to_string(),
            category: category.to_string(),
            bytes: text.len(),
        });
    }

    // Category 22: Empty and minimal content
    let minimal_docs = vec![
        ("", "minimal"),
        (" ", "minimal"),
        ("a", "minimal"),
        ("ab", "minimal"),
        ("...", "minimal"),
        ("---", "minimal"),
        ("___", "minimal"),
    ];
    
    for (text, category) in minimal_docs {
        documents.push(TestDocument {
            content: text.to_string(),
            category: category.to_string(),
            bytes: text.len(),
        });
    }

    let total_bytes: usize = documents.iter().map(|d| d.bytes).sum();
    
    TestContent {
        documents,
        expected_clean,
        expected_polluted,
        total_bytes,
    }
}

// ============================================================================
// Pollution Config
// ============================================================================

#[derive(Debug, Clone)]
pub struct PollutionConfig {
    pub min_frequency: u64,
    pub window_sizes: Vec<usize>,
    pub min_fragment_length: usize,
    pub punctuation_ratio: f32,
    pub utility_threshold: f32,
    pub full_boundary_bonus: f32,
    pub edge_boundary_bonus: f32,
    pub no_boundary_penalty: f32,
    pub global_corroboration_freq: u64,
}

impl Default for PollutionConfig {
    fn default() -> Self {
        Self {
            min_frequency: 3,
            window_sizes: vec![3, 4, 5, 6, 7, 8],
            min_fragment_length: 5,
            punctuation_ratio: 0.40,
            utility_threshold: 0.10,
            full_boundary_bonus: 0.22,
            edge_boundary_bonus: 0.03,
            no_boundary_penalty: -0.18,
            global_corroboration_freq: 4,
        }
    }
}

impl PollutionConfig {
    pub fn to_unit_builder_config(&self) -> UnitBuilderConfig {
        UnitBuilderConfig {
            min_frequency_threshold: self.min_frequency,
            rolling_hash_window_sizes: self.window_sizes.clone(),
            min_fragment_length: self.min_fragment_length,
            punctuation_ratio_limit: self.punctuation_ratio,
            utility_full_boundary_bonus: self.full_boundary_bonus,
            utility_edge_boundary_bonus: self.edge_boundary_bonus,
            utility_no_boundary_penalty: self.no_boundary_penalty,
            global_corroboration_frequency_threshold: self.global_corroboration_freq,
            ..UnitBuilderConfig::default()
        }
    }
    
    pub fn to_governance_config(&self) -> GovernanceConfig {
        GovernanceConfig {
            pollution_detection_enabled: true,
            pollution_min_length: self.min_fragment_length,
            pollution_edge_trim_limit: 3,
            pollution_overlap_threshold: 0.70,
            pollution_quality_margin: 0.08,
            ..GovernanceConfig::default()
        }
    }
}

// ============================================================================
// Pollution Result
// ============================================================================

#[derive(Debug, Clone)]
pub struct PollutionResult {
    pub total_units: usize,
    pub clean_units: usize,
    pub polluted_units: usize,
    pub pollution_score: f32,
    pub pollution_by_category: HashMap<String, usize>,
    pub top_polluted: Vec<PollutedUnit>,
    pub config_summary: ConfigSummary,
    pub config_yaml_patch: String,
}

#[derive(Debug, Clone)]
pub struct PollutedUnit {
    pub content: String,
    pub normalized: String,
    pub score: f32,
    pub reasons: Vec<String>,
    pub level: UnitLevel,
    pub frequency: u64,
}

#[derive(Debug, Clone)]
pub struct ConfigSummary {
    pub min_frequency: u64,
    pub window_sizes: Vec<usize>,
    pub min_fragment_length: usize,
    pub punctuation_ratio: f32,
}

// ============================================================================
// Pollution Test Runner
// ============================================================================

/// Run a pollution test with the given content and config
pub fn run_pollution_test(
    test_content: &TestContent,
    config: &PollutionConfig,
    db_path: &Path,
) -> Result<PollutionResult, Box<dyn std::error::Error>> {
    let mut store = MemoryStore::new_with_governance(
        db_path.to_str().expect("valid path"),
        &config.to_governance_config(),
    );
    
    let builder_config = config.to_unit_builder_config();
    
    // Batch documents for efficiency (combine multiple small docs into one)
    let batch_size = 1000;
    let mut batch = String::new();
    let mut batch_count = 0;
    
    for (i, doc) in test_content.documents.iter().enumerate() {
        batch.push_str(&doc.content);
        batch.push('\n');
        
        if batch.len() >= 50000 || i == test_content.documents.len() - 1 {
            let packet = input::ingest_raw(&batch, true);
            let output = UnitBuilder::ingest_with_config(&packet, &builder_config);
            let hierarchy = HierarchicalUnitOrganizer::organize(&output, &builder_config);
            store.ingest_hierarchy(&hierarchy, SourceKind::TrainingDocument, &format!("batch_{}", batch_count));
            
            batch.clear();
            batch_count += 1;
            
            if batch_count % 100 == 0 {
                print!(".");
                std::io::Write::flush(&mut std::io::stdout()).ok();
            }
        }
    }
    
    println!(" Processed {} batches", batch_count);
    
    // Flush any remaining pending writes
    store.flush_pending_writes();
    
    // Analyze pollution
    analyze_pollution(&store, config, test_content)
}

fn analyze_pollution(
    store: &MemoryStore,
    config: &PollutionConfig,
    _test_content: &TestContent,
) -> Result<PollutionResult, Box<dyn std::error::Error>> {
    let units = store.all_units();
    let mut polluted_units = Vec::new();
    let mut pollution_by_category: HashMap<String, usize> = HashMap::new();
    
    for unit in &units {
        let pollution = detect_pollution(&unit.content, &unit.normalized, unit.level, unit.frequency);
        if !pollution.reasons.is_empty() {
            for reason in &pollution.reasons {
                *pollution_by_category.entry(reason.clone()).or_insert(0) += 1;
            }
            polluted_units.push(PollutedUnit {
                content: unit.content.clone(),
                normalized: unit.normalized.clone(),
                score: pollution.score,
                reasons: pollution.reasons,
                level: unit.level,
                frequency: unit.frequency,
            });
        }
    }
    
    // Sort by pollution score
    polluted_units.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
    
    let total_units = units.len();
    let polluted_count = polluted_units.len();
    let clean_count = total_units.saturating_sub(polluted_count);
    
    // Calculate pollution score (0 = no pollution, 1 = all polluted)
    let pollution_ratio = if total_units > 0 {
        polluted_count as f32 / total_units as f32
    } else {
        0.0
    };
    
    // Factor in severity
    let avg_severity = if !polluted_units.is_empty() {
        polluted_units.iter().map(|u| u.score).sum::<f32>() / polluted_units.len() as f32
    } else {
        0.0
    };
    
    let pollution_score = pollution_ratio * (0.5 + 0.5 * avg_severity);
    
    Ok(PollutionResult {
        total_units,
        clean_units: clean_count,
        polluted_units: polluted_count,
        pollution_score,
        pollution_by_category,
        top_polluted: polluted_units.into_iter().take(20).collect(),
        config_summary: ConfigSummary {
            min_frequency: config.min_frequency,
            window_sizes: config.window_sizes.clone(),
            min_fragment_length: config.min_fragment_length,
            punctuation_ratio: config.punctuation_ratio,
        },
        config_yaml_patch: String::new(),
    })
}

struct PollutionDetection {
    score: f32,
    reasons: Vec<String>,
}

fn detect_pollution(content: &str, normalized: &str, level: UnitLevel, frequency: u64) -> PollutionDetection {
    let mut score: f32 = 0.0;
    let mut reasons = Vec::new();
    
    // Check 1: Escaped unicode patterns (uXXXX or uXXXXXX)
    if normalized.contains("u0") && normalized.len() <= 10 {
        let unicode_pattern = regex::Regex::new(r"u[0-9a-f]{4,6}").unwrap();
        if unicode_pattern.is_match(normalized) {
            score += 0.8;
            reasons.push("escaped_unicode".to_string());
        }
    }
    
    // Check 2: Outer punctuation (trailing/leading non-alphanumeric)
    let trimmed = content.trim_matches(|c: char| !c.is_alphanumeric());
    if trimmed.len() < content.len() && trimmed.len() >= 2 {
        let has_leading_punct = !content.starts_with(|c: char| c.is_alphanumeric());
        let has_trailing_punct = !content.ends_with(|c: char| c.is_alphanumeric());
        
        if has_leading_punct || has_trailing_punct {
            // Check if it's a legitimate hyphenated word
            let punct_count = content.chars().filter(|c| !c.is_alphanumeric()).count();
            let alpha_count = content.chars().filter(|c| c.is_alphanumeric()).count();
            
            if punct_count > 0 && alpha_count > 0 {
                let punct_ratio = punct_count as f32 / content.len() as f32;
                if punct_ratio > 0.15 && !content.contains(|c: char| c.is_whitespace()) {
                    score += 0.6;
                    reasons.push("outer_punct".to_string());
                }
            }
        }
    }
    
    // Check 3: Edge-trimmed partial words
    if level == UnitLevel::Word || level == UnitLevel::Subword {
        // Check if it's a partial of a common word
        let partial_patterns = [
            ("atholic", "catholic"),
            ("audel", "baudelaire"),
            ("ubang", "subang"),
            ("her team", "her team"), // legitimate phrase
        ];
        
        for (partial, _) in &partial_patterns {
            if normalized.contains(partial) && normalized.len() < 12 {
                // Check if it's not a full word
                if normalized != *partial {
                    score += 0.5;
                    reasons.push("edge_trim".to_string());
                    break;
                }
            }
        }
    }
    
    // Check 4: URL/path fragments
    if normalized.contains('/') || normalized.contains(':') {
        if normalized.len() < 20 && !normalized.contains(' ') {
            // Exclude time formats like "10:30", "12:45:00"
            let is_time_format = normalized.chars().all(|c| c.is_ascii_digit() || c == ':');
            if !is_time_format {
                score += 0.7;
                reasons.push("urlish".to_string());
            }
        }
    }
    
    // Check 5: File extension fragments
    let ext_pattern = regex::Regex::new(r"\.(png|jpg|pdf|txt|gz|tar|zip)$").unwrap();
    if ext_pattern.is_match(normalized) && normalized.len() < 15 {
        score += 0.5;
        reasons.push("file_extension".to_string());
    }
    
    // Check 6: Subword pollution (too short for word level)
    if level == UnitLevel::Subword && normalized.len() >= 4 && normalized.len() <= 6 {
        // Check if it's all lowercase and looks like a word fragment
        if normalized.chars().all(|c| c.is_ascii_lowercase()) {
            // Common English word fragments that are pollution
            let common_fragments = ["tion", "atio", "ness", "ment", "able", "ible"];
            if common_fragments.contains(&normalized.as_ref()) {
                // These might be legitimate suffixes, lower score
                score += 0.2;
                reasons.push("suffix_fragment".to_string());
            }
        }
    }
    
    // Check 7: High frequency pollution (repeated bad patterns)
    if frequency > 5 && score > 0.0 {
        score *= 1.2; // Amplify pollution score for frequent pollution
        reasons.push("high_frequency".to_string());
    }
    
    PollutionDetection { score: score.min(1.0_f32), reasons }
}

// ============================================================================
// Config Sweep
// ============================================================================

pub fn config_sweep(test_content: &TestContent) -> Result<Vec<(String, PollutionResult)>, Box<dyn std::error::Error>> {
    let mut results = Vec::new();
    
    // Define config variations to test
    let configs = generate_config_variations();
    
    for (name, config) in configs {
        let db_path = std::env::temp_dir().join(format!("pollution_sweep_{}.db", uuid::Uuid::new_v4()));
        
        let result = run_pollution_test(test_content, &config, &db_path)?;
        
        let mut result_with_config = result.clone();
        result_with_config.config_yaml_patch = generate_yaml_patch(&config);
        
        results.push((name, result_with_config));
        
        // Cleanup
        let _ = std::fs::remove_file(&db_path);
    }
    
    Ok(results)
}

fn generate_config_variations() -> Vec<(String, PollutionConfig)> {
    let mut configs = Vec::new();
    
    // Default config
    configs.push(("default".to_string(), PollutionConfig::default()));
    
    // Variation 1: Higher frequency threshold (stricter)
    configs.push(("high_freq_threshold".to_string(), PollutionConfig {
        min_frequency: 4,
        ..PollutionConfig::default()
    }));
    
    // Variation 2: Larger minimum window (avoid small fragments)
    configs.push(("larger_min_window".to_string(), PollutionConfig {
        window_sizes: vec![4, 5, 6, 7, 8],
        min_fragment_length: 5,
        ..PollutionConfig::default()
    }));
    
    // Variation 3: Stricter punctuation handling
    configs.push(("strict_punct".to_string(), PollutionConfig {
        punctuation_ratio: 0.35,
        no_boundary_penalty: -0.25,
        edge_boundary_bonus: 0.02,
        ..PollutionConfig::default()
    }));
    
    // Variation 4: Higher utility threshold
    configs.push(("high_utility".to_string(), PollutionConfig {
        utility_threshold: 0.15,
        full_boundary_bonus: 0.25,
        ..PollutionConfig::default()
    }));
    
    // Variation 5: Combined strict settings
    configs.push(("strict_combined".to_string(), PollutionConfig {
        min_frequency: 3,
        window_sizes: vec![3, 4, 5, 6, 7, 8],
        min_fragment_length: 5,
        punctuation_ratio: 0.40,
        utility_threshold: 0.12,
        full_boundary_bonus: 0.22,
        edge_boundary_bonus: 0.03,
        no_boundary_penalty: -0.18,
        ..PollutionConfig::default()
    }));
    
    // Variation 6: Very strict (may filter too much)
    configs.push(("very_strict".to_string(), PollutionConfig {
        min_frequency: 5,
        window_sizes: vec![5, 6, 7, 8],
        min_fragment_length: 6,
        punctuation_ratio: 0.30,
        utility_threshold: 0.20,
        full_boundary_bonus: 0.30,
        edge_boundary_bonus: 0.01,
        no_boundary_penalty: -0.30,
        ..PollutionConfig::default()
    }));
    
    // Variation 7: Focus on boundary detection
    configs.push(("boundary_focused".to_string(), PollutionConfig {
        full_boundary_bonus: 0.35,
        edge_boundary_bonus: 0.08,
        no_boundary_penalty: -0.35,
        ..PollutionConfig::default()
    }));
    
    // Variation 8: Lenient (for comparison)
    configs.push(("lenient".to_string(), PollutionConfig {
        min_frequency: 1,
        window_sizes: vec![2, 3, 4, 5, 6, 7, 8],
        min_fragment_length: 3,
        punctuation_ratio: 0.65,
        utility_threshold: 0.05,
        no_boundary_penalty: -0.05,
        ..PollutionConfig::default()
    }));
    
    // Variation 9: Anti-unicode pollution specific
    configs.push(("anti_unicode".to_string(), PollutionConfig {
        min_fragment_length: 5,
        punctuation_ratio: 0.35,
        ..PollutionConfig::default()
    }));
    
    // Variation 10: Balanced optimal (educated guess)
    configs.push(("balanced_optimal".to_string(), PollutionConfig {
        min_frequency: 3,
        window_sizes: vec![3, 4, 5, 6, 7, 8],
        min_fragment_length: 4,
        punctuation_ratio: 0.45,
        utility_threshold: 0.10,
        full_boundary_bonus: 0.20,
        edge_boundary_bonus: 0.04,
        no_boundary_penalty: -0.15,
        global_corroboration_freq: 4,
    }));
    
    configs
}

fn generate_yaml_patch(config: &PollutionConfig) -> String {
    format!(
        r#"
layer_2_unit_builder:
  min_frequency_threshold: {}
  rolling_hash_window_sizes: {:?}
  min_fragment_length: {}
  punctuation_ratio_limit: {}
  utility_full_boundary_bonus: {}
  utility_edge_boundary_bonus: {}
  utility_no_boundary_penalty: {}

layer_21_memory_governance:
  pollution_min_length: {}
  pollution_detection_enabled: true
"#,
        config.min_frequency,
        config.window_sizes,
        config.min_fragment_length,
        config.punctuation_ratio,
        config.full_boundary_bonus,
        config.edge_boundary_bonus,
        config.no_boundary_penalty,
        config.min_fragment_length
    )
}

// ============================================================================
// Pollution Report
// ============================================================================

#[derive(Debug)]
pub struct PollutionReport {
    pub total_units: usize,
    pub polluted_units: usize,
    pub pollution_ratio: f32,
    pub pollution_by_category: HashMap<String, usize>,
    pub top_polluted: Vec<PollutedUnit>,
    pub recommendations: Vec<String>,
}

pub fn pollution_report(db_path: &Path) -> Result<PollutionReport, Box<dyn std::error::Error>> {
    let store = MemoryStore::new(db_path.to_str().expect("valid path"));
    let units = store.all_units();
    
    let mut polluted_units = Vec::new();
    let mut pollution_by_category: HashMap<String, usize> = HashMap::new();
    
    for unit in &units {
        let pollution = detect_pollution(&unit.content, &unit.normalized, unit.level, unit.frequency);
        if !pollution.reasons.is_empty() {
            for reason in &pollution.reasons {
                *pollution_by_category.entry(reason.clone()).or_insert(0) += 1;
            }
            polluted_units.push(PollutedUnit {
                content: unit.content.clone(),
                normalized: unit.normalized.clone(),
                score: pollution.score,
                reasons: pollution.reasons,
                level: unit.level,
                frequency: unit.frequency,
            });
        }
    }
    
    polluted_units.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
    
    let total_units = units.len();
    let polluted_count = polluted_units.len();
    let pollution_ratio = if total_units > 0 {
        polluted_count as f32 / total_units as f32
    } else {
        0.0
    };
    
    // Generate recommendations
    let mut recommendations = Vec::new();
    
    if pollution_by_category.get("escaped_unicode").unwrap_or(&0) > &5 {
        recommendations.push(
            "Add unicode escape detection in normalize_window() to reject \\uXXXX patterns".to_string()
        );
    }
    
    if pollution_by_category.get("outer_punct").unwrap_or(&0) > &10 {
        recommendations.push(
            "Increase min_fragment_length and add outer punctuation stripping in should_reject_fragment()".to_string()
        );
    }
    
    if pollution_by_category.get("edge_trim").unwrap_or(&0) > &5 {
        recommendations.push(
            "Increase rolling_hash_window_sizes minimum to avoid byte-boundary cuts".to_string()
        );
    }
    
    if pollution_by_category.get("urlish").unwrap_or(&0) > &3 {
        recommendations.push(
            "Add URL/path detection in normalize_window() to reject fragments containing / or :".to_string()
        );
    }
    
    if pollution_ratio > 0.15 {
        recommendations.push(
            "Overall pollution is high. Consider running config sweep with --sweep to find optimal settings".to_string()
        );
    }
    
    Ok(PollutionReport {
        total_units,
        polluted_units: polluted_count,
        pollution_ratio,
        pollution_by_category,
        top_polluted: polluted_units.into_iter().take(20).collect(),
        recommendations,
    })
}

// ============================================================================
// Pattern Analysis
// ============================================================================

#[derive(Debug)]
pub struct PatternAnalysis {
    pub patterns: Vec<PollutionPattern>,
    pub suggested_fixes: Vec<SuggestedFix>,
}

#[derive(Debug)]
pub struct PollutionPattern {
    pub name: String,
    pub frequency: usize,
    pub example: String,
    pub root_cause: String,
    pub suggested_fix: String,
}

#[derive(Debug)]
pub struct SuggestedFix {
    pub description: String,
    pub location: String,
    pub priority: String,
    pub implementation_hint: String,
}

pub fn analyze_pollution_patterns(test_content: &TestContent) -> Result<PatternAnalysis, Box<dyn std::error::Error>> {
    let mut patterns = Vec::new();
    
    // Pattern 1: Escaped Unicode
    patterns.push(PollutionPattern {
        name: "escaped_unicode".to_string(),
        frequency: test_content.documents.iter()
            .filter(|d| d.category == "escaped_unicode")
            .count(),
        example: "u0259, u00B0, u00E9".to_string(),
        root_cause: "JSON escape sequences (\\uXXXX) are not filtered during text normalization".to_string(),
        suggested_fix: "Add regex filter in normalize_window() to detect and reject \\uXXXX patterns".to_string(),
    });
    
    // Pattern 2: Outer Punctuation
    patterns.push(PollutionPattern {
        name: "outer_punct".to_string(),
        frequency: test_content.documents.iter()
            .filter(|d| d.category == "outer_punct")
            .count(),
        example: "claude-, sudan_, file.".to_string(),
        root_cause: "Byte-window captures text with trailing/leading punctuation that isn't stripped".to_string(),
        suggested_fix: "Strengthen should_reject_fragment() to reject units with outer punctuation unless full_token_boundary_hits > 0".to_string(),
    });
    
    // Pattern 3: Edge Trimmed
    patterns.push(PollutionPattern {
        name: "edge_trim".to_string(),
        frequency: test_content.documents.iter()
            .filter(|d| d.category == "edge_trim")
            .count(),
        example: "atholic, audel, ubang".to_string(),
        root_cause: "Rolling hash window cuts words at byte boundaries, creating partial words".to_string(),
        suggested_fix: "Increase minimum window size and require full_token_boundary_hits for single-token units".to_string(),
    });
    
    // Pattern 4: URL Fragments
    patterns.push(PollutionPattern {
        name: "url_fragments".to_string(),
        frequency: test_content.documents.iter()
            .filter(|d| d.category == "url_fragments")
            .count(),
        example: "-map.png, api/, v1/".to_string(),
        root_cause: "URLs and paths are processed as regular text, creating fragments".to_string(),
        suggested_fix: "Add URL detection in normalize_window() and reject fragments matching URL patterns".to_string(),
    });
    
    // Generate suggested fixes
    let suggested_fixes = vec![
        SuggestedFix {
            description: "Add unicode escape detection".to_string(),
            location: "src/layers/builder.rs:normalize_window()".to_string(),
            priority: "high".to_string(),
            implementation_hint: r#"
// Add after line 228 in normalize_window():
// Reject unicode escape sequences
if normalized.contains("\\u") || normalized.contains("u0") {
    let unicode_escape = regex::Regex::new(r"u[0-9a-fA-F]{4,6}").unwrap();
    if unicode_escape.is_match(&normalized) {
        return None;
    }
}
"#.to_string(),
        },
        SuggestedFix {
            description: "Strengthen outer punctuation rejection".to_string(),
            location: "src/layers/builder.rs:should_reject_fragment()".to_string(),
            priority: "high".to_string(),
            implementation_hint: r#"
// Modify should_reject_fragment() to check for outer punctuation:
fn should_reject_fragment(stats: &WindowStats, config: &UnitBuilderConfig) -> bool {
    if stats.full_token_boundary_hits > 0 {
        return false;
    }
    
    // NEW: Reject if has outer punctuation
    let content = &stats.content;
    let has_leading_punct = content.starts_with(|c: char| !c.is_alphanumeric() && c != ' ');
    let has_trailing_punct = content.ends_with(|c: char| !c.is_alphanumeric() && c != ' ');
    
    if (has_leading_punct || has_trailing_punct) && !content.contains(' ') {
        return true;
    }
    
    // ... rest of existing logic
}
"#.to_string(),
        },
        SuggestedFix {
            description: "Add URL fragment detection".to_string(),
            location: "src/layers/builder.rs:normalize_window()".to_string(),
            priority: "medium".to_string(),
            implementation_hint: r#"
// Add URL/path fragment detection in normalize_window():
// Reject URL-like fragments
if normalized.contains('/') || (normalized.contains(':') && normalized.len() < 20) {
    return None;
}

// Reject file extension fragments
if normalized.matches('.').count() > 0 && normalized.len() < 15 {
    let ext_pattern = regex::Regex::new(r"\.(png|jpg|pdf|txt|gz|tar|zip|html?|css|js)$").unwrap();
    if ext_pattern.is_match(&normalized) {
        return None;
    }
}
"#.to_string(),
        },
        SuggestedFix {
            description: "Increase default minimum window size".to_string(),
            location: "config/config.yaml:layer_2_unit_builder".to_string(),
            priority: "medium".to_string(),
            implementation_hint: r#"
# Change in config.yaml:
layer_2_unit_builder:
  rolling_hash_window_sizes: [3, 4, 5, 6, 7, 8]  # Start from 3 instead of 2
  min_fragment_length: 5  # Increase from 4
"#.to_string(),
        },
    ];
    
    Ok(PatternAnalysis {
        patterns,
        suggested_fixes,
    })
}

fn main() {
    // Library binary - no-op main for compilation
    println!("pollution_dev_lib is a library, not a runnable binary");
}
