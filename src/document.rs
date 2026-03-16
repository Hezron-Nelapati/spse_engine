use crate::classification::input;
use crate::config::DocumentIngestionConfig;
use crate::types::{IntentKind, MetadataSummary, RetrievedDocument};
use chrono::Utc;
use lopdf::Document as PdfDocument;
use quick_xml::escape::unescape;
use quick_xml::events::Event;
use quick_xml::Reader;
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::io::{Cursor, Read, Seek};
use std::path::{Path, PathBuf};
use zip::ZipArchive;

#[derive(Debug, Clone)]
pub struct InlineDocumentRequest {
    pub prompt: String,
    pub paths: Vec<PathBuf>,
}

#[derive(Debug, Clone, Default)]
pub struct DocumentQueryState {
    pub previous_prompt: Option<String>,
    pub carry_terms: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct DocumentAnswer {
    pub text: String,
    pub confidence: f32,
    pub supporting_sources: Vec<String>,
    pub supporting_passages: Vec<String>,
    pub carry_terms: Vec<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ResponseMode {
    Summary,
    Focused,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum QuestionFocus {
    Person,
    Time,
    Place,
    Quantity,
    Definition,
    Unknown,
}

#[derive(Debug, Clone)]
struct QueryProfile {
    effective_terms: Vec<String>,
    trigrams: BTreeSet<String>,
    follow_up_hint: bool,
    broad_query: bool,
    intent: IntentKind,
    question_focus: QuestionFocus,
}

#[derive(Debug, Clone)]
struct DocumentChunk {
    source_url: String,
    heading: String,
    text: String,
    sentences: Vec<String>,
    terms: Vec<String>,
    term_set: BTreeSet<String>,
    heading_terms: BTreeSet<String>,
    trigrams: BTreeSet<String>,
    salience: f32,
}

#[derive(Debug, Clone)]
struct ScoredChunk {
    chunk: DocumentChunk,
    score: f32,
    lexical_overlap: f32,
    coverage: f32,
    heading_overlap: f32,
    focus_overlap: f32,
    trigram_overlap: f32,
}

#[derive(Debug, Clone)]
struct StructuredTable {
    source_url: String,
    section_title: Option<String>,
    headers: Vec<String>,
    records: Vec<StructuredRecord>,
    kind: StructuredTableKind,
    quality: f32,
}

#[derive(Debug, Clone)]
struct StructuredRecord {
    values: Vec<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum StructuredTableKind {
    Numbered,
    Matrix,
    KeyValue,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DocumentFormat {
    Docx,
    Pdf,
    Text,
}

#[derive(Debug, Clone)]
struct RecordSelection {
    index: usize,
    matched_columns: Vec<usize>,
    score: f32,
    matched_by_number: bool,
}

#[derive(Debug, Clone)]
struct DocumentSection {
    source_url: String,
    title: String,
    body: String,
    path: Vec<String>,
    order: usize,
}

pub fn parse_inline_document_request(input: &str) -> InlineDocumentRequest {
    let mut paths = BTreeSet::new();
    let mut stripped = input.to_string();

    for quoted in quoted_segments(input) {
        let path = PathBuf::from(&quoted);
        if is_supported_document_path(&path) {
            paths.insert(path);
            stripped = stripped.replace(&format!("\"{quoted}\""), " ");
            stripped = stripped.replace(&format!("'{quoted}'"), " ");
        }
    }

    let mut prompt_tokens = Vec::new();
    for token in stripped.split_whitespace() {
        let candidate =
            token.trim_matches(|ch: char| ch == '"' || ch == '\'' || ch == ',' || ch == ';');
        let path = PathBuf::from(candidate);
        if is_supported_document_path(&path) {
            paths.insert(path);
        } else {
            prompt_tokens.push(token);
        }
    }

    InlineDocumentRequest {
        prompt: prompt_tokens.join(" ").trim().to_string(),
        paths: paths.into_iter().collect(),
    }
}

pub fn load_documents_from_paths(paths: &[PathBuf]) -> Result<Vec<RetrievedDocument>, String> {
    load_documents_from_paths_with_config(paths, &DocumentIngestionConfig::default())
}

pub fn load_documents_from_paths_with_config(
    paths: &[PathBuf],
    config: &DocumentIngestionConfig,
) -> Result<Vec<RetrievedDocument>, String> {
    let mut docs = Vec::new();
    let mut errors = Vec::new();

    for path in paths {
        match load_document_from_path_with_config(path, config) {
            Ok(doc) => docs.push(doc),
            Err(err) => errors.push(err),
        }
    }

    if docs.is_empty() && !errors.is_empty() {
        return Err(errors.join("; "));
    }

    Ok(docs)
}

pub fn load_document_from_path(path: &Path) -> Result<RetrievedDocument, String> {
    load_document_from_path_with_config(path, &DocumentIngestionConfig::default())
}

pub fn load_document_from_path_with_config(
    path: &Path,
    config: &DocumentIngestionConfig,
) -> Result<RetrievedDocument, String> {
    let content = load_path_content_with_config(path, config)?;
    Ok(RetrievedDocument {
        source_url: path.display().to_string(),
        title: path
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("local_document")
            .to_string(),
        raw_content: content.clone(),
        normalized_content: input::normalize_text(&content),
        retrieved_at: Utc::now(),
        trust_score: 0.98,
        cached: true,
        metadata_summary: MetadataSummary::default(),
    })
}

pub fn answer_question(
    prompt: &str,
    documents: &[RetrievedDocument],
    state: Option<&DocumentQueryState>,
    intent_hint: Option<IntentKind>,
) -> Option<DocumentAnswer> {
    answer_question_with_config(
        prompt,
        documents,
        state,
        intent_hint,
        &DocumentIngestionConfig::default(),
    )
}

pub fn answer_question_with_config(
    prompt: &str,
    documents: &[RetrievedDocument],
    state: Option<&DocumentQueryState>,
    intent_hint: Option<IntentKind>,
    config: &DocumentIngestionConfig,
) -> Option<DocumentAnswer> {
    if documents.is_empty() {
        return None;
    }

    let query = build_query_profile(prompt, state, intent_hint);
    if let Some(answer) = section_answer(prompt, documents, &query, config) {
        return Some(answer);
    }
    if let Some(answer) = structured_answer(prompt, documents, &query, config) {
        return Some(answer);
    }

    let chunks = build_chunks(documents, config.max_chunk_chars);
    if chunks.is_empty() {
        let fallback = documents
            .iter()
            .flat_map(|document| split_sentences(&document.raw_content))
            .take(3)
            .collect::<Vec<_>>();
        let text = if fallback.is_empty() {
            "I couldn't extract a useful answer from the document.".to_string()
        } else {
            finalize_answer(&fallback.join(" "))
        };
        return Some(DocumentAnswer {
            text,
            confidence: 0.2,
            supporting_sources: unique_strings(
                documents
                    .iter()
                    .map(|document| document.source_url.clone())
                    .collect(),
            ),
            supporting_passages: fallback,
            carry_terms: state
                .map(|value| value.carry_terms.clone())
                .unwrap_or_default(),
        });
    }

    let weights = query_term_weights(&chunks, &query.effective_terms);
    let mut scored = if chunks.len() > 64 {
        use rayon::prelude::*;
        chunks
            .into_par_iter()
            .map(|chunk| score_chunk(chunk, &query, &weights))
            .collect::<Vec<_>>()
    } else {
        chunks
            .into_iter()
            .map(|chunk| score_chunk(chunk, &query, &weights))
            .collect::<Vec<_>>()
    };
    scored.sort_by(|lhs, rhs| rhs.score.total_cmp(&lhs.score));

    let mode = select_response_mode(&query, &scored);
    let selected = select_chunks(&scored, mode, config.max_selected_passages);
    let passages = if matches!(mode, ResponseMode::Focused) {
        let specialized = specialized_focus_passages(&scored, &query, &weights);
        if specialized.is_empty() {
            supporting_passages(&selected, &query, mode, &weights)
        } else {
            specialized
        }
    } else {
        supporting_passages(&selected, &query, mode, &weights)
    };
    let answer = match mode {
        ResponseMode::Summary => summarize_passages(&passages),
        ResponseMode::Focused => answer_from_passages(&passages),
    };
    let confidence = estimate_confidence(&selected, &query, mode);

    Some(DocumentAnswer {
        text: answer,
        confidence,
        supporting_sources: unique_strings(
            selected
                .iter()
                .map(|entry| entry.chunk.source_url.clone())
                .collect(),
        ),
        supporting_passages: passages.clone(),
        carry_terms: carry_terms(&query, &selected, config.max_carry_terms),
    })
}

pub fn load_path_content(path: &Path) -> Result<String, String> {
    load_path_content_with_config(path, &DocumentIngestionConfig::default())
}

pub fn load_path_content_with_config(
    path: &Path,
    config: &DocumentIngestionConfig,
) -> Result<String, String> {
    if !path.exists() {
        return Err(format!("document path not found: {}", path.display()));
    }
    validate_file_size(path, config)?;
    let extension = path
        .extension()
        .and_then(|ext| ext.to_str())
        .unwrap_or_default()
        .to_lowercase();

    match extension.as_str() {
        "docx" => extract_docx_text(path, config),
        "pdf" => extract_pdf_text(path, config),
        "txt" | "md" | "text" => fs::read_to_string(path)
            .map(|text| clean_document_text(&text))
            .map_err(|err| format!("failed to read text file {}: {err}", path.display())),
        _ => Err(format!(
            "unsupported document type for {}. Supported: .docx, .pdf, .txt, .md",
            path.display()
        )),
    }
}

pub fn is_supported_document_path(path: &Path) -> bool {
    if !path.exists() {
        return false;
    }
    matches!(
        path.extension()
            .and_then(|ext| ext.to_str())
            .unwrap_or_default()
            .to_lowercase()
            .as_str(),
        "docx" | "pdf" | "txt" | "md" | "text"
    )
}

pub fn validate_document_mime(path: &Path, mime: &str) -> Result<(), String> {
    let extension = path
        .extension()
        .and_then(|ext| ext.to_str())
        .unwrap_or_default()
        .to_lowercase();
    let format = document_format_from_mime(mime)
        .ok_or_else(|| format!("unsupported mime type for {}: {mime}", path.display()))?;
    let extension_format = document_format_from_extension(&extension)
        .ok_or_else(|| format!("unsupported document extension for {}", path.display()))?;
    if format != extension_format {
        return Err(format!(
            "mime type {mime} conflicts with extension .{} for {}",
            extension,
            path.display()
        ));
    }
    Ok(())
}

pub fn load_document_bytes(
    bytes: &[u8],
    mime: Option<&str>,
    config: &DocumentIngestionConfig,
) -> Result<String, String> {
    validate_inline_size(bytes, config)?;
    let format = if let Some(mime) = mime {
        document_format_from_mime(mime)
            .ok_or_else(|| format!("unsupported inline document mime type: {mime}"))?
    } else if bytes.starts_with(b"%PDF-") {
        DocumentFormat::Pdf
    } else if bytes.starts_with(b"PK\x03\x04") {
        DocumentFormat::Docx
    } else {
        DocumentFormat::Text
    };

    match format {
        DocumentFormat::Docx => extract_docx_text_from_reader(
            Cursor::new(bytes),
            "inline_docx",
            config.max_docx_xml_chars,
        ),
        DocumentFormat::Pdf => {
            extract_pdf_text_from_bytes(bytes, "inline_pdf", config.max_pdf_pages)
        }
        DocumentFormat::Text => Ok(clean_document_text(&String::from_utf8_lossy(bytes))),
    }
}

fn extract_docx_text(path: &Path, config: &DocumentIngestionConfig) -> Result<String, String> {
    let file = fs::File::open(path)
        .map_err(|err| format!("failed to open docx {}: {err}", path.display()))?;
    extract_docx_text_from_reader(file, &path.display().to_string(), config.max_docx_xml_chars)
}

fn extract_docx_text_from_reader<R: Read + Seek>(
    reader: R,
    source_label: &str,
    max_xml_chars: usize,
) -> Result<String, String> {
    let mut archive = ZipArchive::new(reader)
        .map_err(|err| format!("failed to read docx archive {source_label}: {err}"))?;
    let mut document_bytes = Vec::new();
    {
        let mut xml_entry = archive
            .by_name("word/document.xml")
            .map_err(|err| format!("missing word/document.xml in {source_label}: {err}"))?;
        let mut bounded = xml_entry.by_ref().take((max_xml_chars + 1) as u64);
        bounded
            .read_to_end(&mut document_bytes)
            .map_err(|err| format!("failed to read document.xml in {source_label}: {err}"))?;
    }
    if document_bytes.len() > max_xml_chars {
        return Err(format!(
            "docx XML exceeds configured limit for {source_label}: {} chars",
            max_xml_chars
        ));
    }
    let document_xml = String::from_utf8(document_bytes)
        .map_err(|err| format!("invalid UTF-8 in document.xml for {source_label}: {err}"))?;
    let mut reader = Reader::from_str(&document_xml);
    reader.trim_text(true);
    let mut buffer = Vec::new();
    let mut output = String::new();

    loop {
        match reader.read_event_into(&mut buffer) {
            Ok(Event::Text(text)) => {
                let decoded = text
                    .unescape()
                    .ok()
                    .and_then(|value| unescape(&value).ok().map(|v| v.into_owned()))
                    .unwrap_or_else(|| String::from_utf8_lossy(text.as_ref()).to_string());
                if !decoded.is_empty() {
                    output.push_str(&decoded);
                    output.push(' ');
                }
            }
            Ok(Event::Empty(tag)) if tag.name().as_ref() == b"w:tab" => output.push('\t'),
            Ok(Event::End(end)) if end.name().as_ref() == b"w:p" => output.push('\n'),
            Ok(Event::Eof) => break,
            Err(err) => {
                return Err(format!("failed to parse docx XML {source_label}: {err}"));
            }
            _ => {}
        }
        buffer.clear();
    }

    Ok(clean_document_text(&output))
}

fn extract_pdf_text(path: &Path, config: &DocumentIngestionConfig) -> Result<String, String> {
    let bytes =
        fs::read(path).map_err(|err| format!("failed to read pdf {}: {err}", path.display()))?;
    extract_pdf_text_from_bytes(&bytes, &path.display().to_string(), config.max_pdf_pages)
}

fn extract_pdf_text_from_bytes(
    bytes: &[u8],
    source_label: &str,
    max_pdf_pages: usize,
) -> Result<String, String> {
    let document = PdfDocument::load_mem(bytes)
        .map_err(|err| format!("failed to open pdf {source_label}: {err}"))?;
    let page_numbers = document
        .get_pages()
        .keys()
        .copied()
        .take(max_pdf_pages)
        .collect::<Vec<_>>();
    if page_numbers.is_empty() {
        return Ok(String::new());
    }
    let text = document
        .extract_text(&page_numbers)
        .map_err(|err| format!("failed to extract pdf text {source_label}: {err}"))?;
    let repaired = text.replace("-\n", "");
    Ok(clean_document_text(&repaired))
}

fn validate_file_size(path: &Path, config: &DocumentIngestionConfig) -> Result<(), String> {
    let metadata =
        fs::metadata(path).map_err(|err| format!("failed to stat {}: {err}", path.display()))?;
    let max_bytes = (config.max_file_size_mb.max(0.0) * 1024.0 * 1024.0) as u64;
    if max_bytes > 0 && metadata.len() > max_bytes {
        return Err(format!(
            "document exceeds configured size limit for {}: {} bytes > {} bytes",
            path.display(),
            metadata.len(),
            max_bytes
        ));
    }
    Ok(())
}

fn validate_inline_size(bytes: &[u8], config: &DocumentIngestionConfig) -> Result<(), String> {
    let max_bytes = (config.max_file_size_mb.max(0.0) * 1024.0 * 1024.0) as usize;
    if max_bytes > 0 && bytes.len() > max_bytes {
        return Err(format!(
            "inline document exceeds configured size limit: {} bytes > {} bytes",
            bytes.len(),
            max_bytes
        ));
    }
    Ok(())
}

fn document_format_from_extension(extension: &str) -> Option<DocumentFormat> {
    match extension {
        "docx" => Some(DocumentFormat::Docx),
        "pdf" => Some(DocumentFormat::Pdf),
        "txt" | "md" | "text" => Some(DocumentFormat::Text),
        _ => None,
    }
}

fn document_format_from_mime(mime: &str) -> Option<DocumentFormat> {
    match mime.trim().to_ascii_lowercase().as_str() {
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document" => {
            Some(DocumentFormat::Docx)
        }
        "application/pdf" => Some(DocumentFormat::Pdf),
        "text/plain" | "text/markdown" | "text/x-markdown" => Some(DocumentFormat::Text),
        _ => None,
    }
}

fn build_query_profile(
    prompt: &str,
    state: Option<&DocumentQueryState>,
    intent_hint: Option<IntentKind>,
) -> QueryProfile {
    let normalized = input::normalize_text(prompt);
    let prompt_terms = informative_terms(&normalized);
    let follow_up_hint = follow_up_hint(&normalized, &prompt_terms);
    let mut effective_terms = prompt_terms.clone();

    if (follow_up_hint || effective_terms.len() < 2) && state.is_some() {
        for term in &state.expect("state checked").carry_terms {
            if !effective_terms.contains(term) {
                effective_terms.push(term.clone());
            }
        }
    }

    if effective_terms.is_empty() {
        if let Some(state) = state {
            if let Some(previous_prompt) = &state.previous_prompt {
                for term in informative_terms(previous_prompt) {
                    if !effective_terms.contains(&term) {
                        effective_terms.push(term);
                    }
                }
            }
        }
    }

    let intent = intent_hint.unwrap_or_else(|| infer_query_intent(prompt, &normalized));
    let broad_query = matches!(
        intent,
        IntentKind::Summarize
            | IntentKind::Compare
            | IntentKind::Analyze
            | IntentKind::Plan
            | IntentKind::Critique
            | IntentKind::Recommend
            | IntentKind::Brainstorm
    ) || effective_terms.len() <= 2
        || normalized.trim().len() <= 18;

    QueryProfile {
        trigrams: char_trigrams(&normalized),
        broad_query,
        effective_terms,
        follow_up_hint,
        intent,
        question_focus: infer_question_focus(prompt, &normalized, intent),
    }
}

fn structured_answer(
    prompt: &str,
    documents: &[RetrievedDocument],
    query: &QueryProfile,
    config: &DocumentIngestionConfig,
) -> Option<DocumentAnswer> {
    let tables = extract_structured_tables(documents);
    let mut best: Option<(f32, String, String)> = None;

    for table in tables {
        if let Some((score, answer)) = match_structured_table(prompt, query, &table) {
            match &best {
                Some((best_score, _, _)) if *best_score >= score => {}
                _ => best = Some((score, answer, table.source_url.clone())),
            }
        }
    }

    best.map(|(score, answer, source)| DocumentAnswer {
        text: finalize_answer(&answer),
        confidence: (0.68 + (score * 0.24)).clamp(0.55, 0.96),
        supporting_sources: vec![source],
        supporting_passages: vec![answer.clone()],
        carry_terms: carry_terms_from_text(query, &answer, config.max_carry_terms),
    })
}

fn section_answer(
    prompt: &str,
    documents: &[RetrievedDocument],
    query: &QueryProfile,
    config: &DocumentIngestionConfig,
) -> Option<DocumentAnswer> {
    let sections = extract_sections(documents);
    let scoped_sections = scoped_sections(prompt, query, &sections)?;
    let scoped_documents = scoped_sections
        .iter()
        .map(section_to_document)
        .collect::<Vec<_>>();
    let chunks = build_chunks(&scoped_documents, config.max_chunk_chars);
    if chunks.is_empty() {
        return None;
    }

    let weights = query_term_weights(&chunks, &query.effective_terms);
    let mut scored = chunks
        .into_iter()
        .map(|chunk| score_chunk(chunk, query, &weights))
        .collect::<Vec<_>>();
    scored.sort_by(|lhs, rhs| rhs.score.total_cmp(&lhs.score));
    let mode = select_response_mode(query, &scored);
    let selected = select_chunks(&scored, mode, config.max_selected_passages);
    let passages = if matches!(mode, ResponseMode::Focused) {
        let specialized = specialized_focus_passages(&scored, query, &weights);
        if specialized.is_empty() {
            supporting_passages(&selected, query, mode, &weights)
        } else {
            specialized
        }
    } else {
        supporting_passages(&selected, query, mode, &weights)
    };
    let text = match mode {
        ResponseMode::Summary => summarize_passages(&passages),
        ResponseMode::Focused => answer_from_passages(&passages),
    };
    let best_scope_score = scoped_sections
        .iter()
        .map(|section| section_relevance(prompt, query, section))
        .fold(0.0, f32::max);

    Some(DocumentAnswer {
        text,
        confidence: (estimate_confidence(&selected, query, mode) + (0.18 * best_scope_score))
            .clamp(0.35, 0.96),
        supporting_sources: unique_strings(
            selected
                .iter()
                .map(|entry| entry.chunk.source_url.clone())
                .collect(),
        ),
        supporting_passages: passages.clone(),
        carry_terms: carry_terms(query, &selected, config.max_carry_terms),
    })
}

fn extract_sections(documents: &[RetrievedDocument]) -> Vec<DocumentSection> {
    let mut sections = Vec::new();
    let mut order = 0usize;

    for document in documents {
        let blocks = structural_blocks(&document.raw_content);
        let mut current_title = document.title.clone();
        let mut current_path = vec![document.title.clone()];
        let mut heading_stack = Vec::new();
        let mut body_parts = Vec::new();

        for block in blocks {
            if is_heading(&block) {
                if !body_parts.is_empty() {
                    sections.push(DocumentSection {
                        source_url: document.source_url.clone(),
                        title: current_title.clone(),
                        body: body_parts.join(" "),
                        path: current_path.clone(),
                        order,
                    });
                    order += 1;
                    body_parts.clear();
                }
                let depth = heading_depth(&block);
                if depth <= 1 {
                    heading_stack.clear();
                } else if heading_stack.len() >= depth {
                    heading_stack.truncate(depth - 1);
                }
                heading_stack.push(block.clone());
                current_title = block;
                current_path = heading_stack.clone();
            } else {
                body_parts.push(block);
            }
        }

        if !body_parts.is_empty() {
            sections.push(DocumentSection {
                source_url: document.source_url.clone(),
                title: current_title,
                body: body_parts.join(" "),
                path: current_path,
                order,
            });
            order += 1;
        }
    }

    for table in extract_structured_tables(documents) {
        if !matches!(table.kind, StructuredTableKind::KeyValue) {
            continue;
        }
        for record in table.records {
            if record.values.len() < 2 {
                continue;
            }
            let title = clean_inline_whitespace(&record.values[0]);
            let body = clean_inline_whitespace(&record.values[1]);
            if title.is_empty() || body.len() < 16 {
                continue;
            }
            sections.push(DocumentSection {
                source_url: table.source_url.clone(),
                title: title.clone(),
                body,
                path: table
                    .section_title
                    .clone()
                    .map(|section_title| vec![section_title, title.clone()])
                    .unwrap_or_else(|| vec![title.clone()]),
                order,
            });
            order += 1;
        }
    }

    sections
}

fn scoped_sections(
    prompt: &str,
    query: &QueryProfile,
    sections: &[DocumentSection],
) -> Option<Vec<DocumentSection>> {
    if sections.is_empty() {
        return None;
    }

    let mut ranked = sections
        .iter()
        .cloned()
        .map(|section| {
            let relevance = section_relevance(prompt, query, &section);
            (relevance, section)
        })
        .collect::<Vec<_>>();
    ranked.sort_by(|lhs, rhs| rhs.0.total_cmp(&lhs.0));

    let (best_score, seed) = ranked.first()?.clone();
    let exact_seed_focus = exact_section_focus(prompt, query, &seed);
    let minimum_scope_score = if exact_seed_focus {
        0.30
    } else if query.broad_query {
        0.22
    } else {
        0.28
    };
    if best_score < minimum_scope_score {
        return None;
    }

    let mut selected = Vec::new();
    let limit = if exact_seed_focus {
        2
    } else if query.broad_query {
        5
    } else {
        3
    };
    for (score, section) in ranked.into_iter().take(18) {
        let relation = section_relation(&seed, &section);
        let scoped_score = (0.72 * score) + (0.28 * relation);
        let qualifies = if exact_seed_focus {
            section.order == seed.order
                || score >= best_score * 0.82
                || (relation >= 0.72 && score >= best_score * 0.70)
        } else {
            section.order == seed.order
                || scoped_score >= if query.broad_query { 0.26 } else { 0.34 }
                || (relation >= 0.62 && score >= best_score * 0.55)
        };
        if qualifies {
            selected.push(section);
        }
        if selected.len() >= limit {
            break;
        }
    }

    if selected.is_empty() {
        selected.push(seed);
    }

    Some(selected)
}

fn section_relevance(prompt: &str, query: &QueryProfile, section: &DocumentSection) -> f32 {
    let path_focus = section
        .path
        .iter()
        .map(|part| heading_focus_score(prompt, &query.effective_terms, part))
        .fold(0.0, f32::max)
        .max(heading_focus_score(
            prompt,
            &query.effective_terms,
            &section.title,
        ));
    let body_terms = informative_terms(&section.body)
        .into_iter()
        .collect::<BTreeSet<_>>();
    let body_overlap = overlap_ratio(&query.effective_terms, &body_terms);
    let density = sentence_shape_score(&section.body);
    let scope_bonus = if section.path.len() > 1 { 0.06 } else { 0.0 };
    let exact_bonus = if exact_section_focus(prompt, query, section) {
        0.28
    } else {
        0.0
    };

    if query.effective_terms.is_empty() {
        (0.55 * path_focus + 0.25 * density + scope_bonus + exact_bonus).clamp(0.0, 1.0)
    } else {
        (0.46 * path_focus + 0.28 * body_overlap + 0.12 * density + scope_bonus + exact_bonus)
            .clamp(0.0, 1.0)
    }
}

fn exact_section_focus(prompt: &str, query: &QueryProfile, section: &DocumentSection) -> bool {
    let prompt_norm = input::normalize_text(prompt);
    let prompt_terms = query
        .effective_terms
        .iter()
        .cloned()
        .collect::<BTreeSet<_>>();
    let title_terms = informative_terms(&section.title)
        .into_iter()
        .collect::<BTreeSet<_>>();
    let path_heading_focus = section
        .path
        .iter()
        .map(|part| heading_focus_score(prompt, &query.effective_terms, part))
        .fold(0.0, f32::max);
    let numeric_match = first_number(&prompt_norm).is_some()
        && section
            .path
            .iter()
            .any(|part| first_number(part) == first_number(&prompt_norm))
        || first_number(&section.title) == first_number(&prompt_norm);

    (!prompt_terms.is_empty()
        && prompt_terms.len() == title_terms.len()
        && prompt_terms.iter().all(|term| title_terms.contains(term)))
        || path_heading_focus >= 0.84
        || numeric_match
}

fn section_relation(seed: &DocumentSection, candidate: &DocumentSection) -> f32 {
    let shared_prefix = shared_path_ratio(&seed.path, &candidate.path);
    let order_gap = seed.order.abs_diff(candidate.order);
    let proximity = (1.0 - (order_gap.min(6) as f32 / 6.0)).clamp(0.0, 1.0);
    let same_source = if seed.source_url == candidate.source_url {
        1.0
    } else {
        0.0
    };

    (0.5 * shared_prefix + 0.3 * proximity + 0.2 * same_source).clamp(0.0, 1.0)
}

fn shared_path_ratio(lhs: &[String], rhs: &[String]) -> f32 {
    if lhs.is_empty() || rhs.is_empty() {
        return 0.0;
    }

    let shared = lhs
        .iter()
        .zip(rhs.iter())
        .take_while(|(left, right)| left.eq_ignore_ascii_case(right))
        .count();
    shared as f32 / lhs.len().max(rhs.len()) as f32
}

fn section_to_document(section: &DocumentSection) -> RetrievedDocument {
    let title = if section.path.is_empty() {
        section.title.clone()
    } else {
        section.path.join(" > ")
    };
    RetrievedDocument {
        source_url: section.source_url.clone(),
        title,
        raw_content: section.body.clone(),
        normalized_content: input::normalize_text(&section.body),
        retrieved_at: Utc::now(),
        trust_score: 0.98,
        cached: true,
        metadata_summary: MetadataSummary::default(),
    }
}

fn extract_structured_tables(documents: &[RetrievedDocument]) -> Vec<StructuredTable> {
    let mut tables = Vec::new();
    let mut seen = BTreeSet::new();

    for document in documents {
        let lines = document
            .raw_content
            .lines()
            .map(clean_inline_whitespace)
            .filter(|line| !line.is_empty())
            .collect::<Vec<_>>();
        for table in parse_document_tables(&lines, &document.source_url) {
            let signature = format!(
                "{}::{:?}::{:?}::{:?}",
                table.source_url,
                table.section_title,
                table.headers,
                table.records.first().map(|record| record.values.clone())
            );
            if seen.insert(signature) {
                tables.push(table);
            }
        }
    }

    tables
}

fn parse_document_tables(lines: &[String], source_url: &str) -> Vec<StructuredTable> {
    let mut tables = Vec::new();
    let mut index = 0usize;

    while index < lines.len() {
        if let Some((table, _next_index)) = parse_numbered_table(lines, index, source_url) {
            tables.push(table);
        }

        if let Some((table, _next_index)) = parse_matrix_table(lines, index, source_url) {
            tables.push(table);
        }

        if let Some((table, _next_index)) = parse_key_value_table(lines, index, source_url) {
            tables.push(table);
        }

        index += 1;
    }

    tables
}

fn parse_numbered_table(
    lines: &[String],
    start: usize,
    source_url: &str,
) -> Option<(StructuredTable, usize)> {
    let mut first_numeric = start;
    while first_numeric < lines.len()
        && !is_numeric_marker(&lines[first_numeric])
        && first_numeric - start < 6
        && looks_like_table_header(&lines[first_numeric])
    {
        first_numeric += 1;
    }

    let header_count = first_numeric.checked_sub(start)?;
    if header_count < 2 || first_numeric >= lines.len() || !is_numeric_marker(&lines[first_numeric])
    {
        return None;
    }

    let headers = lines[start..first_numeric].to_vec();
    if !headers
        .iter()
        .all(|header| is_plausible_header_cell(header, true))
        || headers
            .iter()
            .skip(1)
            .filter(|header| informative_terms(header).is_empty())
            .count()
            > 0
    {
        return None;
    }

    let mut records = Vec::new();
    let mut index = first_numeric;

    while index + header_count <= lines.len() {
        if !is_numeric_marker(&lines[index]) {
            break;
        }
        let row = lines[index..index + header_count].to_vec();
        if !is_plausible_row(&headers, &row, StructuredTableKind::Numbered) {
            break;
        }
        records.push(StructuredRecord { values: row });
        index += header_count;
    }

    if records.len() < 2 {
        return None;
    }

    let quality = score_table_quality(StructuredTableKind::Numbered, &records);

    Some((
        StructuredTable {
            source_url: source_url.to_string(),
            section_title: preceding_section_title(lines, start),
            headers,
            records,
            kind: StructuredTableKind::Numbered,
            quality,
        },
        index,
    ))
}

fn parse_matrix_table(
    lines: &[String],
    start: usize,
    source_url: &str,
) -> Option<(StructuredTable, usize)> {
    let mut best: Option<(StructuredTable, usize)> = None;

    for header_count in (3..=5).rev() {
        if start + header_count + header_count > lines.len() {
            continue;
        }

        let headers = lines[start..start + header_count].to_vec();
        if !headers
            .iter()
            .all(|header| is_plausible_header_cell(header, false))
            || headers
                .iter()
                .any(|header| is_numeric_marker(header) || header == "#")
            || headers
                .iter()
                .filter(|header| is_generic_table_header(header))
                .count()
                < headers.len().saturating_sub(1)
        {
            continue;
        }

        let mut records = Vec::new();
        let mut index = start + header_count;

        while index + header_count <= lines.len() {
            let row = lines[index..index + header_count].to_vec();
            if !is_plausible_row(&headers, &row, StructuredTableKind::Matrix) {
                break;
            }
            records.push(StructuredRecord { values: row });
            index += header_count;
        }

        if records.len() < 2 {
            continue;
        }

        let quality = score_table_quality(StructuredTableKind::Matrix, &records);
        let table = StructuredTable {
            source_url: source_url.to_string(),
            section_title: preceding_section_title(lines, start),
            headers,
            records,
            kind: StructuredTableKind::Matrix,
            quality,
        };

        match &best {
            Some((current, _)) if current.quality >= table.quality => {}
            _ => best = Some((table, index)),
        }
    }

    best
}

fn parse_key_value_table(
    lines: &[String],
    start: usize,
    source_url: &str,
) -> Option<(StructuredTable, usize)> {
    if start + 3 >= lines.len() {
        return None;
    }
    if !is_plausible_header_cell(&lines[start], false)
        || !is_plausible_header_cell(&lines[start + 1], false)
        || lines[start] == "#"
        || lines[start + 1] == "#"
        || lines[start].chars().any(|ch| ch.is_ascii_digit())
        || lines[start + 1].chars().any(|ch| ch.is_ascii_digit())
        || informative_terms(&lines[start]).is_empty()
        || informative_terms(&lines[start + 1]).is_empty()
        || !is_generic_table_header(&lines[start])
        || !is_generic_table_header(&lines[start + 1])
    {
        return None;
    }
    if is_numeric_marker(&lines[start + 2]) {
        return None;
    }
    if start + 4 < lines.len()
        && is_plausible_header_cell(&lines[start + 2], false)
        && is_plausible_header_cell(&lines[start + 3], false)
        && !is_numeric_marker(&lines[start + 4])
    {
        return None;
    }

    let headers = vec![lines[start].clone(), lines[start + 1].clone()];
    let mut records = Vec::new();
    let mut index = start + 2;

    while index + 1 < lines.len() {
        if is_numeric_marker(&lines[index]) {
            break;
        }
        let key = lines[index].clone();
        let value = lines[index + 1].clone();
        if !is_plausible_row(
            &headers,
            &[key.clone(), value.clone()],
            StructuredTableKind::KeyValue,
        ) {
            break;
        }
        records.push(StructuredRecord {
            values: vec![key, value],
        });
        index += 2;
    }

    if records.len() < 2 {
        return None;
    }

    let quality = score_table_quality(StructuredTableKind::KeyValue, &records);

    Some((
        StructuredTable {
            source_url: source_url.to_string(),
            section_title: preceding_section_title(lines, start),
            headers,
            records,
            kind: StructuredTableKind::KeyValue,
            quality,
        },
        index,
    ))
}

fn match_structured_table(
    prompt: &str,
    query: &QueryProfile,
    table: &StructuredTable,
) -> Option<(f32, String)> {
    let prompt_terms = informative_terms(prompt);
    let selection = select_record(prompt, &prompt_terms, table)?;
    let record = table.records.get(selection.index)?;
    let column_index = select_column(prompt, &prompt_terms, table, record, &selection);
    let answer = render_record_answer(table, record, column_index)?;
    if is_degenerate_structured_answer(table, record, &answer) {
        return None;
    }

    let record_terms = informative_terms(&record.values.join(" "));
    let coverage = if query.effective_terms.is_empty() {
        0.4
    } else {
        let matched = query
            .effective_terms
            .iter()
            .filter(|term| record_terms.contains(*term))
            .count() as f32;
        matched / query.effective_terms.len() as f32
    };
    let header_alignment = table_alignment_score(prompt, &prompt_terms, table);
    let selection_alignment = selection.score.clamp(0.0, 1.0);
    let first_data_column = if first_column_is_index(table, record) {
        1
    } else {
        0
    };
    let entity_lookup = starts_with_entity_question(prompt)
        && selection
            .matched_columns
            .iter()
            .any(|column| *column != first_data_column);
    let key_value_lookup = matches!(table.kind, StructuredTableKind::KeyValue)
        && selection.score >= 0.6
        && selection
            .matched_columns
            .iter()
            .any(|column| *column == first_data_column || *column == 0);
    if !selection.matched_by_number
        && header_alignment < 0.28
        && !entity_lookup
        && !key_value_lookup
    {
        return None;
    }
    let score = (0.28 * table.quality
        + 0.28 * coverage
        + 0.24 * header_alignment
        + 0.20 * selection_alignment
        + if entity_lookup || key_value_lookup {
            0.08
        } else {
            0.0
        })
    .clamp(0.0, 1.0);
    if score < 0.42 {
        return None;
    }
    Some((score, answer))
}

fn select_record(
    prompt: &str,
    prompt_terms: &[String],
    table: &StructuredTable,
) -> Option<RecordSelection> {
    if let Some(position) = ordinal_position(prompt) {
        let zero_based = position.saturating_sub(1);
        if zero_based < table.records.len()
            && (matches!(table.kind, StructuredTableKind::Numbered)
                || table
                    .headers
                    .first()
                    .map(|header| header.eq_ignore_ascii_case("step"))
                    .unwrap_or(false))
        {
            return Some(RecordSelection {
                index: zero_based,
                matched_columns: vec![0],
                score: 0.96,
                matched_by_number: true,
            });
        }
    }

    if let Some(number) = first_number(prompt) {
        if let Some(index) = table.records.iter().position(|record| {
            record
                .values
                .first()
                .map(|value| value == &number)
                .unwrap_or(false)
        }) {
            return Some(RecordSelection {
                index,
                matched_columns: vec![0],
                score: 1.0,
                matched_by_number: true,
            });
        }
    }

    let mut best: Option<(f32, usize)> = None;
    for (index, record) in table.records.iter().enumerate() {
        let column_scores = record
            .values
            .iter()
            .enumerate()
            .map(|(column, value)| {
                let terms = informative_terms(value);
                let overlap = prompt_terms
                    .iter()
                    .filter(|term| terms.contains(*term))
                    .count() as f32;
                let score = if prompt_terms.is_empty() {
                    0.0
                } else {
                    overlap / prompt_terms.len() as f32
                };
                (column, score)
            })
            .collect::<Vec<_>>();

        let max_overlap = column_scores
            .iter()
            .map(|(_, score)| *score)
            .fold(0.0, f32::max);
        let coverage = coverage_ratio(
            prompt_terms,
            &record
                .values
                .iter()
                .flat_map(|value| informative_terms(value))
                .collect::<BTreeSet<_>>(),
        );
        let score = (0.65 * max_overlap + 0.35 * coverage).clamp(0.0, 1.0);
        match best {
            Some((best_score, _)) if best_score >= score => {}
            _ => best = Some((score, index)),
        }
    }

    best.and_then(|(score, index)| {
        if score < 0.24 {
            return None;
        }
        let matched_columns = table.records[index]
            .values
            .iter()
            .enumerate()
            .filter_map(|(column, value)| {
                let terms = informative_terms(value);
                let overlap = prompt_terms
                    .iter()
                    .filter(|term| terms.contains(*term))
                    .count() as f32;
                let ratio = if prompt_terms.is_empty() {
                    0.0
                } else {
                    overlap / prompt_terms.len() as f32
                };
                if ratio >= 0.20 {
                    Some(column)
                } else {
                    None
                }
            })
            .collect::<Vec<_>>();
        Some(RecordSelection {
            index,
            matched_columns,
            score,
            matched_by_number: false,
        })
    })
}

fn select_column(
    prompt: &str,
    prompt_terms: &[String],
    table: &StructuredTable,
    record: &StructuredRecord,
    selection: &RecordSelection,
) -> Option<usize> {
    let first_data_column = if first_column_is_index(table, record) {
        1
    } else {
        0
    };
    let entity_lookup = starts_with_entity_question(prompt)
        && selection
            .matched_columns
            .iter()
            .any(|column| *column != first_data_column);

    if entity_lookup {
        return Some(first_data_column);
    }

    if table.headers.len() == 2 {
        if matches!(table.kind, StructuredTableKind::KeyValue) || first_data_column == 1 {
            if selection.matched_columns.contains(&first_data_column) {
                return Some((first_data_column + 1).min(record.values.len().saturating_sub(1)));
            }
            return Some((first_data_column + 1).min(record.values.len().saturating_sub(1)));
        }
        return Some(first_data_column);
    }

    let unique_prompt_terms = prompt_terms
        .iter()
        .cloned()
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect::<Vec<_>>();
    let mut best: Option<(f32, usize)> = None;
    for (index, header) in table.headers.iter().enumerate().skip(first_data_column) {
        let header_terms = informative_terms(header);
        let header_overlap = overlap_ratio(
            &unique_prompt_terms,
            &header_terms.into_iter().collect::<BTreeSet<_>>(),
        );
        let value_terms =
            informative_terms(record.values.get(index).map(String::as_str).unwrap_or(""))
                .into_iter()
                .collect::<BTreeSet<_>>();
        let value_overlap = overlap_ratio(&unique_prompt_terms, &value_terms);
        let novelty = 1.0 - value_overlap;
        let content_score =
            sentence_shape_score(record.values.get(index).map(String::as_str).unwrap_or(""));
        let identifier_penalty = if selection.matched_columns.contains(&index) {
            0.65
        } else {
            0.0
        };
        let subject_bonus = if entity_lookup && index == first_data_column {
            0.55
        } else {
            0.0
        };
        let score =
            (0.65 * header_overlap) + (0.30 * novelty) + (0.15 * content_score) + subject_bonus
                - identifier_penalty;
        match best {
            Some((best_score, _)) if best_score >= score => {}
            _ => best = Some((score, index)),
        }
    }

    best.map(|(_, index)| index).or_else(|| {
        if record.values.len() > first_data_column {
            Some(first_data_column.max(1))
        } else {
            None
        }
    })
}

fn render_record_answer(
    table: &StructuredTable,
    record: &StructuredRecord,
    column_index: Option<usize>,
) -> Option<String> {
    let answer = match column_index {
        Some(index) if index < record.values.len() => record.values[index].clone(),
        _ if record.values.len() > 1 => record.values[1..].join(" "),
        _ => record.values.join(" "),
    };
    if is_degenerate_structured_answer(table, record, &answer) {
        None
    } else {
        Some(answer)
    }
}

fn first_column_is_index(table: &StructuredTable, record: &StructuredRecord) -> bool {
    matches!(table.kind, StructuredTableKind::Numbered)
        || table
            .headers
            .first()
            .map(|value| value == "#" || value.eq_ignore_ascii_case("step"))
            == Some(true)
        || record
            .values
            .first()
            .map(|value| is_numeric_marker(value))
            .unwrap_or(false)
}

fn table_alignment_score(prompt: &str, prompt_terms: &[String], table: &StructuredTable) -> f32 {
    let header_terms = table
        .headers
        .iter()
        .flat_map(|header| informative_terms(header))
        .collect::<BTreeSet<_>>();
    let header_overlap = overlap_ratio(prompt_terms, &header_terms);
    let section_overlap = table
        .section_title
        .as_ref()
        .map(|title| heading_focus_score(prompt, prompt_terms, title))
        .unwrap_or(0.0);
    let numeric_bias =
        if first_number(prompt).is_some() && matches!(table.kind, StructuredTableKind::Numbered) {
            0.12
        } else {
            0.0
        };
    let entity_bias = if starts_with_entity_question(prompt) {
        0.08
    } else {
        0.0
    };

    (0.45 * header_overlap + 0.55 * section_overlap + numeric_bias + entity_bias).clamp(0.0, 1.0)
}

fn is_degenerate_structured_answer(
    table: &StructuredTable,
    record: &StructuredRecord,
    answer: &str,
) -> bool {
    let normalized = clean_inline_whitespace(answer);
    if normalized.is_empty() || normalized.len() < 3 {
        return true;
    }
    if table
        .headers
        .iter()
        .any(|header| clean_inline_whitespace(header).eq_ignore_ascii_case(&normalized))
    {
        return true;
    }
    if record.values.iter().take(2).any(|value| {
        clean_inline_whitespace(value).eq_ignore_ascii_case(&normalized)
            && informative_terms(&normalized).len() <= 2
    }) && normalized.split_whitespace().count() <= 2
    {
        return true;
    }
    false
}

fn carry_terms_from_text(query: &QueryProfile, text: &str, max_carry_terms: usize) -> Vec<String> {
    let mut carry = query.effective_terms.clone();
    for term in informative_terms(text) {
        if !carry.contains(&term) {
            carry.push(term);
        }
        if carry.len() >= max_carry_terms {
            break;
        }
    }
    carry
}

fn build_chunks(documents: &[RetrievedDocument], max_chunk_chars: usize) -> Vec<DocumentChunk> {
    let mut chunks = Vec::new();

    for document in documents {
        let blocks = structural_blocks(&document.raw_content);
        let total_blocks = blocks.len().max(1);
        let mut heading = document.title.clone();
        let mut position = 0usize;

        for block in blocks {
            if is_heading(&block) {
                heading = block.clone();
                continue;
            }

            for window in sentence_windows(&block, max_chunk_chars) {
                let normalized = input::normalize_text(&window);
                let terms = informative_terms(&normalized);
                if terms.is_empty() && normalized.len() < 20 {
                    continue;
                }
                let term_set = terms.iter().cloned().collect::<BTreeSet<_>>();
                let heading_terms = informative_terms(&heading)
                    .into_iter()
                    .collect::<BTreeSet<_>>();
                let progress = position as f32 / total_blocks as f32;
                let information_density =
                    (term_set.len() as f32 / (normalized.len().max(1) as f32 / 40.0)).min(1.0);
                let heading_bonus = if heading == document.title {
                    0.08
                } else {
                    0.15
                };
                let salience = (0.3
                    + ((1.0 - progress).clamp(0.0, 1.0) * 0.35)
                    + heading_bonus
                    + (information_density * 0.2))
                    .clamp(0.0, 1.0);

                chunks.push(DocumentChunk {
                    source_url: document.source_url.clone(),
                    heading: heading.clone(),
                    text: window.clone(),
                    sentences: split_sentences(&window),
                    terms,
                    term_set,
                    heading_terms,
                    trigrams: char_trigrams(&normalized),
                    salience,
                });
                position += 1;
            }
        }
    }

    chunks
}

fn query_term_weights(chunks: &[DocumentChunk], query_terms: &[String]) -> BTreeMap<String, f32> {
    let mut df = BTreeMap::new();
    for term in query_terms {
        df.insert(term.clone(), 0usize);
    }

    for chunk in chunks {
        for term in query_terms {
            if chunk.term_set.contains(term) {
                *df.entry(term.clone()).or_default() += 1;
            }
        }
    }

    let chunk_count = chunks.len().max(1) as f32;
    df.into_iter()
        .map(|(term, count)| {
            let rarity = (1.0 + (chunk_count / (count.max(1) as f32)).ln()).clamp(1.0, 3.0);
            (term, rarity)
        })
        .collect()
}

fn score_chunk(
    chunk: DocumentChunk,
    query: &QueryProfile,
    weights: &BTreeMap<String, f32>,
) -> ScoredChunk {
    let lexical_overlap = weighted_overlap(&query.effective_terms, &chunk.term_set, weights);
    let coverage = coverage_ratio(&query.effective_terms, &chunk.term_set);
    let heading_overlap = overlap_ratio(&query.effective_terms, &chunk.heading_terms);
    let focus_overlap = heading_focus_score(
        &query.effective_terms.join(" "),
        &query.effective_terms,
        &chunk.heading,
    );
    let trigram_overlap = jaccard_score(&query.trigrams, &chunk.trigrams);

    let mut score = 0.16 * chunk.salience
        + 0.10 * heading_overlap
        + 0.30 * focus_overlap
        + 0.10 * trigram_overlap
        + 0.20 * lexical_overlap
        + 0.14 * coverage;

    if query.effective_terms.is_empty() {
        score = (0.72 * chunk.salience)
            + (0.1 * heading_overlap)
            + (0.1 * focus_overlap)
            + (0.08 * trigram_overlap)
            + (0.08 * sentence_shape_score(&chunk.text));
    } else if query.broad_query {
        score += 0.03 * chunk.salience + 0.12 * focus_overlap;
    }

    ScoredChunk {
        chunk,
        score: score.clamp(0.0, 1.0),
        lexical_overlap,
        coverage,
        heading_overlap,
        focus_overlap,
        trigram_overlap,
    }
}

fn select_response_mode(query: &QueryProfile, scored: &[ScoredChunk]) -> ResponseMode {
    let top = scored.first().map(|entry| entry.score).unwrap_or(0.0);
    let second = scored.get(1).map(|entry| entry.score).unwrap_or(0.0);
    let top_coverage = scored.first().map(|entry| entry.coverage).unwrap_or(0.0);
    let diversity = scored
        .iter()
        .take(4)
        .map(|entry| format!("{}::{}", entry.chunk.source_url, entry.chunk.heading))
        .collect::<BTreeSet<_>>()
        .len();
    let concentration = if second <= f32::EPSILON {
        2.0
    } else {
        top / second
    };

    match query.intent {
        IntentKind::Summarize
        | IntentKind::Compare
        | IntentKind::Analyze
        | IntentKind::Plan
        | IntentKind::Critique
        | IntentKind::Recommend
        | IntentKind::Brainstorm => {
            return ResponseMode::Summary;
        }
        IntentKind::Extract | IntentKind::Classify | IntentKind::Debug => {
            if top >= 0.26 {
                return ResponseMode::Focused;
            }
        }
        IntentKind::Explain => {
            if query.broad_query && diversity >= 2 {
                return ResponseMode::Summary;
            }
        }
        _ => {}
    }

    if query.effective_terms.is_empty()
        || top < 0.34
        || top_coverage < 0.24
        || (query.broad_query && diversity >= 2 && concentration < 1.2)
    {
        ResponseMode::Summary
    } else {
        ResponseMode::Focused
    }
}

fn select_chunks(
    scored: &[ScoredChunk],
    mode: ResponseMode,
    max_selected_passages: usize,
) -> Vec<ScoredChunk> {
    let limit = match mode {
        ResponseMode::Summary => max_selected_passages,
        ResponseMode::Focused => 2,
    };
    let mut selected = Vec::new();
    let best_focus = scored
        .iter()
        .map(|candidate| candidate.focus_overlap)
        .fold(0.0, f32::max);

    for candidate in scored.iter().take(12) {
        if best_focus >= 0.6 && candidate.focus_overlap + 0.15 < best_focus {
            continue;
        }
        let too_similar = selected
            .iter()
            .any(|chosen: &ScoredChunk| chunk_similarity(&chosen.chunk, &candidate.chunk) > 0.72);
        let repeated_section = selected.iter().any(|chosen| {
            chosen.chunk.source_url == candidate.chunk.source_url
                && chosen.chunk.heading == candidate.chunk.heading
        });

        if too_similar {
            continue;
        }
        if matches!(mode, ResponseMode::Summary) && repeated_section && !selected.is_empty() {
            continue;
        }

        selected.push(candidate.clone());
        if selected.len() >= limit {
            break;
        }
    }

    if selected.is_empty() {
        if let Some(first) = scored.first() {
            selected.push(first.clone());
        }
    }

    selected
}

fn supporting_passages(
    selected: &[ScoredChunk],
    query: &QueryProfile,
    mode: ResponseMode,
    weights: &BTreeMap<String, f32>,
) -> Vec<String> {
    let mut passages = Vec::new();
    for entry in selected {
        if let Some(sentence) = representative_sentence(entry, query, mode, weights) {
            push_unique(&mut passages, sentence);
        } else {
            push_unique(&mut passages, entry.chunk.text.clone());
        }
    }
    passages
}

fn specialized_focus_passages(
    scored: &[ScoredChunk],
    query: &QueryProfile,
    weights: &BTreeMap<String, f32>,
) -> Vec<String> {
    if matches!(query.question_focus, QuestionFocus::Unknown) {
        return Vec::new();
    }

    let mut ranked = Vec::new();
    for entry in scored.iter().take(12) {
        let sentences = if entry.chunk.sentences.is_empty() {
            vec![entry.chunk.text.clone()]
        } else {
            entry.chunk.sentences.clone()
        };
        for sentence in sentences {
            let normalized = input::normalize_text(&sentence);
            let terms = informative_terms(&normalized);
            let term_set = terms.iter().cloned().collect::<BTreeSet<_>>();
            let lexical = weighted_overlap(&query.effective_terms, &term_set, weights);
            let coverage = coverage_ratio(&query.effective_terms, &term_set);
            let focus = question_focus_score(query.question_focus, &sentence);
            let heading = heading_focus_score(
                &query.effective_terms.join(" "),
                &query.effective_terms,
                &entry.chunk.heading,
            );
            let score = (0.22 * lexical)
                + (0.16 * coverage)
                + (0.42 * focus)
                + (0.10 * heading)
                + (0.10 * entry.score);
            ranked.push((score, sentence));
        }
    }

    ranked.sort_by(|lhs, rhs| rhs.0.total_cmp(&lhs.0));
    let mut passages = Vec::new();
    for (_, sentence) in ranked {
        push_unique(&mut passages, sentence);
        if passages.len() >= 2 {
            break;
        }
    }
    passages
}

fn representative_sentence(
    entry: &ScoredChunk,
    query: &QueryProfile,
    mode: ResponseMode,
    weights: &BTreeMap<String, f32>,
) -> Option<String> {
    let sentences = if entry.chunk.sentences.is_empty() {
        vec![entry.chunk.text.clone()]
    } else {
        entry.chunk.sentences.clone()
    };

    let mut best: Option<(f32, String)> = None;
    for (index, sentence) in sentences.into_iter().enumerate() {
        let normalized = input::normalize_text(&sentence);
        let terms = informative_terms(&normalized);
        let term_set = terms.iter().cloned().collect::<BTreeSet<_>>();
        let lexical = weighted_overlap(&query.effective_terms, &term_set, weights);
        let coverage = coverage_ratio(&query.effective_terms, &term_set);
        let trigram = jaccard_score(&query.trigrams, &char_trigrams(&normalized));
        let brevity = sentence_shape_score(&sentence);
        let sentence_position = 1.0 - (index as f32 / entry.chunk.sentences.len().max(1) as f32);
        let focus_alignment = question_focus_score(query.question_focus, &sentence);
        let mut score = 0.1 * sentence_position + 0.2 * brevity + 0.15 * entry.chunk.salience;

        if query.effective_terms.is_empty() {
            score += 0.55 * brevity;
        } else {
            score += 0.32 * lexical + 0.18 * coverage + 0.12 * trigram;
        }

        if matches!(mode, ResponseMode::Focused) {
            score += 0.08 * entry.lexical_overlap
                + 0.06 * entry.heading_overlap
                + 0.40 * focus_alignment;
        }

        match &best {
            Some((best_score, _)) if *best_score >= score => {}
            _ => best = Some((score, sentence)),
        }
    }

    best.map(|(_, sentence)| sentence)
}

fn summarize_passages(passages: &[String]) -> String {
    if passages.is_empty() {
        return "I couldn't ground a summary in the document.".to_string();
    }
    finalize_answer(
        &passages
            .iter()
            .take(3)
            .cloned()
            .collect::<Vec<_>>()
            .join(" "),
    )
}

fn answer_from_passages(passages: &[String]) -> String {
    if passages.is_empty() {
        return "I couldn't ground a focused answer in the document.".to_string();
    }
    finalize_answer(
        &passages
            .iter()
            .take(2)
            .cloned()
            .collect::<Vec<_>>()
            .join(" "),
    )
}

fn estimate_confidence(selected: &[ScoredChunk], query: &QueryProfile, mode: ResponseMode) -> f32 {
    if selected.is_empty() {
        return 0.15;
    }

    let top = selected.first().map(|entry| entry.score).unwrap_or(0.0);
    let average = selected.iter().map(|entry| entry.score).sum::<f32>() / selected.len() as f32;
    let coverage = selected
        .iter()
        .map(|entry| entry.coverage)
        .fold(0.0, f32::max);
    let heading_alignment = selected
        .iter()
        .map(|entry| entry.heading_overlap.max(entry.focus_overlap))
        .fold(0.0, f32::max);
    let trigram_alignment = selected
        .iter()
        .map(|entry| entry.trigram_overlap)
        .fold(0.0, f32::max);
    let follow_up_bonus = if query.follow_up_hint { 0.04 } else { 0.0 };
    let mode_bonus = if matches!(mode, ResponseMode::Focused) {
        0.08
    } else {
        0.0
    };

    (0.42 * top
        + 0.18 * average
        + 0.22 * coverage
        + 0.08 * heading_alignment
        + 0.05 * trigram_alignment
        + follow_up_bonus
        + mode_bonus)
        .clamp(0.15, 0.96)
}

fn carry_terms(
    query: &QueryProfile,
    selected: &[ScoredChunk],
    max_carry_terms: usize,
) -> Vec<String> {
    let mut terms = query.effective_terms.clone();
    let mut frequencies = BTreeMap::new();

    for entry in selected {
        for term in &entry.chunk.terms {
            if term.len() <= 3 || is_stopword(term) {
                continue;
            }
            *frequencies.entry(term.clone()).or_insert(0usize) += 1;
        }
    }

    let mut ranked = frequencies.into_iter().collect::<Vec<_>>();
    ranked.sort_by(|lhs, rhs| rhs.1.cmp(&lhs.1).then_with(|| lhs.0.cmp(&rhs.0)));

    for (term, _) in ranked {
        if !terms.contains(&term) {
            terms.push(term);
        }
        if terms.len() >= max_carry_terms {
            break;
        }
    }

    terms.truncate(max_carry_terms);
    terms
}

fn structural_blocks(text: &str) -> Vec<String> {
    let mut blocks = Vec::new();
    let mut current = String::new();

    for raw_line in text.lines() {
        let line = raw_line.trim();
        if line.is_empty() {
            flush_block(&mut current, &mut blocks);
            continue;
        }

        if is_heading(line) {
            flush_block(&mut current, &mut blocks);
            blocks.push(line.to_string());
            continue;
        }

        if current.is_empty() {
            current.push_str(line);
        } else if should_merge_line(&current, line) {
            current.push(' ');
            current.push_str(line);
        } else {
            flush_block(&mut current, &mut blocks);
            current.push_str(line);
        }
    }

    flush_block(&mut current, &mut blocks);
    blocks
}

fn sentence_windows(text: &str, max_chunk_chars: usize) -> Vec<String> {
    let sentences = split_sentences(text);
    if sentences.is_empty() {
        return vec![text.trim().to_string()];
    }

    let mut windows = Vec::new();
    let mut current = String::new();

    for sentence in sentences {
        if current.is_empty() {
            current.push_str(&sentence);
            continue;
        }

        if current.len() + sentence.len() + 1 <= max_chunk_chars {
            current.push(' ');
            current.push_str(&sentence);
        } else {
            windows.push(current);
            current = sentence;
        }
    }

    if !current.trim().is_empty() {
        windows.push(current);
    }

    windows
}

fn split_sentences(text: &str) -> Vec<String> {
    let mut sentences = Vec::new();
    let mut current = String::new();

    for ch in text.chars() {
        current.push(ch);
        if matches!(ch, '.' | '!' | '?' | '\n') {
            let trimmed = current.trim();
            if !trimmed.is_empty() {
                sentences.push(trimmed.to_string());
            }
            current.clear();
        }
    }

    let trailing = current.trim();
    if !trailing.is_empty() {
        sentences.push(trailing.to_string());
    }

    sentences
        .into_iter()
        .map(|sentence| clean_inline_whitespace(&sentence))
        .filter(|sentence| !sentence.is_empty())
        .collect()
}

fn informative_terms(text: &str) -> Vec<String> {
    input::normalize_text(text)
        .split_whitespace()
        .map(|token| {
            token
                .trim_matches(|ch: char| !ch.is_alphanumeric())
                .to_lowercase()
        })
        .filter(|token| token.len() > 2)
        .filter(|token| !is_stopword(token))
        .collect()
}

fn is_stopword(token: &str) -> bool {
    matches!(
        token,
        "the"
            | "and"
            | "for"
            | "with"
            | "that"
            | "this"
            | "from"
            | "into"
            | "about"
            | "what"
            | "does"
            | "where"
            | "when"
            | "which"
            | "have"
            | "your"
            | "their"
            | "there"
            | "would"
            | "could"
            | "should"
            | "mainly"
            | "document"
            | "file"
            | "please"
            | "give"
            | "tell"
            | "show"
            | "summarize"
            | "summarise"
            | "summary"
            | "overview"
            | "explain"
            | "describe"
    )
}

fn weighted_overlap(
    query_terms: &[String],
    chunk_terms: &BTreeSet<String>,
    weights: &BTreeMap<String, f32>,
) -> f32 {
    if query_terms.is_empty() {
        return 0.0;
    }

    let mut matched = 0.0;
    let mut total = 0.0;
    for term in query_terms {
        let weight = *weights.get(term).unwrap_or(&1.0);
        total += weight;
        if chunk_terms.contains(term) {
            matched += weight;
        }
    }

    if total <= f32::EPSILON {
        0.0
    } else {
        (matched / total).clamp(0.0, 1.0)
    }
}

fn coverage_ratio(query_terms: &[String], chunk_terms: &BTreeSet<String>) -> f32 {
    if query_terms.is_empty() {
        return 0.0;
    }
    let matched = query_terms
        .iter()
        .filter(|term| chunk_terms.contains(*term))
        .count() as f32;
    (matched / query_terms.len() as f32).clamp(0.0, 1.0)
}

fn overlap_ratio(query_terms: &[String], chunk_terms: &BTreeSet<String>) -> f32 {
    coverage_ratio(query_terms, chunk_terms)
}

fn char_trigrams(text: &str) -> BTreeSet<String> {
    let compact = text
        .to_lowercase()
        .chars()
        .filter(|ch| ch.is_alphanumeric() || ch.is_whitespace())
        .collect::<String>();
    let chars = compact.chars().collect::<Vec<_>>();
    if chars.len() < 3 {
        return BTreeSet::new();
    }

    let mut trigrams = BTreeSet::new();
    for window in chars.windows(3) {
        trigrams.insert(window.iter().collect());
    }
    trigrams
}

fn jaccard_score(lhs: &BTreeSet<String>, rhs: &BTreeSet<String>) -> f32 {
    if lhs.is_empty() || rhs.is_empty() {
        return 0.0;
    }

    let intersection = lhs.iter().filter(|value| rhs.contains(*value)).count() as f32;
    let union = lhs.len() as f32 + rhs.len() as f32 - intersection;
    if union <= f32::EPSILON {
        0.0
    } else {
        (intersection / union).clamp(0.0, 1.0)
    }
}

fn chunk_similarity(lhs: &DocumentChunk, rhs: &DocumentChunk) -> f32 {
    let lexical = jaccard_score(&lhs.term_set, &rhs.term_set);
    let trigrams = jaccard_score(&lhs.trigrams, &rhs.trigrams);
    (0.6 * lexical + 0.4 * trigrams).clamp(0.0, 1.0)
}

fn follow_up_hint(normalized: &str, prompt_terms: &[String]) -> bool {
    let lowered = normalized.to_lowercase();
    prompt_terms.len() <= 3
        || lowered.starts_with("and ")
        || lowered.starts_with("also ")
        || lowered.starts_with("what about ")
        || lowered.starts_with("how about ")
        || lowered.starts_with("then ")
        || lowered.contains(" it ")
        || lowered.contains(" they ")
        || lowered.contains(" that ")
        || lowered.contains(" those ")
}

fn starts_with_entity_question(prompt: &str) -> bool {
    let lowered = prompt.trim().to_lowercase();
    lowered.starts_with("who ") || lowered.starts_with("which ")
}

fn heading_focus_score(prompt: &str, prompt_terms: &[String], title: &str) -> f32 {
    let title_terms = informative_terms(title)
        .into_iter()
        .collect::<BTreeSet<_>>();
    let lexical = overlap_ratio(prompt_terms, &title_terms);
    let prompt_norm = input::normalize_text(prompt);
    let title_norm = input::normalize_text(title);
    let phrase_match = if !title_norm.is_empty()
        && (prompt_norm.contains(&title_norm) || title_norm.contains(&prompt_norm))
    {
        1.0
    } else {
        0.0
    };
    let exact_heading = if !prompt_terms.is_empty()
        && prompt_terms.len() == title_terms.len()
        && prompt_terms.iter().all(|term| title_terms.contains(term))
    {
        1.0
    } else {
        0.0
    };
    let numeric_match = if first_number(&prompt_norm).is_some()
        && first_number(&prompt_norm) == first_number(&title_norm)
    {
        1.0
    } else {
        0.0
    };
    let trigram = jaccard_score(&char_trigrams(&prompt_norm), &char_trigrams(&title_norm));
    (0.35 * lexical
        + 0.24 * phrase_match
        + 0.16 * trigram
        + 0.15 * exact_heading
        + 0.10 * numeric_match)
        .clamp(0.0, 1.0)
}

fn ordinal_position(text: &str) -> Option<usize> {
    let normalized = input::normalize_text(text);
    for token in normalized.split_whitespace() {
        let cleaned = token
            .trim_matches(|ch: char| !ch.is_alphanumeric())
            .to_lowercase();
        let value = match cleaned.as_str() {
            "first" | "1st" => Some(1),
            "second" | "2nd" => Some(2),
            "third" | "3rd" => Some(3),
            "fourth" | "4th" => Some(4),
            "fifth" | "5th" => Some(5),
            "sixth" | "6th" => Some(6),
            "seventh" | "7th" => Some(7),
            "eighth" | "8th" => Some(8),
            "ninth" | "9th" => Some(9),
            "tenth" | "10th" => Some(10),
            _ => None,
        };
        if value.is_some() {
            return value;
        }
    }
    None
}

fn preceding_section_title(lines: &[String], start: usize) -> Option<String> {
    if start == 0 {
        return None;
    }

    for index in (0..start).rev() {
        let candidate = &lines[index];
        if candidate.is_empty() {
            continue;
        }
        if is_heading(candidate) && !looks_like_table_header(candidate) {
            return Some(candidate.clone());
        }
        if candidate.len() <= 40
            && !candidate.ends_with('.')
            && !candidate.ends_with('!')
            && !candidate.ends_with('?')
            && informative_terms(candidate).len() <= 6
        {
            return Some(candidate.clone());
        }
        if start - index > 3 {
            break;
        }
    }

    None
}

fn is_plausible_header_cell(line: &str, allow_hash: bool) -> bool {
    let trimmed = line.trim();
    if trimmed.is_empty() || trimmed.len() > 64 {
        return false;
    }
    if trimmed == "#" {
        return allow_hash;
    }
    if trimmed.ends_with('.') || trimmed.ends_with('!') || trimmed.ends_with('?') {
        return false;
    }
    let words = trimmed.split_whitespace().count();
    if words == 0 || words > 7 {
        return false;
    }
    if trimmed.chars().all(|ch| ch.is_ascii_digit()) {
        return false;
    }
    informative_terms(trimmed).len() <= 6
}

fn is_generic_table_header(line: &str) -> bool {
    let trimmed = line.trim();
    let word_count = trimmed.split_whitespace().count();
    let alpha_count = trimmed.chars().filter(|ch| ch.is_alphabetic()).count();
    let allowed_symbol_count = trimmed
        .chars()
        .filter(|ch| matches!(ch, '/' | '&' | '+' | '-'))
        .count();
    word_count >= 1
        && word_count <= 4
        && trimmed.len() <= 40
        && alpha_count >= (trimmed.len() / 3).max(3)
        && allowed_symbol_count <= 2
}

fn is_plausible_row(headers: &[String], row: &[String], kind: StructuredTableKind) -> bool {
    if row.len() != headers.len() || row.iter().any(|value| value.trim().is_empty()) {
        return false;
    }

    match kind {
        StructuredTableKind::Numbered => {
            if !row
                .first()
                .map(|value| is_numeric_marker(value))
                .unwrap_or(false)
            {
                return false;
            }
        }
        StructuredTableKind::Matrix => {
            if row
                .first()
                .map(|value| is_numeric_marker(value) || value == "#")
                .unwrap_or(true)
            {
                return false;
            }
        }
        StructuredTableKind::KeyValue => {
            if row.first().map(|value| value.len() > 96).unwrap_or(true) {
                return false;
            }
        }
    }

    let repeated_headers = row
        .iter()
        .filter(|value| {
            headers
                .iter()
                .any(|header| header.eq_ignore_ascii_case(value))
        })
        .count();
    repeated_headers == 0
}

fn score_table_quality(kind: StructuredTableKind, records: &[StructuredRecord]) -> f32 {
    let row_count = records.len().min(8) as f32 / 8.0;
    let avg_width = if records.is_empty() {
        0.0
    } else {
        records
            .iter()
            .map(|record| {
                record
                    .values
                    .iter()
                    .map(|value| value.len().min(120) as f32)
                    .sum::<f32>()
                    / record.values.len().max(1) as f32
            })
            .sum::<f32>()
            / records.len() as f32
    };
    let width_score = if (12.0..=140.0).contains(&avg_width) {
        1.0
    } else {
        0.65
    };
    let kind_bonus = match kind {
        StructuredTableKind::Numbered => 0.14,
        StructuredTableKind::Matrix => 0.1,
        StructuredTableKind::KeyValue => 0.06,
    };

    (0.45 + 0.3 * row_count + 0.15 * width_score + kind_bonus).clamp(0.0, 1.0)
}

fn looks_like_table_header(line: &str) -> bool {
    let trimmed = line.trim();
    if trimmed.is_empty() || trimmed.len() > 60 {
        return false;
    }
    !trimmed.ends_with('.') && !trimmed.ends_with('!') && !trimmed.ends_with('?')
}

fn is_numeric_marker(line: &str) -> bool {
    let trimmed = line.trim();
    !trimmed.is_empty() && trimmed.chars().all(|ch| ch.is_ascii_digit())
}

fn first_number(text: &str) -> Option<String> {
    text.split_whitespace()
        .map(|token| token.trim_matches(|ch: char| !ch.is_ascii_digit()))
        .find(|token| !token.is_empty())
        .map(str::to_string)
}

fn is_heading(line: &str) -> bool {
    let trimmed = line.trim();
    if trimmed.len() < 3 || trimmed.len() > 90 {
        return false;
    }
    if trimmed.ends_with('.') || trimmed.ends_with('!') || trimmed.ends_with('?') {
        return false;
    }
    if trimmed.ends_with(':') {
        return true;
    }

    let words = trimmed.split_whitespace().collect::<Vec<_>>();
    if words.len() > 10 {
        return false;
    }

    let uppercase_words = words
        .iter()
        .filter(|word| {
            word.chars()
                .next()
                .map(|ch| ch.is_uppercase())
                .unwrap_or(false)
        })
        .count();
    let uppercase_ratio = uppercase_words as f32 / words.len().max(1) as f32;

    trimmed == trimmed.to_uppercase()
        || uppercase_ratio > 0.7
        || starts_with_section_marker(trimmed)
}

fn starts_with_section_marker(line: &str) -> bool {
    let first = line.chars().next().unwrap_or_default();
    if first.is_ascii_digit() {
        return true;
    }

    let prefix = line
        .chars()
        .take_while(|ch| ch.is_ascii_alphabetic() || *ch == '.')
        .collect::<String>();
    matches!(prefix.as_str(), "I" | "II" | "III" | "IV" | "V" | "VI")
}

fn heading_depth(line: &str) -> usize {
    let trimmed = line.trim();
    let numeric_prefix = trimmed
        .split_whitespace()
        .next()
        .unwrap_or_default()
        .trim_matches(|ch: char| !ch.is_alphanumeric() && ch != '.');
    if numeric_prefix.chars().any(|ch| ch.is_ascii_digit()) {
        let segments = numeric_prefix
            .split('.')
            .filter(|segment| !segment.is_empty())
            .count();
        if segments > 0 {
            return segments;
        }
    }

    if trimmed == trimmed.to_uppercase() {
        return 1;
    }

    let word_count = trimmed.split_whitespace().count();
    if word_count <= 3 {
        1
    } else if word_count <= 6 {
        2
    } else {
        3
    }
}

fn should_merge_line(current: &str, next: &str) -> bool {
    if is_heading(next) {
        return false;
    }

    let ends_sentence = current.ends_with('.')
        || current.ends_with('!')
        || current.ends_with('?')
        || current.ends_with(':');

    !ends_sentence || next.len() < 80
}

fn sentence_shape_score(text: &str) -> f32 {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return 0.0;
    }

    let length = trimmed.len();
    if (35..=220).contains(&length) {
        1.0
    } else if (20..=280).contains(&length) {
        0.7
    } else {
        0.35
    }
}

fn infer_query_intent(prompt: &str, normalized: &str) -> IntentKind {
    let prompt_lower = prompt.to_lowercase();
    if prompt_lower.contains('?')
        || prompt_lower
            .split_whitespace()
            .next()
            .map(|token| {
                matches!(
                    token,
                    "who" | "what" | "when" | "where" | "which" | "why" | "how"
                )
            })
            .unwrap_or(false)
    {
        return IntentKind::Question;
    }

    if normalized.contains("summarize")
        || normalized.contains("summarise")
        || normalized.contains("summary")
        || normalized.contains("overview")
        || normalized.contains("gist")
    {
        return IntentKind::Summarize;
    }
    if normalized.contains("compare") || normalized.contains("difference") {
        return IntentKind::Compare;
    }
    if normalized.contains("plan")
        || normalized.contains("roadmap")
        || normalized.contains("next steps")
        || normalized.contains("step by step")
    {
        return IntentKind::Plan;
    }
    if normalized.contains("analyze") || normalized.contains("analysis") {
        return IntentKind::Analyze;
    }
    if normalized.contains("debug")
        || normalized.contains("fix")
        || normalized.contains("error")
        || normalized.contains("broken")
    {
        return IntentKind::Debug;
    }
    if normalized.contains("recommend") || normalized.contains("suggest") {
        return IntentKind::Recommend;
    }
    if normalized.contains("classify") || normalized.contains("categorize") {
        return IntentKind::Classify;
    }
    if normalized.contains("translate") || normalized.contains("translation") {
        return IntentKind::Translate;
    }
    if normalized.contains("critique") || normalized.contains("review") {
        return IntentKind::Critique;
    }
    if normalized.contains("brainstorm") || normalized.contains("ideas") {
        return IntentKind::Brainstorm;
    }
    if normalized.starts_with("implement ")
        || normalized.starts_with("run ")
        || normalized.starts_with("execute ")
        || normalized.starts_with("create ")
        || normalized.starts_with("add ")
        || normalized.starts_with("remove ")
        || normalized.starts_with("install ")
        || normalized.starts_with("patch ")
        || normalized.starts_with("update ")
    {
        return IntentKind::Act;
    }
    if normalized.contains("extract") || normalized.contains("list") {
        return IntentKind::Extract;
    }
    if normalized.contains("explain") || normalized.contains("describe") {
        return IntentKind::Explain;
    }

    IntentKind::Unknown
}

fn infer_question_focus(prompt: &str, normalized: &str, intent: IntentKind) -> QuestionFocus {
    if !matches!(
        intent,
        IntentKind::Question | IntentKind::Explain | IntentKind::Extract | IntentKind::Classify
    ) {
        return QuestionFocus::Unknown;
    }

    let first = prompt
        .split_whitespace()
        .next()
        .unwrap_or_default()
        .trim_matches(|ch: char| !ch.is_alphanumeric())
        .to_lowercase();
    match first.as_str() {
        "who" => QuestionFocus::Person,
        "when" => QuestionFocus::Time,
        "where" => QuestionFocus::Place,
        "what" => QuestionFocus::Definition,
        "which" => {
            if normalized.contains("person") || normalized.contains("president") {
                QuestionFocus::Person
            } else {
                QuestionFocus::Definition
            }
        }
        "how" => {
            if normalized.contains("how many") || normalized.contains("how much") {
                QuestionFocus::Quantity
            } else {
                QuestionFocus::Unknown
            }
        }
        _ => QuestionFocus::Unknown,
    }
}

fn question_focus_score(focus: QuestionFocus, sentence: &str) -> f32 {
    match focus {
        QuestionFocus::Person => person_answer_score(sentence),
        QuestionFocus::Time => time_answer_score(sentence),
        QuestionFocus::Place => place_answer_score(sentence),
        QuestionFocus::Quantity => quantity_answer_score(sentence),
        QuestionFocus::Definition => definition_answer_score(sentence),
        QuestionFocus::Unknown => 0.0,
    }
}

fn person_answer_score(sentence: &str) -> f32 {
    let tokens = sentence
        .split_whitespace()
        .map(|token| token.trim_matches(|ch: char| !ch.is_alphanumeric() && ch != '-'))
        .filter(|token| !token.is_empty())
        .collect::<Vec<_>>();
    let mut longest_run = 0usize;
    let mut current_run = 0usize;

    for token in &tokens {
        let is_name_like = token
            .chars()
            .next()
            .map(|ch| ch.is_uppercase())
            .unwrap_or(false)
            && token.chars().skip(1).any(|ch| ch.is_lowercase())
            && !matches!(token.to_lowercase().as_str(), "the" | "a" | "an");
        if is_name_like {
            current_run += 1;
            longest_run = longest_run.max(current_run);
        } else {
            current_run = 0;
        }
    }

    let copula = if sentence.contains(" is ") || sentence.contains(" was ") {
        0.25
    } else {
        0.0
    };
    let early_bonus = if tokens.iter().take(4).any(|token| {
        token
            .chars()
            .next()
            .map(|ch| ch.is_uppercase())
            .unwrap_or(false)
            && !matches!(token.to_lowercase().as_str(), "the" | "a" | "an")
    }) {
        0.12
    } else {
        0.0
    };

    match longest_run {
        0 => 0.0,
        1 => (0.22_f32 + (copula * 0.6) + (early_bonus * 0.4)).clamp(0.0, 1.0),
        _ => (0.62_f32 + copula + early_bonus).clamp(0.0, 1.0),
    }
}

fn time_answer_score(sentence: &str) -> f32 {
    let month_like = [
        "january",
        "february",
        "march",
        "april",
        "may",
        "june",
        "july",
        "august",
        "september",
        "october",
        "november",
        "december",
    ]
    .iter()
    .any(|month| sentence.to_lowercase().contains(month));
    let digit_count = sentence.chars().filter(|ch| ch.is_ascii_digit()).count();
    let tense_bonus = if sentence.contains(" on ") || sentence.contains(" in ") {
        0.15
    } else {
        0.0
    };

    ((if month_like { 0.55 } else { 0.0 })
        + ((digit_count.min(8) as f32 / 8.0) * 0.35)
        + tense_bonus)
        .clamp(0.0, 1.0)
}

fn place_answer_score(sentence: &str) -> f32 {
    let lowered = sentence.to_lowercase();
    let locative = if lowered.contains(" in ")
        || lowered.contains(" at ")
        || lowered.contains(" from ")
        || lowered.contains(" located ")
    {
        0.4
    } else {
        0.0
    };
    let capitalized_tokens = sentence
        .split_whitespace()
        .filter(|token| {
            token
                .chars()
                .next()
                .map(|ch| ch.is_uppercase())
                .unwrap_or(false)
        })
        .count();

    (locative + (capitalized_tokens.min(4) as f32 / 4.0) * 0.35).clamp(0.0, 1.0)
}

fn quantity_answer_score(sentence: &str) -> f32 {
    let digits = sentence.chars().filter(|ch| ch.is_ascii_digit()).count();
    let lowered = sentence.to_lowercase();
    let quantity_word = ["million", "billion", "percent", "%", "dozen"]
        .iter()
        .any(|token| lowered.contains(token));
    (((digits.min(8) as f32 / 8.0) * 0.7) + if quantity_word { 0.25 } else { 0.0 }).clamp(0.0, 1.0)
}

fn definition_answer_score(sentence: &str) -> f32 {
    let lowered = sentence.to_lowercase();
    let definition_like = if lowered.contains(" is ")
        || lowered.contains(" refers to ")
        || lowered.contains(" means ")
    {
        0.55
    } else {
        0.0
    };
    (definition_like + (sentence_shape_score(sentence) * 0.3)).clamp(0.0, 1.0)
}

fn quoted_segments(input: &str) -> Vec<String> {
    let mut segments = Vec::new();
    let mut active_quote = None;
    let mut current = String::new();

    for ch in input.chars() {
        match (active_quote, ch) {
            (None, '"' | '\'') => {
                active_quote = Some(ch);
                current.clear();
            }
            (Some(quote), current_ch) if current_ch == quote => {
                if !current.is_empty() {
                    segments.push(current.clone());
                }
                current.clear();
                active_quote = None;
            }
            (Some(_), current_ch) => current.push(current_ch),
            _ => {}
        }
    }

    segments
}

fn finalize_answer(text: &str) -> String {
    let collapsed = clean_inline_whitespace(text);
    let cleaned = collapsed.trim();
    if cleaned.is_empty() {
        return "I couldn't extract a useful answer from the document.".to_string();
    }
    let mut chars = cleaned.chars();
    let first = chars
        .next()
        .map(|ch| ch.to_uppercase().to_string())
        .unwrap_or_default();
    let rest = chars.collect::<String>();
    let sentence = format!("{first}{rest}");
    if sentence.ends_with('.') || sentence.ends_with('!') || sentence.ends_with('?') {
        sentence
    } else {
        format!("{sentence}.")
    }
}

fn clean_document_text(text: &str) -> String {
    text.replace("\r\n", "\n")
        .replace('\r', "\n")
        .lines()
        .map(clean_inline_whitespace)
        .filter(|line| !line.is_empty())
        .collect::<Vec<_>>()
        .join("\n")
}

fn clean_inline_whitespace(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn flush_block(current: &mut String, blocks: &mut Vec<String>) {
    let trimmed = clean_inline_whitespace(current);
    if !trimmed.is_empty() {
        blocks.push(trimmed);
    }
    current.clear();
}

fn push_unique(values: &mut Vec<String>, value: String) {
    if !values.iter().any(|existing| existing == &value) {
        values.push(value);
    }
}

fn unique_strings(values: Vec<String>) -> Vec<String> {
    let mut unique = Vec::new();
    for value in values {
        push_unique(&mut unique, value);
    }
    unique
}
