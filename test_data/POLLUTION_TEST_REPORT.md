# Pollution Detection Test Report

## Test Corpus Summary

### Small Corpus (Initial Testing)
- **Size**: ~50 KB
- **Documents**: 124
- **Categories**: 22

### Large Corpus (Stress Testing)
- **Size**: 70.42 MB
- **Documents**: 480,784
- **Categories**: 14 (5 MB each)

---

## Test Data Categories

### 1. Escaped Unicode (`escaped_unicode.txt` - 5.13 MB)
**Purpose**: Test detection of JSON unicode escape sequences that leak into text.

**Pollution Patterns**:
```
\u0259    → schwa symbol escaped
\u00B0    → degree symbol escaped
\u00E9    → accented e escaped
\u20AC    → euro symbol escaped
```

**Expected Clean**:
```
café, naïve, façade, München, Zürich
```

**Detection Logic**: `looks_like_unicode_escape()` in `builder.rs`

---

### 2. Outer Punctuation Fragments (`outer_punct.txt` - 5.02 MB)
**Purpose**: Test rejection of fragments with leading/trailing punctuation.

**Pollution Patterns**:
```
Claude-    (trailing hyphen)
Sudan_     (trailing underscore)
file.      (trailing period)
"nested    (leading quote)
valid"     (trailing quote)
```

**Detection Logic**: `has_outer_punctuation()` in `builder.rs`

---

### 3. URL/Path Fragments (`url_fragments.txt` - 5.02 MB)
**Purpose**: Test rejection of URL and file path fragments.

**Pollution Patterns**:
```
https://example.com/page-1
/home/user/documents/report.pdf
api/v1/users
file.png
```

**Detection Logic**: `looks_like_url_fragment()` in `builder.rs`

---

### 4. Broken JSON (`broken_json.txt` - 5.07 MB)
**Purpose**: Test handling of malformed JSON fragments.

**Pollution Patterns**:
```
{"key": "value", "broken: missing_quote}
{"nested": {"deep": {"unclosed": "yes"
{"unicode": "\u00", "truncated": true}
```

**Current Issue**: Fragments like `"nested"`, `{"nested`, `true}` still pass through.

---

### 5. Encoding Issues (`encoding_issues.txt` - 5.02 MB)
**Purpose**: Test handling of mojibake and mixed encodings.

**Pollution Patterns**:
```
CafÃ© with mojibake      (UTF-8 misdecoded as Latin-1)
Ã¼ber Ã¶ffentlich        (German with wrong encoding)
ÐÑÑÑÐ¸Ð¹                 (Russian in wrong encoding)
æ—¥æœ¬èªž                (Japanese mojibake)
```

---

### 6. SQL Patterns (`sql_patterns.txt` - 5.02 MB)
**Purpose**: Test that SQL fragments don't pollute the unit database.

**Test Patterns**:
```
SELECT * FROM users WHERE id = 1
'; DROP TABLE users; --
UNION SELECT password FROM admin
INSERT INTO logs VALUES ('test')
```

---

### 7. HTML/XML Markup (`html_markup.txt` - 5.02 MB)
**Purpose**: Test filtering of markup fragments.

**Pollution Patterns**:
```
<div class="container">
&lt;escaped&gt; &amp; &quot;entities&quot;
<!-- comment --><span>visible</span>
<script>alert('xss')</script>
```

**Current Issue**: `quot;` still detected as pollution.

---

### 8. Control Characters (`control_chars.txt` - 5.02 MB)
**Purpose**: Test handling of null bytes and control chars.

**Pollution Patterns**:
```
Text with\x00null\x00bytes
Line1\nLine2\nLine3
Tab\there\tand\tthere
Bell\x07and\x08backspace
```

---

### 9. Emoji Content (`emoji_content.txt` - 5.02 MB)
**Purpose**: Test that emoji and special unicode are handled correctly.

**Test Patterns**:
```
Hello 👋 World 🌍!
Math: 2×3=6, a²+b²=c²
Arrows: → ← ↑ ↓ ↔
Currency: $ € £ ¥ ₹ ₿
```

---

### 10. Code Fragments (`code_fragments.txt` - 5.02 MB)
**Purpose**: Test that programming language syntax doesn't pollute.

**Test Patterns**:
```
fn main() { println!("Hello"); }
def function(): return True
const x = () => { return 42; };
public static void main(String[] args)
```

**Current Issue**: `main(` detected as pollution (outer punct).

---

### 11. Adversarial Patterns (`adversarial.txt` - 5.03 MB)
**Purpose**: Test resilience against intentionally malformed input.

**Pollution Patterns**:
```
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
a a a a a a a a a a a a a a a a a a a a
wordwordwordwordwordwordwordwordwordword
AaAaAaAaAaAaAaAaAaAaAaAaAaAaAaAaAaAaAaAa
```

---

### 12. Whitespace Anomalies (`whitespace_anomaly.txt` - 5.03 MB)
**Purpose**: Test handling of unusual whitespace patterns.

**Pollution Patterns**:
```
Multiple   spaces   between   words
Tabs\t\t\tmultiple\t\ttabs
Non-breaking\u00A0space\u00A0here
Zero\u200Bwidth\u200Bspace
```

---

### 13. Numeric Fragments (`numeric_fragments.txt` - 5.02 MB)
**Purpose**: Test handling of dates, times, UUIDs, and numeric patterns.

**Test Patterns**:
```
Date: 2024-01-15T10:30:00Z
Time: 10:30:45.123
UUID: 550e8400-e29b-41d4-a716-446655440000
Version: v2.3.1-beta.2+build.123
```

**NOTE**: Time formats like `10:30`, `30:45` are now correctly excluded from pollution detection.

---

### 14. Mixed Realistic (`mixed_realistic.txt` - 5.00 MB)
**Purpose**: Real-world messy data combining all patterns.

**Test Patterns**:
```
{"question":"What is 2+2?","context":"Math basics","answer":"4"}
User123 commented: 'Great article!!! 5/5 stars 👍👍👍'
RT @user: Check this out! https://t.co/abc123 #hashtag
[ERROR] 2024-01-15 10:30:45 - Connection failed (code: 500)
```

---

## Critical Issue: Edge-Trimmed Date Fragments

### Problem Description

Before the pollution detection improvements, the byte-window rolling hash would create **edge-trimmed partial fragments** from structured data like dates.

### Example: Date `2025-05-10`

**Before fixes**, the database would contain these polluted units:

| Fragment | Reason | Status |
|----------|--------|--------|
| `2025-05-10` | Full date (clean) | ✅ Valid |
| `2025-05-` | Edge-trimmed (trailing hyphen) | ❌ Pollution |
| `-05-10` | Edge-trimmed (leading hyphen) | ❌ Pollution |
| `-05-1` | Edge-trimmed (partial) | ❌ Pollution |
| `-10` | Edge-trimmed (too short) | ❌ Pollution |
| `05-` | Edge-trimmed (trailing hyphen) | ❌ Pollution |

### Root Cause

The rolling hash byte-window operates on byte boundaries, not semantic boundaries. When processing text like:

```
Date: 2025-05-10 for the event
```

The window would slide across the date string and create fragments at every position:

```
Position 0: "Date: 2025"
Position 1: "ate: 2025-"
Position 2: "te: 2025-0"
...
Position 10: "2025-05-1"
Position 11: "025-05-10"
Position 12: "25-05-10 "  → "25-05-10"
Position 13: "5-05-10 f"  → "5-05-10"
...
```

### Why These Were Not Filtered

1. **`min_fragment_length: 4`** - Fragments like `-10` (length 3) were filtered, but `-05-1` (length 5) passed.
2. **No outer punctuation check** - Fragments like `2025-05-` had trailing punctuation but no detection.
3. **No edge-trim detection** - Partial words from mid-string cuts were not identified.

### Fixes Applied

1. **Increased `min_fragment_length` to 5** - Filters shorter meaningless fragments.
2. **Added `has_outer_punctuation()`** - Rejects fragments with leading/trailing punctuation unless they have full token boundary hits.
3. **Added `looks_like_url_fragment()`** - Rejects URL/path-like fragments including file extensions.
4. **Increased `min_frequency_threshold` to 3** - Requires more corroboration before unit activation.
5. **Reduced `punctuation_ratio_limit` to 0.40** - Stricter punctuation filtering.

### After Fixes

For the same date `2025-05-10`:

| Fragment | Status | Reason |
|----------|--------|--------|
| `2025-05-10` | ✅ Valid | Full date, proper boundaries |
| `2025-05-` | ❌ Rejected | Outer punctuation (trailing hyphen) |
| `-05-10` | ❌ Rejected | Outer punctuation (leading hyphen) |
| `-05-1` | ❌ Rejected | Outer punctuation + low frequency |
| `-10` | ❌ Rejected | Below min_fragment_length |

---

## Test Results Comparison

| Metric | Before Fixes | After Fixes | Improvement |
|--------|--------------|-------------|-------------|
| **Small Corpus** ||||
| Pollution score | 0.116 (11.6%) | 0.012 (1.2%) | **91.5% reduction** |
| Polluted units | 42 / 295 | 3 / 207 | **93% fewer** |
| **Large Corpus** ||||
| Pollution score | N/A (too slow) | 0.057 (5.7%) | Baseline established |
| Polluted units | N/A | 26 / 383 | 26 remaining |
| Processing time | 398s | 214s | **46% faster** |

---

## Remaining Pollution Patterns

The 26 remaining polluted units from large corpus testing:

1. **JSON fragments with outer punctuation** (24 units):
   - `"nested"`, `"valid"`, `{"nested`, `true}`, `quot;`
   
2. **Code fragments with outer punctuation** (1 unit):
   - `main(`

3. **URL-like fragments** (1 unit):
   - Single `:` character

### Recommended Future Fixes

1. **Add JSON fragment detection** - Detect patterns like `"key"`, `{"key`, `value}` as JSON pollution.
2. **Add code syntax detection** - Detect `function(`, `main(` as code fragments.
3. **Stricter outer punctuation** - Require higher boundary hit ratio for punctuation-wrapped fragments.

---

## Configuration Changes

### Before
```yaml
layer_2_unit_builder:
  min_frequency_threshold: 2
  rolling_hash_window_sizes: [2, 3, 4, 5, 6, 7, 8]
  punctuation_ratio_limit: 0.55
  min_fragment_length: 4
  utility_full_boundary_bonus: 0.18
  utility_edge_boundary_bonus: 0.05
  utility_no_boundary_penalty: -0.12
```

### After
```yaml
layer_2_unit_builder:
  min_frequency_threshold: 3
  rolling_hash_window_sizes: [3, 4, 5, 6, 7, 8]
  punctuation_ratio_limit: 0.40
  min_fragment_length: 5
  utility_full_boundary_bonus: 0.22
  utility_edge_boundary_bonus: 0.03
  utility_no_boundary_penalty: -0.18
```

---

## Performance Optimizations

### Issue
Processing 480,784 documents took 398 seconds due to synchronous SQLite writes per unit.

### Solution
Added deferred write batching:
- `pending_writes` buffer in `MemoryStore`
- `batch_upsert_units()` in `Db` for transactional bulk writes
- Flush threshold of 500 units per transaction

### Result
- **46% faster** (398s → 214s)
- Same pollution detection accuracy
- No data loss (flush on completion)

---

## How to Run Tests

### Small Corpus (Quick)
```bash
cargo run --release --bin pollution_dev -- --dry-run
```

### Large Corpus (Full)
```bash
# Generate corpus first (one-time)
python3 scripts/generate_large_corpus.py

# Run test
cargo run --release --bin pollution_dev -- --large
```

### Config Sweep
```bash
cargo run --release --bin pollution_dev -- --sweep
```

---

## Files Modified

| File | Changes |
|------|---------|
| `config/config.yaml` | Updated pollution detection parameters |
| `src/layers/builder.rs` | Added `looks_like_unicode_escape()`, `looks_like_url_fragment()`, `has_outer_punctuation()` |
| `src/memory/store.rs` | Added `pending_writes`, `flush_pending_writes()`, `set_write_deferred()` |
| `src/persistence.rs` | Added `batch_upsert_units()` |
| `src/bin/pollution_dev_lib.rs` | Added `load_large_corpus()`, fixed time format detection |
| `scripts/generate_large_corpus.py` | New: 70MB corpus generator |

---

*Report generated: 2025-03-14*
*SPSE Engine v0.1.0*
