#!/usr/bin/env python3
"""
Large-scale corpus generator for pollution testing.
Generates 5MB+ files per category with realistic pollution patterns.
"""

import os
import random
import string
import json
from pathlib import Path

OUTPUT_DIR = Path(__file__).parent.parent / "test_data" / "large_corpus"
TARGET_SIZE_MB = 5
TARGET_SIZE_BYTES = TARGET_SIZE_MB * 1024 * 1024

# Word lists for realistic content generation
CLEAN_WORDS = [
    "the", "be", "to", "of", "and", "a", "in", "that", "have", "I",
    "it", "for", "not", "on", "with", "he", "as", "you", "do", "at",
    "this", "but", "his", "by", "from", "they", "we", "say", "her", "she",
    "or", "an", "will", "my", "one", "all", "would", "there", "their", "what",
    "so", "up", "out", "if", "about", "who", "get", "which", "go", "me",
    "when", "make", "can", "like", "time", "no", "just", "him", "know", "take",
    "people", "into", "year", "your", "good", "some", "could", "them", "see", "other",
    "than", "then", "now", "look", "only", "come", "its", "over", "think", "also",
    "back", "after", "use", "two", "how", "our", "work", "first", "well", "way",
    "even", "new", "want", "because", "any", "these", "give", "day", "most", "us",
    # Technical words
    "system", "data", "process", "memory", "function", "method", "class", "object",
    "variable", "constant", "parameter", "argument", "return", "value", "type",
    "string", "integer", "float", "boolean", "array", "list", "dictionary", "map",
    "algorithm", "structure", "pattern", "design", "architecture", "component",
    "module", "package", "library", "framework", "interface", "implementation",
    "reasoning", "analysis", "synthesis", "evaluation", "validation", "testing",
    "production", "development", "deployment", "configuration", "optimization",
]

TECHNICAL_WORDS = [
    "api", "endpoint", "request", "response", "header", "body", "status", "code",
    "authentication", "authorization", "token", "session", "cookie", "cache",
    "database", "query", "index", "table", "column", "row", "schema", "migration",
    "server", "client", "proxy", "gateway", "load", "balance", "cluster", "node",
    "microservice", "monolith", "container", "docker", "kubernetes", "pod", "service",
    "async", "sync", "parallel", "concurrent", "thread", "process", "socket", "stream",
    "json", "xml", "yaml", "csv", "protobuf", "graphql", "rest", "grpc", "websocket",
    "encryption", "decryption", "hash", "salt", "cipher", "key", "certificate", "tls",
]

UNICODE_CHARS = [
    "café", "naïve", "façade", "résumé", "coöperate", "über", "münchen", "zürich",
    "tokyo", "beijing", "seoul", "bangkok", "jerusalem", "cairo", "moscow", "prague",
    "日本語", "한국어", "العربية", "中文", "ελληνικά", "русский", "עברית", "हिन्दी",
]


def random_unicode_escape():
    """Generate random unicode escape sequences like \\uXXXX"""
    codepoint = random.randint(0x0020, 0xFFFF)
    return f"\\u{codepoint:04X}"


def random_punctuation():
    """Generate random punctuation pollution"""
    puncts = ["-", "_", ".", "!", "?", ",", ":", ";", "'", '"', "(", ")", "[", "]", "{", "}"]
    return random.choice(puncts)


def random_url():
    """Generate random URL patterns"""
    domains = ["example.com", "api.service.io", "cdn.example.org", "static.assets.net"]
    paths = ["/api/v1/users", "/data/export", "/files/download", "/images/photo.png"]
    return f"https://{random.choice(domains)}{random.choice(paths)}"


def random_file_path():
    """Generate random file path patterns"""
    dirs = ["/home/user", "/var/log", "/etc/config", "/usr/local/bin", "/tmp/cache"]
    files = ["config.yaml", "data.json", "report.pdf", "image.png", "script.py"]
    return f"{random.choice(dirs)}/{random.choice(files)}"


def random_email():
    """Generate random email patterns"""
    names = ["john", "jane", "admin", "support", "info", "contact", "user", "test"]
    domains = ["example.com", "company.org", "service.io", "app.net"]
    return f"{random.choice(names)}@{random.choice(domains)}"


def random_sql():
    """Generate random SQL-like patterns"""
    patterns = [
        "SELECT * FROM users WHERE id = 1",
        "INSERT INTO logs VALUES ('test')",
        "UPDATE config SET value = 'new'",
        "DELETE FROM cache WHERE expired = true",
        "UNION SELECT password FROM admin",
        "'; DROP TABLE users; --",
    ]
    return random.choice(patterns)


def random_html():
    """Generate random HTML/XML fragments"""
    patterns = [
        '<div class="container"><p>Text</p></div>',
        '&lt;escaped&gt; &amp; &quot;entities&quot;',
        '<!-- comment --><span>visible</span>',
        '<script>alert("xss")</script>',
        '<a href="link">click</a>',
        '<img src="image.png" alt="desc"/>',
    ]
    return random.choice(patterns)


def random_json_fragment():
    """Generate random JSON fragments (some malformed)"""
    patterns = [
        '{"key": "value", "number": 42}',
        '{"nested": {"deep": {"data": true}}}',
        '{"array": [1, 2, 3, "mixed"]}',
        '{"broken": "missing_quote}',
        '{"unclosed": {"nested": "yes"',
        '{"unicode": "\\u00E9\\u00E8"}',
        '{"escaped": "line\\nbreak"}',
    ]
    return random.choice(patterns)


def random_base64():
    """Generate random base64-like strings"""
    chars = string.ascii_letters + string.digits + "+/="
    length = random.randint(16, 64)
    return ''.join(random.choice(chars) for _ in range(length))


def random_control_chars():
    """Generate strings with control characters"""
    base = "Text with"
    control = random.choice(['\x00', '\x07', '\x08', '\x1b', '\x7f'])
    return f"{base}{control}control{control}chars"


def random_emoji():
    """Generate strings with emoji"""
    emojis = ["👋", "🌍", "🎉", "✨", "🔥", "💯", "👍", "🚀", "💡", "⚡"]
    return f"Hello {random.choice(emojis)} World {random.choice(emojis)}!"


def random_code_fragment():
    """Generate random code fragments"""
    patterns = [
        'fn main() { println!("Hello"); }',
        'def function(): return True',
        'const x = () => { return 42; };',
        'public static void main(String[] args)',
        '#include <stdio.h>\nint main()',
        'package main\nimport "fmt"',
        'if (condition) { doSomething(); }',
        'for (let i = 0; i < n; i++) { process(i); }',
        'try { risky() } catch (e) { handle(e) }',
        'SELECT u.name FROM users u WHERE u.active = true',
    ]
    return random.choice(patterns)


def random_adversarial():
    """Generate adversarial patterns"""
    patterns = [
        'A' * 100,
        'a' * 100,
        'word' * 25,
        'AaAaAaAaAa' * 10,
        '123' * 33,
        ''.join(random.choice('!@#$%^&*()') for _ in range(50)),
    ]
    return random.choice(patterns)


def random_whitespace_anomaly():
    """Generate whitespace anomalies"""
    patterns = [
        'Multiple   spaces   between   words',
        'Tabs\t\t\tmultiple\t\ttabs',
        '  Leading and trailing  ',
        'Non-breaking\u00A0space\u00A0here',
        'Zero\u200Bwidth\u200Bspace',
    ]
    return random.choice(patterns)


def random_numeric():
    """Generate numeric/date patterns"""
    patterns = [
        'Date: 2024-01-15T10:30:00Z',
        'Time: 10:30:45.123',
        'Money: $1,234.56 USD',
        'Percentage: 99.9%',
        'Scientific: 1.23e-10',
        'Hex color: #FF5733',
        'UUID: 550e8400-e29b-41d4-a716-446655440000',
        'Version: v2.3.1-beta.2+build.123',
        'IP: 192.168.1.1',
        'Port: 8080',
    ]
    return random.choice(patterns)


def generate_sentence(clean_ratio=0.7):
    """Generate a sentence with mix of clean and polluted content"""
    words = []
    word_count = random.randint(5, 20)
    
    for _ in range(word_count):
        if random.random() < clean_ratio:
            # Clean word
            words.append(random.choice(CLEAN_WORDS + TECHNICAL_WORDS))
        else:
            # Potentially polluted
            pollution_type = random.choice([
                'unicode_escape', 'punctuation', 'url', 'email', 
                'json', 'html', 'control', 'emoji', 'code', 'numeric'
            ])
            
            if pollution_type == 'unicode_escape':
                words.append(f"symbol{random_unicode_escape()}")
            elif pollution_type == 'punctuation':
                word = random.choice(CLEAN_WORDS)
                punct = random_punctuation()
                if random.random() < 0.5:
                    words.append(f"{word}{punct}")
                else:
                    words.append(f"{punct}{word}")
            elif pollution_type == 'url':
                words.append(random_url())
            elif pollution_type == 'email':
                words.append(random_email())
            elif pollution_type == 'json':
                words.append(random_json_fragment())
            elif pollution_type == 'html':
                words.append(random_html())
            elif pollution_type == 'control':
                words.append(random_control_chars())
            elif pollution_type == 'emoji':
                words.append(random_emoji())
            elif pollution_type == 'code':
                words.append(random_code_fragment())
            elif pollution_type == 'numeric':
                words.append(random_numeric())
    
    return ' '.join(words)


def generate_paragraph(sentence_count=5):
    """Generate a paragraph of sentences"""
    return '. '.join(generate_sentence() for _ in range(sentence_count)) + '.'


def generate_category_escaped_unicode(output_path: Path):
    """Category 1: Escaped Unicode patterns"""
    print(f"Generating escaped_unicode corpus...")
    content = []
    current_size = 0
    
    while current_size < TARGET_SIZE_BYTES:
        # Mix of clean unicode text and escaped sequences
        if random.random() < 0.3:
            # Clean unicode
            text = f"The {random.choice(UNICODE_CHARS)} word is valid. "
        else:
            # Escaped unicode pollution
            escapes = [random_unicode_escape() for _ in range(random.randint(1, 5))]
            text = f"Text with {' '.join(escapes)} sequences. "
        
        content.append(text)
        current_size += len(text.encode('utf-8'))
    
    output_path.write_text('\n'.join(content), encoding='utf-8')
    print(f"  Written {current_size / 1024 / 1024:.2f} MB")


def generate_category_outer_punct(output_path: Path):
    """Category 2: Outer punctuation fragments"""
    print(f"Generating outer_punct corpus...")
    content = []
    current_size = 0
    
    while current_size < TARGET_SIZE_BYTES:
        if random.random() < 0.4:
            # Clean text
            text = generate_paragraph(3)
        else:
            # Punctuation pollution
            words = [random.choice(CLEAN_WORDS) for _ in range(random.randint(5, 15))]
            polluted = []
            for word in words:
                if random.random() < 0.3:
                    punct = random_punctuation()
                    if random.random() < 0.5:
                        polluted.append(f"{word}{punct}")
                    else:
                        polluted.append(f"{punct}{word}")
                else:
                    polluted.append(word)
            text = ' '.join(polluted) + '. '
        
        content.append(text)
        current_size += len(text.encode('utf-8'))
    
    output_path.write_text('\n'.join(content), encoding='utf-8')
    print(f"  Written {current_size / 1024 / 1024:.2f} MB")


def generate_category_url_fragments(output_path: Path):
    """Category 3: URL and path fragments"""
    print(f"Generating url_fragments corpus...")
    content = []
    current_size = 0
    
    while current_size < TARGET_SIZE_BYTES:
        if random.random() < 0.3:
            # Clean text
            text = generate_paragraph(3)
        else:
            # URL/path pollution
            url_count = random.randint(1, 5)
            urls = []
            for _ in range(url_count):
                if random.random() < 0.5:
                    urls.append(random_url())
                else:
                    urls.append(random_file_path())
            
            text = f"Visit {' '.join(urls)} for more info. "
            text += generate_sentence(clean_ratio=0.8) + ' '
        
        content.append(text)
        current_size += len(text.encode('utf-8'))
    
    output_path.write_text('\n'.join(content), encoding='utf-8')
    print(f"  Written {current_size / 1024 / 1024:.2f} MB")


def generate_category_broken_json(output_path: Path):
    """Category 4: Broken/malformed JSON"""
    print(f"Generating broken_json corpus...")
    content = []
    current_size = 0
    
    while current_size < TARGET_SIZE_BYTES:
        if random.random() < 0.3:
            # Valid JSON
            data = {
                "id": random.randint(1, 10000),
                "text": generate_sentence(clean_ratio=0.9),
                "timestamp": "2024-01-15T10:30:00Z",
                "valid": True
            }
            text = json.dumps(data)
        else:
            # Malformed JSON
            text = random_json_fragment()
        
        content.append(text)
        current_size += len(text.encode('utf-8'))
    
    output_path.write_text('\n'.join(content), encoding='utf-8')
    print(f"  Written {current_size / 1024 / 1024:.2f} MB")


def generate_category_encoding_issues(output_path: Path):
    """Category 5: Encoding issues and mojibake"""
    print(f"Generating encoding_issues corpus...")
    content = []
    current_size = 0
    
    mojibake_patterns = [
        "CafÃ© with mojibake",
        "Ã¼ber Ã¶ffentlich",
        "ÐÑÑÑÐ¸Ð¹ text",
        "æ—¥æœ¬èªž mojibake",
        "Valid UTF-8: 日本語 한국어 العربية",
        "Mixed: café, über, naïve",
    ]
    
    while current_size < TARGET_SIZE_BYTES:
        if random.random() < 0.5:
            text = generate_paragraph(3)
        else:
            text = random.choice(mojibake_patterns) + ' '
            text += generate_sentence(clean_ratio=0.7) + ' '
        
        content.append(text)
        current_size += len(text.encode('utf-8'))
    
    output_path.write_text('\n'.join(content), encoding='utf-8')
    print(f"  Written {current_size / 1024 / 1024:.2f} MB")


def generate_category_sql_patterns(output_path: Path):
    """Category 6: SQL injection patterns"""
    print(f"Generating sql_patterns corpus...")
    content = []
    current_size = 0
    
    while current_size < TARGET_SIZE_BYTES:
        if random.random() < 0.4:
            text = generate_paragraph(3)
        else:
            text = random_sql() + ' '
            text += generate_sentence(clean_ratio=0.8) + ' '
        
        content.append(text)
        current_size += len(text.encode('utf-8'))
    
    output_path.write_text('\n'.join(content), encoding='utf-8')
    print(f"  Written {current_size / 1024 / 1024:.2f} MB")


def generate_category_html_markup(output_path: Path):
    """Category 7: HTML/XML markup"""
    print(f"Generating html_markup corpus...")
    content = []
    current_size = 0
    
    while current_size < TARGET_SIZE_BYTES:
        if random.random() < 0.3:
            text = generate_paragraph(3)
        else:
            text = random_html() + ' '
            text += generate_sentence(clean_ratio=0.7) + ' '
        
        content.append(text)
        current_size += len(text.encode('utf-8'))
    
    output_path.write_text('\n'.join(content), encoding='utf-8')
    print(f"  Written {current_size / 1024 / 1024:.2f} MB")


def generate_category_control_chars(output_path: Path):
    """Category 8: Control characters"""
    print(f"Generating control_chars corpus...")
    content = []
    current_size = 0
    
    while current_size < TARGET_SIZE_BYTES:
        if random.random() < 0.3:
            text = generate_paragraph(3)
        else:
            text = random_control_chars() + ' '
            text += generate_sentence(clean_ratio=0.8) + ' '
        
        content.append(text)
        current_size += len(text.encode('utf-8'))
    
    # Write as binary to preserve control chars
    with open(output_path, 'wb') as f:
        f.write('\n'.join(content).encode('utf-8'))
    print(f"  Written {current_size / 1024 / 1024:.2f} MB")


def generate_category_emoji(output_path: Path):
    """Category 9: Emoji and special unicode"""
    print(f"Generating emoji corpus...")
    content = []
    current_size = 0
    
    while current_size < TARGET_SIZE_BYTES:
        if random.random() < 0.5:
            text = generate_paragraph(3)
        else:
            text = random_emoji() + ' '
            text += generate_sentence(clean_ratio=0.7) + ' '
        
        content.append(text)
        current_size += len(text.encode('utf-8'))
    
    output_path.write_text('\n'.join(content), encoding='utf-8')
    print(f"  Written {current_size / 1024 / 1024:.2f} MB")


def generate_category_code_fragments(output_path: Path):
    """Category 10: Programming code fragments"""
    print(f"Generating code_fragments corpus...")
    content = []
    current_size = 0
    
    while current_size < TARGET_SIZE_BYTES:
        if random.random() < 0.3:
            text = generate_paragraph(3)
        else:
            text = random_code_fragment() + ' '
            text += generate_sentence(clean_ratio=0.8) + ' '
        
        content.append(text)
        current_size += len(text.encode('utf-8'))
    
    output_path.write_text('\n'.join(content), encoding='utf-8')
    print(f"  Written {current_size / 1024 / 1024:.2f} MB")


def generate_category_adversarial(output_path: Path):
    """Category 11: Adversarial patterns"""
    print(f"Generating adversarial corpus...")
    content = []
    current_size = 0
    
    while current_size < TARGET_SIZE_BYTES:
        if random.random() < 0.3:
            text = generate_paragraph(3)
        else:
            text = random_adversarial() + ' '
        
        content.append(text)
        current_size += len(text.encode('utf-8'))
    
    output_path.write_text('\n'.join(content), encoding='utf-8')
    print(f"  Written {current_size / 1024 / 1024:.2f} MB")


def generate_category_whitespace(output_path: Path):
    """Category 12: Whitespace anomalies"""
    print(f"Generating whitespace corpus...")
    content = []
    current_size = 0
    
    while current_size < TARGET_SIZE_BYTES:
        if random.random() < 0.4:
            text = generate_paragraph(3)
        else:
            text = random_whitespace_anomaly() + ' '
        
        content.append(text)
        current_size += len(text.encode('utf-8'))
    
    output_path.write_text('\n'.join(content), encoding='utf-8')
    print(f"  Written {current_size / 1024 / 1024:.2f} MB")


def generate_category_numeric(output_path: Path):
    """Category 13: Numeric/date fragments"""
    print(f"Generating numeric corpus...")
    content = []
    current_size = 0
    
    while current_size < TARGET_SIZE_BYTES:
        if random.random() < 0.4:
            text = generate_paragraph(3)
        else:
            text = random_numeric() + ' '
            text += generate_sentence(clean_ratio=0.8) + ' '
        
        content.append(text)
        current_size += len(text.encode('utf-8'))
    
    output_path.write_text('\n'.join(content), encoding='utf-8')
    print(f"  Written {current_size / 1024 / 1024:.2f} MB")


def generate_category_mixed(output_path: Path):
    """Category 14: Mixed realistic content"""
    print(f"Generating mixed corpus...")
    content = []
    current_size = 0
    
    generators = [
        generate_paragraph,
        lambda: random_url(),
        lambda: random_email(),
        lambda: random_json_fragment(),
        lambda: random_html(),
        lambda: random_emoji(),
        lambda: random_code_fragment(),
        lambda: random_numeric(),
    ]
    
    while current_size < TARGET_SIZE_BYTES:
        # Mix of all types
        parts = []
        for _ in range(random.randint(3, 10)):
            parts.append(random.choice(generators)())
        text = ' '.join(parts) + '\n'
        
        content.append(text)
        current_size += len(text.encode('utf-8'))
    
    output_path.write_text(''.join(content), encoding='utf-8')
    print(f"  Written {current_size / 1024 / 1024:.2f} MB")


def main():
    print(f"=== Large Corpus Generator ===")
    print(f"Target: {TARGET_SIZE_MB} MB per category")
    print(f"Output: {OUTPUT_DIR}")
    print()
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    categories = [
        ("escaped_unicode", generate_category_escaped_unicode),
        ("outer_punct", generate_category_outer_punct),
        ("url_fragments", generate_category_url_fragments),
        ("broken_json", generate_category_broken_json),
        ("encoding_issues", generate_category_encoding_issues),
        ("sql_patterns", generate_category_sql_patterns),
        ("html_markup", generate_category_html_markup),
        ("control_chars", generate_category_control_chars),
        ("emoji_content", generate_category_emoji),
        ("code_fragments", generate_category_code_fragments),
        ("adversarial", generate_category_adversarial),
        ("whitespace_anomaly", generate_category_whitespace),
        ("numeric_fragments", generate_category_numeric),
        ("mixed_realistic", generate_category_mixed),
    ]
    
    total_size = 0
    for name, generator in categories:
        output_path = OUTPUT_DIR / f"{name}.txt"
        generator(output_path)
        size = output_path.stat().st_size
        total_size += size
    
    print()
    print(f"=== Summary ===")
    print(f"Total corpus size: {total_size / 1024 / 1024:.2f} MB")
    print(f"Categories: {len(categories)}")
    
    # List all files with sizes
    print("\nFile sizes:")
    for name, _ in categories:
        path = OUTPUT_DIR / f"{name}.txt"
        size = path.stat().st_size
        print(f"  {name}.txt: {size / 1024 / 1024:.2f} MB")


if __name__ == "__main__":
    main()
