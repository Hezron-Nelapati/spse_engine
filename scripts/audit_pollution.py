#!/usr/bin/env python3
"""
Pollution Audit Script for Structured Predictive Search (SPS) Architecture
Detects polluted unit variants and proposes cleaner canonical forms.

Usage:
  DB=/path/to/spse_memory.db TOP_K=200 python3 audit_pollution.py > pollution_candidates.tsv
"""

import os, sys, sqlite3, math, re, unicodedata
from collections import defaultdict, Counter
from typing import Set, Dict, List, Optional, Tuple

# === Configuration via environment ===
DB = os.environ.get("DB", "/Volumes/SSD/Github/spse_engine/spse_memory.db")
TOP_K = int(os.environ.get("TOP_K", "200"))
MIN_LEN = int(os.environ.get("MIN_LEN", "4"))
EDGE = int(os.environ.get("EDGE_TRIM", "3"))
THRESHOLD = float(os.environ.get("POLLUTION_THRESHOLD", "0.65"))
QUALITY_MARGIN = float(os.environ.get("QUALITY_MARGIN", "0.08"))
USE_RARE_SIG_WEIGHTING = os.environ.get("RARE_SIG_WEIGHT", "false").lower() == "true"

# === Regex patterns ===
escaped_re = re.compile(r'u[0-9a-f]{2,6}', re.I)
url_re = re.compile(r'(https?://|www\.|\.[a-z]{2,4}$)', re.I)

# === Core utilities ===

def trim_outer_punct(s: str) -> str:
    if not s:
        return s
    i, j = 0, len(s)
    while i < j and not s[i].isalnum():
        i += 1
    while j > i and not s[j - 1].isalnum():
        j -= 1
    return s[i:j]

def punct_ratio(s: str) -> float:
    if not s:
        return 1.0
    punct = sum(1 for ch in s if not ch.isalnum())
    return punct / max(len(s), 1)

def is_urlish(s: str) -> bool:
    s = s.lower()
    return bool(url_re.search(s) or "://" in s)

def char_entropy(s: str) -> float:
    if not s:
        return 0.0
    freq = Counter(s.lower())
    total = len(s)
    entropy = -sum((c/total) * math.log2(c/total) for c in freq.values() if c > 0)
    return min(entropy / 4.5, 1.0)

def has_mixed_script(s: str) -> bool:
    scripts = set()
    for ch in s:
        if ch.isalpha():
            try:
                script = unicodedata.name(ch, '').split()[0]
                scripts.add(script)
            except:
                pass
    return len(scripts) > 1

def has_repeated_pattern(s: str, min_repeat: int = 3) -> bool:
    s = trim_outer_punct(s)
    if len(s) < min_repeat * 2:
        return False
    for length in range(1, len(s) // 2 + 1):
        pattern = s[:length]
        if pattern * (len(s) // length) == s[:length * (len(s) // length)]:
            return True
    return False

def signatures(s: str) -> Set[str]:
    s = (s or "").strip()
    out = set()
    core = trim_outer_punct(s)
    
    bases = {s, core}
    if core != core.lower():
        bases.add(core.lower())
    if s != s.lower():
        bases.add(s.lower())
    
    for base in bases:
        if len(base) >= MIN_LEN:
            out.add(base)
        for n in range(1, EDGE + 1):
            if len(base) - n >= MIN_LEN:
                out.add(base[n:])
                out.add(base[:-n])
    return out

def level_bonus(level: str) -> float:
    return {"word": 1.0, "subword": 0.4}.get(level, 0.0)

# === Scoring functions ===

def clean_score(r: Dict) -> float:
    s = r["normalized"]
    score = 0.0
    
    if s and s[0].isalnum() and s[-1].isalnum():
        score += 0.35
    
    score += 0.20 * level_bonus(r["level"])
    score += 0.15 * (1.0 - punct_ratio(s))
    score += 0.10 * min(max(r["confidence"], 0.0), 1.0)
    score += 0.10 * min(max(r.get("utility_score", 0), 0.0) / 2.0, 1.0)
    score += 0.05 * min(max(r.get("trust_score", 0), 0.0), 1.0)
    score += 0.05 * min(math.log1p(max(r["frequency"], 0)) / 12.0, 1.0)
    
    if r.get("memory_type") == "core":
        score += 0.05
    
    if trim_outer_punct(s) == s and len(s) >= MIN_LEN:
        score += 0.10
    
    score += 0.05 * char_entropy(s)
    
    return min(score, 1.0)

def noise_score(r: Dict) -> float:
    s = r["normalized"]
    score = 0.0
    
    if trim_outer_punct(s) != s:
        score += 0.30
    
    if escaped_re.search(s):
        score += 0.30
    
    if is_urlish(s):
        score += 0.25
    
    if r["level"] == "subword":
        score += 0.10
    
    if len(trim_outer_punct(s)) < 6:
        score += 0.05
    
    if has_mixed_script(s):
        score += 0.15
    
    if has_repeated_pattern(s):
        score += 0.10
    
    return min(score, 1.0)

def _alnum_removed_fragment(shorter: str, longer: str, n: int) -> bool:
    if len(longer) - n < MIN_LEN:
        return False

    leading = longer[n:]
    trailing = longer[:-n]

    if leading == shorter:
        removed = longer[:n]
        return bool(removed) and all(ch.isalnum() for ch in removed)
    if trailing == shorter:
        removed = longer[-n:]
        return bool(removed) and all(ch.isalnum() for ch in removed)
    return False

def edge_trim_match(a: str, b: str) -> float:
    for n in range(1, EDGE + 1):
        if _alnum_removed_fragment(a, b, n):
            return 1.0
        if _alnum_removed_fragment(b, a, n):
            return 1.0
    return 0.0

def longest_shared_signature_len(a_sig: Set[str], b_sig: Set[str]) -> int:
    shared = a_sig & b_sig
    if not shared:
        return 0
    return max(len(x) for x in shared)

def relation_profile(u_norm: str, c_norm: str) -> Tuple[bool, float, List[str]]:
    u_core = trim_outer_punct(u_norm)
    c_core = trim_outer_punct(c_norm)
    reasons: List[str] = []
    strength = 0.0

    if not u_core or not c_core:
        return False, 0.0, reasons

    if u_core == c_core and u_norm != c_norm:
        reasons.append("same_core")
        strength = max(strength, 1.0)

    if edge_trim_match(u_core, c_core):
        reasons.append("edge_core")
        strength = max(strength, 0.95)

    shorter, longer = sorted((u_core, c_core), key=len)
    if (
        len(shorter) >= MIN_LEN
        and shorter in longer
        and len(shorter) / max(len(longer), 1) >= 0.78
    ):
        reasons.append("core_contains")
        strength = max(strength, 0.85)

    if not reasons:
        return False, 0.0, reasons

    if "same_core" not in reasons and abs(len(u_core) - len(c_core)) > EDGE + 1:
        return False, 0.0, []

    return True, strength, reasons

def compute_match_score(u_sig: Set[str], c_sig: Set[str], u_norm: str, c_norm: str, sig_freq: Optional[Counter] = None) -> float:
    shared = u_sig & c_sig
    if not shared:
        return 0.0
    
    if USE_RARE_SIG_WEIGHTING and sig_freq:
        shared_weighted = sum(1.0 / math.log1p(sig_freq.get(sig, 1)) for sig in shared)
        max_len = max(len(u_norm), len(c_norm), 1)
        return min(shared_weighted / max_len, 1.0)
    else:
        shared_len = max(len(x) for x in shared)
        return shared_len / max(len(u_norm), len(c_norm), 1)

# === Main execution ===

def main():
    print("score\tpolluted\tpolluted_norm\tlevel\tfreq\tcanonical\tcanonical_norm\tcanonical_level\tcanonical_freq\tpolluted_clean\tcanonical_clean\treasons", flush=True)
    
    conn = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    
    rows = [
        dict(r) for r in cur.execute("""
            SELECT id, content, normalized, level, frequency, utility_score,
                   confidence, salience_score, trust_score, memory_type
            FROM units
            WHERE level IN ('subword', 'word')
              AND length(normalized) >= ?
            ORDER BY frequency DESC
        """, (MIN_LEN,))
    ]
    
    for r in rows:
        r["sig"] = signatures(r["normalized"])
        r["clean"] = clean_score(r)
        r["noise"] = noise_score(r)
    
    bucket: Dict[str, List[int]] = defaultdict(list)
    for idx, r in enumerate(rows):
        for sig in r["sig"]:
            bucket[sig].append(idx)
    
    sig_freq = None
    if USE_RARE_SIG_WEIGHTING:
        sig_freq = Counter(sig for r in rows for sig in r["sig"])
    
    results = []
    
    for i, u in enumerate(rows):
        candidate_ids = set()
        for sig in u["sig"]:
            candidate_ids.update(bucket.get(sig, []))
        candidate_ids.discard(i)
        
        best = None
        best_score = 0.0
        
        for j in candidate_ids:
            c = rows[j]
            if u["normalized"] == c["normalized"]:
                continue

            related, relation_strength, relation_reasons = relation_profile(
                u["normalized"],
                c["normalized"],
            )
            if not related:
                continue
            
            if c["clean"] <= u["clean"] + QUALITY_MARGIN:
                continue
            
            match = compute_match_score(u["sig"], c["sig"], u["normalized"], c["normalized"], sig_freq)
            if match == 0:
                continue

            shared_len = longest_shared_signature_len(u["sig"], c["sig"])
            if shared_len < MIN_LEN:
                continue
            
            edge = edge_trim_match(u["normalized"], c["normalized"])
            
            pollution = (
                0.30 * match +
                0.25 * max(0.0, c["clean"] - u["clean"]) +
                0.20 * u["noise"] +
                0.25 * relation_strength
            )
            
            if pollution > best_score:
                best_score = pollution
                best = (c, relation_reasons)
        
        if best and best_score >= THRESHOLD:
            best_record, relation_reasons = best
            reasons = []
            if trim_outer_punct(u["normalized"]) != u["normalized"]:
                reasons.append("outer_punct")
            if escaped_re.search(u["normalized"]):
                reasons.append("escaped_unicode")
            if is_urlish(u["normalized"]):
                reasons.append("urlish")
            if edge_trim_match(u["normalized"], best_record["normalized"]):
                reasons.append("edge_trim")
            if u["level"] == "subword":
                reasons.append("subword")
            if has_mixed_script(u["normalized"]):
                reasons.append("mixed_script")
            if has_repeated_pattern(u["normalized"]):
                reasons.append("repeated_pattern")
            reasons.extend(relation_reasons)
            
            results.append({
                "score": round(best_score, 4),
                "polluted": u["content"],
                "polluted_norm": u["normalized"],
                "level": u["level"],
                "freq": u["frequency"],
                "canonical": best_record["content"],
                "canonical_norm": best_record["normalized"],
                "canonical_level": best_record["level"],
                "canonical_freq": best_record["frequency"],
                "polluted_clean": round(u["clean"], 4),
                "canonical_clean": round(best_record["clean"], 4),
                "reasons": ",".join(reasons) or "shared_signature",
            })
    
    results.sort(key=lambda x: (-x["score"], -x["freq"], x["polluted_norm"]))
    
    for row in results[:TOP_K]:
        print(
            f"{row['score']}\t{row['polluted']}\t{row['polluted_norm']}\t{row['level']}\t{row['freq']}\t"
            f"{row['canonical']}\t{row['canonical_norm']}\t{row['canonical_level']}\t{row['canonical_freq']}\t"
            f"{row['polluted_clean']}\t{row['canonical_clean']}\t{row['reasons']}",
            flush=True
        )
    
    conn.close()
    print(f"\n# Audit complete: {len(results)} candidates found, showing top {min(TOP_K, len(results))}", file=os.sys.stderr)

if __name__ == "__main__":
    try:
        main()
    except BrokenPipeError:
        try:
            sys.stdout.close()
        except Exception:
            pass
