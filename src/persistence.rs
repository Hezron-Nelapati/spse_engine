use crate::types::{
    CandidateStatus, MemoryChannel, MemoryType, SequenceState, TrainingJobStatus, Unit,
    UnitCandidate, UnitLevel,
};
use rusqlite::{params, Connection, Result as SqlResult, Row};
use serde::{Deserialize, Serialize};
use std::fs::{self, File};
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::Duration;
use uuid::Uuid;

const SNAPSHOT_SCHEMA_VERSION: u16 = 1;

#[derive(Debug, Serialize, Deserialize)]
struct SnapshotEnvelope {
    schema_version: u16,
    created_at: chrono::DateTime<chrono::Utc>,
    units: Vec<Unit>,
    sequence_state: SequenceState,
}

#[derive(Debug, Serialize, Deserialize)]
struct LegacySnapshot {
    units: Vec<Unit>,
    #[serde(default)]
    sequence_state: SequenceState,
}

pub struct Db {
    conn: Mutex<Connection>,
    snapshot_path: PathBuf,
    legacy_snapshot_path: PathBuf,
    jobs_path: PathBuf,
}

impl Db {
    pub fn new(path: impl AsRef<Path>) -> SqlResult<Self> {
        let path = path.as_ref().to_path_buf();
        let conn = Connection::open(&path)?;
        configure_connection(&conn)?;
        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS units (
                id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                normalized TEXT NOT NULL,
                level TEXT NOT NULL,
                frequency INTEGER NOT NULL,
                utility_score REAL NOT NULL,
                pos_x REAL NOT NULL,
                pos_y REAL NOT NULL,
                pos_z REAL NOT NULL,
                anchor_status INTEGER NOT NULL,
                memory_type TEXT NOT NULL,
                memory_channels_json TEXT NOT NULL DEFAULT '[\"main\"]',
                created_at TEXT NOT NULL,
                last_seen_at TEXT NOT NULL,
                salience_score REAL NOT NULL,
                confidence REAL NOT NULL,
                trust_score REAL NOT NULL,
                corroboration_count INTEGER NOT NULL,
                links_json TEXT NOT NULL,
                contexts_json TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS archived_units (
                id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                normalized TEXT NOT NULL,
                level TEXT NOT NULL,
                frequency INTEGER NOT NULL,
                utility_score REAL NOT NULL,
                pos_x REAL NOT NULL,
                pos_y REAL NOT NULL,
                pos_z REAL NOT NULL,
                anchor_status INTEGER NOT NULL,
                memory_type TEXT NOT NULL,
                memory_channels_json TEXT NOT NULL DEFAULT '[\"main\"]',
                created_at TEXT NOT NULL,
                last_seen_at TEXT NOT NULL,
                salience_score REAL NOT NULL,
                confidence REAL NOT NULL,
                trust_score REAL NOT NULL,
                corroboration_count INTEGER NOT NULL,
                links_json TEXT NOT NULL,
                contexts_json TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS growth_ledger (
                date TEXT PRIMARY KEY,
                growth_kb REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS unit_candidates (
                id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                normalized TEXT NOT NULL,
                level TEXT NOT NULL,
                observation_count INTEGER NOT NULL DEFAULT 1,
                utility_score REAL NOT NULL,
                status TEXT NOT NULL,
                first_seen_at TEXT NOT NULL,
                last_seen_at TEXT NOT NULL,
                promoted_at TEXT,
                memory_type TEXT NOT NULL,
                memory_channels_json TEXT NOT NULL DEFAULT '[\"main\"]'
            );
            CREATE TABLE IF NOT EXISTS pattern_combinations (
                parent_unit_id TEXT NOT NULL,
                child_unit_id TEXT NOT NULL,
                co_occurrence_count INTEGER NOT NULL DEFAULT 1,
                compression_gain REAL NOT NULL,
                PRIMARY KEY (parent_unit_id, child_unit_id)
            );
            CREATE INDEX IF NOT EXISTS idx_units_normalized ON units (normalized);
            CREATE INDEX IF NOT EXISTS idx_units_prune_cover
                ON units (anchor_status, memory_type, utility_score DESC, last_seen_at DESC, frequency DESC);
            CREATE INDEX IF NOT EXISTS idx_units_spatial_cover
                ON units (pos_x, pos_y, pos_z, anchor_status, memory_type);
            CREATE INDEX IF NOT EXISTS idx_archived_units_normalized ON archived_units (normalized);
            CREATE INDEX IF NOT EXISTS idx_candidates_by_count
                ON unit_candidates (observation_count DESC)
                WHERE status = 'candidate';
            CREATE INDEX IF NOT EXISTS idx_candidates_by_utility
                ON unit_candidates (utility_score DESC)
                WHERE status IN ('candidate', 'validated');
            CREATE INDEX IF NOT EXISTS idx_combinations_by_gain
                ON pattern_combinations (compression_gain DESC);",
        )?;
        ensure_column(
            &conn,
            "units",
            "memory_channels_json",
            "TEXT NOT NULL DEFAULT '[\"main\"]'",
        )?;
        ensure_column(
            &conn,
            "archived_units",
            "memory_channels_json",
            "TEXT NOT NULL DEFAULT '[\"main\"]'",
        )?;

        let parent = path.parent().unwrap_or_else(|| Path::new("."));
        Ok(Self {
            conn: Mutex::new(conn),
            snapshot_path: parent.join("memory_snapshot.msgpack"),
            legacy_snapshot_path: parent.join("memory_snapshot.json"),
            jobs_path: parent.join("training_jobs.json"),
        })
    }

    pub fn load_all_units(&self) -> SqlResult<Vec<Unit>> {
        let conn = self.conn.lock().expect("db mutex poisoned");
        let mut stmt = conn.prepare(
            "SELECT id, content, normalized, level, frequency, utility_score, pos_x, pos_y, pos_z,
                    anchor_status, memory_type, memory_channels_json, created_at, last_seen_at, salience_score, confidence,
                    trust_score, corroboration_count, links_json, contexts_json
             FROM units",
        )?;

        let rows = stmt.query_map([], hydrate_unit_from_row)?;

        Ok(rows.filter_map(Result::ok).collect())
    }

    pub fn load_units_batch(&self, offset: usize, limit: usize) -> SqlResult<Vec<Unit>> {
        let conn = self.conn.lock().expect("db mutex poisoned");
        let mut stmt = conn.prepare(
            "SELECT id, content, normalized, level, frequency, utility_score, pos_x, pos_y, pos_z,
                    anchor_status, memory_type, memory_channels_json, created_at, last_seen_at, salience_score, confidence,
                    trust_score, corroboration_count, links_json, contexts_json
             FROM units
             ORDER BY normalized, id
             LIMIT ?1 OFFSET ?2",
        )?;

        let rows = stmt.query_map(params![limit as i64, offset as i64], hydrate_unit_from_row)?;
        Ok(rows.filter_map(Result::ok).collect())
    }

    pub fn upsert_unit(&self, unit: &Unit) -> SqlResult<()> {
        let conn = self.conn.lock().expect("db mutex poisoned");
        upsert_unit_in_table(&conn, "units", unit)?;
        Ok(())
    }

    pub fn archive_units(&self, units: &[Unit]) -> SqlResult<()> {
        if units.is_empty() {
            return Ok(());
        }

        let mut conn = self.conn.lock().expect("db mutex poisoned");
        let tx = conn.transaction()?;
        for unit in units {
            upsert_unit_in_table(&tx, "archived_units", unit)?;
            tx.execute(
                "DELETE FROM units WHERE id = ?1",
                params![unit.id.to_string()],
            )?;
        }
        tx.commit()?;
        Ok(())
    }

    pub fn delete_units(&self, ids: &[String]) -> SqlResult<()> {
        if ids.is_empty() {
            return Ok(());
        }
        let conn = self.conn.lock().expect("db mutex poisoned");
        for id in ids {
            conn.execute("DELETE FROM units WHERE id = ?1", params![id])?;
        }
        Ok(())
    }

    pub fn load_archived_unit(&self, id: &str) -> SqlResult<Option<Unit>> {
        let conn = self.conn.lock().expect("db mutex poisoned");
        let mut stmt = conn.prepare(
            "SELECT id, content, normalized, level, frequency, utility_score, pos_x, pos_y, pos_z,
                    anchor_status, memory_type, memory_channels_json, created_at, last_seen_at, salience_score, confidence,
                    trust_score, corroboration_count, links_json, contexts_json
             FROM archived_units
             WHERE id = ?1",
        )?;
        let mut rows = stmt.query(params![id])?;
        if let Some(row) = rows.next()? {
            Ok(Some(hydrate_unit_from_row(row)?))
        } else {
            Ok(None)
        }
    }

    pub fn restore_archived_unit(&self, id: &str) -> SqlResult<Option<Unit>> {
        let mut conn = self.conn.lock().expect("db mutex poisoned");
        let tx = conn.transaction()?;
        let unit = {
            let mut stmt = tx.prepare(
                "SELECT id, content, normalized, level, frequency, utility_score, pos_x, pos_y, pos_z,
                        anchor_status, memory_type, memory_channels_json, created_at, last_seen_at, salience_score, confidence,
                        trust_score, corroboration_count, links_json, contexts_json
                 FROM archived_units
                 WHERE id = ?1",
            )?;
            let mut rows = stmt.query(params![id])?;
            if let Some(row) = rows.next()? {
                Some(hydrate_unit_from_row(row)?)
            } else {
                None
            }
        };

        if let Some(unit) = unit {
            upsert_unit_in_table(&tx, "units", &unit)?;
            tx.execute("DELETE FROM archived_units WHERE id = ?1", params![id])?;
            tx.commit()?;
            Ok(Some(unit))
        } else {
            tx.commit()?;
            Ok(None)
        }
    }

    pub fn load_snapshot(&self) -> io::Result<Option<(Vec<Unit>, SequenceState)>> {
        if self.snapshot_path.exists() {
            let bytes = fs::read(&self.snapshot_path)?;
            let snapshot: SnapshotEnvelope =
                rmp_serde::from_slice(&bytes).map_err(io::Error::other)?;
            if snapshot.schema_version == SNAPSHOT_SCHEMA_VERSION {
                return Ok(Some((snapshot.units, snapshot.sequence_state)));
            }
        }

        if self.legacy_snapshot_path.exists() {
            let raw = fs::read_to_string(&self.legacy_snapshot_path)?;
            let legacy = if raw.trim_start().starts_with('[') {
                LegacySnapshot {
                    units: serde_json::from_str(&raw).map_err(io::Error::other)?,
                    sequence_state: SequenceState::default(),
                }
            } else {
                serde_json::from_str::<LegacySnapshot>(&raw).map_err(io::Error::other)?
            };
            let migrated = (legacy.units.clone(), legacy.sequence_state.clone());
            let _ = self.save_snapshot(&legacy.units, &legacy.sequence_state);
            return Ok(Some(migrated));
        }

        Ok(None)
    }

    pub fn save_snapshot(
        &self,
        units: &[Unit],
        sequence_state: &SequenceState,
    ) -> std::io::Result<PathBuf> {
        let envelope = SnapshotEnvelope {
            schema_version: SNAPSHOT_SCHEMA_VERSION,
            created_at: chrono::Utc::now(),
            units: units.to_vec(),
            sequence_state: sequence_state.clone(),
        };
        let bytes = rmp_serde::to_vec_named(&envelope).map_err(io::Error::other)?;
        let mut file = File::create(&self.snapshot_path)?;
        file.write_all(&bytes)?;
        Ok(self.snapshot_path.clone())
    }

    pub fn save_training_jobs(&self, jobs: &[TrainingJobStatus]) -> std::io::Result<PathBuf> {
        let json = serde_json::to_string_pretty(jobs).unwrap_or_else(|_| "[]".to_string());
        let tmp_path = self.jobs_path.with_extension("json.tmp");
        let mut file = File::create(&tmp_path)?;
        file.write_all(json.as_bytes())?;
        file.sync_all()?;
        fs::rename(&tmp_path, &self.jobs_path)?;
        Ok(self.jobs_path.clone())
    }

    pub fn load_training_jobs(&self) -> std::io::Result<Vec<TrainingJobStatus>> {
        if !self.jobs_path.exists() {
            return Ok(Vec::new());
        }
        let raw = fs::read_to_string(&self.jobs_path)?;
        serde_json::from_str::<Vec<TrainingJobStatus>>(&raw).map_err(io::Error::other)
    }

    pub fn snapshot_path(&self) -> PathBuf {
        self.snapshot_path.clone()
    }

    pub fn record_growth(&self, date: &str, delta_kb: f32) -> SqlResult<f32> {
        let mut conn = self.conn.lock().expect("db mutex poisoned");
        let tx = conn.transaction()?;
        let existing = tx
            .query_row(
                "SELECT growth_kb FROM growth_ledger WHERE date = ?1",
                params![date],
                |row| row.get::<_, f32>(0),
            )
            .unwrap_or(0.0);
        let updated = existing + delta_kb.max(0.0);
        tx.execute(
            "INSERT OR REPLACE INTO growth_ledger (date, growth_kb) VALUES (?1, ?2)",
            params![date, updated],
        )?;
        tx.commit()?;
        Ok(updated)
    }

    pub fn growth_for_date(&self, date: &str) -> SqlResult<f32> {
        let conn = self.conn.lock().expect("db mutex poisoned");
        conn.query_row(
            "SELECT growth_kb FROM growth_ledger WHERE date = ?1",
            params![date],
            |row| row.get::<_, f32>(0),
        )
        .or_else(|err| match err {
            rusqlite::Error::QueryReturnedNoRows => Ok(0.0),
            _ => Err(err),
        })
    }

    pub fn load_all_candidates(&self) -> SqlResult<Vec<UnitCandidate>> {
        let conn = self.conn.lock().expect("db mutex poisoned");
        let mut stmt = conn.prepare(
            "SELECT id, content, normalized, level, observation_count, utility_score, status,
                    first_seen_at, last_seen_at, promoted_at, memory_type, memory_channels_json
             FROM unit_candidates",
        )?;
        let rows = stmt.query_map([], hydrate_candidate_from_row)?;
        Ok(rows.filter_map(Result::ok).collect())
    }

    pub fn load_candidates_batch(&self, offset: usize, limit: usize) -> SqlResult<Vec<UnitCandidate>> {
        let conn = self.conn.lock().expect("db mutex poisoned");
        let mut stmt = conn.prepare(
            "SELECT id, content, normalized, level, observation_count, utility_score, status,
                    first_seen_at, last_seen_at, promoted_at, memory_type, memory_channels_json
             FROM unit_candidates
             ORDER BY normalized, id
             LIMIT ?1 OFFSET ?2",
        )?;
        let rows =
            stmt.query_map(params![limit as i64, offset as i64], hydrate_candidate_from_row)?;
        Ok(rows.filter_map(Result::ok).collect())
    }

    pub fn upsert_candidate(&self, candidate: &UnitCandidate) -> SqlResult<()> {
        let conn = self.conn.lock().expect("db mutex poisoned");
        conn.execute(
            "INSERT INTO unit_candidates (
                id, content, normalized, level, observation_count, utility_score, status,
                first_seen_at, last_seen_at, promoted_at, memory_type, memory_channels_json
            ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12)
            ON CONFLICT(id) DO UPDATE SET
                content = excluded.content,
                normalized = excluded.normalized,
                level = excluded.level,
                observation_count = excluded.observation_count,
                utility_score = excluded.utility_score,
                status = excluded.status,
                first_seen_at = excluded.first_seen_at,
                last_seen_at = excluded.last_seen_at,
                promoted_at = excluded.promoted_at,
                memory_type = excluded.memory_type,
                memory_channels_json = excluded.memory_channels_json",
            params![
                candidate.id.to_string(),
                candidate.content,
                candidate.normalized,
                level_to_str(candidate.level),
                candidate.observation_count as i64,
                candidate.utility_score,
                candidate_status_to_str(candidate.status),
                candidate.first_seen_at.to_rfc3339(),
                candidate.last_seen_at.to_rfc3339(),
                candidate.promoted_at.map(|value| value.to_rfc3339()),
                memory_type_to_str(candidate.memory_type),
                memory_channels_to_json(&candidate.memory_channels),
            ],
        )?;
        Ok(())
    }

    pub fn delete_candidate(&self, id: &Uuid) -> SqlResult<()> {
        let conn = self.conn.lock().expect("db mutex poisoned");
        conn.execute(
            "DELETE FROM unit_candidates WHERE id = ?1",
            params![id.to_string()],
        )?;
        Ok(())
    }

    pub fn upsert_pattern_combination(
        &self,
        parent_unit_id: Uuid,
        child_unit_id: Uuid,
        compression_gain: f32,
    ) -> SqlResult<bool> {
        let conn = self.conn.lock().expect("db mutex poisoned");
        let existed = conn
            .query_row(
                "SELECT 1 FROM pattern_combinations
                 WHERE parent_unit_id = ?1 AND child_unit_id = ?2",
                params![parent_unit_id.to_string(), child_unit_id.to_string()],
                |_| Ok(()),
            )
            .is_ok();
        conn.execute(
            "INSERT INTO pattern_combinations (
                parent_unit_id, child_unit_id, co_occurrence_count, compression_gain
            ) VALUES (?1, ?2, 1, ?3)
            ON CONFLICT(parent_unit_id, child_unit_id) DO UPDATE SET
                co_occurrence_count = co_occurrence_count + 1,
                compression_gain = MAX(pattern_combinations.compression_gain, excluded.compression_gain)",
            params![
                parent_unit_id.to_string(),
                child_unit_id.to_string(),
                compression_gain,
            ],
        )?;
        Ok(existed)
    }

    pub fn wal_size_mb(&self) -> f32 {
        let wal_path = self.conn_path().with_extension(
            self.conn_path()
                .extension()
                .map(|ext| format!("{}-wal", ext.to_string_lossy()))
                .unwrap_or_else(|| "wal".to_string()),
        );
        fs::metadata(wal_path)
            .map(|meta| meta.len() as f32 / (1024.0 * 1024.0))
            .unwrap_or(0.0)
    }

    pub fn index_fragmentation(&self) -> f32 {
        let conn = self.conn.lock().expect("db mutex poisoned");
        let page_count = conn
            .pragma_query_value(None, "page_count", |row| row.get::<_, i64>(0))
            .unwrap_or(0)
            .max(1);
        let freelist_count = conn
            .pragma_query_value(None, "freelist_count", |row| row.get::<_, i64>(0))
            .unwrap_or(0)
            .max(0);
        (freelist_count as f32 / page_count as f32).clamp(0.0, 1.0)
    }

    fn conn_path(&self) -> PathBuf {
        let conn = self.conn.lock().expect("db mutex poisoned");
        conn.path()
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from("spse_memory.db"))
    }
}

fn hydrate_unit_from_row(row: &Row<'_>) -> SqlResult<Unit> {
    let id_text: String = row.get(0)?;
    let links_json: String = row.get(18)?;
    let contexts_json: String = row.get(19)?;
    let created_at_text: String = row.get(12)?;
    let last_seen_text: String = row.get(13)?;
    let memory_channels_json: String = row.get(11)?;

    Ok(Unit {
        id: uuid::Uuid::parse_str(&id_text).unwrap_or_else(|_| uuid::Uuid::nil()),
        content: row.get(1)?,
        normalized: row.get(2)?,
        level: level_from_str(&row.get::<_, String>(3)?),
        frequency: row.get::<_, i64>(4)? as u64,
        utility_score: row.get(5)?,
        semantic_position: [row.get(6)?, row.get(7)?, row.get(8)?],
        anchor_status: row.get::<_, i64>(9)? != 0,
        memory_type: memory_type_from_str(&row.get::<_, String>(10)?),
        memory_channels: memory_channels_from_json(&memory_channels_json),
        created_at: chrono::DateTime::parse_from_rfc3339(&created_at_text)
            .map(|dt| dt.with_timezone(&chrono::Utc))
            .unwrap_or_else(|_| chrono::Utc::now()),
        last_seen_at: chrono::DateTime::parse_from_rfc3339(&last_seen_text)
            .map(|dt| dt.with_timezone(&chrono::Utc))
            .unwrap_or_else(|_| chrono::Utc::now()),
        salience_score: row.get(14)?,
        confidence: row.get(15)?,
        trust_score: row.get(16)?,
        corroboration_count: row.get::<_, i64>(17)? as u32,
        links: serde_json::from_str(&links_json).unwrap_or_default(),
        contexts: serde_json::from_str(&contexts_json).unwrap_or_default(),
    })
}

fn hydrate_candidate_from_row(row: &Row<'_>) -> SqlResult<UnitCandidate> {
    let id_text: String = row.get(0)?;
    let promoted_at_text: Option<String> = row.get(9)?;
    let channels_json: String = row.get(11)?;
    Ok(UnitCandidate {
        id: Uuid::parse_str(&id_text).unwrap_or_else(|_| Uuid::nil()),
        content: row.get(1)?,
        normalized: row.get(2)?,
        level: level_from_str(&row.get::<_, String>(3)?),
        observation_count: row.get::<_, i64>(4)?.max(0) as u64,
        utility_score: row.get(5)?,
        status: candidate_status_from_str(&row.get::<_, String>(6)?),
        first_seen_at: chrono::DateTime::parse_from_rfc3339(&row.get::<_, String>(7)?)
            .map(|value| value.with_timezone(&chrono::Utc))
            .unwrap_or_else(|_| chrono::Utc::now()),
        last_seen_at: chrono::DateTime::parse_from_rfc3339(&row.get::<_, String>(8)?)
            .map(|value| value.with_timezone(&chrono::Utc))
            .unwrap_or_else(|_| chrono::Utc::now()),
        promoted_at: promoted_at_text.and_then(|value| {
            chrono::DateTime::parse_from_rfc3339(&value)
                .ok()
                .map(|value| value.with_timezone(&chrono::Utc))
        }),
        memory_type: memory_type_from_str(&row.get::<_, String>(10)?),
        memory_channels: memory_channels_from_json(&channels_json),
    })
}

fn upsert_unit_in_table(conn: &Connection, table: &str, unit: &Unit) -> SqlResult<()> {
    let statement = format!(
        "INSERT OR REPLACE INTO {table} (
            id, content, normalized, level, frequency, utility_score, pos_x, pos_y, pos_z,
            anchor_status, memory_type, memory_channels_json, created_at, last_seen_at, salience_score, confidence,
            trust_score, corroboration_count, links_json, contexts_json
         ) VALUES (
            ?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15, ?16, ?17, ?18, ?19, ?20
         )"
    );
    conn.execute(
        &statement,
        params![
            unit.id.to_string(),
            unit.content,
            unit.normalized,
            level_to_str(unit.level),
            unit.frequency as i64,
            unit.utility_score,
            unit.semantic_position[0],
            unit.semantic_position[1],
            unit.semantic_position[2],
            i64::from(unit.anchor_status),
            memory_type_to_str(unit.memory_type),
            memory_channels_to_json(&unit.memory_channels),
            unit.created_at.to_rfc3339(),
            unit.last_seen_at.to_rfc3339(),
            unit.salience_score,
            unit.confidence,
            unit.trust_score,
            unit.corroboration_count as i64,
            serde_json::to_string(&unit.links).unwrap_or_else(|_| "[]".to_string()),
            serde_json::to_string(&unit.contexts).unwrap_or_else(|_| "[]".to_string()),
        ],
    )?;
    Ok(())
}

fn ensure_column(conn: &Connection, table: &str, column: &str, definition: &str) -> SqlResult<()> {
    let pragma = format!("PRAGMA table_info({table})");
    let mut stmt = conn.prepare(&pragma)?;
    let existing = stmt
        .query_map([], |row| row.get::<_, String>(1))?
        .filter_map(Result::ok)
        .any(|name| name == column);
    if existing {
        return Ok(());
    }

    let alter = format!("ALTER TABLE {table} ADD COLUMN {column} {definition}");
    conn.execute(&alter, [])?;
    Ok(())
}

fn configure_connection(conn: &Connection) -> SqlResult<()> {
    conn.pragma_update(None, "journal_mode", "WAL")?;
    conn.pragma_update(None, "synchronous", "NORMAL")?;
    conn.pragma_update(None, "temp_store", "MEMORY")?;
    conn.pragma_update(None, "foreign_keys", "ON")?;
    conn.pragma_update(None, "wal_autocheckpoint", 1000)?;
    conn.busy_timeout(Duration::from_secs(5))?;
    Ok(())
}

fn level_to_str(level: UnitLevel) -> &'static str {
    match level {
        UnitLevel::Char => "char",
        UnitLevel::Subword => "subword",
        UnitLevel::Word => "word",
        UnitLevel::Phrase => "phrase",
        UnitLevel::Pattern => "pattern",
    }
}

fn level_from_str(value: &str) -> UnitLevel {
    match value {
        "char" => UnitLevel::Char,
        "subword" => UnitLevel::Subword,
        "phrase" => UnitLevel::Phrase,
        "pattern" => UnitLevel::Pattern,
        _ => UnitLevel::Word,
    }
}

fn memory_type_to_str(memory_type: MemoryType) -> &'static str {
    match memory_type {
        MemoryType::Episodic => "episodic",
        MemoryType::Core => "core",
    }
}

fn memory_type_from_str(value: &str) -> MemoryType {
    match value {
        "core" => MemoryType::Core,
        _ => MemoryType::Episodic,
    }
}

fn memory_channels_to_json(channels: &[MemoryChannel]) -> String {
    let normalized = if channels.is_empty() {
        vec![MemoryChannel::Main]
    } else {
        channels.to_vec()
    };
    serde_json::to_string(&normalized).unwrap_or_else(|_| "[\"main\"]".to_string())
}

fn memory_channels_from_json(value: &str) -> Vec<MemoryChannel> {
    let mut channels = serde_json::from_str::<Vec<MemoryChannel>>(value).unwrap_or_default();
    if channels.is_empty() {
        channels.push(MemoryChannel::Main);
    }
    if !channels.contains(&MemoryChannel::Main) {
        channels.push(MemoryChannel::Main);
    }
    channels
}

fn candidate_status_to_str(status: CandidateStatus) -> &'static str {
    match status {
        CandidateStatus::Candidate => "candidate",
        CandidateStatus::Validated => "validated",
        CandidateStatus::Active => "active",
        CandidateStatus::Rejected => "rejected",
    }
}

fn candidate_status_from_str(value: &str) -> CandidateStatus {
    match value {
        "validated" => CandidateStatus::Validated,
        "active" => CandidateStatus::Active,
        "rejected" => CandidateStatus::Rejected,
        _ => CandidateStatus::Candidate,
    }
}
