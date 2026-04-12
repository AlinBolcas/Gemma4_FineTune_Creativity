"""
memory.py — Memory system with SQLite backend, cognitive scoping, and GDrive sync.

Architecture:
  ┌────────────────────────────────────────────────────────┐
  │ memory.db                                              │
  │ ├── short_term     chat turns (volatile, auto-trimmed) │
  │ ├── long_term      structured insights by scope:       │
  │ │   ├── identity       AI personality/tone/rules       │
  │ │   ├── self_awareness architecture/capabilities       │
  │ │   ├── facts          objective truths, data           │
  │ │   ├── preferences    subjective choices, style        │
  │ │   ├── procedures     how-to, workflows, methods       │
  │ │   ├── relationships  people, connections, roles       │
  │ │   ├── decisions      choices made, rationale          │
  │ │   └── context        current projects, goals          │
  │ └── journal        AI workspace/scratchpad (notes)     │
  └────────────────────────────────────────────────────────┘

Key features:
  - Flush-before-compaction (extracts insights before trimming STM)
  - Scoring-based extraction (replaces old binary Yes/No gate)
  - Cognitive scopes modeled on human memory categorization
  - Identity preserved across clear() (foundational knowledge)
  - GDrive sync for backup/access

Public API (backward-compatible):
  save_short_term, retrieve_short_term_formatted, retrieve_long_term,
  extract_to_long_term, clear, save, load, get_formatted_context
"""

import os
import json
import sys
import re
import sqlite3
import threading
from typing import List, Dict, Optional, Any
from datetime import datetime
from pathlib import Path
import logging

# Configure logging
logger = logging.getLogger(__name__)
if not logger.hasHandlers():
    logger.addHandler(logging.StreamHandler())
    logger.setLevel(logging.INFO)

# --- LLM Provider Configuration ---
LLM_PROVIDER = "openai"

# Import Utils using dynamic import system
import importlib.util

def _import_utils():
    """Dynamically import Utils by finding utils.py in the project."""
    try:
        from src.VI_utils.utils import Utils
        return Utils
    except ImportError:
        pass
    current_file = Path(__file__).resolve()
    project_root = current_file.parent
    while not (project_root / "main.py").exists() and not (project_root / ".git").exists():
        if project_root.parent == project_root:
            break
        project_root = project_root.parent
    utils_path = project_root / "src" / "VI_utils" / "utils.py"
    if utils_path.exists():
        spec = importlib.util.spec_from_file_location("utils", str(utils_path))
        if spec and spec.loader:
            utils_module = importlib.util.module_from_spec(spec)
            sys.modules["utils"] = utils_module
            spec.loader.exec_module(utils_module)
            return utils_module.Utils
    raise ImportError(f"Could not find or import Utils from {utils_path}")

Utils = _import_utils()

# Load the chosen LLM API wrapper using Utils
LLMAPIWrapper = None
try:
    if LLM_PROVIDER == "ollama":
        ollama_module = Utils.import_file("ollama_API.py")
        LLMAPIWrapper = ollama_module.OllamaWrapper
    elif LLM_PROVIDER == "openai":
        openai_module = Utils.import_file("openai_API.py")
        LLMAPIWrapper = openai_module.OpenAIWrapper
except (ImportError, Exception) as e:
    logger.warning(f"Could not import {LLM_PROVIDER.upper()} API for Memory: {e}")
    LLMAPIWrapper = None


class Memory:
    """
    Memory system with SQLite backend, cognitive scoping, and structured LTM.
    Flush-before-compaction ensures no silent memory loss.
    Scoring-based extraction captures more insights than a binary gate.
    """

    # Cognitive scopes — modeled on how the human mind categorizes knowledge
    COGNITIVE_SCOPES = {
        "facts":         "Objective truths, definitions, data, knowledge about the world",
        "preferences":   "Subjective choices, opinions, taste profile, style, aesthetics",
        "procedures":    "How-to knowledge, workflows, methods, techniques, processes",
        "relationships": "People, connections, social/professional context, roles",
        "decisions":     "Choices made, rationale, trade-offs, commitments",
        "context":       "Current projects, goals, situations, temporal info",
        "goals":         "Active objectives, milestones, what the user is building",
        "stack":         "Technical stack: languages, infra, preferred libs, tools, models",
        "experiments":   "Ongoing experiments, what was tried, what failed/worked, learnings",
    }

    # Special scopes — managed deliberately, not by organic extraction
    SPECIAL_SCOPES = {
        "identity":       "AI personality, tone, rules, behavioral guidelines",
        "self_awareness": "Understanding of own architecture, capabilities, codebase",
    }

    # ─── Initialization ──────────────────────────────────────────────

    def __init__(
        self,
        llm_api: Optional[Any] = None,
        short_term_limit: int = 25000,
        long_term_interval: int = 3,
        memory_dir: Optional[str] = None,
        agent_name: Optional[str] = None,
    ):
        """
        Initialize the memory system.

        Args:
            llm_api: LLM API wrapper instance (OpenAIWrapper or OllamaWrapper)
            short_term_limit: Max character count for short-term memory
            long_term_interval: Interactions between automatic LTM extraction
            memory_dir: Custom directory for storing memory files
            agent_name: Agent name for memory isolation (separate DB per agent)
        """
        # Initialize LLM API
        if llm_api is None:
            if LLMAPIWrapper is not None:
                try:
                    self.llm_api = LLMAPIWrapper(auto_pull=True) if LLM_PROVIDER == "ollama" else LLMAPIWrapper()
                except Exception as e:
                    raise ImportError(f"Failed to auto-initialize {LLM_PROVIDER} API: {e}") from e
            else:
                raise ImportError(f"{LLM_PROVIDER.upper()} API required but not available.")
        else:
            self.llm_api = llm_api

        self.agent_name = agent_name

        # Memory directory setup
        if memory_dir:
            self.memory_dir = Path(memory_dir)
        else:
            project_root = Path(__file__).parent.parent.parent
            self.memory_dir = project_root / "data" / "output" / "memory"
        self.memory_dir.mkdir(parents=True, exist_ok=True)

        # Agent-specific subfolder
        if self.agent_name:
            self.agent_memory_dir = self.memory_dir / self.agent_name
            self.agent_memory_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.agent_memory_dir = self.memory_dir

        # File paths
        self.db_path = self.agent_memory_dir / "memory.db"
        self.short_term_markdown = self.agent_memory_dir / "short_term_memory.md"
        self.long_term_markdown = self.agent_memory_dir / "long_term_memory.md"
        self.journal_markdown = self.agent_memory_dir / "journal.md"
        self.ops_board_path = self.agent_memory_dir / "ops_board.json"

        # Config
        self.short_term_limit = short_term_limit
        self.long_term_interval = long_term_interval
        self._short_term_counter = 0
        self._db_lock = threading.Lock()
        self._last_system_prompt: Optional[str] = None  # Dedup: skip storing if unchanged

        # Initialize DB, migrate old JSON, cleanup expired
        self._init_db()
        self._migrate_from_json()
        self._cleanup_expired()
        
        # Auto-seed smart defaults on first init
        self._seed_defaults()

        logger.info(
            f"Memory initialized (SQLite)"
            f"{' for agent ' + self.agent_name if self.agent_name else ''}. "
            f"STM limit: {short_term_limit}, LTM interval: {long_term_interval}."
        )

    # ─── SQLite internals ────────────────────────────────────────────

    def _get_conn(self) -> sqlite3.Connection:
        """Get a SQLite connection (thread-safe)."""
        return sqlite3.connect(str(self.db_path), check_same_thread=False)

    def _init_db(self):
        """Create tables if they don't exist."""
        with self._db_lock, self._get_conn() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS short_term (
                    id            INTEGER PRIMARY KEY AUTOINCREMENT,
                    system_msg    TEXT,
                    user_msg      TEXT,
                    assistant_msg TEXT,
                    kind          TEXT DEFAULT 'chat',
                    created_at    TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS long_term (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    content     TEXT NOT NULL,
                    scope       TEXT DEFAULT 'facts',
                    tags        TEXT DEFAULT '',
                    importance  REAL DEFAULT 0.5,
                    created_at  TEXT NOT NULL,
                    expires_at  TEXT,
                    source      TEXT DEFAULT ''
                );

                CREATE TABLE IF NOT EXISTS journal (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    content     TEXT NOT NULL,
                    tags        TEXT DEFAULT '',
                    created_at  TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS identity_versions (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    version     INTEGER NOT NULL,
                    content     TEXT NOT NULL,
                    reason      TEXT DEFAULT '',
                    created_at  TEXT NOT NULL,
                    is_active   INTEGER DEFAULT 1
                );
            """)

    def _migrate_from_json(self):
        """One-time migration from old JSON files to SQLite (non-destructive)."""
        stm_json = self.agent_memory_dir / "short_term_memory.json"
        ltm_json = self.agent_memory_dir / "long_term_memory.json"
        migrated = False

        with self._db_lock, self._get_conn() as conn:
            # Migrate short-term memory
            if stm_json.exists():
                try:
                    with open(stm_json, "r", encoding="utf-8") as f:
                        stm_data = json.load(f)
                    count = conn.execute("SELECT COUNT(*) FROM short_term").fetchone()[0]
                    if count == 0 and stm_data:
                        for entry in stm_data:
                            conn.execute(
                                "INSERT INTO short_term (system_msg, user_msg, assistant_msg, kind, created_at) "
                                "VALUES (?,?,?,?,?)",
                                (entry.get("system", ""), entry.get("user", ""),
                                 entry.get("assistant", ""), entry.get("kind", "chat"),
                                 entry.get("timestamp", datetime.now().isoformat())),
                            )
                        migrated = True
                        logger.info(f"Migrated {len(stm_data)} STM entries from JSON → SQLite.")
                    backup = stm_json.with_suffix(".json.bak")
                    if not backup.exists():
                        stm_json.rename(backup)
                except Exception as e:
                    logger.warning(f"STM JSON migration failed: {e}")

            # Migrate long-term memory
            if ltm_json.exists():
                try:
                    with open(ltm_json, "r", encoding="utf-8") as f:
                        ltm_data = json.load(f)
                    count = conn.execute("SELECT COUNT(*) FROM long_term").fetchone()[0]
                    if count == 0 and ltm_data:
                        for entry in ltm_data:
                            conn.execute(
                                "INSERT INTO long_term (content, scope, tags, importance, created_at) "
                                "VALUES (?,?,?,?,?)",
                                (entry.get("insight", ""), "facts", "", 0.5,
                                 entry.get("timestamp", datetime.now().isoformat())),
                            )
                        migrated = True
                        logger.info(f"Migrated {len(ltm_data)} LTM entries from JSON → SQLite.")
                    backup = ltm_json.with_suffix(".json.bak")
                    if not backup.exists():
                        ltm_json.rename(backup)
                except Exception as e:
                    logger.warning(f"LTM JSON migration failed: {e}")

        if migrated:
            self._export_markdown()

    def _cleanup_expired(self):
        """Remove expired long-term memories (TTL enforcement)."""
        try:
            with self._db_lock, self._get_conn() as conn:
                deleted = conn.execute(
                    "DELETE FROM long_term WHERE expires_at IS NOT NULL AND expires_at < ?",
                    (datetime.now().isoformat(),),
                ).rowcount
                if deleted:
                    logger.info(f"Cleaned up {deleted} expired LTM entries.")
        except Exception as e:
            logger.warning(f"TTL cleanup failed: {e}")

    def _seed_defaults(self):
        """Auto-seed smart defaults for identity and self-awareness on first init."""
        try:
            with self._db_lock, self._get_conn() as conn:
                # Check if identity already exists
                identity_exists = conn.execute(
                    "SELECT COUNT(*) FROM long_term WHERE scope = 'identity'"
                ).fetchone()[0] > 0
                
                # Check if self_awareness already exists
                self_awareness_exists = conn.execute(
                    "SELECT COUNT(*) FROM long_term WHERE scope = 'self_awareness'"
                ).fetchone()[0] > 0
            
            # Load defaults from config files
            config_dir = Path(__file__).parent.parent.parent / "data" / "input" / "system_config"
            identity_file = config_dir / "init_identity.md"
            self_awareness_file = config_dir / "init_self_awareness.md"
            
            # Seed identity if missing
            if not identity_exists:
                if identity_file.exists():
                    with open(identity_file, "r", encoding="utf-8") as f:
                        default_identity = f.read().strip()
                    logger.info(f"Loading identity from {identity_file}")
                else:
                    # Fallback if file missing
                    default_identity = (
                        "I am ArX, a creative AI assistant by Arvolve. "
                        "I speak concisely and directly. I have opinions. "
                        "I balance technical precision with artistic insight."
                    )
                    logger.warning(f"Identity file not found, using fallback")
                
                self.update_identity(default_identity)
                logger.info("Auto-seeded default identity")
            
            # Seed self-awareness if missing
            if not self_awareness_exists:
                if self_awareness_file.exists():
                    with open(self_awareness_file, "r", encoding="utf-8") as f:
                        default_self_awareness = f.read().strip()
                    logger.info(f"Loading self-awareness from {self_awareness_file}")
                else:
                    # Fallback if file missing
                    default_self_awareness = (
                        "ArX runs on a modular Python codebase with TextGen, Memory, "
                        "RAG, Tools, and AgentGen components. I can invoke tools, edit "
                        "my identity, and learn from conversations."
                    )
                    logger.warning(f"Self-awareness file not found, using fallback")
                
                self.seed_self_awareness(default_self_awareness)
                logger.info("Auto-seeded default self-awareness")
                
        except Exception as e:
            logger.warning(f"Default seeding failed (non-critical): {e}")

    # ─── Public API (backward-compatible) ────────────────────────────

    def save_short_term(
        self, system_prompt: str, user_prompt: str,
        assistant_response: str, kind: str = "chat",
    ):
        """Save an interaction to short-term memory.
        System prompt is only stored when it changes (dedup to save token budget).
        """
        now = datetime.now().isoformat()
        
        # Dedup: only store system prompt if it changed from last interaction
        sys_to_store = None
        if system_prompt and system_prompt != self._last_system_prompt:
            sys_to_store = system_prompt
            self._last_system_prompt = system_prompt
        
        with self._db_lock, self._get_conn() as conn:
            conn.execute(
                "INSERT INTO short_term (system_msg, user_msg, assistant_msg, kind, created_at) "
                "VALUES (?,?,?,?,?)",
                (sys_to_store, user_prompt, assistant_response, kind or "chat", now),
            )

        # Trim if over limit (flush-before-compaction extracts insights first)
        trimmed = self._trim_short_term_memory()
        if trimmed > 0:
            logger.info(f"Trimmed {trimmed} oldest entries (insights extracted first).")

        # Export markdown
        self._export_markdown()

        # Periodic LTM extraction
        self._short_term_counter += 1
        if self._short_term_counter >= self.long_term_interval:
            logger.info(f"Reached LTM interval ({self.long_term_interval}), extracting insights.")
            self.extract_to_long_term()
            self._short_term_counter = 0

    def retrieve_short_term_formatted(
        self, include_system: bool = True, kinds: Optional[List[str]] = None,
    ) -> List[Dict[str, str]]:
        """Retrieve STM formatted for LLM API calls (OpenAI message format)."""
        with self._db_lock, self._get_conn() as conn:
            if kinds:
                placeholders = ",".join("?" for _ in kinds)
                rows = conn.execute(
                    f"SELECT system_msg, user_msg, assistant_msg FROM short_term "
                    f"WHERE LOWER(TRIM(kind)) IN ({placeholders}) ORDER BY id ASC",
                    [k.strip().lower() for k in kinds],
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT system_msg, user_msg, assistant_msg FROM short_term ORDER BY id ASC"
                ).fetchall()

        formatted = []
        for system_msg, user_msg, assistant_msg in rows:
            if include_system and system_msg:
                formatted.append({"role": "system", "content": system_msg})
            if user_msg:
                formatted.append({"role": "user", "content": user_msg})
            if assistant_msg:
                formatted.append({"role": "assistant", "content": assistant_msg})
        return formatted

    def retrieve_long_term(self) -> List[str]:
        """
        Retrieve organic LTM insights as strings (backward-compatible).
        Excludes identity/self_awareness scopes (those are injected separately).
        """
        with self._db_lock, self._get_conn() as conn:
            rows = conn.execute(
                "SELECT content FROM long_term "
                "WHERE scope NOT IN ('identity', 'self_awareness') "
                "AND (expires_at IS NULL OR expires_at > ?) "
                "ORDER BY importance DESC, created_at DESC",
                (datetime.now().isoformat(),),
            ).fetchall()
        return [row[0] for row in rows if row[0]]

    def retrieve_long_term_detailed(self) -> List[Dict]:
        """Retrieve organic LTM insights with full structured metadata."""
        with self._db_lock, self._get_conn() as conn:
            rows = conn.execute(
                "SELECT id, content, scope, tags, importance, created_at, expires_at, source "
                "FROM long_term WHERE scope NOT IN ('identity', 'self_awareness') "
                "AND (expires_at IS NULL OR expires_at > ?) "
                "ORDER BY importance DESC, created_at DESC",
                (datetime.now().isoformat(),),
            ).fetchall()
        return [
            {"id": r[0], "content": r[1], "scope": r[2], "tags": r[3],
             "importance": r[4], "created_at": r[5], "expires_at": r[6], "source": r[7]}
            for r in rows
        ]

    def retrieve_by_scope(self, scope: str, limit: int = 50) -> List[str]:
        """Retrieve LTM entries filtered by a specific scope."""
        with self._db_lock, self._get_conn() as conn:
            rows = conn.execute(
                "SELECT content FROM long_term WHERE scope = ? "
                "AND (expires_at IS NULL OR expires_at > ?) "
                "ORDER BY importance DESC, created_at DESC LIMIT ?",
                (scope, datetime.now().isoformat(), limit),
            ).fetchall()
        return [row[0] for row in rows if row[0]]

    def extract_to_long_term(self, num_interactions: Optional[int] = None, force_insight: Optional[str] = None):
        """
        Extract insights from recent STM into structured LTM.

        Args:
            num_interactions: Number of recent turns to analyze
            force_insight: Save this string directly as an insight (bypasses LLM)
        """
        if force_insight:
            self._save_long_term_insight(content=force_insight)
            logger.info(f"Saved forced insight: '{force_insight[:50]}...'")
            return

        if num_interactions is None:
            num_interactions = self.long_term_interval

        with self._db_lock, self._get_conn() as conn:
            rows = conn.execute(
                "SELECT id, user_msg, assistant_msg FROM short_term ORDER BY id DESC LIMIT ?",
                (num_interactions,),
            ).fetchall()

        if not rows:
            logger.info("No recent memories for extraction.")
            return

        entries = [{"id": r[0], "user": r[1], "assistant": r[2]} for r in reversed(rows)]
        self._extract_insights_scored(entries, source="periodic")

    def clear(self, full_reset: bool = False):
        """
        Clear memory. Identity and self_awareness are preserved by default.
        Use full_reset=True to wipe everything including foundational knowledge.
        """
        with self._db_lock, self._get_conn() as conn:
            conn.execute("DELETE FROM short_term")
            conn.execute("DELETE FROM journal")
            if full_reset:
                conn.execute("DELETE FROM long_term")
            else:
                conn.execute("DELETE FROM long_term WHERE scope NOT IN ('identity', 'self_awareness')")

        self._short_term_counter = 0

        for f in [self.short_term_markdown, self.long_term_markdown, self.journal_markdown]:
            if f.exists():
                try:
                    f.unlink()
                except OSError:
                    pass

        logger.info(f"Memory cleared{' (full reset)' if full_reset else ' (identity preserved)'}.")

    def save(self, filename: Optional[str] = None) -> str:
        """Sync memory exports (markdown). Returns memory directory path."""
        self._export_markdown()
        logger.info(f"Memory synced to: {self.memory_dir}")
        return str(self.memory_dir)

    def load(self, filename: str):
        """Load memory from a JSON file (backward-compatible format)."""
        if not filename.lower().endswith(".json"):
            filename += ".json"

        load_path = self.memory_dir / filename
        if not load_path.exists():
            raise FileNotFoundError(f"Memory file not found: {load_path}")

        with open(load_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.clear()

        with self._db_lock, self._get_conn() as conn:
            if "short_term" in data and isinstance(data["short_term"], list):
                msgs = data["short_term"]
                for i in range(0, len(msgs), 3):
                    if i + 2 < len(msgs):
                        conn.execute(
                            "INSERT INTO short_term (system_msg, user_msg, assistant_msg, kind, created_at) "
                            "VALUES (?,?,?,?,?)",
                            (msgs[i].get("content", ""), msgs[i + 1].get("content", ""),
                             msgs[i + 2].get("content", ""), "chat", datetime.now().isoformat()),
                        )

            if "long_term" in data and isinstance(data["long_term"], list):
                for item in data["long_term"]:
                    content = item if isinstance(item, str) else item.get("content", str(item))
                    conn.execute(
                        "INSERT INTO long_term (content, scope, tags, importance, created_at) "
                        "VALUES (?,?,?,?,?)",
                        (content, "facts", "", 0.5, datetime.now().isoformat()),
                    )

        self._export_markdown()
        logger.info(f"Memory loaded from: {load_path}")
        return True

    def get_formatted_context(
        self, include_short_term: bool = True,
        include_long_term: bool = True,
        max_short_term_entries: Optional[int] = None,
    ) -> str:
        """Get a formatted context string combining STM and LTM."""
        parts = []

        if include_long_term:
            insights = self.retrieve_long_term()
            if insights:
                parts.append("# LONG-TERM MEMORY (Key Insights)")
                for i, entry in enumerate(insights, 1):
                    parts.append(f"{i}. {entry}")
                parts.append("")

        if include_short_term:
            with self._db_lock, self._get_conn() as conn:
                rows = conn.execute(
                    "SELECT system_msg, user_msg, assistant_msg FROM short_term ORDER BY id ASC"
                ).fetchall()
            if rows:
                if max_short_term_entries and max_short_term_entries > 0:
                    rows = rows[-max_short_term_entries:]
                parts.append("# RECENT CONVERSATION HISTORY")
                for i, (sys_msg, usr_msg, ast_msg) in enumerate(rows, 1):
                    parts.append(f"## Interaction {i}")
                    if sys_msg:
                        parts.append(f"System: {sys_msg}")
                    parts.append(f"User: {usr_msg}")
                    parts.append(f"Assistant: {ast_msg}")
                    parts.append("")

        return "\n".join(parts)

    # ─── Identity & Self-Awareness (versioned) ─────────────────────────

    def update_identity(self, content: str, reason: str = "") -> None:
        """
        Update the AI's identity definition with version tracking.
        Old versions are preserved for rollback.
        """
        if not content or not content.strip():
            return
        now = datetime.now().isoformat()
        with self._db_lock, self._get_conn() as conn:
            # Get next version number
            max_ver = conn.execute("SELECT MAX(version) FROM identity_versions").fetchone()[0]
            new_ver = (max_ver or 0) + 1
            # Deactivate all previous versions
            conn.execute("UPDATE identity_versions SET is_active = 0")
            # Save new version
            conn.execute(
                "INSERT INTO identity_versions (version, content, reason, created_at, is_active) "
                "VALUES (?,?,?,?,1)",
                (new_ver, content.strip(), reason, now),
            )
            # Update long_term scope (used for injection into prompts)
            conn.execute("DELETE FROM long_term WHERE scope = 'identity'")
            conn.execute(
                "INSERT INTO long_term (content, scope, tags, importance, created_at, source) "
                "VALUES (?, 'identity', 'personality,rules,tone', 1.0, ?, ?)",
                (content.strip(), now, f"identity_v{new_ver}"),
            )
        self._export_long_term_markdown()
        logger.info(f"Identity v{new_ver} saved ({len(content)} chars). Reason: {reason or 'N/A'}")

    def rollback_identity(self, steps: int = 1) -> str:
        """Rollback identity to a previous version. Returns the restored version info."""
        with self._db_lock, self._get_conn() as conn:
            # Get current active version
            current = conn.execute(
                "SELECT version FROM identity_versions WHERE is_active = 1"
            ).fetchone()
            if not current:
                return "No active identity version to rollback from."
            
            current_ver = current[0]
            target_ver = max(1, current_ver - steps)
            
            # Find target version
            target = conn.execute(
                "SELECT version, content, reason FROM identity_versions WHERE version = ?",
                (target_ver,),
            ).fetchone()
            if not target:
                return f"Version {target_ver} not found."
            
            # Deactivate all, reactivate target
            conn.execute("UPDATE identity_versions SET is_active = 0")
            conn.execute("UPDATE identity_versions SET is_active = 1 WHERE version = ?", (target_ver,))
            
            # Update long_term scope
            conn.execute("DELETE FROM long_term WHERE scope = 'identity'")
            conn.execute(
                "INSERT INTO long_term (content, scope, tags, importance, created_at, source) "
                "VALUES (?, 'identity', 'personality,rules,tone', 1.0, ?, ?)",
                (target[1], datetime.now().isoformat(), f"rollback_to_v{target_ver}"),
            )
        self._export_long_term_markdown()
        logger.info(f"Identity rolled back: v{current_ver} → v{target_ver}")
        return f"Rolled back from v{current_ver} to v{target_ver}. Reason was: {target[2] or 'N/A'}"

    def list_identity_versions(self, limit: int = 10) -> List[Dict]:
        """List identity version history."""
        with self._db_lock, self._get_conn() as conn:
            rows = conn.execute(
                "SELECT version, content, reason, created_at, is_active "
                "FROM identity_versions ORDER BY version DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [
            {"version": r[0], "content": r[1][:100] + "..." if len(r[1]) > 100 else r[1],
             "reason": r[2], "created_at": r[3], "active": bool(r[4])}
            for r in rows
        ]

    def seed_self_awareness(self, content: str) -> None:
        """
        Set the AI's self-awareness (architecture, capabilities, codebase).
        Replaces any existing self_awareness entries. scope='self_awareness'.
        """
        if not content or not content.strip():
            return
        with self._db_lock, self._get_conn() as conn:
            conn.execute("DELETE FROM long_term WHERE scope = 'self_awareness'")
            conn.execute(
                "INSERT INTO long_term (content, scope, tags, importance, created_at, source) "
                "VALUES (?, 'self_awareness', 'architecture,capabilities,system', 0.9, ?, 'system_seed')",
                (content.strip(), datetime.now().isoformat()),
            )
        self._export_long_term_markdown()
        logger.info(f"Self-awareness seeded ({len(content)} chars).")

    def update_self_awareness_tools(self, tool_schemas: dict) -> None:
        """
        Patch the '## Available Tools' section of the self-awareness LTM entry
        with a dynamically generated list based on actual registered tool schemas.

        This keeps the LLM's knowledge of its tools in sync with reality.
        Called from TextGen.__init__() after tools and memory are both ready.

        Args:
            tool_schemas: Dict of {name: schema_dict} from textGen._tool_schema_cache
        """
        if not tool_schemas:
            return

        # Read current self-awareness from LTM
        with self._db_lock, self._get_conn() as conn:
            row = conn.execute(
                "SELECT content FROM long_term WHERE scope = 'self_awareness' LIMIT 1"
            ).fetchone()
        if not row:
            return

        current = row[0]

        # Build dynamic tools section from schemas
        lines = ["## Available Tools\n"]
        for name, schema in sorted(tool_schemas.items()):
            func_info = schema.get("function", schema)
            desc = func_info.get("description", "").split("\n")[0]  # first line only
            params = func_info.get("parameters", {}).get("properties", {})
            param_names = ", ".join(params.keys()) if params else ""
            lines.append(f"- `{name}({param_names})` — {desc}")
        lines.append("")  # trailing newline

        new_tools_section = "\n".join(lines)

        # Replace the old section (everything between ## Available Tools and the next ##)
        marker_start = "## Available Tools"
        if marker_start in current:
            before = current.split(marker_start)[0]
            after_parts = current.split(marker_start)[1].split("\n## ", 1)
            if len(after_parts) > 1:
                after = "\n## " + after_parts[1]
            else:
                after = ""
            updated = before + new_tools_section + after
        else:
            # No existing section — append before the last section
            updated = current.rstrip() + "\n\n" + new_tools_section

        # Write back
        self.seed_self_awareness(updated)
        logger.info(f"Self-awareness tools section updated ({len(tool_schemas)} tools).")

    # ─── Journal / Workspace ─────────────────────────────────────────

    def write_journal(self, content: str, tags: str = "") -> None:
        """Write a note to the AI workspace/journal (scratchpad)."""
        if not content or not content.strip():
            return
        with self._db_lock, self._get_conn() as conn:
            conn.execute(
                "INSERT INTO journal (content, tags, created_at) VALUES (?,?,?)",
                (content.strip(), tags, datetime.now().isoformat()),
            )
        self._export_journal_markdown()
        logger.info(f"Journal entry written: {content[:50]}...")

    def read_journal(self, limit: int = 20, tags: Optional[str] = None) -> List[Dict]:
        """Read recent journal entries, optionally filtered by tags."""
        with self._db_lock, self._get_conn() as conn:
            if tags:
                rows = conn.execute(
                    "SELECT id, content, tags, created_at FROM journal "
                    "WHERE tags LIKE ? ORDER BY id DESC LIMIT ?",
                    (f"%{tags}%", limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT id, content, tags, created_at FROM journal ORDER BY id DESC LIMIT ?",
                    (limit,),
                ).fetchall()
        return [{"id": r[0], "content": r[1], "tags": r[2], "created_at": r[3]} for r in rows]

    # ─── Ops Board (outcome-driven state tracking) ───────────────────

    def retrieve_ops_board(self) -> Dict[str, List[Dict]]:
        """
        Retrieve the current Ops Board state.
        
        Returns a structured dict with 5 sections:
          NOW: top 1-3 priorities (with next_action, owner, deadline)
          WAITING: delegated/blocked items (who, what, blocking_on)
          UPCOMING: deadlines next 7 days
          INBOX: raw captures to triage
          METRICS: optional (sleep/energy, spend, etc.)
        """
        default_board = {
            "NOW": [],
            "WAITING": [],
            "UPCOMING": [],
            "INBOX": [],
            "METRICS": {}
        }
        
        if not self.ops_board_path.exists():
            return default_board
        
        try:
            with open(self.ops_board_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, Exception):
            return default_board

    def update_ops_board(self, board: Dict[str, Any]) -> None:
        """
        Replace the entire Ops Board state.
        
        Args:
            board: Full board dict with NOW/WAITING/UPCOMING/INBOX/METRICS keys
        """
        try:
            # Ensure required sections exist
            for section in ["NOW", "WAITING", "UPCOMING", "INBOX", "METRICS"]:
                if section not in board:
                    board[section] = [] if section != "METRICS" else {}
            
            # Atomic write
            tmp = self.ops_board_path.with_suffix(".json.tmp")
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(board, f, indent=2)
            os.replace(tmp, self.ops_board_path)
        except Exception as e:
            logger.warning(f"Failed to update ops board: {e}")

    def add_to_ops_inbox(self, item: str, tags: str = "") -> str:
        """Add a raw capture to INBOX for later triage."""
        board = self.retrieve_ops_board()
        entry = {
            "content": item,
            "tags": tags,
            "added_at": datetime.now().isoformat(),
        }
        board["INBOX"].append(entry)
        self.update_ops_board(board)
        return f"Added to INBOX: {item[:60]}..."

    def move_ops_item(self, item_content: str, from_section: str, to_section: str) -> str:
        """Move an item between board sections (e.g. INBOX → NOW, NOW → WAITING)."""
        board = self.retrieve_ops_board()
        
        # Find and remove from source
        source = board.get(from_section, [])
        if isinstance(source, list):
            # Handle both dict items and legacy string items
            match = None
            for i in source:
                if isinstance(i, dict):
                    if item_content.lower() in str(i.get("content", "")).lower():
                        match = i
                        break
                elif isinstance(i, str):
                    if item_content.lower() in i.lower():
                        match = i
                        break
            
            if match:
                source.remove(match)
                # Add to destination
                dest = board.get(to_section, [])
                if isinstance(dest, list):
                    dest.append(match)
                    self.update_ops_board(board)
                    return f"Moved '{item_content[:40]}...' from {from_section} → {to_section}"
        return f"Item not found in {from_section}"

    def mark_ops_done(self, item_content: str, section: str = "NOW") -> str:
        """Remove a completed item from the board."""
        board = self.retrieve_ops_board()
        source = board.get(section, [])
        if isinstance(source, list):
            before = len(source)
            # Handle both dict items and legacy string items
            def should_keep(i):
                if isinstance(i, dict):
                    return item_content.lower() not in str(i.get("content", "")).lower()
                elif isinstance(i, str):
                    return item_content.lower() not in i.lower()
                return True
            
            board[section] = [i for i in source if should_keep(i)]
            if len(board[section]) < before:
                self.update_ops_board(board)
                # Log to journal
                self.write_journal(f"Completed: {item_content}", tags="done,ops")
                return f"Marked done and removed from {section}"
        return f"Item not found in {section}"

    def get_ops_summary(self) -> str:
        """Human-readable summary of current Ops Board state."""
        board = self.retrieve_ops_board()
        lines = []
        for section in ["NOW", "WAITING", "UPCOMING", "INBOX"]:
            items = board.get(section, [])
            if isinstance(items, list) and items:
                lines.append(f"\n{section} ({len(items)}):")
                for item in items[:5]:  # max 5 per section
                    # Guard against mixed types (dict vs string)
                    if isinstance(item, dict):
                        content = item.get("content", str(item))[:60]
                    else:
                        content = str(item)[:60]
                    lines.append(f"  - {content}")
                if len(items) > 5:
                    lines.append(f"  ... and {len(items) - 5} more")
        metrics = board.get("METRICS", {})
        if metrics:
            lines.append(f"\nMETRICS: {json.dumps(metrics, indent=2)}")
        return "\n".join(lines) if lines else "(Ops Board empty)"

    # ─── GDrive sync ─────────────────────────────────────────────────

    def sync_to_gdrive(self, drive=None, folder_id: Optional[str] = None) -> bool:
        """Upload memory.db + markdown files to a GDrive folder."""
        try:
            if drive is None:
                gdrive_module = Utils.import_file("gdrive_API.py")
                drive = gdrive_module.GDriveWrapper()

            if not folder_id:
                folder = drive.get_or_create_folder("ArX_Memory")
                folder_id = folder["id"]
                if self.agent_name:
                    agent_folder = drive.get_or_create_folder(self.agent_name, parent_id=folder_id)
                    folder_id = agent_folder["id"]

            self._export_markdown()

            if self.db_path.exists():
                drive.upload_file(str(self.db_path), parent_id=folder_id)
            for md in [self.short_term_markdown, self.long_term_markdown, self.journal_markdown]:
                if md.exists():
                    drive.upload_file(str(md), parent_id=folder_id)
            if self.ops_board_path.exists():
                drive.upload_file(str(self.ops_board_path), parent_id=folder_id)

            logger.info(f"Memory synced to GDrive folder: {folder_id}")
            return True
        except Exception as e:
            logger.warning(f"GDrive sync failed: {e}")
            return False

    # ─── Internal: trim + extraction ─────────────────────────────────

    def _trim_short_term_memory(self) -> int:
        """Trim STM to stay under limit. FLUSH-BEFORE-COMPACTION: extracts insights first."""
        with self._db_lock, self._get_conn() as conn:
            rows = conn.execute(
                "SELECT id, system_msg, user_msg, assistant_msg FROM short_term ORDER BY id ASC"
            ).fetchall()

        total = sum(len(r[1] or "") + len(r[2] or "") + len(r[3] or "") for r in rows)
        if total <= self.short_term_limit:
            return 0

        # Collect oldest entries to remove until under limit
        to_remove = []
        running = total
        for r in rows:
            if running <= self.short_term_limit:
                break
            to_remove.append({"id": r[0], "user": r[2] or "", "assistant": r[3] or ""})
            running -= len(r[1] or "") + len(r[2] or "") + len(r[3] or "")

        if not to_remove:
            return 0

        # Flush: extract insights from entries about to die
        try:
            self._extract_insights_scored(to_remove, source="compaction_flush")
        except Exception as e:
            logger.warning(f"Flush-before-compaction failed (continuing trim): {e}")

        # Delete trimmed entries
        ids = [e["id"] for e in to_remove]
        with self._db_lock, self._get_conn() as conn:
            ph = ",".join("?" for _ in ids)
            conn.execute(f"DELETE FROM short_term WHERE id IN ({ph})", ids)

        return len(to_remove)

    def _extract_insights_scored(self, entries: List[Dict], source: str = "periodic"):
        """
        Scoring-based insight extraction using cognitive scopes.
        Single LLM call → JSON array with importance scores.
        Saves all insights ≥ 0.3 importance.
        """
        if not entries:
            return

        conv_text = ""
        for i, entry in enumerate(entries, 1):
            conv_text += f"Turn {i}:\nUser: {entry.get('user', '')}\nAssistant: {entry.get('assistant', '')}\n\n"

        # Build scope descriptions for the prompt
        scope_lines = "\n".join(
            f"  '{k}' ({v})" for k, v in self.COGNITIVE_SCOPES.items()
        )

        extraction_system = (
            "You are an AI memory curator. Extract facts, preferences, decisions, "
            "or any useful context from conversations.\n"
            "For each insight provide:\n"
            "- content: clear concise statement\n"
            "- scope: categorize as one of:\n"
            f"{scope_lines}\n"
            "- tags: relevant keywords (comma-separated)\n"
            "- importance: 0.0-1.0 (0.3=minor, 0.5=moderate, 0.7=important, 1.0=critical)\n\n"
            "Return ONLY a JSON array. If nothing worth saving, return: []\n"
            'Example: [{"content": "User prefers dark mode", "scope": "preferences", '
            '"tags": "ui,settings", "importance": 0.5}]'
        )

        try:
            raw = self.llm_api.chat_completion(
                user_prompt=f"Extract memorable insights from this conversation:\n\n{conv_text}",
                system_prompt=extraction_system,
                temperature=0.3,
                max_tokens=500,
            ).strip()

            insights = self._parse_json_array(raw)

            saved = 0
            for ins in insights:
                if not isinstance(ins, dict):
                    continue
                content = ins.get("content", "").strip()
                importance = float(ins.get("importance", 0.5))
                # Validate scope — fallback to 'facts' if unrecognized
                scope = ins.get("scope", "facts")
                if scope not in self.COGNITIVE_SCOPES:
                    scope = "facts"

                if content and importance >= 0.3:
                    self._save_long_term_insight(
                        content=content, scope=scope,
                        tags=ins.get("tags", ""), importance=importance, source=source,
                    )
                    saved += 1

            if saved:
                logger.info(f"Extracted {saved} insight(s) (source: {source}).")
            else:
                logger.info(f"No insights above threshold (source: {source}).")

        except json.JSONDecodeError:
            if raw and len(raw) > 15:
                self._save_long_term_insight(content=raw, source=source)
                logger.info("Extracted 1 insight (fallback parse).")
        except Exception as e:
            logger.warning(f"Insight extraction failed: {e}")

    def _parse_json_array(self, text: str) -> list:
        """Robustly extract a JSON array from LLM output."""
        cleaned = text
        if "```" in cleaned:
            match = re.search(r"```(?:json)?\s*(.*?)```", cleaned, re.DOTALL)
            if match:
                cleaned = match.group(1).strip()
        start = cleaned.find("[")
        end = cleaned.rfind("]") + 1
        if start >= 0 and end > start:
            cleaned = cleaned[start:end]
        result = json.loads(cleaned)
        return result if isinstance(result, list) else [result] if isinstance(result, dict) else []

    def _save_long_term_insight(
        self, content: str, scope: str = "facts", tags: str = "",
        importance: float = 0.5, source: str = "",
    ):
        """Save a structured insight to long-term memory."""
        if not content or not content.strip():
            return
        with self._db_lock, self._get_conn() as conn:
            conn.execute(
                "INSERT INTO long_term (content, scope, tags, importance, created_at, source) "
                "VALUES (?,?,?,?,?,?)",
                (content.strip(), scope, tags, importance, datetime.now().isoformat(), source),
            )
        self._export_long_term_markdown()
        logger.info(f"LTM [{scope}|{importance:.1f}]: {content[:60]}...")

    # ─── Markdown exports ────────────────────────────────────────────

    def _export_markdown(self):
        """Export STM, LTM, and journal to human-readable markdown."""
        self._export_short_term_markdown()
        self._export_long_term_markdown()
        self._export_journal_markdown()

    def _export_short_term_markdown(self):
        try:
            with self._db_lock, self._get_conn() as conn:
                rows = conn.execute(
                    "SELECT system_msg, user_msg, assistant_msg, kind, created_at "
                    "FROM short_term ORDER BY id DESC"
                ).fetchall()
            md = "# Short-Term Memory Log\n\n"
            for sys_msg, usr_msg, ast_msg, kind, ts in rows:
                md += f"## {ts}\n\n"
                if sys_msg:
                    md += f"**System:** {sys_msg}\n\n"
                md += f"**User:** {usr_msg}\n\n"
                md += f"**Assistant:** {ast_msg}\n\n---\n\n"
            with open(self.short_term_markdown, "w", encoding="utf-8") as f:
                f.write(md)
        except Exception as e:
            logger.warning(f"STM markdown export failed: {e}")

    def _export_long_term_markdown(self):
        try:
            with self._db_lock, self._get_conn() as conn:
                rows = conn.execute(
                    "SELECT content, scope, tags, importance, created_at, expires_at "
                    "FROM long_term ORDER BY scope ASC, importance DESC, created_at DESC"
                ).fetchall()
            md = "# Long-Term Memory (All Scopes)\n\n"
            current_scope = None
            for content, scope, tags, importance, ts, expires in rows:
                if scope != current_scope:
                    current_scope = scope
                    md += f"## Scope: {scope}\n\n"
                # Compact: importance + date + tags on one line, content below (no duplicate title)
                compact_ts = ts[:16].replace("T", " ") if ts else ""
                header = f"**[{importance:.1f}]** *{compact_ts}*"
                if tags:
                    header += f" | {tags}"
                if expires:
                    header += f" | exp: {expires[:10]}"
                md += f"{header}\n{content}\n\n---\n\n"
            with open(self.long_term_markdown, "w", encoding="utf-8") as f:
                f.write(md)
        except Exception as e:
            logger.warning(f"LTM markdown export failed: {e}")

    def _export_journal_markdown(self):
        try:
            with self._db_lock, self._get_conn() as conn:
                rows = conn.execute(
                    "SELECT content, tags, created_at FROM journal ORDER BY id DESC"
                ).fetchall()
            if not rows:
                return
            md = "# Journal / Workspace\n\n"
            for content, tags, ts in rows:
                md += f"## {ts}"
                if tags:
                    md += f" | {tags}"
                md += f"\n\n{content}\n\n---\n\n"
            with open(self.journal_markdown, "w", encoding="utf-8") as f:
                f.write(md)
        except Exception as e:
            logger.warning(f"Journal markdown export failed: {e}")


# ─── Demo ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    if LLMAPIWrapper is None:
        logger.critical(f"Could not import {LLM_PROVIDER.upper()} API wrapper.")
        exit(1)
    try:
        llm_api = LLMAPIWrapper(auto_pull=True) if LLM_PROVIDER == "ollama" else LLMAPIWrapper()
    except Exception as e:
        logger.critical(f"Error initializing {LLM_PROVIDER.upper()} API: {e}")
        exit(1)

    memory = Memory(llm_api=llm_api, short_term_limit=20000)
    memory.clear(full_reset=True)

    print("\n===== MEMORY SYSTEM DEMO (Cognitive Scopes + Journal) =====\n")

    # Seed identity
    memory.update_identity(
        "I am ArX, a creative AI assistant by Arvolve. I speak concisely, "
        "have opinions, and balance technical precision with artistic insight."
    )

    # Seed self-awareness
    memory.seed_self_awareness(
        "ArX runs on a modular Python codebase: TextGen (LLM hub), Memory (SQLite), "
        "RAG (FAISS vector search), Tools (web crawl, image gen, email, etc.), "
        "AgentGen (multi-agent orchestration). Memory uses cognitive scopes for categorization."
    )

    # Write some journal notes
    memory.write_journal("Exploring music-to-image generation pipeline ideas.", tags="project,brainstorm")
    memory.write_journal("Need to benchmark FAISS vs ChromaDB for vector search.", tags="research,todo")

    # Simulate conversation
    def chat(user_input):
        history = memory.retrieve_short_term_formatted()
        response = llm_api.chat_completion(user_prompt=user_input, message_history=history)
        memory.save_short_term("You are ArX.", user_input, response)
        return response

    for msg in [
        "I prefer dark mode for all UIs. Remember that.",
        "My timezone is UTC+0 and I work from London.",
        "We decided to use FAISS for vector search. Key technical decision.",
        "The deployment process is: build Docker → push to registry → helm upgrade.",
    ]:
        print(f"User: {msg}")
        print(f"ArX: {chat(msg)[:120]}...\n")

    # Force some insights
    memory.extract_to_long_term(force_insight="User prefers dark mode for all UIs.")

    # Show results
    print("\n===== MEMORY CONTENTS =====")
    print(f"\nIdentity: {memory.retrieve_by_scope('identity')}")
    print(f"\nSelf-awareness: {memory.retrieve_by_scope('self_awareness')}")
    print(f"\nOrganic LTM ({len(memory.retrieve_long_term())} entries):")
    for d in memory.retrieve_long_term_detailed():
        print(f"  [{d['scope']}|{d['importance']:.1f}] {d['content'][:70]}...")
    print(f"\nJournal ({len(memory.read_journal())} entries):")
    for j in memory.read_journal():
        print(f"  [{j['tags']}] {j['content'][:70]}...")
    print(f"\nDB: {memory.db_path}")
    print("===== DEMO COMPLETE =====")
