"""Independent player identities, scoped evidence, and explicitly requested turns."""

from __future__ import annotations

from copy import deepcopy
import json
import time
from uuid import uuid4

from .campaigns import Campaigns, order_conversation, required
from .context import pack_context, _Counter
from .decisions import recall
from .model import LocalModel
from .player_contract import PLAYER_SYSTEM, PLAN_SYSTEM, PlayerDecision, reply_contract
from .rules import search_rules, active_documents
from .settings import load as model_settings
from .store import digest, now, packed


class Players:
    def __init__(self, store):
        self.store, self.campaigns = store, Campaigns(store)
        with store.db() as db:
            db.executescript("""
            CREATE TABLE IF NOT EXISTS player_profiles (
                id TEXT PRIMARY KEY,name TEXT NOT NULL,instructions TEXT NOT NULL,
                adapter_id INTEGER,revision INTEGER NOT NULL,updated TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS player_bindings (
                id TEXT PRIMARY KEY,campaign_id TEXT NOT NULL,profile_id TEXT NOT NULL,
                character_id TEXT NOT NULL,active INTEGER NOT NULL,revision INTEGER NOT NULL,
                UNIQUE(campaign_id,character_id));
            CREATE TABLE IF NOT EXISTS player_runs (
                id TEXT PRIMARY KEY,binding_id TEXT NOT NULL,session_id TEXT NOT NULL,
                status TEXT NOT NULL,request TEXT NOT NULL,result TEXT NOT NULL,
                created TEXT NOT NULL,finished TEXT,message_id TEXT);
            CREATE UNIQUE INDEX IF NOT EXISTS player_active_turn
                ON player_runs(binding_id,session_id) WHERE status IN ('queued','running');
            """)

    def profiles(self):
        with self.store.db() as db:
            return [
                dict(r)
                for r in db.execute("SELECT * FROM player_profiles ORDER BY name,id")
            ]

    def profile(self, identifier):
        with self.store.db() as db:
            row = db.execute(
                "SELECT * FROM player_profiles WHERE id=?", (identifier,)
            ).fetchone()
        if row is None:
            raise ValueError("Unknown player profile.")
        return dict(row)

    def save_profile(
        self,
        name,
        instructions,
        *,
        adapter_id=None,
        identifier=None,
        expected_revision=None,
    ):
        if adapter_id is not None and (type(adapter_id) is not int or adapter_id < 0):
            raise ValueError(
                "Choose a nonnegative server adapter ID, or the base model."
            )
        name, instructions = (
            required(name, "Player name", 100),
            required(instructions, "Player personality", 4000),
        )
        identifier = identifier or uuid4().hex
        with self.store.db() as db:
            db.execute("BEGIN IMMEDIATE")
            prior = db.execute(
                "SELECT * FROM player_profiles WHERE id=?", (identifier,)
            ).fetchone()
            if prior and expected_revision != prior["revision"]:
                raise ValueError("This player profile changed. Reload before saving.")
            revision = prior["revision"] + 1 if prior else 1
            db.execute(
                "INSERT INTO player_profiles VALUES(?,?,?,?,?,?) ON CONFLICT(id) DO UPDATE SET name=excluded.name,instructions=excluded.instructions,adapter_id=excluded.adapter_id,revision=excluded.revision,updated=excluded.updated",
                (identifier, name, instructions, adapter_id, revision, now()),
            )
        return identifier

    def bindings(self, campaign_id):
        self.campaigns.campaign(campaign_id)
        with self.store.db() as db:
            return [
                dict(r)
                for r in db.execute(
                    "SELECT * FROM player_bindings WHERE campaign_id=? ORDER BY rowid",
                    (campaign_id,),
                )
            ]

    def binding(self, identifier):
        with self.store.db() as db:
            row = db.execute(
                "SELECT * FROM player_bindings WHERE id=?", (identifier,)
            ).fetchone()
        if row is None:
            raise ValueError("Unknown player assignment.")
        return dict(row)

    def assign(self, campaign_id, profile_id, character_id, *, active=True):
        self.profile(profile_id)
        if not any(
            c["id"] == character_id for c in self.campaigns.characters(campaign_id)
        ):
            raise ValueError("Choose a character from this campaign.")
        with self.store.db() as db:
            db.execute("BEGIN IMMEDIATE")
            prior = db.execute(
                "SELECT * FROM player_bindings WHERE campaign_id=? AND character_id=?",
                (campaign_id, character_id),
            ).fetchone()
            identifier = prior["id"] if prior else uuid4().hex
            revision = prior["revision"] + 1 if prior else 1
            db.execute(
                "INSERT INTO player_bindings VALUES(?,?,?,?,?,?) ON CONFLICT(campaign_id,character_id) DO UPDATE SET profile_id=excluded.profile_id,active=excluded.active,revision=excluded.revision",
                (
                    identifier,
                    campaign_id,
                    profile_id,
                    character_id,
                    int(bool(active)),
                    revision,
                ),
            )
        profile = self.profile(profile_id)
        self.campaigns.map_participant(
            campaign_id,
            profile["name"] + " (AI)",
            "agent:" + identifier,
            role="player",
            character_id=character_id,
        )
        return identifier

    def perspective(self, session_id, binding_id):
        """Construct access boundaries before any model call or retrieval.

        Deliberately do not use the facilitator snapshot or its inferred state:
        those inferences were made with privileged context and can contain secrets.
        """
        binding = self.binding(binding_id)
        session = self.campaigns.session(session_id)
        if binding["campaign_id"] != session["campaign_id"] or not binding["active"]:
            raise ValueError("Choose an active player assigned to this campaign.")
        profile = self.profile(binding["profile_id"])
        characters = self.campaigns.characters(session["campaign_id"])
        own = next((c for c in characters if c["id"] == binding["character_id"]), None)
        if own is None:
            raise ValueError("The assigned character is no longer available.")
        public_characters = {
            c["name"] for c in characters if c["visibility"] == "public"
        }
        visible = []
        for message in self.campaigns.messages(session_id):
            recipient = message.get("recipient") or (
                "table" if message["visibility"] == "public" else "facilitator"
            )
            own_speech = message["role"] == "player" and (
                message["character"] == own["name"]
                or message.get("source", {}).get("character_id") == own["id"]
            )
            if recipient not in {"table", own["id"]} and not own_speech:
                continue
            safe = {
                k: message[k]
                for k in ["id", "ordinal", "revision", "speaker", "role", "text"]
            }
            safe["character"] = (
                message["character"]
                if message["character"] in public_characters | {own["name"]}
                else ""
            )
            safe["visibility"] = "public" if recipient == "table" else "private"
            # Keep chronology without exposing transport paths or provenance prompts.
            origin = message.get("source", {})
            if origin.get("kind") == "player_agent":
                safe["speaker"] = origin.get("player_name", "Player") + " (AI)"
            timing = origin.get("timeline", {})
            timing = timing if isinstance(timing, dict) else {}
            timing = {
                k: v
                for k, v in timing.items()
                if k
                in {
                    "kind",
                    "domain",
                    "utc_start",
                    "utc_end",
                    "start_seconds",
                    "end_seconds",
                }
            }
            if timing.get("domain"):
                timing["domain"] = digest(str(timing["domain"]))[:16]
            safe["source"] = {"kind": origin.get("kind", "manual"), "timeline": timing}
            safe["created"] = message["created"]
            visible.append(safe)
        conversation = order_conversation(visible)
        for message in conversation["messages"]:
            message.pop("source", None)
            message.pop("created", None)
        documents = [
            {
                k: d[k]
                for k in ["id", "title", "text", "visibility", "sha256", "metadata"]
            }
            for d in active_documents({"documents": self.campaigns.documents(session["campaign_id"], public_only=True)})
        ]
        # A public document's filesystem/upload metadata is unnecessary to the model.
        for document in documents:
            meta = document["metadata"]
            meta = json.loads(meta) if isinstance(meta, str) else meta
            document["metadata"] = {"kind": meta.get("kind", "reference")}
        recipients = [
            {"id": "table", "name": "Everyone at the table"},
            {"id": "facilitator", "name": "Private to the facilitator"},
        ]
        recipients += [
            {"id": c["id"], "name": c["name"]}
            for c in characters
            if c["visibility"] == "public" and c["id"] != own["id"]
        ]
        return {
            "session_id": session_id,
            "binding": binding,
            "profile": profile,
            "character": {k: own[k] for k in ["id", "name", "sheet", "revision"]},
            "messages": conversation["messages"],
            "documents": documents,
            "recipients": recipients,
        }

    @staticmethod
    def fingerprint(perspective):
        value = deepcopy(perspective)
        for message in value["messages"]:
            message.pop("order_index", None)
            message.pop("ingestion_ordinal", None)
            message.get("chronology", {}).pop("overlaps_prior", None)
        return digest(packed(value))

    def runs(self, session_id, binding_id=None):
        self.campaigns.session(session_id)
        with self.store.db() as db:
            rows = db.execute(
                "SELECT * FROM player_runs WHERE session_id=?"
                + (" AND binding_id=?" if binding_id else "")
                + " ORDER BY rowid DESC",
                (session_id, binding_id) if binding_id else (session_id,),
            )
            return [
                {
                    **dict(r),
                    "request": json.loads(r["request"]),
                    "result": json.loads(r["result"]),
                }
                for r in rows
            ]

    def request(
        self,
        session_id,
        binding_id,
        *,
        workers=None,
        model_factory=None,
        context_options=None,
        on_complete=None,
    ):
        perspective = self.perspective(session_id, binding_id)
        with self.store.db() as db:
            db.execute("BEGIN IMMEDIATE")
            active = db.execute(
                "SELECT id FROM player_runs WHERE session_id=? AND binding_id=? AND status IN ('queued','running')",
                (session_id, binding_id),
            ).fetchone()
            if active:
                return active["id"]
            previous = db.execute(
                "SELECT request,message_id FROM player_runs WHERE session_id=? AND binding_id=? AND status='complete' ORDER BY rowid DESC LIMIT 1",
                (session_id, binding_id),
            ).fetchone()
            if previous:
                before_own_turn = deepcopy(perspective)
                before_own_turn["messages"] = [
                    m
                    for m in before_own_turn["messages"]
                    if m["id"] != previous["message_id"]
                ]
                if json.loads(previous["request"]).get(
                    "context_hash"
                ) == self.fingerprint(before_own_turn):
                    raise ValueError(
                        "This player has already responded to the current conversation. Add a new contribution before another turn."
                    )
            identifier = uuid4().hex
            db.execute(
                "INSERT INTO player_runs VALUES(?,?,?,?,?,?,?,?,?)",
                (
                    identifier,
                    binding_id,
                    session_id,
                    "queued",
                    "{}",
                    "{}",
                    now(),
                    None,
                    None,
                ),
            )

        def work():
            self._run(
                identifier, session_id, binding_id, model_factory, context_options
            )
            if on_complete:
                on_complete(session_id)

        if workers is None:
            work()
        else:
            try:
                workers.submit(work)
            except RuntimeError as exc:
                self._finish(identifier, "failed", {"error": str(exc)})
        return identifier

    def _finish(self, identifier, status, result, message_id=None):
        with self.store.db() as db:
            db.execute(
                "UPDATE player_runs SET status=?,result=?,finished=?,message_id=? WHERE id=? AND status IN ('queued','running')",
                (status, packed(result), now(), message_id, identifier),
            )

    def cancel(self, session_id, identifier):
        with self.store.db() as db:
            changed = db.execute(
                "UPDATE player_runs SET status='cancelled',finished=? WHERE id=? AND session_id=? AND status IN ('queued','running')",
                (now(), identifier, session_id),
            ).rowcount
        if not changed:
            raise ValueError("Choose a queued or running turn from this session.")

    def _active(self, identifier):
        with self.store.db() as db:
            row = db.execute(
                "SELECT status FROM player_runs WHERE id=?", (identifier,)
            ).fetchone()
        return bool(row and row["status"] in {"queued", "running"})

    def _run(self, identifier, session_id, binding_id, model_factory, options):
        started = time.monotonic()
        trace = {
            "steps": [],
            "world_state_mutations": False,
            "source_protocol": "player-turn-v1",
        }
        try:
            if not self._active(identifier):
                return
            view = self.perspective(session_id, binding_id)
            fingerprint = self.fingerprint(view)
            config = model_settings(self.store.home)
            if options is None:
                from .runtime_context import context_options

                options = context_options(self.store)
            options = dict(options)
            counter = _Counter(options.get("tokenizer"), options.get("token_counter"))
            budget = options["context_limit"] - options.get("safety_margin", 512) - 700
            selected = view["profile"]["adapter_id"]
            ids = [a["id"] for a in config["routing"]["adapters"]]
            if selected is not None and (
                config["backend"] != "llama.cpp" or selected not in ids
            ):
                raise ValueError(
                    "This player's adapter is not in the configured local server inventory."
                )
            player_config = deepcopy(config)
            player_config["routing"]["tasks"]["player"] = selected
            factory = model_factory or (
                lambda task: LocalModel(task=task, configuration=player_config)
            )
            documents = deepcopy(view["documents"])
            documents.append(
                {
                    "id": "starting-character",
                    "title": "Your starting character sheet",
                    "visibility": "private",
                    "pinned": True,
                    "text": packed(view["character"]),
                }
            )
            packed_view = pack_context(
                turns=view["messages"],
                state={},
                player_input=view["messages"][-1]["text"]
                if view["messages"]
                else "Introduce your character without inventing a scene or other participants.",
                direction=view["profile"]["instructions"],
                documents=documents,
                rules=[],
                system_prompt=PLAYER_SYSTEM,
                active_entities=[view["character"]["name"]],
                **options,
            )
            body = packed_view["body"]
            body["player_personality"] = body.pop("private_facilitator_direction")
            body["instruction"] = (
                "Take one turn for your assigned character using only these visible sources. Starting-sheet values may have changed in the conversation."
            )
            body["available_recipients"] = view["recipients"]
            body["assigned_character"] = {
                k: view["character"][k] for k in ["id", "name"]
            }
            observations, seen = [], set()
            with self.store.db() as db:
                db.execute(
                    "UPDATE player_runs SET status='running',request=? WHERE id=?",
                    (
                        packed(
                            {
                                "perspective": view,
                                "body": body,
                                "context_hash": fingerprint,
                                "packing": packed_view["trace"],
                            }
                        ),
                        identifier,
                    ),
                )
            planning_started = time.monotonic()
            for step_number in range(3):
                if not self._active(identifier):
                    return
                if (
                    self.fingerprint(self.perspective(session_id, binding_id))
                    != fingerprint
                ):
                    self._finish(
                        identifier,
                        "stale",
                        {
                            **trace,
                            "reason": "The visible conversation or player configuration changed.",
                        },
                    )
                    return
                decision_input = {"context": body, "tool_results": observations}
                if counter.count(decision_input, PLAN_SYSTEM) > budget:
                    trace["evidence_limit"] = (
                        "No further tool call fits; answer from visible evidence or ask the facilitator."
                    )
                    break
                remaining = 45 - (time.monotonic() - planning_started)
                if remaining <= 0:
                    trace["evidence_limit"] = "Planning time budget reached."
                    break
                decision, metrics = factory("auditor").complete(
                    [
                        {"role": "system", "content": PLAN_SYSTEM},
                        {"role": "user", "content": packed(decision_input)},
                    ],
                    PlayerDecision,
                    max_tokens=400,
                    temperature=0,
                    timeout=remaining,
                )
                step = {"decision": decision.model_dump(), "model": metrics}
                trace["steps"].append(step)
                if decision.action == "speak":
                    break
                key = (decision.action, decision.query)
                if key in seen:
                    step["error"] = "Repeated lookup was stopped."
                    break
                seen.add(key)
                if decision.action == "sheet":
                    result = {"starting_sheet": view["character"]}
                elif not decision.query.strip():
                    result = {"missing": "A query is required."}
                elif decision.action == "recall":
                    result = recall({"messages": view["messages"]}, decision.query)
                else:
                    result = search_rules(
                        self.store,
                        decision.query,
                        limit=3,
                        snapshot={"documents": view["documents"], "rule_profile": {}},
                    )
                observation = {
                    "tool": decision.action,
                    "query": decision.query,
                    "result": result,
                }
                if (
                    counter.count(
                        {"context": body, "tool_results": [*observations, observation]},
                        PLAYER_SYSTEM,
                    )
                    > budget
                ):
                    observation = {
                        "tool": decision.action,
                        "gap": "The complete result does not fit. Ask the facilitator instead of inventing details.",
                    }
                observations.append(observation)
                step["observation"] = observation
            request = {
                "context": body,
                "tool_results": observations,
                "turn_plan": trace["steps"][-1].get("decision", {}).get("reason", "")
                if trace["steps"]
                else "",
            }
            if counter.count(request, PLAYER_SYSTEM) > budget:
                raise ValueError(
                    "This player's permitted context does not fit the configured window."
                )
            if not self._active(identifier):
                return
            answer, metrics = factory("player").complete(
                [
                    {"role": "system", "content": PLAYER_SYSTEM},
                    {"role": "user", "content": packed(request)},
                ],
                reply_contract([r["id"] for r in view["recipients"]]),
                max_tokens=600,
                temperature=0.7,
                timeout=90,
            )
            trace["player"] = {
                "answer": answer.model_dump(),
                "model": metrics,
                "prompt_tokens": counter.count(request, PLAYER_SYSTEM),
            }
            if not answer.utterance.strip() or answer.recipient not in {
                r["id"] for r in view["recipients"]
            }:
                raise ValueError(
                    "The player returned an empty contribution or an unavailable recipient."
                )
            if (
                self.fingerprint(self.perspective(session_id, binding_id))
                != fingerprint
            ):
                self._finish(
                    identifier,
                    "stale",
                    {
                        **trace,
                        "reason": "Context changed during the player's response.",
                    },
                )
                return
            # The final source check and utterance insert share one write lock.
            # Another conversation edit cannot slip between validation and posting.
            with self.store.db() as db:
                db.execute("BEGIN IMMEDIATE")
                status = db.execute(
                    "SELECT status FROM player_runs WHERE id=?", (identifier,)
                ).fetchone()
                if not status or status["status"] != "running":
                    return
                if (
                    self.fingerprint(self.perspective(session_id, binding_id))
                    != fingerprint
                ):
                    db.execute(
                        "UPDATE player_runs SET status='stale',result=?,finished=? WHERE id=?",
                        (
                            packed(
                                {**trace, "reason": "Context changed before posting."}
                            ),
                            now(),
                            identifier,
                        ),
                    )
                    return
                speaker = "agent:" + binding_id
                participant = db.execute(
                    "SELECT id FROM play_participants WHERE campaign_id=? AND speaker=?",
                    (view["binding"]["campaign_id"], speaker),
                ).fetchone()
                participant_id = participant["id"] if participant else uuid4().hex
                db.execute(
                    "INSERT INTO play_participants VALUES(?,?,?,?,?,?) ON CONFLICT(campaign_id,speaker) DO UPDATE SET name=excluded.name,role=excluded.role,character_id=excluded.character_id",
                    (
                        participant_id,
                        view["binding"]["campaign_id"],
                        view["profile"]["name"] + " (AI)",
                        "player",
                        speaker,
                        view["character"]["id"],
                    ),
                )
                message_id, stamp = uuid4().hex, now()
                ordinal = db.execute(
                    "SELECT coalesce(max(ordinal),0)+1 FROM play_messages WHERE session_id=?",
                    (session_id,),
                ).fetchone()[0]
                source = {
                    "kind": "player_agent",
                    "player_name": view["profile"]["name"],
                    "profile_id": view["profile"]["id"],
                    "binding_id": binding_id,
                    "character_id": view["character"]["id"],
                    "run_id": identifier,
                    "adapter_id": selected,
                    "recipient": answer.recipient,
                    "timeline": {
                        "kind": "manual_entry",
                        "domain": "agent-turn-utc",
                        "utc_start": time.time(),
                    },
                }
                db.execute(
                    "INSERT INTO play_messages VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                    (
                        message_id,
                        session_id,
                        ordinal,
                        speaker,
                        "player",
                        view["character"]["name"],
                        answer.utterance.strip(),
                        "public" if answer.recipient == "table" else "private",
                        packed(source),
                        "player-run:" + identifier,
                        stamp,
                    ),
                )
                trace["seconds"] = round(time.monotonic() - started, 3)
                db.execute(
                    "UPDATE player_runs SET status='complete',result=?,finished=?,message_id=? WHERE id=?",
                    (packed(trace), stamp, message_id, identifier),
                )
        except Exception as exc:
            self._finish(
                identifier,
                "failed",
                {
                    **trace,
                    "error": str(exc),
                    "model_error": getattr(exc, "trace", {}),
                    "seconds": round(time.monotonic() - started, 3),
                },
            )
