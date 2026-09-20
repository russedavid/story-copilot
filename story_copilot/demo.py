"""An original, reusable fictional campaign; not actual play or evaluation gold.

No podcast transcripts, published scenarios, or rulebook passages are included.
Creation is explicit, atomic and idempotent; reopening never overwrites edits.
"""

from __future__ import annotations

import argparse
from uuid import NAMESPACE_URL, uuid5

from .campaigns import Campaigns
from .store import Store, digest, now, packed

SLUG = "the-unsent-signal-v1"
TITLE = "Example · The Unsent Signal"
DISCLOSURE = (
    "This campaign began as an original fictional example written for Story Copilot. "
    "Its initial conversation is authored, not recorded play or an evaluation gold answer. "
    "You can change the scene, choose a response, and continue in your own direction."
)
BRIEFING = """The Unsent Signal

A storm has cut the road to a coastal weather station. At 9:10 p.m., its disconnected
receiver prints a warning dated eleven minutes in the future: DO NOT OPEN THE NORTH DOOR.

Jo Bell, a radio engineer, and Rowan Hart, a reporter, have come to find the operator
who missed his evening check-in. The station's caretaker, Mara Vale, offers them dry
coats but keeps watching the corridor to the north. Rain drums against the windows.
The visitor book on the counter has a torn stub where today's page should be.
The wall clock reads 9:21; Jo's pocket watch reads 9:10. Somewhere behind a sealed
panel, three soft clicks repeat at uneven intervals.

The characters are still in the entrance room. Nobody has opened the panel or
the north door. No attack, injury, or expenditure has occurred. What they do next
belongs to the players.
"""
STORY_NOTES = """Facilitator-only possibilities for this original example

Mara quietly disconnected the main relay herself. She heard her missing brother's
voice on the line and became frightened. She wants help but fears that admitting
what she heard will make the visitors leave. She is not secretly attacking them.

The eleven-minute discrepancy may come from the station clock, an old delayed
transmission, or something stranger. Choose the explanation as Facilitator; these are
possibilities, not established facts. The clicks can point toward an accessible
maintenance hatch without forcing anyone to open it.

Respond to both characters. Give Jo concrete sensory or technical observations
that distinguish what is obvious from what would require closer examination. Let
Rowan's question expose Mara's competing motives through a reply or a hesitation.
Do not silently resolve an unmade roll, choose a player's action, reveal these
notes wholesale, or force the party through the north door.
"""
DIRECTION = (
    "Run an intimate, grounded mystery. Keep the characters' choices open. "
    "Answer their immediate questions before introducing another complication. "
    "Suggest a response I can use or adapt; I remain the Facilitator."
)
EXAMPLE_RULES = """Original rules for this example, not a published game system.
Players choose what their characters attempt; the facilitator describes observable results.
Obvious sounds and visible objects need no roll. Uncertainty calls for a question, not an invented check.
Powering the portable relay tester costs 1 battery charge. A character must have at least the cost available.
A suggestion to use the tester does not spend anything; only a confirmed use in the conversation does.
A character learns a secret only by hearing it or discovering it. Private facilitator notes are not shared knowledge.
"""
RULE_PROFILE = {
    "resource_aliases": {"charges": "battery_charges"},
    "query_aliases": {},
    "tools": [
        {
            "name": "can_afford",
            "operation": "greater_equal",
            "description": "Compare the character's current battery charges with the stated cost, in that order.",
        }
    ],
}
STYLE = (
    "Specific sounds, objects, and human motives; restrained unease. Brief scenes, "
    "natural NPC dialogue, and clear opportunities for the players to act."
)
CHARACTERS = [
    (
        "jo",
        "Jo Bell",
        {
            "occupation": "Radio engineer",
            "pronouns": "she/her",
            "expertise": [
                "electrical repair",
                "careful listening",
                "weather instruments",
            ],
            "resources": {"battery_charges": 3},
            "inventory": ["pocket watch", "insulated hand tools", "small torch"],
            "motivation": "Find the missing operator without putting Rowan in danger.",
            "relationship": "Rowan has helped explain Jo's technical work to the public.",
        },
    ),
    (
        "rowan",
        "Rowan Hart",
        {
            "occupation": "Reporter",
            "pronouns": "they/them",
            "expertise": ["reporting", "archival research", "patient questioning"],
            "resources": {"film_frames": 8},
            "inventory": ["notebook", "pencil", "folded coastal map"],
            "motivation": "Find a missing person before turning the mystery into a story.",
            "relationship": "Jo is a trusted friend, not a source to pressure for a quote.",
        },
    ),
]
CONVERSATION = [
    (
        "Facilitator",
        "facilitator",
        "",
        "Mara sets two dry coats on a chair. 'The operator sometimes takes a walk when the weather turns,' she says, although the road outside is already flooding. The visitor book on the counter has a torn stub where today's page should be. The disconnected receiver prints the warning. Three clicks come from behind the sealed panel. You are both still in the entrance room.",
    ),
    (
        "Ari",
        "player",
        "Jo Bell",
        "I leave the north door shut and switch off the receiver before touching it. Can I trace where the clicking comes from without opening the sealed panel?",
    ),
    (
        "Bea",
        "player",
        "Rowan Hart",
        "I ask Mara when she last saw the operator and show her the empty space where today's page should be in the visitor book. I keep my voice calm; I am not threatening or restraining her.",
    ),
    (
        "Ari",
        "player",
        "Jo Bell",
        "Before I touch any wiring, I listen beside the panel. Is the clicking obvious enough to locate without powering the test equipment? And which clock does the warning's time agree with?",
    ),
]


def identity(part):
    return uuid5(NAMESPACE_URL, "story-copilot/original-demo/" + SLUG + "/" + part).hex


def demo_info(campaigns, campaign_id):
    with campaigns.store.db() as db:
        row = db.execute(
            "SELECT * FROM play_demo_imports WHERE campaign_id=?", (campaign_id,)
        ).fetchone()
    return dict(row) if row else None


def ensure_demo(campaigns):
    """Build in one transaction so concurrent clicks cannot leave half a demo."""
    campaign_id, session_id = identity("campaign"), identity("session")
    stamp = now()
    with campaigns.store.db() as db:
        db.execute("BEGIN IMMEDIATE")
        existing = db.execute(
            "SELECT * FROM play_demo_imports WHERE slug=?", (SLUG,)
        ).fetchone()
        if existing:
            return {**dict(existing), "created_now": False}
        db.execute(
            "INSERT INTO play_campaigns(id,title,system,direction,style,created,updated) VALUES(?,?,?,?,?,?,?)",
            (campaign_id, TITLE, "Custom rules", DIRECTION, STYLE, stamp, stamp),
        )
        for part, title, text, vis in [
            ("about", "About this example", DISCLOSURE, "public"),
            ("briefing", "The Unsent Signal · player briefing", BRIEFING, "public"),
            ("notes", "The Unsent Signal · Facilitator notes", STORY_NOTES, "private"),
            ("rules", "Original example rules", EXAMPLE_RULES, "public"),
        ]:
            db.execute(
                "INSERT INTO play_documents(id,campaign_id,title,text,visibility,sha256,metadata,created) VALUES(?,?,?,?,?,?,?,?)",
                (
                    identity(part),
                    campaign_id,
                    title,
                    text,
                    vis,
                    digest(text),
                    packed(
                        {
                            "kind": "rules" if part == "rules" else "reference",
                            "source": "original_authored_demo",
                            "demo": SLUG,
                            "not_real_play": True,
                            "not_evaluation_gold": True,
                        }
                    ),
                    stamp,
                ),
            )
        db.execute(
            "INSERT INTO play_rule_profiles VALUES(?,?,1,?)",
            (campaign_id, packed(RULE_PROFILE), stamp),
        )
        for part, name, sheet in CHARACTERS:
            db.execute(
                "INSERT INTO play_characters(id,campaign_id,name,sheet,visibility,revision,updated) VALUES(?,?,?,?,?,?,?)",
                (identity(part), campaign_id, name, packed(sheet), "public", 1, stamp),
            )
        for speaker, role, character in [
            ("Facilitator", "facilitator", None),
            ("Ari", "player", identity("jo")),
            ("Bea", "player", identity("rowan")),
        ]:
            db.execute(
                "INSERT INTO play_participants(id,campaign_id,name,role,speaker,character_id) VALUES(?,?,?,?,?,?)",
                (
                    identity("speaker-" + speaker),
                    campaign_id,
                    speaker + " (fictional)",
                    role,
                    speaker,
                    character,
                ),
            )
        db.execute(
            "INSERT INTO play_sessions(id,campaign_id,ordinal,title,parent_id,proactive,created) VALUES(?,?,?,?,?,?,?)",
            (session_id, campaign_id, 1, "At the weather station", None, 0, stamp),
        )
        db.execute("INSERT INTO play_session_kinds VALUES(?,?)", (session_id, "fresh"))
        db.execute(
            "INSERT INTO play_campaign_progress VALUES(?,?)", (campaign_id, session_id)
        )
        for ordinal, (speaker, role, character, text) in enumerate(CONVERSATION, 1):
            source = {
                "kind": "manual",
                "authored_demo": SLUG,
                "not_real_play": True,
                "not_evaluation_gold": True,
                "note": "Original fictional conversation, authored for this demo; entry time is not fictional-world time.",
            }
            db.execute(
                "INSERT INTO play_messages(id,session_id,ordinal,speaker,role,character,text,visibility,source,external_id,created) VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                (
                    identity("message-" + str(ordinal)),
                    session_id,
                    ordinal,
                    speaker,
                    role,
                    character,
                    text,
                    "public",
                    packed(source),
                    "demo:" + SLUG + ":" + str(ordinal),
                    now(),
                ),
            )
        db.execute(
            "INSERT INTO play_demo_imports VALUES(?,?,?,?)",
            (SLUG, campaign_id, session_id, stamp),
        )
    return {
        "slug": SLUG,
        "campaign_id": campaign_id,
        "session_id": session_id,
        "created": stamp,
        "created_now": True,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--home", help="Private Story Copilot data directory.")
    args = parser.parse_args()
    result = ensure_demo(Campaigns(Store(args.home)))
    print("Original fictional example ready at /play/" + result["session_id"])
    print(
        "Automatic suggestions are off on first creation. Request a suggestion when ready."
    )


if __name__ == "__main__":
    main()
