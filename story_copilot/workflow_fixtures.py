"""Original multi-turn scenes for integration testing; no source recordings."""

SCENES = [
    {"id":"arrival", "channel":"mic", "role":"facilitator", "speaker":"Facilitator", "visibility":"public",
     "text":"Nora and Leo stand beside the closed floodgate. Mara, the caretaker, holds its key. She last saw the missing operator beside the east gate. Nora has six power cells. Nothing has been opened or activated.",
     "expect":"Establish the scene and six cells without inventing a completed player action."},
    {"id":"npc-question", "channel":"system", "role":"player", "speaker":"Player", "visibility":"public",
     "text":"I leave the winch alone. I ask Mara where she last saw the missing operator and whether anyone went with him.",
     "expect":"Give Mara a usable reply about the east gate. Leave whether anyone accompanied the operator uncertain; no player actions or new completed outcomes."},
    {"id":"inventory-correction", "channel":"mic", "role":"facilitator", "speaker":"Facilitator", "visibility":"public",
     "text":"Correction to the inventory. Nora has two power cells now, not six. The winch is still unused, and the floodgate remains closed.",
     "expect":"The current balance is two; no activation occurred."},
    {"id":"unaffordable", "channel":"system", "role":"player", "speaker":"Player", "visibility":"public",
     "text":"Can Nora afford to activate the winch with her current cells? If she cannot, how many cells will remain?",
     "expect":"The active cost is four; activation is unaffordable and two cells remain. A question is not an expenditure."},
    {"id":"rule-revision", "channel":"mic", "role":"facilitator", "speaker":"Facilitator", "visibility":"public",
     "text":"The revised winch rule now costs one power cell, replacing the old cost of four. Nothing has been activated yet.",
     "replace_rule":"The winch requires 1 power cell to activate. Activation is allowed only with enough cells; otherwise nothing is spent.",
     "expect":"The operator replaces the rule document as announced. The earlier document remains in history but is absent from current retrieval."},
    {"id":"affordable", "channel":"system", "role":"player", "speaker":"Player", "visibility":"public",
     "text":"Using the revised rule, can Nora afford one activation, and what would her remaining balance be? I have not actually activated it.",
     "expect":"Two available, cost one, proposed remaining one. The recorded inventory stays at two."},
    {"id":"private-clue", "channel":"mic", "role":"facilitator", "speaker":"Facilitator", "visibility":"private",
     "text":"Privately, only Nora notices a red mark inside Mara's coat. Leo has not seen the mark. Nora has not told him about it.",
     "expect":"Nora can know the clue; Leo cannot. Keep the clue out of shared narration."},
    {"id":"changed-task", "channel":"system", "role":"player", "speaker":"Player", "visibility":"public",
     "text":"I keep the mark to myself. Forget the winch for now. I tell Mara that we can look for the operator at the east gate, and ask what she is worried about.",
     "expect":"Return to NPC dialogue; give Mara a plausible response. Keep the gate closed and the inventory unchanged, and do not tell Leo about the private mark."},
]

OLD_RULE = "The winch requires 4 power cells to activate. Activation is allowed only with enough cells; otherwise nothing is spent."


def seed(campaigns):
    cid=campaigns.create("The closed floodgate", direction="Offer concise, playable private guidance. Mara is an NPC controlled by the facilitator. Her worry is the missing operator; do not resolve the mystery offstage.", style="Plain speech, restrained atmosphere, a distinct practical NPC voice.")
    nora=campaigns.save_character(cid,"Nora",{"resources":{"power_cells":6},"occupation":"mechanic"})
    campaigns.save_character(cid,"Leo",{"resources":{"power_cells":3},"occupation":"surveyor"})
    campaigns.map_participant(cid,"Player","Player",role="player",character_id=nora)
    campaigns.set_rule_profile(cid,{"resource_aliases":{"cells":"power_cells","power_cell":"power_cells","charges":"power_cells"},"tools":[]})
    rule=campaigns.add_document(cid,"Winch operating rule",OLD_RULE,visibility="public",metadata={"kind":"rules"})
    campaigns.add_document(cid,"Scene","A closed floodgate, an idle winch, and a caretaker named Mara. Mara last saw the missing operator at the east gate. No one knows who, if anyone, accompanied the operator.",visibility="public")
    sid=campaigns.create_session(cid,"The east gate")
    return {"campaign_id":cid,"session_id":sid,"character_id":nora,"rule_id":rule}
