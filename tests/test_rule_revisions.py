import pytest

from story_copilot.campaigns import Campaigns
from story_copilot.rules import active_documents, search_rules
from story_copilot.store import Store


def test_revised_material_keeps_history_but_only_current_rule_is_retrieved(tmp_path):
    store=Store(tmp_path/'data');c=Campaigns(store);cid=c.create('Original scenario');sid=c.create_session(cid,'Start')
    old=c.add_document(cid,'Beacon v1','Beacon use costs 8 cells.',metadata={'kind':'rules'},visibility='public')
    before=c.snapshot(sid)
    new=c.add_document(cid,'Beacon v2','Beacon use costs 6 cells.',metadata={'kind':'rules','supersedes':old},visibility='public')
    after=c.snapshot(sid)
    assert {d['id'] for d in after['documents']}=={old,new}
    assert [d['id'] for d in active_documents(after)]==[new]
    assert search_rules(store,'beacon cost',snapshot=before)[0]['text']=='Beacon use costs 8 cells.'
    assert search_rules(store,'beacon cost',snapshot=after)[0]['text']=='Beacon use costs 6 cells.'
    assert c.evidence_hash(before)!=c.evidence_hash(after)
    with pytest.raises(ValueError,match='already replaced'):
        c.add_document(cid,'Conflicting replacement','A different cost.',metadata={'kind':'rules','supersedes':old})


def test_replacement_cannot_cross_campaigns_or_reveal_a_private_update(tmp_path):
    c=Campaigns(Store(tmp_path/'data'));a,b=c.create('A'),c.create('B')
    old=c.add_document(a,'Public rule','Old public text',visibility='public',metadata={'kind':'rules'})
    with pytest.raises(ValueError,match='this campaign'):
        c.add_document(b,'New','New',metadata={'kind':'rules','supersedes':old})
    c.add_document(a,'Private draft rule','Secret draft',metadata={'kind':'rules','supersedes':old})
    visible=active_documents({'documents':c.documents(a,public_only=True)})
    assert [d['id'] for d in visible]==[old]
