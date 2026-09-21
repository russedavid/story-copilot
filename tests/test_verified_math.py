import pytest

from story_copilot.mechanics import numeric_sources, grounded_calculation, calculation_profile
from story_copilot.rule_advice import Calculation, RulesAnswer, display_advice, validate_advice


def evidence():
    messages=[{"id":"correction","role":"facilitator","text":"Mira now has 2 charges, not 12."}]
    state={"resources":{"Mira:charges":{"value":2}}}
    rules=[{"id":"rule","text":"The obsolete cost was 8 charges. The revised cost is 6 charges; activation requires enough and otherwise spends nothing."}]
    return numeric_sources(messages,state,rules,actors=["Mira"]),rules


def test_current_cost_rejects_obsolete_and_negated_values_but_computes_a_valid_denial():
    sources,rules=evidence();profile=calculation_profile({"tools":[]})
    assert next(x for x in sources if x['id']=='rule:rule:0')['status']=='superseded'
    assert next(x for x in sources if x['id']=='message:correction:1')['status']=='superseded'
    call=Calculation(tool='balance_after_cost',inputs=[{'source':'resource:Mira:charges','value':2},{'source':'rule:rule:1','value':6}])
    result=grounded_calculation(call,profile,sources)
    assert result['result']=={'allowed':False,'remaining':2}
    advice=RulesAnswer(answer='Untrusted model arithmetic',citations=[{'id':'rule','quote':rules[0]['text']}],calculation=call,missing_information=[])
    assert 'Remaining balance: 2' in display_advice(advice,result)
    for source,value in [('rule:rule:0',8),('message:correction:1',12)]:
        wrong=call.model_copy(deep=True);wrong.inputs[1].source=source;wrong.inputs[1].value=value
        with pytest.raises(ValueError,match='superseded'):
            grounded_calculation(wrong,profile,sources)


def test_historical_comparison_is_explicit_and_cannot_authorize_spending():
    sources,_=evidence();profile=calculation_profile({})
    call=Calculation(tool='arithmetic_difference',scope='historical',inputs=[{'source':'rule:rule:0','value':8},{'source':'rule:rule:1','value':6}])
    assert grounded_calculation(call,profile,sources)['result']==2
    call.tool='balance_after_cost'
    with pytest.raises(ValueError,match='Historical'):
        grounded_calculation(call,profile,sources)


def test_context_words_do_not_make_current_numbers_obsolete_and_multiple_characters_remain_named():
    rules=[{'id':'bridge','text':'The old bridge costs 5 supplies to repair. It needs not less than 3 workers.'}]
    sources=numeric_sources([],{'resources':{'A:charge':{'value':1},'B:charge':{'value':4}}},rules,actors=['A','B'])
    assert all(x['status']!='superseded' for x in sources)
    assert {x['id'] for x in sources} >= {'resource:A:charge','resource:B:charge'}


def test_numerical_answer_needs_calculation_when_inputs_exist():
    sources,rules=evidence()
    answer=RulesAnswer(answer='Yes, spend it.',citations=[{'id':'rule','quote':rules[0]['text']}],calculation=None,missing_information=[])
    with pytest.raises(ValueError,match='calculator'):
        validate_advice(answer,rules,calculation_profile({}),sources,question='Can Mira afford another use?')
