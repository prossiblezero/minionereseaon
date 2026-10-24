"""Whole-event history selection; no target-dependent arguments."""
import re
from data import TASK, SYSTEM, codes


def render_user(view, indices):
    selected = set(indices)
    text = view['context_text']
    for i in reversed(range(len(view['history']))):
        if i not in selected:
            a,b=view['history'][i]['span']
            text=text[:a]+text[b:]
    text=re.sub(r'(?:,\s*){2,}', ', ', text)
    text=re.sub(r'([:：])\s*,\s*', r'\1 ', text)
    return text.strip()+'\n\n'+TASK


def prompt_text(view, tok, indices, thinking):
    messages=[dict(role='system',content=SYSTEM),dict(role='user',content=render_user(view,indices))]
    result=tok.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=thinking)
    if thinking:
        result+='<think>\n'
    return result


def select_history(view,tok,budget,strategy='recent'):
    events=view['history']; all_indices=list(range(len(events)))
    def cost(ids):
        return len(tok.encode(prompt_text(view,tok,ids,False),add_special_tokens=False))
    def fits(ids):
        return cost(ids)<=budget
    if fits(all_indices):
        return all_indices
    if not fits([]):
        raise ValueError('non_event_context_exceeds_input_budget')
    reverse=all_indices[::-1]
    base=cost([]);available=budget-base;selected=[]
    def fill(priority,ceiling):
        for i in priority:
            if i in selected:continue
            candidate=sorted(selected+[i])
            if cost(candidate)<=ceiling:
                selected.append(i)
    product=[i for i in reverse if events[i]['domain']=='prod']
    if strategy=='domain_recent':
        fill(product,base+int(available*.5))
    elif strategy=='balanced':
        fill(reverse,base+int(available*.5))
        fill(product,base+int(available*.75))
        seen={(events[i]['domain'],codes(events[i]['sid'])[0]) for i in selected}
        diverse=[]
        for i in reverse:
            key=(events[i]['domain'],codes(events[i]['sid'])[0])
            if key not in seen:
                seen.add(key);diverse.append(i)
        fill(diverse,budget)
    elif strategy!='recent':
        raise ValueError(strategy)
    fill(reverse,budget)
    selected.sort()
    return selected


def history_gate(report):
    fractions=report['length_gate']['test']['trimmed_fraction']
    for budget in (2048,1024,512):
        if fractions[str(budget)]>=0.3:
            return budget
    raise RuntimeError('History gate failed: reconstruct timestamped UserProfile evaluation before running this experiment')
