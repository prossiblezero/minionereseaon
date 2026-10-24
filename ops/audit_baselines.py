"""Recompute whether held-out targets have any train-only ID support."""
import collections
import sys
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from data import CODE,ROOT,load_catalog,load_split,parse_answer,read_jsonl,write_json


def main():
    train=load_split('main','train');catalog=load_catalog()
    targets={r['train_target'] for r in train}
    history={e['sid'] for r in train for e in r['history']}
    counts=collections.Counter(r['train_target'] for r in train)
    popular={sid for sid,_ in sorted(counts.items(),key=lambda item:(-item[1],item[0]))[:64]}
    result=dict(training_target_unique=len(targets),training_history_unique=len(history),catalog_sids=len(catalog),splits={},
                interpretation='Cold-start and training support are measured separately; all-zero baseline remains a weak reference, not evidence of broad superiority.')
    for split in ('validation','test'):
        rows=load_split('main',split);by_id={r['sample_id']:r for r in rows}
        predictions=read_jsonl(ROOT/'predictions/main'/f'sasrec_42_{split}.jsonl')
        assert {p['sample_id'] for p in predictions}==by_id.keys()
        valid=attempts=hits=trained=0
        for prediction in predictions:
            recommendations=[parse_answer(o['text'],catalog)[0] for o in prediction['outputs']]
            attempts+=len(recommendations);valid+=sum(sid is not None for sid in recommendations)
            trained+=sum(sid in targets for sid in recommendations)
            hits+=bool(set(recommendations)&set(by_id[prediction['sample_id']]['targets']))
        result['splits'][split]=dict(examples=len(rows),target_seen_as_training_target=sum(bool(set(r['targets'])&targets) for r in rows),
            target_seen_in_training_history=sum(bool(set(r['targets'])&history) for r in rows),
            target_unseen_in_training=sum(not bool(set(r['targets'])&(targets|history)) for r in rows),
            popularity_hits64=sum(bool(set(r['targets'])&popular) for r in rows),sasrec_valid_attempts=valid,sasrec_attempts=attempts,
            sasrec_hits64=hits,sasrec_recommendations_seen_as_training_target=trained)
    write_json(CODE/'reports/baseline_support_audit.json',result);print(result)


if __name__=='__main__':main()
