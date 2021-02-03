from thexp.nest.trainer.acc import *
from eval_cap.bleu.bleu_scorer import BleuScorer
from trainers import GlobalParams
from eval_cap.cider.cider import Cider


class AccBleu(AccMixin):
    def acc_bleu_(self, gts, res, vid_order=None, n=4,
                  meter: Meter = None, name='Ablue'):
        # assert(gts.keys() == res.keys())
        if vid_order is None:
            vid_order = gts.keys()

        bleu_scorer = BleuScorer(n=n)
        for id in vid_order:
            hypo = res[id]
            ref = gts[id]

            # Sanity check.
            assert (type(hypo) is list)
            assert (len(hypo) == 1)
            assert (type(ref) is list)
            assert (len(ref) > 0)

            bleu_scorer += (hypo[0], ref)

        # score, scores = bleu_scorer.compute_score(option='shortest')
        score, scores = bleu_scorer.compute_score(option='closest', verbose=0)
        # score, scores = bleu_scorer.compute_score(option='average', verbose=1)

        # return (bleu, bleu_info)
        # return score, scores
        if meter is not None:
            meter[name] = score[-1] * 100
        return score


class AccCider():
    def acc_cider_(self, vid2refs, vid2tsts, vid_order=None, is_init_refs=False,
                   meter: Meter = None, name='Acider'):
        cider = Cider()

        if not is_init_refs:
            cider.init_refs(vid2refs)
        if vid_order is None:
            vid_order = vid2refs.keys()
        score, scores = cider.compute_cider(vid2refs, vid2tsts, vid_order)
        if meter is not None:
            meter[name] = score * 100
        return score, scores
