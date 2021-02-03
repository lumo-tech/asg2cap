from thexp.nest.trainer.losses import *


class CaptionLoss(CELoss):

    def loss_cap_(self, logits, caption_ids, caption_masks,
                  reduce_mean=True,
                  meter: Meter = None, name: str = 'Lcap'):
        """
          logits: shape=(batch*(seq_len-1), num_words)
          caption_ids: shape=(batch, seq_len)
          caption_masks: shape=(batch, seq_len)
        """
        batch_size, seq_len = caption_ids.size()
        losses = F.cross_entropy(logits, caption_ids[:, 1:].contiguous().view(-1),
                                 reduction='none')
        onehot_caption_masks = caption_masks[:, 1:] > 0
        onehot_caption_masks = onehot_caption_masks.float()
        caption_masks = caption_masks[:, 1:].reshape(-1).float()
        if reduce_mean:
            loss = torch.sum(losses * caption_masks) / torch.sum(onehot_caption_masks)
        else:
            loss = torch.div(
                torch.sum((losses * caption_masks).view(batch_size, seq_len - 1), 1),
                torch.sum(onehot_caption_masks, 1)
            )
        if meter is not None:
            meter[name] = loss

        return loss
