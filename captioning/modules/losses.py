from numpy import dtype
import torch
import torch.nn as nn
from ..utils.rewards import get_scores, get_self_cider_scores


class RewardCriterion(nn.Module):
    def __init__(self):
        super(RewardCriterion, self).__init__()

    def forward(self, input, seq, reward, reduction='mean'):
        N,L = input.shape[:2]
        input = input.gather(2, seq.unsqueeze(2)).squeeze(2)
        
        input = input.reshape(-1)
        reward = reward.reshape(-1)
        mask = (seq>0).to(input)
        mask = torch.cat([mask.new(mask.size(0), 1).fill_(1), mask[:, :-1]], 1).reshape(-1)
        output = - input * reward * mask
        
        if reduction == 'none':
            output = output.view(N,L).sum(1) / mask.view(N,L).sum(1)
        elif reduction == 'mean':
            output = torch.sum(output) / torch.sum(mask)

        return output


class StructureLosses(nn.Module):
    """
    This loss is inspired by Classical Structured Prediction Losses for Sequence to Sequence Learning (Edunov et al., 2018).
    """
    def __init__(self, opt):
        super(StructureLosses, self).__init__()
        self.opt = opt
        self.loss_type = opt.structure_loss_type

    def forward(self, input, seq, data_gts, reduction='mean'):
        """
        Input is either logits or log softmax
        """
        out = {}

        batch_size = input.size(0)# batch_size = sample_size * seq_per_img
        seq_per_img = batch_size // len(data_gts)

        assert seq_per_img == self.opt.train_sample_n, seq_per_img

        mask = (seq>0).to(input)
        mask = torch.cat([mask.new_full((mask.size(0), 1), 1), mask[:, :-1]], 1)
        
        scores = get_scores(data_gts, seq, self.opt)
        scores = torch.from_numpy(scores).type_as(input).view(-1, seq_per_img)
        out['reward'] = scores #.mean()
        if self.opt.entropy_reward_weight > 0:
            entropy = - (F.softmax(input, dim=2) * F.log_softmax(input, dim=2)).sum(2).data
            entropy = (entropy * mask).sum(1) / mask.sum(1)
            print('entropy', entropy.mean().item())
            scores = scores + self.opt.entropy_reward_weight * entropy.view(-1, seq_per_img)
        # rescale cost to [0,1]
        costs = - scores
        if self.loss_type == 'risk' or self.loss_type == 'softmax_margin': 
            costs = costs - costs.min(1, keepdim=True)[0]
            costs = costs / costs.max(1, keepdim=True)[0]
        # in principle
        # Only risk need such rescale
        # margin should be alright; Let's try.

        # Gather input: BxTxD -> BxT
        input = input.gather(2, seq.unsqueeze(2)).squeeze(2)

        if self.loss_type == 'seqnll':
            # input is logsoftmax
            input = input * mask
            input = input.sum(1) / mask.sum(1)
            input = input.view(-1, seq_per_img)

            target = costs.min(1)[1]
            output = F.cross_entropy(input, target, reduction=reduction)
        elif self.loss_type == 'risk':
            # input is logsoftmax
            input = input * mask
            input = input.sum(1)
            input = input.view(-1, seq_per_img)

            output = (F.softmax(input.exp()) * costs).sum(1).mean()
            assert reduction=='mean'

            # test
            # avg_scores = input
            # probs = F.softmax(avg_scores.exp_())
            # loss = (probs * costs.type_as(probs)).sum() / input.size(0)
            # print(output.item(), loss.item())            

        elif self.loss_type == 'max_margin':
            # input is logits
            input = input * mask
            input = input.sum(1) / mask.sum(1)
            input = input.view(-1, seq_per_img)
            _, __ = costs.min(1, keepdim=True)
            costs_star = _
            input_star = input.gather(1, __)
            output = F.relu(costs - costs_star - input_star + input).max(1)[0] / 2
            output = output.mean()
            assert reduction=='mean'

            # sanity test
            # avg_scores = input + costs
            # scores_with_high_target = avg_scores.clone()
            # scores_with_high_target.scatter_(1, costs.min(1)[1].view(-1, 1), 1e10)

            # target_and_offender_index = scores_with_high_target.sort(1, True)[1][:, 0:2]
            # avg_scores = avg_scores.gather(1, target_and_offender_index)
            # target_index = avg_scores.new_zeros(avg_scores.size(0), dtype=torch.long)
            # loss = F.multi_margin_loss(avg_scores, target_index, size_average=True, margin=0)
            # print(loss.item() * 2, output.item()) 

        elif self.loss_type == 'multi_margin':
            # input is logits
            input = input * mask
            input = input.sum(1) / mask.sum(1)
            input = input.view(-1, seq_per_img)
            _, __ = costs.min(1, keepdim=True)
            costs_star = _
            input_star = input.gather(1, __)
            output = F.relu(costs - costs_star - input_star + input)
            output = output.mean()
            assert reduction=='mean'

            # sanity test
            # avg_scores = input + costs
            # loss = F.multi_margin_loss(avg_scores, costs.min(1)[1], margin=0)
            # print(output, loss)

        elif self.loss_type == 'softmax_margin':
            # input is logsoftmax
            input = input * mask
            input = input.sum(1) / mask.sum(1)
            input = input.view(-1, seq_per_img)

            input = input + costs
            target = costs.min(1)[1]
            output = F.cross_entropy(input, target, reduction=reduction)

        elif self.loss_type == 'real_softmax_margin':
            # input is logits
            # This is what originally defined in Kevin's paper
            # The result should be equivalent to softmax_margin
            input = input * mask
            input = input.sum(1) / mask.sum(1)
            input = input.view(-1, seq_per_img)

            input = input + costs
            target = costs.min(1)[1]
            output = F.cross_entropy(input, target, reduction=reduction)

        elif self.loss_type == 'new_self_critical':
            """
            A different self critical
            Self critical uses greedy decoding score as baseline;
            This setting uses the average score of the rest samples as baseline
            (suppose c1...cn n samples, reward1 = score1 - 1/(n-1)(score2+..+scoren) )
            """
            baseline = (scores.sum(1, keepdim=True) - scores) / (scores.shape[1] - 1)
            scores = scores - baseline
            # self cider used as reward to promote diversity (not working that much in this way)
            if getattr(self.opt, 'self_cider_reward_weight', 0) > 0:
                _scores = get_self_cider_scores(data_gts, seq, self.opt)
                _scores = torch.from_numpy(_scores).type_as(scores).view(-1, 1)
                _scores = _scores.expand_as(scores - 1)
                scores += self.opt.self_cider_reward_weight * _scores
            output = - input * mask * scores.view(-1, 1)
            if reduction == 'none':
                output = output.sum(1) / mask.sum(1)
            elif reduction == 'mean':
                output = torch.sum(output) / torch.sum(mask)

        out['loss'] = output
        return out

class LanguageModelCriterion(nn.Module):
    def __init__(self):
        super(LanguageModelCriterion, self).__init__()

    def forward(self, input, target, mask, reduction='mean'):
        # input: 5B * len * vocab
        # target: 5B * len
        # mask: 5B * len
        if target.ndim == 3:
            target = target.reshape(-1, target.shape[2])
            mask = mask.reshape(-1, mask.shape[2])
        N,L = input.shape[:2] # N: 5B  L:len
        # truncate to the same size
        target = target[:, :input.size(1)]
        mask = mask[:, :input.size(1)].to(input)

        output = -input.gather(2, target.unsqueeze(2)).squeeze(2) * mask

        if reduction == 'none':
            output = output.view(N,L).sum(1) / mask.view(N,L).sum(1)
        elif reduction == 'mean':
            output = torch.sum(output) / torch.sum(mask)

        return output

class LanguageModelCriterion_NNAIC(nn.Module):
    def __init__(self):
        super(LanguageModelCriterion_NNAIC, self).__init__()

    def forward(self, length_N, word_logprob, target, mask, reduction='mean'):
        # length_N: 5B
        # word_logprob: 5B * len * vocab
        # target: 5B * len
        # mask: 5B * len
        if target.ndim == 3:
            target = target.reshape(-1, target.shape[2])
            mask = mask.reshape(-1, mask.shape[2])
        N,L = word_logprob.shape[:2] # N: 5B  L:len
        # truncate to the same size
        target = target[:, :word_logprob.size(1)]
        mask = mask[:, :word_logprob.size(1)].to(word_logprob)
        word_loss = -word_logprob.gather(2, target.unsqueeze(2)).squeeze(2) * mask

        length = mask.sum(1)
        length_loss_f = nn.MSELoss(reduction='none')
        length_loss = length_loss_f(length, length_N)

        if reduction == 'none':
            output = (word_loss.view(N,L).sum(1) + (length_loss * 0.1)) / mask.view(N,L).sum(1)
        elif reduction == 'mean':
            output = (torch.sum(word_loss) + torch.sum(length_loss)) / torch.sum(mask)

        return output

class PB_pad_LanguageModelCriterion(nn.Module):
    def __init__(self):
        super(PB_pad_LanguageModelCriterion, self).__init__()
    
    def forward(self, phrase_label, predict_phrase_logprob, phrase_mask, phrase_num, phrase_length_label, predict_phrase_length, predict_phrase_length_logprob, reduction='mean'):
        if phrase_label.ndim == 3:
            phrase_label = phrase_label.reshape(-1, phrase_label.shape[2])
            phrase_mask = phrase_mask.reshape(-1, phrase_mask.shape[2])
            phrase_num = phrase_num.reshape(-1)
            phrase_length_label = phrase_length_label.reshape(-1, phrase_length_label.shape[2])

        N, PL = predict_phrase_logprob.shape[0:2]
        LL = predict_phrase_length_logprob.shape[1]
        
        phrase_loss = -predict_phrase_logprob.gather(2, phrase_label.unsqueeze(2)).squeeze(2) * phrase_mask

        length_mask = predict_phrase_length_logprob.new_full([N, LL], False, dtype=torch.bool)
        for i in range(0, N):
            length_mask[i, 0:phrase_num[i] ] = True
        # length_loss_f = nn.MSELoss(reduction='none')
        length_loss = -predict_phrase_length_logprob.gather(2, phrase_length_label.unsqueeze(2)).squeeze(2) * length_mask
        # length_loss = length_loss_f(predict_phrase_length.to(torch.float), phrase_length_label.to(torch.float)) * length_mask

        # print(predict_phrase_length[0:3])
        # print(phrase_length_label[0:3])
        # print(length_loss[0:3])

        if reduction == 'none':
            output = (phrase_loss.view(N, PL).sum(1) + length_loss.view(N, LL).sum(1) ) / phrase_mask.view(N, PL).sum(1)
            length_loss_mean = None
            phrase_loss_mean = None
        elif reduction == 'mean':
            length_loss_mean = torch.sum(length_loss) / torch.sum(phrase_mask)
            phrase_loss_mean = torch.sum(phrase_loss) / torch.sum(phrase_mask)
            output = length_loss_mean + phrase_loss_mean

        return output, length_loss_mean, phrase_loss_mean

class LanguageModelCriterion_NAIC(nn.Module):
    def __init__(self):
        super(LanguageModelCriterion_NAIC, self).__init__()

    def forward(self, predict_phrase_length_logprob, predict_phrase_syn_logprob, predict_phrase_logprob,\
        phrase_num, phrase_length_label, phrase_syn_label, phrase_label, reduction='mean'):
        if phrase_length_label.ndim == 3:
            phrase_num = phrase_num.reshape(-1)
            phrase_length_label = phrase_length_label.reshape(-1, phrase_length_label.shape[2])
            phrase_syn_label = phrase_syn_label.reshape(-1, phrase_syn_label.shape[2])
            phrase_label = phrase_label.reshape(-1, phrase_label.shape[2])
        
        B = phrase_label.shape[0]
        real_phrase_label = phrase_label[:, 1:-1]
        phrase_mask = predict_phrase_logprob.new_full(real_phrase_label.shape, False, dtype=torch.bool)
        for i in range(B):
            phrase_mask[i, 0:sum(phrase_length_label[i])-1] = True # because phrase has no eos/bos to compare
        phrase_loss = -predict_phrase_logprob.gather(2, real_phrase_label.unsqueeze(2)).squeeze(2) * phrase_mask

        real_phrase_length_label = phrase_length_label[:, 1:]
        real_phrase_syn_label = phrase_syn_label[:, 1:]
        phrase_label_mask = predict_phrase_length_logprob.new_full(real_phrase_length_label.shape, False, dtype=torch.bool)
        phrase_syn_mask = predict_phrase_syn_logprob.new_full(real_phrase_syn_label.shape, False, dtype=torch.bool)
        for i in range(B):
            phrase_label_mask[i, 0:phrase_num[i]] = True
            phrase_syn_mask[i, 0:phrase_num[i]] = True
        phrase_length_loss = -predict_phrase_length_logprob.gather(2, real_phrase_length_label.unsqueeze(2)).squeeze(2) * phrase_label_mask
        phrase_syn_loss = -predict_phrase_syn_logprob.gather(2, real_phrase_syn_label.unsqueeze(2)).squeeze(2) * phrase_syn_mask

        if reduction == 'none':
            output = (phrase_loss.sum(1) + phrase_length_loss.sum(1) + phrase_syn_loss.sum(1)) / phrase_mask.sum(1)
            length_loss_mean = None
            phrase_loss_mean = None
            syn_loss_mean = None
        elif reduction == 'mean':
            length_loss_mean = torch.sum(phrase_length_loss) / torch.sum(phrase_mask)
            phrase_loss_mean = torch.sum(phrase_loss) / torch.sum(phrase_mask)
            syn_loss_mean = torch.sum(phrase_syn_loss) / torch.sum(phrase_mask)
            output = length_loss_mean + phrase_loss_mean + syn_loss_mean
        return output, length_loss_mean, phrase_loss_mean, syn_loss_mean


class LanguageModelCriterion_UIC(nn.Module):
    def __init__(self):
        super(LanguageModelCriterion_UIC, self).__init__()

    def forward(self, SA_predict_phrase_length_logprob, SA_predict_phrase_syn_logprob, SA_predict_phrase_logprob, \
        NA_predict_phrase_length_logprob, NA_predict_phrase_syn_logprob, NA_predict_phrase_logprob, \
        phrase_num, phrase_length_label, phrase_syn_label, phrase_label, reduction='mean', self_dis=False):
        if phrase_length_label.ndim == 3:
            phrase_num = phrase_num.reshape(-1)
            phrase_length_label = phrase_length_label.reshape(-1, phrase_length_label.shape[2])
            phrase_syn_label = phrase_syn_label.reshape(-1, phrase_syn_label.shape[2])
            phrase_label = phrase_label.reshape(-1, phrase_label.shape[2])
        
        B = phrase_label.shape[0]
        real_phrase_label = phrase_label[:, 1:-1]
        phrase_mask = SA_predict_phrase_logprob.new_full(real_phrase_label.shape, False, dtype=torch.bool)
        for i in range(B):
            phrase_mask[i, 0:sum(phrase_length_label[i])-1] = True # because phrase has no eos/bos to compare
        SA_phrase_loss = -SA_predict_phrase_logprob.gather(2, real_phrase_label.unsqueeze(2)).squeeze(2) * phrase_mask
        NA_phrase_loss = -NA_predict_phrase_logprob.gather(2, real_phrase_label.unsqueeze(2)).squeeze(2) * phrase_mask

        if self_dis:
            SA_predict_phrase_prob = torch.exp(SA_predict_phrase_logprob)
            KL_loss_cal = nn.KLDivLoss(reduction='none')
            KL_loss = KL_loss_cal(NA_predict_phrase_logprob, SA_predict_phrase_prob.detach()) * phrase_mask.unsqueeze(2)

        real_phrase_length_label = phrase_length_label[:, 1:]
        real_phrase_syn_label = phrase_syn_label[:, 1:]
        phrase_label_mask = SA_predict_phrase_length_logprob.new_full(real_phrase_length_label.shape, False, dtype=torch.bool)
        phrase_syn_mask = SA_predict_phrase_syn_logprob.new_full(real_phrase_syn_label.shape, False, dtype=torch.bool)
        for i in range(B):
            phrase_label_mask[i, 0:phrase_num[i]] = True
            phrase_syn_mask[i, 0:phrase_num[i]] = True
        SA_phrase_length_loss = -SA_predict_phrase_length_logprob.gather(2, real_phrase_length_label.unsqueeze(2)).squeeze(2) * phrase_label_mask
        SA_phrase_syn_loss = -SA_predict_phrase_syn_logprob.gather(2, real_phrase_syn_label.unsqueeze(2)).squeeze(2) * phrase_syn_mask
        NA_phrase_length_loss = -NA_predict_phrase_length_logprob.gather(2, real_phrase_length_label.unsqueeze(2)).squeeze(2) * phrase_label_mask
        NA_phrase_syn_loss = -NA_predict_phrase_syn_logprob.gather(2, real_phrase_syn_label.unsqueeze(2)).squeeze(2) * phrase_syn_mask

        if reduction == 'none':
            output = (SA_phrase_loss.sum(1) + SA_phrase_length_loss.sum(1) + SA_phrase_syn_loss.sum(1) + NA_phrase_loss.sum(1) + NA_phrase_length_loss.sum(1) + NA_phrase_syn_loss.sum(1)) / phrase_mask.sum(1)
            length_loss_mean = None
            phrase_loss_mean = None
            syn_loss_mean = None
        elif reduction == 'mean':
            SA_length_loss_mean = torch.sum(SA_phrase_length_loss) / torch.sum(phrase_mask)
            SA_phrase_loss_mean = torch.sum(SA_phrase_loss) / torch.sum(phrase_mask)
            SA_syn_loss_mean = torch.sum(SA_phrase_syn_loss) / torch.sum(phrase_mask)
            NA_length_loss_mean = torch.sum(NA_phrase_length_loss) / torch.sum(phrase_mask)
            NA_phrase_loss_mean = torch.sum(NA_phrase_loss) / torch.sum(phrase_mask)
            NA_syn_loss_mean = torch.sum(NA_phrase_syn_loss) / torch.sum(phrase_mask)
            output = SA_length_loss_mean + SA_phrase_loss_mean + SA_syn_loss_mean + NA_length_loss_mean + NA_phrase_loss_mean + NA_syn_loss_mean
            if self_dis:
                KL_loss_mean = torch.sum(KL_loss) / torch.sum(phrase_mask)
                output += KL_loss_mean
        return output, SA_length_loss_mean, SA_phrase_loss_mean, SA_syn_loss_mean, NA_length_loss_mean, NA_phrase_loss_mean, NA_syn_loss_mean


class LanguageModelCriterion_TUIC(nn.Module):
    def __init__(self):
        super(LanguageModelCriterion_TUIC, self).__init__()

    def forward(self, predict_phrase_length_logprob, predict_phrase_syn_logprob, A_predict_phrase_prob, A_predict_phrase_logprob,
                    SA_predict_phrase_prob, SA_predict_phrase_logprob, NA_predict_phrase_logprob, 
                    phrase_num, phrase_length_label, phrase_syn_label, phrase_label, reduction='mean'):
        if phrase_length_label.ndim == 3:
            phrase_num = phrase_num.reshape(-1)
            phrase_length_label = phrase_length_label.reshape(-1, phrase_length_label.shape[2])
            phrase_syn_label = phrase_syn_label.reshape(-1, phrase_syn_label.shape[2])
            phrase_label = phrase_label.reshape(-1, phrase_label.shape[2])
        
        B = phrase_label.shape[0]
        real_phrase_label = phrase_label[:, 1:-1]
        phrase_mask = SA_predict_phrase_logprob.new_full(real_phrase_label.shape, False, dtype=torch.bool)
        for i in range(B):
            phrase_mask[i, 0:sum(phrase_length_label[i])-1] = True # because phrase has no eos/bos to compare
        A_phrase_loss = -A_predict_phrase_logprob.gather(2, real_phrase_label.unsqueeze(2)).squeeze(2) * phrase_mask
        SA_phrase_loss = -SA_predict_phrase_logprob.gather(2, real_phrase_label.unsqueeze(2)).squeeze(2) * phrase_mask
        NA_phrase_loss = -NA_predict_phrase_logprob.gather(2, real_phrase_label.unsqueeze(2)).squeeze(2) * phrase_mask

        KL_loss = nn.KLDivLoss(reduction='none')
        SA_KL_loss = KL_loss(SA_predict_phrase_logprob, A_predict_phrase_prob.detach()) * phrase_mask.unsqueeze(2)
        NA_KL_loss = (KL_loss(NA_predict_phrase_logprob, SA_predict_phrase_prob.detach()) + KL_loss(NA_predict_phrase_logprob, A_predict_phrase_prob.detach()))* phrase_mask.unsqueeze(2)

        real_phrase_length_label = phrase_length_label[:, 1:]
        real_phrase_syn_label = phrase_syn_label[:, 1:]
        phrase_length_mask = predict_phrase_length_logprob.new_full(real_phrase_length_label.shape, False, dtype=torch.bool)
        phrase_syn_mask = predict_phrase_syn_logprob.new_full(real_phrase_syn_label.shape, False, dtype=torch.bool)
        for i in range(B):
            phrase_length_mask[i, 0:phrase_num[i]] = True
            phrase_syn_mask[i, 0:phrase_num[i]] = True
        phrase_length_loss = -predict_phrase_length_logprob.gather(2, real_phrase_length_label.unsqueeze(2)).squeeze(2) * phrase_length_mask
        phrase_syn_loss = -predict_phrase_syn_logprob.gather(2, real_phrase_syn_label.unsqueeze(2)).squeeze(2) * phrase_syn_mask

        if reduction == 'none':
            # output = (SA_phrase_loss.sum(1) + phrase_length_loss.sum(1) + phrase_syn_loss.sum(1) + NA_phrase_loss.sum(1) + NA_phrase_length_loss.sum(1) + NA_phrase_syn_loss.sum(1)) / phrase_mask.sum(1)
            length_loss_mean = None
            phrase_loss_mean = None
            syn_loss_mean = None
        elif reduction == 'mean':
            length_loss_mean = torch.sum(phrase_length_loss) / torch.sum(phrase_mask)
            syn_loss_mean = torch.sum(phrase_syn_loss) / torch.sum(phrase_mask)
            A_phrase_loss_mean = torch.sum(A_phrase_loss) / torch.sum(phrase_mask)
            SA_phrase_loss_mean = torch.sum(SA_phrase_loss) / torch.sum(phrase_mask)
            NA_phrase_loss_mean = torch.sum(NA_phrase_loss) / torch.sum(phrase_mask)
            SA_KL_loss_mean = torch.sum(SA_KL_loss) / torch.sum(phrase_mask)
            NA_KL_loss_mean = torch.sum(NA_KL_loss) / torch.sum(phrase_mask)
            # output = length_loss_mean + syn_loss_mean + A_phrase_loss_mean
            output = length_loss_mean + syn_loss_mean + A_phrase_loss_mean + SA_phrase_loss_mean + NA_phrase_loss_mean + SA_KL_loss_mean + NA_KL_loss_mean
        return output, length_loss_mean, syn_loss_mean, A_phrase_loss_mean, SA_phrase_loss_mean, NA_phrase_loss_mean, SA_KL_loss_mean, NA_KL_loss_mean


class LabelSmoothing(nn.Module):
    "Implement label smoothing."
    def __init__(self, size=0, padding_idx=0, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(size_average=False, reduce=False)
        # self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        # self.size = size
        self.true_dist = None
        
    def forward(self, input, target, mask, reduction='mean'):
        N,L = input.shape[:2]
        # truncate to the same size
        target = target[:, :input.size(1)]
        mask =  mask[:, :input.size(1)]

        input = input.reshape(-1, input.size(-1))
        target = target.reshape(-1)
        mask = mask.reshape(-1).to(input)

        # assert x.size(1) == self.size
        self.size = input.size(1)
        # true_dist = x.data.clone()
        true_dist = input.data.clone()
        # true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.fill_(self.smoothing / (self.size - 1))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        # true_dist[:, self.padding_idx] = 0
        # mask = torch.nonzero(target.data == self.padding_idx)
        # self.true_dist = true_dist
        output = self.criterion(input, true_dist).sum(1) * mask
        
        if reduction == 'none':
            output = output.view(N,L).sum(1) / mask.view(N,L).sum(1)
        elif reduction == 'mean':
            output = torch.sum(output) / torch.sum(mask)

        return output
