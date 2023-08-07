import torch
import torch.nn as nn
from . import losses
from ..utils.rewards import init_scorer, get_self_critical_reward

class LossWrapper(torch.nn.Module):
    def __init__(self, model, opt):
        super(LossWrapper, self).__init__()
        self.opt = opt
        self.model = model
        self.train_mode = getattr(opt, 'train_mode', 'AIC')
        self.self_dis = getattr(opt, 'self_dis', False)
        self.rl_kl = getattr(opt, 'rl_kl', False)
        if self.train_mode == 'AIC' or self.train_mode == 'auto':
            if opt.label_smoothing > 0:
                self.crit = losses.LabelSmoothing(smoothing=opt.label_smoothing)
            else:
                self.crit = losses.LanguageModelCriterion()
        elif self.train_mode == 'NNAIC':
            self.crit = losses.LanguageModelCriterion_NNAIC()
        elif self.train_mode == 'NAIC' or self.train_mode == 'SAIC':
            self.crit = losses.LanguageModelCriterion_NAIC()
        elif self.train_mode == 'UIC' or self.train_mode == 'UIC_ds':
            self.crit = losses.LanguageModelCriterion_UIC()
        elif self.train_mode == 'UIC_s' or self.train_mode == 'UIC_u':
            self.crit = losses.LanguageModelCriterion_TUIC()
        elif self.train_mode == 'PB_pad':
            self.crit = losses.PB_pad_LanguageModelCriterion()
        self.rl_crit = losses.RewardCriterion()
        self.struc_crit = losses.StructureLosses(opt)

    def forward(self, fc_feats, att_feats, labels, masks, att_masks, gts, gt_indices,
                sc_flag, struc_flag, drop_worst_flag, phrase=None, phrase_num=None, 
                phrase_length=None, phrase_syn=None, extend_phrase_syn_seq=None, extend_phrase_seq=None, extend_phrase_seq_mask=None, glat_p=0.3):
        opt = self.opt
        
        out = {}

        reduction = 'none' if drop_worst_flag else 'mean'
        if self.train_mode == 'AIC' or self.train_mode == 'auto':
            if struc_flag:
                if opt.structure_loss_weight < 1:
                    lm_loss = self.crit(self.model(fc_feats, att_feats, labels[..., :-1], att_masks), labels[..., 1:], masks[..., 1:], reduction=reduction)
                else:
                    lm_loss = torch.tensor(0).type_as(fc_feats)
                if opt.structure_loss_weight > 0:
                    gen_result, sample_logprobs, t = self.model(fc_feats, att_feats, att_masks,
                        opt={'sample_method':opt.train_sample_method,
                            'beam_size':opt.train_beam_size,
                            'output_logsoftmax': opt.struc_use_logsoftmax or opt.structure_loss_type == 'softmax_margin'\
                                or not 'margin' in opt.structure_loss_type,
                            'sample_n': opt.train_sample_n},
                        mode='sample')
                    gts = [gts[_] for _ in gt_indices.tolist()]
                    struc_loss = self.struc_crit(sample_logprobs, gen_result, gts, reduction=reduction)
                else:
                    struc_loss = {'loss': torch.tensor(0).type_as(fc_feats),
                                'reward': torch.tensor(0).type_as(fc_feats)}
                loss = (1-opt.structure_loss_weight) * lm_loss + opt.structure_loss_weight * struc_loss['loss']
                out['lm_loss'] = lm_loss
                out['struc_loss'] = struc_loss['loss']
                out['reward'] = struc_loss['reward']
            elif not sc_flag:
                loss = self.crit(self.model(fc_feats, att_feats, labels[..., :-1], att_masks), labels[..., 1:], masks[..., 1:], reduction=reduction)
            else:
                self.model.eval()
                with torch.no_grad():
                    greedy_res, _ = self.model(fc_feats, att_feats, att_masks,
                        mode='sample',
                        opt={'sample_method': opt.sc_sample_method,
                            'beam_size': opt.sc_beam_size})
                self.model.train()
                gen_result, sample_logprobs = self.model(fc_feats, att_feats, att_masks,
                        opt={'sample_method':opt.train_sample_method,
                            'beam_size':opt.train_beam_size,
                            'sample_n': opt.train_sample_n},
                        mode='sample')
                gts = [gts[_] for _ in gt_indices.tolist()]
                reward = get_self_critical_reward(greedy_res, gts, gen_result, self.opt)
                reward = torch.from_numpy(reward).to(sample_logprobs)
                loss = self.rl_crit(sample_logprobs, gen_result.data, reward, reduction=reduction)
                out['reward'] = reward[:,0].mean()
        elif self.train_mode == 'NNAIC':
            if struc_flag:
                if opt.structure_loss_weight < 1:
                    length_N, word_logprob = self.model(fc_feats, att_feats, labels[..., :-1], att_masks)
                    lm_loss = self.crit(length_N, word_logprob, labels[..., 1:], masks[..., 1:], reduction=reduction)
                else:
                    lm_loss = torch.tensor(0).type_as(fc_feats)
                if opt.structure_loss_weight > 0:
                    seq, seq_logprobs, length_N, t = self.model(fc_feats, att_feats, att_masks,
                        opt={'sample_method':opt.train_sample_method, # sample
                            'beam_size':opt.train_beam_size,
                            'output_logsoftmax': opt.struc_use_logsoftmax or opt.structure_loss_type == 'softmax_margin'\
                                or not 'margin' in opt.structure_loss_type, # True when using nscl
                            'sample_n': opt.train_sample_n,
                            'train_mode': self.train_mode},
                        mode='sample')
                    gts = [gts[_] for _ in gt_indices.tolist()]
                    struc_loss = self.struc_crit(seq_logprobs, seq, gts, reduction=reduction)
                else:
                    struc_loss = {'loss': torch.tensor(0).type_as(fc_feats),
                                'reward': torch.tensor(0).type_as(fc_feats)}
                loss = (1-opt.structure_loss_weight) * lm_loss + opt.structure_loss_weight * struc_loss['loss']
                out['lm_loss'] = lm_loss
                out['struc_loss'] = struc_loss['loss']
                out['reward'] = struc_loss['reward']
            else:
                length_N, word_logprob = self.model(fc_feats, att_feats, labels[..., :-1], att_masks)
                loss = self.crit(length_N, word_logprob, labels[..., 1:], masks[..., 1:], reduction=reduction)
        elif self.train_mode == 'NAIC':
            if struc_flag:
                if opt.structure_loss_weight < 1:
                    predict_phrase_length_logprob, predict_phrase_syn_logprob, predict_phrase_logprob = \
                        self.model(fc_feats, att_feats, labels, att_masks, phrase_num, phrase_length, phrase_syn, extend_phrase_syn_seq, extend_phrase_seq, extend_phrase_seq_mask)
                    lm_loss, length_loss_mean, phrase_loss_mean, syn_loss_mean = self.crit(predict_phrase_length_logprob, predict_phrase_syn_logprob,\
                        predict_phrase_logprob, phrase_num, phrase_length, phrase_syn, labels, reduction=reduction)
                else:
                    lm_loss = torch.tensor(0).type_as(fc_feats)
                if opt.structure_loss_weight > 0:
                    seq, seq_logprobs, p_phrase_num, p_phrase_length, p_phrase_syn, t = self.model(fc_feats, att_feats, att_masks,
                        opt={'sample_method':opt.train_sample_method, # sample
                            'beam_size':opt.train_beam_size,
                            'output_logsoftmax': opt.struc_use_logsoftmax or opt.structure_loss_type == 'softmax_margin'\
                                or not 'margin' in opt.structure_loss_type, # True when using nscl
                            'sample_n': opt.train_sample_n,
                            'train_mode': self.train_mode},
                        mode='sample')
                    gts = [gts[_] for _ in gt_indices.tolist()]
                    struc_loss = self.struc_crit(seq_logprobs, seq, gts, reduction=reduction)
                else:
                    struc_loss = {'loss': torch.tensor(0).type_as(fc_feats),
                                'reward': torch.tensor(0).type_as(fc_feats)}
                loss = (1-opt.structure_loss_weight) * lm_loss + opt.structure_loss_weight * struc_loss['loss']
                out['lm_loss'] = lm_loss
                out['struc_loss'] = struc_loss['loss']
                out['reward'] = struc_loss['reward']
            else:
                predict_phrase_length_logprob, predict_phrase_syn_logprob, predict_phrase_logprob = \
                    self.model(fc_feats, att_feats, labels, att_masks, phrase_num, phrase_length, phrase_syn, extend_phrase_syn_seq, extend_phrase_seq, extend_phrase_seq_mask)
                loss, length_loss_mean, phrase_loss_mean, syn_loss_mean = self.crit(predict_phrase_length_logprob, predict_phrase_syn_logprob,\
                    predict_phrase_logprob, phrase_num, phrase_length, phrase_syn, labels, reduction=reduction)
                out['length_loss'] = length_loss_mean
                out['phrase_loss'] = phrase_loss_mean
                out['syn_loss'] = syn_loss_mean
        elif self.train_mode == 'SAIC':
            if struc_flag:
                if opt.structure_loss_weight < 1:
                    predict_phrase_length_logprob, predict_phrase_syn_logprob, predict_phrase_logprob = \
                        self.model(fc_feats, att_feats, labels, att_masks, phrase_num, phrase_length, phrase_syn, extend_phrase_syn_seq, extend_phrase_seq, extend_phrase_seq_mask)
                    lm_loss, length_loss_mean, phrase_loss_mean, syn_loss_mean = self.crit(predict_phrase_length_logprob, predict_phrase_syn_logprob,\
                        predict_phrase_logprob, phrase_num, phrase_length, phrase_syn, labels, reduction=reduction)
                else:
                    lm_loss = torch.tensor(0).type_as(fc_feats)
                if opt.structure_loss_weight > 0:
                    seq, seq_logprobs, p_phrase_num, p_phrase_length, p_phrase_syn, t = self.model(fc_feats, att_feats, att_masks,
                        opt={'sample_method':opt.train_sample_method, # sample
                            'beam_size':opt.train_beam_size,
                            'output_logsoftmax': opt.struc_use_logsoftmax or opt.structure_loss_type == 'softmax_margin'\
                                or not 'margin' in opt.structure_loss_type, # True when using nscl
                            'sample_n': opt.train_sample_n,
                            'train_mode': self.train_mode},
                        mode='sample')
                    gts = [gts[_] for _ in gt_indices.tolist()]
                    struc_loss = self.struc_crit(seq_logprobs, seq, gts, reduction=reduction)
                else:
                    struc_loss = {'loss': torch.tensor(0).type_as(fc_feats),
                                'reward': torch.tensor(0).type_as(fc_feats)}
                loss = (1-opt.structure_loss_weight) * lm_loss + opt.structure_loss_weight * struc_loss['loss']
                out['lm_loss'] = lm_loss
                out['struc_loss'] = struc_loss['loss']
                out['reward'] = struc_loss['reward']
            else:
                predict_phrase_length_logprob, predict_phrase_syn_logprob, predict_phrase_logprob = \
                    self.model(fc_feats, att_feats, labels, att_masks, phrase_num, phrase_length, phrase_syn, extend_phrase_syn_seq, extend_phrase_seq, extend_phrase_seq_mask)
                loss, length_loss_mean, phrase_loss_mean, syn_loss_mean = self.crit(predict_phrase_length_logprob, predict_phrase_syn_logprob,\
                    predict_phrase_logprob, phrase_num, phrase_length, phrase_syn, labels, reduction=reduction)
                out['length_loss'] = length_loss_mean
                out['phrase_loss'] = phrase_loss_mean
                out['syn_loss'] = syn_loss_mean
        elif self.train_mode == 'UIC' or self.train_mode == 'UIC_ds':
            if struc_flag:
                if opt.structure_loss_weight < 1:
                    SA_predict_phrase_length_logprob, SA_predict_phrase_syn_logprob, SA_predict_phrase_logprob, \
                        NA_predict_phrase_length_logprob, NA_predict_phrase_syn_logprob, NA_predict_phrase_logprob = \
                        self.model(fc_feats, att_feats, labels, att_masks, phrase_num, phrase_length, phrase_syn, extend_phrase_syn_seq, extend_phrase_seq, extend_phrase_seq_mask)
                    lm_loss, SA_length_loss_mean, SA_phrase_loss_mean, SA_syn_loss_mean, NA_length_loss_mean, NA_phrase_loss_mean, NA_syn_loss_mean = \
                        self.crit(SA_predict_phrase_length_logprob, SA_predict_phrase_syn_logprob, SA_predict_phrase_logprob, \
                            NA_predict_phrase_length_logprob, NA_predict_phrase_syn_logprob, NA_predict_phrase_logprob, \
                                phrase_num, phrase_length, phrase_syn, labels, reduction=reduction)
                else:
                    lm_loss = torch.tensor(0).type_as(fc_feats)
                if opt.structure_loss_weight > 0:
                    SAIC_seq, SAIC_seq_logprobs, p_phrase_num, p_phrase_length, p_phrase_syn, t = self.model(fc_feats, att_feats, att_masks,
                        opt={'sample_method':opt.train_sample_method, # sample
                            'beam_size':opt.train_beam_size,
                            'output_logsoftmax': opt.struc_use_logsoftmax or opt.structure_loss_type == 'softmax_margin'\
                                or not 'margin' in opt.structure_loss_type, # True when using nscl
                            'sample_n': opt.train_sample_n,
                            'train_mode': 'SAIC'},
                        mode='sample')
                    NAIC_seq, NAIC_seq_logprobs, p_phrase_num, p_phrase_length, p_phrase_syn, t = self.model(fc_feats, att_feats, att_masks,
                        opt={'sample_method':opt.train_sample_method, # sample
                            'beam_size':opt.train_beam_size,
                            'output_logsoftmax': opt.struc_use_logsoftmax or opt.structure_loss_type == 'softmax_margin'\
                                or not 'margin' in opt.structure_loss_type, # True when using nscl
                            'sample_n': opt.train_sample_n,
                            'train_mode': 'NAIC'},
                        mode='sample')
                    gts = [gts[_] for _ in gt_indices.tolist()]
                    SAIC_struc_loss = self.struc_crit(SAIC_seq_logprobs, SAIC_seq, gts, reduction=reduction)
                    NAIC_struc_loss = self.struc_crit(NAIC_seq_logprobs, NAIC_seq, gts, reduction=reduction)
                    struc_loss = {'loss': SAIC_struc_loss['loss'] + NAIC_struc_loss['loss'],
                                'reward': SAIC_struc_loss['reward'] + NAIC_struc_loss['reward']} 
                else:
                    struc_loss = {'loss': torch.tensor(0).type_as(fc_feats),
                                'reward': torch.tensor(0).type_as(fc_feats)}
                SAIC_loss = (1-opt.structure_loss_weight) * lm_loss + opt.structure_loss_weight * SAIC_struc_loss['loss']
                NAIC_loss = (1-opt.structure_loss_weight) * lm_loss + opt.structure_loss_weight * NAIC_struc_loss['loss']
                loss = SAIC_loss + NAIC_loss
                if self.rl_kl:
                    SAIC_mask = SAIC_seq > 0
                    KL_loss = nn.KLDivLoss(reduction='none')
                    SAIC_seq_probs = torch.exp(SAIC_seq_logprobs)
                    NA_KL_loss = KL_loss(NAIC_seq_logprobs, SAIC_seq_probs.detach()) * SAIC_mask.unsqueeze(2)
                    NA_KL_loss_mean = torch.sum(NA_KL_loss) / (torch.sum(SAIC_mask) + 1e-6)
                    loss += NA_KL_loss_mean
                out['lm_loss'] = lm_loss
                out['struc_loss'] = struc_loss['loss']
                out['reward'] = struc_loss['reward']
            else:
                SA_predict_phrase_length_logprob, SA_predict_phrase_syn_logprob, SA_predict_phrase_logprob, \
                    NA_predict_phrase_length_logprob, NA_predict_phrase_syn_logprob, NA_predict_phrase_logprob = \
                    self.model(fc_feats, att_feats, labels, att_masks, phrase_num, phrase_length, phrase_syn, extend_phrase_syn_seq, extend_phrase_seq, extend_phrase_seq_mask, glat_p)
                loss, SA_length_loss_mean, SA_phrase_loss_mean, SA_syn_loss_mean, NA_length_loss_mean, NA_phrase_loss_mean, NA_syn_loss_mean = \
                    self.crit(SA_predict_phrase_length_logprob, SA_predict_phrase_syn_logprob, SA_predict_phrase_logprob, \
                        NA_predict_phrase_length_logprob, NA_predict_phrase_syn_logprob, NA_predict_phrase_logprob, \
                            phrase_num, phrase_length, phrase_syn, labels, reduction=reduction, self_dis=self.self_dis)
                out['SA_length_loss'] = SA_length_loss_mean
                out['SA_phrase_loss'] = SA_phrase_loss_mean
                out['SA_syn_loss'] = SA_syn_loss_mean
                out['NA_length_loss'] = NA_length_loss_mean
                out['NA_phrase_loss'] = NA_phrase_loss_mean
                out['NA_syn_loss'] = NA_syn_loss_mean
        elif self.train_mode == 'UIC_s' or self.train_mode == 'UIC_u':
            if struc_flag:
                if opt.structure_loss_weight < 1:
                    predict_phrase_length_logprob, predict_phrase_syn_logprob, A_predict_phrase_prob, A_predict_phrase_logprob, \
                        SA_predict_phrase_prob, SA_predict_phrase_logprob, NA_predict_phrase_logprob = \
                        self.model(fc_feats, att_feats, labels, att_masks, phrase_num, phrase_length, phrase_syn, extend_phrase_syn_seq, extend_phrase_seq, extend_phrase_seq_mask)
                    lm_loss, length_loss_mean, syn_loss_mean, A_phrase_loss_mean, SA_phrase_loss_mean, NA_phrase_loss_mean, SA_KL_loss_mean, NA_KL_loss_mean = \
                        self.crit(predict_phrase_length_logprob, predict_phrase_syn_logprob, A_predict_phrase_prob, A_predict_phrase_logprob,
                            SA_predict_phrase_prob, SA_predict_phrase_logprob, NA_predict_phrase_logprob,
                            phrase_num, phrase_length, phrase_syn, labels, reduction=reduction)
                else:
                    lm_loss = torch.tensor(0).type_as(fc_feats)
                if opt.structure_loss_weight > 0:
                    AIC_seq, AIC_seq_logprobs, p_phrase_num, p_phrase_length, p_phrase_syn, t = self.model(fc_feats, att_feats, att_masks,
                        opt={'sample_method':opt.train_sample_method, # sample
                            'beam_size':opt.train_beam_size,
                            'output_logsoftmax': opt.struc_use_logsoftmax or opt.structure_loss_type == 'softmax_margin'\
                                or not 'margin' in opt.structure_loss_type, # True when using nscl
                            'sample_n': opt.train_sample_n,
                            'train_mode': 'UIC_s_AIC'},
                        mode='sample')
                    SAIC_seq, SAIC_seq_logprobs, p_phrase_num, p_phrase_length, p_phrase_syn, t = self.model(fc_feats, att_feats, att_masks,
                        opt={'sample_method':opt.train_sample_method, # sample
                            'beam_size':opt.train_beam_size,
                            'output_logsoftmax': opt.struc_use_logsoftmax or opt.structure_loss_type == 'softmax_margin'\
                                or not 'margin' in opt.structure_loss_type, # True when using nscl
                            'sample_n': opt.train_sample_n,
                            'train_mode': 'UIC_s_SAIC'},
                        mode='sample')
                    NAIC_seq, NAIC_seq_logprobs, p_phrase_num, p_phrase_length, p_phrase_syn, t = self.model(fc_feats, att_feats, att_masks,
                        opt={'sample_method':opt.train_sample_method, # sample
                            'beam_size':opt.train_beam_size,
                            'output_logsoftmax': opt.struc_use_logsoftmax or opt.structure_loss_type == 'softmax_margin'\
                                or not 'margin' in opt.structure_loss_type, # True when using nscl
                            'sample_n': opt.train_sample_n,
                            'train_mode': 'UIC_s_NAIC'},
                        mode='sample')
                    gts = [gts[_] for _ in gt_indices.tolist()]
                    AIC_struc_loss = self.struc_crit(AIC_seq_logprobs, AIC_seq, gts, reduction=reduction)
                    SAIC_struc_loss = self.struc_crit(SAIC_seq_logprobs, SAIC_seq, gts, reduction=reduction)
                    NAIC_struc_loss = self.struc_crit(NAIC_seq_logprobs, NAIC_seq, gts, reduction=reduction)
                    struc_loss = {'loss': AIC_struc_loss['loss'] + SAIC_struc_loss['loss'] + NAIC_struc_loss['loss'],
                                'reward': AIC_struc_loss['reward'] + SAIC_struc_loss['reward'] + NAIC_struc_loss['reward']} 
                else:
                    struc_loss = {'loss': torch.tensor(0).type_as(fc_feats),
                                'reward': torch.tensor(0).type_as(fc_feats)}
                AIC_loss = (1-opt.structure_loss_weight) * lm_loss + opt.structure_loss_weight * AIC_struc_loss['loss']
                SAIC_loss = (1-opt.structure_loss_weight) * lm_loss + opt.structure_loss_weight * SAIC_struc_loss['loss']
                NAIC_loss = (1-opt.structure_loss_weight) * lm_loss + opt.structure_loss_weight * NAIC_struc_loss['loss']
                loss = AIC_loss + SAIC_loss + NAIC_loss
                if self.rl_kl:
                    AIC_mask = AIC_seq > 0
                    SAIC_mask = SAIC_seq > 0
                    KL_loss = nn.KLDivLoss(reduction='none')
                    AIC_seq_probs = torch.exp(AIC_seq_logprobs)
                    SAIC_seq_probs = torch.exp(SAIC_seq_logprobs)
                    SA_KL_loss = KL_loss(SAIC_seq_logprobs, AIC_seq_probs.detach()) * AIC_mask.unsqueeze(2)
                    NA_KL_loss = KL_loss(NAIC_seq_logprobs, AIC_seq_probs.detach()) * AIC_mask.unsqueeze(2) + KL_loss(NAIC_seq_logprobs, SAIC_seq_probs.detach()) * SAIC_mask.unsqueeze(2)
                    SA_KL_loss_mean = torch.sum(SA_KL_loss) / (torch.sum(AIC_mask) + 1e-6)
                    NA_KL_loss_mean = torch.sum(NA_KL_loss) / (torch.sum(SAIC_mask) + 1e-6)
                    loss += (SA_KL_loss_mean + NA_KL_loss_mean) 
                out['lm_loss'] = lm_loss
                out['struc_loss'] = struc_loss['loss']
                out['reward'] = struc_loss['reward']
            else:
                predict_phrase_length_logprob, predict_phrase_syn_logprob, A_predict_phrase_prob, A_predict_phrase_logprob, \
                    SA_predict_phrase_prob, SA_predict_phrase_logprob, NA_predict_phrase_logprob = \
                    self.model(fc_feats, att_feats, labels, att_masks, phrase_num, phrase_length, phrase_syn, extend_phrase_syn_seq, extend_phrase_seq, extend_phrase_seq_mask)
                loss, length_loss_mean, syn_loss_mean, A_phrase_loss_mean, SA_phrase_loss_mean, NA_phrase_loss_mean, SA_KL_loss_mean, NA_KL_loss_mean = \
                    self.crit(predict_phrase_length_logprob, predict_phrase_syn_logprob, A_predict_phrase_prob, A_predict_phrase_logprob,
                            SA_predict_phrase_prob, SA_predict_phrase_logprob, NA_predict_phrase_logprob,
                            phrase_num, phrase_length, phrase_syn, labels, reduction=reduction)
                out['length_loss'] = length_loss_mean
                out['syn_loss'] = syn_loss_mean
                out['A_phrase_loss'] = A_phrase_loss_mean
                out['SA_phrase_loss'] = SA_phrase_loss_mean
                out['NA_phrase_loss'] = NA_phrase_loss_mean
                out['SA_KL_loss'] = SA_KL_loss_mean
                out['NA_KL_loss'] = NA_KL_loss_mean
        elif self.train_mode == 'PB_pad':
            if struc_flag:
                if opt.structure_loss_weight < 1:
                    lm_loss = self.crit(self.model(fc_feats, att_feats, labels[..., :-1], att_masks), labels[..., 1:], masks[..., 1:], reduction=reduction)
                else:
                    lm_loss = torch.tensor(0).type_as(fc_feats)
                if opt.structure_loss_weight > 0:
                    seq, seq_logprobs, p_phrase_num, p_phrase_length, t = self.model(fc_feats, att_feats, att_masks,
                        opt={'sample_method':opt.train_sample_method, # sample
                            'beam_size':opt.train_beam_size,
                            'output_logsoftmax': opt.struc_use_logsoftmax or opt.structure_loss_type == 'softmax_margin'\
                                or not 'margin' in opt.structure_loss_type, # True when using nscl
                            'sample_n': opt.train_sample_n,
                            'train_mode': self.train_mode},
                        mode='sample')
                    gts = [gts[_] for _ in gt_indices.tolist()]
                    struc_loss = self.struc_crit(seq_logprobs, seq, gts, reduction=reduction)
                else:
                    struc_loss = {'loss': torch.tensor(0).type_as(fc_feats),
                                'reward': torch.tensor(0).type_as(fc_feats)}
                loss = (1-opt.structure_loss_weight) * lm_loss + opt.structure_loss_weight * struc_loss['loss']
                out['lm_loss'] = lm_loss
                out['struc_loss'] = struc_loss['loss']
                out['reward'] = struc_loss['reward']
            else:
                predict_phrase_length, predict_phrase_length_logprob, predict_phrase_logprob = self.model(fc_feats, att_feats, labels, att_masks, phrase_num, phrase_length, phrase_syn, extend_phrase_syn_seq, extend_phrase_seq, extend_phrase_seq_mask)
                loss, length_loss, phrase_loss = self.crit(phrase[..., 1:], predict_phrase_logprob, masks[..., 1:], phrase_num, phrase_length[..., 1:], predict_phrase_length, predict_phrase_length_logprob, reduction=reduction)
                out['length_loss'] = length_loss
                out['phrase_loss'] = phrase_loss
                out['syn_loss'] = 0
        out['loss'] = loss
        return out
