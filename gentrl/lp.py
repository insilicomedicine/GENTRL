import torch
import torch.nn as nn
from math import sqrt, pi
from sklearn.mixture import GaussianMixture
import numpy as np


class LP(nn.Module):
    """
    Class for a Learnable Prior.
    """

    def __init__(self, distr_descr, tt_int=30, distr_init='rand',
                 tt_type='usual', eps=1e-10, order=None, **kwargs):
        """
        Args:
            distr_descr: list of n tuples, where n is a number
                of variables in lp model distribution, i-th tuple describes
                the distribution of the i-th variable; if i-th variable is
                continuous the tuple should contain ('c', d, lr, rb), where d
                is a number of gaussians to model this variable, lr and rb are
                optional elements that descibe lower and upper bounds for
                means of the gaussians; if i-th variable is discrete, then it
                should be described as ('d', d) where d is number of values
                this variable can take

                example: [('c', 10), ('c', 10, -2, 5), ('d', 2), ('c', 20)]
            tt_int: int; internal dimension of Tensor-Train decomposition
            distr_init: 'rand' or 'uniform'; method to initialize
                the distribution
            tt_type: 'usual' or 'ring'; type of Tensor Train decomposition
            eps: float; small number to avoid devision by zero
            order: None or list of int; if None then order of cores corresponds
                to distr_descr, otherwise it should be a permutation of
                [0, 1, ..., len(distr_descr) - 1]
        """
        super(LP, self).__init__()

        self.tt_int = tt_int
        self.tt_type = tt_type

        self.distr_descr = distr_descr
        self.distr_init = distr_init

        self.tt_cores = []
        self.means = []
        self.log_stds = []

        self.eps = eps

        if order is None:
            self.order = list(range(len(distr_descr)))
        else:
            self.order = order

        # initialize cores, means, and stds for the distribution

        if self.tt_type not in ['ring', 'usual']:
            raise ValueError("Use 'ring' or 'usual' in tt_type, "
                             "found {}".format(self.tt_type))

        for var_descr in self.distr_descr:
            if distr_init == 'rand':
                cur_core = torch.randn(var_descr[1], self.tt_int, self.tt_int)
            elif distr_init == 'uniform':
                cur_core = torch.ones(var_descr[1], self.tt_int, self.tt_int)
            else:
                raise ValueError("Use 'rand' or 'uniform' in distr_init, "
                                 "found {}".format(distr_init))

            cur_core = cur_core / (self.tt_int ** 2 * var_descr[1])

            self.tt_cores.append(nn.Parameter(cur_core))

            if var_descr[0] == 'd':  # discrete variable
                self.means.append(None)
                self.log_stds.append(None)
            elif var_descr[0] == 'c':  # continous variable
                if len(var_descr) == 4:
                    lb = var_descr[2]
                    rb = var_descr[3]
                else:
                    lb = -1
                    rb = 1

                if distr_init == 'rand':
                    cur_means = torch.rand(var_descr[1]) * (rb - lb) + lb
                elif distr_init == 'uniform':
                    cur_means = (torch.arange(var_descr[1]).float() /
                                 (var_descr[1] - 1)) * (rb - lb) + lb

                cur_log_stds = 2 * torch.log(
                    torch.ones(var_descr[1]) * (rb - lb) / var_descr[1]
                )

                self.means.append(nn.Parameter(cur_means))
                self.log_stds.append(nn.Parameter(cur_log_stds))
            else:
                raise ValueError("Use 'c' or 'd' in distribution desciption, "
                                 "found {}".format(var_descr[1]))

        self._make_model_parameters()

    @staticmethod
    def __make_contr_vec(x, var, missed, means, log_stds):
        if missed is None:
            missed = torch.isnan(x).byte()
            x[missed] = 0
        if var[0] == 'd':
            contr_vect = torch.zeros(x.shape[0], var[1])
            contr_vect[missed] = 1
            contr_vect[torch.arange(x.shape[0]), x.long().cpu()] = 1
        elif var[0] == 'c':
            cur_vals = x[:, None]
            cur_stds = torch.exp(log_stds)[None, :]
            cur_means = means[None, :]

            contr_vect = (cur_vals - cur_means) / cur_stds
            contr_vect = torch.exp(-0.5 * (contr_vect ** 2))
            contr_vect = contr_vect / (sqrt(2 * pi) * cur_stds)
            contr_vect = contr_vect + 1e-10

            m = missed.float()[:, None].to(x.device)
            contr_vect = contr_vect * (1 - m) + m

        contr_vect = contr_vect.to(x.device)

        return contr_vect

    def log_prob(self, x, marg=None):
        '''
        Computes logits for each token input_tensor by given latent code

        Args:
            x: tensor of shape [num_objects, num_components];
                missing values encoded as nan
            marg: None or list of bools; if None, no variables will
                be marginalized, else if i-th value of list is True,
                then i-th variable will be marginalized
        Returns:
            log_probs: tensor of shape [num_objects]
        '''
        num_objects = x.shape[0]

        if marg is None:
            marg = x.shape[1] * [False]

        perm_marg = [marg[i] for i in self.order]
        perm_dist_descr = [self.distr_descr[i] for i in self.order]
        perm_x = x[:, self.order]
        perm_cores = [self.tt_cores[i] for i in self.order]
        perm_means = [self.means[i] for i in self.order]
        perm_log_stds = [self.log_stds[i] for i in self.order]

        # compute log probabilities
        log_probs = torch.zeros(num_objects).to(x.device)

        if self.tt_type == 'usual':
            pref = torch.ones(num_objects, 1, perm_cores[0].shape[1])
            norm_pref = torch.ones(num_objects, 1, perm_cores[0].shape[1])
        elif self.tt_type == 'ring':
            pref = torch.eye(perm_cores[0].shape[1])
            pref = pref[None, :, :].repeat(num_objects, 1, 1)

            norm_pref = torch.eye(perm_cores[0].shape[1])
        pref = pref.to(x.device)
        norm_pref = norm_pref.to(x.device)

        for i, (core, var) in enumerate(zip(perm_cores, perm_dist_descr)):
            core = self._pos_func(core)

            if perm_marg[i]:
                cond_core = core.sum(dim=0)[None, :, :]
                cond_core = cond_core.repeat(num_objects, 1, 1)
            else:
                cur_contr_vect = self.__make_contr_vec(perm_x[:, i],
                                                       var, None,
                                                       perm_means[i],
                                                       perm_log_stds[i])
                cond_core = core[None, :, :, :]
                cond_core = cond_core * cur_contr_vect[:, :, None, None]
                cond_core = cond_core.sum(dim=1)

            norm_core = core.sum(dim=0)

            pref = torch.bmm(pref, cond_core)
            norm_pref = norm_pref @ norm_core

            cur_norm_const = torch.sum(norm_pref) + self.eps
            pref = pref / cur_norm_const
            norm_pref = norm_pref / cur_norm_const

            cur_prob_addition = pref.sum(dim=-1).sum(dim=-1) + self.eps
            log_probs = log_probs + torch.log(cur_prob_addition)

            pref = pref / cur_prob_addition[:, None, None]

        if self.tt_type == 'ring':
            eye = torch.eye(perm_cores[-1].shape[-1])[None, :, :]
            eye = eye.to(x.device)
            cur_prob_addition = (pref * eye).sum(dim=-1).sum(dim=-1)
            cur_div = (norm_pref * eye).sum(dim=-1).sum(dim=-1) + self.eps
            cur_prob_addition = cur_prob_addition / cur_div
            log_probs = log_probs + torch.log(cur_prob_addition)
        elif self.tt_type == 'usual':
            cur_prob_addition = pref.sum(dim=-1).sum(dim=-1)
            cur_div = norm_pref.sum(dim=-1).sum(dim=-1) + self.eps
            cur_prob_addition = cur_prob_addition / cur_div
            log_probs = log_probs + torch.log(cur_prob_addition)

        return log_probs

    def sample(self, num_samples, sample_descr, conds=None):
        '''
        Sample from the distribution

        Args:
            num_samples: int, number objects to sample
            sample_descr: list of chars, containining
                's' if we should sample this variable
                'm' if we should marginalise this variable
                'c' if we should condition on this variable

                example: ['s', 's', 'c', 's', 'm', 's']
            conditions: tensor of shape [num_sampled, total_num_of_variables],
                if sample_descr has variables for conditioning, then
                condition values should be set by this parameter
        Returns:
            samples: tensor of shape [num_objects, num_vars_to_sample]
        '''

        perm_dist_descr = [self.distr_descr[i] for i in self.order]

        perm_sample_descr = [sample_descr[i] for i in self.order]
        perm_cores = [self.tt_cores[i] for i in self.order]
        perm_means = [self.means[i] for i in self.order]
        perm_log_stds = [self.log_stds[i] for i in self.order]

        if conds is not None:
            perm_conds = conds[:, self.order]

        # computing contraction vectors
        contr_vect_list = []
        for i, (action, var) in enumerate(
                zip(perm_sample_descr, perm_dist_descr)):
            if action == 'c':
                contr_vect_list.append(self.__make_contr_vec(perm_conds[:, i],
                                                             var,
                                                             None,
                                                             perm_means[i],
                                                             perm_log_stds[i]))
            elif action in ['m', 's']:
                contr_vect_list.append(
                    torch.ones(num_samples, var[1]).to(self.tt_cores[0].device)
                )

        # computing suffixes to sample via chainrule
        sufxs = []
        if self.tt_type == 'usual':
            cur_suf = torch.ones(num_samples, perm_cores[-1].shape[-1], 1)
        else:
            cur_suf = torch.eye(perm_cores[-1].shape[-1])
            cur_suf = cur_suf[None, :, :].repeat(num_samples, 1, 1)
        cur_suf = cur_suf.to(self.tt_cores[0])
        sufxs.append(cur_suf)

        for var_descr, core, contr_vect in zip(perm_dist_descr[::-1],
                                               perm_cores[::-1],
                                               contr_vect_list[::-1]):
            core = self._pos_func(core)

            cond_core = (core[None, :, :, :] * contr_vect[:, :, None, None])
            cond_core = cond_core.sum(dim=1)

            cur_suf = torch.bmm(cond_core, cur_suf)

            norm_const = torch.sum(cur_suf + self.eps, dim=-1, keepdim=True)
            norm_const = torch.sum(norm_const, dim=-2, keepdim=True)

            cur_suf /= norm_const

            sufxs.append(cur_suf)
        sufxs = sufxs[-2::-1]

        # sampling
        if self.tt_type == 'usual':
            pref = torch.ones(num_samples, 1, perm_cores[0].shape[1])
        else:
            pref = torch.eye(perm_cores[0].shape[1])[None, :, :]
            pref = pref.repeat(num_samples, 1, 1)

        pref = pref.to(self.tt_cores[0])

        samples_list = []
        for i, (action, var, core, suf, prev_contr_vect) in enumerate(
                zip(perm_sample_descr,
                    perm_dist_descr,
                    perm_cores,
                    sufxs,
                    contr_vect_list)):
            core = self._pos_func(core)
            if action == 's':
                # compute current mixture/discr dist weights
                part_to_contract = torch.bmm(suf, pref).permute(0, 2, 1)
                part_to_contract = part_to_contract[:, None, :, :]
                weights = part_to_contract * core[None, :, :, :]
                weights = weights.sum(dim=-1).sum(dim=-1) + self.eps
                weights /= torch.sum(weights, dim=-1, keepdim=True)

                # sample
                discr_comp_sample = torch.multinomial(weights, num_samples=1)
                discr_comp_sample = discr_comp_sample.view(-1)

                # construct cur_contr_vect
                if var[0] == 'd':
                    cur_samples = discr_comp_sample
                elif var[0] == 'c':
                    cur_means = perm_means[i][discr_comp_sample]
                    cur_log_stds = perm_log_stds[i][discr_comp_sample]

                    cur_samples = cur_means + torch.exp(
                        cur_log_stds) * torch.randn_like(cur_log_stds)

                samples_list.append(cur_samples)
                contr_vect = self.__make_contr_vec(cur_samples,
                                                   var,
                                                   None,
                                                   perm_means[i],
                                                   perm_log_stds[i])
            elif action in ['m', 'c']:
                samples_list.append(None)
                contr_vect = prev_contr_vect

            cond_core = (core[None, :, :, :] * contr_vect[:, :, None, None])
            cond_core = cond_core.sum(dim=1)

            pref = torch.bmm(pref, cond_core) + self.eps

            norm_const = torch.sum(pref + self.eps, dim=-1, keepdim=True)
            norm_const = torch.sum(norm_const, dim=-2, keepdim=True)

            pref /= norm_const

        inv_perm_samples_list = len(samples_list) * [None]
        for i in range(len(samples_list)):
            inv_perm_samples_list[self.order[i]] = samples_list[i]

        return torch.cat(
            [s.float()[:, None] for s in inv_perm_samples_list if
             s is not None], dim=-1).detach()

    def reinit_from_data(self, data, var_types=None):
        """
        Reinitializing Gaussians' parameters to better
        cover the latent space
        Also resets TT cores

        Args:
            data: tensor of shape [num_objects, num_vars], data to
                reinitialize the Gaussians
            var_types:
        """
        new_tt_cores = []
        new_means = []
        new_log_stds = []

        components = []
        for i, var_descr in enumerate(self.distr_descr):
            cur_core = torch.randn(var_descr[1], self.tt_int, self.tt_int)
            cur_core = cur_core / (self.tt_int ** 2 * var_descr[1])
            new_tt_cores.append(nn.Parameter(cur_core))

            if torch.sum(torch.isnan(data[:, i])) == data.shape[0]:
                new_means.append(self.means[i])
                new_log_stds.append(self.log_stds[i])

                components.append(-1 * np.ones(data.shape[0]))
                continue

            if var_descr[0] == 'd':
                new_means.append(None)
                new_log_stds.append(None)
                cur_components = data[:, i].cpu().detach().numpy()
                cur_components[np.isnan(cur_components)] = -1

            elif var_descr[0] == 'c':
                gmm = GaussianMixture(n_components=var_descr[1])
                cur_data = data[:, i].cpu().detach().numpy()
                non_missings = np.logical_not(np.isnan(cur_data))

                cur_components = -1 * np.ones_like(cur_data)

                non_missed_data = cur_data[non_missings].reshape(-1, 1)
                gmm.fit(non_missed_data)
                cur_gmm_comp = gmm.predict(non_missed_data)
                cur_components[non_missings] = cur_gmm_comp

                cur_means = torch.from_numpy(gmm.means_[:, 0])
                cur_means = cur_means.float()
                cur_log_stds = torch.from_numpy(gmm.covariances_[:, 0, 0])
                cur_log_stds = torch.log(cur_log_stds.float() + self.eps) / 2

                new_means.append(nn.Parameter(cur_means))
                new_log_stds.append(nn.Parameter(cur_log_stds))

            components.append(cur_components.astype(np.int))

        if var_types is not None:
            usual_vars_idxs = [i for i in range(len(var_types)) if
                               var_types[i] == 0]
            target_vars_idxs = [i for i in range(len(var_types)) if
                                var_types[i] == 1]

            scores = np.zeros((len(target_vars_idxs), len(usual_vars_idxs)))

            for i in range(len(target_vars_idxs)):
                for j in range(len(usual_vars_idxs)):
                    tg_n = self.distr_descr[target_vars_idxs[i]][1]
                    us_n = self.distr_descr[usual_vars_idxs[j]][1]
                    mx = np.zeros((tg_n, us_n))

                    tg_comp = components[target_vars_idxs[i]]
                    us_comp = components[usual_vars_idxs[j]]
                    for x, y in zip(tg_comp, us_comp):
                        if x != -1 and y != -1:
                            mx[x, y] += 1

                    if mx.sum() == 0:
                        continue

                    mx += 1e-10

                    s = mx.sum(axis=0)
                    mx = mx / s[None, :]

                    scores[i, j] = -((np.log(mx) * mx).sum(
                        axis=0) * s).sum() / s.sum()

            groups = np.argmin(scores, axis=0)

            new_order = []

            for group_i in range(len(target_vars_idxs)):
                g_members = np.where(groups == group_i)[0]
                g_members = sorted(g_members, key=lambda s: scores[group_i, s])
                new_group = [target_vars_idxs[group_i]]

                for i, member in enumerate(g_members):
                    if i % 2 == 0:
                        new_group = new_group + [usual_vars_idxs[member]]
                    else:
                        new_group = [usual_vars_idxs[member]] + new_group

                new_order += new_group

            self.order = new_order

        for i in range(len(self.tt_cores)):
            self.tt_cores[i].data = new_tt_cores[i].data
            if new_means[i] is not None:
                self.means[i].data = new_means[i].data
                self.log_stds[i].data = new_log_stds[i].data

    @staticmethod
    def _pos_func(x):
        return x * x

    def _make_model_parameters(self):
        parameters = []
        for mean, log_std in zip(self.means, self.log_stds):
            if mean is None:
                continue
            parameters += [mean, log_std]

        for core in self.tt_cores:
            parameters.append(core)

        self.parameters = nn.ParameterList(parameters)
