# Adapted REDQSAC agent to use diffusion samples.

import numpy as np
from redq.algos.core import ReplayBuffer
from redq.algos.redq_sac import REDQSACAgent
from torch import Tensor
import torch


def combine_two_tensors(tensor1, tensor2):
    return Tensor(np.concatenate([tensor1, tensor2], axis=0))


class REDQRLPDAgent(REDQSACAgent):

    def __init__(self, diffusion_buffer_size=int(1e6), diffusion_sample_ratio=0.5, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.diffusion_buffer = ReplayBuffer(obs_dim=self.obs_dim, act_dim=self.act_dim, size=diffusion_buffer_size)
        self.diffusion_sample_ratio = diffusion_sample_ratio

    # def sample_data(self, batch_size):
    #     diffusion_batch_size = int(batch_size * self.diffusion_sample_ratio)
    #     online_batch_size = int(batch_size - diffusion_batch_size)
    #     # Sample from the diffusion buffer
    #     if self.diffusion_buffer.size < diffusion_batch_size:
    #         return super().sample_data(batch_size)
    #     diffusion_batch = self.diffusion_buffer.sample_batch(batch_size=diffusion_batch_size)
    #     online_batch = self.replay_buffer.sample_batch(batch_size=online_batch_size)
    #     obs_tensor = combine_two_tensors(online_batch['obs1'], diffusion_batch['obs1']).to(self.device)
    #     obs_next_tensor = combine_two_tensors(online_batch['obs2'], diffusion_batch['obs2']).to(self.device)
    #     acts_tensor = combine_two_tensors(online_batch['acts'], diffusion_batch['acts']).to(self.device)
    #     rews_tensor = combine_two_tensors(online_batch['rews'], diffusion_batch['rews']).unsqueeze(1).to(self.device)
    #     done_tensor = combine_two_tensors(online_batch['done'], diffusion_batch['done']).unsqueeze(1).to(self.device)
    #     return obs_tensor, obs_next_tensor, acts_tensor, rews_tensor, done_tensor
    def sample_diff_data(self, batch_size):
        # sample data from diffsion buffer
        if self.diffusion_buffer.size < batch_size:
            print("###############################diffsuion samples is not enough")
            return super().sample_data(batch_size)
        batch = self.diffusion_buffer.sample_batch(batch_size)
        obs_tensor = Tensor(batch['obs1']).to(self.device)
        obs_next_tensor = Tensor(batch['obs2']).to(self.device)
        acts_tensor = Tensor(batch['acts']).to(self.device)
        rews_tensor = Tensor(batch['rews']).unsqueeze(1).to(self.device)
        done_tensor = Tensor(batch['done']).unsqueeze(1).to(self.device)
        return obs_tensor, obs_next_tensor, acts_tensor, rews_tensor, done_tensor

    def soft_update_model1_with_model2(self, model1, model2, rou):
        """
        used to polyak update a target network
        :param model1: a pytorch model
        :param model2: a pytorch model of the same class
        :param rou: the update is model1 <- rou*model1 + (1-rou)model2
        """
        for model1_param, model2_param in zip(model1.parameters(), model2.parameters()):
            model1_param.data.copy_(rou*model1_param.data + (1-rou)*model2_param.data)  

    def train(self, logger):
        # 确保 Critic 和 Actor 使用不同的 buffer
        num_update = 0 if self._get_current_num_data() <= self.delay_update_steps else self.utd_ratio

        # 更新 Critic 网络（从 diffusion_buffer 中采样数据）
        for i_update in range(num_update):
            # 从 diffusion_buffer 中采样
            if self.diffusion_buffer.size > 0:
                obs_tensor, obs_next_tensor, acts_tensor, rews_tensor, done_tensor = self.sample_data(self.batch_size)
            else:
                # 如果 diffusion_buffer 中没有数据，直接跳过更新
                obs_tensor, obs_next_tensor, acts_tensor, rews_tensor, done_tensor = self.sample_data(self.batch_size)

            """Critic Q loss"""
            y_q, sample_idxs = self.get_redq_q_target_no_grad(obs_next_tensor, rews_tensor, done_tensor)
            q_prediction_list = []
            for q_i in range(self.num_Q):
                q_prediction = self.q_net_list[q_i](torch.cat([obs_tensor, acts_tensor], 1))
                q_prediction_list.append(q_prediction)
            q_prediction_cat = torch.cat(q_prediction_list, dim=1)
            y_q = y_q.expand((-1, self.num_Q)) if y_q.shape[1] == 1 else y_q
            q_loss_all = self.mse_criterion(q_prediction_cat, y_q) * self.num_Q

            for q_i in range(self.num_Q):
                self.q_optimizer_list[q_i].zero_grad()
            q_loss_all.backward()
                
                
            if ((i_update + 1) % self.policy_update_delay == 0) or i_update == num_update - 1:
                # 从 replay_buffer 中采样
                if self.replay_buffer.size > 0:
                    obs_tensor, obs_next_tensor, acts_tensor, rews_tensor, done_tensor = self.sample_data(self.batch_size)
                else:
                    # 如果 replay_buffer 中没有数据，直接跳过更新
                    continue

                """Actor policy loss"""
                a_tilda, mean_a_tilda, log_std_a_tilda, log_prob_a_tilda, _, pretanh = self.policy_net.forward(obs_tensor)
                q_a_tilda_list = []
                for sample_idx in range(self.num_Q):
                    self.q_net_list[sample_idx].requires_grad_(False)
                    q_a_tilda = self.q_net_list[sample_idx](torch.cat([obs_tensor, a_tilda], 1))
                    q_a_tilda_list.append(q_a_tilda)
                q_a_tilda_cat = torch.cat(q_a_tilda_list, 1)
                ave_q = torch.mean(q_a_tilda_cat, dim=1, keepdim=True)
                policy_loss = (self.alpha * log_prob_a_tilda - ave_q).mean()
                self.policy_optimizer.zero_grad()
                policy_loss.backward()
                for sample_idx in range(self.num_Q):
                    self.q_net_list[sample_idx].requires_grad_(True)
                self.policy_optimizer.step()

                # Alpha loss（可选）
                if self.auto_alpha:
                    alpha_loss = -(self.log_alpha * (log_prob_a_tilda + self.target_entropy).detach()).mean()
                    self.alpha_optim.zero_grad()
                    alpha_loss.backward()
                    self.alpha_optim.step()
                    self.alpha = self.log_alpha.cpu().exp().item()
                else:
                    alpha_loss = Tensor([0])
            
            """update networks"""
            for q_i in range(self.num_Q):
                self.q_optimizer_list[q_i].step()

            if ((i_update + 1) % self.policy_update_delay == 0) or i_update == num_update - 1:
                self.policy_optimizer.step()

            # polyak averaged Q target networks
            for q_i in range(self.num_Q):
                self.soft_update_model1_with_model2(self.q_target_net_list[q_i], self.q_net_list[q_i], self.polyak)

            # by default only log for the last update out of <num_update> updates
            if i_update == num_update - 1:
                logger.store(LossPi=policy_loss.cpu().item(), LossQ1=q_loss_all.cpu().item() / self.num_Q,
                             LossAlpha=alpha_loss.cpu().item(), Q1Vals=q_prediction.detach().cpu().numpy(),
                             Alpha=self.alpha, LogPi=log_prob_a_tilda.detach().cpu().numpy(),
                             PreTanh=pretanh.abs().detach().cpu().numpy().reshape(-1))

        # 如果没有更新，记录 0 防止日志记录问题
        if num_update == 0:
            logger.store(LossPi=0, LossQ1=0, LossAlpha=0, Q1Vals=0, Alpha=0, LogPi=0, PreTanh=0)

    def reset_diffusion_buffer(self):
        self.diffusion_buffer = ReplayBuffer(obs_dim=self.obs_dim, act_dim=self.act_dim,
                                             size=self.diffusion_buffer.max_size)
