import math
import torch
from torch.distributions.kl import register_kl
import torch.nn.functional as F
from toolbox.hyperspherical_vae.ops.ive import ive, ive_fraction_approx, ive_fraction_approx2
from toolbox.hyperspherical_vae.distributions.hyperspherical_uniform import (
    HypersphericalUniform,
)


class VonMisesFisher(torch.distributions.Distribution):

    arg_constraints = {
        "loc": torch.distributions.constraints.real,
        "scale": torch.distributions.constraints.positive,
    }
    support = torch.distributions.constraints.real
    has_rsample = True
    _mean_carrier_measure = 0

    @property
    def mean(self):
        # option 1:
        return self.loc * (
            ive(self.__m / 2, self.scale) / ive(self.__m / 2 - 1, self.scale)
        )
        # option 2:
        # return self.loc * ive_fraction_approx(torch.tensor(self.__m / 2), self.scale)
        # options 3:
        # return self.loc * ive_fraction_approx2(torch.tensor(self.__m / 2), self.scale)

    @property
    def stddev(self):
        return self.scale

    def __init__(self, loc, scale, validate_args=None, k=1):
        self.dtype = loc.dtype
        self.loc = loc
        self.scale = scale
        self.device = loc.device
        self.__m = loc.shape[-1]
        self.__e1 = (torch.Tensor([1.0] + [0] * (loc.shape[-1] - 1))).to(self.device)
        self.k = k

        super().__init__(self.loc.size(), validate_args=validate_args)

    def sample(self, shape=torch.Size()):
        with torch.no_grad():
            return self.rsample(shape)

    # def rsample(self, shape=torch.Size()):
    #     shape = shape if isinstance(shape, torch.Size) else torch.Size([shape])
    #
    #     w = (
    #         self.__sample_w3(shape=shape)
    #         if self.__m == 3
    #         else self.__sample_w_rej(shape=shape)
    #     )
    #
    #     v = (
    #         torch.distributions.Normal(0, 1)
    #         .sample(shape + torch.Size(self.loc.shape))
    #         .to(self.device)
    #         .transpose(0, -1)[1:]
    #     ).transpose(0, -1)
    #     v = v / v.norm(dim=-1, keepdim=True)
    #
    #     w_ = torch.sqrt(torch.clamp(1 - (w ** 2), 1e-10))
    #     x = torch.cat((w, w_ * v), -1)
    #     z = self.__householder_rotation(x)
    #
    #     return z.type(self.dtype)
    def rsample(self, shape=torch.Size()):
        """
        从 vMF 分布中采样
        Args:
            shape: 额外的采样形状
        Returns:
            采样结果，形状为 [batch_size, dim]
        """
        shape = shape if isinstance(shape, torch.Size) else torch.Size([shape])
        batch_size = self.loc.shape[0]

        # 采样 w
        w = (
            self.__sample_w3(shape=shape)
            if self.__m == 3
            else self.__sample_w_rej(shape=shape)
        )

        # 确保 w 的形状正确 [batch_size, 1]
        w = w.view(batch_size, 1)

        # 在切空间中采样 v [batch_size, dim-1]
        v = torch.randn(batch_size, self.__m - 1, device=self.device)
        v = F.normalize(v, dim=1)  # 归一化

        # 计算 w_
        w_ = torch.sqrt(torch.clamp(1 - (w ** 2), 1e-10))

        # 构造完整向量 [batch_size, dim]
        x = torch.cat((w, w_ * v), dim=1)

        # 应用 Householder 变换
        z = self.__householder_rotation(x)

        return z.type(self.dtype)

    def __sample_w3(self, shape):
        shape = shape + torch.Size(self.scale.shape)
        u = torch.distributions.Uniform(0, 1).sample(shape).to(self.device)
        self.__w = (
            1
            + torch.stack(
                [torch.log(u), torch.log(1 - u) - 2 * self.scale], dim=0
            ).logsumexp(0)
            / self.scale
        )
        return self.__w

    # def __sample_w_rej(self, shape):
    #     c = torch.sqrt((4 * (self.scale ** 2)) + (self.__m - 1) ** 2)
    #     b_true = (-2 * self.scale + c) / (self.__m - 1)
    #
    #     # using Taylor approximation with a smooth swift from 10 < scale < 11
    #     # to avoid numerical errors for large scale
    #     b_app = (self.__m - 1) / (4 * self.scale)
    #     s = torch.min(
    #         torch.max(
    #             torch.tensor([0.0], dtype=self.dtype, device=self.device),
    #             self.scale - 10,
    #         ),
    #         torch.tensor([1.0], dtype=self.dtype, device=self.device),
    #     )
    #     b = b_app * s + b_true * (1 - s)
    #
    #     a = (self.__m - 1 + 2 * self.scale + c) / 4
    #     d = (4 * a * b) / (1 + b) - (self.__m - 1) * math.log(self.__m - 1)
    #
    #     self.__b, (self.__e, self.__w) = b, self.__while_loop(b, a, d, shape, k=self.k)
    #     return self.__w
    def __sample_w_rej(self, shape):
        """
        使用拒绝采样方法采样 w
        """
        batch_size = self.loc.shape[0]  # 获取批次大小

        c = torch.sqrt((4 * (self.scale ** 2)) + (self.__m - 1) ** 2)
        b_true = (-2 * self.scale + c) / (self.__m - 1)

        # 使用Taylor近似，在scale在10到11之间平滑过渡
        b_app = (self.__m - 1) / (4 * self.scale)
        s = torch.min(
            torch.max(
                torch.tensor([0.0], dtype=self.dtype, device=self.device),
                self.scale - 10,
            ),
            torch.tensor([1.0], dtype=self.dtype, device=self.device),
        )
        b = b_app * s + b_true * (1 - s)

        a = (self.__m - 1 + 2 * self.scale + c) / 4
        d = (4 * a * b) / (1 + b) - (self.__m - 1) * math.log(self.__m - 1)

        # 扩展参数到批次大小
        b1 = b.expand (batch_size)
        a1 = a.expand (batch_size)
        d1 = d.expand (batch_size)

        self.__b, (self.__e, self.__w) = b1, self.__while_loop(b1, a1, d1, batch_size, k=self.k)
        return self.__w

    @staticmethod
    def first_nonzero(x, dim, invalid_val=-1):
        mask = x > 0
        idx = torch.where(
            mask.any(dim=dim),
            mask.float().argmax(dim=1).squeeze(),
            torch.tensor(invalid_val, device=x.device),
        )
        return idx

    # def __while_loop(self, b, a, d, shape, k=20, eps=1e-20):
    #     #  matrix while loop: samples a matrix of [A, k] samples, to avoid looping all together
    #     b, a, d = [
    #         e.repeat(*shape, *([1] * len(self.scale.shape))).reshape(-1, 1)
    #         for e in (b, a, d)
    #     ]
    #     w, e, bool_mask = (
    #         torch.zeros_like(b).to(self.device),
    #         torch.zeros_like(b).to(self.device),
    #         (torch.ones_like(b) == 1).to(self.device),
    #     )
    #
    #     sample_shape = torch.Size([b.shape[0], k])
    #     shape = shape + torch.Size(self.scale.shape)
    #
    #     while bool_mask.sum() != 0:
    #         con1 = torch.tensor((self.__m - 1) / 2, dtype=torch.float64)
    #         con2 = torch.tensor((self.__m - 1) / 2, dtype=torch.float64)
    #         e_ = (
    #             torch.distributions.Beta(con1, con2)
    #             .sample(sample_shape)
    #             .to(self.device)
    #             .type(self.dtype)
    #         )
    #
    #         u = (
    #             torch.distributions.Uniform(0 + eps, 1 - eps)
    #             .sample(sample_shape)
    #             .to(self.device)
    #             .type(self.dtype)
    #         )
    #
    #         w_ = (1 - (1 + b) * e_) / (1 - (1 - b) * e_)
    #         t = (2 * a * b) / (1 - (1 - b) * e_)
    #
    #         accept = ((self.__m - 1.0) * t.log() - t + d) > torch.log(u)
    #         accept_idx = self.first_nonzero(accept, dim=-1, invalid_val=-1).unsqueeze(1)
    #         accept_idx_clamped = accept_idx.clamp(0)
    #         # we use .abs(), in order to not get -1 index issues, the -1 is still used afterwards
    #         w_ = w_.gather(1, accept_idx_clamped.view(-1, 1))
    #         e_ = e_.gather(1, accept_idx_clamped.view(-1, 1))
    #
    #         reject = accept_idx < 0
    #         accept = ~reject if torch.__version__ >= "1.2.0" else 1 - reject
    #
    #         w[bool_mask * accept] = w_[bool_mask * accept]
    #         e[bool_mask * accept] = e_[bool_mask * accept]
    #
    #         bool_mask[bool_mask * accept] = reject[bool_mask * accept]
    #
    #     return e.reshape(shape), w.reshape(shape)
    def __while_loop(self, b, a, d, batch_size, k=20, eps=1e-20):
        """
        矩阵形式的while循环：一次采样 [batch_size, k] 个样本
        """
        # 初始化结果张量
        w = torch.zeros(batch_size, device=self.device)
        e = torch.zeros(batch_size, device=self.device)
        bool_mask = torch.ones(batch_size, dtype=torch.bool, device=self.device)

        while bool_mask.any():
            # 只为未接受的样本生成新的候选
            n_remaining = bool_mask.sum().item()  # 使用 item() 获取标量值

            # Beta分布采样
            con1 = torch.tensor((self.__m - 1) / 2, dtype=torch.float64)
            con2 = torch.tensor((self.__m - 1) / 2, dtype=torch.float64)
            e_ = (
                torch.distributions.Beta(con1, con2)
                    .sample((n_remaining, k))
                    .to(self.device)
                    .type(self.dtype)
            )

            # 均匀分布采样
            u = (
                torch.distributions.Uniform(0 + eps, 1 - eps)
                    .sample((n_remaining, k))
                    .to(self.device)
                    .type(self.dtype)
            )

            # 计算候选值
            b_remaining = b[bool_mask].unsqueeze(1)
            a_remaining = a[bool_mask].unsqueeze(1)
            d_remaining = d[bool_mask].unsqueeze(1)

            w_ = (1 - (1 + b_remaining) * e_) / (1 - (1 - b_remaining) * e_)
            t = (2 * a_remaining * b_remaining) / (1 - (1 - b_remaining) * e_)

            # 计算接受概率
            accept = ((self.__m - 1.0) * t.log() - t + d_remaining) > torch.log(u)

            # 找到每行第一个接受的样本
            accept_idx = self.first_nonzero(accept, dim=1)
            valid_samples = accept_idx >= 0

            if valid_samples.any():
                # 获取接受的样本
                w_accepted = w_.gather(1, accept_idx[valid_samples].unsqueeze(1)).squeeze()
                e_accepted = e_.gather(1, accept_idx[valid_samples].unsqueeze(1)).squeeze()

                # 更新结果
                remaining_indices = torch.where(bool_mask)[0]
                valid_indices = remaining_indices[valid_samples]

                w[valid_indices] = w_accepted
                e[valid_indices] = e_accepted

                # 更新 bool_mask
                new_bool_mask = bool_mask.clone()  # 克隆 bool_mask
                new_bool_mask[valid_indices] = False
                bool_mask = new_bool_mask

        return e, w
    #
    # def __householder_rotation(self, x):
    #     u = self.__e1 - self.loc
    #     u = u / (u.norm(dim=-1, keepdim=True) + 1e-5)
    #     z = x - 2 * (x * u).sum(-1, keepdim=True) * u
    #     return z
    def __householder_rotation(self, x):
        """
        应用 Householder 变换
        Args:
            x: 输入张量 [batch_size, dim]
        Returns:
            变换后的张量 [batch_size, dim]
        """
        # 确保 e1 的形状正确
        e1 = self.__e1.expand(x.shape[0], -1)

        # 计算 Householder 向量
        u = e1 - self.loc
        u = u / (u.norm(dim=-1, keepdim=True) + 1e-5)

        # 应用变换
        z = x - 2 * (x * u).sum(-1, keepdim=True) * u
        return z

    def entropy(self):
        # option 1:
        output = (
            -self.scale
            * ive(self.__m / 2, self.scale)
            / ive((self.__m / 2) - 1, self.scale)
        )
        # option 2:
        # output = - self.scale * ive_fraction_approx(torch.tensor(self.__m / 2), self.scale)
        # option 3:
        # output = - self.scale * ive_fraction_approx2(torch.tensor(self.__m / 2), self.scale)

        return output.view(*(output.shape[:-1])) + self._log_normalization()

    def log_prob(self, x):
        return self._log_unnormalized_prob(x) - self._log_normalization()

    def _log_unnormalized_prob(self, x):
        output = self.scale * (self.loc * x).sum(-1, keepdim=True)

        return output.view(*(output.shape[:-1]))

    def _log_normalization(self):
        output = -(
            (self.__m / 2 - 1) * torch.log(self.scale)
            - (self.__m / 2) * math.log(2 * math.pi)
            - (self.scale + torch.log(ive(self.__m / 2 - 1, self.scale)))
        )

        return output.view(*(output.shape[:-1]))
# class VonMisesFisher(torch.distributions.Distribution):
#     def __init__(self, loc, scale, validate_args=None):
#         self.dtype = loc.dtype
#         self.loc = F.normalize(loc, dim=-1)
#         self.scale = scale
#         batch_shape = self.loc.shape[:-1]
#         super(VonMisesFisher, self).__init__(batch_shape, validate_args=validate_args)
#         self.__m = loc.shape[-1]
#         self.__e1 = torch.zeros(loc.shape[-1], device=loc.device)
#         self.__e1[0] = 1.
#
#     def rsample(self, shape=torch.Size()):
#         shape = shape if isinstance(shape, torch.Size) else torch.Size([shape])
#
#         batch_size = self.loc.shape[0]
#
#         # 使用批量采样方法
#         w = self._sample_w_batch(batch_size)
#
#         # 采样 v
#         v = torch.randn(batch_size, self.__m - 1, device=self.loc.device)
#         v = F.normalize(v, dim=-1)
#
#         # 构造完整的向量
#         w_ = torch.sqrt(torch.clamp(1 - (w ** 2), 1e-10))
#         w = w.view(batch_size, 1)
#         w_ = w_.view(batch_size, 1)
#
#         x = torch.cat([w, w_ * v], dim=1)
#
#         # 应用 Householder 变换
#         u = self.__e1.expand(batch_size, -1) - self.loc
#         u = u / (u.norm(dim=-1, keepdim=True) + 1e-5)
#         z = x - 2 * (x * u).sum(-1, keepdim=True) * u
#
#         return z
#
#     def _sample_w_batch(self, batch_size):
#         """批量采样方法"""
#         dim = self.__m
#         kappa = self.scale
#
#         # 使用固定数量的样本
#         num_samples = batch_size
#
#         # 生成 Beta 分布样本
#         beta = torch.distributions.Beta(0.5 * (dim - 1), 0.5 * (dim - 1))
#         beta_samples = beta.sample((num_samples,)).to(self.loc.device)
#
#         # 计算 w
#         w = 2 * beta_samples - 1
#
#         # 应用 von Mises-Fisher 分布的变换
#         w = w * torch.tanh(kappa)
#
#         return w
#
#     def log_prob(self, value):
#         return self._log_unnormalized_prob(value) - self._log_normalization()
#
#     def _log_unnormalized_prob(self, value):
#         return self.scale * torch.sum(self.loc * value, dim=-1)
#
#     def _log_normalization(self):
#         dim = self.__m
#         kappa = self.scale
#         return (
#                 (dim / 2 - 1) * torch.log(kappa)
#                 - (dim / 2) * math.log(2 * math.pi)
#                 - (kappa + torch.log(torch.tensor(1.0).to(kappa.device)))
#         )
@register_kl(VonMisesFisher, HypersphericalUniform)
def _kl_vmf_uniform(vmf, hyu):
    return -vmf.entropy() + hyu.entropy()
