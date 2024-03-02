import sys
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
	from torch import block_diag
except ImportError:
	def block_diag(*m):
		m = torch.cat([m1.unsqueeze(-3) for m1 in m], -3)
		eye = torch.eye(m.shape[-3], device=m.device)[:, None, :, None]
		return (m.unsqueeze(-2) * eye).reshape(torch.Size(torch.tensor(m.shape[-2:]) * m.shape[-3]))

try:
	from torch import kron
except ImportError:
	def kron(a, b):
		siz1 = torch.Size(torch.tensor(a.shape[-2:]) * torch.tensor(b.shape[-2:]))
		res = a.unsqueeze(-1).unsqueeze(-3) * b.unsqueeze(-2).unsqueeze(-4)
		siz0 = res.shape[:-4]
		return res.reshape(siz0 + siz1)

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'figure.max_open_warning': 0})
# plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams.update({'font.size': 16})


def init_NN(model, W_var, b_var):
	for m in model.modules():
		if isinstance(m, (nn.Linear, nn.Conv2d)):
			with torch.no_grad():
				fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
				m.weight.normal_(0, math.sqrt(W_var/fan_in))
				if m.bias is not None:
					if math.sqrt(b_var) > 0:
						m.bias.normal_(0, math.sqrt(b_var))
					else:
						m.bias.fill_(0.)
		elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
			nn.init.constant_(m.weight, 1)
			nn.init.constant_(m.bias, 0)
			nn.init.constant_(m.running_mean, 0)
			nn.init.constant_(m.running_var, 1)

def regularize_on_weight(model, anchor_model, method, W_var, b_var):
	if method == 'free':
		return torch.tensor(0.)

	if method == 'anc':
		assert anchor_model is not None
		anchor_model_modules = list(anchor_model.modules())

	l2 = 0
	for module_idx, m1 in enumerate(model.modules()):
		if isinstance(m1, (nn.Linear, nn.Conv2d)):
			fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m1.weight)
			if method == 'anc':
				m2 = anchor_model_modules[module_idx]
				l2 += ((m1.weight - m2.weight)**2).sum() / W_var * fan_in
				if m1.bias is not None:
					l2 += ((m1.bias - m2.bias)**2).sum() / b_var
			else:
				l2 += ((m1.weight)**2).sum() / W_var * fan_in
				if m1.bias is not None:
					l2 += ((m1.bias)**2).sum() / b_var
	return l2/2.

def regularize_on_weight_ll(model, method, W_var, b_var, n_ensemble):
	if method == 'free':
		return torch.tensor(0.)

	if method == 'anc':
		assert False, 'Anchored ensemble does not support weight sharing'

	l2 = 0
	for module_idx, m1 in enumerate(model.modules()):
		if isinstance(m1, (nn.Linear, nn.Conv2d)):
			fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m1.weight)
			l2 += ((m1.weight)**2).sum() / W_var * fan_in * n_ensemble
			if m1.bias is not None:
				l2 += ((m1.bias)**2).sum() / b_var * n_ensemble
		elif isinstance(m1, (ParallelLinear)):
			l2 += ((m1.weights)**2).sum() / W_var * m1.in_features
			if m1.biases is not None:
				l2 += ((m1.biases)**2).sum() / b_var

	return l2/2.

class ParallelLinear(nn.Module):
	def __init__(self, n_ensemble, in_features, out_features, bias=True):
		super(ParallelLinear, self).__init__()
		self.n_ensemble = n_ensemble
		self.in_features = in_features
		self.out_features = out_features
		self.bias = bias
		self.weights = nn.Parameter(torch.Tensor(n_ensemble, in_features, out_features))
		if self.bias:
			self.biases = nn.Parameter(torch.Tensor(n_ensemble, out_features))
		else:
			self.register_parameter('biases', None)

	def forward(self, input):
		if input.dim() == 2:
			out = torch.einsum('bi,nij->bnj', [input, self.weights])
		else:
			out = torch.einsum('bni,nij->bnj', [input, self.weights])
		if self.bias:
			out += self.biases
		return out

class ParallelPriorLinear(nn.Module):
	def __init__(self, n_ensemble, in_features, out_features, bias=True, W_var=2., b_var=0.01):
		super(ParallelPriorLinear, self).__init__()
		self.n_ensemble = n_ensemble
		self.in_features = in_features
		self.out_features = out_features
		self.bias = bias and b_var > 0
		self.W_std = math.sqrt(W_var/in_features)
		self.b_std = math.sqrt(b_var)

		self.register_buffer('weight_m', torch.empty(self.n_ensemble, self.in_features, self.out_features))
		if self.bias:
			self.register_buffer('bias_m', torch.empty(self.n_ensemble, self.out_features))

	@torch.no_grad()
	def forward(self, input):
		if input.dim() != 3: assert input.dim() == 2; input = input.unsqueeze(1)
		if input.size(1) < self.n_ensemble:
			assert self.n_ensemble % input.size(1) == 0
			input = input.repeat(1, self.n_ensemble//input.size(1), 1)
		self.weight_m.normal_(0, self.W_std)
		out = torch.einsum('bni,nij->bnj', [input, self.weight_m])
		if self.bias:
			self.bias_m.normal_(0, self.b_std)
			out += self.bias_m
		return out


class ParallelPriorConv2d(nn.Module):
	def __init__(self, n_ensemble, in_channels, out_channels, kernel_size, stride=1,
				 padding=0, dilation=1, groups=1, bias=True, W_var=2., b_var=0.01):
		super(ParallelPriorConv2d, self).__init__()
		self.n_ensemble = n_ensemble
		self.in_channels = in_channels
		self.out_channels = out_channels
		self.kernel_size = kernel_size
		self.stride = stride
		self.padding = padding
		self.dilation = dilation
		self.groups = groups
		self.bias = bias and b_var > 0
		self.W_std = math.sqrt(W_var/(in_channels // groups * self.kernel_size * self.kernel_size))
		self.b_std = math.sqrt(b_var)

		self.register_buffer('weight_m', torch.empty(self.n_ensemble*self.out_channels, self.in_channels//self.groups,
							 self.kernel_size, self.kernel_size))
		if self.bias:
			self.register_buffer('bias_m', torch.empty(self.n_ensemble, self.out_channels, 1, 1))

	@torch.no_grad()
	def forward(self, input):
		if input.dim() != 5: assert input.dim() == 4; input = input.unsqueeze(1)
		if input.size(1) < self.n_ensemble:
			assert self.n_ensemble % input.size(1) == 0
			input = input.repeat(1, self.n_ensemble // input.size(1), 1, 1, 1)
		self.weight_m.normal_(0, self.W_std)
		out = F.conv2d(input.flatten(1, 2),
					   self.weight_m,
					   bias=None,
					   stride=self.stride, dilation=self.dilation,
					   groups=self.groups*self.n_ensemble,
					   padding=self.padding)
		out = out.view(out.size(0), self.n_ensemble, self.out_channels, out.shape[2], out.shape[3])
		if self.bias:
			self.bias_m.normal_(0, self.b_std)
			out += self.bias_m
		return out


class ParallelPriorMaxPool2d(nn.Module):
	def __init__(self, kernel_size):
		super(ParallelPriorMaxPool2d, self).__init__()
		self.kernel_size = kernel_size

	def forward(self, x):
		assert x.dim() == 5
		out = F.max_pool2d(x.flatten(0,1), self.kernel_size)
		return out.view(x.size(0), x.size(1), x.size(2), out.size(-2), out.size(-1))

def gp_sample_and_estimate_kl(y_preds_original, y_preds_ref, n_sample, epsilon, W_var, b_var, label=None, tau=1):
	y_preds = y_preds_original.permute(0, 2, 1).flatten(1)
	y_pred_mean = y_preds.mean(0)
	y_preds_normalized = (y_preds-y_pred_mean)/math.sqrt(y_preds.shape[0])
	with torch.no_grad():
		eps = (y_preds_normalized.norm(dim=0, p=2)**2).mean() * epsilon
	y_pred_dist = torch.distributions.lowrank_multivariate_normal.LowRankMultivariateNormal(
					y_pred_mean, y_preds_normalized.T, eps*torch.ones_like(y_pred_mean))
	y_pred_samples = y_pred_dist.rsample((n_sample,)).view(n_sample, y_preds_original.shape[-1], y_preds_original.shape[-2])

	with torch.no_grad():
		y_preds_ref = y_preds_ref * tau
		y_pred_mean_ref = torch.zeros_like(y_pred_mean)
		cov_ref = (y_preds_ref.permute(1,0,2) @ y_preds_ref.permute(1,2,0)).mean(0)/y_preds_ref.shape[-1]
		cov_ref = cov_ref * W_var + b_var
		if label is not None:
			cov_ref[:label.shape[0], :label.shape[0]] *= (label[:, None] == label[None, :]).type_as(cov_ref)
		eps_ref = torch.trace(cov_ref) / cov_ref.shape[0] * epsilon
		cov_ref += eps_ref*torch.eye(cov_ref.shape[1], device=cov_ref.device)
		L_ref = torch.cholesky(cov_ref).contiguous()
		scale_tril = kron(torch.eye(y_preds_original.shape[-1], device=L_ref.device), L_ref)
		y_pred_dist_ref = torch.distributions.multivariate_normal.MultivariateNormal(
			y_pred_mean_ref, scale_tril=scale_tril)

	kl = torch.distributions.kl.kl_divergence(y_pred_dist, y_pred_dist_ref)
	return y_pred_samples, kl

def gp_sample(y_preds, epsilon, n_sample, diag_only=False):
	if diag_only:
		y_pred_mean = y_preds.mean(0)
		y_pred_std = y_preds.std(0, unbiased=False)
		y_pred_samples = torch.randn(n_sample, *y_pred_mean.shape, device=y_pred_mean.device).mul_(y_pred_std).add_(y_pred_mean)
		return y_pred_samples
	else:
		y_pred_mean = y_preds.mean(0)
		y_preds_normalized = (y_preds-y_pred_mean)/math.sqrt(y_preds.shape[0])
		eps = (y_preds_normalized.norm(dim=0, p=2)**2).mean(-1) * epsilon
		covs = y_preds_normalized.permute(1, 2, 0) @ y_preds_normalized.permute(1, 0, 2)
		covs += eps[:, None, None] * torch.eye(y_preds.shape[-1], device=y_preds.device).unsqueeze(0)
		L = torch.cholesky(covs).contiguous()
		scale_tril = block_diag(*L)
		y_pred_dist = torch.distributions.multivariate_normal.MultivariateNormal(y_pred_mean.flatten(), scale_tril=scale_tril)
		y_pred_samples = y_pred_dist.sample((n_sample,)).view(n_sample, y_preds.shape[-2], y_preds.shape[-1])
		return y_pred_samples

class _ECELoss(torch.nn.Module):
	def __init__(self, n_bins=15):
		"""
		n_bins (int): number of confidence interval bins
		"""
		super(_ECELoss, self).__init__()
		bin_boundaries = torch.linspace(0, 1, n_bins + 1)
		self.bin_lowers = bin_boundaries[:-1]
		self.bin_uppers = bin_boundaries[1:]

		bin_boundaries_plot = torch.linspace(0, 1, 11)
		self.bin_lowers_plot = bin_boundaries_plot[:-1]
		self.bin_uppers_plot = bin_boundaries_plot[1:]

	def forward(self, confidences, predictions, labels, title=None):
		accuracies = predictions.eq(labels)
		ece = torch.zeros(1, device=confidences.device)
		for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
			# Calculated |confidence - accuracy| in each bin
			in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
			prop_in_bin = in_bin.float().mean()
			if prop_in_bin.item() > 0:
				accuracy_in_bin = accuracies[in_bin].float().mean()
				avg_confidence_in_bin = confidences[in_bin].mean()
				ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

		accuracy_in_bin_list = []
		for bin_lower, bin_upper in zip(self.bin_lowers_plot, self.bin_uppers_plot):
			in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
			prop_in_bin = in_bin.float().mean()
			accuracy_in_bin = 0
			if prop_in_bin.item() > 0:
				accuracy_in_bin = accuracies[in_bin].float().mean().item()
			accuracy_in_bin_list.append(accuracy_in_bin)

		if title:
			fig = plt.figure(figsize=(8,6))
			p1 = plt.bar(np.arange(10) / 10., accuracy_in_bin_list, 0.1, align = 'edge', edgecolor ='black')
			p2 = plt.plot([0,1], [0,1], '--', color='gray')

			plt.ylabel('Accuracy', fontsize=18)
			plt.xlabel('Confidence', fontsize=18)
			#plt.title(title)
			plt.xticks(np.arange(0, 1.01, 0.2), fontsize=12)
			plt.yticks(np.arange(0, 1.01, 0.2), fontsize=12)
			plt.xlim(left=0,right=1)
			plt.ylim(bottom=0,top=1)
			plt.grid(True)
			#plt.legend((p1[0], p2[0]), ('Men', 'Women'))
			plt.text(0.1, 0.83, 'ECE: {:.4f}'.format(ece.item()), fontsize=18)
			fig.tight_layout()
			plt.savefig(title, format='pdf', dpi=600, bbox_inches='tight')

		return ece