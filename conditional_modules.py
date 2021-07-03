import math
import numpy as np
import torch
import torch.nn as nn

def convert_archcond(module, num_archs, cond_tracking=True):

	module_output = module
	if isinstance(module, (torch.nn.BatchNorm2d)):
		bn = module
		module_output = CondDynamicBatchNorm2d(bn.num_features,
											   bn.eps, bn.momentum,
											   bn.affine,
											   bn.track_running_stats,
											   cond_tracking,
											   num_archs)
		if bn.affine:
			with torch.no_grad():
				module_output.weight.data.copy_(bn.weight.data.expand_as(module_output.weight))
				module_output.bias.data.copy_(bn.bias.data.expand_as(module_output.bias))
		if bn.track_running_stats:
			with torch.no_grad():
				module_output.running_mean.data.copy_(
					bn.running_mean.data.expand_as(module_output.running_mean))
				module_output.running_var.data.copy_(
					bn.running_var.data.expand_as(module_output.running_var))
				module_output.num_batches_tracked.data.copy_(
					bn.num_batches_tracked.data.expand_as(module_output.num_batches_tracked))
		del module
		return module_output

	for name, child in module.named_children():
		module_output.add_module(name, convert_archcond(child, num_archs, cond_tracking))
	del module
	return module_output

class CondDynamicBatchNorm2d(nn.Module):

	def __init__(self, num_features, eps=1e-5, momentum=0.1,
				 affine=True, track_running_stats=True, cond_tracking=False,
				 num_archs=None):
		super(CondDynamicBatchNorm2d, self).__init__()

		self.num_features = num_features
		self.eps = eps
		self.momentum = momentum
		self.affine = affine
		self.track_running_stats = track_running_stats

		self.num_archs = num_archs
		self.cond_tracking = cond_tracking

		self.temp_not_track = False

		if cond_tracking:
			assert num_archs

		if self.affine:
			self.weight = nn.Parameter(torch.Tensor(num_features))
			self.bias = nn.Parameter(torch.Tensor(num_features))
		else:
			self.register_parameter('weight', None)
			self.register_parameter('bias', None)
		if self.track_running_stats:
			if self.cond_tracking:
				self.register_buffer('running_mean', torch.zeros(num_archs, num_features))
				self.register_buffer('running_var', torch.ones(num_archs, num_features))
				self.register_buffer('num_batches_tracked', torch.zeros(num_archs, dtype=torch.long))
			else:
				self.register_buffer('running_mean', torch.zeros(num_features))
				self.register_buffer('running_var', torch.ones(num_features))
				self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
		else:
			self.register_parameter('running_mean', None)
			self.register_parameter('running_var', None)
			self.register_parameter('num_batches_tracked', None)
		self.reset_parameters()
		self.active_arch = None

	def reset_running_stats(self):
		if self.track_running_stats:
			self.running_mean.zero_()
			self.running_var.fill_(1)
			self.num_batches_tracked.zero_()

	def reset_parameters(self):
		self.reset_running_stats()
		if self.affine:
			self.weight.data.fill_(1.0)
			self.bias.data.zero_()

	def forward(self, input):
		if self.cond_tracking:
			assert self.active_arch is not None

		feature_dim = input.size(1)

		exponential_average_factor = 0.0

		if self.training and self.track_running_stats:
			if self.cond_tracking:
				self.num_batches_tracked[self.active_arch] += 1
			else:
				self.num_batches_tracked += 1
			if self.momentum is None:  # use cumulative moving average
				num_tracked = self.num_batches_tracked[self.active_arch].item() \
					if self.cond_tracking else self.num_batches_tracked.item()
				exponential_average_factor = 1.0 / num_tracked
			else:  # use exponential moving average
				exponential_average_factor = self.momentum

		if self.track_running_stats and not self.temp_not_track:
			if self.cond_tracking:
				running_mean = self.running_mean[self.active_arch, :feature_dim]
				running_var = self.running_var[self.active_arch, :feature_dim]
			else:
				running_mean = self.running_mean[:feature_dim]
				running_var = self.running_var[:feature_dim]
		else:
			running_mean = None
			running_var = None

		if self.affine:
			weight = self.weight[:feature_dim]
			bias = self.bias[:feature_dim]
		else:
			weight = None
			bias = None

		out = nn.functional.batch_norm(
			input, running_mean, running_var, weight, bias,
			self.training or not self.track_running_stats,
			exponential_average_factor, self.eps)
		return out

	def extra_repr(self):
		return '{num_features}, num_archs={num_archs}, eps={eps}, momentum={momentum}, ' \
			   'affine={affine}, track_running_stats={track_running_stats}, ' \
			   'cond_tracking={cond_tracking}'.format(**self.__dict__)

class CondLinear(nn.Module):
    __constants__ = ['bias', 'in_features', 'out_features', 'num_archs']
    def __init__(self, in_features, out_features, bias=True, num_archs=1):
        super(CondLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.active_arch = None
        self.num_archs = num_archs
        self.weight = nn.Parameter(torch.Tensor(num_archs, out_features, in_features))
        if bias is None or bias is False:
            self.register_parameter('bias', None)
        else:
            self.bias = nn.Parameter(torch.Tensor(num_archs, out_features))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(2))
        for i in range(self.num_archs):
            self.weight[i].data.uniform_(-stdv, stdv)
            if self.bias is not None:
                self.bias[i].data.uniform_(-stdv, stdv)

    def forward(self, input):
        weight = self.weight[self.active_arch]
        bias = self.bias[self.active_arch] if self.bias is not None else None
        out = torch.nn.functional.linear(input, weight, bias)
        return out

    def extra_repr(self):
        r"""
        Overriden.
        """
        return 'in_features={}, out_features={}, bias={}, num_archs={}'.format(
            self.in_features, self.out_features, self.bias is not None, self.num_archs)

def set_active_arch(model, args, active_arch=None):
	if active_arch is None:
		active_arch = np.random.randint(args.num_archs)
	config = args.config[active_arch]
	if isinstance(model, torch.nn.parallel.DistributedDataParallel):
		model.module.set_active_subnet(ks=config[0], e=config[1], d=config[2])
	else:
		model.set_active_subnet(ks=config[0], e=config[1], d=config[2])
	for m in model.modules():
		if isinstance(m, (CondDynamicBatchNorm2d, CondLinear)):
			m.active_arch = active_arch

def reset_stats_for_condbn(model, active_arch):
	for m in model.modules():
		if isinstance(m, CondDynamicBatchNorm2d):
			if m.cond_tracking:
				mask = torch.zeros(m.num_archs, 1, dtype=torch.float, device=m.running_mean.device)
				mask[active_arch, 0] = 1
				m.running_mean.data.mul_(mask)
				m.running_var.data.mul_(mask)

def scale_stats_for_condbn(model, scale):
	for m in model.modules():
		if isinstance(m, CondDynamicBatchNorm2d):
			if m.cond_tracking:
				m.running_mean.data.mul_(scale)
				m.running_var.data.mul_(scale)
