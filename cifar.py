import argparse
import os
import shutil
import time
from functools import partial
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Variable

import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from timm.utils import AverageMeter

from utils_degp import init_NN, regularize_on_weight, ParallelPriorLinear, \
	ParallelPriorConv2d, ParallelPriorMaxPool2d, gp_sample_and_estimate_kl, \
	gp_sample, _ECELoss
from conditional_modules import convert_archcond, CondDynamicBatchNorm2d

parser = argparse.ArgumentParser(description='DEGP on CIFAR-10')
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
					help='number of data loading workers (default: 8)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
					help='random seed (default: 1)')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
					help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
					help='manual epoch number (useful on restarts)')
parser.add_argument('--milestones', default=[100, 150], type=int, nargs='+')
parser.add_argument('-b', '--batch-size', default=128, type=int,
					metavar='N', help='mini-batch size (default: 128)')
parser.add_argument('--test-batch-size', default=1000, type=int)
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
					metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
					help='momentum')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
					help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
					help='evaluate model on validation set')
parser.add_argument('--save-dir', dest='save_dir',
					help='The directory used to save the trained models',
					default='/data/zhijie/snapshots_degp/', type=str)
parser.add_argument('--dataset', type=str, default='cifar10')
parser.add_argument('--data-root', type=str, default='/data/LargeData/Regular/cifar')

parser.add_argument('--n_ensemble', type=int, default=10)
parser.add_argument('--arch', type=str, default='resnet20')
parser.add_argument('--W_var', type=float, default=2.)
parser.add_argument('--b_var', type=float, default=0.01)

parser.add_argument('--method', type=str, default='free')
parser.add_argument('--w_alpha', type=float, default=0.1)
parser.add_argument('--f_alpha', type=float, default=1.)
parser.add_argument('--ip', type=str, default=None)
parser.add_argument('--prior_arch', type=str, default='resnet20')
parser.add_argument('--n_ensembles_prior', type=int, default=10)
parser.add_argument('--epsilon', type=float, default=5e-2)
parser.add_argument('--n_sample', type=int, default=256)
parser.add_argument('--tau_lr', type=float, default=1e-3)
parser.add_argument('--with_w_reg', action='store_true', default=False)
parser.add_argument('--remove_residual', action='store_true', default=False)
parser.add_argument('--lr_warmup', type=int, default=5)
parser.add_argument('--tau_init', type=float, default=0.1)
parser.add_argument('--not_use_bn', action='store_true', default=False)
parser.add_argument('--diag_only', action='store_true', default=False)

best_prec1 = 0


def main():
	global args, best_prec1
	args = parser.parse_args()
	args.use_bn = not args.not_use_bn
	args.token = '{}_{}_ens{}_alpha{}-{}_ip{}_wreg{}_{}{}{}{}{}'.format(
		args.arch, args.method, args.n_ensemble, args.w_alpha,
		args.f_alpha, args.ip, args.with_w_reg, args.seed,
		'_{}'.format(args.prior_arch) if args.prior_arch != args.arch and args.method == 'our' else '',
		'_rr' if args.remove_residual else '',
		'_nobn' if not args.use_bn else '',
		'_cifar100' if args.dataset=='cifar100' else '')
	args.save_dir = os.path.join(args.save_dir, args.token)

	# Check the save_dir exists or not
	if not os.path.exists(args.save_dir):
		os.makedirs(args.save_dir)

	np.random.seed(args.seed)
	torch.manual_seed(args.seed)
	torch.cuda.manual_seed_all(args.seed)

	model_func = partial(eval(args.arch), num_classes=100 if args.dataset == 'cifar100' else 10)
	models = []
	for _ in range(args.n_ensemble):
		model = model_func(args.use_bn)
		if args.method == 'our':
			model = convert_archcond(model, None, False)
		model.cuda()
		init_NN(model, args.W_var, args.b_var)
		models.append(model)
	tau = nn.Parameter(torch.tensor(args.tau_init).cuda())

	if args.method == 'our':
		prior_model = to_prior_nn(eval(args.prior_arch)(args.use_bn), args).cuda()
	else:
		prior_model = None

	# print(models[-1])
	# print(prior_model)

	if args.method == 'anc':
		anchor_models = [model_func(args.use_bn).cuda() for _ in range(args.n_ensemble)]
		for anchor_model in anchor_models:
			init_NN(anchor_model, args.W_var, args.b_var)
			for p in anchor_model.parameters():
				p.requires_grad_(False)
	else:
		anchor_models = None

	# define loss function (criterion) and optimizer
	criterion = nn.CrossEntropyLoss().cuda()

	params = []
	for model in models:
		for param in model.parameters():
			if param.requires_grad:
				params.append(param)
	optimizer = torch.optim.SGD(params, args.lr,
								momentum=args.momentum,
								weight_decay=0)

	optimizer_tau = torch.optim.Adam([tau], lr=args.tau_lr)

	# optionally resume from a checkpoint
	if args.resume:
		if args.resume == 'auto':
			args.resume = os.path.join(args.save_dir, 'checkpoint.th')
		if os.path.isfile(args.resume):
			print("=> loading checkpoint '{}'".format(args.resume))
			checkpoint = torch.load(args.resume, map_location='cpu')
			args.start_epoch = checkpoint['epoch']
			best_prec1 = checkpoint['best_prec1']
			for model_idx, model in enumerate(models):
				model.load_state_dict(checkpoint['state_dict_{}'.format(model_idx)])
			if 'optimizer' in checkpoint:
				optimizer.load_state_dict(checkpoint['optimizer'])
			if 'optimizer_tau' in checkpoint:
				optimizer_tau.load_state_dict(checkpoint['optimizer_tau'])
			if 'tau' in checkpoint:
				tau.data.fill_(checkpoint['tau'])
			print("=> loaded checkpoint '{}' (epoch {} acc {})"
				  .format(args.resume, checkpoint['epoch'], checkpoint['prec1']))
		else:
			print("=> no checkpoint found at '{}'".format(args.resume))

	lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
														milestones=args.milestones,
														last_epoch=args.start_epoch - 1)

	cudnn.benchmark = True

	if args.dataset == 'cifar10':
		normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
										 std=[0.229, 0.224, 0.225])
		data_mean = torch.tensor([0.485, 0.456, 0.406]).view(1,-1,1,1).cuda()
		data_std = torch.tensor([0.229, 0.224, 0.225]).view(1,-1,1,1).cuda()

		train_loader = torch.utils.data.DataLoader(
			datasets.CIFAR10(root=args.data_root, train=True,
			transform=transforms.Compose([
				transforms.RandomHorizontalFlip(),
				transforms.RandomCrop(32, 4),
				transforms.ToTensor(),
				normalize,
			]), download=True),
			batch_size=args.batch_size, shuffle=True,
			num_workers=args.workers, pin_memory=True)

		val_loader = torch.utils.data.DataLoader(
			datasets.CIFAR10(root=args.data_root, train=False,
			transform=transforms.Compose([
				transforms.ToTensor(),
				normalize,
			])),
			batch_size=args.test_batch_size, shuffle=False,
			num_workers=args.workers, pin_memory=True)
	elif args.dataset == 'cifar100':
		normalize = transforms.Normalize(mean=[x / 255 for x in [129.3, 124.1, 112.4]],
										 std=[x / 255 for x in [68.2, 65.4, 70.4]])
		data_mean = torch.tensor([x / 255 for x in [129.3, 124.1, 112.4]]).view(1,-1,1,1).cuda()
		data_std = torch.tensor([x / 255 for x in [68.2, 65.4, 70.4]]).view(1,-1,1,1).cuda()

		train_loader = torch.utils.data.DataLoader(
			datasets.CIFAR100(root=args.data_root, train=True,
			transform=transforms.Compose([
				transforms.RandomHorizontalFlip(),
				transforms.RandomCrop(32, 4),
				transforms.ToTensor(),
				normalize,
			]), download=True),
			batch_size=args.batch_size, shuffle=True,
			num_workers=args.workers, pin_memory=True)

		val_loader = torch.utils.data.DataLoader(
			datasets.CIFAR100(root=args.data_root, train=False,
			transform=transforms.Compose([
				transforms.ToTensor(),
				normalize,
			])),
			batch_size=args.test_batch_size, shuffle=False,
			num_workers=args.workers, pin_memory=True)
	else:
		raise NotImplementedError

	if args.evaluate:
		validate(args, val_loader, models, criterion, tau)
		eval_corrupted_data(args, models, criterion, tau, data_mean, data_std)
		return

	start_time = time.time()
	epoch_time = AverageMeter()

	w_norms = []
	for epoch in range(args.start_epoch, args.epochs):

		need_hour, need_mins, need_secs = convert_secs2time(epoch_time.avg * (args.epochs-epoch))
		need_time = '[Need: {:02d}:{:02d}:{:02d}]'.format(need_hour, need_mins, need_secs)
		print('==>>{:s} [Epoch={:03d}/{:03d}] {:s}'.format(
									time_string(), epoch, args.epochs, need_time) \
					+ ' [Best : Accuracy={:.4f}]'.format(best_prec1))

		# train for one epoch
		train(args, train_loader, models, prior_model, anchor_models, tau,
			  criterion, optimizer, optimizer_tau, epoch, data_mean, data_std)
		lr_scheduler.step()

		with torch.no_grad():
			tmp = 0
			for model_idx, model in enumerate(models):
				for param in model.parameters():
					tmp += ((param)**2).sum()
			tmp = tmp.item() / len(models)
			print(tmp)
			w_norms.append(tmp)

		# evaluate on validation set
		_, prec1, _ = validate(args, val_loader, models, criterion, tau)

		# remember best prec@1 and save checkpoint
		best_prec1 = max(prec1, best_prec1)

		if epoch % 10 == 0 or epoch == args.epochs - 1:
			ckpt = {'epoch': epoch + 1, 'best_prec1': best_prec1,
					'prec1': prec1, 'tau': tau.item()}
			for model_idx, model in enumerate(models):
				ckpt['state_dict_{}'.format(model_idx)] = model.state_dict()
			ckpt['optimizer'.format(model_idx)] = optimizer.state_dict()
			ckpt['optimizer_tau'.format(model_idx)] = optimizer_tau.state_dict()
			torch.save(ckpt, os.path.join(args.save_dir, 'checkpoint.th'))

		epoch_time.update(time.time() - start_time)
		start_time = time.time()

	probs, labels, mis = validate(args, val_loader, models, criterion, tau,
								  ret_probs_and_labels=True, suffix=None)
	ood_dataset = datasets.SVHN(args.data_root.replace('cifar', 'svhn'),
		split='test', download=True,
		transform=transforms.Compose([
			transforms.ToTensor(),
			normalize,
		]))
	ood_loader = torch.utils.data.DataLoader(ood_dataset, batch_size=args.test_batch_size,
		num_workers=args.workers, pin_memory=True, shuffle=False)
	ood_probs, ood_labels, ood_mis = validate(args, ood_loader, models,
											  criterion, tau,
											  ret_probs_and_labels=True,
											  suffix=None)

	is_correct = np.concatenate([(probs.argmax(1) == labels).astype(np.float32),
								 np.zeros((ood_labels.shape[0]))])

	mis = np.concatenate([mis, ood_mis])
	mis /= mis.max()
	mis = 1 - mis
	ths = np.linspace(0, 1, 300)[:299]
	accs = []
	for th in ths:
		if th >= mis.max():
			accs.append(accs[-1])
		else:
			accs.append(is_correct[mis > th].mean())
	np.save('accs/cifar/{}.npy'.format(args.token), np.array(accs))

	w_norms = np.array(w_norms)
	np.save('accs/cifar/wnorms_{}.npy'.format(args.token), w_norms)

	acc_wrt_ens = np.zeros((args.n_ensemble, 3))
	for i in range(1, args.n_ensemble + 1):
		test_loss, test_acc, test_ece = validate(args, val_loader, models[:i],
												 criterion, tau, suffix=None)
		acc_wrt_ens[i-1, 0] = test_loss
		acc_wrt_ens[i-1, 1] = test_acc
		acc_wrt_ens[i-1, 2] = test_ece
	np.save('accs/cifar/acc_wrt_ens_{}.npy'.format(args.token), acc_wrt_ens)

	eval_corrupted_data(args, models, criterion, tau, data_mean, data_std)

def train(args, train_loader, models, prior_model, anchor_models, tau,
		  criterion, optimizer, optimizer_tau, epoch, data_mean, data_std):
	batch_time = AverageMeter()
	data_time = AverageMeter()
	losses = AverageMeter()
	f_regs = AverageMeter()
	w_regs = AverageMeter()
	top1 = AverageMeter()

	# switch to train mode
	for model in models: model.train()

	end = time.time()
	for i, (data, label) in enumerate(train_loader):
		# measure data loading time
		data_time.update(time.time() - end)

		if epoch < args.lr_warmup:
			for param_group in optimizer.param_groups:
				param_group['lr'] = args.lr * float(i + epoch*len(train_loader)) \
									/(args.lr_warmup*len(train_loader))

		data = data.cuda(non_blocking=True)
		label = label.cuda(non_blocking=True)

		# compute output
		outputs = [model(data) for model in models]

		if args.method == 'our':
			assert args.n_ensemble > 1

			ip = None
			if args.ip:
				if args.ip == 'uniform':
					ip = (torch.empty_like(data).uniform_(0, 1) - data_mean) / data_std

			if ip is not None:
				for model in models:
					for m in model.modules():
						if isinstance(m, (CondDynamicBatchNorm2d)):
							m.temp_not_track = True
				ip_outputs = [model(ip) for model in models]
				for model in models:
					for m in model.modules():
						if isinstance(m, (CondDynamicBatchNorm2d)):
							m.temp_not_track = False
				outputs = [torch.cat([output, ip_output], 0)
					for output, ip_output in zip(outputs, ip_outputs)]

				data = torch.cat([data, ip], 0)

			with torch.no_grad():
				with torch.cuda.amp.autocast():
					y_preds_ref = prior_model(data).float()

			y_pred_samples, kl = gp_sample_and_estimate_kl(torch.stack(outputs, 0),
				y_preds_ref, args.n_sample, args.epsilon, args.W_var, args.b_var)
			y_pred_samples = y_pred_samples[:, :, :label.shape[0]].permute(0, 2, 1).flatten(0,1)

			loss = criterion(y_pred_samples.div(tau),
							 label.repeat(args.n_sample)) * args.n_ensemble
			f_reg = kl / label.size(0) * args.n_ensemble

			pred = y_pred_samples.argmax(dim=1, keepdim=True)
			correct = pred.eq(label.repeat(args.n_sample).view_as(pred)).sum().item()/float(args.n_sample)

			if args.with_w_reg:
				w_reg = sum([regularize_on_weight(model, None, 'reg', args.W_var, args.b_var)
								for model in models]) / len(train_loader.dataset)
			else:
				w_reg = torch.tensor(0.).cuda()
		else:
			loss = sum([criterion(output, label) for output in outputs])
			f_reg = torch.tensor(0.).cuda()
			w_reg = 0
			for model_idx, model in enumerate(models):
				anchor_model = None if anchor_models is None else anchor_models[model_idx]
				w_reg += regularize_on_weight(model, anchor_model, args.method,
											  args.W_var, args.b_var)
			w_reg = w_reg / len(train_loader.dataset)

			correct = 0
			for output in outputs:
				pred = output.argmax(dim=1, keepdim=True)
				correct += pred.eq(label.view_as(pred)).sum().item()/float(args.n_ensemble)

		optimizer.zero_grad()
		optimizer_tau.zero_grad()
		(loss + f_reg * args.f_alpha + w_reg * args.w_alpha).backward()
		optimizer.step()
		optimizer_tau.step()

		# record loss
		losses.update(loss.item()/args.n_ensemble, label.size(0))
		f_regs.update(f_reg.item()/args.n_ensemble, 1)
		w_regs.update(w_reg.item()/args.n_ensemble, 1)
		top1.update(correct/label.size(0), label.size(0))

		# measure elapsed time
		batch_time.update(time.time() - end)
		end = time.time()

	print('\tLr: {lr:.4f}, '
		  'Time {batch_time.avg:.3f}, '
		  'Data {data_time.avg:.3f}, '
		  'Loss {loss.avg:.4f}, '
		  'F Reg {f_reg.avg:.4f}, '
		  'W Reg {w_reg.avg:.4f}, '
		  'Tau {tau:.4f}, '
		  'Prec@1 {top1.avg:.4f}'.format(
			  lr=optimizer.param_groups[0]['lr'], batch_time=batch_time,
			  data_time=data_time, loss=losses, f_reg=f_regs, w_reg=w_regs,
			  tau=tau.item(), top1=top1))


def validate(args, val_loader, models, criterion, tau,
			 ret_probs_and_labels=False, suffix=''):
	# switch to evaluate mode
	for model in models: model.eval()

	test_loss, correct = 0, 0
	probs, labels, mis = [], [], []
	with torch.no_grad():
		for data, target in val_loader:
			data = data.cuda(non_blocking=True)
			target = target.cuda(non_blocking=True)

			with torch.cuda.amp.autocast():
				output = torch.stack([model(data) for model in models], 0).float()
			if args.method == 'our' and len(models) > 1:
				output = gp_sample(output, args.epsilon, 1000, args.diag_only).div(tau)
			mis.append(ent(output.softmax(-1).mean(0)) - ent(output.softmax(-1)).mean(0))
			output = output.softmax(-1).mean(0).log()
			test_loss += criterion(output, target).item() * target.size(0)  # sum up batch loss
			pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
			correct += pred.eq(target.view_as(pred)).sum().item()

			probs.append(output)
			labels.append(target)

		labels = torch.cat(labels)
		probs = torch.cat(probs).softmax(-1)
		mis = torch.cat(mis)
		confidences, predictions = torch.max(probs, 1)
		ece_func = _ECELoss().cuda()
		ece = ece_func(confidences, predictions, labels,
					   title=None if suffix is None else 'eces/cifar/{}{}.pdf'.format(args.token, suffix))

	test_loss /= len(val_loader.dataset)
	top1 = float(correct) / len(val_loader.dataset)
	print('\tN_ensemble: {}, Test set: Average loss: {:.4f}, Accuracy: {:.4f}, ECE: {:.4f}'.format(
		len(models), test_loss, top1, ece.item()))
	if ret_probs_and_labels:
		return probs.data.cpu().numpy(), labels.data.cpu().numpy(), mis.data.cpu().numpy()
	return test_loss, top1, ece.item()

def eval_corrupted_data(args, models, criterion, tau, data_mean, data_std):
	corrupted_data_path = './CIFAR-10-C/CIFAR-10-C' if args.dataset == 'cifar10' else './CIFAR-100-C/CIFAR-100-C'
	corrupted_data_files = os.listdir(corrupted_data_path)
	corrupted_data_files.remove('labels.npy')
	corrupted_data_files.remove('README.txt')
	results = np.zeros((5, len(corrupted_data_files), 3))
	labels = torch.from_numpy(np.load(os.path.join(corrupted_data_path, 'labels.npy'), allow_pickle=True)).long()
	for ii, corrupted_data_file in enumerate(corrupted_data_files):
		corrupted_data = np.load(os.path.join(corrupted_data_path, corrupted_data_file), allow_pickle=True)
		for i in range(5):
			print(corrupted_data_file, i)
			images = torch.from_numpy(corrupted_data[i*10000:(i+1)*10000]).float().permute(0, 3, 1, 2)/255.
			images = (images - data_mean.cpu())/data_std.cpu()
			corrupted_dataset = torch.utils.data.TensorDataset(images, labels[i*10000:(i+1)*10000])
			corrupted_loader = torch.utils.data.DataLoader(corrupted_dataset,
				batch_size=args.test_batch_size, shuffle=False, num_workers=args.workers,
				pin_memory=False, sampler=None, drop_last=False)
			r1, r2, r3 = validate(args, corrupted_loader, models, criterion, tau, suffix=None)
			results[i, ii] = np.array([r1, r2, r3])
	print(results.mean(1)[:, 2])
	np.save('corrupted_results/{}.npy'.format(args.token), results)

def ent(p):
    return -(p*p.add(1e-8).log()).sum(-1)

class Identity(nn.Module):
	def __init__(self, *args):
		super(Identity, self).__init__()
	def forward(self, x):
		return x

class Zero(nn.Module):
	def __init__(self):
		super(Zero, self).__init__()
	def forward(self, x):
		return 0

def to_prior_nn(input, args):
	if isinstance(input, (nn.Conv2d, nn.Linear, nn.BatchNorm2d)):
		if isinstance(input, (nn.Conv2d)):
			output = ParallelPriorConv2d(args.n_ensembles_prior, input.in_channels,
										 input.out_channels,
										 input.kernel_size[0], input.stride,
										 input.padding, input.dilation,
										 input.groups, input.bias is not None,
										 args.W_var, args.b_var)
		elif isinstance(input, (nn.Linear, nn.BatchNorm2d)):
			output = Identity()
		del input
		return output

	if isinstance(input, (LambdaLayer)) and args.remove_residual:
		output = Zero()
		del input
		return output

	output = input
	for name, module in input.named_children():
		output.add_module(name, to_prior_nn(module, args))
	del input
	return output

class LambdaLayer(nn.Module):
	def __init__(self, lambd):
		super(LambdaLayer, self).__init__()
		self.lambd = lambd

	def forward(self, x):
		return self.lambd(x)

class BasicBlock(nn.Module):
	expansion = 1

	def __init__(self, in_planes, planes, use_bn, norm_fn, stride=1, option='A'):
		super(BasicBlock, self).__init__()
		self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride,
							   padding=1, bias=not use_bn)
		self.bn1 = norm_fn(planes)
		self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
							   padding=1, bias=not use_bn)
		self.bn2 = norm_fn(planes)

		self.shortcut = LambdaLayer(lambda x: x)
		if stride != 1 or in_planes != planes:
			if option == 'A':
				"""
				For CIFAR10 ResNet paper uses option A.
				"""
				self.shortcut = LambdaLayer(lambda x:
											F.pad(x[..., ::2, ::2],
												  (0, 0, 0, 0, planes//4, planes//4),
												  "constant", 0
												  )
											)
			elif option == 'B':
				self.shortcut = nn.Sequential(
					 nn.Conv2d(in_planes, self.expansion * planes,
					 		   kernel_size=1, stride=stride, bias=not use_bn),
					 nn.BatchNorm2d(self.expansion * planes)
				)

	def forward(self, x):
		out = F.relu(self.bn1(self.conv1(x)))
		out = self.bn2(self.conv2(out))
		out += self.shortcut(x)
		out = F.relu(out)
		return out

class ResNet(nn.Module):
	def __init__(self, block, num_blocks, num_classes=10, use_bn=True):
		super(ResNet, self).__init__()
		self.in_planes = 16

		norm_fn = nn.BatchNorm2d if use_bn else Identity
		self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=not use_bn)
		self.bn1 = norm_fn(16)
		self.layer1 = self._make_layer(block, 16, num_blocks[0], use_bn, norm_fn, stride=1)
		self.layer2 = self._make_layer(block, 32, num_blocks[1], use_bn, norm_fn, stride=2)
		self.layer3 = self._make_layer(block, 64, num_blocks[2], use_bn, norm_fn, stride=2)
		self.linear = nn.Linear(64, num_classes)

	def _make_layer(self, block, planes, num_blocks, use_bn, norm_fn, stride):
		strides = [stride] + [1]*(num_blocks-1)
		layers = []
		for stride in strides:
			layers.append(block(self.in_planes, planes, use_bn, norm_fn, stride))
			self.in_planes = planes * block.expansion
		return nn.Sequential(*layers)

	def forward(self, x):
		out = F.relu(self.bn1(self.conv1(x)))
		out = self.layer1(out)
		out = self.layer2(out)
		out = self.layer3(out)
		out = out.mean([-2, -1])
		out = self.linear(out)
		return out

def resnet20(use_bn, num_classes=10):
	return ResNet(BasicBlock, [3, 3, 3], use_bn=use_bn, num_classes=num_classes)

def resnet32(use_bn, num_classes=10):
	return ResNet(BasicBlock, [5, 5, 5], use_bn=use_bn, num_classes=num_classes)

def resnet44(use_bn, num_classes=10):
	return ResNet(BasicBlock, [7, 7, 7], use_bn=use_bn, num_classes=num_classes)

def resnet56(use_bn, num_classes=10):
	return ResNet(BasicBlock, [9, 9, 9], use_bn=use_bn, num_classes=num_classes)

def resnet110(use_bn, num_classes=10):
	return ResNet(BasicBlock, [18, 18, 18], use_bn=use_bn, num_classes=num_classes)

def resnet1202(use_bn, num_classes=10):
	return ResNet(BasicBlock, [200, 200, 200], use_bn=use_bn, num_classes=num_classes)

def time_string():
  ISOTIMEFORMAT='%Y-%m-%d %X'
  string = '[{}]'.format(time.strftime( ISOTIMEFORMAT, time.gmtime(time.time()) ))
  return string

def convert_secs2time(epoch_time):
  need_hour = int(epoch_time / 3600)
  need_mins = int((epoch_time - 3600*need_hour) / 60)
  need_secs = int(epoch_time - 3600*need_hour - 60*need_mins)
  return need_hour, need_mins, need_secs


if __name__ == '__main__':
	main()
