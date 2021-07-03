from __future__ import print_function
import copy
import math
import numpy as np
import argparse
import itertools
from functools import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import Tensor
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR

from utils_degp import init_NN, regularize_on_weight, ParallelPriorLinear, \
	ParallelPriorConv2d, ParallelPriorMaxPool2d, gp_sample_and_estimate_kl, \
	gp_sample, _ECELoss
from conditional_modules import convert_archcond, CondDynamicBatchNorm2d

def ent(p):
    return -(p*p.add(1e-8).log()).sum(-1)

def fn_make_NN(arch, dropout=0., D_in=[1, 28, 28], D_out=10):
	if arch == 'fashionCNN':
		model = torch.nn.Sequential(
			nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
			nn.BatchNorm2d(32),
            nn.ReLU(), nn.MaxPool2d(2),
			nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
			nn.BatchNorm2d(64),
            nn.ReLU(), nn.MaxPool2d(2),
			nn.Flatten(1),
			nn.Linear(64*6*6, 256), nn.ReLU(),
			nn.Linear(256, 10)
		)
	elif 'mlp' in arch:
		n_layer = int(float(arch.split('_')[1]))
		n_hidden = int(float(arch.split('_')[2]))
		layers = [torch.nn.Flatten(1),
			torch.nn.Linear(np.prod(D_in), n_hidden),
			torch.nn.ReLU(),
			torch.nn.Dropout(dropout),
			torch.nn.Linear(n_hidden, D_out)]
		for _ in range(n_layer - 1):
			layers.insert(4, torch.nn.Linear(n_hidden, n_hidden))
			layers.insert(5, torch.nn.ReLU())
			layers.insert(6, torch.nn.Dropout(dropout))
		model = torch.nn.Sequential(*layers)
	return model

def fn_make_prior_NN(n_ensembles, W_var, b_var, arch, D_in=[1, 28, 28], D_out=10):
	if arch == 'fashionCNN':
		if len(n_ensembles) == 1:
			n_ensembles = n_ensembles * 3
		assert len(n_ensembles) == 3
		model = torch.nn.Sequential(
			ParallelPriorConv2d(n_ensembles[0], 1, 32, 3, padding=1,
								W_var=W_var, b_var=b_var),
			nn.ReLU(),
			ParallelPriorMaxPool2d(2),
			ParallelPriorConv2d(n_ensembles[1], 32, 64, 3,
								W_var=W_var, b_var=b_var),
			nn.ReLU(),
			ParallelPriorMaxPool2d(2),
			nn.Flatten(2),
			ParallelPriorLinear(n_ensembles[2], 64*6*6, 256,
								W_var=W_var, b_var=b_var),
			nn.ReLU(),
		)
	elif arch == 'small_fashionCNN':
		if len(n_ensembles) == 1:
			n_ensembles = n_ensembles * 2
		assert len(n_ensembles) == 2
		model = torch.nn.Sequential(
			ParallelPriorConv2d(n_ensembles[0], 1, 64, 5,
								W_var=W_var, b_var=b_var),
			nn.ReLU(),
			ParallelPriorMaxPool2d(4),
			nn.Flatten(2),
			ParallelPriorLinear(n_ensembles[1], 64*6*6, 256,
								W_var=W_var, b_var=b_var),
			nn.ReLU(),
		)
	elif 'mlp' in arch:
		n_layer = int(float(arch.split('_')[1]))
		n_hidden = int(float(arch.split('_')[2]))
		if len(n_ensembles) == 1:
			n_ensembles = n_ensembles * (n_layer)
		assert len(n_ensembles) == n_layer
		layers = [nn.Flatten(1)]
		for i, n_ensemble in enumerate(n_ensembles):
			in_features, out_features = n_hidden, n_hidden
			if i == 0:
				in_features = np.prod(D_in)
			layers.append(ParallelPriorLinear(n_ensemble, in_features, out_features,
											  W_var=W_var, b_var=b_var))
			layers.append(nn.ReLU())
		model = torch.nn.Sequential(*layers)
	return model

def train(args, epoch, models, device, train_loader, tau, optimizer_tau,
		  optimizer, prior_model=None, anchor_models=None):
	for model in models: model.train()
	correct = 0

	for batch_idx, (data, label) in enumerate(train_loader):
		data, label = data.to(device), label.to(device)

		outputs = [model(data) for model in models]
		if args.method == 'our':
			assert args.n_ensemble > 1 and args.dropout == 0

			ip = None
			if args.ip:
				if args.ip == 'uniform':
					ip = torch.empty_like(data).uniform_(-1, 1)
				elif 'kde' in args.ip:
					kde_intensity = float(args.ip.replace('kde', ''))
					ip = torch.randn_like(data)*kde_intensity + data

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

			loss = F.cross_entropy(y_pred_samples.div(tau),
								   label.repeat(args.n_sample)) * args.n_ensemble
			reg = kl / label.size(0) * args.n_ensemble

			pred = y_pred_samples.argmax(dim=1, keepdim=True)
			correct += pred.eq(label.repeat(args.n_sample).view_as(pred)).sum().item()/args.n_sample
		else:
			loss = sum([F.cross_entropy(output, label) for output in outputs])
			reg = 0
			for model_idx, model in enumerate(models):
				anchor_model = None if anchor_models is None else anchor_models[model_idx]
				reg += regularize_on_weight(model, anchor_model, args.method, args.W_var, args.b_var)
			reg = reg / len(train_loader.dataset)

			for output in outputs:
				pred = output.argmax(dim=1, keepdim=True)
				correct += pred.eq(label.view_as(pred)).sum().item()/args.n_ensemble

		optimizer.zero_grad()
		optimizer_tau.zero_grad()
		(loss + reg * args.alpha).backward()
		optimizer.step()
		optimizer_tau.step()
		if batch_idx % args.log_interval == 0:
			print('Train Epoch: {} {}  Acc: {:.4f}  Loss: {:.4f}  Reg: {:.4f}  Tau: {:.4f}'.format(
				epoch, batch_idx, float(correct) / min(len(train_loader.dataset),
				(batch_idx+1)*args.batch_size), loss.item(), reg.item(), tau.item()))

def test(args, models, device, test_loader, tau, ret_probs_and_labels=False):
	for model in models:
		model.eval()
		if args.dropout > 0 and args.n_dropout_inf > 1:
			model.train()
	test_loss = 0
	correct = 0
	probs, labels, mis = [], [], []
	with torch.no_grad():
		for data, target in test_loader:
			data, target = data.to(device), target.to(device)
			output = torch.stack([model(data) for model, _ in
				itertools.product(models, range(args.n_dropout_inf))], 0)

			if args.method == 'our' and len(models) > 1:
				output = gp_sample(output, args.epsilon, 1000).div(tau)
			mis.append(ent(output.softmax(-1).mean(0)) - ent(output.softmax(-1)).mean(0))
			output = output.softmax(-1).mean(0).log()
			test_loss += F.cross_entropy(output, target, reduction='sum').item()  # sum up batch loss
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
			title=None if ret_probs_and_labels else
				  'eces/{}_{}_{}_ens{}_alpha{}_ip{}_{}{}.pdf'.format(
				  	args.arch, args.dataset, args.method, args.n_ensemble,
					args.alpha, args.ip, args.seed,
					'_{}'.format(args.prior_arch) if args.prior_arch != args.arch and args.method == 'our' else ''))

	test_loss /= len(test_loader.dataset)
	print('N_ensemble: {}, Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%), ECE: {:.4f}'.format(
		len(models), test_loss, correct, len(test_loader.dataset),
		100. * correct / len(test_loader.dataset), ece.item()))
	if ret_probs_and_labels:
		return probs.data.cpu().numpy(), labels.data.cpu().numpy(), mis.data.cpu().numpy()
	return test_loss, 100. * correct / len(test_loader.dataset)

def fit(args, models, device, train_loader, test_loader, tau, prior_model=None,
		anchor_models=None):
	params = []
	for model in models:
		for p in model.parameters():
			if p.requires_grad:
				params.append(p)

	optimizer = optim.SGD(params, lr=args.lr)
	optimizer_tau = optim.Adam([tau], lr=args.tau_lr)
	scheduler = CosineAnnealingLR(optimizer, args.epochs)
	for epoch in range(1, args.epochs + 1):
		train(args, epoch, models, device, train_loader, tau, optimizer_tau,
			  optimizer, prior_model, anchor_models)
		test(args, models, device, test_loader, tau)
		scheduler.step()
	if args.save_model:
		torch.save(model.state_dict(), "mnist.pt")

def main():
	# Training settings
	parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
	parser.add_argument('--batch-size', type=int, default=64, metavar='N',
						help='input batch size for training (default: 64)')
	parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
						help='input batch size for testing (default: 1000)')
	parser.add_argument('--epochs', type=int, default=24, metavar='N',
						help='number of epochs to train (default: 24)')
	parser.add_argument('--lr', type=float, default=1e-1, metavar='LR')
	parser.add_argument('--tau_lr', type=float, default=1e-3)
	parser.add_argument('--no-cuda', action='store_true', default=False,
						help='disables CUDA training')
	parser.add_argument('--seed', type=int, default=1, metavar='S',
						help='random seed (default: 1)') #1
	parser.add_argument('--log-interval', type=int, default=100, metavar='N',
						help='how many batches to wait before logging training status')
	parser.add_argument('--save-model', action='store_true', default=False,
						help='For Saving the current Model')
	parser.add_argument('--data-path', type=str, default='.')
	parser.add_argument('--dataset', type=str, default='fmnist')
	parser.add_argument('--n_ensemble', type=int, default=10)
	parser.add_argument('--n_dropout_inf', type=int, default=1)
	parser.add_argument('--arch', type=str, default='fashionCNN')
	parser.add_argument('--dropout', type=float, default=0.)
	parser.add_argument('--W_var', type=float, default=2.)
	parser.add_argument('--b_var', type=float, default=0.01)

	parser.add_argument('--method', type=str, default='free')
	parser.add_argument('--alpha', type=float, default=0.1)
	parser.add_argument('--ip', type=str, default=None)
	parser.add_argument('--prior_arch', type=str, default='fashionCNN')
	parser.add_argument('--n_ensembles_prior', type=int, nargs='+', default=[10])
	parser.add_argument('--epsilon', type=float, default=5e-2)
	parser.add_argument('--n_sample', type=int, default=256)

	args = parser.parse_args()
	use_cuda = not args.no_cuda and torch.cuda.is_available()
	torch.manual_seed(args.seed)
	device = torch.device("cuda" if use_cuda else "cpu")

	dataset = datasets.FashionMNIST if args.dataset == 'fmnist' else datasets.MNIST
	transform=transforms.Compose([
		transforms.ToTensor(),
		transforms.Normalize((0.5,), (0.5,))
		])
	dataset1 = dataset(args.data_path, train=True, download=True,
					   transform=transform)
	dataset2 = dataset(args.data_path, train=False,
					   transform=transform)
	train_loader = torch.utils.data.DataLoader(dataset1, batch_size=args.batch_size,
		num_workers=4, pin_memory=True, shuffle=True)
	test_loader = torch.utils.data.DataLoader(dataset2, batch_size=args.test_batch_size,
		num_workers=4, pin_memory=True, shuffle=False)

	models = []
	for _ in range(args.n_ensemble):
		model = fn_make_NN(args.arch, args.dropout)
		model = convert_archcond(model, None, False)
		model = model.to(device)
		init_NN(model, args.W_var, args.b_var)
		models.append(model)
	tau = nn.Parameter(torch.tensor(1.).cuda())

	if args.method == 'our':
		prior_model = fn_make_prior_NN(args.n_ensembles_prior, args.W_var,
									   args.b_var, args.prior_arch).to(device)
	else:
		prior_model = None

	if args.method == 'anc':
		anchor_models = [fn_make_NN(args.arch).to(device) for _ in range(args.n_ensemble)]
		for anchor_model in anchor_models:
			init_NN(anchor_model, args.W_var, args.b_var)
			for p in anchor_model.parameters():
				p.requires_grad_(False)
	else:
		anchor_models = None

	print(models[-1])
	print(prior_model)

	fit(args, models, device, train_loader, test_loader, tau, prior_model, anchor_models)

	acc_wrt_ens = np.zeros((args.n_ensemble, 2))
	for i in range(1, args.n_ensemble + 1):
		test_loss, test_acc = test(args, models[:i], device, test_loader, tau)
		acc_wrt_ens[i-1, 0] = test_loss
		acc_wrt_ens[i-1, 1] = test_acc
	np.save('accs/acc_wrt_ens_{}_{}_{}_ens{}_alpha{}_ip{}_{}{}.npy'.format(
			args.arch, args.dataset, args.method, args.n_ensemble, args.alpha, args.ip, args.seed,
			'_{}'.format(args.prior_arch) if args.prior_arch != args.arch and args.method == 'our' else ''),
		acc_wrt_ens)

	probs, labels, mis = test(args, models, device, test_loader, tau, ret_probs_and_labels=True)
	if args.dataset == 'fmnist':
		ood_dataset = datasets.MNIST(args.data_path, train=False, transform=transform)
	else:
		ood_dataset = datasets.FashionMNIST(args.data_path, train=False, transform=transform)
	ood_loader = torch.utils.data.DataLoader(ood_dataset, batch_size=args.test_batch_size,
		num_workers=1, pin_memory=True, shuffle=False)
	ood_probs, ood_labels, ood_mis = test(args, models, device, ood_loader, tau, ret_probs_and_labels=True)

	is_correct = np.concatenate([(probs.argmax(1) == labels).astype(np.float32), np.zeros((ood_labels.shape[0]))])
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
	np.save('accs/{}_{}_{}_ens{}_alpha{}_ip{}_{}{}.npy'.format(
			args.arch, args.dataset, args.method, args.n_ensemble, args.alpha, args.ip, args.seed,
			'_{}'.format(args.prior_arch) if args.prior_arch != args.arch and args.method == 'our' else ''),
		np.array(accs))

if __name__ == '__main__':
	main()
