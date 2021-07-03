#!/usr/bin/env python
# coding: utf-8

import argparse
import matplotlib.pyplot as plt
import math
from tqdm import tqdm
import numpy as np
import random
import copy

import pandas as pd
import zipfile
import urllib.request
from sklearn.model_selection import KFold
import scipy.stats as stats

# for NN
import torch

# for NNGP
import neural_tangents as nt
from neural_tangents import stax

# for HMC and VI
import theano
floatX = theano.config.floatX
import pymc3 as pm
import theano.tensor as T
import sklearn

from utils_degp import ParallelPriorLinear, gp_sample_and_estimate_kl

#prepare
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams.update({'font.size': 10})

parser = argparse.ArgumentParser(description='UCI',
	formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--not_use_gpu', action='store_true', default=False)
parser.add_argument('--manualSeed', type=int, default=123, help='manual seed')

# utils
class Erf(torch.nn.Module):
	def __init__(self):
		super(Erf, self).__init__()

	def forward(self, x):
		return x.erf()

def fn_make_data(dataset):
	if dataset=="boston":
		path = 'uci_data//01_boston_house//housing_data.csv'
		data = np.loadtxt(path,skiprows=0)

	elif dataset=="kin8":
		path = 'uci_data//04_kin8nm//Dataset.csv'
		data = np.loadtxt(path, skiprows=0)

	elif dataset=="power":
		path = 'uci_data//06_power_plant//Folds5x2_pp.csv'
		data = np.loadtxt(path, delimiter=',',skiprows=1)

	elif dataset=="protein":
		path = 'uci_data//07_protein//CASP.csv'
		data_1 = np.loadtxt(path, delimiter=',',skiprows=1)
		# we are predicting the first column
		# put first column at end
		data = np.c_[data_1[:,1:], data_1[:,0]]

	elif dataset=="wine":
		path = 'uci_data//08_wine//winequality-red.csv'
		data = np.loadtxt(path, delimiter=';',skiprows=1)
	return data

def fn_make_NN(activation_fn, n_layer=1, n_hidden=100, D_in=1, dropout=0.):
	D_out = 1

	if activation_fn == 'relu':
		mid_act = torch.nn.ReLU()
	elif activation_fn == 'erf':
		mid_act = Erf()

	layers = [torch.nn.Linear(D_in, n_hidden),
		mid_act,
		torch.nn.Dropout(dropout),
		torch.nn.Linear(n_hidden, D_out, bias=False)]
	for _ in range(n_layer - 1):
		layers.insert(3, torch.nn.Linear(n_hidden, n_hidden))
		layers.insert(4, mid_act)
		layers.insert(5, torch.nn.Dropout(dropout))
	model = torch.nn.Sequential(*layers)
	return model

def fn_make_prior_NN(activation_fn, n_ensembles, W1_var, b1_var, W2_var, b2_var, n_hidden=100, D_in=1):
	D_out = 1 # input and output dimension

	if activation_fn == 'relu':
		mid_act = torch.nn.ReLU()
	elif activation_fn == 'erf':
		mid_act = Erf()

	layers = []
	for i, n_ensemble in enumerate(n_ensembles):
		in_features, out_features, bias = n_hidden, n_hidden, True
		if i == 0:
			in_features = D_in

		if i == 0:
			W_var = W1_var * D_in
			b_var = b1_var
		else:
			W_var = W2_var * n_hidden
			b_var = b2_var

		layers.append(ParallelPriorLinear(n_ensemble, in_features, out_features, W_var=W_var, b_var=b_var, bias=bias))
		layers.append(mid_act)

	model = torch.nn.Sequential(*layers)
	return model

def init_NN(model, W1_var, b1_var, W2_var, b2_var):
	# initialise weights
	model[0].weight.data.normal_(0.0, np.sqrt(W1_var))
	model[0].bias.data.normal_(0.0, np.sqrt(b1_var))
	for n, p in model[1:].named_parameters():
		if 'bias' in n:
			p.data.normal_(0.0, np.sqrt(b2_var))
		else:
			p.data.normal_(0.0, np.sqrt(W2_var))

def fn_predict_ensemble(NNs, x_test, tau=1.):
	''' fn to predict given a list of NNs (an ensemble)'''
	y_preds = []
	with torch.no_grad():
		for m in range(len(NNs)):
			y_preds.append(NNs[m](x_test).data.div(tau).cpu().numpy())
	y_preds = np.array(y_preds)

	y_preds_mu = np.mean(y_preds,axis=0)
	y_preds_std = np.std(y_preds,axis=0)

	return y_preds, y_preds_mu, y_preds_std

def nngp(x_train, y_train, x_test, data_noise, n_ensemble, activation_fn,
		 W1_var, b1_var, W2_var, b2_var, n_layer, n_hidden, empirical=False):

	layers = [stax.Dense(n_hidden, W_std=math.sqrt(W1_var*x_train.shape[1]), b_std=math.sqrt(b1_var), parameterization='standard'),
		stax.Relu() if activation_fn == 'relu' else stax.Erf(),
		stax.Dense(1, W_std=math.sqrt(W2_var*n_hidden), b_std=0, parameterization='standard')
	]
	for _ in range(n_layer - 1):
		layers.insert(2, stax.Dense(n_hidden, W_std=math.sqrt(W2_var*n_hidden), b_std=math.sqrt(b2_var),
									parameterization='standard'))
		layers.insert(3, stax.Relu() if activation_fn == 'relu' else stax.Erf())
	init_fn, apply_fn, kernel_fn = stax.serial(*layers)

	if empirical:
		from jax import random
		key1, key2, key3 = random.split(random.PRNGKey(1), 3)
		_, params = init_fn(key3, x_train.shape)
		kernel_fn = nt.empirical_kernel_fn(apply_fn, trace_axes=(-1,), vmap_axes=0, implementation=1)
		cov_dd = kernel_fn(x_train, None, 'nngp', params) + np.identity(x_train.shape[0])*data_noise
		cov_xd = kernel_fn(x_test, x_train, 'nngp', params)
		cov_xx = kernel_fn(x_test, None, 'nngp', params)
	else:
		cov_dd = kernel_fn(x_train, None, 'nngp') + np.identity(x_train.shape[0])*data_noise
		cov_xd = kernel_fn(x_test, x_train, 'nngp')
		cov_xx = kernel_fn(x_test, None, 'nngp')

	L = np.linalg.cholesky(cov_dd)
	alpha = np.linalg.solve(L.T,np.linalg.solve(L,y_train))
	y_pred_mu = np.matmul(cov_xd,alpha)
	v = np.linalg.solve(L,cov_xd.T)
	cov_pred = cov_xx - np.matmul(v.T,v)

	y_pred_var = np.atleast_2d(np.diag(cov_pred) + data_noise).T
	y_pred_std = np.sqrt(y_pred_var)
	return None, y_pred_mu, y_pred_std

def anchored_ensemble(x_train, y_train, x_test, reg, data_noise, n_ensemble, activation_fn,
					  W1_var, b1_var, W2_var, b2_var, n_layer, n_hidden, gamma=1., alpha_w=1):
	x = torch.tensor(x_train).float().to(device)
	y = torch.tensor(y_train).float().to(device)
	x_test = torch.tensor(x_test).float().to(device)
	# set up loss
	loss_fn = torch.nn.MSELoss().to(device)

	# create the NNs
	anchor_NNs=[]
	for m in range(n_ensemble):
		anchor_NNs.append(fn_make_NN(activation_fn, n_layer, n_hidden, D_in = x_train.shape[1]).to(device))
		init_NN(anchor_NNs[-1], W1_var, b1_var, W2_var, b2_var)
		for p in anchor_NNs[-1].parameters():
			p.requires_grad_(False)

	NNs, params = [], []
	for m in range(n_ensemble):
		NNs.append(fn_make_NN(activation_fn, n_layer, n_hidden, D_in = x_train.shape[1]).to(device))
		init_NN(NNs[-1], W1_var, b1_var, W2_var, b2_var)
		for p in NNs[-1].parameters():
			params.append(p)
	optimizer = torch.optim.Adam(params, lr=l_rate)
	scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=gamma)

	# do training
	for _ in range(epochs):
		perturb = torch.randperm(x.shape[0])
		# loss_sum, kl_sum = 0., 0.
		for idx in range(0, x.shape[0], 256):
			batch_x = x[perturb[idx:min(idx+256, x.shape[0])]]
			batch_y = y[perturb[idx:min(idx+256, x.shape[0])]]
			losses, l2s = [], []
			for m in range(n_ensemble):
				losses.append(loss_fn(NNs[m](batch_x), batch_y))

				# set up reg loss
				l2 = 0
				if reg == 'anc':
					l2 += ((NNs[m][0].weight - anchor_NNs[m][0].weight)**2).sum() / W1_var
					l2 += ((NNs[m][0].bias - anchor_NNs[m][0].bias)**2).sum() / b1_var
					for p1, p2 in zip(list(NNs[m][1:].parameters()), list(anchor_NNs[m][1:].parameters())):
						l2 += ((p1 - p2)**2).sum() / (b2_var if p1.dim() == 1 else W2_var)
				elif reg == 'reg':
					l2 += (NNs[m][0].weight**2).sum() / W1_var
					l2 += (NNs[m][0].bias**2).sum() / b1_var
					for p in NNs[m][1:].parameters():
						l2 += (p**2).sum() / (b2_var if p.dim() == 1 else W2_var)
				elif reg == 'free':
					# do nothing
					l2 += 0.0
				else:
					raise NotImplementedError
				l2 = l2 / x.size(0) * data_noise
				l2s.append(l2)

			# run gradient update
			optimizer.zero_grad()
			(sum(losses) + sum(l2s)*alpha_w).backward()
			optimizer.step()
		scheduler.step()

	# run predictions
	y_preds, y_preds_mu, y_preds_std = fn_predict_ensemble(NNs,x_test)
	return y_preds, y_preds_mu, y_preds_std

def gp_ensemble(x_train, y_train, x_test, data_noise, n_ensemble, activation_fn,
				W1_var, b1_var, W2_var, b2_var, n_layer, n_hidden, ip_max, ip_min,
				gamma=1., n_ensembles_ref=None, n_hidden_ref=None, ip=None,
				n_sample=256, epsilon=5e-2, alpha=1):

	x = torch.tensor(x_train).float().to(device)
	y = torch.tensor(y_train).float().to(device)
	x_test = torch.tensor(x_test).float().to(device)
	# set up loss
	loss_fn = torch.nn.MSELoss().to(device)

	if n_ensembles_ref is None:
		n_ensembles_ref = [n_ensemble,] * (n_layer)
	if n_hidden_ref is None:
		n_hidden_ref = n_hidden
	NN_ref = fn_make_prior_NN(activation_fn, n_ensembles_ref, W1_var, b1_var,
		W2_var, b2_var, n_hidden_ref, x_train.shape[1]).to(device)

	# create the NNs
	NNs, params = [], []
	for m in range(n_ensemble):
		NNs.append(fn_make_NN(activation_fn, n_layer, n_hidden, D_in = x_train.shape[1]).to(device))
		init_NN(NNs[-1], W1_var, b1_var, W2_var, b2_var)
		for p in NNs[-1].parameters():
			params.append(p)
	tau = torch.nn.Parameter(torch.tensor(1.).to(device))
	optimizer = torch.optim.Adam(params + [tau], lr=l_rate)
	scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=gamma)

	for _ in range(epochs):
		perturb = torch.randperm(x.shape[0])
		loss_sum, kl_sum = 0., 0.
		for idx in range(0, x.shape[0], 256):
			batch_x = x[perturb[idx:min(idx+256, x.shape[0])]]
			batch_y = y[perturb[idx:min(idx+256, x.shape[0])]]

			if ip is None:
				input = batch_x
			elif ip == 'uniform':
				input = torch.cat([batch_x, torch.empty_like(batch_x).uniform_(0, 1) * (ip_max-ip_min) + ip_min], 0)
			else:
				raise NotImplementedError

			outputs = [NNs[m](input) for m in range(n_ensemble)]
			with torch.no_grad():
				with torch.cuda.amp.autocast():
					y_preds_ref = NN_ref(input).float()

			y_pred_samples, kl = gp_sample_and_estimate_kl(torch.stack(outputs, 0),
				y_preds_ref, n_sample, epsilon, W2_var * n_hidden, 0)
			y_pred_samples = y_pred_samples[:, :, :batch_y.shape[0]].permute(0, 2, 1).flatten(0,1)

			loss = loss_fn(y_pred_samples.div(tau), batch_y.repeat(n_sample, 1)) * n_ensemble
			reg = kl / batch_x.size(0) * data_noise * 2 * n_ensemble

			# run gradient update
			optimizer.zero_grad()
			(loss + kl * alpha).backward()
			optimizer.step()

			loss_sum += loss.item() * batch_x.shape[0]
			kl_sum += kl.item() * batch_x.shape[0]
		scheduler.step()

	# run predictions
	y_preds, y_preds_mu, y_preds_std = fn_predict_ensemble(NNs,x_test, tau)
	return y_preds, y_preds_mu, y_preds_std

def mc_dropout(x_train, y_train, x_test, n_ensemble, activation_fn,
			   W1_var, b1_var, W2_var, b2_var, n_layer, n_hidden, dropout=0., gamma=1.):
	x = torch.tensor(x_train).float().to(device)
	y = torch.tensor(y_train).float().to(device)
	x_test = torch.tensor(x_test).float().to(device)

	# create the NN
	NN = fn_make_NN(activation_fn, n_layer, n_hidden, dropout=dropout, D_in = x_train.shape[1]).to(device)
	init_NN(NN, W1_var, b1_var, W2_var, b2_var)

	loss_fn = torch.nn.MSELoss().to(device)

	params = []
	for p in NN.parameters():
		if p.requires_grad:
			params.append(p)
	optimizer = torch.optim.Adam(params, lr=l_rate)
	scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=gamma)

	for _ in range(epochs):
		perturb = torch.randperm(x.shape[0])
		for idx in range(0, x.shape[0], 256):
			batch_x = x[perturb[idx:min(idx+256, x.shape[0])]]
			batch_y = y[perturb[idx:min(idx+256, x.shape[0])]]
			loss = loss_fn(NN(batch_x), batch_y)
			# run gradient update
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
		scheduler.step()

	# run predictions
	y_preds = []
	with torch.no_grad():
		for m in range(n_ensemble):
			y_preds.append(NN(x_test).data.cpu().numpy())
	y_preds = np.array(y_preds)
	y_preds_mu = np.mean(y_preds,axis=0)
	y_preds_std = np.std(y_preds,axis=0)
	return y_preds, y_preds_mu, y_preds_std

def build_model_pm(total_size, D_in, ann_input, ann_output, data_noise, activation_fn,
				   W1_var, b1_var, W2_var, b2_var, n_layer, n_hidden):
	with pm.Model() as model:
		for i in range(n_layer+1):
			n_in = D_in if i == 0 else n_hidden
			n_out = 1 if i == n_layer else n_hidden
			W_var = W1_var if i == 0 else W2_var
			b_var = b1_var if i == 0 else b2_var

			init_w = np.random.normal(loc=0, scale=np.sqrt(W_var), size=[n_in, n_out]).astype(floatX)
			init_b = np.random.normal(loc=0, scale=np.sqrt(b_var), size=[n_out]).astype(floatX)
			weights_w = pm.Normal('w_{}'.format(i), 0, sd=np.sqrt(W_var), shape=(n_in, n_out), testval=init_w)
			weights_b = pm.Normal('b_{}'.format(i), 0, sd=np.sqrt(b_var), shape=(n_out), testval=init_b) if i<n_layer else 0

			act_pre = pm.math.dot(ann_input if i == 0 else act_out, weights_w) + weights_b
			if i < n_layer:
				if activation_fn == 'relu':
					act_out = pm.math.maximum(act_pre, 0)
				elif activation_fn =='erf':
					act_out = pm.math.erf(act_pre)

		out = pm.Normal('out', act_pre,sd=np.sqrt(data_noise),
						observed=ann_output,
						total_size=total_size)
	return model, out

def hmc_vi(x_train, y_train, x_test, data_noise, n_ensemble, activation_fn,
		   W1_var, b1_var, W2_var, b2_var, n_layer, n_hidden, is_hmc=True):
	ann_input = theano.shared(x_train)
	ann_output = theano.shared(y_train)
	BNN, out = build_model_pm(len(y_train), x_train.shape[1], ann_input, ann_output, data_noise, activation_fn,
								W1_var, b1_var, W2_var, b2_var, n_layer, n_hidden)
	# run inference
	if is_hmc:
		step = pm.HamiltonianMC(path_length=0.5, adapt_step_size=True, step_scale=0.04,
			gamma=0.05, k=0.9, t0=1, target_accept=0.95, model=BNN)
		trace = pm.sample(n_inf_samples, step=step, model=BNN, chains=1, n_jobs=1, tune=300)
		# reduce path_length if failing - 5.0 is ok with cos_lin data
	else:
		# https://docs.pymc.io/notebooks/bayesian_neural_network_advi.html
		inference = pm.ADVI(model=BNN)
		approx = pm.fit(n=vi_steps, method=inference, model=BNN)
		trace = approx.sample(draws=n_inf_samples)

		if True:
			fig = plt.figure(figsize=(8, 4))
			ax = fig.add_subplot(111)
			ax.plot(-inference.hist, label='new ADVI', alpha=.3)
			ax.plot(approx.hist, label='old ADVI', alpha=.3)
			ax.set_ylabel('ELBO');
			ax.set_xlabel('iteration');
			fig.show()

	# make predictions
	ann_input.set_value(x_test.astype('float32'))
	ann_output.set_value(x_test[:, 0:1].astype('float32'))

	ppc = pm.sample_ppc(trace, model=BNN, samples=n_pred_samples) # this does new set of preds per point
	y_preds = ppc['out']
	y_preds_mu = y_preds.mean(axis=0)
	y_preds_std = y_preds.std(axis=0)
	return y_preds, y_preds_mu, y_preds_std

def gauss_neg_log_like(y_true, y_pred_gauss_mid, y_pred_gauss_dev, scale_c):
	"""
	return negative gaussian log likelihood
	"""

	n = y_true.shape[0]
	y_true=y_true.reshape(-1)*scale_c
	y_pred_gauss_mid=y_pred_gauss_mid*scale_c
	y_pred_gauss_dev=y_pred_gauss_dev*scale_c
	neg_log_like = -np.sum(stats.norm.logpdf(y_true.squeeze(), loc=y_pred_gauss_mid.squeeze(), scale=y_pred_gauss_dev.squeeze()))
	neg_log_like = neg_log_like/n

	return neg_log_like

def cross_validation(kf, data, method, scale_c, data_noise, n_ensemble, activation_fn,
					 W1_var, b1_var, W2_var, b2_var, n_layer, n_hidden, gamma, alpha):

	in_dim = data.shape[1] - 1
	test_nlls, test_rmses = [], []
	test_nlls_ref, test_rmses_ref = [], []

	for idx in kf.split(data):
		train_index, test_index = idx

		x_train, y_train = data[train_index, :in_dim], data[train_index, in_dim:]
		x_test, y_test = data[test_index, :in_dim], data[test_index, in_dim:]

		if method == 'nngp':
			y_preds, y_preds_mu, y_preds_std = nngp(x_train, y_train, x_test, data_noise, n_ensemble,
													activation_fn, W1_var, b1_var, W2_var, b2_var,
													n_layer, n_hidden)
		elif method == 'anchored_ens':
			y_preds, y_preds_mu, y_preds_std = anchored_ensemble(x_train, y_train, x_test, 'anc',
																 data_noise, n_ensemble, activation_fn,
																 W1_var, b1_var, W2_var, b2_var,
																 n_layer, n_hidden, gamma=gamma)
		elif method == 'reg_ens':
			y_preds, y_preds_mu, y_preds_std = anchored_ensemble(x_train, y_train, x_test, 'reg',
																 data_noise, n_ensemble, activation_fn,
																 W1_var, b1_var, W2_var, b2_var,
																 n_layer, n_hidden, gamma=gamma)
		elif method == 'free_ens':
			y_preds, y_preds_mu, y_preds_std = anchored_ensemble(x_train, y_train, x_test, 'free',
																 data_noise, n_ensemble, activation_fn,
																 W1_var, b1_var, W2_var, b2_var,
																 n_layer, n_hidden, gamma=gamma)
		elif method == 'our':
			y_preds, y_preds_mu, y_preds_std = gp_ensemble(x_train, y_train, x_test, data_noise,
														   n_ensemble, activation_fn, W1_var, b1_var,
														   W2_var, b2_var, n_layer, n_hidden,
														   torch.from_numpy(data[:, :-1].min(0)).to(device).float(),
														   torch.from_numpy(data[:, :-1].max(0)).to(device).float(),
														   gamma=gamma, ip='uniform', alpha=alpha)
		elif method == 'mc_dropout':
			y_preds, y_preds_mu, y_preds_std = mc_dropout(x_train, y_train, x_test, n_ensemble,
														  activation_fn, W1_var, b1_var, W2_var, b2_var,
														  n_layer, n_hidden, dropout=0.4, gamma=gamma)
		elif method == 'hmc':
			y_preds, y_preds_mu, y_preds_std = hmc_vi(x_train, y_train, x_test, data_noise, n_ensemble,
													  activation_fn, W1_var, b1_var, W2_var, b2_var,
													  n_layer, n_hidden, is_hmc=True)
		elif method == 'vi':
			y_preds, y_preds_mu, y_preds_std = hmc_vi(x_train, y_train, x_test, data_noise, n_ensemble,
													  activation_fn, W1_var, b1_var, W2_var, b2_var,
													  n_layer, n_hidden, is_hmc=False)
		if method != 'nngp' and method != 'hmc' and method != 'vi':
			y_preds_std = np.sqrt(np.square(y_preds_std) + data_noise)

		rmse = np.sqrt(np.mean(np.square(scale_c*(y_test - y_preds_mu))))
		neg_log_like = gauss_neg_log_like(y_test, y_preds_mu, y_preds_std, scale_c)
		test_nlls.append(neg_log_like)
		test_rmses.append(rmse)

	print(method + '  NLL = %7.3f +/- %.3f    RMSE = %7.3f +/- %.3f' % (np.array(test_nlls).mean(), np.array(test_nlls).var()**0.5, np.array(test_rmses).mean(), np.array(test_rmses).var()**0.5))

if __name__ == '__main__':
	args = parser.parse_args()
	if not args.not_use_gpu:
		device = torch.device('cuda')
	else:
		device = torch.device('cpu')

	np.random.seed(args.manualSeed)
	random.seed(args.manualSeed)
	torch.manual_seed(args.manualSeed)
	if not args.not_use_gpu: torch.cuda.manual_seed_all(args.manualSeed)

	# general settings
	n_hidden = 256 	# no. hidden units in NN
	n_layer = 2
	activation_fn = 'relu' # relu erf

	# optimisation options for NN
	epochs = 1000 		# run reg for 15+ epochs seems to mess them up
	l_rate = 0.01 		# learning rate
	gamma = 0.99

	# for ensemble
	n_ensemble = 10	# no. NNs in ensemble

	n_inf_samples = 2000 # number samples to take during inference
	n_pred_samples = 200 # number samples to take during prediction
	vi_steps = 20000 # number optimisation steps to run for VI

	# cross validation
	n_splits = 5

	for dataset in ['boston', 'power', 'wine', 'kin8', 'protein']:
		print("------------------------------" + dataset + "------------------------------")
		data = fn_make_data(dataset)
		perm = np.random.permutation(data.shape[0])
		data = data[perm]
		scale_c = np.std(data[:,-1])
		# normalise data
		for i in range(0, data.shape[1]):
			# avoid zero variance features (exist one or two)
			sdev_norm = np.std(data[:,i])
			sdev_norm = 0.001 if sdev_norm == 0 else sdev_norm
			data[:,i] = (data[:,i] - np.mean(data[:,i]) )/sdev_norm
		kf = KFold(n_splits=n_splits)
		if dataset == 'boston':
			data_noise = 0.06
			W1_var = 10./(data.shape[1]-1)
			b1_var = 10.
			W2_var = 1./n_hidden
			b2_var = 1
			alpha = 1e-4
		elif dataset == 'power':
			data_noise = 0.05
			W1_var = 4./(data.shape[1]-1)
			b1_var = 4.
			W2_var = 2./n_hidden
			b2_var = 2
			alpha = 1e-3
		elif dataset == 'wine':
			data_noise = 0.5
			W1_var = 10./(data.shape[1]-1)
			b1_var = 10.
			W2_var = 2./n_hidden
			b2_var = 2
			alpha = 1e-1
		elif dataset == 'kin8':
			data_noise = 0.02
			W1_var = 40./(data.shape[1]-1)
			b1_var = 40.
			W2_var = 1./n_hidden
			b2_var = 1
			alpha = 1e-3
		elif dataset == 'protein':
			data_noise = 0.5
			W1_var = 50./(data.shape[1]-1)
			b1_var = 50.
			W2_var = 1./n_hidden
			b2_var = 1
			alpha = 1e-6

		for method in ['our', 'nngp', 'anchored_ens', 'reg_ens', 'free_ens', 'mc_dropout', 'vi']:
			cross_validation(kf, data, method, scale_c, data_noise, n_ensemble, activation_fn,
							 W1_var, b1_var, W2_var, b2_var, n_layer, n_hidden, gamma, alpha)
