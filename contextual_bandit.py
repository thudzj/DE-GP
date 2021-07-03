'''
CUDA_VISIBLE_DEVICES=5 python dcb.py --run-experiment --download --f_alpha 0.1
CUDA_VISIBLE_DEVICES=1 python dcb.py --run-experiment --download --bandit adult --f_alpha 0.1
python dcb.py --run-experiment --download --bandit mushroom
'''


from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'figure.max_open_warning': 0})
# plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams.update({'font.size': 14})


from typing import Optional
import numpy as np
import torch
import torch.nn as nn
import seaborn as sns

# from matplotlib.colors import ListedColormap
# construct cmap
# flatui = ["#9b59b6", "#3498db", "#95a5a6", "#e74c3c", "#34495e", "#2ecc71"]
# my_cmap = ListedColormap(sns.color_palette().as_hex())

import genrl
import genrl.agents
import genrl.agents.bandits
import genrl.agents.bandits.contextual
from genrl.agents.bandits.contextual.base import DCBAgent
from genrl.agents.bandits.contextual.common import TransitionDB
# from genrl.agents.bandits.contextual.bootstrap_neural import BootstrapNeuralAgent
from genrl.agents.bandits.contextual.fixed import FixedAgent
# from genrl.agents.bandits.contextual.linpos import LinearPosteriorAgent
# from genrl.agents.bandits.contextual.neural_greedy import NeuralGreedyAgent
# from genrl.agents.bandits.contextual.neural_linpos import NeuralLinearPosteriorAgent
# from genrl.agents.bandits.contextual.neural_noise_sampling import NeuralNoiseSamplingAgent
# from genrl.agents.bandits.contextual.variational import VariationalAgent
from genrl.bandit.trainer import DCBTrainer

import copy

from utils_degp import init_NN, regularize_on_weight, ParallelPriorLinear, gp_sample_and_estimate_kl, gp_sample


class Ensemble(nn.Module):
	def __init__(self, args, **kwself):
		super(Ensemble, self).__init__()
		self.context_dim = kwself.get("context_dim")
		self.hidden_dims = kwself.get("hidden_dims")
		self.n_actions = kwself.get("n_actions")
		self.device = kwself.get("device")

		self.n_ensemble = args.n_ensemble
		self.W_var = args.W_var
		self.b_var = args.b_var
		self.method = args.method
		self.alpha = args.alpha
		self.epsilon = args.epsilon
		self.n_sample = args.n_sample
		self.ip = args.ip
		self.n_ensembles_prior = args.n_ensembles_prior

		t_hidden_dims = [self.context_dim, *self.hidden_dims, self.n_actions]
		self.models = nn.ModuleList([])
		for _ in range(self.n_ensemble):
			layers = []
			for i in range(len(t_hidden_dims) - 1):
				layers.append(nn.Linear(t_hidden_dims[i], t_hidden_dims[i + 1]))
				if i < len(t_hidden_dims) - 2:
					layers.append(nn.ReLU())
			self.models.append(nn.Sequential(*layers))

		self.models.to(torch.float).to(self.device)
		self.anchor_models = copy.deepcopy(self.models)
		init_NN(self.models, self.W_var, self.b_var)
		init_NN(self.anchor_models, self.W_var, self.b_var)
		for p in self.anchor_models.parameters():
			p.requires_grad_(False)

		self.init_lr = kwself.get("init_lr", 3e-4)
		self.optimizer = torch.optim.Adam(self.models.parameters(), lr=self.init_lr)
		self.lr_decay = kwself.get("lr_decay", None)
		self.lr_scheduler = (
			torch.optim.lr_scheduler.LambdaLR(
				self.optimizer, lambda i: 1 / (1 + self.lr_decay * (i))
			)
			if self.lr_decay is not None
			else None
		)
		self.lr_reset = kwself.get("lr_reset", False)

		layers = []
		for i in range(len(t_hidden_dims) - 2):
			layers.append(ParallelPriorLinear(self.n_ensembles_prior, t_hidden_dims[i], t_hidden_dims[i + 1]))
			layers.append(nn.ReLU())
		self.prior_model = nn.Sequential(*layers)

		# print(self.models[-1], self.prior_model)
		self.not_trained=True
		self.data_mean = 0
		self.data_std = 1

	def forward(self, context, **kwself):
		output = torch.stack([model(context.to(self.device).sub(self.data_mean).div(self.data_std)) for model in self.models], 0)
		# print(context.shape, self.models[0](context).shape, output.shape)
		if self.method == 'our' and len(self.models) > 1:
			output = gp_sample(output, self.epsilon, output.shape[0])
		output = output[np.random.randint(output.shape[0])].view(-1)
		return dict(x=context, pred_rewards=output)

	def train_model(self, db, epochs: int, batch_size: int):
		if self.lr_decay is not None and self.lr_reset is True:
				self._reset_lr(self.init_lr)
		# self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, epochs)

		contexts = torch.stack(db.db["contexts"]).to(self.device)
		# self.data_mean = contexts.mean(0)
		# self.data_std = contexts.std(0)
		context_max = contexts.max(0)[0]
		context_min = contexts.min(0)[0]
		db_size = len(db.db["contexts"])
		# print(context_max, context_min)

		# rewards = np.array(db.db["rewards"])
		# r_mean = rewards.mean()
		# r_std = rewards.std()

		# losses = []
		# regs = []

		for _ in range(1000 if self.not_trained else epochs):
			x, a, y = db.get_data(batch_size)
			x = x.to(self.device)
			a = a.to(self.device)
			y = y.to(self.device)#.sub(r_mean).div(r_std)
			action_mask = torch.nn.functional.one_hot(a.view(-1), num_classes=self.n_actions)
			reward_vec = y.view(-1).repeat(self.n_actions, 1).T * action_mask

			outputs = [model(x.sub(self.data_mean).div(self.data_std)) for model in self.models]

			# print(x.sub(self.data_mean).div(self.data_std + 1e-8).mean(), x.sub(self.data_mean).div(self.data_std + 1e-8).std(), torch.stack(outputs, 0).mean(), torch.stack(outputs, 0).std())
			if self.method == 'our':
				ip = None
				if self.ip:
					if self.ip == 'uniform':
						ip = torch.empty_like(x).uniform_(0, 1) * (context_max - context_min) + context_min
					elif 'kde' in self.ip:
						kde_intensity = float(self.ip.replace('kde', ''))
						ip = torch.randn_like(x)*kde_intensity + x

				if ip is not None:
					ip_outputs = [model(ip.sub(self.data_mean).div(self.data_std)) for model in self.models]
					outputs = [torch.cat([output, ip_output], 0) for output, ip_output in zip(outputs, ip_outputs)]

					x = torch.cat([x, ip], 0)

				with torch.no_grad():
					y_preds_ref = self.prior_model(x.sub(self.data_mean).div(self.data_std)).float()

				y_pred_samples, kl = gp_sample_and_estimate_kl(torch.stack(outputs, 0),
					y_preds_ref, self.n_sample, self.epsilon, self.W_var, self.b_var)
				y_pred_samples = y_pred_samples[:, :, :batch_size].permute(0, 2, 1).flatten(0,1)

				# print(a.shape, action_mask.shape, reward_vec.shape)
				loss = torch.sum(action_mask.repeat(self.n_sample, 1) * (reward_vec.repeat(self.n_sample, 1) - y_pred_samples) ** 2) / batch_size / self.n_sample * self.n_ensemble
				reg = kl / batch_size * args.n_ensemble
			else:
				loss = sum([torch.sum(action_mask * (reward_vec - output) ** 2) / batch_size for output in outputs])
				reg = 0.
				for model_idx, model in enumerate(self.models):
					anchor_model = self.anchor_models[model_idx]
					reg += regularize_on_weight(model, anchor_model, self.method, self.W_var, self.b_var)
				reg = reg / db_size

			# print(loss, reg)

			self.optimizer.zero_grad()
			(loss + reg * self.alpha).backward()
			self.optimizer.step()

			# losses.append(loss.item() / self.n_ensemble)
			# regs.append(reg.item() / self.n_ensemble)

			if _ % 100 == 99:
				print("Num_data {}, training loss {:.4f}, reg {:.4f}, lr {:.4f}".format(db_size, loss.item(), reg.item(), self.optimizer.param_groups[0]['lr']))

		if self.lr_decay is not None:
			self.lr_scheduler.step()

		if self.not_trained:
			self.not_trained = False

	def _reset_lr(self, lr: float) -> None:
		"""Resets learning rate of optimizer.
		self:
			lr (float): New value of learning rate
		"""
		for o in self.optimizer.param_groups:
			o["lr"] = lr
		self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
			self.optimizer, lambda i: 1 / (1 + self.lr_decay * (i))
		)

class EnsembleAgent(DCBAgent):
	def __init__(self, bandit, args, **kwargs):
		super(EnsembleAgent, self).__init__(bandit, kwargs.get("device", "cpu"))
		self.init_pulls = kwargs.get("init_pulls", 3)
		self.model = (
			Ensemble(
				args,
				context_dim=self.context_dim,
				hidden_dims=kwargs.get("hidden_dims", None),
				n_actions=self.n_actions,
				init_lr=kwargs.get("init_lr", 0.1),
				# max_grad_norm=kwargs.get("max_grad_norm", 0.5),
				lr_decay=kwargs.get("lr_decay", 0.5),
				lr_reset=kwargs.get("lr_reset", False),
				device=self.device
			)
		)
		self.db = TransitionDB(self.device)
		self.t = 0
		self.update_count = 0

	def select_action(self, context: torch.Tensor) -> int:
		self.t += 1
		if self.t < self.n_actions * self.init_pulls:
			return torch.tensor(
				self.t % self.n_actions, device=self.device, dtype=torch.int
			).view(1)
		with torch.no_grad():
			results = self.model(context.view(1, -1))
		action = torch.argmax(results["pred_rewards"]).to(torch.int).view(1)
		return action

	def update_db(self, context: torch.Tensor, action: int, reward: int):
		self.db.add(context, action, reward)

	def update_params(
		self,
		action: Optional[int] = None,
		batch_size: int = 512,
		train_epochs: int = 20,
	):
		self.update_count += 1
		self.model.train_model(self.db, train_epochs, batch_size)



ALGOS = {
	# "bootstrap": BootstrapNeuralAgent,
	"fixed": FixedAgent,
	"ens": EnsembleAgent,
	# "linpos": LinearPosteriorAgent,
	# "neural-greedy": NeuralGreedyAgent,
	# "neural-linpos": NeuralLinearPosteriorAgent,
	# "neural-noise": NeuralNoiseSamplingAgent,
	# "variational": VariationalAgent,
}
BANDITS = {
	"adult": genrl.utils.data_bandits.AdultDataBandit,
	"census": genrl.utils.data_bandits.CensusDataBandit,
	"covertype": genrl.utils.data_bandits.CovertypeDataBandit,
	"magic": genrl.utils.data_bandits.MagicDataBandit,
	"mushroom": genrl.utils.data_bandits.MushroomDataBandit,
	"statlog": genrl.utils.data_bandits.StatlogDataBandit,
	"bernoulli": genrl.bandit.BernoulliMAB,
	"gaussian": genrl.bandit.GaussianMAB,
}


def run(args, agent, bandit, plot=True):
	logdir = Path(args.logdir).joinpath(
		f"{agent.__class__.__name__}-on-{bandit.__class__.__name__}-{datetime.now():%d%m%y%H%M%S}"
	)
	trainer = DCBTrainer(
		agent, bandit, logdir=logdir, log_mode=["stdout", "tensorboard"]
	)

	results = trainer.train(
		timesteps=args.timesteps,
		update_interval=args.update_interval,
		update_after=args.update_after,
		batch_size=args.batch_size,
		train_epochs=args.train_epochs,
		log_every=args.log_every,
		ignore_init=args.ignore_init,
		init_train_epochs=args.init_train_epochs,
		train_epochs_decay_steps=args.train_epochs_decay_steps,
	)

	if plot:
		fig, axs = plt.subplots(3, 2, figsize=(10, 10))
		fig.suptitle(
			f"{agent.__class__.__name__} on {bandit.__class__.__name__}",
			fontsize=14,
		)
		axs[0, 0].scatter(list(range(len(bandit.regret_hist))), results["regrets"])
		axs[0, 0].set_title("Regret History")
		axs[0, 1].scatter(list(range(len(bandit.reward_hist))), results["rewards"])
		axs[0, 1].set_title("Reward History")
		axs[1, 0].plot(results["cumulative_regrets"])
		axs[1, 0].set_title("Cumulative Regret")
		axs[1, 1].plot(results["cumulative_rewards"])
		axs[1, 1].set_title("Cumulative Reward")
		axs[2, 0].plot(results["regret_moving_avgs"])
		axs[2, 0].set_title("Regret Moving Avg")
		axs[2, 1].plot(results["reward_moving_avgs"])
		axs[2, 1].set_title("Reward Moving Avg")

		fig.savefig(
			Path(logdir).joinpath(
				f"{agent.__class__.__name__}-on-{bandit.__class__.__name__}.png"
			)
		)
		return results


def plot_multi_runs(args, multi_results, title):
	with sns.color_palette():
		fig = plt.figure(figsize=(12, 4.5), dpi=600)
		# fig.suptitle(title, fontsize=14)
		# axs[0, 0].set_title("Cumulative Regret")
		# axs[0, 1].set_title("Cumulative Reward")
		# axs[1, 0].set_title("Regret Moving Avg")
		# axs[1, 1].set_title("Reward Moving Avg")
		ax = fig.add_subplot(121)
		ax.tick_params(axis='both', which='major', labelsize=12)
		ax.tick_params(axis='both', which='minor', labelsize=12)
		ax.set_xlabel('Timestep', fontsize=16)
		ax.set_ylabel('Cumulative Regret', fontsize=16)
		for name, results in multi_results.items():
			ax.plot(results["cumulative_regrets"], label=name)
			ax.grid(color='lightgray', linestyle='--')

		ax = fig.add_subplot(122)
		ax.tick_params(axis='both', which='major', labelsize=12)
		ax.tick_params(axis='both', which='minor', labelsize=12)
		ax.set_xlabel('Timestep', fontsize=16)
		ax.set_ylabel('Cumulative Reward', fontsize=16)
		for name, results in multi_results.items():
			ax.plot(results["cumulative_rewards"], label=name)
			ax.grid(color='lightgray', linestyle='--')

		plt.legend()
		# ax.legend()
		# handles, labels = ax.get_legend_handles_labels()
		# ax.get_legend().remove()
		# fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=2, prop={'size':16})

		fig.tight_layout()
		fig.savefig(Path(args.logdir).joinpath(f"{title}"))


def run_multi_algos(args):
	bandit_class = BANDITS[args.bandit.lower()]
	bandit = bandit_class(download=args.download)
	multi_results = {}
	for name, algo in ALGOS.items():
		agent = algo(bandit)
		multi_results[name] = run(args, agent, bandit)
	plot_multi_runs(args, multi_results, title=f"DCBs-on-{bandit_class.__name__}")


def run_multi_bandits(args):
	algo = ALGOS[args.algo.lower()]
	multi_results = {}
	for name, bandit_class in BANDITS.items():
		bandit = bandit_class(download=args.download)
		agent = algo(bandit)
		multi_results[name] = run(args, agent, bandit)
	plot_multi_runs(args, multi_results, title=f"{algo.__name__}-Performance")


def run_single_algos_on_bandit(args):
	algo = ALGOS[args.algo.lower()]
	bandit_class = BANDITS[args.bandit.lower()]
	bandit = bandit_class(download=args.download)
	agent = algo(bandit)
	run(args, agent, bandit)


def run_experiment(args):
	start_time = datetime.now
	print(f"\nStarting experiment at {start_time():%d-%m-%y %H:%M:%S}\n")
	results = {}

	bandit_class = BANDITS[args.bandit.lower()]
	bandit = bandit_class(download=args.download)

	# bootstrap = BootstrapNeuralAgent(bandit=bandit)
	# logdir = Path(args.logdir).joinpath(
	#     f"{bootstrap.__class__.__name__}-on-{bandit.__class__.__name__}"
	# )
	# bootstrap_trainer = DCBTrainer(
	#     bootstrap, bandit, logdir=logdir, log_mode=["stdout", "tensorboard"]
	# )
	# results["bootstrap"] = bootstrap_trainer.train(
	#     timesteps=args.timesteps,
	#     update_interval=50,
	#     update_after=args.update_after,
	#     batch_size=args.batch_size,
	#     train_epochs=100,
	#     log_every=args.log_every,
	#     ignore_init=args.ignore_init,
	#     init_train_epochs=None,
	#     train_epochs_decay_steps=None,
	# )

	for method in ['our', 'free', 'reg', 'anc']:
		args.method = method
		args.alpha = 1
		if method == 'our':
			args.alpha = args.f_alpha
		if method == 'reg':
			args.alpha = 1.
		if method == 'anc':
			args.alpha = 1.
		ensemble = EnsembleAgent(bandit=bandit, args=args, device='cuda', hidden_dims=[256, 256])
		logdir = Path(args.logdir).joinpath(
			f"{ensemble.__class__.__name__}-{args.method}-on-{bandit.__class__.__name__}"
		)
		ensemble_trainer = DCBTrainer(
			ensemble, bandit, logdir=logdir, log_mode=["stdout", "tensorboard"]
		)
		results["ensemble_{}".format(args.method)] = ensemble_trainer.train(
			timesteps=args.timesteps,
			update_interval=50,
			update_after=args.update_after,
			batch_size=args.batch_size,
			train_epochs=100,
			log_every=args.log_every,
			ignore_init=args.ignore_init,
			init_train_epochs=None,
			train_epochs_decay_steps=None,
		)

	fixed = FixedAgent(bandit)
	logdir = Path(args.logdir).joinpath(
		f"{fixed.__class__.__name__}-on-{bandit.__class__.__name__}"
	)
	fixed_trainer = DCBTrainer(
		fixed, bandit, logdir=logdir, log_mode=["stdout", "tensorboard"]
	)
	results["fixed"] = fixed_trainer.train(
		timesteps=args.timesteps,
		update_interval=1,
		update_after=0,
		batch_size=1,
		train_epochs=1,
		log_every=args.log_every,
		ignore_init=args.ignore_init,
		init_train_epochs=None,
		train_epochs_decay_steps=None,
	)

	np.save("logs/{}_{}.npy".format(args.bandit, args.seed), results)
	results = np.load("logs/{}_{}.npy".format(args.bandit, args.seed), allow_pickle=True).item()

	# linpos = LinearPosteriorAgent(bandit)
	# logdir = Path(args.logdir).joinpath(
	#     f"{linpos.__class__.__name__}-on-{bandit.__class__.__name__}-{datetime.now():%d%m%y%H%M%S}"
	# )
	# linpos_trainer = DCBTrainer(
	#     linpos, bandit, logdir=logdir, log_mode=["stdout", "tensorboard"]
	# )
	# results["linpos"] = linpos_trainer.train(
	#     timesteps=args.timesteps,
	#     update_interval=1,
	#     update_after=args.update_after,
	#     log_every=args.log_every,
	#     ignore_init=args.ignore_init,
	# )

	# neural_linpos = NeuralLinearPosteriorAgent(bandit)
	# logdir = Path(args.logdir).joinpath(
	#     f"{neural_linpos.__class__.__name__}-on-{bandit.__class__.__name__}-{datetime.now():%d%m%y%H%M%S}"
	# )
	# neural_linpos_trainer = DCBTrainer(
	#     neural_linpos, bandit, logdir=logdir, log_mode=["stdout", "tensorboard"]
	# )
	# results["neural-linpos"] = neural_linpos_trainer.train(
	#     timesteps=args.timesteps,
	#     update_interval=50,
	#     update_after=args.update_after,
	#     batch_size=args.batch_size,
	#     train_epochs=100,
	#     log_every=args.log_every,
	#     ignore_init=args.ignore_init,
	#     init_train_epochs=None,
	#     train_epochs_decay_steps=None,
	# )

	# neural_noise = NeuralNoiseSamplingAgent(bandit)
	# logdir = Path(args.logdir).joinpath(
	#     f"{neural_noise.__class__.__name__}-on-{bandit.__class__.__name__}-{datetime.now():%d%m%y%H%M%S}"
	# )
	# neural_noise_trainer = DCBTrainer(
	#     neural_noise, bandit, logdir=logdir, log_mode=["stdout", "tensorboard"]
	# )
	# results["neural-noise"] = neural_noise_trainer.train(
	#     timesteps=args.timesteps,
	#     update_interval=50,
	#     update_after=args.update_after,
	#     batch_size=args.batch_size,
	#     train_epochs=100,
	#     log_every=args.log_every,
	#     ignore_init=args.ignore_init,
	#     init_train_epochs=None,
	#     train_epochs_decay_steps=None,
	# )

	# variational = VariationalAgent(bandit)
	# logdir = Path(args.logdir).joinpath(
	#     f"{variational.__class__.__name__}-on-{bandit.__class__.__name__}-{datetime.now():%d%m%y%H%M%S}"
	# )
	# variational_trainer = DCBTrainer(
	#     variational, bandit, logdir=logdir, log_mode=["stdout", "tensorboard"]
	# )
	# results["variational"] = variational_trainer.train(
	#     timesteps=args.timesteps,
	#     update_interval=50,
	#     update_after=args.update_after,
	#     batch_size=args.batch_size,
	#     train_epochs=100,
	#     log_every=args.log_every,
	#     ignore_init=args.ignore_init,
	#     init_train_epochs=10000,
	#     train_epochs_decay_steps=100,
	# )

	# plot_multi_runs(args, results, title="{}.pdf".format(args.bandit))
	# print(f"\nCompleted experiment at {(datetime.now() - start_time).seconds}\n")


import argparse

def main(args):
	if args.algo.lower() == "all" and args.bandit.lower() != "all":
		run_multi_algos(args)
	elif args.algo.lower() != "all" and args.bandit.lower() == "all":
		run_multi_bandits(args)
	elif args.algo.lower() == "all" and args.bandit.lower() == "all":
		raise ValueError("all argument cannot be used for both bandit and algorithm")
	else:
		run_single_algos_on_bandit(args)


if __name__ == "__main__":

	parser = argparse.ArgumentParser(description="Train Deep Contextual Bandits")

	parser.add_argument(
		"--run-experiment",
		help="Run pre written experiment with all algos",
		action="store_true",
	)

	parser.add_argument(
		"-a",
		"--algo",
		help="Which algorithm to train",
		default="neural-greedy",
		type=str,
	)
	parser.add_argument(
		"-b", "--bandit", help="Which bandit to train on", default="covertype", type=str
	)
	parser.add_argument(
		"-t",
		"--timesteps",
		help="How many timesteps to train for",
		default=5000,
		type=int,
	)
	parser.add_argument("--batch-size", help="Batch Size", default=256, type=int)
	parser.add_argument(
		"--update-interval", help="Update Interval", default=20, type=int
	)
	parser.add_argument(
		"--update-after",
		help="Timesteps to start updating after",
		default=500,
		type=int,
	)
	parser.add_argument(
		"--train-epochs",
		help="Epochs to train for each update",
		default=20,
		type=int,
	)
	parser.add_argument(
		"--log-every",
		help="Timesteps interval for logging",
		default=100,
		type=int,
	)
	parser.add_argument(
		"--logdir",
		help="Directory to store logs in",
		default="./logs/",
		type=str,
	)
	parser.add_argument(
		"--ignore-init",
		help="Initial no. of step to ignore",
		default=10,
		type=int,
	)
	parser.add_argument(
		"--init-train-epochs",
		help="Initial no. of step to ignore",
		default=None,
		type=int,
	)
	parser.add_argument(
		"--train-epochs-decay-steps",
		help="Initial no. of step to ignore",
		default=None,
		type=int,
	)
	parser.add_argument(
		"--download",
		help="Download data for bandit",
		action="store_true",
	)

	parser.add_argument('--method', type=str, default='free')
	# parser.add_argument('--alpha', type=float, default=1.)
	parser.add_argument('--ip', type=str, default=None)
	parser.add_argument('--n_ensembles_prior', type=int, default=64)
	parser.add_argument('--epsilon', type=float, default=5e-2)
	parser.add_argument('--n_sample', type=int, default=256)
	parser.add_argument('--W_var', type=float, default=2.)
	parser.add_argument('--b_var', type=float, default=0.01)
	parser.add_argument('--n_ensemble', type=int, default=10)
	parser.add_argument('--f_alpha', type=float, default=1.)
	parser.add_argument('--seed', type=int, default=0)

	args = parser.parse_args()

	np.random.seed(args.seed)
	torch.manual_seed(args.seed)
	torch.cuda.manual_seed_all(args.seed)

	if args.run_experiment:
		run_experiment(args)
	else:
		main(args)
