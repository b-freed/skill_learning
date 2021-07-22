class EpisodeDataBuffer():
	'''
	Data buffer for when we're using entire episodes
	'''
	def __init__(self,batch_size):
		self.batch_size = batch_size
		self.inputs = None
		self.targets = None

	def add_data(self,states,actions,next_states,rewards):

		# self.inputs should be a K x T x (state_dim+a_dim) array
		# self.targets should be a K x T x (state_dim+1) array

		assert len(states.shape) == 3

		states_actions      = np.concatenate([states,actions],axis=-1)
		next_states_rewards = np.concatenate([next_states,rewards],axis=-1)

		if self.inputs is not None:
			self.inputs  = np.concatenate([self.inputs,states_actions],    axis=0)
			self.targets = np.concatenate([self.targets,next_states_rewards], axis=0)

		else:
			self.inputs  = states_actions
			self.targets = next_states_rewards

	def get_dataloader(self):

		train_data = TensorDataset(torch.tensor(self.inputs, dtype=torch.float32),
								   torch.tensor(self.targets,dtype=torch.float32))

		train_loader = torch.utils.data.DataLoader(
			train_data,
			batch_size=self.batch_size,
			num_workers=1)

		return train_loader

	def __len__(self):
		if self.inputs is not None:
			return self.inputs.shape[0]
		return 0

	def load_data(self,data_file):
		data = np.load(data_file)
		states = data['states']
		actions = data['actions']
		next_states = data['next_states']
		rewards = data['rewards']
		self.add_data(states,actions,next_states,rewards)