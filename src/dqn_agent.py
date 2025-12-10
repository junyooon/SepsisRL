class DQNAgent:
    def __init__(self, s_size, a_size, h_size, gamma, lr, buffer_capacity, b_size, print_every, device):
        self.s_size = s_size
        self.a_size = a_size
        self.h_size = h_size
        self.gamma = gamma
        self.lr = lr
        self.b_size = b_size
        self.print_every = print_every
        self.device = device
        self.train_step_count = 0

        # Policy and Target Network
        self.policy_net = Policy(s_size, a_size, h_size).to(device)
        self.target_net = Policy(s_size, a_size, h_size).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # Optimizer and Loss
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr)
        self.criterion = nn.SmoothL1Loss()

        # Replay Buffer
        self.memory = ReplayBuffer(buffer_capacity)

    def select_action(self, state, epsilon):
        if random.random() < epsilon:
            # Exploration: Choose a random admissible action
            return random.randrange(self.a_size)
        else:
            # Exploitation: Choose the action with the maximum Q-value
            self.policy_net.eval()
            with torch.no_grad():
                state_one_hot = torch.zeros(1, self.s_size, device=self.device)
                state_one_hot[0, state] = 1.0
                q_values = self.policy_net(state_one_hot)
                action = q_values.argmax().item()
            self.policy_net.train()
            
            return action

    def _convert_batch_to_tensors(self, transitions):
        batch = self.memory.transition(*zip(*transitions))

        # Convert discrete state indices
        s_batch = torch.zeros(len(transitions), self.s_size, device=self.device)
        next_s_batch = torch.zeros(len(transitions), self.s_size, device=self.device)

        for i, s_idx in enumerate(batch.state):
            s_batch[i, s_idx] = 1.0
        for i, s_idx in enumerate(batch.next_state):
            # Only set next state if not terminal
            if not batch.done[i]:
                next_s_batch[i, s_idx] = 1.0

        # Convert other components
        a_batch = torch.tensor(batch.action, dtype=torch.long, device=self.device).unsqueeze(1)
        r_batch = torch.tensor(batch.reward, dtype=torch.float, device=self.device).unsqueeze(1)
        done_mask = torch.tensor(batch.done, dtype=torch.float, device=self.device).unsqueeze(1)
        prev_action_batch = torch.tensor(batch.prev_action, dtype=torch.long, device=self.device).unsqueeze(1)

        return s_batch, a_batch, r_batch, next_s_batch, done_mask, prev_action_batch

    def learn(self):
        transitions = self.memory.sample(self.b_size)
        if transitions is None:
            return

        s_batch, a_batch, r_batch, next_s_batch, done_mask, _ = self._convert_batch_to_tensors(transitions)

        # Q(s_t, a_t)
        q_values = self.policy_net(s_batch).gather(1, a_batch)

        # V(s_(t+1)) = max_a Q_target(s_(t+1), a)
        with torch.no_grad():
            next_q_values = self.target_net(next_s_batch).max(1)[0].unsqueeze(1)

        # target_q_values = R_t + gamma * V(s_(t+1)) * (1 - done_mask)
        target_q_values = r_batch + self.gamma * next_q_values * (1 - done_mask) # mask 'done' states

        # Huber Loss
        loss = self.criterion(q_values, target_q_values)

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_value_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()

        # Update Target Network
        self.train_step_count += 1
        if self.train_step_count % self.print_every == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        return loss.item()

class ConstrainedDQNAgent(DQNAgent):
    def __init__(self, delta_max, lambda_penalty, **kwargs):
        super().__init__(**kwargs)
        self.delta_max = delta_max
        self.lambda_penalty = lambda_penalty

    def learn(self):
        transitions = self.memory.sample(self.b_size)
        if transitions is None:
            return

        state_batch, action_batch, reward_batch, next_state_batch, done_mask, prev_action_batch = self._convert_batch_to_tensors(transitions)

        penalties = torch.zeros_like(reward_batch, device=self.device)
        for i in range(self.b_size):
            curr_action_int = action_batch[i].item()
            prev_action_int = prev_action_batch[i].item()
            
            penalty_val = calculate_penalty(curr_action_int, prev_action_int, self.delta_max)
            penalties[i] = penalty_val

        # R(s,t) = R_original(s_t, a_t) - λ * Penalty(a_t, a_(t-1))
        final_reward_batch = reward_batch - self.lambda_penalty * penalties

        # Q(s_t, a_t)
        current_q_values = self.policy_net(state_batch).gather(1, action_batch)

        # V(s_(t+1)) = max_a Q_target(s_(t+1), a) - DQN
        with torch.no_grad():
            next_q_values = self.target_net(next_state_batch).max(1)[0].unsqueeze(1)

        target_q_values = final_reward_batch + self.gamma * next_q_values * (1 - done_mask)

        # Optimize
        loss = self.criterion(current_q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_value_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()

        # Update Target Network
        self.train_step_count += 1
        if self.train_step_count % self.print_every == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        return loss.item()
