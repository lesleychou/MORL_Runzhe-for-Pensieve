from __future__ import absolute_import, division, print_function
import random
import torch
import copy
import numpy as np
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
from collections import namedtuple
from collections import deque

use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
Tensor = FloatTensor


class MetaAgent(object):
    '''
    (1) act: how to sample an action to examine the learning
        outcomes or explore the environment;
    (2) memorize: how to store observed observations in order to
        help learing or establishing the empirical model of the
        enviroment;
    (3) learn: how the agent learns from the observations via
        explicitor implicit inference, how to optimize the policy
        model.
    '''

    def __init__(self, model, args, is_train=False):
        self.model_ = model
        self.model = copy.deepcopy(model)
        self.is_train = is_train
        self.gamma = args.gamma
        self.epsilon = args.epsilon
        self.epsilon_decay = args.epsilon_decay
        self.epsilon_delta = (args.epsilon - 0.05) / args.episode_num

        self.mem_size = args.mem_size
        self.batch_size = args.batch_size
        self.weight_num = args.weight_num
        self.trans_mem = deque()

        # state, action, next state, reward, terminal_state
        self.trans = namedtuple('trans', ['s', 'a', 's_', 'r', 'd'])
        self.priority_mem = deque()

        if args.optimizer == 'Adam':
            self.optimizer = optim.Adam(self.model_.parameters(), lr=args.lr)
        elif args.optimizer == 'RMSprop':
            self.optimizer = optim.RMSprop(self.model_.parameters(), lr=args.lr)

        self.w_kept = None
        self.update_count = 0
        self.update_freq = args.update_freq

        if self.is_train:
            self.model_.train()
        if use_cuda:
            self.model.cuda()
            self.model_.cuda()

    def act(self, state, preference=None):
        '''
        what is the reward_size for VS?  3
        the state of VS: "past chunk throughput, past chunk download time,
            next chunk sizes, current buffer size, # of chunks left, last chunk bitrate"

        How to calculate Q here? use NaiveOnelayerCQN, add preferences in the forward step
        '''
        # random pick a preference if it is not specified
        if preference is None:
            if self.w_kept is None:
                self.w_kept = torch.randn(self.model_.reward_size)
                self.w_kept = (torch.abs(self.w_kept) / \
                               torch.norm(self.w_kept, p=1)).type(FloatTensor)
            preference = self.w_kept
            # weights is a 'reward_size' tensor
            # reward: [bitrate, rebuffering, smoothness], (1, 3)

        state = torch.from_numpy(state).type(FloatTensor)

        # model return q, scalar value for each action with given state
        _, Q = self.model_(
            Variable(state.unsqueeze(0), requires_grad=False),
            Variable(preference.unsqueeze(0), requires_grad=False))

        action = Q.max(1)[1].data.cpu().numpy()
        action = int(action[0])

        # what is this if doing??
        # if model is training, do the random action to explore
        # when trans_mem_size < each batch_size, OR the random number < epsilon, do the random action
        if self.is_train and (len(self.trans_mem) < self.batch_size or \
                              torch.rand(1)[0] < self.epsilon):
            action = np.random.choice(self.model.action_size, 1)[0]
            action = int(action)

        return action

    def memorize(self, state, action, next_state, reward, terminal, roi=False):
        self.trans_mem.append(self.trans(
            torch.from_numpy(state).type(FloatTensor),  # state
            action,  # action
            torch.from_numpy(next_state).type(FloatTensor),  # next state
            torch.from_numpy(reward).type(FloatTensor),  # reward
            terminal))  # terminal

        # randomly produce a preference for calculating priority
        #
        #if roi: 
        #    preference = self.w_kept
        #else:
        preference = torch.randn(self.model_.reward_size)
        preference = (torch.abs(preference) / \
                      torch.norm(preference, p=1)).type(FloatTensor)

        state = torch.from_numpy(state).type(FloatTensor)

        # input state and action to the model
        # generate q value from the model
        _, q = self.model_(Variable(state.unsqueeze(0), requires_grad=False),
                           Variable(preference.unsqueeze(0), requires_grad=False))
        # why is this?
        q = q[0, action].data
        # what is 'wr'? weighted/scalarized reward
        wr = preference.dot(torch.from_numpy(reward).type(FloatTensor))
        if not terminal:
            next_state = torch.from_numpy(next_state).type(FloatTensor)
            # hq is q.detach
            hq, _ = self.model_(Variable(next_state.unsqueeze(0), requires_grad=False),
                                Variable(preference.unsqueeze(0), requires_grad=False))
            hq = hq.data[0]
            # what is 'p'? Priority
            p = abs(wr + self.gamma * hq - q)
        else:
            self.w_kept = None
            if self.epsilon_decay:
                self.epsilon -= self.epsilon_delta
            p = abs(wr - q)
        # why?
        p += 1e-5
	
        #if roi: 
        #    p = 1

        # create and store priority
        self.priority_mem.append(
            p
        )
        # when trans_mem_size grow more than mem_size, pop the first
        if len(self.trans_mem) > self.mem_size:
            self.trans_mem.popleft()
            self.priority_mem.popleft()

    # how does sample work?
    def sample(self, pop, pri, k):
        ''' Sample a minibatch of transitions from our memory
        pop: transition memory
        pri: priority
        k: batch size
        '''
        # reformat priority as a numpy array, with type float
        pri = np.array(pri).astype(np.float)

        # grab random indices
        inds = np.random.choice(
            range(len(pop)), k,
            replace=False,
            p=pri / pri.sum()
        )

        # return the sample
        return [pop[i] for i in inds]

    def actmsk(self, num_dim, index):
        mask = ByteTensor(num_dim).zero_()
        mask[index] = 1
        return mask.unsqueeze(0)

    # what is this? non-terminal-indices
    def nontmlinds(self, terminal_batch):
        mask = ByteTensor(terminal_batch)
        inds = torch.arange(0, len(terminal_batch)).type(LongTensor)
        inds = inds[mask.eq(0)]
        return inds

    def learn(self, preference=None):
        # why larger?
        # in DQN, we sample from the trans_mem a batch of of size
        # only when the replay buffer at least larger than batch_size, do sample
        # Until we have that, do nothing
        if len(self.trans_mem) > self.batch_size:

            self.update_count += 1

            # Sample a minibatch from the transition memory
            minibatch = self.sample(self.trans_mem, self.priority_mem, self.batch_size)
            # why to do batchify? how does it used
            # copy the trans for weight_num (default 32) times, then combine with the w_batch,
            # where for each trans only the weight is different
            batchify = lambda x: list(x) * self.weight_num
            state_batch = batchify(map(lambda x: x.s.unsqueeze(0), minibatch))
            action_batch = batchify(map(lambda x: LongTensor([x.a]), minibatch))
            reward_batch = batchify(map(lambda x: x.r.unsqueeze(0), minibatch))
            next_state_batch = batchify(map(lambda x: x.s_.unsqueeze(0), minibatch))
            terminal_batch = batchify(map(lambda x: x.d, minibatch))

            if preference is None:
                w_batch = np.random.randn(self.weight_num, self.model_.reward_size)
                # normalized w_batch
                w_batch = np.abs(w_batch) / \
                          np.linalg.norm(w_batch, ord=1, axis=1, keepdims=True)
                # repeat each w for 'batch_size' times
                w_batch = torch.from_numpy(w_batch.repeat(self.batch_size, axis=0)).type(FloatTensor)
            else:
                w_batch = preference.cpu().numpy()
                w_batch = np.expand_dims(w_batch, axis=0)
                w_batch = np.abs(w_batch) / \
                          np.linalg.norm(w_batch, ord=1, axis=1, keepdims=True)
                w_batch = torch.from_numpy(w_batch.repeat(self.batch_size, axis=0)).atype(FloatTensor)
            
            # give the 'w_batch', output the Q or learn the Q?
            __, Q = self.model_(Variable(torch.cat(state_batch, dim=0)),
                                Variable(w_batch))
            # detach since we don't want gradients to propagate
            # HQ, _    = self.model_(Variable(torch.cat(next_state_batch, dim=0), volatile=True),
            # 					  Variable(w_batch, volatile=True))
            _, DQ = self.model(Variable(torch.cat(next_state_batch, dim=0), requires_grad=False),
                               Variable(w_batch, requires_grad=False))
            _, act = self.model_(Variable(torch.cat(next_state_batch, dim=0), requires_grad=False),
                                 Variable(w_batch, requires_grad=False))[1].max(1)
            # what is HQ, DQ, act here
            # why use them? how to explain to others

            # for .gather(), see https://stackoverflow.com/questions/50999977
            # gather will index the rows of the DQ by the list of act.
            # HQ is current Q values
            HQ = DQ.gather(1, act.unsqueeze(dim=1)).squeeze()

            # w_reward_batch is weight_batch multiply reward_batch
            w_reward_batch = torch.bmm(w_batch.unsqueeze(1),
                                       torch.cat(reward_batch, dim=0).unsqueeze(2)
                                       ).squeeze()

            # non terminal indexes on the terminal mask
            nontmlmask = self.nontmlinds(terminal_batch)
            with torch.no_grad():
                # Tau_Q is the real target Q
                Tau_Q = Variable(torch.zeros(self.batch_size * self.weight_num).type(FloatTensor))
                # Tau_Q want all the non-terminal values and replace it by gamma*HQ
                # for terminal, return w_reward
                # for non-terminal, calculate Bellman
                Tau_Q[nontmlmask] = self.gamma * HQ[nontmlmask]
                Tau_Q += Variable(w_reward_batch)

            # what is act before?
            # act is directly from the model
            # action_batch is from the trans_mem
            actions = Variable(torch.cat(action_batch, dim=0))

            # Compute Huber loss
            # Q is the predicted Q
            # loss between Q and Tau_Q
            loss = F.smooth_l1_loss(Q.gather(1, actions.unsqueeze(dim=1)), Tau_Q.unsqueeze(dim=1))

            self.optimizer.zero_grad()
            loss.backward()
            for param in self.model_.parameters():
                param.grad.data.clamp_(-1, 1)
            self.optimizer.step()

            if self.update_count % self.update_freq == 0:
                self.model.load_state_dict(self.model_.state_dict())

            return loss.data

        return 0.0

    def reset(self):
        self.w_kept = None
        if self.epsilon_decay:
            self.epsilon -= self.epsilon_delta

    def predict(self, probe):
        return self.model(Variable(FloatTensor([0, 0]).unsqueeze(0), requires_grad=False),
                          Variable(probe.unsqueeze(0), requires_grad=False))

    def save(self, save_path, model_name):
        torch.save(self.model, "{}{}.pkl".format(save_path, model_name))
