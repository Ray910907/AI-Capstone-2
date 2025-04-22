import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
from torch.utils.data import DataLoader, TensorDataset
import tqdm
import  gym # Setting up new environment
from tqdm import trange,tqdm

# 設置環境和框架
class Network(nn.Module):

    def __init__(self, action_size, seed = 42): #constructor init method we have multidimentional array here hence, action size passed here.
        super(Network, self).__init__() # Initilizer of the base class to correctly activate inheretance
        self.seed = torch.manual_seed(seed)
        self.conv1 = nn.Conv2d(3, 32, kernel_size = 8, stride = 4) # Convulation --> 3-inputchannels(RGB), outputchannels-32, kernel size(8x8), stride = 4. -- EYES of AI
        self.bn1 = nn.BatchNorm2d(32) # Helps in training Normalization
        self.conv2 = nn.Conv2d(32, 64, kernel_size = 4, stride = 2) # These parameters are changed on the basis of experimentation and hyperparameter tuning for optimal result.
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, kernel_size = 3, stride = 1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 128, kernel_size = 3, stride = 1)
        self.bn4 = nn.BatchNorm2d(128) # EYES of AI finished.
        # Neurons in Flattening layer -- (Inputsize-Kernelsize + 2*padding)/stride + 1 -- for each convolutional layer.
        self.fc1 = nn.Linear(10 * 10 * 128, 512) # Fulllyconnected Layers - Brain of AI (inpur , Output Neurons(Taken by Experimentation))
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, action_size) # Brain Finished

    def forward(self, state): # Foreward propogation
        x = F.relu(self.bn1(self.conv1(state))) # Using conv1(Convolutional) in bn1(BatchNormalization) this in total taken in relu activation function.
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = x.view(x.size(0), -1) # Flattening -- first dimentions are kept and other are flattened to the first i.e 1 dimension
        x = F.relu(self.fc1(x)) # Forewarding to fully connected layer.
        x = F.relu(self.fc2(x))
        return self.fc3(x)
  
env = gym.make('MsPacman-v0', full_action_space = False) # Small part of other agents/ monsters will be deterministic.  full_action_space = False -- agent uses a set of simplified actions (Less computations).
state_shape = env.observation_space.shape
state_size = env.observation_space.shape[0]
number_actions = env.action_space.n
print('State shape: ', state_shape)
print('State size: ', state_size)
print('Number of actions: ', number_actions)

learning_rate = 5e-4
minibatch_size = 64
discount_factor = 0.99
#replay class wont be implemented but experience replay will still be implemented.
#soft update will not be done as it does not improve performance here

# preprocessing is necessary to feed it into AI
from PIL import Image
from torchvision import transforms

def preprocess_frame(frame): #frame converted to pytorch tensors
    frame = Image.fromarray(frame) #numpy array to a PIL image class as image is now in numpyarray form
    preprocess = transforms.Compose([transforms.Resize((128, 128)), transforms.ToTensor()]) # First resizing image and the turning it to a tensor(Also normalizes the frame).
    return preprocess(frame).unsqueeze(0) #adding extra batch dimension. (Which batch each frame belongs to)


class Agent(): #Creating Agent while integrating Experience Replay and Image Preprocessing

    def __init__(self, action_size, n_step=4):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.action_size = action_size
        self.local_qnetwork = Network(action_size).to(self.device)
        self.target_qnetwork = Network(action_size).to(self.device)
        self.optimizer = optim.Adam(self.local_qnetwork.parameters(), lr = learning_rate)
        self.memory = deque(maxlen = 10000) #double ended queue maxlen = capacity
        self.tau = 0.001
        self.n_step = n_step
        self.n_step_buffer = deque(maxlen=n_step)

    def soft_update(self, local_model, target_model, tau):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)
    
    def _get_n_step_transition(self):
        state, action, _, _, _ = self.n_step_buffer[0]
        reward, next_state, done = 0.0, None, False
        
        for idx, (_, _, r, ns, d) in enumerate(self.n_step_buffer):
            reward += (discount_factor ** idx) * r
            next_state, done = ns, d
            if done:
                break

        return (state, action, reward, next_state, done)

    def step(self, state, action, reward, next_state, done):
        state = preprocess_frame(state) # preprossing both the states
        next_state = preprocess_frame(next_state)
        
        self.memory.append((state, action, reward, next_state, done))
        # n-step-Q-learning
        '''
        
        self.n_step_buffer.append((state, action, reward, next_state, done))

        if len(self.n_step_buffer) == self.n_step:
            n_step_transition = self._get_n_step_transition()
            self.memory.append(n_step_transition)

        if done:
            while len(self.n_step_buffer) > 0:
                n_step_transition = self._get_n_step_transition()
                self.memory.append(n_step_transition)
                self.n_step_buffer.popleft()
        '''

        if len(self.memory) > minibatch_size: #if greater than batch size then learn.
            experiences = random.sample(self.memory, k = minibatch_size) # take experiences total i.e of the batchsize
            self.learn(experiences, discount_factor) # then learn form them

    def act(self, state, epsilon = 0.):
        state = preprocess_frame(state).to(self.device) # preprossing the images
        self.local_qnetwork.eval() #eval firm nn.Module
        with torch.no_grad():
            action_values = self.local_qnetwork(state)
        self.local_qnetwork.train()
        if random.random() > epsilon:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences, discount_factor):
        states, actions, rewards, next_states, dones = zip(*experiences) # unzipping the experiences (all states are already mytorch tensors)
        states = torch.from_numpy(np.vstack(states)).float().to(self.device) # vstack function can take a pytorch tensor and convert it to a numpy array.(i.e we dont have to convert it to a numpy array) After this we reconvert them into Pytorch tensors
        actions = torch.from_numpy(np.vstack(actions)).long().to(self.device) # alternative could be torch.cat function
        rewards = torch.from_numpy(np.vstack(rewards)).float().to(self.device)
        next_states = torch.from_numpy(np.vstack(next_states)).float().to(self.device)
        dones = torch.from_numpy(np.vstack(dones).astype(np.uint8)).float().to(self.device)
        next_q_targets = self.target_qnetwork(next_states).detach().max(1)[0].unsqueeze(1)
        q_targets = rewards + discount_factor * next_q_targets * (1 - dones)
        q_expected = self.local_qnetwork(states).gather(1, actions)
        loss = F.mse_loss(q_expected, q_targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        #self.soft_update(self.local_qnetwork, self.target_qnetwork, self.tau)

agent = Agent(number_actions)

number_episodes = 500
maximum_number_timesteps_per_episode = 10000
epsilon_starting_value  = 1.0
epsilon_ending_value  = 0.01
epsilon_decay_value  = 0.995
epsilon = epsilon_starting_value
scores_on_100_episodes = deque(maxlen = 100)
import matplotlib.pyplot as plt

scores_on_100_episodes = []
scores = []

import numpy as np
from PIL import Image

pbar = trange(1, number_episodes + 1, desc="Training", ncols=100)
best = 0.0
for episode in pbar:
    state, _ = env.reset()
    score = 0
    for t in range(maximum_number_timesteps_per_episode):
        action = agent.act(state, epsilon)
        next_state, reward, done, _, _ = env.step(action)
        agent.step(state, action, reward, next_state, done)
        state = next_state
        score += reward
        if done:
            break
    scores_on_100_episodes.append(score)
    scores.append(score)
    epsilon = max(epsilon_ending_value, epsilon_decay_value * epsilon)

    avg_score = np.mean(scores_on_100_episodes[-100:])
    pbar.set_description(f"Episode {episode} | Avg: {avg_score:.2f} | Score: {score:.2f}")

    if avg_score >= best:
        best = avg_score
        torch.save(agent.local_qnetwork.state_dict(), 'checkpoint.pth')

plt.plot(scores)
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.title("DQN on PacMan")
plt.savefig("./videos/Pacman_nq.png")

import glob
import io
import base64
import imageio
from IPython.display import HTML, display
from gym.wrappers.monitoring.video_recorder import VideoRecorder

def show_video_of_model(agent, env_name):
    env = gym.make(env_name, render_mode='rgb_array')
    state, _ = env.reset()
    done = False
    frames = []
    while not done:
        frame = env.render()
        frames.append(frame)
        action = agent.act(state)
        state, reward, done, _, _ = env.step(action)
    env.close()
    imageio.mimsave('pac_nq.mp4', frames, fps=30)

show_video_of_model(agent, 'MsPacman-v0')

def show_video():
    mp4list = glob.glob('*.mp4')
    if len(mp4list) > 0:
        mp4 = mp4list[0]
        video = io.open(mp4, 'r+b').read()
        encoded = base64.b64encode(video)
        display(HTML(data=''''''.format(encoded.decode('ascii'))))
    else:
        print("Could not find video")

show_video()