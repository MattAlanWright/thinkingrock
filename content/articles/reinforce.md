---
title: "Practical REINFORCE in PyTorch"
date: 2020-01-23T21:31:24-04:00
description: "Hands-on practical introduction to the policy gradient theorem and REINFORCE using PyTorch"
draft: false
toc: false
---

This article is a hands-on introduction to building gradient-based reinforcement learning algorithms in PyTorch. We'll review the policy gradient theorem, the foundation for gradient-based learning methods, and how it's used in practice. Then we'll implement the classic REINFORCE learning algorithm, as it appears in [Sutton and Barto's *Reinforcement Learning*](http://www.andrew.cmu.edu/course/10-703/textbook/BartoSutton.pdf) and use it to teach an agent to solve the OpenAI Gym [CartPole environment](https://gym.openai.com/envs/CartPole-v0/). This algorithm is a great way to gain experience with gradient-based learning, and understanding it will help pave the way to building more complex learning algorithms, like the Actor-Critic.

If you've found this page then I'm assuming that you know the basics of reinforcement learning methods and terminology. By the end of this article I hope that you'll have developed an intuition for how gradient-based learning builds on these foundations.

## The Policy Gradient Theorem

Before we look at REINFORCE, let's dive into the *policy gradient theorem*, which provides the foundation of the algorithm. For the case of an episodic task, the policy gradient theorem as described in Sutton and Barto states that

$$ \nabla J(\theta) \propto  \sum_s{\mu(s)}\sum_a{q_{\pi}(s,a)\nabla\pi(a|s,\theta)}$$

where

- $J(\theta)$ is a scalar performance measure of the current policy
- $\mu(s)$ is the distribution of states $s$ over policy $\pi$
- $q_{\pi}(s,a)$ is the action value function evaluated at state $s$ and action $a$
- $\pi(a|s,\theta)$ is the policy, evaluated at state $s$ and action $a$, and parameterized by $\theta$

We'll explain what this theorem is saying soon, but first we should be clear on what all those pieces are. If you're reading this post, I'm going to assume you're pretty comfortable with the action value function $q_{\pi}(s,a)$ . The state distribution is self-explanatory* and we'll come back to the performance measure in a bit. For now, let's spend a bit more time with this formulation of $\pi$, which probably looks a little different than you're used to.

There are two key things to notice about the policy under this formulation. First, it's parameterized on $\theta$. If this is your first look at REINFORCE or the policy gradient theorem, you might not be used to seeing that $\theta$ there. While the policy gradient theorem doesn't require any specific formulation of $\pi$, it does require that it's a parameterized function, i.e. that it uses some set of parameters $\theta$ to calculate the final result when passed in an action $a$ and state $s$. We'll assume (or assert/decide/whatever) that $\theta$ is a vector or matrix of weights used in some simple linear function or neural network, since this is almost always the case in practice. Second, notice that we're taking the gradient of this function, so it must be differentiable - one more good reason to use a neural network. To be clear, it must be differentiable with respect to the weight vector $\theta$, with $a$ and $s$ fixed. This is because we ultimately want to figure out what direction we can nudge those weights in order to increase or decrease the probability of taking action $a$ at state $s$ in the future.

Now let's take another look at the performance measure $J(\theta)$. I'm not going to get into the nitty-gritty here, since this is much better handled in Sutton and Barto and because I'm more concerned with getting to the implementation, but the idea here is that $J(\theta)$ is a scalar measurement of "how good" your policy is as a function of its parameter vector $\theta$. Of course, it's not at all obvious what this function should be, and there are actually plenty of candidates depending on whether this is an episodic or continuing environment. One possible measurement that makes pretty good intuitive sense is the value of your starting state:

$$J(\theta) \equiv V_{\pi_{\theta}}(s_{0})$$

where $V_{\pi_{\theta}}(s_{0})$ is a standard state value function, i.e. an estimate of the total return you expect to see starting in state $s_{0}$ and following your policy $\pi_{\theta}$. This should make intuitive sense, since if the value of $s_0$ is high, then your policy is obviously a good one and whatever $\theta$ you chose must have been good. Conversely, if it's low then your policy isn't great and you want to adjust $\theta$ to increase the reward you expect to see. Of course, in order to do that, you need to know to know how $\theta$ should be adjusted to increase your expected return. That's where the policy gradient theorem comes in.

The policy gradient theorem gives you a tractable, useful definition of the gradient of your performance measure (or at least something proportional to the gradient, which is just as good). The proof is available elsewhere (check Sutton and Barto), but it's true for several different candidate performance measures, including $J(\theta) \equiv V_{\pi_{\theta}}(s_{0})$. This is described pretty off-handedly in Sutton and Barto, but it's worth taking a second to let it sink in: using the formula from the beginning of this section (and a few more steps we'll describe below) we can determine an update to our weight vector $\theta$ that will increase the performance of our agent, regardless of the environment or the task.

**\*Just in case you *don't* think it's self-explanatory, this is, roughly, the probability of being in state $s$, or the frequency with which state $s$ appears. We're going to make this term disappear later, so don't sweat it too much right now.**

### Simplifying the formula

That's a really good start, but while it's mathematically tractable, the policy gradient as shown above can't be used in a real-world learning algorithm. First of all, there's that state distribution $\mu(s)$, which we don't know. Second, the equation shown above is generalized over the state and action variables $s$ and $a$, but we don't want a general formula - we want an algorithm that can operate on samples of the random variables $S_{t}$ and $A_{t}$. What do I mean by that? The definition of the policy gradient above is based on $q_{\pi}(s,a)$ and  $\pi(a|s,\theta)$ over the entire domain  of $s$ and $a$. But we don't know what $q_{\pi}(s,a)$ and $\pi(a|s,\theta)$ look like over all $s$ and $a$. We likely don't know what they look like at all. What we *can* do, is sample over those domains. In other words, we have random variables $S_{t}$ and $A_{t}$ that we can sample by letting episodes play out under our policy $\pi$ to try to get an estimate of those functions. So let's start chipping away at that formula to see if we can come up with something more useful.

First, notice that

$$\sum_s{\mu(s)}\sum_a{q_{\pi}(s,a)\nabla\pi(a|s,\theta)}$$

is just an expectation over $s$. This is simply the definition of an expectation, bearing in mind that $\mu(s)$ is effectively the probability distribution of $s$. So, keeping in mind that

$$\mathbb{E}[f(S_{t})] \equiv \sum_{s}Pr(S_{t} = s)f(s)$$

we can re-write this as

$$\mathbb{E}[{\sum_a{q(S_{t},a)\nabla\pi(a|S_{t},\theta)}}]$$

We got rid of that $s$ and replaced it with something we can actually sample, namely the random variable $S_{t}$. Now let's try to do the same with that $a$. For this we're going to use some trickery. If we multiply everything inside that expectation by $\frac{\pi(a|S_{t},\theta)}{\pi(a|S_{t},\theta)}$ we end up with 

$$\mathbb{E}[\sum_{a}   \pi(a|S_{t},\theta)   q(S_{t},a)   \frac{\nabla\pi(a|S_{t},\theta)}{\pi(a|S_{t},\theta)}  ]$$

This might not look better at first, but remember that $\pi(a|S_{t},\theta)$ is really the distribution over actions. This means that summation over actions becomes another expectation. Since the expectation of an expectation is an expectation, we can re-write this as

$$\mathbb{E}[q(S_{t},A_{t})   \frac{\nabla\pi(A_{t}|S_{t},\theta)}{\pi(A_{t}|S_{t},\theta)}  ]$$

Our $a$ has been replaced with the much more useful $A_{t}$. And since the expectation of that action value function is by definition the expected return, we can simplify a bit more to

$$\mathbb{E}[G_{t}  \frac{\nabla\pi(A_{t}|S_{t},\theta)}{\pi(A_{t}|S_{t},\theta)}  ]$$

Voila! We now have a dead-simple formula over two random variables, $S_{t}$ and $A_{t}$, both of which we can sample repeatedly to build up that expectation. For one last simplifying step, let's recall that

$$\nabla \ln{f} = \frac{\nabla{f}}{f}$$

and tweak that last formula just a bit. What we end up with is

$$\nabla{J(\theta)} \propto \mathbb{E}[G_{t}\nabla \ln \pi(A_{t}|S_{t},\theta)]$$

That last step might seem needless, but logs just happen to be nicer to play with (they're much more numerically stable because they grow so slowly). The fact that its proportional rather than equivalent is fine: that proportionality rule gets absorbed into whatever learning rate we choose and that all comes out in the wash.

So that's it. That formula says that if we can somehow approximate the expected value of that thing on the right (not necessarily trivial, but doable) then we have an answer to the question "How can I tweak these weights to improve my policy?"

Let's see how we might go about that.

### The Policy Gradient, Monte Carlo, and REINFORCE

The beauty of REINFORCE is its simplicity. It's pretty much the most straight-forward method you could come up with to put the policy gradient theorem to work. The theorem says we need to evaluate the expectation of $G_{t}\nabla \ln \pi(A_{t}|S_{t},\theta)$, so how might we do that? Rather than sprain any muscles trying to be clever, REINFORCE does the obvious: Monte Carlo sampling. For any given episode of a reinforcement learning task, $G_{t}$ is measurable and samples of $A_{t}$ and $S_{t}$ can be recorded. As long as you don't mind waiting until the end of the episode to calculate the expectation and perform your update, that's all you need.

The idea behind REINFORCE is: let an episode play out and record all states, actions and rewards as you go. Once it's done, calculate $G_{t}$ for all $t$. Then loop through all your recorded state-action pairs, feed those back into your policy and update your weights according to the gradient of that policy. Below is the definition of REINFORCE, taken directly from Sutton and Barto:

![reinforce_algo](/reinforce_algo.png)

Since I've lifted most of this post right from the book, the notation should look familiar. The constant $\alpha$ is our learning rate and takes care of the proportionality in our original formula. One new item that we haven't seen before is the constant $\gamma$. Anyone coming from a reinforcement learning background will be familiar with this. This is the *discount factor*, and it's there to ensure that the later a reward comes in time, the less effect it has on $G_{t}$. This is pretty standard reinforcement learning and explained better elsewhere so if you're uncomfortable with this, take a quick tour through the internet and come back. It doesn't make a lot of difference and in fact you could implement REINFORCE without it (though I do include it in my code).

## PyTorch Implementation

Let's finally get to the implementation. I'm going to assume you have PyTorch installed. You'll also need the OpenAI gym package, since we'll be teaching an agent to solve the CartPole environment. The final code will be available on GitHub as a Jupyter notebook.

### Policy

The first thing we'll do is define our policy, but before we get into the code we need to discuss a subtle but important point about the definition of an agent's "policy". Typically, an agent's policy is its decision-making tool: it takes in a state, produces a probability distribution over the possible actions, and then chooses an action by sampling from that probability distribution. When we talk about policies, this is usually what we mean. But in the formula for the policy gradient given above, this is not quite what we mean by $\pi(a|s,\theta)$. Our policy $\pi$ does *not* include the sampling mechanism, it's just the probability distribution over actions. And when we write  $\pi(a|s,\theta)$, we're referring to the probability of selecting the *specific action* $a$. This is a pretty subtle point, but it's important to know precisely what the policy gradient means when you're trying to code it.

The diagram below illustrates our policy for the CartPole environment, whose state is a tensor of length four and in which we can take one of two possible actions:

![policy](/policy.png)

Spoiler alert: our policy is going to be a simple linear network with no hidden layers. This network, which we'll refer to specifically as the *policy network*, produces a probability distribution over the two actions using a softmax. From here, that probability distribution is fed into a multinomial sampler which chooses an action based on that probability distribution. Why is it important for us to keep all of these components separate? Because the policy gradient theorem requires us to take the gradient of our policy, and there's no such thing as the gradient of a sampler. Once we select an action, we *can* take the gradient of our policy network, defining the probability of that action as the output.

So, with all that in mind, let's define our *Policy Network*. This is going to be our $\pi(a|s,\theta)$, without the sampling mechanism:

```python
class PolicyNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(PolicyNetwork, self).__init__()
        self.output = nn.Linear(input_size, output_size)

    def forward(self, x):
        x = F.softmax(self.output(x))
        return x
```

If you aren't familiar with the PyTorch-isms above, then take a look at [PyTorch's neural network tutorial](https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html#sphx-glr-beginner-blitz-neural-networks-tutorial-py "PyTorch Neural Network Tutorial"). All we're doing is defining a simple network that takes the state as an *input_size*-length vector and produces a probability distribution over *output_size* possible actions. The softmax is used to produce a proper probability distribution from our network's output. Pretty simple so far.

### REINFORCE Algorithm

I decided to implement REINFORCE as a single function that internally creates a *PolicyNetwork* instance:

```python
def REINFORCE(env, state_space_size, action_space_size, num_episodes, gamma=0.99):
    
    # Check to see if we can run on the GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Create policy network (i.e. differentiable policy function)
    policy = PolicyNetwork(state_space_size, action_space_size).to(device)
    optimizer = optim.Adam(policy.parameters(), lr=1e-3)
    
    ...
```

Of course we do the obligatory check to see if we have a GPU available. We create our network as well as an optimizer for training later.

REINFORCE plays out some number of episodes and updates its parameters after each one. Let's start adding the scaffolding for that. Assume the next few code blocks are being appended to previous one (minus the correct indentation). I'll let you know when that changes.

```python
# Record all of the scores to review later
scores = []

for i in range(num_episodes):

    # All states, actions and rewards need to be recorded for training
    states  = []
    actions = []
    rewards = []

    # Reset the score and the environment for this episode
    score = 0
    state = env.reset()

    # Play out an episode using the current policy
    while True:

        ...
```

To play through an episode, we feed the current state to our policy network, sample an action from the probabilities it produces, feed the action to the environment and repeat. Along the way, we record everything for training. Remember, this is a Monte Carlo algorithm and all our training happens later:

```python
# Play out an episode using the current policy
while True:

    # Take a step and generate action probabilities for current state
    # The state must first be turned into a tensor and sent to the device
    state_tensor = torch.from_numpy(state).float().to(device)
    action_probs = policy.forward(state_tensor)

    # Sample from softmax output to get next action
    # 'Categorical' is the same as 'Multinomial'
    m = Categorical(action_probs)
    action = m.sample()

    # Take another step, update the state, and check the reward
    # Calling item retrieves the action value from the action tensor
    next_state, reward, done, _ = env.step(action.item())
    score += reward

    # Record all of our states, actions and rewards
    rewards.append(reward)
    states.append(state)
    actions.append(action)

    # Update the state for the next step
    state = next_state

    if done:
        print("Episode: {} Score: {}".format(i, score))
        break
```

There are a couple of PyTorch-specific things to look at here. First, our policy needs the state to be a PyTorch tensor, so we need to convert it from a numpy array. Second, in case you're not familiar with 'Categorical', it is imported from PyTorch's *distributions* package and allows us to sample from a multinomial distribution over the action probabilities we generated from our policy. Basically, we can feed it a tensor of probabilites and then sample from that vector according to those probabilities. This is that non-differentiable sampler we talked about earlier.

Now it's time to train. We update our weights after every single episode:

```python
# Now that the episode is done, update out policy for each timestep
for t in range(len(states)):

    # Get returns at all times, i.e. G[t] for all t
    G = sum([r * gamma ** i for i, r in enumerate(rewards[t:])])

    # Update our weights. First, zero the gradients
    optimizer.zero_grad()

    # Convert state to a tensor and re-evaluate probability distribution
    state_tensor = torch.from_numpy(states[t]).float().to(device)
    probs = policy(state_tensor)

    # Evaluate performanc as per the policy gradient theorem and update our
    # weights to take a step in the direction of increased performance.
    m = Categorical(probs)
    performance = -m.log_prob(actions[t]) * G
    performance.backward()
    optimizer.step()
```

This is the busiest section, so let's walk through it slowly. That first line inside the for-loop is just using list comprehension to generate our list of returns $G$, index-able by `t`. I'm pretty sure this could be more efficiently done, but this way is simpler.

Then we zero our gradients. Pretty standard.

Next, we re-feed the state at each timestep back into our policy network. This is done because we're about to take the gradient of that policy network with respect to our weights, given a fixed state and action selection. We want to know what direction we could have tweaked the weights when fed state $s_{t}$ in order to increase or decrease the probability of action $a_{t}$, depending on how good it was.

Next, we re-build our *Categorical* sampler. Yes, I know, I'm re-doing a lot of work here. PyTorch is actually capable of letting us avoid re-feeding the state and re-creating the sampler but at the cost of my intention to keep this is as simple and one-to-one with Sutton and Barto as I can. The sampler is rebuilt specifically because PyTorch's distribution classes have a built-in mechanism that let's us take the gradient of our policy network as if our policy network had only a single output, the chosen action.

If that didn't make sense, then take another look at the diagram of the policy above. The output of the network is *two* action probabilities. It's effectively two different functions operating in parallel. But we want the gradient of $\pi(a|s,\theta)$, as in the gradient of whichever of those functions produced the selected action. PyTorch's distribution classes let you do this: you can feed all probabilities into the distribution class (for us, this is the 'Categorical' class) and then call `log_prob(action)` on that instance. This not only gives you the log of the probability of that action but provides that log probability as a one-dimensional tensor that you can call `backward()` on, i.e. take the gradient of. This is exactly what the policy gradient theorem wants from us.

If you want to dive into those last points a bit more, then check out [PyTorch's distributions documentation](https://pytorch.org/docs/stable/distributions.html) where they explain it more thoroughly. Be careful though: they use terminology that's slightly out of sync with us and with Sutton and Barto.

After that we: multiply by the return at that time step, as prescribed by the policy gradient theorem; multiply by -1 to turn our gradient descent machine into a gradient *ascent* machine; let PyTorch update the weights. Remember that, according to the policy gradient theorem, the line

```python
m.log_prob(actions[t]) * G
```

*is* the gradient of our performance measure (well, proportional to it). This isn't the gradient of a loss function that we want to minimize, it's the gradient of the performance metric we want to maximize (don't get confused: it's not the performance metric itself, it's the performance metric's gradient). Since PyTorch automatically performs gradient descent, we reverse this with the negative sign.

And that's it! Now we just need to create an environment and let it do its thing (this code block is no longer inside the definition of `REINFORCE`):

```python
env = gym.make("CartPole-v0")
action_space_size = env.action_space.n
state_space_size = env.observation_space.shape[0]
scores = REINFORCE(env,
                   state_space_size=state_space_size,
                   action_space_size=action_space_size,
                   num_episodes=5000)
```

Below is a screenshot of the training curve for this code on the CartPole-v0 environment.

![results](/results.png)

Not too shabby for an algorithm from 1992. You can check out the full implementation at https://github.com/MattAlanWright/pytorch-reinforce.

This article barely scratches the surface. There's a lot more you can do with the policy gradient theorem. Despite its age, it's still a major component in most modern reinforcement learning techniques. For a bit of background reading, check out [Williams' original paper](http://www-anw.cs.umass.edu/~barto/courses/cs687/williams92simple.pdf) where he defines the REINFORCE family of algorithms, and the [paper by Sutton et. al.](https://homes.cs.washington.edu/~todorov/courses/amath579/reading/PolicyGradient.pdf) where they formalize the policy gradient theorem.

I hope this helped someone out there. Feel free to email me with questions, comments or critiques.

