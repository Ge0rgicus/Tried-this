# Tried-this
Learning Reinforcement learning the four attempts of generals.io

I wanted to learn more about DQN and CNN’s to create a self modifying AI that does not need human input to improve itself, like in the famous novel I Have No Mouth And I Must Scream. For that I picked generals because of its easy to implement AI’s and its strategic gameplay, which relies on resource management and predicting the opponent. The goal of this Project was to create an agent which could consistently beat the Expander agent the repository already gives us.

In the end I tried 4 different approaches with all of them over 1000 episodes. However, in the end none of the four managed to beat the Expander agent consistently, all falling under 20% success rate. Although I failed in making a successful agent I have learned why Reinforcement Learning is so difficult and what really matters.

This writeup will focus on my approach and what worked and what didn't, and what to focus on in the future

BACKGROUND.

Generals is a game that forces resource allocation to beat the other opponent. Each round the castle (the main base) generates one soldier and every 10 rounds each area gains one soldier. During the game the players can also take over other castles and areas. By using these soldiers the two players must take over each other's castle. 

It is challenging for the AI because of its partial visibility of around its own area, so it doesn't know exactly where the enemy is and also the fact that it relies heavily on future strategies the agent must focus on to beat its opponent. Because of its many possibilities the agent struggles to find the right strategy out of thousands.

The Expander agent focuses on one thing. Expand. It takes over as much area as possible to gain as many soldiers as possible to overwhelm the opponent, so the longer it is alive the stronger it becomes. Meaning it focuses mostly on the end game, until it finds the opponent, by which then it has more soldiers than the opponent.

Because of its strength, the agent must eliminate him quickly, but because it doesn't know where the enemy is it is often too late, so having 50%+ means it found a better strategy than expanding.

ATTEMPT 1

The first AI agent I tried was the Q-learning algorithm. It uses the Bellman equation to decide what it's going to do based on the future reward from the state and current action. The state has 5 features: army advantage, land advantage, distance, territory, and general alive. And 5 actions:  army advantage, land advantage, distance, territory, general alive from the strongest cell.

After more than 400 episodes it plateaued at 18-20% with 71 unique states and the table never grew more than 100 states. It had many flaws: it couldn't distinguish between armies clustered and spread out, it put many board positions into the same state, and lost nearly all spatial information.

From this I learned that Q learning is meant for smaller state spaces, also that features dictate how good the agent will be and that tabular methods cant solve complex problems.

ATTEMPT 2

Next I tried using a simple DQN with the same features as last time. So after swapping the Q table with a neural network, I implemented an experience replay buffer that collects state action reward and next state to be sampled in batches to stabilize the learning and a target network to effectively guide the algorithm to learn correctly in the right direction.

The reason I did this is because DQN is the most common type of RL algorithm out there and also because it would generalize better than a Q table. It had 3 hidden layers with 5 outputs. But in the end it had a success rate of 7%, even though it had a very low loss rate.

It had the same problem like last tim of having way too little of features and it also learned incorrectly, meaning it took in garbage and outputted garbage. But I learned from this that Neural networks can magically fix everything and that low loss doesn't mean right learning.

ATTEMPT 3

Next time I tried using a spatial DQN. with a CNN it managed to look at the terrain much more clearly: owned cells, enemy cells, armies, mountains, generals. And I also increased the actions space to 256 for every cell and action it can make. 

After convoluting and dense neural networks, it would go through an action mask to check all valid moves and with a larger batch to use I was sure it would be better than the expander agent. However it still fell short, with a 4.3% win rate and a HUGE loss rate.

Because of its large states it was a nightmare to compute. It needed much and much more than 1000 episodes for it to be somewhat efficient. But I learned that more doesn't mean better. The bigger it is the more data it needs.

ATTEMPT 4

The last attempt was to use the CNN full board but with much less actions (meaning only 5). Because it would have rich states but only 5 actions. Decreasing the load heavily. 
It had a more aggressive learning method. Very high epsilon decay and higher learning rate. But in the end it only reached 9%. 

From this I learned that I should not rely on AI to write my AI agents, and that I should be competent myself.

SECTION 7

I learned that the more actions there are the more attempts you need. I only had 1000 episodes even though I had 256 actions and that's why it failed. Why did it fail? Well imagine you are a knight and you know 5 moves, after a few dozen fights you put the moves in correct order for maximum fighting efficiency. But now imagine you have 1000 moves, and you must find the most efficient combination for that. Even after 1000 fights you would not be able to try every combination, and at that rate you would have died. So that is why my agent failed. It didn't fight enough before finding the optimal strategy. This is called the sample efficiency problem - the more actions you have, the more episodes you need to explore them all. My 400 episodes were enough for 5 actions but nowhere near enough for 256.

The other reason all 4 of my agents failed was because of bad state representations.Q-table and simple DQN: 5 features lost all spatial information - the agent couldn't tell WHERE armies were, just HOW MANY. 256-action CNN: Had the right state (full board), but wrong action space size. 5-action CNN: Best combination, but Expander was too strong for 400 episodes. If there is only garbage that gets collected then only garbage gets outputted.

I also learned that loss doesn't mean performance. Even if you have a low loss rate the agent might just be always expecting to lose every time. Like shooting a basketball hoop. Lets say loss is the difference between result and expectations, and you expect to miss and you do, even though you missed you have met your expectations.

"I also learned why experience replay is critical for DQN. Without it, training on sequential steps (N, N+1, N+2) means the network only sees very similar states and overfits to recent patterns, forgetting older lessons. Random sampling from past experiences breaks this temporal correlation and forces the network to learn general patterns instead of memorizing one specific game path.

Looking back, here's what I'd do differently:
1. Self-play instead of Expander - Train two copies of my agent against each other. They start equally bad and improve together, rather than one agent trying to beat a strong fixed opponent.
2. Curriculum learning - Start by beating RandomAgent first (easy), then a weak heuristic, then gradually harder opponents. Don't jump straight to Expander.
3. Simpler problem to validate - Test my approach on a 4×4 board first to make sure it works, then scale up to 8×8 once I know the method is sound.
4. Try policy gradients (PPO) - Policy-based methods like PPO are often more sample efficient than value-based methods like DQN for strategy games.
The real bottleneck wasn't compute or episodes - it was my training strategy. Even with 10,000 episodes, beating a strong heuristic baseline requires either massive amounts of data OR a smarter approach like self-play and curriculum learning.


	

