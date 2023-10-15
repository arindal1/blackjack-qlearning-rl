# AI Blackjack w/ Q-learning

This project implements a Q-learning-based reinforcement learning solution to play the popular casino card game, Blackjack. The implementation consists of two main components: `BlackjackQLearning`, representing the Q-learning agent, and `BlackjackGame`, modeling the game of Blackjack.

## Introduction

The Q-learning agent (`BlackjackQLearning`) employs Q-tables to learn an optimal strategy for playing Blackjack. It uses an ε-greedy strategy for action selection during training and updates the Q-table based on rewards obtained during gameplay. The Blackjack game (`BlackjackGame`) simulates the actions of the player and dealer, allowing the Q-learning agent to learn and improve its strategy through interactions.

## Project Structure

The project structure is as follows:
- `qlearning.py`: Contains the Q-learning agent implementation.
- `blackjack.py`: Contains the Blackjack game implementation.
- `main.py`: Entry point for training the Q-learning agent and evaluating its performance.
- `ABOUT_THE_CODE.md`: Markdown that explains the code.

## How to Run

To run the project and train the Q-learning agent, follow these steps:

1. Ensure you have Python installed on your machine.

2. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/blackjack-q-learning.git
   cd blackjack-q-learning
   ```

3. Run the training script:
   ```bash
   python main.py
   ```

4. The agent will be trained for a specified number of episodes, and test games will be played to evaluate its performance.

***Note:*** You can refer to [ABOUT_THE_CODE](ABOUT_THE_CODE.md) for a line-by-line breakdown.

## Configuration and Parameters

You can modify the training parameters and other configurations by editing the constants and variables in `main.py` and the relevant parts of `qlearning.py`.

- `alpha`: Learning rate for the Q-learning algorithm.
- `gamma`: Discount factor for future rewards.
- `epsilon`: Exploration probability for the ε-greedy strategy.
- `episodes`: Number of training episodes for the Q-learning agent.
- `test_games`: Number of test games to evaluate the agent's performance.

## Results

After training and testing, the project will display the win rate of the Q-learning agent in playing Blackjack.

## Contributing

Feel free to contribute to this project by opening issues or creating pull requests. Contributions are welcome and encouraged!

## Contact

If you have any questions or suggestions related to this project, you can reach out to me at:

- GitHub: [arindal1](https://github.com/arindal1)
- LinkedIn: [arindalchar](https://www.linkedin.com/arindalchar/)
- Twitter: [arindal_17](https://twitter.com/arindal_17)

---

## Keep creating 🚀
