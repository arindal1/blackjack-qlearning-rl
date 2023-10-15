[qlearning.py](#qlearning.py) || [blackjack.py](#blackjack.py) || [main.py](#main.py)

# qlearning.py

```python
import numpy as np
from blackjack import BlackjackGame
```
- This imports the necessary libraries. `numpy` is used for numerical operations, and `BlackjackGame` is a custom class representing the Blackjack game.

```python
class BlackjackQLearning:
    def __init__(self, alpha=0.1, gamma=0.9, epsilon=0.1):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

        # Initialize Q-table
        self.Q = np.zeros((33, 12, 2, 2))  # Player sum, dealer card, action
```
- This defines a class called `BlackjackQLearning` representing the Q-learning agent for playing Blackjack.
- The `__init__` method initializes the Q-learning agent with given parameters like `alpha`, `gamma`, and `epsilon`. It also creates a Q-table to store the Q-values for different states (player sum, dealer card, usable ace, and action).

```python
    def choose_action(self, player_sum, dealer_card, usable_ace):
        if np.random.uniform(0, 1) < self.epsilon:
            return np.random.choice(["hit", "stay"])
        else:
            return "hit" if self.Q[player_sum, dealer_card, usable_ace, 0] > self.Q[player_sum, dealer_card, usable_ace, 1] else "stay"
```
- This method chooses an action for the agent based on the ε-greedy policy. With probability `epsilon`, it chooses a random action (`"hit"` or `"stay"`), otherwise, it selects the action (`"hit"` or `"stay"`) that maximizes the Q-value.

```python
    def update(self, player_sum, dealer_card, usable_ace, action, reward, new_player_sum, new_dealer_card, new_usable_ace):
        action_idx = 0 if action == "hit" else 1
        old_value = self.Q[player_sum, dealer_card, usable_ace, action_idx]
        future_max = np.max(self.Q[new_player_sum, new_dealer_card, new_usable_ace])
        self.Q[player_sum, dealer_card, usable_ace, action_idx] = old_value + self.alpha * (reward + self.gamma * future_max - old_value)
```
- This method updates the Q-value of the chosen action based on the Q-learning update rule using the provided reward and the maximum Q-value for the next state.

```python
    @staticmethod
    def has_usable_ace(hand):
        """Check if the hand has a usable ace."""
        value, ace = 0, False
        for card in hand:
            card_number = card['number']
            value += min(10, int(card_number) if card_number not in ['J', 'Q', 'K', 'A'] else 11)
            ace |= (card_number == 'A')
        return int(ace and value + 10 <= 21)
```
- This is a static method that checks if a hand has a usable ace. It iterates through the cards in the hand, keeping track of the hand value and whether an ace is present.

```python
    def train(self, episodes):
        one_percent = round(episodes / 100)

        for ep in range(episodes):
            game = BlackjackGame()
            game.start_game()

            if ep % one_percent == 0:
                progress = (ep/episodes) * 100
                print(f"Training progress: {progress:.2f}%")
```
- This method trains the Q-learning agent through a specified number of episodes (games).
- It iterates through episodes, creating a new Blackjack game for each episode.

```python
            dealer_card = int(game.dealer_hand[0]['number']) if game.dealer_hand[0]['number'] not in ['J', 'Q', 'K', 'A'] else (10 if game.dealer_hand[0]['number'] != 'A' else 11)
            status = "continue"

            while status == "continue":
                player_sum = game.hand_value(game.player_hand)
                usable_ace = self.has_usable_ace(game.player_hand)
                action = self.choose_action(player_sum, dealer_card, usable_ace)
                status = game.player_action(action)
                new_player_sum = game.hand_value(game.player_hand)
                new_usable_ace = self.has_usable_ace(game.player_hand)

                reward = 0  # Intermediate reward, only final matters

                if status == "player_blackjack":
                    reward = 1
                elif status == "player_bust":
                    reward = -1
```
- Within each episode, the agent interacts with the Blackjack game environment.
- It gets the dealer's face-up card and the current game status.
- It iteratively selects actions based on the Q-learning policy, observes rewards and new states, and updates the Q-values accordingly.

```python
                if reward != 0:
                    self.update(player_sum, dealer_card, usable_ace, action, reward, new_player_sum, dealer_card, new_usable_ace)

                if action == "stay":
                    break

            final_result = game.game_result()
            final_reward = 1 if final_result == "win" else (-1 if final_result == "loss" else 0)
            self.update(player_sum, dealer_card, usable_ace, action, final_reward, new_player_sum, dealer_card, new_usable_ace)
```
- The agent updates the Q-values based on the final outcome of the game for the episode.

```python
    def play(self):
        game = BlackjackGame()
        game.start_game()

        print("Dealer shows:", game.format_cards(game.dealer_hand[:1]))

        status = "continue"
        print(game.format_cards(game.player_hand), game.hand_value(game.player_hand))
        while status == "continue":
            player_sum = game.hand_value(game.player_hand)
            usable_ace = self.has_usable_ace(game.player_hand)
            dealer_card = int(game.dealer_hand[0]['number']) if game.dealer_hand[0]['number'] not in ['J', 'Q', 'K', 'A'] else (10 if game.dealer_hand[0]['number'] != 'A' else 11)
            action = "hit" if self.Q[player_sum, dealer_card, usable_ace, 0] > self.Q[player_sum, dealer_card, usable_ace, 1] else "stay"
            status = game.player_action(action)
            
            if action == "stay":
                break
                
            print(game.format_cards(game.player_hand), game.hand_value(game.player_hand))
```
- This method allows the agent to play a game of Blackjack after being trained.
- It interacts with the game environment based on the Q-learning policy and prints the game progression.

```python
        if status == "continue":
            print("Dealer has:", game.format_cards(game.dealer_hand), game.hand_value(game.dealer_hand))
            game.dealer_action()

        final_result = game.game_result()
        return final_result
```
- After the agent completes its turn, it allows the dealer to play and calculates the final result of the game.

```python


# Train the agent
agent = BlackjackQLearning()
agent.train(5000000)
```
- This section creates the `BlackjackQLearning` agent and trains it over a specified number of episodes.

```python
test_games = 1000000
wins = 0
losses = 0
draws = 0

for _ in range(test_games):
    print("-----")
    result = agent.play()
    print(result)
    if result == "win":
        wins += 1
    elif result == "loss":
        losses += 1
    else:
        draws += 1
```
- This section tests the trained agent by playing a number of test games and recording the results.

```python
print(f"Wins: {wins}, Losses: {losses}, Draws: {draws}")
print(f"Win rate: {wins/(wins + losses)*100:.2f}%")
```
- Finally, the code prints the results of the test games, including the number of wins, losses, draws, and the win rate.

---

# blackjack.py

This code defines a Blackjack game simulator with a simple console interface where the player can interact with the game by choosing actions such as "hit" or "stay." Here's a detailed explanation of each part of the code:

```python
import random
```
- This imports the `random` module for generating random numbers, which is used to shuffle the deck.

```python
class BlackjackGame:
    # Constants representing card suits using unicode characters
    heart = "\u2665"
    spade = "\u2660"
    diamond = "\u2666"
    club = "\u2663"

    suits = {
        "diamonds": diamond,
        "hearts": heart,
        "spades": spade,
        "clubs": club
    }
```
- This defines a class called `BlackjackGame` to simulate the Blackjack game.
- Constants for card suits are defined using Unicode characters.
- A dictionary `suits` is created to map suit names to their corresponding Unicode symbols.

```python
    def __init__(self):
        self.deck = self.generate_deck()
        random.shuffle(self.deck)
        self.player_hand = []
        self.dealer_hand = []
```
- The `__init__` method initializes the `BlackjackGame` object.
- It generates a deck of cards using the `generate_deck` method, shuffles the deck, and initializes empty hands for the player and dealer.

```python
    @staticmethod
    def generate_deck():
        numbers = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']
        suits = ['hearts', 'diamonds', 'clubs', 'spades']
        deck = [{'number': number, 'suit': suit} for number in numbers for suit in suits]
        return deck
```
- This static method generates a standard deck of 52 cards, represented as a list of dictionaries where each dictionary represents a card.

```python
    def deal_card(self):
        return self.deck.pop()
```
- The `deal_card` method simulates dealing a card from the deck.

```python
    def start_game(self):
        self.player_hand = [self.deal_card(), self.deal_card()]
        self.dealer_hand = [self.deal_card(), self.deal_card()]
```
- The `start_game` method initializes the game by dealing two cards to the player and two cards to the dealer.

```python
    @staticmethod
    def hand_value(hand):
        # Calculate the value of the hand (considering aces as 1 or 11)
        value = 0
        aces = 0
        for card in hand:
            if card['number'] in ['J', 'Q', 'K']:
                value += 10
            elif card['number'] == 'A':
                value += 11
                aces += 1
            else:
                value += int(card['number'])

        while value > 21 and aces:
            value -= 10
            aces -= 1
        return value
```
- The `hand_value` method calculates the total value of a hand, considering the possibility of aces being worth either 1 or 11 points.

```python
    def player_action(self, action):
        if action == "hit":
            self.player_hand.append(self.deal_card())
        return self.game_status()
```
- The `player_action` method allows the player to either "hit" (take another card) or "stay" (end their turn). If the action is "hit," a card is added to the player's hand, and the game status is checked.

```python
    def dealer_action(self, output=False):
        while self.hand_value(self.dealer_hand) < 17:
            self.dealer_hand.append(self.deal_card())
            if output:
                print("Dealer hits and has:", self.format_cards(self.dealer_hand), self.hand_value(self.dealer_hand))
```
- The `dealer_action` method simulates the dealer's turn, where the dealer continues to draw cards until their hand value is 17 or higher.

```python
    def game_status(self):
        player_value = self.hand_value(self.player_hand)
        if player_value > 21:
            return "player_bust"
        elif player_value == 21:
            return "player_blackjack"
        else:
            return "continue"
```
- The `game_status` method checks the current game status, such as if the player has bust or achieved Blackjack.

```python
    def game_result(self):
        self.dealer_action()
        player_value = self.hand_value(self.player_hand)
        dealer_value = self.hand_value(self.dealer_hand)

        if player_value > 21:
            return "loss"
        elif dealer_value > 21 or player_value > dealer_value:
            return "win"
        elif player_value == dealer_value:
            return "draw"
        else:
            return "loss"
```
- The `game_result` method calculates the final result of the game (win, loss, or draw) by comparing the values of the player's and dealer's hands.

```python
    @staticmethod
    def format_cards(cards):
        result = ""
        for card in cards:
            suit = BlackjackGame.suits[card["suit"]]
            result += f"{card['number']}{suit} "
        
        return result.strip()
```
- The `format_cards` method formats the cards in a hand with their numbers and Unicode suit symbols.

```python
game = BlackjackGame()
game.start_game()
```
- An instance of the `BlackjackGame` class is created, and a new game is started.

```python
def main():
    print("Dealer shows:", game.format_cards(game.dealer_hand[:1]))

    status = "continue"
    while status == "continue":
        print(game.format_cards(game.player_hand), game.hand_value(game.player_hand))
        action

 = input("Enter an action (hit/stay): ")
        status = game.player_action(action)
        
        if action == "stay":
            break

    if status == "continue":
        print("Dealer has:", game.format_cards(game.dealer_hand), game.hand_value(game.dealer_hand))
        game.dealer_action()

    print(game.game_result())
```
- The `main` function runs the main logic of the game, displaying the initial state, handling player actions, and determining the game result.

```python
if __name__ == "__main__":
    main()
```
- Finally, the `main` function is called when the script is run, starting the Blackjack game.

---

# main.py

This script combines the previously defined `BlackjackGame` and `BlackjackQLearning` classes to train a Q-learning agent to play Blackjack and test its performance. Let's go through each part step by step:

```python
from blackjack import BlackjackGame
from qlearning import BlackjackQLearning
```
- This imports the `BlackjackGame` class from the `blackjack` module and the `BlackjackQLearning` class from the `qlearning` module.

```python
def main():
    # Train the agent
    agent = BlackjackQLearning()
    episodes = 5000000  # Number of training episodes
    agent.train(episodes)
```
- The `main` function is defined, which is the entry point for the script.
- An instance of the `BlackjackQLearning` class is created as the agent.
- The number of training episodes is set to 5,000,000, and the `train` method of the agent is called to train the agent using Q-learning.

```python
    # Test the agent's performance
    test_games = 1000000
    wins = 0
    losses = 0
    draws = 0
```
- The script sets the number of test games to 1,000,000 and initializes counters for wins, losses, and draws.

```python
    for _ in range(test_games):
        print("-----")
        result = agent.play()
        print(result)
        if result == "win":
            wins += 1
        elif result == "loss":
            losses += 1
        else:
            draws += 1
```
- A loop is executed for the specified number of test games.
- For each game, the `play` method of the agent is called, simulating a game and getting the result.
- The result (win, loss, or draw) is printed, and the win, loss, or draw counters are updated accordingly.

```python
    print(f"Wins: {wins}, Losses: {losses}, Draws: {draws}")
    print(f"Win rate: {wins/(wins + losses)*100:.2f}%")
```
- Finally, the script prints the total number of wins, losses, and draws, as well as the win rate calculated as a percentage.

```python
if __name__ == "__main__":
    main()
```
- This conditional block ensures that the `main` function is called when the script is run.


---

**Return to the [TOP](#qlearning.py).**
