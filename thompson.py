# Initialize parameters for Beta distributions
import main
alpha = [1] * num_slot_machines  # Number of wins
beta = [1] * num_slot_machines  # Number of losses

# Number of rounds to play
num_rounds = 1000

# Thompson Sampling loop
for round in range(num_rounds):
    # Sample from the Beta distributions for each slot machine
    sampled_probs = [random.beta(alpha[i], beta[i]) for i in range(num_slot_machines)]

    # Choose the slot machine with the highest sampled value
    chosen_machine = sampled_probs.index(max(sampled_probs))

    # Simulate pulling the chosen machine and observing the reward
    reward = simulate_pull(chosen_machine)

    # Update the parameters of the Beta distribution based on the reward
    if reward == 1:
        alpha[chosen_machine] += 1
    else:
        beta[chosen_machine] += 1

# After the loop, the slot machine with the highest alpha parameter is the "best" machine to pull.
