import random
import statistics

numbers = [0, 1, 2, 3, 4, 5]
probabilities = [0.15, 0.2, 0.3, 0.2, 0.1, 0.05]

times = random.choices(numbers, weights=probabilities, k=100)

for i, draw in enumerate(times, start=1):
    print(f"Time {i}: {draw}")

average = statistics.mean(times)
std_dev = statistics.stdev(times)

print("\nAverage:", average)
print("Std:", std_dev)