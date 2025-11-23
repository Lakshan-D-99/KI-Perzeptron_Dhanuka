import numpy as np
import matplotlib.pyplot as plt

# Step 1: Generate 10 Random values between -1 and 1
m = 10
X = np.random.uniform(-1,1,(m,2))

# We need to choose two Points to define the Linear Function. Based on this Linear Function, we can decide which data
# is on the Positive Side and the Negative Side
pointOne = np.random.uniform(-1,1,2)
pointTwo = np.random.uniform(-1,1,2)

# The following Function will check on which side the Point is
def check_point(x,pOne,pTwo):
    value = (pTwo[1] - pOne[1]) * x[0] - (pTwo[0] - pOne[0]) * x[1] + (pTwo[0] * pOne[1] - pTwo[1] * pOne[0])
    if value > 0:
        return 1
    else:
        return -1

# Now we have the expected Outputs for our Inputs
y = np.array([check_point(x, pointOne, pointTwo) for x in X])

# Now we can define the Perception Algorithem for Prediction
def perceptron_train(X, y, alpha=1):
    # m and n are the values
    m, n = X.shape
    # At the Start the Weight should be 0
    weight = np.zeros(n + 1)
    X_bias = np.hstack((np.ones((m, 1)), X))

    steps = 0
    while True:
        h = np.sign(X_bias @ weight)
        h[h == 0] = -1

        wrong_points = np.where(h != y)[0]
        if len(wrong_points) == 0:
            break

        # Pick a random Wrong value to perform the Train until there are no Wrong Predictions
        i = np.random.choice(wrong_points)
        weight += alpha * y[i] * X_bias[i]
        steps += 1

    return weight, steps

# We want to run our Perception Train func for 1000 times and calculate the average Steps
steps_list = []
for _ in range(1000):
    _, steps = perceptron_train(X, y,1)
    steps_list.append(steps)

avg_steps = np.mean(steps_list)
print(f"Average Steps:{avg_steps}")

# Lastly, we have to perform an experiment where the amount of Points are 100 and 1000 and the Alpha values are 1 and 0.1
def experiment(amount_points,alpha,runs=1000):
    steps_list = []

    for _ in range(runs):
        X = np.random.uniform(-1, 1, (m, 2))
        pointOne = np.random.uniform(-1, 1, 2)
        pointTwo = np.random.uniform(-1, 1, 2)
        y = np.array([check_point(x, pointOne, pointTwo) for x in X])

        _, steps = perceptron_train(X, y,alpha)
        steps_list.append(steps)

    avg_step = np.mean(steps_list)
    print(f"m={amount_points}, alpha={alpha} -> Average Steps: {avg_step:.2f}")
    return avg_step

# Now we can perform some tests
# m = 100 and Alpha = 1 und Alpha = 0.1
experiment(100,1)
experiment(100,0.1)

# m = 1000 and Alpha = 1 and Alpha = 0.1
experiment(1000,1)
experiment(1000,0.1)


