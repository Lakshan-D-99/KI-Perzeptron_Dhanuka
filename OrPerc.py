# This is an OR Gate implementation in Perception

# Activation function determines the class or category
def activation(summation, threshold):
    if summation > threshold:
        return 1
    else:
        return 0

def perceptron(or_input):
    # Training Data set for a OR Gate and the expected Outpur
    inputs = [[0, 0],[0, 1],[1, 0],[1, 1]]
    expected_output = [0, 1, 1, 1]

    # Initial  Weights, Threshold and the Learning Rate
    weights = [0.0, 0.3]
    threshold = 0.4
    learning_rate = 0.5

    print("Perceptron Training Start")

    i = 0
    while i < len(inputs):
        x = inputs[i]
        summation = x[0]*weights[0] + x[1]*weights[1]
        output = activation(summation, threshold)

        print(f"Input: {x}, Weights: {weights}, Summation: {summation}, Threshold: {threshold}")
        print(f"Expected: {expected_output[i]}, Predicted: {output}")

        # If our Prediction is wrong, then we have to update the Weights
        if output != expected_output[i]:
            print("Incorrect output. Updating weights...")
            weights[0] += learning_rate * (expected_output[i] - output) * x[0]
            weights[1] += learning_rate * (expected_output[i] - output) * x[1]
            print(f"Updated Weights: {weights}\n")
            # We have to start again to predict
            i = -1
        i += 1
        print("------------------------")

    # Prediction for new input
    summation = or_input[0]*weights[0] + or_input[1]*weights[1]
    final_output = activation(summation, threshold)
    return final_output

# Example prediction
or_input = [0, 0]
print(f"OR Gate Output for {or_input}: {perceptron(or_input)}")
