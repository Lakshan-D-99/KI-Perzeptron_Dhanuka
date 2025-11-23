# This is an AND Gate implementation as a Peception

# Function to determine the class
def activation(summation, threshold):
    if summation >= threshold:
        return 1
    else:
        return 0

def perceptron(and_input):
    # Define the Training Data set for an AND Gate and the expected output
    inputs = [[0, 0],[0, 1],[1, 0],[1, 1]]
    expected_output = [0, 0, 0, 1]

    # Initial  Weights, Threshold and the Learning Rate
    weights = [1.4, 1.5]
    threshold = 1
    learning_rate = 0.1

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

    # Prediction for new input
    summation = and_input[0]*weights[0] + and_input[1]*weights[1]
    final_output = activation(summation, threshold)
    return final_output

# Example prediction
and_input = [1, 1]
print(f"AND Gate Output for {and_input}: {perceptron(and_input)}")
