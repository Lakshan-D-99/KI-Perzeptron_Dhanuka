# Implement NOT Gate using a Perceptron

# Again the Activation Function will check for the class
def activation(summation, threshold):
    if summation >= threshold:
        return 1
    else:
        return 0

def perceptron(not_input):
    # Define Training Set and the expected Output
    inputs = [0, 1]
    expected_output = [1, 0]

    # Initial weight, threshold, and learning rate
    weight = 0
    threshold = 0
    learning_rate = 0.5

    print("NOT Gate Perceptron Training Start")

    i = 0
    while i < len(inputs):
        x = inputs[i]
        summation = x * weight
        output = activation(summation, threshold)

        print(f"Input: {x}, Weight: {weight}, Summation: {summation}, Threshold: {threshold}")
        print(f"Expected: {expected_output[i]}, Predicted: {output}")

        # If our Prediction is wrong, then we have to update the Weights
        if output != expected_output[i]:
            print("Incorrect output. Updating weight...")
            weight += learning_rate * (expected_output[i] - output) * x
            print(f"Updated Weight: {weight}\n")
            # We have to start again to predict
            i = -1
        i += 1
        print("------------------------")

    # Prediction for new input
    summation = not_input * weight
    final_output = activation(summation, threshold)
    return final_output

# Example prediction
not_input = 1
print(f"NOT Gate Output for {not_input}: {perceptron(not_input)}")
