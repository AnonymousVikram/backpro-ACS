import numpy as np
import random

dataSet = np.empty((0, 3), int)

# Loop to generate Training Data
for i in range(1000):
    input1 = random.randint(0, 1)
    input2 = random.randint(0, 1)
    expectedVal = input1 or input2
    dataSet = np.append(dataSet, np.array([[input1, input2, expectedVal]]), axis=0)

'''
    The array "dataFrame" will have three columns:
      1         2            3
    Input1   Input2    Expected Value
'''


def neuralNetworkPrediction(inputVal1, inputVal2, weightVal1, weightVal2, biasVal):
    # 2x1 array of Inputs
    inputLayer = np.array([[inputVal1],
                           [inputVal2]])

    # 1x2 array of Weights
    weights = np.array([[weightVal1, weightVal2]])

    # Matrix Operations
    output = weights @ inputLayer + biasVal

    # Sigmoid Function
    output = 1 / (1 + np.exp(output))
    return output


def machineLearning(inputVal1, inputVal2, weightVal1, weightVal2, biasVal, predictVal, expectVal):
    # Derivative of the Cost Function with respect to the Predicted Value
    derCostPred = (expectVal - predictVal) / (predictVal - predictVal ** 2)

    # Just a variable to make the next computations look cleaner
    eNumber = np.exp(-(weightVal1 * inputVal1) - (weightVal2 * inputVal2) - biasVal)

    # Derivative of the Predicted Value with respect to Weight1
    derPredWeight1 = (inputVal1 * eNumber) / ((eNumber + 1) ** 2)

    # Derivative of the Predicted Value with respect to Weight2
    derPredWeight2 = (inputVal2 * eNumber) / ((eNumber + 1) ** 2)

    # Derivative of the Predicted Value with respect to Cost
    derPredBias = eNumber / ((eNumber + 1) ** 2)

    # Gradient Descent Array
    gradDescCost = np.array([[derCostPred * derPredWeight1],
                             [derCostPred * derPredWeight2],
                             [derCostPred * derPredBias]])

    # Inputted Weights and Biases
    weightBias = np.array([[weightVal1],
                           [weightVal2],
                           [biasVal]])

    # Next round of Weights and Biases
    outputVals = weightBias - (0.5 * gradDescCost)

    return outputVals


# Random Initial values
print("Weights and Biases at the beginning:")
weight1 = 0
weight2 = 0
bias = 0
print("Weight 1:", weight1, "\n" + "Weight 2:", weight2, "\n" + "Bias:", bias, "\n", )


# Algorithm Training using Data produced Earlier
print("Training the Algorithm...")
for i in range(1000):
    outputPrediction = neuralNetworkPrediction(dataSet[i - 1][0], dataSet[i - 1][1], weight1, weight2, bias)[0][0]
    updatedVals = machineLearning(dataSet[i - 1][0], dataSet[i - 1][1], weight1, weight2, bias, outputPrediction,
                                  dataSet[i - 1][2])
    weight1 = updatedVals[0][0]
    weight2 = updatedVals[1][0]
    bias = updatedVals[2][0]

print("Training Completed!\n")
print("Weights and Biases after Training:")
print("Weight 1:", weight1, "\n" + "Weight 2:", weight2, "\n" + "Bias:", bias, "\n", )

# User Interface (includes rounding)
again = True
while again:
    input1 = int(input("Input #1:"))
    input2 = int(input("Input #2:"))
    dataSet = np.append(dataSet, np.array([[input1, input2, input1 or input2]]), axis=0)

    out = neuralNetworkPrediction(input1, input2, weight1, weight2, bias)[0][0]
    updatedVals = machineLearning(input1, input2, weight1, weight2, bias, out, input1 or input2)

    weight1 = updatedVals[0][0]
    weight2 = updatedVals[1][0]
    bias = updatedVals[2][0]

    prob = 0
    if round(out) == 1:
        prob = out
    else:
        prob = 1-out

    prob = round(prob, 3)
    prob *= 100
    print("With", str(prob) + "% probabiliy,", input1, "or", input2, "is", round(out))

    againIn = input("Would you like to run this again?")
    againIn = againIn.lower()
    if againIn == 'yes' or againIn == 'true' or againIn == 1:
        again = True
    else:
        again = False
