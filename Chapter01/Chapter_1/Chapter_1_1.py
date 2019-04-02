inputs = [1,2]
weights = [1,1,1]

def perceptron_predict(inputs, weights):
    activation = weights[0]    
    for i in range(len(inputs)-1):
      activation += weights[i+1] * inputs[i]
      return 1.0 if activation >= 0.0 else 0.0

print(perceptron_predict(inputs,weights))
