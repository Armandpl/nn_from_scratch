import numpy as np

inputs = [np.matrix('0;0'), np.matrix('0;1'), np.matrix('1;0'), np.matrix('1;1')]
outputs = np.array([0,1,1,0])

layers = []

layers.append(np.random.random((4,2)))

layers.append(np.random.random((4, 4)))

layers.append(np.random.random((1,4)))

print("layers:")
for i in range(3):
    print(layers[i])
    print("")

##forward :
print("input :")

print(inputs[1])

#first
output_first = layers[0].dot(inputs[1])
# RELU
output_first = np.maximum(output_first, 0)

print("first layer output")
print(output_first)

#second
output_second = layers[1].dot(output_first)
np.maximum(output_second, 0)

print("second layer output")
print(output_second)

#third
output_third = layers[2].dot(output_second)
np.maximum(output_third,0)

print("output :")
print(output_third)
