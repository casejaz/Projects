#!/usr/bin/python
import random
import math
from random import shuffle
import copy

class ActivationFunction:
    """ ActivationFunction packages a function together with its derivative. """
    """ This prevents getting the wrong derivative for a given function.     """
    """ Because some derivatives are computable from the function's value,   """
    """ the derivative has two arguments: one for the argument and one for   """
    """ the value of the corresponding function. Typically only one is use.  """

    def __init__(af, name, fun, deriv):
        af.name = name
        af.fun = fun
        af.deriv = deriv

    def fun(af, x):
        return af.fun(x)

    def deriv(af, x, y):
        return af.deriv(x, y)

logsig = ActivationFunction("logsig",
                            lambda x: 1.0/(1.0 + math.exp(-x)),
                            lambda x,y: y*(1.0-y))

tansig = ActivationFunction("tansig",
                            lambda x: math.tanh(x),
                            lambda x,y: 1.0 - y*y)

purelin = ActivationFunction("purelin",
                             lambda x: x,
                             lambda x,y: 1)

def randomWeight():
    """ returns a random weight value between -0.5 and 0.5 """
    return random.random()-0.5

def inner(x, y):
    """ Returns the inner product of two equal-length vectors. """
    n = len(x)
    assert len(y) == n
    sum = 0
    for i in range(0, n):
        sum += x[i]*y[i]
    return sum

def subtract(x, y):
    """ Returns the first vector minus the second. """
    n = len(x)
    assert len(y) == n
    return map(lambda i: x[i]-y[i], range(0, n))

def countWrong(L, tolerance):
    """ Returns the number of elements of L with an absolute """
    """ value above the specified tolerance. """
    return reduce(lambda x,y:x+y, \
                  map(lambda x:1 if abs(x)>tolerance else 0, L))

def roundall(item, n):
    """ Round a list, list of lists, etc. to n decimal places. """
    if type(item) is list:
        return map(lambda x:roundall(x, n), item)
    return round(item, n)

class rtrl:
    def __init__(nn, numNeurons, neuronType, rate, pattern, numSlots, runLength):
        """
        numNeurons: The total number of neurons to be used.
        neuronType: The type of neuron to use, such as logsig, tansig, etc.
        learningRate: A real number indicating the learning rate for the network.
        pattern: A list of lists of slot numbers, with each list being a pattern for one percussion instrument.
        numSlots: The number of time steps in a repetition cycle.
        runLength: The total number of steps for which to run training and playing.
        """
        nn.numNeurons = numNeurons

        nn.neuronType = neuronType

        nn.rate = rate

        nn.pattern = pattern

        nn.numSlots = numSlots

        nn.runLength = runLength



        nn.output = [0 for neuro_m in range(nn.numNeurons)]


        nn.timer = 3
        # first m are external input values, following n are neuron outputs
        nn.weight = [[randomWeight() for output in range(1 + nn.numNeurons)]
                                for neuron in range(nn.numNeurons)]
        # initialize first m neurons


        nn.bias = [randomWeight() for neuron_j in range(nn.numNeurons)]
        #nn.sensitivity = [0 for neuron_k in range(nn.numNeurons)]
        nn.act = [0 for neuron_l in range(nn.numNeurons)]
        nn.error = [0 for neuron_m in range(nn.numNeurons)]

        nn.X = [0 for idx in range(nn.numSlots)]
        nn.X[0] = 1
        nn.Z = [0 for i in range(nn.numNeurons + 1)]
        nn.Z[0] = nn.X[0]
        nn.X1 = [[0 for instrument in range(len(pattern))] for time in range(nn.numSlots)]
        for time_i in range(nn.numSlots):
            for instru_i in range(len(pattern)):
                if time_i in pattern[instru_i]:
                    nn.X1[time_i][instru_i] = 1


    def describe(nn, noisy):
        """ describe prints a description of this network. """
        print "---------------------------------------------------------------"
        print "size =", nn.numNeurons
        print "function =", map(lambda x:x.name, [nn.neuronType])
        print "learning rate =", nn.rate
        if noisy:
            print "weight =", roundall(nn.weight, 3)
            print "bias =", roundall(nn.bias, 3)

    def forward(nn, t):
        nn.Z[0] = nn.X[t%nn.numSlots]
        fun = nn.neuronType.fun

        for neuron in range(nn.numNeurons):
            # compute and save the activations
            nn.act[neuron] = nn.bias[neuron] + inner(nn.weight[neuron], nn.Z)
            # compute the output
            nn.output[neuron] = fun(nn.act[neuron])
        J = 0
        wrong = 0
        hits = 0
        for instru in range(len(nn.pattern)):
            # TODO: check time of output later
            desired = nn.X1[t][instru]
            if nn.output[instru] > 0.5:
                output = 1
                hits += 1
            else:
                output = 0
            error = nn.X1[t][instru] - nn.output[instru]
            nn.error[instru] = error
            J += (error ** 2)
            if output != desired:
                wrong +=  1
        for idx in range(1,len(nn.Z)):
            nn.Z[idx] = nn.output[idx-1]
        return J, wrong, hits, nn.output[0]

    def train(nn, displayInterval, noisy):

        ptL = [[[0 for j in range(1 + nn.numNeurons)]
                    for i in range(nn.numNeurons)]
                    for k in range(nn.numNeurons)]
        for step in range(0, nn.runLength):
            if step % nn.numSlots == 0:
                MSE = 0
            if step % displayInterval == 0:
                print "Step {}".format(step)


            result = nn.forward(step%nn.numSlots)
            errorsq = result[0]
            wrong = result[1]
            hits = result[2]
            output = result[3]
            slot_MSE = errorsq / len(nn.pattern)
            MSE += slot_MSE


            if step % displayInterval == 0:
                print "slot= {} click= {} MSE= {} hit= {} wrong={} output={}".format(step%nn.numSlots, nn.X[step%nn.numSlots], round(slot_MSE,3), hits, wrong, round(output,3))
                if noisy:
                    print "desired " # figure this out later
            ptL_pre = copy.copy(ptL)
            for i in range(nn.numNeurons):
                for j in range(1+nn.numNeurons):
                    sumK = 0
                    for k in range(nn.numNeurons):
                        sumL = 0
                        for l in range(nn.numNeurons):
                            sumL += nn.weight[k][1+l] * ptL_pre[l][i][j]
                            # sensitivity z(t) term
                        if i == k:
                            sumL += nn.Z[j]
                        deriv =  nn.neuronType.deriv(nn.act[k],nn.output[k])
                        ptL[k][i][j] = deriv * sumL
                        sumK += nn.error[k] * ptL_pre[k][i][j]
                    nn.weight[i][j] += (nn.rate * sumK )
        return
def main():
    rtrlnet = rtrl(10,logsig, 0.3,[[1,2,3]],4,10000)
    rtrlnet.describe(True)
    rtrlnet.train(1, False)



main()
