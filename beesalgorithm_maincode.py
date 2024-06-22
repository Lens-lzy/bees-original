import numpy as np
import math
# import random


print("Parameters")
d= 2 #"Please insert your dimension - minimum 2 dimension (try: 3):")
n = 50 #Population
m = 20 #Best sites
e = 8 #Elite sites
nep = 50
nsp = 25 
ngh = 0.1 #local search radius
MaxIt = 1000 #itereation

#Shpere Function to optimize
def Sphere(x):
    return np.sum(np.power(x, 2))

def Cost (x):
    return Sphere(x)

def Foraging(x,ngh):
    nVar = x.size
    k = np.random.randint(0, nVar)
    y = x
    y[0, k] = x[0, k] + np.random.uniform(-ngh, ngh)
    return y

class Bee:
    def __init__(self,Position,Cost):
        self.Position = Position
        self.Cost = Cost

# Setting Parameters of Bees Algorithm
varMin = float(-10)
varMax = float(10)
maxIt = MaxIt  # Maximum Number of Iterations
nScoutBee = n  # n = Number of Scout Bees
nSelectedSite = m  # m = Number of Selected Sites
nEliteSite = e  # e = Number of Selected Elite Sites
nSelectedSiteBee = nsp  # nsp = Number of Selected Recruited Bees for Selected (m-e) Sites
nEliteSiteBee = nep  # nep = Number of Recruited Bees for Elite Sites
shrink = 0.95

# Initialization
bee=[]
for i in range(nScoutBee):
    bee.append([])
    position = np.random.uniform(varMin,varMax,size=(1,d-1))
    bee[i]=Bee(position,Cost(position))

bee.sort(key=lambda bee: bee.Cost, reverse=False)

BestSol = ([],math.inf)
BestSol = bee[0]

BestCost = np.zeros([maxIt,1])
BestPos = []
for i in range(maxIt):
    BestPos.append([])

newbee=Bee([],[])






# Main Loop
for it in range(maxIt):
    
    # Elite Sites
    for i in range(nEliteSite):

        bestnewbee=Bee([],math.inf)

        for j in range(nEliteSiteBee):
            newbee.Position = Foraging(bee[i].Position,ngh)
            newbee.Cost = Cost(newbee.Position)
            if newbee.Cost < bestnewbee.Cost:
                bestnewbee.Cost = newbee.Cost
                bestnewbee.Position = newbee.Position

        if bestnewbee.Cost < bee[i].Cost:
            bee[i].Cost = bestnewbee.Cost
            bee[i].Position = bestnewbee.Position

    # Selected Non-Elite Sites
    for i in range(nEliteSite,nSelectedSite):

        bestnewbee=Bee([],math.inf)

        for j in range(nSelectedSiteBee):
            newbee.Position = Foraging(bee[i].Position,ngh)
            newbee.Cost = Cost(newbee.Position)
            if newbee.Cost < bestnewbee.Cost:
                bestnewbee.Cost = newbee.Cost
                bestnewbee.Position = newbee.Position

        if bestnewbee.Cost < bee[i].Cost:
            bee[i].Cost = bestnewbee.Cost
            bee[i].Position = bestnewbee.Position

    # Non - Selected Sites
    for i in range(nSelectedSite,nScoutBee):
        position = np.random.uniform(varMin,varMax,size=(1,d-1))
        bee[i]=Bee(position,Cost(position))


    # Sort
    bee.sort(key=lambda bee: bee.Cost, reverse=False)

    # Update
    BestSol.Cost = bee[0].Cost
    BestSol.Position = bee[0].Position

    # Store Best Cost ever found
    BestCost[it] = BestSol.Cost
    BestPos [it] = BestSol.Position
    # Display Iteration Information
    print(['Iteration ' + str(it) + ': Best Cost = ' + str(BestCost[it])+ ': Best Position = ' + str(BestPos[it])])

    ngh = shrink * ngh




## Results ##

import matplotlib.pyplot as plot

Y = BestCost
X = []
for i in range(len(Y)):
    X.append(i)

# Display grid

plot.grid(True, which="both")
# Linear X axis, Logarithmic Y axis

plot.semilogy(X, Y)
plot.ylim([0, 10])
plot.xlim([0, X.__len__()])
# Provide the title for the semilog plot
plot.title('Sphere Function')
plot.xlabel('Iteration')
plot.ylabel('Best Cost')
plot.show()
