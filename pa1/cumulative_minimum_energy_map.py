import numpy as numpy

def getElement(r ,c , energyImage):
    rows, cols = energyImage.shape
    if(r < 0) or (r >= rows) or (c < 0) or (c >= cols):
        return float("inf")
    else:
        return energyImage[r][c]

def cumulative_minimum_energy_map(energyImage, seamDirection):
    rows, cols = energyImage.shape
    cum_cost = np.empty_like(energyImage)
    if(seamDirection == 'VERTICAL'):
        cum_cost[0, :] = energyImage[0, :]
        for r in range(1, rows):
            for c in range(cols):
                cum_cost[r][c] = energyImage[r][c] + min(getElement(r-1, c-1, energyImage),getElement(r-1, c, energyImage), getElement(r-1, c+1, energyImage))
    elif(seamDirection == 'HORIZONTAL'):
        cum_cost[:,0] = energyImage[:,0]
        for c in range(1, cols):
            for r in range(rows):
                cum_cost[r][c] = energyImage[r][c] + min(getElement(r, c-1, energyImage),getElement(r-1,c-1,energyImage),getElement(r+1, c-1, energyImage))
    else:
        print "Blunder"
    
    return cum_cost
