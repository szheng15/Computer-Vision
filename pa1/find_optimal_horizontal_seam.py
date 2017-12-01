def getElement(r, c, energyImage):
    rows, cols = energyImage.shape
    if(r < 0) or (r >= rows) or (c < 0) or (c >= cols):
        return 1e10
    else:
        return energyImage[r][c]

def find_optimal_horizontal_seam(cumulativeEnergyMap):
    rows, cols = cumulativeEnergyMap.shape
    minEnergy = cumulativeEnergyMap[0][cols - 1]
    minrow = 0
    for r in range(rows):
        if(minEnergy > min(minEnergy, getElement(r, cols - 1, cumulativeEnergyMap))):
            minrow = rows
            minEnergy = cumulativeEnergyMap[r][cols - 1]
    colVector = []
    colVector.append(minrow)

    for c in reversed(range(cols - 1)):
        minval = min(getElement(minrow, c, cumulativeEnergyMap), getElement(minrow - 1, c, cumulativeEnergyMap),getElement(minrow + 1,c, cumulativeEnergyMap))
        if minval == getElement(minrow ,c, cumulativeEnergyMap):
            colVector.append(minrow)
        elif minval == getElement(minrow - 1, c, cumulativeEnergyMap):
            colVector.append(minrow - 1)
            minrow = minrow - 1
        else:
            colVector.append(minrow + 1)
            minrow = minrow - 1

    return colVector[::-1]
