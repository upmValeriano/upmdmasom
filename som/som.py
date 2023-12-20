import numpy as np

class som:

    """
    Self Organizing Map
    input:
    - nrows: number of rows
    - ncols: number of columns
    - dimension: dimension of the neurons
    - vmin: minimum value of the neurons
    - vmax: maximum value of the neurons
    """

    def __init__(self,
                 nrows = 5,
                 ncols = 5,
                 dimension = 3,
                 vmin =  0,
                 vmax =  1,
                 vicinity = "rectangular"):
        self.nrows = nrows
        self.ncols = ncols
        self.dimension = dimension
        self.vmin = vmin
        self.vmax = vmax
        self.vicinity = vicinity
        self._createSOM()
        print('clase iniciada')

    def _createSOM(self):
        SOM = np.random.rand(self.nrows,
                             self.ncols,
                             self.dimension)
        SOM = SOM * (self.vmax-self.vmin)+self.vmin
        self.SOM = SOM

    #
    def find_BMU(SOM, x):
        distSq = (np.square(SOM - x)).sum(axis=2)
        return np.unravel_index(np.argmin(distSq, axis=None), distSq.shape)

    def update_weights(SOM, train_ex, learn_rate, radius_sq,
                       BMU_coord, step=3):
        g, h = BMU_coord

        if radius_sq < 1e-3:
            SOM[g, h, :] += learn_rate * (train_ex - SOM[g, h, :])
            return SOM

        for i in range(max(0, g - step), min(SOM.shape[0], g + step)):
            for j in range(max(0, h - step), min(SOM.shape[1], h + step)):
                dist_sq = np.square(i - g) + np.square(j - h)
                dist_func = np.exp(-dist_sq / 2 / radius_sq)
                SOM[i, j, :] += learn_rate * dist_func * (train_ex - SOM[i, j, :])
        return SOM

    def train_SOM(SOM,
                  train_data,
                  learn_rate=.1,
                  radius_sq=1,
                  lr_decay=.1,
                  radius_decay=.1,
                  epochs=10
                  ):
        learn_rate_0 = learn_rate
        radius_0 = radius_sq
        for epoch in np.arange(0, epochs):
            rand.shuffle(train_data)
            for train_ex in train_data:
                g, h = find_BMU(SOM, train_ex)
                SOM = update_weights(SOM, train_ex,
                                     learn_rate, radius_sq, (g, h))

            learn_rate = learn_rate_0 * np.exp(-epoch * lr_decay)
            radius_sq = radius_0 * np.exp(-epoch * radius_decay)
        return SOM





