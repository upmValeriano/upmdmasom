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
                 vicinity = "rectangular",
                 learn_rate=.1,
                 radius_sq=1,
                 lr_decay=.1,
                 radius_decay=.1,
                 #epochs=10,
                 randomState = 0
                 ):
        self.RDS = np.random.RandomState(randomState)
        self.nrows = nrows
        self.ncols = ncols
        self.dimension = dimension
        self.vmin = vmin
        self.vmax = vmax
        self.vicinity = vicinity
        self._createSOM()
        self.learn_rate = learn_rate
        self.radius_sq = radius_sq
        self.lr_decay = lr_decay
        self.radius_decay = radius_decay
        #self.epochs = epochs

    def _createSOM(self):
        SOM = self.RDS.rand(self.nrows,
                             self.ncols,
                             self.dimension)
        SOM = SOM * (self.vmax-self.vmin)+self.vmin
        self.SOM = SOM

    #
    def find_BMU(self, x):
        distSq = (np.square(self.SOM - x)).sum(axis=2)
        return np.unravel_index(np.argmin(distSq, axis=None), distSq.shape)

    def update_weights(self,
                       train_ex,
                       learn_rate_t,
                       radius_sq_t,
                       BMU_coord,
                       step=3):
        g, h = BMU_coord

        if radius_sq_t < 1e-3:
            self.SOM[g, h, :] += learn_rate_t * (train_ex - self.SOM[g, h, :])
            return self

        for i in range(max(0, g - step), min(self.SOM.shape[0], g + step)):
            for j in range(max(0, h - step), min(self.SOM.shape[1], h + step)):
                dist_sq = np.square(i - g) + np.square(j - h)
                dist_func = np.exp(-dist_sq / 2 / radius_sq_t)
                self.SOM[i, j, :] += learn_rate_t * dist_func * (train_ex - self.SOM[i, j, :])
        return self

    def train_SOM(self,train_data,epochs):
        for epoch in np.arange(0, epochs):

            learn_rate_t = self.learn_rate * np.exp(-epoch * self.lr_decay)
            radius_sq_t = self.radius_sq * np.exp(-epoch * self.radius_decay)

            self.RDS.shuffle(train_data)
            for train_ex in train_data:
                g, h = self.find_BMU(train_ex)
                self.update_weights(train_ex,
                                    learn_rate_t,
                                    radius_sq_t,
                                    (g, h))

        return self





