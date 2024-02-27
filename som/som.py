import numpy as np
from tqdm import tqdm 

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
        #self._createSOM()
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

    def storeMapping(self, train_data):
        self.data_ = train_data
        self.data_map = [ [[] for c in range(self.ncols)] for r in range(self.nrows)]
        self.mat_count = np.zeros((self.nrows, self.ncols), np.int_)
        
        for index, x in enumerate(self.data_):
            r, c = self.find_BMU(x)
            self.data_map[r][c].append(index)
            self.mat_count[r,c] +=1

    def get_topologiaXY(self):
        topoXY = np.asarray([[[0,0] for j in range(self.ncols)] for i in range(self.nrows)])
        xc, yc = 0, 0 #Se inicializa el valor del centro del hexagono
        r = 1 #Valor del radio del hexagono
        a = r*np.sqrt(3)/2 # Valor de la apotema
        for i in range(self.nrows): 
            xc = 0 if i%2==0 else a  ## Se vuelve a poner la xc a su valor inicial
            for j in range(self.ncols):
                topoXY[i,j]=[xc,yc]
                xc = xc + 2*a
            yc = yc + r + a/2
        topoXY.resize((topoXY.shape[0]*topoXY.shape[1],topoXY.shape[2]))
        return topoXY

    def getCodes(self):
        _codes = self.SOM
        _codes = _codes.reshape(_codes.shape[0]*_codes.shape[1], _codes.shape[2])
        return _codes


    def getDataNeuron(self, row, col):
        return self.data_[self.data_map[row][col]]
    
    def getIndexDataNeur(self, row, col):
        return self.data_map[row][col]

    def getDistanceNode(self, r, c, data):
        ## Devuelve la distancia de las observaciones del array data a la neurona (r,c)
        
        return np.sqrt([np.dot((self.SOM[r][c]-x), (self.SOM[r][c]-x).T) for x in data])


    def train_SOM(self,train_data,epochs):
        
        self.dimension = train_data.shape[1]
        self._createSOM()


        lstEpocas = tqdm([i for i in range(0, epochs)])
        for epoch in lstEpocas:

            learn_rate_t = self.learn_rate * np.exp(-epoch * self.lr_decay)
            radius_sq_t = self.radius_sq * np.exp(-epoch * self.radius_decay)

            self.RDS.shuffle(train_data)
            for train_ex in train_data:
                g, h = self.find_BMU(train_ex)
                self.update_weights(train_ex,
                                    learn_rate_t,
                                    radius_sq_t,
                                    (g, h))

        self.storeMapping(train_data)
        return self

    def train_super_SOM(self, train_data, y_train, epochs):
        data_train = np.zeros((train_data.shape[0],train_data.shape[1]+1))
        for i, value in enumerate(zip(train_data, y_train)):
            for j, val in enumerate(value[0]):
                data_train[i,j]=val
            data_train[i, data_train.shape[1]-1] = value[1]

        self.train_SOM(train_data=data_train, epochs=epochs)

        self.SOM_atrib = np.zeros((self.SOM.shape[0],self.SOM.shape[1], self.SOM.shape[2]-1))
        self.SOM_label = np.zeros((self.SOM.shape[0],self.SOM.shape[1]))
        for i in range(self.SOM.shape[0]):
            for j in range(self.SOM.shape[1]):
                for k in range(self.SOM.shape[2]-1):
                    self.SOM_atrib[i,j,k]=self.SOM[i,j,k]
                self.SOM_label[i,j]=np.round(self.SOM[i,j,-1])

        return self

    def getNeuronaCercana(self, x):
        iNeu, jNeu = -1, -1
        dist = np.inf
        for i in range(self.SOM_atrib.shape[0]):
            for j in range(self.SOM_atrib.shape[1]):
                dif_ = self.SOM_atrib[i,j] - x
                w_dist = np.sqrt(np.dot(dif_.T,dif_))
                if w_dist < dist:
                   dist = w_dist
                   iNeu, jNeu = i, j
    
        return iNeu, jNeu

    def getPredict(self, X):
        
        y_pred = []
        for x_ in X:
            neur = self.getNeuronaCercana(x_)
            y_pred.append(self.SOM_label[neur])

        return np.asarray(y_pred)

    def score(self, y_pred, y_test):
        assert len(y_pred.shape) == len(y_test.shape), "y_pred e y_test deben ser 2 arrays de una dimensión"
        assert y_pred.shape[0] == y_test.shape[0], "y_pred e y_test deben ser 2 arrays de igual longitud"
        return len(y_pred[y_pred == y_test])/len(y_pred)
        
    def coordenadasNeurona(self,fila,columna):
        posx=posy=0
        
        if self.vicinity == "regular":
            posx = fila
            posy = columna
        elif self.vicinity == "hexagonal":
            
            # dimensiones del hexagono regular:
            # radio circulo cincunscrito
            ru = 1 # igual a longitud de un lado
            # radio circulo inscrito
            ri = np.sqrt(3)/2*ru # cos(30)*ru
            
            dx = 2*ri # desplazamiento en x
            dy = 3*ru/2

            if fila%2==0:
                # fila par
                posxinit = ri
            else:
                posxinit = 0
                
            posx = posxinit + (columna-1)*dx
            posy = (fila-1)*dy
            
	return posx,posy

