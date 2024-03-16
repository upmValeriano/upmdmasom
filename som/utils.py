import numpy as np
from matplotlib import pyplot as plt, patches, cm, colors
import matplotlib as mpl
import seaborn as sns
from sklearn.cluster import AgglomerativeClustering
from scipy.stats import norm
from typing import List, Optional, Tuple
import itertools
import random as rd

class somutils:

    def plot_Cluster_hexagon(pSom, n_clusters, figsize=(15,15), title='SOM con clustering y recuento de nodos'):
        ## Reformateo de las coordenadas de los pesos en cada nodo para obtener un matriz X que pasar el clustering
        X_pesos = pSom.SOM.reshape(pSom.SOM.shape[0]*pSom.SOM.shape[1], pSom.SOM.shape[2])
        cluster_ag = AgglomerativeClustering(distance_threshold=None, n_clusters=n_clusters, compute_distances=True)
        y_ag = cluster_ag.fit_predict(X_pesos)
        y_pesos = y_ag.reshape(pSom.SOM.shape[0], pSom.SOM.shape[1])
        cmap = mpl.colormaps["Spectral"]
        norm = colors.Normalize(0, n_clusters)
        
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_title(title, loc='left', fontstyle='oblique', fontsize='medium')
        xc, yc = 0, 0 #Se inicializa el valor del centro del hexagono
        r = 1 #Valor del radio del hexagono
        a = r*np.sqrt(3)/2 # Valor de la apotema
        for i in range(pSom.nrows):
            xc = 0 if i%2==0 else a  ## Se vuelve a poner la xc a su valor inicial
            for j in range(pSom.ncols):

                hexa = patches.RegularPolygon((xc,yc), 6, radius=r, orientation=0, facecolor=cmap(norm(y_pesos[pSom.nrows-i-1][j]))[0:3],
                                             edgecolor='w', fill=True)
                ax.add_patch(hexa)
                plt.text(xc, yc, str(pSom.mat_count[pSom.nrows-i-1,j]), color='k', va='center', ha='center')
                xc = xc + 2*a
            yc = yc + r + a/2

        fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax)
        ax.axis('equal')
        ax.set_xticks([])
        ax.set_yticks([])
        plt.show()

    def plot_neurPointsCount(pSom, figsize=(15,15), title = "Conteo por nodo"):
        ## En la variable self.mat_count se encuentra el recuento de puntos
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_title(title, loc='left', fontstyle='oblique', fontsize='medium')

        # Set the font size and the distance of the title from the plot

        #plt.title(title,fontsize=18)
        #ttl = ax.title
        #ttl.set_position([0.5,1.05])



        # Remove the axes
        ax.axis('off')

        if pSom.vicinity == 'rectangular':

            im = ax.imshow(pSom.mat_count)
            ax.figure.colorbar(im, ax=ax)
            # Loop over data dimensions and create text annotations.
            for i in range(pSom.nrows):
                for j in range(pSom.ncols):
                    text = ax.text(j, i, pSom.mat_count[i, j], ha="center", va="center", color="w")

        elif pSom.vicinity == 'hexagonal':

            # print('hexagonal')

            maxCount = np.max(pSom.mat_count)
            print('maxCount', maxCount)

            cmap = mpl.colormaps["Spectral"]
            norm = colors.Normalize(0, maxCount)

            xc, yc = 0, 0  # Se inicializa el valor del centro del hexagono
            r = 1  # Valor del radio del hexagono
            a = r * np.sqrt(3) / 2  # Valor de la apotema
            for i in range(pSom.nrows):
                xc = 0 if i % 2 == 0 else a  ## Se vuelve a poner la xc a su valor inicial
                for j in range(pSom.ncols):
                    countValueNeuron = pSom.mat_count[i,j]
                    hexa = patches.RegularPolygon((xc, yc), 6,
                                                  radius=r,
                                                  orientation=0,
                                                  facecolor=cmap(norm(countValueNeuron)),
                                                  # facecolor='red',
                                                  edgecolor='w',
                                                  fill=True)
                    ax.add_patch(hexa)
                    plt.text(xc,yc,str(countValueNeuron),color='k',va="center",ha="center")
                    #plt.text(xc, yc, str(pSom.mat_count[pSom.nrows - i - 1, j]), color='k', va='center', ha='center')
                    xc = xc + 2 * a
                yc = yc + r + a / 2

            fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax)
        else:
            raise ValueError("Vicinity")

        ax.axis('equal')
        # Hide ticks for X & Y axis
        ax.set_xticks([])
        ax.set_yticks([])

        plt.show()

    def plot_valuesMap(pSom, labels, figsize=(15,15), title='Distribución del valor de las observaciones por neurona'):
        # Dibuja el peso de cada neurona
        # Se aplana la matriz con los pesos de las neuronas
        somFlat = pSom.SOM.reshape((pSom.SOM.shape[0]*pSom.SOM.shape[1], pSom.SOM.shape[2]))
        media = np.mean(somFlat, axis=0)
        desv = np.std(somFlat, axis=0)
        # Make figure and axes
        colores = ['b', 'g', 'r', 'c', 'm', 'y', 'k','#653700','#E6DAA6','#DDA0DD','#BBF90F','#990000','#D5A6BD']
        if pSom.data_.shape[1] <= len(colores):
            fig, axs = plt.subplots(pSom.nrows, pSom.ncols, figsize=figsize, subplot_kw=dict(polar=True))
        else:
            fig, axs = plt.subplots(pSom.nrows, pSom.ncols, figsize=figsize)
        plt.suptitle(title)
        if pSom.data_.shape[1] <= len(colores):
            colores = colores[0:pSom.data_.shape[1]]
        N=pSom.SOM.shape[2]
        theta = np.linspace(0.0, 2 * np.pi, N, endpoint=False)
        width = [2*np.pi/N for k in range(N)]
        for i in range(pSom.nrows):
            for j in range(pSom.ncols):
                dat_est = [norm(media[k], desv[k]).cdf(pSom.SOM[pSom.nrows-i-1,j,k]) for k in range(pSom.SOM.shape[2])]
                if pSom.data_.shape[1] > len(colores):
                    axs[i, j].plot(dat_est)
                    axs[i, j].set_xticks([])
                    axs[i, j].set_yticks([])

                else:
                    axs[i, j].bar(theta, dat_est, width=width, bottom=0.0, color=colores, alpha=0.5)
                    tic = mpl.ticker
                    no_labels = mpl.ticker.NullFormatter()
                    #axs[i, j].yaxis.set_major_locator(tic.MultipleLocator(2))
                    axs[i, j].xaxis.set_major_formatter(no_labels)
                    axs[i, j].yaxis.set_major_formatter(no_labels)
                #axs[i, j].axis('equal')

        if pSom.data_.shape[1] == len(colores):
            leg = fig.legend(labels, loc='lower right')
            for k in range(pSom.SOM.shape[2]):
                #leg.legendHandles[k].set_color(colores[k])
                leg.legend_handles[k].set_color(colores[k])
        plt.show()

    def plot_heatmaps(pSom, labels, ilabMap, figsize=(13,7), title = "mapas de Calor"):
        n_mapas = len(ilabMap)
        fig, axs = plt.subplots(n_mapas, figsize=figsize)


        for i in range(n_mapas):
            annot = np.empty((pSom.nrows, pSom.ncols))
            mat = np.zeros((pSom.nrows, pSom.ncols))
            iL = ilabMap[i]
            for f in range(pSom.nrows):
                for c in range(pSom.ncols):
                    #arr_map = pSom.data_[pSom.data_map[f][c]]
                    arr_map = pSom.getDataNeuron(f, c)
                    if arr_map.shape[0]>0:
                        #mat[f,c] = np.sum(arr_map[:,iL])/arr_map.shape[0]
                        mat[f,c] = np.mean(arr_map[:,iL], axis=0)
            # Hide ticks for X & Y axis
            if n_mapas>1:
                axs[i].set_title(labels[iL])
                axs[i].set_xticks([])
                axs[i].set_yticks([])
                # Remove the axes
                axs[i].axis('off')
                sns.heatmap(mat,cmap='RdYlGn',ax=axs[i])
            else:
                axs.set_title(labels[iL])
                axs.set_xticks([])
                axs.set_yticks([])
                # Remove the axes
                axs.axis('off')
                sns.heatmap(mat,cmap='RdYlGn',ax=axs)

        plt.show()

    def plot_pointsMap(pSom, figsize=(15,15), title='Mapeo de observaciones por neurona'):
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_title(title, loc='left', fontstyle='oblique', fontsize='medium')
        yc = 1 #Valor inicial de la coordenada y del centro del circulo principal
        rp = 1 #Valor del radio del circulo externo principal
        for i in range(pSom.nrows):
            xc = 1 ## Valor inicial de la coordenada x del centro del circulo principal
            for j in range(pSom.ncols):
                circle = patches.Circle((xc,yc), radius=rp, color='black', fill = False)
                ax.add_patch(circle)
                arr_map = pSom.data_[pSom.data_map[pSom.nrows-i-1][j]]
                if arr_map.shape[0]>0:
                    radios = pSom.getDistanceNode(pSom.nrows-i-1, j, arr_map)
                    max_radio = max(radios)
                    for r in radios:
                        ## Se toma un angulo random para situar el centro
                        t = rd.random() ## Un random entre 0 y 1 que se multiplicará por 2*np.pi
                        centro = [xc+r*np.cos(2*t*np.pi)/(2*max_radio), yc+r*np.sin(2*t*np.pi)/(2*max_radio)]
                        circle = patches.Circle(centro, radius=r/(2*max_radio), color='blue', fill = False)
                        ax.add_patch(circle)
                xc = xc + 2*rp
            yc = yc + 2*rp

        ax.axis('equal')
        ax.set_xticks([])
        ax.set_yticks([])
        plt.show()

    def plot_u_matrix(pSom, cmap: str = "Greys", fontsize: int = 18, figsize: Tuple = (6, 6)):
        u_matrix = somutils.get_u_matrix(pSom=pSom)
        _, ax = plt.subplots(figsize=figsize)
        img = ax.imshow(u_matrix.squeeze(), cmap=cmap)
        ax.set_xticks(np.arange(0, pSom.ncols * 2 + 1, 20))
        ax.set_xticklabels(np.arange(0, pSom.ncols + 1, 10))
        ax.set_yticks(np.arange(0, pSom.nrows * 2 + 1, 20))
        ax.set_yticklabels(np.arange(0, pSom.nrows + 1, 10))

        # ticks and labels
        ax.xaxis.set_tick_params(labelsize=fontsize)
        ax.yaxis.set_tick_params(labelsize=fontsize)
        #for tick in ax.xaxis.get_major_ticks():
            #tick.label.set_fontsize(fontsize)
        #for tick in ax.yaxis.get_major_ticks():
            #tick.label.set_fontsize(fontsize)
        ax.set_ylabel("SOM rows", fontsize=fontsize)
        ax.set_xlabel("SOM columns", fontsize=fontsize)

        # colorbar
        cbar = plt.colorbar(img, ax=ax, fraction=0.04, pad=0.04)
        cbar.ax.set_ylabel(
            "Distance measure (a.u.)", rotation=90, fontsize=fontsize, labelpad=20
        )
        cbar.ax.tick_params(labelsize=fontsize)

        return
        
    def get_u_matrix(pSom, mode: str = "mean") -> np.ndarray:
        """Calculate unified distance matrix (u-matrix).
        Parameters
        ----------
        mode : str, optional (default="mean)
            Choice of the averaging algorithm
        Returns
        -------
        u_matrix : np.ndarray
            U-matrix containing the distances between all nodes of the
            unsupervised SOM. Shape = (nrows*2-1, ncols*2-1)
        """
        pSom.u_mean_mode_ = mode

        pSom.u_matrix = np.zeros(shape=(pSom.nrows * 2 - 1, pSom.ncols * 2 - 1, 1), dtype=float)

        # step 1: fill values between SOM nodes
        somutils._calc_u_matrix_distances(pSom=pSom)

        # step 2: fill values at SOM nodes and on diagonals
        somutils._calc_u_matrix_means(pSom=pSom)

        return pSom.u_matrix

    def _calc_u_matrix_distances(pSom) -> None:
        """Calcula la distancia euclidea entre los nodos del SOM.
           En lugar de hacerlo con la dimensión del SOM - shape = (nrows, ncols, 1)
           La hace más granular hasta shape (nrows*2-1, ncols*2-1, 1), pero en los cálculo usa los datos del 
           SOM accediendo al indice que resulta del divisor entero row // 2 o col // 2
           
           Hace un primer cálculo de la distancia con el vecino de la derecha o inferior en la topología
        """
        for row in range(pSom.nrows * 2 - 1):
            for col in range(pSom.ncols * 2 - 1):
                # Vector vecino por defecto
                nb = (0, 0)

                if not (row % 2) and (col % 2):  # Vector vecino si Fila impar, columna par
                    # mean horizontally
                    nb = (0, 1)

                elif (row % 2) and not (col % 2):  # Vector vecino si Fila par, columna impar
                    # mean vertically
                    nb = (1, 0)

                pSom.u_matrix[row, col] = np.linalg.norm(
                    pSom.SOM[row // 2, col // 2] - 
                    pSom.SOM[row // 2 + nb[0], col // 2 + nb[1]], 
                    axis=0,
                )

    def _calc_u_matrix_means(pSom) -> None:
        """
        Calcula el resto de partes de la matriz U. Que son las entradas de las posiciones de los nodos
        del SOM actual y las entradas entre los nodos de la matriz de distancias con la granularidad aumentada.
        """
        for row in range(pSom.nrows * 2 - 1):
            for col in range(pSom.ncols * 2 - 1):
                if not (row % 2) and not (col % 2):
                    # SOM nodes -> mean over 2-4 values
                    nodelist = []
                    if row > 0:
                        nodelist.append((row - 1, col))
                    if row < pSom.nrows * 2 - 2:
                        nodelist.append((row + 1, col))
                    if col > 0:
                        nodelist.append((row, col - 1))
                    if col < pSom.ncols * 2 - 2:
                        nodelist.append((row, col + 1))
                    pSom.u_matrix[row, col] = somutils._get_u_mean(pSom, nodelist)
                elif (row % 2) and (col % 2):
                    # mean over four
                    pSom.u_matrix[row, col] = somutils._get_u_mean(pSom,
                        [
                            (row - 1, col),
                            (row + 1, col),
                            (row, col - 1),
                            (row, col + 1),
                        ]
                    )

    def _get_u_mean(pSom, nodelist: List[Tuple[int, int]]) -> Optional[float]:
        """
        Calcula el valor medio de una lista de nodos en la matriz U. 
        Alternativamente se puede usar la mediana, el mínimo o máximo 
        """
        meanlist = [pSom.u_matrix[u_node] for u_node in nodelist]
        u_mean = None
        if pSom.u_mean_mode_ == "mean":
            u_mean = np.mean(meanlist)
        elif pSom.u_mean_mode_ == "median":
            u_mean = np.median(meanlist)
        elif pSom.u_mean_mode_ == "min":
            u_mean = np.min(meanlist)
        elif pSom.u_mean_mode_ == "max":
            u_mean = np.max(meanlist)
        return u_mean
    def get_topologiaXY(pSom):
        topoXY = np.asarray([[[0,0] for j in range(pSom.ncols)] for i in range(pSom.nrows)])
        xc, yc = 0, 0 #Se inicializa el valor del centro del hexagono
        r = 1 #Valor del radio del hexagono
        a = r*np.sqrt(3)/2 # Valor de la apotema
        for i in range(pSom.nrows): 
            xc = 0 if i%2==0 else a  ## Se vuelve a poner la xc a su valor inicial
            for j in range(pSom.ncols):
                topoXY[i,j]=[xc,yc]
                xc = xc + 2*a
            yc = yc + r + a/2
        topoXY.resize((topoXY.shape[0]*topoXY.shape[1],topoXY.shape[2]))
        return topoXY
    def getCodes(pSom):
        _codes = pSom.SOM
        _codes = _codes.reshape(_codes.shape[0]*_codes.shape[1], _codes.shape[2])
        return _codes

    def plotEstimationMap(pSom, figsize=(13,7), title = "mapa de estimaciones"):

        mat = np.zeros((pSom.nrows, pSom.ncols))
        for f in range(pSom.nrows):
            for c in range(pSom.ncols):
                mat[f,c]=pSom.SOM_label[f,c]
    
        fig, axs = plt.subplots(1, figsize=figsize)
        sns.set()
        axs.set_xticks([])
        axs.set_yticks([])
        # Remove the axes
        axs.axis('off')
        sns.heatmap(mat,cmap='RdYlGn',ax=axs)
        plt.show()

