import numpy as np
from matplotlib import pyplot as plt, patches, cm, colors
import matplotlib as mpl
import seaborn as sns
from tqdm import tqdm
import random as rd
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from typing import List, Optional, Tuple
from scipy.stats import norm
import itertools
from susi.SOMPlots import plot_umatrix

class somutils:

    def getSUSI_mat_count(susi):
        mat = np.zeros(shape=(susi.unsuper_som_.shape[0], susi.unsuper_som_.shape[1]), dtype=int)
        for node in susi.bmus_:
            mat[node[0], node[1]] += 1
        return mat

    def plotSUSI_Cluster_hexagon(susi, n_clusters, figsize=(15, 15),
                                 title='unsuper_som con clustering y recuento de nodos'):
        ## Reformateo de las coordenadas de los pesos en cada nodo para obtener un matriz X que pasar el clustering
        X_pesos = susi.unsuper_som_.reshape(susi.unsuper_som_.shape[0] * susi.unsuper_som_.shape[1],
                                            susi.unsuper_som_.shape[2])
        cluster_ag = AgglomerativeClustering(distance_threshold=None, n_clusters=n_clusters, compute_distances=True)
        y_ag = cluster_ag.fit_predict(X_pesos)
        y_pesos = y_ag.reshape(susi.unsuper_som_.shape[0], susi.unsuper_som_.shape[1])
        mat_count = getSUSI_mat_count(susi)
        # Make figure and axes
        # colores = ['b', 'g', 'r', 'c', 'm', 'y', 'k','#653700','#E6DAA6','#DDA0DD','#BBF90F']
        # colorlist=["darkorange", "gold", "lawngreen", "lightseagreen"]
        # cmap=LinearSegmentedColormap.from_list('clusters', colors=colorlist, N=n_clusters)
        cmap = mpl.colormaps["Spectral"]
        # need to normalize because color maps are defined in [0, 1]
        norm = colors.Normalize(0, n_clusters)

        fig, ax = plt.subplots(figsize=figsize)
        ax.set_title(title, loc='left', fontstyle='oblique', fontsize='medium')
        xc, yc = 0, 0  # Se inicializa el valor del centro del hexagono
        r = 1  # Valor del radio del hexagono
        a = r * np.sqrt(3) / 2  # Valor de la apotema
        for i in range(susi.n_rows):
            xc = 0 if i % 2 == 0 else a  ## Se vuelve a poner la xc a su valor inicial
            for j in range(susi.n_columns):
                hexa = patches.RegularPolygon((xc, yc), 6, radius=r, orientation=0,
                                              facecolor=cmap(norm(y_pesos[susi.n_rows - i - 1][j]))[0:3],
                                              edgecolor='w', fill=True)
                ax.add_patch(hexa)
                plt.text(xc, yc, str(mat_count[susi.n_rows - i - 1, j]), color='k', va='center', ha='center')
                xc = xc + 2 * a
            yc = yc + r + a / 2

        fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax)
        ax.axis('equal')
        ax.set_xticks([])
        ax.set_yticks([])
        plt.show()

    def plotSUSI_neurPointsCount(susi, figsize=(13, 7), title="Conteo por nodo"):
        ## En la variable susi.mat_count se encuentra el recuento de puntos
        fig, ax = plt.subplots(figsize=figsize)

        # Set the font size and the distance of the title from the plot
        plt.title(title, fontsize=18)
        ttl = ax.title
        ttl.set_position([0.5, 1.05])

        # Hide ticks for X & Y axis
        ax.set_xticks([])
        ax.set_yticks([])

        # Remove the axes
        ax.axis('off')
        # sns.heatmap(susi.mat_count,annot=susi.mat_count,fmt="",cmap='RdYlGn',linewidths=0.30,ax=ax
        # fig, ax = plt.subplots(figsize=figsize)
        # ax.set_title(title
        mat_count = getSUSI_mat_count(susi)
        im = ax.imshow(mat_count)
        ax.figure.colorbar(im, ax=ax)
        # Loop over data dimensions and create text annotations.
        for i in range(susi.n_rows):
            for j in range(susi.n_columns):
                text = ax.text(j, i, mat_count[i, j], ha="center", va="center", color="w")
        plt.show()

    def plotSUSI_valuesMap(susi, labels, figsize=(15, 15),
                           title='Distribución del valor de las observaciones por neurona'):
        # Dibuja el peso de cada neurona
        # Se aplana la matriz con los pesos de las neuronas
        somFlat = susi.unsuper_som_.reshape(
            (susi.unsuper_som_.shape[0] * susi.unsuper_som_.shape[1], susi.unsuper_som_.shape[2]))
        media = np.mean(somFlat, axis=0)
        desv = np.std(somFlat, axis=0)
        # Make figure and axes
        colores = ['b', 'g', 'r', 'c', 'm', 'y', 'k', '#653700', '#E6DAA6', '#DDA0DD', '#BBF90F', '#990000', '#D5A6BD']
        if susi.X_.shape[1] <= len(colores):
            fig, axs = plt.subplots(susi.n_rows, susi.n_columns, figsize=figsize, subplot_kw=dict(polar=True))
        else:
            fig, axs = plt.subplots(susi.n_rows, susi.n_columns, figsize=figsize)
        plt.suptitle(title)
        if susi.X_.shape[1] <= len(colores):
            colores = colores[0:susi.X_.shape[1]]
        N = susi.unsuper_som_.shape[2]
        theta = np.linspace(0.0, 2 * np.pi, N, endpoint=False)
        width = [2 * np.pi / N for k in range(N)]
        for i in range(susi.n_rows):
            for j in range(susi.n_columns):
                dat_est = [norm(media[k], desv[k]).cdf(susi.unsuper_som_[susi.n_rows - i - 1, j, k]) for k in
                           range(susi.unsuper_som_.shape[2])]
                if susi.X_.shape[1] > len(colores):
                    axs[i, j].plot(dat_est)
                    axs[i, j].set_xticks([])
                    axs[i, j].set_yticks([])

                else:
                    axs[i, j].bar(theta, dat_est, width=width, bottom=0.0, color=colores, alpha=0.5)
                    tic = mpl.ticker
                    no_labels = mpl.ticker.NullFormatter()
                    # axs[i, j].yaxis.set_major_locator(tic.MultipleLocator(2))
                    axs[i, j].xaxis.set_major_formatter(no_labels)
                    axs[i, j].yaxis.set_major_formatter(no_labels)
                # axs[i, j].axis('equal')

        if susi.X_.shape[1] == len(colores):
            leg = fig.legend(labels, loc='lower right')
            for k in range(susi.unsuper_som_.shape[2]):
                # leg.legendHandles[k].set_color(colores[k])
                leg.legend_handles[k].set_color(colores[k])
        plt.show()

    def getSUSI_IndexDataNeuron(susi, r, c):
        indexes = []
        for index, bmu in enumerate(susi.bmus_):
            if bmu[0] == r and bmu[1] == c:
                indexes.append(index)
        return np.asarray(indexes)

    def getSUSI_DataNeuron(susi, r, c):
        data = []
        for index, bmu in enumerate(susi.bmus_):
            if bmu[0] == r and bmu[1] == c:
                data.append(susi.X_[index])
        return np.asarray(data)

    def plotSUSI_heatmaps(susi, labels, ilabMap, figsize=(13, 7), title="mapas de Calor"):
        n_mapas = len(ilabMap)
        fig, axs = plt.subplots(n_mapas, figsize=figsize)

        for i in range(n_mapas):
            annot = np.empty((susi.n_rows, susi.n_columns))
            mat = np.zeros((susi.n_rows, susi.n_columns))
            iL = ilabMap[i]
            for f in range(susi.n_rows):
                for c in range(susi.n_columns):
                    arr_map = getSUSI_DataNeuron(susi, f, c)
                    if arr_map.shape[0] > 0:
                        # mat[f,c] = np.sum(arr_map[:,iL])/arr_map.shape[0]
                        mat[f, c] = np.mean(arr_map[:, iL], axis=0)
            # Hide ticks for X & Y axis
            if n_mapas > 1:
                axs[i].set_title(labels[iL])
                axs[i].set_xticks([])
                axs[i].set_yticks([])
                # Remove the axes
                axs[i].axis('off')
                sns.heatmap(mat, cmap='RdYlGn', ax=axs[i])
            else:
                axs.set_title(labels[iL])
                axs.set_xticks([])
                axs.set_yticks([])
                # Remove the axes
                axs.axis('off')
                sns.heatmap(mat, cmap='RdYlGn', ax=axs)

        plt.show()

    def getDistanceNode(susi, r, c, data):
        return np.sqrt([np.dot((susi.unsuper_som_[r][c] - x), (susi.unsuper_som_[r][c] - x).T) for x in data])

    def plotSUSI_pointsMap(susi, figsize=(15, 15), title='Mapeo de observaciones por neurona'):
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_title(title, loc='left', fontstyle='oblique', fontsize='medium')
        yc = 1  # Valor inicial de la coordenada y del centro del circulo principal
        rp = 1  # Valor del radio del circulo externo principal
        for i in range(susi.n_rows):
            xc = 1  ## Valor inicial de la coordenada x del centro del circulo principal
            for j in range(susi.n_columns):
                circle = patches.Circle((xc, yc), radius=rp, color='black', fill=False)
                ax.add_patch(circle)
                arr_map = getSUSI_DataNeuron(susi, susi.n_rows - i - 1, j)
                if arr_map.shape[0] > 0:
                    radios = getDistanceNode(susi, susi.n_rows - i - 1, j, arr_map)
                    max_radio = max(radios)
                    for r in radios:
                        ## Se toma un angulo random para situar el centro
                        t = rd.random()  ## Un random entre 0 y 1 que se multiplicará por 2*np.pi
                        centro = [xc + r * np.cos(2 * t * np.pi) / (2 * max_radio),
                                  yc + r * np.sin(2 * t * np.pi) / (2 * max_radio)]
                        circle = patches.Circle(centro, radius=r / (2 * max_radio), color='blue', fill=False)
                        ax.add_patch(circle)
                xc = xc + 2 * rp
            yc = yc + 2 * rp

        ax.axis('equal')
        ax.set_xticks([])
        ax.set_yticks([])
        plt.show()

    def plotSUSI_u_matrix(susi, cmap: str = "Greys", fontsize: int = 18, figsize: Tuple = (6, 6)):
        u_matrix = susi.get_u_matrix()
        _, ax = plt.subplots(figsize=figsize)
        img = ax.imshow(u_matrix.squeeze(), cmap=cmap)
        ax.set_xticks(np.arange(0, susi.n_columns * 2 + 1, 20))
        ax.set_xticklabels(np.arange(0, susi.n_columns + 1, 10))
        ax.set_yticks(np.arange(0, susi.n_rows * 2 + 1, 20))
        ax.set_yticklabels(np.arange(0, susi.n_rows + 1, 10))

        # ticks and labels
        ax.xaxis.set_tick_params(labelsize=fontsize)
        ax.yaxis.set_tick_params(labelsize=fontsize)
        ax.set_ylabel("SOM rows", fontsize=fontsize)
        ax.set_xlabel("SOM columns", fontsize=fontsize)

        # colorbar
        cbar = plt.colorbar(img, ax=ax, fraction=0.04, pad=0.04)
        cbar.ax.set_ylabel(
            "Distance measure (a.u.)", rotation=90, fontsize=fontsize, labelpad=20
        )
        cbar.ax.tick_params(labelsize=fontsize)

        return

    def getSUSI_topologiaXY(susi):
        topoXY = np.asarray([[[0, 0] for j in range(susi.n_columns)] for i in range(susi.n_rows)])
        xc, yc = 0, 0  # Se inicializa el valor del centro del hexagono
        r = 1  # Valor del radio del hexagono
        a = r * np.sqrt(3) / 2  # Valor de la apotema
        for i in range(susi.n_rows):
            xc = 0 if i % 2 == 0 else a  ## Se vuelve a poner la xc a su valor inicial
            for j in range(susi.n_columns):
                topoXY[i, j] = [xc, yc]
                xc = xc + 2 * a
            yc = yc + r + a / 2
        topoXY.resize((topoXY.shape[0] * topoXY.shape[1], topoXY.shape[2]))
        return topoXY

    def getSUSI_Codes(susi):
        _codes = susi.unsuper_som_
        _codes = _codes.reshape(_codes.shape[0] * _codes.shape[1], _codes.shape[2])
        return _codes