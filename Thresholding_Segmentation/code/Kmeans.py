import numpy as np
import matplotlib.pyplot as plt

import cv2


np.random.seed(42)

def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2)**2))

class grayKmeans():
    
    def __init__(self):
         
        self.res = None
        self.grey_l = [10,80,100,150,230,255]
           
    def Kmeans_Gray(self,grayImage):
        rows,columns = grayImage.shape
        self.histogram,self.bins = np.histogram(grayImage,256,[0,256])
        self.res = self.find_centroids_grayImage(self.histogram)
        end = np.zeros((grayImage.shape))
        
        if len(self.res[1]) > len(self.res[0]):
            '''bacground is res1'''
            flag = 1
        else:
            flag = 0


        for i in range(rows):
            for j in range(columns):
                if flag == 1:
                    if (grayImage[i][j] in self.res[1]):
                        end[i][j] = 0

                    else:
                        end[i][j] = 255
                else:
                    if (grayImage[i][j] in self.res[1]):
                        end[i][j] = 255

                    else:
                        end[i][j] = 0
        return(end)
        
    def find_centroids_grayImage(self,histogram):
        rand_points = [ np.random.randint(0, 255) for i in range(2) ]
        centroid1_avg = 0
        centroid2_avg = 0   
        for k in range(0,10):
            if k == 0:
                cent1, cent2 = rand_points
                
            else:
                cent1 = centroid1_avg
                cent2 = centroid2_avg

            point1_centroid = []
            point2_centroid = []
            w1_centroid = []
            w2_centroid = []
            sum1 = 0
            sum2 = 0
            for i,val in enumerate(histogram):
                ''' computing absolute distance from each of the cluster and assigning it to a particular cluster based on distance'''
                if  abs(i - cent1) <  abs(i - cent2):
                    point1_centroid.append(i)
                    w1_centroid.append(val)
                    sum1 = sum1 + (i * val)
                else:
                    point2_centroid.append(i)
                    w2_centroid.append(val)
                    sum2 = sum2 + (i * val)
            
            
            centroid1_avg = int(sum1)/sum(w1_centroid)	
            centroid2_avg = int(sum2)/sum(w2_centroid)			
        return [point1_centroid,point2_centroid] 
    

class KMeans():

    def __init__(self, K=5, max_iters=100, plot_steps=False):
        self.K = K
        self.max_iters = max_iters
        self.plot_steps = plot_steps

        # list of sample indices for each cluster
        self.clusters = [[] for _ in range(self.K)]
        # the centers (mean feature vector) for each cluster
        self.centroids = []

    def predict(self, X):
        self.X = X
        self.n_samples, self.n_features = X.shape
        
        # initialize 
        random_sample_idxs = np.random.choice(self.n_samples, self.K, replace=False)
        self.centroids = [self.X[idx] for idx in random_sample_idxs]

        # Optimize clusters
        for _ in range(self.max_iters):
            # Assign samples to closest centroids (create clusters)
            self.clusters = self._create_clusters(self.centroids)
            if self.plot_steps:
                self.plot()

            # Calculate new centroids from the clusters
            centroids_old = self.centroids
            self.centroids = self._get_centroids(self.clusters)
            
            # check if clusters have changed
            if self._is_converged(centroids_old, self.centroids):
                break

            if self.plot_steps:
                self.plot()

        # Classify samples as the index of their clusters
        return self._get_cluster_labels(self.clusters)


    def _get_cluster_labels(self, clusters):
        # each sample will get the label of the cluster it was assigned to
        labels = np.empty(self.n_samples)

        for cluster_idx, cluster in enumerate(clusters):
            for sample_index in cluster:
                labels[sample_index] = cluster_idx
        return labels

    def _create_clusters(self, centroids):
        # Assign the samples to the closest centroids to create clusters
        clusters = [[] for _ in range(self.K)]
        for idx, sample in enumerate(self.X):
            centroid_idx = self._closest_centroid(sample, centroids)
            clusters[centroid_idx].append(idx)
        return clusters

    def _closest_centroid(self, sample, centroids):
        # distance of the current sample to each centroid
        distances = [euclidean_distance(sample, point) for point in centroids]
        closest_index = np.argmin(distances)
        return closest_index

    def _get_centroids(self, clusters):
        # assign mean value of clusters to centroids
        centroids = np.zeros((self.K, self.n_features))
        for cluster_idx, cluster in enumerate(clusters):
            cluster_mean = np.mean(self.X[cluster], axis=0)
            centroids[cluster_idx] = cluster_mean
        return centroids

    def _is_converged(self, centroids_old, centroids):
        # distances between each old and new centroids, fol all centroids
        distances = [euclidean_distance(centroids_old[i], centroids[i]) for i in range(self.K)]
        return sum(distances) == 0

    def cent(self):
        return self.centroids


def draw_kmeans(image):
    pixel_values = image.reshape((-1, 3))
    pixel_values = np.float32(pixel_values)
    k = KMeans(K=3, max_iters=100) 
    y_pred = k.predict(pixel_values)
    k.cent()
    centers = np.uint8(k.cent())
    y_pred = y_pred.astype(int)
    np.unique(y_pred)
    labels = y_pred.flatten()
    segmented_image = centers[labels.flatten()]
    segmented_image = segmented_image.reshape(image.shape)
    masked_image = np.copy(image)
    masked_image = masked_image.reshape((-1, 3))
    cluster = 2
    masked_image[labels == cluster] = [0, 0, 0]
    masked_image = masked_image.reshape(image.shape)
    return(masked_image)




