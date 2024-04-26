import random
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsRegressor
import numpy as np

#Creating a random test data to work with
def create_rand_data(nodes, x_num, y_num):
    x = [random.randint(0, x_num) for i in range(nodes)]
    y = [random.randint(0, y_num) for i in range(nodes)]

    plt.title("Random Data")
    plt.scatter(x,y)
    plt.show()
    return zip(x, y)

#Inserting data into Kmeans and getting its 'Elbow'(not essenssial cos we need 4>= clusters for this method to show its impact)
def elbow(data):
    inertias = []
    for i in range(1,15):
        kmeans = KMeans(n_clusters=i)
        kmeans.fit(data)
        inertias.append(kmeans.inertia_)
    plt.plot(range(1,15), inertias, marker='o')
    plt.title('Elbow method')
    plt.xlabel('Number of clusters')
    plt.ylabel('Inertia')
    plt.show() 

# implementingg KMeans with target no. of clusters
def kmeans_centroids(data, clusters):
    #Applying Kmeans on input data
    x, y = zip(*data)
    kmeans = KMeans(n_clusters= clusters)
    kmeans.fit(data)
    plt.scatter(x, y, c=kmeans.labels_)
    plt.title(f"KMeans Clustered data with n={clusters}")
    plt.show()
    
    #Finding cetroids of the clusters
    centroids = kmeans.cluster_centers_
    X_train = []
    y_train = []
    for i, centroid in enumerate(centroids):
        print(f"Centroid {i + 1}: ({centroid[0]}, {centroid[1]})")
        X_train.append([centroid[0], centroid[1]])
        y_train.append(i + 1)
        
    #placing them into numpy arryas
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    plt.scatter(X_train[:, 0], X_train[:, 1])
    plt.title(f"KMeans Centroids")
    plt.show()
    labels = kmeans.labels_
    
    #Getting nodes with their coresponing clusters
    print("Nodes with their corresponding clusters:")
    clusters = {i: [] for i in range(kmeans.n_clusters)}
    for i, label in enumerate(labels):
        clusters[label].append(data[i])
    for cluster, points in clusters.items():
        print(f"Cluster {cluster + 1}: {points}")
    return X_train, y_train, clusters

def knn_classify(new_node, n_neighbors):
    #Applying KNN on kmeans centroids
    X_train, y_train, clusters = kmeans_centroids(data, 8)
    knn_regressor = KNeighborsRegressor(n_neighbors=3)
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    knn_regressor.fit(X_train, y_train)    
    
    neighbors_distances , neighbors_indices = knn_regressor.kneighbors(
        new_node, n_neighbors=3)

    closest_nodes = X_train[neighbors_indices]
    closest_nodes_targets = y_train[neighbors_indices]

    print("Closest Nodes:", closest_nodes)#shows position of closest centroids
    print("Distances:", neighbors_distances)
    print("Target Values of Closest Nodes:", closest_nodes_targets, "\n")

    #Put all values off close clusters inside a list
    values = [] 
    X = []
    y = []
    cluster_nodes = []
    for i in closest_nodes_targets[0]:
        i = int(i)
        cluster_nodes.append(i)
        values.append(clusters[i-1])
    
    #seprate nodes and their cluster number is places inside a list for training
    for c in range(len(values)):
        for train in values[c]:
            X.append([train[0],train[1]])
            y.append(int(cluster_nodes[c]))
    X = np.array(X)
    y = np.array(y)
    plt.scatter(X[:, 0], X[:, 1])
    plt.show() #plotting not needed

    #fit(train) the node in KNN
    knn_regressor = KNeighborsRegressor(n_neighbors=n_neighbors)
    knn_regressor.fit(X, y)
    
    closest_values = X[neighbors_indices]
    closest_values_targets = y[neighbors_indices]

    print("Closest values:", closest_values)#shows position off closest nodes in space
    print("Distances:", neighbors_distances)
    print("Target Values of Closest targets:", closest_values_targets)#shows the cluster of the closest nodes
    
data = list(create_rand_data(1000, 10000, 10000))

new_node = np.array([[3699, 4500]])

knn_classify(new_node, 3)
