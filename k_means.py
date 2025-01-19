import random
from PIL import Image
import numpy as np
import copy
import matplotlib.pyplot as plt

#Centroid computation
def computeCentroid(data):
    # Return empty array in case of empty data or 0 dimensions data
    if len(data) == 0 or len(data[0]) == 0:
        return []

    # Compute sum
    res = [0] * len(data[0])
    for point in data:
        for i in range(len(point)):
            feature = point[i]
            res[i] += feature

    # Divide to get mean
    for i in range(len(res)):
        res[i] /= len(data)

    return res

# Squared distance
def distance(point1, point2):
    res = 0
    for i in range(max(len(point1), len(point2))):
        cur = point1[i] - point2[i]
        res += cur * cur
    return res

# Find clusters
def mykmeans(X, k):
    # Basic information
    dim = len(X[0])
    # Randomly generate k clusters
    clusters = []
    for i in range(k):
        clusters.append(copy.deepcopy(X[random.randint(0, len(X)-1)]))

    dist = 1

    # Loop based on variation in function of dataset distance
    while np.sqrt(dist) > 0.00001:
        # Classification of dataset
        classification = [[] for _ in range(k)]
        for point in X:
            dists = []
            for cluster in clusters:
                cur_dist = distance(cluster, point)
                dists.append(cur_dist)
            n = dists.index(min(dists))
            classification[n].append(point)

        # Compute new clusters
        last = copy.deepcopy(clusters)
        for i in range(len(classification)):
            if len(classification[i]) != 0:
                clusters[i] = computeCentroid(classification[i])


        # Compute variation and keep the highest variation
        far = distance(clusters[0], last[0])
        for i in range(len(clusters)-1):
            cur_dist = distance(clusters[i+1], last[i+1])
            if cur_dist > far:
                far = cur_dist

        dist = far

    return clusters

# Data from image
def collect_pixels(path):
    # Collect image
    image = Image.open(path)
    width, height = image.size

    # Fill pixels array
    pixels = []
    for j in range(height):
        for i in range(width):
            rgb = image.getpixel((i,j))
            pixels.append([rgb[0], rgb[1], rgb[2]])

    return pixels, height, width



# Compress and display image
def compress_image(path,k):
    # Collect data
    data, height, width = collect_pixels(path)

    k += 1


    # Find clusters
    clusters = mykmeans(data, k)

    # Pixels classification
    for i in range(len(data)):
        # Find the closest cluster
        pixel = data[i]
        n, dist = 0, distance(pixel, clusters[0])

        for j in range(len(clusters) - 1):
            cluster = clusters[j+1]
            cur_dist = distance(pixel, cluster)

            if cur_dist < dist:
                n, dist = j, cur_dist

        data[i] = n

    # Convert clusters to int
    for i in range(len(clusters)):
        for j in range(len(clusters[0])):
            clusters[i][j] = int(clusters[i][j])

    # Generate new image
    compressed = Image.new("RGB", (width, height))
    for i in range(len(data)):
        x = i%width
        y = i // width


        rgb = tuple(clusters[data[i]][:3])
        compressed.putpixel((x,y), rgb)

    #Display image
    res = np.array(compressed)
    plt.imshow(res)
    plt.show()

path = "test.jpg"
compress_image(path, 3)
compress_image(path, 5)
compress_image(path, 8)
