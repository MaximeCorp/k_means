self made K means model (very long to run so there are pics in the repository).

K means is an unsupervised machine learning model that simplifies datasets by using k "clusters" which are initialized randomly at the first step.
At each step, every point in the dataset will be assigned to the closest cluster (euclidian distance) and the cluster's position is updated to the mean of the points assigned to the cluster.
The aim is to minimize the sum of the distance between the points and their assigned cluster.
When the difference is considered to be low enough, the points can be replaced by their cluster.
In the case of image compression, the points are the pixels and their coordinates (x,y,z) are their rgb, the final result is a picture with only k different colors.
