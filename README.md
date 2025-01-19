This project implements the K-Means clustering algorithm from scratch. It also includes a variety of utility functions to visualize clusters and perform image quantization. The K-Means algorithm is used to partition data points into k clusters based on their features.

## **Overview**

The K-Means algorithm aims to partition data into k clusters such that the points within each cluster are as close as possible to each other, while being as far as possible from points in other clusters. It is an iterative algorithm that minimizes the within-cluster sum of squared distances between data points and their corresponding cluster centers (centroids).

In this implementation, multiple utility functions are provided for:

    Visualizing the clusters and centroids
    Performing K-Means with multiple trials to avoid local minima
    Quantizing image colors using K-Means
    Generating synthetic K-Means data for testing purposes

## **Libraries and Dependencies**

This project requires the following Python libraries:

  numpy: For numerical operations, especially matrix and vector manipulation.
  matplotlib: For plotting the clusters and visualizing the results.
  itertools: For generating markers for cluster visualization.
