""" 3. Hypothesis: The Drone (Signal vs. Noise)
This is a classic signal processing problem.
The Signal (Drone): The dataset says drone_moving and drone_idle. A drone has multiple, extremely fast-spinning propellers (5000-6500 RPM). Each propeller is a high-frequency event generator, just like the fan but much faster and more chaotic. The drone's body is also moving, creating edge events.
Result: A small, localized area on the sensor that is constantly flooded with events. This is a high-density spatial cluster.
The Noise (Tree): The dataset mentions a "wobbling tree on the background".
Result: A leaf sways, creating a few events, then stops. A branch drifts, creating a sparse line of events. This activity is spatially and temporally sparse. It's "popcorn" noise.
Your plan to use DBSCAN is perfect because it's designed for this. DBSCAN's core idea is "find dense areas" and "label everything else as noise."
Conceptual Code (for Phase 3)
This is the core logic you'll add to the play_dat.py rendering loop. """

import numpy as np
import sklearn
from sklearn.cluster import DBSCAN

# Assume 'events' is a packet you just got from the evio iterator
# It contains events from the last --window (e.g., 20ms)
# events['x_coords'] [cite: 78], events['y_coords'] [cite: 80]

if len(events['x_coords']) > 50: # Only run if there's enough data
    
    # 1. Stack coordinates into an (N, 2) array
    X = np.stack([events['x_coords'], events['y_coords']], axis=1)

    # 2. Run DBSCAN
    # eps: Max distance between two points to be neighbors.
    #      This is the parameter you'll tune by *looking* at the data.
    #      Maybe 10 pixels?
    # min_samples: How many points to form a dense region.
    #      Maybe 20 points?
    db = DBSCAN(eps=10, min_samples=20, n_jobs=-1).fit(X)
    
    # 'labels' is an array. -1 means "noise". 0, 1, 2... are cluster IDs.
    labels = db.labels_
    
    # Find the largest cluster that isn't noise
    unique_labels, counts = np.unique(labels[labels != -1], return_counts=True)
    
    if len(counts) > 0:
        # Get the label of the biggest cluster
        main_cluster_label = unique_labels[np.argmax(counts)]
        
        # Get all points belonging to this cluster
        cluster_points = X[labels == main_cluster_label]
        
        # 3. Find the centroid
        # This is the drone's estimated position
        centroid = np.mean(cluster_points, axis=0)
        (centroid_x, centroid_y) = centroid.astype(int)
        
        # 4. (For visualization) Get a bounding box
        min_x, min_y = np.min(cluster_points, axis=0)
        max_x, max_y = np.max(cluster_points, axis=0)
        
        # Now, in your rendering code (e.g., OpenCV):
        # cv2.rectangle(frame, (min_x, min_y), (max_x, max_y), (0, 255, 0), 2)
        # cv2.circle(frame, (centroid_x, centroid_y), 5, (0, 0, 255), -1)