1. [Overview](#overview) 
2. [Features: Visual Metric Analyzer](#vma) 
3. [Features: Data Record Browser](#drb)
3. [Installation](#installation)
4. [Loading Data](#loading)
5. [Details on File Formatting](#formatting)
6. [Authors and Acknowledgments](#authors)
7. [License](#license)

# VEMOS

VEMOS (Visual Explorer for Metrics of Similarity) is a Python package that provides a graphical user interface for exploring and evaluating distance metrics.

In many applications, there can be many possible distance metrics that take into account different aspects of a complex data set. It's often necessary to test these metrics' ability to group, match, or classify items and to compare the performance of competing metrics. However, most existing packages for testing metrics do not link the analyses back to the original data records, which is crucial for spotting outliers, clusters, or trends in a metric's performance.

VEMOS provides analyses for exploring, evaluating, and comparing distance metrics while linking performance back to individual data records. With a user-friendly GUI, VEMOS makes it intuitive to follow the connections between data records and distance scores, for both broad analyses and in-depth evaluations of subsets of interest. It seamlessly integrates both data set exploration and distance metric evaluation and experimentation. 

<a name="features"></a>
## Overview

VEMOS consists of two interconnected interfaces, the Visual Metric Analyzer and the Data Record Browser. The Data Record Browser enables closer examination of the details of individual data records, including matches to other data records. The Metric Analyzer focuses on a broader exploration and analysis of the data set and the distance matrices, relating data visualizations back to individual records. 

<a name="vma"></a>
## Features: Visual Metric Analyzer

### Visualizations of the whole data set

#### Central Table

The central table displays the current distance matrix, where each entry is the similarity/dissimilarity score of a data record pair. 

##### Interactive features:
* Highlight table entries with the mouse/keyboard and right-click to view the corresponding record pairs.

##### Settings
* `Data Types to Display`: which data types to see when examining a record.
* `Matrices to Use in Analyses`: which distance matrices to include when performing analyses.
* `Display Format`: For incomplete matrices, it may be more convenient to view the scores as a list in the form (i, j, dist) rather than a table, where dist is the matrix entry for the pair of data records (i, j).

#### Heat Map
`Analyses ? Heat Map`

Displays the distance matrix as an image; the similarity/dissimilarity score of each pair of data records is a colored (x, y) point, with red being least similar and blue most similar. Provides a visual summary of the areas of similarity or dissimilarity in the data.

##### Interactive features:

* Click on a point in the heat map to view the corresponding record pairs.

#### Multidimensional Scaling
`Analyses ? MDS`

Since comparison metrics provide a measure of the similarity of data records, it often helps to view the records as points in space with the distance between them determined by their similarity. This makes it easier to identify clusters and outliers visually and note patterns in the spatial layout of the data. However, heterogeneous data sets often do not reside in a low-dimensional vector space. Instead, a distance matrix can be used to represent the data as a set of 2D or 3D points that have approximately the same distances between them using multi-dimensional scaling (MDS), which lets you examine the 2D or 3D spatial layout of the data records.

##### Interactive features:

* Auto-color the points according to different groupings to see how the spatial layout correlates with preexisting groups or clustering results
* Hover over a point to display its ID.
* Double-click on a point or select an area on the plot to display the corresponding data records. The Lasso/Rectangle option toggles between selecting a rectangular or freeform area.

### Clustering
`Clustering`

The data set can come with one or more user-defined groupings, but a grouping can also be generated by clustering algorithms. These data-driven algorithms group the data records based on their distances from each other. For many applications, it is critical to understand whether a given distance metric can lead to a clear grouping or clustering of the data records. Clusters are automatically stored as groupings in the data set and can later be used to color other visualizations by grouping. VEMOS uses the scikit-learn implementations of spectral and hierarchical clustering.

Use `Clustering Preferences` to perform clustering with your preferred settings.

##### Interactive features: Hierarchical clustering
* Hover over a node to display all records in that cluster.
* Clicking on a node to display all associated records in a new window.

##### Interactive features: Spectral clustering

* Hover over a point to display its ID.
* Double-click on a point or select an area on the plot to display the corresponding data records. The Lasso/Rectangle option toggles between selecting a rectangular or freeform area.
* `View`: View all records in that cluster.

### Binary Classification Performance

A common question when testing similarity metrics is whether the metric is effective in determining if two given items match�for example, whether the metric can determine that two fingerprints correspond to the same person. These questions amount to binary classification problems, as the metric must be able to classify scores as matches or nonmatches. VEMOS provides several analyses of binary classification performance:

#### Histogram
`Binary Classification Performance` ? `Histogram`

Displays the distribution of similarity/dissimilarity scores for matched and unmatched data records. Greater separation between the distributions of matched and unmatched scores indicates that the metric is better at discriminating between matches and nonmatches.

##### Interactive features:

* Hover to display the score at that point and to display the quantity and percent of match and nonmatch scores to either side of that score.
* Click to pin the score to the figure.

#### Smooth Histogram
`Binary Classification Performance` ? `Smooth Histogram`

Displays the distribution of similarity/dissimilarity scores for matched and unmatched records, smoothed using Gaussian kernel density estimation.

##### Interactive features:

* Hover to display the score at that point and the height of the match and nonmatch distributions at that point, and display the the quantity and percent of matched and unmatched scores to either side of that score.
*  Click to pin the score to the figure.


#### Linear Ordering
`Binary Classification Performance` ? `Linear Ordering`

Plots the scores of all matched and unmatched record pairs on two lines, making it easier to spot outliers or see the distribution of a smaller set of scores.

##### Interactive features:
* Hover to display the score at that point and the quantity and percent of match and nonmatch scores to either side of that score.
* Click to pin the score to the plot.

#### ROC Curve
`Binary Classification Performance` ? `ROC Curve`

The receiver operating characteristic (ROC) curve plots the true positive rate against the false positive rate when different thresholds are chosen for distinguishing matches from non-matches. The curve�s proximity to the upper left of the plot, measured by the area under the curve (AUC), indicates the accuracy of the algorithm. 

##### Interactive features:
* Hover over the plot to display the true and false positive rates at that point. 
* Hover over a threshold to display information about the quantity and percent of match and nonmatch scores to either side of that threshold.
* Click to pin the true and false positive rates to the figure.
* Highlight an area on the plot to display all scores in that range.
* `Show on Single Plot/Show on Separate Plots`:  Display the ROC curves for multiple matrices (if present) on the same plot or multiple plots.

#### Statistics
`Binary Classification Performance` ? `Statistics`

Displays the standard deviation and mean and median score of each matrix.

### Combining Metrics and Generating Matrices

#### Matrix Generation

`Edit Matrices` ? `Generate Matrix`

Although some data sets come with matrices of similarity scores under different metrics already provided, some data sets may include image files but no preexisting scores to compare them. In other cases, although the data set may already contain scores under some metrics, it may help to compare a custom image comparison metric to a standard one. VEMOS provides the ability to create distance matrices using standard image comparison techniques.

##### Metric Options
* Structural similarity (SSIM)
* Mean squared error (MSE)
* Normalized root mean squared error (NRMSE)
* Hamming distance: ? A(i, j) ? B(i, j) for pixels (i, j) in each image pair A, B
* Euclidean distance: Calculates the Euclidean or Frobenius norm of the difference between corresponding pixels in the two images.

SSIM, MSE, and MRNSE use the scikit-learn implementation.

#### Combining Metrics
`Edit Matrices` ? `Fuse Matrices`

Individual metrics may take into account different features of the data records. For example,one metric may calculate the similarity of leaves based on the shapes of their boundary curves,while another may examine the colors of the images. Thus, combining these metrics to create a fused metric that accounts for both of these features may result in increased accuracy.

VEMOS provides the functionality to combine metrics to improve separation between matched and unmatched scores. Each pair of records is a data point classified as matching or non-matching, and its scores under each selected metric form its feature vector. The data points can then be represented in high-dimensional space. Support vector classification then searches for a hyperplane in that space that provides the best boundary between matching and non-matching data points, acting as a classifier. The decision function then gives a data point�s distance from the hyperplane (positive if classified as a match, and negative if not).  Adding the minimum distance to this matrix of distances results in the non-negative fused matrix. By taking information from multiple metrics into account, this new matrix can improve retrieval performance.

##### Options
Select a linear, polynomial, or radial basis function kernel for support vector classification, which determines the shape of the classifier.

### Settings
`Settings`

* `Data Types to Display`: which data types to see when examining a record.
* `Matrices to Use in Analyses`: which distance matrices to include when performing analyses.
* `Display Format`: For incomplete matrices, it may be more convenient to view the scores as a list in the form (i, j, dist) rather than a table, where dist is the matrix entry for the pair of data records (i, j).

<a name="drb"></a>
## Features: Data Record Browser
The Data Record Browser lets users examine individual data records more closely. Browse the data files and view the records� matches and groups while also gaining an overview of a record�s similarity to other records under different metrics. You can also assign matches and groups for later use in the Visual Metric Analyzer.

<a name="installation"></a>
## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install VEMOS.

1. Install required modules, if necessary:
```bash
pip install pyqt5
pip install numpy
pip install scipy
pip install matplotlib
pip install scikit-image
pip install scikit-learn
```

2. Install VEMOS:
```bash
pip install vemos
```

3. Run VEMOS:
```bash
python -c "import vemos.__init__ as I; I.run()"
```

<a name="loading"></a>
## Loading Data
VEMOS works with (a) a set of data files and/or (b) comparison scores under one or more metrics.

Though both the Data Record Browser and the Visual Metric Analyzer are designed to work with matrices *and* scores, the Data Record Browser only *requires* data files and the Visual Metric Analyzer only *requires* scores.

### Loading Data Files
#### Loading from a Directory

If loading from a file directory, the data may be organized using:

(a) Named folders: The directory of files should contain subdirectories with the names of the data types.  Check the box `Read Data Types from Folders`.

(b) File naming conventions: If the data types aren't in the folder names, then individual files should have different names/file extensions for each data type. Under `Set Data Types`, list the conventions for each data type.

#### Loading from a Description FIle
If the files aren't stored in an organized way, they can be manually organized using a description file. A description file should be a text file in the format:
ID; (group_1, group_2); (match1, match2,...); data_type_1: file_1; data_type_2: file_2; ...

### Loading Matrices
Add matrices to the `Load Similarity/Dissimilarity Matrices` box.
`Matrix Type` lets you specify whether the matrix contains similarity or dissimilarity scores.
`Format` lets you specify whether the file is a matrix or list of scores (see "Accepted Score Matrix Formats" below for details).

<a name="formatting"></a>
## Details on File Formatting

### Accepted data file formats:
 
+ Images and binary masks/segmentations: All formats that can be processed with matplotlib.image.imread, which includes .png, .jpg, .jpeg, .tif, and .tiff
+ Curves and point clouds: Text file with a list of points in the format
  ```bash
  x1, y1
  x2, y2
  ```
+ Text description: Text file.


### Accepted score matrix formats:

+ Recommended for complete matrices: Text or CSV file of a 2D matrix of scores, where each row and column corresponds to a data record.
+ Recommended for incomplete matrices: Text or CSV file in tabular form, as below:

  |   |   | Metric 1 | Metric 2 | Ground Truth|
  |---|---|----------|----------|-------------|
  |ID1|ID2|.123      |.234      |Y            |
  |ID1|ID4|.567      |.678      |N            |

  such that in the first row, all but the first two columns are the names of the metrics; in each subsequent row, the first two columns are the IDs of the records being compared, the last column stores the ground-truth value, if available (i.e., whether or not the two records are matches; `Y` if so and `N` if not), and the other entries are scores under the metrics.



Asymmetric matrices can be symmetrized using the average, minimum, or maximum of entry (i,j) and entry (j, i).

<a name="authors"></a>
## Authors and acknowledgment
VEMOS was made by Eve Fleisig under the guidance of Gunay Dogan. The work took place at the National Institute of Standards and Technology.

Special thanks to Martin Herman, Hari Iyer, Steve Lund, Yooyoung Lee, Gautham Venkatasubramanian, and the rest 
of the NIST Footwear Forensics Group for their contributions, suggestions, and support.

<a name="license"></a>
## License
[BSD](https://opensource.org/licenses/BSD-3-Clause)