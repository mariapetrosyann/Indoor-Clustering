# WiFi Signal-Based Indoor Environment Clustering and Classification

## Introduction
This project utilizes a WiFi signals dataset to classify and cluster indoor environments into specific rooms: Kitchen, Hallway, Livingroom, and Patio. Initially, a Decision Tree Classifier was used for supervised classification, followed by a K-Means algorithm for unsupervised clustering to identify inherent patterns in the data.

## Getting Started

### Prerequisites
Ensure you have the following libraries installed:
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn

You can install these libraries using pip:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

### Installation
1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/Indoor-Clustering.git
    ```
2. Navigate to the project directory:
    ```bash
    cd IndoorClustering
    ```
3. Open the Jupyter Notebook `IndoorClustering.ipynb` in your preferred environment.

## Running the Project

### Running Locally
1. Ensure all required libraries are installed.
2. Open and run all cells in the Jupyter Notebook to execute the code, visualize data, and evaluate the model.

## Libraries and Functions Used

- **pandas**: For data manipulation and analysis.
- **numpy**: For numerical computations.
- **matplotlib**: For creating static, animated, and interactive visualizations.
- **seaborn**: For statistical data visualization.
- **scikit-learn**: For machine learning algorithms, model evaluation metrics, and data preprocessing.

## Project Steps and Outputs

### Dataset Description
Provided an overview of the dataset to understand its structure and statistical properties.

### Data Analysis and Visualization
Generated a correlation matrix to understand the relationships between different features.

### Data Preprocessing and Decision Tree Classification
Used a Decision Tree Classifier to classify the data and analyzed the confusion matrix.

#### Confusion Matrix Interpretation
- Hallway is misclassified as Livingroom 7 times.
- Kitchen is correctly classified with no errors.
- Livingroom has 2 instances misclassified as Hallway, 2 as Kitchen, and 2 as Patio.
- Patio is misclassified as Kitchen 1 time.

#### Overall Performance
- Accuracy: 98.33%
- Precision, Recall, and F1 Score for each class demonstrate a good balance between precision and recall.

### Data Clustering with K-Means Algorithm
Trained the model using the K-Means algorithm, excluding the target column, and analyzed the WCSS plot.

#### Observations from the WCSS Plot
- The clear "elbow" point is at 4 clusters, aligning with the pre-defined expectation of having 4 clusters in the data.

### Mismatched Train Results after Clustering
Identified instances where the original 'Target' values do not match the duplicated 'Targetu' values.

## Results

### Classification Metrics
- **Accuracy**: 98.33%
- **Precision**:
  - Class 1 (Kitchen): 99.28%
  - Class 2 (Hallway): 98.17%
  - Class 3 (Livingroom): 97.30%
  - Class 4 (Patio): 98.66%
- **Recall**:
  - Class 1 (Kitchen): 97.18%
  - Class 2 (Hallway): 100.00%
  - Class 3 (Livingroom): 96.64%
  - Class 4 (Patio): 99.32%
- **F1 Score**:
  - Class 1 (Kitchen): 98.22%
  - Class 2 (Hallway): 99.08%
  - Class 3 (Livingroom): 96.97%
  - Class 4 (Patio): 98.99%

## Conclusion
The project demonstrates the effectiveness of using Decision Tree Classifier for supervised classification and K-Means for unsupervised clustering in the context of indoor room classification based on WiFi signal data. The model shows high accuracy and balanced precision, recall, and F1 scores, indicating a well-performing classification model. The WCSS plot further supports the choice of 4 clusters, aligning with the room categories.
