# Customer Segmentation Using Clustering Algorithms

## Aim
To implement K-Means, Agglomerative Clustering, and DBSCAN on the Mall Customers dataset and evaluate clustering quality.

## Dataset
Mall Customers dataset from Kaggle:  
https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python

Features used: Annual Income and Spending Score.

## Technologies Used
- Python
- Pandas
- Matplotlib
- Scikit-learn
- KaggleHub

## Algorithms
1. K-Means Clustering
2. Agglomerative Clustering
3. DBSCAN

## Evaluation Metrics
- Silhouette Score: Higher is better.
- Davies–Bouldin Index: Lower is better.

## How to Run

Install dependencies:

```bash
pip install kagglehub pandas matplotlib scikit-learn
```

Run the program:

```bash
python customer_clustering.py
```

## Output
- Clustering performance comparison table.
- Scatter plots of customer groups.
- `Customer_Clustering_Results.csv` containing cluster assignments.

## Conclusion
The project groups customers based on income and spending behavior and compares three clustering algorithms using evaluation metrics and visualizations.
