# DL--Beginner

A beginner-friendly repository for learning Deep Learning fundamentals, starting with the basics of neural networks.

## 📚 Contents

### Perceptron Implementation

This repository contains Jupyter notebooks demonstrating the implementation and visualization of the Perceptron algorithm, one of the simplest forms of artificial neural networks.

#### Files:
- **perceptron-demo.ipynb** - Complete demonstration of Perceptron classification with visualization

## 🎯 Project Overview

The project demonstrates:
- **Binary Classification**: Using Perceptron to classify student placement packages based on CGPA
- **Data Preprocessing**: Converting continuous target values into binary categories using median thresholding
- **Model Training**: Implementing sklearn's Perceptron algorithm
- **Visualization**: Decision boundary plotting using mlxtend library

## 🔧 Technologies Used

- **Python 3.x**
- **NumPy** - Numerical computing
- **Pandas** - Data manipulation and analysis
- **Matplotlib** - Data visualization
- **Seaborn** - Statistical data visualization
- **Scikit-learn** - Machine learning algorithms
- **mlxtend** - Machine learning extensions for decision boundary visualization
- **gdown** - Google Drive file downloading

## 📊 Dataset

The project uses a student placement dataset with:
- **Feature**: CGPA (Cumulative Grade Point Average)
- **Target**: Package (Salary package) - converted to binary classification
- Classification threshold: Median package value

## 🚀 Getting Started

### Prerequisites

Install the required dependencies:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn mlxtend gdown
```

### Running the Notebooks

1. Clone the repository:
```bash
git clone https://github.com/Raghav0079/DL--Beginner.git
cd DL--Beginner
```

2. Open the notebook in Jupyter or Google Colab:
```bash
jupyter notebook perceptron-demo.ipynb
```

3. Run the cells sequentially to:
   - Load and explore the dataset
   - Visualize the relationship between CGPA and package
   - Train the Perceptron model
   - Evaluate model performance
   - Visualize the decision boundary

## 📈 Model Performance

The Perceptron model is trained to classify whether a student will receive a package above or below the median value based on their CGPA. Model accuracy is evaluated on a 20% test split.

## 🔍 Key Concepts Covered

- **Perceptron Algorithm**: Understanding the basic building block of neural networks
- **Binary Classification**: Converting regression problems to classification
- **Train-Test Split**: Proper model evaluation methodology
- **Decision Boundaries**: Visualizing how the model separates different classes
- **Model Evaluation**: Using accuracy metrics to assess performance

## 📝 Learning Outcomes

By working through this repository, you will learn:
1. How to implement a Perceptron classifier
2. Data preprocessing for binary classification
3. Model training and evaluation techniques
4. Visualization of machine learning decision boundaries
5. Practical application of linear classifiers

## 🤝 Contributing

This is a learning repository. Feel free to fork, experiment, and suggest improvements!

## 📧 Contact

**Raghav** - [GitHub Profile](https://github.com/Raghav0079)

## 📄 License

This project is open source and available for educational purposes.

---

*Happy Learning! 🎓*
