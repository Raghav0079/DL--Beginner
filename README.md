# Deep Learning for Beginners 🚀

A comprehensive collection of Jupyter notebooks covering fundamental deep learning concepts, implementations, and techniques. Perfect for beginners starting their journey in deep learning and neural networks.

## 📚 Table of Contents

- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Notebook Contents](#notebook-contents)
  - [Core Concepts](#core-concepts)
  - [Optimization Techniques](#optimization-techniques)
  - [Regularization Methods](#regularization-methods)
  - [Practical Applications](#practical-applications)
- [Getting Started](#getting-started)
- [Requirements](#requirements)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## 🔍 Overview

This repository contains hands-on implementations and explanations of essential deep learning concepts. Each notebook is designed to provide both theoretical understanding and practical coding experience, making complex topics accessible to beginners.

## 🛠️ Prerequisites

- Basic knowledge of Python programming
- Understanding of basic mathematics (linear algebra, calculus)
- Familiarity with NumPy and matplotlib
- Basic machine learning concepts (optional but helpful)

## 📖 Notebook Contents

### Core Concepts

| Notebook | Description |
|----------|-------------|
| [`perceptron.ipynb`](perceptron%20.ipynb) | Introduction to the fundamental building block of neural networks |
| [`perceptron-demo.ipynb`](perceptron-demo.ipynb) | Interactive demonstration of perceptron learning |
| [`backpropagation-regression.ipynb`](backpropogation%20-%20regression%20.ipynb) | Understanding backpropagation for regression problems |
| [`backpropagation-classification.ipynb`](backpropogation-classification.ipynb) | Backpropagation algorithm for classification tasks |

### Optimization Techniques

| Notebook | Description |
|----------|-------------|
| [`batch-vs-stochastic-gd.ipynb`](batch%20vs%20stochastic%20GD.ipynb) | Comparison of batch and stochastic gradient descent methods |
| [`feature-scaling.ipynb`](feature-scaling.ipynb) | Importance and techniques of feature scaling in deep learning |

### Regularization Methods

| Notebook | Description |
|----------|-------------|
| [`regularization.ipynb`](regularization.ipynb) | L1, L2 regularization and their effects on model performance |
| [`dropout-regression.ipynb`](dropout-regression.ipynb) | Dropout technique for preventing overfitting in regression |
| [`dropout-classification.ipynb`](dropout-classification.ipynb) | Dropout implementation for classification problems |
| [`early-stopping.ipynb`](early-stopping.ipynb) | Early stopping to prevent overfitting |

### Weight Initialization & Common Problems

| Notebook | Description |
|----------|-------------|
| [`xavier-he-initialization.ipynb`](xavier_he_initialization.ipynb) | Xavier and He weight initialization techniques |
| [`zero-initialization-relu.ipynb`](zero-initialization-relu.ipynb) | Problems with zero initialization when using ReLU activation |
| [`zero-initialization-sigmoid.ipynb`](zero-initialization-sigmoid.ipynb) | Zero initialization issues with sigmoid activation |
| [`vanishing-gradient.ipynb`](vanishing%20gradient.ipynb) | Understanding and solving the vanishing gradient problem |

### Practical Applications

| Notebook | Description |
|----------|-------------|
| [`mnist-classification.ipynb`](mnist-classification.ipynb.ipynb) | Classic handwritten digit recognition using MNIST dataset |
| [`customer-churn-prediction.ipynb`](customer%20churn%20prediction.ipynb) | Predicting customer churn using neural networks |
| [`graduate-admission-regression.ipynb`](graduate%20admission%20regression%20.ipynb) | Predicting graduate admission chances using regression |

## 🚀 Getting Started

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Raghav0079/DL--Beginner.git
   cd DL--Beginner
   ```

2. **Set up your environment:**
   ```bash
   # Using conda (recommended)
   conda create -n dl-beginner python=3.8
   conda activate dl-beginner
   
   # Or using venv
   python -m venv dl-beginner
   source dl-beginner/bin/activate  # On Windows: dl-beginner\Scripts\activate
   ```

3. **Install required packages:**
   ```bash
   pip install jupyter numpy pandas matplotlib seaborn scikit-learn tensorflow keras
   ```

4. **Start Jupyter Notebook:**
   ```bash
   jupyter notebook
   ```

## 📋 Requirements

- Python 3.7+
- Jupyter Notebook
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- TensorFlow/Keras
- Additional dependencies as specified in individual notebooks

## 💡 Usage

### Recommended Learning Path

1. **Start with basics:** `perceptron.ipynb` → `perceptron-demo.ipynb`
2. **Learn backpropagation:** `backpropagation-regression.ipynb` → `backpropagation-classification.ipynb`
3. **Understand optimization:** `batch-vs-stochastic-gd.ipynb` → `feature-scaling.ipynb`
4. **Master regularization:** `regularization.ipynb` → `dropout-*.ipynb` → `early-stopping.ipynb`
5. **Handle initialization:** `xavier-he-initialization.ipynb` → `zero-initialization-*.ipynb`
6. **Solve common problems:** `vanishing-gradient.ipynb`
7. **Apply to real projects:** `mnist-classification.ipynb` → `customer-churn-prediction.ipynb` → `graduate-admission-regression.ipynb`

### Running Individual Notebooks

Each notebook is self-contained and can be run independently. However, following the recommended learning path will provide the best educational experience.

```bash
# Open a specific notebook
jupyter notebook "perceptron .ipynb"
```

## 🎯 Learning Objectives

After completing these notebooks, you will understand:

- ✅ How neural networks learn through backpropagation
- ✅ Different optimization techniques and their trade-offs
- ✅ Regularization methods to prevent overfitting
- ✅ Proper weight initialization techniques
- ✅ Common problems in deep learning and their solutions
- ✅ How to apply these concepts to real-world problems

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

### How to Contribute

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is open source and available under the [MIT License](LICENSE).

## 🙏 Acknowledgments

- Inspired by various deep learning courses and tutorials
- Thanks to the open-source community for the amazing tools and libraries
- Special thanks to all contributors and learners who help improve this repository

## 📬 Contact

If you have any questions or suggestions, please feel free to reach out:

- GitHub: [@Raghav0079](https://github.com/Raghav0079)
- Issues: [Create an issue](https://github.com/Raghav0079/DL--Beginner/issues)

---

⭐ **If you find this repository helpful, please consider giving it a star!** ⭐

Happy Learning! 🎉