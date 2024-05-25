
### Heart Disease Prediction using Machine Learning

This project focuses on predicting heart disease using machine learning models and a neural network. The dataset used contains various features such as age, sex, cholesterol levels, and other clinical parameters that are believed to be correlated with heart disease. The goal is to analyze these features and build models that can accurately predict the presence or absence of heart disease in patients.

## Acknowledgements

This project was based on a code by Shreekant Gosavi. The original code can be found https://github.com/g-shreekant/Heart-Disease-Prediction-using-Machine-Learning.git. 
I have made the following changes to the original code:
- Enhanced data visualization
- Optimized model training loops
- Added more detailed exploratory data analysis
- Using Visual Studio Code instead of Jupyter Notebook


### Key Features

1. **Data Visualization**: The dataset is visualized using various techniques such as count plots, histograms, and bar plots to analyze the distribution of features and their relationship with the target variable (heart disease presence).
  
2. **Machine Learning Models**: Several popular classification algorithms are implemented and evaluated, including Logistic Regression, Naive Bayes, Support Vector Machine, K-Nearest Neighbors, Decision Tree, Random Forest, and XGBoost. Each model is trained on the dataset and evaluated using accuracy scores to determine its effectiveness in predicting heart disease.

3. **Neural Network Implementation**: A neural network is built using TensorFlow and Keras. The model architecture includes an input layer, hidden layers with ReLU activation, and an output layer with a sigmoid activation function. The neural network is trained on the dataset and evaluated for its accuracy in predicting heart disease.

### Usage

1. **Dataset**: The dataset (`heart.csv`) used in this project contains 303 instances and 14 columns, where each row corresponds to a patient. Columns include clinical features and the target variable, indicating the presence of heart disease.
  
2. **Requirements**: Ensure you have Python installed with necessary libraries such as pandas, numpy, scikit-learn, seaborn, matplotlib, TensorFlow, and xgboost. These can be installed using `pip install -r requirements.txt`.

3. **Execution**: Run the main script (`heart_disease_prediction.py`) to load the dataset, visualize the data, train machine learning models, and evaluate their performance. The script concludes with a summary of accuracy scores for each model.

### Conclusion

This project serves as a practical example of applying machine learning techniques to predict heart disease based on clinical parameters. The comprehensive analysis includes data visualization, model training, and evaluation, showcasing the effectiveness of different algorithms and a neural network in this predictive task. By exploring and utilizing the provided code and models, you can gain insights into the predictive modeling process for heart disease and apply these techniques to similar healthcare datasets.
