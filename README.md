# neural-network-challenge-2
This activity aims to predict employee attrition using neural networks. The dataset used contains various attributes of employees such as age, job role, marital status, etc., along with the target variable 'Attrition', which indicates whether an employee has left the company or not. Additionally, the 'Department' variable is included to predict the department to which an employee belongs.

## Dependencies
The following Python libraries are used in this project:

pandas: For data manipulation and analysis.
numpy: For numerical computations.
scikit-learn: For machine learning tasks such as data preprocessing and model evaluation.
tensorflow: For building and training neural networks.

## Dataset
The dataset used in this project is sourced from a CSV file available at the following URL: [attrition.csv](https://static.bc-edx.com/ai/ail-v-1-0/m19/lms/datasets/attrition.csv). It contains various attributes of employees such as age, job role, marital status, etc., along with the target variable 'Attrition' indicating whether an employee has left the company or not.

## Data Preprocessing
-Feature Selection: A subset of columns is selected as features for the model. These columns include attributes such as age, distance from home, education, etc.
-One-Hot Encoding: Categorical variables ('JobRole', 'MaritalStatus', 'WorkLifeBalance') are one-hot encoded to convert them into numerical format suitable for machine learning models.
-Normalization: The numerical features are standardized using StandardScaler to bring them to a similar scale, which helps in improving model performance.

## Model Architecture

The neural network model architecture consists of the following components:

-Input Layer: An input layer with the number of neurons equal to the number of features.
-Shared Layers: Two shared dense layers with ReLU activation, allowing the network to learn representations common to both prediction tasks.
-Department Branch: This branch predicts the department to which an employee belongs. It consists of a hidden layer and an output layer with softmax activation.
-Attrition Branch: This branch predicts whether an employee will leave the company or not. Similar to the department branch, it consists of a hidden layer and an output layer with softmax.

## Training

The model is trained using the training data split into features (X_train_categorical_encoded) and labels (y_train_department_encoded and y_train_encoded). It is compiled with the Adam optimizer and categorical cross-entropy loss for both department and attrition branches. The training process involves 50 epochs with a batch size of 64.

## Evaluation
The model is evaluated using the testing data (X_test_categorical_encoded). Evaluation metrics include accuracy for both department and attrition predictions. The evaluation results provide insights into the model's performance in predicting employee attrition and department allocation.