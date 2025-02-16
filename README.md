# README

## Adversarial Training and Poison Detection with TensorFlow

### Overview
This project implements a neural network model using TensorFlow to detect poisoned samples in a dataset. It also enhances robustness by generating adversarial examples using the Fast Gradient Sign Method (FGSM) and retraining the model on both clean and adversarial samples. The script:
- Loads and preprocesses training and test datasets
- Builds and trains a simple neural network classifier
- Generates adversarial examples to improve model robustness
- Retrains the model with both clean and adversarial samples
- Evaluates the model using AUC-ROC
- Saves predictions to a CSV file

### Requirements
Ensure you have the following dependencies installed before running the script:
```bash
pip install numpy pandas tensorflow scikit-learn
```

### Dataset
The script expects two CSV files:
- `train.csv`: Contains training data with labels (`poisoned` column)
- `test.csv`: Contains test data without labels

Both datasets should have a column named `index`, which is dropped during processing.

### Running the Script
Run the script using:
```bash
python script.py
```

### Steps in the Script
1. **Data Loading & Preprocessing**
   - Reads `train.csv` and `test.csv`.
   - Splits features (`X_train`) and labels (`y_train`).
   - Normalizes features using `StandardScaler`.

2. **Model Building & Training**
   - Defines a simple feedforward neural network with ReLU activations.
   - Trains the model using binary cross-entropy loss and Adam optimizer.

3. **Generating Adversarial Examples (FGSM Attack)**
   - Computes the gradient of the loss w.r.t input features.
   - Perturbs inputs slightly to create adversarial samples.

4. **Adversarial Training**
   - Retrains the model with a mix of original and adversarial examples.

5. **Model Evaluation & Predictions**
   - Computes AUC-ROC score on training data.
   - Outputs predictions for submission in `submission.csv`.

### Output
- `submission.csv`: Contains predicted `poisoned` labels for the dataset.

### Future Improvements
- Experiment with different adversarial attack strengths (`epsilon`).
- Fine-tune model hyperparameters.
- Evaluate on additional adversarial attack methods.

### Author
William James

