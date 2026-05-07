import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib

def main():
    print("Loading data...")
    # Load dataset
    df = pd.read_csv('creditcard.csv')
    
    print("Balancing dataset...")
    # Separate the classes
    legit = df[df.Class == 0]
    fraud = df[df.Class == 1]
    
    # Under-sample the legitimate transactions to balance the classes
    # Fraud transactions are around 492
    legit_sample = legit.sample(n=len(fraud), random_state=42)
    
    # Concatenate the two DataFrames
    new_df = pd.concat([legit_sample, fraud], axis=0)
    
    print("Preparing training data...")
    # Split features and targets
    X = new_df.drop(columns='Class', axis=1)
    Y = new_df['Class']
    
    # Split the data into Training data & Testing Data
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=42)
    
    print("Training model...")
    # Train Logistic Regression model
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, Y_train)
    
    print("Evaluating model...")
    # Accuracy on training data
    train_pred = model.predict(X_train)
    train_acc = accuracy_score(Y_train, train_pred)
    print(f'Accuracy on Training data : {train_acc:.4f}')
    
    # Accuracy on test data
    test_pred = model.predict(X_test)
    test_acc = accuracy_score(Y_test, test_pred)
    print(f'Accuracy on Test data : {test_acc:.4f}')
    
    print("Classification Report:")
    print(classification_report(Y_test, test_pred))
    
    print("Saving model...")
    # Save the model
    joblib.dump(model, 'model.pkl')
    print("Model saved to model.pkl successfully!")

if __name__ == '__main__':
    main()
