from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, recall_score
import config

def build_model(input_shape):
    """
    Builds the Sequential neural network model for binary fire prediction.
    """
    model = Sequential([
        Dense(128, activation='relu', input_shape=(input_shape,)),
        Dropout(config.DROPOUT_RATE),
        Dense(64, activation='relu'),
        Dropout(config.DROPOUT_RATE),
        Dense(32, activation='relu'),
        Dropout(config.DROPOUT_RATE),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=config.LEARNING_RATE),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model

def train_model(model, x_train, y_train, x_test, y_test):
    """
    Trains the deep learning model.
    """
    print(f"Training model for {config.EPOCHS} epochs with batch size {config.BATCH_SIZE}...")
    
    history = model.fit(
        x_train, y_train,
        epochs=config.EPOCHS,
        batch_size=config.BATCH_SIZE,
        validation_data=(x_test, y_test),
        verbose=1
    )
    return history

def evaluate_model(model, x_test, y_test):
    """
    Evaluates the model on the test set and prints metrics.
    """
    y_pred_prob = model.predict(x_test)
    y_pred = (y_pred_prob > 0.5).astype(int)
    
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    print("\n--- Model Evaluation ---")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Recall:   {recall:.4f}")
    print("Confusion Matrix:")
    print(conf_matrix)
    print("------------------------")
    
    return {'accuracy': accuracy, 'f1': f1, 'recall': recall, 'confusion_matrix': conf_matrix}
