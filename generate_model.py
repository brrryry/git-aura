import torch
import numpy as np
import pandas as pd

torch.manual_seed(69)
np.random.seed(69)

INPUT_LAYERS = 9
OUTPUT_LAYERS = 1
HIDDEN_LAYERS = [27, 18, 6]

LEARNING_RATE = 0.001
NUM_EPOCHS = 10000

class NN:
    def __init__(self, model_path = None):
        if model_path is None:
            # make the model here
            self.model = torch.nn.Sequential(
                torch.nn.Linear(INPUT_LAYERS, HIDDEN_LAYERS[0]),
                torch.nn.ReLU(),
                torch.nn.Linear(HIDDEN_LAYERS[0], HIDDEN_LAYERS[1]),
                torch.nn.ReLU(),
                torch.nn.Linear(HIDDEN_LAYERS[1], HIDDEN_LAYERS[2]),
                torch.nn.ReLU(),
                torch.nn.Linear(HIDDEN_LAYERS[2], OUTPUT_LAYERS),
                torch.nn.Sigmoid()
            )
        else:
            self.model = torch.load(model_path)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()

        self.criterion = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=LEARNING_RATE)

    def train(self, X, y):
        X = torch.from_numpy(X).to(self.device)
        y = torch.from_numpy(y).to(self.device)


        for epoch in range(NUM_EPOCHS):
            # Forward pass
            y_pred = self.model(X)

            loss = self.criterion(y_pred, y)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if epoch % 100 == 0:
                print(f'Epoch {epoch} loss: {loss.item()}')
            
    def predict(self, X):
        X = torch.tensor(X, dtype=torch.float32).to(self.device)
        y_pred = self.model(X)
        return y_pred.cpu().detach().numpy()

    
    def save_model(self, path):
        torch.save(self.model, path)


if __name__ == '__main__':
    # Load the data
    df = pd.read_csv('data/users_final.csv')
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values

    # Remove the login, link column

    X = np.delete(X, 0, axis=1)
    X = np.delete(X, 0, axis=1)


    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)

    # Remove all the rows with NaN values
    mask = ~np.isnan(X).any(axis=1)
    X = X[mask]
    y = y[mask]



    # apply sigmoid function to y
    y = np.expand_dims(y, axis=1)
    y = 1 / (1 + np.exp(-y))

    # Split the data into training and testing
    mask = np.random.rand(len(X)) < 0.8
    X_train = X[mask]
    y_train = y[mask]
    X_test = X[~mask]
    y_test = y[~mask]


    # Train the model
    model = NN()
    model.train(X_train, y_train)

    # make a histogram of the difference between the predicted and the actual values
    y_pred = model.predict(X_test)
    y_pred = y_pred.flatten()
    y_test = y_test.flatten()

    import matplotlib.pyplot as plt
    plt.hist(y_pred - y_test, bins=50)
    plt.show()






    # Save the model
    model.save_model('model/model.pth')