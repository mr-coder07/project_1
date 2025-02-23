import numpy as np

class LSTMCell:
    def __init__(self, input_size, hidden_size, output_size=1, learning_rate=0.001):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.W_f = np.random.randn(hidden_size, hidden_size + input_size) * 0.01
        self.b_f = np.zeros((hidden_size, 1))
        self.W_i = np.random.randn(hidden_size, hidden_size + input_size) * 0.01
        self.b_i = np.zeros((hidden_size, 1))
        self.W_C = np.random.randn(hidden_size, hidden_size + input_size) * 0.01
        self.b_C = np.zeros((hidden_size, 1))
        self.W_o = np.random.randn(hidden_size, hidden_size + input_size) * 0.01
        self.b_o = np.zeros((hidden_size, 1))
        self.W_y = np.random.randn(output_size, hidden_size) * 0.01
        self.b_y = np.zeros((output_size, 1))
        self.h = np.zeros((hidden_size, 1))
        self.C = np.zeros((hidden_size, 1))

    def sigmoid(self, x):
        x = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-x))

    def forward(self, x):
        combined = np.concatenate((self.h, x), axis=0)
        f_t = self.sigmoid(np.dot(self.W_f, combined) + self.b_f)
        i_t = self.sigmoid(np.dot(self.W_i, combined) + self.b_i)
        C_tilde = np.tanh(np.dot(self.W_C, combined) + self.b_C)
        self.C = f_t * self.C + i_t * C_tilde
        o_t = self.sigmoid(np.dot(self.W_o, combined) + self.b_o)
        self.h = o_t * np.tanh(self.C)
        return self.h

    def predict_output(self):
        y = np.dot(self.W_y, self.h) + self.b_y
        return y

    def train(self, input_sequences, target_sequence, epochs=100):
        m = target_sequence.shape[1]
        for epoch in range(epochs):
            total_loss = 0
            dW_f = np.zeros_like(self.W_f)
            dW_i = np.zeros_like(self.W_i)
            dW_C = np.zeros_like(self.W_C)
            dW_o = np.zeros_like(self.W_o)
            dW_y = np.zeros_like(self.W_y)
            db_f = np.zeros_like(self.b_f)
            db_i = np.zeros_like(self.b_i)
            db_C = np.zeros_like(self.b_C)
            db_o = np.zeros_like(self.b_o)
            db_y = np.zeros_like(self.b_y)
            for t in range(m):
                x_t = input_sequences[:, t].reshape(-1, 1)
                h_t = self.forward(x_t)
                y_pred = self.predict_output()
                loss = 0.5 * (y_pred - target_sequence[:, t])**2
                total_loss += loss
                dy = y_pred - target_sequence[:, t]
                dW_y += np.dot(dy, self.h.T)
                db_y += dy
                dh = np.dot(self.W_y.T, dy)
                do = dh * np.tanh(self.C)
                dC = dh * self.h * (1 - np.tanh(self.C)**2)
                di = dC * self.C
                df = dC * self.C
                dC_tilde = dC * self.h
                combined_input = np.concatenate((self.h, x_t), axis=0)
                dcombined = np.dot(self.W_f.T, df) + np.dot(self.W_i.T, di) + np.dot(self.W_C.T, dC_tilde) + np.dot(self.W_o.T, do)
                dW_f += np.dot(df * (self.h - self.C), combined_input.T)
                db_f += df
                dW_i += np.dot(di * (self.h - self.C), combined_input.T)
                db_i += di
                dW_C += np.dot(dC_tilde * (self.h - self.C), combined_input.T)
                db_C += dC_tilde
                dW_o += np.dot(do * (self.h - self.C), combined_input.T)
                db_o += do
            self.W_f -= self.learning_rate * dW_f
            self.b_f -= self.learning_rate * db_f
            self.W_i -= self.learning_rate * dW_i
            self.b_i -= self.learning_rate * db_i
            self.W_C -= self.learning_rate * dW_C
            self.b_C -= self.learning_rate * db_C
            self.W_o -= self.learning_rate * dW_o
            self.b_o -= self.learning_rate * db_o
            self.W_y -= self.learning_rate * dW_y
            self.b_y -= self.learning_rate * db_y
            print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss[0, 0] / m}")

    def predict(self, input_sequences):
        m = input_sequences.shape[1]
        predictions = []
        for t in range(m):
            x_t = input_sequences[:, t].reshape(-1, 1)
            h_t = self.forward(x_t)
            y_pred = self.predict_output()
            predictions.append(y_pred.flatten())
        return np.array(predictions)
