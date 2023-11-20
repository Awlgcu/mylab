import numpy as np
import blzs.Hebb as Hebb
x1 = np.array([[1], [-1], [1], [1]])
x2 = np.array([[-1], [1], [1], [-1]])
y1 = np.array([[1], [-1], [0], [0]])
y2 = np.array([[-1], [1], [0], [0]])
hebb_x = np.zeros((4, 2))
hebb_x[:, 0] = x1[:, 0]
hebb_x[:, 1] = x2[:, 0]
hebb = Hebb.Hebb(4,4)
hebb.train(hebb_x, hebb_x, alpha=0.1)
pred_y1 = hebb.prediction(y1)
print(pred_y1)
pred_y2 = hebb.prediction(y2)
print(pred_y2)