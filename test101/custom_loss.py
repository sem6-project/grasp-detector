from keras.backend as K
import numpy as np
from utils import IOU

def rectangle_guessing_loss(y_true, y_pred):
	y_true_np = K.eval(y_true)
	y_pred_np = K.eval(y_pred)
	y_true_iter = np.nditer(y_true_np, order='C')
	y_pred_iter = np.nditer(y_pred_np, order='C')
	loss = 0.0
	count = 0
	for y1, y2 in zip(y_true_iter, y_pred_iter):
		loss.append(IOU(y1, y2))
		count += 1
	
	return (loss / count)
