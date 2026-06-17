import numpy as np

class TDRC:
	def __init__(self, features, params):
		self.features = features
		self.params = params
		self.gamma = params['gamma']
		self.alpha = params['alpha']
		self.beta = params['beta']
		self.eta = params.get('eta', 1)

		self.w = np.zeros(features,dtype=np.float64)
		self.h = np.zeros(features,dtype=np.float64)

	def update(self, x, a, r, xp, rho):
		#print(x,xp)
		x=np.clip(x,-1e6,1e6).astype(np.float64)
		xp=np.clip(xp,-1e6,1e6).astype(np.float64)
		v = self.w.dot(x)
		vp = self.w.dot(xp)
		delta   = np.clip(r + self.gamma * vp - v, -50000, 50000)
		delta_hat = np.clip(self.h.dot(x),         -50000, 50000)
		dw = rho * (delta * x - self.gamma * delta_hat * xp)
		dh = (rho * delta - delta_hat) * x - self.beta * self.h
		dw = np.clip(dw, -500, 500)
		dh = np.clip(dh, -500, 500)
		self.w = self.w + self.alpha * dw
		self.h = self.h + self.eta * self.alpha * dh
	def getWeights(self):
		return self.w
