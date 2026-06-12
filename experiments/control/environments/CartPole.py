"""
Classic cart-pole system implemented by Rich Sutton et al.
Copied from http://incompleteideas.net/sutton/book/code/pole.c
permalink: https://perma.cc/C9ZM-652R
"""
import math
import numpy as np
from RlGlue import BaseEnvironment

BACK = 0
FORWARD = 1

class CartPole(BaseEnvironment):
	def __init__(self):
		self.position = -0.6 + np.random.random() * 0.6
		self.velocity = 0.0
		self.pole_angle=-0.6 + np.random.random() * 0.6
		self.pole_angle_velocity=0
		self.gravity = 9.8
		self.masscart = 1.0
		self.masspole = 0.1
		self.total_mass = self.masspole + self.masscart
		self.length = 0.5  # actually half the pole's length
		self.polemass_length = self.masspole * self.length
		self.force_mag = 10.0
		self.tau = 0.02  # seconds between state updates
		self.kinematics_integrator = "euler"
		self.theta_threshold_radians = 12 * 2 * math.pi / 360
		self.x_threshold = 2.4
		self.features = 4
		self.num_actions = 2
		high = np.array([
			self.x_threshold * 2,
			np.inf,
			self.theta_threshold_radians * 2,
			np.inf,
		],
		dtype=np.float32,
		)

	def start(self):
		self.position = -0.6 + np.random.random() * 0.6
		self.velocity = 0.0
		self.pole_angle=-0.6 + np.random.random() * 0.6
		self.pole_angle_velocity=0
		return (self.position, self.velocity,self.pole_angle,self.pole_angle_velocity)

	# give all actions for a given state
	def actions(self, s):
		return [BACK, FORWARD]

	# give the rewards associated with a given state, action, next state tuple
	def rewards(self, s, a, sp):
		return 1

	# get the next state and termination status
	def next_state(self, s, a):
		action = a
		x, x_dot, theta, theta_dot = s
		force = self.force_mag if action == 1 else -self.force_mag
		costheta = np.cos(theta)
		sintheta = np.sin(theta)
		temp = (
			force + self.polemass_length * np.square(theta_dot) * sintheta
			) / self.total_mass
		thetaacc = (self.gravity * sintheta - costheta * temp) / (
			self.length
			* (4.0 / 3.0 - self.masspole * np.square(costheta) / self.total_mass)
		)
		xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass
		if self.kinematics_integrator == "euler":
			x = x + self.tau * x_dot
			x_dot = x_dot + self.tau * xacc
			theta = theta + self.tau * theta_dot
			theta_dot = theta_dot + self.tau * thetaacc
		else:
			x_dot = x_dot + self.tau * xacc
			x = x + self.tau * x_dot
			theta_dot = theta_dot + self.tau * thetaacc
			theta = theta + self.tau * theta_dot

		if x < -self.x_threshold or x>self.x_threshold or theta < -self.theta_threshold_radians or theta >self.theta_threshold_radians:
			return (x,x_dot, theta,theta_dot), True
		
		return (x,x_dot, theta,theta_dot), False

	def step(self, a):
		s = (self.position, self.velocity,self.pole_angle,self.pole_angle_velocity)
		sp, t = self.next_state(s, a)
		self.position = sp[0]
		self.velocity = sp[1]
		self.pole_angle = sp[2]
		self.pole_angle_velocity = sp[3]

		r = self.rewards(s, a, sp)

		return (r, sp, t)
