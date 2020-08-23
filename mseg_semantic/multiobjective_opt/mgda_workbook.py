

import matplotlib.pyplot as plt
import numpy as np
import pdb

LIGHT_BLUE = np.array([221, 237, 255]) / 255

def main():
	""" """
	v1 = np.array([-1,1])
	v2 = np.array([3,1])

	# v1 = np.array([-2,2])
	# v2 = np.array([-1,2])

	# v1 = np.array([2,2])
	# v2 = np.array([0.5,2])

	plt.arrow(0,0, v1[0], v1[1], color="r", width=0.03, zorder=1.5)
	plt.arrow(0,0, v2[0], v2[1], color="m", width=0.03, zorder=1.5)

	method = 'heuristic' # 'analytic'

	print('Gamma = 1: ', v2.dot(v1) >= v1.T.dot(v1))
	print('Gamma = 0: ', v2.dot(v1) >= v2.T.dot(v2))

	if method == 'heuristic':
		alphas = np.linspace(0,1,20)
		p = np.zeros((20,2))
		for i, alpha in enumerate(alphas):
			p[i,:] = alpha * v1 + (1-alpha) * v2

		norms = np.linalg.norm(p, axis=1)
		min_norm_idx = np.argmin(norms)
		for i, alpha in enumerate(alphas):
			if i == min_norm_idx:
				color = 'g'
				zorder = 2
			else:
				color = LIGHT_BLUE
				zorder = 1

			dx = p[i,0]
			dy = p[i,1]
			plt.arrow(0,0, dx, dy, color=color, width=0.01, zorder=zorder)
	elif method == 'analytic':

		num = (v2 - v1).T.dot(v2)
		denom = np.linalg.norm(v1 - v2) ** 2
		alpha = num / denom
		# clip to range [0,1]
		alpha = max(min(alpha,1),0)
		p = alpha * v1 + (1-alpha) * v2
		dx, dy = p
		color = 'g'
		zorder = 2
		plt.arrow(0,0, dx, dy, color=color, width=0.01, zorder=zorder)


	plt.xlim([-2.5,3.5])
	plt.ylim([-0.5,2.5])
	plt.show()


if __name__ == '__main__':
	main()