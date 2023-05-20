import math
import numpy as np 


def matprint(mat, fmt="g"):
    col_maxes = [max([len(("{:"+fmt+"}").format(x)) for x in col]) for col in mat.T]
    for x in mat:
        for i, y in enumerate(x):
            print(("{:"+str(col_maxes[i])+fmt+"}").format(y), end="  ")
        print("")

def rot_matrix(direction, theta):
	"""
	rotation matrix to rotate around direction dir by angle theta

	Taken from: https://en.wikipedia.org/wiki/Rotation_matrix#General_rotations
	"""

	u1 = direction[0]
	u2 = direction[1]
	u3 = direction[2]

	ct = math.cos(theta)
	st = math.sin(theta)

	rot = [
			[ct + u1*u1*(1 - ct), u1*u2*(1 - ct) - u3*st, u1*u3*(1-ct) + u2*st], 
			[u2*u1*(1-ct) + u3*st, ct + u2*u2*(1-ct), u2*u3*(1-ct) - u1*st], 
			[u3*u1*(1-ct) - u2*st, u3*u2*(1-ct) + u1*st, ct + u3*u3*(1-ct)]
		]

	return rot


def is_in_pos_region(vec):
	"""
	returns True if the vector is part of the 'positive' set.

	A vector is either in the 'postive' region of space or the 'negative'. We
	use this to set e_k = e_-k^*
	"""

	if vec[2] > 0:

		return True

	elif vec[2] == 0:

		if vec[0] > 0:

			return True

		elif vec[0] == 0:

			if vec[1] >= 0:

				return True

			else:
				
				return False
		else:

			return False  

	else:
		return False


def create_perp_orthonormal_basis(q):
	"""
	given a q vector, returns a 2x3 matrix e_(nu i), such that q . e_(nu i) = 0
	"""

	x_hat = np.array([1, 0, 0])
	y_hat = np.array([0, 1, 0])
	z_hat = np.array([0, 0, 1])

	q_dir = q/np.linalg.norm(q)

	if not is_in_pos_region(q):

		# put q in positive region

		q_dir = -q_dir

		theta = math.acos(np.dot(z_hat, q_dir))
		phi = math.atan2(np.dot(q_dir, y_hat),np.dot(q_dir, x_hat))

		r1 = rot_matrix(y_hat, theta)
		r2 = rot_matrix(z_hat, phi)

		rot = np.matmul(r2, r1)

		e1_pre = np.matmul(rot, np.array(x_hat, dtype=complex))
		e2_pre = np.matmul(rot, np.array(y_hat, dtype=complex))

		e1 = np.sqrt(0.5)*(e1_pre - 1j*e2_pre)
		e2 = np.sqrt(0.5)*(e1_pre + 1j*e2_pre)

		# e1 = e1_pre
		# e2 = e2_pre

	else:

		theta = math.acos(np.dot(z_hat, q_dir))
		phi = math.atan2(np.dot(q_dir, y_hat),np.dot(q_dir, x_hat))

		r1 = rot_matrix(y_hat, theta)
		r2 = rot_matrix(z_hat, phi)

		rot = np.matmul(r2, r1)

		e1_pre = np.matmul(rot, np.array(x_hat, dtype=complex))
		e2_pre = np.matmul(rot, np.array(y_hat, dtype=complex))

		e1 = np.sqrt(0.5)*(e1_pre + 1j*e2_pre)
		e2 = np.sqrt(0.5)*(e1_pre - 1j*e2_pre)

		# e1 = e1_pre
		# e2 = e2_pre

	return [e1, e2]