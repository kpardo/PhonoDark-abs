'''
test.py
unit tests
needs to be fully fixed up to actually work. right now just collecting
tests functions from previous scripts.
'''



def check_T_mat_properties(q_XYZ_list, MATERIAL):
    '''
    originally from rate_calculator.py
    '''

	print('!! CHECKING PROPERTIES OF T matrix AT EACH Q')
	print('!! COMMENT OUT FOR FASTER SPEED')

	[pol_energy_list, pol_T_list] = diagonalization.calculate_pol_E_T(MATERIAL, q_XYZ_list)

	[pol_energy_list_mq, pol_T_list_mq] = diagonalization.calculate_pol_E_T(MATERIAL, -1*q_XYZ_list)

	num_pol = len(pol_energy_list_mq[0])

	for q in range(len(q_XYZ_list)):

		T11q = pol_T_list[q][:num_pol, :num_pol]
		T12q = pol_T_list[q][:num_pol, num_pol:]
		T21q = pol_T_list[q][num_pol:, :num_pol]
		T22q = pol_T_list[q][num_pol:, num_pol:]

		T11mq = pol_T_list_mq[q][:num_pol, :num_pol]
		T12mq = pol_T_list_mq[q][:num_pol, num_pol:]
		T21mq = pol_T_list_mq[q][num_pol:, :num_pol]
		T22mq = pol_T_list_mq[q][num_pol:, num_pol:]

		tol_param = 10**(-3)

		if not compare_mats(T11q, np.conj(T22mq), tol_param)[0]:
			print("T11_q != T22_(-q)^*")
			print(q)
			print('Max difference :')
			print(compare_mats(T11q, np.conj(T22mq), tol_param)[1])

			# for nu in range(num_pol):
			# 	print(nu)
			# 	print(compare_mats(T11q[nu], np.conj(T22mq)[nu], tol_param)[1])

			# print(T11q[28])
			# print(np.conj(T22mq)[28])
			# print((T11q[28] - np.conj(T22mq)[28])*np.conj(T11q[28] - np.conj(T22mq)[28]))
			# print()
			# exit()
		if not compare_mats(T12q, np.conj(T21mq), tol_param):
			print("T12q != T21_(-q)^*")
			print(q)
			print(compare_mats(T12q, np.conj(T21mq), tol_param)[1])
			print()

		if not compare_mats(T21q, np.conj(T12mq), tol_param):
			print("T21q != T12_(-q)^*")
			print(q)
			print(compare_mats(T21q, np.conj(T12mq), tol_param)[1])
			print()

		if not compare_mats(T22q, np.conj(T11mq), tol_param):
			print("T22_q != T11_(-q)^*")
			print(q)
			print(compare_mats(T22q, np.conj(T11mq), tol_param)[1])
			print()


def check_acoustic_sum_rule(born):
    '''
    originally from rate_calculator.py
    '''

	if not np.allclose(sum(born), np.zeros((3,3))):
		print('Acoustic sum rule violated.')
		print(sum(born))



def compare_mats(m1, m2, tol):
	"""
	compares two matrices, if the maximum difference between them is less than tol
	returns True, otherwise False

    originally from rate_calculator.py
	"""

	diff_mat = m1 - m2

	diff_mat_sq = (m1 - m2)*np.conj(m1 - m2)

	max_diff_sq = np.amax(diff_mat_sq)

	if max_diff_sq < tol:
		return [True, max_diff_sq]
	else:
		return [False, max_diff_sq]




def check_U_plus_V(T_mat):
    '''
    originally from rate_calculator.py
    '''

	num_pol = len(T_mat)//2

	print(num_pol)
	T11_conj = np.conj(T_mat[:num_pol, :num_pol])

	T21_conj = np.conj(T_mat[num_pol:, :num_pol])

	UpV = T11_conj + T21_conj

	print('U_(nup, nu)^*_p + V_(nup, nu)_(-p) : ')
	print()

	for i in range(num_pol):
		for j in range(num_pol):

			if UpV[i][j]*np.conj(UpV[i][j]) < 10**(-6):
				UpV[i][j] = 0

	my_math.matprint(UpV*np.conj(UpV))
	print()
