'''
transfer_matrix.py

Defines the transfer matrix class and subclasses.
'''

@dataclass
class TransferMatrix:
    pass

@dataclass
class TMatPol(TransferMatrix):
    '''
    Calculates the T matrix for phonon contributions.
    Inputs: h matrix
    Outputs: diagonalized T matrix
    '''

@dataclass
class TMatPhonon(TransferMatrix):
    '''
    Inputs: energies, operator, approxs
    '''
    pass

@dataclass
class TMatMagnon(TransferMatrix):
    pass
