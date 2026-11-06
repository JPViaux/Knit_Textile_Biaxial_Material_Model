import numpy as np
from scipy.optimize import fsolve

def dfgrd_to_cauchystrain(dfgrd: np.array) -> np.array:
    '''Calculates Right Cauchy Strain C from deformation gradient F. Returns 3x3'''
    cauchystrain = np.zeros((3,3),float)
    dfgrdT = np.zeros((3,3),float)
    dfgrdT = dfgrd.transpose()
    cauchystrain = np.dot(dfgrdT,dfgrd)

    return cauchystrain

def cauchystress_to_pk2(cauchy: np.array, dfgrd: np.array) -> np.array:
    '''Calculates the 2nd Piola Kirchhoff Stress S from Cauchy Stress Sigma. Returns 3x3'''
    pk2 = np.zeros((3,3),float)
    idfgrd = np.linalg.inv(dfgrd)
    it_dfgrd = np.transpose(idfgrd)
    Jacob = Jacobian(dfgrd)    
    pk2 = Jacob * np.dot(idfgrd,np.dot(cauchy,it_dfgrd))
    return pk2

def pk1_to_pk2(pk1: np.array,dfgrd: np.array) -> np.array:
    '''1st PK P to 2nd PK S. Returns 3x3'''
    pk2 = np.zeros((3,3),float)
    idfgrd = np.linalg.inv(dfgrd)
    pk2 = np.dot(idfgrd,pk1)
    return pk2

def pk1_to_cauchystress(pk1: np.array,dfgrd: np.array) -> np.array:
    '''1st PK P to Cauchy Stress Sigma. Returns 3x3 array'''
    cauchystress = np.zeros((3,3),float)
    dfgrdT = dfgrd.transpose()
    Jacob = Jacobian(dfgrd)
    cauchystress = 1/Jacob * np.dot(pk1,dfgrdT)
    return cauchystress

def pk2_to_pk1(pk2: np.array, dfgrd: np.array) -> np.array:
    '''2nd PK S to 1st PK P. Returns 3x3'''
    pk1 = np.zeros((3,3),float)
    pk1 = np.dot(dfgrd,pk2)
    return pk1

def pk2_to_cauchystress(pk2: np.array, dfgrd: np.array) -> np.array:
    '''2nd PK S to Cauchy Stress Sigma. Returns 3x3 array'''
    cauchystress = np.zeros((3,3),float)
    cauchystress = pk1_to_cauchystress(pk2_to_pk1(pk2,dfgrd),dfgrd)
    return cauchystress

def Jacobian(dfgrd: np.array) -> float:
    '''Calculates the Jacobian using F. Returns float'''
    try:
        J = np.linalg.det(dfgrd)
    except Exception as e:
        print("There was en error calculating the Jacobian: ",e)
    return J
  
def matrix_to_voigt(matrix: np.array) -> np.array:
    '''Takes a 3x3 matrix and returns it in voigt notation'''
    if len(matrix[0]) != 3:
        raise ValueError("The matrix to transform to voigt must be 3x3")
    voigt = np.array([matrix[0][0],matrix[1][1],matrix[2][2],matrix[1][2],matrix[0][2],matrix[0][1]])
    return voigt
    
def voigt_to_matrix(vec: np.array) -> np.array:
    '''Converts a Voigt vector to a 3x3 matrix'''
    if len(vec) != 6:
        raise ValueError("The vector must be in column form and of length 6")
    matrix = np.zeros((3,3),float)
    matrix[0,0] = vec[0] 
    matrix[1,1] = vec[1] 
    matrix[2,2] = vec[2] 
    matrix[1,2] = vec[3]
    matrix[2,1] = vec[3] 
    matrix[0,2] = vec[4]
    matrix[2,0] = vec[4] 
    matrix[0,1] = vec[5]
    matrix[1,0] = vec[5]
    return matrix

def dfgrd_to_greenstrain(dfgrd: np.array) -> np.array:
    '''Converts the deformation gradient F into the Right Green Strain E. Returns a 3x3 array'''
    C = dfgrd.transpose() @ dfgrd
    E = 1/2 * (C - np.eye(3))
    return E

def greenstrain_to_dfgrd(E: np.array) -> np.array:
    '''Converts the Green Strain E into the Deformation Gradient F. Returns a 3x3 array'''
    C = 2 * E + np.eye(3)
    F = cauchystrain_to_dfgrd(C)
    return F

def cauchystrain_to_dfgrd(C: np.array) -> np.array:
    '''Converts the Cauchy Strain C into the Deformation Gradient F. It assumes
    symmetry for F (no rotations). Returns a 3x3 array'''
    C = matrix_to_voigt(C)
    x11,x12,x13,x22,x23,x33 = fsolve(c_to_f_equations, (1, 0, 0, 1, 0, 1), args = C)
    F = np.zeros((3,3))
    F[0,0] = x11 
    F[0,1] = x12 
    F[0,2] = x13 
    F[1,0] = x12 
    F[1,1] = x22 
    F[1,2] = x23 
    F[2,0] = x13 
    F[2,1] = x23 
    F[2,2] = x33 
    return F

def c_to_f_equations(p, strains: np.array) -> np.array:
    '''Create the equations to solve C = F^T * F'''
    x11,x12,x13,x22,x23,x33 = p
    return np.array([x11**2 +  x12**2  + x13**2  - strains[0], 
                x12**2 +  x22**2  + x23**2  - strains[1],
                x13**2 +  x23**2  + x33**2  - strains[2],
                x11*x12 + x12*x22 + x13*x23 - strains[3],
                x11*x13 + x12*x23 + x13*x33 - strains[4],
                x12*x13 + x22*x23 + x23*x33 - strains[5]
            ])

def cauchystrain_to_greenstrain(C: np.array) -> np.array:
    '''Converts right Cauchy strain to rigth Green strain'''
    return 1/2*(C - np.eye(3))

def greenstrain_to_cauchystrain(E: np.array) -> np.array:
    '''Converts rigth Green strain to right Cauchy strain'''
    return 2 * E + np.eye(3)

def displgrd_to_dfgrd(H: np.array) -> np.array:
    '''Converts displacement gradient to deformation gradient'''
    return H + np.eye(3)

def displgrd_to_cauchystrain(H: np.array) -> np.array:
    '''Converts displacement gradient to rigth Cauchy strain'''
    F = displgrd_to_dfgrd(H)
    return dfgrd_to_cauchystrain(F)

def displgrd_to_greenstrain(H: np.array) -> np.array:
    '''Converts displacement gradient to rigth Green strain'''
    F = displgrd_to_dfgrd(H)
    return dfgrd_to_greenstrain(F)

def dfgrd_to_displgrd(F: np.array) -> np.array:
    '''Converts deformation gradient to displacement gradient'''
    return F - np.eye(3)

def cauchystrain_to_displgrd(C: np.array) -> np.array:
    '''Converts rigth Cauchy strain to displacement gradient'''
    F = cauchystrain_to_dfgrd(C)
    return dfgrd_to_displgrd(F)

def greenstrain_to_displgrd(E: np.array) -> np.array:
    '''Converts rigth Green strain to displacement gradient'''
    F = greenstrain_to_dfgrd(E)
    return dfgrd_to_greenstrain(F)

def cauchystress_to_pk1(Sigma: np.array, F: np.array) -> np.array:
    '''Converts the Cauchy stress to the 1st PK'''
    J = Jacobian(F)
    i_F = np.linalg.inv(F)
    iT_F = np.transpose(i_F)
    return J * np.dot(Sigma, iT_F)

