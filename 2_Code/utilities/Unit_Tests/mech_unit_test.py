import numpy as np
import sys
import os
curr_file = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(curr_file)
sys.path.insert(0, parent_dir)
from Python_Wrapper import mech

def main() -> None:
    # Initialization of variables to validate the functions
    vec = np.array([1,2,3,4,5,6])
    matrix = np.array([[1,6,5],[6,2,4],[5,4,3]])
    F = np.array([[2,0,0],[0,1.5,0],[0,0,1]])
    sigma = np.array([[2,0,0],[0,0,0],[0,0,0]])
    P = np.linalg.det(F) * sigma @ np.linalg.inv(F.transpose())
    S = np.linalg.inv(F) @ P
    C = F.transpose() @ F
    E = 1/2 * (C - np.eye(3))
    # When calculating F from C, we need a tolerance. It is done
    # by solving second degree equations
    tolerance = 10e-10
  
    # Testing the voigt to matrix functions
    test_voigt(vec,matrix)
    # Testing the Jacobian
    test_Jacobian(F)
    # Test all the other functions
    test_F_to_C(F)
    test_sigma_to_S(sigma,F)
    test_P_to_S(P,F)
    test_P_to_sigma(P,F)
    test_S_to_P(S,F)
    test_S_to_sigma(S,F)
    test_C_to_F(C,F,tolerance)
    test_F_to_E(F)
    test_E_to_F(E,F,tolerance)
    test_C_to_E(C,E)
    test_E_to_C(E,C)

def test_voigt(vec: np.array,mat: np.array) -> None:
    '''Function to test voigt_to_matrix and matrix_to_voigt'''
    # Comparing the matrices    
    matrix = mech.voigt_to_matrix(vec)
    if not are_equal(matrix,mat):
        raise ValueError("The voigt_to_matrix function is not working properly")
    print("The voigt_to_matrix function works well!")
    # Comparing the vectors
    vector = mech.matrix_to_voigt(mat)
    for i in range(len(vector)):
        if vector[i] != vec[i]:
            raise ValueError("The voigt_to_matrix function is not working properly")
    print("The matrix_to_voigt function works well!")

def test_Jacobian(F: np.array) -> None:
     '''Function to test the Jacobian function'''
     if np.linalg.det(F) != mech.Jacobian(F):
        raise ValueError("The Jacobian function is not working well :(")
     print("The Jacobian function works great!")

def are_equal(m1: np.array, m2: np.array) -> bool:
    '''Function that compares two matrices. Returns true if all the elements are equal'''
    comparison = m1 == m2
    for row in comparison:
         for el in row:
              if not el:
                   return False
    return True

def are_almost_equal(m1: np.array, m2: np.array, tolerance: float) -> bool:
    '''Function that compares two matrices element by element with a tolerance. Returns true 
    if all the elements are within tolerance'''
    comparison = m1 - m2
    for row in comparison:
        for el in row:
            if abs(el) > tolerance:
                return False
    return True

def test_F_to_C(F: np.array) -> None:
    '''Tests the function dfgrd_to_cauchystrain'''
    if not are_equal(np.dot(F.transpose(),F),mech.dfgrd_to_cauchystrain(F)):
        raise ValueError("The dfgrd_to_cauchystrain isn't working")
    print("The dfgrd_to_cauchystrain works!")

def test_sigma_to_S(sigma: np.array, F: np.array) -> None:
    '''Tests the function cauchystress_to_pk2'''
    S = np.linalg.det(F) * np.linalg.inv(F) @ sigma @ np.linalg.inv(F.transpose())
    if not are_equal(S,mech.cauchystress_to_pk2(sigma,F)):
        raise ValueError("The cauchystress_to_pk2 is not working")
    print("The cauchystress_to_pk2 function works amazing!")

def test_P_to_S(P: np.array, F: np.array) -> None:
    '''Tests the function pk1_to_pk2'''
    S = np.linalg.inv(F) @ P
    if not are_equal(S,mech.pk1_to_pk2(P,F)):
        raise ValueError("The pk1_to_pk2 function has a problem")
    print("Yes! The pk1_to_pk2 works")

def test_P_to_sigma(P: np.array, F: np.array) -> None:
    '''Test the function pk1_to_cauchystress'''
    sigma = 1/np.linalg.det(F) * P @ F.transpose()
    if not are_equal(sigma,mech.pk1_to_cauchystress(P,F)):
        raise ValueError("The pk1_to_cauchystress doesn't work")
    print("You got the pk1_to_cauchystress function right!")

def test_S_to_P(S: np.array, F: np.array) -> None:
    '''Test pk2_to_pk1 function'''
    P = np.dot(F,S)
    if not are_equal(P,mech.pk2_to_pk1(S,F)):
        raise ValueError("Sorry, the function pk2_to_pk1 has a problem")
    print("Come on! The function pk2_to_pk1 works!")

def test_S_to_sigma(S: np.array, F: np.array) -> None:
    '''Test the function pk2_to_cauchystress'''
    sigma = 1/np.linalg.det(F) * F @ S @ F.transpose()
    if not are_equal(sigma,mech.pk2_to_cauchystress(S,F)):
        raise ValueError("You need to check pk2_to_cauchystress")
    print("It works! pk2_to_cauchystress")

def test_C_to_F(C: np.array, F: np.array, tolerance: float) -> None:
    '''Test the cauchystrain_to_dfgrd function'''
    if not are_almost_equal(F,mech.cauchystrain_to_dfgrd(C),tolerance):
        print("F: ",F)
        print("Calculated F: ",mech.cauchystrain_to_dfgrd(C))
        raise ValueError("Double-check the cauchystrain_to_dfgrd function")
    print("YES! The function cauchystrain_to_dfgrd works!")

def test_F_to_E(F: np.array) -> None:
    '''Test the function dfgrd_to_greenstrain'''
    E = 1/2 * (F.transpose() @ F - np.eye(3))
    if not are_equal(E,mech.dfgrd_to_greenstrain(F)):
        raise ValueError("The function dfgrd_to_greenstrain has an error")
    print("Woohoo! The function dfgrd_to_greenstrain works great!")

def test_E_to_F(E: np.array, F: np.array, tolerance: float) -> None:
    '''Tests the function greenstrain_to_dfgrd'''
    if not are_almost_equal(F,mech.greenstrain_to_dfgrd(E),tolerance):
        raise ValueError("Sorry, the greenstrain_to_dfgrd function is wrong")
    print("Vamos! The greenstrain_to_dfgrd function works!")

def test_C_to_E(C: np.array, E: np.array) -> None:
    '''Tests the cauchystrain_to_greenstrain function'''
    if not are_equal(E,mech.cauchystrain_to_greenstrain(C)):
        raise ValueError("The cauchystrain_to_greenstrain function has a problem")
    print("Yes! The cauchystrain_to_greenstrain function works!")

def test_E_to_C(E: np.array, C: np.array) -> None:
    '''Tests the greenstrain_to_cauchystrain function'''
    if not are_equal(C,mech.greenstrain_to_cauchystrain(E)):
        raise ValueError("The greenstrain_to_cauchystrain function has a problem")
    print("Come on! The greenstrain_to_cauchystrain function works!")

if __name__ == '__main__':
    main()

