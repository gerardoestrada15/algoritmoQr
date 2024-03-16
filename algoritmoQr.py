import numpy as np

def gram_schmidt_ortonormalizacion(A):
    # Convertir la matriz de entrada a tipo de datos flotante
    A = A.astype(float)
    
    n, m = A.shape
    
    Q = np.zeros((n, m))
    
    R = np.zeros((m, m))
    
    for j in range(m):
        v_j = A[:, j]
        
        for i in range(j):
            producto_interno = np.dot(Q[:, i], v_j)
            
            v_j -= producto_interno * Q[:, i]
        
        norma_v_j = np.linalg.norm(v_j)
        
        q_j = v_j / norma_v_j
        
        for i in range(j+1):
            producto_interno = np.dot(Q[:, i], A[:, j])
            R[i, j] = producto_interno
        
        # Asignar q_j a la j-ésima columna de Q
        Q[:, j] = q_j
    
    return Q, R

# Llamar a la función e imprimir los resultados
A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
Q, R = gram_schmidt_ortonormalizacion(A)
print("Matriz Q:")
print(Q)
print("\nMatriz R:")
print(R)