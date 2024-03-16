import numpy as np

def gram_schmidt_ortonormalizacion(A):
    # Convertir la matriz de entrada a tipo de datos flotante
    A = A.astype(float)
    
    # Obtener las dimensiones de la matriz A
    n, m = A.shape
    
    # Inicializar una matriz vacía Q del mismo tamaño que A
    Q = np.zeros((n, m))
    
    # Inicializar una matriz vacía R del mismo tamaño que A
    R = np.zeros((m, m))
    
    # Para cada columna de A
    for j in range(m):
        # Inicializar v_j como la j-ésima columna de A
        v_j = A[:, j]
        
        # Para cada columna anterior de A
        for i in range(j):
            # Calcular el producto interno entre v_j y q_i
            producto_interno = np.dot(Q[:, i], v_j)
            
            # Restar la proyección de v_j sobre q_i de v_j
            v_j -= producto_interno * Q[:, i]
        
        # Calcular la norma de v_j
        norma_v_j = np.linalg.norm(v_j)
        
        # Asignar la j-ésima columna de Q como v_j normalizada
        q_j = v_j / norma_v_j
        
        # Asignar los coeficientes de la proyección en R
        for i in range(j+1):
            producto_interno = np.dot(Q[:, i], A[:, j])
            R[i, j] = producto_interno
        
        # Asignar q_j a la j-ésima columna de Q
        Q[:, j] = q_j
    
    return Q, R

# Ejemplo de uso
A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
Q, R = gram_schmidt_ortonormalizacion(A)
print("Matriz Q:")
print(Q)
print("\nMatriz R:")
print(R)