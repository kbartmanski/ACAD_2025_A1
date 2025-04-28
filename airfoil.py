import numpy as np
from typing import Tuple
import matplotlib.pyplot as plt

class Airfoil(object):

    def __init__(self) -> None:
        self.z_c = None
        self.z_c_prime = None
        self.normal_vector = None

    def z_c(self, x: float) -> float:
        pass

    def z_c_prime(self, x: float) -> float:
        pass

    def normal_vector(self, x: float) -> np.ndarray:
        pass

class ParabolicAirfoil(Airfoil):

    def __init__(self, eps: float) -> None:
        
        # Check if eps is a scalar
        if not isinstance(eps, (int, float)):
            raise ValueError("eps must be a scalar")
        
        # Check if eps is positive
        if eps <= 0:
            raise ValueError("eps must be positive")
        
        self.eps = eps

        # Define the z_c function
        self.z_c = lambda x: 4 * eps * x * (1 - x)

        # Define the z_c_prime function
        self.z_c_prime = lambda x: 4 * eps * (1 - 2 * x)

        # Assign normal to None
        self.normal_vector = None

    def __repr__(self) -> str:
        return 'ParabolicAirfoil'

class NacaAirfoil(Airfoil):

    def __init__(self, serial_number: str) -> None:
        
        # Check if teh serial number is 4 digits
        if len(serial_number) != 4:
            raise ValueError("Serial number must be 4 digits")
        
        self.serial_number = serial_number

        # Extract the parameters from the serial number
        self.m = int(serial_number[0]) / 100
        self.p = int(serial_number[1]) / 10

        # Define the z_c function

        def z_c(x: float) -> float:

            # Check if x is a scalar
            if not isinstance(x, (int, float)):
                raise ValueError("x must be a scalar")
        
            # Check if x is in the range [0, 1]
            if x < 0 or x > 1:
                raise ValueError("x must be in the range [0, 1]")
            

            if x < self.p:
                return self.m / self.p**2 * (2 * self.p * x - x**2)
            else:
                return self.m / (1 - self.p)**2 * (1 - 2 * self.p + 2 * self.p * x - x**2)
            
        # Define the z_c_prime function
        def z_c_prime(x: float) -> float:
            
            # Check if x is a scalar
            if not isinstance(x, (int, float)):
                raise ValueError("x must be a scalar")
        
            # Check if x is in the range [0, 1]
            if x < 0 or x > 1:
                raise ValueError("x must be in the range [0, 1]")
            
            if x < self.p:
                return 2 * self.m / self.p**2 * (self.p - x)
            else:
                return 2 * self.m / (1 - self.p)**2 * (self.p - x)
            
        # Get normal vector at x
        def normal_vector(x: float) -> np.ndarray:
            
            # Check if x is a scalar
            if not isinstance(x, (int, float)):
                raise ValueError("x must be a scalar")
        
            # Check if x is in the range [0, 1]
            if x < 0 or x > 1:
                raise ValueError("x must be in the range [0, 1]")
            
            # Get the angle of the tangent vector
            theta = np.arctan(z_c_prime(x))
            
            # Get the normal vector
            return np.array([-np.sin(theta), np.cos(theta)])
        
        # Return the functions
        self.z_c = z_c
        self.z_c_prime = z_c_prime
        self.normal_vector = normal_vector
    
    def plot(self, N_plot: int=100) -> None:
        
        # Define the x values
        x = np.linspace(0, 1, N_plot)
        
        # Define the y values
        y = np.array([self.z_c(x_i) for x_i in x])

        # Get the normal vectors
        normal_vectors = np.array([self.normal_vector(x_i) for x_i in x])

        # Plot the normal vectors
        for i in range(N_plot):
            x1, y1 = x[i], y[i]
            x2, y2 = x[i] + normal_vectors[i][0], y[i] + normal_vectors[i][1]

            plt.plot([x1, x2], [y1, y2], 'b-')
        
        # Plot the airfoil
        plt.axis('equal')
        plt.plot(x, y, 'r-')
        plt.show()

    def __repr__(self) -> str:
        return str(self.serial_number)


if __name__ == "__main__":
    airfoil = NacaAirfoil("2412")
    airfoil.plot(N_plot=100)



