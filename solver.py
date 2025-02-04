import numpy as np
import inspect
import matplotlib.pyplot as plt
from airfoil import NacaAirfoil, Airfoil, ParabolicAirfoil
from typing import Tuple, Literal

class PanelSolver(object):

    def __init__(self, serial_number: str, Q_inf: float, alpha: int, rho: float, N_panels: int, *, airfoil: Airfoil=None, verbose: bool=True) -> None:
        
        # Print info
        if verbose:
            print(f"Creating a PanelSolver object with {N_panels} panels for NACA {serial_number} at {Q_inf} m/s and {alpha} rad and {rho} kg/m^3...")

        # Save attributes
        self.serial_number = serial_number
        self.N_panels = N_panels
        self.Q_inf = Q_inf  # Freestream velocity in m/s
        self.alpha = alpha  # Angle of attack in radians
        self.rho = rho      # Air density in kg/m^3

        # Retrieve U_inf and W_inf
        self.U_inf = Q_inf * np.cos(alpha)
        self.W_inf = Q_inf * np.sin(alpha)

        # Generate the airfoil
        self.airfoil = NacaAirfoil(serial_number)

        # If airfoil is provided, use it
        if airfoil is not None:
            if verbose:
                print("Using provided airfoil. Ignoring serial number.")
            self.airfoil = airfoil

        # Generate the grid
        self.grid_pos = self._generate_grid()

        # Get the quarter-chord and three-quarter-chord points
        self.quarter_chord_pos = self._get_alpha_point(self.grid_pos, 0.25)
        self.three_quarter_chord_pos = self._get_alpha_point(self.grid_pos, 0.75)

        # Get angles of the panels (in rad)
        self.panel_angles = self._get_panel_angles()

        # Get the normal vectors of the panels
        self.panel_normals = self._get_panel_normals()

        # Calculate lhs matrix
        self.__lhs = self._calculate_lhs()
        self.__rhs = self._calculate_rhs()
        
        # print that initialized properly
        if verbose:
            print("Initialized successfully.")

        # Solve the system
        if verbose:
            print("Solving the system...")
        self.Gamma_solution = self._solve_system()
        
        if verbose:
            print("Solved successfully.")

        # Post-process the solution
        self.gamma, self.DL, self.Dp, self.DCp, self.L, self.M0, self.Cl, self.Cm0, self.Cm4 = self._post_process()


    def _generate_grid(self) -> np.ndarray:
        
        # Sampling points
        x = np.linspace(0, 1, self.N_panels+1)

        # Initialize grid
        z = np.zeros(x.shape)

        # Get the camber function
        z_c = self.airfoil.z_c

        # Write in the grid
        for i in range(x.shape[0]):
            z[i] = z_c(x[i])

        # Return np.ndarray (n x 2)
        return np.vstack((x, z)).T

    def _get_panel_angles(self) -> np.ndarray:

        # Initialize slopes
        angles = np.zeros(self.N_panels)

        # Compute the slopes
        for i in range(self.N_panels):
            x1, z1 = self.grid_pos[i]
            x2, z2 = self.grid_pos[i+1]

            angles[i] = np.arctan2(z2-z1, x2-x1)

        return angles

    def _get_panel_normals(self) -> np.ndarray:
            
            # Initialize normals
            normals = np.zeros((self.N_panels, 2))
    
            # Compute the normals
            for i in range(self.N_panels):
                angle = self.panel_angles[i]
                normals[i] = np.array([-np.sin(angle), np.cos(angle)])
    
            return normals
     
    @staticmethod
    def _get_alpha_point(grid: np.ndarray, alpha: float) -> np.ndarray:

        # Check if grid is np.ndarray
        assert isinstance(grid, np.ndarray), "grid must be a np.ndarray"

        # Check if grid has the right shape
        assert grid.shape[1] == 2, "grid must have shape (n x 2)"

        # Check if alpha is a float
        assert isinstance(alpha, float), "alpha must be a float"

        # Check if alpha is between 0 and 1
        assert 0 <= alpha <= 1, "alpha must be between 0 and 1"

        # Initialise the result
        res = np.zeros((len(grid)-1, 2))

        # Compute the points
        for i in range(len(grid)-1):
            res[i] = (1-alpha)*grid[i] + alpha*grid[i+1]

        return res

    def _calculate_influence_coefficient(self, i: int, j: int) -> float:

        # Check if i and j are integers
        assert isinstance(i, int), "i must be an integer"

        # Check if i and j are within the range
        assert 0 <= i < self.N_panels, "i must be within the range of the panels"
        assert 0 <= j < self.N_panels, "j must be within the range of the panels"

        # Get quarter-chord point of panel i (around which you sum)
        x_i, z_i = self.three_quarter_chord_pos[i]

        # Get three-quarter-chord point of panel j (from where you get contribution of vorticity)
        x_j, z_j = self.quarter_chord_pos[j]

        # Get the normal vector of panel i
        n_i = self.panel_normals[i]

        # Distance from i to j
        r_ij = np.sqrt((x_j - x_i)**2 + (z_j - z_i)**2)

        # u_ij
        u_ij = (z_i - z_j) * (1 / (2 * np.pi * r_ij**2))
        w_ij = -(x_i - x_j) * (1 / (2 * np.pi * r_ij**2))

        # Compile velocity vector
        u_w_ij = np.array([u_ij, w_ij])

        # Return the dot product
        return np.dot(u_w_ij, n_i)

    def _calculate_lhs(self) -> np.ndarray:

        # Initialize the lhs matrix
        lhs = np.zeros((self.N_panels, self.N_panels))

        # Compute the lhs matrix
        for i in range(self.N_panels):
            for j in range(self.N_panels):
                a_ij = self._calculate_influence_coefficient(i, j)
                lhs[i, j] = a_ij

        return lhs

    def _calculate_rhs(self) -> np.ndarray:
            
            # Initialize the rhs vector
            rhs = np.zeros(self.N_panels)
    
            # Compute the rhs vector
            for i in range(self.N_panels):
                
                # Get i-th panel normal
                n_i = self.panel_normals[i]

                # U_W_inf
                U_W_inf = np.array([self.U_inf, self.W_inf])

                # Compute the rhs
                rhs[i] = -np.dot(U_W_inf, n_i)
    
            return rhs
    
    def _solve_system(self) -> np.ndarray:
            
            # Solve the system
            try:
                return np.linalg.solve(self.__lhs, self.__rhs)
            except np.linalg.LinAlgError:
                raise RuntimeError("Singular matrix. Cannot solve.")

    def _post_process(self):

        # Get quarter-chord points for moment calculation
        quarter_chord = self.quarter_chord_pos

        # Get vorticity distribution (gamma = Gamma / Dx)
        gamma = np.zeros(self.N_panels)

        for i in range(self.N_panels):
            gamma[i] = self.Gamma_solution[i] / (self.grid_pos[i+1, 0] - self.grid_pos[i,0])

        # Get Lift increments DL
        DL = np.zeros(self.N_panels)

        for i in range(self.N_panels):
            DL[i] = self.rho * self.Q_inf * self.Gamma_solution[i]

        # Get pressure increment Dp and pressure coefficient increment DCp
        Dp = np.zeros(self.N_panels)
        DCp = np.zeros(self.N_panels)

        for i in range(self.N_panels):
            Dp[i] = self.rho * self.Q_inf * gamma[i]
            DCp[i] = Dp[i] / (0.5 * self.rho * self.Q_inf**2)

        # Calculate the total lift
        L = np.sum(DL)

        # Calculate the total Moment about LE
        M0 = 0
        for i in range(self.N_panels):
            M0 += DL[i] * quarter_chord[i,0] * np.cos(self.alpha)

        # Get non-dimensional lift and moment coefficients
        Cl = L / (0.5 * self.rho * self.Q_inf**2)
        Cm0 = M0 / (0.5 * self.rho * self.Q_inf**2)
        Cm4 = 0.25 * Cl - Cm0
        
        return gamma, DL, Dp, DCp, L, M0, Cl, Cm0, Cm4

    def plot_mesh(self, N_plot: int=1_000) -> None:

        # Get the camber function
        z_c = self.airfoil.z_c

        # Plot the camber line
        x = np.linspace(0, 1, N_plot)
        z_c_array = np.zeros(x.shape)
        for i in range(x.shape[0]):
            z_c_array[i] = z_c(x[i])

        plt.plot(x, z_c_array, label="Camber line")

        # Plot the grid
        plt.plot(self.grid_pos[:,0], self.grid_pos[:,1], "o", label="Grid points")

        # Plot the panels
        for i in range(self.grid_pos.shape[0]-1):
            plt.plot(self.grid_pos[i:i+2,0], self.grid_pos[i:i+2,1], "r-")

        # Get quarter-chord points
        quarter_chord = self._get_alpha_point(self.grid_pos, 0.25)

        # Plot the quarter-chord points
        plt.plot(quarter_chord[:,0], quarter_chord[:,1], "o", label="Quarter-chord points")

        # At quarter chord plot the normal vectors
        for i in range(quarter_chord.shape[0]):
            normal = self.panel_normals[i]
            
            plt.quiver(quarter_chord[i,0], quarter_chord[i,1], normal[0], normal[1], scale=8, width=0.002, scale_units="xy", angles="xy")

        # Make equal aspect ratio
        plt.axis("equal")

        # Set the labels
        plt.xlabel("x")
        plt.ylabel("z")
        plt.legend()
        plt.show()

    def plot_res(self, var: Literal['gamma', 'Dp', 'DCp'], plot_camber: bool=False) -> None:

        # camber scale
        scale_camber = 0.1

        # Get x_quarter_chord
        x_quarter_chord = self.quarter_chord_pos[:,0]

        # Check if var is valid
        assert var in ['gamma', 'Dp', 'DCp'], "var must be either 'gamma', 'Dp' or 'DCp'"

        # Get the variable
        if var == "gamma":
            var_array = self.gamma
            ylabel = r"$\gamma,\, [m/s]$"
        elif var == "Dp":
            var_array = self.Dp
            ylabel = r"$\Delta p,\, [Pa]$"
        elif var == "DCp":
            var_array = self.DCp
            ylabel = r"$\Delta C_p,\, [-]$"

        # Plot the variable
        plt.plot(x_quarter_chord, var_array, "bo-")

        # Plot the camber line if requested
        if plot_camber:
            x = np.linspace(0, 1, 300)

            # Get the camber function
            z_c = self.airfoil.z_c

            # Initialize the array
            z_c_array = np.zeros(x.shape)

            # Compute the camber line points
            for i, x_i in enumerate(x):
                z_c_array[i] = z_c(x_i)

            # Get appropriate scale
            var_max = np.max(var_array)
            var_min = np.min(var_array)
            var_range = var_max - var_min

            scale_camber = var_range / (np.max(z_c_array) - np.min(z_c_array)) * scale_camber
            # Plot the camber line
            plt.plot(x, z_c_array * scale_camber, 'k--', label="Camber line", linewidth=0.5)

        # Set the labels
        plt.xlabel(r"$x/c$")
        plt.ylabel(ylabel)

        # Grid and legend
        plt.grid()
        plt.legend()
        plt.tight_layout()
        plt.show()

class PanelAnalyzer(object):

    def __init__(self, ps: PanelSolver, tol: float) -> None:

        # Save the PanelSolver object
        self.ps = ps
        self.tol = tol

        # Get all attributes used for initialization of PanelSolver

        # Get the signature of the __init__ method of ps
        init_signature = inspect.signature(self.ps.__init__)
        # Extract the parameter names passed in the constructor
        init_keys = init_signature.parameters.keys()
        self.init_params_dict = {param: getattr(self.ps, param) for param in init_keys if hasattr(self.ps, param)}

    def find_var_alpha(self, var: Literal['Cl', 'Cm4'], N_panels: int, *,
                        N_res: int=100,
                       alpha_min_deg: float=-5, alpha_max_deg: float=5) -> float:

        # Find range of alphas
        alpha_min = -6 * np.pi / 180
        alpha_max = 6 * np.pi / 180

        # Get the range of alphas
        alphas = np.linspace(alpha_min, alpha_max, N_res)

        # Initialize the array
        vartab = np.zeros(N_res)

        # Loop over alphas
        for i, alpha in enumerate(alphas):

            # Set the alpha and verbose to False and N_panels
            self.init_params_dict["alpha"] = alpha
            self.init_params_dict["verbose"] = False
            self.init_params_dict["N_panels"] = N_panels

            # Create a new PanelSolver object
            ps_new = PanelSolver(**self.init_params_dict)

            # Get the var
            if var == "Cl":
                vartab[i] = ps_new.Cl
            elif var == "Cm4":
                vartab[i] = ps_new.Cm4

        if var == "Cl":
            label = r"$C_l,\, [-]$"
        elif var == "Cm4":
            label = r"$C_{m_4},\, [-]$"

        plt.plot(alphas * 180 / np.pi, vartab, "bx-")
        plt.xlabel(r"$\alpha,\, [deg]$")
        plt.ylabel(label)
        plt.grid()
        plt.tight_layout()

        plt.savefig(f"2-1-a_{var}.pdf")
        plt.show()

    def _compute_Cl_alpha(self, N_panels: int, N_res: int=100, plot: bool=True) -> float:

        # Find range of alphas
        alpha_min = -5 * np.pi / 180
        alpha_max = 5 * np.pi / 180

        # Get the range of alphas
        alphas = np.linspace(alpha_min, alpha_max, N_res)

        # Initialize the array
        Cl_alphas = np.zeros(N_res)

        # Loop over alphas
        for i, alpha in enumerate(alphas):

            # Set the alpha and verbose to False and N_panels
            self.init_params_dict["alpha"] = alpha
            self.init_params_dict["verbose"] = False
            self.init_params_dict["N_panels"] = N_panels

            # Create a new PanelSolver object
            ps_new = PanelSolver(**self.init_params_dict)

            # Get the Cl
            Cl_alphas[i] = ps_new.Cl

        # Plot if requested
        if plot:
            plt.plot(alphas * 180 / np.pi, Cl_alphas, "bo-")
            plt.xlabel(r"$\alpha,\, [deg]$")
            plt.ylabel(r"$C_l,\, [-]$")
            plt.grid()
            plt.show()

        # Fit a line to the Cl vs alpha
        p = np.polyfit(alphas, Cl_alphas, 1)

        # Return the slope
        return p[0]

    def convergence_study(self, *, N_max: int=500):
        
        Cl_alpha_ex = 2 * np.pi

        # Initialize the Cl_alpha list
        Cl_alpha_list = []

        # Initialize the error and relative error lists
        error_list = []
        rel_error_list = []
        N_panels_list = []

        # Initialize the N_panels
        N_panels = 1

        # Initialize the relative error
        rel_error = np.inf

        # Solve for Cl_alpha for each N_panels
        while not (N_panels > N_max):

            # Print info
            print(f"N_panels = {N_panels}")

            # Add N_panels to the list
            N_panels_list.append(N_panels)

            # Compute Cl_alpha
            Cl_alpha_list.append(self._compute_Cl_alpha(N_panels, plot=False))

            # Compute the error and append to the list
            error = np.abs(Cl_alpha_list[-1] - Cl_alpha_ex)
            error_list.append(error)

            # Compute the relative error and append to the list
            rel_error = error / Cl_alpha_ex
            rel_error_list.append(rel_error)

            # Increment N_panels
            N_panels += 5


        # If len(Cl_alpha_list) == N_max, print a warning
        if len(Cl_alpha_list) == N_max:
            print("Warning: Maximum number of panels reached. Stopping the convergence study.")
        else:
            # If rel_error <= tol, print that the convergence study is successful
            print("Convergence study successful.")
            print(f"rel_error = {rel_error}, tol = {self.tol}")

        # Plot the error and relative error on two subplots
        _, ax = plt.subplots(1, 2, figsize=(8.3, 5))
        ax[0].plot(N_panels_list, error_list, "bo-")
        ax[0].set_xlabel(r"$N_p$")
        ax[0].set_ylabel(r"$|C_{l_{\alpha}}^{N_p} - 2 \pi|$")
        ax[0].set_xticks(N_panels_list[::2])
        ax[0].grid()

        ax[1].plot(N_panels_list, rel_error_list, "bo-")
        ax[1].set_xlabel(r"$N_p$")
        ax[1].set_ylabel(r"$\frac{|C_{l_{\alpha}}^{N_p} - 2 \pi|}{2 \pi}$")
        ax[1].set_xticks(N_panels_list[::2])
        ax[1].grid()

        plt.tight_layout()

        plt.savefig("1-3-b_4418.pdf")
        plt.show()

if __name__ == "__main__":

    serial_number = "4418"
    N_panels = 20
    Q_inf = 1   # m/s
    alpha = 10 * np.pi /180   # rad
    rho = 1.225 # kg/m^3

    # For test case, use parabolic airfoil
    test_airfoil = None
    # test_airfoil = ParabolicAirfoil(eps=0.1)
    
    ps = PanelSolver(serial_number, Q_inf, alpha, rho, N_panels, airfoil=test_airfoil)
    # ps.plot_res("DCp", plot_camber=True)

    # ps.plot_mesh()
    # print(ps.Cl)
    # ps.plot_res("DCp", plot_camber=True)

    pa = PanelAnalyzer(ps, tol=1e-1)
    # pa._compute_Cl_alpha(N_panels=20, N_res=10, plot=True)
    # pa.convergence_study(N_max=76)
    pa.find_var_alpha("Cm4", N_panels=60, N_res=10)


