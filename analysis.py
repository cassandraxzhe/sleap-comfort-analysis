import h5py
import numpy as np
from scipy.interpolate import interp1d




class SLEAP_Analysis:
    def __init__(self, filename, start_frame=0, end_frame=None, framerate=30, threshold=1):
        # Body part indices
        self.L_BACK_FOOT_INDEX = 3
        self.R_BACK_FOOT_INDEX = 1
        self.L_BACK_FETLOCK_INDEX = 2
        self.R_BACK_FETLOCK_INDEX = 0

        # Other parameters
        self.start_frame = start_frame
        self.end_frame = end_frame
        self.framerate = framerate # Default framerate, can be adjusted
        self.threshold = threshold  # Default threshold for gait analysis, can be adjusted

        # Load the SLEAP data from the specified file
        self.dset_names, self.locations, self.node_names = self.open_file(filename)
        print("\n===filename===")
        print(self.filename)
        print("\n===HDF5 datasets===")
        print(self.dset_names)
        print("\n===locations data shape===")
        print(self.locations.shape)
        print("\n===nodes===")
        for i, name in enumerate(self.node_names):
            print(f"{i}: {name}")


        self.frame_count, self.node_count, _, self.instance_count = self.locations.shape
        print("\nframe count:", self.frame_count)
        print("node count:", self.node_count)
        print("instance count:", self.instance_count)
        

    def open_file(filename):
        with h5py.File(filename, "r") as f:
            dset_names = list(f.keys())
            # print(dset_names)
            locations = f["tracks"][:].T
            node_names = [n.decode() for n in f["node_names"][:]]
        return dset_names, locations, node_names

    
    def analyze(self):
        # Placeholder for analysis logic
        print("Analyzing SLEAP data...")

        self.locations = self.fill_missing(self.locations)

        return 
    
    def visualize(self):
        # Placeholder for visualization logic
        print("Visualizing SLEAP data...")

        return
    

    def fill_missing(self, Y, kind="linear"):
        """
        Fills missing values independently along each dimension after the 
        first.
        """

        # Store initial shape.
        initial_shape = Y.shape

        # Flatten after first dim.
        Y = Y.reshape((initial_shape[0], -1))

        # Interpolate along each slice.
        for i in range(Y.shape[-1]):
            y = Y[:, i]

            # Build interpolant.
            x = np.flatnonzero(~np.isnan(y))

            # Check if x is empty, if so, skip interpolation for this slice
            if x.size == 0:
                continue

            f = interp1d(x, y[x], kind=kind, fill_value="extrapolate", bounds_error=False)

            # Fill missing
            xq = np.flatnonzero(np.isnan(y))
            y[xq] = f(xq)

            # Fill leading or trailing NaNs with the nearest non-NaN values
            mask = np.isnan(y)
            y[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), y[~mask])

            # Save slice
            Y[:, i] = y

        # Restore to initial shape.
        Y = Y.reshape(initial_shape)

        return Y
    
    def get_gait_data(self):
        return 