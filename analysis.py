import h5py
import numpy as np
from scipy.interpolate import interp1d
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy.differentiate import derivative


class SLEAP_Analysis:
    def __init__(self, filename, start_frame=0, end_frame=None, framerate=30, threshold=1, subject="Achilles", group="Preop", session="1"):
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

        # Experiment metadata
        self.subject = subject
        self.group = group
        self.session = session

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


    def open_file(self, filename):
        """
        Reads h5 file data.

        `filename` the name of the h5 file, extensions included
        @returns 

        """
        with h5py.File(filename, "r") as f:
            dset_names = list(f.keys())
            # print(dset_names)
            locations = f["tracks"][:].T
            node_names = [n.decode() for n in f["node_names"][:]]
        return dset_names, locations, node_names

    
    def analyze(self):
        """
        Conducts stance time analysis of the h5 data.
        """
        # Placeholder for analysis logic
        print("Analyzing SLEAP data...")

        self.locations = self.fill_missing(self.locations)

        return 
    
    def visualize(self):
        """
        Visualizes the data as a series of plots that show the location of 
        the data points over time.
        """

        self.lbfoot_loc = self.locations[self.start_frame:self.end_frame, self.L_BACK_FOOT_INDEX, :, :]
        self.rbfoot_loc = self.locations[self.start_frame:self.end_frame, self.R_BACK_FOOT_INDEX, :, :]

        sns.set('notebook', 'ticks', font_scale=1.2)
        mpl.rcParams['figure.figsize'] = [15,6]

        self.plot_locs(self.lbfoot_loc, self.rbfoot_loc, f'{self.subject} {self.group} Session {self.session}')
    

    def plot_locs(self, left_loc, right_loc, title, y_coords=False, x_coords=True):
        """
        Helper function to plot the given locations of the data.
        """
        plt.figure()

        # x-coordinate tracking
        if x_coords:
            plt.plot(left_loc[:,0,0], 'y',label='left x')
            plt.plot(right_loc[:,0,0], 'g',label='right x')

        # y-coordinate tracking
        if y_coords:
            plt.plot(-1*self.lbfoot_loc[:,1,0], 'y', label='left y')
            plt.plot(-1*self.rbfoot_loc[:,1,0], 'g', label='right y')

        plt.legend(loc="center right")
        plt.title(title)

        # visualize locations
        plt.figure(figsize=(7,7))
        plt.plot(self.lbfoot_loc[:,0,0],-1*self.lbfoot_loc[:,1,0], 'y',label='left')
        plt.plot(self.rbfoot_loc[:,0,0],-1*self.rbfoot_loc[:,1,0], 'g',label='right')
        plt.legend()
        plt.title(title + 'back feet tracks')
    

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
    
    
    def smooth_diff(self, node_loc, win=25, poly=3):
        """
        node_loc is a [frames, 2] array

        win defines the window to smooth over

        poly defines the order of the polynomial
        to fit with

        """
        node_loc_vel = np.zeros_like(node_loc)

        for c in range(node_loc.shape[-1]):
            node_loc_vel[:, c] = savgol_filter(node_loc[:, c], win, poly, deriv=1)

        node_vel = np.linalg.norm(node_loc_vel,axis=1)

        return node_vel


    def dx(self, node_loc):
        """
        Parameters
        ----------
        node_loc : array
            location array for the node to examine
        framerate : int
            framerate in fps

        Returns
        ----------
        deriv : array
            array of derivative of node position at each frame
        """
        deriv = []

        x_prev = node_loc[0, 0, 0]
        for x in node_loc[:, 0, 0]:
            deriv.append((x-x_prev)/(24/self.framerate))
            x_prev = x
        return deriv
    
    
    def calc_downtime(self, deriv):
        """
        Parameters
        ----------
            deriv : list
            derivative of coordinates for each frame
            sigma : int | float
            threshold for foot down
            total : int
            total number of frames


        Returns
        -------
            tuple (foot_down, percent_down, indices)
            foot_down : int
                number of frames where the foot is down
            percent_down : float
                percent time down out of total stride
            indices : list
                list of indices when the foot is down
        """
        foot_down = 0
        indices= []
        total = self.end_frame-self.start_frame

        for i, x in enumerate(deriv):
            if x < self.threshold:
                foot_down += 1
                indices.append(i)
        print("Down for", foot_down, "frames:", foot_down/total*100, "percent of total time.")
        return foot_down, foot_down / total * 100, indices


    def get_gait_data(self):
        """
        Returns the left and right gait data as a 2d list `data` such that
        data[0] = the left leg stance time list and data[1] = the right leg
        stance time list.
        """
        # apply savitzky-golay
        self.lbfoot = self.smooth_diff(self.lbfoot_loc[:, :, 0])
        self.rbfoot = self.smooth_diff(self.rbfoot_loc[:, :, 0])

        # calculate dx/dt at each frame
        self.lbfoot_deriv = self.dx(self.lbfoot_loc)
        self.rbfoot_deriv = self.dx(self.rbfoot_loc)

        # Calculate the downtime for each foot
        self.lbfoot_down = self.calc_downtime(self.lbfoot_deriv)
        self.rbfoot_down = self.calc_downtime(self.rbfoot_deriv)

        return [self.lbfoot_down, self.rbfoot_down]