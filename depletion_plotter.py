import pandas as pd
import openmc
import os
import numpy as np
import matplotlib.pyplot as plt
import openmc.deplete
import operator
import h5py
import re



class DepletionPlotter:
    """A class for plotting the results of a depletion simulation.

    This class provides functionality for post-processing the results of a depletion simulation
    performed using OpenMC. It takes in the path to the HDF5 output file, the units in which
    the results should be presented, and an optional threshold value for the minimum value of
    the nuclides to include in the plot.

    The postprocess method processes the depletion results, extracting information on the
    evolution of the atomic number density over time for each nuclide in the simulation,
    and aggregating the data by element. Finally, it calculates the average of the sum
    of the number density of all elements, which is stored in the `total` attribute.

    Attributes:
      input_depletion (str): Path to the HDF5 depletion output file.
      flag_units (str): Units in which the results should be presented ('atoms', 'atom/b-cm', 'atom/cm3', 'g/cm3').
      threshold (float, optional): Minimum value of the nuclides to include in the plot. Defaults to 0.0.

    Attributes (after post-processing):
      results (openmc.deplete.ResultsList): An object that stores the depletion results from the HDF5 file.
      units (str): The units in which the results should be presented.
      av_num (float): Avogadro's number.
      f (h5py.File): An object that stores the contents of the HDF5 file.
      threshold (float): The minimum value of the nuclides to include in the plot.
      atoms_evolution (dict): A dictionary that stores the time-evolution of the atomic number density for each nuclide.
      elements_dict (dict): A dictionary that aggregates the atomic number density of each nuclide by element.
      total (float): The average of the sum of the number density of all elements over time.
    """
    def __init__(self, input_depletion, material, flag_units, threshold=None):

        self.results = openmc.deplete.ResultsList.from_hdf5(input_depletion)
        self.units = flag_units
        self.threshold = threshold
        self.av_num = 6.023 * 1e23
        self.hdf5 = h5py.File(input_depletion, "r")
        self.material = material
        self.unit_options = {"atoms": "atoms",
        "atom/b-cm": "atom/b-cm",
        "atom/cm3": "atom/cm3",
        "g/cm3": "atom/cm3"}

    def postprocess(self):
        """Process the depletion results and store them in the object.
        """
        results = self.results
        hdf5 = self.hdf5
        flag = self.units
        unit_options = self.unit_options
        atoms_evolution = {}
        nuclides = hdf5["nuclides"]

        for i, result in enumerate(nuclides):
            s = [float(x) for x in re.findall(r"-?\d+\.?\d*", result)]
            time, atom = results.get_atoms( self.material, result, nuc_units=unit_options.get(flag, "atoms"))
            if flag == "g/cm3":
                atom = s[0] * atom / self.av_num
            atoms_evolution[result] = atom

        elements_dict = {}
        for key in atoms_evolution:
            element = " ".join(re.findall("[a-zA-Z]+", key))
            if key.startswith(element):
                current = atoms_evolution[key]
                if element not in elements_dict:
                    elements_dict[element] = []
                elements_dict[element].append(current)

        for key in elements_dict:
            X = np.zeros(len(time))
            for i, value in enumerate(elements_dict[key]):
                X += value
            elements_dict[key] = X

        self.atoms_evolution = atoms_evolution
        self.atoms_evolution = {key: value for key, value in atoms_evolution.items() if not all(v == 0 for v in value)}
        self.elements_dict = elements_dict
        self.elements_dict = {key: value for key, value in elements_dict.items() if not all(v == 0 for v in value)}
        self.time = time
        self.total = sum(elements_dict.values()) / len(time)
