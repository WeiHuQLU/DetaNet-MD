from schnetpack.md.data import IRSpectrum,RamanSpectrum
import numpy as np
from ase import units as ase_units
from schnetpack import units as spk_units
from schnetpack import properties
from md.hdf5_property import Property







class IR_Spectrum(IRSpectrum):
    """
    Compute infrared spectra from a molecular dynamics HDF5 dataset. This class requires the dipole moments
    to be present in the HDF5 dataset.

    Args:
        data (schnetpack.md.utils.HDF5Loader): Loaded dataset.
        resolution (int, optional): Resolution used when computing the spectrum. Indicates how many time lags
                                    are considered in the autocorrelation function is used.
        dipole_moment_handle (str, optional): Indentifier used for extracting dipole data.
    """

    def __init__(
        self,
        data: Property,
        resolution: int = 4096,
        dipole_moment_handle: str = properties.dipole_moment,
    ):
        super(IR_Spectrum, self).__init__(data, resolution=resolution)
        self.dipole_moment_handle = dipole_moment_handle

    def _get_data(self, molecule_idx: int):
        """
        Extract the molecular dipole moments and compute their time derivative via central
        difference.

        Args:
            molecule_idx (int): Index of the molecule for which the spectrum should be computed.
                                Uses the same conventions as schnetpack.md.System.

        Returns:
            numpy.array: Array holding the dipole moment derivatives.
        """
        relevant_data = self.data.get_property(
            self.dipole_moment_handle, False, mol_idx=molecule_idx
        )


        # Compute numerical derivative via central differences
        relevant_data = (relevant_data[2:, ...] - relevant_data[:-2, ...]) / (
            2 * self.timestep
        )
        return relevant_data

    def _process_autocorrelation(self, autocorrelation: np.array):
        """
        Sum over the three Cartesian components.

        Args:
            autocorrelation (numpy.array): Dipole moment flux autorcorrelation functions.

        Returns:
            numpy.array: Updated autocorrelation.
        """
        dipole_autocorrelation = np.sum(autocorrelation, axis=0)
        return [dipole_autocorrelation]
class Raman_Spectrum(RamanSpectrum):
    """
    Compute Raman spectra from a molecular dynamics HDF5 dataset. This class requires the polarizabilities
    to be present in the HDF5 dataset.

    Args:
        data (schnetpack.md_test.utils.HDF5Loader): Loaded dataset.
        incident_frequency (float): laser frequency used for spectrum (in cm$^{-1}$).
                                    One typical value would be 19455.25 cm^-1 (514 nm)
        temperature (float): temperature used for spectrum (in K).
        polarizability_handle (str, optional): Identifier used for extracting polarizability data.
        resolution (int, optional): Resolution used when computing the spectrum. Indicates how many time lags
                                    are considered in the autocorrelation function is used.
        averaged (bool): compute rotationally averaged Raman spectrum.
    """

    def __init__(
        self,
        data: Property,
        incident_frequency: float,
        temperature: float,
        polarizability_handle: str = properties.polarizability,
        resolution: int = 4096,
        averaged: bool = False,
    ):
        super(RamanSpectrum, self).__init__(data, resolution=resolution)
        self.incident_frequency = incident_frequency
        self.temperature = temperature
        self.averaged = averaged
        self.polarizability_handle = polarizability_handle

    def _get_data(self, molecule_idx: int):
        """
        Extract the molecular dipole moments and compute their time derivative via central
        difference.

        Args:
            molecule_idx (int): Index of the molecule for which the spectrum should be computed.
                                Uses the same conventions as schnetpack.md_test.System.

        Returns:
            numpy.array: Array holding the dipole moment derivatives.
        """
        relevant_data = self.data.get_property(
            self.polarizability_handle, False, mol_idx=molecule_idx
        )


        # Compute numerical derivative via central differences
        relevant_data = (relevant_data[2:, ...] - relevant_data[:-2, ...]) / (
            2 * self.timestep
        )

        # Compute isotropic and anisotropic part
        if self.averaged:
            # Setup for random orientations of the molecule
            polar_data = np.zeros((relevant_data.shape[0], 7))
            # Isotropic contribution:
            polar_data[:, 0] = np.trace(relevant_data, axis1=1, axis2=2) / 3
            # Anisotropic contributions
            polar_data[:, 1] = relevant_data[..., 0, 0] - relevant_data[..., 1, 1]
            polar_data[:, 2] = relevant_data[..., 1, 1] - relevant_data[..., 2, 2]
            polar_data[:, 3] = relevant_data[..., 2, 2] - relevant_data[..., 0, 0]
            polar_data[:, 4] = relevant_data[..., 0, 1]
            polar_data[:, 5] = relevant_data[..., 0, 2]
            polar_data[:, 6] = relevant_data[..., 1, 2]
        else:
            polar_data = np.zeros((relevant_data.shape[0], 2))
            # Typical experimental setup
            # xx
            polar_data[:, 0] = relevant_data[..., 0, 0]
            # xy
            polar_data[:, 1] = relevant_data[..., 0, 1]

        return polar_data

    def _process_autocorrelation(self, autocorrelation):
        """
        Compute isotropic and anisotropic components.

        Args:
            autocorrelation (numpy.array): Dipole moment flux autorcorrelation functions.

        Returns:
            numpy.array: Updated autocorrelation.
        """
        if self.averaged:
            isotropic = autocorrelation[0, :]
            anisotropic = (
                0.5 * autocorrelation[1, :]
                + 0.5 * autocorrelation[2, :]
                + 0.5 * autocorrelation[3, :]
                + 3.0 * autocorrelation[4, :]
                + 3.0 * autocorrelation[5, :]
                + 3.0 * autocorrelation[6, :]
            )
        else:
            isotropic = autocorrelation[0, :]
            anisotropic = autocorrelation[1, :]

        autocorrelation = [isotropic, anisotropic]

        return autocorrelation

    def _process_spectrum(self):
        """
        Apply temperature and frequency dependent cross section.
        """
        frequencies = self.frequencies[0]
        cross_section = (
                (self.incident_frequency - frequencies) ** 4
                / frequencies
                / (
                        1
                        - np.exp(
                    -(spk_units.hbar2icm * frequencies)
                    / (spk_units.kB * self.temperature)
                )
                )
        )
        cross_section[0] = 0

        for i in range(len(self.intensities)):
            self.intensities[i] *= cross_section
            self.intensities[i] *= 4.160440e-18  # Where does this come from?
            self.intensities[i][0] = 0.0

        if self.averaged:
            isotropic, anisotropic = self.intensities
            parallel = isotropic + 4 / 45 * anisotropic
            orthogonal = anisotropic / 15

            self.intensities = [parallel, orthogonal]
        
