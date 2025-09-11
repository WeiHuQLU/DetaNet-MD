import json
import logging
import h5py
import numpy as np
from ase import Atoms
from typing import Optional
from tqdm import trange

from schnetpack import properties, units

log = logging.getLogger(__name__)

from schnetpack.md.data import HDF5Loader


log = logging.getLogger(__name__)


class HDF5LoaderError(Exception):
    """
    Exception for HDF5 loader class.
    """

    pass



class Property(HDF5Loader):
    def __init__(
        self,
        hdf5_database: str,
        skip_initial: Optional[int] = 0,
        load_properties: Optional[bool] = True,
    ):

        super(Property,self).__init__(
            hdf5_database=hdf5_database,
            skip_initial=skip_initial,
            load_properties=load_properties

        )
        self.database = h5py.File(hdf5_database, "r", swmr=True, libver="latest")
        self.skip_initial = skip_initial
        self.data_groups = list(self.database.keys())

        self.properties = {}

        # Load basic structure properties and MD info
        if "molecules" not in self.data_groups:
            raise HDF5LoaderError(
                "Molecule data not found in {:s}".format(hdf5_database)
            )
        else:
            self._load_molecule_data()

        # If requested, load other properties predicted by the model stored via PropertyStream
        if load_properties:
            if "properties" not in self.data_groups:
                raise HDF5LoaderError(
                    "Molecule properties not found in {:s}".format(hdf5_database)
                )
            else:
                self._load_property_data()

        # Do formatting for info
        loaded_properties = list(self.properties.keys())
        if len(loaded_properties) == 1:
            loaded_properties = str(loaded_properties[0])
        else:
            loaded_properties = (
                ", ".join(loaded_properties[:-1]) + " and " + loaded_properties[-1]
            )

        log.info(
            "Loaded properties {:s} from {:s}".format(loaded_properties, hdf5_database)
        )

    def get_property(
        self,
        property_name: str,
        atomistic: bool,
        mol_idx: Optional[int] = 0,
        replica_idx: Optional[int] = None,
    ):
        """
        Extract property from dataset.

        Args:
            property_name (str): Name of the property as contained in the self.properties dictionary.
            atomistic (bool): Whether the property is atomistic (e.g. forces) or defined for the whole molecule
                              (e.g. energies, dipole moments).
            mol_idx (int): Index of the molecule to extract, by default uses the first molecule (mol_idx=0)
            replica_idx (int): Replica of the molecule to extract (e.g. for ring polymer molecular dynamics). If
                               replica_idx is set to None (default), the centroid is returned if multiple replicas are
                               present.

        Returns:
            np.array: N_steps x property dimensions array containing the requested property collected during
                      the simulation.
        """
        if atomistic:
            mol_idx = slice(
                self.molecule_range[mol_idx], self.molecule_range[mol_idx + 1]
            )
        else:
            mol_idx = mol_idx

        # Check whether property is present
        if property_name not in self.properties:
            raise HDF5LoaderError(f"Property {property_name} not found in database.")

        if self.properties[property_name] is None:
            # Typically used for cells
            return None
        elif property_name == properties.Z or property_name == properties.masses:
            # Special case for atom types and masses
            return self.properties[property_name][mol_idx]
        else:
            # Standard properties
            a= 3 * mol_idx - 3
            b= 3 * mol_idx
            target_property = self.properties[property_name][:, :, a:b, ...]
#            target_property=np.sum(dipole_property**2, axis=2)

        # Compute the centroid unless requested otherwise
        if replica_idx is None:
            if target_property is not None:
                target_property = np.mean(target_property, axis=1)
        else:
            if target_property is not None:
                target_property = target_property[:, replica_idx, ...]

        return target_property
