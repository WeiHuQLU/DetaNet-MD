from typing import Union, List, Dict
from schnetpack.md.neighborlist_md import NeighborListMD
import  torch
import numpy as np
from md_test.models_load import model_force,model_energy ,model_dipole ,model_polar
from schnetpack.model import AtomisticModel
from schnetpack.md import System
from schnetpack.md.calculators import SchNetPackCalculator
import logging
log = logging.getLogger(__name__)
class DetaNetCalculator(SchNetPackCalculator):
    def __init__(self,
                model_files: list[str],
                force_key: str,
                energy_unit: Union[str, float],
                position_unit: Union[str, float],
                neighbor_list: NeighborListMD,
                energy_key: str = None,
                stress_key: str = None,
                required_properties: List = [],
                property_conversion: Dict[str, Union[str, float]] = {},
                script_model: bool = False,
                device:torch.device=torch.device('cuda')
                ):
        self.device=device
        super(DetaNetCalculator, self).__init__(
            model_file=model_files,
            required_properties=required_properties,
            neighbor_list=neighbor_list,
            force_key=force_key,
            energy_unit=energy_unit,
            position_unit=position_unit,
            energy_key=energy_key,
            stress_key=stress_key,
            property_conversion=property_conversion,
            script_model=script_model
        )
        # Convert list of models to module list
        self.models = torch.nn.ModuleList(self.model)
        self.neighbor_list = neighbor_list

    def _prepare_model(self, model_files: List[str]) -> List[AtomisticModel]:
        """
        Load multiple models.

        Args:
            model_files (list(str)): List of stored models. ,model_polar

        Returns:
            list(AtomisticModel): list of loaded models.
        """
        get_models=[model_force,model_dipole]
        load_models= [i(j,self.device) for i, j in zip(get_models, model_files)]
        models_list=[]
        for model in load_models:
                model.eval()
                model.to(self.device)
                models_list.append(model)

        return models_list


    def calculate(self, system: System):
        """
        Perform all calculations and compyte properties .

        Args:
            system (schnetpack.md.System): System from the molecular dynamics simulation.#,'polarizability'
        """
        inputs = self._generate_input(system)
        #cell models
        idx_m = inputs['_idx_m']
        numbers=inputs['_atomic_numbers']
        positions=inputs['_positions']
        idx_i = inputs['_idx_i']
        idx_j = inputs['_idx_j']
        idx_ji = torch.cat((idx_j.unsqueeze(0), idx_i.unsqueeze(0)), dim=0).long()
        prediction = [model(z=numbers, pos=positions,edge_index=idx_ji,batch=idx_m) for model in self.models]
        properties = ['forces', 'dipole_moment']
        self.results = {i: j for i, j in zip(properties, prediction)}
        if 'forces' in self.results:
            self.results['forces'] = self.results['forces'].reshape(-1, 3)
        if 'energy' in self.results:
            self.results['energy'] = self.results['energy'].reshape(-1)
        if 'dipole_moment' in self.results:
            self.results['dipole_moment']=self.results['dipole_moment'].reshape(-1)
        
        if 'polarizability'in self.results:
            self.results['polarizability']=self.results['polarizability'].reshape(-1,3)
        
        self._update_system(system)

    def _generate_input(self, system: System) -> Dict[str, torch.Tensor]:
        """
        Function to extracts neighbor lists, atom_types, positions e.t.c. from the system and generate a properly
        formatted input for the schnetpack model.

        Args:
            system (schnetpack.md.System): System object containing current state of the simulation.

        Returns:
            dict(torch.Tensor): Schnetpack inputs in dictionary format.
        """
        inputs={}
        inputs_list = self._get_system_molecules(system)
        for key,value in inputs_list.items():
            inputs[key]=value.to(self.device)

        neighbors = self.neighbor_list.get_neighbors(inputs)
        inputs.update(neighbors)
        return inputs



