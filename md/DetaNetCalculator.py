from typing import Union, List, Dict
from schnetpack.md.neighborlist_md import NeighborListMD
import  torch
import numpy as np
from schnetpack.model import AtomisticModel
from schnetpack.md import System
from schnetpack.md.calculators import SchNetPackCalculator
from md.models_load import model_force,model_energy ,model_dipole ,model_polar  
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
        self.required_properties=required_properties

    def _prepare_model(self, model_files: List[str]) -> List[AtomisticModel]:
        """
        Load models corresponding to the required properties.
    
        Args:
            model_files (List[str]): List of paths to model files. Must match required_properties.
    
        Returns:
            List[AtomisticModel]: Loaded and prepared models.
        """
        # required_property -> model_loader 
        model_loader_map = {
            "forces": model_force,
            "energy": model_energy,
            "dipole_moment": model_dipole,
            "polarizability": model_polar
        }
    
        if len(model_files) != len(self.required_properties):
            raise ValueError(f"Mismatch between number of model files ({len(model_files)}) and "
                             f"required properties ({len(self.required_properties)}).")
    
        models_list = []
        for prop, path in zip(self.required_properties, model_files):
            if prop not in model_loader_map:
                raise ValueError(f"Unknown property '{prop}' for model loading. Supported keys: {list(model_loader_map.keys())}")
            model = model_loader_map[prop](path, self.device)
            model.to(self.device)
            model.eval()
            models_list.append(model)
    
        return models_list


    def calculate(self, system: System):
        """
        Perform all calculations and compyte properties .

        Args:
            system (schnetpack.md.System): System from the molecular dynamics simulation.
        """

        inputs = self._generate_input(system)
        #cell models
        numbers=inputs['_atomic_numbers']
        positions=inputs['_positions']
        idx_m = inputs['_idx_m']
        box=inputs['_cell']
        if torch.any(inputs['_pbc']==False):
            prediction = [model(z=numbers, pos=positions,box=None,batch=idx_m) for model in self.models]
            
        else:
             prediction = [model(z=numbers, pos=positions,box=box,batch=idx_m) for model in self.models]
        
        self.results = {i: j for i, j in zip(self.required_properties, prediction)}
       
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



