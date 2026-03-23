from typing import Union, List, Dict
from schnetpack.md.neighborlist_md import NeighborListMD
import torch
import numpy as np
from schnetpack.model import AtomisticModel
from schnetpack.md import System
from schnetpack.md.calculators import SchNetPackCalculator
from md_dipole.models_load import model_force_energy, model_dipole, model_polar, model_energy, model_force
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
                 separate_energy_force: bool = False,  
                 device: torch.device = torch.device('cuda')):
        self.device = device
        self.separate_energy_force = separate_energy_force  
        self.model_to_props = []  
        
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
        self.required_properties = required_properties

    def _prepare_model(self, model_files: List[str]) -> List[AtomisticModel]:
  
        load_models = []
        file_idx = 0  

        if "energy" in self.required_properties or "forces" in self.required_properties:
            if not self.separate_energy_force:
                m = model_force_energy(model_files[file_idx], self.device)
                m.eval()
                m.to(self.device)
                load_models.append(m)
                
                current_props = []
                if "energy" in self.required_properties:
                    current_props.append(("energy", 0))
                if "forces" in self.required_properties:
                    current_props.append(("forces", 1))
                self.model_to_props.append(current_props)
                
                file_idx += 1
            else:
                if "energy" in self.required_properties:
                    m = model_energy(model_files[file_idx], self.device)
                    m.eval()
                    m.to(self.device)
                    load_models.append(m)
                    self.model_to_props.append([("energy", 0)])
                    file_idx += 1
                
                if "forces" in self.required_properties:
                    m = model_force(model_files[file_idx], self.device)
                    m.eval()
                    m.to(self.device)
                    load_models.append(m)
                    self.model_to_props.append([("forces", 0)])
                    file_idx += 1

        if "dipole_moment" in self.required_properties:
            m = model_dipole(model_files[file_idx], self.device)
            m.eval()
            m.to(self.device)
            load_models.append(m)
            self.model_to_props.append([("dipole_moment", 0)])
            file_idx += 1

        if "polarizability" in self.required_properties:
            m = model_polar(model_files[file_idx], self.device)
            m.eval()
            m.to(self.device)
            load_models.append(m)
            self.model_to_props.append([("polarizability", 0)])
            file_idx += 1

        return load_models

    def calculate(self, system: System):

        inputs = self._generate_input(system)
        numbers = inputs['_atomic_numbers']
        positions = inputs['_positions']
        idx_m = inputs['_idx_m']
        box = inputs['_cell']

        if torch.any(inputs['_pbc'] == False):
            prediction = [model(z=numbers, pos=positions, cell=None, batch=idx_m) for model in self.models]
        else:
            prediction = [model(z=numbers, pos=positions, cell=box, batch=idx_m) for model in self.models]

        self.results = {}
        for pred, prop_info in zip(prediction, self.model_to_props):
            for prop_name, pred_idx in prop_info:
                if isinstance(pred, (tuple, list)):
                    val = pred[pred_idx]
                else:
                    val = pred
                self.results[prop_name] = val

        if 'forces' in self.results:
            self.results['forces'] = self.results['forces'].reshape(-1, 3)
          
        if 'energy' in self.results:
            self.results['energy'] = self.results['energy'].reshape(-1)
           
        if 'dipole_moment' in self.results:
            self.results['dipole_moment'] = self.results['dipole_moment'].reshape(-1)
           
        if 'polarizability' in self.results:
            self.results['polarizability'] = self.results['polarizability'].reshape(-1, 3)
          
        self._update_system(system)

    def _generate_input(self, system: System) -> Dict[str, torch.Tensor]:
        inputs = {}
        inputs_list = self._get_system_molecules(system)
        for key, value in inputs_list.items():
            inputs[key] = value.to(self.device)

        neighbors = self.neighbor_list.get_neighbors(inputs)
        inputs.update(neighbors)
        return inputs


