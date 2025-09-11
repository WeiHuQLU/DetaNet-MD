import torch
from PIMD3.calculator.models_load import model_force,model_energy
from schnetpack.interfaces import AtomsConverter
import os
import ase
from ase import units
from ase.constraints import FixAtoms
from ase.calculators.calculator import Calculator, all_changes
import numpy as np
import schnetpack.task

class ASEDetaNetCalculator(Calculator):
    implemented_properties = ["energy", "forces"]

    def __init__(
            self,
            model_files:list[str],
            neighbor_list: schnetpack.transform.Transform,
            properties:list[str],
            device:torch.device=torch.device('cuda'),
            **kwargs
    ):
         self.required_properties= properties
         Calculator.__init__(self, **kwargs)
         self.device = device
         self.models_files=model_files
         self.neighbor_list=neighbor_list
         self.models=self.load_model(model_files)

         

    def load_model(self, model_files: list) :
        """
        Load an individual model, activate stress computation

        Args:
            model_file (str): path to model.

        Returns:
           AtomisticTask: loaded schnetpack model
        """

        get_models=[]
        
        for str in self.required_properties:
             if str == "forces":
                get_models.append(model_force)
             elif str == "energy":
                get_models.append(model_energy)
        load_models= [i(j,self.device) for i, j in zip(get_models, model_files)]
        self.models=[]
        for model in load_models:
                model.eval()
                model.to(self.device)
                self. models.append(model)
        return self.models


    def calculate(
            self,
            atoms=None,
            Properties=("energy", "forces"),
            system_changes=all_changes
    ):
        """
        Args:
            atoms (ase.Atoms): ASE atoms object.
            properties (list of str): Properties to calculate.
            system_changes (list of str): List of changes for ASE.
        """
        # First call original calculator to set atoms attribute
        # (see https://wiki.fysik.dtu.dk/ase/_modules/ase/calculators/calculator.html#Calculator)

        Calculator.calculate(self, atoms, Properties, system_changes)

        # set up converter

        self.converter =AtomsConverter(
                self.neighbor_list(cutoff=5.0), dtype=torch.float32, device=self.device
            )

        # Call model
        inputs = self.converter(atoms)
        numbers = inputs['_atomic_numbers']
        positions = inputs['_positions']
        #idx_i = inputs['_idx_i']
        #idx_j = inputs['_idx_j']
        idx_m = inputs['_idx_m']
        #idx_ji = torch.cat((idx_j.unsqueeze(0), idx_i.unsqueeze(0)), dim=0).long()
        box=inputs['_cell']
        if torch.any(inputs['_pbc']==False):
            prediction = [model(z=numbers, pos=positions,box=None,batch=idx_m) for model in self.models]
            
        else:
             prediction = [model(z=numbers, pos=positions,box=box,batch=idx_m) for model in self.models]

        # Convert outputs to ASE calculator format
        self.results = {i: j for i, j in zip(self.required_properties, prediction)}
        if 'forces' in self.results:
            self.results['forces'] = self.results['forces'].reshape(-1, 3).detach().cpu().numpy().astype(np.float64)
        if 'energy' in self.results:
            self.results['energy'] = self.results['energy'].reshape(-1).detach().cpu().numpy().astype(np.float64)


