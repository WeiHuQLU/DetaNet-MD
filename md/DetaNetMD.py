import torch
import os
from schnetpack.md import UniformInit,MaxwellBoltzmannInit
from schnetpack.md.integrators import VelocityVerlet
from schnetpack.md.neighborlist_md import NeighborListMD
from schnetpack.transform import ASENeighborList
from md.DetaNetCalculator import DetaNetCalculator
from schnetpack.md.simulation_hooks import NHCThermostat
from schnetpack.md.simulation_hooks import callback_hooks
from schnetpack.md import Simulator
from schnetpack import properties
from schnetpack.md.integrators import RingPolymer
from schnetpack.md.simulation_hooks import NHCRingPolymerThermostat
from schnetpack.md import System
class  MolecularDynamics :
    def __init__(self,molecule,model_files,md_params,md_workdir,md_device='cuda'):
        self.molecule = molecule
        self.md_params = md_params
        self.md_device=md_device
        self.model_files=model_files
        self.md_workdir=md_workdir
        self.md_system = System()
        self.md_system.load_molecules(
                 molecule,
            self.md_params["n_replicas"],
    position_unit_input="Angstrom")


        # Set up the initializer
        '''
        md_initializer = UniformInit(
            self.md_params["system_temperature"],
            remove_center_of_mass=True,
            remove_translation=True,
            remove_rotation=True,
        )
        '''
        #MaxwellBoltzmannInit
        md_initializer=MaxwellBoltzmannInit(self.md_params["system_temperature"],
        remove_translation=True,
        remove_rotation=True,
       )

        # Initialize the system momenta
        md_initializer.initialize_system(self.md_system)
        # Set up the integrator
        self.md_integrator = VelocityVerlet(self.md_params["time_step"])
        # set cutoff and buffer region
        cutoff = 5.0  # Angstrom (units used in model)
        cutoff_shell = 2.0  # Angstrom

        # initialize neighbor list for MD using the ASENeighborlist as basis ,'polarizability'
        md_neighborlist = NeighborListMD(
            cutoff,
            cutoff_shell,
            ASENeighborList,
        )

        self.md_calculator = DetaNetCalculator(
            model_files=self.model_files,
            force_key="forces",
            energy_unit="eV",
            position_unit="Angstrom",
            neighbor_list=md_neighborlist,
            energy_key=None,
            required_properties=self.md_params["required_properties"],
            device=torch.device(self.md_device)
        )

        # Nose-Hover chain thermostat

        # Set temperature and thermostat constant
        bath_temperature =self.md_params["bath_temperature"]
        time_constant = self.md_params["time_constant"]
        chain_length = self.md_params["chain_length"]

        # Initialize the thermostat
        Nose_Hover = NHCThermostat(bath_temperature,time_constant,chain_length)
        self.simulation_hooks = [Nose_Hover]

    def run (self,MDRestart=False):

        # Gnerate a directory of not present
        if not os.path.exists(self.md_workdir):
            os.mkdir(self.md_workdir)

        # Path to database
        log_file = os.path.join(self.md_workdir, "simulation.hdf5")

        # Size of the buffer
        buffer_size = 100

        # Set up data streams to store positions, momenta and the energy ,properties.polarizability
        data_streams = [
            callback_hooks.MoleculeStream(store_velocities=True),
            callback_hooks.PropertyStream(
                target_properties=self.md_params["required_properties"]),
        ]

        # Create the file logger
        file_logger = callback_hooks.FileLogger(
            log_file,
            buffer_size,
            data_streams=data_streams,
            every_n_steps=1,  # logging frequency
            precision=32,  # floating point precision used in hdf5 database
        )

        # Update the simulation hooks
        self.simulation_hooks.append(file_logger)

        # Set the path to the checkpoint file
        chk_file = os.path.join(self.md_workdir, 'simulation.chk')

        # Create the checkpoint logger
        checkpoint = callback_hooks.Checkpoint(chk_file, every_n_steps=100)

        # Update the simulation hooks
        self.simulation_hooks.append(checkpoint)

        # use single precision
        md_precision = torch.float32

        md_simulator =Simulator(self.md_system,
                                self.md_integrator,
                                self.md_calculator,
                                simulator_hooks=self.simulation_hooks)

        md_simulator.to(md_precision)
        md_simulator.to(self.md_device)
        

        if MDRestart:
            checkpoint = torch.load(chk_file)
            md_simulator.restart_simulation(checkpoint)
            md_simulator.to(md_precision)
            md_simulator.to(self.md_device)
            run=md_simulator.simulate(self.md_params["n_steps"])
        else:
            run = md_simulator.simulate(self.md_params["n_steps"])

        return run
        

class RPMolecularDynamics:
    def __init__(self,molecule,model_files,rpmd_params,rpmd_workdir,rpmd_device='cuda'):

        self.model_files = model_files
        self.rpmd_params = rpmd_params
        self.molecule= molecule
        self.rpmd_device = rpmd_device
        self.rpmd_workdir =rpmd_workdir
        self.rpmd_system = System()
        self.rpmd_system.load_molecules(
                  molecule,
                  self.rpmd_params["n_replicas"],
                 position_unit_input="Angstrom")

        # Set up the initializer
                # Set up the initializer
        '''
        rpmd_initializer = UniformInit(
            self.rpmd_params["system_temperature"],
            remove_center_of_mass=True,
            remove_translation=True,
            remove_rotation=True,
        )
        '''
        # MaxwellBoltzmannInit
        rpmd_initializer = MaxwellBoltzmannInit(self.rpmd_params["system_temperature"],
                                              remove_translation=True,
                                              remove_rotation=True,
                                              )
        # Initialize the system momenta
        rpmd_initializer.initialize_system(self.rpmd_system)

        # Here, a smaller time step is required for numerical stability

        # Initialize the integrator, RPMD also requires a polymer temperature which determines the coupling of beads.
        # Here, we set it to the system temperature
        self.rpmd_integrator = RingPolymer(
            self.rpmd_params["rpmd_time_step"],
            self.rpmd_params["n_replicas"],
            self.rpmd_params["system_temperature"]
        )

        # set cutoff and buffer region
        cutoff = 5.0  # Angstrom (units used in model)
        cutoff_shell = 2.0  # Angstrom

        # initialize neighbor list for MD using the ASENeighborlist as basis
        md_neighborlist = NeighborListMD(
            cutoff,
            cutoff_shell,
            ASENeighborList,
        )

        self.rpmd_calculator = DetaNetCalculator(
            model_files,  # path to stored model
            "forces",  # force key
            "eV",  # energy units
            "Angstrom",  # length units
            md_neighborlist,  # neighbor list
            energy_key="energy",  # name of potential energies
            required_properties=self.md_params["required_properties"],  # additional properties extracted from the model
            device=torch.device(self.rpmd_device)
        )
        # Set temperature and thermostat constant
        bath_temperature = self.rpmd_params["bath_temperature"]
        time_constant =self.rpmd_params["time_constant"]
        chain_length=self.rpmd_params["chain_length"]

        # Initialize the thermostat
        Nose_Hover = NHCRingPolymerThermostat(bath_temperature, time_constant, chain_length)
        self.RPsimulation_hooks = [Nose_Hover]

    def run(self,MDRestart=False):

        if not os.path.exists(self.rpmd_workdir):
            os.mkdir(self.rpmd_workdir)

        # Path to database
        log_file = os.path.join(self.rpmd_workdir, "simulation1.hdf5")

        # Size of the buffer
        buffer_size = 100

        # Set up data streams to store positions, momenta and the energy
        data_streams = [
            callback_hooks.MoleculeStream(store_velocities=True),
            callback_hooks.PropertyStream(
                target_properties=self.md_params["required_properties"]),
        ]

        # Create the file logger
        file_logger = callback_hooks.FileLogger(
            log_file,
            buffer_size,
            data_streams=data_streams,
            every_n_steps=1,  # logging frequency
            precision=32,  # floating point precision used in hdf5 database
        )

        # Update the simulation hooks
        self.RPsimulation_hooks.append(file_logger)

        # Set the path to the checkpoint file
        chk_file = os.path.join(self.rpmd_workdir, 'simulation.chk')

        # Create the checkpoint logger
        checkpoint = callback_hooks.Checkpoint(chk_file, every_n_steps=100)

        # Update the simulation hooks
        self.RPsimulation_hooks.append(checkpoint)

        # use single precision
        rpmd_precision = torch.float32

        rpmd_simulator =Simulator(self.rpmd_system,
                                self.rpmd_integrator,
                                self.rpmd_calculator,
                                simulator_hooks=self.RPsimulation_hooks)

        rpmd_simulator.to(rpmd_precision)
        rpmd_simulator.to(self.rpmd_device)
       

        if MDRestart:
            checkpoint = torch.load(chk_file)
            rpmd_simulator.restart_simulation(checkpoint)
            rpmd_simulator.to(self.rpmd_device)
            rpmd_simulator.to(rpmd_precision)
            run=rpmd_simulator.simulate(self.rpmd_params["n_steps"])
        else:
            run=rpmd_simulator.simulate(self.rpmd_params["n_steps"])

        return run
