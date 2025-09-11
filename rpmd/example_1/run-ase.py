from ase.io import read
import sys
sys.path.append('/public/home/')
from PIMD3.calculator.ase_calculator import ASEDetaNetCalculator
from ase.calculators.socketio import SocketClient
from schnetpack.transform import ASENeighborList
import torch
import warnings
warnings.filterwarnings("ignore")


# Define atoms object
molecule_path='/Urea_MS.cif'
atoms = read(molecule_path)

model_energy_path= '/model/energy7.pth'
model_force_path = '/model/force7.pth'
model_files=[model_force_path,model_energy_path]

# Set ASE calculator #################
calcs = []
calc = ASEDetaNetCalculator(
        model_files,
        neighbor_list=ASENeighborList,
        properties=["forces","energy"],
        device=torch.device("cpu"))
atoms.set_calculator(calc)

# Create Client
host = "driver_urea"
client = SocketClient(unixsocket=host)
client.run(atoms)
