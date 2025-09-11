import torch
import os
from detanet_model.detanet_pbc import DetaNet
def model_force (params='D:\My_code\detanet\\trained_param\qm12\\force.pth',device:torch.device=torch.device('cuda')):

    model_force = DetaNet(num_features=128,
                          act='swish',
                          maxl=3,
                          num_block=3,
                          radial_type='trainable_bessel',
                          num_radial=64 , # qm7X,12   32
                          attention_head=8,
                          cutoff_lower=0.0,
                         cutoff_upper=4.5,
                         max_num_neighbors=120,
                         check_errors=True,
                         box_vecs=None,
                          dropout=0.0,
                          use_cutoff=False,
                          max_atomic_number=35,  # qm7X,12   17
                          atom_ref=None,
                          scale=1.0,
                          scalar_outsize=1,
                          irreps_out=None,
                          summation=True,
                          norm=False,
                          out_type='scalar',
                          grad_type='force',
                          device=device)
    force_trained_params = torch.load(params,map_location=device)
    model_force.load_state_dict(state_dict=force_trained_params)
    return model_force
def model_energy(params='D:\My_code\detanet\\trained_param\qm12\\energy.pth',device:torch.device=torch.device('cuda')):
    model_energy = DetaNet(num_features=128,
                           act='swish',
                           maxl=3,
                           num_block=3,
                           radial_type='trainable_bessel',
                           num_radial=128,  # qm7X  32
                           attention_head=8,
                           cutoff_lower=0.0,
                           cutoff_upper=4.5,
                           max_num_neighbors=120,
                           check_errors=True,
                           box_vecs=None,
                           dropout=0.0,
                           use_cutoff=False,
                           max_atomic_number=35,  # qm7X  17
                           atom_ref=None,
                           scale=1.0,
                           scalar_outsize=1,
                           irreps_out=None,
                           summation=True,
                           norm=False,
                           out_type='scalar',
                           grad_type=None,
                           device=device)
    energy_trained_params = torch.load(params,map_location=device)
    model_energy.load_state_dict(state_dict=energy_trained_params)
    return model_energy



def model_dipole(params='D:\My_code\detanet\\trained_param\qm12\\dipole.pth',device:torch.device=torch.device('cuda')):
        model_dipole = DetaNet(num_features=128,
                               act='swish',
                               maxl=3,
                               num_block=3,
                               radial_type='trainable_bessel',
                               num_radial=32,  # qm7X  32
                               attention_head=8,
                               cutoff_lower=0.0,
                               cutoff_upper=5.0,
                               max_num_neighbors=120,
                               check_errors=True,
                               box_vecs=None,
                               dropout=0.0,
                               use_cutoff=False,
                               max_atomic_number=35,  # qm7X  17
                               atom_ref=None,
                               scale=1.0,
                               scalar_outsize=1,
                               irreps_out='1o',
                               summation=True,
                               norm=False,
                               out_type='dipole',
                               grad_type=None,
                               device=device)

        dipole_trained_params = torch.load(params)
        model_dipole.load_state_dict(state_dict=dipole_trained_params)
        return model_dipole

def model_polar(params='D:\My_code\detanet\\trained_param\qm12\\polar.pth',device:torch.device=torch.device('cuda')):
    model_polar = DetaNet(num_features=128,
                          act='swish',
                          maxl=3,
                          num_block=3,
                          radial_type='trainable_bessel',
                          num_radial=32,  # qm7X   32
                          attention_head=8,
                          cutoff_lower=0.0,
                         cutoff_upper=5.0,
                         max_num_neighbors=120,
                         check_errors=True,
                         box_vecs=None,
                          dropout=0.0,
                          use_cutoff=False,
                          max_atomic_number=35,  # qm7X   17
                          atom_ref=None,
                          scale=None,
                          scalar_outsize=2,
                          irreps_out='2e',
                          summation=True,
                          norm=False,
                          out_type='2_tensor',
                          grad_type=None,
                          device=device)

    polar_trained_params = torch.load(params)
    model_polar.load_state_dict(state_dict=polar_trained_params)
    return model_polar
