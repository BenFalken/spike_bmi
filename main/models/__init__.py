"""
Object Tracking Module (OTM) - Models Package

This package contains neural network model definitions for spiking neural networks
used in object tracking tasks on neuromorphic hardware.
"""

from .model_bmi import SNN_Speck, NeuronType, create_model as create_model

__all__ = ['SNN_Speck', 'NeuronType', 'create_model']