import pytest
import torch
import numpy as np
import os
import tempfile
import shutil
from pathlib import Path
import h5py
import json
from unittest.mock import MagicMock, patch

from diffusion_trainer import DiffusionTrainer
import train_diffu

class TestConfig:
    """Test configuration helper with settings matching production config"""
    def __init__(self):
        self.base_config = {
            # File paths and dataset info
            'FILES': ['dataset_1_photons_1.hdf5'],
            'EVAL': ['dataset_1_photons_2.hdf5'],
            'BIN_FILE': "/work1/cms_mlsim/oamram/CaloDiffusion/CaloChallenge/code/binning_dataset_1_photons.xml",
            'PART_TYPE': 'photon',
            'AVG_SHOWER_LOC': "/wclustre/cms_mlsim/denoise/CaloChallenge/dataset_2_avg_showers.hdf5",
            'DATASET_NUM': 1,
            'HOLDOUT': 0,

            # Shape configurations
            'SHAPE_ORIG': [-1, 368],
            'SHAPE': [-1, 5, 10, 30, 1],
            'SHAPE_PAD': [-1, 1, 5, 10, 30],

            # Training parameters
            'BATCH': 128,
            'LR': 4e-4,
            'MAXEPOCH': 1000,
            'NLAYERS': 3,
            'EARLYSTOP': 20,

            # Architecture parameters
            'LAYER_SIZE_AE': [32, 64, 64, 32],
            'DIM_RED_AE': [0, 2, 0, 2],
            'LAYER_SIZE_UNET': [16, 16, 16, 32],
            'COND_SIZE_UNET': 128,
            'KERNEL': [3, 3, 3],
            'STRIDE': [3, 2, 2],
            'BLOCK_ATTN': True,
            'MID_ATTN': True,
            'COMPRESS_Z': True,
            'ACT': 'swish',
            'EMBED': 128,

            # Energy parameters
            'EMAX': 4194.304,
            'EMIN': 0.256,
            'ECUT': 0.0000001,
            'logE': True,
            'MAXDEP': 3.1,

            # Geometry and input configuration
            'CYLINDRICAL': True,
            'SHOWERMAP': 'logit-norm',
            'R_Z_INPUT': True,
            'PHI_INPUT': True,

            # Diffusion parameters
            'BETA_MAX': 0.02,
            'NOISE_SCHED': 'cosine',
            'NSTEPS': 400,
            'COLD_DIFFU': False,
            'COLD_NOISE': 1.0,
            'TRAINING_OBJ': 'noise_pred',
            'LOSS_TYPE': 'l2',
            'TIME_EMBED': 'sigma',
            'COND_EMBED': 'id',
            'SHOWER_EMBED': 'orig-NN',
            'CHECKPOINT_NAME': 'dataset1_phot'
        }
        
        self.args = type('Args', (), {
            'model': 'Diffu',
            'config': self.base_config,
            'data_folder': '',
            'nevts': 1000,
            'frac': 0.8,
            'load': False,
            'seed': 42,
            'reset_training': False,
            'layer_sizes': None,
            'binning_file': None,
            'save_folder_append': None,
            'model_loc': None
        })()

@pytest.fixture
def test_config():
    return TestConfig()

@pytest.fixture
def temp_dir():
    """Create temporary directory for test artifacts"""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)

@pytest.fixture
def mock_data(temp_dir):
    """Create mock training data file matching the expected structure"""
    data_path = os.path.join(temp_dir, "dataset_1_photons_1.hdf5")
    
    # Create synthetic data matching the expected shapes
    num_events = 1000
    shower_shape = (num_events, 5, 10, 30)  # Matches SHAPE config
    data = np.random.normal(0, 1, shower_shape).astype(np.float32)
    
    # Generate energies in the correct range
    log_emin = np.log(0.256)  # EMIN from config
    log_emax = np.log(4194.304)  # EMAX from config
    energies = np.exp(np.random.uniform(log_emin, log_emax, num_events)).astype(np.float32)
    
    with h5py.File(data_path, 'w') as f:
        f.create_dataset('incident_energies', data=energies)
        f.create_dataset('showers', data=data)
        
        # Add geometry information
        f.create_dataset('R', data=np.random.uniform(0, 100, shower_shape).astype(np.float32))
        f.create_dataset('Z', data=np.random.uniform(-100, 100, shower_shape).astype(np.float32))
        f.create_dataset('Phi', data=np.random.uniform(-np.pi, np.pi, shower_shape).astype(np.float32))
        
        # Add metadata
        f.attrs['num_events'] = num_events
    
    return data_path

@pytest.fixture
def mock_binning_file(temp_dir):
    """Create mock binning file matching the photon dataset configuration"""
    binning_file = os.path.join(temp_dir, "binning_dataset_1_photons.xml")
    
    binning_content = """<?xml version="1.0" ?>
    <Binning>
        <Bins>
            <Axis id="0" number_of_bins="5" title="R" minimum="0" maximum="100" />
            <Axis id="1" number_of_bins="10" title="Z" minimum="-100" maximum="100" />
            <Axis id="2" number_of_bins="30" title="Phi" minimum="-3.14159" maximum="3.14159" />
        </Bins>
    </Binning>
    """
    
    with open(binning_file, 'w') as f:
        f.write(binning_content)
    
    return binning_file

@pytest.fixture
def mock_avg_showers(temp_dir):
    """Create mock average showers file for cold diffusion"""
    avg_file = os.path.join(temp_dir, "avg_showers.hdf5")
    
    with h5py.File(avg_file, 'w') as f:
        shape = (10, 5, 10, 30)  # 10 energy bins
        f.create_dataset('avg_showers', data=np.random.normal(0, 1, shape).astype(np.float32))
        f.create_dataset('std_showers', data=np.abs(np.random.normal(0, 0.1, shape)).astype(np.float32))
        f.create_dataset('E_bins', data=np.logspace(np.log10(0.256), np.log10(4194.304), 11).astype(np.float32))
    
    return avg_file

@pytest.fixture
def setup_identical_seeds():
    """Ensure both implementations use the same random seeds"""
    torch.manual_seed(42)
    np.random.seed(42)

class TestDiffusionImplementations:
    """Test suite comparing original and new implementations"""
    
    def test_data_loading(self, test_config, mock_data, mock_binning_file, temp_dir, setup_identical_seeds):
        """Test that both implementations load and preprocess data identically"""
        test_config.args.data_folder = temp_dir
        test_config.args.config["BIN_FILE"] = mock_binning_file
        test_config.args.binning_file = mock_binning_file
        
        trainer = DiffusionTrainer(test_config.args)
        trainer.load_data()
        
        # Load data with original implementation
        data_orig, energies_orig = train_diffu.utils.DataLoader(
            os.path.join(temp_dir, test_config.base_config["FILES"][0]),
            test_config.base_config["SHAPE_PAD"],
            emax=test_config.base_config["EMAX"],
            emin=test_config.base_config["EMIN"],
            nevts=test_config.args.nevts,
            max_deposit=test_config.base_config["MAXDEP"],
            logE=test_config.base_config["logE"],
            showerMap=test_config.base_config["SHOWERMAP"],
            dataset_num=test_config.base_config["DATASET_NUM"],
            nholdout=test_config.base_config["HOLDOUT"],
            orig_shape="orig" in test_config.base_config["SHOWER_EMBED"]
        )
        
        assert trainer.train_loader.dataset.tensors[0].shape == torch.from_numpy(energies_orig).shape
        assert trainer.train_loader.dataset.tensors[1].shape == torch.from_numpy(data_orig).shape
        assert torch.allclose(
            trainer.train_loader.dataset.tensors[0].cpu(),
            torch.from_numpy(energies_orig),
            rtol=1e-5
        )
        assert torch.allclose(
            trainer.train_loader.dataset.tensors[1].cpu(),
            torch.from_numpy(data_orig),
            rtol=1e-5
        )

    def test_model_initialization(self, test_config, mock_binning_file, setup_identical_seeds):
        """Test that models are initialized identically"""
        test_config.args.binning_file = mock_binning_file
        
        trainer = DiffusionTrainer(test_config.args)
        trainer.data_shape = test_config.base_config["SHAPE_PAD"][1:]
        trainer.setup_model()
        
        model_orig = train_diffu.CaloDiffu(
            test_config.base_config["SHAPE_PAD"][1:],
            config=test_config.base_config,
            training_obj=test_config.base_config["TRAINING_OBJ"],
            R_Z_inputs=test_config.base_config["R_Z_INPUT"],
            nsteps=test_config.base_config["NSTEPS"],
            cold_diffu=test_config.base_config["COLD_DIFFU"]
        )
        
        assert str(trainer.model) == str(model_orig)
        
        for (n1, p1), (n2, p2) in zip(trainer.model.named_parameters(), model_orig.named_parameters()):
            assert n1 == n2, f"Parameter names differ: {n1} vs {n2}"
            assert torch.allclose(p1, p2), f"Parameter values differ for {n1}"
            
        assert trainer.model._num_embed == test_config.base_config["EMBED"]
        assert trainer.model.R_Z_inputs == test_config.base_config["R_Z_INPUT"]

    def test_training_loop(self, test_config, mock_data, mock_binning_file, temp_dir, setup_identical_seeds):
        """Test that training produces identical results"""
        test_config.args.data_folder = temp_dir
        test_config.args.config["BIN_FILE"] = mock_binning_file
        test_config.args.binning_file = mock_binning_file
        test_config.base_config["MAXEPOCH"] = 2  # Reduce epochs for testing
        
        # Train with new implementation
        trainer = DiffusionTrainer(test_config.args)
        trainer.load_data()
        trainer.setup_model()
        trainer.setup_training()
        
        new_losses = []
        def mock_save(*args, **kwargs):
            pass
        trainer.save_checkpoint = mock_save
        
        train_metrics = trainer.train()
        
        # Train with original implementation
        with patch('train_diffu.torch.save'):
            orig_metrics = train_diffu.train_diffu(test_config.args)
            
        assert train_metrics['loss'] == pytest.approx(orig_metrics['loss'], rel=1e-5)

    @pytest.mark.parametrize("model_type", ["Diffu", "Latent_Diffu"])
    def test_model_variants(self, test_config, mock_data, mock_binning_file, temp_dir, setup_identical_seeds, model_type):
        """Test different model variants"""
        test_config.args.model = model_type
        test_config.args.data_folder = temp_dir
        test_config.args.config["BIN_FILE"] = mock_binning_file
        test_config.args.binning_file = mock_binning_file
        
        if model_type == "Latent_Diffu":
            ae_path = os.path.join(temp_dir, "ae_checkpoint.pth")
            mock_ae_state = {
                'model_state_dict': {
                    'encoder.embed.weight': torch.randn(test_config.base_config['EMBED'], 
                                                      test_config.base_config['LAYER_SIZE_AE'][0])
                }
            }
            torch.save(mock_ae_state, ae_path)
            test_config.args.model_loc = ae_path
        
        trainer = DiffusionTrainer(test_config.args)
        trainer.load_data()
        trainer.setup_model()
        
        if model_type == "Diffu":
            assert not hasattr(trainer, 'autoencoder')
        else:
            assert hasattr(trainer, 'autoencoder')
            assert trainer.encoded_stats is not None

    def test_checkpoint_saving_loading(self, test_config, mock_data, mock_binning_file, temp_dir):
        """Test checkpoint functionality"""
        test_config.args.data_folder = temp_dir
        test_config.args.config["BIN_FILE"] = mock_binning_file
        test_config.args.binning_file = mock_binning_file
        
        trainer = DiffusionTrainer(test_config.args)
        trainer.load_data()
        trainer.setup_model()
        trainer.setup_training()
        
        # Save checkpoint
        trainer.save_checkpoint()
        
        # Create new trainer and load checkpoint
        new_trainer = DiffusionTrainer(test_config.args)
        new_trainer.load_data()
        new_trainer.setup_model()
        new_trainer.setup_training()
        new_trainer.load_checkpoint()
        
        # Verify loaded state matches
        assert new_trainer.epoch == trainer.epoch
        assert new_trainer.best_metrics == trainer.best_metrics
        
        # Compare model parameters
        for (p1, p2) in zip(new_trainer.model.parameters(), trainer.model.parameters()):
            assert torch.allclose(p1, p2)

if __name__ == "__main__":
    pytest.main([__file__, "-v"])