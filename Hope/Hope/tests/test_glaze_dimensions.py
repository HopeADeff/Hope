import sys
import unittest
from unittest.mock import MagicMock, patch, ANY
import torch
import numpy as np
from PIL import Image
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.modules['diffusers'] = MagicMock()
sys.modules['transformers'] = MagicMock()
sys.modules['gpu_utils'] = MagicMock()

from glaze_protection import GlazeStyleProtector

class TestGlazeDimensions(unittest.TestCase):
    def setUp(self):
        self.mock_vae = MagicMock()
        self.mock_vae.encode.return_value.latent_dist.sample.return_value = torch.zeros(1, 4, 64, 64)
        self.mock_vae.decode.return_value.sample = torch.zeros(1, 3, 512, 512) 
        self.mock_scheduler = MagicMock()
        self.mock_scheduler.add_noise.return_value = torch.zeros(1, 4, 64, 64)
        self.mock_unet = MagicMock()
        self.mock_unet.return_value.sample = torch.zeros(1, 4, 64, 64)

    @patch('glaze_protection.get_device', return_value='cpu')
    @patch('glaze_protection.get_model_path', return_value='models')
    @patch('glaze_protection.AutoencoderKL')
    @patch('glaze_protection.UNet2DConditionModel')
    @patch('glaze_protection.PNDMScheduler')
    @patch('glaze_protection.CLIPTokenizer')
    @patch('glaze_protection.CLIPTextModel')
    @patch('glaze_protection.Image.open')
    @patch('glaze_protection.validate_image_path')
    @patch('glaze_protection.Path') 
    def test_dimensions_preserved(self, MockPath, MockValidate, MockImageOpen, MockClipText, MockClipTok, MockScheduler, MockUNet, MockVAE, MockPathFunc, MockDevice):
        
        MockVAE.from_pretrained.return_value.to.return_value = self.mock_vae
        MockUNet.from_pretrained.return_value.to.return_value = self.mock_unet
        MockScheduler.from_pretrained.return_value = self.mock_scheduler
        
        input_width, input_height = 300, 600
        real_input_img = Image.new('RGB', (input_width, input_height))
        
        MockImageOpen.return_value = real_input_img
        
        protector = GlazeStyleProtector(verbose=False)
        protector.iterations = 1 
        
        with patch('glaze_protection.Image.fromarray') as MockFromInfo:
            mock_output_img = MagicMock()
            MockFromInfo.return_value = mock_output_img
            
            protector.protect_image('fake_input.jpg', 'fake_output.jpg')
            
            mock_output_img.resize.assert_called_with((300, 600), Image.LANCZOS)

if __name__ == '__main__':
    unittest.main()
