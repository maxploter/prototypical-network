import unittest
import torch
from loss.autoencoder_loss import AutoencoderLoss


class TestAutoencoderLoss(unittest.TestCase):
    def test_forward(self):
        """Test that forward returns correct dictionary structure"""
        loss_fn = AutoencoderLoss()

        # Create dummy data
        batch_size = 4
        img_size = 28
        outputs = torch.randn(batch_size, 1, img_size, img_size)
        targets = torch.randn(batch_size, 1, img_size, img_size)

        # Get loss dictionary
        loss_dict = loss_fn(outputs, targets)

        # Check that loss_dict is a dictionary
        self.assertIsInstance(loss_dict, dict)

        # Check that required keys are present
        self.assertIn('loss_mse', loss_dict)
        self.assertIn('psnr', loss_dict)

        # Check loss_mse
        mse_loss = loss_dict['loss_mse']
        self.assertIsNotNone(mse_loss)
        self.assertTrue(torch.is_tensor(mse_loss))
        self.assertGreaterEqual(mse_loss.item(), 0.0, "MSE loss should be non-negative")

        # Check psnr
        psnr = loss_dict['psnr']
        self.assertIsNotNone(psnr)
        self.assertTrue(torch.is_tensor(psnr))
        # PSNR can be negative for large MSE, just check it's finite
        self.assertTrue(torch.isfinite(psnr), "PSNR should be finite")

    def test_weight_dict(self):
        """Test that weight_dict is properly defined"""
        loss_fn = AutoencoderLoss()

        self.assertTrue(hasattr(loss_fn, 'weight_dict'))
        self.assertIn('loss_mse', loss_fn.weight_dict)
        self.assertEqual(loss_fn.weight_dict['loss_mse'], 1.0)

    def test_perfect_reconstruction(self):
        """Test PSNR is very high for perfect reconstruction"""
        loss_fn = AutoencoderLoss()

        # Perfect reconstruction (same input and output)
        batch_size = 2
        img_size = 28
        data = torch.randn(batch_size, 1, img_size, img_size)

        loss_dict = loss_fn(data, data)

        # MSE should be very close to 0
        mse_loss = loss_dict['loss_mse']
        self.assertLess(mse_loss.item(), 1e-6, "MSE should be near zero for perfect reconstruction")

        # PSNR should be very high (or inf)
        psnr = loss_dict['psnr']
        self.assertGreater(psnr.item(), 100.0, "PSNR should be very high for perfect reconstruction")

    def test_only_loss_mse_in_weight_dict(self):
        """Test that only loss_mse is in weight_dict (psnr should not be)"""
        loss_fn = AutoencoderLoss()

        # psnr should NOT be in weight_dict (it's for logging only)
        self.assertNotIn('psnr', loss_fn.weight_dict)

        # Only loss_mse should be in weight_dict
        self.assertEqual(len(loss_fn.weight_dict), 1)


if __name__ == '__main__':
    unittest.main()
