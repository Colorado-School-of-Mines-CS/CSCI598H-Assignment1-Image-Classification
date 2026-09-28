"""Small public checks that use synthetic tensors instead of CIFAR-10."""

import unittest

import torch
from torch import nn

from models import (
    ConvClassifier,
    LinearClassifier,
    ThreeLayerClassifier,
    VisionTransformerClassifier,
    build_model,
)
from train import build_optimizer, compute_accuracy, softmax_cross_entropy


class PublicSmokeTests(unittest.TestCase):
    def test_all_models_return_logits(self):
        cases = [
            (LinearClassifier(), (2, 3, 32, 32)),
            (ThreeLayerClassifier(), (2, 3, 32, 32)),
            (ConvClassifier(), (2, 3, 32, 32)),
            (VisionTransformerClassifier(), (2, 3, 32, 32)),
            (ConvClassifier(num_layers=5), (2, 3, 32, 32)),
            (VisionTransformerClassifier(num_layers=5), (2, 3, 32, 32)),
        ]
        for model, shape in cases:
            with self.subTest(model=type(model).__name__):
                output = model(torch.randn(*shape))
                self.assertEqual(tuple(output.shape), (2, 10))

    def test_loss_is_finite_and_differentiable(self):
        logits = torch.tensor([[10000.0, 9999.0], [-9000.0, 9000.0]],
                              requires_grad=True)
        loss = softmax_cross_entropy(logits, torch.tensor([0, 1]))
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_accuracy(self):
        logits = torch.tensor([[3.0, 1.0], [0.0, 2.0], [4.0, 1.0]])
        self.assertAlmostEqual(compute_accuracy(logits, torch.tensor([0, 0, 0])), 2 / 3)

    def test_model_factory(self):
        self.assertIsInstance(build_model("linear"), LinearClassifier)
        self.assertIsInstance(build_model("three_layer"), ThreeLayerClassifier)
        self.assertIsInstance(build_model("conv"), ConvClassifier)
        self.assertIsInstance(build_model("vit"), VisionTransformerClassifier)
        self.assertEqual(build_model("conv", num_layers=5).num_layers, 5)

    def test_optimizer_factory(self):
        parameter = nn.Parameter(torch.tensor([1.0]))
        self.assertIsInstance(build_optimizer("sgd", [parameter], 0.01), torch.optim.SGD)
        self.assertIsInstance(build_optimizer("adam", [parameter], 0.001), torch.optim.Adam)
        sgd_momentum = build_optimizer("sgd", [parameter], 0.01, momentum=0.9)
        self.assertEqual(sgd_momentum.param_groups[0]["momentum"], 0.9)
        self.assertIsInstance(build_optimizer("adamw", [parameter], 0.001), torch.optim.AdamW)


if __name__ == "__main__":
    unittest.main()
