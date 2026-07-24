import unittest

from train.pretrain.optimizer_utils import (
    is_coupling_parameter,
    partition_optimizer_parameters,
)


class PretrainOptimizerUtilsTest(unittest.TestCase):
    def test_coupling_modules_include_attention_adaln_ffn_and_gates(self):
        coupling_names = [
            "diffusion.view_to_arm_attention.in_proj_weight",
            "diffusion.arm_to_view_attention.out_proj.weight",
            "diffusion.arm_coupling_norm.weight",
            "diffusion.view_coupling_norm.bias",
            "diffusion.coupling_timestep_encoder.3.weight",
            "diffusion.role_adaln_coupling.timestep_encoder.1.weight",
            "diffusion.role_adaln_coupling.attention_modulation.1.bias",
            "diffusion.role_adaln_coupling.arm_ffn.0.weight",
            "diffusion.role_adaln_coupling.ffn_modulation.1.weight",
        ]
        for parameter_name in coupling_names:
            with self.subTest(parameter_name=parameter_name):
                self.assertTrue(is_coupling_parameter(parameter_name))

        self.assertFalse(is_coupling_parameter("diffusion.arm_unet.final_conv.weight"))
        self.assertFalse(
            is_coupling_parameter("diffusion.rgb_encoder.backbone.conv1.weight")
        )

    def test_partition_is_complete_and_disjoint(self):
        parameters = {
            "diffusion.arm_unet.weight": object(),
            "diffusion.view_to_arm_attention.weight": object(),
            "image_encoder.weight": object(),
        }

        main, coupling, backbone = partition_optimizer_parameters(
            parameters.items(),
            is_backbone_parameter=lambda name: name.startswith("image_encoder"),
        )

        self.assertEqual(main, [parameters["diffusion.arm_unet.weight"]])
        self.assertEqual(
            coupling,
            [parameters["diffusion.view_to_arm_attention.weight"]],
        )
        self.assertEqual(backbone, [parameters["image_encoder.weight"]])


if __name__ == "__main__":
    unittest.main()
