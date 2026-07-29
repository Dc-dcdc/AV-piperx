import unittest

from train.s1_pretrain.train.optimizer_utils import (
    is_coupling_parameter,
    is_visual_backbone_parameter,
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

    def test_visual_backbone_recognizes_real_diffusion_parameter_prefix(self):
        """验证DiffusionRgbEncoder的真实ResNet参数进入低学习率Backbone组。"""
        backbone_names = [
            "diffusion.rgb_encoder.backbone.0.weight",
            "model.backbone.conv1.weight",
            "image_encoder.resnet.layer1.0.weight",
            "policy.visual_encoders.front.layer4.weight",
        ]
        for parameter_name in backbone_names:
            with self.subTest(parameter_name=parameter_name):
                self.assertTrue(is_visual_backbone_parameter(parameter_name))

        self.assertFalse(
            is_visual_backbone_parameter(
                "diffusion.rgb_encoder.pool.temperature"
            )
        )
        self.assertFalse(
            is_visual_backbone_parameter("diffusion.arm_unet.final_conv.weight")
        )

    def test_partition_is_complete_and_disjoint(self):
        parameters = {
            "diffusion.arm_unet.weight": object(),
            "diffusion.view_to_arm_attention.weight": object(),
            "diffusion.rgb_encoder.backbone.0.weight": object(),
        }

        main, coupling, backbone = partition_optimizer_parameters(
            parameters.items(),
            is_backbone_parameter=is_visual_backbone_parameter,
        )

        self.assertEqual(main, [parameters["diffusion.arm_unet.weight"]])
        self.assertEqual(
            coupling,
            [parameters["diffusion.view_to_arm_attention.weight"]],
        )
        self.assertEqual(
            backbone,
            [parameters["diffusion.rgb_encoder.backbone.0.weight"]],
        )


if __name__ == "__main__":
    unittest.main()
