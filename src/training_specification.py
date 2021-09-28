import json


class TrainingSpecification:

    template = """    
{
    "TrainingSpecification": {
    "TrainingImage": "IMAGE_REPLACE_ME",
    "SupportedHyperParameters": [
        {
            "Name": "batch_size",
            "Description": "The number of samples in each batch.",
            "Type": "Integer",
            "Range": {
                "IntegerParameterRangeSpecification": {
                    "MinValue": "1",
                    "MaxValue": "64"
                }
            },
            "IsTunable": false,
            "IsRequired": false,
            "DefaultValue": "8"
        },
        {
            "Name": "num_of_trainable_layers",
            "Description": "Number of trainable layers. By default, only the Logits layer is trained.",
            "Type": "Integer",
            "Range": {
                "IntegerParameterRangeSpecification": {
                    "MinValue": "1",
                    "MaxValue": "4"
                }
            },
            "IsTunable": false,
            "IsRequired": false,
            "DefaultValue": "2"
        },
        {
            "Name": "max_number_of_steps",
            "Description": "The maximum number of training steps.",
            "Type": "Integer",
            "Range": {
                "IntegerParameterRangeSpecification": {
                    "MinValue": "1",
                    "MaxValue": "100000"
                }
            },
            "IsTunable": false,
            "IsRequired": false,
            "DefaultValue": "100"
        },
        {
            "Name": "random_image_flip",
            "Description": "Enable random image flip (horizontally). Only Enabled if apply_image_augmentation flag is also enabled",
            "Type": "Integer",
            "Range": {
                "IntegerParameterRangeSpecification": {
                    "MinValue": "0",
                    "MaxValue": "1"
                }
            },
            "IsTunable": false,
            "IsRequired": false,
            "DefaultValue": "1"
        },
        {
            "Name": "random_image_crop",
            "Description": "Enable random cropping of images. Only Enabled if apply_image_augmentation flag is also enabled",
            "Type": "Integer",
            "Range": {
                "IntegerParameterRangeSpecification": {
                    "MinValue": "0",
                    "MaxValue": "1"
                }
            },
            "IsTunable": false,
            "IsRequired": false,
            "DefaultValue": "1"
        },
        {
            "Name": "min_object_covered",
            "Description": "The remaining cropped image must contain at least this fraction of the whole image. Only Enabled if apply_image_augmentation flag is also enabled",
            "Type": "Continuous",
            "Range": {
                "ContinuousParameterRangeSpecification": {
                    "MinValue": "0.1",
                    "MaxValue": "1.0"
                }
            },
            "IsTunable": false,
            "IsRequired": false,
            "DefaultValue": "0.9"
        },
        {
            "Name": "random_image_rotation",
            "Description": "Enable random image rotation counter-clockwise by 90, 180, 270, or 360 degrees. Only Enabled if apply_image_augmentation flag is also enabled",
            "Type": "Integer",
            "Range": {
                "IntegerParameterRangeSpecification": {
                    "MinValue": "0",
                    "MaxValue": "1"
                }
            },
            "IsTunable": false,
            "IsRequired": false,
            "DefaultValue": "1"
        },
        {
            "Name": "apply_image_augmentation",
            "Description": "Enable random image augmentation during preprocessing for training.",
            "Type": "Integer",
            "Range": {
                "IntegerParameterRangeSpecification": {
                    "MinValue": "0",
                    "MaxValue": "1"
                }
            },
            "IsTunable": false,
            "IsRequired": false,
            "DefaultValue": "1"
        },
        {
            "Name": "learning_rate",
            "Description": "Initial learning rate.",
            "Type": "Continuous",
            "Range": {
                "ContinuousParameterRangeSpecification": {
                    "MinValue": "0.0001",
                    "MaxValue": "0.1"
                }
            },
            "IsTunable": false,
            "IsRequired": false,
            "DefaultValue": "0.01"
        },
        {
            "Name": "model_name",
            "Description": "The name of the architecture to train.",
            "Type": "Categorical",
            "Range": {
                "CategoricalParameterRangeSpecification": {
                    "Values": [
                        "mobilenet_v1_075",
                        "mobilenet_v1_050",
                        "mobilenet_v1_025",
                        "mobilenet_v1",
                        "inception_v1"
                    ]
                }
            },
            "IsTunable": false,
            "IsRequired": false,
            "DefaultValue": "mobilenet_v1"
        },
        {
            "Name": "optimizer",
            "Description": "The name of the optimizer, one of adadelta, adagrad, adam, ftrl, momentum, sgd or rmsprop",
            "Type": "Categorical",
            "Range": {
                "CategoricalParameterRangeSpecification": {
                    "Values": [
                        "adadelta",
                        "adagrad",
                        "adam",
                        "ftrl",
                        "momentum",
                        "sgd",
                        "rmsprop"
                    ]
                }
            },
            "IsTunable": false,
            "IsRequired": false,
            "DefaultValue": "adam"
        }
    ],
    "SupportedTrainingInstanceTypes": INSTANCES_REPLACE_ME,
    "SupportsDistributedTraining": false,
    "MetricDefinitions": METRICS_REPLACE_ME,
    "TrainingChannels": CHANNELS_REPLACE_ME,
    "SupportedTuningJobObjectiveMetrics": TUNING_OBJECTIVES_REPLACE_ME
    }
}
"""

    def get_training_specification_dict(
        self,
        ecr_image,
        supports_gpu,
        supported_channels=None,
        supported_metrics=None,
        supported_tuning_job_objective_metrics=None,
    ):
        return json.loads(
            self.get_training_specification_json(
                ecr_image,
                supports_gpu,
                supported_channels,
                supported_metrics,
                supported_tuning_job_objective_metrics,
            )
        )

    def get_training_specification_json(
        self,
        ecr_image,
        supports_gpu,
        supported_channels=None,
        supported_metrics=None,
        supported_tuning_job_objective_metrics=None,
    ):
        if supported_channels is None:
            print("Please provide at least one supported channel")
            raise ValueError("Please provide at least one supported channel")

        if supported_metrics is None:
            supported_metrics = []
        if supported_tuning_job_objective_metrics is None:
            supported_tuning_job_objective_metrics = []

        return (
            self.template.replace("IMAGE_REPLACE_ME", ecr_image)
            .replace("INSTANCES_REPLACE_ME", self.get_supported_instances(supports_gpu))
            .replace(
                "CHANNELS_REPLACE_ME",
                json.dumps([ob.__dict__ for ob in supported_channels], indent=4, sort_keys=True),
            )
            .replace(
                "METRICS_REPLACE_ME",
                json.dumps([ob.__dict__ for ob in supported_metrics], indent=4, sort_keys=True),
            )
            .replace(
                "TUNING_OBJECTIVES_REPLACE_ME",
                json.dumps(
                    [ob.__dict__ for ob in supported_tuning_job_objective_metrics],
                    indent=4,
                    sort_keys=True,
                ),
            )
        )

    @staticmethod
    def get_supported_instances(supports_gpu):
        cpu_list = [
            "ml.m4.xlarge",
            "ml.m4.2xlarge",
            "ml.m4.4xlarge",
            "ml.m4.10xlarge",
            "ml.m4.16xlarge",
            "ml.m5.large",
            "ml.m5.xlarge",
            "ml.m5.2xlarge",
            "ml.m5.4xlarge",
            "ml.m5.12xlarge",
            "ml.m5.24xlarge",
            "ml.c4.xlarge",
            "ml.c4.2xlarge",
            "ml.c4.4xlarge",
            "ml.c4.8xlarge",
            "ml.c5.xlarge",
            "ml.c5.2xlarge",
            "ml.c5.4xlarge",
            "ml.c5.9xlarge",
            "ml.c5.18xlarge",
        ]
        gpu_list = [
            "ml.p2.xlarge",
            "ml.p2.8xlarge",
            "ml.p2.16xlarge",
            "ml.p3.2xlarge",
            "ml.p3.8xlarge",
            "ml.p3.16xlarge",
        ]
        gpu_list_1 = [
            "ml.p2.xlarge",
        ]

        list_to_return = cpu_list

        if supports_gpu:
            list_to_return = gpu_list_1

        return json.dumps(list_to_return)