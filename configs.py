import os


class Configs:
    fold = 0
    input_path = os.environ["SEVERSTAL_INPUT_PATH"]
    encoder = "resnet34"
    arch = "Unet"
    classify_model = "resnet34"
    output_path = os.path.join(os.chdir(), "./output/segmentation/", arch, encoder_name)
    num_epochs = 10
    batch_size = 2
    output_class = 4
    lr = 0.001
    cuda_idx = 0
    SEED = 1119
    threshold = 0.5
    use_kaeru_model = True
    description = "encoder: {} arch: {}".format(encoder, arch)
