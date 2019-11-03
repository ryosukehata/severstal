import os


class Configs:
    fold = 0
    segmentation = True
    use_kaeru_model = True
    optim_metric = "avg_val_dice"
    test_size = 0.2
    input_path = os.environ["SEVERSTAL_INPUT_PATH"]
    output_path = os.path.join(input_path, "output")
    encoder = "resnet34"
    arch = "Unet"
    classify_model = "resnet34"
    if segmentation:
        output_path = os.path.join(
            output_path, "segmentation", arch, encoder)
    else:
        output_path = os.path.join(
            output_path, "classification", arch, encoder)
    num_epochs = 10
    batch_size = 2
    output_class = 4
    lr = 0.001
    cuda_idx = 0
    SEED = 1119
    threshold = 0.5
    debug = True
    description = "encoder: {} arch: {}, optimize_metric:{}".format(
        encoder, arch, optim_metric)
    if debug:
        fast_dev_run = True
    else:
        fast_dev_run = False
    train_percet_check = 0.1


if __name__ == '__main__':
    config = Configs()
    print(config)
