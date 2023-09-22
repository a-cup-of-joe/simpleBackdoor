import torch
import transformers
def load_model(args):
    match args.model:
        case "simple":
            from models.simple import SimpleNet

            model = SimpleNet(num_classes=args.num_classes)
        case "resnet18":
            from torchvision.models import resnet18
            model = resnet18(num_classes=args.num_classes)
        case "resnet20":
            from models.resnet import resnet20
            net = resnet20()
            model = torch.nn.DataParallel(net).cuda()
        case "bert":
            from transformers import (AutoConfig,
                                      AutoModelForSequenceClassification,
                                      AutoTokenizer)

            if args.pretrain:
                # load pretrained model
                model_config = AutoConfig.from_pretrained(args.model_path)
                model_config.num_labels = args.num_classes
                model = AutoModelForSequenceClassification.from_pretrained(
                    args.model_path, config=model_config
                )
                tokenizer = AutoTokenizer.from_pretrained(args.model_path)
                args.tokenizer = tokenizer
            else:
                # load a new model
                raise NotImplementedError("not supported.")
        # case "deepspeech":
        #     from torchaudio.models import DeepSpeech

        #     model = DeepSpeech(
        #         n_feature=args.input_size, n_class=args.num_classes
        #     )
        case "audiocnn":
            from models.simple import AudioCNN
            model = AudioCNN()
        case "r3d":
            from torchvision.models.video import R3D_18_Weights, r3d_18

            weights = R3D_18_Weights.DEFAULT
            trans = weights.transforms()
            args.transforms = trans
            model = r3d_18(weights=weights)
        case _:
            raise NotImplementedError("Model %s not supported." % args.model)
    return model
