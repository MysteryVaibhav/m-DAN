import argparse
from model import *
from properties import *
from data_loader import DataLoader
from evaluator import Evaluator
from trainer import Trainer


def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Argument Parser for m-DAN')
    parser.add_argument("--image_dir", dest="image_dir", type=str, default=IMAGES_DIR)
    parser.add_argument("--image_features_dir", dest="image_features_dir", type=str, default=TRAIN_IMAGES_DIR)
    parser.add_argument("--caption_file", dest="caption_file", type=str, default=CAPTION_INFO)
    parser.add_argument("--split_file", dest="split_file", type=str, default=SPLIT_INFO)

    parser.add_argument("--hidden_dimension", dest="hidden_dimension", type=int, default=HIDDEN_DIMENSION)
    parser.add_argument("--embedding_dimension", dest="embedding_dimension", type=int, default=EMBEDDING_DIMENSION)

    parser.add_argument("--mini_batch_size", dest="mini_batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--learning_rate", dest="learning_rate", type=float, default=LEARNING_RATE)
    parser.add_argument("--num_epochs", dest="num_epochs", type=int, default=EPOCHS)
    parser.add_argument("--number_of_steps", dest="number_of_steps", type=int, default=NO_OF_STEPS)
    parser.add_argument("--margin", dest="margin", type=int, default=MARGIN)
    parser.add_argument("--clip_value", dest="clip_value", type=float, default=CLIP_VALUE)
    parser.add_argument("--wdecay", dest="wdecay", type=float, default=0.0001)
    parser.add_argument("--step_size", dest="step_size", type=int, default=30)
    parser.add_argument("--gamma", dest="gamma", type=int, default=10)

    parser.add_argument("--mode", dest="mode", type=int, default=0)
    parser.add_argument("--model_dir", dest="model_dir", type=str, default=MODEL_DIR)
    parser.add_argument("--model_file_name", dest="model_file_name", type=str, default="model_weights_0_0.00.t7")

    parser.add_argument("--vocab_size", dest="vocab_size", type=int, default=VOCAB_SIZE)
    parser.add_argument("--max_caption_len", dest="max_caption_len", type=int, default=MAX_CAPTION_LEN)
    parser.add_argument("--visual_feature_dimension", dest="visual_feature_dimension", type=int,
                        default=VISUAL_FEATURE_DIMENSION)
    parser.add_argument("--regions_in_image", dest="regions_in_image", type=int, default=NO_OF_REGIONS_IN_IMAGE)
    return parser.parse_args()


def main():
    params = parse_arguments()
    print("Constructing data loaders...")
    dl = DataLoader(params)
    evaluator = Evaluator(params, dl)
    print("Constructing data loaders...[OK]")

    if params.mode == 0:
        print("Training...")
        t = Trainer(params, dl, evaluator)
        t.train()
        print("Training...[OK]")
    elif params.mode == 1:
        print("Loading model...")
        model = mDAN(params)
        model_file_path = os.path.join(params.model_dir, params.model_file_name)
        model.load_state_dict(torch.load(model_file_path))
        if torch.cuda.is_available():
            model = model.cuda()
        print("Loading model...[OK]")

        print("Evaluating model on test set...")
        r_1, r_5, r_10 = evaluator.recall(model, is_test=True)
        print("R@1 : {}".format(r_1))
        print("R@5 : {}".format(r_5))
        print("R@10 : {}".format(r_10))
        print("Evaluating model on test set...[OK]")


if __name__ == '__main__':
    main()
