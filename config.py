import argparse


def getConfig():
    parser = argparse.ArgumentParser()

    parser.add_argument('--random-seed', default=111, type=int, help='random seed')

    parser.add_argument('--dataset', default='cub_bird', help='name of the dataset (cub_bird or stanford_dog or aircraft or vegfru)')

    parser.add_argument('--code-length', default=64, type=int, help='length of hash codes')

    parser.add_argument('--lr', default=0.0001 , type=float, help='initial learning rate')

    parser.add_argument('--batch-size', default=16, type=int, help='batch size')

    parser.add_argument('--epoch', default=90, type=int, help='total training epochs')

    parser.add_argument('--gpu-ids', default='0', help='gpu id list(eg: 0,1,2...)')

    parser.add_argument('--model-name', default='resnet50', type=str, help='model name (resnet18 or alexnet)')

    parser.add_argument('--tao', default=0.5, type=float, help='beta')

    parser.add_argument('--theta', default=0.01, type=float, help='theta')


    args = parser.parse_args()

    return args
