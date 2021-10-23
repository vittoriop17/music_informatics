from utils.utils import upload_args
import train


def main():
    args = upload_args()
    train.train(args)

if __name__=='__main__':
    main()