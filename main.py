from utils.utils import upload_args
import train
import warnings

def main():
    warnings.filterwarnings("ignore")
    args = upload_args()
    train.train(args)

if __name__=='__main__':
    main()