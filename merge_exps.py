import os
import argparse 





def main(merge_into_path: str, merge_from_path: str):
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--merge_into', type=str
                    help='A path to directory of experiments, merge into')
    parser.add_argument('--merge_from', type=str,
                        help='A path to directory of experiments, merge from')
    
    
    args = parser.parse_args()


    main(merge_into_path=args.merge_into, merge_from_path=args.merge_from)
