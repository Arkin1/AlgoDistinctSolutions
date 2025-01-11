from sklearn.model_selection import train_test_split
import os
import json
import logging
from typing import Any
from model_facades.Word2VecFacade import Word2VecFacade

logger = logging.getLogger()

def split_dataset(dataset_path: str,
                  val_ratio: float = 0.1,
                  test_ratio: float = 0.1,
                  random_state = 42) -> None:
    
    def group_submissions_by_dict(dataset:list[dict[str, Any]]) -> dict[str, list[dict[str,Any]]]:
        submissions_per_problem = {}

        for submission in dataset:
            problem = submission['problem']
            if problem not in submissions_per_problem:
                submissions_per_problem[problem] = []

            submissions_per_problem[problem].append(submission)
        return submissions_per_problem
    
    def print_small_statistics(dataset:list[dict[str, Any]]):
        logger.debug("Small statistics about the dataset")
        submissions_per_problem = group_submissions_by_dict(dataset)
        logger.debug(f"Number of problems: {len(submissions_per_problem)}")
        
        for problem, submissions in submissions_per_problem.items():
            logger.debug(f"\t{problem}:")
            logger.debug(f"\t\tnumber submissions: {len(submissions)}")
            logger.debug(f"\t\talgorithmic solutions: {list(set([s['algorithmic_solution'] for s in submissions]))}")

    logger.info(f"Splitting dataset with val_ratio={val_ratio} and test_ratio={test_ratio}.")

    if val_ratio is None or test_ratio is None or val_ratio == 0 or test_ratio == 0:
        return Exception("val_ratio or test_ratio must be greater than 0.")

    with open(os.path.join(dataset_path, 'dataset.json'), 'r', encoding='utf-8') as fp:
        dataset_info = json.load(fp)

    print_small_statistics(dataset_info)

    submissions_per_problem  = group_submissions_by_dict(dataset_info)

    train_dataset_info = []
    val_dataset_info = []
    test_dataset_info = []

    for problem, submissions in submissions_per_problem.items(): 
        logger.info(f"Splitting problem {problem}...")

        remaining_submissions, test_submissions = train_test_split(submissions,
                                                                   stratify = [s['algorithmic_solution'] for s in submissions],
                                                                   test_size = test_ratio,
                                                                   random_state = random_state)
        
        train_submissions, val_submissions = train_test_split(remaining_submissions,
                                                              stratify = [s['algorithmic_solution'] for s in remaining_submissions],
                                                              test_size = val_ratio / (1 - test_ratio),
                                                              random_state = random_state)
        
        train_dataset_info.extend(train_submissions)
        val_dataset_info.extend(val_submissions)
        test_dataset_info.extend(test_submissions)

    with open(os.path.join(dataset_path, 'train.json'), 'w', encoding='utf-8') as fp:
        json.dump(train_dataset_info, fp)

    logger.info("Created train split!")
    print_small_statistics(train_dataset_info)

    with open(os.path.join(dataset_path, 'val.json'), 'w', encoding='utf-8') as fp:
        json.dump(val_dataset_info, fp)

    logger.info("Created val split!")
    print_small_statistics(val_dataset_info)

    with open(os.path.join(dataset_path, 'test.json'), 'w', encoding='utf-8') as fp:
        json.dump(test_dataset_info, fp)

    logger.info("Created test split!")  
    print_small_statistics(test_dataset_info)

def pretrain_embeddings(args: dict[str, Any]):
    dataset_info_path = args['dataset_info_path']
    destination_path = args['destination_path']

    destination_folder = os.path.dirname(destination_path)
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    if 'w2v' in args:
        w2v_facade = Word2VecFacade()

        w2v_args = args['w2v']
        
        w2v_facade.pretrain(dataset_info_path, destination_path, **w2v_args)




    



