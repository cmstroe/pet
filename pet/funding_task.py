import csv
import os
from typing import List

from pet.task_helpers import MultiMaskTaskHelper
from pet.tasks import DataProcessor, PROCESSORS, TASK_HELPERS
from pet.utils import InputExample


class FundingClassificationDataProcessor(DataProcessor):
    """
    Example for a data processor.
    """

    # Set this to the name of the task
    TASK_NAME = "funding"

    # Set this to the name of the file containing the train examples
    TRAIN_FILE_NAME = "funding_train.csv"

    # Set this to the name of the file containing the dev examples
    DEV_FILE_NAME = "funding_test.csv"

    # Set this to the name of the file containing the test examples
    TEST_FILE_NAME = "funding_test.csv"

    # Set this to the name of the file containing the unlabeled examples
    UNLABELED_FILE_NAME = "funding_unlabeled.csv"

    # Set this to a list of all labels in the train + test data
    LABELS = ["0", "1"]

    # Set this to the column of the train/test csv files containing the input's text a
    TEXT_A_COLUMN = 1

    # Set this to the column of the train/test csv files containing the input's gold label
    LABEL_COLUMN = 0

    def get_train_examples(self, data_dir: str) -> List[InputExample]:
        """
        This method loads train examples from a file with name `TRAIN_FILE_NAME` in the given directory.
        :param data_dir: the directory in which the training data can be found
        :return: a list of train examples
        """
        x = os.path.join(data_dir,  FundingClassificationDataProcessor.TRAIN_FILE_NAME)
        return self._create_examples(x, "train")

    def get_dev_examples(self, data_dir: str) -> List[InputExample]:
        """
        This method loads dev examples from a file with name `DEV_FILE_NAME` in the given directory.
        :param data_dir: the directory in which the dev data can be found
        :return: a list of dev examples
        """
        x  = os.path.join(data_dir, FundingClassificationDataProcessor.DEV_FILE_NAME)
        print(x)
        return self._create_examples(x, "dev")
    
    def get_test_examples(self, data_dir) -> List[InputExample]:
        """
        This method loads test examples from a file with name `TEST_FILE_NAME` in the given directory.
        :param data_dir: the directory in which the test data can be found
        :return: a list of test examples
        """
        x  = os.path.join(data_dir, FundingClassificationDataProcessor.TEST_FILE_NAME)
        print(x)
        return self._create_examples(x, "test")

    def get_unlabeled_examples(self, data_dir) -> List[InputExample]:
        """
        This method loads unlabeled examples from a file with name `UNLABELED_FILE_NAME` in the given directory.
        :param data_dir: the directory in which the unlabeled data can be found
        :return: a list of unlabeled examples
        """
        return self._create_examples(os.path.join(data_dir, FundingClassificationDataProcessor.UNLABELED_FILE_NAME), "unlabeled")

    def get_labels(self) -> List[str]:
        """This method returns all possible labels for the task."""
        return FundingClassificationDataProcessor.LABELS

    def _create_examples(self, path: str, set_type: str) -> List[InputExample]:
        examples = []

        with open(path,  encoding= 'unicode_escape') as f:
            reader = csv.reader(f, delimiter=',')
            for idx, row in enumerate(reader):
                print(idx)
                body = row[0]
                label = 0 
                guid = "%s-%s" % (set_type, idx)
                text_a = body.replace('\\n', ' ').replace('\\', ' ')

                example = InputExample(guid=guid, text_a=text_a, label=label)
                examples.append(example)

        return examples

# register the processor for this task with its name
PROCESSORS[FundingClassificationDataProcessor.TASK_NAME] = FundingClassificationDataProcessor
