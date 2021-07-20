# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
To add a new task to PET, both a DataProcessor and a PVP for this task must
be added. The DataProcessor is responsible for loading training and test data.
This file shows an example of a DataProcessor for a new task.
"""

import csv
import os
from typing import List

from pet.task_helpers import MultiMaskTaskHelper
from pet.tasks import DataProcessor, PROCESSORS, TASK_HELPERS
from pet.utils import InputExample


class MarketClassificationDataProcessor(DataProcessor):
    """
    Example for a data processor.
    """

    # Set this to the name of the task
    TASK_NAME = "causes"

    # Set this to the name of the file containing the train examples
    TRAIN_FILE_NAME = "train0.csv"

    # Set this to the name of the file containing the dev examples
    DEV_FILE_NAME = "test0.csv"

    # Set this to the name of the file containing the test examples
    TEST_FILE_NAME = ""

    # Set this to the name of the file containing the unlabeled examples
    UNLABELED_FILE_NAME = "unlabeled.csv"

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
        x = os.path.join(data_dir,  MarketClassificationDataProcessor.TRAIN_FILE_NAME)
        return self._create_examples(x, "train")

    def get_dev_examples(self, data_dir: str) -> List[InputExample]:
        """
        This method loads dev examples from a file with name `DEV_FILE_NAME` in the given directory.
        :param data_dir: the directory in which the dev data can be found
        :return: a list of dev examples
        """
        x  = os.path.join(data_dir, MarketClassificationDataProcessor.DEV_FILE_NAME)
        print(x)
        return self._create_examples(x, "dev")
    
    def get_test_examples(self, data_dir) -> List[InputExample]:
        """
        This method loads test examples from a file with name `TEST_FILE_NAME` in the given directory.
        :param data_dir: the directory in which the test data can be found
        :return: a list of test examples
        """
        raise NotImplementedError()

    def get_unlabeled_examples(self, data_dir) -> List[InputExample]:
        """
        This method loads unlabeled examples from a file with name `UNLABELED_FILE_NAME` in the given directory.
        :param data_dir: the directory in which the unlabeled data can be found
        :return: a list of unlabeled examples
        """
        return self._create_examples(os.path.join(data_dir, MarketClassificationDataProcessor.UNLABELED_FILE_NAME), "unlabeled")

    def get_labels(self) -> List[str]:
        """This method returns all possible labels for the task."""
        return MarketClassificationDataProcessor.LABELS

    def _create_examples(self, path: str, set_type: str) -> List[InputExample]:
        examples = []

        with open(path) as f:
            reader = csv.reader(f, delimiter=',')
            for idx, row in enumerate(reader):
                print(idx)
                label, body = row
                guid = "%s-%s" % (set_type, idx)
                text_a = body.replace('\\n', ' ').replace('\\', ' ')

                example = InputExample(guid=guid, text_a=text_a, label=label)
                examples.append(example)

        return examples


class FundingClassificationDataProcessor(DataProcessor):
    """
    Example for a data processor.
    """

    # Set this to the name of the task
    TASK_NAME = "funding"

    # Set this to the name of the file containing the train examples
    TRAIN_FILE_NAME = "train0.csv"

    # Set this to the name of the file containing the dev examples
    DEV_FILE_NAME = "test0.csv"

    # Set this to the name of the file containing the test examples
    TEST_FILE_NAME = ""

    # Set this to the name of the file containing the unlabeled examples
    UNLABELED_FILE_NAME = "unlabeled.csv"

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
        x = os.path.join(data_dir,  MarketClassificationDataProcessor.TRAIN_FILE_NAME)
        return self._create_examples(x, "train")

    def get_dev_examples(self, data_dir: str) -> List[InputExample]:
        """
        This method loads dev examples from a file with name `DEV_FILE_NAME` in the given directory.
        :param data_dir: the directory in which the dev data can be found
        :return: a list of dev examples
        """
        x  = os.path.join(data_dir, MarketClassificationDataProcessor.DEV_FILE_NAME)
        print(x)
        return self._create_examples(x, "dev")
    
    def get_test_examples(self, data_dir) -> List[InputExample]:
        """
        This method loads test examples from a file with name `TEST_FILE_NAME` in the given directory.
        :param data_dir: the directory in which the test data can be found
        :return: a list of test examples
        """
        raise NotImplementedError()

    def get_unlabeled_examples(self, data_dir) -> List[InputExample]:
        """
        This method loads unlabeled examples from a file with name `UNLABELED_FILE_NAME` in the given directory.
        :param data_dir: the directory in which the unlabeled data can be found
        :return: a list of unlabeled examples
        """
        return self._create_examples(os.path.join(data_dir, MarketClassificationDataProcessor.UNLABELED_FILE_NAME), "unlabeled")

    def get_labels(self) -> List[str]:
        """This method returns all possible labels for the task."""
        return MarketClassificationDataProcessor.LABELS

    def _create_examples(self, path: str, set_type: str) -> List[InputExample]:
        examples = []

        with open(path) as f:
            reader = csv.reader(f, delimiter=',')
            for idx, row in enumerate(reader):
                print(idx)
                label, body = row
                guid = "%s-%s" % (set_type, idx)
                text_a = body.replace('\\n', ' ').replace('\\', ' ')

                example = InputExample(guid=guid, text_a=text_a, label=label)
                examples.append(example)

        return examples


class PartnerShipClassificationDataProcessor(DataProcessor):
    """
    Example for a data processor.
    """

    # Set this to the name of the task
    TASK_NAME = "partner"

    # Set this to the name of the file containing the train examples
    TRAIN_FILE_NAME = "train0.csv"

    # Set this to the name of the file containing the dev examples
    DEV_FILE_NAME = "test0.csv"

    # Set this to the name of the file containing the test examples
    TEST_FILE_NAME = ""

    # Set this to the name of the file containing the unlabeled examples
    UNLABELED_FILE_NAME = "unlabeled.csv"

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
        x = os.path.join(data_dir,  MarketClassificationDataProcessor.TRAIN_FILE_NAME)
        return self._create_examples(x, "train")

    def get_dev_examples(self, data_dir: str) -> List[InputExample]:
        """
        This method loads dev examples from a file with name `DEV_FILE_NAME` in the given directory.
        :param data_dir: the directory in which the dev data can be found
        :return: a list of dev examples
        """
        x  = os.path.join(data_dir, MarketClassificationDataProcessor.DEV_FILE_NAME)
        print(x)
        return self._create_examples(x, "dev")
    
    def get_test_examples(self, data_dir) -> List[InputExample]:
        """
        This method loads test examples from a file with name `TEST_FILE_NAME` in the given directory.
        :param data_dir: the directory in which the test data can be found
        :return: a list of test examples
        """
        raise NotImplementedError()

    def get_unlabeled_examples(self, data_dir) -> List[InputExample]:
        """
        This method loads unlabeled examples from a file with name `UNLABELED_FILE_NAME` in the given directory.
        :param data_dir: the directory in which the unlabeled data can be found
        :return: a list of unlabeled examples
        """
        return self._create_examples(os.path.join(data_dir, MarketClassificationDataProcessor.UNLABELED_FILE_NAME), "unlabeled")

    def get_labels(self) -> List[str]:
        """This method returns all possible labels for the task."""
        return MarketClassificationDataProcessor.LABELS

    def _create_examples(self, path: str, set_type: str) -> List[InputExample]:
        examples = []

        with open(path) as f:
            reader = csv.reader(f, delimiter=',')
            for idx, row in enumerate(reader):
                print(idx)
                label, body = row
                guid = "%s-%s" % (set_type, idx)
                text_a = body.replace('\\n', ' ').replace('\\', ' ')

                example = InputExample(guid=guid, text_a=text_a, label=label)
                examples.append(example)

        return examples



class MAClassificationDataProcessor(DataProcessor):
    """
    Example for a data processor.
    """

    # Set this to the name of the task
    TASK_NAME = "ma"

    # Set this to the name of the file containing the train examples
    TRAIN_FILE_NAME = "train0.csv"

    # Set this to the name of the file containing the dev examples
    DEV_FILE_NAME = "test0.csv"

    # Set this to the name of the file containing the test examples
    TEST_FILE_NAME = ""

    # Set this to the name of the file containing the unlabeled examples
    UNLABELED_FILE_NAME = "unlabeled.csv"

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
        x = os.path.join(data_dir,  MarketClassificationDataProcessor.TRAIN_FILE_NAME)
        return self._create_examples(x, "train")

    def get_dev_examples(self, data_dir: str) -> List[InputExample]:
        """
        This method loads dev examples from a file with name `DEV_FILE_NAME` in the given directory.
        :param data_dir: the directory in which the dev data can be found
        :return: a list of dev examples
        """
        x  = os.path.join(data_dir, MarketClassificationDataProcessor.DEV_FILE_NAME)
        print(x)
        return self._create_examples(x, "dev")
    
    def get_test_examples(self, data_dir) -> List[InputExample]:
        """
        This method loads test examples from a file with name `TEST_FILE_NAME` in the given directory.
        :param data_dir: the directory in which the test data can be found
        :return: a list of test examples
        """
        raise NotImplementedError()

    def get_unlabeled_examples(self, data_dir) -> List[InputExample]:
        """
        This method loads unlabeled examples from a file with name `UNLABELED_FILE_NAME` in the given directory.
        :param data_dir: the directory in which the unlabeled data can be found
        :return: a list of unlabeled examples
        """
        return self._create_examples(os.path.join(data_dir, MarketClassificationDataProcessor.UNLABELED_FILE_NAME), "unlabeled")

    def get_labels(self) -> List[str]:
        """This method returns all possible labels for the task."""
        return MarketClassificationDataProcessor.LABELS

    def _create_examples(self, path: str, set_type: str) -> List[InputExample]:
        examples = []

        with open(path) as f:
            reader = csv.reader(f, delimiter=',')
            for idx, row in enumerate(reader):
                print(idx)
                label, body = row
                guid = "%s-%s" % (set_type, idx)
                text_a = body.replace('\\n', ' ').replace('\\', ' ')

                example = InputExample(guid=guid, text_a=text_a, label=label)
                examples.append(example)

        return examples





# register the processor for this task with its name
PROCESSORS[MarketClassificationDataProcessor.TASK_NAME] = MarketClassificationDataProcessor
PROCESSORS[FundingClassificationDataProcessor.TASK_NAME] = FundingClassificationDataProcessor
PROCESSORS[PartnerShipClassificationDataProcessor.TASK_NAME] = PartnerShipClassificationDataProcessor
PROCESSORS[MAClassificationDataProcessor.TASK_NAME] = MAClassificationDataProcessor

# optional: if you have to use verbalizers that correspond to multiple tokens, uncomment the following line
# TASK_HELPERS[MyTaskDataProcessor.TASK_NAME] = MultiMaskTaskHelper
