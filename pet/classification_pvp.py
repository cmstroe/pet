from typing import List

from pet.pvp import PVP, PVPS
from pet.utils import InputExample


class BusinessStatussPVP(PVP):
    """
    Example for a pattern-verbalizer pair (PVP).
    """

    # Set this to the name of the task
    TASK_NAME = "b-status"

    # Set this to the verbalizer for the given task: a mapping from the task's labels (which can be obtained using
    # the corresponding DataProcessor's get_labels method) to tokens from the language model's vocabulary
   
    VERBALIZER = {
        "0": ["no"],
        "1": ["yes"]
    }

    def get_parts(self, example: InputExample):
       # register the PVP for this task with its name
        text = self.shortenable(example.text_a)

        if self.pattern_id == 0:
            return [text,' talks about market status?', self.mask], []
        elif self.pattern_id == 1:
            return [self.mask, ' , ' , text,'describes the market status.'], []
        else:
            raise ValueError("No pattern implemented for id {}".format(self.pattern_id))
    #does this contain general information about the market?
    def verbalize(self, label) -> List[str]:
        return BusinessStatussPVP.VERBALIZER[label]

PVPS[BusinessStatussPVP.TASK_NAME] = BusinessStatussPVP
