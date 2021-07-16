import json 
import torch
import log
from typing import List, Dict
from pet.modeling import EvalConfig
from pet.wrapper import TransformerModelWrapper
from pet.utils import InputExample, exact_match, save_logits, save_predictions, softmax, LogitsList, set_seed, eq_div


# with open("preprocessing_new_tags/company_sentences_euro.json") as jsonFile:
#     json_obj = json.load(jsonFile)

# sentences = [], j = 0

# for i in range (100, 300):
#     obj = json_obj[i]
#     sentences[j] = obj['sentences']
#     j+=1

# wrapper = TransformerModelWrapper.__new__(TransformerModelWrapper)
# wrapper.config = wrapper._load_config(path)
# tokenizer_class = MODEL_CLASSES[wrapper.config.model_type]['tokenizer']
# model_class = MODEL_CLASSES[wrapper.config.model_type][wrapper.config.wrapper_type]
# wrapper.model = model_class.from_pretrained(path)
# wrapper.tokenizer = tokenizer_class.from_pretrained(path)
# wrapper.preprocessor = PREPROCESSORS[wrapper.config.wrapper_type](subj, verb, 
#             wrapper, wrapper.config.task_name, wrapper.config.pattern_id, wrapper.config.verbalizer_file)
# wrapper.task_helper = TASK_HELPERS[wrapper.config.task_name](wrapper) \
#             if wrapper.config.task_name in TASK_HELPERS else None

def evaluate(model: TransformerModelWrapper, eval_data: List[InputExample], config: EvalConfig,
             priming_data: List[InputExample] = None) -> Dict:
    """
    Evaluate a model.

    :param model: the model to evaluate
    :param eval_data: the examples for evaluation
    :param config: the evaluation config
    :param priming_data: an optional list of priming data to use
    :return: a dictionary containing the model's logits, predictions and (if any metrics are given) scores
    """

    if config.priming:
        for example in eval_data:
            example.meta['priming_data'] = priming_data

    metrics = config.metrics if config.metrics else ['acc']
    # device = torch.device(config.device if config.device else "cuda:0") #if torch.cuda.is_available() else "cpu")
    device = torch.device("cuda:0")
    model.model.to(device)
    results = model.eval(eval_data, device, per_gpu_eval_batch_size=config.per_gpu_eval_batch_size,
                         n_gpu=config.n_gpu, decoding_strategy=config.decoding_strategy, priming=config.priming)

    predictions = np.argmax(results['logits'], axis=1)
    scores = {}

    for metric in metrics:
        if metric == 'acc':
            scores[metric] = simple_accuracy(predictions, results['labels'])
        elif metric == 'f1':
            scores[metric] = f1_score(results['labels'], predictions)
        elif metric == 'f1-macro':
            scores[metric] = f1_score(results['labels'], predictions, average='macro')
        elif metric == 'em':
            scores[metric] = exact_match(predictions, results['labels'], results['question_ids'])
        else:
            raise ValueError(f"Metric '{metric}' not implemented")

    results['scores'] = scores
    results['predictions'] = predictions
    return results

def main():

    logger = log.get_logger('root')

    subj = 'a'
    verb = 'b'
    pattern_iter_output_dir = "outputs_size500_one_patter_highlight_reason/final/p0-i2/wrapper_config.json"
    eval_data = "inference_data.csv"
    eval_config = "outputs_size500_one_patter_highlight_reason/final/p0-i2/eval_config.json"

    wrapper = TransformerModelWrapper.from_pretrained(subj, verb, pattern_iter_output_dir)

    eval_result = evaluate(wrapper, eval_data, eval_config, priming_data=None)

    save_predictions(os.path.join(pattern_iter_output_dir, 'predictions_10k.jsonl'), wrapper, eval_result)
    save_logits(os.path.join(pattern_iter_output_dir, 'eval_logits_10k.txt'), eval_result['logits'])

    scores = eval_result['scores']
    logger.info("--- RESULT (pattern_id={}, iteration={}) ---".format(pattern_id, iteration))
    logger.info(scores)

    results_dict['test_set_after_training'] = scores
    with open(os.path.join(pattern_iter_output_dir, 'results_10k.json'), 'w') as fh:
        json.dump(results_dict, fh)

    for metric, value in scores.items():
        results[metric][pattern_id].append(value)

if __name__ == "__main__":
    main()
                

    # model = torch.load('pytorch_model.bin', map_location='cpu')

