import json 
import torch
import log
import argparse
from typing import List, Dict

from pet.modeling import EvalConfig
from pet.wrapper import TransformerModelWrapper, WrapperConfig, SEQUENCE_CLASSIFIER_WRAPPER
from pet.utils import InputExample, exact_match, save_logits, save_predictions, softmax, LogitsList, set_seed, eq_div
from pet.tasks import METRICS, DEFAULT_METRICS, load_examples
from pet.modeling import TrainConfig, EvalConfig, WrapperConfig, train_classifier
from pet.funding_task import PROCESSORS
from run import load_sequence_classifier_configs


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

    verb = "total"
    subj = "patterns"
    pattern_iter_output_dir = "outputs_funding_total_patterns/final/p0-i2/"
    TEST_SET = "inference_data"
    eval_data = load_examples(
        "funding", ".", TEST_SET, num_examples=-1)

    eval_config = "outputs_size500_one_patter_highlight_reason/final/p0-i2/eval_config.json"



    sc_model_cfg = WrapperConfig(model_type='roberta', model_name_or_path='outputs_funding_total_patterns/final/p0-i2/wrapper_config.json',
                              wrapper_type=SEQUENCE_CLASSIFIER_WRAPPER, task_name='funding',
                              label_list=["0", "1"], max_seq_length=64,
                              verbalizer_file=None, cache_dir= ".")

                              

    sc_train_cfg = TrainConfig(device="cuda:0", per_gpu_train_batch_size=4,
                                per_gpu_unlabeled_batch_size=4, n_gpu=1,
                                num_train_epochs=3, max_steps=-1,
                                temperature=2,
                                gradient_accumulation_steps=1,
                                weight_decay=0.01, learning_rate=1e-5,
                                adam_epsilon=1e-8, warmup_steps=0,
                                max_grad_norm=1.0, use_logits=False)
    
    sc_train_cfg.load(TrainConfig, "outputs_funding_total_patterns/final/p0-i2/wrapper_config.json")

    metrics = METRICS.get("funding", DEFAULT_METRICS)
    sc_eval_cfg = EvalConfig(device="cuda:0", n_gpu=1, metrics=metrics,
                              per_gpu_eval_batch_size=8)


    train_classifier(subj, verb, model_config = sc_model_cfg, train_config = sc_train_cfg, eval_config = sc_eval_cfg, output_dir="./predictions",
                             repetitions=1, train_data=None, unlabeled_data=None,
                             eval_data=eval_data, do_train=False, do_eval=True, seed=42)

    # wrapper = WrapperConfig(model_type= 'roberta', model_name_or_path= (pattern_iter_output_dir + "/pythorch_model.bin"),
    #                           wrapper_type=SEQUENCE_CLASSIFIER_WRAPPER, task_name= "funding",
    #                           label_list=[0,1], max_seq_length=64,
    #                           verbalizer_file=None, cache_dir="/results")

    # eval_cfg = pet.EvalConfig(
    #                           per_gpu_eval_batch_size=args.sc_per_gpu_eval_batch_size)



    # eval_result = evaluate(wrapper, eval_data, eval_config)

    # save_predictions(os.path.join(pattern_iter_output_dir, 'predictions_10k.jsonl'), wrapper, eval_result)
    # save_logits(os.path.join(pattern_iter_output_dir, 'eval_logits_10k.txt'), eval_result['logits'])

    # scores = eval_result['scores']
    # logger.info("--- RESULT (pattern_id={}, iteration={}) ---".format(pattern_id, iteration))
    # logger.info(scores)

    # results_dict['test_set_after_training'] = scores
    # with open(os.path.join(pattern_iter_output_dir, 'results_10k.json'), 'w') as fh:
    #     json.dump(results_dict, fh)

    # for metric, value in scores.items():
    #     results[metric][pattern_id].append(value)

if __name__ == "__main__":
    main()
                

    # model = torch.load('pytorch_model.bin', map_location='cpu')

