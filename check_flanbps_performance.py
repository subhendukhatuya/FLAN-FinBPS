import os
import sys
import time
from rouge_score import rouge_scorer
import pandas as pd
import numpy as np

from evaluate import load
bertscore = load("bertscore")


f1 = open('./lora_prediction_t5_large_ectsum_train_summary.txt', 'r')

predicted_summary_list = []
actual_summary_list = []
#for line in f1:
 #   print(line)
  #  predicted_summary = line.split('Pred:')[1]
  #  actual_summary = line.split('Pred:')[0].split('True:')[1]
  #  predicted_summary_list.append(predicted_summary)
  #  actual_summary_list.append(actual_summary)



#df = pd.read_csv('./true_vs_pred_question_based.csv')
#df = pd.read_csv('true_vs_pred_question_based_flant5.csv')
#df = pd.read_csv('corrected_true_vs_pred_question_based_flant5_0.5.csv')
df = pd.read_csv('corrected_true_vs_pred_question_based_Flant5_from_ect.csv')
for i, row in df.iterrows():
    actual_summary_list.append(row['True'].replace('True:',''))
    predicted_summary_list.append(row['Corrected_Pred'].replace('Corrected_Pred:',''))


#predictions = ["hello there", "general kenobi"]
#references = ["hello there", "general kenobi"]
results = bertscore.compute(predictions=predicted_summary_list, references=actual_summary_list, lang="en")
bert_score_precision = np.mean(results['precision'])
bert_score_recall = np.mean(results['recall'])
bert_score_f1 = np.mean(results['f1'])
print('bert score results', bert_score_precision, bert_score_recall, bert_score_f1)


## SummaC Score

from summac.model_summac import SummaCConv

model_conv = SummaCConv(models=["vitc"], bins='percentile', granularity="sentence", nli_labels="e", device="cuda", start_file="default", agg="mean")

score_conv1 = model_conv.score(predicted_summary_list, actual_summary_list)
print(" SummacConv score: %.3f" % (score_conv1["scores"][0])) # [Summary 1] SummaCZS Score: 0.582; SummacConv score: 0.536



class RougeEvaluator:
    def __init__(self):
        self.scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

    def calculate_rouge_score(self, generated_output, targets):
        assert len(generated_output) == len(targets), 'generation and targets length mismatch, force exit'

        rouge1_precision, rouge1_recall, rouge1_fmeasure = 0, 0, 0
        rouge2_precision, rouge2_recall, rouge2_fmeasure = 0, 0, 0
        rougeL_precision, rougeL_recall, rougeL_fmeasure = 0, 0, 0

        for i, row in enumerate(generated_output):
            rouge_score = self.scorer.score(generated_output[i], targets[i])
            rouge1_precision += rouge_score['rouge1'].precision
            rouge1_recall += rouge_score['rouge1'].recall
            rouge1_fmeasure += rouge_score['rouge1'].fmeasure
            rouge2_precision += rouge_score['rouge2'].precision
            rouge2_recall += rouge_score['rouge2'].recall
            rouge2_fmeasure += rouge_score['rouge2'].fmeasure
            rougeL_precision += rouge_score['rougeL'].precision
            rougeL_recall += rouge_score['rougeL'].recall
            rougeL_fmeasure += rouge_score['rougeL'].fmeasure

        print('-' * 50)
        print('ROUGE1 Precision: ', rouge1_precision / len(generated_output))
        print('ROUGE1 Recall: ', rouge1_recall / len(generated_output))
        print('ROUGE1 F1: ', rouge1_fmeasure / len(generated_output))
        print('-' * 50)
        print('ROUGE2 Precision: ', rouge2_precision / len(generated_output))
        print('ROUGE2 Recall: ', rouge2_recall / len(generated_output))
        print('ROUGE2 F1: ', rouge2_fmeasure / len(generated_output))
        print('-' * 50)
        print('ROUGEL Precision: ', rougeL_precision / len(generated_output))
        print('ROUGEL Recall: ', rougeL_recall / len(generated_output))
        print('ROUGEL F1: ', rougeL_fmeasure / len(generated_output))

        result_dict = {
            'rouge1-pre': rouge1_precision / len(generated_output),
            'rouge1-rec': rouge1_recall / len(generated_output),
            'rouge1-f1': rouge1_fmeasure / len(generated_output),
            'rouge2-pre': rouge2_precision / len(generated_output),
            'rouge2-rec': rouge2_recall / len(generated_output),
            'rouge2-f1': rouge2_fmeasure / len(generated_output),
            'rougeL-pre': rougeL_precision / len(generated_output),
            'rougeL-rec': rougeL_recall / len(generated_output),
            'rougeL-f1': rougeL_fmeasure / len(generated_output),
        }
        return result_dict


evaluator = RougeEvaluator()
results = evaluator.calculate_rouge_score(predicted_summary_list, actual_summary_list)

print(results)
