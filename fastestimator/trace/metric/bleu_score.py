# Copyright 2022 The FastEstimator Authors. All Rights Reserved.
#
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
# ==============================================================================
import math
from collections import Counter
from fractions import Fraction
from typing import Iterable, List, Tuple, Union

import numpy as np
from nltk.translate.bleu_score import SmoothingFunction, brevity_penalty, modified_precision, sentence_bleu

from fastestimator.trace.meta._per_ds import per_ds
from fastestimator.trace.trace import Trace
from fastestimator.util.data import Data
from fastestimator.util.traceability_util import traceable
from fastestimator.util.util import to_number


def get_formated_list(input_data: np.ndarray) -> List[str]:
    """
        Filter the padding(elements with 0 value) and typecast the elements of list to str.

        Returns:
            Formated list.
    """
    return [str(i) for i in input_data if i != 0]


def get_formated_reference(input_data: np.ndarray) -> List[List[str]]:
    """
        Encapsulate formated list in another list.

        Returns:
            List encapsulated formated list.
    """
    return [get_formated_list(input_data)]


@per_ds
@traceable()
class BleuScore(Trace):
    """Calculate the Bleu score for a nlp task and report it back to the logger.

       Calculate BLEU score (Bilingual Evaluation Understudy) from Papineni, Kishore, Salim Roukos, Todd Ward, and
       Wei-Jing Zhu. 2002."BLEU: a method for automatic evaluation of machine translation."In Proceedings of ACL.
       https://www.aclweb.org/anthology/P02-1040.pdf

       The BLEU metric scores a translation on a scale of 0 to 1, in an attempt to measure the adequacy and fluency of
       the Machine Translation output. The closer to 1 the test sentences score, the more overlap there is with their
       human reference translations and thus, the better the system is deemed to be. The MT output would score 1 only
       if it is identical to the reference human translation. But even two competent human translations of the exact
       same material may only score in the 0.6 or 0.7 range as they are likely to use different vocabulary and phrasing.
       We should be wary of very high BLEU scores (in excess of 0.7) as it is probably measuring improperly or overfitting.

       The default BLEU calculates a score for up to 4-grams using uniform weights (this is called BLEU-4). To evaluate
       your translations with lower order ngrams, use customized "n_gram". E.g. when accounting for up to 2-grams
       with uniform weights (this is called BLEU-2) use n_gram=2.

       If there is no ngrams overlap for any order of n-grams, BLEU returns the value 0. This is because the precision
       for the order of n-grams withoutoverlap is 0, and the geometric mean in the final BLEU score computation multiplies
       the 0 with the precision of other n-grams. This results in 0. Shorter translations may have inflated precision values due to having
       smaller denominators; therefore, we give them proportionally smaller smoothed counts. Instead of scaling to 1/(2^k),
       Chen and Cherry suggests dividing by 1/ln(len(T)), where T is the length of the translation.


    Args:
        true_key: Name of the key that corresponds to ground truth in the batch dictionary.
        pred_key: Name of the key that corresponds to predicted score in the batch dictionary.
        mode: What mode(s) to execute this Trace in. For example, "train", "eval", "test", or "infer". To execute
            regardless of mode, pass None. To execute in all modes except for a particular one, you can pass an argument
            like "!infer" or "!train".
        ds_id: What dataset id(s) to execute this Trace in. To execute regardless of ds_id, pass None. To execute in all
            ds_ids except for a particular one, you can pass an argument like "!ds1".
        output_name: Name of the key to store back to the state.
        n_gram: Number of grams used to calculate bleu score.
        per_ds: Whether to automatically compute this metric individually for every ds_id it runs on, in addition to
            computing an aggregate across all ds_ids on which it runs. This is automatically False if `output_name`
            contains a "|" character.

    """
    def __init__(self,
                 true_key: str,
                 pred_key: str,
                 mode: Union[None, str, Iterable[str]] = ("eval", "test"),
                 ds_id: Union[None, str, Iterable[str]] = None,
                 output_name: str = 'bleu_score',
                 n_gram: int = 4,
                 per_ds: bool = True) -> None:
        super().__init__(inputs=(true_key, pred_key), outputs=output_name, mode=mode, ds_id=ds_id)
        self.n_gram = n_gram
        self.weights = self.get_output_weights()
        self.per_ds = per_ds
        self.smoothing_function = SmoothingFunction().method4
        self.no_of_correct_predicted = Counter()
        self.no_of_total_predicted = Counter()
        self.total_hypotheses_length = 0
        self.total_references_length = 0

    @property
    def true_key(self) -> str:
        return self.inputs[0]

    @property
    def pred_key(self) -> str:
        return self.inputs[1]

    def get_output_weights(self) -> Tuple[float, ...]:
        """
            Generate weights tuple based on n_gram.

            Returns:
                Tuple of n_gram weights

            Raises:
                ValueError: When n_gram provided is less than or equal to 0..
        """
        if self.n_gram > 0:
            return (1 / self.n_gram, ) * self.n_gram
        else:
            raise ValueError("N Gram should be a positive integer.")

    def on_epoch_begin(self, data: Data) -> None:
        self.no_of_correct_predicted = Counter()
        self.no_of_total_predicted = Counter()
        self.total_hypotheses_length = 0
        self.total_references_length = 0

    def get_brevity_penalty(self) -> float:
        """
            Calculate the brevity penalty of the corpus.

            Returns:
                Brevity penalty for corpus.
        """
        return brevity_penalty(self.total_references_length, self.total_hypotheses_length)

    def get_smoothened_modified_precision(self) -> List[float]:
        """
            Calculate the smoothened modified precision.

            Returns:
                List of smoothened modified precision of n_grams.
        """
        # Collects the various precision values for the different ngram orders.
        p_n = [
            Fraction(self.no_of_correct_predicted[i], self.no_of_total_predicted[i], _normalize=False)
            for i in range(1, self.n_gram + 1)
        ]
        # Smoothen the modified precision.
        return self.smoothing_function(p_n, [], [], hyp_len=self.total_hypotheses_length)

    def get_corpus_bleu_score(self) -> float:
        """
            Calculate the bleu score using corpus level brevity penalty and geometric average precision.

            Returns:
                Corpus level bleu score.
        """
        # Calculate corpus-level brevity penalty.
        bp = self.get_brevity_penalty()

        # Returns 0 if there's no matching 1-gram
        if self.no_of_correct_predicted[1] == 0:
            return 0

        n_gram_precision = self.get_smoothened_modified_precision()

        geometric_average_precision = math.exp(
            math.fsum((w_i * math.log(p_i) for w_i, p_i in zip(self.weights, n_gram_precision) if p_i > 0)))
        bleu_score = bp * geometric_average_precision

        return bleu_score

    def batch_precision_parameters(self, references: List[np.ndarray], hypotheses: List[np.ndarray]) -> List[float]:
        """
            Calculate modified precision per n_gram for input references and hypotheses combinations.

            Args:
                references: Ground truth sentences.
                hypotheses: Predicted sentences.

            Returns:
                List of sentence level bleu scores
        """

        assert len(references) == len(hypotheses), (
            "The number of hypotheses and their reference(s) should be the same ")

        sentence_level_scores = []
        # Iterate through each hypothesis and their corresponding references.
        for reference, hypothesis in zip(references, hypotheses):

            # For each order of ngram, calculate the correct predicted words and
            # total predicted words for the corpus-level modified precision.
            reference = get_formated_reference(reference)
            hypothesis = get_formated_list(hypothesis)
            for i in range(1, self.n_gram + 1):
                p_i = modified_precision(reference, hypothesis, i)
                self.no_of_correct_predicted[i] += p_i.numerator
                self.no_of_total_predicted[i] += p_i.denominator

            sentence_level_scores.append(sentence_bleu(reference, hypothesis, self.weights, self.smoothing_function))

            # Calculate the hypothesis length and the closest reference length.
            # Adds them to the corpus-level hypothesis and reference counts.
            hyp_len = len(hypothesis)
            self.total_hypotheses_length += hyp_len
            ref_lens = (len(ref) for ref in reference)
            self.total_references_length += min(ref_lens, key=lambda ref_len: (abs(ref_len - hyp_len), ref_len))

        return sentence_level_scores

    def on_batch_end(self, data: Data) -> None:
        y_pred, y_true = to_number(data['pred']), to_number(data['target_real'])
        if y_true.shape[-1] > 1 and y_true.ndim > 2:
            y_true = np.argmax(y_true, axis=-1)
        if y_pred.shape[-1] > 1 and y_pred.ndim > 2:
            y_pred = np.argmax(y_pred, axis=-1)
        sentence_level_scores = self.batch_precision_parameters(y_true, y_pred)
        data.write_per_instance_log(self.outputs[0], sentence_level_scores)

    def on_epoch_end(self, data: Data) -> None:
        data.write_with_log(self.outputs[0], round(self.get_corpus_bleu_score(), 5))
