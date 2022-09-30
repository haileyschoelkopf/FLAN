# Copyright 2021 The FLAN Authors.
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

"""Define light-weight seqio tasks for FLAN."""
import collections
import dataclasses
import functools
from typing import Any, Callable, List, Optional, Tuple

import seqio
from t5.data import glue_utils
from t5.data import postprocessors as t5_post
from t5.evaluation import metrics as t5_metrics
import tensorflow.compat.v1 as tf

from flan import baseline_templates
from flan import few_shot
from flan import metrics as gm_metrics
from flan import postprocessors
from flan import preprocessors
from flan import templates
from flan import utils

ShotConfig = few_shot.ShotConfig

# This is a placeholder, for the paper we used an internal vocabulary and model.
VOCAB_FILE = '/fsx/hailey/jaxtest/t5-tokenizer/spiece.model'
FLAN_VOCABULARY = seqio.SentencePieceVocabulary(VOCAB_FILE)

FLAN_OUTPUT_FEATURES = {
    'inputs':
        seqio.Feature(
            vocabulary=FLAN_VOCABULARY, add_eos=True, required=False),
    'targets':
        seqio.Feature(vocabulary=FLAN_VOCABULARY, add_eos=True)
}
FLAN_OUTPUT_FEATURES_LM = {
    'targets': seqio.Feature(vocabulary=FLAN_VOCABULARY, add_eos=True)
}


TASK_CONFIGS = {}


@dataclasses.dataclass
class _TaskConfig:
  source: seqio.DataSource
  preprocessors: List[Callable[..., tf.data.Dataset]]
  postprocess_fn: Optional[Callable[..., Any]]
  metric_fns: List[seqio.MetricFnCallable]
  num_multi_shots: int = 1
  # TODO(kguu): we should manually decide `num_multi_shots` for every task.


NUM_TRAIN_EXAMPLES = 30000
NUM_VAL_EXAMPLES = 200
SPLITS_DICT = {
    'train': f'train[:{NUM_TRAIN_EXAMPLES}]',
    'validation': f'validation[:{NUM_VAL_EXAMPLES}]',
    'test': 'test',
}

PARACRAWL_SPLITS_DICT = {
    'train': f'train[:{NUM_TRAIN_EXAMPLES}]',
    'validation': f'train[-{1000+NUM_VAL_EXAMPLES}:-1000]',
    'test': 'train[-1000:]',
}

WMT16_SPLITS_DICT = {
    'train': f'train[:{NUM_TRAIN_EXAMPLES}]',
    'validation': f'validation[:{NUM_VAL_EXAMPLES}]',
    'test': 'test',
}

NUM_VAL_EXAMPLES_WSC = 50
WSC_SPLITS_DICT = {
    'train': f'train[:-{NUM_VAL_EXAMPLES_WSC}]',
    'validation': f'train[-{NUM_VAL_EXAMPLES_WSC}:]',
    'test': 'validation',
}

# Number of templates per task for ablation study.
NUM_TEMPLATES_LIST = [1, 2, 4, 7, 10]


def enumerate_items(items_list):
  num_items = tf.shape(items_list)[0]
  number_list = tf.strings.as_string(tf.range(1, 1 + num_items, 1))
  numbered_items = tf.strings.join([number_list, items_list], separator='. ')
  numbered_items_str = tf.strings.reduce_join(numbered_items, separator='\n')
  return numbered_items_str


# =============================== BoolQ ========================================
@seqio.map_over_dataset
def _process_boolq(example):
  one_hot = tf.one_hot(tf.cast(example['answer'], tf.int32), 2)
  options = tf.constant(['no', 'yes'])
  return {
      'title': example['title'],
      'text': example['passage'],
      'question': example['question'],
      'options': options,
      'answer': tf.boolean_mask(options, one_hot)[0],
  }


TASK_CONFIGS['bool_q'] = _TaskConfig(
    source=seqio.TfdsDataSource(
        tfds_name='bool_q:1.0.0',
        splits={
            'train': f'train[:-{NUM_VAL_EXAMPLES}]',
            'validation': f'train[-{NUM_VAL_EXAMPLES}:]',
            'test': 'validation',
        }),
    preprocessors=[
        _process_boolq,
        preprocessors.format_options,
    ],
    postprocess_fn=None,
    metric_fns=glue_utils.get_super_glue_metric('boolq'),
)


# =============================== RTE ========================================
@seqio.map_over_dataset
def _process_rte(example):
  one_hot = tf.one_hot(tf.cast(example['label'], tf.int32), 2)
  options = tf.constant(['yes', 'no'])
  glm_options = tf.constant(['true', 'false'])
  return {
      'premise': example['premise'],
      'hypothesis': example['hypothesis'],
      'options': options,
      'glm_options': glm_options,
      'answer': tf.boolean_mask(options, one_hot)[0],
      'glm_answer': tf.boolean_mask(glm_options, one_hot)[0],
  }


TASK_CONFIGS['rte'] = _TaskConfig(
    source=seqio.TfdsDataSource(
        tfds_name='super_glue/rte:1.0.2',
        splits={
            'train': f'train[:-{NUM_VAL_EXAMPLES}]',
            'validation': f'train[-{NUM_VAL_EXAMPLES}:]',
            'test': 'validation',
        }),
    preprocessors=[
        _process_rte,
        preprocessors.format_options,
    ],
    postprocess_fn=None,
    metric_fns=glue_utils.get_super_glue_metric('rte'),
)


# =============================== Wsc ========================================
@seqio.map_over_dataset
def _process_wsc(example):
  one_hot = tf.one_hot(tf.cast(example['label'], tf.int32), 2)
  options = tf.constant(['no', 'yes'])
  return {
      'context': example['text'],
      'text1': example['span1_text'],
      'text2': example['span2_text'],
      'options': options,
      'answer': tf.boolean_mask(options, one_hot)[0],
  }


TASK_CONFIGS['wsc'] = _TaskConfig(
    source=seqio.TfdsDataSource(
        tfds_name='super_glue/wsc:1.0.2',
        splits=WSC_SPLITS_DICT,
    ),
    preprocessors=[
        _process_wsc,
        preprocessors.format_options,
    ],
    postprocess_fn=None,
    # Metric function same as in t5/data/tasks.py
    metric_fns=[t5_metrics.accuracy],
)


# =============================== WSC273 =======================================
@seqio.map_over_dataset
def _process_wsc273(example):
  """WSC 273 dataset."""
  prefix = tf.strings.strip(
      tf.strings.substr(example['text'], 0, example['pronoun_start']))
  suffix = tf.strings.strip(
      tf.strings.substr(example['text'], example['pronoun_end'], -1))
  entities = tf.stack(
      [example['option1_normalized'], example['option2_normalized']])
  options_list_endings = tf.fill(tf.shape(entities), value=suffix)
  options = tf.strings.join([entities, options_list_endings], separator=' ')
  one_hot = tf.one_hot(example['label'], 2)
  return {
      'context': prefix,
      'options': options,
      'answer': tf.boolean_mask(options, one_hot)[0],
  }


TASK_CONFIGS['wsc273'] = _TaskConfig(
    source=seqio.TfdsDataSource(
        tfds_name='wsc273:1.0.0',  # Only the test split is available.
        splits={
            'test': 'test',
        }
    ),
    preprocessors=[
        _process_wsc273,
        preprocessors.format_options,
    ],
    postprocess_fn=None,
    metric_fns=[t5_metrics.accuracy],
)


# =============================== Wic ========================================
@seqio.map_over_dataset
def _process_wic(example):
  one_hot = tf.one_hot(tf.cast(example['label'], tf.int32), 2)
  options = tf.constant(['different meanings', 'the same meaning'])
  return {
      'sentence1': example['sentence1'],
      'sentence2': example['sentence2'],
      'word': example['word'],
      'options': options,
      'answer': tf.boolean_mask(options, one_hot)[0],
  }


TASK_CONFIGS['wic'] = _TaskConfig(
    source=seqio.TfdsDataSource(
        tfds_name='super_glue/wic:1.0.2',
        splits={
            'train': f'train[:-{NUM_VAL_EXAMPLES}]',
            'validation': f'train[-{NUM_VAL_EXAMPLES}:]',
            'test': 'validation',
        }),
    preprocessors=[
        _process_wic,
        preprocessors.format_options,
    ],
    postprocess_fn=None,
    metric_fns=glue_utils.get_super_glue_metric('wic'),
)



# =============================== ReCoRD ============================
@seqio.map_over_dataset
def _process_record(example):
  """Processing function for ReCoRD dataset."""

  query_left = tf.strings.strip(
      tf.strings.split(
          example['query'], '@placeholder', result_type='RaggedTensor')[0][0])

  # query_right needs to be appended to all options and answers.
  query_right = tf.strings.split(
      example['query'], '@placeholder', result_type='RaggedTensor')[0][1]

  # Append query_right to options.
  entities = example['entities']
  options_list_endings = tf.fill(tf.shape(entities), value=query_right)
  options = tf.strings.join([entities, options_list_endings], separator='')

  # Append query_right to answers.
  answers = example['answers']
  answers_list_endings = tf.fill(tf.shape(answers), value=query_right)
  answers = tf.strings.join([answers, answers_list_endings])

  # Because options is variable length, make it into a string.
  options_str = tf.strings.reduce_join(
      ['OPTIONS:\n- ',
       tf.strings.reduce_join(options, separator='\n- ')])

  # Remove the "@highlights".
  passage = tf.strings.split(
      example['passage'], '\n@highlight', result_type='RaggedTensor')[0][0]

  return {
      'answer': answers[0],
      'passage': passage,
      'query': query_left,
      'answers': answers,
      'options_str': options_str,
      'options': options,
  }


TASK_CONFIGS['record'] = _TaskConfig(
    source=seqio.TfdsDataSource(
        tfds_name='super_glue/record:1.0.2',
        splits={
            'train': f'train[:{NUM_TRAIN_EXAMPLES}]',
            'validation': f'train[-{NUM_VAL_EXAMPLES}:]',
            'test': 'validation',
        }),
    preprocessors=[
        _process_record,
    ],
    postprocess_fn=t5_post.qa,
    metric_fns=glue_utils.get_super_glue_metric('record'),
)





# =============================== Arc ==========================================
@seqio.map_over_dataset
def _process_arc(example):
  return {
      'question': example['question'],
      'options': example['choices']['text'],
      'answer': example['choices']['text'][example['answerKey']],
      'label': int(example['answerKey']),
  }


def _filter_arc(dataset):

  def my_fn(example):
    return tf.equal(tf.shape(example['options'])[0], 4)

  return dataset.filter(my_fn)


for config_name in ['Challenge', 'Easy']:
  TASK_CONFIGS[f'arc_{config_name.lower()}'] = _TaskConfig(
      source=seqio.TfdsDataSource(
          tfds_name=f'ai2_arc/ARC-{config_name}:1.0.0',
          splits={
              'train': f'train[:-{NUM_VAL_EXAMPLES}]',
              'validation': f'train[-{NUM_VAL_EXAMPLES}:]',
              'test': 'test',
          }),
      preprocessors=[
          _process_arc,
          _filter_arc,
          preprocessors.format_options,
      ],
      postprocess_fn=None,
      metric_fns=[t5_metrics.accuracy],
  )





# ========================== ADD TASKS TO REGISTRY. ============================
def register_few_shot_versions_of_task(zero_shot_name: str,
                                       prune_exemplars: bool = False,
                                       multishot_max_num_shots: int = 16,
                                       max_input_length: Optional[int] = None):
  """Registers one-shot and multi-shot versions of a zero-shot Task.

  Args:
    zero_shot_name: the base zero-shot task name.
    prune_exemplars: If True, prune excessive exemplars by input length.
    multishot_max_num_shots: Maximum number of exemplars for multi-shot tasks.
    max_input_length: Maximum length of all exemplars.
  """
  for shot_config in ShotConfig:
    if shot_config in [ShotConfig.ZERO, ShotConfig.MULTI]:
      continue
    # X-shot version of the task.
    few_shot.register_few_shot_version_of_task(
        base_task_name=zero_shot_name,
        new_task_name=zero_shot_name + shot_config.name_suffix,
        num_shots=shot_config.value,
        prune_exemplars=prune_exemplars,
        max_input_length=max_input_length)

  # Multi-shot version of the task.
  few_shot.register_few_shot_version_of_task(
      base_task_name=zero_shot_name,
      new_task_name=zero_shot_name + ShotConfig.MULTI.name_suffix,
      num_shots=multishot_max_num_shots,
      prune_exemplars=prune_exemplars,
      max_input_length=max_input_length)


def register_few_shot_versions_of_continuations_task(
    zero_shot_name: str) -> None:
  """Registers one, two, three, and five-shot versions of a zero-shot Task."""

  x_y_delimiter = ' '
  inputs_prefix = ''
  targets_prefix = ''
  example_separator = '\n\n'
  final_suffix = ''

  for shot_config in ShotConfig:
    if shot_config in [ShotConfig.ZERO, ShotConfig.MULTI]:
      continue
    few_shot.register_few_shot_version_of_task(
        base_task_name=zero_shot_name,
        new_task_name=zero_shot_name + shot_config.name_suffix,
        num_shots=shot_config.value,
        x_y_delimiter=x_y_delimiter,
        inputs_prefix=inputs_prefix,
        targets_prefix=targets_prefix,
        example_separator=example_separator,
        final_suffix=final_suffix,
    )


for t_name, config in TASK_CONFIGS.items():
  flan_pattern_name = utils.t_name_to_flan_pattern_name(t_name)
  for idx, patterns in enumerate(templates.PATTERNS[flan_pattern_name]):
    # Zero-shot task instantiated from template {idx}.
    # Note: Used primarily for zeroshot eval.
    inputs_pattern, targets_pattern = patterns

    # Task names:
    # Zero-shot version: f'{t_name}_type_{idx}'
    # One-shot version: f'{t_name}_type_{idx}_one_shot'
    # Multi-shot version: f'{t_name}_type_{idx}_multi_shot'

    # Zero-shot version of the task.
    zero_shot_task_name = utils.ZeroshotEvalTaskName.get(t_name, idx)
    seqio.TaskRegistry.add(
        zero_shot_task_name,
        source=config.source,
        preprocessors=config.preprocessors +
        # Format inputs and outputs according to the patterns. This should be
        # the same for all tasks.
        preprocessors.get_flan_formatter(inputs_pattern, targets_pattern) +
        # Tokenization for the prefix-LM. This should be the same for all tasks.
        preprocessors.FLAN_TOKENIZE,
        postprocess_fn=config.postprocess_fn,
        output_features=FLAN_OUTPUT_FEATURES,
        metric_fns=config.metric_fns)

    # Few-shot versions of the task.
    register_few_shot_versions_of_task(
        zero_shot_task_name,
        prune_exemplars=True,
        # The current input length is 1024. Save 64 for seperators.
        max_input_length=960,
        multishot_max_num_shots=16)

    if utils.is_classification(flan_pattern_name):
      zero_shot_task_name = utils.ZeroshotScoreEvalTaskName.get(
          t_name, idx)
      # Zeroshot rank-classifcation tasks from template {idx}.
      # Note: These are only used for scoring/rank-classification eval.
      # Task names:
      # Zero-shot version: f'{t_name}_type_{idx}_scoring_eval'
      # One-shot version: f'{t_name}_type_{idx}_scoring_eval_one_shot'
      # Multi-shot version: f'{t_name}_type_{idx}_scoring_eval_multi_shot'

      seqio.TaskRegistry.add(
          # Task name: f'{t_name}_type_{idx}_scoring_eval'
          zero_shot_task_name,
          source=config.source,
          preprocessors=config.preprocessors +
          # Format inputs and outputs according to the patterns. This should be
          # the same for all tasks.
          preprocessors.get_flan_formatter(inputs_pattern, targets_pattern) +
          [preprocessors.rank_classification_from_options] +
          # Tokenization for the prefix-LM. This should be the same for all
          # tasks.
          preprocessors.FLAN_TOKENIZE,
          postprocess_fn=t5_post.rank_classification,
          output_features=FLAN_OUTPUT_FEATURES,
          metric_fns=[t5_metrics.rank_classification])

      register_few_shot_versions_of_task(
          zero_shot_task_name,
          prune_exemplars=True,
          max_input_length=960,
          multishot_max_num_shots=16)

    # Zeroshot rank-classifcation tasks from template {idx}
    # WITHOUT OPTIONS OR FLAN.
    # These are only used for scoring eval for the baseline.
    if utils.is_classification(flan_pattern_name):
      inputs_pattern_no_options = utils.remove_input_patterns_options(
          inputs_pattern)
      seqio.TaskRegistry.add(
          # Task name: f'{t_name}_type_{idx}_score_eval_no_options'
          utils.ZeroshotScoreEvalNoOptionTaskName.get(t_name, idx),
          source=config.source,
          preprocessors=config.preprocessors +
          # Format inputs and outputs according to the patterns. This should be
          # the same for all tasks.
          preprocessors.get_flan_formatter(inputs_pattern_no_options,
                                            targets_pattern) +
          [preprocessors.rank_classification_from_options] +
          # Tokenization for the prefix-LM. This should be the same for all
          # tasks.
          preprocessors.FLAN_TOKENIZE,
          postprocess_fn=t5_post.rank_classification,
          output_features=FLAN_OUTPUT_FEATURES,
          metric_fns=[t5_metrics.rank_classification])

    # Zeroshot rank-classifcation tasks from template {idx}
    # Without options, but keep flan clean.
    # These are only used for scoring eval for flan.
    if utils.is_classification(flan_pattern_name):
      inputs_pattern_no_options = utils.remove_input_patterns_options(
          inputs_pattern)
      seqio.TaskRegistry.add(
          # Task name: f'{t_name}_type_{idx}_score_flan_no_options'
          utils.ZeroshotScoreFLANNoOptionTaskName.get(t_name, idx),
          source=config.source,
          preprocessors=config.preprocessors +
          # Format inputs and outputs according to the patterns. This should be
          # the same for all tasks.
          preprocessors.get_flan_formatter(inputs_pattern_no_options,
                                            targets_pattern) +
          [preprocessors.rank_classification_from_options] +
          # Tokenization for the prefix-LM. This should be the same for all
          # tasks.
          preprocessors.FLAN_TOKENIZE,
          postprocess_fn=t5_post.rank_classification,
          output_features=FLAN_OUTPUT_FEATURES,
          metric_fns=[t5_metrics.rank_classification])

  # Add a single task with all templates for that task.
  patterns_list = templates.PATTERNS[flan_pattern_name]
  for num_templates in NUM_TEMPLATES_LIST:
    selected_patterns = patterns_list[:num_templates]

    # Task names:
    # Zero-shot version: f'{t_name}_{num_templates}templates'
    # One-shot version: f'{t_name}_{num_templates}templates_one_shot'
    # Multi-shot version: f'{t_name}_{num_templates}templates_multi_shot'

    # Zero-shot version of the task.
    # Note: Used primarily for training.
    zero_shot_task_name = utils.ZeroshotTemplatedTaskName.get(
        t_name, num_templates)
    seqio.TaskRegistry.add(
        zero_shot_task_name,
        source=config.source,
        preprocessors=config.preprocessors +
        # This batch formatter applies many prompts to a single task.
        preprocessors.get_batch_flan_formatter(selected_patterns) +
        preprocessors.FLAN_TOKENIZE,
        output_features=FLAN_OUTPUT_FEATURES,
        metric_fns=config.metric_fns)

    # Few-shot versions of the task.
    register_few_shot_versions_of_task(
        zero_shot_task_name,
        prune_exemplars=True,
        max_input_length=960,
        multishot_max_num_shots=16)

  # For backwards compatibility.
  seqio.TaskRegistry.add(
      # Task name: f'{t_name}_all_prompts'.
      utils.AllPromptsTaskName.get(t_name),
      source=config.source,
      preprocessors=config.preprocessors +
      # This batch formatter applies many prompts to a single task.
      preprocessors.get_batch_flan_formatter(patterns_list) +
      preprocessors.FLAN_TOKENIZE,
      output_features=FLAN_OUTPUT_FEATURES,
      metric_fns=config.metric_fns)

  # Add task for non-flan baseline evaluation.
  if flan_pattern_name in baseline_templates.PATTERNS:
    continuation_patterns = baseline_templates.PATTERNS[flan_pattern_name]
    for idx, patterns in enumerate(continuation_patterns):
      inputs_pattern, targets_pattern = patterns
      name = f'continuations_{t_name}_type_{idx}'
      if utils.is_classification(flan_pattern_name):
        name += '_scoring_eval'
        seqio.TaskRegistry.add(
            name,
            source=config.source,
            preprocessors=config.preprocessors +
            preprocessors.get_glm_formatter(inputs_pattern, targets_pattern) +
            [preprocessors.GLM_RANK_CLASSIFICATION] +
            preprocessors.FLAN_TOKENIZE,
            postprocess_fn=t5_post.rank_classification,
            output_features=FLAN_OUTPUT_FEATURES,
            metric_fns=[t5_metrics.rank_classification])
      else:
        seqio.TaskRegistry.add(
            name,
            source=config.source,
            preprocessors=config.preprocessors +
            preprocessors.get_glm_formatter(inputs_pattern, targets_pattern) +
            preprocessors.FLAN_TOKENIZE,
            postprocess_fn=postprocessors.parse_glm_qa_answer,
            output_features=FLAN_OUTPUT_FEATURES,
            metric_fns=config.metric_fns)

      # Few-shot versions of the task.
      register_few_shot_versions_of_continuations_task(name)

seqio.MixtureRegistry.add(
  name=split.eval_mixture_name,
  tasks=split.test_tasks,
  default_rate=seqio.mixing_rate_num_examples)