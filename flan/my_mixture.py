import functools
import seqio

from flan import few_shot
from flan import task_splits
from flan import tasks  # pylint: disable=unused-import
from flan import templates  # pylint: disable=unused-import

mixing_rate_3k = functools.partial(seqio.mixing_rate_num_examples, maximum=3000)

split = task_splits.generate_superglue_num_tasks_ablation()[-1]

seqio.MixtureRegistry.add(
  name="flan_trial",
  tasks=list(seqio.TaskRegistry.names()),
  default_rate=mixing_rate_3k)