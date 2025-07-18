"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE_Lavis file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

from recAtk.models.minigpt4.common.registry import registry
from recAtk.models.minigpt4.tasks.base_task import BaseTask
from recAtk.models.minigpt4.tasks.rec_base_task import RecBaseTask
# from minigpt4.tasks.image_text_pretrain import ImageTextPretrainTask
from recAtk.models.minigpt4.tasks.rec_pretrain import RecPretrainTask


def setup_task(cfg):
    assert "task" in cfg.run_cfg, "Task name must be provided."

    task_name = cfg.run_cfg.task
    task = registry.get_task_class(task_name).setup_task(cfg=cfg)
    assert task is not None, "Task {} not properly registered.".format(task_name)

    return task


__all__ = [
    "BaseTask",
    # "ImageTextPretrainTask",
    "RecPretrainTask"
]
