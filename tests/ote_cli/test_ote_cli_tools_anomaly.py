import os
import shutil
from subprocess import run

import pytest

from ote_cli.registry import Registry

registry = Registry("external")


def get_template_rel_dir(template):
    return os.path.dirname(os.path.relpath(template["path"]))


def get_some_vars(template, root):
    template_dir = get_template_rel_dir(template)
    task_type = template["task_type"]
    work_dir = os.path.join(root, task_type)
    template_work_dir = os.path.join(work_dir, template_dir)
    algo_backend_dir = "/".join(template_dir.split("/")[:2])
    return template_dir, work_dir, template_work_dir, algo_backend_dir


def gen_parse_model_template_tests(task_type):
    class MyTests:
        pass

    args = {
        "--train-ann-file": "tmp/anomalib/train.json",
        "--train-data-roots": "tmp/anomalib/dataset/",
        "--val-ann-file": "tmp/anomalib/val.json",
        "--val-data-roots": "tmp/anomalib/dataset/",
        "--test-ann-files": "tmp/anomalib/test.json",
        "--test-data-roots": "tmp/anomalib/dataset/",
    }

    root = "/tmp/ote_cli/"
    ote_dir = os.getcwd()

    test_id = 0
    for template in registry.filter(task_type=task_type).templates:

        @pytest.mark.run(order=test_id)
        def test_ote_train(self, template=template):
            template_dir, work_dir, template_work_dir, algo_backend_dir = get_some_vars(template, root)
            assert run(f"./{algo_backend_dir}/init_venv.sh {work_dir}/venv", check=True, shell=True).returncode == 0
            os.makedirs(template_work_dir, exist_ok=True)
            print(f"{template_work_dir=}")

            command_line = (
                f"ote_train "
                f'--train-ann-file {os.path.join(ote_dir, args["--train-ann-file"])} '
                f'--train-data-roots {os.path.join(ote_dir, args["--train-data-roots"])} '
                f'--val-ann-file {os.path.join(ote_dir, args["--val-ann-file"])} '
                f'--val-data-roots {os.path.join(ote_dir, args["--val-data-roots"])} '
                f"--save-weights {template_work_dir}/trained.pth "
            )
            print(f". {work_dir}/venv/bin/activate && pip install -e ote_cli && cd {template_dir} && {command_line}")
            assert (
                run(
                    f". {work_dir}/venv/bin/activate && pip install -e ote_cli && cd {template_dir} && {command_line}",
                    check=True,
                    shell=True,
                ).returncode
                == 0
            )

        setattr(
            MyTests,
            "test_ote_train_" + template["task_type"] + "__" + get_template_rel_dir(template),
            test_ote_train,
        )
        test_id += 1

        # @pytest.mark.run(order=test_id)
        # def test_ote_eval(self, template=template):
        #     template_dir, work_dir, template_work_dir, algo_backend_dir = get_some_vars(template, root)
        #     assert run(f"./{algo_backend_dir}/init_venv.sh {work_dir}/venv", check=True, shell=True).returncode == 0
        #     os.makedirs(template_work_dir, exist_ok=True)
        #     print(f"{template_work_dir=}")

        #     command_line = (
        #         f"ote_eval "
        #         f'--test-ann-file {os.path.join(ote_dir, args["--test-ann-files"])} '
        #         f'--test-data-roots {os.path.join(ote_dir, args["--test-data-roots"])} '
        #         f"--load-weights {template_work_dir}/trained.pth "
        #     )

        #     assert (
        #         run(
        #             f". {work_dir}/venv/bin/activate && pip install -e ote_cli && cd {template_dir} && {command_line}",
        #             check=True,
        #             shell=True,
        #         ).returncode
        #         == 0
        #     )

        # setattr(
        #     MyTests, "test_ote_eval_" + template["task_type"] + "__" + get_template_rel_dir(template), test_ote_eval
        # )
        # test_id += 1

        # @pytest.mark.run(order=test_id)
        # def test_ote_export(self, template=template):
        #     template_dir, work_dir, template_work_dir, algo_backend_dir = get_some_vars(template, root)
        #     assert run(f"./{algo_backend_dir}/init_venv.sh {work_dir}/venv", check=True, shell=True).returncode == 0
        #     os.makedirs(template_work_dir, exist_ok=True)
        #     print(f"{template_work_dir=}")

        #     command_line = (
        #         f"ote_export "
        #         f"--labels person "
        #         f"--load-weights {template_work_dir}/trained.pth "
        #         f"--save-model-to {template_work_dir}/exported"
        #     )

        #     assert (
        #         run(
        #             f". {work_dir}/venv/bin/activate && pip install -e ote_cli && cd {template_dir} && {command_line}",
        #             check=True,
        #             shell=True,
        #         ).returncode
        #         == 0
        #     )

        # setattr(
        #     MyTests, "test_ote_export_" + template["task_type"] + "__" + get_template_rel_dir(template), test_ote_export
        # )
        # test_id += 1

    return MyTests


class TestOteCliWithANOMALY(gen_parse_model_template_tests(task_type="ANOMALY_CLASSIFICATION")):
    pass
