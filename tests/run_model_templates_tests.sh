python3 -m venv venv || exit 1
. venv/bin/activate || exit 1
pip install --upgrade pip || exit 1
pip install -e ote || exit 1

python tests/run_model_templates_tests.py `pwd` $@ || exit 1
