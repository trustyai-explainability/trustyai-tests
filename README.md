# trustyai-tests
Framework for TrustyAI functional integration tests. Work in progress.

## Overview
- Leverages [kubernetes python client](https://github.com/kubernetes-client/python) and [openshift-python-wrapper](https://github.com/RedHatQE/openshift-python-wrapper)
- The idea is to have the flexibility to do anything we would do through the command line, but programmatically through the K8S/OpenShift APIs, and to be able to work with cluster resources using simple Python objects.
- In this PoC, pytest was used (since pytest fixtures integrate nicely with openshift-python-wrapper), but similar results could be achieved using other testing frameworks
- These tests in principle could be run in different environments (vanilla K8S and OpenShift with OpenDataHub or OpenShift AI) with minimum effort, since they use the K8S/OpenShift API directly

## Directory
- [model_data](https://github.com/adolfo-ab/trustyai-tests/tree/main/data): train/test data to feed the models
- [resources](https://github.com/adolfo-ab/trustyai-tests/tree/main/resources): classes that define different K8S/OpenShift resources used in the tests. Most of these could be moved directly to openshift-python-wrapper.
- [tests](https://github.com/adolfo-ab/trustyai-tests/tree/main/tests): tests and pytest fixtures used in the PoC. Only a very simple test is provided here, just to demonstrate the possibilities of this approach.
- [utils](https://github.com/adolfo-ab/trustyai-tests/tree/main/utils): constants and other utils in tests and resources.

## Running the tests
- `export KUBECONFIG=${path to the kubeconfig of your cluster}`, or alternatively `oc login` into your cluster.
- Make sure you have Poetry installed.
- Install the project's dependencies with `poetry install`.
- Configure `pre-commit`
- Run the tests with `poetry run pytest -s --log-cli-level=DEBUG tests/your_tests.py`
