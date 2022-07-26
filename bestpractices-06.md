# week 6: Best Practices

# Part A:  Testing with pytest

We want to implement tests for the Lambda function.

Installing pytest as a dev dependency using `pipenv`:
```
pipenv install --dev pytest
```
Then create a directory where tests are stored in the current directory (Lambda/GCP Function Directory); Example: `tests` with an `__init__.py` file (could be empty).

We can also configure the editor to run tests directly.

To write a test, add a file to the `tests` folder with the test you want. Example:
```python
def test_prediction():

    ride = {
    'PULocationID': 43,
    'DOLocationID': 215,
    'datetime': '2021-01-01 00:15:56',
    'trip_distance': 14
}

    main.download_files()

    import sys
    sys.path.insert(0,'/tmp')
    from preprocessor import preprocess_dict

    D = preprocess_dict(ride)
    X = main.vectorize(ride)
    predicted_duration = round(main.predict(X))

    assert predicted_duration == 36
```

**Note:** To easily deal with environment variables, put them in a `.env` file and install the package: `pytest-dotenv`. Example
```
PROJECT_ID=project-30393
```

#### Best practices for Unit testing:

+ Use many unit tests for smaller units
+ Use Mocks; 
### Integration Tests:
Integration tests are tests which cover the entire pipeline to assess how well the parts fit together.

#### Lambda functions: 
We Interface with the Docker container, the returned object is a dictionary and we use `deepdiff` to see the difference between the expected dictionary and the returned dictionary.
#### Google Functions:
Use `functions-framework`, print the result and look for it in the output pipe. In addition to running the function locally, it also "simulates" a PubSub topic.
```python
#! integration_test.py
import os
import requests
import subprocess
import base64
import json

from pathlib import Path

from requests.packages.urllib3.util.retry import Retry


def test_framework():

    ride = {
    'PULocationID': 43,
    'DOLocationID': 215,
    'datetime': '2021-01-01 00:15:56',
    'trip_distance': 14
}

    port = 8888
    ride_json = json.dumps(ride)
    encoded_ride = base64.b64encode(ride_json.encode('utf-8')).decode('utf-8')

    pubsub_message = {
        'data' : {'data': encoded_ride}
    }

    current_path = Path(os.path.dirname(__file__))

    parent_path = current_path.parent
    process = subprocess.Popen(
      [
        'functions-framework',
        '--target', 'predict_duration',
        '--signature-type', 'event',
        '--port', str(port)
      ],
      cwd=parent_path,
      stdout=subprocess.PIPE
    )

    url = f'http://localhost:{port}/'

    retry_policy = Retry(total=6, backoff_factor=1)
    retry_adapter = requests.adapters.HTTPAdapter(
      max_retries=retry_policy)

    session = requests.Session()
    session.mount(url, retry_adapter)

    response = session.post(url, json=pubsub_message)

    assert response.status_code == 200

    # Stop the functions framework process
    process.kill()
    process.wait()
    out, err = process.communicate()

    print(out, err, response.content)

    assert '36' in str(out) #Search for the result
```

### Pipeline tests:

In addition to seeing how well our entire function works. We may also want to see how well it interacts with the pipeline itself. For AWS, we use localstack. For GCP, we can use functions-framework like above, or also `gcloud beta emulators` (Docs: [here ](https://cloud.google.com/sdk/gcloud/reference/beta/emulators))

# Code Style, Linting, Formatting:

In Python, it is recommended to follow the [PEP8 guidelines](https://pep8.org/). This ensures clean, standard code formats and helps  code readability as well as minimizing diffs. To do this automatically, we use Formatters

In addition to following style guides, we also want our code to be free of bad practices and deprecated language features.
### Linting:

We may want styled code, however, conforming to the PEP8 style manually may be cumbersome. So we can use Linters instead. Linters are pieces of software that make sure the code conforms to a certain style with minimal hinderence to the developer. For Python, a common linter is pylint. To use pylint, simply install it via `pip` or `pipenv` and run `pylint <file>` where `<file>` is the file we want to lint.

To ignore certain errors in certain regions, we use `# pylint: disable=[ERROR CODE]` blocks in the code; Example:
```python

# pylint: disable=C0413, W0621

import os
# ...
# Many bad imports that we're aware of
# pylint: enable=C0413, W0621
```

To ignore certain errors in the entire file(s) in the directory. We use a `pylintrc` file.

### Formatting:

For Formatting Python code, a common tool is Black. Run in the same way as `pylint`.
For formatting import statements, we use `isort` same as earlier.

# Git pre-commit hoooks:

Testing, Formatting, and Linting should be run every time the code changes. However, it is very easy to forget doing it. So we may want to run them every time we commit to the repository.

To install pre-commit hooks package, simply install `pre-commit` via `pip` or `pipenv`

#### Initializing pre-commit-hooks config file:

Pre-commit (the program), uses a file named: `.pre-commit-config.yaml` to specify what programs will run at every commit (the hooks). We can initialize this file (optional) with a set of common provided hooks. A sample config file is output via `pre-commit sample-config`, output the `stdout` to the `.pre-commit-config.yaml` by: `pre-commit sample-config > .pre-commit-config.yaml` in Bash.

#### Adding hooks:

**Note:** It is imperative to follow the indentation in the `yaml` file. Bad indentations will cause the program to halt. One that was annoying for me was the spaces between `repo` and the dash.

We can add hooks that run either from local programs, or hooks provided by certain common software like `isort`:

##### Online repos:

Usually online repos will provide a config like:
```yaml
-   repo: https://github.com/pycqa/isort
        rev: 5.10.1
        hooks:
        - id: isort
          name: isort (python)
```

##### Local:

Certain programs need to run locally. Example: Pytest

To run a hook from the local "repo" we use a config similar to:
```yaml
-   repo: local
        hooks:
        - id: pylint
          name: pylint
          entry: pylint
          language: system
          types: [python]
          args:
            [
            "-rn", # Only display messages
            "-sn", # Don't display the score
            ]
```

# Makefiles and Make:

Make is a tool for automating various steps for production.

`make` looks for the `Makefile`, which contains the steps needed to automate the build.

`make` makes use of aliases, like:
```makefile
run:
    echo 123
```
here `run` is an alias. Now when `make run` is executed, `echo 123` is executed as a result.

We can also make aliases depend on other aliases
```makefile
test: 
    echo test
run: test
    echo 123
```
here `run` depends on `test`, and when `make run` is executed, both `echo test` and `echo 123` are executed in that order.

In our case, we want to run many things before running the program or commiting or deploying to AWS Lambda/GCP Functions/.... To do so we can make use of `Makefile`. We want to run:

1. Tests (Unit tests and integration tests): using `pytest`
2. Quality checks: `pylint`, `black`, `isort`

An example on how `Makefile` can be used in our case:

```makefile
LOCAL_TAG:=$(shell date +"%Y-%m-%d-%H-%M")

test:
	pytest 
quality_check: test
	black .
	isort .
	pylint --recursive=y -sn -rn .
build: test quality_check
	./set_envs.sh
	echo ${LOCAL_TAG}
	cd web-service/ && ./docker-build.sh web-prediction-service:${LOCAL_TAG}
publish: test quality_check build
	echo $(PROJECT_ID)
	cd function/ && ./deploy.sh 
```

# Part B:  Infrastructure as Code ( IaC)
