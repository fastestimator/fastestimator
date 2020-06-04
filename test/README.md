## Run nightly-test

```bash
$ python3 run_nightly_build.py
```

## Run PR-test

```bash
$ python3 run_pr_test.py
```

or

```bash
$ python3 -m unittest discover PR_test
```

### Run single test file

```bash
$ python3 -m unittest <file_path>
```

### Run specific test case(class)

```bash
$ python3 -m unittest <module_path>
```

Please be aware that the "<module_path>" here uses dot instead of slash
ex:

```bash
$ python3 -m unittest PR_test.unit_test.backend.test_abs.TestAbs
```

## Run PR-test and check its test coverage

* step 1: running PR test using coverage command line

    ```bash
    $ coverage run --source ../fastestimator -m unittest discover PR_test
    ```

    it will generate .coverage file in your current folder

* step 2: checkout report
    * plain way

        ```bash
        $ coverage report
        ```

    * html way

        ```bash
        $ coverage html
        ```

        It will generate a web project called "htmlcov" in your current folder.
        Double click the inside index.html to view the report in browser.
