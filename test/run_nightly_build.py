import os
import shutil
import time
from typing import Dict, List


def print_report(report: Dict[str, Dict[str, float]]) -> None:
    print("------------------------ Test report ------------------------")
    for test_name, test_report in report.items():
        result = "pass" if test_report["result"] == 0 else "fail"
        print("{}: {}, (spend {:.2f} min)".format(test_name, result, test_report["time"]))


def print_fail_list(fail_list: List[str]) -> None:
    print("------------------------ fail list ------------------------")
    if not fail_list:
        print("all test passed")

    for fail_test in fail_list:
        print(fail_test)


def print_test_error(test_name: str, stderr: str) -> None:
    os.system("echo ------------------------ {} fails ------------------------".format(test_name))
    os.system("echo ================================ error log ================================")
    os.system("cat " + stderr)
    os.system("echo ===========================================================================")


def test_apphub(report: Dict[str, Dict[str, float]], fail_list: List[str]) -> None:
    apphub_scripts_dir = os.path.abspath(os.path.join(__file__, "..", "apphub_scripts"))
    for dirpath, _, filenames in os.walk(apphub_scripts_dir):
        if dirpath.endswith("/template"):
            continue

        for f in filenames:
            if not f.endswith(".py"):
                continue

            print(os.path.abspath(os.path.join(dirpath, f)))
            test_name = os.path.join(os.path.relpath(dirpath, apphub_scripts_dir), f)

            if test_name in report:
                raise ValueError("some testing share the same name")

            start_time = time.time()
            result = os.system("python3 " + os.path.join(dirpath, f))
            exec_time = (time.time() - start_time) / 60
            report[test_name] = {"result": result, "time": exec_time}

            stderr_file = os.path.join(dirpath, f.replace(".py", "_stderr.txt"))

            if result:
                print_test_error(test_name, stderr_file)
                fail_list.append(test_name)


def test_tutorial(report: Dict[str, Dict[str, float]], fail_list: List[str]) -> None:
    src_tutorial = os.path.abspath(os.path.join(__file__, "..", "..", "tutorial"))
    dis_tutorial = os.path.abspath(os.path.join(__file__, "..", "tutorial_result"))
    if os.path.exists(dis_tutorial):
        shutil.rmtree(dis_tutorial)

    for dirpath, _, filenames in os.walk(src_tutorial):
        for f in filenames:
            if not f.endswith(".ipynb"):
                continue

            print(os.path.join(dirpath, f))
            rel_path = os.path.relpath(dirpath, src_tutorial)
            dis_dir = os.path.join(dis_tutorial, rel_path, f.split(".ipynb")[0])
            os.makedirs(dis_dir)

            nb_in_file = os.path.join(dirpath, f)
            nb_out_file = os.path.join(dis_dir, f.replace(".ipynb", "_out.ipynb"))
            stderr_file = os.path.join(dis_dir, "stderr.txt")

            start_time = time.time()
            result = os.system("papermill {} {} 2>> {} -k nightly_build".format(nb_in_file, nb_out_file, stderr_file))
            exec_time = (time.time() - start_time) / 60
            time.sleep(10)

            test_name = os.path.join(rel_path, f)
            if test_name in report:
                raise ValueError("some testing share the same name")

            report[test_name] = {"result": result, "time": exec_time}
            if result:
                print_test_error(test_name, stderr_file)
                fail_list.append(test_name)


# os.system() will return 0 when it execute sucessfully
if __name__ == "__main__":
    report = {}
    fail_list = []

    test_apphub(report, fail_list)
    test_tutorial(report, fail_list)

    print_report(report)
    print_fail_list(fail_list)

    os.system("rm -rf /tmp/tmp*")

    if fail_list:
        raise ValueError("not all tests passed")
