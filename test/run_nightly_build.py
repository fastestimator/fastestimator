import papermill as pm
import os
import time

# os.system() will return 0 when it execute sucessfully
if __name__ == "__main__":
    test_apphub_script_dir = os.path.abspath(os.path.join(__file__, "..", "apphub_scripts"))
    test_tutorial_script_dir = os.path.abspath(os.path.join(__file__, "..", "tutorial_scripts"))

    report={}
    fail_list=[]
    for dirpath, _, filenames in os.walk(test_apphub_script_dir):
        if dirpath.endswith("/template"):
            continue

        for f in filenames:
            if not f.endswith(".py"):
                continue

            print(os.path.abspath(os.path.join(dirpath, f)))
            test_name = os.path.join(dirpath.split("/")[-1], f)
            start_time = time.time()
            result = os.system("python3 " + os.path.abspath(os.path.join(dirpath, f)))
            exec_time = int((time.time() - start_time) / 60)
            report[test_name] = {"fail": result, "time":exec_time}
            if result:
                os.system("echo ------------------------ {} fails ------------------------".format(test_name))
                error_report_file = ".".join(f.split(".")[:-1]) + ".txt"
                os.system("echo ================================ error log ================================")
                os.system("cat " + os.path.abspath(os.path.join(dirpath, error_report_file)))
                os.system("echo ===========================================================================")
                fail_list.append(test_name)

    for dirpath, _, filenames in os.walk(test_tutorial_script_dir):
        if dirpath.endswith("/template"):
            continue

        for f in filenames:
            if not f.endswith(".py"):
                continue

            print(os.path.abspath(os.path.join(dirpath, f)))
            test_name = os.path.join(dirpath.split("/")[-1], f)
            start_time = time.time()
            result = os.system("python3 " + os.path.abspath(os.path.join(dirpath, f)))
            exec_time = int((time.time() - start_time) / 60)
            report[test_name] = {"fail": result, "time":exec_time}
            if result:
                os.system("echo ------------------------ {} fails ------------------------".format(test_name))
                error_report_file = ".".join(f.split(".")[:-1]) + ".txt"
                os.system("echo ================================ error log ================================")
                os.system("cat " + os.path.abspath(os.path.join(dirpath, error_report_file)))
                os.system("echo ===========================================================================")
                fail_list.append(test_name)

    print("the report is: {}".format(report))
    print("the fail list is: {}".format(fail_list))
    os.system("rm -rf /tmp/tmp*")

    if fail_list:
        raise ValueError("not all tests passed")