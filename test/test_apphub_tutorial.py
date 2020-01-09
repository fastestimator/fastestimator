import papermill as pm
import os
import time

# os.system() will return 0 when it execute sucessfully 
if __name__ == "__main__":
    test_apphub_script_dir = os.path.abspath(os.path.join(__file__, "..", "apphub_scripts"))
    test_tutorial_script_dir = os.path.abspath(os.path.join(__file__, "..", "tutorial_scripts"))
    print(test_apphub_script_dir)
    print(test_tutorial_script_dir)

    report={}
    fail_list=[]
    # for dirpath, _, filenames in os.walk(test_apphub_script_dir):
    #     if dirpath.split("/")[-1] == "template": 
    #         continue

    #     for f in filenames:
    #         if f.split(".")[-1] != "py":
    #             continue

    #         print(os.path.abspath(os.path.join(dirpath, f)))
    #         test_name = os.path.join(dirpath.split("/")[-1], f)
    #         start_time = time.time()
    #         result = os.system("python3 " + os.path.abspath(os.path.join(dirpath, f)))
    #         exec_time = (time.time() - start_time) / 60 
    #         report[test_name] = {"fail": result, "time":exec_time}
    #         if result:
    #             fail_list.append(test_name)

    for dirpath, _, filenames in os.walk(test_tutorial_script_dir):
        print(dirpath)
        if dirpath.split("/")[-1] == "template": 
            continue

        for f in filenames:
            if f.split(".")[-1] != "py":
                continue

            print(os.path.abspath(os.path.join(dirpath, f)))
            test_name = os.path.join(dirpath.split("/")[-1], f)
            start_time = time.time()
            result = os.system("python3 " + os.path.abspath(os.path.join(dirpath, f)))
            exec_time = (time.time() - start_time) / 60 
            report[test_name] = {"fail": result, "time":exec_time}
            if result:
                fail_list.append(test_name)

    print("the report is: {}".format(report))
    print("the fail list is: {}".format(fail_list))


