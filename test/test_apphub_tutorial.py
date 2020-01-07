import papermill as pm
import os

test_apphub_script_dir = os.path.abspath(os.path.join(__file__, "..", "apphub_scripts"))
print(test_apphub_script_dir)

for dirpath, _, filenames in os.walk(test_apphub_script_dir):
    for f in filenames:
           print(os.path.abspath(os.path.join(dirpath, f)))
           os.system("python3 " + os.path.abspath(os.path.join(dirpath, f)))

