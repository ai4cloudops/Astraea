"""
------------------------
Issue the new sampling policy at every period
------------------------
"""
import os
from configparser import SafeConfigParser
from config import load_config

class AstraeaOrc():
    def __init__(self,app,span_states_file_txt,s3):
        parser = SafeConfigParser()
        self.app = app
        self.span_states_file_txt = span_states_file_txt
        self.s3=s3
        self.config = load_config('../conf/astraea-config.ini')
    def issue_sampling_policy(self, splits):
        ### Read file and if it includes the span from before, replace it or insert new
        # Opening JSON file
        if os.path.exists(self.span_states_file):
            f = open(self.span_states_file)
            # returns JSON object as 
            # a dictionary
            data = json.load(f)
            f.close()
        else:
            data = {}

        ## issue all the new sampling values to shared policy file
        for item in splits:
            data[item] = splits[item]

        f_write = open(self.span_states_file, "w+")
        json.dump(data, f_write)
        f_write.close()

    def issue_sampling_policy_txt(self, splits):
        ## check if remained in splits
        used = set()

        # opening the file in read mode
        file = open(self.span_states_file_txt, "r")
        replacement = ""

        # using the for loop
        for line in file:
            line = line.strip()
            line_arr = line.split(" ")
            if line_arr[0] in splits:
                changes = line.replace(line_arr[1], str(splits[line_arr[0]]))
                used.add(line_arr[0])
            else:
                changes = line

            #     changes = line.replace("hardships", "situations")
            replacement = replacement + changes + "\n"

        ## there is some new spans in split
        remaining = list(set(splits.keys()) - set(used))
        if len(remaining) > 0:
            for item in remaining:
                print("*-*-*-*-*-* Adding new span ", item, splits[item])
                replacement = replacement + item + " " + str(splits[item]) + "\n"

        file.close()
        # opening the file in write mode
        fout = open(self.span_states_file_txt, "w")
        fout.write(replacement)
        fout.close()
        try:
            self.s3.upload_file(self.span_states_file_txt, self.config["bucket_name"], self.config["s3_key"])
            print(f'Successfully uploaded {self.span_states_file_txt} to {self.config["bucket_name"]}/{self.config["s3_key"]}')
        except Exception as e:
            print(
                f'Error uploading {self.span_states_file_txt} to {self.config["bucket_name"]}/{self.config["s3_key"]}: {str(e)}')
