import xml.etree.ElementTree as ET
import os
from Constants import TMP_FOLDER_PATH
from uuid import uuid4
from preprocessing_operations.PreprocessingOp import PreprocessingOp

#Based on AlgoLabel Repository
class RemoveUnusedFunctionsPreprocessingOp(PreprocessingOp):
    def preprocess(self, source_code:str) -> str:
        if not isinstance(source_code, str):
            raise Exception(f'Source code should be a string. Instead it is {type(source_code)}')
        
        id_ = f'tmp_{uuid4()}'
        source_code_tmp_path = os.path.join(TMP_FOLDER_PATH, f'{id_}.cpp')
        cpp_check_result_tmp_path =  os.path.join(TMP_FOLDER_PATH, f'{id_}.xml')
        
        with open(source_code_tmp_path, 'w', encoding='utf8') as fp:
            fp.write(source_code)

        os.system(f"cppcheck --enable=all --xml -q --output-file=\"{cpp_check_result_tmp_path}\" {source_code_tmp_path}")

        lines        = source_code.split("\n")
        tree         = ET.parse(cpp_check_result_tmp_path)
        root         = tree.getroot()
        errors       = root.find("errors")
        remove_lines = set()

        if not errors:
            preprocessed_source_code = "\n".join(lines)
            os.remove(source_code_tmp_path)
            os.remove(cpp_check_result_tmp_path)
            return preprocessed_source_code

        for error in errors.findall("error"):
            if error.get('id') == "unusedFunction":
                location = int(error.find('location').get('line')) - 1
                count_ph = 0
                seen_the_end = False
                index = location

                for line in lines[location:]:
                    remove_lines.add(index)
                    index += 1
                    for ch in line:
                        if ch == "{":
                            count_ph += 1
                        elif ch == "}":
                            count_ph -= 1
                            seen_the_end = True

                    if count_ph == 0 and seen_the_end:
                        break

        lines = [line for idx, line in enumerate(lines)
                    if idx not in remove_lines and len(line) > 0]

        preprocessed_source_code = "\n".join(lines)
        os.remove(source_code_tmp_path)
        os.remove(cpp_check_result_tmp_path)
        return preprocessed_source_code
