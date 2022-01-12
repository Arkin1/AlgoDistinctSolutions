import xml.etree.ElementTree as ET
import os
from Constants import TMP_PATH
#Taken from AlgoLabel Repository
class RemoveUnusedFunctionsPreprocessingOp:

    def preprocess(self, source_code_data):
        code = source_code_data['preprocessed']
        id = source_code_data['id']

        f = open(f'{TMP_PATH}/{id}.cpp', 'w', encoding='utf8')
        f.write(code)
        f.close()

        os.system(f"cppcheck --enable=all --xml -q --output-file=\"{TMP_PATH}/{id}.xml\" {TMP_PATH}/{id}.cpp")

        lines        = code.split("\n")
        tree         = ET.parse(f'{TMP_PATH}/{id}.xml')
        root         = tree.getroot()
        errors       = root.find("errors")
        remove_lines = set()

        if not errors:
            source_code_data['preprocessed'] = "\n".join(lines)
            os.remove(f'{TMP_PATH}/{id}.xml')
            os.remove(f'{TMP_PATH}/{id}.cpp')
            return

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

        source_code_data['preprocessed'] = "\n".join(lines)

        os.remove(f'{TMP_PATH}/{id}.xml')
        os.remove(f'{TMP_PATH}/{id}.cpp')

        return source_code_data
