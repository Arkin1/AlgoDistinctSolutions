import os
from uuid import uuid4
from Constants import TMP_FOLDER_PATH, TOKENIZER_PATH
from subprocess import run

class TokenizePreprocessingOp:
    def preprocess(self, source_code:str) -> list[str]:
        id_ = f'tmp_{uuid4()}'

        source_code_tmp_path = os.path.join(TMP_FOLDER_PATH, f'{id_}.cpp')
        source_code_tokens_tmp_path = os.path.join(TMP_FOLDER_PATH, f'{id_}.tokens')

        with open(source_code_tmp_path, 'w', encoding='utf8') as fp:                          
            fp.write(source_code)

        with open(source_code_tokens_tmp_path, "w", encoding = 'utf8') as fp:
            rc = run(f"{TOKENIZER_PATH} -b {source_code_tmp_path}", shell=True, stdout=fp, stderr=fp, check = True)
            if not rc:
                raise Exception("Error while tokenizing source code")

        with open(source_code_tokens_tmp_path, "r", encoding = 'utf8') as fp:
            tokens = []
            for line in fp:
                if "//" not in line and "/*" not in line:
                    tokens.append(line.strip())
                elif "EOF encountered" in line:
                    raise Exception("Failure occured during tokenizer!")
                
        os.remove(source_code_tmp_path)
        os.remove(source_code_tokens_tmp_path)

        return tokens