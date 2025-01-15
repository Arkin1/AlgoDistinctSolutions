# SAFE TEAM
# distributed under license: GPL 3 License http://www.gnu.org/licenses/

from safe.asm_embedding.FunctionAnalyzerRadare import RadareFunctionAnalyzer
from argparse import ArgumentParser
from safe.asm_embedding.FunctionNormalizer import FunctionNormalizer
from safe.asm_embedding.InstructionsConverter import InstructionsConverter
from safe.neural_network.SAFEEmbedder import SAFEEmbedder
from safe.utils import utils
from pathlib import Path

from pprint import pprint as pp
import logging
from os import path
from tqdm import tqdm


class SAFE:

    def __init__(self,
                 model_path="./data/safe_trained_X86.pb",
                 instr_conv="./data/i2v/word2id.json",
                 max_instr=150):

        self.converter = InstructionsConverter(instr_conv)
        self.normalizer = FunctionNormalizer(max_instruction=max_instr)
        self.embedder = SAFEEmbedder(model_path)
        self.embedder.loadmodel()
        self.embedder.get_tensor()

    def embedd_function(self, filename, address):
        analyzer = RadareFunctionAnalyzer(filename, use_symbol=False, depth=0)
        functions = analyzer.analyze()
        instructions_list = None
        for function in functions:
            if functions[function]['address'] == address:
                instructions_list = functions[function]['filtered_instructions']
                break
        if instructions_list is None:
            print("Function not found")
            return None
        converted_instructions = self.converter.convert_to_ids(instructions_list)
        instructions, length = self.normalizer.normalize_functions([converted_instructions])
        embedding = self.embedder.embedd(instructions, length)
        return embedding

    def embedd_functions(self, filename):
        analyzer = RadareFunctionAnalyzer(filename, use_symbol=False, depth=0)
        functions = analyzer.analyze()
        embeddings = []
        for function in functions:
            instructions_list = functions[function]['filtered_instructions']
            converted_instructions = self.converter.convert_to_ids(instructions_list)
            instructions, length = self.normalizer.normalize_functions([converted_instructions])
            embedding = self.embedder.embedd(instructions, length)
            embeddings.append(embedding)
        return embeddings

if __name__ == '__main__':

    utils.print_safe()

    parser = ArgumentParser(description="Safe Embedder")

    parser.add_argument("-m", "--model",   help="Safe trained model to generate function embeddings")
    parser.add_argument("-i", "--input",   help="Input executable that contains the function to embedd")
    parser.add_argument("-a", "--address", help="Hexadecimal address of the function to embedd")

    args = parser.parse_args()
    address = int(args.address, 16)
    safe = SAFE(args.model)

    embedding = safe.embedd_function(args.input, address)
    print(embedding[0])




