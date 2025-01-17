import tree_sitter

class ExtractFunctionsAndMethodsPreprocessingOp:
    def __init__(self, language = 'cpp'):
        if language == 'cpp':
            import tree_sitter_cpp as tcpp
            self.language = tree_sitter.Language(tcpp.language())
            self.parser = tree_sitter.Parser(self.language)

    def preprocess(self, source_code:str) -> str:
        tree = self.parser.parse(bytes(source_code, "utf8"))

        functions = []
        for x in self._traverse_tree(tree):
            if x.type == 'function_definition':
                functions.append(x.text.decode('utf-8'))
                
        return functions

    def _traverse_tree(self, tree):
        cursor = tree.walk()

        visited_children = False
        while True:
            if not visited_children:
                yield cursor.node
                if not cursor.goto_first_child():
                    visited_children = True
            elif cursor.goto_next_sibling():
                visited_children = False
            elif not cursor.goto_parent():
                break