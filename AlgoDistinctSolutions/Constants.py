TMP_FOLDER_PATH = 'tmp'
TOKENIZER_PATH = '/app/tokenizer/src/tokenizer'
TFIDF_STOP_WORDS = ["(", ")", ".", "#", ";", ",", ">>", "<<", "{", "}", "[", "]", "'.'", "\"...\"", ">", "<", "std", 
                    "fout", "fin", "++", "--", "<=", ">=", "auto", "auto", "bool","break", "case", "char", "char8_t",
                    "char16_t", "char32_t", "class", "concept", "const", "consteval", "constexpr", "continue", 
                    "decltype", "default", "do", "double", "else", "enum", "explicit","false", "float", "for",
                    "if", "inline", "int", "long", "new", "noexcept", "not", "operator", "private", "protected", 
                    "public", "return", "short", "signed", "sizeof", "static", "struct", "switch", "template", 
                    "this", "true", "union", "unsigned", "using", "virtual", "void", "volatile", "wchar_t", "while"]

TFIDF_TOKEN_PATTERN = "\w+|\+|-|=|!="

NOT_COMPILABLE_PATH = 'not_compilable'

SEED  = 42