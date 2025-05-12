import ast
import tokenize
import io
import math
import re
from collections import Counter
import random

# ---------------------------------------------------------------
# Utility Functions
# ---------------------------------------------------------------

def remove_c_comments(code: str) -> str:
    """
    Removes C/C++ single-line (//) and multi-line (/* */) comments.
    """
    return code
    # Remove multi-line comments
    code = re.sub(r'/\*.*?\*/', '', code, flags=re.DOTALL)
    # Remove single-line comments
    code = re.sub(r'//.*', '', code)
    return code


# ---------------------------------------------------------------
# 2. Halstead Difficulty
# ---------------------------------------------------------------

def halstead_difficulty(code: str, language: str="python") -> float:
    """
    Computes a simplified Halstead Difficulty metric.
    
    For Python, uses the tokenize module.
    For C/C++, uses a regex tokenizer with a heuristic classification:
      - Tokens that are pure alphanumeric (or underscore) are considered operands.
      - All other tokens are considered operators.
    
    The formula used is:
       Difficulty = (n1 / 2) * (N2 / n2)
    where:
       n1: number of distinct operators
       n2: number of distinct operands
       N2: total occurrences of operands
    """
    if language.lower() in ["c", "c++", "cpp"]:
        # Remove comments first
        code = remove_c_comments(code)
        # A simple regex tokenizer: words and non-whitespace non-word characters.
        tokens = re.findall(r'\w+|[^\s\w]', code)
        operators = set()
        operands = []
        for token in tokens:
            # A token is an operand if it is composed of alphanumerics/underscore.
            if re.fullmatch(r'[A-Za-z_]\w*', token) or re.fullmatch(r'\d+', token):
                operands.append(token)
            else:
                operators.add(token)
        n1 = len(operators)
        distinct_operands = set(operands)
        n2 = len(distinct_operands)
        N2 = len(operands)
        if n2 == 0:
            return 0.0
        return (n1 / 2) * (N2 / n2)
    elif language.lower() in ["python"]:
        # Python code using tokenize.
        operators = set()
        operands = []
        try:
            tokens = tokenize.generate_tokens(io.StringIO(code).readline)
        except tokenize.TokenError:
            return 0.0  # In case of tokenization error.
        for tok in tokens:
            tok_type = tok.type
            tok_string = tok.string
            if tok_type == tokenize.OP:
                operators.add(tok_string)
            elif tok_type in (tokenize.NAME, tokenize.NUMBER, tokenize.STRING):
                operands.append(tok_string)
        n1 = len(operators)
        distinct_operands = set(operands)
        n2 = len(distinct_operands)
        N2 = len(operands)
        if n2 == 0:
            return 0.0
        return (n1 / 2) * (N2 / n2)
    else:
        raise ValueError("Unsupported language. Use 'python' or 'c'.")

# ---------------------------------------------------------------
# 3. McCabe Cyclomatic Complexity
# ---------------------------------------------------------------

# Python implementation using AST.
class ComplexityVisitor(ast.NodeVisitor):
    """
    AST visitor to compute cyclomatic complexity.
    Counts decision points (if, for, while, with, try) and boolean operators.
    """
    def __init__(self):
        self.complexity = 0

    def generic_visit(self, node):
        if isinstance(node, (ast.If, ast.For, ast.While, ast.With, ast.Try)):
            self.complexity += 1
        if isinstance(node, ast.BoolOp):
            self.complexity += len(node.values) - 1
        super().generic_visit(node)

def cyclomatic_complexity(code: str, language: str="cpp") -> int:
    """
    Computes the cyclomatic complexity.
    
    For Python, parses the AST.
    For C/C++, uses regex to count keywords and operators that increase complexity.
    A base value of 1 is added.
    """
    if language.lower() in ["c", "c++", "cpp"]:
        code = remove_c_comments(code)
        # Keywords and symbols that likely add a decision point.
        patterns = [
            r'\bif\b',
            r'\bfor\b',
            r'\bwhile\b',
            r'\bcase\b',
            r'\bdefault\b',
            r'\belse\s+if\b',
            r'\?\s*',         # Ternary operator '?'
            r'&&',
            r'\|\|'
        ]
        count = 0
        for pat in patterns:
            count += len(re.findall(pat, code))
        return count + 1
    elif language.lower() in ["python"]:
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            print(f"Syntax error in code: {e}")
            print(f"Syntax error in code: {e}")
            return 0  # or raise an exception if preferred
        visitor = ComplexityVisitor()
        visitor.visit(tree)
        return visitor.complexity + 1
    else:
        raise ValueError("Unsupported language. Use 'python' or 'c'.")

# ---------------------------------------------------------------
# 4. Depth of Nesting
# ---------------------------------------------------------------

# Python implementation using AST.
class NestingDepthVisitor(ast.NodeVisitor):
    """
    AST visitor to compute maximum depth of nested control structures.
    """
    def __init__(self):
        self.max_depth = 0
        self.current_depth = 0

    def generic_visit(self, node):
        if isinstance(node, (ast.FunctionDef, ast.For, ast.While, ast.If, ast.With, ast.Try)):
            self.current_depth += 1
            self.max_depth = max(self.max_depth, self.current_depth)
            super().generic_visit(node)
            self.current_depth -= 1
        else:
            super().generic_visit(node)

def depth_of_nesting(code: str, language: str="python") -> int:
    """
    Computes the maximum depth of nested control structures.
    
    For Python, uses AST traversal.
    For C/C++, a simple heuristic is applied by scanning for '{' and '}'.
    """
    if language.lower() in ["c", "c++", "cpp"]:
        code = remove_c_comments(code)
        max_depth = 0
        current_depth = 0
        # Simple scan: increase depth on '{', decrease on '}'.
        for char in code:
            if char == '{':
                current_depth += 1
                max_depth = max(max_depth, current_depth)
            elif char == '}':
                current_depth = max(current_depth - 1, 0)
        return max_depth
    elif language.lower() in ["python"]:
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return 0
        visitor = NestingDepthVisitor()
        visitor.visit(tree)
        return visitor.max_depth
    
    else:
        raise ValueError("Unsupported language. Use 'python' or 'c'.")

# ---------------------------------------------------------------
# 5. Byte Entropy
# ---------------------------------------------------------------

def byte_entropy(code: str) -> float:
    """
    Computes the Shannon entropy of the code’s byte-level representation.
    (Same for any language.)
    """
    byte_data = code.encode('utf-8')
    if not byte_data:
        return 0.0
    freq = Counter(byte_data)
    total = len(byte_data)
    return -sum((count / total) * math.log2(count / total) for count in freq.values())


def sample_code(corpus_dict, sample_num=10, halstead_range=(8, 10), cyclomatic_range=(1,3), don_range=(1,1.5), language='python', seed=42):

    metric_dict = {}

    for cid, code in corpus_dict.items():
        metric_dict[cid] = {}
        try:
            hal = halstead_difficulty(code, language=language)
            cyc = cyclomatic_complexity(code, language=language)
            don = depth_of_nesting(code, language=language)
        except:
            continue

        metric_dict[cid]['halstead'] = hal
        metric_dict[cid]['cyclomatic'] = cyc
        metric_dict[cid]['depth_of_nesting'] = don

        keys_sample_from = []

        if hal >= halstead_range[0] and hal <= halstead_range[1] and cyc >= cyclomatic_range[0] and cyc <= cyclomatic_range[1] and don >= don_range[0] and don <= don_range[1]:
            keys_sample_from.append(cid)

        if len(keys_sample_from) >= sample_num:
            return random.sample(keys_sample_from, sample_num)
        return keys_sample_from, metric_dict
