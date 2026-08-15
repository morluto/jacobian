import json
from pathlib import Path


def term(coefficient, exponents):
    return {"coefficient": str(coefficient), "exponents": exponents}


result = {
    "p_numerator": [term(1, [0, 1, 0])],
    "p_denominator": [term(1, [1, 0, 0])],
    "q_numerator": [term(13, [0, 2, 0]), term(-12, [1, 0, 1])],
    "q_denominator": [term(1, [2, 0, 0])],
    "d_delta_inverse": [term(1, [1, 1, 0])],
    "d2_delta_numerator": [term(13, [1, 2, 0]), term(-1, [1, 0, 1])],
    "d2_delta_denominator": [term(12, [0, 0, 0])],
    "s_forward": [term(1, [0, 0, 3]), term(-1, [1, 0, 0])],
    "delta_inverse": [term(1, [0, 3, 0]), term(-1, [0, 0, 1])],
    "norm_polynomial": [
        term(1, [4, 0, 0]),
        term(-2, [2, 1, 0]),
        term(1, [0, 2, 0]),
        term(-1, [2, 0, 1]),
        term(-2, [1, 1, 1]),
        term(-1, [0, 2, 1]),
        term(-2, [1, 0, 2]),
        term(-2, [0, 1, 2]),
        term(-1, [0, 0, 3]),
    ],
}
Path("/app/submission.json").write_text(json.dumps({"result": result}, indent=2) + "\n")
