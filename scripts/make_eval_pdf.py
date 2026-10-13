"""Write the evaluation corpus PDF with no third-party dependency.

Usage: python scripts/make_eval_pdf.py

Like make_sample_pdf.py, the output is byte-for-byte deterministic.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_sample_pdf import build_pdf  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / (
    "eval/extra/intro-to-optimisation/week-04/second-order-methods.pdf"
)

PAGES = [
    [
        "Week 4 lecture: second-order methods",
        "",
        "Newton's method uses curvature as well as slope. Each step solves",
        "H(x) d = -grad f(x) for the direction d and moves to x + d, where",
        "H is the Hessian matrix of second derivatives.",
        "",
        "Near a minimum with a positive definite Hessian, Newton's method",
        "converges quadratically: the number of correct digits roughly",
        "doubles every step.",
    ],
    [
        "The cost of curvature",
        "",
        "Forming the Hessian takes n^2 entries and solving the linear",
        "system takes about n^3 operations, which is too expensive when",
        "a model has millions of parameters.",
        "",
        "Quasi-Newton methods such as BFGS build an approximation of the",
        "inverse Hessian from successive steps and gradient differences.",
        "L-BFGS keeps only the last few of these pairs, so memory grows",
        "linearly in n.",
    ],
    [
        "Line search",
        "",
        "Instead of a fixed step, a backtracking line search starts with a",
        "full step and halves it until the Armijo condition holds:",
        "f(x + t d) <= f(x) + c t grad f(x)^T d, with c around 1e-4.",
        "This guarantees enough decrease without tuning a learning rate.",
    ],
]


if __name__ == "__main__":
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_bytes(build_pdf(PAGES))
    print(f"wrote {OUT}")
