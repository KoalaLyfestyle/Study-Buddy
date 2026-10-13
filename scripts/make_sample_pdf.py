"""Write the sample course PDF with no third-party dependency.

Usage: python scripts/make_sample_pdf.py

The output is byte-for-byte deterministic, so re-running it does not
change the committed file.
"""

from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / (
    "sample_course/intro-to-optimisation/week-02/convexity-lecture.pdf"
)

PAGES = [
    [
        "Week 2 lecture: convex sets and convex functions",
        "",
        "A set is convex when the straight segment between any two of its",
        "points stays inside the set. Balls, half-spaces, and polyhedra are",
        "convex. A crescent shape is not.",
        "",
        "A function f is convex when its epigraph is a convex set.",
        "Equivalently, for every x, y and every t between 0 and 1:",
        "f(t x + (1 - t) y) <= t f(x) + (1 - t) f(y).",
    ],
    [
        "Why convexity matters for optimisation",
        "",
        "For a convex function every local minimum is a global minimum.",
        "That is why gradient descent with a small enough step size finds",
        "the best answer on a convex problem, and why it can get stuck on a",
        "non-convex one.",
        "",
        "Second-order test: a twice-differentiable function is convex when",
        "its Hessian matrix is positive semidefinite everywhere.",
    ],
]


def _escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def build_pdf(pages: list[list[str]]) -> bytes:
    objects: list[bytes] = []
    n_pages = len(pages)
    font_id = 3 + 2 * n_pages
    kids = " ".join(f"{3 + 2 * i} 0 R" for i in range(n_pages))
    objects.append(b"<< /Type /Catalog /Pages 2 0 R >>")
    objects.append(f"<< /Type /Pages /Kids [{kids}] /Count {n_pages} >>".encode())
    for i, lines in enumerate(pages):
        content_id = 4 + 2 * i
        objects.append(
            (
                f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
                f"/Resources << /Font << /F1 {font_id} 0 R >> >> /Contents {content_id} 0 R >>"
            ).encode()
        )
        ops = ["BT", "/F1 12 Tf", "16 TL", "72 720 Td"]
        for line in lines:
            ops.append(f"({_escape(line)}) Tj T*")
        ops.append("ET")
        stream = "\n".join(ops).encode("latin-1")
        objects.append(b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream")
    objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")

    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += b"%d 0 obj\n" % number + body + b"\nendobj\n"
    xref = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objects) + 1)
    for offset in offsets:
        out += b"%010d 00000 n \n" % offset
    out += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (
        len(objects) + 1,
        xref,
    )
    return bytes(out)


if __name__ == "__main__":
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_bytes(build_pdf(PAGES))
    print(f"wrote {OUT}")
