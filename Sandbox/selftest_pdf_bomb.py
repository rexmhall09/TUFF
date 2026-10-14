"""Runs INSIDE the sandbox container, from Scripts/sandbox_selftest.sh:

  container exec --user 10001:10001 NAME python3 -c "$(cat Sandbox/selftest_pdf_bomb.py)"

(With `python3 -` the reader's child processes would try to re-import the main
program from standard input and never start, so the bomb would be "refused"
without being parsed. That is why this is passed with -c, and why a small valid
PDF is read first as a control.)

It reads a small valid PDF with the server's own bounded PDF reader, which must
return its text, then builds a PDF whose page content inflates to 3 GB of zeros
(a Flate bomb) and gives it to the same reader. The reader must refuse it
(`pdf_unreadable`) or leave it without text, in its child process, within its time limit. Prints one
line, "ok ..." or "FAIL ...", and exits 0 or 1.
"""
import io
import sys
import time
import zlib

sys.path.insert(0, "/app")
import server  # noqa: E402


def bomb(megabytes):
    packer = zlib.compressobj(9)
    piece = b"\x00" * 1_000_000
    packed = b"".join(packer.compress(piece) for _ in range(megabytes)) + packer.flush()
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R >>",
        b"<< /Length %d /Filter /FlateDecode >>\nstream\n" % len(packed) + packed + b"\nendstream",
    ]
    out = io.BytesIO(b"%PDF-1.4\n")
    out.seek(0, io.SEEK_END)
    offsets = []
    for number, body in enumerate(objects, 1):
        offsets.append(out.tell())
        out.write(b"%d 0 obj\n" % number + body + b"\nendobj\n")
    start = out.tell()
    out.write(b"xref\n0 5\n0000000000 65535 f \n")
    for offset in offsets:
        out.write(b"%010d 00000 n \n" % offset)
    out.write(b"trailer\n<< /Size 5 /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % start)
    return out.getvalue()


def small(text):
    content = b"BT /F1 12 Tf 72 720 Td (" + text.encode() + b") Tj ET"
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R "
        b"/Resources << /Font << /F1 5 0 R >> >> >>",
        b"<< /Length %d >>\nstream\n" % len(content) + content + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    out = io.BytesIO(b"%PDF-1.4\n")
    out.seek(0, io.SEEK_END)
    offsets = []
    for number, body in enumerate(objects, 1):
        offsets.append(out.tell())
        out.write(b"%d 0 obj\n" % number + body + b"\nendobj\n")
    start = out.tell()
    out.write(b"xref\n0 6\n0000000000 65535 f \n")
    for offset in offsets:
        out.write(b"%010d 00000 n \n" % offset)
    out.write(b"trailer\n<< /Size 6 /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % start)
    return out.getvalue()


def main():
    if server.PdfReader is None:
        print("FAIL pypdf is not installed in the image")
        return 1
    try:
        control = server.read_pdf_bounded(small("sandbox control text"))
    except server.ToolError as error:
        print(f"FAIL a small valid PDF was refused ({error.code}); the PDF child does not run")
        return 1
    if "sandbox control text" not in control["text"]:
        print(f"FAIL a small valid PDF came back as {control['text']!r}")
        return 1
    data = bomb(3000)
    started = time.monotonic()
    try:
        result = server.read_pdf_bounded(data)
    except server.ToolError as error:
        took = time.monotonic() - started
        if error.code != "pdf_unreadable":
            print(f"FAIL refused with {error.code}, not pdf_unreadable")
            return 1
        outcome = "refused as pdf_unreadable"
    else:
        # pypdf's own stream limit (PDF_STREAM_LIMIT) can stop the stream from
        # being decoded at all: the page then has no text (the server answers
        # `pdf_unreadable` for a PDF without text). The bomb is harmless as long
        # as nothing was read from it and the child ended in time.
        took = time.monotonic() - started
        if result["text"]:
            print(f"FAIL the bomb was read: {len(result['text'])} characters")
            return 1
        outcome = "not decoded (no text came out of it)"
    if took > server.PDF_TIMEOUT + 6:
        print(f"FAIL took {took:.1f} s, more than the limit allows")
        return 1
    print(f"ok a valid PDF is read, and the bomb was {outcome} after {took:.1f} s")
    return 0


sys.exit(main())
