import subprocess


def test_main():
    assert subprocess.check_output(["surfwetter-ml", "foo", "foobar"], text=True) == "foobar\n"
