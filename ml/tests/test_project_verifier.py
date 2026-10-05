from pathlib import Path

from ml.src.verification.project_verifier import (
    EXPECTED_CLASS_NAMES,
    build_verification_report,
    save_verification_report,
    verify_artifacts,
    verify_class_mapping,
    verify_required_sources,
    verify_runtime_configuration,
)


PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[2]
)


def test_expected_class_mapping():

    assert (
        EXPECTED_CLASS_NAMES
        == [
            "crazing",
            "inclusion",
            "patches",
            "pitted_surface",
            "rolled-in_scale",
            "scratches",
        ]
    )


def test_project_class_mapping():

    result = (
        verify_class_mapping()
    )

    assert (
        result[
            "passed"
        ]
        is True
    )


def test_runtime_configuration():

    result = (
        verify_runtime_configuration()
    )

    assert (
        result[
            "passed"
        ]
        is True
    )


def test_required_sources():

    result = (
        verify_required_sources(
            PROJECT_ROOT
        )
    )

    assert (
        result[
            "passed"
        ]
        is True
    )


def test_non_strict_artifacts_are_optional(
    tmp_path: Path,
):

    result = (
        verify_artifacts(
            tmp_path,
            strict=False,
        )
    )

    assert (
        result[
            "passed"
        ]
        is True
    )


def test_strict_artifacts_fail_when_missing(
    tmp_path: Path,
):

    result = (
        verify_artifacts(
            tmp_path,
            strict=True,
        )
    )

    assert (
        result[
            "passed"
        ]
        is False
    )


def test_complete_non_strict_report():

    report = (
        build_verification_report(
            project_root=(
                PROJECT_ROOT
            ),
            strict_artifacts=False,
        )
    )

    assert (
        report[
            "passed"
        ]
        is True
    )


def test_save_verification_report(
    tmp_path: Path,
):

    report = {
        "passed":
            True
    }

    output = (
        tmp_path
        / "verification.json"
    )

    save_verification_report(
        report,
        output,
    )

    assert (
        output.exists()
    )

    assert (
        '"passed": true'
        in output.read_text(
            encoding="utf-8"
        )
    )