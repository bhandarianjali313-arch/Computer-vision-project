import argparse
from pathlib import Path

from ml.src.verification.project_verifier import (
    build_verification_report,
    save_verification_report,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run final reproducibility checks "
            "for the industrial defect "
            "detection ML pipeline."
        )
    )

    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path("."),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "outputs/verification/"
            "day22_project_verification.json"
        ),
    )

    parser.add_argument(
        "--strict-artifacts",
        action="store_true",
        help=(
            "Require local dataset and "
            "model artifacts to exist."
        ),
    )

    return parser.parse_args()


def print_section(
    title,
    section,
):
    status = (
        "PASS"
        if section[
            "passed"
        ]
        else "FAIL"
    )

    print(
        f"{title:<28}: "
        f"{status}"
    )


def main():
    args = parse_args()

    report = (
        build_verification_report(
            project_root=(
                args.project_root
            ),
            strict_artifacts=(
                args.strict_artifacts
            ),
        )
    )

    save_verification_report(
        report,
        args.output,
    )

    print("=" * 78)
    print(
        "DAY 22 - FINAL ML "
        "PROJECT VERIFICATION"
    )
    print("=" * 78)

    print(
        f"Python              : "
        f"{report['environment']['python_version']}"
    )

    print(
        f"Executable          : "
        f"{report['environment']['python_executable']}"
    )

    print(
        f"Project root        : "
        f"{report['project_root']}"
    )

    print()

    sections = (
        report[
            "sections"
        ]
    )

    print_section(
        "Required sources",
        sections[
            "required_sources"
        ],
    )

    print_section(
        "NEU class mapping",
        sections[
            "class_mapping"
        ],
    )

    print_section(
        "Runtime configuration",
        sections[
            "runtime_configuration"
        ],
    )

    print_section(
        "Quality policy",
        sections[
            "quality_policy"
        ],
    )

    print_section(
        (
            "Local artifacts"
            if args.strict_artifacts
            else "Local artifacts (optional)"
        ),
        sections[
            "artifacts"
        ],
    )

    print(
        "\nArtifact status:"
    )

    for (
        name,
        information,
    ) in (
        sections[
            "artifacts"
        ][
            "checks"
        ]
        .items()
    ):
        marker = (
            "FOUND"
            if information[
                "exists"
            ]
            else "MISSING"
        )

        print(
            f"  {name:<30} "
            f"{marker}"
        )

    print("\n" + "=" * 78)

    final_status = (
        "PASSED"
        if report[
            "passed"
        ]
        else "FAILED"
    )

    print(
        f"FINAL STATUS: "
        f"{final_status}"
    )

    print(
        f"Report      : "
        f"{args.output}"
    )

    print("=" * 78)

    if not report[
        "passed"
    ]:
        raise SystemExit(
            1
        )


if __name__ == "__main__":
    main()