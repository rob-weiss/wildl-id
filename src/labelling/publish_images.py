import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

DEFAULT_PROFILE = "wildl-id"
DEFAULT_REGION = "eu-central-1"
DEFAULT_STACK_NAME = "wildl-id-images"
IMAGES_DIRECTORY = Path(__file__).resolve().parents[2] / "images"


def build_sync_command(
    source: Path, bucket: str, dry_run: bool, progress: bool = False
) -> list[str]:
    command = [
        "aws",
        "s3",
        "sync",
        f"{source}/",
        f"s3://{bucket}/images/",
        "--delete",
        "--checksum-algorithm",
        "SHA256",
    ]
    if dry_run:
        command.append("--dryrun")
    if progress:
        command.append("--progress-multiline")
    return command


def build_force_upload_command(
    source: Path, bucket: str, relative_path: str
) -> list[str]:
    source_root = source.resolve()
    relative = Path(relative_path)
    if relative.is_absolute() or not relative.parts or ".." in relative.parts:
        raise ValueError("force-upload path must stay beneath the images directory")

    image_path = source / relative
    if not image_path.resolve().is_relative_to(source_root):
        raise ValueError("force-upload path must stay beneath the images directory")
    if not image_path.is_file():
        raise FileNotFoundError(image_path)

    return [
        "aws",
        "s3",
        "cp",
        str(image_path),
        f"s3://{bucket}/images/{relative.as_posix()}",
        "--checksum-algorithm",
        "SHA256",
    ]


def run_aws(
    command: list[str],
    profile: str,
    region: str,
    *,
    echo_output: bool = True,
    stream_output: bool = False,
) -> subprocess.CompletedProcess:
    command = [
        *command,
        "--profile",
        profile,
        "--region",
        region,
        "--no-cli-pager",
    ]
    try:
        if stream_output:
            return subprocess.run(command, check=True, text=True)
        result = subprocess.run(command, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as error:
        if error.stdout:
            sys.stdout.write(error.stdout)
        if error.stderr:
            sys.stderr.write(error.stderr)
        raise
    if echo_output:
        if result.stdout:
            sys.stdout.write(result.stdout)
        if result.stderr:
            sys.stderr.write(result.stderr)
    return result


def stack_outputs(stack_name: str, profile: str, region: str) -> dict[str, str]:
    result = run_aws(
        [
            "aws",
            "cloudformation",
            "describe-stacks",
            "--stack-name",
            stack_name,
            "--output",
            "json",
        ],
        profile,
        region,
        echo_output=False,
    )
    try:
        outputs = json.loads(result.stdout)["Stacks"][0]["Outputs"]
    except (IndexError, KeyError, json.JSONDecodeError) as error:
        raise RuntimeError(
            f"CloudFormation stack {stack_name} has no outputs"
        ) from error
    return {output["OutputKey"]: output["OutputValue"] for output in outputs}


def wait_for_invalidation(
    distribution_id: str, invalidation_id: str, profile: str, region: str
) -> None:
    command = [
        "aws",
        "cloudfront",
        "get-invalidation",
        "--distribution-id",
        distribution_id,
        "--id",
        invalidation_id,
        "--query",
        "Invalidation.Status",
        "--output",
        "text",
    ]
    for attempt in range(30):
        result = run_aws(command, profile, region, echo_output=False)
        status = result.stdout.strip()
        print(f"CloudFront invalidation: {status}", flush=True)
        if status == "Completed":
            return
        if status != "InProgress":
            raise RuntimeError(f"Unexpected CloudFront invalidation status: {status}")
        if attempt < 29:
            time.sleep(20)
    raise TimeoutError("CloudFront invalidation did not complete after 30 checks")


def publish_images(
    stack_name: str,
    profile: str,
    region: str,
    source: Path,
    dry_run: bool,
    force_upload: str | None,
    progress: bool = False,
) -> None:
    if dry_run and force_upload is not None:
        raise ValueError("--dry-run and --force-upload cannot be used together")

    outputs = stack_outputs(stack_name, profile, region)
    try:
        bucket = outputs["ImagesBucketName"]
        distribution_id = outputs["CloudFrontDistributionId"]
    except KeyError as error:
        raise RuntimeError(
            f"CloudFormation stack output {error.args[0]} is missing"
        ) from error

    if force_upload is not None:
        transfer_command = build_force_upload_command(source, bucket, force_upload)
    else:
        transfer_command = build_sync_command(source, bucket, dry_run, progress)
    run_aws(transfer_command, profile, region, stream_output=progress)

    if not dry_run:
        result = run_aws(
            [
                "aws",
                "cloudfront",
                "create-invalidation",
                "--distribution-id",
                distribution_id,
                "--paths",
                "/*",
            ],
            profile,
            region,
            echo_output=not progress,
        )
        if progress:
            try:
                invalidation_id = json.loads(result.stdout)["Invalidation"]["Id"]
            except (KeyError, json.JSONDecodeError) as error:
                raise RuntimeError("CloudFront invalidation response has no ID") from error
            wait_for_invalidation(distribution_id, invalidation_id, profile, region)


def main() -> int:
    parser = argparse.ArgumentParser(description="Publish local images to S3.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--force-upload", metavar="RELATIVE_PATH")
    parser.add_argument(
        "--progress",
        action="store_true",
        help="show transfer progress and wait for CloudFront invalidation",
    )
    parser.add_argument("--stack-name", default=DEFAULT_STACK_NAME)
    parser.add_argument("--profile", default=DEFAULT_PROFILE)
    parser.add_argument("--region", default=DEFAULT_REGION)
    arguments = parser.parse_args()
    try:
        publish_images(
            arguments.stack_name,
            arguments.profile,
            arguments.region,
            IMAGES_DIRECTORY,
            arguments.dry_run,
            arguments.force_upload,
            arguments.progress,
        )
    except (
        FileNotFoundError,
        RuntimeError,
        ValueError,
        subprocess.CalledProcessError,
    ) as error:
        parser.exit(1, f"publish-images: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
