# S3 Image Storage Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Publish the local `images/` tree to S3 and display original images through public CloudFront URLs while preserving local processing.

**Architecture:** CloudFormation provisions a private, versioned S3 bucket and a public CloudFront distribution restricted to that bucket through OAC. A Python wrapper invokes native AWS CLI sync with deletion mirroring, then invalidates CloudFront; the web app maps original local paths to the distribution URL and retains local paths for processing and annotated-image fallback.

**Tech Stack:** AWS CloudFormation, AWS CLI v2, Python standard library, NiceGUI, FastAPI, unittest.

**Spec:** [2026-10-29-s3-image-storage-design.md](../specs/2026-10-29-s3-image-storage-design.md)

## Global Constraints

- Store objects in `eu-central-1`.
- Keep S3 Block Public Access enabled; serve public objects only through CloudFront OAC.
- Expire incomplete multipart uploads after 1 day.
- Retain noncurrent object versions for 30 days; treat this as the proposed default pending review before deployment.
- AWS CLI sync detects local changes by size and modification time; `--checksum-algorithm SHA256` validates transfer integrity, not change detection.
- Keep `images/` authoritative and do not upload `data/`, `labels/`, `models/`, or `logs/`.
- Never provide AWS credentials to the web container.
- Do not edit images during a publish; multi-object sync is not atomic.
- Use profile `wildl-id` and stack name `wildl-id-images` by default.

## Review Focus

- Dry-run and failed transfers must not cause invalidation; pin both in publisher tests.
- A same-size edit with preserved modification time needs the documented force-upload path; pin its generated `aws s3 cp --checksum-algorithm SHA256` command in publisher tests.
- Nested and URL-sensitive image paths must map to the same S3 key; pin this in URL-mapping tests.
- Missing originals, annotated-image fallback, and unset CDN configuration must preserve local behavior; pin these in review-app tests.
- S3 must deny direct public reads while CloudFront OAC can read only the image prefix; verify template policy and deployed bucket/distribution settings.

---

## Files

- Create `infra/images-stack.yaml` for the bucket, lifecycle rules, OAC, distribution, and scoped bucket policy.
- Create `src/labelling/publish_images.py` for stack-output lookup, dry-run/publish, and cache invalidation.
- Create `tests/test_publish_images.py` for publisher command sequencing and failure behavior.
- Modify `src/labelling/review_app.py` to select a CDN URL for original images while retaining local paths for Pillow and annotated fallbacks.
- Modify `web/app.py` and `compose.yaml` to pass optional `WILDL_ID_IMAGE_BASE_URL` configuration.
- Extend `tests/test_web_app.py` for URL mapping and local fallback behavior.
- Create `docs/image-storage.md` and link it from `README.md` for setup, publish, public-access, recovery, and cost guidance.
- Keep `docs/superpowers/specs/2026-10-29-s3-image-storage-design.md` and this plan as design records.

## Task 1: Add AWS Infrastructure Template

**Files:**
- Create: `infra/images-stack.yaml`

**Interfaces:**
- Produces CloudFormation stack `wildl-id-images` with outputs `ImagesBucketName`, `CloudFrontDistributionId`, and `CloudFrontImageBaseUrl`.
- Stores objects below the `images/` prefix; CloudFront base URL ends with `/images`.

- [ ] **Step 1: Write the CloudFormation template**

Define an S3 bucket with versioning enabled, all four Block Public Access settings enabled, and lifecycle rules to abort incomplete multipart uploads after 1 day and expire noncurrent versions after 30 days. Define a CloudFront distribution with an S3 REST origin, Origin Access Control, HTTPS viewer access, and a bucket policy granting `s3:GetObject` only to that distribution for `images/*`. Do not add a custom domain.

- [ ] **Step 2: Validate the template**

Run: `uvx --from cfn-lint cfn-lint infra/images-stack.yaml`
Expected: No lint errors. CloudFormation API validation remains gated on granting `cloudformation:ValidateTemplate`.

- [ ] **Step 3: Review security invariants**

Confirm the template has no public bucket policy, keeps all Block Public Access settings enabled, scopes OAC access to `images/*`, and sets lifecycle values to 1 and 30 days.

## Task 2: Build the Native AWS CLI Publisher

**Files:**
- Create: `src/labelling/publish_images.py`
- Create: `tests/test_publish_images.py`

**Interfaces:**
- `build_sync_command(source: Path, bucket: str, dry_run: bool) -> list[str]`
- `build_force_upload_command(source: Path, bucket: str, relative_path: str) -> list[str]`
- `publish_images(stack_name: str, profile: str, region: str, source: Path, dry_run: bool, force_upload: str | None) -> None`
- CLI: `uv run python src/labelling/publish_images.py [--dry-run] [--force-upload RELATIVE_PATH] [--stack-name NAME] [--profile PROFILE] [--region REGION]`

- [ ] **Step 1: Write tests for command construction and dry-run safety**

Assert that sync targets `s3://<bucket>/images/`, includes `--delete` and `--checksum-algorithm SHA256`, and adds AWS CLI `--dryrun` when requested. Cover missing stack outputs, sync failure, force-upload key validation, and no invalidation after failed sync or during dry-run.

- [ ] **Step 2: Run the focused tests and confirm failure**

Run: `PYTHONPATH=. uv run --locked python -m unittest discover -s tests -p test_publish_images.py -v`
Expected: FAIL because the publisher module does not exist.

- [ ] **Step 3: Implement stack output lookup and sync**

Use `subprocess.run(..., check=True, capture_output=True, text=True)` with argument arrays, never shell interpolation. Read bucket and distribution outputs with `aws cloudformation describe-stacks`. Run `aws s3 sync <source>/ s3://<bucket>/images/ --delete --checksum-algorithm SHA256`; add `--dryrun` only in dry-run mode. For `--force-upload`, validate the relative path stays beneath the source directory, then run `aws s3 cp <file> s3://<bucket>/images/<key> --checksum-algorithm SHA256`.

- [ ] **Step 4: Invalidate CloudFront only after a successful real sync**

After a successful non-dry-run sync or force-upload, call `aws cloudfront create-invalidation --distribution-id <id> --paths '/*'`. A transfer failure must stop execution before invalidation. Let AWS CLI errors propagate with a nonzero exit.

- [ ] **Step 5: Run the focused tests**

Run: `PYTHONPATH=. uv run --locked python -m unittest discover -s tests -p test_publish_images.py -v`
Expected: PASS, including dry-run, missing stack outputs, sync failure, unsafe paths, and no invalidation after failure.

## Task 3: Display CloudFront URLs in the Review App

**Files:**
- Modify: `src/labelling/review_app.py`
- Modify: `web/app.py`
- Modify: `compose.yaml`
- Modify: `tests/test_web_app.py`

**Interfaces:**
- `cloud_image_url(image_path: Path, images_root: Path, base_url: str | None) -> str | None`
- `ReviewApp` receives optional `image_base_url`; `None` preserves current local serving.

- [ ] **Step 1: Write URL mapping tests**

Cover a nested original path, spaces and Unicode in path segments, path outside `images/`, missing CDN base URL, and an annotated-image fallback. Assert URLs encode path segments while preserving `/` separators.

- [ ] **Step 2: Run the focused tests and confirm failure**

Run: `PYTHONPATH=. uv run --locked python -m unittest discover -s tests -p test_web_app.py -v`
Expected: New URL mapping tests fail because CDN mapping is absent.

- [ ] **Step 3: Add the pure URL mapping helper**

In `src/labelling/review_app.py`, map only paths beneath the local `images/` root. Strip trailing slashes from the base URL, URL-encode each relative path while keeping separators, and return `None` when the path is outside the root or no base URL is configured.

- [ ] **Step 4: Keep local reads and use the CDN only as the browser source**

Continue opening `current_image_path` with Pillow for dimensions and overlays. Use the CDN URL only when the selected original path is beneath `images/`; leave annotated fallback paths local. In `web/app.py`, pass `WILDL_ID_IMAGE_BASE_URL` into `ReviewApp`. Add the optional environment variable to the web service in `compose.yaml` without adding AWS credentials.

- [ ] **Step 5: Run focused app tests**

Run: `PYTHONPATH=. uv run --locked python -m unittest discover -s tests -p test_web_app.py -v`
Expected: PASS, including existing local-only tests and new CDN/fallback cases.

## Task 4: Document Setup, Publishing, and Recovery

**Files:**
- Create: `docs/image-storage.md`
- Modify: `README.md`

- [ ] **Step 1: Document AWS prerequisites and stack deployment**

Document the `wildl-id` profile, `eu-central-1`, stack name, CloudFormation permission prerequisite, deployment command, stack outputs, setting `WILDL_ID_IMAGE_BASE_URL`, and rebuilding/restarting the web container.

- [ ] **Step 2: Document safe publish workflow**

Document dry-run first, then the real command; explain `--delete` mirrors local deletions, edits during publish are unsupported, and retries converge after interruption. Document restore using `aws s3api list-object-versions` and `aws s3api get-object --version-id`, then republish the restored local file. Explain public URLs are usable by anyone who obtains them and warn about EXIF/GPS and sensitive locations.

Document that ordinary sync compares size and modification time. For a file whose content changed without either changing, provide the publisher's `--force-upload <relative-path>` command and explain that its SHA-256 transfer checksum validates the replacement.

- [ ] **Step 3: Link the guide from README and verify docs**

Run: `git diff --check`
Expected: No whitespace errors; README links to `docs/image-storage.md`.

## Task 5: Deploy and Verify AWS Resources

**Gate:** Do not deploy until the user has approved the plan and 30-day version retention, and the required CloudFormation permissions are attached. Review the dry-run after deployment and before any image upload. Before the initial real publish, report the local image count/size and obtain confirmation for that transfer. Scope stack operations to this stack where supported; `cloudformation:ValidateTemplate` requires `Resource: "*"`.

**Files:**
- No source changes; deploy `infra/images-stack.yaml`.

- [ ] **Step 1: Deploy the stack**

Run: `aws cloudformation deploy --template-file infra/images-stack.yaml --stack-name wildl-id-images --profile wildl-id --region eu-central-1 --no-cli-pager`
Expected: Stack reaches `CREATE_COMPLETE`; save the bucket name, distribution ID, and base URL from stack outputs.

- [ ] **Step 2: Verify private bucket and recovery settings**

Use `aws s3api get-public-access-block`, `get-bucket-versioning`, and `get-bucket-lifecycle-configuration`. Confirm public access blocks are all enabled, versioning is enabled, and lifecycle values match the approved retention.

- [ ] **Step 3: Configure the web app and preview publishing**

Set `WILDL_ID_IMAGE_BASE_URL` to the stack output, restart the web container, and run publisher dry-run. Confirm the destination is only the intended `images/` prefix and review every proposed deletion.

- [ ] **Step 4: Obtain approval before first upload, then publish**

Report local file count and total bytes, public-access implications, and estimated costs where available. Run the real publisher only after approval. Verify one known image returns successfully over HTTPS through CloudFront and direct anonymous S3 access is denied.

## Task 6: Run Focused Regression Checks

- [ ] Run: `PYTHONPATH=. uv run --locked python -m unittest discover -s tests`
  Expected: PASS.
- [ ] Run: `HOST_UID="$(id -u)" HOST_GID="$(id -g)" docker compose config --quiet`
  Expected: PASS with valid compose configuration.
- [ ] Run: `git diff --check`
  Expected: PASS.
