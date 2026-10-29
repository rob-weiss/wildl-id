# Cloud Image Storage

The local `images/` directory remains the source of truth. AWS S3 stores its published copy; CloudFront serves browser images through public HTTPS URLs. The S3 bucket itself stays private.

## First-Time Setup

Install AWS CLI v2, then sign in with the IAM user profile:

```bash
aws login --profile wildl-id --region eu-central-1
aws sts get-caller-identity --profile wildl-id --no-cli-pager
```

The caller identity must be the intended IAM user, not root. Do not create access keys or put AWS credentials in the repository or web container.

The IAM user needs CloudFormation permission to validate and deploy the `wildl-id-images` stack, in addition to the existing S3 and CloudFront permissions. Scope stack operations to this stack where AWS supports resource-level permissions. `cloudformation:ValidateTemplate` requires `Resource: "*"`.

Deploy the private bucket and CloudFront distribution:

```bash
aws cloudformation deploy \
  --template-file infra/images-stack.yaml \
  --stack-name wildl-id-images \
  --profile wildl-id \
  --region eu-central-1 \
  --no-cli-pager
```

Read the generated bucket name, distribution ID, and image URL:

```bash
aws cloudformation describe-stacks \
  --stack-name wildl-id-images \
  --query 'Stacks[0].Outputs' \
  --output table \
  --profile wildl-id \
  --region eu-central-1 \
  --no-cli-pager
```

Set `WILDL_ID_IMAGE_BASE_URL` to the `CloudFrontImageBaseUrl` output. For local Compose use, store that public URL in the ignored root `.env` file, not credentials:

```dotenv
WILDL_ID_IMAGE_BASE_URL=https://YOUR_DISTRIBUTION.cloudfront.net/images
```

Rebuild and restart the web container:

```bash
HOST_UID="$(id -u)" HOST_GID="$(id -g)" docker compose up -d --build web
```

The reviewer continues reading local originals for image dimensions and processing. It uses CloudFront only for browser display. Annotated-image fallbacks remain local.

## Publish Changes

First inspect the local mirror plan. This dry-run performs no S3 writes, deletions, or CloudFront invalidations:

```bash
uv run --locked python src/labelling/publish_images.py --dry-run
```

Review every planned deletion. The real command mirrors all files below `images/`, including deletions, to the bucket's `images/` prefix:

```bash
uv run --locked python src/labelling/publish_images.py
```

Before the first upload, check the transfer size and file count, then review the dry-run:

```bash
du -sh images
find images -type f | wc -l
uv run --locked python src/labelling/publish_images.py --dry-run
```

Do not run the real publish until the initial size, file count, and planned deletions have been reviewed and approved.

Do not edit files during a publish. S3 writes each completed object atomically, but a directory sync is not an atomic snapshot; the web app may briefly see a mix of old and new images. If a command fails or is interrupted, rerun it after local edits stop. Completed objects remain valid; the next run retries remaining work. A successful publish invalidates CloudFront so overwritten and deleted images stop using cached responses.

AWS CLI sync compares file size and modification time. Its SHA-256 option validates transfer integrity, but does not detect content edits that preserve both values. Force-upload that specific file when needed, using a path relative to `images/`:

```bash
uv run --locked python src/labelling/publish_images.py \
  --force-upload 2026/Grenzweg/image_0000001.jpg
```

## Restore an Image Version

S3 versioning retains overwritten and deleted object versions for 30 days. List versions for the object and choose a previous object version, not a delete marker:

```bash
aws s3api list-object-versions \
  --bucket YOUR_BUCKET \
  --prefix images/2026/Grenzweg/image_0000001.jpg \
  --profile wildl-id \
  --region eu-central-1 \
  --no-cli-pager
```

Download that version, replace the local source file, then publish the restored local copy:

```bash
aws s3api get-object \
  --bucket YOUR_BUCKET \
  --key images/2026/Grenzweg/image_0000001.jpg \
  --version-id VERSION_ID \
  /tmp/image_0000001.jpg \
  --profile wildl-id \
  --region eu-central-1 \
  --no-cli-pager

cp /tmp/image_0000001.jpg images/2026/Grenzweg/image_0000001.jpg
uv run --locked python src/labelling/publish_images.py
```

Versions older than 30 days expire and cannot be restored. Do not delete the CloudFormation stack or empty the versioned bucket as routine cleanup; review image retention and backup needs first.

## Public Access and Costs

CloudFront URLs work for anyone who obtains them; they are public links, not access control. CloudFront is public, while S3 blocks direct public access and grants reads only to this distribution. Check EXIF GPS metadata and sensitive camera locations before the first publish. The publisher does not strip metadata.

Costs depend on stored image volume, retained versions, requests, data transfer, and CloudFront invalidations. The distribution uses `PriceClass_100`; no custom domain or DNS records are required. Review AWS billing after the first publish and as usage grows.
