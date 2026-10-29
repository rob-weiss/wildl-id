# S3 Image Storage Design

## Goal

Keep `images/` as the authoritative local source while publishing additions, edits, and deletions to AWS. Let the review web app display images from public CloudFront URLs. Keep local filesystem access for image processing and metadata reads.

## Decisions

- Store image objects in an S3 bucket in `eu-central-1`.
- Keep the bucket private and block all public S3 access.
- Serve objects through a public CloudFront distribution using Origin Access Control (OAC). The distribution's default hostname is sufficient; no custom domain is required.
- Enable S3 versioning so sync deletions create recoverable delete markers and overwrites retain prior versions.
- Configure lifecycle cleanup for incomplete multipart uploads after one day. Propose expiring noncurrent object versions after 30 days; review this retention before deployment.
- Use AWS CLI with the `wildl-id` profile. A repository command mirrors `images/` to one dedicated bucket prefix, including deletions, then invalidates CloudFront so changed paths stop serving cached copies.
- Do not give the web container AWS credentials. Configure its public image base URL and derive object keys from the existing relative image paths.

## Data Flow

1. User changes local files under `images/`.
2. User runs the repository's image-publish command. It previews or executes local-to-S3 sync with deletion mirroring.
3. On successful sync, the command invalidates the CloudFront cache. A failed sync does not invalidate; rerunning the command converges the bucket to the unchanged local source.
4. The web app continues opening local files for Pillow dimensions and existing processing. It sets the browser image source to the corresponding CloudFront URL.
5. Existing local-only behavior remains available when no CloudFront base URL is configured.

## Consistency and Recovery

- S3 exposes each completed object write atomically, but a multi-object sync is not a transaction. The app can observe a mixed old/new dataset during publication.
- Do not edit images during a publish. Rerun after interruption before treating the cloud copy as current.
- AWS CLI sync generally compares size and modification time, not full file content. SHA-256 upload checks validate transfer integrity, but do not detect a local edit that preserves both size and modification time. Document this limitation and provide a force-upload command using `aws s3 cp --checksum-algorithm SHA256` for that rare case.
- Versioning supports recovery, but a delete marker does not itself restore a removed object. Document the restore operation.
- CloudFront invalidation is required after overwrites/deletions when immediate freshness matters. Public URLs are accessible to anyone who obtains them; they are not access control.

## Scope

- Infrastructure-as-code for the bucket, versioning, lifecycle rules, private-origin CloudFront distribution, OAC, and narrowly scoped origin policy.
- A local CLI command for dry-run and exact mirroring, with explicit target prefix, failure propagation, and post-success cache invalidation.
- Web-app configuration for CloudFront URLs, retaining local paths for processing.
- Documentation for setup, authentication, publishing, public access implications, restore, and teardown cautions.
- Focused tests for object-key mapping, URL construction, missing configuration fallback, and sync-command argument construction where practical.

## Non-Goals

- Moving image processing, labels, or model files to S3.
- Uploading `data/`, `labels/`, or generated logs.
- Editing or deleting objects directly in the AWS console as a normal workflow.
- Making the S3 bucket itself public.
- Automatic background sync while files are changing.
- Creating a custom domain or DNS records.

## Acceptance Criteria

- A dry-run shows planned uploads, updates, and deletions without changing AWS state.
- A successful publish makes local additions and edits available through CloudFront and removes cloud objects deleted locally.
- An interrupted publish can be rerun; completed objects remain valid and missing work is retried.
- S3 versioning allows recovery of an accidentally overwritten or deleted object within the retention window.
- Incomplete multipart uploads expire automatically.
- The bucket denies direct public access; CloudFront can read only through its OAC.
- The review app displays a configured CloudFront URL while retaining its current local-file processing behavior.
- No AWS credentials enter source control or the web container.

## Risks and Open Questions

- Public CloudFront URLs expose every object under the distribution path to anyone with a URL. Check image EXIF for GPS coordinates and review sensitive locations before first publication.
- Storage, CloudFront requests, data transfer, invalidations beyond the free allowance, and retained object versions can incur charges. Estimate cost after measuring image volume and expected traffic.
- Confirm whether 30-day noncurrent-version retention is acceptable before deployment.
- Confirm whether all files currently under `images/` should be public, or whether the publish command should filter extensions.
- Confirm image edits do not intentionally preserve both size and modification time; otherwise use content checksums to detect them.
