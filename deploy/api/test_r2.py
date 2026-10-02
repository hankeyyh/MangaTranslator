"""R2 request validation, upload contract, and completion event ordering."""

from __future__ import annotations

import importlib
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import MagicMock, Mock, patch

from PIL import Image
from pydantic import ValidationError

from deploy.api.schemas import CreateJobRequest
from deploy.api.storage import upload_r2_object


class R2SchemaTests(unittest.TestCase):
    def request(self, **output):
        return CreateJobRequest.model_validate(
            {"images": [{"image_id": "page"}], "output": output}
        )

    def test_accepts_r2_with_optional_paths(self):
        for paths in ([], ["user/task/page.webp"]):
            request = self.request(type="r2", bucket="results", paths=paths)
            self.assertEqual(request.output.type, "r2")
            self.assertEqual(request.output.paths, paths)

    def test_requires_bucket_and_matching_paths(self):
        for output in (
            {"type": "r2"},
            {"type": "r2", "bucket": ""},
            {"type": "r2", "bucket": "results", "paths": ["a", "b"]},
        ):
            with self.subTest(output=output), self.assertRaises(ValidationError):
                self.request(**output)

    def test_existing_output_types_still_work(self):
        self.assertEqual(CreateJobRequest(images=[{"image_id": "page"}]).output.type, "none")
        for output_type in ("none", "volume", "supabase"):
            self.request(type=output_type, bucket="results")


class R2UploadTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.env = patch.dict(
            os.environ,
            {
                "R2_ENDPOINT": "https://account.r2.cloudflarestorage.com/",
                "R2_ACCESS_KEY_ID": "test-access",
                "R2_SECRET_ACCESS_KEY": "test-secret",
            },
            clear=True,
        )
        self.env.start()
        self.addCleanup(self.env.stop)

    def test_put_uses_requested_bucket_key_bytes_and_content_type(self):
        for suffix, content_type in (("webp", "image/webp"), ("png", "image/png"), ("jpg", "image/jpeg")):
            file = Path(self.tmp.name) / f"page.{suffix}"
            file.write_bytes(b"translated-image")
            key = f"user/task/中文 page.{suffix}"
            client = Mock(spec=["put_object", "close"])
            def check_put(**kwargs):
                self.assertEqual(kwargs["Bucket"], "results")
                self.assertEqual(kwargs["Key"], key)
                self.assertEqual(kwargs["Body"].read(), b"translated-image")
                self.assertEqual(kwargs["ContentType"], content_type)
            client.put_object.side_effect = check_put
            with self.subTest(suffix=suffix), patch("boto3.client", return_value=client) as factory:
                self.assertEqual(upload_r2_object("results", key, file), key)
                options = factory.call_args.kwargs
                self.assertEqual(options["endpoint_url"], "https://account.r2.cloudflarestorage.com")
                self.assertEqual(options["region_name"], "auto")
                self.assertEqual(options["aws_access_key_id"], "test-access")
                self.assertEqual(options["aws_secret_access_key"], "test-secret")
                self.assertEqual(options["config"].signature_version, "s3v4")
                self.assertEqual(options["config"].s3["addressing_style"], "path")
                client.put_object.assert_called_once()
                client.close.assert_called_once()

    def test_real_s3_client_upload_without_context_manager(self):
        import boto3
        from botocore.stub import ANY, Stubber

        client = boto3.client(
            "s3",
            endpoint_url="https://account.r2.cloudflarestorage.com",
            aws_access_key_id="test-access",
            aws_secret_access_key="test-secret",
            region_name="auto",
        )
        self.addCleanup(client.close)
        file = Path(self.tmp.name) / "page.webp"
        file.write_bytes(b"image")
        with Stubber(client) as stubber:
            stubber.add_response(
                "put_object",
                {"ETag": '"test-etag"'},
                {"Bucket": "results", "Key": "key", "Body": ANY, "ContentType": "image/webp"},
            )
            with patch("boto3.client", return_value=client), patch.object(client, "close", wraps=client.close) as close:
                self.assertEqual(upload_r2_object("results", "key", file), "key")
                close.assert_called_once()
            stubber.assert_no_pending_responses()

    def test_upload_failure_propagates(self):
        file = Path(self.tmp.name) / "page.webp"
        file.write_bytes(b"image")
        client = Mock(spec=["put_object", "close"])
        client.put_object.side_effect = RuntimeError("upload denied")
        with patch("boto3.client", return_value=client), self.assertRaisesRegex(RuntimeError, "upload denied"):
            upload_r2_object("results", "key", file)
        client.close.assert_called_once()

    def test_missing_credentials_fail_before_client_creation(self):
        for missing in ("R2_ENDPOINT", "R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY"):
            with self.subTest(missing=missing), patch.dict(os.environ, {missing: ""}), patch("boto3.client") as factory:
                with self.assertRaisesRegex(RuntimeError, "are required"):
                    upload_r2_object("results", "key", Path("unused.webp"))
                factory.assert_not_called()


class R2WorkerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Persistence tests do not load the GPU translation / model stack.
        config_map = types.ModuleType("deploy.api.config_map")
        config_map.build_mt_config = MagicMock()
        with patch.dict(sys.modules, {"deploy.api.config_map": config_map}):
            cls.worker = importlib.import_module("deploy.api.worker")

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.file = Path(self.tmp.name) / "page.webp"
        Image.new("RGB", (16, 16), "red").save(self.file)
        self.request = {
            "job_id": "job",
            "output": {"type": "r2", "bucket": "results", "paths": ["user/task/page.webp"]},
        }

    def test_completion_event_follows_compression_and_successful_upload(self):
        events = []
        def upload(bucket, key, file):
            self.assertEqual((bucket, key), ("results", "user/task/page.webp"))
            self.assertEqual(events, [])
            with Image.open(file) as image:
                self.assertEqual(image.format, "WEBP")
            return key
        with patch.object(self.worker, "upload_r2_object", side_effect=upload), patch.object(
            self.worker, "_append_job_event", side_effect=lambda store, job, event, data: events.append((event, data))
        ):
            self.assertTrue(self.worker._persist_translated_image(None, "job", self.request, {"image_id": "page"}, 0, self.file))
        self.assertEqual(events, [("image_completed", {"image_id": "page", "index": 0, "output_path": "user/task/page.webp"})])

    def test_failed_upload_emits_failure_without_completion(self):
        events = []
        with patch.object(self.worker, "upload_r2_object", side_effect=RuntimeError("upload denied")), patch.object(
            self.worker, "_append_job_event", side_effect=lambda store, job, event, data: events.append(event)
        ), patch.object(self.worker.traceback, "print_exc"):
            self.assertFalse(self.worker._persist_translated_image(None, "job", self.request, {"image_id": "page"}, 0, self.file))
        self.assertEqual(events, ["image_failed"])

    def test_default_key_and_none_output(self):
        self.request["output"].pop("paths")
        with patch.object(self.worker, "upload_r2_object", return_value="job/page.webp") as upload:
            self.assertEqual(self.worker._write_output(self.request, {"image_id": "page"}, 0, self.file), "job/page.webp")
            upload.assert_called_once_with("results", "job/page.webp", self.file)
        self.assertIsNone(self.worker._write_output({"output": {"type": "none"}}, {}, 0, self.file))


if __name__ == "__main__":
    unittest.main()
