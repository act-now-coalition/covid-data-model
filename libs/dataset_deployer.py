import os
import io
import logging
import boto3

_logger = logging.getLogger(__name__)


class DatasetDeployer(object):
    """Common deploy operations for persisting files to s3 or local folder.

    Deploys to S3 if the BUCKET_NAME environment variable is set.
    """

    def __init__(self, key="filename.csv", body="a random data", output_dir="."):
        self.s3 = boto3.client("s3")
        # Supplied by ENV on AWS
        # BUCKET_NAME format is s3://{BUCKET_NAME}
        self.bucket_name = os.environ.get("BUCKET_NAME")
        self.key = key
        self.body = body
        self.output_dir = output_dir

    def _persist_to_s3(self):
        """Persists specific data onto an s3 bucket.
        This method assumes versioned is handled on the bucket itself.
        """
        _logger.info(f"persisting {self.key} to s3")

        response = self.s3.put_object(
            Bucket=self.bucket_name, Key=self.key, Body=self.body, ACL="public-read"
        )
        return response

    def _persist_to_local(self):
        """Persists specific data onto an s3 bucket.
        This method assumes versioned is handled on the bucket itself.
        """
        _logger.info(f"persisting {self.key} {self.output_dir}")

        with open(os.path.join(self.output_dir, self.key), "wb") as f:
            # hack to allow the local writer to take either bytes or a string
            # note this assumes that all strings are given in utf-8 and not,
            # like, ASCII
            f.write(
                self.body.encode("UTF-8") if isinstance(self.body, str) else self.body
            )

        pass

    def persist(self):
        if self.bucket_name:
            self._persist_to_s3()
        else:
            self._persist_to_local()
        return


def upload_csv(key_name: str, csv: str, output_dir: str):
    blob = {
        "key": f"{key_name}.csv",
        "body": csv,
        "output_dir": output_dir,
    }
    obj = DatasetDeployer(**blob)
    obj.persist()
    _logger.info(f"Generated csv for {key_name}")


def upload_json(key_name, json: str, output_dir: str):
    DatasetDeployer(f"{key_name}.json", json, output_dir).persist()


def deploy_shape_files(
    output_dir: str,
    key: str,
    shp_bytes: io.BytesIO,
    shx_bytes: io.BytesIO,
    dbf_bytes: io.BytesIO,
):
    """Deploys shape files to specified output dir.

    Args:
        output_dir: Output directory to save shapefiles to.
        key: stem of filename to save shapefiles to.
        shp_bytes:
        shx_bytes:
        dbf_bytes:
    """
    DatasetDeployer(
        key=f"{key}.shp", body=shp_bytes.getvalue(), output_dir=output_dir
    ).persist()
    DatasetDeployer(
        key=f"{key}.shx", body=shx_bytes.getvalue(), output_dir=output_dir
    ).persist()
    DatasetDeployer(
        key=f"{key}.dbf", body=dbf_bytes.getvalue(), output_dir=output_dir
    ).persist()
