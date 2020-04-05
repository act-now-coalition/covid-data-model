from io import BytesIO
import boto3
import os

from libs.build_dod_dataset import get_usa_by_county_df, get_usa_by_states_df, get_usa_county_shapefile, get_usa_state_shapefile
from libs.build_dod_dataset import get_usa_by_county_with_projection_df

class DatasetDeployer():

    def __init__(self, key='filename.csv', body='a random data'):
        self.s3 = boto3.client('s3')
        # Supplied by ENV on AWS
        # BUCKET_NAME format is s3://{BUCKET_NAME}
        self.bucket_name = os.environ.get('BUCKET_NAME')
        self.key = key
        self.body = body

    def _persist_to_s3(self):
        """Persists specific data onto an s3 bucket.
        This method assumes versioned is handled on the bucket itself.
        """
        print('persisting {} to s3'.format(self.key))

        response = self.s3.put_object(Bucket=self.bucket_name,
                                      Key=self.key,
                                      Body=self.body,
                                      ACL='public-read')
        return response

    def _persist_to_local(self):
        """Persists specific data onto an s3 bucket.
        This method assumes versioned is handled on the bucket itself.
        """
        print('persisting {} to local'.format(self.key))

        with open(self.key, 'wb') as f:
            # hack to allow the local writer to take either bytes or a string
            # note this assumes that all strings are given in utf-8 and not,
            # like, ASCII
            f.write(self.body.encode('UTF-8') if isinstance(self.body, str) else self.body)

        pass

    def persist(self):
        if self.bucket_name:
            self._persist_to_s3()
        else:
            self._persist_to_local()
        return


def deploy():
    """The entry function for invokation

    """
    states_blob = {
        'key': 'states.csv',
        'body': get_usa_by_states_df().to_csv()
        }
    statesObj = DatasetDeployer(**states_blob)
    statesObj.persist()

    counties_blob = {
        'key': 'counties.csv',
        'body': get_usa_by_county_with_projection_df().to_csv()
        }
    countiesObj = DatasetDeployer(**counties_blob)
    countiesObj.persist()

    states_shp = BytesIO()
    states_shx = BytesIO()
    states_dbf = BytesIO()
    get_usa_state_shapefile(states_shp, states_shx, states_dbf)
    DatasetDeployer(key='states.shp', body=states_shp.getvalue()).persist()
    DatasetDeployer(key='states.shx', body=states_shx.getvalue()).persist()
    DatasetDeployer(key='states.dbf', body=states_dbf.getvalue()).persist()

    counties_shp = BytesIO()
    counties_shx = BytesIO()
    counties_dbf = BytesIO()
    get_usa_county_shapefile(counties_shp, counties_shx, counties_dbf)
    DatasetDeployer(key='counties.shp', body=counties_shp.getvalue()).persist()
    DatasetDeployer(key='counties.shx', body=counties_shx.getvalue()).persist()
    DatasetDeployer(key='counties.dbf', body=counties_dbf.getvalue()).persist()

    print('finished dod job')


if __name__ == "__main__":
    """Used for manual trigger

    # triggering persistance to s3
    AWS_PROFILE=covidactnow BUCKET_NAME=covidactnow-deleteme python deploy_dod_dataset.py

    # triggering persistance to local
    python deploy_dod_dataset.py
    """

    deploy()
