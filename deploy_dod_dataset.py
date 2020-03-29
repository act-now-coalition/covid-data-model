import boto3
import os

from libs.build_dod_dataset import get_usa_by_county_df, get_usa_by_states_df, get_usa_county_shapefile, get_usa_state_shapefile


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

        with open(self.key, 'w') as f:
            f.write(self.body)

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
        'body': get_usa_by_county_df().to_csv()
        }
    countiesObj = DatasetDeployer(**counties_blob)
    countiesObj.persist()

    get_usa_county_shapefile('shapefiles/counties')
    get_usa_state_shapefile('shapefiles/states')

    print('finished dod job')


if __name__ == "__main__":
    """Used for manual trigger

    # triggering persistance to s3
    AWS_PROFILE=covidactnow BUCKET_NAME=covidactnow-models-staging python deploy_dod_dataset.py

    # triggering persistance to local
    python deploy_dod_dataset.py
    """
    deploy()
