import argparse
from datetime import datetime
import os
import subprocess
import sys

from run import INTERVENTIONS, OUTPUT_DIR
from libs.CovidDatasets import us_state_abbrev

STATE_COUNT = 53  # states + a few territories

def run_run_py(datasource: str) -> None:
  '''
  Check that we successfully ran `python run.py`
  '''
  env = dict(os.environ)
  env['COVID_DATA_PUBLIC'] = datasource
  start = datetime.now()
  result = subprocess.run(['python', 'run.py'], env=env, capture_output=True)
  end = datetime.now()
  # Will raise an appropriate exception if run.py crashed
  result.check_returncode()
  duration = end - start
  print(f'duration {duration}')

def clear_result_dir(result_dir: str) -> None:
  for f in os.listdir(result_dir):
    os.unlink(f)

def validate_results(result_dir: str) -> None:
  '''
  For each state, check that we have a file for each intervention,
  and that the file is non-empty
  '''
  per_state_expected = len(INTERVENTIONS)
  missing_or_empty = []
  for state in us_state_abbrev.values():
    for i in range(per_state_expected):
      fname = os.path.join(result_dir, '.'.join([state, str(i), 'json']))
      try:
        result = os.stat(fname)
        if result.st_size == 0:
          missing_or_empty.append(fname)
      except FileNotFoundError:
        missing_or_empty.append(fname)
  if len(missing_or_empty) > 0:
    raise RuntimeError(f'Missing or empty expected files: {", ".join(missing_or_empty)}')


if __name__ == '__main__':
  parser = argparse.ArgumentParser(
    description='Validate run.py against a dataset. For example, with a local checkout, try `python validate.py -d `pwd`/../covid-data-public`',
  )
  parser.add_argument('-d', '--data-source', required=True, help='Path or URL to an instance of the covid-data-public repo')

  args = parser.parse_args()
  try:
    run_run_py(args.data_source)
  except subprocess.CalledProcessError as e:
    print(f'run.py failed with code {e.returncode}')
    print(e.stderr.decode('utf-8'))
    sys.exit(e.returncode)
  validate_results(OUTPUT_DIR)
