from typing import Optional
import datetime
import base64
import os
import tempfile
import structlog
import gspread
import pathlib
from libs.datasets import dataset_utils
from libs.datasets import dataset_pointer
from libs.datasets.dataset_utils import DatasetType
from libs.datasets.dataset_pointer import DatasetPointer


_logger = structlog.getLogger(__name__)

# base 64 encoded service account json file.
SERVICE_ACCOUNT_DATA_ENV_NAME = "GOOGLE_SHEETS_SERVICE_ACCOUNT_DATA"


def init_client() -> gspread.Client:
    service_account_data = os.environ.get(SERVICE_ACCOUNT_DATA_ENV_NAME)
    if service_account_data:
        _logger.info("Loading service account from env variable data.")
        service_account_data = base64.b64decode(service_account_data)

        with tempfile.NamedTemporaryFile() as tmp_file:
            tmp_file.write(service_account_data)
            tmp_file.flush()
            return gspread.service_account(filename=tmp_file.name)

    return gspread.service_account()


def create_or_replace_worksheet(
    spreadsheet: gspread.Spreadsheet, worksheet_name: str
) -> gspread.Worksheet:
    """Creates or replaces a worksheet with name `worksheet_name`.

    Note(chris): Taking the approach of deleting worksheet to make sure that
    state of worksheet is totally clean.  Other methods of clearing worksheet using gspread
    did not clear conditional formatting rules.

    Args:
        sheet: Spreadsheet
        worksheet_name: Name of worksheet.

    Returns: Newly created Worksheet.
    """
    try:
        worksheet = spreadsheet.worksheet(worksheet_name)
        try:
            spreadsheet.del_worksheet(worksheet)
        except Exception:
            # If worksheet name exists but is the only worksheet, need to add a new tmp sheet
            # first then delete the old one
            new_worksheet = spreadsheet.add_worksheet("tmp", 100, 100)
            spreadsheet.del_worksheet(worksheet)
            new_worksheet.update_title(worksheet_name)
            return new_worksheet
    except gspread.WorksheetNotFound:
        pass

    return spreadsheet.add_worksheet(worksheet_name, 100, 100)


def create_or_clear_worksheet(sheet: gspread.Spreadsheet, worksheet_name: str) -> gspread.Worksheet:
    """Creates or clears a worksheet with name `worksheet_name`.

    Args:
        sheet: Spreadsheet
        worksheet_name: Name of worksheet.

    Returns: Worksheet with name `worksheet_name`.
    """
    try:
        worksheet = sheet.worksheet(worksheet_name)
        worksheet.clear()
        return worksheet
    except gspread.WorksheetNotFound:
        pass

    return sheet.add_worksheet(worksheet_name, 100, 100)


def open_spreadsheet(
    sheet_id: str, gspread_client: Optional[gspread.Client] = None,
) -> gspread.Spreadsheet:
    """Opens or creates a spreadsheet, optionally sharing with `share_email`.

    Args:
        sheet_name: Name of sheet to open or create.
        share_email: Email to share sheet with.

    Returns: Spreadsheet.
    """

    gspread_client = gspread_client or init_client()

    return gspread_client.open_by_key(sheet_id)


def open_or_create_spreadsheet(
    sheet_name: str,
    share_email: Optional[str] = None,
    gspread_client: Optional[gspread.Client] = None,
) -> gspread.Spreadsheet:
    """Opens or creates a spreadsheet, optionally sharing with `share_email`.

    Args:
        sheet_name: Name of sheet to open or create.
        share_email: Email to share sheet with.

    Returns: Spreadsheet.
    """

    gspread_client = gspread_client or init_client()

    try:
        sheet = gspread_client.open(sheet_name)
    except gspread.SpreadsheetNotFound:
        _logger.info("Sheet not found, creating.", sheet_name=sheet_name)
        sheet = gspread_client.create(sheet_name)

    if share_email:
        _logger.info("Sharing sheet", email=share_email)
        sheet.share(share_email, perm_type="user", role="writer")

    return sheet


def update_info_sheet(
    sheet: gspread.Spreadsheet,
    sheet_name: str = "Update Info",
    pointer_directory: pathlib.Path = dataset_utils.DATA_DIRECTORY,
):
    filename = dataset_pointer.form_filename(DatasetType.MULTI_REGION)
    pointer_path = pointer_directory / filename
    pointer = DatasetPointer.parse_raw(pointer_path.read_text())
    data = [
        ("Field", "Value"),
        ("Updated at", datetime.datetime.utcnow().isoformat()),
        ("Covid Data Model SHA", pointer.model_git_info.sha),
    ]
    worksheet = create_or_replace_worksheet(sheet, sheet_name)
    worksheet.update(data)

    _logger.info("Successfully updated Info worksheet")
    return worksheet
