import collections
import dataclasses
import datetime
import enum
import json
from abc import ABC
from abc import abstractmethod
from dataclasses import dataclass
from typing import Any
from typing import ClassVar
from typing import Iterable
from typing import List
from typing import Mapping
from typing import MutableMapping
from typing import Optional
from typing import Tuple

import pandas as pd
from datapublic.common_fields import CommonFields
from datapublic.common_fields import DemographicBucket
from datapublic.common_fields import FieldName
from datapublic.common_fields import GetByValueMixin
from datapublic.common_fields import PdFields
from datapublic.common_fields import ValueAsStrMixin

from libs.dataclass_utils import dataclass_with_default_init


@enum.unique
class TagField(GetByValueMixin, ValueAsStrMixin, FieldName, enum.Enum):
    """The attributes of a tag, columns in a table with one row per tag."""

    LOCATION_ID = CommonFields.LOCATION_ID
    # VARIABLE values should be a metric name, typically one in CommonFields.
    VARIABLE = PdFields.VARIABLE
    DEMOGRAPHIC_BUCKET = PdFields.DEMOGRAPHIC_BUCKET
    # TYPE values must be a string from TagType
    TYPE = "tag_type"
    # CONTENT values vary depending on TYPE, either a JSON string or bare string.
    CONTENT = "content"


@enum.unique
class TagType(GetByValueMixin, ValueAsStrMixin, str, enum.Enum):
    """The type of the annotation.

    Each enumeration refers to the method used to generate the annotation.
    """

    # Currently 'annotation' refers to tag types that must have a specific real value for DATE.
    # Other tags (currently only PROVENANCE, but may be expanded for things such as storing
    # a processing step name) do not have a date field. Putting the type of all
    # tags in a single enum makes it easy to represent all tags in the same structure but
    # makes the concept of an 'annotation type' less explicit in our code.

    CUMULATIVE_TAIL_TRUNCATED = "cumulative_tail_truncated"
    CUMULATIVE_LONG_TAIL_TRUNCATED = "cumulative_long_tail_truncated"
    ZSCORE_OUTLIER = "zscore_outlier"
    KNOWN_ISSUE = "known_issue"
    KNOWN_ISSUE_NO_DATE = "known_issue_no_date"
    DERIVED = "derived"
    DROP_FUTURE_OBSERVATION = "drop_future_observation"
    # When adding a TagType remember to update timeseries_stats.StatName so it is displayed in
    # the dashboard.

    PROVENANCE = PdFields.PROVENANCE
    SOURCE_URL = "source_url"
    SOURCE = "source"


@dataclass(frozen=True)
class TagInTimeseries(ABC):
    """Represents a tag in the context of a particular timeseries"""

    # TAG_TYPE must be set by subclasses.
    TAG_TYPE: ClassVar[TagType]

    @property
    def tag_type(self) -> TagType:
        return self.TAG_TYPE

    @property
    @abstractmethod
    def content(self) -> str:
        pass

    @staticmethod
    def make(tag_type: TagType, *, content: str) -> "TagInTimeseries":
        """Given a `tag_type` and content, returns an instance of the appropriate class."""
        return TAG_TYPE_TO_CLASS[tag_type].make_instance(content=content)

    @classmethod
    @abstractmethod
    def make_instance(cls, *, content: str) -> "TagInTimeseries":
        """Deserializes the content and returns an instance of the TagInTimeseries subclass."""
        pass

    def as_record(
        self, location_id: str, variable: CommonFields, bucket: DemographicBucket
    ) -> Mapping[str, Any]:
        """Returns this tag in a record with the content serialized."""
        return {
            TagField.LOCATION_ID: location_id,
            TagField.VARIABLE: variable,
            TagField.DEMOGRAPHIC_BUCKET: bucket,
            TagField.TYPE: self.tag_type,
            TagField.CONTENT: self.content,
        }


@dataclass(frozen=True)
class ProvenanceTag(TagInTimeseries):
    source: str

    TAG_TYPE = TagType.PROVENANCE

    @classmethod
    def make_instance(cls, *, content: str) -> "TagInTimeseries":
        return cls(source=content)

    @property
    def content(self) -> str:
        return self.source


class UrlStr(str):
    """Wraps str to provide some type safety."""

    # If we need to do more with URLs consider replacing UrlStr with https://pypi.org/project/yarl/
    @staticmethod
    def make_optional(str_in: Optional[str]) -> Optional["UrlStr"]:
        return UrlStr(str_in) if str_in else None


@dataclass(frozen=True)
class SourceUrl(TagInTimeseries):
    source: UrlStr

    TAG_TYPE = TagType.SOURCE_URL

    @classmethod
    def make_instance(cls, *, content: str) -> "TagInTimeseries":
        return cls(source=UrlStr(content))

    @property
    def content(self) -> str:
        return self.source


@dataclass_with_default_init(frozen=True)
class Source(TagInTimeseries):
    type: str
    url: Optional[UrlStr] = None
    name: Optional[str] = None

    TAG_TYPE = TagType.SOURCE

    def __init__(self, type, *, url=None, name=None):
        # pylint: disable=E1101
        self.__default_init__(type=type, url=(url or None), name=(name or None))

    @staticmethod
    def rename_and_make_tag_df(
        in_df: pd.DataFrame, *, source_type: Optional[str] = None, rename: Mapping[str, str]
    ) -> pd.DataFrame:
        """Creates a Source for each row of `in_df`.

        Args:
            in_df: DataFrame with columns to be used to create Source tags and MultiIndex
            starting with a location and variable. All index levels are returned in columns.
            rename: Maps from column of in_df to Source attribute name. Columns not in `rename` are
              ignored; add an identity mapping if a column in in_df is to be used as an attribute
              of Source.
            source_type: optional static value for Source `type` attribute

        Returns:
            Source JSONs in a pd.DataFrame suitable for passing to MultiRegionDataset
        """

        assert in_df.index.names[0] in [CommonFields.FIPS, CommonFields.LOCATION_ID]
        assert in_df.index.names[1] == PdFields.VARIABLE
        # Make a DataFrame with columns for each attribute of Source
        columns_to_keep = in_df.columns.intersection(rename.keys())
        attribute_df = in_df.loc[:, columns_to_keep].rename(columns=rename)
        if source_type:
            attribute_df["type"] = source_type
        json_series = Source.attribute_df_to_json_series(attribute_df)
        source_df = json_series.rename(TagField.CONTENT).reset_index()
        source_df[TagField.TYPE] = TagType.SOURCE
        return source_df

    @staticmethod
    def attribute_df_to_json_series(attribute_df: pd.DataFrame) -> pd.Series:
        """Turns a pd.DataFrame of attributes in columns into a pd.Series with the same index."""
        assert attribute_df.columns.isin([f.name for f in dataclasses.fields(Source)]).all()
        # Convert any kind of NA into an empty string so that join/merge below works. Currently
        # it doesn't work because copying NA between column and index changes it from np.nan to
        # NA (or something like that). We could fix this by changing to consistent use of np.nan
        # but that depends on buggy behavior documented at https://stackoverflow.com/a/53719315.
        # Instead convert to empty string and let Source.__init__ convert that back to None.
        attribute_df = attribute_df.fillna("")
        attribute_columns = attribute_df.columns.to_list()
        # Use slow Source.content instead of something like https://stackoverflow.com/a/64700027
        # because Pandas to_json encodes slightly differently, breaking tests that compare JSON
        # objects as strings.

        # Make a pd.Series that has an index of the unique rows of attribute_df and values of
        # Source.content (a JSON str). This is done so the very slow lambda is only called once
        # per unique values. This speeds up 'data update' by several minutes. Too bad Pandas doesn't
        # cache apply return values internally like it does for datetime conversion.
        unique_contents = (
            attribute_df.drop_duplicates()
            .set_index(attribute_columns, drop=False)
            .apply(lambda row: Source(**row.to_dict()).content, axis=1, result_type="reduce")
            .rename(TagField.CONTENT)
        )
        assert unique_contents.index.names == attribute_columns
        # Left join to return a pd.Series with the same index as attribute_df and values from
        # unique_contents.
        return attribute_df.join(unique_contents, on=attribute_columns)[TagField.CONTENT]

    @classmethod
    def make_instance(cls, *, content: str) -> "TagInTimeseries":
        content_parsed = json.loads(content)
        return cls(
            type=content_parsed["type"],
            url=UrlStr.make_optional(content_parsed.get("url", None)),
            name=content_parsed.get("name", None),
        )

    @property
    def content(self) -> str:
        d = {"type": self.type}
        if self.url:
            d["url"] = self.url
        if self.name:
            d["name"] = self.name
        return json.dumps(d, separators=(",", ":"))


@dataclass(frozen=True)
class AnnotationWithDate(TagInTimeseries, ABC):
    """Represents an annotation, a tag added when modifying or removing one or more observations"""

    original_observation: float
    date: pd.Timestamp

    @classmethod
    def make_instance(cls, *, content: str) -> "TagInTimeseries":
        content_parsed = json.loads(content)
        date = pd.to_datetime(content_parsed.pop("date"))
        return cls(date=date, **content_parsed)

    @property
    def content(self) -> str:
        return json.dumps(
            {
                "date": self.date.date().isoformat(),
                "original_observation": self.original_observation,
            }
        )


@dataclass(frozen=True)
class CumulativeTailTruncated(AnnotationWithDate):
    TAG_TYPE = TagType.CUMULATIVE_TAIL_TRUNCATED


@dataclass(frozen=True)
class CumulativeLongTailTruncated(AnnotationWithDate):
    TAG_TYPE = TagType.CUMULATIVE_LONG_TAIL_TRUNCATED


@dataclass(frozen=True)
class ZScoreOutlier(AnnotationWithDate):
    TAG_TYPE = TagType.ZSCORE_OUTLIER


@dataclass(frozen=True)
class KnownIssue(TagInTimeseries):
    public_note: str
    date: datetime.date

    TAG_TYPE = TagType.KNOWN_ISSUE

    @classmethod
    def make_instance(cls, *, content: str) -> "TagInTimeseries":
        content_parsed = json.loads(content)
        # Support loading old "disclaimer" and new "public_note".
        public_note = content_parsed.get("disclaimer", content_parsed.get("public_note", ""))
        return cls(
            public_note=public_note, date=datetime.date.fromisoformat(content_parsed["date"]),
        )

    @property
    def content(self) -> str:
        d = {"public_note": self.public_note, "date": self.date.isoformat()}
        return json.dumps(d, separators=(",", ":"))


@dataclass(frozen=True)
class KnownIssueNoDate(TagInTimeseries):
    public_note: str

    TAG_TYPE = TagType.KNOWN_ISSUE_NO_DATE

    @classmethod
    def make_instance(cls, *, content: str) -> "TagInTimeseries":
        content_parsed = json.loads(content)
        return cls(public_note=content_parsed["public_note"])

    @property
    def content(self) -> str:
        d = {"public_note": self.public_note}
        return json.dumps(d, separators=(",", ":"))


@dataclass(frozen=True)
class Derived(TagInTimeseries):
    TAG_TYPE = TagType.DERIVED
    # Name of the function which added this derived tag.
    function_name: str

    @classmethod
    def make_instance(cls, *, content: str) -> "TagInTimeseries":
        content_parsed = json.loads(content)
        return cls(function_name=content_parsed.get("f", ""))

    @property
    def content(self) -> str:
        d = {"f": self.function_name}
        return json.dumps(d, separators=(",", ":"))


@dataclass(frozen=True)
class DropFutureObservation(TagInTimeseries):
    TAG_TYPE = TagType.DROP_FUTURE_OBSERVATION
    after: datetime.date

    @classmethod
    def make_instance(cls, *, content: str) -> "TagInTimeseries":
        content_parsed = json.loads(content)
        return cls(after=datetime.date.fromisoformat(content_parsed["after"]),)

    @property
    def content(self) -> str:
        d = {"after": self.after.isoformat()}
        return json.dumps(d, separators=(",", ":"))


TAG_TYPE_TO_CLASS = {
    TagType.CUMULATIVE_TAIL_TRUNCATED: CumulativeTailTruncated,
    TagType.CUMULATIVE_LONG_TAIL_TRUNCATED: CumulativeLongTailTruncated,
    TagType.ZSCORE_OUTLIER: ZScoreOutlier,
    TagType.PROVENANCE: ProvenanceTag,
    TagType.SOURCE_URL: SourceUrl,
    TagType.SOURCE: Source,
    TagType.KNOWN_ISSUE: KnownIssue,
    TagType.KNOWN_ISSUE_NO_DATE: KnownIssueNoDate,
    TagType.DERIVED: Derived,
    TagType.DROP_FUTURE_OBSERVATION: DropFutureObservation,
}


@dataclass(frozen=True)
class TagCollection:
    """A collection of TagInTimeseries, organized by location, bucket and field name. The collection
    object itself is frozen but the dict within it may be modified."""

    _location_var_map: MutableMapping[Tuple, List[TagInTimeseries]] = dataclasses.field(
        default_factory=lambda: collections.defaultdict(list)
    )

    def add(
        self,
        tag: TagInTimeseries,
        *,
        location_id: str,
        variable: CommonFields,
        bucket: DemographicBucket
    ) -> None:
        """Adds a tag to this collection."""
        self._location_var_map[(location_id, variable, bucket)].append(tag)

    def _as_records(self) -> Iterable[Mapping]:
        for (location_id, variable, bucket), tags in self._location_var_map.items():
            for tag in tags:
                yield tag.as_record(location_id, variable, bucket)

    def as_dataframe(self) -> pd.DataFrame:
        """Returns all tags in this collection in a DataFrame."""
        return pd.DataFrame.from_records(self._as_records())


def series_string_to_object(tag: pd.Series) -> pd.Series:
    """Converts a Series of content strings (generally JSONs) into a Series of TagInTimeseries
    objects."""
    type_idx_offset = tag.index.names.index(TagField.TYPE)
    assert type_idx_offset > 0
    assert tag.name == TagField.CONTENT
    # Apply a function to each element in the Series self.tag with the function having access to
    # the index of each element. From https://stackoverflow.com/a/47645833/341400.
    # result_type reduce forces the return value to be a Series, even when tag is empty.
    return tag.to_frame().apply(
        lambda row: TagInTimeseries.make(row.name[type_idx_offset], content=row.content),
        axis=1,
        result_type="reduce",
    )
