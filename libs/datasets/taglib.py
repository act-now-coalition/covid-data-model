import collections
import dataclasses
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
from covidactnow.datapublic.common_fields import CommonFields
from covidactnow.datapublic.common_fields import FieldName
from covidactnow.datapublic.common_fields import GetByValueMixin
from covidactnow.datapublic.common_fields import PdFields
from covidactnow.datapublic.common_fields import ValueAsStrMixin


@enum.unique
class TagField(GetByValueMixin, ValueAsStrMixin, FieldName, enum.Enum):
    """The attributes of a tag, columns in a table with one row per tag."""

    LOCATION_ID = CommonFields.LOCATION_ID
    # VARIABLE values should be a metric name, typically one in CommonFields.
    VARIABLE = PdFields.VARIABLE
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

    def as_record(self, location_id: str, variable: CommonFields) -> Mapping[str, Any]:
        """Returns this tag in a record with the content serialized."""
        return {
            TagField.LOCATION_ID: location_id,
            TagField.VARIABLE: variable,
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


@dataclass(frozen=True)
class Source(TagInTimeseries):
    type: str
    url: Optional[UrlStr] = None
    name: Optional[str] = None

    TAG_TYPE = TagType.SOURCE

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


TAG_TYPE_TO_CLASS = {
    TagType.CUMULATIVE_TAIL_TRUNCATED: CumulativeTailTruncated,
    TagType.CUMULATIVE_LONG_TAIL_TRUNCATED: CumulativeLongTailTruncated,
    TagType.ZSCORE_OUTLIER: ZScoreOutlier,
    TagType.PROVENANCE: ProvenanceTag,
    TagType.SOURCE_URL: SourceUrl,
    TagType.SOURCE: Source,
}


@dataclass(frozen=True)
class TagCollection:
    """A collection of TagInTimeseries, organized by location and field name. The collection
    object itself is frozen but the dict within it may be modified."""

    _location_var_map: MutableMapping[Tuple, List[TagInTimeseries]] = dataclasses.field(
        default_factory=lambda: collections.defaultdict(list)
    )

    def add(self, tag: TagInTimeseries, *, location_id: str, variable: CommonFields) -> None:
        """Adds a tag to this collection."""
        self._location_var_map[(location_id, variable)].append(tag)

    def _as_records(self) -> Iterable[Mapping]:
        for (location_id, variable), tags in self._location_var_map.items():
            for tag in tags:
                yield tag.as_record(location_id, variable)

    def as_dataframe(self) -> pd.DataFrame:
        """Returns all tags in this collection in a DataFrame."""
        return pd.DataFrame.from_records(self._as_records())
