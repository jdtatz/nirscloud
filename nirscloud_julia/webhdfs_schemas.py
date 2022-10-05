from typing import Any, Dict, List

from typing_extensions import Literal, NotRequired, Required, TypedDict


class fileStatusProperties(TypedDict):
    accessTime: Required[int]
    """The access time."""
    blockSize: Required[int]
    """The block size of a file."""
    group: Required[str]
    """The group owner."""
    length: Required[int]
    """The number of bytes in a file."""
    modificationTime: Required[int]
    """The modification time."""
    owner: Required[str]
    """The user who is the owner."""
    pathSuffix: Required[str]
    """The path suffix."""
    permission: Required[str]
    """The permission represented as a octal string."""
    replication: Required[int]
    """The number of replication of a file."""
    symlink: NotRequired[str]
    """The link target of a symlink."""
    type: Required[Literal["FILE", "DIRECTORY", "SYMLINK"]]
    """The type of the path object."""


class tokenProperties(TypedDict):
    urlString: Required[str]
    """A delegation token encoded as a URL safe string."""


class blockStoragePolicyProperties(TypedDict):
    id: Required[int]
    """Policy ID."""
    name: Required[str]
    """Policy name."""
    storageTypes: Required[List[str]]
    """An array of storage types for block placement."""
    replicationFallbacks: Required[List[str]]
    """An array of fallback storage types for replication."""
    creationFallbacks: Required[List[str]]
    """An array of fallback storage types for file creation."""
    copyOnCreateFile: Required[bool]
    """If set then the policy cannot be changed after file creation."""


class diffReportEntries(TypedDict):
    sourcePath: Required[str]
    """Source path name relative to snapshot root"""
    targetPath: Required[str]
    """Target path relative to snapshot root used for renames"""
    type: Required[Literal["CREATE", "MODIFY", "DELETE", "RENAME"]]
    """Type of diff report entry"""


class snapshottableDirectoryStatus(TypedDict):
    dirStatus: fileStatusProperties
    parentFullPath: Required[str]
    """Full path of the parent of snapshottable directory"""
    snapshotNumber: Required[int]
    """Number of snapshots created on the snapshottable directory"""
    snapshotQuota: Required[int]
    """Total number of snapshots allowed on the snapshottable directory"""


class blockLocationProperties(TypedDict):
    cachedHosts: Required[List[str]]
    """Datanode hostnames with a cached replica"""
    corrupt: Required[bool]
    """True if the block is corrupted"""
    hosts: Required[List[str]]
    """Datanode hostnames store the block"""
    length: Required[int]
    """Length of the block"""
    names: Required[List[str]]
    """Datanode IP:xferPort for accessing the block"""
    offset: Required[int]
    """Offset of the block in the file"""
    storageTypes: Required[List[Literal["RAM_DISK", "SSD", "DISK", "ARCHIVE"]]]
    """Storage type of each replica"""
    topologyPaths: Required[List[str]]
    """Datanode addresses in network topology"""


class AclStatus_AclStatus(TypedDict):
    entries: NotRequired[List[str]]
    group: Required[str]
    """The group owner."""
    owner: Required[str]
    """The user who is the owner."""
    stickyBit: Required[bool]
    """True if the sticky bit is on."""


class AclStatus(TypedDict):
    AclStatus: NotRequired[AclStatus_AclStatus]


class XAttrs_XAttrs_item(TypedDict):
    name: Required[str]
    """XAttr name."""
    value: NotRequired[str]
    """XAttr value."""


class XAttrs(TypedDict):
    XAttrs: NotRequired[List[XAttrs_XAttrs_item]]


class XAttrNames(TypedDict):
    XAttrNames: Required[str]
    """XAttr names."""


class boolean(TypedDict):
    boolean: Required[bool]
    """A boolean value"""


class ContentSummary_ContentSummary_typeQuota_ARCHIVE(TypedDict):
    consumed: Required[int]
    """The storage type space consumed."""
    quota: Required[int]
    """The storage type quota."""


class ContentSummary_ContentSummary_typeQuota_DISK(TypedDict):
    consumed: Required[int]
    """The storage type space consumed."""
    quota: Required[int]
    """The storage type quota."""


class ContentSummary_ContentSummary_typeQuota_SSD(TypedDict):
    consumed: Required[int]
    """The storage type space consumed."""
    quota: Required[int]
    """The storage type quota."""


class ContentSummary_ContentSummary_typeQuota(TypedDict):
    ARCHIVE: NotRequired[ContentSummary_ContentSummary_typeQuota_ARCHIVE]
    DISK: NotRequired[ContentSummary_ContentSummary_typeQuota_DISK]
    SSD: NotRequired[ContentSummary_ContentSummary_typeQuota_SSD]


class ContentSummary_ContentSummary(TypedDict):
    directoryCount: Required[int]
    """The number of directories."""
    fileCount: Required[int]
    """The number of files."""
    length: Required[int]
    """The number of bytes used by the content."""
    quota: Required[int]
    """The namespace quota of this directory."""
    spaceConsumed: Required[int]
    """The disk space consumed by the content."""
    spaceQuota: Required[int]
    """The disk space quota."""
    typeQuota: NotRequired[ContentSummary_ContentSummary_typeQuota]


class ContentSummary(TypedDict):
    ContentSummary: NotRequired[ContentSummary_ContentSummary]


class QuotaUsage_QuotaUsage_typeQuota_ARCHIVE(TypedDict):
    consumed: Required[int]
    """The storage type space consumed."""
    quota: Required[int]
    """The storage type quota."""


class QuotaUsage_QuotaUsage_typeQuota_DISK(TypedDict):
    consumed: Required[int]
    """The storage type space consumed."""
    quota: Required[int]
    """The storage type quota."""


class QuotaUsage_QuotaUsage_typeQuota_SSD(TypedDict):
    consumed: Required[int]
    """The storage type space consumed."""
    quota: Required[int]
    """The storage type quota."""


class QuotaUsage_QuotaUsage_typeQuota(TypedDict):
    ARCHIVE: NotRequired[QuotaUsage_QuotaUsage_typeQuota_ARCHIVE]
    DISK: NotRequired[QuotaUsage_QuotaUsage_typeQuota_DISK]
    SSD: NotRequired[QuotaUsage_QuotaUsage_typeQuota_SSD]


class QuotaUsage_QuotaUsage(TypedDict):
    fileAndDirectoryCount: Required[int]
    """The number of files and directories."""
    quota: Required[int]
    """The namespace quota of this directory."""
    spaceConsumed: Required[int]
    """The disk space consumed by the content."""
    spaceQuota: Required[int]
    """The disk space quota."""
    typeQuota: NotRequired[QuotaUsage_QuotaUsage_typeQuota]


class QuotaUsage(TypedDict):
    QuotaUsage: NotRequired[QuotaUsage_QuotaUsage]


class FileChecksum_FileChecksum(TypedDict):
    algorithm: Required[str]
    """The name of the checksum algorithm."""
    bytes: Required[str]
    """The byte sequence of the checksum in hexadecimal."""
    length: Required[int]
    """The length of the bytes (not the length of the string)."""


class FileChecksum(TypedDict):
    FileChecksum: NotRequired[FileChecksum_FileChecksum]


class DirectoryListing_DirectoryListing(TypedDict):
    partialListing: Required[Dict[str, Any]]
    """A partial directory listing"""
    remainingEntries: Required[int]
    """Number of remaining entries"""


class DirectoryListing(TypedDict):
    DirectoryListing: NotRequired[DirectoryListing_DirectoryListing]


class long(TypedDict):
    long: Required[int]
    """A long integer value"""


class Path(TypedDict):
    Path: Required[str]
    """The string representation a Path."""


class RemoteException_RemoteException(TypedDict):
    exception: Required[str]
    """Name of the exception"""
    message: Required[str]
    """Exception message"""
    javaClassName: NotRequired[str]
    """Java class name of the exception"""


class RemoteException(TypedDict):
    RemoteException: NotRequired[RemoteException_RemoteException]


class BlockStoragePolicies_BlockStoragePolicies(TypedDict):
    BlockStoragePolicy: NotRequired[List[blockStoragePolicyProperties]]
    """An array of BlockStoragePolicy"""


class BlockStoragePolicies(TypedDict):
    BlockStoragePolicies: NotRequired[BlockStoragePolicies_BlockStoragePolicies]


class SnapshotDiffReport_SnapshotDiffReport(TypedDict):
    diffList: Required[List[diffReportEntries]]
    """An array of DiffReportEntry"""
    fromSnapshot: Required[str]
    """Source snapshot"""
    snapshotRoot: Required[str]
    """String representation of snapshot root path"""
    toSnapshot: Required[str]
    """Destination snapshot"""


class SnapshotDiffReport(TypedDict):
    SnapshotDiffReport: NotRequired[SnapshotDiffReport_SnapshotDiffReport]


class SnapshottableDirectoryList(TypedDict):
    SnapshottableDirectoryList: Required[List[snapshottableDirectoryStatus]]
    """An array of SnapshottableDirectoryStatus"""


class BlockLocations_BlockLocations(TypedDict):
    BlockLocation: NotRequired[List[blockLocationProperties]]
    """An array of BlockLocation"""


class BlockLocations(TypedDict):
    BlockLocations: NotRequired[BlockLocations_BlockLocations]


class BlockLocation(TypedDict):
    BlockLocation: blockLocationProperties


class BlockStoragePolicy(TypedDict):
    BlockStoragePolicy: blockStoragePolicyProperties


class Token(TypedDict):
    Token: tokenProperties


class FileStatus(TypedDict):
    FileStatus: fileStatusProperties


class FileStatuses_FileStatuses(TypedDict):
    FileStatus: NotRequired[List[fileStatusProperties]]
    """An array of FileStatus"""


class FileStatuses(TypedDict):
    FileStatuses: NotRequired[FileStatuses_FileStatuses]
