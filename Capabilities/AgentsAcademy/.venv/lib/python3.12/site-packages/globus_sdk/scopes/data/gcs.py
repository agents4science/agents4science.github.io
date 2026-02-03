from ..builder import ScopeBuilder


class GCSEndpointScopeBuilder(ScopeBuilder):
    """
    A ScopeBuilder with a named property for the GCS manage_collections scope.
    "manage_collections" is a scope on GCS Endpoints. The resource_server string should
    be the GCS Endpoint ID.

    **Examples**

    >>> sb = GCSEndpointScopeBuilder("xyz")
    >>> mc_scope = sb.manage_collections
    """

    _classattr_scope_names = ["manage_collections"]

    @property
    def manage_collections(self) -> str:
        return self.urn_scope_string("manage_collections")


class GCSCollectionScopeBuilder(ScopeBuilder):
    """
    A ScopeBuilder with a named property for the GCS data_access scope.
    "data_access" is a scope on GCS Collections. The resource_server string should
    be the GCS Collection ID.

    **Examples**

    >>> sb = GCSCollectionScopeBuilder("xyz")
    >>> da_scope = sb.data_access
    >>> https_scope = sb.https
    """

    _classattr_scope_names = ["data_access", "https"]

    @property
    def data_access(self) -> str:
        return self.url_scope_string("data_access")

    @property
    def https(self) -> str:
        return self.url_scope_string("https")
