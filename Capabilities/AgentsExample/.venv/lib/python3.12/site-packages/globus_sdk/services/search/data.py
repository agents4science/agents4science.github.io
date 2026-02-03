from __future__ import annotations

import typing as t

from globus_sdk._missing import MISSING, MissingType
from globus_sdk._payload import GlobusPayload


class SearchQueryV1(GlobusPayload):
    """
    A specialized dict which has helpers for creating and modifying a Search
    Query document.

    :param q: The query string. Required unless filters are used.
    :param limit: A limit on the number of results returned in a single page
    :param offset: An offset into the set of all results for the query
    :param advanced: Whether to enable (``True``) or not to enable (``False``) advanced
        parsing of query strings. The default of ``False`` is robust and guarantees that
        the query will not error with "bad query string" errors
    :param filters: a list of filters to apply to the query
    :param facets: a list of facets to apply to the query
    :param post_facet_filters: a list of filters to apply after facet
        results are returned
    :param boosts: a list of boosts to apply to the query
    :param sort: a list of fields to sort results
    :param additional_fields: additional data to include in the query document
    """

    def __init__(
        self,
        *,
        q: str | MissingType = MISSING,
        limit: int | MissingType = MISSING,
        offset: int | MissingType = MISSING,
        advanced: bool | MissingType = MISSING,
        filters: list[dict[str, t.Any]] | MissingType = MISSING,
        facets: list[dict[str, t.Any]] | MissingType = MISSING,
        post_facet_filters: list[dict[str, t.Any]] | MissingType = MISSING,
        boosts: list[dict[str, t.Any]] | MissingType = MISSING,
        sort: list[dict[str, t.Any]] | MissingType = MISSING,
        additional_fields: dict[str, t.Any] | None = None,
    ) -> None:
        super().__init__()
        self["@version"] = "query#1.0.0"

        self["q"] = q
        self["limit"] = limit
        self["offset"] = offset
        self["advanced"] = advanced
        self["filters"] = filters
        self["facets"] = facets
        self["post_facet_filters"] = post_facet_filters
        self["boosts"] = boosts
        self["sort"] = sort
        self.update(additional_fields or {})


class SearchScrollQuery(GlobusPayload):
    """
    A scrolling query type, for scrolling the full result set for an index.

    Scroll queries have more limited capabilities than general searches.
    They cannot boost fields, sort, or apply facets. They can, however, still apply the
    same filtering mechanisms which are available to normal queries.

    Scrolling also differs in that it supports the use of the ``marker`` field, which is
    used to paginate results.

    :param q: The query string
    :param limit: A limit on the number of results returned in a single page
    :param advanced: Whether to enable (``True``) or not to enable (``False``) advanced
        parsing of query strings. The default of ``False`` is robust and guarantees that
        the query will not error with "bad query string" errors
    :param marker: the marker value
    :param additional_fields: additional data to include in the query document
    """

    def __init__(
        self,
        q: str | MissingType = MISSING,
        *,
        limit: int | MissingType = MISSING,
        advanced: bool | MissingType = MISSING,
        marker: str | MissingType = MISSING,
        additional_fields: dict[str, t.Any] | None = None,
    ) -> None:
        super().__init__()
        self["q"] = q
        self["limit"] = limit
        self["advanced"] = advanced
        self["marker"] = marker
        self.update(additional_fields or {})
