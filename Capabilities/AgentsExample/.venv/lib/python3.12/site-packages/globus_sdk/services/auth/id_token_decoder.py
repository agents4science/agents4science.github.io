from __future__ import annotations

import datetime
import sys
import typing as t

import jwt
from cryptography.hazmat.primitives.asymmetric.rsa import RSAPublicKey

from globus_sdk.response import GlobusHTTPResponse

from ._common import SupportsJWKMethods

if t.TYPE_CHECKING:
    from globus_sdk import AuthLoginClient, GlobusAppConfig

    if sys.version_info >= (3, 11):
        from typing import Self
    else:
        from typing_extensions import Self


class IDTokenDecoder:
    """
    JWT decoder for OIDC ID tokens issued by Globus Auth.
    Decoding uses a client object to fetch necessary data from Globus Auth.

    By default, the OIDC configuration data and JWKs will be cached in an internal dict.
    An alternative cache can be provided on init to use an alternative storage
    mechanism.

    The ``get_jwt_audience`` and ``get_jwt_leeway`` methods supply parameters to
    decoding. Subclasses can override these methods to customize the decoder.

    :param auth_client: The client which should be used to callout to Globus Auth as
        needed. Any AuthClient or AuthLoginClient will work for this purpose.
    :param jwt_leeway: The JWT leeway to use during decoding, as a number of seconds
        or a timedelta. The default is 5 minutes.
    :param jwt_options: The ``options`` passed to the underlying JWT decode function.
        Defaults to an empty dict.
    """

    def __init__(
        self,
        auth_client: SupportsJWKMethods,
        *,
        # default to 300 seconds
        #
        # valuable inputs to this number:
        # - expected clock drift per day (6s for a bad clock)
        # - Windows time sync interval (64s)
        # - Windows' stated goal of meeting the Kerberos 5 clock skew requirement (5m)
        # - ntp panic threshold (1000s of drift)
        # - the knowledge that VM clocks typically run slower and may skew significantly
        #
        # NTP panic should be understood as a critical error; 1000s of drift is
        # therefore too high for us to allow.
        #
        # 300s (5m) is therefore chosen to match the Windows desired maximum for
        # clock drift, and the underlying Kerberos requirement.
        jwt_leeway: float | datetime.timedelta = 300.0,
        jwt_options: dict[str, t.Any] | None = None,
    ) -> None:
        self._auth_client = auth_client
        self._openid_configuration: dict[str, t.Any] | None = None
        self._jwk: RSAPublicKey | None = None

        self.jwt_leeway: float | datetime.timedelta = jwt_leeway
        self.jwt_options: dict[str, t.Any] = (
            jwt_options if jwt_options is not None else {}
        )

    @classmethod
    def for_globus_app(
        cls,
        *,
        app_name: str,  # pylint: disable=unused-argument
        config: GlobusAppConfig,  # pylint: disable=unused-argument
        login_client: AuthLoginClient,
    ) -> Self:
        """
        Create an ``IDTokenDecoder`` for use in a GlobusApp.

        :param app_name: The name supplied to the GlobusApp.
        :param config: The configuration supplied to the GlobusApp.
        :param login_client: A login client to use for instantiating an
            ``IDTokenDecoder``.
        """
        return cls(login_client)

    def decode(self, id_token: str, /) -> dict[str, t.Any]:
        """
        The ``decode()`` method takes an ``id_token`` as a string, and decodes it to
        a dictionary.

        This method should implicitly retrieve the OpenID configuration and JWK
        for Globus Auth.

        :param id_token: The token to decode
        """
        audience = self.get_jwt_audience()
        openid_configuration = self.get_openid_configuration()
        jwk = self.get_jwk()

        signing_algos = openid_configuration["id_token_signing_alg_values_supported"]

        return jwt.decode(
            id_token,
            key=jwk,
            algorithms=signing_algos,
            audience=audience,
            options=self.jwt_options,
            leeway=self.jwt_leeway,
        )

    def get_jwt_audience(self) -> str | None:
        """
        The audience for JWT verification defaults to the client's client ID.
        """
        return self._auth_client.client_id

    def store_openid_configuration(
        self, openid_configuration: dict[str, t.Any] | GlobusHTTPResponse
    ) -> None:
        """
        Store openid_configuration data for future use in ``decode()``.

        :param openid_configuration: The configuration data, as fetched via
            :meth:`AuthClient.get_openid_configuration`
        """
        if isinstance(openid_configuration, GlobusHTTPResponse):
            self._openid_configuration = openid_configuration.data
        else:
            self._openid_configuration = openid_configuration

    def store_jwk(self, jwk: RSAPublicKey) -> None:
        """
        Store a JWK for future use in ``decode()``.

        :param jwk: The JWK, as fetched via :meth:`AuthClient.get_jwk`
            with ``as_pem=True``.
        """
        self._jwk = jwk

    def get_openid_configuration(self) -> dict[str, t.Any]:
        """
        Fetch the OpenID Configuration for Globus Auth, and cache the result before
        returning it.

        If a config was previously stored, return that instead.
        """
        if self._openid_configuration is None:
            self._openid_configuration = (
                self._auth_client.get_openid_configuration().data
            )
        return self._openid_configuration

    def get_jwk(self) -> RSAPublicKey:
        """
        Fetch the JWK for Globus Auth, and cache the result before returning it.

        If a key was previously stored, return that instead.
        """
        if not self._jwk:
            self._jwk = self._auth_client.get_jwk(
                openid_configuration=self.get_openid_configuration(), as_pem=True
            )
        return self._jwk
