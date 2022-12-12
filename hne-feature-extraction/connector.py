"""connector.py.

Interface for reading and manipulating tables from Dremio.

Test with -
$ python breastana/connector.py
"""
import urllib.parse
from pathlib import Path
from typing import Any, Callable, Optional
import sys

import pandas as pd
from pyarrow import flight


class TableLoader:
    """Generic table loading interface."""

    def __init__(self, user: str, password: str, flight_port: int = 32010):
        self.mod_dict: dict[str, Any] = {}
        self.mod_mask = pd.DataFrame()
        self.user = user
        self.password = password
        self.flight_port = flight_port

    def load_from_dremio(self, url: Any) -> pd.DataFrame:
        """Load table from Dremio."""
        dremio_session = DremioDataframeConnector(
            scheme="grpc+tcp",
            hostname=url.hostname,
            flightport=self.flight_port,
            dremio_user=self.user,
            dremio_password=self.password,
            connection_args={},
        )
        return dremio_session.get_table(url.query)


class FeatureTableLoader(TableLoader):
    """Trying to make loading from dremio nicer."""

    def add_table(
        self,
        table: str,
        feature_tag: str,
        index_cols: list[str] = ["main_index"],
        fx_transform: Optional[Callable[[Any], Any]] = None,
        check_uniqueness: bool = True,
    ) -> pd.DataFrame:
        """Add table to FeatureTableLoader object."""
        if not type(table) == pd.DataFrame:
            path = Path(table)
            url = urllib.parse.urlparse(table)

            print(f"Info: {url}")

            if url.scheme == "file" and path.suffix == ".csv":
                df = pd.read_csv(path)
            if "dremio" in url.scheme:
                df = self.load_from_dremio(url)
        else:
            df = table.reset_index()

        df = df.set_index(index_cols)

        if fx_transform is not None:
            df = fx_transform(df)

        if not len(df.index.unique()) == len(df.index) and check_uniqueness:
            raise RuntimeError(
                "Feature tables must have unique indicies post-transform",
                f"N={len(df.index)-len(df.index.unique())} try something else!",
            )

        self.mod_dict[feature_tag] = df

        return df

    def calculate_mask(self) -> pd.DataFrame:
        """Calculate index mask."""
        for key in self.mod_dict.keys():
            df = self[key]
            main_index = df.index.get_level_values(0)

            self.mod_mask = self.mod_mask.reindex(self.mod_mask.index.union(main_index))
            self.mod_mask.loc[main_index, key] = True
            self.mod_mask = self.mod_mask.fillna(False)

        return self.mod_mask

    def __getitem__(self, key: str) -> Any:
        """Get item."""
        return self.mod_dict[key]

    def __setitem__(self, key: str, item: Any) -> None:
        """Set item."""
        self.mod_dict[key] = item


class DremioClientAuthMiddlewareFactory(flight.ClientMiddlewareFactory):  # type: ignore
    """A factory that creates DremioClientAuthMiddleware(s)."""

    def __init__(self) -> None:
        self.call_credential: list[Any] = []

    def start_call(self, info: Any) -> Any:  # type ignore:
        """Start call."""
        return DremioClientAuthMiddleware(self)

    def set_call_credential(self, call_credential: list[Any]) -> None:
        """Set call credentials."""
        self.call_credential = call_credential


class DremioClientAuthMiddleware(flight.ClientMiddleware):  # type: ignore
    """Dremio ClientMiddleware used for authentication.

    Extracts the bearer token from
    the authorization header returned by the Dremio
    Flight Server Endpoint.

    Parameters
    ----------
    factory : ClientHeaderAuthMiddlewareFactory
        The factory to set call credentials if an
        authorization header with bearer token is
        returned by the Dremio server.
    """

    def __init__(self, factory: DremioClientAuthMiddlewareFactory):
        self.factory = factory

    def received_headers(self, headers: dict[str, Any]) -> None:
        """Process header."""
        auth_header_key = "authorization"
        authorization_header: list[Any] = []
        for key in headers:
            if key.lower() == auth_header_key:
                authorization_header = headers.get(auth_header_key)  # type: ignore
        self.factory.set_call_credential(
            [b"authorization", authorization_header[0].encode("utf-8")]
        )


class DremioDataframeConnector:
    """Dremio connector.

    Iterfaces with a Dremio instance/cluster
    via Apache Arrow Flight for fast read performance.

    Parameters
    ----------
    scheme: connection scheme
    hostname: host of main dremio name
    flightport: which port dremio exposes to flight requests
    dremio_user: username to use
    dremio_password: associated password
    connection_args: anything else to pass to the FlightClient initialization
    """

    def __init__(
        self,
        scheme: str,
        hostname: str,
        flightport: int,
        dremio_user: str,
        dremio_password: str,
        connection_args: dict[str, Any],
    ):
        # Skipping tls...

        # Two WLM settings can be provided upon initial authentication
        # with the Dremio Server Flight Endpoint:
        # - routing-tag
        # - routing queue
        initial_options = flight.FlightCallOptions(
            headers=[
                (b"routing-tag", b"test-routing-tag"),
                (b"routing-queue", b"Low Cost User Queries"),
            ]
        )
        client_auth_middleware = DremioClientAuthMiddlewareFactory()
        client = flight.FlightClient(
            f"{scheme}://{hostname}:{flightport}",
            middleware=[client_auth_middleware],
            **connection_args,
        )
        self.bearer_token = client.authenticate_basic_token(
            dremio_user, dremio_password, initial_options
        )
        self.client = client

    def run(self, project: str, table_name: str) -> pd.DataFrame:
        """Get a fixed table.

        Returns the virtual table at project(or "space").table_name
        as a pandas dataframe

        Parameters
        ----------
        project: Project ID to read from
        table_name:  Table name to load

        """
        sqlquery = f'''SELECT * FROM "{project}"."{table_name}"'''

        # flight_desc = flight.FlightDescriptor.for_command(sqlquery)
        print("[INFO] Query: ", sqlquery)

        options = flight.FlightCallOptions(headers=[self.bearer_token])
        # schema = self.client.get_schema(flight_desc, options)

        # Get the FlightInfo message to retrieve the Ticket corresponding
        # to the query result set.
        flight_info = self.client.get_flight_info(
            flight.FlightDescriptor.for_command(sqlquery), options
        )

        # Retrieve the result set as a stream of Arrow record batches.
        reader = self.client.do_get(flight_info.endpoints[0].ticket, options)
        return reader.read_pandas()

    def get_table(self, sqlquery: str) -> pd.DataFrame:
        """Run a query.

        Returns the virtual table at project(or "space").table_name
        as a pandas dataframe.

        Parameters
        ----------
        project: Project ID to read from
        table_name:  Table name to load

        """
        # flight_desc = flight.FlightDescriptor.for_command(sqlquery)
        print("[INFO] Query: ", sqlquery)

        options = flight.FlightCallOptions(headers=[self.bearer_token])
        # schema = self.client.get_schema(flight_desc, options)

        # Get the FlightInfo message to retrieve the Ticket corresponding
        # to the query result set.
        flight_info = self.client.get_flight_info(
            flight.FlightDescriptor.for_command(sqlquery), options
        )

        # Retrieve the result set as a stream of Arrow record batches.
        reader = self.client.do_get(flight_info.endpoints[0].ticket, options)

        return reader.read_pandas()


if __name__ == "__main__":
    import getpass

    # set username and password
    # (or Personal Access Token) when prompted at the command prompt
    DREMIO_USER = input("Username: ")
    DREMIO_PASSWORD = getpass.getpass(prompt="Password or PAT: ", stream=None)

    dremio_session = DremioDataframeConnector(
        scheme="grpc+tcp",
        hostname="tlvidreamcord1",
        flightport=32010,
        dremio_user=DREMIO_USER,
        dremio_password=DREMIO_PASSWORD,
        connection_args={},
    )
    query = 'SELECT merged_hne_inventory.spectrum_sample_id, merged_hne_inventory.slide_image FROM merged_hne_inventory'

    df = dremio_session.get_table(query)
    df['merged_hne_inventory.slide_image'] = df['merged_hne_inventory.slide_image'].str.removeprefix("file://")
    df.to_csv(sys.stdout)
