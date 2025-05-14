import importlib.metadata
import requests
from loguru import logger

from .cli import refresh_bearer_token, validate_api_http_response


class AtlasUser:
    def __init__(self):
        self.credentials = refresh_bearer_token()


class AtlasClass(object):
    def __init__(self):
        """
        Initializes the Atlas client.
        """
        try:
            tenant = self.credentials["tenant"]
            if tenant == "staging":
                api_hostname = "staging-api-atlas.nomic.ai"
                web_hostname = "staging-atlas.nomic.ai"
            elif tenant == "production":
                api_hostname = "api-atlas.nomic.ai"
                web_hostname = "atlas.nomic.ai"
            elif tenant == "enterprise":
                api_hostname = self.credentials["api_domain"]
                web_hostname = self.credentials["frontend_domain"]
            else:
                raise ValueError("Invalid tenant value")

            self.atlas_api_path = f"https://{api_hostname}"
            self.web_path = f"https://{web_hostname}"

            token = self.credentials["token"]
            self.token = token

            try:
                version = importlib.metadata.version("nomic")
            except importlib.metadata.PackageNotFoundError:
                version = "unknown"

            self.header = {
                "Authorization": f"Bearer {token}",
                "User-Agent": f"py-nomic/{version}",
            }

            if not self.token:
                raise ValueError("Token is missing")

        except KeyError as e:
            logger.error(f"Missing key in credentials: {e}")
            raise
        except ValueError as e:
            logger.error(f"Value error: {e}")
            raise

    @property
    def credentials(self):
        return refresh_bearer_token()

    def _get_current_user(self):
        api_base_path = self.atlas_api_path

        # Se agregó un tiempo de espera para evitar bloqueos indefinidos
        response = requests.get(
            api_base_path + "/v1/user",
            headers=self.header,
            timeout=10,
        )
        response = validate_api_http_response(response)
        if response.status_code != 200:
            logger.error("Failed to fetch current user")
            response.raise_for_status()

        return response.json()

    def _validate_map_data_inputs(self, colorable_fields, data_sample):
        """Validates inputs to map data calls."""

        if not isinstance(colorable_fields, list):
            raise TypeError("colorable_fields must be a list")

        for field in colorable_fields:
            if field not in data_sample:
                raise ValueError(f"Field {field} is not in data sample")

    def _get_current_users_main_organization(self):
        """
        Retrieves the ID of the current user's default organization.

        **Returns:** The ID of the current user's default organization
        """

        user = self._get_current_user()

        for organization in user.get("organizations", []):
            if organization.get("is_default"):
                return organization["id"]

        raise ValueError("No default organization found")


class AtlasDataset:
    def __init__(self, name: str, unique_id_field: str = None, identifier: str = None, description: str = None):
        """
        Inicializa un conjunto de datos Atlas.

        Args:
            name (str): Nombre del conjunto de datos.
            unique_id_field (str, opcional):
                Campo único para identificar los datos.
            identifier (str, opcional):
                Identificador único del conjunto de datos.
            description (str, opcional):
                Descripción del conjunto de datos.
        """
        self.name = name
        self.unique_id_field = unique_id_field
        self.identifier = identifier
        self.description = description

    def delete(self):
        """
        Simula la eliminación del conjunto de datos.
        """
        print(f"Dataset {self.name} eliminado.")

    def add_data(self, data):
        """
        Agrega datos al conjunto de datos.

        Args:
            data: Datos a agregar.
        """
        print(f"Agregando datos al conjunto {self.name}: {data}")


class AtlasDataStream:
    def __init__(self, data):
        """
        Inicializa un flujo de datos Atlas.

        Args:
            data: Los datos a procesar en el flujo.
        """
        self.data = data

    def process(self):
        """
        Simula el procesamiento de datos en el flujo.
        """
        print("Procesando datos en el flujo...")

    def get_credentials(self):
        """
        Obtiene las credenciales necesarias para el flujo de datos.
        """
        print("Obteniendo credenciales para el flujo de datos...")
