import typer
from pydantic import BaseModel, ConfigDict
from stringcase import camelcase
from specklepy.transports.server import ServerTransport
from specklepy.api.client import SpeckleClient

from testing_local import run

class SpeckleProjectData(BaseModel):
    """Values of the project / model that triggered the run of this function."""

    project_id: str
    model_id: str
    version_id: str
    speckle_server_url: str

    model_config = ConfigDict(alias_generator=camelcase, protected_namespaces=())


class FunctionInputs(BaseModel):
    """
    These are function author defined values, automate will make sure to supply them.
    """

    keyword: str

    class Config:
        alias_generator = camelcase


def main(speckle_project_data: str, function_inputs: str, speckle_token: str):
    
    # schema comes from automate 
    project_data = SpeckleProjectData.model_validate_json(speckle_project_data)
    # defined by function author (above). Optional 
    inputs = FunctionInputs.model_validate_json(function_inputs)

    client = SpeckleClient(project_data.speckle_server_url, use_ssl=False)
    client.authenticate_with_token(speckle_token)
    server_transport = ServerTransport(project_data.project_id, client)
    run(client, server_transport, inputs.keyword) 

    print(
        "Ran function with",
        f"{speckle_project_data} {function_inputs}",
    )


if __name__ == "__main__":
    typer.run(main)
