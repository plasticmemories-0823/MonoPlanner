"""Start the local server with all mutable storage outside the application."""
import json
import os
import sys

from agent.settings import PROJECT_ROOT
from agent.storage.preferences import state_root


def main():
    directory = state_root()/'service'
    directory.mkdir(parents=True, exist_ok=True)
    configuration = json.loads((PROJECT_ROOT/'langgraph.json').read_text(encoding='utf-8'))
    configuration['dependencies'] = [PROJECT_ROOT.as_posix()]
    configuration['graphs'] = {'monoplaner': (PROJECT_ROOT/'agent/server/graph.py').as_posix()+':graph'}
    configuration['http']['app'] = (PROJECT_ROOT/'agent/server/app.py').as_posix()+':app'
    config = directory/'langgraph.json'
    config.write_text(json.dumps(configuration), encoding='utf-8')
    os.chdir(directory)
    sys.path.insert(0, str(PROJECT_ROOT))
    from langgraph_cli.cli import cli
    cli(['dev', '--host', '127.0.0.1', '--port', os.getenv('MONOPLANER_BACKEND_PORT', '2024'),
         '--config', str(config), '--no-browser', '--no-reload'])


if __name__ == '__main__':
    main()
