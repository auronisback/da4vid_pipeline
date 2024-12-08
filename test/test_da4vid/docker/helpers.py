"""
Helper functions to duplicate images across tests of docker containers.
"""
import docker


def duplicate_image(client: docker.DockerClient, original_name: str, duplicate_name):
  if client is None:
    raise RuntimeError('Docker client not specified')
  client.images.get(original_name).tag(duplicate_name, 'latest')


def remove_duplicate_image(client: docker.DockerClient, duplicate_name: str):
  if client is None:
    raise RuntimeError('Docker client not specified')
  client.images.remove(f'{duplicate_name}:latest')
