"""Postgres connections for the configured local application database."""

import os

import psycopg
from pgvector.psycopg import register_vector
from psycopg.rows import dict_row


def connect():
    connection = psycopg.connect(
        os.environ["DATABASE_URL"], row_factory=dict_row, autocommit=True
    )
    register_vector(connection)
    return connection
