from fastapi import FastAPI
from pydantic import BaseModel


class File(BaseModel):
    id: int
    name: str | None = None
    content: str